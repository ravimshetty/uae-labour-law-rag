"""End-to-end orchestration: EN / code-switched query -> grounded Arabic answer.

    ┌ input guardrails ┐  ┌ privacy ┐  ┌ code-switch ┐  ┌ router ┐  ┌ hybrid retrieval ┐
    query ─► size/injection/scope ─► PII vault ─► normalise+expand ─► federal|difc|both ─► EN+AR, BM25+dense, RRF
           ─► prompt (paired AR/EN articles) ─► LLM ─► output guardrails (cite / numbers / arabic / PII)
           ─► (regenerate once) ─► disclaimer ─► audit log (redacted) ─► response
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from .config import Settings, load_settings
from .generation.llm import LLM, build_llm
from .generation.prompts import SYSTEM_PROMPT, build_user_prompt
from .guardrails.input_guard import REFUSALS_AR, check_input, sanitize_context
from .guardrails.output_guard import check_output, finalize
from .index.embeddings import build_embedder
from .index.store import KnowledgeBase
from .models import Citation, GuardrailFlag, Jurisdiction, PipelineResult
from .privacy.audit import AuditLogger, sha256
from .privacy.pii import PIIVault, redact
from .retrieval.hybrid import ArticleHit, CrossEncoderReranker, HybridRetriever
from .retrieval.router import route
from .text.arabic import to_arabic_indic_digits
from .text.codeswitch import Glossary, normalize_query

log = logging.getLogger(__name__)
MAX_SOURCE_CHARS = 2500  # per article per language in the prompt


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, kb: KnowledgeBase | None = None, llm: LLM | None = None):
        self.s = settings or load_settings()
        self.kb = kb or KnowledgeBase.load(self.s.path("paths.index_dir"))
        self.embedder = build_embedder(self.s.get("embeddings.provider"), self.s.get("embeddings.model"),
                                       self.s.get("embeddings.dim", 1024))
        reranker = None
        if self.s.get("retrieval.use_reranker"):
            try:
                reranker = CrossEncoderReranker(self.s.get("retrieval.reranker_model"))
            except ImportError:
                log.warning("sentence-transformers not installed – reranker disabled")
        self.retriever = HybridRetriever(
            self.kb, self.embedder, top_k_bm25=self.s.get("retrieval.top_k_bm25", 20),
            top_k_dense=self.s.get("retrieval.top_k_dense", 20), rrf_k=self.s.get("retrieval.rrf_k", 60),
            reranker=reranker)
        self.llm = llm or build_llm(self.s.get("llm.provider"), self.s.get("llm.model"),
                                    temperature=self.s.get("llm.temperature", 0.0),
                                    max_tokens=self.s.get("llm.max_tokens", 900),
                                    timeout_s=self.s.get("llm.timeout_s", 60))
        self.glossary = Glossary.load(self.s.path("paths.glossary"))
        self.audit = AuditLogger(self.s.path("paths.audit_log"), self.s.get("privacy.audit_store_raw_query", False))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _source_payload(hits: list[ArticleHit]) -> list[dict]:
        out = []
        for i, h in enumerate(hits, 1):
            p = h.primary
            out.append({
                "ref": f"S{i}", "article": p.article,
                "title": (h.en or p).title or (h.ar or p).title,
                "title_ar": h.ar.title if h.ar else "",
                "jurisdiction": p.jurisdiction.value,
                "authoritative_lang": p.lang.value,
                "law_en": (h.en.law_name if h.en else p.meta.get("law_name_en", p.law_name)),
                "law_ar": (h.ar.law_name if h.ar else p.meta.get("law_name_ar", p.law_name)),
                "text_ar": sanitize_context(h.ar.text[:MAX_SOURCE_CHARS]) if h.ar else "",
                "text_en": sanitize_context(h.en.text[:MAX_SOURCE_CHARS]) if h.en else "",
                "url": p.source_url,
            })
        return out

    @staticmethod
    def _split_gist(raw: str) -> tuple[str, str]:
        if "GIST_EN:" in raw:
            body, gist = raw.rsplit("GIST_EN:", 1)
            return body.strip(), gist.strip()
        return raw.strip(), ""

    def _refuse(self, rid: str, key: str, flags: list[GuardrailFlag], t0: float, red_q: str,
                raw: str, jur: str = "n/a", vault: PIIVault | None = None) -> PipelineResult:
        self.audit.log({"request_id": rid, "query_sha256": sha256(raw), "redacted_query": red_q,
                        "blocked": True, "reason": key, "flags": [f.name for f in flags],
                        "latency_ms": int((time.time() - t0) * 1000)})
        return PipelineResult(rid, REFUSALS_AR[key], "", jur, [], flags, blocked=True,
                              redactions=vault.summary() if vault else {})

    # ------------------------------------------------------------------ main
    def answer(self, query: str, jurisdiction_hint: str | None = None, debug: bool = False) -> PipelineResult:
        t0, rid = time.time(), uuid.uuid4().hex[:12]

        # 1) privacy first: redact so even refusals/audit never contain raw PII
        red_q, vault = redact(query)

        # 2) code-switch normalisation + cross-lingual expansion (also feeds the scope check)
        nq = normalize_query(red_q[:5000], self.glossary)

        # 3) input guardrails
        chk = check_input(red_q, self.s.get("guardrails.max_query_chars", 1500),
                          self.s.get("guardrails.block_prompt_injection", True),
                          has_legal_concepts=bool(nq.concepts))
        flags = list(chk.flags)
        if vault.counts:
            flags.append(GuardrailFlag("pii_redacted", "info", str(vault.summary())))
        if not chk.allowed:
            return self._refuse(rid, chk.refusal_key or "out_of_scope", flags, t0, red_q, query, vault=vault)
        high_stakes = any(f.name == "high_stakes" for f in flags)

        if nq.is_code_switched:
            flags.append(GuardrailFlag("code_switched_query", "info", ",".join(nq.arabizi_tokens)))

        # 4) jurisdiction routing
        r = route(red_q, self.s.get("router.default_when_ambiguous", "both"))
        if r.out_of_scope == "adgm":
            flags.append(GuardrailFlag("adgm_out_of_scope", "block", "ADGM not in knowledge base"))
            return self._refuse(rid, "adgm", flags, t0, red_q, query, vault=vault)
        jur = Jurisdiction(jurisdiction_hint) if jurisdiction_hint in {"federal", "difc", "both"} else r.jurisdiction

        # 5) retrieval
        hits = self.retriever.retrieve(nq.query_en, nq.query_ar, jur, k=self.s.get("retrieval.final_k", 6))
        if not hits:
            return self._refuse(rid, "no_context", flags, t0, red_q, query, jur.value, vault)
        sources = self._source_payload(hits)
        source_texts = {s["ref"]: f"{s['text_ar']} {s['text_en']} {s['article']}" for s in sources}

        # 6) generation + output guardrails (+1 regeneration with feedback)
        feedback, out, gist = None, None, ""
        for attempt in range(1 + self.s.get("guardrails.max_regenerations", 1)):
            raw = self.llm.complete(SYSTEM_PROMPT, build_user_prompt(red_q, sources, jur.value, nq.concepts, feedback))
            body, gist = self._split_gist(raw)
            if body.strip() == "NO_ANSWER":
                return self._refuse(rid, "no_context", flags, t0, red_q, query, jur.value, vault)
            out = check_output(body, source_texts=source_texts, query=red_q, vault=vault,
                               min_arabic_ratio=self.s.get("guardrails.min_arabic_ratio", 0.6),
                               require_citations=self.s.get("guardrails.require_citations", True),
                               numeric_grounding=self.s.get("guardrails.numeric_grounding", True))
            if not out.needs_regen:
                break
            feedback = "; ".join(f"{f.name}({f.detail})" for f in out.flags)
        assert out is not None
        flags += out.flags
        if self.s.get("guardrails.require_citations", True) and not out.citations_used:
            return self._refuse(rid, "no_context", flags, t0, red_q, query, jur.value, vault)

        answer = finalize(out.text, high_stakes)
        if self.s.get("privacy.rehydrate_user_pii_in_answer", False):
            answer = vault.rehydrate(answer)
        if self.s.get("app.digits") == "arabic_indic":
            answer = to_arabic_indic_digits(answer)

        by_ref = {s["ref"]: s for s in sources}
        citations = []
        for ref in out.citations_used:
            s = by_ref[ref]
            citations.append(Citation(
                ref=ref, law_name_ar=s["law_ar"], law_name_en=s["law_en"], article=s["article"],
                title=s["title_ar"] or s["title"], snippet=(s["text_ar"] or s["text_en"])[:280], lang=s["authoritative_lang"],
                source_url=s["url"], authoritative=True))

        latency = int((time.time() - t0) * 1000)
        self.audit.log({
            "request_id": rid, "query_sha256": sha256(query), "redacted_query": red_q,
            "pii": vault.summary(), "jurisdiction": jur.value, "route_reason": r.reason,
            "concepts": nq.concepts, "retrieved": [h.key for h in hits], "cited": out.citations_used,
            "flags": [f.name for f in flags], "model": self.llm.name, "latency_ms": latency, "blocked": False,
        })
        dbg = {"normalized": nq.debug(), "route": r.reason, "retrieved": [h.key for h in hits],
               "latency_ms": latency} if debug else {}
        return PipelineResult(rid, answer, gist if self.s.get("app.include_english_gist", True) else "",
                              jur.value, citations, flags, redactions=vault.summary(), debug=dbg)


def get_pipeline(config: str | Path | None = None) -> RAGPipeline:
    return RAGPipeline(load_settings(config))
