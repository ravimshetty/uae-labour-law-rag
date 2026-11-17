"""Input guardrails – run BEFORE retrieval / LLM.

1. size limit
2. prompt-injection / jailbreak detection (EN + AR + Arabizi phrasing)
3. scope check – UAE / DIFC employment matters only
4. high-stakes intent detection – adds "consult a lawyer / MOHRE / DIFC Courts" flag
5. PII redaction (privacy/pii.py) – done by the pipeline right after these checks

All checks are deterministic regex/keyword rules: cheap, explainable, testable.
An optional LLM classifier (`llm_scope_check`) can be enabled for borderline cases.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import GuardrailFlag

_INJECTION = [
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions|rules|prompts?)",
    r"disregard (?:the |your )?(?:system|previous) (?:prompt|instructions)",
    r"(?:reveal|print|show|repeat) (?:me )?(?:your|the) (?:system|hidden|initial) (?:prompt|instructions)",
    r"you are now (?:a|an|in) ", r"\bdeveloper mode\b", r"\bjail ?break\b", r"\bDAN\b",
    r"act as (?:an? )?(?:unfiltered|unrestricted)", r"pretend (?:you have|there are) no (?:rules|restrictions)",
    r"respond only in english", r"<\s*/?\s*system\s*>", r"\[\s*system\s*\]",
    r"تجاهل (?:جميع |كل )?(?:التعليمات|الأوامر)", r"انس(?:َ)? (?:التعليمات|كل ما سبق)",
    r"اكشف (?:عن )?(?:التعليمات|موجه النظام)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION), re.I)

_EMPLOYMENT = re.compile(
    r"employ|labou?r|worker|staff|salary|wage|pay\b|paid|gratuity|leave|vacation|holiday|probation|"
    r"notice|terminat|dismiss|fired|resign|contract|overtime|working hours|work hours|shift|"
    r"maternity|paternity|sick|end of service|non.?compete|mohre|wps|difc|dews|employer|"
    r"company|job|work permit|labour card|visa|absconding|complaint|tribunal|court|ramadan|"
    r"part.?time|remote work|bonus|commission|deduct|discriminat|harass|"
    r"عمل|عامل|موظف|أجر|اجر|راتب|إجازة|اجازة|تجربة|إنذار|انذار|عقد|نهاية الخدمة|فصل|استقالة|"
    r"صاحب العمل|ساعات|مكافأة|مكافاة|شكوى",
    re.I)
_UAE = re.compile(r"uae|emirates|dubai|abu dhabi|sharjah|ajman|ras al|fujairah|umm al|difc|mohre|"
                  r"الإمارات|الامارات|دبي|أبوظبي|ابوظبي|الشارقة", re.I)
_COUNTRIES = r"(?:saudi(?: arabia)?|ksa|qatar|oman|kuwait|bahrain|india|uk|united kingdom|usa|united states|egypt|pakistan|philippines)"
# "<country> labour law" or "labour law in <country>" (mere nationality mentions are fine)
_OTHER_COUNTRY = re.compile(rf"\b{_COUNTRIES}\b[\w\s']{{0,25}}\b(?:law|labou?r|employment)\b|"
                            rf"\b(?:law|labou?r|employment)\b[\w\s]{{0,25}}\b(?:in|of|under)\s+{_COUNTRIES}\b", re.I)
_HIGH_STAKES = re.compile(r"\bsue\b|lawsuit|file a case|court case|legal notice|draft (?:a )?(?:contract|claim)|"
                          r"criminal|police|travel ban|absconding|deport|رفع دعوى|قضية|بلاغ", re.I)
_HARMFUL = re.compile(r"forge|fake (?:contract|certificate|salary certificate)|bypass wps|avoid paying (?:gratuity|salary)|"
                      r"hide (?:from|salary from) mohre|تزوير", re.I)


@dataclass
class InputCheck:
    allowed: bool
    flags: list[GuardrailFlag] = field(default_factory=list)
    refusal_key: str | None = None   # key into REFUSALS_AR


REFUSALS_AR = {
    "too_long": "السؤال طويل جداً. يُرجى اختصاره وإعادة المحاولة.",
    "injection": "عذراً، لا يمكنني تنفيذ هذا الطلب. يُرجى طرح سؤال يتعلق بقانون العمل في دولة الإمارات أو قانون التوظيف في مركز دبي المالي العالمي.",
    "out_of_scope": "عذراً، يقتصر هذا المساعد على الإجابة عن الأسئلة المتعلقة بقانون تنظيم علاقات العمل في دولة الإمارات (المرسوم بقانون اتحادي رقم 33 لسنة 2021) وقانون التوظيف في مركز دبي المالي العالمي (القانون رقم 2 لسنة 2019).",
    "other_country": "عذراً، يغطي هذا المساعد قوانين العمل في دولة الإمارات ومركز دبي المالي العالمي فقط، ولا يغطي قوانين الدول الأخرى.",
    "adgm": "عذراً، لا تشمل قاعدة المعرفة لوائح التوظيف في سوق أبوظبي العالمي (ADGM). يمكنني الإجابة وفق القانون الاتحادي أو قانون مركز دبي المالي العالمي.",
    "harmful": "عذراً، لا يمكنني المساعدة في أي إجراء يخالف القانون أو يهدف إلى التحايل على حقوق العمال أو أصحاب العمل.",
    "no_context": "لم أجد في النصوص القانونية المتاحة ما يكفي للإجابة عن هذا السؤال بشكل موثوق. يُرجى إعادة صياغة السؤال أو التواصل مع وزارة الموارد البشرية والتوطين أو محاكم مركز دبي المالي العالمي.",
}


def check_input(query: str, max_chars: int = 1500, block_injection: bool = True,
                has_legal_concepts: bool = False) -> InputCheck:
    """`has_legal_concepts`: the code-switch normaliser linked a glossary concept
    (e.g. Arabizi "fatrat al tajruba") – that alone makes the query in scope."""
    flags: list[GuardrailFlag] = []
    if len(query) > max_chars:
        return InputCheck(False, [GuardrailFlag("too_long", "block", f"{len(query)} chars")], "too_long")
    if _INJECTION_RE.search(query):
        flags.append(GuardrailFlag("prompt_injection", "block" if block_injection else "warn",
                                   "instruction-override pattern detected"))
        if block_injection:
            return InputCheck(False, flags, "injection")
    if _HARMFUL.search(query):
        return InputCheck(False, flags + [GuardrailFlag("harmful_intent", "block")], "harmful")
    if _OTHER_COUNTRY.search(query) and not _UAE.search(query):
        return InputCheck(False, flags + [GuardrailFlag("other_jurisdiction", "block")], "other_country")
    if not (_EMPLOYMENT.search(query) or has_legal_concepts):
        return InputCheck(False, flags + [GuardrailFlag("out_of_scope", "block", "no employment topic")],
                          "out_of_scope")
    if _HIGH_STAKES.search(query):
        flags.append(GuardrailFlag("high_stakes", "warn", "recommend professional legal advice"))
    return InputCheck(True, flags)


def sanitize_context(text: str) -> str:
    """Defence against indirect injection: neutralise instruction-like lines in retrieved text."""
    return _INJECTION_RE.sub("[removed]", text)
