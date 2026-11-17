from uae_labour_rag.ingest.chunker import split_articles
from uae_labour_rag.models import Jurisdiction
from uae_labour_rag.retrieval.router import route


def test_article_split_removes_toc_duplicates():
    text = "Contents\nArticle (1) Definitions 10\nArticle (2) Objectives 13\n\nArticle (1) Definitions\n" + \
           "In this law the following words have the meanings stated, unless the context requires otherwise.\n" + \
           "Article (2) Objectives\nThis decree-law aims to regulate the employment relationship fairly.\n"
    arts = split_articles(text, r"^\s*Article\s*\(\s*(\d+)\s*\)")
    assert [a.number for a in arts] == ["1", "2"]
    assert arts[0].title == "Definitions" and "following words" in arts[0].body


def test_arabic_article_numbers_with_indic_digits():
    text = "المادة (٥١) مكافأة نهاية الخدمة\nيستحق العامل الأجنبي مكافأة نهاية الخدمة عند انتهاء خدمته وفق الأحكام التالية."
    arts = split_articles(text, r"^\s*(?:ال)?مادة\s*\(\s*([0-9٠-٩]+)\s*\)")
    assert arts[0].number == "51"


def test_kb_pairs_en_ar(kb):
    pair = kb.pairs["fdl-33-2021:art51"]
    assert set(pair) == {"en", "ar"} and pair["ar"].authoritative


def test_router():
    assert route("DIFC vacation leave").jurisdiction == Jurisdiction.DIFC
    assert route("MOHRE complaint for unpaid salary").jurisdiction == Jurisdiction.FEDERAL
    assert route("annual leave days").jurisdiction == Jurisdiction.BOTH
    assert route("ADGM notice period").out_of_scope == "adgm"


def test_cross_lingual_retrieval(pipe):
    from uae_labour_rag.text.codeswitch import normalize_query
    for q, expected in [("gratuity after 7 years mainland", "fdl-33-2021:art51"),
                        ("كم مدة فترة التجربة", "fdl-33-2021:art9"),
                        ("ijaza days DIFC", "difc-2-2019:art27")]:
        nq = normalize_query(q, pipe.glossary)
        hits = pipe.retriever.retrieve(nq.query_en, nq.query_ar, route(q).jurisdiction, k=3)
        assert expected in [h.key for h in hits], (q, [h.key for h in hits])
