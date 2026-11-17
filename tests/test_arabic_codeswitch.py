from uae_labour_rag.config import ROOT
from uae_labour_rag.text.arabic import arabic_ratio, normalize_for_search, to_western_digits
from uae_labour_rag.text.codeswitch import Glossary, normalize_query, segment

G = Glossary.load(ROOT / "config/glossary.yaml")


def test_digits_and_folding():
    assert to_western_digits("٣٠ يوماً") == "30 يوماً"
    assert normalize_for_search("مكافأة") == normalize_for_search("مكافاة")
    assert normalize_for_search("الإجازة") == normalize_for_search("الاجازة")
    assert normalize_for_search("العَامِل") == "العامل"


def test_segmentation_mixed_scripts():
    segs = segment("notice period فترة الإنذار for DIFC ٣٠")
    assert [s.script for s in segs] == ["latin", "arabic", "latin", "numeric"]


def test_arabic_in_english_query_links_concept():
    nq = normalize_query("how many days مكافأة نهاية الخدمة after 3 years?", G)
    assert "end_of_service_gratuity" in nq.concepts
    assert nq.is_code_switched
    assert "gratuity" in nq.query_en           # AR concept surfaced in EN query
    assert "مكافأة نهاية الخدمة" in nq.query_ar


def test_arabizi():
    nq = normalize_query("how many days ijaza do I get", G)
    assert "annual_leave" in nq.concepts
    assert "ijaza" in nq.arabizi_tokens
    assert "ijaza" not in nq.query_en            # noise removed from EN retrieval query
    assert "الإجازة السنوية" in nq.query_ar


def test_pure_english_gets_arabic_expansion():
    nq = normalize_query("What is the probation period?", G)
    assert nq.concepts == ["probation"]
    assert "فترة التجربة" in nq.query_ar
    assert not nq.is_code_switched


def test_arabic_ratio():
    assert arabic_ratio("مرحبا") == 1.0
    assert arabic_ratio("hello") == 0.0
