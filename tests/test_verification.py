from uae_labour_rag.verification import build_report

SRC = {"S1": {"law_en": "FDL 33/2021", "article": "29", "title": "Annual Leave", "text_ar": "",
              "text_en": "The worker is entitled to paid annual leave of not less than 30 days for each year of service."}}


def test_correct_sentence_verified(pipe):
    rep = build_report("يستحق العامل إجازة سنوية لا تقل عن 30 يوماً [S1]", SRC, pipe.glossary, pipe.llm)
    assert rep["sentences"][0]["verdict"] == "verified"
    assert rep["sentences"][0]["numbers"] == {"30": True}


def test_hallucinated_number_flagged(pipe):
    rep = build_report("يستحق العامل إجازة سنوية مدتها 45 يوماً [S1]", SRC, pipe.glossary, pipe.llm)
    assert rep["sentences"][0]["verdict"] == "unsupported"


def test_topic_mismatch_flagged(pipe):
    rep = build_report("يستحق العامل مكافأة نهاية الخدمة [S1]", SRC, pipe.glossary, pipe.llm)
    s = rep["sentences"][0]
    assert s["verdict"] == "review" and not s["concepts_match"]


def test_pipeline_verify_end_to_end(pipe):
    r = pipe.answer("sick leave كم يوم بأجر كامل in mainland?", verify=True)
    assert r.verification["summary"]["unsupported"] == 0
    assert r.verification["summary"]["verified"] >= 1
    assert pipe.answer("annual leave days").verification is None   # opt-in only
