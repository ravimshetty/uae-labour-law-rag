import json

from uae_labour_rag.privacy.audit import AuditLogger
from uae_labour_rag.privacy.pii import contains_pii, redact


def test_uae_pii_redaction_roundtrip():
    q = ("My Emirates ID is 784-1990-1234567-1, call +971 50 123 4567 or mail a.b@x.ae. "
         "IBAN AE070331234567890123456, passport N1234567. My name is Ravi Kumar")
    red, vault = redact(q)
    for raw in ["784-1990-1234567-1", "+971 50 123 4567", "a.b@x.ae", "AE070331234567890123456", "N1234567", "Ravi Kumar"]:
        assert raw not in red
    assert {"EMIRATES_ID", "PHONE", "EMAIL", "IBAN", "PASSPORT", "PERSON"} <= set(vault.counts)
    assert vault.rehydrate(red) == q


def test_arabic_indic_digits_emirates_id():
    red, vault = redact("رقم الهوية ٧٨٤-١٩٩٠-١٢٣٤٥٦٧-١")
    assert "<EMIRATES_ID_1>" in red


def test_salary_and_years_not_redacted():
    red, vault = redact("salary 12000 AED after 7 years, notice 30 days")
    assert red == "salary 12000 AED after 7 years, notice 30 days"
    assert not vault.counts


def test_card_needs_luhn():
    assert "CARD" in contains_pii("4111 1111 1111 1111")
    assert "CARD" not in contains_pii("1234 5678 9012 3456")


def test_audit_never_stores_raw(tmp_path):
    a = AuditLogger(tmp_path / "a.jsonl", store_raw=False)
    a.log({"request_id": "x", "raw_query": "secret 784-1990-1234567-1", "redacted_query": "<EMIRATES_ID_1>"})
    rec = json.loads((tmp_path / "a.jsonl").read_text())
    assert "raw_query" not in rec and "784" not in json.dumps(rec)
    assert a.purge_older_than(-1) == 1   # retention purge works
