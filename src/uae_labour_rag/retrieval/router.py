"""Jurisdiction router.

Onshore (mainland + most free zones) employment is governed by Federal Decree-Law
33/2021; DIFC has its own Employment Law (DIFC Law 2/2019) and courts. Mixing them
gives wrong answers (e.g. gratuity vs. DEWS/Qualifying Scheme), so we route first.

Rules are explicit and auditable; when nothing matches we fall back to
`router.default_when_ambiguous` (default `both` -> the answer compares the regimes).
ADGM is detected and flagged as out of scope.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import Jurisdiction

_DIFC = re.compile(
    r"\bdifc\b|dubai international financial cent(?:re|er)|\bdews\b|qualifying scheme|"
    r"gate (?:village|avenue)|مركز دبي المالي|محاكم المركز", re.I)
_FEDERAL = re.compile(
    r"\bmohre\b|\buae labou?r law\b|\bmainland\b|\bonshore\b|federal (?:decree|law)|decree[- ]law|\b33 of 2021\b|"
    r"labour court|\bwps\b|\btasheel\b|tawjeeh|\bjafza\b|\bdmcc\b|\bdso\b|"
    r"وزارة الموارد البشرية|المرسوم بقانون|القانون الاتحادي|البر الرئيسي", re.I)
_ADGM = re.compile(r"\badgm\b|abu dhabi global market|سوق أبوظبي العالمي", re.I)


@dataclass
class Route:
    jurisdiction: Jurisdiction
    reason: str
    out_of_scope: str | None = None


def route(query: str, default: str = "both") -> Route:
    difc, fed, adgm = bool(_DIFC.search(query)), bool(_FEDERAL.search(query)), bool(_ADGM.search(query))
    if adgm and not (difc or fed):
        return Route(Jurisdiction.BOTH, "ADGM mentioned", out_of_scope="adgm")
    if difc and fed:
        return Route(Jurisdiction.BOTH, "both regimes mentioned")
    if difc:
        return Route(Jurisdiction.DIFC, "DIFC keyword")
    if fed:
        return Route(Jurisdiction.FEDERAL, "onshore/federal keyword")
    return Route(Jurisdiction(default if default != "ask" else "both"), "no jurisdiction signal – default")
