"""High-recall keyword/regex safety net. Runs before (and overrides) the ML classifier.

The rules are intentionally conservative: false alarms cost a template message; misses can cost lives.
"""
from __future__ import annotations

import re

EMERGENCY_PATTERNS = [
    r"\b(faint(ed|ing)?|pass(ed)? out|passing out|collaps(e|ed|ing)|unconscious|not (responding|answering|waking)|difficult to wake|won'?t wake)\b",
    r"\b(seizure|fitting|convuls\w*)\b",
    r"\bfruity\b",
    r"\b(can'?t|cannot|unable to) (swallow|keep (any )?(water|fluids?|food|anything) down)\b",
    r"\b(chest pain|chest is tight|left arm pain|face is drooping|can'?t speak|one side of my body|numbness on one side)\b",
    r"\b(large|high) ketones\b|\bketones (are )?(high|large)\b",
    r"\b(deep|fast|heavy|heavily) (fast )?breath\w*|\bbreathing (fast|heavily|deep)\b",
    r"\b(too much|double|extra) insulin by mistake\b|\binjected too much\b",
    r"\bmeter (says|shows|reads) ['\"]?(hi|lo)\b|\b(says|reads?) ['\"]?(HI|LO)['\"]?\b",
    r"\b(pus|black|spreading)\b.*\b(foot|leg|wound|sore)\b|\b(foot|leg|wound|sore)\b.*\b(pus|black|spreading)\b",
]

# blood glucose value with mmol/L context: <3.0 (level 2) or >= 25 treated as emergency
_MMOL = re.compile(r"\b(?:sugar|glucose|reading|level|is|was|at|of|dropped to)\D{0,15}(\d{1,2}(?:\.\d)?)\s*(mmol)?", re.I)

CLINICIAN_PATTERNS = [
    r"\b(insulin|metformin|glibenclamide|gliclazide|mixtard|actrapid|ozempic|semaglutide|empagliflozin|dapagliflozin|sitagliptin|dexamethasone|tablets?|medicine|medication|drugs?|dose|doses|dosage|units?)\b",
    r"\bdo i have\b|\bam i diabetic\b|\bconfirm\b.*\bdiabet|\bdiagnos\w*",
    r"\b(lab report|creatinine|my result|test showed)\b",
    r"\b(herbal (mixture|medicine)|what treatment|prescri\w+|chemical shop)\b",
]


def rule_label(text: str) -> tuple[str | None, str]:
    t = text.lower()
    for p in EMERGENCY_PATTERNS:
        if re.search(p, t, flags=re.I):
            return "emergency", f"rule:{p[:40]}"
    for m in _MMOL.finditer(text):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if v < 3.0 and ("low" in t or "sugar" in t or "glucose" in t) or (v >= 25 and v < 60 and ("sugar" in t or "reading" in t or "glucose" in t)):
            return "emergency", f"rule:glucose_value={v}"
    for p in CLINICIAN_PATTERNS:
        if re.search(p, t, flags=re.I):
            return "needs_clinician", f"rule:{p[:40]}"
    return None, "no rule matched"
