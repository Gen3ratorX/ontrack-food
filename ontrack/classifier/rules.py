"""High-recall keyword/regex safety net. Runs before (and overrides) the ML classifier.

The rules are intentionally conservative: false alarms cost a template message; misses can cost lives.
"""
from __future__ import annotations

import re

from ..glucose import emergency_reason

EMERGENCY_PATTERNS = [
    r"\b(faint(ed|ing)?|pass(ed)? out|passing out|collaps(e|ed|ing)|unconscious|not (responding|answering|waking)|difficult to wake|won'?t wake)\b",
    # "can't wake him" / "unable to wake her" — not covered by "won't wake" or "difficult to wake"
    r"\b(can'?t|cannot|cant|couldn'?t|couldnt|unable to|wont) wake\b",
    r"\b(seizures?|fitting|convuls\w*)\b|\b(having|had|has|getting|got) (a |some )?fits?\b|\bfits\b(?!\s+(?:a|an|the|my|your|his|her|their|this|that|into|in\b|well|ok|okay))",
    # Neuroglycopenic signs. "confused about/whether/how..." is a question, not a symptom report.
    r"\b(trembl(?:e|es|ed|ing)|tremors?)\b",
    r"\bconfusion\b|\bconfused\b(?!\s+(?:about|whether|if\b|how\b|what\b|which\b|why\b))",
    r"\bfruity\b",
    r"\b(can'?t|cannot|unable to) (swallow|keep (any )?(water|fluids?|food|anything) down)\b",
    r"\b(chest pain|chest is tight|left arm pain|face is drooping|can'?t speak|one side of my body|numbness on one side)\b",
    r"\b(large|high) ketones\b|\bketones (are )?(high|large)\b",
    r"\b(deep|fast|heavy|heavily) (fast )?breath\w*|\bbreathing (fast|heavily|deep)\b",
    r"\b(too much|double|extra) insulin by mistake\b|\binjected too much\b",
    r"\bmeter (says|shows|reads) ['\"]?(hi|lo)\b|\b(says|reads?) ['\"]?(HI|LO)['\"]?\b",
    r"\b(pus|black|spreading)\b.*\b(foot|leg|wound|sore)\b|\b(foot|leg|wound|sore)\b.*\b(pus|black|spreading)\b",
]

# Milder hypo signs are an emergency together (shaking + sweating, dizzy + sweating,
# blurred vision + sweating, weak + another sign). One of them alone is not — "I sweat
# after exercise", blurry eyes asked about as retinopathy, or dizzy on standing stays quiet.
_HYPO_SIGNS = (
    ("shaking", re.compile(r"\bshak(?:e|es|ing|y)\b", re.I)),
    ("sweating", re.compile(r"\bsweat(?:s|ing|y|ed)?\b", re.I)),
    ("dizzy", re.compile(r"\bdizz(?:y|iness)\b|\blight[-\s]?headed\b", re.I)),
    ("blurred_vision", re.compile(
        r"\bblur(?:red|ry)?\s+vision\b"
        r"|\bvision\b(?:\s+\w+){0,4}\s+(?:blur(?:red|ry)?|double)\b"
        r"|\bblur(?:red|ry)?\s+eyes?\b"
        r"|\beyes?\b(?:\s+\w+){0,4}\s+blur(?:red|ry)?\b"
        r"|\b(?:trouble|difficulty)\s+seeing\b"
        r"|\bcan(?:not|'?t|t)\s+see\s+(?:well|clearly|properly)\b",
        re.I,
    )),
    ("weak", re.compile(r"\bweak(?:ness)?\b", re.I)),
)


def _hypo_cluster(text: str) -> str | None:
    hit = [name for name, cre in _HYPO_SIGNS if cre.search(text)]
    if len(hit) >= 2:
        return "rule:hypo_cluster:" + "+".join(hit)
    return None


CLINICIAN_PATTERNS = [
    r"\b(insulin|metformin|glibenclamide|gliclazide|mixtard|actrapid|ozempic|semaglutide|empagliflozin|dapagliflozin|sitagliptin|dexamethasone|tablets?|medicine|medication|drugs?|dose|doses|dosage|units?)\b",
    r"\bdo i have\b|\bam i diabetic\b|\bconfirm\b.*\bdiabet|\bdiagnos\w*",
    r"\b(lab report|creatinine|my result|test showed)\b",
    r"\b(herbal (mixture|medicine)|what treatment|prescri\w+|chemical shop)\b",
]


# Ordinary questions about sugar rising after a meal are education, not a crisis.
# Emergency rules (very high readings, vomiting, fruity breath, confusion, etc.) are checked first.
_POST_MEAL_SUGAR = re.compile(
    r"\b(?:why|how come|what makes|does|do|can|is it normal)\b.{0,80}\b(?:blood\s+)?(?:sugar|glucose)\b.{0,50}\b(?:go(?:es|ing)?\s+up|rise[sd]?|rising|increase[sd]?|spike[sd]?)\b"
    r"|"
    r"\b(?:blood\s+)?(?:sugar|glucose)\b.{0,40}\b(?:go(?:es|ing)?\s+up|rise[sd]?|rising|increase[sd]?|spike[sd]?)\b.{0,50}\b(?:after|when)\b.{0,40}\b(?:eat(?:ing)?|food|a meal|meal|rice|kenkey|fufu|banku|jollof)\b",
    re.I | re.S,
)


def post_meal_sugar_question(text: str) -> bool:
    return _POST_MEAL_SUGAR.search(text) is not None


def rule_label(text: str) -> tuple[str | None, str]:
    t = text.lower()
    for p in EMERGENCY_PATTERNS:
        if re.search(p, t, flags=re.I):
            return "emergency", f"rule:{p[:40]}"
    cluster = _hypo_cluster(t)
    if cluster:
        return "emergency", cluster
    # Severe low <3.0 mmol/L (54 mg/dL) or very high >=16.7 mmol/L (300 mg/dL).
    why = emergency_reason(text)
    if why:
        return "emergency", why
    for p in CLINICIAN_PATTERNS:
        if re.search(p, t, flags=re.I):
            return "needs_clinician", f"rule:{p[:40]}"
    return None, "no rule matched"
