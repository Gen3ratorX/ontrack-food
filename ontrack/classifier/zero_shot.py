"""Optional zero-shot triage with MedGemma (GPU). Compare against the TF-IDF baseline on the same holdout."""
from __future__ import annotations

import re

from . import LABELS

PROMPT = """Classify the user's message to a diabetes education assistant into exactly one label:
- general_education: questions about food, portions, Ghanaian dishes, exercise, or understanding diabetes in general.
- needs_clinician: anything about medicines, insulin or doses, starting/stopping/changing treatment, interpreting their own test results, or asking for a diagnosis.
- emergency: signs of possible emergency now (very low or very high sugar with symptoms, confusion, fainting, seizure, vomiting and unable to keep fluids down, fruity breath, fast deep breathing, chest pain, stroke signs, spreading infected foot wound with fever).
If unsure between two labels, choose the more serious one.

Message: "{text}"
Answer with only the label."""


def zero_shot_label(generator, text: str) -> str:
    msgs = [{"role": "user", "content": [{"type": "text", "text": PROMPT.format(text=text)}]}]
    out = generator.generate(msgs, max_new_tokens=10).strip().lower()
    for lab in ["emergency", "needs_clinician", "general_education"]:  # most serious first
        if lab in out or lab.replace("_", " ") in out:
            return lab
    m = re.search(r"(emergency|clinician|general)", out)
    return {"emergency": "emergency", "clinician": "needs_clinician", "general": "general_education"}.get(m.group(1) if m else "", "needs_clinician")


def evaluate_zero_shot(generator, rows: list[dict]) -> dict:
    from sklearn.metrics import classification_report

    preds = [zero_shot_label(generator, r["text"]) for r in rows]
    return classification_report([r["label"] for r in rows], preds, labels=LABELS, output_dict=True, zero_division=0)
