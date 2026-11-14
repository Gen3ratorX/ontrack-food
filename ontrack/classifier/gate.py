"""Safety gate in front of the assistant.

Order: (1) emergency rules -> emergency; (2) post-meal "why does my sugar go up after eating"
questions stay general_education (a model must not turn that into an emergency);
(3) otherwise the ML model; (4) clinician rules can escalate a general_education
prediction to needs_clinician. Escalation only ever moves toward the safer label.
"""
from __future__ import annotations

from pathlib import Path

from ..config import CONFIG
from .rules import post_meal_sugar_question, rule_label

SEVERITY = {"general_education": 0, "needs_clinician": 1, "emergency": 2}


class SafetyGate:
    def __init__(self, model=None, model_path: str | Path | None = None, emergency_threshold: float = 0.45):
        if model is None and model_path is not None and Path(model_path).exists():
            import joblib

            model = joblib.load(model_path)
        self.model = model
        self.emergency_threshold = emergency_threshold

    @classmethod
    def from_config(cls, cfg=CONFIG) -> "SafetyGate":
        return cls(model_path=cfg.classifier_path)

    def classify(self, text: str, use_model: bool = True) -> tuple[str, str]:
        rlab, rwhy = rule_label(text)
        if rlab == "emergency":
            return "emergency", rwhy
        label, why = "general_education", "default"
        # "Why does my sugar go up after eating rice?" is education. A weak model
        # confuses that wording with a high-sugar emergency; do not let it.
        if post_meal_sugar_question(text):
            label, why = "general_education", "rule:post_meal_sugar_rise"
        elif use_model and self.model is not None:
            proba = self.model.predict_proba([text])[0]
            classes = list(self.model.classes_)
            p = dict(zip(classes, proba))
            label = classes[int(proba.argmax())]
            why = f"model p={p[label]:.2f}"
            if p.get("emergency", 0) >= self.emergency_threshold and label != "emergency":
                label, why = "emergency", f"model emergency p={p['emergency']:.2f} >= {self.emergency_threshold}"
        if rlab and SEVERITY[rlab] > SEVERITY[label]:
            label, why = rlab, rwhy
        return label, why
