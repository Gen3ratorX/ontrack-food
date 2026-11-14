import json

import pytest

from ontrack.classifier.gate import SafetyGate
from ontrack.classifier.rules import rule_label
from ontrack.config import CONFIG


@pytest.fixture(scope="module")
def gate():
    from ontrack.classifier.baseline import load_jsonl, make_model

    rows = load_jsonl(CONFIG.triage_data)
    model = make_model().fit([r["text"] for r in rows], [r["label"] for r in rows])
    return SafetyGate(model=model)


def test_seed_data_is_marked_synthetic_and_balanced():
    rows = [json.loads(l) for l in open(CONFIG.triage_data, encoding="utf-8")]
    assert len(rows) >= 150
    assert all(r["synthetic"] is True for r in rows)
    labels = {r["label"] for r in rows}
    assert labels == {"general_education", "needs_clinician", "emergency"}


@pytest.mark.parametrize("text", [
    "My sugar is 2.1 and I am shaking",
    "He has fainted and is not responding",
    "my breath smells fruity and I'm vomiting",
    "she is having a seizure",
    "the meter says LO",
])
def test_rules_catch_emergencies(text):
    assert rule_label(text)[0] == "emergency"


@pytest.mark.parametrize("text,label", [
    ("How many units of insulin should I take for jollof?", "needs_clinician"),
    ("Can I stop my metformin?", "needs_clinician"),
    ("Is kenkey better than banku for my sugar?", "general_education"),
    ("What is the plate method?", "general_education"),
    ("I collapsed and my sugar was 2.0", "emergency"),
])
def test_gate_examples(gate, text, label):
    assert gate.classify(text)[0] == label


def test_gate_never_downgrades_emergency_rule(gate):
    assert gate.classify("my child is unconscious, should I give insulin?")[0] == "emergency"


# Reported as happening now. Rules alone must call these emergency; the TF-IDF model is not consulted.
@pytest.mark.parametrize("text", [
    "shaking, sweating, confused",
    "dizzy and sweating",
    "trembling",
    "confusion",
    "fainting",
    "he passed out",
    "blurred vision with sweating",
    "can't wake him",
    "she is having a seizure",
    "he is having fits",
    "my body is shaking and I'm sweating",
    "feeling weak and confused",
    "Chale, my body is shaking and I am sweating",
])
def test_hypo_red_flags_are_emergency_by_rules_alone(text):
    label, why = SafetyGate().classify(text, use_model=False)
    assert label == "emergency", why
    assert why.startswith("rule:")


@pytest.mark.parametrize("text", [
    "I sweat after exercise",
    "I sweat a lot when I walk in the Accra sun",
    "I'm confused about the plate method",
    "Which meal fits a diabetic plate?",
    "My eyes are blurry, is it diabetic retinopathy?",
    "I feel dizzy if I stand up too fast",
    "I am shaking the bottle of sobolo",
    "What are the signs of low blood sugar?",
])
def test_benign_mentions_are_not_emergency_by_rules_alone(text):
    label, why = SafetyGate().classify(text, use_model=False)
    assert label == "general_education", why


class _FixedProba:
    """Stand-in classifier so a test can force an emergency prediction."""

    classes_ = ["general_education", "needs_clinician", "emergency"]

    def __init__(self, proba):
        import numpy as np
        self._p = np.array(proba, dtype=float)

    def predict_proba(self, texts):
        import numpy as np
        return np.tile(self._p, (len(texts), 1))


@pytest.mark.parametrize("text", [
    "Why does my sugar go up after eating rice?",
    "Why does my blood sugar rise after eating?",
    "Does my sugar go up after I eat fufu?",
    "My sugar goes up after eating kenkey",
])
def test_post_meal_sugar_questions_stay_general_even_if_model_says_emergency(text):
    from ontrack.classifier.rules import rule_label
    assert rule_label(text)[0] != "emergency"
    gate = SafetyGate(model=_FixedProba([0.05, 0.10, 0.85]))
    label, why = gate.classify(text)
    assert label == "general_education", why
    assert why == "rule:post_meal_sugar_rise"


@pytest.mark.parametrize("text", [
    "I'm vomiting, my breath smells fruity and I'm breathing fast",
    "My sugar reading is 30 mmol/L and I feel drowsy",
    "shaking, sweating, confused",
    "Why does my sugar go up after eating rice? I can't wake him",
])
def test_real_emergencies_still_override_a_post_meal_wording(text):
    gate = SafetyGate(model=_FixedProba([0.9, 0.05, 0.05]))
    assert gate.classify(text)[0] == "emergency"


def test_portion_without_a_scale_is_not_a_clinician_rule():
    """qa1-046 asks how to measure food by hand. No red-flag or clinician rule matches.
    The TF-IDF gate can still call it needs_clinician because 'how do' is shared with
    dose questions and the margin is a few thousandths. The label stays general_education."""
    from ontrack.classifier.rules import rule_label
    import json
    from ontrack.config import CONFIG

    question = "How do I measure food portions without a scale?"
    assert rule_label(question)[0] is None
    rows = [json.loads(l) for l in open(CONFIG.qa_data, encoding="utf-8") if l.strip()]
    match = next(r for r in rows if r["id"] == "qa1-046")
    assert match["question"] == question
    assert match["expected_triage"] == "general_education"


def test_pregnancy_insulin_question_is_clinician_by_rules_not_an_emergency_rule():
    """Dose question. The rule is needs_clinician. A model may still escalate to emergency;
    that is the safer-label policy, not a red-flag bug."""
    from ontrack.classifier.rules import rule_label
    text = "I am pregnant and diabetic, how much insulin do I need now?"
    assert rule_label(text)[0] == "needs_clinician"
    assert SafetyGate().classify(text, use_model=False)[0] == "needs_clinician"
    assert SafetyGate(model=_FixedProba([0.05, 0.15, 0.80])).classify(text)[0] == "emergency"


def test_assistant_uses_template_when_gated(gate):
    from ontrack.pipeline import OnTrackAssistant
    from ontrack.rag.generator import EchoGenerator

    a = OnTrackAssistant(EchoGenerator(), None, gate, use_rag=False)
    r = a.answer("My father fainted and is not responding")
    assert r.mode == "template" and "112" in r.answer
    r = a.answer("How many units of insulin for fufu?")
    assert r.mode == "template" and "cannot diagnose or change any medicine" in r.answer
