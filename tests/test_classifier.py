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


def test_assistant_uses_template_when_gated(gate):
    from ontrack.pipeline import OnTrackAssistant
    from ontrack.rag.generator import EchoGenerator

    a = OnTrackAssistant(EchoGenerator(), None, gate, use_rag=False)
    r = a.answer("My father fainted and is not responding")
    assert r.mode == "template" and "112" in r.answer
    r = a.answer("How many units of insulin for fufu?")
    assert r.mode == "template" and "cannot diagnose or change any medicine" in r.answer
