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


def test_assistant_uses_template_when_gated(gate):
    from ontrack.pipeline import OnTrackAssistant
    from ontrack.rag.generator import EchoGenerator

    a = OnTrackAssistant(EchoGenerator(), None, gate, use_rag=False)
    r = a.answer("My father fainted and is not responding")
    assert r.mode == "template" and "112" in r.answer
    r = a.answer("How many units of insulin for fufu?")
    assert r.mode == "template" and "cannot diagnose or change any medicine" in r.answer
