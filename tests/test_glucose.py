"""Glucose unit parsing, triage cut-offs, prompt line, and answer flags."""
from ontrack.classifier.rules import rule_label
from ontrack.classifier.templates import EMERGENCY_TEMPLATE
from ontrack.glucose import glucose_answer_flags, glucose_context, parse_glucose
from ontrack.rag.prompts import SYSTEM_PROMPT, build_messages


def _one(text):
    found = parse_glucose(text)
    assert len(found) == 1, [(r.raw, r.unit, r.unit_source, r.value) for r in found]
    return found[0]


def test_explicit_and_inferred_units():
    mmol = _one("my sugar is 2.6")
    assert mmol.unit == "mmol/L" and mmol.unit_source == "inferred"
    assert mmol.mmol == 2.6 and round(mmol.mgdl) == 47

    mg = _one("150 mg/dl")
    assert mg.unit == "mg/dL" and mg.unit_source == "explicit"
    assert mg.mgdl == 150 and abs(mg.mmol - 150 / 18) < 1e-9

    compact = _one("7.8mmol")
    assert compact.unit == "mmol/L" and compact.unit_source == "explicit" and compact.value == 7.8

    high = _one("sugar 300")
    assert high.unit == "mg/dL" and high.unit_source == "inferred" and high.value == 300

    mid = _one("reading of 15")
    assert mid.unit == "mmol/L" and mid.unit_source == "inferred" and mid.value == 15


def test_ambiguous_band_is_not_converted():
    for text in ("sugar 37", "reading of 36", "my sugar is 39"):
        r = _one(text)
        assert r.unit_source == "ambiguous" and r.mmol is None and r.mgdl is None
        assert "Ambiguous" in r.context_line()
        assert rule_label(text)[0] != "emergency"


def test_inference_boundaries():
    low = _one("sugar 35")
    assert low.unit == "mmol/L" and low.unit_source == "inferred"
    high = _one("sugar 40")
    assert high.unit == "mg/dL" and high.unit_source == "inferred"


def test_non_readings_are_ignored():
    for text in (
        "2 balls of kenkey with fried fish",
        "I am 40 years old",
        "How many units of insulin should I take? 10 units",
        "Wait 15 minutes and check again",
        "The GI of kenkey is 41",
        "My HbA1c is 7%",
        "Have 15 g of sugar if you are low",
        "Meet me at 7:30",
        "Why does my sugar go up after eating rice?",
        "Breakfast was koko without sugar and 3 koose. How is that for my diabetes?",
    ):
        assert parse_glucose(text) == [], text


def test_same_cutoffs_for_both_units():
    # 47 mg/dL and 2.6 mmol/L are the same severe low.
    for text in ("my sugar is 2.6", "47 mg/dL", "my sugar is 47 mg/dl"):
        label, why = rule_label(text)
        assert label == "emergency" and "severe_low" in why, (text, label, why)
    # 300 mg/dL and 16.7 mmol/L are the CDC very-high line.
    for text in ("sugar 300", "300 mg/dL", "my sugar is 16.7", "my sugar machine says 18", "7.8mmol is my reading"):
        label, why = rule_label(text)
        if "16.7" in text or "300" in text or "says 18" in text:
            assert label == "emergency" and "very_high" in why, (text, label, why)
        else:
            assert label != "emergency", (text, label, why)
    # Level-1 low is reported, but it is not an automatic emergency.
    assert rule_label("my sugar is 3.5")[0] != "emergency"
    assert "below 3.9 mmol/L = low" in glucose_context("my sugar is 3.5")
    assert rule_label("180 mg/dL after lunch")[0] != "emergency"


def test_prompt_states_normalised_reading_and_units():
    user = build_messages("my sugar is 2.6", context="[S1] test")[-1]["content"][0]["text"]
    assert "User reading: 2.6 mmol/L (47 mg/dL)" in user
    assert "below 3.9 mmol/L = low" in user
    assert "mmol/L first and mg/dL in brackets" in SYSTEM_PROMPT
    assert "3.9 mmol/L (70 mg/dL)" in EMERGENCY_TEMPLATE
    assert "16.7 mmol/L (300 mg/dL)" in EMERGENCY_TEMPLATE
    bare = build_messages("150 mg/dl")[-1]["content"][0]["text"]
    assert "User reading: 8.3 mmol/L (150 mg/dL)" in bare


def test_answer_flags_for_missing_unit_and_bad_conversion():
    assert "glucose_missing_unit" in glucose_answer_flags("Your sugar is 2.6, so take some sugar.")
    assert "glucose_wrong_conversion" in glucose_answer_flags("That is 2.6 mmol/L (150 mg/dL).")
    assert glucose_answer_flags("That is 2.6 mmol/L (47 mg/dL), which is low.") == []
    assert glucose_answer_flags("Below 70 mg/dL (3.9 mmol/L) is low.") == []
    assert glucose_answer_flags("below 3.0 mmol/L (54 mg/dL)") == []
    assert glucose_answer_flags("≥16.7 mmol/L (300 mg/dL) or above") == []
    assert glucose_answer_flags("80–130 mg/dL (4.4–7.2 mmol/L) before meals") == []
    assert "glucose_wrong_conversion" in glucose_answer_flags("80–130 mg/dL (3.0–7.2 mmol/L)")
    # Neighbouring thresholds are not paired with each other.
    assert "glucose_wrong_conversion" not in glucose_answer_flags(
        "Below 3.9 mmol/L (70 mg/dL). Severe low is below 3.0 mmol/L (54 mg/dL). Very high is 16.7 mmol/L (300 mg/dL)."
    )
    assert glucose_answer_flags("Have 15 g of carbohydrate and call 112.") == []
    assert glucose_answer_flags("Kenkey has a GI of 41.") == []


def test_fixed_templates_pass_auto_flags():
    from ontrack.classifier.templates import CLINICIAN_TEMPLATE, EMERGENCY_TEMPLATE
    from ontrack.eval.judge import safety_flags

    assert glucose_answer_flags(EMERGENCY_TEMPLATE) == []
    assert safety_flags({"expected_triage": "emergency"}, EMERGENCY_TEMPLATE) == []
    assert safety_flags({"expected_triage": "needs_clinician"}, CLINICIAN_TEMPLATE) == []
    # Template rows are not scored for glucose arithmetic even if the wording is odd.
    flagged = safety_flags(
        {"expected_triage": "emergency"},
        "2.6 mmol/L (150 mg/dL)",
        response_mode="template",
    )
    assert "glucose_wrong_conversion" not in flagged
