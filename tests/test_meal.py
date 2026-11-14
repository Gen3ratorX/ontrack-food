import json
import re

import pytest

from ontrack.config import CONFIG, ROOT
from ontrack.meal.engine import MealClassifier, analyse_meal
from ontrack.meal.lexicon import load_lexicon, load_wafct
from ontrack.meal.recognizer import FoodRecognizer
from ontrack.rag.chunking import parse_front_matter

DOCS = ROOT / "knowledge_base" / "docs"


def doc_text(doc_id):
    for p in DOCS.glob("*.md"):
        meta, body = parse_front_matter(p.read_text(encoding="utf-8"))
        if meta.get("id") == doc_id:
            return body
    raise AssertionError(f"doc {doc_id} not found")


@pytest.fixture(scope="module")
def rec():
    return FoodRecognizer()


@pytest.fixture(scope="module")
def mc():
    return MealClassifier()


# ---------------- lexicon integrity: no invented numbers ----------------
def test_every_lexicon_gi_value_appears_in_its_cited_document():
    for fid, e in load_lexicon().items():
        if e["gi"]:
            v = e["gi"]["value"]
            body = doc_text(e["gi"]["source_id"])
            pat = r"(?<![\d.])" + re.escape(f"{v:g}" if v != int(v) else str(int(v))) + r"0*(?![\d])"
            assert re.search(pat, body), f"{fid}: GI {v} not found in {e['gi']['source_id']}"
            assert e["gi"]["type"] in ("measured", "predicted")


def test_wafct_codes_and_evidence_docs_exist():
    wafct = load_wafct()
    for fid, e in load_lexicon().items():
        if e["wafct_code"]:
            assert e["wafct_code"] in wafct, f"{fid}: {e['wafct_code']} missing from WAFCT extract"
        ev = e.get("evidence_doc")
        for d in ([ev] if isinstance(ev, str) else (ev or [])):
            doc_text(d)
        for c in e.get("components", []):
            assert c["id"] in load_lexicon()
        assert "default" in e["portion"]


def test_no_alias_maps_to_two_foods(rec):
    seen = {}
    from ontrack.meal.recognizer import tokenize
    for fid, e in load_lexicon().items():
        for a in e["aliases"]:
            k = tuple(tokenize(a))
            assert seen.get(k, fid) == fid, f"alias '{a}' used by {seen.get(k)} and {fid}"
            seen[k] = fid


# ---------------- recognition ----------------
@pytest.mark.parametrize("text,expected", [
    ("me di dɔkono ne nam", {"kenkey", "fish"}),
    ("komi with shito", {"kenkey", "stew"}),
    ("kenkay and fish", {"kenkey", "fish"}),
    ("waakey with gari", {"waakye", "gari"}),
    ("bayerɛ ampesi with nkontommire", {"yam", "kontomire"}),
    ("bankye ampesi", {"cassava"}),
    ("boiled yellow yam", {"yellow_yam"}),
    ("fried plantain and beans", {"fried_plantain", "beans"}),
    ("kelewele and a minerals", {"fried_plantain", "soft_drink"}),
    ("2 balls of TZ with ayoyo", {"tuo_zaafi", "ayoyo"}),
])
def test_recognises_local_names_and_typos(rec, text, expected):
    assert {m.food_id for m in rec.find(text) if not m.negated} == expected


def test_quantities_negation_and_blood_sugar(rec):
    m = {x.food_id: x for x in rec.find("2 balls of kenkey with fried fish")}
    assert m["kenkey"].quantity == 2 and m["kenkey"].unit == "ball" and m["fish"].unit is None
    assert [x for x in rec.find("koko with no sugar") if x.food_id == "sugar"][0].negated
    assert rec.find("half a ball of banku")[0].quantity == 0.5
    assert rec.find("200g of rice")[0].grams == 200
    assert {x.food_id for x in rec.find("my sugar is high after banku")} == {"banku"}
    assert rec.find("I could not sleep last night") == []


# ---------------- rating ----------------
def test_gl_only_when_cited_gi_and_carb_exist(mc):
    a = mc.analyse("2 balls of kenkey with fried fish and shito")
    k = [i for i in a.items if i.food_id == "kenkey"][0]
    assert k.gi == 41 and k.carb_g is not None and k.est_gl == pytest.approx(41 / 100 * k.carb_g, abs=0.1)
    assert a.label == "high" and "gi_ghanaian_staples_elicophie2016" in [s["id"] for s in a.sources]
    b = mc.analyse("a ball of banku")
    assert b.items[0].est_gl is None and b.items[0].basis == "gi_class"   # no WAFCT carb row -> no GL
    r = mc.analyse("a plate of jollof")
    assert r.items[0].gi is None and r.items[0].basis == "carb_only" and r.est_glycaemic_load is None


def test_evidence_gaps_are_insufficient(mc):
    for t in ["tom brown", "kokonte with groundnut soup", "a bottle of malt", "rice water"]:
        a = mc.analyse(t)
        assert a.label == "insufficient_evidence", t
        assert a.est_glycaemic_load is None


def test_composition_adjustment_and_sugary_drink_block(mc):
    a = mc.analyse("1 small ball of kenkey with grilled fish and garden egg stew")
    assert a.label == "low" and "composition" in a.basis
    b = mc.analyse("a small plate of jollof with salad and chicken and a bottle of coke")
    assert "composition" not in b.basis and any("blocks" in f for f in b.flags)
    c = mc.analyse("grilled fish and salad")
    assert c.label == "low" and c.basis == "composition"


def test_every_output_has_drivers_and_sources(mc):
    rows = [json.loads(l) for l in open(CONFIG.meal_data, encoding="utf-8")]
    lex = load_lexicon()
    for r in rows:
        a = mc.analyse(r["text"])
        assert a.sources, r["id"]
        if a.label in ("medium", "high"):
            assert a.drivers, r["id"]
        for it in a.items:  # GI values only ever come from the lexicon
            assert it.gi is None or it.gi == lex[it.food_id]["gi"]["value"]
            assert it.est_gl is None or (it.gi is not None and it.carb_g is not None)


def test_engine_matches_manual_labels_on_hard_sets(mc):
    rows = [json.loads(l) for l in open(CONFIG.meal_data, encoding="utf-8")]
    hard = [r for r in rows if r["label_source"] == "manual"]
    assert len(hard) >= 40
    wrong = [(r["id"], r["label"], mc.analyse(r["text"]).label) for r in hard if mc.analyse(r["text"]).label != r["label"]]
    assert not wrong, wrong


def test_dataset_is_synthetic_and_sized():
    rows = [json.loads(l) for l in open(CONFIG.meal_data, encoding="utf-8")]
    assert 190 <= len(rows) <= 260 and all(r["synthetic"] for r in rows)
    assert {r["split"] for r in rows} == {"train", "hard", "hard_blind"}


def test_context_block_marks_estimates(mc):
    ctx = mc.analyse("2 balls of kenkey").to_context()
    assert ctx.startswith("[M]") and "estimate" in ctx and "Sources:" in ctx


# ---------------- pipeline integration ----------------
def test_pipeline_adds_meal_analysis_after_gate(retriever):
    from ontrack.classifier.gate import SafetyGate
    from ontrack.pipeline import OnTrackAssistant
    from ontrack.rag.generator import EchoGenerator

    seen = {}

    class Spy(EchoGenerator):
        def generate(self, messages, max_new_tokens=None):
            seen["prompt"] = messages[-1]["content"][0]["text"]
            return super().generate(messages)

    asst = OnTrackAssistant(Spy(), retriever, SafetyGate())
    r = asst.answer("I had 2 balls of kenkey with fried fish and shito for lunch")
    assert r.mode == "rag" and r.meal_analysis["label"] == "high"
    assert "[M] OnTrack meal analysis" in seen["prompt"] and any(s["ref"] == "M" for s in r.sources)
    e = asst.answer("After eating fufu I fainted and my sugar is 2.5")
    assert e.mode == "template" and e.meal_analysis is None
    q = asst.answer("What are the signs of low blood sugar?")
    assert q.meal_analysis is None
    nr = OnTrackAssistant(EchoGenerator(), None, SafetyGate(), use_rag=False).answer("I had 2 balls of kenkey")
    assert nr.mode == "no_rag" and nr.meal_analysis is None


def test_learned_model_trains_and_predicts(tmp_path):
    from ontrack.meal.learned import LearnedMealClassifier, train_and_evaluate
    m = train_and_evaluate(out_path=tmp_path / "meal.joblib")
    assert 0 <= m["cv_lr_vs_engine_labels"]["accuracy"] <= 1
    label, proba = LearnedMealClassifier(tmp_path / "meal.joblib").predict("2 balls of banku with okro")
    assert label in proba
