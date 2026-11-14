"""Interactive / one-shot CLI.

python -m ontrack.cli "Is kenkey good for diabetes?" --mode rag --generator medgemma
python -m ontrack.cli --retrieve-only "fruity breath vomiting"
python -m ontrack.cli --classify-meal "2 balls of kenkey with fried fish and shito"   (meal classifier only, no model)
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--mode", default="rag", choices=["rag", "no_rag"])
    ap.add_argument("--generator", default="medgemma", choices=["medgemma", "echo"])
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--retrieve-only", action="store_true")
    ap.add_argument("--classify-meal", action="store_true", help="run only the meal glycaemic-impact classifier and print JSON")
    ap.add_argument("--no-meal", action="store_true", help="do not add the meal analysis to the RAG context")
    a = ap.parse_args()
    if a.classify_meal:
        from .meal.engine import analyse_meal
        qs = [a.question] if a.question else iter(lambda: input("Meal: "), "")
        for q in qs:
            m = analyse_meal(q)
            print(m.to_context())
            print(json.dumps({"label": m.label, "basis": m.basis, "confidence": m.confidence, "est_glycaemic_load": m.est_glycaemic_load,
                              "total_carb_g": m.total_carb_g, "drivers": m.drivers, "flags": m.flags,
                              "sources": [s["url"] for s in m.sources]}, indent=1, ensure_ascii=False))
        return
    from .classifier.gate import SafetyGate
    from .config import CONFIG
    from .rag.retriever import Retriever

    retriever = Retriever.from_config() if (a.mode == "rag" or a.retrieve_only) else None
    if a.retrieve_only:
        for h in retriever.search(a.question):
            print(f"{h.score:.3f}  {h.chunk_id}  {h.title}")
        return
    from .pipeline import OnTrackAssistant
    from .rag.generator import EchoGenerator, MedGemmaGenerator

    gen = EchoGenerator() if a.generator == "echo" else MedGemmaGenerator(CONFIG.model_id, CONFIG.load_in_4bit, CONFIG.max_new_tokens)
    asst = OnTrackAssistant(gen, retriever, None if a.no_gate else SafetyGate.from_config(), use_rag=a.mode == "rag", classify_meals=not a.no_meal)
    qs = [a.question] if a.question else iter(lambda: input("You: "), "")
    for q in qs:
        r = asst.answer(q)
        meal = f" | meal={r.meal_analysis['label']}" if r.meal_analysis else ""
        print(f"\n[{r.mode} | triage={r.triage_label}{meal}]\n{r.answer}\n")
        if r.sources:
            print(json.dumps(r.sources, indent=1))


if __name__ == "__main__":
    main()
