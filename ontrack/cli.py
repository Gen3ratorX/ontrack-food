"""Interactive / one-shot CLI.

python -m ontrack.cli "Is kenkey good for diabetes?" --mode rag --generator medgemma
python -m ontrack.cli --retrieve-only "fruity breath vomiting"
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
    a = ap.parse_args()
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
    asst = OnTrackAssistant(gen, retriever, None if a.no_gate else SafetyGate.from_config(), use_rag=a.mode == "rag")
    qs = [a.question] if a.question else iter(lambda: input("You: "), "")
    for q in qs:
        r = asst.answer(q)
        print(f"\n[{r.mode} | triage={r.triage_label}]\n{r.answer}\n")
        if r.sources:
            print(json.dumps(r.sources, indent=1))


if __name__ == "__main__":
    main()
