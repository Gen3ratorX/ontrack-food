"""Evaluation harness: run the assistant with and without RAG on the held-out test set, score, export.

Examples
  # CPU smoke test (no model download):
  python -m ontrack.eval.harness --generator echo --judge heuristic --limit 10
  # Colab T4, full run:
  python -m ontrack.eval.harness --generator medgemma --judge llm --modes rag no_rag
  # After humans fill in the CSV:
  python -m ontrack.eval.harness --score-human artifacts/runs/<run>/human_scoring.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

from ..config import CONFIG
from .judge import HeuristicJudge, LLMJudge, safety_flags
from .rubric import CRITERIA, aggregate

HUMAN_COLS = ["row_id", "item_id", "question", "reference_answer", "culture_tag", "expected_triage", "response",
              "auto_flags", "accuracy", "relevance", "cultural_relevance", "safety", "clarity", "critical_unsafe", "reviewer", "notes"]


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def build_assistant(mode: str, generator, retriever, gate):
    from ..pipeline import OnTrackAssistant

    return OnTrackAssistant(generator, retriever if mode == "rag" else None, gate, use_rag=(mode == "rag"))


def run(items, modes, generator, retriever, gate, judge=None, out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir or CONFIG.runs_dir / time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in modes:
        asst = build_assistant(mode, generator, retriever, gate)
        for i, it in enumerate(items, 1):
            t0 = time.time()
            r = asst.answer(it["question"])
            row = {"item_id": it["id"], "mode": mode, "response_mode": r.mode, "triage_label": r.triage_label,
                   "expected_triage": it["expected_triage"], "culture_tag": it["culture_tag"],
                   "question": it["question"], "reference_answer": it["reference_answer"], "response": r.answer,
                   "sources": r.sources, "latency_s": round(time.time() - t0, 2),
                   "auto_flags": safety_flags(it, r.answer)}
            if judge is not None:
                row.update(judge.score(it, r.answer))
            rows.append(row)
            print(f"[{mode} {i}/{len(items)}] {it['id']} triage={r.triage_label} flags={row['auto_flags']}")
    with open(out_dir / "responses.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    export_human_csv(rows, out_dir)
    summary = {"n_items": len(items), "modes": modes, "generator": getattr(generator, "model_id", "?"),
               "gate": gate is not None, "judge": getattr(judge, "name", None),
               "triage_agreement": {m: sum(r["triage_label"] == r["expected_triage"] for r in rows if r["mode"] == m) / max(1, len(items)) for m in modes},
               "auto_flag_counts": {m: sum(bool(r["auto_flags"]) for r in rows if r["mode"] == m) for m in modes}}
    if judge is not None:
        summary["judge_scores"] = aggregate(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("outputs in", out_dir)
    return summary


def export_human_csv(rows, out_dir: Path, seed: int = 13):
    """Blinded CSV for human/clinician scoring (mode hidden, order shuffled); key saved separately."""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    key = {}
    with open(out_dir / "human_scoring.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HUMAN_COLS)
        w.writeheader()
        for n, idx in enumerate(order, 1):
            r = rows[idx]
            rid = f"R{n:04d}"
            key[rid] = {"item_id": r["item_id"], "mode": r["mode"]}
            w.writerow({"row_id": rid, "item_id": r["item_id"], "question": r["question"], "reference_answer": r["reference_answer"],
                        "culture_tag": r["culture_tag"], "expected_triage": r["expected_triage"], "response": r["response"],
                        "auto_flags": ";".join(r["auto_flags"])})
    (out_dir / "human_scoring_key.json").write_text(json.dumps(key, indent=2))


def score_human(csv_path: Path) -> dict:
    key = json.loads((csv_path.parent / "human_scoring_key.json").read_text())
    rows = []
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        if not r.get("accuracy"):
            continue
        rec = {"mode": key[r["row_id"]]["mode"], "culture_tag": r["culture_tag"],
               "critical_unsafe": str(r.get("critical_unsafe", "")).strip().lower() in ("y", "yes", "true", "1")}
        for c in CRITERIA:
            rec[c] = float(r[c]) if r.get(c) not in (None, "", "NA", "N/A") else None
        rows.append(rec)
    res = aggregate(rows)
    (csv_path.parent / "human_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--modes", nargs="+", default=["rag", "no_rag"], choices=["rag", "no_rag"])
    ap.add_argument("--generator", default="medgemma", choices=["medgemma", "echo"])
    ap.add_argument("--judge", default="none", choices=["none", "llm", "heuristic"])
    ap.add_argument("--judge-model", default=None, help="HF model id for the judge (default: reuse the generator)")
    ap.add_argument("--no-gate", action="store_true", help="evaluate the raw model without the triage gate")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--score-human", default=None, help="path to a completed human_scoring.csv")
    a = ap.parse_args()

    if a.score_human:
        score_human(Path(a.score_human)); return

    items = load_jsonl(CONFIG.splits_dir / f"{a.split}.jsonl")[: a.limit]
    from ..classifier.gate import SafetyGate
    from ..rag.generator import EchoGenerator, MedGemmaGenerator
    from ..rag.retriever import Retriever

    gen = EchoGenerator() if a.generator == "echo" else MedGemmaGenerator(CONFIG.model_id, CONFIG.load_in_4bit, CONFIG.max_new_tokens)
    retriever = Retriever.from_config() if "rag" in a.modes else None
    gate = None if a.no_gate else SafetyGate.from_config()
    judge = None
    if a.judge == "heuristic":
        judge = HeuristicJudge()
    elif a.judge == "llm":
        jgen = gen if not a.judge_model else MedGemmaGenerator(a.judge_model, CONFIG.load_in_4bit, 200)
        judge = LLMJudge(jgen)
    run(items, a.modes, gen, retriever, gate, judge, Path(a.out) if a.out else None)


if __name__ == "__main__":
    main()
