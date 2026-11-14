"""Pull diabetes-related items from AfriMed-QA v2 (gated: accept terms on HF and set HF_TOKEN).

python -m ontrack.eval.afrimedqa --out data/eval/afrimedqa_diabetes_candidates.jsonl
Outputs CANDIDATES for manual curation (not auto-added to the test set).
"""
from __future__ import annotations

import argparse
import json
import re

KEYWORDS = re.compile(r"diabet|glucose|insulin|hba1c|glycaem|glycem|hypoglyc|hyperglyc|ketoacidosis|metformin", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval/afrimedqa_diabetes_candidates.jsonl")
    ap.add_argument("--types", nargs="+", default=["Consumer Queries", "SAQ"], help="question_type values to keep")
    a = ap.parse_args()
    from datasets import load_dataset

    dsd = load_dataset("intronhealth/afrimedqa_v2")  # DatasetDict; iterate every split it contains
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for r in (row for name in dsd for row in dsd[name]):
            q = r.get("question_clean") or r.get("question") or ""
            if not KEYWORDS.search(q + " " + str(r.get("prompt") or "")):
                continue
            if a.types and not any(t.lower() in str(r.get("question_type", "")).lower() for t in a.types):
                continue
            f.write(json.dumps({
                "afrimedqa_sample_id": r.get("sample_id"), "question_type": r.get("question_type"),
                "prompt": r.get("prompt"), "question": q, "answer_rationale": r.get("answer_rationale"),
                "specialty": r.get("specialty"), "country": r.get("country"), "afrimedqa_split": r.get("split"),
                "licence": "CC BY-SA 4.0 (AfriMed-QA v2)", "synthetic": False,
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} candidate items to {a.out}")


if __name__ == "__main__":
    main()
