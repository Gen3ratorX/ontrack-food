"""Retrieval check: for each Q&A item with source_ids, does the retriever return at least one of them (hit@k),
and what fraction of the cited documents does it return (doc recall@k)?

python -m ontrack.eval.retrieval_eval --k 5
"""
from __future__ import annotations

import argparse
import json

from ..config import CONFIG
from ..rag.retriever import Retriever


def evaluate(retriever: Retriever, items: list[dict], k: int = 5) -> dict:
    items = [it for it in items if it.get("source_ids") and it["expected_triage"] == "general_education"]
    hits, recalls, misses = 0, [], []
    for it in items:
        got = {h.doc_id for h in retriever.search(it["question"], k=k)}
        want = set(it["source_ids"])
        if got & want:
            hits += 1
        else:
            misses.append({"id": it["id"], "question": it["question"], "want": sorted(want), "got": sorted(got)})
        recalls.append(len(got & want) / len(want))
    return {"n": len(items), "k": k, "hit_at_k": hits / max(1, len(items)),
            "doc_recall_at_k": sum(recalls) / max(1, len(recalls)), "misses": misses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dense-only", action="store_true")
    a = ap.parse_args()
    r = Retriever.from_config()
    if a.dense_only:
        r.hybrid = False
    items = [json.loads(l) for l in open(CONFIG.qa_data, encoding="utf-8")]
    res = evaluate(r, items, a.k)
    print(f"{'dense' if a.dense_only else 'hybrid'}: n={res['n']} hit@{a.k}={res['hit_at_k']:.3f} doc_recall@{a.k}={res['doc_recall_at_k']:.3f}")
    for m in res["misses"]:
        print("  miss", m)


if __name__ == "__main__":
    main()
