"""Stratified dev/test split that holds out a fixed number of items (default 60) for final evaluation.

python -m ontrack.eval.split --n-test 60
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from ..config import CONFIG


def strat_key(r: dict) -> str:
    return f"{r['expected_triage']}|{r['culture_tag'] if r['culture_tag'] != 'west_african' else 'ghanaian'}"


def split(rows: list[dict], n_test: int = 60, seed: int = 42) -> tuple[list[dict], list[dict]]:
    if n_test >= len(rows):
        raise ValueError(f"n_test={n_test} must be smaller than dataset size {len(rows)}")
    keys = [strat_key(r) for r in rows]
    counts = Counter(keys)
    keys = [k if counts[k] >= 2 else k.split("|")[0] for k in keys]  # fall back to triage label only
    counts = Counter(keys)
    keys = [k if counts[k] >= 2 else "rare" for k in keys]
    if Counter(keys).get("rare", 2) < 2:
        keys = [k if k != "rare" else max(counts, key=counts.get) for k in keys]
    dev, test = train_test_split(rows, test_size=n_test, stratify=keys, random_state=seed)
    return dev, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=[str(CONFIG.qa_data)], help="one or more JSONL files (batches)")
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(CONFIG.splits_dir))
    a = ap.parse_args()
    rows = [json.loads(l) for p in a.data for l in open(p, encoding="utf-8") if l.strip()]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids across batches"
    dev, test = split(rows, a.n_test, a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, part in (("dev", dev), ("test", test)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in sorted(part, key=lambda r: r["id"]):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"total={len(rows)} dev={len(dev)} test={len(test)}")
    print("test by triage:", Counter(r["expected_triage"] for r in test))
    print("test by culture:", Counter(r["culture_tag"] for r in test))


if __name__ == "__main__":
    main()
