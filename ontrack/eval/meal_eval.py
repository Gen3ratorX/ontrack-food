"""Evaluate the rule-based meal classifier (no training): food recognition P/R/F1 and label accuracy per split,
plus the meal-label check on the QA meal items.

Run:  python -m ontrack.eval.meal_eval
"""
from __future__ import annotations

import json
from collections import defaultdict

from ..config import CONFIG, ROOT
from ..meal.engine import MealClassifier
from ..meal.learned import food_recognition_scores


def evaluate(data_path=CONFIG.meal_data) -> dict:
    rows = [json.loads(l) for l in open(data_path, encoding="utf-8")]
    mc = MealClassifier()
    by = defaultdict(list)
    for r in rows:
        by[r["split"]].append(r)
    out = {}
    for split, rs in by.items():
        preds = [mc.analyse(r["text"]).label for r in rs]
        out[split] = {"n": len(rs), "label_source": rs[0]["label_source"],
                      "label_accuracy": round(sum(p == r["label"] for p, r in zip(preds, rs)) / len(rs), 3),
                      "food_recognition": {k: v for k, v in food_recognition_scores(rs, mc.recognizer).items() if k != "errors"}}
    qa = [json.loads(l) for l in open(ROOT / "data" / "eval" / "qa_batch2_meals.jsonl", encoding="utf-8")]
    out["qa_batch2_meals"] = {"n": len(qa), "engine_label_accuracy":
                              round(sum(mc.analyse(q["question"]).label == q["expected_meal_label"] for q in qa) / len(qa), 3)}
    out["note"] = ("'train' labels are engine output (accuracy 1.0 by construction); 'hard' was used while developing the rules; "
                   "'hard_blind' was labelled before running the engine and not tuned on. All rows synthetic.")
    return out


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=1))
