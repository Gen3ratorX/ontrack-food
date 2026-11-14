"""Learned meal classifier: TF-IDF (word + char n-grams) + logistic regression.

Run:  python -m ontrack.meal.learned   (5-fold CV on engine-labelled rows, hard-case comparison, saves model + metrics)

HONESTY NOTE - label leakage: the 'train' rows were labelled by the rule engine (ontrack/meal/engine.py).
Cross-validated accuracy therefore measures how well the model IMITATES THE ENGINE, not how well it
predicts real blood-glucose impact. The only engine-independent check is the small hand-labelled 'hard'
set, and the engine rules were adjusted once after looking at it (see README), so it is not fully blind either.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ..classifier.baseline import load_jsonl, make_model
from ..config import CONFIG
from . import LABELS
from .engine import MealClassifier


def _report(y_true, y_pred) -> dict:
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {"accuracy": round(rep["accuracy"], 3), "macro_f1": round(rep["macro avg"]["f1-score"], 3),
            "per_class_f1": {l: round(rep[l]["f1-score"], 3) for l in LABELS},
            "confusion_matrix": {"labels": LABELS, "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist()}}


def food_recognition_scores(rows: list[dict], recognizer=None) -> dict:
    from .recognizer import FoodRecognizer
    rec = recognizer or FoodRecognizer()
    tp = fp = fn = 0
    errors = []
    for r in rows:
        got = {m.food_id for m in rec.find(r["text"]) if not m.negated}
        exp = set(r["expected_foods"])
        tp += len(got & exp); fp += len(got - exp); fn += len(exp - got)
        if got != exp:
            errors.append({"id": r["id"], "text": r["text"], "missed": sorted(exp - got), "extra": sorted(got - exp)})
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {"precision": round(p, 3), "recall": round(rc, 3), "f1": round(f1, 3), "n": len(rows), "errors": errors}


def train_and_evaluate(data_path=CONFIG.meal_data, out_path=CONFIG.meal_model_path, seed: int = 42) -> dict:
    rows = load_jsonl(data_path)
    train = [r for r in rows if r["split"] == "train"]
    hard = [r for r in rows if r["split"] == "hard"]
    blind = [r for r in rows if r["split"] == "hard_blind"]
    X = np.array([r["text"] for r in train]); y = np.array([r["label"] for r in train])
    n_splits = min(5, min(int((y == l).sum()) for l in set(y)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_pred = cross_val_predict(make_model(seed), X, y, cv=cv)
    model = make_model(seed).fit(X, y)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)

    engine = MealClassifier()
    Xh = [r["text"] for r in hard]; yh = [r["label"] for r in hard]
    lr_h = list(model.predict(Xh))
    eng_h = [engine.analyse(t).label for t in Xh]
    per_item = [{"id": r["id"], "text": r["text"], "manual": r["label"], "engine": e, "lr": l}
                for r, e, l in zip(hard, eng_h, lr_h)]
    metrics = {
        "n_train_engine_labelled": len(train), "n_hard_manual": len(hard), "cv_folds": n_splits, "seed": seed,
        "label_counts_train": {l: int((y == l).sum()) for l in LABELS},
        "cv_lr_vs_engine_labels": _report(y, cv_pred),
        "hard_engine_vs_manual": _report(yh, eng_h),
        "hard_lr_vs_manual": _report(yh, lr_h),
        "hard_lr_engine_agreement": round(float(np.mean([a == b for a, b in zip(lr_h, eng_h)])), 3),
        "food_recognition_hard": food_recognition_scores(hard),
        "food_recognition_train": {k: v for k, v in food_recognition_scores(train).items() if k != "errors"},
        "hard_items": per_item,
        **_blind(blind, model, engine),
        "note": ("All data synthetic. CV labels come from the rule engine (label leakage): CV accuracy = how well "
                 "TF-IDF+LR imitates the engine. Hard set is small (n=%d) and hand-labelled with the documented method." % len(hard)),
    }
    Path(out_path).with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
    return metrics


def _blind(blind, model, engine) -> dict:
    if not blind:
        return {}
    Xb = [r["text"] for r in blind]; yb = [r["label"] for r in blind]
    lr_b = list(model.predict(Xb)); eng_b = [engine.analyse(t).label for t in Xb]
    return {"n_hard_blind": len(blind),
            "hard_blind_engine_vs_manual": _report(yb, eng_b),
            "hard_blind_lr_vs_manual": _report(yb, lr_b),
            "food_recognition_hard_blind": food_recognition_scores(blind),
            "hard_blind_items": [{"id": r["id"], "text": r["text"], "manual": r["label"], "engine": e, "lr": l}
                                 for r, e, l in zip(blind, eng_b, lr_b)]}


class LearnedMealClassifier:
    def __init__(self, path=CONFIG.meal_model_path):
        self.model = joblib.load(path)

    def predict(self, text: str) -> tuple[str, dict]:
        proba = self.model.predict_proba([text])[0]
        classes = list(self.model.classes_)
        return classes[int(proba.argmax())], {c: round(float(p), 3) for c, p in zip(classes, proba)}


if __name__ == "__main__":
    m = train_and_evaluate()
    print(json.dumps({k: v for k, v in m.items() if k not in ("hard_items",)}, indent=1)[:4000])
    print("\nHard cases (manual | engine | LR):")
    for it in m["hard_items"] + [{"id": "---"} | {"text": "", "manual": "", "engine": "", "lr": ""}] + m.get("hard_blind_items", []):
        flag = "" if it["manual"] == it["engine"] == it["lr"] else "  *"
        print(f"  {it['id']} {it['text'][:50]:50s} {it['manual']:22s} {it['engine']:22s} {it['lr']}{flag}")
