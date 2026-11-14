"""TF-IDF (word + character n-grams) + logistic regression triage classifier.

Run:  python -m ontrack.classifier.baseline   (trains, evaluates, saves model + metrics)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from ..config import CONFIG
from . import LABELS


def load_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def make_model(seed: int = 42) -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)),
    ])
    clf = LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced", random_state=seed)
    return Pipeline([("feats", feats), ("clf", clf)])


def _report(y_true, y_pred) -> dict:
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    return {"report": rep, "confusion_matrix": {"labels": LABELS, "matrix": cm},
            "accuracy": rep["accuracy"], "macro_f1": rep["macro avg"]["f1-score"],
            "emergency_recall": rep["emergency"]["recall"]}


def train_and_evaluate(data_path=CONFIG.triage_data, out_path=CONFIG.classifier_path, seed: int = 42, test_size: float = 0.25) -> dict:
    from .gate import SafetyGate

    rows = load_jsonl(data_path)
    X = np.array([r["text"] for r in rows]); y = np.array([r["label"] for r in rows])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

    model = make_model(seed).fit(Xtr, ytr)
    holdout_model = _report(yte, model.predict(Xte))

    gate = SafetyGate(model=model)
    holdout_gate = _report(yte, [gate.classify(t)[0] for t in Xte])
    holdout_rules = _report(yte, [gate.classify(t, use_model=False)[0] for t in Xte])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_pred = cross_val_predict(make_model(seed), X, y, cv=cv)
    cv_model = _report(y, cv_pred)
    gate_cv_pred = np.empty_like(y)
    for tr, te in cv.split(X, y):
        g = SafetyGate(model=make_model(seed).fit(X[tr], y[tr]))
        gate_cv_pred[te] = [g.classify(t)[0] for t in X[te]]
    cv_gate = _report(y, gate_cv_pred)

    final = make_model(seed).fit(X, y)  # deploy model trained on all seed data
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out_path)

    metrics = {
        "n_examples": len(rows), "n_train": len(Xtr), "n_holdout": len(Xte), "seed": seed,
        "label_counts": {l: int((y == l).sum()) for l in LABELS},
        "holdout_model_only": holdout_model,
        "holdout_rules_only": holdout_rules,
        "holdout_gate_rules_plus_model": holdout_gate,
        "cv5_model_only": cv_model,
        "cv5_gate_rules_plus_model": cv_gate,
        "note": "Synthetic seed data; metrics are optimistic and only show the pipeline works.",
    }
    errs = [{"text": t, "true": a, "gate": gate.classify(t)[0]} for t, a in zip(Xte, yte) if gate.classify(t)[0] != a]
    metrics["holdout_gate_errors"] = errs
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CONFIG.triage_data))
    ap.add_argument("--out", default=str(CONFIG.classifier_path))
    ap.add_argument("--metrics", default=str(Path(CONFIG.classifier_path).with_suffix(".metrics.json")))
    a = ap.parse_args()
    m = train_and_evaluate(a.data, a.out)
    Path(a.metrics).write_text(json.dumps(m, indent=2))

    def line(name, r):
        return (f"{name:32s} acc={r['accuracy']:.3f} macroF1={r['macro_f1']:.3f} "
                f"emergency_recall={r['emergency_recall']:.3f}")
    print(f"n={m['n_examples']} {m['label_counts']}  holdout={m['n_holdout']}")
    print(line("holdout: model only", m["holdout_model_only"]))
    print(line("holdout: rules only", m["holdout_rules_only"]))
    print(line("holdout: gate (rules+model)", m["holdout_gate_rules_plus_model"]))
    print(line("5-fold CV: model only", m["cv5_model_only"]))
    print(line("5-fold CV: gate", m["cv5_gate_rules_plus_model"]))
    print("5-fold CV gate confusion", m["cv5_gate_rules_plus_model"]["confusion_matrix"]["matrix"])
    print("gate confusion (rows=true, cols=pred)", LABELS, m["holdout_gate_rules_plus_model"]["confusion_matrix"]["matrix"])
    for e in m["holdout_gate_errors"]:
        print("  miss:", e)
    print("saved", a.out, "and", a.metrics)


if __name__ == "__main__":
    main()
