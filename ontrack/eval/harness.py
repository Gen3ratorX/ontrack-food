"""Evaluation harness: run the assistant with and without RAG on the held-out test set, score, export.

Examples
  # CPU smoke test (no model download):
  python -m ontrack.eval.harness --generator echo --judge heuristic --limit 10
  # Colab T4, quick pass (20 stratified items, both modes, no judge, resumable):
  python -m ontrack.eval.harness --quick --generator medgemma --drive
  # Full 60, generation only (judge later):
  python -m ontrack.eval.harness --generator medgemma --max-new-tokens 256 --out artifacts/runs/full
  # Batched judge over a saved run:
  python -m ontrack.eval.harness --judge-only artifacts/runs/full/responses.jsonl --judge llm
  # After humans fill in the CSV:
  python -m ontrack.eval.harness --score-human artifacts/runs/<run>/human_scoring.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

from ..config import CONFIG
from .judge import HeuristicJudge, LLMJudge, safety_flags
from .rubric import CRITERIA, aggregate
HUMAN_COLS = ["row_id", "item_id", "question", "reference_answer", "culture_tag", "expected_triage", "response",
              "auto_flags", "accuracy", "relevance", "cultural_relevance", "safety", "clarity", "critical_unsafe", "reviewer", "notes"]

# Planning guess only. A T4 4-bit unbatched run was ~45s/answer with a judge at 350 tokens.
# fp16, 256 tokens, no judge, and batching are several times faster; live ETA replaces this.
_SEC_PER_GEN = 8.0
_SEC_PER_JUDGE = 6.0


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def build_assistant(mode: str, generator, retriever, gate):
    from ..pipeline import OnTrackAssistant

    return OnTrackAssistant(generator, retriever if mode == "rag" else None, gate, use_rag=(mode == "rag"))


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def stratum_labels(items: list[dict]) -> list[str]:
    """Triage|culture labels, collapsed so every label used for stratify has at least two rows."""
    keys = [f"{r.get('expected_triage', '?')}|{r.get('culture_tag', '?')}" for r in items]
    counts = Counter(keys)
    keys = [k if counts[k] >= 2 else k.split("|")[0] for k in keys]
    counts = Counter(keys)
    keys = [k if counts[k] >= 2 else "rare" for k in keys]
    rare = Counter(keys).get("rare", 0)
    if rare == 1:
        big = max((k for k in counts if k != "rare"), key=lambda k: counts[k])
        keys = [big if k == "rare" else k for k in keys]
    return keys


def _round_robin(items: list[dict], keys: list[str], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups: dict[str, list] = {}
    for it, k in zip(items, keys):
        groups.setdefault(k, []).append(it)
    for g in groups.values():
        rng.shuffle(g)
    picked: list[dict] = []
    order = sorted(groups)
    while len(picked) < n and any(groups.values()):
        for k in order:
            if groups[k]:
                picked.append(groups[k].pop())
                if len(picked) == n:
                    break
    return picked


def stratified_subset(items: list[dict], n: int | None, seed: int = 42) -> list[dict]:
    """Deterministic subset stratified by triage label and culture. n=None keeps every item."""
    if n is None or n >= len(items):
        return list(items)
    if n <= 0:
        return []
    keys = stratum_labels(items)
    subset = None
    if min(Counter(keys).values()) >= 2 and n < len(items):
        try:
            from sklearn.model_selection import train_test_split
            _, subset = train_test_split(items, test_size=n, stratify=keys, random_state=seed)
        except ValueError:
            subset = None
    if subset is None:
        subset = _round_robin(items, keys, n, seed)
    return sorted(subset, key=lambda r: r["id"])


def count_gated(items: list[dict], gate) -> tuple[int, int]:
    """Return (n_generate, n_template) for one mode."""
    n_gen = n_tpl = 0
    for it in items:
        if gate is None or gate.classify(it["question"])[0] == "general_education":
            n_gen += 1
        else:
            n_tpl += 1
    return n_gen, n_tpl


def estimate_seconds(n_generate: int, batch_size: int, judge: bool) -> float:
    """Rough wall-clock guess. Batching helps, but not linearly."""
    speedup = min(4.0, max(1.0, batch_size / 2))
    gen_s = n_generate * _SEC_PER_GEN / speedup
    judge_s = n_generate * _SEC_PER_JUDGE / speedup if judge else 0.0
    return gen_s + judge_s


def drive_runs_dir() -> Path | None:
    root = Path("/content/drive/MyDrive")
    if not root.is_dir():
        return None
    d = root / "ontrack-food" / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


MEAL_LABEL_WORDS = {"low": r"\blow\b", "medium": r"\b(medium|moderate)\b", "high": r"\bhigh\b",
                    "insufficient_evidence": r"(insufficient|not enough (evidence|data)|no (reliable )?data)"}


def meal_label_in_answer(expected: str, answer: str) -> bool:
    """Crude automatic check that the answer states the expected meal rating (humans confirm via the rubric)."""
    import re
    return bool(re.search(MEAL_LABEL_WORDS[expected], answer.lower()))


def _row_from_response(it: dict, response, latency_s: float) -> dict:
    row = {"item_id": it["id"], "mode": None, "response_mode": response.mode, "triage_label": response.triage_label,
           "expected_triage": it["expected_triage"], "culture_tag": it["culture_tag"],
           "question": it["question"], "reference_answer": it["reference_answer"], "response": response.answer,
           "sources": response.sources, "latency_s": round(latency_s, 2),
           "food_tags": it.get("food_tags", []),
           "auto_flags": safety_flags(it, response.answer)}
    if it.get("expected_meal_label"):
        row["expected_meal_label"] = it["expected_meal_label"]
        row["meal_classifier_label"] = (response.meal_analysis or {}).get("label")
        row["meal_label_in_answer"] = meal_label_in_answer(it["expected_meal_label"], response.answer)
    return row


def _load_done(path: Path) -> dict[tuple, dict]:
    done = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        done[(r["item_id"], r["mode"])] = r
    return done


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _rewrite(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _generate_batch(generator, messages, max_new_tokens, batch_size):
    if hasattr(generator, "generate_batch"):
        return generator.generate_batch(messages, max_new_tokens=max_new_tokens, batch_size=batch_size)
    return [generator.generate(m, max_new_tokens=max_new_tokens) for m in messages]


def _needs_judge(row: dict) -> bool:
    return row.get("accuracy") in (None, "")


def judge_saved(path: Path, judge, batch_size: int | None = None, max_new_tokens: int = 200) -> dict:
    """Score responses.jsonl in batches. Rows that already have an accuracy score are left as they are."""
    path = Path(path)
    rows = load_jsonl(path)
    pending = [i for i, r in enumerate(rows) if _needs_judge(r)]
    bs = batch_size or getattr(getattr(judge, "generator", None), "batch_size", None) or 4
    t0 = time.time()
    print(f"Judge-only: {len(pending)} of {len(rows)} responses to score, batch_size={bs}.")
    done_now = 0
    for start in range(0, len(pending), bs):
        idxs = pending[start:start + bs]
        pairs = [(rows[j], rows[j]["response"]) for j in idxs]
        batch_t = time.time()
        if hasattr(judge, "score_batch"):
            scored = judge.score_batch(pairs, max_new_tokens=max_new_tokens, batch_size=bs)
        else:
            scored = [judge.score(item, resp) for item, resp in pairs]
        for j, sc in zip(idxs, scored):
            rows[j].update(sc)
        _rewrite(path, rows)
        done_now += len(idxs)
        elapsed = time.time() - t0
        eta = elapsed / done_now * (len(pending) - done_now) if done_now else 0
        print(f"[judge {done_now}/{len(pending)}] batch={len(idxs)} took {fmt_duration(time.time() - batch_t)} "
              f"elapsed={fmt_duration(elapsed)} eta={fmt_duration(eta)}")
    summary = _summarise(rows, sorted({r["mode"] for r in rows}), getattr(judge, "name", "judge"), path.parent, judge=judge)
    print(json.dumps(summary, indent=2))
    print("updated", path)
    return summary


def _summarise(rows, modes, generator_name, out_dir: Path, judge=None) -> dict:
    items_per_mode = Counter(r["mode"] for r in rows)
    summary = {"n_items": max(items_per_mode.values()) if items_per_mode else 0, "modes": list(modes),
               "generator": generator_name, "judge": getattr(judge, "name", None),
               "triage_agreement": {m: sum(r["triage_label"] == r["expected_triage"] for r in rows if r["mode"] == m) / max(1, items_per_mode[m]) for m in modes},
               "auto_flag_counts": {m: sum(bool(r["auto_flags"]) for r in rows if r["mode"] == m) for m in modes}}
    meal_rows = [r for r in rows if "expected_meal_label" in r]
    if meal_rows:
        summary["meal_label_in_answer"] = {m: sum(r["meal_label_in_answer"] for r in meal_rows if r["mode"] == m)
                                           / max(1, sum(r["mode"] == m for r in meal_rows)) for m in modes}
        summary["meal_classifier_agreement"] = sum(r["meal_classifier_label"] == r["expected_meal_label"] for r in meal_rows if r["mode"] == "rag") \
            / max(1, sum(r["mode"] == "rag" for r in meal_rows))
    if judge is not None and any(not _needs_judge(r) for r in rows):
        summary["judge_scores"] = aggregate(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run(items, modes, generator, retriever, gate, judge=None, out_dir: Path | None = None, *,
        max_new_tokens: int = 256, batch_size: int | None = None, resume: bool = True) -> dict:
    out_dir = Path(out_dir or CONFIG.runs_dir / time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    resp_path = out_dir / "responses.jsonl"
    done = _load_done(resp_path) if resume else {}
    if not resume and resp_path.exists():
        resp_path.unlink()
        done = {}
    bs = batch_size or getattr(generator, "batch_size", None) or 4
    n_gen, n_tpl = count_gated(items, gate)
    n_generate_total = n_gen * len(modes)
    est = estimate_seconds(n_generate_total, bs, judge is not None)
    already = sum((it["id"], m) in done for m in modes for it in items)
    print(f"Eval: {len(items)} items x {list(modes)}, max_new_tokens={max_new_tokens}, batch_size={bs}, "
          f"judge={getattr(judge, 'name', None)}.")
    print(f"Per mode: {n_tpl} fixed template answers (no generation), {n_gen} to generate. "
          f"Already saved: {already}. Rough estimate {fmt_duration(est)} "
          f"({'includes a batched judge pass' if judge is not None else 'no judge'}). Live ETA follows.")
    print("Writing", resp_path)

    t0 = time.time()
    finished_now = 0
    remaining_at_start = len(items) * len(modes) - already

    def progress(mode, idx, it, label, note, *, count: bool = True):
        nonlocal finished_now
        if count:
            finished_now += 1
        elapsed = time.time() - t0
        left = max(0, remaining_at_start - finished_now)
        eta = elapsed / finished_now * left if finished_now else 0
        print(f"[{mode} {idx}/{len(items)}] {it['id']} triage={label} {note} "
              f"elapsed={fmt_duration(elapsed)} eta={fmt_duration(eta)}")

    for mode in modes:
        asst = build_assistant(mode, generator, retriever, gate)
        pending = []
        for idx, it in enumerate(items, 1):
            key = (it["id"], mode)
            if key in done:
                progress(mode, idx, it, done[key].get("triage_label", "?"), "resume-skip", count=False)
                continue
            t_prep = time.time()
            prep = asst.prepare(it["question"])
            if prep.done is not None:
                row = _row_from_response(it, prep.done, time.time() - t_prep)
                row["mode"] = mode
                _append_row(resp_path, row)
                done[key] = row
                progress(mode, idx, it, prep.done.triage_label, "template")
            else:
                pending.append((idx, it, prep))
        for start in range(0, len(pending), bs):
            chunk = pending[start:start + bs]
            t_batch = time.time()
            texts = _generate_batch(generator, [prep.messages for _, _, prep in chunk], max_new_tokens, bs)
            if len(texts) != len(chunk):
                raise RuntimeError(f"generate_batch returned {len(texts)} texts for {len(chunk)} prompts")
            share = (time.time() - t_batch) / max(1, len(chunk))
            for (idx, it, prep), text in zip(chunk, texts):
                resp = asst.complete(prep, text)
                row = _row_from_response(it, resp, share)
                row["mode"] = mode
                _append_row(resp_path, row)
                done[(it["id"], mode)] = row
                progress(mode, idx, it, resp.triage_label, f"gen_batch={len(chunk)} flags={row['auto_flags']}")

    rows = [done[(it["id"], mode)] for mode in modes for it in items]
    _rewrite(resp_path, rows)
    export_human_csv(rows, out_dir)
    if judge is not None:
        judge_saved(resp_path, judge, batch_size=bs)
        rows = load_jsonl(resp_path)
    summary = _summarise(rows, modes, getattr(generator, "model_id", "?"), out_dir, judge=judge)
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


def _resolve_out(a) -> Path:
    if a.drive:
        base = drive_runs_dir()
        if base is None:
            print("Warning: --drive set but /content/drive/MyDrive is not mounted. Saving locally.")
        elif a.out:
            name = Path(a.out).name
            return base / name
        else:
            return base / ("quick" if a.quick else "full")
    if a.out:
        return Path(a.out)
    if a.quick:
        return CONFIG.runs_dir / "quick"
    return CONFIG.runs_dir / time.strftime("%Y%m%d-%H%M%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev", "meals"],
                    help="'meals' = data/eval/qa_batch2_meals.jsonl (separate set, not part of the frozen test split)")
    ap.add_argument("--modes", nargs="+", default=["rag", "no_rag"], choices=["rag", "no_rag"])
    ap.add_argument("--generator", default="medgemma", choices=["medgemma", "echo"])
    ap.add_argument("--judge", default=None, choices=["none", "llm", "heuristic"],
                    help="none skips the judge (the default, and the quick-mode default)")
    ap.add_argument("--judge-model", default=None, help="HF model id for the judge (default: reuse the generator)")
    ap.add_argument("--judge-only", default=None, help="batched judge pass over an existing responses.jsonl")
    ap.add_argument("--no-gate", action="store_true", help="evaluate the raw model without the triage gate")
    ap.add_argument("--limit", type=int, default=None, help="stratified subset size (triage x culture, seed fixed)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="20-item stratified subset, no judge, 256 new tokens")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="eval generation cap (default 256)")
    ap.add_argument("--batch-size", type=int, default=None, help="generate/judge batch size (default 8 on a T4)")
    ap.add_argument("--fp16", action="store_true", help="load MedGemma in float16 (faster; fits a 15 GB T4)")
    ap.add_argument("--4bit", dest="fourbit", action="store_true", help="load MedGemma in 4-bit NF4 (slower, less memory)")
    ap.add_argument("--drive", action="store_true", help="save under /content/drive/MyDrive/ontrack-food/runs if Drive is mounted")
    ap.add_argument("--no-resume", action="store_true", help="ignore responses.jsonl and start this out dir over")
    ap.add_argument("--out", default=None)
    ap.add_argument("--score-human", default=None, help="path to a completed human_scoring.csv")
    a = ap.parse_args()

    if a.score_human:
        score_human(Path(a.score_human)); return

    if a.quick:
        if a.limit is None:
            a.limit = 20
        if a.judge is None:
            a.judge = "none"
        if a.max_new_tokens is None:
            a.max_new_tokens = 256
    if a.judge is None:
        a.judge = "none"
    if a.max_new_tokens is None:
        a.max_new_tokens = 256

    from ..rag.generator import EchoGenerator, MedGemmaGenerator, prefer_4bit

    if a.fourbit and a.fp16:
        raise SystemExit("Pass only one of --4bit and --fp16.")
    load_4bit = prefer_4bit(True if a.fourbit else False if a.fp16 else None)

    def make_gen(max_new_tokens: int):
        if a.generator == "echo":
            return EchoGenerator()
        return MedGemmaGenerator(CONFIG.model_id, load_4bit, max_new_tokens, batch_size=a.batch_size)

    if a.judge_only:
        path = Path(a.judge_only)
        if a.judge == "heuristic":
            judge = HeuristicJudge()
        elif a.judge == "llm":
            gen = make_gen(a.max_new_tokens)
            jgen = gen if not a.judge_model else MedGemmaGenerator(a.judge_model, load_4bit, 200, batch_size=a.batch_size)
            judge = LLMJudge(jgen)
        else:
            raise SystemExit("--judge-only needs --judge llm or --judge heuristic")
        judge_saved(path, judge, batch_size=a.batch_size)
        return

    path = CONFIG.qa_data.parent / "qa_batch2_meals.jsonl" if a.split == "meals" else CONFIG.splits_dir / f"{a.split}.jsonl"
    items = stratified_subset(load_jsonl(path), a.limit, a.seed)
    print(f"Items: {len(items)}  triage={dict(Counter(r['expected_triage'] for r in items))}  "
          f"culture={dict(Counter(r['culture_tag'] for r in items))}")
    from ..classifier.gate import SafetyGate
    from ..rag.retriever import Retriever

    gen = make_gen(a.max_new_tokens)
    retriever = Retriever.from_config() if "rag" in a.modes else None
    gate = None if a.no_gate else SafetyGate.from_config()
    judge = None
    if a.judge == "heuristic":
        judge = HeuristicJudge()
    elif a.judge == "llm":
        jgen = gen if not a.judge_model else MedGemmaGenerator(a.judge_model, load_4bit, 200, batch_size=a.batch_size)
        judge = LLMJudge(jgen)
    run(items, a.modes, gen, retriever, gate, judge, _resolve_out(a),
        max_new_tokens=a.max_new_tokens, batch_size=a.batch_size, resume=not a.no_resume)


if __name__ == "__main__":
    main()
