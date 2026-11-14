"""Load the food lexicon and WAFCT 2019 composition rows."""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

from ..config import ROOT

LEXICON_PATH = ROOT / "knowledge_base" / "data" / "food_lexicon.json"
WAFCT_PATH = ROOT / "knowledge_base" / "data" / "wafct2019_selected.csv"
WAFCT_URL = "https://www.fao.org/fileadmin/user_upload/faoweb/2020/WAFCT_2019.xlsx"


def _num(v: str) -> float | None:
    v = (v or "").strip().strip("[]")
    try:
        return float(v)
    except ValueError:
        return None


@lru_cache(maxsize=1)
def load_wafct(path: str | Path = WAFCT_PATH) -> dict[str, dict]:
    rows = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["code"]] = {"code": r["code"], "name": r["food_name_en"], "carb_g": _num(r["carb_available_g"]),
                               "fibre_g": _num(r["fibre_g"]), "energy_kcal": _num(r["energy_kcal"]),
                               "estimated": "[" in (r["carb_available_g"] or "")}
    return rows


@lru_cache(maxsize=1)
def load_lexicon(path: str | Path = LEXICON_PATH) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {e["id"]: e for e in data["foods"]}
