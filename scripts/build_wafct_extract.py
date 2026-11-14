"""Extract Ghana-relevant foods from the FAO/INFOODS West African Food Composition Table (WAFCT 2019).

If knowledge_base/data/wafct2019_selected.csv is already present, the download and rebuild are
skipped unless --force is passed (Colab just needs the committed extract). Otherwise the official
Excel file is downloaded when it is not already on disk, a curated list of food codes is selected, and:
  knowledge_base/data/wafct2019_selected.csv   (machine-readable, per 100 g edible portion)
  knowledge_base/docs/wafct2019_ghana_foods.md (text for the RAG index)
are written.

Values are copied exactly from sheet "03 NV_sum_39 (per 100g EP)". Values in [brackets] are
FAO's notation for lower-quality/ estimated values and are kept as-is.

Required citation: Vincent, A. et al. 2020. FAO/INFOODS Food Composition Table for Western Africa (2019)
User Guide & Condensed Food Composition Table. Rome, FAO.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.fao.org/fileadmin/user_upload/faoweb/2020/WAFCT_2019.xlsx"
SHEET = "03 NV_sum_39 (per 100g EP)"
# FAO returns 403 to urllib's default User-Agent.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)

# code -> OnTrack local name(s)
SELECTED = {
    "01_099": "kenkey (cooked fermented maize dough)",
    "01_102": "Hausa koko (fermented millet porridge, no sugar)",
    "01_134": "white rice, boiled (base of jollof / plain rice)",
    "01_152": "brown rice, boiled",
    "01_033": "millet, whole grain, boiled",
    "01_046": "white bread (wheat, unfortified)",
    "01_048": "wholemeal bread",
    "02_041": "fermented cassava paste (Nigerian-style wet fufu; not the same as Ghanaian pounded fufu)",
    "02_040": "dry fufu flour (fermented cassava)",
    "02_039": "gari (white, toasted)",
    "02_003": "cassava, boiled (ampesi)",
    "02_006": "cocoyam, white, boiled",
    "02_064": "plantain, unripe (green), boiled",
    "02_057": "plantain, ripe, boiled",
    "02_084": "plantain, deep fried (e.g. kelewele / fried plantain)",
    "02_020": "yam, boiled",
    "02_082": "yam, deep fried",
    "02_018": "water yam, boiled",
    "02_072": "sweet potato, orange flesh, boiled",
    "03_007": "cowpea / black-eyed beans, boiled (waakye, red red)",
    "03_019": "Bambara groundnut (aboboi), Ghana, boiled",
    "04_029": "kontomire (cocoyam leaves), boiled",
    "04_034": "garden egg (eggplant), boiled",
    "04_103": "native garden egg, boiled",
    "04_042": "okro (okra), boiled",
    "04_024": "amaranth leaves (aleefi), boiled",
    "04_039": "ayoyo (jute mallow) leaves, boiled",
    "05_002": "avocado (pear)",
    "05_015": "mango, ripe",
    "05_017": "pawpaw (papaya), ripe",
    "05_018": "pineapple",
    "05_016": "orange",
    "05_022": "watermelon",
    "05_003": "banana, ripe",
    "06_025": "groundnuts, Ghana, raw",
    "06_026": "groundnut paste",
    "08_002": "egg, boiled",
    "09_043": "tilapia, grilled",
    "09_037": "sardine, canned in oil",
    "07_035": "chicken, light meat without skin, grilled",
    "11_004": "red palm oil",
    "12_024": "carbonated soft drink (cola, Sprite-type)",
    "12_011": "orange juice, fresh, unsweetened",
    "12_023": "coconut water",
    "13_002": "white sugar",
    "13_001": "honey",
    # added for the meal classifier (v0.2)
    "01_133": "oats porridge (no sugar)",
    "01_053": "macaroni / spaghetti, boiled (proxy for talia)",
    "01_188": "sweet biscuit",
    "03_040": "soya beans, Ghana, boiled",
    "08_003": "egg, fried",
    "07_011": "beef, lean, grilled",
    "12_008": "tea, infusion (no sugar)",
}

COLS = {
    0: "code", 1: "food_name_en", 7: "energy_kcal", 9: "protein_g", 10: "fat_g",
    11: "carb_available_g", 12: "fibre_g", 19: "sodium_mg", 18: "potassium_mg",
}


def csv_path() -> Path:
    return ROOT / "knowledge_base" / "data" / "wafct2019_selected.csv"


def download_xlsx(dest: Path) -> None:
    """Download the WAFCT workbook. FAO blocks the default urllib user agent."""
    print("Downloading", URL)
    req = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def main(xlsx: Path | None = None, force: bool = False) -> int:
    out_csv = csv_path()
    if out_csv.exists() and not force:
        print(f"WAFCT extract already present ({out_csv.relative_to(ROOT)}); skipping download and rebuild.")
        return 0

    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    xlsx = xlsx or raw / "WAFCT_2019.xlsx"
    if not xlsx.exists():
        try:
            download_xlsx(xlsx)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            rel = out_csv.relative_to(ROOT)
            if out_csv.exists():
                print(f"Warning: could not download {URL} ({e}). Keeping the committed extract at {rel}.")
                return 0
            print(f"Error: could not download {URL} ({e}) and {rel} is missing.", file=sys.stderr)
            return 1
    df = pd.read_excel(xlsx, sheet_name=SHEET, header=None).iloc[3:]
    df = df[list(COLS)].rename(columns=COLS)
    df = df[df["code"].isin(SELECTED)].copy()
    missing = set(SELECTED) - set(df["code"])
    if missing:
        raise SystemExit(f"Codes not found in WAFCT: {sorted(missing)}")
    df.insert(1, "ontrack_name", df["code"].map(SELECTED))
    df["code"] = pd.Categorical(df["code"], categories=list(SELECTED), ordered=True)
    df = df.sort_values("code")
    for c in ["energy_kcal", "protein_g", "fat_g", "carb_available_g", "fibre_g", "sodium_mg", "potassium_mg"]:
        df[c] = df[c].map(lambda v: "" if pd.isna(v) else (f"{v:g}" if isinstance(v, (int, float)) else str(v).strip()))
    df.to_csv(out_csv, index=False)

    lines = [
        "---",
        "id: wafct2019_ghana_foods",
        "title: Nutrient composition of Ghanaian foods (FAO/INFOODS WAFCT 2019 extract, per 100 g edible portion)",
        "publisher: FAO/INFOODS. Vincent A. et al. 2020. FAO/INFOODS Food Composition Table for Western Africa (2019). Rome, FAO.",
        f"url: {URL}",
        "licence: FAO encourages use for private study, research and non-commercial purposes with acknowledgement (see Introduction sheet). Values copied exactly; [brackets] = FAO lower-quality/estimated value.",
        "accessed: 2026-11-14",
        "topic: food composition, carbohydrate, fibre, energy, Ghanaian foods",
        "chunk_size: 250",
        "---",
        "# Food composition of common Ghanaian foods (WAFCT 2019)",
        "",
        "All values per 100 g edible portion as described. Available carbohydrate is what mainly raises blood glucose; fibre slows absorption. "
        "Composition is not the same as glycaemic index (see the GI documents). WAFCT code in brackets.",
        "",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"- {r['ontrack_name']} — WAFCT '{r['food_name_en']}' [{r['code']}]: energy {r['energy_kcal']} kcal, "
            f"available carbohydrate {r['carb_available_g']} g, fibre {r['fibre_g']} g, protein {r['protein_g']} g, "
            f"fat {r['fat_g']} g, sodium {r['sodium_mg']} mg per 100 g."
        )
    out_md = ROOT / "knowledge_base" / "docs" / "wafct2019_ghana_foods.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(df)} foods -> {out_csv.relative_to(ROOT)} and {out_md.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default=None)
    ap.add_argument("--force", action="store_true", help="re-download and rebuild even if the committed CSV exists")
    a = ap.parse_args()
    sys.exit(main(a.xlsx, force=a.force))
