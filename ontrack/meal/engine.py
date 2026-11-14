"""Evidence-grounded meal glycaemic-impact engine.

Labels: low | medium | high | insufficient_evidence.

Per food item (portion grams = stated grams, or lexicon portion x quantity x size word):
  1. cited GI AND WAFCT carbohydrate  -> estimated GL = GI/100 x carb  (GI News cut-offs: <=10 low, 11-19 medium, >=20 high)
  2. WAFCT carbohydrate only          -> carbohydrate-only rating (CDC 15 g serving; <=30 g low, 31-60 medium, >60 high; OnTrack heuristic)
  3. cited GI only                    -> GI class (<=55 low, 56-69 medium, >=70 high), moved up one level for a large portion (>=1.5x)
  4. neither                          -> evidence gap (contributes no rating)
Meal: GL items are summed, carb-only items are summed, GI-class items taken individually; the meal level is the max.
Composition: >=2 of {protein, vegetable, legume/fibre, fat}, from >=2 different foods, moves the meal down one level, unless a sugary
drink/added sugar is present, a portion was large, or the meal's known carbohydrate exceeds 90 g.
No GI number is ever created here: GI values come only from the lexicon, whose values are tested against the cited docs.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from ..config import ROOT
from ..rag.chunking import parse_front_matter
from .lexicon import load_lexicon, load_wafct
from .recognizer import FoodMention, FoodRecognizer

LEVELS = ["low", "medium", "high"]
CARB_CATEGORIES = {"starch", "composite", "sugary_drink", "sugar", "fruit", "legume"}
MODERATOR_TYPES = {"protein", "vegetable", "legume", "fat"}
WAFCT_DOC = "wafct2019_ghana_foods"
METHOD_DOC = "glycaemic_load_method"
CARB_DOC = "cdc_healthy_carbs"
COMPOSITION_DOC = "cdc_meal_planning"
GAP_DOC = "gi_evidence_gaps"
CARB_ADJUST_CAP_G = 90


def gl_level(gl: float) -> str:
    gl = round(gl)  # GL is an estimate; GI News cut-offs are whole numbers
    return "low" if gl <= 10 else ("medium" if gl < 20 else "high")


def carb_level(carb: float) -> str:
    return "low" if carb <= 30 else ("medium" if carb <= 60 else "high")


def gi_level(gi: float) -> str:
    return "low" if gi <= 55 else ("medium" if gi < 70 else "high")


def _up(level: str) -> str:
    return LEVELS[min(LEVELS.index(level) + 1, 2)]


def _down(level: str) -> str:
    return LEVELS[max(LEVELS.index(level) - 1, 0)]


def _gi_phrase(gi: float, gi_type: str | None, gi_source: str | None) -> str:
    """GI number with its class, so a high meal rating cannot be read as a high GI."""
    kind = gi_type or "cited"
    src = gi_source or "cited source"
    return f"GI {gi:g} = {gi_level(gi).upper()} GI ({kind}, {src})"


def _portion_phrase(it: "ItemAnalysis") -> str:
    desc = it.portion_desc.strip()
    m = re.match(r"^(\d+(?:\.\d+)?) (.+)$", desc)
    if m:
        n = float(m.group(1))
        unit = m.group(2)
        if n > 1 and not unit.endswith("s"):
            unit += "s"
        num = str(int(n)) if n.is_integer() else f"{n:g}"
        desc = f"{num} {unit}"
    if it.carb_g is not None:
        return f"{desc}, ~{it.carb_g:.0f} g carbohydrate"
    return desc


def _large_portion(it: "ItemAnalysis") -> bool:
    if it.carb_g is not None and it.carb_g >= 60:
        return True
    if re.search(r"\blarge\b", it.portion_desc):
        return True
    m = re.match(r"^(\d+(?:\.\d+)?)", it.portion_desc)
    return bool(m and float(m.group(1)) >= 2)


def _plain_rating_reason(a: "MealAnalysis") -> str:
    """One plain sentence for why the meal rating is what it is. GI class is named separately."""
    if a.label not in LEVELS:
        return ""
    label = a.label.upper()
    starch = [i for i in a.items if i.category in CARB_CATEGORIES]
    # Meal impact above the food's own GI class: the portion / GL did it, not a high GI.
    lower = [i for i in starch if i.gi is not None and LEVELS.index(gi_level(i.gi)) < LEVELS.index(a.label)]
    if lower:
        sentences = []
        for i in lower:
            size = "large portion" if _large_portion(i) else "portion"
            sentences.append(
                f"Rated {label} because of the {size} ({_portion_phrase(i)}), "
                f"not because {i.name} has a high GI; {i.name} has a {gi_level(i.gi)} GI."
            )
        return " ".join(sentences)
    with_gi = [i for i in starch if i.gi is not None]
    if "composition" in a.basis and with_gi:
        bits = [f"{i.name} has a {gi_level(i.gi)} GI ({i.gi:g})" for i in with_gi]
        return (f"Rated {label} after moderating foods lowered the meal. "
                + "; ".join(bits) + ". Do not describe the GI as lower than that.")
    if with_gi and all(i.basis == "gi_class" or i.carb_g is None for i in with_gi):
        bits = [f"{i.name} has a {gi_level(i.gi)} GI ({_gi_phrase(i.gi, i.gi_type, i.gi_source)})" for i in with_gi]
        return f"Rated {label} because " + "; ".join(bits) + ". This follows the GI class."
    if with_gi:
        bits = [f"{i.name} {_gi_phrase(i.gi, i.gi_type, i.gi_source)}" for i in with_gi]
        return f"Rated {label} from glycaemic load or carbohydrate, and the GI class agrees: " + "; ".join(bits) + "."
    carb_bits = [f"{i.name}: {_portion_phrase(i)}" for i in starch if i.carb_g is not None]
    if carb_bits:
        return (f"Rated {label} because of the carbohydrate in the portion ({'; '.join(carb_bits)}). "
                "No cited GI is given, so do not call the GI high or low.")
    return f"Rated {label} from the meal composition, not from a glycaemic index."


@lru_cache(maxsize=1)
def doc_meta() -> dict[str, dict]:
    out = {}
    for p in (ROOT / "knowledge_base" / "docs").glob("*.md"):
        meta, _ = parse_front_matter(p.read_text(encoding="utf-8"))
        if meta.get("id"):
            out[meta["id"]] = {"id": meta["id"], "title": meta.get("title", ""), "url": meta.get("url", "")}
    return out


@dataclass
class ItemAnalysis:
    food_id: str
    name: str
    category: str
    matched_text: str
    match: str
    portion_g: float
    portion_assumed: bool
    portion_desc: str
    carb_g: float | None = None
    fibre_g: float | None = None
    wafct_code: str | None = None
    wafct_proxy: bool = False
    gi: float | None = None
    gi_type: str | None = None
    gi_source: str | None = None
    est_gl: float | None = None
    basis: str = "none"  # gl | carb_only | gi_class | none
    level: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class MealAnalysis:
    text: str
    label: str
    basis: str
    confidence: str
    est_glycaemic_load: float | None
    total_carb_g: float | None
    items: list[ItemAnalysis]
    drivers: list[str]
    moderators: list[str]
    flags: list[str]
    sources: list[dict]
    explanation: str
    negated: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_context(self) -> str:
        """Compact block for the RAG prompt. Numbers here must be kept as-is by the generator."""
        lines = [f"[M] OnTrack meal analysis (rule engine; estimates, not a diagnosis)",
                 f"Rating: {self.label.upper()} expected blood-sugar impact (basis: {self.basis}; confidence: {self.confidence})"]
        reason = _plain_rating_reason(self)
        if reason:
            lines.append(reason)
        if self.est_glycaemic_load is not None:
            lines.append(f"Estimated glycaemic load (estimate, from cited GI x WAFCT carbohydrate, assumed portions): {self.est_glycaemic_load:.0f}")
        no_carb = [i.name for i in self.items if i.category in CARB_CATEGORIES and i.carb_g is None]
        if self.total_carb_g is not None and not no_carb:
            lines.append(f"Estimated available carbohydrate (estimate, WAFCT, assumed portions): {self.total_carb_g:.0f} g")
        elif no_carb:
            part = f"; carbohydrate from the other foods ≈ {self.total_carb_g:.0f} g" if self.total_carb_g else ""
            lines.append(f"Carbohydrate could NOT be estimated for: {', '.join(no_carb)} (no WAFCT row){part}")
        for it in self.items:
            bits = [f"{it.name}: ~{it.portion_g:.0f} g ({it.portion_desc}{', assumed portion' if it.portion_assumed else ''})"]
            if it.carb_g is not None:
                bits.append(f"carb {it.carb_g:.0f} g (WAFCT {it.wafct_code}{', proxy' if it.wafct_proxy else ''})")
            if it.gi is not None:
                bits.append(_gi_phrase(it.gi, it.gi_type, it.gi_source))
            if it.est_gl is not None:
                bits.append(f"est. GL {it.est_gl:.0f}")
            if it.level:
                bits.append(f"impact {it.level} via {it.basis} (not the GI class)")
            else:
                bits.append("no GI or carbohydrate data")
            lines.append("- " + "; ".join(bits))
        if self.drivers:
            lines.append("Drivers of the rating: " + ", ".join(self.drivers))
        if self.moderators:
            lines.append("Moderating foods (protein/vegetable/legume/fat): " + ", ".join(self.moderators))
        for f in self.flags:
            lines.append("Note: " + f)
        lines.append("Sources: " + "; ".join(s["id"] for s in self.sources))
        return "\n".join(lines)


class MealClassifier:
    def __init__(self, recognizer: FoodRecognizer | None = None):
        self.lex = load_lexicon()
        self.wafct = load_wafct()
        self.recognizer = recognizer or FoodRecognizer(self.lex)

    # ------------------------------------------------------------------
    def _portion(self, e: dict, m: FoodMention, base_g: float | None = None) -> tuple[float, bool, str]:
        if m.grams:
            return m.grams, False, f"{m.grams:g} g stated"
        portions = e.get("portion") or {"default": 100}
        if base_g is not None:  # component of a composite
            g = base_g
        elif m.unit and m.unit in portions:
            g = portions[m.unit]
        else:
            g = portions["default"]
        mult = m.quantity * m.size
        unit = m.unit or e.get("default_unit", "serving")
        q = "half" if m.quantity == 0.5 else (f"{m.quantity:g}" if m.quantity != 1 else "1")
        size = {0.4: "tiny ", 0.6: "small ", 1.5: "large ", 2.0: "very large "}.get(m.size, "")
        return g * mult, True, f"{q} {size}{unit}".strip()

    def _item(self, fid: str, m: FoodMention, base_g: float | None = None) -> ItemAnalysis:
        e = self.lex[fid]
        g, assumed, desc = self._portion(e, m, base_g)
        it = ItemAnalysis(fid, e["name"], e["category"], m.matched_text, m.match, round(g, 1), assumed, desc,
                          wafct_code=e.get("wafct_code"), wafct_proxy=bool(e.get("wafct_proxy")))
        it.notes += list(m.notes)
        if e.get("proxy_note"):
            it.notes.append(e["proxy_note"])
        row = self.wafct.get(e["wafct_code"]) if e.get("wafct_code") else None
        if row and row["carb_g"] is not None:
            it.carb_g = round(row["carb_g"] * g / 100, 1)
            if row["fibre_g"] is not None:
                it.fibre_g = round(row["fibre_g"] * g / 100, 1)
        if e.get("gi"):
            it.gi, it.gi_type, it.gi_source = e["gi"]["value"], e["gi"]["type"], e["gi"]["source_id"]
        if e.get("gi_note"):
            it.notes.append(e["gi_note"])
        if e["category"] not in CARB_CATEGORIES and e["category"] != "soup":
            # vegetables / protein / fat: carbs counted if known, but they don't set the rating on their own
            it.basis = "carb_only" if it.carb_g is not None else "none"
            it.level = carb_level(it.carb_g) if it.carb_g is not None else None
            return it
        if it.gi is not None and it.carb_g is not None:
            it.est_gl = round(it.gi / 100 * it.carb_g, 1)
            it.basis, it.level = "gl", gl_level(it.est_gl)
        elif it.carb_g is not None:
            it.basis, it.level = "carb_only", carb_level(it.carb_g)
        elif it.gi is not None:
            it.basis, it.level = "gi_class", gi_level(it.gi)
            if m.quantity * m.size >= 1.5 or (m.grams and m.grams >= 1.5 * e["portion"]["default"]):
                it.level = _up(it.level)
                it.notes.append("moved up one level for a large portion (no carbohydrate data to compute GL)")
        return it

    # ------------------------------------------------------------------
    def analyse(self, text: str) -> MealAnalysis:
        mentions = self.recognizer.find(text)
        items: list[ItemAnalysis] = []
        negated: list[str] = []
        big_portion = False
        for m in mentions:
            e = self.lex[m.food_id]
            if m.negated:
                negated.append(e["name"])
                continue
            if m.quantity * m.size >= 1.5:
                big_portion = True
            if e.get("components"):
                mult = m.quantity * m.size
                for c in e["components"]:
                    sub = FoodMention(c["id"], m.matched_text, m.match, 1.0, None, 1.0, None, False, m.start,
                                      [f"part of {e['name']} ({e.get('component_note', '')})"])
                    it = self._item(c["id"], sub, base_g=c["grams"] * mult)
                    if m.grams:  # scale stated grams across components
                        tot = sum(cc["grams"] for cc in e["components"])
                        it = self._item(c["id"], sub, base_g=m.grams * c["grams"] / tot)
                    it.name = f"{e['name']}: {it.name}"
                    it.notes.append(e.get("gi_note") or "")
                    it.notes = [n for n in it.notes if n]
                    items.append(it)
            else:
                items.append(self._item(m.food_id, m))
        return self._rate(text, items, negated, big_portion, mentions)

    def _rate(self, text, items, negated, big_portion, mentions) -> MealAnalysis:
        flags: list[str] = []
        src_ids: list[str] = []
        carb_items = [i for i in items if i.category in CARB_CATEGORIES or i.category == "soup"]
        carb_items = [i for i in carb_items if i.category != "soup" or i.level]
        rated = [i for i in carb_items if i.level]
        gaps = [i for i in carb_items if not i.level]
        mods = sorted({t for i in items for t in self.lex[i.food_id].get("moderators", []) if t in MODERATOR_TYPES})
        moderator_foods = [i.name for i in items if set(self.lex[i.food_id].get("moderators", [])) & MODERATOR_TYPES]
        # need >=2 moderator TYPES coming from >=2 different foods (a handful of nuts alone does not count twice)
        sugary = [i for i in items if i.category in ("sugary_drink", "sugar")]

        if not items:
            return MealAnalysis(text, "insufficient_evidence", "no_foods_recognised", "low", None, None, [], [], [],
                                ["No foods were recognised. Describe the meal with food names and portions (e.g. '1 ball of kenkey with fish')."],
                                [], "No recognisable foods.", negated)

        gl_items = [i for i in rated if i.basis == "gl"]
        carbo_items = [i for i in rated if i.basis == "carb_only"]
        gi_items = [i for i in rated if i.basis == "gi_class"]
        candidates: list[tuple[str, str]] = []
        sum_gl = round(sum(i.est_gl for i in gl_items), 1) if gl_items else None
        if gl_items:
            candidates.append((gl_level(sum_gl), "gl"))
        sum_carb_only = sum(i.carb_g for i in carbo_items)
        if carbo_items:
            candidates.append((carb_level(sum_carb_only), "carb_only"))
        for i in gi_items:
            candidates.append((i.level, "gi_class"))
        known_carb = [i.carb_g for i in items if i.carb_g is not None]
        total_carb = round(sum(known_carb), 1) if known_carb else None

        if not carb_items:
            label, basis = "low", "composition"
            flags.append("No starchy or sugary foods recognised; rating based on meal composition (protein/vegetables). Portions and cooking oil still matter for weight.")
        elif not rated:
            label, basis = "insufficient_evidence", "evidence_gap"
            flags.append("None of the starchy/sugary foods here has a cited GI or WAFCT carbohydrate value: " + ", ".join(i.name for i in gaps) + ".")
        else:
            order = {"high": 2, "medium": 1, "low": 0}
            label, basis = max(candidates, key=lambda c: (order[c[0]], c[1] == "gl"))
            bases = sorted({b for _, b in candidates})
            basis = "+".join(bases) if len(bases) > 1 else bases[0]
            if gaps:
                flags.append("Partial evidence: no GI or carbohydrate data for " + ", ".join(i.name for i in gaps)
                             + "; the rating may underestimate this meal.")
            # composition adjustment
            if len(mods) >= 2 and len(moderator_foods) >= 2 and label != "low":
                if sugary:
                    flags.append("Moderating foods present, but a sugary drink/added sugar blocks the composition adjustment.")
                elif big_portion:
                    flags.append("Moderating foods present, but a large portion blocks the composition adjustment.")
                elif total_carb is not None and total_carb > CARB_ADJUST_CAP_G:
                    flags.append(f"Moderating foods present, but estimated carbohydrate above {CARB_ADJUST_CAP_G} g blocks the composition adjustment.")
                else:
                    flags.append(f"Moved down one level ({label} -> {_down(label)}) because the meal includes " + ", ".join(mods) + ".")
                    label = _down(label)
                    basis += "+composition"

        drivers = []
        key = [i for i in rated if i.category in CARB_CATEGORIES and (i.level in ("medium", "high") or label == "low")]
        if not key:  # several small carbohydrate portions that add up
            key = [i for i in rated if i.category in CARB_CATEGORIES]
        for i in key:
            d = f"{i.name} (impact {i.level} via {i.basis}"
            if i.gi is not None:
                d += f"; {_gi_phrase(i.gi, i.gi_type, i.gi_source)}"
            d += f", est. GL {i.est_gl:.0f}" if i.est_gl is not None else (f", ~{i.carb_g:.0f} g carb" if i.carb_g is not None else "")
            drivers.append(d + ")")
        # flags
        if gi_items:
            flags.append("GI-only rating for " + ", ".join(i.name for i in gi_items) + ": no WAFCT carbohydrate value, so portion size could not be "
                         "counted. A low GI does not make a food 'free' - a large ball still contains a lot of carbohydrate.")
        if any(i.gi_type == "predicted" for i in rated):
            flags.append("Uses an in-vitro PREDICTED GI (cassava); not measured in people.")
        if any(i.basis == "carb_only" and i.category in CARB_CATEGORIES for i in rated):
            flags.append("Carbohydrate-only rating for foods without a cited GI (OnTrack heuristic on CDC 15 g servings, needs dietitian validation).")
        if any(i.portion_assumed for i in items):
            flags.append("Portion sizes are OnTrack assumptions for a typical Ghanaian serving; actual portions change the result.")
        if any(i.wafct_proxy for i in items):
            flags.append("Some foods use the closest WAFCT row as a proxy (see item notes).")
        if any(i.match == "fuzzy" for i in items):
            flags.append("Some food names were matched approximately: " + ", ".join(f"'{i.matched_text}' -> {i.name}" for i in items if i.match == "fuzzy") + ".")
        if any(self.lex[i.food_id].get("fried") for i in items) or "fat" in mods:
            flags.append("Fried foods/oil/fatty soups add calories; fat slows the sugar rise but is not 'free'.")
        for i in sugary:
            flags.append(f"{i.name}: sugary drinks and added sugar raise blood sugar quickly; CDC advises choosing water or zero-calorie drinks.")
        if any(i.food_id == "yam" for i in items):
            flags.append("Yam GI differs by variety (44-75); say white, water or yellow yam for a GI-based estimate.")
        if any(i.food_id == "plantain" for i in items):
            flags.append("Ripeness not stated; ripe plantain carbohydrate values were used.")
        if negated:
            flags.append("Excluded (stated as not eaten): " + ", ".join(negated) + ".")

        # sources
        for i in items:
            if i.gi_source:
                src_ids.append(i.gi_source)
            if i.carb_g is not None:
                src_ids.append(WAFCT_DOC)
            if not i.level and i.category in CARB_CATEGORIES:
                ev = self.lex[i.food_id].get("evidence_doc") or GAP_DOC
                src_ids.extend(ev if isinstance(ev, list) else [ev])
            elif i.basis == "carb_only" and self.lex[i.food_id].get("evidence_doc") == GAP_DOC:
                src_ids.append(GAP_DOC)
        if any(i.food_id in ("yam", "fried_yam") for i in items):
            src_ids.append("gi_yam_ampofo2021")
        src_ids.append(METHOD_DOC)
        if any(i.basis == "carb_only" for i in rated):
            src_ids.append(CARB_DOC)
        if "composition" in basis or (len(mods) >= 2 and len(moderator_foods) >= 2):
            src_ids.append(COMPOSITION_DOC)
        meta = doc_meta()
        seen, sources = set(), []
        for s in src_ids:
            if s not in seen and s in meta:
                seen.add(s)
                sources.append(meta[s])

        # confidence
        if label == "insufficient_evidence":
            conf = "low"
        else:
            score = 2
            if gaps or any(i.match == "fuzzy" for i in items) or any(i.gi_type == "predicted" for i in rated):
                score -= 1
            if any(i.basis != "gl" for i in rated if i.category in CARB_CATEGORIES):
                score -= 1
            if gi_items and label == "low":
                score = 0
            if all(not i.portion_assumed for i in items) and score < 2:
                score += 1
            conf = ["low", "medium", "high"][max(score, 0)]
            if any(i.portion_assumed for i in items) and conf == "high":
                conf = "medium"

        expl = self._explain(label, basis, drivers, mods, sum_gl, total_carb, gaps + gi_items, bool(gl_items or carbo_items))
        return MealAnalysis(text, label, basis, conf, sum_gl, total_carb, items, drivers, moderator_foods, flags, sources, expl, negated)

    @staticmethod
    def _explain(label, basis, drivers, mods, sum_gl, total_carb, gaps, basis_has_carb=True) -> str:
        if label == "insufficient_evidence":
            return ("There is not enough cited evidence (no GI or carbohydrate data) to rate this meal. "
                    "Check your blood sugar about 2 hours after eating to learn how it affects you.")
        s = f"Expected blood-sugar impact: {label}."
        if drivers:
            s += " Main contributors: " + "; ".join(drivers) + "."
        if sum_gl is not None:
            s += f" Estimated glycaemic load of the GI-rated foods ≈ {sum_gl:.0f} (estimate)."
        if total_carb is not None and not gaps and basis_has_carb:
            s += f" Estimated carbohydrate ≈ {total_carb:.0f} g (estimate)."
        if mods:
            s += " Moderating components: " + ", ".join(mods) + "."
        if gaps:
            s += " Some foods have no data, so this may be an underestimate."
        return s


_default: MealClassifier | None = None


def analyse_meal(text: str) -> MealAnalysis:
    global _default
    if _default is None:
        _default = MealClassifier()
    return _default.analyse(text)
