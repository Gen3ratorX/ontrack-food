"""Recognise Ghanaian / West African foods and portions in free text.

Strategy: normalise text -> longest-first exact alias matching on token n-grams -> fuzzy matching
(difflib) for remaining tokens of length >= 5 -> quantity/size/unit parsing from the 4 tokens before a
match -> modifier-based variants (fried / ripe / white yam ...) -> negation ("no sugar", "without rice").
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import asdict, dataclass, field

from .lexicon import load_lexicon

NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "half": 0.5,
             "quarter": 0.25, "couple": 2, "few": 3, "double": 2, "some": 1}
SIZE_WORDS = {"small": 0.6, "little": 0.6, "tiny": 0.4, "medium": 1.0, "big": 1.5, "large": 1.5, "huge": 2.0, "heavy": 1.5, "full": 1.0}
UNITS = {"ball": "ball", "balls": "ball", "plate": "plate", "plates": "plate", "bowl": "bowl", "bowls": "bowl", "cup": "cup", "cups": "cup",
         "ladle": "ladle", "ladles": "ladle", "spoon": "spoon", "spoons": "spoon", "spoonful": "spoon", "tablespoon": "tablespoon",
         "tablespoons": "tablespoon", "tbsp": "tablespoon", "teaspoon": "teaspoon", "teaspoons": "teaspoon", "tsp": "teaspoon",
         "slice": "slice", "slices": "slice", "piece": "piece", "pieces": "piece", "finger": "finger", "fingers": "finger",
         "bottle": "bottle", "bottles": "bottle", "can": "can", "glass": "glass", "glasses": "glass", "handful": "handful",
         "pack": "pack", "packs": "pack", "loaf": "loaf", "tin": "tin", "sachet": "sachet", "cube": "cube", "cubes": "cube",
         "fruit": "fruit", "egg": "egg", "eggs": "egg", "g": "g", "grams": "g", "gram": "g"}
BOUNDARIES = {"with", "and", "plus", "&", "then", "also", "or", ",", ";", ":", "(", ")", "ne", "na"}
BLOOD_SUGAR_PREV = {"my", "blood", "his", "her", "their", "fasting", "your", "high", "low"}
BLOOD_SUGAR_NEXT = {"level", "levels", "is", "was", "reading", "readings", "went", "dropped", "drops", "keeps", "goes", "has", "test", "meter", "reads", "of"}
NEGATORS = {"no", "without", "not", "never", "skip", "skipped", "avoid", "minus"}
# English words that must never fuzzy-match a food
NO_FUZZ = {"would", "should", "could", "about", "think", "there", "their", "where", "which", "sugar", "having", "eating", "plate",
           "lunch", "dinner", "supper", "breakfast", "diabetes", "diabetic", "level", "levels", "after", "before", "today", "night",
           "morning", "little", "small", "large", "water", "fried", "boiled", "roasted", "grilled", "green", "white", "yellow", "ripe",
           "unripe", "steamed", "pieces", "piece", "balls", "bottle", "glass", "slice", "slices", "spoon", "with", "things", "drink",
           "drank", "wanna", "gonna", "please", "doctor", "nurse", "blood", "check", "every", "times", "heavy", "light", "sauce", "stew",
           "soup", "plenty", "really", "maybe", "today", "chop", "chopped", "cooked", "cooking", "palava", "pepper", "onion", "onions",
           "tomato", "tomatoes", "local", "market", "price"}


def normalise(text: str) -> str:
    t = text.lower().replace("ɛ", "e").replace("ɔ", "o").replace("’", "'")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"(\d)\s*/\s*(\d)", r"\1/\2", t)
    t = re.sub(r"(\d+)\s*(g|grams?)\b", r"\1 g", t)
    return t


def tokenize(text: str) -> list[str]:
    return re.findall(r"\d+/\d+|\d+(?:\.\d+)?|[a-z]+(?:[-.'][a-z]+)*|[,;:()]", normalise(text))


@dataclass
class FoodMention:
    food_id: str
    matched_text: str
    match: str  # exact | fuzzy
    quantity: float = 1.0
    unit: str | None = None
    size: float = 1.0
    grams: float | None = None  # explicit grams if stated
    negated: bool = False
    start: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class FoodRecognizer:
    def __init__(self, lexicon: dict | None = None, fuzzy_cutoff: float = 0.8):
        self.lex = lexicon or load_lexicon()
        self.alias2id: dict[tuple, str] = {}
        for fid, e in self.lex.items():
            for a in e["aliases"] + [e["id"].replace("_", " ")]:
                toks = tuple(tokenize(a))
                if toks:
                    self.alias2id.setdefault(toks, fid)
        self.alias_tokens = {k[0] for k in self.alias2id if len(k) == 1} - set(UNITS) - {"half"}
        self.max_n = max(len(k) for k in self.alias2id)
        self.single = {k[0]: v for k, v in self.alias2id.items() if len(k) == 1 and len(k[0]) >= 5}
        self.fuzzy_cutoff = fuzzy_cutoff

    def _qty(self, toks: list[str], i: int) -> tuple[float, str | None, float, float | None, bool]:
        qty, unit, size, grams, neg = 1.0, None, 1.0, None, False
        window = []
        for w in reversed(toks[max(0, i - 4): i]):
            if w in BOUNDARIES or w in self.alias_tokens:
                break
            window.insert(0, w)
        for j, w in enumerate(window):
            if w in NEGATORS:
                neg = True
            if re.fullmatch(r"\d+(\.\d+)?", w):
                nxt = window[j + 1] if j + 1 < len(window) else None
                if nxt == "g":
                    grams = float(w)
                else:
                    qty = float(w)
            elif re.fullmatch(r"\d+/\d+", w):
                a, b = w.split("/"); qty = float(a) / float(b)
            elif w in NUM_WORDS:
                qty = NUM_WORDS[w] if w not in ("a", "an", "some") or qty == 1.0 else qty
                if w == "half" and j + 1 < len(window) and window[j + 1] in ("a", "an"):
                    qty = 0.5
            elif w in SIZE_WORDS:
                size = SIZE_WORDS[w]
            elif w in UNITS:
                unit = UNITS[w]
        # "half a ball" / "a ball and a half"
        if "half" in window:
            qty = qty * 0.5 if qty not in (0.5, 1.0) else 0.5
        # a negator only counts if it is directly before the food or before its quantity phrase
        if neg and not (window and (window[-1] in NEGATORS or (len(window) > 1 and window[-2] in NEGATORS))):
            neg = False
        return qty, unit, size, grams, neg

    def find(self, text: str) -> list[FoodMention]:
        toks = tokenize(text)
        used = [False] * len(toks)
        found: list[FoodMention] = []
        # 1) exact, longest first
        for n in range(self.max_n, 0, -1):
            for i in range(len(toks) - n + 1):
                if any(used[i:i + n]):
                    continue
                key = tuple(toks[i:i + n])
                if key in self.alias2id:
                    q, u, s, g, neg = self._qty(toks, i)
                    found.append(FoodMention(self.alias2id[key], " ".join(key), "exact", q, u, s, g, neg, i))
                    for k in range(i, i + n):
                        used[k] = True
        # 2) fuzzy single tokens (typos like 'kenkay', 'waakey', 'kontomere')
        for i, t in enumerate(toks):
            if used[i] or len(t) < 5 or t in NO_FUZZ or t in UNITS or t in SIZE_WORDS:
                continue
            m = difflib.get_close_matches(t, list(self.single), n=1, cutoff=self.fuzzy_cutoff)
            if m:
                q, u, s, g, neg = self._qty(toks, i)
                found.append(FoodMention(self.single[m[0]], t, "fuzzy", q, u, s, g, neg, i, [f"fuzzy match '{t}' -> '{m[0]}'"]))
                used[i] = True
        found.sort(key=lambda m: m.start)
        return self._apply_variants(toks, found)

    def _apply_variants(self, toks, found):
        ids = {m.food_id for m in found}
        out = []
        for m in found:
            e = self.lex[m.food_id]
            ctx = toks[max(0, m.start - 3): m.start + len(m.matched_text.split()) + 2]
            for mod, target in (e.get("variants") or {}).items():
                if mod in ctx or (mod == "fried" and "fry" in ctx):
                    m.notes.append(f"'{mod}' -> {target}")
                    m.food_id = target
                    break
            if m.food_id == "sugar" and m.match == "exact":
                prev = toks[m.start - 1] if m.start else ""
                nxt = toks[m.start + 1] if m.start + 1 < len(toks) else ""
                if prev in BLOOD_SUGAR_PREV or nxt in BLOOD_SUGAR_NEXT:
                    continue  # "my sugar is high" = blood glucose, not a food
            drop = set(e.get("drop_if_cooccurs") or [])
            if drop & ids:
                continue
            out.append(m)
        # de-duplicate repeated mentions of the same food (keep the one with an explicit quantity)
        best: dict[str, FoodMention] = {}
        for m in out:
            if m.food_id not in best or (best[m.food_id].quantity == 1.0 and m.quantity != 1.0):
                best[m.food_id] = m
        return sorted(best.values(), key=lambda m: m.start)
