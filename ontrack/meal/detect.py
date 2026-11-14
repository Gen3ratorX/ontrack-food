"""Decide whether a user question describes a meal (so the meal classifier should run)."""
from __future__ import annotations

import re

from .lexicon import load_lexicon
from .recognizer import FoodMention, normalise

MEAL_CUES = re.compile(r"\b(eat|eats|ate|eaten|eating|meal|lunch|dinner|supper|breakfast|plate|bowl|ball|balls|chop|chopped|"
                       r"had|have|having|take|took|drink|drank|drinking|food|snack|portion|serving|cup|slice|slices|bottle|"
                       r"rate this|okay for me|ok for me|good for me|bad for)\b")


def is_meal_description(text: str, mentions: list[FoodMention]) -> bool:
    lex = load_lexicon()
    foods = [m for m in mentions if not m.negated and lex[m.food_id]["category"] != "drink_zero"]
    if not foods:
        return False
    if len({m.food_id for m in foods}) >= 2:
        return True
    if any(m.unit or m.grams or m.quantity != 1.0 or m.size != 1.0 for m in foods):
        return True
    return bool(MEAL_CUES.search(normalise(text)))
