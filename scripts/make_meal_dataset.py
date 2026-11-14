"""Build the SYNTHETIC meal-description dataset for the learned meal classifier.

Two parts (every row has synthetic=true):
  * templated  (~170): generated from templates; label = rule-engine output (label_source="engine").
                       A model trained on these learns to imitate the engine -> label leakage, by design.
  * hard cases (~35) : hand-written by the OnTrack team (typos, Twi names, negation, evidence gaps,
                       'blood sugar' phrases, unusual portions); label assigned MANUALLY by applying the
                       documented method in knowledge_base/docs/glycaemic_load_method.md
                       (label_source="manual"). Used only as a held-out challenge set.
Run:  python scripts/make_meal_dataset.py
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ontrack.meal.engine import MealClassifier  # noqa: E402

random.seed(7)
FRAMES = ["I ate {m}.", "For lunch I had {m}.", "Dinner was {m}.", "Is {m} okay for me?", "Can I eat {m}?", "Breakfast: {m}.",
          "I'm about to eat {m}, how will it affect my sugar?", "Today I took {m}.", "What about {m}?", "I want to have {m} tonight.",
          "Rate this meal: {m}", "My lunch was {m} — is that bad for diabetes?"]
STARCH = ["1 ball of kenkey", "2 balls of kenkey", "a small ball of kenkey", "half a ball of kenkey", "1 ball of banku", "2 balls of banku",
          "a small ball of banku", "fufu", "a big ball of fufu", "fufu flour fufu", "tuo zaafi", "2 balls of TZ", "akple", "kafa", "abolo",
          "a plate of jollof", "a big plate of jollof rice", "a small plate of jollof", "waakye", "a big plate of waakye", "plain rice",
          "a cup of rice", "brown rice", "2 slices of bread", "4 slices of tea bread", "2 slices of wholemeal bread", "boiled white yam",
          "2 slices of boiled yellow yam", "boiled water yam", "yam", "fried yam", "boiled plantain", "boiled unripe plantain",
          "ripe plantain", "kelewele", "cassava ampesi", "cocoyam", "a cup of gari soakings", "kokonte", "a bowl of oats",
          "a plate of spaghetti", "koko", "tom brown", "red red", "200 g of rice", "100 g of kenkey", "kofi brokeman"]
SIDES = ["fried fish", "grilled tilapia", "shito", "pepper sauce", "okro stew", "okro soup", "kontomire stew", "garden egg stew",
         "light soup", "groundnut soup", "palm nut soup", "ayoyo soup", "2 boiled eggs", "an egg", "chicken", "goat meat", "salad",
         "beans", "avocado", "sardines", "koose", "groundnuts", "vegetables", "khebab"]
DRINKS = ["a bottle of coke", "a malt", "sobolo", "a glass of orange juice", "water", "tea with 2 teaspoons of sugar", "tea without sugar",
          "a Fanta", "Milo with sugar"]
FRUIT = ["a mango", "a slice of pawpaw", "a slice of watermelon", "a banana", "an orange", "2 slices of pineapple"]


def compose():
    s = random.choice(STARCH)
    k = random.choice([0, 1, 1, 2, 2, 3])
    sides = random.sample(SIDES, k)
    parts = [s] + sides
    if random.random() < 0.25:
        parts.append(random.choice(DRINKS))
    if random.random() < 0.1:
        parts.append(random.choice(FRUIT))
    m = parts[0] if len(parts) == 1 else (parts[0] + " with " + ", ".join(parts[1:-1]) + (" and " if len(parts) > 2 else "") + parts[-1]
                                          if len(parts) > 1 else parts[0])
    return random.choice(FRAMES).format(m=m)


NO_STARCH = ["grilled fish and salad", "2 boiled eggs and garden egg stew", "chicken and vegetables", "khebab and avocado",
             "light soup with goat meat only, no fufu", "groundnuts", "a slice of watermelon", "a mango", "a banana and groundnuts",
             "kokonte and light soup", "a cup of tom brown", "rice water", "ampesi with kontomire stew", "akyeke", "a malt and koose",
             "a bottle of coke", "2 bottles of coke", "a glass of orange juice", "Milo with sugar", "sobolo", "a malt"]

# (text, manual label, expected canonical foods, note)
HARD = [
 ("me di dɔkono ne nam", "high", ["kenkey", "fish"], "Twi: 'I ate kenkey and fish'; 1 ball default -> est. GL > 20"),
 ("I had komi and fried fish with pepper", "high", ["kenkey", "fish"], "Ga name for kenkey"),
 ("kenkay with shito and fish", "medium", ["kenkey", "stew", "fish"], "typo; GL ~26 high, moved down (fish protein + shito oil)"),
 ("2 balls of bankuu and okro", "high", ["banku", "okro"], "typo; GI 73 high"),
 ("waakey with gari and spaghetti", "high", ["waakye", "gari", "spaghetti"], "typo; lots of starch"),
 ("small plate of jolof with salad and grilled chicken", "low", ["jollof", "vegetables", "chicken"], "typo; ~47 g carb medium, moved down for veg+protein"),
 ("koko with no sugar", "low", ["koko"], "negation: sugar excluded"),
 ("koko with 3 teaspoons of sugar and bread", "high", ["koko", "sugar", "white_bread"], "added sugar"),
 ("fufu with light soup but without meat", "low", ["fufu", "light_soup"], "negation of meat; GI 55 low (GI-only)"),
 ("I skipped the rice and had only fish and salad", "low", ["fish", "vegetables"], "negated starch"),
 ("my sugar is 9 after eating banku", "high", ["banku"], "'sugar' here is blood glucose, not food"),
 ("kokonte with groundnut soup", "insufficient_evidence", ["kokonte", "groundnut_soup"], "conflicting GI evidence, no carb data"),
 ("tom brown porridge", "insufficient_evidence", ["tom_brown"], "evidence gap"),
 ("rice water", "insufficient_evidence", ["rice_water"], "evidence gap"),
 ("a bottle of malta guinness", "insufficient_evidence", ["malt"], "no composition data (flag sugary)"),
 ("sobolo", "insufficient_evidence", ["sobolo"], "no composition data"),
 ("ampesi and kontomire", "insufficient_evidence", ["ampesi", "kontomire"], "ampesi food type unspecified"),
 ("bayerɛ ampesi with nkontommire", "medium", ["yam", "kontomire"], "Twi yam + kontomire; carb-only ~54 g medium; one moderator only"),
 ("boiled white yam with kontomire stew and egg", "medium", ["white_yam", "kontomire", "egg"], "GL 24 high -> medium with veg+protein"),
 ("yellow yam with garden egg stew", "high", ["yellow_yam", "garden_egg"], "GI 75; GL 40 high; only vegetable moderator"),
 ("bankye ampesi", "high", ["cassava"], "Twi cassava; predicted GI 77.3 -> GL ~42"),
 ("borɔdeɛ ampesi with nkontommire and fish", "low", ["plantain", "kontomire", "fish"], "Twi plantain; ~52 g carb medium, moved down for vegetable + fish"),
 ("kofi brokeman and groundnuts", "medium", ["roasted_plantain", "groundnut"], "roasted plantain proxy ~52 g; a handful of nuts alone should not move it down"),
 ("redred", "medium", ["red_red"], "composite; ~88 g carb high, moved down (legume+fat)"),
 ("nkatenkwan ne fufu", "low", ["groundnut_soup", "fufu"], "Twi groundnut soup with fufu; GI 55 low (GI-only)"),
 ("abenkwan and 2 balls of fufu", "medium", ["palmnut_soup", "fufu"], "fufu GI 55 low moved up for 2 balls"),
 ("3 balls of banku with tilapia and pepper", "high", ["banku", "fish"], "large portion blocks adjustment; 'pepper' (fresh ground pepper) deliberately not counted as a food"),
 ("half ball of kenkey with fish and salad", "low", ["kenkey", "fish", "vegetables"], "GL ~13 medium -> low with fish+salad"),
 ("100 g kenkey and sardines", "low", ["kenkey", "sardine"], "stated grams: GL ~10 low"),
 ("200g rice", "medium", ["white_rice"], "stated grams: ~53 g carb"),
 ("gari soakings with sugar and groundnuts", "high", ["gari", "sugar", "groundnut"], "gari 58 g + sugar"),
 ("kelewele and a minerals", "high", ["fried_plantain", "soft_drink"], "local word 'minerals' = soft drink"),
 ("a cup of tea without sugar and 2 boiled eggs", "low", ["tea", "egg"], "no starch"),
 ("ayoyo soup with tz", "medium", ["ayoyo", "tuo_zaafi"], "GI 68 medium (GI-only), short alias tz"),
 ("akple with okro soup and crab", "medium", ["akple", "okro"], "GI 69 medium; crab not in lexicon"),
]

# Second hand-labelled set, written and labelled BEFORE the engine was run on it, evaluated once and frozen.
# (No rule or label changes were made after seeing these results.)
HARD_BLIND = [
 ("Two fingers of boiled green plantain with garden egg stew and boiled egg", "high", ["unripe_plantain", "garden_egg", "egg"], "~91 g carb; large portion blocks adjustment"),
 ("I drank 2 bottles of Fanta", "high", ["soft_drink"], "~69 g sugar"),
 ("one slice of wholemeal bread with avocado and egg", "low", ["wholemeal_bread", "avocado", "egg"], "~11 g carb"),
 ("big bowl of oats with honey", "medium", ["oats", "honey"], "~39 g + ~16 g carb"),
 ("Nkate nkwan with a ball of omo tuo", "high", ["groundnut_soup", "white_rice"], "rice ~79 g; one moderator food"),
 ("tuo zaafi with ayoyo and guinea fowl", "low", ["tuo_zaafi", "ayoyo", "chicken"], "GI 68 medium, moved down veg + protein"),
 ("3 slices of pawpaw and a banana", "high", ["pawpaw", "banana"], "~86 g carb from fruit"),
 ("kontomire stew with boiled eggs, no yam", "low", ["kontomire", "egg"], "starch negated"),
 ("abolo and grilled fish with pepper sauce", "low", ["abolo", "fish", "stew"], "GI 58 medium, moved down protein + fat"),
 ("a plate of fried rice with chicken and coleslaw and a coke", "high", ["white_rice", "chicken", "vegetables", "soft_drink"], "rice + soft drink; sugar blocks"),
 ("half plate of waakye with fish and salad", "low", ["waakye", "fish", "vegetables"], "~36 g carb medium, moved down"),
 ("100g fufu", "low", ["fufu"], "GI 55 low (GI-only)"),
 ("a medium ball of kenkey", "high", ["kenkey"], "GL ~26"),
 ("gari and beans with fried plantain", "high", ["gari", "beans", "fried_plantain"], "~137 g carb"),
 ("plain rice with okro stew", "high", ["white_rice", "okro"], "~79 g carb, one moderator"),
 ("Is a cup of rice okay?", "medium", ["white_rice"], "cup = 160 g -> ~42 g carb"),
 ("I want to eat jollof but my sugar is 15", "high", ["jollof"], "'my sugar' = blood glucose"),
 ("akyeke with fried fish and shito", "medium", ["akyeke", "fish", "stew"], "predicted GI 79 high (GI-only), moved down for fish + shito"),
]


def main():
    mc = MealClassifier()
    rows, seen = [], set()
    i = 0
    while len(rows) < 155:
        t = compose()
        if t in seen:
            continue
        seen.add(t)
        a = mc.analyse(t)
        rows.append(dict(id=f"meal-t{len(rows):03d}", text=t, label=a.label, label_source="engine", split="train",
                         expected_foods=sorted({m.food_id for m in mc.recognizer.find(t) if not m.negated}), synthetic=True))
    for t in NO_STARCH:
        a = mc.analyse(t)
        rows.append(dict(id=f"meal-t{len(rows):03d}", text=t, label=a.label, label_source="engine", split="train",
                         expected_foods=sorted({m.food_id for m in mc.recognizer.find(t) if not m.negated}), synthetic=True))
    for j, (t, lab, foods, note) in enumerate(HARD):
        rows.append(dict(id=f"meal-h{j:03d}", text=t, label=lab, label_source="manual", split="hard", expected_foods=sorted(foods),
                         note=note, synthetic=True))
    for j, (t, lab, foods, note) in enumerate(HARD_BLIND):
        rows.append(dict(id=f"meal-b{j:03d}", text=t, label=lab, label_source="manual", split="hard_blind", expected_foods=sorted(foods),
                         note=note, synthetic=True))
    out = ROOT / "data" / "meal" / "meal_descriptions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(out, len(rows), Counter((r["split"], r["label"]) for r in rows))


if __name__ == "__main__":
    main()
