"""Writes data/eval/qa_batch2_meals.jsonl: meal-classification evaluation items (synthetic, draft).

These are NOT merged into the frozen 60-item test split (data/eval/splits/test.jsonl); they are scored as a
separate 'meals' set. Extra field `expected_meal_label` = the rating the meal classifier should give,
used by the rubric's meal criterion. Reference numbers come from the engine output (estimates, assumed portions).
"""
import json
from pathlib import Path

ORIGIN = "OnTrack team synthetic batch 2 (meal classifier; AI-assisted draft)"
ITEMS = [
 ("I'm about to eat 2 balls of kenkey with fried fish and shito. How will it affect my blood sugar?",
  "High expected impact (estimate). Ga kenkey has a low GI of 41 in a Ghanaian study, but two balls (about 500 g, an assumed portion) give roughly 126 g of carbohydrate based on FAO/INFOODS WAFCT 2019, so the estimated glycaemic load is about 52, which is high (20 or more). A low GI does not make a big portion free. Try one smaller ball, keep the fish, add vegetables such as garden egg or salad, and use less oily shito.",
  "high", ["kenkey", "fish"], ["gi_ghanaian_staples_elicophie2016", "wafct2019_ghana_foods", "glycaemic_load_method", "cdc_meal_planning"]),
 ("For lunch I want half a ball of banku with okro stew and grilled tilapia. Is that okay?",
  "Medium expected impact. Banku had a high GI of 73 in a Ghanaian study, and there is no WAFCT carbohydrate value for banku, so the amount of carbohydrate cannot be estimated. A half ball with okro (a non-starchy vegetable) and grilled fish (protein) moves the rating down one level, from high to medium. This is a sensible plate: keep the banku portion small and fill half the plate with okro or other vegetables.",
  "medium", ["banku", "okro", "fish"], ["gi_ghanaian_staples_elicophie2016", "glycaemic_load_method", "cdc_meal_planning"]),
 ("Rate this meal: a plate of jollof rice with fried chicken, salad and a bottle of Coke.",
  "High expected impact (estimate). There is no measured GI for Ghanaian jollof, so the rating is based on carbohydrate only: a plate of rice (about 300 g, assumed) gives around 79 g of carbohydrate and a bottle of cola around 35 g (WAFCT 2019). The sugary drink raises blood sugar quickly and cancels the benefit of the chicken and salad. Choose water instead of Coke, take a smaller portion of jollof, and make salad half the plate.",
  "high", ["jollof", "chicken", "vegetables", "soft_drink"], ["wafct2019_ghana_foods", "gi_evidence_gaps", "cdc_healthy_carbs", "cdc_meal_planning"]),
 ("Is kokonte with groundnut soup good for my sugar?",
  "There is not enough evidence to rate this. Published GI values for kokonte conflict (very low values in one small study using an unusual method, versus moderate in-vitro predictions) and there is no WAFCT carbohydrate value for kokonte, so OnTrack does not give a number. Keep the portion moderate, add vegetables and fish or lean meat, watch the oil in groundnut soup, and check your blood sugar about 2 hours after eating to see how it affects you.",
  "insufficient_evidence", ["kokonte", "groundnut_soup"], ["gi_corn_cassava_yeboah2019", "gi_cassava_predicted_2020", "glycaemic_load_method"]),
 ("Breakfast was koko without sugar and 3 koose. How is that for my diabetes?",
  "Low expected impact for the koko part, with a caveat. A cup of Hausa koko without sugar (about 300 g, assumed) gives roughly 27 g of carbohydrate (WAFCT 2019), which is low on OnTrack's carbohydrate scale. Koose are bean fritters (protein and fibre, but fried); there is no WAFCT value for them, so the rating may underestimate the meal. Keeping the koko unsweetened is a good choice.",
  "low", ["koko", "koose"], ["wafct2019_ghana_foods", "glycaemic_load_method", "cdc_healthy_carbs"]),
 ("I had boiled white yam with kontomire stew and an egg. What impact will it have?",
  "Medium expected impact (estimate). Boiled white yam had a GI of about 44 in a Ghanaian study (Ampofo 2021), but a typical serving of about 200 g (assumed) has roughly 54 g of carbohydrate (WAFCT 2019), so its estimated glycaemic load is about 24, which is high. Kontomire (a leafy vegetable) and egg (protein) move the meal down one level to medium. A smaller piece of yam and more kontomire would lower it further.",
  "medium", ["white_yam", "kontomire", "egg"], ["gi_yam_ampofo2021", "wafct2019_ghana_foods", "glycaemic_load_method", "cdc_meal_planning"]),
 ("Me di bayerɛ ampesi ne nkontommire. Is that okay for my sugar?",
  "Medium expected impact (estimate). Yam ampesi with kontomire is a common meal. The yam variety is not stated and GI differs by variety (about 44 to 75 when boiled), so the rating uses carbohydrate only: about 54 g from a 200 g serving (assumed, WAFCT 2019), which is medium. Adding fish or egg and keeping the yam portion moderate helps. Tell OnTrack whether it is white, water or yellow yam for a more specific estimate.",
  "medium", ["yam", "kontomire"], ["wafct2019_ghana_foods", "gi_yam_ampofo2021", "glycaemic_load_method", "cdc_healthy_carbs"]),
]


def main():
    out = Path(__file__).resolve().parents[1] / "data" / "eval" / "qa_batch2_meals.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for i, (q, ref, lab, foods, srcs) in enumerate(ITEMS, 1):
            f.write(json.dumps({"id": f"qa2-meal-{i:03d}", "question": q, "reference_answer": ref, "category": "meal_classification",
                                "food_tags": foods, "culture_tag": "ghanaian", "risk_level": "low", "expected_triage": "general_education",
                                "expected_meal_label": lab, "source_ids": srcs, "synthetic": True, "origin": ORIGIN,
                                "review_status": "draft"}, ensure_ascii=False) + "\n")
    print(out, len(ITEMS))


if __name__ == "__main__":
    main()
