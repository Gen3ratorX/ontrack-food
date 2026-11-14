"""Writes knowledge_base/data/food_lexicon.json — the curated food lexicon for the meal classifier.

Rules for editing:
- `gi` may ONLY hold a value that appears in the cited knowledge-base document (`source_id`); a test checks this.
- Carbohydrate/fibre values are NOT stored here; they are read at runtime from wafct2019_selected.csv via `wafct_code`.
- `wafct_proxy: true` means the WAFCT row is the closest available food, not the exact dish.
- `portion` grams are OnTrack ASSUMPTIONS for a typical Ghanaian serving (to be validated by a dietitian).
- Local-language aliases (Twi/Ga/Ewe) are marked in `alias_notes` and need native-speaker review.
"""
import json
from pathlib import Path

E16 = "gi_ghanaian_staples_elicophie2016"; Y19 = "gi_corn_cassava_yeboah2019"; A21 = "gi_yam_ampofo2021"; P20 = "gi_cassava_predicted_2020"
GAP = "gi_evidence_gaps"

def gi(value, source_id, study, kind="measured", note=""):
    return {"value": value, "type": kind, "source_id": source_id, "study": study, "note": note}

BALL = {"ball": 250, "small ball": 150, "default": 250}
PLATE = {"plate": 300, "bowl": 300, "cup": 160, "ladle": 100, "spoon": 60, "default": 300}

F = [
 # ---------------- starchy staples ----------------
 dict(id="kenkey", name="kenkey", category="starch", aliases=["kenkey", "ga kenkey", "fante kenkey", "dokono", "dɔkono", "komi", "kenky"],
      alias_notes="dokono/dɔkono (Twi), komi (Ga)", wafct_code="01_099", gi=gi(41, E16, "Eli-Cophie 2016", note="measured for Ga kenkey; Fante kenkey not tested"),
      portion={"ball": 250, "small ball": 150, "half": 125, "default": 250}, default_unit="ball"),
 dict(id="banku", name="banku", category="starch", aliases=["banku", "banku ball"], wafct_code=None,
      gi=gi(73, E16, "Eli-Cophie 2016", note="corn:cassava dough 80:20"), gi_note="no WAFCT row for banku, so no carbohydrate or GL estimate",
      portion=BALL, default_unit="ball"),
 dict(id="fufu", name="fufu (pounded)", category="starch", aliases=["fufu", "fufuo", "foofoo", "pounded fufu"], wafct_code=None,
      gi=gi(55, E16, "Eli-Cophie 2016", note="locally pounded fufu"), gi_note="no WAFCT row for Ghanaian pounded fufu",
      portion=BALL, default_unit="ball"),
 dict(id="fufu_flour", name="fufu from processed flour", category="starch", aliases=["fufu flour", "fufu powder", "instant fufu", "powdered fufu"],
      wafct_code=None, gi=gi(31, E16, "Eli-Cophie 2016", note="industrial plantain/cassava/potato flour"), portion=BALL, default_unit="ball"),
 dict(id="tuo_zaafi", name="tuo zaafi (TZ)", category="starch", aliases=["tuo zaafi", "tuo zafi", "tuozaafi", "tz", "t.z", "tuo"], wafct_code=None,
      gi=gi(68, E16, "Eli-Cophie 2016"), portion=BALL, default_unit="ball"),
 dict(id="akple", name="akple", category="starch", aliases=["akple", "akplé"], alias_notes="Ewe", wafct_code=None, gi=gi(69, Y19, "Yeboah 2019"),
      portion=BALL, default_unit="ball"),
 dict(id="abolo", name="abolo", category="starch", aliases=["abolo"], alias_notes="Ewe", wafct_code=None, gi=gi(58, Y19, "Yeboah 2019"),
      portion={"piece": 100, "default": 150}, default_unit="piece"),
 dict(id="kafa", name="kafa", category="starch", aliases=["kafa"], alias_notes="Ewe", wafct_code=None, gi=gi(29, Y19, "Yeboah 2019"),
      portion=BALL, default_unit="ball"),
 dict(id="kokonte", name="kokonte", category="starch", aliases=["kokonte", "konkonte", "face the wall"], wafct_code=None, gi=None,
      gi_note="conflicting evidence: measured 7/18 (Yeboah 2019, unusual method) vs predicted 40/61 (in vitro, 2020)", evidence_doc=[Y19, P20],
      portion=BALL, default_unit="ball"),
 dict(id="white_rice", name="white rice", category="starch", aliases=["rice", "white rice", "plain rice", "omo tuo", "rice balls", "fried rice"],
      wafct_code="01_134", gi=None, gi_note="no cited Ghanaian GI", evidence_doc=GAP, portion=PLATE, default_unit="plate"),
 dict(id="brown_rice", name="brown rice", category="starch", aliases=["brown rice"], wafct_code="01_152", gi=None, gi_note="no cited GI",
      portion=PLATE, default_unit="plate"),
 dict(id="jollof", name="jollof rice", category="starch", aliases=["jollof", "jollof rice", "jolof", "jellof", "jolloff"], wafct_code="01_134",
      wafct_proxy=True, proxy_note="carbohydrate from WAFCT boiled white rice (jollof has no WAFCT row)", gi=None,
      gi_note="no measured GI for Ghanaian jollof", evidence_doc=GAP, portion=PLATE, default_unit="plate"),
 dict(id="waakye", name="waakye", category="composite", aliases=["waakye", "waache", "waakyi", "wakye", "waakey"], gi=None,
      gi_note="no measured GI for waakye", evidence_doc=GAP,
      components=[{"id": "white_rice", "grams": 200}, {"id": "beans", "grams": 100}],
      component_note="assumed split per plate: 200 g rice + 100 g beans", portion={"plate": 1.0, "default": 1.0}, default_unit="plate"),
 dict(id="red_red", name="red red (beans stew with fried plantain)", category="composite", aliases=["red red", "redred", "red-red"],
      components=[{"id": "beans", "grams": 200}, {"id": "fried_plantain", "grams": 100}],
      component_note="assumed split: 200 g beans stew + 100 g fried plantain", portion={"plate": 1.0, "default": 1.0}, default_unit="plate"),
 dict(id="gari", name="gari (dry weight)", category="starch", aliases=["gari", "garri", "eba", "gari soakings", "gari soakings"], wafct_code="02_039",
      gi=None, gi_note="no numeric GI in sources", evidence_doc=GAP, portion={"cup": 70, "spoon": 15, "handful": 30, "default": 70}, default_unit="cup"),
 dict(id="yam", name="yam, boiled (variety unspecified)", category="starch", aliases=["yam", "bayere", "bayerɛ", "boiled yam", "yam ampesi"],
      alias_notes="bayerɛ (Twi)", wafct_code="02_020", gi=None, gi_note="GI 44–75 depending on variety (Ampofo 2021) — say white, yellow or water yam",
      evidence_doc=A21, variants={"fried": "fried_yam", "white": "white_yam", "yellow": "yellow_yam", "water": "water_yam"},
      portion={"slice": 100, "piece": 100, "default": 200}, default_unit="piece"),
 dict(id="white_yam", name="white yam, boiled", category="starch", aliases=["white yam", "puna", "pona"], alias_notes="pona/puna: white yam variety",
      wafct_code="02_020", gi=gi(44.26, A21, "Ampofo 2021"), variants={"fried": "fried_white_yam"}, portion={"slice": 100, "piece": 100, "default": 200}, default_unit="piece"),
 dict(id="yellow_yam", name="yellow yam, boiled", category="starch", aliases=["yellow yam"], wafct_code="02_020", wafct_proxy=True,
      proxy_note="WAFCT pale yam row used for carbohydrate", gi=gi(75.18, A21, "Ampofo 2021"), portion={"slice": 100, "piece": 100, "default": 200}, default_unit="piece"),
 dict(id="water_yam", name="water yam, boiled", category="starch", aliases=["water yam", "afase", "avaye"], alias_notes="afase (Twi)", wafct_code="02_018",
      gi=gi(50.12, A21, "Ampofo 2021"), portion={"slice": 100, "piece": 100, "default": 200}, default_unit="piece"),
 dict(id="fried_yam", name="fried yam (variety unspecified)", category="starch", aliases=["fried yam", "yam chips"], wafct_code="02_082", gi=None,
      gi_note="fried yam GI 59–69 by variety (Ampofo 2021)", evidence_doc=A21, fried=True, portion={"piece": 60, "default": 200}, default_unit="piece"),
 dict(id="fried_white_yam", name="fried white yam", category="starch", aliases=["fried white yam"], wafct_code="02_082", gi=gi(59.13, A21, "Ampofo 2021"),
      fried=True, portion={"piece": 60, "default": 200}, default_unit="piece"),
 dict(id="plantain", name="plantain, boiled (ripeness unspecified)", category="starch", aliases=["plantain", "borode", "borɔdeɛ", "brodeɛ", "brodee", "boiled plantain", "plantain ampesi"],
      alias_notes="borɔdeɛ (Twi)", wafct_code="02_057", wafct_proxy=True, proxy_note="ripe plantain values used when ripeness not stated (higher carbohydrate)",
      gi=None, gi_note="no reliable measured GI for plain plantain", evidence_doc=GAP,
      variants={"fried": "fried_plantain", "unripe": "unripe_plantain", "green": "unripe_plantain", "ripe": "ripe_plantain", "roasted": "roasted_plantain"},
      portion={"finger": 150, "piece": 75, "default": 150}, default_unit="finger"),
 dict(id="unripe_plantain", name="unripe (green) plantain, boiled", category="starch", aliases=["green plantain", "unripe plantain"], wafct_code="02_064", gi=None,
      gi_note="no reliable measured GI", evidence_doc=GAP, variants={"fried": "fried_plantain"}, portion={"finger": 150, "piece": 75, "default": 150}, default_unit="finger"),
 dict(id="ripe_plantain", name="ripe plantain, boiled", category="starch", aliases=["ripe plantain"], wafct_code="02_057", gi=None, gi_note="no reliable measured GI",
      evidence_doc=GAP, variants={"fried": "fried_plantain"}, portion={"finger": 150, "piece": 75, "default": 150}, default_unit="finger"),
 dict(id="fried_plantain", name="fried plantain (kelewele / dodo)", category="starch", aliases=["kelewele", "kele wele", "fried plantain", "dodo", "plantain chips"],
      wafct_code="02_084", gi=None, gi_note="no reliable measured GI", evidence_doc=GAP, fried=True, portion={"plate": 150, "piece": 30, "default": 100}, default_unit="plate"),
 dict(id="roasted_plantain", name="roasted plantain (kofi brokeman / boli)", category="starch", aliases=["kofi brokeman", "roasted plantain", "roast plantain", "boli"],
      wafct_code="02_057", wafct_proxy=True, proxy_note="no WAFCT row for roasted plantain; boiled ripe plantain used", gi=None, evidence_doc=GAP,
      portion={"finger": 150, "piece": 75, "default": 150}, default_unit="finger"),
 dict(id="cassava", name="cassava, boiled", category="starch", aliases=["cassava", "bankye", "boiled cassava", "cassava ampesi"], alias_notes="bankye (Twi)",
      wafct_code="02_003", gi=gi(77.30, P20, "J Food Nutr Sci 2020", kind="predicted", note="in-vitro predicted GI for boiled cassava (ampesi)"),
      portion={"piece": 100, "default": 200}, default_unit="piece"),
 dict(id="akyeke", name="akyeke (steamed fermented cassava grits)", category="starch", aliases=["akyeke", "attieke", "acheke"], wafct_code=None,
      gi=gi(79.05, P20, "J Food Nutr Sci 2020", kind="predicted", note="in-vitro predicted GI"), gi_note="no WAFCT row for akyeke",
      portion={"plate": 250, "default": 250}, default_unit="plate"),
 dict(id="cocoyam", name="cocoyam, boiled", category="starch", aliases=["cocoyam", "mankani", "mankaninini"], alias_notes="mankani (Twi)", wafct_code="02_006",
      gi=None, gi_note="no cited GI", portion={"piece": 100, "default": 200}, default_unit="piece"),
 dict(id="ampesi", name="ampesi (boiled starchy food, type unspecified)", category="starch", aliases=["ampesi"], wafct_code=None, gi=None,
      gi_note="say which food (yam, plantain, cocoyam, cassava)", drop_if_cooccurs=["yam", "white_yam", "yellow_yam", "water_yam", "plantain", "unripe_plantain", "ripe_plantain", "cassava", "cocoyam"],
      portion={"default": 250}, default_unit="plate"),
 dict(id="koko", name="koko / Hausa koko (no sugar)", category="starch", aliases=["koko", "hausa koko", "millet porridge", "kooko"], wafct_code="01_102",
      gi=None, gi_note="no cited GI; sugar added at the stall raises it", portion={"cup": 300, "bowl": 350, "default": 300}, default_unit="cup"),
 dict(id="tom_brown", name="tom brown", category="starch", aliases=["tom brown", "tombrown"], wafct_code=None, gi=None, gi_note="no data", portion={"cup": 300, "default": 300}, default_unit="cup"),
 dict(id="rice_water", name="rice water (rice porridge)", category="starch", aliases=["rice water"], wafct_code=None, gi=None, gi_note="no data", portion={"cup": 300, "default": 300}, default_unit="cup"),
 dict(id="oats", name="oats porridge", category="starch", aliases=["oats", "oat porridge", "oatmeal", "quaker oats"], wafct_code="01_133", gi=None, gi_note="no cited GI",
      portion={"bowl": 250, "cup": 250, "default": 250}, default_unit="bowl"),
 dict(id="white_bread", name="white bread", category="starch", aliases=["bread", "white bread", "sugar bread", "tea bread", "butter bread", "agege bread"],
      wafct_code="01_046", wafct_proxy=True, proxy_note="unfortified white bread row; sugar bread has added sugar", gi=None, gi_note="no cited GI",
      portion={"slice": 30, "loaf": 400, "small loaf": 100, "default": 60}, default_unit="slice"),
 dict(id="wholemeal_bread", name="wholemeal bread", category="starch", aliases=["wholemeal bread", "whole wheat bread", "brown bread", "wheat bread"], wafct_code="01_048",
      gi=None, gi_note="no cited GI", portion={"slice": 30, "default": 60}, default_unit="slice"),
 dict(id="spaghetti", name="spaghetti / talia", category="starch", aliases=["spaghetti", "talia", "macaroni", "pasta"], wafct_code="01_053", wafct_proxy=True,
      proxy_note="boiled macaroni row", gi=None, gi_note="no cited GI", portion={"plate": 200, "default": 100}, default_unit="plate"),
 dict(id="biscuit", name="sweet biscuit", category="sugar", aliases=["biscuit", "biscuits", "cookies"], wafct_code="01_188", gi=None, portion={"pack": 50, "piece": 10, "default": 50}, default_unit="pack"),
 # ---------------- legumes & nuts ----------------
 dict(id="beans", name="beans / cowpea, boiled", category="legume", aliases=["beans", "black eyed beans", "black-eyed beans", "cowpea", "cowpeas", "bean stew"],
      wafct_code="03_007", gi=None, gi_note="no cited GI", portion={"plate": 200, "ladle": 150, "default": 150}, default_unit="ladle"),
 dict(id="koose", name="koose (bean fritters)", category="legume", aliases=["koose", "kose", "akara", "kosai"], wafct_code=None, fried=True, gi=None,
      portion={"piece": 20, "default": 60}, default_unit="piece", moderators=["protein", "legume", "fat"]),
 dict(id="soya", name="soya beans, boiled", category="legume", aliases=["soya", "soya beans", "soybeans", "soy beans"], wafct_code="03_040", gi=None, portion={"default": 100}, default_unit="cup"),
 dict(id="groundnut", name="groundnuts", category="legume", aliases=["groundnut", "groundnuts", "peanut", "peanuts", "nkatie", "nkate"], alias_notes="nkatie (Twi)",
      wafct_code="06_025", gi=None, portion={"handful": 30, "default": 30}, default_unit="handful", moderators=["legume", "fat", "protein"]),
 dict(id="bambara", name="Bambara beans (aboboi)", category="legume", aliases=["aboboi", "bambara beans", "bambara groundnut"], wafct_code="03_019", gi=None, portion={"default": 150}, default_unit="ladle"),
 # ---------------- vegetables & soups ----------------
 dict(id="kontomire", name="kontomire (cocoyam leaves)", category="vegetable", aliases=["kontomire", "kontommire", "nkontommire", "kontomire stew", "palava sauce", "cocoyam leaves"],
      alias_notes="nkontommire (Twi)", wafct_code="04_029", gi=None, portion={"ladle": 150, "default": 150}, default_unit="ladle"),
 dict(id="garden_egg", name="garden eggs", category="vegetable", aliases=["garden egg", "garden eggs", "nyaadewa", "garden egg stew", "eggplant"], alias_notes="nyaadewa (Twi)",
      wafct_code="04_034", gi=None, portion={"default": 150}, default_unit="ladle"),
 dict(id="okro", name="okro (okra)", category="vegetable", aliases=["okro", "okra", "nkruma", "okro stew", "okro soup", "okra soup"], alias_notes="nkruma (Twi)",
      wafct_code="04_042", gi=None, portion={"default": 150}, default_unit="ladle"),
 dict(id="ayoyo", name="ayoyo (jute mallow) leaves", category="vegetable", aliases=["ayoyo", "ayoyo soup", "jute leaves", "jute mallow"], wafct_code="04_039", gi=None, portion={"default": 150}, default_unit="ladle"),
 dict(id="vegetables", name="vegetables / salad", category="vegetable", aliases=["salad", "vegetables", "veggies", "cabbage", "carrots", "lettuce", "cucumber", "green beans", "spinach", "coleslaw", "veg"],
      wafct_code=None, gi=None, portion={"default": 100}, default_unit="cup"),
 dict(id="light_soup", name="light soup", category="soup", aliases=["light soup", "pepper soup", "nkrakra"], alias_notes="nkrakra (Twi)", wafct_code=None, gi=None,
      portion={"default": 300}, default_unit="bowl", moderators=["vegetable"]),
 dict(id="groundnut_soup", name="groundnut soup", category="soup", aliases=["groundnut soup", "nkate nkwan", "nkatenkwan", "peanut soup"], wafct_code=None, gi=None,
      portion={"default": 300}, default_unit="bowl", moderators=["fat", "protein"]),
 dict(id="palmnut_soup", name="palm nut soup", category="soup", aliases=["palm nut soup", "palmnut soup", "abenkwan", "abɛnkwan", "banga soup"], wafct_code=None, gi=None,
      portion={"default": 300}, default_unit="bowl", moderators=["fat"]),
 dict(id="stew", name="tomato stew / shito", category="fat", aliases=["stew", "tomato stew", "shito", "pepper sauce", "gravy"], wafct_code=None, gi=None, portion={"default": 80}, default_unit="ladle"),
 dict(id="palm_oil", name="palm oil", category="fat", aliases=["palm oil", "zomi", "red oil"], wafct_code="11_004", gi=None, portion={"spoon": 15, "default": 15}, default_unit="spoon"),
 dict(id="avocado", name="avocado (pear)", category="fat", aliases=["avocado", "pear", "avocado pear"], wafct_code="05_002", gi=None, portion={"half": 100, "default": 100}, default_unit="half"),
 # ---------------- protein ----------------
 dict(id="fish", name="fish (e.g. tilapia, grilled)", category="protein", aliases=["fish", "tilapia", "grilled tilapia", "fried fish", "mackerel", "salmon", "herring", "kpanla", "keta school boys", "momoni", "tuna", "nam"],
      alias_notes="nam (Twi) = fish or meat; mapped to fish",
      wafct_code="09_043", wafct_proxy=True, proxy_note="grilled tilapia row", gi=None, portion={"piece": 100, "default": 100}, default_unit="piece"),
 dict(id="sardine", name="sardines", category="protein", aliases=["sardine", "sardines"], wafct_code="09_037", gi=None, portion={"tin": 100, "default": 100}, default_unit="tin"),
 dict(id="chicken", name="chicken", category="protein", aliases=["chicken", "fried chicken", "grilled chicken", "chicken thigh", "guinea fowl"], wafct_code="07_035", wafct_proxy=True,
      proxy_note="grilled light meat without skin", gi=None, portion={"piece": 100, "default": 100}, default_unit="piece"),
 dict(id="meat", name="meat (beef/goat)", category="protein", aliases=["beef", "meat", "goat", "goat meat", "chevon", "wele", "kpomo", "mutton", "khebab", "kebab", "chofi"],
      wafct_code="07_011", wafct_proxy=True, proxy_note="lean grilled beef row", gi=None, portion={"piece": 50, "default": 100}, default_unit="piece"),
 dict(id="egg", name="egg", category="protein", aliases=["egg", "eggs", "boiled egg", "boiled eggs", "fried egg", "omelette", "omelet"], wafct_code="08_002", gi=None,
      portion={"egg": 50, "default": 50}, default_unit="egg"),
 # ---------------- fruit ----------------
 dict(id="mango", name="mango", category="fruit", aliases=["mango", "mangoes"], wafct_code="05_015", gi=None, portion={"default": 200}, default_unit="fruit"),
 dict(id="pawpaw", name="pawpaw (papaya)", category="fruit", aliases=["pawpaw", "papaya"], wafct_code="05_017", gi=None, portion={"slice": 150, "default": 150}, default_unit="slice"),
 dict(id="pineapple", name="pineapple", category="fruit", aliases=["pineapple", "pineapples", "abrobe"], wafct_code="05_018", gi=None, portion={"slice": 100, "default": 100}, default_unit="slice"),
 dict(id="orange", name="orange", category="fruit", aliases=["orange", "oranges", "ankaa"], alias_notes="ankaa (Twi)", wafct_code="05_016", gi=None, portion={"default": 130}, default_unit="fruit"),
 dict(id="watermelon", name="watermelon", category="fruit", aliases=["watermelon"], wafct_code="05_022", gi=None, portion={"slice": 200, "default": 200}, default_unit="slice"),
 dict(id="banana", name="banana", category="fruit", aliases=["banana", "bananas", "kwadu"], alias_notes="kwadu (Twi)", wafct_code="05_003", gi=None, portion={"default": 100}, default_unit="fruit"),
 # ---------------- drinks & sugar ----------------
 dict(id="soft_drink", name="soft drink (cola/Fanta/Sprite)", category="sugary_drink", aliases=["coke", "coca cola", "coca-cola", "fanta", "sprite", "soft drink", "soda", "minerals", "mineral", "pepsi"],
      alias_notes="'minerals' = soft drinks in Ghanaian English", wafct_code="12_024", gi=None, portion={"bottle": 350, "can": 330, "glass": 250, "default": 350}, default_unit="bottle"),
 dict(id="malt", name="malt drink", category="sugary_drink", aliases=["malt", "malta guinness", "malta", "alvaro"], wafct_code=None, gi=None, gi_note="no data; sugar-sweetened",
      portion={"bottle": 330, "default": 330}, default_unit="bottle"),
 dict(id="sobolo", name="sobolo (sweetened)", category="sugary_drink", aliases=["sobolo", "bissap", "zobo"], wafct_code=None, gi=None, gi_note="no data; usually sweetened",
      portion={"default": 350}, default_unit="bottle"),
 dict(id="juice", name="fruit juice", category="sugary_drink", aliases=["juice", "fruit juice", "orange juice", "pineapple juice"], wafct_code="12_011", wafct_proxy=True,
      proxy_note="fresh unsweetened orange juice row", gi=None, portion={"glass": 250, "cup": 250, "default": 250}, default_unit="glass"),
 dict(id="sugar", name="added sugar", category="sugar", aliases=["sugar", "with sugar"], wafct_code="13_002", gi=None,
      portion={"teaspoon": 4, "tablespoon": 12, "spoon": 4, "cube": 4, "default": 8}, default_unit="teaspoon"),
 dict(id="honey", name="honey", category="sugar", aliases=["honey"], wafct_code="13_001", gi=None, portion={"tablespoon": 20, "spoon": 20, "teaspoon": 7, "default": 20}, default_unit="tablespoon"),
 dict(id="milo", name="Milo / chocolate drink", category="sugary_drink", aliases=["milo", "bournvita", "cocoa drink", "hot chocolate"], wafct_code=None, gi=None, gi_note="no data", portion={"default": 250}, default_unit="cup"),
 dict(id="tea", name="tea (no sugar)", category="drink_zero", aliases=["tea", "lipton", "green tea"], wafct_code="12_008", gi=None, portion={"default": 250}, default_unit="cup"),
 dict(id="water", name="water", category="drink_zero", aliases=["water", "sachet water", "pure water"], wafct_code=None, gi=None, portion={"default": 500}, default_unit="sachet"),
]

DEFAULT_MOD = {"protein": ["protein"], "legume": ["legume", "protein"], "vegetable": ["vegetable"], "fat": ["fat"], "soup": [], "fruit": [], "starch": [], "composite": [],
               "sugary_drink": [], "sugar": [], "drink_zero": []}

def main():
    out = []
    for e in F:
        e = dict(e)
        e.setdefault("wafct_code", None); e.setdefault("wafct_proxy", False); e.setdefault("gi", None); e.setdefault("fried", False)
        e.setdefault("moderators", DEFAULT_MOD[e["category"]] + (["fat"] if e["fried"] and "fat" not in DEFAULT_MOD[e["category"]] else []))
        out.append(e)
    ids = [e["id"] for e in out]
    assert len(ids) == len(set(ids)), "duplicate ids"
    p = Path(__file__).resolve().parents[1] / "knowledge_base" / "data" / "food_lexicon.json"
    p.write_text(json.dumps({"version": "0.2", "note": __doc__.strip(), "foods": out}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(p, len(out), "foods,", sum(len(e["aliases"]) for e in out), "aliases")

if __name__ == "__main__":
    main()
