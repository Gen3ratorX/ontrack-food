import pytest


@pytest.mark.parametrize("query,expected_doc", [
    ("What is the glycaemic index of banku?", "gi_ghanaian_staples_elicophie2016"),
    ("Is kenkey low GI?", "gi_ghanaian_staples_elicophie2016"),
    ("vomiting, fruity breath and fast breathing", "cdc_dka"),
    ("what is the 15-15 rule for low sugar", "cdc_treat_low_blood_sugar"),
    ("does boiled yam have lower GI than fried yam", "gi_yam_ampofo2021"),
    ("how much carbohydrate is in gari", "wafct2019_ghana_foods"),
    ("is kontomire good for diabetes", "wafct2019_ghana_foods"),
    ("what number do I call for an ambulance in Ghana", "ghana_emergency_contacts"),
    ("how should I care for my feet", "cdc_feet"),
    ("what is the GI of waakye", "gi_evidence_gaps"),
])
def test_expected_source_in_top5(retriever, query, expected_doc):
    docs = [h.doc_id for h in retriever.search(query, k=5)]
    assert expected_doc in docs, docs


def test_scores_sorted_and_formatted(retriever):
    from ontrack.rag.retriever import format_context

    hits = retriever.search("plate method", k=3)
    assert 1 <= len(hits) <= 3
    ctx = format_context(hits)
    assert ctx.startswith("[S1] Source:") and "http" in ctx


def test_irrelevant_query_returns_few_or_none(retriever):
    hits = retriever.search("xqzv blorp")
    assert len(hits) <= 5


def test_retrieval_hit_rate_on_qa_set(retriever):
    import json

    from ontrack.config import CONFIG
    from ontrack.eval.retrieval_eval import evaluate

    items = [json.loads(l) for l in open(CONFIG.qa_data, encoding="utf-8")]
    res = evaluate(retriever, items, k=5)
    assert res["hit_at_k"] >= 0.85, res["misses"]
