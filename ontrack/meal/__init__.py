"""Meal / food glycaemic-impact classifier (text-based).

recognizer.py : find foods + portions in free text (curated lexicon, local names, fuzzy matching)
engine.py     : evidence engine -> low / medium / high / insufficient_evidence with sources
learned.py    : TF-IDF + logistic regression trained on the synthetic meal dataset
"""
LABELS = ["low", "medium", "high", "insufficient_evidence"]
