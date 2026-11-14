---
id: afrimedqa_dataset_card
title: AfriMed-QA v2 dataset (evaluation resource, not clinical guidance)
publisher: Intron Health, SisonkeBiotik, BioRAMP, Georgia Tech, MasakhaneNLP, Google Research; Olatunji et al. arXiv:2411.15640
url: https://huggingface.co/datasets/intronhealth/afrimedqa_v2
licence: CC BY-SA 4.0 (gated on Hugging Face: accept terms to download)
accessed: 2026-11-14
topic: dataset, evaluation
index: false
---
# AfriMed-QA v2

A pan-African medical QA dataset of 15,275 questions from 621 contributors in 16 countries (including Ghana) across 32+ specialties, including Endocrinology (236 questions). Types: consumer queries (10,000), expert MCQ (3,910), expert SAQ (359), crowdsourced SAQ (877), crowdsourced MCQ (129). Key columns: question / question_clean, prompt (consumer queries), answer_options, correct_answer, answer_rationale, specialty, country, split, plus reviewer ratings (rated_correct, rated_harmful, etc.).

OnTrack uses it only as a source of diabetes-related evaluation questions (filtered by keyword and specialty), not as knowledge for answers. The CSV is afri_med_qa_15k_v2.4_phase_2_15275.csv. Load with `datasets.load_dataset("intronhealth/afrimedqa_v2")` after accepting the terms with a Hugging Face token.
