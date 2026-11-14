"""Central configuration. Override with environment variables when running on Colab."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # Generation model (gated on Hugging Face: accept the Health AI Developer Foundations terms first)
    model_id: str = os.environ.get("ONTRACK_MODEL_ID", "google/medgemma-1.5-4b-it")
    load_in_4bit: bool = os.environ.get("ONTRACK_4BIT", "1") == "1"
    max_new_tokens: int = int(os.environ.get("ONTRACK_MAX_NEW_TOKENS", "400"))
    # Retrieval
    embedding_model: str = os.environ.get("ONTRACK_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = 900  # characters
    chunk_overlap: int = 150
    top_k: int = 5
    min_score: float = 0.25  # cosine similarity floor for a chunk to be used
    # Paths
    kb_dir: Path = field(default_factory=lambda: ROOT / "knowledge_base" / "docs")
    index_dir: Path = field(default_factory=lambda: ROOT / "artifacts" / "index")
    classifier_path: Path = field(default_factory=lambda: ROOT / "artifacts" / "triage_tfidf_lr.joblib")
    meal_model_path: Path = field(default_factory=lambda: ROOT / "artifacts" / "meal_tfidf_lr.joblib")
    meal_data: Path = field(default_factory=lambda: ROOT / "data" / "meal" / "meal_descriptions.jsonl")
    triage_data: Path = field(default_factory=lambda: ROOT / "data" / "classifier" / "triage_seed.jsonl")
    qa_data: Path = field(default_factory=lambda: ROOT / "data" / "eval" / "qa_batch1.jsonl")
    splits_dir: Path = field(default_factory=lambda: ROOT / "data" / "eval" / "splits")
    runs_dir: Path = field(default_factory=lambda: ROOT / "artifacts" / "runs")


CONFIG = Config()
