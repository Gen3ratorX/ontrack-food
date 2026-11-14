"""Exercises the MedGemma code path (processor chat template + generate) with a tiny random Gemma 3 model.
Opt-in because it downloads ~20 MB: ONTRACK_TEST_TINY_GEMMA=1 pytest tests/test_generator_tiny.py"""
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("ONTRACK_TEST_TINY_GEMMA") != "1", reason="opt-in network test")


def test_tiny_gemma3_generates():
    from ontrack.rag.generator import MedGemmaGenerator
    from ontrack.rag.prompts import build_messages

    g = MedGemmaGenerator("trl-internal-testing/tiny-Gemma3ForConditionalGeneration", load_in_4bit=False,
                          max_new_tokens=5, dtype="float32")
    out = g.generate(build_messages("Is kenkey okay?", "[S1] Source: test\nkenkey"))
    assert isinstance(out, str)
