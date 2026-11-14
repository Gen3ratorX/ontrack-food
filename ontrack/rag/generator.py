"""Generation backends.

MedGemmaGenerator: google/medgemma-1.5-4b-it via transformers (GPU; 4-bit via bitsandbytes optional).
EchoGenerator: deterministic stand-in for CPU tests and pipeline smoke runs (no model download).
"""
from __future__ import annotations

import re


def _strip_thinking(text: str) -> str:
    # Some MedGemma variants can emit a thought block delimited by <unused94> ... <unused95>.
    text = re.sub(r"<unused94>.*?<unused95>", "", text, flags=re.S)
    return text.strip()


class MedGemmaGenerator:
    def __init__(self, model_id: str = "google/medgemma-1.5-4b-it", load_in_4bit: bool = True,
                 max_new_tokens: int = 400, dtype: str = "auto"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if dtype == "auto":
            # T4 (compute capability 7.5) has no native bfloat16 -> use float16 there.
            bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            torch_dtype = torch.bfloat16 if bf16_ok else torch.float16
        else:
            torch_dtype = getattr(torch, dtype)
        import transformers
        from packaging.version import Version

        dtype_key = "dtype" if Version(transformers.__version__) >= Version("4.56.0") else "torch_dtype"
        kwargs = {"device_map": "auto", dtype_key: torch_dtype}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype, bnb_4bit_use_double_quant=True,
            )
        self.torch = torch
        self.torch_dtype = torch_dtype
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        self.model.eval()

    def generate(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        n = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False)
        return _strip_thinking(self.processor.decode(out[0][n:], skip_special_tokens=True))


class EchoGenerator:
    """Offline stand-in: returns a canned educational answer quoting the first source. For tests only."""

    model_id = "echo-test-generator"

    def generate(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        user = messages[-1]["content"][0]["text"]
        meal = re.search(r"Rating: (\w+) expected blood-sugar impact", user)
        if meal:
            return (f"(test generator) Meal rating: {meal.group(1).lower()} [M]. These are estimates based on assumed portions. "
                    "Please check with your health worker.")
        m = re.search(r"\[S1\] Source: (.*?)\n(.*?)(\n\n\[S2\]|\n\nQuestion:)", user, flags=re.S)
        if m:
            snippet = " ".join(m.group(2).split()[:40])
            return f"(test generator) Based on {m.group(1)}: {snippet} [S1]. Please check with your health worker."
        return "(test generator) General guidance: watch portion sizes and use the plate method. Please check with your health worker."
