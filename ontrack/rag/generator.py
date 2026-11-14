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


def default_batch_size() -> int:
    """8 on a T4 (or any GPU with at least 14 GB). Smaller GPUs get a smaller batch."""
    try:
        import torch
    except Exception:
        return 1
    if not torch.cuda.is_available():
        return 1
    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    name = torch.cuda.get_device_name(0).lower()
    if "t4" in name or mem_gb >= 14:
        return 8
    if mem_gb >= 8:
        return 4
    return 1


def prefer_4bit(force: bool | None = None) -> bool:
    """Float16 MedGemma 4B is about 8.6 GB and fits in a 15 GB T4, and it is faster than 4-bit.

    Pass True/False to force a choice. With force=None, use 4-bit only when a GPU is present
    and has under 14 GB of memory.
    """
    if force is not None:
        return force
    try:
        import torch
        if torch.cuda.is_available():
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return mem_gb < 14
    except Exception:
        pass
    return False


class MedGemmaGenerator:
    def __init__(self, model_id: str = "google/medgemma-1.5-4b-it", load_in_4bit: bool = True,
                 max_new_tokens: int = 400, dtype: str = "auto", batch_size: int | None = None):
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
        self.batch_size = batch_size or default_batch_size()
        self.load_in_4bit = load_in_4bit
        self.processor = AutoProcessor.from_pretrained(model_id)
        tok = self.processor.tokenizer
        # Decoder-only generation needs left padding so the last prompt token lines up in a batch.
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        self.model.eval()

    def generate(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        return self.generate_batch([messages], max_new_tokens=max_new_tokens, batch_size=1)[0]

    def _encode(self, batch_messages: list[list[dict]]):
        try:
            inputs = self.processor.apply_chat_template(
                batch_messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True,
            )
        except (TypeError, ValueError):
            texts = [self.processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                     for m in batch_messages]
            inputs = self.processor.tokenizer(texts, return_tensors="pt", padding=True)
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)
        else:
            inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        keep = {k: inputs[k] for k in ("input_ids", "attention_mask") if k in inputs}
        return keep or inputs

    def generate_batch(self, batch_messages: list[list[dict]], max_new_tokens: int | None = None,
                       batch_size: int | None = None) -> list[str]:
        """Generate one completion per conversation. Left-padded, in chunks of `batch_size`."""
        if not batch_messages:
            return []
        bs = batch_size or self.batch_size
        n_new = max_new_tokens or self.max_new_tokens
        decode = getattr(self.processor, "batch_decode", None) or self.processor.tokenizer.batch_decode
        pad_id = self.processor.tokenizer.pad_token_id
        outs: list[str] = []
        for i in range(0, len(batch_messages), bs):
            chunk = batch_messages[i:i + bs]
            inputs = self._encode(chunk)
            n_prompt = inputs["input_ids"].shape[-1]
            with self.torch.inference_mode():
                gen = self.model.generate(**inputs, max_new_tokens=n_new, do_sample=False, pad_token_id=pad_id)
            texts = decode(gen[:, n_prompt:], skip_special_tokens=True)
            outs.extend(_strip_thinking(t) for t in texts)
        return outs


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

    def generate_batch(self, batch_messages: list[list[dict]], max_new_tokens: int | None = None,
                       batch_size: int | None = None) -> list[str]:
        return [self.generate(m, max_new_tokens=max_new_tokens) for m in batch_messages]
