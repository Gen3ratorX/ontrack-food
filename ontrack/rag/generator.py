"""Generation backends.

MedGemmaGenerator: google/medgemma-1.5-4b-it via transformers (GPU; 4-bit via bitsandbytes optional).
EchoGenerator: deterministic stand-in for CPU tests and pipeline smoke runs (no model download).
"""
from __future__ import annotations

import gc
import logging
import re

log = logging.getLogger(__name__)


def _strip_thinking(text: str) -> str:
    # Some MedGemma variants can emit a thought block delimited by <unused94> ... <unused95>.
    text = re.sub(r"<unused94>.*?<unused95>", "", text, flags=re.S)
    return text.strip()


# RAG prompts past this many tokens are batched in pairs so left-padding stays small.
_LONG_PROMPT_TOKENS = 1024


def default_batch_size(max_prompt_tokens: int | None = None, *, load_in_4bit: bool = False) -> int:
    """T4 float16 starts at 4 (2 when prompts are long or little memory is free).

    4-bit on a T4 stays at 4 as well: the previous default of 8 OOMs during float16 prefill
    on a 14.5 GiB card, and a smaller batch is the safe shared default.
    """
    try:
        import torch
    except Exception:
        return 1
    if not torch.cuda.is_available():
        return 1
    props = torch.cuda.get_device_properties(0)
    mem_gb = props.total_memory / (1024 ** 3)
    name = torch.cuda.get_device_name(0).lower()
    free_gb = None
    try:
        free, _total = torch.cuda.mem_get_info()
        free_gb = free / (1024 ** 3)
    except Exception:
        free_gb = None
    t4 = "t4" in name or mem_gb >= 14
    if t4:
        bs = 4
    elif mem_gb >= 8:
        bs = 4
    else:
        bs = 1
    if load_in_4bit and t4:
        bs = 4
    if max_prompt_tokens is not None and max_prompt_tokens >= _LONG_PROMPT_TOKENS:
        bs = min(bs, 2)
    if free_gb is not None:
        if free_gb < 1.5:
            bs = 1
        elif free_gb < 3.5:
            bs = min(bs, 2)
    return max(1, bs)


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
        self.batch_size = batch_size if batch_size is not None else default_batch_size(load_in_4bit=load_in_4bit)
        self._oom_logged = False
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
        # padding belongs in processor_kwargs. Passing it as a direct kwarg makes
        # transformers warn: "Kwargs passed to processor.__call__ have to be in processor_kwargs dict".
        try:
            inputs = self.processor.apply_chat_template(
                batch_messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
                processor_kwargs={"padding": True},
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

    def _release_cuda(self) -> None:
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _fit_batch_to_prompts(self, longest: int) -> None:
        """Drop to 2 for long RAG prompts, and further if the GPU is already tight."""
        cap = self.batch_size
        if longest >= _LONG_PROMPT_TOKENS:
            cap = min(cap, 2)
        if self.torch.cuda.is_available():
            try:
                free, _total = self.torch.cuda.mem_get_info()
                free_gb = free / (1024 ** 3)
            except Exception:
                free_gb = None
            if free_gb is not None and free_gb < 1.5:
                cap = 1
            elif free_gb is not None and free_gb < 3.5:
                cap = min(cap, 2)
        if cap < self.batch_size:
            log.info("Prompts up to %s tokens; batch size %s -> %s.", longest, self.batch_size, cap)
            self.batch_size = cap

    def _note_oom(self, new_bs: int) -> None:
        self.batch_size = new_bs
        if not self._oom_logged:
            msg = f"CUDA out of memory; using batch size {new_bs} for the rest of this run."
            log.warning(msg)
            print(msg, flush=True)
            self._oom_logged = True

    def _generate_chunk(self, chunk: list[list[dict]], n_new: int) -> list[str]:
        inputs = None
        generated = None
        try:
            inputs = self._encode(chunk)
            n_prompt = int(inputs["input_ids"].shape[-1])
            pad_id = self.processor.tokenizer.pad_token_id
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs, max_new_tokens=n_new, do_sample=False, pad_token_id=pad_id,
                )
            decode = getattr(self.processor, "batch_decode", None) or self.processor.tokenizer.batch_decode
            texts = [_strip_thinking(t) for t in decode(generated[:, n_prompt:], skip_special_tokens=True)]
            return texts, self._truncation_flags(generated, n_prompt, n_new)
        except self.torch.cuda.OutOfMemoryError:
            inputs = None
            generated = None
            self._release_cuda()
            raise
        finally:
            inputs = None
            generated = None
            gc.collect()

    def _truncation_flags(self, generated, n_prompt: int, n_new: int) -> list[bool]:
        """True when a row used the full new-token budget and did not end on EOS."""
        eos = getattr(self.processor.tokenizer, "eos_token_id", None)
        if eos is None:
            eos_ids: set[int] = set()
        elif isinstance(eos, int):
            eos_ids = {eos}
        else:
            eos_ids = {int(x) for x in eos}
        hit_cap = int(generated.shape[-1]) >= n_prompt + n_new
        flags = []
        for row in generated:
            if not hit_cap:
                flags.append(False)
                continue
            last = int(row[n_prompt + n_new - 1])
            flags.append(last not in eos_ids)
        return flags

    def generate_batch(self, batch_messages: list[list[dict]], max_new_tokens: int | None = None,
                       batch_size: int | None = None) -> list[str]:
        """Generate one completion per conversation.

        Prompts are sorted by length so left-padding is not dominated by one long RAG prompt.
        On CUDA OOM the chunk is retried at half the batch size, down to 1, and that size is kept.
        """
        if not batch_messages:
            return []
        n_new = max_new_tokens or self.max_new_tokens
        lengths = []
        for messages in batch_messages:
            ids = self._encode([messages])["input_ids"]
            lengths.append(int(ids.shape[-1]))
            del ids
        self._fit_batch_to_prompts(max(lengths) if lengths else 0)
        requested = batch_size if batch_size is not None else self.batch_size
        bs = max(1, min(requested, self.batch_size))
        order = sorted(range(len(batch_messages)), key=lambda i: (lengths[i], i))
        ordered = [batch_messages[i] for i in order]
        results: list[str | None] = [None] * len(batch_messages)
        truncated: list[bool] = [False] * len(batch_messages)
        index = 0
        saw_oom = False
        while index < len(ordered):
            this = min(bs, len(ordered) - index)
            chunk = ordered[index:index + this]
            try:
                texts, flags = self._generate_chunk(chunk, n_new)
            except self.torch.cuda.OutOfMemoryError:
                if this == 1:
                    raise
                bs = max(1, this // 2)
                self.batch_size = bs
                saw_oom = True
                continue
            if saw_oom:
                self._note_oom(bs)
                saw_oom = False
            if len(texts) != len(chunk):
                raise RuntimeError(f"generate_batch returned {len(texts)} texts for {len(chunk)} prompts")
            for offset, (text, flag) in enumerate(zip(texts, flags)):
                results[order[index + offset]] = text
                truncated[order[index + offset]] = bool(flag)
            index += this
        self.last_truncated = truncated
        return [t if t is not None else "" for t in results]


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
        self.last_truncated = [False] * len(batch_messages)
        return [self.generate(m, max_new_tokens=max_new_tokens) for m in batch_messages]
