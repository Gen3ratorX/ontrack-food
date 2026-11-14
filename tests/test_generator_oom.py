"""CUDA OOM in generate_batch halves the batch and keeps prompt order."""
import torch

from ontrack.rag.generator import MedGemmaGenerator


def test_generate_batch_oom_halves_and_restores_order(capsys):
    lengths = {"tiny": 2, "short": 4, "mid": 9, "medium": 14, "long": 28}
    codes = {name: i + 1 for i, name in enumerate(lengths)}

    gen = MedGemmaGenerator.__new__(MedGemmaGenerator)
    gen.torch = torch
    gen.batch_size = 4
    gen._oom_logged = False
    gen.max_new_tokens = 4
    gen.processor = type("P", (), {})()

    def encode(chunk):
        width = max(lengths[m[-1]["content"][0]["text"]] for m in chunk)
        rows = []
        for m in chunk:
            name = m[-1]["content"][0]["text"]
            row = torch.zeros(width, dtype=torch.long)
            row[-1] = codes[name]
            rows.append(row)
        ids = torch.stack(rows)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    gen._encode = encode

    class Model:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            n = int(kwargs["input_ids"].shape[0])
            self.calls.append(n)
            if n > 2:
                raise torch.cuda.OutOfMemoryError(f"simulated OOM for batch {n}")
            extra = kwargs["input_ids"][:, -1:].clone()
            return torch.cat([kwargs["input_ids"], extra], dim=1)

    gen.model = Model()
    gen.processor.tokenizer = type("T", (), {"pad_token_id": 0})()
    gen.processor.batch_decode = lambda tokens, skip_special_tokens=True: [
        next(name for name, code in codes.items() if code == int(row[0])) for row in tokens
    ]

    prompts = ["long", "tiny", "medium", "short", "mid"]
    messages = [[{"role": "user", "content": [{"type": "text", "text": name}]}] for name in prompts]
    out = gen.generate_batch(messages, batch_size=4)

    assert out == prompts
    assert all(n <= 2 for n in gen.model.calls[1:])
    assert any(n > 2 for n in gen.model.calls)
    assert gen.batch_size == 2
    # A second call stays at the reduced size and does not log again.
    gen.model.calls.clear()
    again = gen.generate_batch(messages, batch_size=8)
    assert again == prompts
    assert gen.model.calls
    assert max(gen.model.calls) <= 2
    logged = [line for line in capsys.readouterr().out.splitlines() if "CUDA out of memory" in line]
    assert logged == ["CUDA out of memory; using batch size 2 for the rest of this run."]
