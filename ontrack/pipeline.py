"""OnTrack assistant: triage gate -> (template | RAG generation | no-RAG generation)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .classifier.gate import SafetyGate
from .classifier.templates import template_for
from .rag.prompts import build_messages
from .rag.retriever import Retriever, format_context


@dataclass
class AssistantResponse:
    question: str
    answer: str
    mode: str  # "rag" | "no_rag" | "template"
    triage_label: str
    triage_reason: str
    sources: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class OnTrackAssistant:
    def __init__(self, generator, retriever: Retriever | None = None, gate: SafetyGate | None = None, use_rag: bool = True):
        if use_rag and retriever is None:
            raise ValueError("RAG mode needs a retriever")
        self.generator, self.retriever, self.gate, self.use_rag = generator, retriever, gate, use_rag

    def answer(self, question: str) -> AssistantResponse:
        label, reason = ("general_education", "gate disabled")
        if self.gate is not None:
            label, reason = self.gate.classify(question)
            if label != "general_education":
                return AssistantResponse(question, template_for(label), "template", label, reason)
        if self.use_rag:
            hits = self.retriever.search(question)
            msgs = build_messages(question, format_context(hits))
            text = self.generator.generate(msgs)
            srcs = [{"ref": f"S{i}", "doc_id": h.doc_id, "title": h.title, "url": h.url, "score": round(h.score, 3)} for i, h in enumerate(hits, 1)]
            return AssistantResponse(question, text, "rag", label, reason, srcs)
        text = self.generator.generate(build_messages(question, None))
        return AssistantResponse(question, text, "no_rag", label, reason)
