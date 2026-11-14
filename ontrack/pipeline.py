"""OnTrack assistant: triage gate -> (template | [meal classifier ->] RAG generation | no-RAG generation).

The safety gate always runs first. The meal classifier only runs for general_education questions that
describe a meal, and only in RAG mode (the no-RAG baseline stays a plain model, for a fair comparison)."""
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
    meal_analysis: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreparedAnswer:
    """Either a finished template response, or messages waiting for a (possibly batched) generate call."""
    question: str
    triage_label: str
    triage_reason: str
    done: AssistantResponse | None = None
    messages: list | None = None
    mode: str = "template"
    sources: list = field(default_factory=list)
    meal_analysis: dict | None = None


class OnTrackAssistant:
    def __init__(self, generator, retriever: Retriever | None = None, gate: SafetyGate | None = None, use_rag: bool = True,
                 classify_meals: bool = True):
        if use_rag and retriever is None:
            raise ValueError("RAG mode needs a retriever")
        self.generator, self.retriever, self.gate, self.use_rag = generator, retriever, gate, use_rag
        self.classify_meals = classify_meals
        self._meal = None

    def analyse_meal(self, question: str):
        """Return a MealAnalysis if the question describes a meal, else None."""
        from .meal.detect import is_meal_description
        from .meal.engine import MealClassifier
        if self._meal is None:
            self._meal = MealClassifier()
        mentions = self._meal.recognizer.find(question)
        if not is_meal_description(question, mentions):
            return None
        return self._meal.analyse(question)

    def prepare(self, question: str) -> PreparedAnswer:
        """Gate first. Emergency and clinician questions are fixed templates and do not need generation."""
        label, reason = ("general_education", "gate disabled")
        if self.gate is not None:
            label, reason = self.gate.classify(question)
            if label != "general_education":
                done = AssistantResponse(question, template_for(label), "template", label, reason)
                return PreparedAnswer(question, label, reason, done=done)
        if self.use_rag:
            meal = self.analyse_meal(question) if self.classify_meals else None
            query = question
            if meal is not None:  # steer retrieval toward the recognised foods' evidence
                query = question + " " + " ".join(sorted({i.name.split(":")[0].split("(")[0] for i in meal.items}))
            hits = self.retriever.search(query)
            msgs = build_messages(question, format_context(hits), meal.to_context() if meal is not None else None)
            srcs = [{"ref": f"S{i}", "doc_id": h.doc_id, "title": h.title, "url": h.url, "score": round(h.score, 3)} for i, h in enumerate(hits, 1)]
            meal_dict = None
            if meal is not None:
                srcs += [{"ref": "M", "doc_id": s["id"], "title": s["title"], "url": s["url"], "score": None} for s in meal.sources]
                meal_dict = meal.to_dict()
            return PreparedAnswer(question, label, reason, messages=msgs, mode="rag", sources=srcs, meal_analysis=meal_dict)
        return PreparedAnswer(question, label, reason, messages=build_messages(question, None), mode="no_rag")

    def complete(self, prepared: PreparedAnswer, text: str | None = None) -> AssistantResponse:
        if prepared.done is not None:
            return prepared.done
        return AssistantResponse(prepared.question, text or "", prepared.mode, prepared.triage_label,
                                 prepared.triage_reason, prepared.sources, prepared.meal_analysis)

    def answer(self, question: str) -> AssistantResponse:
        prepared = self.prepare(question)
        if prepared.done is not None:
            return prepared.done
        return self.complete(prepared, self.generator.generate(prepared.messages))
