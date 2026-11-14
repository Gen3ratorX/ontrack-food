"""Blood glucose readings in free text.

Ghana uses mmol/L. CDC and ADA sources use mg/dL. 1 mmol/L = 18 mg/dL.

Cited cut-offs (canonical mmol/L, compared in the unit the person used when it is explicit):
- low (ADA/CDC level 1): below 3.9 mmol/L (70 mg/dL)
- severe low (ADA level 2): below 3.0 mmol/L (54 mg/dL) — emergency
- very high (CDC DKA: go to emergency care if glucose stays at 300 mg/dL or above):
  16.7 mmol/L (300 mg/dL) — emergency

A number with no unit is mmol/L at or below 35, and mg/dL at or above 40.
Values in between are ambiguous and are not converted. Portions, ages, insulin
doses, clock times and A1C/GI figures are not readings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MGDL_PER_MMOL = 18.0

# No-unit inference. 35 mmol/L is 630 mg/dL; 40 mg/dL is 2.2 mmol/L.
INFER_MMOL_MAX = 35.0
INFER_MGDL_MIN = 40.0

LOW_MMOL = 3.9
LOW_MGDL = 70.0
SEVERE_MMOL = 3.0
SEVERE_MGDL = 54.0
# 300/18 = 16.6 recurring; the knowledge base cites this as 16.7 mmol/L.
VERY_HIGH_MMOL = 300.0 / MGDL_PER_MMOL
VERY_HIGH_MGDL = 300.0

_NUM = r"(\d{1,4}(?:\.\d+)?)"
_UNIT = r"mg\s*/\s*dl|mmol\s*/\s*l|mmols?(?!\s*/\s*mol)"
_NUM_THEN_UNIT = re.compile(rf"(?<![\d.]){_NUM}(?![\d.])\s*(?P<unit>{_UNIT})\b", re.I)
_UNIT_THEN_NUM = re.compile(rf"\b(?P<unit>{_UNIT})\s*(?<![\d.]){_NUM}(?![\d.])", re.I)
_CUE = re.compile(
    r"\b(?:blood\s*sugars?|sugars?|glucoses?|glucometers?|meters?|readings?|rbs|fbs)\b",
    re.I,
)
# Words that may sit between "sugar" / "reading" and the number. "sugar and 3 koose" does not qualify.
_BRIDGE = re.compile(
    r"(?:\s+(?:is|was|were|of|at|reads?|says?|showing|shows|level|levels|around|about|"
    r"approximately|approx|to|dropped|now|machine|meter|glucometer))*\s*",
    re.I,
)
_NUMBER = re.compile(rf"(?<![\d.]){_NUM}(?![\d.])")
_SUFFIX_REJECT = re.compile(
    r"^\s*(?:%|units?\b|iu\b|u\b|minutes?\b|mins?\b|hours?\b|hrs?\b|days?\b|"
    r"years?\b|yrs?\b|g\b|grams?\b|kg\b|ml\b|kcal\b|calories?\b|"
    r"balls?\b|plates?\b|cups?\b|slices?\b|pieces?\b|tablespoons?\b|tbsp\b|teaspoons?\b|tsp\b|"
    r"am\b|pm\b|o'?clock\b|mg\b(?!\s*/\s*dl))",
    re.I,
)
_REJECT_BEFORE = re.compile(
    r"\b(?:age|aged|ages|gi|hba1c|a1c|units?|doses?)\s*$",
    re.I,
)


@dataclass(frozen=True)
class GlucoseReading:
    value: float
    unit: str | None  # "mmol/L", "mg/dL", or None when ambiguous
    unit_source: str  # "explicit", "inferred", "ambiguous"
    mmol: float | None
    mgdl: float | None
    start: int
    end: int
    raw: str

    def context_line(self) -> str:
        if self.unit_source == "ambiguous" or self.mmol is None or self.mgdl is None:
            return (
                f"User reading: {self.value:g} (no unit). Ambiguous: between 35 and 40, "
                "so it is not assumed to be mmol/L or mg/dL."
            )
        line = f"User reading: {self.mmol:.1f} mmol/L ({round(self.mgdl):.0f} mg/dL)"
        if is_severe_low(self):
            line += ", below 3.0 mmol/L (54 mg/dL) = severe low; also below 3.9 mmol/L = low"
        elif is_low(self):
            line += ", below 3.9 mmol/L = low"
        elif is_very_high(self):
            line += ", at or above 16.7 mmol/L (300 mg/dL) = very high"
        return line


def _canon_unit(token: str) -> str:
    t = re.sub(r"\s+", "", token.lower())
    if t.startswith("mg"):
        return "mg/dL"
    return "mmol/L"


def _pair(value: float, unit: str) -> tuple[float, float]:
    if unit == "mmol/L":
        return value, value * MGDL_PER_MMOL
    return value / MGDL_PER_MMOL, value


def _rejected_number(text: str, start: int, end: int) -> bool:
    if start > 0 and text[start - 1] == ":":
        return True
    if end < len(text) and text[end] == ":":
        return True
    if _SUFFIX_REJECT.match(text[end:end + 24]):
        return True
    before = text[max(0, start - 16):start]
    if _REJECT_BEFORE.search(before):
        return True
    return False


def _make(value: float, unit: str | None, source: str, start: int, end: int, raw: str) -> GlucoseReading:
    if unit is None:
        return GlucoseReading(value, None, source, None, None, start, end, raw)
    mmol, mgdl = _pair(value, unit)
    return GlucoseReading(value, unit, source, mmol, mgdl, start, end, raw)


def _from_explicit(value: float, unit_token: str, start: int, end: int, raw: str) -> GlucoseReading:
    return _make(value, _canon_unit(unit_token), "explicit", start, end, raw)


def _from_bare(value: float, start: int, end: int, raw: str) -> GlucoseReading:
    if value <= INFER_MMOL_MAX:
        return _make(value, "mmol/L", "inferred", start, end, raw)
    if value >= INFER_MGDL_MIN:
        return _make(value, "mg/dL", "inferred", start, end, raw)
    return _make(value, None, "ambiguous", start, end, raw)


def parse_glucose(text: str) -> list[GlucoseReading]:
    """Readings in order. Explicit units win; a bare number needs a sugar/reading word before it."""
    found: dict[int, GlucoseReading] = {}

    def add(reading: GlucoseReading) -> None:
        # Keep the longer span if two matches start together (number vs number+unit).
        prev = found.get(reading.start)
        if prev is None or reading.end > prev.end:
            found[reading.start] = reading

    for cre in (_NUM_THEN_UNIT, _UNIT_THEN_NUM):
        for m in cre.finditer(text):
            value = float(m.group(1))
            if _rejected_number(text, m.start(1), m.end(1)) and cre is _UNIT_THEN_NUM:
                continue
            add(_from_explicit(value, m.group("unit"), m.start(1), m.end(1), text[m.start():m.end()]))

    for cue in _CUE.finditer(text):
        window = text[cue.end():cue.end() + 48]
        for m in _NUMBER.finditer(window):
            between = window[:m.start(1)]
            if _BRIDGE.fullmatch(between) is None:
                break
            abs_start = cue.end() + m.start(1)
            abs_end = cue.end() + m.end(1)
            if abs_start in found or _rejected_number(text, abs_start, abs_end):
                break
            add(_from_bare(float(m.group(1)), abs_start, abs_end, m.group(1)))
            break

    return [found[k] for k in sorted(found)]


def _on_mgdl_scale(reading: GlucoseReading) -> bool:
    return reading.unit == "mg/dL"


def is_severe_low(reading: GlucoseReading) -> bool:
    if reading.mmol is None or reading.mgdl is None:
        return False
    if _on_mgdl_scale(reading):
        return reading.mgdl < SEVERE_MGDL
    return reading.mmol < SEVERE_MMOL


def is_low(reading: GlucoseReading) -> bool:
    """Level 1 or worse: below 3.9 mmol/L (70 mg/dL)."""
    if reading.mmol is None or reading.mgdl is None:
        return False
    if _on_mgdl_scale(reading):
        return reading.mgdl < LOW_MGDL
    return reading.mmol < LOW_MMOL


def is_very_high(reading: GlucoseReading) -> bool:
    """CDC emergency line: 300 mg/dL (16.7 mmol/L) or above."""
    if reading.mmol is None or reading.mgdl is None:
        return False
    if _on_mgdl_scale(reading):
        return reading.mgdl >= VERY_HIGH_MGDL
    return reading.mmol + 1e-9 >= VERY_HIGH_MMOL


def glucose_context(text: str) -> str:
    """Lines to prepend to a prompt. Empty when the text has no reading."""
    readings = parse_glucose(text)
    if not readings:
        return ""
    return "\n".join(r.context_line() for r in readings)


def emergency_reason(text: str) -> str | None:
    """Severe low or CDC very-high threshold. Ambiguous numbers do not fire."""
    for r in parse_glucose(text):
        if r.mmol is None:
            continue
        if is_severe_low(r):
            return f"rule:glucose_severe_low={r.mmol:.1f}mmol"
        if is_very_high(r):
            return f"rule:glucose_very_high={r.mmol:.1f}mmol"
    return None


def conversion_ok(mmol: float, mgdl: float) -> bool:
    """True when mg/dL is mmol/L × 18, allowing ordinary rounding."""
    return abs(mgdl - mmol * MGDL_PER_MMOL) <= max(1.5, 0.02 * abs(mgdl))


def glucose_answer_flags(response: str) -> list[str]:
    """Flag a glucose number with no unit, or a pair whose conversion is not ×18."""
    readings = parse_glucose(response)
    flags = []
    if any(r.unit_source != "explicit" for r in readings):
        flags.append("glucose_missing_unit")
    explicit = [r for r in readings if r.unit_source == "explicit" and r.mmol is not None]
    for i, a in enumerate(explicit):
        for b in explicit[i + 1:]:
            if a.unit == b.unit:
                continue
            if min(abs(a.start - b.end), abs(b.start - a.end)) > 50:
                continue
            mmol = a.mmol if a.unit == "mmol/L" else b.mmol
            mgdl = a.mgdl if a.unit == "mg/dL" else b.mgdl
            if mmol is not None and mgdl is not None and not conversion_ok(mmol, mgdl):
                flags.append("glucose_wrong_conversion")
                return flags
    return flags
