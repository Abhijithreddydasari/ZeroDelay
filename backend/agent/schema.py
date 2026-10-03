"""The structured decision the brain must return, plus tolerant JSON parsing.

Gemma is prompted to emit exactly this JSON. We parse it robustly (strip code
fences, extract the first JSON object) and coerce to a `Decision` dataclass, so a
minor formatting slip never crashes the pipeline.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

VALID_ACTIONS = {
    "answer",
    "advance",
    "block",
    "branch",
    "replan",
    "repeat",
    "escalate",
    "emergency",
    "clarify",
    "tool_request",
}


@dataclass
class ToolRequest:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    action: str
    spoken_text: str
    procedure_id: str | None = None
    step_id: int | None = None
    citations: list[str] = field(default_factory=list)
    tool_request: ToolRequest | None = None
    risk: str | None = None
    needs_clarification: str | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.tool_request is None:
            d["tool_request"] = None
        return d


# The JSON contract injected into the prompt.
SCHEMA_HINT = """Return ONLY a single JSON object, no prose, no code fences, with this shape:
{
  "action": "answer | advance | block | branch | replan | repeat | escalate | emergency | clarify | tool_request",
  "spoken_text": "concise words to speak to the technician",
  "procedure_id": "e.g. EVA-PREP-001 or null",
  "step_id": 7,
  "citations": ["manual:EVA-PREP-001#7", "sensor:airlock_pressure_psia=14.6 (critical)"],
  "tool_request": null,
  "risk": "text or null",
  "needs_clarification": "one question or null",
  "confidence": 0.0
}
All the reference facts you need (sensor readings, torque specs, part availability) are
provided above. Keep "tool_request" null and give your final decision now."""


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    # Otherwise grab the first balanced-looking {...} block.
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def parse_decision(text: str) -> Decision | None:
    data = _extract_json(text)
    if not isinstance(data, dict):
        return None

    action = str(data.get("action", "answer")).strip()
    if action not in VALID_ACTIONS:
        action = "answer"

    tr = data.get("tool_request")
    tool_request = None
    if isinstance(tr, dict) and tr.get("tool"):
        tool_request = ToolRequest(
            tool=str(tr.get("tool")),
            args=tr.get("args") if isinstance(tr.get("args"), dict) else {},
        )

    try:
        confidence = float(data.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    citations = data.get("citations") or []
    if not isinstance(citations, list):
        citations = [str(citations)]

    return Decision(
        action=action,
        spoken_text=str(data.get("spoken_text", "")).strip(),
        procedure_id=_none_if_null(data.get("procedure_id")),
        step_id=_int_or_none(data.get("step_id")),
        citations=[str(c) for c in citations],
        tool_request=tool_request,
        risk=_none_if_null(data.get("risk")),
        needs_clarification=_none_if_null(data.get("needs_clarification")),
        confidence=confidence,
    )


def fallback_decision(spoken_text: str) -> Decision:
    """Used when the model's output can't be parsed as JSON."""
    return Decision(action="answer", spoken_text=spoken_text.strip(), confidence=0.0)


def _none_if_null(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"null", "none", ""}:
        return None
    return value


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
