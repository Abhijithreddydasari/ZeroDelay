"""Compose the reasoning prompt from rules + retrieved context + tool results.

Only the selected procedure's structured steps and the relevant sensor snapshot are
injected (not the whole corpus), which keeps prompts small and fast.
"""
from __future__ import annotations

from typing import Any

from ..data_loader import Corpus, Procedure
from ..index.retriever import RetrievalResult
from .schema import SCHEMA_HINT

RULEBOOK = """You are ZeroDelay, a hands-free offline voice copilot guiding a technician
through hazardous maintenance work (space/EVA domain). You cannot reach the cloud or
ground control in real time. Follow these rules:

1. SAFETY FIRST. Never advance past a step whose safety preconditions are not met.
2. GROUND EVERY FACT. Base any sensor value, torque spec, or part availability ONLY on
   the LIVE TELEMETRY and RESOLVED REFERENCE FACTS provided below. Do NOT guess numbers.
3. RESPECT SAFETY TIERS. routine = brief confirm; caution = state warnings then confirm;
   critical = state warnings, require explicit confirmation, and re-check preconditions.
4. CROSS-CHECK. If a live sensor contradicts what a step assumes (e.g. pressure too high
   before opening a hatch), BLOCK the step and explain why.
5. BRANCH ON SYMPTOMS. If the reported symptom changes the repair path, branch.
6. ESCALATE / EMERGENCY. If conditions are unsafe or irreversible, halt and give the
   immediate safety action; note that an alert is queued for when connectivity returns.
7. ASK, DON'T GUESS. If the request is ambiguous, ask ONE clarifying question.
8. SPEAK PLAINLY. spoken_text is read aloud by a voice engine: short, calm, no markdown.
"""


def build_prompt(
    query: str,
    retrieval: RetrievalResult,
    corpus: Corpus,
    sensor_snapshot: list[dict[str, Any]] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    has_live_image: bool = False,
    attached_diagram_ids: list[str] | None = None,
    force_final: bool = False,
    current_step_id: int | None = None,
) -> str:
    parts: list[str] = [RULEBOOK, ""]

    # Selected procedure (structured).
    proc = None
    if retrieval.top_procedure_id:
        proc = corpus.procedures.get(retrieval.top_procedure_id)
    if proc:
        parts.append("RETRIEVED PROCEDURE (structured, source of truth):")
        parts.append(_render_procedure(proc, current_step_id))
        parts.append("")

    # Other candidate procedures (titles only, for disambiguation).
    others = [p for p in retrieval.procedures[1:]]
    if others:
        parts.append("OTHER CANDIDATE PROCEDURES:")
        for p in others:
            parts.append(f"- {p.procedure_id}: {p.title} (score {p.score})")
        parts.append("")

    # Relevant diagrams (images). Only the ids in `attached_diagram_ids` are truly sent
    # to the vision model; the rest are listed by name only so the model does not claim
    # to have visually inspected an image it never received.
    if retrieval.diagrams:
        attached = set(attached_diagram_ids or [])
        parts.append("RELEVANT DIAGRAMS:")
        for d in retrieval.diagrams:
            if d.diagram_id in attached:
                status = "image attached"
            elif d.image_exists:
                status = "reference available, image NOT shown"
            else:
                status = "image not generated yet"
            parts.append(f"- {d.diagram_id}: {d.title} ({status})")
        if has_live_image:
            parts.append(
                "A live camera image from the technician is also attached. Compare it "
                "against the reference diagram to verify the step's expected state."
            )
        parts.append("")

    # Live telemetry snapshot (relevant sensors only).
    if sensor_snapshot:
        parts.append("LIVE TELEMETRY (simulated):")
        for s in sensor_snapshot:
            if not s.get("known"):
                continue
            parts.append(
                f"- {s['sensor']} = {s.get('value')} {s.get('unit') or ''}"
                f" [{s.get('status')}]"
            )
        parts.append("")

    # Reference facts resolved deterministically in code before this turn (sensor reads,
    # torque specs, inventory) so the model never has to make a tool round-trip.
    if tool_results:
        parts.append("RESOLVED REFERENCE FACTS (already looked up — use these, do not ask for tools):")
        for tr in tool_results:
            parts.append(f"- {_render_fact(tr)}")
        parts.append("")

    parts.append(f'TECHNICIAN SAID: "{query}"')
    parts.append("")
    if force_final:
        parts.append(
            "Every reference fact you need is provided above. Do NOT return action "
            "\"tool_request\" — give your final decision now, grounded in those facts."
        )
        parts.append("")
    parts.append(SCHEMA_HINT)
    return "\n".join(parts)


def _render_fact(fact: dict[str, Any]) -> str:
    """One-line view of a resolved tool fact for the prompt."""
    tool = fact.get("tool", "?")
    args = fact.get("args", {})
    result = fact.get("result", {})
    arg_str = ", ".join(f"{k}={v}" for k, v in (args or {}).items())
    return f"{tool}({arg_str}) -> {result}"


def relevant_sensor_names(corpus: Corpus, retrieval: RetrievalResult) -> list[str]:
    if not retrieval.top_procedure_id:
        return []
    proc = corpus.procedures.get(retrieval.top_procedure_id)
    return proc.sensors_watched if proc else []


def _render_procedure(proc: Procedure, current_step_id: int | None = None) -> str:
    """Compact, token-efficient view of the procedure's typed steps.

    Summarizes each step to just what the model needs to enforce order + safety
    (tier, instruction, preconditions, verify, ordering, branches, failure action),
    in a terse line format instead of a verbose YAML dump.
    """
    fm = proc.front_matter
    lines: list[str] = [f"{proc.procedure_id} — {proc.title} (system: {proc.system})"]
    if proc.summary:
        lines.append(f"summary: {_one_line(proc.summary)}")
    entry = fm.get("entry_conditions")
    if entry:
        lines.append(f"entry_conditions: {_compact_conditions(entry)}")
    lines.append("steps:")
    steps = proc.steps if current_step_id is None else [s for s in proc.steps if s.get("id") == current_step_id]
    for step in steps:
        tier = step.get("safety_tier", "routine")
        lines.append(f"  {step.get('id')} [{tier}] {_one_line(step.get('title', ''))}")
        if step.get("instruction"):
            lines.append(f"     do: {_one_line(step['instruction'])}")
        if step.get("preconditions"):
            lines.append(f"     pre: {_compact_conditions(step['preconditions'])}")
        if isinstance(step.get("verify"), dict):
            lines.append(f"     verify: {_compact_verify(step['verify'])}")
        for key in ("depends_on_step", "must_precede", "order_enforced"):
            if step.get(key) is not None:
                lines.append(f"     {key}: {step[key]}")
        if step.get("branches"):
            lines.append(f"     branches: {_compact_branches(step['branches'])}")
        if isinstance(step.get("on_failure"), dict):
            onf = step["on_failure"]
            lines.append(f"     on_fail: {onf.get('action')} — {_one_line(onf.get('note', ''))}")
        if step.get("risk_if_skipped"):
            lines.append(f"     risk_if_skipped: {_one_line(step['risk_if_skipped'])}")
        if step.get("warnings"):
            joined = "; ".join(_one_line(w) for w in step["warnings"])
            lines.append(f"     warnings: {joined}")
    return "\n".join(lines)


def _one_line(text: Any) -> str:
    """Collapse any whitespace/newlines into a single spaced line."""
    return " ".join(str(text or "").split())


def _compact_conditions(conds: list[Any]) -> str:
    out: list[str] = []
    for c in conds or []:
        if not isinstance(c, dict):
            out.append(_one_line(c))
            continue
        subj = c.get("sensor") or c.get("part") or c.get("tool") or "?"
        val = c.get("value", c.get("qty"))
        piece = f"{subj} {c.get('check', '')}".strip()
        if val is not None:
            piece = f"{piece} {val}".strip()
        out.append(piece)
    return "; ".join(out)


def _compact_verify(v: dict[str, Any]) -> str:
    method = v.get("method", "")
    if method == "sensor":
        return _one_line(
            f"sensor {v.get('sensor', '')} {v.get('check', '')} {v.get('value', '')}"
        )
    if method == "visual":
        return _one_line(
            f"visual ref={v.get('visual_ref', '')} expect={v.get('expect_state', '')}"
        )
    if method == "verbal":
        return f"verbal — {_one_line(v.get('prompt', ''))}"
    return method


def _compact_branches(branches: list[Any]) -> str:
    out: list[str] = []
    for b in branches or []:
        if isinstance(b, dict):
            out.append(f"if '{_one_line(b.get('symptom', ''))}' -> {b.get('action', '')}")
    return " | ".join(out)
