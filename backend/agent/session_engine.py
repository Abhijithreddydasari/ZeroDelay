"""Persistent, deterministic procedure state and the low-latency command path."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .. import config
from ..data_loader import Procedure, load_corpus
from ..tools.reference_tools import check_inventory
from ..tools.sensor_sim import get_simulator
from .schema import Decision


def public_procedure(proc: Procedure) -> dict[str, Any]:
    return {
        "id": proc.procedure_id, "title": proc.title, "system": proc.system,
        "summary": proc.summary,
        "steps": [
            {"id": s["id"], "title": s.get("title", ""),
             "safetyTier": s.get("safety_tier", "routine"),
             "instruction": s.get("instruction", ""),
             "warnings": s.get("warnings") or [],
             "diagram": f"/diagrams/{s['verify']['visual_ref']}.png"
             if isinstance(s.get("verify"), dict) and s["verify"].get("visual_ref") else None}
            for s in proc.steps
        ],
    }


def validate_corpus() -> None:
    corpus = load_corpus()
    for proc in corpus.procedures.values():
        ids = [s["id"] for s in proc.steps]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate step IDs in {proc.procedure_id}")
        for step in proc.steps:
            for branch in [*(step.get("branches") or []), step.get("on_failure") or {}]:
                target = corpus.procedures.get(branch.get("goto_procedure", proc.procedure_id))
                if target is None:
                    raise ValueError(f"Unknown branch procedure in {proc.procedure_id}")
                target_id = branch.get("goto_step")
                if target_id is not None and target_id not in {s["id"] for s in target.steps}:
                    raise ValueError(f"Unknown branch step {target_id} in {target.procedure_id}")


class SessionStore:
    def __init__(self, path: Path | None = None):
        self.path = path or config.ARTIFACTS_DIR / "sessions.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._guard = threading.Lock()
        self._locks: dict[str, threading.RLock] = {}
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, state TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS turns (session_id TEXT NOT NULL, turn_id TEXT NOT NULL, result TEXT NOT NULL, PRIMARY KEY(session_id, turn_id))")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def create(self) -> dict[str, Any]:
        state = {"id": uuid.uuid4().hex, "procedure_id": None, "step_id": None,
                 "completed_step_ids": [], "pending_procedure_id": None,
                 "pending_confirmation": False, "step_started_at": None,
                 "completed": False, "halted": False}
        with self._connect() as db:
            db.execute("INSERT INTO sessions VALUES (?, ?)", (state["id"], json.dumps(state)))
        return state

    def get(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT state FROM sessions WHERE id=?", (session_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def get_turn(self, session_id: str, turn_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT result FROM turns WHERE session_id=? AND turn_id=?", (session_id, turn_id)).fetchone()
        return json.loads(row[0]) if row else None

    def commit(self, state: dict[str, Any], turn_id: str, result: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute("UPDATE sessions SET state=? WHERE id=?", (json.dumps(state), state["id"]))
            db.execute("INSERT INTO turns VALUES (?, ?, ?)", (state["id"], turn_id, json.dumps(result)))

    @contextmanager
    def locked(self, session_id: str) -> Iterator[None]:
        with self._guard:
            lock = self._locks.setdefault(session_id, threading.RLock())
        with lock:
            yield


def session_view(state: dict[str, Any]) -> dict[str, Any]:
    view = {k: v for k, v in state.items() if k != "step_started_at"}
    proc = load_corpus().procedures.get(state.get("procedure_id"))
    view["procedure"] = public_procedure(proc) if proc else None
    return view


def _decision(action: str, text: str, state: dict[str, Any], risk: str | None = None) -> Decision:
    return Decision(action=action, spoken_text=text, procedure_id=state.get("procedure_id"),
                    step_id=state.get("step_id"), risk=risk, confidence=1.0)


def _normal(query: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", query.lower()).split())


_ALIASES = {
    "EVA-PREP-001": ("airlock", "air luck", "eva", "spacewalk", "suit"),
    "COOL-JMP-002": ("coolant", "ammonia", "jumper"),
    "BATT-ORU-003": ("battery", "oru"),
    "CDRA-FAULT-004": ("cdra", "see draw", "c draw", "carbon dioxide", "cabin co2", "scrubber"),
}
_REPEAT = {"repeat", "repeat step", "repeat that", "current step"}
_NEXT = {"next", "next step", "step done", "done", "continue"}
_CONFIRM = {"confirm", "confirmed", "i confirm", "step complete", "completed"}


def _candidate(query: str) -> str | None:
    words = _normal(query)
    matches = [pid for pid, aliases in _ALIASES.items() if any(a in words for a in aliases)]
    return matches[0] if len(matches) == 1 else None


def _check_condition(cond: dict[str, Any]) -> bool:
    check = cond.get("check")
    if "part" in cond or "tool" in cond:
        item = check_inventory(cond.get("part") or cond.get("tool"))
        return item.get("available", False) and item.get("qty", 0) >= cond.get("qty", 1)
    if "sensor" not in cond:
        return False
    reading = get_simulator().read(cond["sensor"])
    if not reading.get("known") or reading.get("value") is None or reading.get("status") == "unknown":
        return False
    value = reading.get("value")
    target = cond.get("value")
    if check == "in_range":
        return reading.get("status") == "nominal"
    if check == "out_of_range":
        return reading.get("status") != "nominal"
    if check == "equals":
        return value == target
    if check == "not_equals":
        return value != target
    if check == "less_than":
        return isinstance(value, (int, float)) and isinstance(target, (int, float)) and value < target
    if check == "greater_than":
        return isinstance(value, (int, float)) and isinstance(target, (int, float)) and value > target
    return False


def _current_step(state: dict[str, Any]) -> tuple[Procedure, dict[str, Any]]:
    proc = load_corpus().procedures[state["procedure_id"]]
    return proc, next(s for s in proc.steps if s["id"] == state["step_id"])


def _guidance(state: dict[str, Any]) -> Decision:
    proc, step = _current_step(state)
    state["pending_confirmation"] = True
    warnings = " ".join(step.get("warnings") or [])
    tier = step.get("safety_tier", "routine")
    confirm = " Say confirm when complete." if tier != "critical" else " Give the specific completion read-back before continuing."
    return _decision("answer", f"Step {step['id']}: {warnings} {step.get('instruction', '')}{confirm}".strip(), state)


def _confirmed_for(proc: Procedure, step: dict[str, Any], words: str) -> bool:
    tier = step.get("safety_tier", "routine")
    if tier == "routine":
        return words in _CONFIRM | _NEXT
    if tier == "caution":
        prompt = _normal((step.get("verify") or {}).get("prompt", ""))
        return words in _CONFIRM or ("acknowledged" in prompt and words in {"acknowledged", "acknowledge"})
    # Critical steps need a step-specific read-back, not a bare "confirm".
    specific = {6: {"confirm tether locked", "tether locked and load tested"},
                7: {"confirm airlock below 2 psia", "airlock below 2 psia"},
                8: {"egress complete"}, 25: {"crew safe"}}
    if proc.procedure_id == "EVA-PREP-001" or proc.procedure_id == "CDRA-FAULT-004" and step["id"] == 25:
        return words in specific.get(step["id"], {f"confirm step {step['id']}"})
    return words == f"confirm step {step['id']}"


def fast_decision(state: dict[str, Any], query: str) -> Decision | None:
    words = _normal(query)
    corpus = load_corpus()
    if state["completed"]:
        return _decision("answer", "This procedure is complete. Start a new session for another task.", state)
    if state.get("halted"):
        return _decision("emergency", "This procedure was halted. Follow the emergency guidance and start a new session only after the hazard is resolved.", state)
    if words in {"do not confirm", "don t confirm", "not confirmed", "not done",
                 "do not continue", "don t continue", "not complete"}:
        return _decision("block", "No step was advanced. Tell me when the current step is complete.", state)
    if not state["procedure_id"]:
        selected = _candidate(query)
        if selected and ("procedure" in words or "select" in words or "choose" in words):
            state["pending_procedure_id"] = selected
        if words in {"confirm procedure", "yes confirm procedure", "confirm selection"} and state["pending_procedure_id"]:
            proc = corpus.procedures[state["pending_procedure_id"]]
            state.update(procedure_id=proc.procedure_id, step_id=proc.steps[0]["id"],
                         pending_procedure_id=None, step_started_at=time.time())
            return _guidance(state)
        if state["pending_procedure_id"] and words in {"confirm", "yes"}:
            proc = corpus.procedures[state["pending_procedure_id"]]
            state.update(procedure_id=proc.procedure_id, step_id=proc.steps[0]["id"],
                         pending_procedure_id=None, step_started_at=time.time())
            return _guidance(state)
        if selected:
            state["pending_procedure_id"] = selected
        if state["pending_procedure_id"]:
            proc = corpus.procedures[state["pending_procedure_id"]]
            return _decision("clarify", f"I found {proc.title}. Say confirm procedure to start, or name a different procedure.", state)
        return _decision("clarify", "Which procedure do you need: airlock, coolant, battery, or CDRA?", state)

    proc, step = _current_step(state)
    life_safety = {"cabin_co2_mmHg", "suit_o2_pct", "suit_pressure_psia", "suit_co2_mmHg"}
    critical = [s for s in proc.sensors_watched if s in life_safety and get_simulator().read(s).get("status") == "critical"]
    if critical and not (proc.procedure_id == "CDRA-FAULT-004" and step["id"] == 25):
        state["pending_confirmation"] = False
        if proc.procedure_id == "CDRA-FAULT-004" and "cabin_co2_mmHg" in critical:
            state["step_id"] = 25
            state["step_started_at"] = time.time()
        return _decision("emergency", f"Critical reading: {', '.join(critical)}. Stop the procedure and follow the emergency step.", state)
    # A changed fault must never silently switch the active procedure.
    different = _candidate(query)
    if different and different != proc.procedure_id:
        return _decision("clarify", f"This session is for {proc.title}. Start a new session for a different procedure.", state)
    if proc.procedure_id == "COOL-JMP-002" and step["id"] == 3 and state["pending_confirmation"]:
        if words in {"white flakes", "ammonia flakes", "white flakes or snow", "ammonia snow"}:
            state.update(halted=True, pending_confirmation=False)
            return _decision("emergency", "Possible active ammonia release. Stop the repair, execute decontamination, and alert ground.", state)
        if words in {"no frost pressure dropping", "no visible frost but pressure still dropping"}:
            state.update(halted=True, pending_confirmation=False)
            return _decision("escalate", "The leak is not confirmed at the quick disconnect. Stop replacement and escalate for loop isolation.", state)
        if words in {"frost without flakes", "white frost only no flakes"}:
            state["completed_step_ids"].append(3)
            state.update(step_id=4, step_started_at=time.time())
            guidance = _guidance(state)
            guidance.action = "advance"
            return guidance
    if words in _REPEAT:
        return _guidance(state)
    if words in _NEXT | _CONFIRM or words.startswith("confirm ") or words in {"acknowledged", "acknowledge", "egress complete", "crew safe"}:
        if not state["pending_confirmation"]:
            return _guidance(state)
        if not _confirmed_for(proc, step, words):
            return _decision("block", "That confirmation is not specific enough for this step. Repeat the step and give its completion read-back.", state)
        if proc.procedure_id == "COOL-JMP-002" and step["id"] == 3:
            return _decision("block", "Classify the leak before advancing: report frost without flakes, ammonia flakes, or no frost with falling pressure.", state)
        if proc.procedure_id == "COOL-JMP-002" and step["id"] == 5 and 4 not in state["completed_step_ids"]:
            return _decision("block", "The loop pressure check in step 4 must be confirmed before demating.", state)
        entry = (proc.front_matter.get("entry_conditions") or []) if step["id"] == proc.steps[0]["id"] else []
        failed = [c for c in [*entry, *(step.get("preconditions") or [])] if not _check_condition(c)]
        missing_parts = [part for part in step.get("required_parts") or []
                         if not check_inventory(part).get("available")]
        if proc.procedure_id == "COOL-JMP-002" and step["id"] == 6 and missing_parts == ["QD-JMP-14-CAP"]:
            if check_inventory("CAP-GENERIC-2").get("available"):
                missing_parts = []
        if proc.procedure_id == "CDRA-FAULT-004" and step["id"] == 20:
            if not missing_parts:
                return _decision("block", "A spare sorbent bed is available. Reassess before using the no-spare bypass.", state)
            missing_parts = []
        if missing_parts:
            return _decision("block", f"Required part unavailable: {', '.join(missing_parts)}. Do not advance.", state)
        substitute_cap = proc.procedure_id == "COOL-JMP-002" and step["id"] == 2 and any(c.get("part") == "QD-JMP-14-CAP" for c in failed) and check_inventory("CAP-GENERIC-2").get("available")
        if substitute_cap:
            failed = [c for c in failed if c.get("part") != "QD-JMP-14-CAP"]
        if failed:
            note = (step.get("on_failure") or {}).get("note") or "A required condition is not met."
            action = (step.get("on_failure") or {}).get("action", "block")
            if action not in {"block", "emergency", "escalate"}:
                action = "block"
            return _decision(action, f"Cannot advance. {note}", state, note)
        verify = step.get("verify") or {}
        cdra_branch_read = proc.procedure_id == "CDRA-FAULT-004" and step["id"] == 3
        valve_not_nominal = (proc.procedure_id == "CDRA-FAULT-004" and step["id"] == 8
                             and get_simulator().read("cdra_valve_state").get("status") != "nominal")
        if verify.get("method") == "sensor" and not cdra_branch_read and (valve_not_nominal or not _check_condition(verify)):
            note = (step.get("on_failure") or {}).get("note") or "The verification sensor is not in the required state."
            if proc.procedure_id == "CDRA-FAULT-004" and step["id"] == 8:
                state.update(step_id=20, step_started_at=time.time(), pending_confirmation=True)
                replan = next(s for s in proc.steps if s["id"] == 20)
                return _decision("replan", f"{note} {replan['instruction']}", state)
            return _decision("block", f"Cannot advance. {note}", state, note)
        duration = (step.get("specs") or {}).get("duration_s")
        if duration and time.time() - (state.get("step_started_at") or time.time()) < duration:
            return _decision("block", "The required timed interval has not elapsed. Do not advance yet.", state)
        depends = step.get("depends_on_step")
        if depends is not None and depends not in state["completed_step_ids"]:
            return _decision("block", f"Step {depends} must be confirmed first.", state)
        state["completed_step_ids"].append(step["id"])
        position = next(i for i, s in enumerate(proc.steps) if s["id"] == step["id"])
        if proc.procedure_id == "CDRA-FAULT-004":
            if step["id"] == 3:
                valve = get_simulator().read("cdra_valve_state")["value"]
                temp = get_simulator().read("cdra_bed_temp_c")
                target = 8 if valve in {"fault", "transition"} else 14 if temp["status"] != "nominal" else None
                if target is None:
                    state["completed_step_ids"].pop()
                    return _decision("clarify", "Neither documented fault branch matches. Recheck the valve and bed temperature.", state)
                position = next(i for i, s in enumerate(proc.steps) if s["id"] == target) - 1
            elif step["id"] in {8, 14, 20}:
                position = next(i for i, s in enumerate(proc.steps) if s["id"] == 15) - 1
            elif step["id"] in {15, 25}:
                position = len(proc.steps) - 1
        if position == len(proc.steps) - 1:
            state.update(completed=True, pending_confirmation=False)
            return _decision("advance", "Procedure complete. All steps were confirmed.", state)
        next_step = proc.steps[position + 1]
        state.update(step_id=next_step["id"], step_started_at=time.time())
        guidance = _guidance(state)
        guidance.action = "replan" if substitute_cap else "advance"
        if substitute_cap:
            guidance.spoken_text = "The listed cap is unavailable. Use approved contingency cap CAP-GENERIC-2. " + guidance.spoken_text
        return guidance
    return None
