"""Single-turn orchestration: audio -> STT -> RAG -> reason -> TTS.

Facts a step depends on (sensor readings, torque specs, part availability) are resolved
deterministically in code *before* the model runs, so a turn is a single reasoning pass
(no generative tool round-trips). That single pass is what we stream, so the spoken
answer arrives continuously instead of in stop-start chunks.

State is intentionally NOT held here (the multi-turn procedure state machine is a
later build). This class wires the pieces into one testable turn.
"""
from __future__ import annotations

import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .. import config
from ..data_loader import Procedure, load_corpus
from ..index.retriever import Retriever, RetrievalResult, RetrievedProcedure, RetrievedDiagram
from ..tools import reference_tools
from ..tools.sensor_sim import get_simulator
from . import prompt_builder
from .schema import Decision, fallback_decision, parse_decision

# Cap images passed to the vision model to protect VRAM/latency.
_MAX_DIAGRAM_IMAGES = 1

# Phrases in a technician's utterance that imply a visual check. Kept fairly specific so
# ordinary verbal/sensor turns stay text-only (the fast path) instead of paying for the
# vision encoder on every turn.
_VISION_QUERY_HINTS = (
    "diagram", "schematic", "picture", "photo", "image", "illustration", "callout",
    "look at", "show me", "which one", "which valve", "which switch", "which connector",
    "where is", "where's", "point to", "identify", "locate", "does this look",
    "what does this", "on the panel", "visually", "verify visually",
)

# Matches the opening of the spoken_text JSON string so we can surface it token-by-token
# while the model is still emitting the rest of the decision object.
_SPOKEN_KEY_RE = re.compile(r'"spoken_text"\s*:\s*"')
_JSON_UNESCAPE = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}


def _extract_spoken_prefix(raw: str) -> str | None:
    """Best-effort decode of the `spoken_text` value emitted so far in a streaming JSON.

    Returns the (partial) spoken text once the key appears, or None if it hasn't yet.
    Stops at the closing quote when it arrives, so trailing JSON (action, etc.) is
    ignored. This lets us stream just the words the technician will hear.
    """
    match = _SPOKEN_KEY_RE.search(raw)
    if not match:
        return None
    out: list[str] = []
    escaped = False
    for ch in raw[match.end():]:
        if escaped:
            out.append(_JSON_UNESCAPE.get(ch, ch))
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            break
        else:
            out.append(ch)
    return "".join(out)


def _fastener_from_specs(specs: Any) -> str | None:
    """Pull a fastener_id out of a step's `specs` (either specs.torque.fastener_id or a
    top-level specs.fastener_id), so we can pre-fetch its torque spec."""
    if not isinstance(specs, dict):
        return None
    torque = specs.get("torque")
    if isinstance(torque, dict) and torque.get("fastener_id"):
        return torque["fastener_id"]
    if specs.get("fastener_id"):
        return specs["fastener_id"]
    return None


class Orchestrator:
    def __init__(self, retriever: Retriever | None = None, gemma=None):
        self.corpus = load_corpus()
        self.retriever = retriever or Retriever()
        self._gemma = gemma  # injected lazily to keep imports light for tests
        self.simulator = get_simulator()

    # ------------------------------------------------------------------
    @property
    def gemma(self):
        if self._gemma is None:
            from ..models.gemma import get_runtime

            self._gemma = get_runtime()
        return self._gemma

    # ------------------------------------------------------------------
    def process_audio(
        self,
        audio_path: str | Path,
        live_image_path: str | Path | None = None,
        speak: bool = True,
    ) -> dict[str, Any]:
        query = self.gemma.transcribe(audio_path)
        result = self.process_text(query, live_image_path=live_image_path, speak=speak)
        result["transcribed"] = True
        return result

    def process_text(
        self,
        query: str,
        live_image_path: str | Path | None = None,
        speak: bool = True,
        procedure_id: str | None = None,
        step_id: int | None = None,
    ) -> dict[str, Any]:
        t_start = time.perf_counter()
        ctx = self._prepare_turn(query, live_image_path, procedure_id, step_id)

        t_gen = time.perf_counter()
        raw = self.gemma.reason(ctx["prompt"], images=ctx["images"])
        generate_ms = (time.perf_counter() - t_gen) * 1000

        parsed = parse_decision(raw)
        decision = self._finalize_decision(parsed or fallback_decision(raw))

        tts_path = None
        tts_ms = 0.0
        if speak and decision.spoken_text:
            t_tts = time.perf_counter()
            tts_path = self._speak(decision.spoken_text)
            tts_ms = (time.perf_counter() - t_tts) * 1000

        self._log_timing(t_start, ctx["retrieve_ms"], generate_ms, tts_ms, ctx["images"])
        result = self._build_result(query, decision, ctx, tts_path, parsed is not None)
        result["timing_ms"] = {"retrieve": round(ctx["retrieve_ms"]),
            "prompt": round(ctx["prompt_ms"]), "generation_call": round(generate_ms),
            "tts": round(tts_ms), **self.gemma.last_metrics()}
        return result

    def stream_text(
        self,
        query: str,
        live_image_path: str | Path | None = None,
        speak: bool = True,
        procedure_id: str | None = None,
        step_id: int | None = None,
    ) -> Iterator[tuple[str, Any]]:
        """Single-pass turn that streams the spoken answer as it is generated.

        Yields ("delta", text_chunk) events as the model emits the `spoken_text`, then
        one ("final", result_dict) event with the full decision, retrieval, and TTS.
        Because every fact is resolved up front, the generation is one continuous pass:
        the words stream steadily instead of pausing for mid-answer tool round-trips.
        """
        t_start = time.perf_counter()
        ctx = self._prepare_turn(query, live_image_path, procedure_id, step_id)

        raw_parts: list[str] = []
        emitted = 0
        t_gen = time.perf_counter()
        for chunk in self.gemma.reason_stream(ctx["prompt"], images=ctx["images"]):
            raw_parts.append(chunk)
            spoken = _extract_spoken_prefix("".join(raw_parts))
            if spoken is not None and len(spoken) > emitted:
                yield ("delta", spoken[emitted:])
                emitted = len(spoken)
        generate_ms = (time.perf_counter() - t_gen) * 1000

        raw = "".join(raw_parts)
        parsed = parse_decision(raw)
        decision = self._finalize_decision(parsed or fallback_decision(raw))

        tts_path = None
        tts_ms = 0.0
        if speak and decision.spoken_text:
            t_tts = time.perf_counter()
            tts_path = self._speak(decision.spoken_text)
            tts_ms = (time.perf_counter() - t_tts) * 1000

        self._log_timing(t_start, ctx["retrieve_ms"], generate_ms, tts_ms, ctx["images"])
        yield ("final", self._build_result(query, decision, ctx, tts_path, parsed is not None))

    # ------------------------------------------------------------------
    def _prepare_turn(
        self, query: str, live_image_path: str | Path | None,
        procedure_id: str | None = None, step_id: int | None = None,
    ) -> dict[str, Any]:
        """Everything a single reasoning pass needs: retrieval, telemetry, resolved
        reference facts, the (optional) diagram image, and the built prompt."""
        t_retrieve = time.perf_counter()
        if procedure_id and procedure_id in self.corpus.procedures:
            proc = self.corpus.procedures[procedure_id]
            retrieval = RetrievalResult(query=query,
                procedures=[RetrievedProcedure(proc.procedure_id, proc.title, 1.0)],
                diagrams=[RetrievedDiagram(d.diagram_id, d.title, str(d.image_path), d.image_exists, 0.0)
                          for did in proc.related_diagrams if (d := self.corpus.diagrams.get(did))])
        else:
            retrieval = self.retriever.retrieve(query)
        retrieve_ms = (time.perf_counter() - t_retrieve) * 1000

        # Relevant live telemetry for the selected procedure.
        sensor_names = prompt_builder.relevant_sensor_names(self.corpus, retrieval)
        snapshot = [self.simulator.read(n) for n in sensor_names]

        # Resolve the step preconditions (torque specs / part availability) in code so the
        # model gets exact facts without a generative tool loop. Sensors come via snapshot.
        proc = (
            self.corpus.procedures.get(retrieval.top_procedure_id)
            if retrieval.top_procedure_id
            else None
        )
        resolved_facts = self._resolve_reference_facts(proc, step_id)

        # Vision is the biggest per-turn cost, so only attach a reference diagram when the
        # turn actually needs it (a camera frame, a visual-sounding request, or an explicit
        # override). Otherwise reason text-only, which is far faster.
        want_diagram = self._wants_diagram_vision(query, live_image_path)
        images = self._collect_images(retrieval, live_image_path, want_diagram)
        attached_diagram_ids = self._attached_diagram_ids(retrieval, want_diagram)

        t_prompt = time.perf_counter()
        prompt = prompt_builder.build_prompt(
            query=query,
            retrieval=retrieval,
            corpus=self.corpus,
            sensor_snapshot=snapshot,
            tool_results=resolved_facts or None,
            has_live_image=live_image_path is not None,
            attached_diagram_ids=attached_diagram_ids,
            # Facts are pre-resolved, so there is exactly one pass and no tool requests.
            force_final=True,
            current_step_id=step_id,
        )
        return {
            "retrieval": retrieval,
            "snapshot": snapshot,
            "resolved_facts": resolved_facts,
            "images": images,
            "attached_diagram_ids": attached_diagram_ids,
            "prompt": prompt,
            "retrieve_ms": retrieve_ms,
            "prompt_ms": (time.perf_counter() - t_prompt) * 1000,
        }

    def _finalize_decision(self, decision: Decision | None) -> Decision:
        if decision is None:
            return fallback_decision("I could not process that. Please repeat.")
        # Safety net: with facts pre-resolved the model shouldn't ask for tools, but if it
        # does anyway, never hand the client a tool_request we did not execute.
        if decision.action == "tool_request":
            decision.action = "answer" if decision.spoken_text else "clarify"
            decision.tool_request = None
            if not decision.spoken_text:
                decision.spoken_text = (
                    "I need more information to continue safely. Please repeat the "
                    "current step or the last reading."
                )
        return decision

    def _build_result(
        self, query: str, decision: Decision, ctx: dict[str, Any], tts_path,
        parse_valid: bool = True,
    ) -> dict[str, Any]:
        retrieval = ctx["retrieval"]
        # Only surface to the client the diagram(s) actually shown to the model this turn,
        # so the UI displays a schematic when it's relevant, not on every turn.
        shown = set(ctx["attached_diagram_ids"])
        return {
            "query": query,
            "decision": decision.to_dict(),
            "retrieval": {
                "procedures": [p.__dict__ for p in retrieval.procedures],
                "diagrams": [
                    d.__dict__ for d in retrieval.diagrams if d.diagram_id in shown
                ],
            },
            "tool_calls": ctx["resolved_facts"],
            "sensor_snapshot": ctx["snapshot"],
            "tts_path": str(tts_path) if tts_path else None,
            "parse_valid": parse_valid,
        }

    def _resolve_reference_facts(self, proc: Procedure | None, step_id: int | None = None) -> list[dict[str, Any]]:
        """Look up the torque specs and part/tool availability the procedure references.

        Done deterministically from the corpus (not by the model) so the reasoning pass
        already has every exact value and never needs a tool round-trip. Sensor readings
        are provided separately via the live telemetry snapshot.
        """
        if proc is None:
            return []

        facts: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        def add(tool: str, args: dict[str, Any], result: dict[str, Any]) -> None:
            key = (tool, str(sorted(args.items())))
            if key in seen:
                return
            seen.add(key)
            facts.append({"tool": tool, "args": args, "result": result})

        fm = proc.front_matter
        part_ids: set[str] = set()
        for key in ("required_tools", "required_parts"):
            for pid in fm.get(key) or []:
                part_ids.add(pid)
        for cond in fm.get("entry_conditions") or []:
            if isinstance(cond, dict) and cond.get("part"):
                part_ids.add(cond["part"])

        for step in (s for s in proc.steps if step_id is None or s.get("id") == step_id):
            for cond in step.get("preconditions") or []:
                if isinstance(cond, dict) and cond.get("part"):
                    part_ids.add(cond["part"])
            for pid in step.get("required_parts") or []:
                part_ids.add(pid)
            fastener_id = _fastener_from_specs(step.get("specs"))
            if fastener_id:
                add(
                    "get_torque_spec",
                    {"fastener_id": fastener_id},
                    reference_tools.get_torque_spec(fastener_id),
                )

        for pid in sorted(part_ids):
            add("check_inventory", {"part_id": pid}, reference_tools.check_inventory(pid))
        return facts

    def _log_timing(self, t_start, retrieve_ms, generate_ms, tts_ms, images) -> None:
        if not config.TIMING:
            return
        print(
            f"[timing] retrieve={retrieve_ms:.0f}ms "
            f"generate={generate_ms:.0f}ms x1 "
            f"tts={tts_ms:.0f}ms vision={'on' if images else 'off'} "
            f"total={(time.perf_counter() - t_start) * 1000:.0f}ms",
            flush=True,
        )

    # ------------------------------------------------------------------
    def _wants_diagram_vision(self, query: str, live_image_path) -> bool:
        """Whether to feed a reference diagram to the vision model this turn.

        Running the vision encoder on a ~1MP schematic is the single biggest per-turn
        cost, and most turns are verbal/sensor checks that don't need it. Always use
        vision when the technician sent a camera frame; honor an explicit
        ZD_DIAGRAM_VISION override; otherwise only when the utterance sounds visual.
        """
        if live_image_path is not None:
            return True
        mode = config.DIAGRAM_VISION
        if mode == "always":
            return True
        if mode == "off":
            return False
        q = (query or "").lower()
        return any(hint in q for hint in _VISION_QUERY_HINTS)

    def _collect_images(self, retrieval, live_image_path, want_diagram) -> list[str]:
        images: list[str] = []
        if want_diagram:
            for d in retrieval.diagrams:
                if len(images) >= _MAX_DIAGRAM_IMAGES:
                    break
                if d.image_exists:
                    images.append(d.image_path)
        if live_image_path:
            images.append(str(live_image_path))
        return images

    def _attached_diagram_ids(self, retrieval, want_diagram) -> list[str]:
        """Ids of the diagrams actually sent to the vision model (same cap as images),
        so the prompt only claims an image is 'attached' when it truly is."""
        ids: list[str] = []
        if not want_diagram:
            return ids
        for d in retrieval.diagrams:
            if len(ids) >= _MAX_DIAGRAM_IMAGES:
                break
            if d.image_exists:
                ids.append(d.diagram_id)
        return ids

    def _speak(self, text: str):
        from ..models import tts

        return tts.synthesize_to_file(text)
