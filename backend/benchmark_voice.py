"""Repeatable synthetic-audio benchmark for the warmed procedure voice path.

Run: python -m backend.benchmark_voice
This exercises API/ASR/state/TTS on the local machine; a real microphone check is
still required before treating the latency target as accepted.
"""
from __future__ import annotations

import io
import json
import math
import os
import tempfile
from pathlib import Path

os.environ["ZD_WARMUP"] = "0"
os.environ["ZD_OFFLINE"] = "1"

from fastapi.testclient import TestClient  # noqa: E402
from .api.server import app  # noqa: E402
from .models import asr, tts, vad  # noqa: E402
from .tools.sensor_sim import get_simulator  # noqa: E402


def p(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def main() -> int:
    procedures = (("airlock", "EVA-PREP-001"), ("coolant", "COOL-JMP-002"),
                  ("battery", "BATT-ORU-003"), ("carbon dioxide scrubber", "CDRA-FAULT-004"))
    phrases = tuple({f"{name} procedure" for name, _ in procedures}) + (
        "confirm procedure", "next step", "acknowledged", "repeat")
    audio = {phrase: tts.synthesize_to_bytes(phrase) for phrase in phrases}
    asr.load()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio["next step"])
        warm_path = Path(f.name)
    try:
        asr.transcribe(warm_path)
        vad.detect_speech(warm_path)
    finally:
        warm_path.unlink(missing_ok=True)

    samples: list[float] = []
    with TestClient(app) as client:
        start_cycle = int(os.environ.get("ZD_BENCH_START", "0"))
        cycles = int(os.environ.get("ZD_BENCH_CYCLES", "5"))
        for cycle in range(start_cycle, start_cycle + cycles):
            get_simulator().reset()
            name, procedure_id = procedures[cycle % len(procedures)]
            if procedure_id == "CDRA-FAULT-004":
                get_simulator().inject("cabin_co2_mmHg", 4.0)
            command = "acknowledged" if procedure_id == "CDRA-FAULT-004" else "next step"
            turn_phrases = (f"{name} procedure", "confirm procedure", command, "repeat")
            sid = client.post("/sessions").json()["id"]
            for i, phrase in enumerate(turn_phrases):
                response = client.post("/converse/stream",
                    data={"session_id": sid, "turn_id": f"{cycle}-{i}"},
                    files={"audio": ("input.wav", io.BytesIO(audio[phrase]), "audio/wav")})
                events = [json.loads(line) for line in response.text.splitlines()]
                error = next((e for e in events if e["type"] == "error"), None)
                if error:
                    raise RuntimeError(error["detail"])
                final = next(e["result"] for e in events if e["type"] == "final")
                first_audio = final["timing_ms"]["first_audio"]
                if first_audio is None:
                    raise RuntimeError(f"No audio for {phrase!r}")
                state = final["session"]
                if i == 0 and state["pending_procedure_id"] != procedure_id:
                    raise RuntimeError(f"ASR or selection failed for {phrase!r}: {final['query']!r}")
                if i >= 1 and (state["procedure_id"] != procedure_id or state["step_id"] != (1 if i == 1 else 2)):
                    raise RuntimeError(f"Unexpected state for {phrase!r}, ASR={final['query']!r}: {state}")
                samples.append(first_audio)
                print(f"{phrase!r}: {first_audio:.0f} ms, ASR={final['query']!r}, step={state['step_id']}", flush=True)
    get_simulator().reset()
    median, p95 = p(samples, 0.50), p(samples, 0.95)
    print(f"Warmed synthetic voice: n={len(samples)}, p50={median:.0f} ms, p95={p95:.0f} ms")
    return 0 if p95 < 15000 else 1


if __name__ == "__main__":
    raise SystemExit(main())
