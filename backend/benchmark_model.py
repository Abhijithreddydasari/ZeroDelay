"""Compare local Gemma models on the same four grounded text questions.

Run separately for each model, for example:
  ZD_OFFLINE=1 ZD_GEMMA_MODEL=google/gemma-4-E4B-it python -m backend.benchmark_model
Results are printed; this does not change the configured default model.
"""
from __future__ import annotations

import math
import time

from . import config
from .agent.orchestrator import Orchestrator


CASES = (
    ("EVA-PREP-001", 1, "What battery reading must I confirm before I continue?"),
    ("BATT-ORU-003", 2, "Is it safe to demate the battery connector now?"),
    ("COOL-JMP-002", 1, "Which tools are required for this step?"),
    ("CDRA-FAULT-004", 1, "What does the cabin CO2 reading mean?"),
)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(p * len(ordered)) - 1]


def main() -> None:
    orchestrator = Orchestrator()
    started = time.perf_counter()
    orchestrator.gemma.warmup()
    print(f"model={config.GEMMA_MODEL_ID} load_and_warmup_ms={round((time.perf_counter()-started)*1000)}", flush=True)
    samples = []
    for procedure_id, step_id, query in CASES:
        started = time.perf_counter()
        result = orchestrator.process_text(query, speak=False, procedure_id=procedure_id, step_id=step_id)
        elapsed = (time.perf_counter() - started) * 1000
        samples.append(elapsed)
        decision = result["decision"]
        print(f"{procedure_id} step={step_id} elapsed_ms={round(elapsed)} action={decision['action']} "
              f"parse_valid={result['parse_valid']} input_tokens={result['timing_ms'].get('input_tokens')} "
              f"output_tokens={result['timing_ms'].get('output_tokens')}", flush=True)
    print(f"open_ended_n={len(samples)} p50_ms={round(percentile(samples,.5))} "
          f"p95_ms={round(percentile(samples,.95))}")


if __name__ == "__main__":
    main()
