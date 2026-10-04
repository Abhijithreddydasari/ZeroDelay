"""Small local latency sample window; never stores transcripts or audio."""
from __future__ import annotations

import math
import threading
from collections import deque

_samples: deque[tuple[str, dict[str, float]]] = deque(maxlen=500)
_lock = threading.Lock()


def record(kind: str, timings: dict[str, float]) -> None:
    with _lock:
        _samples.append((kind, {k: float(v) for k, v in timings.items()}))


def summary() -> dict:
    with _lock:
        rows = list(_samples)
    output = {}
    for kind in sorted({k for k, _ in rows}):
        subset = [t for k, t in rows if k == kind]
        values = {}
        for metric in sorted({m for row in subset for m in row}):
            samples = sorted(row[metric] for row in subset if metric in row)
            values[metric] = {
                "p50": samples[math.ceil(0.50 * len(samples)) - 1],
                "p95": samples[math.ceil(0.95 * len(samples)) - 1],
                "samples": len(samples),
            }
        output[kind] = {"turns": len(subset), "timing_ms": values}
    return output
