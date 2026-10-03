# Local voice latency and validation

Measured on an RTX 5060 Laptop GPU with the models cached and `ZD_OFFLINE=1`.
Run `python -m backend.benchmark_voice` for the procedure path and
`python -m backend.benchmark_model` for open-ended text turns. `/metrics/latency`
reports per-stage p50/p95 for turns served by the running API process.

| Path | Samples | p50 | p95 | Measurement boundary |
| --- | ---: | ---: | ---: | --- |
| Warm procedure voice, Piper-generated input | 20 | 2.60 s | 3.15 s | Request start to first server audio event |
| Open-ended E4B text | 4 | 39.94 s | 57.50 s | Request start to full decision |

The four open-ended E4B turns took 31.74–57.50 s, using 1035–1411 prompt
tokens and 127–222 output tokens. Gemma loading and warmup took 60.35 s in
that run. Cold start is tracked separately from the warm-turn figures.

The synthetic voice set exercises procedure selection for all four procedures,
confirmation, next step or acknowledgment, and repeat through ASR, the session
engine, and Piper. Unit tests cover the four
procedure definitions, noncontiguous step IDs, wrong-procedure requests, unsafe
sensors, negated confirmations, silence, persistence, replay, and stream errors.
The original Gemma ASR path failed while loading `llvmlite.dll`; the default
faster-whisper small.en CPU-int8 path completed the offline synthetic run.

The 15-second target is for **first audible guidance at p95**. The table measures
the server's first audio event, so a recorded natural-speech set and a real
microphone/speaker run are still needed to accept that target. E2B was not
benchmarked; E4B remains the default until E2B passes the same answer and
safety checks.
