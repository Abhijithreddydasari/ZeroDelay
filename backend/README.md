# ZeroDelay Backend

Offline voice-copilot backend: **faster-whisper** (speech recognition), **Gemma 4** (reasoning + vision) +
multimodal RAG over [`data/`](../data) + **Piper** TTS, served over FastAPI. Runs 100%
locally after a one-time model download. Built for 8GB GPUs (RTX 5060 / 3070).

## Pipeline

```
audio -> faster-whisper STT -> persisted procedure state + safety checks
      -> routine command: verified step guidance -> Piper TTS
      -> open-ended question: Gemma reasoning + vision -> validated decision -> Piper TTS
```

Retrieval only chooses *which* procedure/diagram applies; exact facts (torque, sensor
ranges, inventory, fault branches) come from deterministic tool calls, not vector search.

## Setup and downloads (one time, then fully offline)

Windows / PowerShell steps. Download the selected models and voice once; the service runs offline after that.

### 1. Virtual environment

```powershell
cd path\to\LMT
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install PyTorch FIRST (GPU-specific)

Install `torch`, `torchvision`, and `torchaudio` together from the same CUDA index.
`torchvision` is required by Gemma 4's image processor.

```powershell
# RTX 3070 (Ampere):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# RTX 5060 (Blackwell - needs newer CUDA):
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify the GPU is visible:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3. Install the rest

```powershell
pip install -r backend\requirements.txt
```

### 4. Hugging Face login + accept Gemma licenses

Gemma weights are gated. Click "agree" once (while logged in) on the model pages you use:
`google/gemma-4-E4B-it` and `google/embeddinggemma-300M`. Then:

```powershell
hf auth login
```

Tip: to keep the large cache off C:, set `$env:HF_HOME="D:\hf-cache"` before downloading.

### 5. Download the models

```powershell
hf download google/gemma-4-E4B-it        # reasoning + vision
hf download google/embeddinggemma-300M   # retrieval embeddings
hf download Systran/faster-whisper-small.en # CPU speech recognition
```

These live in the HF cache and are found automatically (no path config needed).

### 6. Download a Piper voice (TTS)

The code expects the voice pair in `backend\models\piper\`:

```powershell
hf download rhasspy/piper-voices `
  en/en_US/lessac/medium/en_US-lessac-medium.onnx `
  en/en_US/lessac/medium/en_US-lessac-medium.onnx.json `
  --local-dir backend\models\piper
```

If the files land in nested subfolders, move the `.onnx` and `.onnx.json` directly into
`backend\models\piper\`. (Alternative: `python -m piper.download_voices en_US-lessac-medium`,
then move the files there.)

### 7. Diagram images (vision path)

The 8 diagram PNGs live in [`data/diagrams/`](../data/diagrams). Their filenames match
the diagram IDs in [`data/manifest.yaml`](../data/manifest.yaml). The vision model
reads the PNGs directly; no annotation or prompt files are required.

### Download / runtime footprint

| Item | Download | Runtime cost |
| --- | --- | --- |
| Gemma 4 E4B | ~10 GB | ~4-5 GB VRAM (4-bit) |
| EmbeddingGemma | ~1.2 GB | CPU |
| faster-whisper small.en | ~500 MB | CPU (int8) |
| Piper voice | ~60 MB | CPU |

E2B is an optional model for comparison after it has passed the same safety and answer tests.
On Blackwell (5060), a `bitsandbytes` error usually means it needs the
newest `bitsandbytes`; as a last resort run on a machine with headroom or set
`$env:ZD_GEMMA_4BIT="0"`.

## Build the index

```
python -m backend.index.build_index
# or: python -m backend.cli build-index
```

## Test without the front-end

```
python -m backend.smoke_test                       # no ML deps needed
python -m backend.cli info                          # corpus stats
python -m backend.cli retrieve "airlock won't depressurize"
python -m backend.cli ask "coolant loop pressure is dropping"
python -m backend.benchmark_voice                    # synthetic voice path, offline
```

## Run the API

```powershell
$env:ZD_OFFLINE="1" # Set before launching, after downloads and index construction.
python -m uvicorn backend.api.server:app --port 8000
# Models warm in a background thread by default; set ZD_WARMUP=0 to disable.
```

Check `http://127.0.0.1:8000/ready` and wait for `ready=true` before the voice demo.

### Endpoints (for the JS front-end)

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/health` | - | status + model id |
| GET | `/ready` | - | model/index readiness and startup timing |
| GET | `/metrics/latency` | - | local p50/p95 stage timings |
| POST | `/sessions` | - | create a persistent procedure session |
| GET | `/sessions/{id}` | - | authoritative procedure and step state |
| GET | `/procedures` | - | procedure list |
| GET | `/sensors` | - | live telemetry snapshot |
| POST | `/sensors/inject` | `{name, value}` | new reading (demo anomalies) |
| POST | `/sensors/reset` | - | reset to nominal |
| POST | `/transcribe` | multipart `audio` | `{text}` |
| POST | `/ask` | `{query, session_id?, turn_id?, image_base64?, speak?}` | decision + authoritative session state; audio if requested |
| POST | `/converse` | multipart `audio` (+ `session_id`, `turn_id`, `image?`) | decision, session, audio chunks |
| POST | `/converse/stream` | same multipart fields | NDJSON query, audio_chunk, final/error |
| POST | `/tts` | `{text}` | `audio/wav` |

Retrieved diagram PNGs are also served read-only at `/diagrams/<diagram_id>.png`, so the
front-end can show the schematic the vision model just looked at. The decision JSON shape
is documented in [`agent/schema.py`](agent/schema.py).

Persistent turns require both `session_id` and `turn_id`; reuse the same turn ID
when retrying to avoid repeating a state change. Procedure state is stored in
`backend/artifacts/sessions.sqlite`; transcripts remain in frontend `localStorage`.
CLI calls use the model-only path, and CLI `converse` still uses Gemma ASR.

## Configuration

Override via environment variables (see [`config.py`](config.py)): `ZD_OFFLINE`,
`ZD_GEMMA_MODEL`, `ZD_GEMMA_4BIT`, `ZD_GEMMA_DEVICE_MAP`, `ZD_EMBED_MODEL`,
`ZD_EMBED_DEVICE`, `ZD_EMBED_DIM`, `ZD_PIPER_VOICE`, `ZD_TOP_K`, `ZD_TOOL_LOOP`,
`ZD_MAX_NEW_TOKENS`, `ZD_ASR_BACKEND`, `ZD_ASR_MODEL`, `ZD_WARMUP`.

Select a procedure by voice, then say `confirm procedure`. Routine commands use
the deterministic fast path; caution and critical steps require confirmation or
read-back and current sensor checks. `python -m backend.benchmark_voice` measures
warmed synthetic-audio latency. Verify with a real microphone as well.
Recorded results and pending validation are in [`PERFORMANCE.md`](PERFORMANCE.md).

## Troubleshooting (read this before integrating)

These are the exact errors we already hit and fixed. If setup was done correctly you
should not see them, but they're documented so nobody loses time.

### `ModuleNotFoundError: No module named 'torchvision'`
Gemma 4 is multimodal, so `transformers` imports `torchvision` for the image path even on
text-only calls. Install it from the **same CUDA index** as `torch` (a plain
`pip install torchvision` can pull a CPU/mismatched build):

```powershell
# match your torch build - check with:  python -c "import torch; print(torch.__version__)"
# cu128 (RTX 5060):
pip install --pre torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# cu124 (RTX 3070):
pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### `ValueError: Some modules are dispatched on the CPU or the disk`
bitsandbytes 4-bit cannot keep layers on CPU. With `device_map="auto"` on a tight 8GB card,
accelerate offloads part of the model to CPU and the quantizer aborts. We fixed this in
[`models/gemma.py`](models/gemma.py): when a CUDA GPU is present the model is pinned fully to
it (`device_map={"": 0}`) instead of `"auto"`. Override with `ZD_GEMMA_DEVICE_MAP` if needed
(`auto`, `cpu`, `cuda:0`). A smaller model needs its own safety and answer checks before use.

### `FileNotFoundError: Piper voice not found ...`
`hf download rhasspy/piper-voices ...` saves the voice into nested subfolders
(`piper\en\en_US\lessac\medium\`). The code expects the pair directly in
`backend\models\piper\`. Flatten them:

```powershell
Move-Item -Force backend\models\piper\en\en_US\lessac\medium\en_US-lessac-medium.onnx      backend\models\piper\
Move-Item -Force backend\models\piper\en\en_US\lessac\medium\en_US-lessac-medium.onnx.json backend\models\piper\
Remove-Item -Recurse -Force backend\models\piper\en
```

You should end up with exactly `en_US-lessac-medium.onnx` (~63 MB) and
`en_US-lessac-medium.onnx.json` in `backend\models\piper\`.

### First call is slow
Cold start loads the ASR, embedding, TTS, and Gemma models. `/ready` reports when all
required models and the index are available. Warming is enabled by default; use
`ZD_WARMUP=0` only for targeted tests. Procedure commands bypass Gemma after warmup.

### Verify the whole pipeline quickly
```powershell
python -c "import torch, torchvision; print(torch.cuda.is_available(), torchvision.__version__)"
python -m backend.cli ask "coolant loop pressure is dropping"      # text -> structured JSON
python -m backend.benchmark_voice                          # synthetic voice -> ASR -> session -> TTS
```

## Scope

This backend includes persistent procedure sessions, deterministic common commands,
sensor and inventory gates, model-assisted open-ended answers, and in-memory TTS chunks.
The bundled procedures and sensor readings are synthetic demonstration data.
