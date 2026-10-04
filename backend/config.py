"""Central configuration for the ZeroDelay backend.

All paths, model ids, and tunables live here so the rest of the code stays clean.
Everything is designed to run fully offline once the models are downloaded.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    """Parse an int env var, falling back to the default on a bad value."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
DATA_DIR = REPO_ROOT / "data"

PROCEDURES_DIR = DATA_DIR / "procedures"
REFERENCE_DIR = DATA_DIR / "reference"
DIAGRAMS_DIR = DATA_DIR / "diagrams"
ANNOTATIONS_DIR = DIAGRAMS_DIR / "annotations"
MANIFEST_PATH = DATA_DIR / "manifest.yaml"

# Generated artifacts (gitignored).
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
VECTOR_DB_PATH = ARTIFACTS_DIR / "vectorstore.sqlite"
TTS_OUTPUT_DIR = ARTIFACTS_DIR / "tts"

# ---------------------------------------------------------------------------
# Models (Hugging Face ids; downloaded once, then run offline)
# ---------------------------------------------------------------------------
# Gemma handles reasoning + vision, plus optional legacy ASR. E2B needs validation.
GEMMA_MODEL_ID = os.environ.get("ZD_GEMMA_MODEL", "google/gemma-4-E4B-it")
GEMMA_FALLBACK_MODEL_ID = "google/gemma-4-E2B-it"  # No automatic fallback is applied.

# Load Gemma in 4-bit (bitsandbytes) to fit 8GB GPUs. Audio modules stay bf16.
GEMMA_LOAD_IN_4BIT = os.environ.get("ZD_GEMMA_4BIT", "1") == "1"
# Reasoning / "thinking" mode. This model's chat template exposes a hidden reasoning
# channel gated by `enable_thinking` (OFF by default). Enabling it makes Gemma draft a
# reasoning block before its final answer: often better on hard cases, but slower and
# more tokens (bump ZD_MAX_NEW_TOKENS so the JSON decision still fits). STT never thinks.
GEMMA_THINKING = os.environ.get("ZD_GEMMA_THINKING", "0") == "1"
# Offline mode: force local-only model loading (no Hugging Face network calls). Required
# for the offline/privacy demo. Enable with ZD_OFFLINE=1 once all models are cached.
OFFLINE = os.environ.get("ZD_OFFLINE", "0") == "1"
if OFFLINE:
    # Belt-and-suspenders: covers transformers, sentence-transformers, and hf_hub,
    # including any indirect loads we don't pass local_files_only to explicitly.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Device placement. On an 8GB GPU, bitsandbytes 4-bit forbids CPU-offloaded layers,
# so we pin the whole model to one GPU rather than letting "auto" spill to CPU.
# Override with ZD_GEMMA_DEVICE_MAP (e.g. "auto", "cpu", "cuda:0").
GEMMA_DEVICE_MAP = os.environ.get("ZD_GEMMA_DEVICE_MAP", "").strip()
# Modules that must NOT be quantized (Gemma 4 audio pitfall) + the lm_head.
GEMMA_QUANT_SKIP_MODULES = [
    "lm_head",
    "audio_tower",
    "embed_audio",
    "model.audio_tower",
    "model.embed_audio",
]

# Embeddings: EmbeddingGemma via sentence-transformers, on CPU to keep GPU for Gemma.
EMBED_MODEL_ID = os.environ.get("ZD_EMBED_MODEL", "google/embeddinggemma-300M")
EMBED_DEVICE = os.environ.get("ZD_EMBED_DEVICE", "cpu")
# Matryoshka truncation: 256 dims is ~3x faster with minimal quality loss.
EMBED_DIM = _env_int("ZD_EMBED_DIM", 256)
EMBED_QUERY_PROMPT = "Retrieval-query"
EMBED_DOCUMENT_PROMPT = "Retrieval-document"

# TTS: Piper voice (download the .onnx + .onnx.json pair into models/piper/).
PIPER_VOICE_DIR = BACKEND_DIR / "models" / "piper"
PIPER_VOICE_NAME = os.environ.get("ZD_PIPER_VOICE", "en_US-lessac-medium")
PIPER_VOICE_PATH = PIPER_VOICE_DIR / f"{PIPER_VOICE_NAME}.onnx"

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
AUDIO_SAMPLE_RATE = 16000  # Gemma expects 16kHz mono
AUDIO_MAX_SECONDS = 30     # Gemma 4 audio clips are capped at 30s

# ---------------------------------------------------------------------------
# Retrieval / agent tunables
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K = _env_int("ZD_TOP_K", 3)
TOOL_LOOP_MAX_ITERS = _env_int("ZD_TOOL_LOOP", 2)
GEMMA_MAX_NEW_TOKENS = _env_int("ZD_MAX_NEW_TOKENS", 512)
ASR_MAX_NEW_TOKENS = _env_int("ZD_ASR_MAX_NEW_TOKENS", 128)
ASR_BACKEND = os.environ.get("ZD_ASR_BACKEND", "faster-whisper")
ASR_MODEL_ID = os.environ.get("ZD_ASR_MODEL", "Systran/faster-whisper-small.en")

# Vision gating. Feeding a ~1MP schematic runs Gemma's vision encoder — the biggest
# per-turn cost — and most turns don't need it. "auto" attaches a diagram only when the
# turn looks visual (a live camera frame, or a visual-sounding request); "always" keeps
# the old every-turn behavior; "off" never attaches a reference diagram.
DIAGRAM_VISION = os.environ.get("ZD_DIAGRAM_VISION", "auto").strip().lower()
# Log a per-turn latency breakdown (retrieve / generate / tts) to stdout for profiling.
TIMING = os.environ.get("ZD_TIMING", "0") == "1"


def ensure_dirs() -> None:
    """Create generated-artifact directories if missing."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PIPER_VOICE_DIR.mkdir(parents=True, exist_ok=True)
