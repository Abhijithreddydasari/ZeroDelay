"""Small offline ASR on CPU, leaving GPU memory for Gemma."""
from __future__ import annotations

import threading
from pathlib import Path

from .. import config

_model = None
_lock = threading.Lock()


def load():
    global _model
    if config.ASR_BACKEND == "gemma":
        from .gemma import get_runtime
        runtime = get_runtime()
        runtime.warmup()
        return runtime
    if _model is None:
        with _lock:
            if _model is None:
                from faster_whisper import WhisperModel
                _model = WhisperModel(
                    config.ASR_MODEL_ID, device="cpu", compute_type="int8",
                    local_files_only=config.OFFLINE,
                )
    return _model


def is_loaded() -> bool:
    if config.ASR_BACKEND == "gemma":
        from .gemma import get_runtime
        return get_runtime().is_loaded
    return _model is not None


def transcribe(path: str | Path) -> str:
    if config.ASR_BACKEND == "gemma":
        from .gemma import get_runtime
        return get_runtime().transcribe(path)
    segments, _ = load().transcribe(str(path), beam_size=1, language="en", vad_filter=False)
    return " ".join(segment.text.strip() for segment in segments).strip()
