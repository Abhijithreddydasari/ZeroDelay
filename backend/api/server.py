"""FastAPI service exposing the ZeroDelay pipeline to the (JS) front-end.

Run:  uvicorn backend.api.server:app --port 8000
Models warm in the background at startup by default; ZD_WARMUP=0 disables warming.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Guardrail so a huge upload / base64 blob can't exhaust memory or disk.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

from .. import config
from ..data_loader import load_corpus
from ..models import vad
from ..models import asr
from .. import metrics
from ..tools.sensor_sim import get_simulator
from ..agent.session_engine import SessionStore, fast_decision, session_view, validate_corpus

app = FastAPI(title="ZeroDelay Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the diagram PNGs so the front-end can display retrieved diagrams in the
# chat (they live in data/diagrams/<diagram_id>.png). Read-only static mount.
if config.DIAGRAMS_DIR.exists():
    app.mount(
        "/diagrams",
        StaticFiles(directory=str(config.DIAGRAMS_DIR)),
        name="diagrams",
    )

_orchestrator = None
_sessions = None
_warm_status = {"status": "starting", "error": None, "startup_ms": None}


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from ..agent.orchestrator import Orchestrator

        _orchestrator = Orchestrator()
    return _orchestrator


def get_sessions() -> SessionStore:
    global _sessions
    if _sessions is None:
        _sessions = SessionStore()
    return _sessions


def _warm_models() -> None:
    start = time.perf_counter()
    try:
        from ..models import embedder, tts
        asr.load()
        tts._get_voice()
        embedder._get_model()
        get_orchestrator().gemma.warmup()
        _warm_status["status"] = "ready"
    except Exception as exc:
        _warm_status["status"] = "error"
        _warm_status["error"] = str(exc)
    finally:
        _warm_status["startup_ms"] = round((time.perf_counter() - start) * 1000)


@app.on_event("startup")
def _startup() -> None:
    config.ensure_dirs()
    validate_corpus()
    get_sessions()
    if os.environ.get("ZD_WARMUP", "1") == "1":
        threading.Thread(target=_warm_models, daemon=True).start()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    query: str
    image_base64: str | None = None
    speak: bool = True
    session_id: str | None = None
    turn_id: str | None = None


class TTSRequest(BaseModel):
    text: str


class InjectRequest(BaseModel):
    name: str
    value: float | str


# ---------------------------------------------------------------------------
# Info endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "gemma_model": config.GEMMA_MODEL_ID}


@app.get("/ready")
def ready() -> dict:
    """Report index availability and actual ASR, Piper, embedding, and Gemma loading."""
    from ..index.retriever import index_is_ready
    from ..models import embedder, tts

    checks = {
        "vector_index": index_is_ready(),
        "piper_voice": config.PIPER_VOICE_PATH.exists(),
        "piper_loaded": bool(tts._get_voice.cache_info().currsize),
        "embedder_loaded": bool(embedder._get_model.cache_info().currsize),
        "offline_mode": config.OFFLINE,
        "asr_loaded": asr.is_loaded(),
        "gemma_loaded": bool(_orchestrator and _orchestrator.gemma.is_loaded),
        "warmup": _warm_status.copy(),
    }
    checks["ready"] = bool(checks["vector_index"] and checks["piper_loaded"] and
                           checks["embedder_loaded"] and checks["asr_loaded"] and checks["gemma_loaded"])
    return checks


@app.get("/metrics/latency")
def latency_metrics() -> dict:
    return metrics.summary()


@app.post("/sessions")
def create_session() -> dict:
    return session_view(get_sessions().create())


@app.get("/sessions/{session_id}")
def read_session(session_id: str) -> dict:
    state = get_sessions().get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session_view(state)


@app.get("/procedures")
def procedures() -> dict:
    corpus = load_corpus()
    return {
        "procedures": [
            {
                "id": p.procedure_id,
                "title": p.title,
                "system": p.system,
                "summary": p.summary,
                "steps": len(p.steps),
            }
            for p in corpus.procedures.values()
        ]
    }


@app.get("/sensors")
def sensors() -> dict:
    return {"sensors": get_simulator().snapshot()}


@app.post("/sensors/inject")
def inject(req: InjectRequest) -> dict:
    try:
        return {"reading": get_simulator().inject(req.name, req.value)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/sensors/reset")
def reset_sensors() -> dict:
    get_simulator().reset()
    return {"status": "reset"}


def _session_turn(query: str, session_id: str, turn_id: str, speak: bool,
                  image_path: Path | None = None) -> dict:
    """Serialize one session turn and persist its result for safe replay."""
    from ..models import tts

    store = get_sessions()
    with store.locked(session_id):
        cached = store.get_turn(session_id, turn_id)
        if cached is not None:
            if speak and cached["decision"].get("spoken_text"):
                cached["audio_chunks_base64"] = [
                    base64.b64encode(data).decode("ascii")
                    for data in tts.synthesize_wav_chunks(cached["decision"]["spoken_text"])
                ]
            cached["replayed"] = True
            return cached
        state = store.get(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        start = time.perf_counter()
        decision = fast_decision(state, query)
        if decision is not None:
            result = {"query": query, "decision": decision.to_dict(),
                      "retrieval": {"procedures": [], "diagrams": []},
                      "sensor_snapshot": get_simulator().snapshot(), "tool_calls": [],
                      "fast_path": True}
        else:
            result = get_orchestrator().process_text(
                query, live_image_path=image_path, speak=False,
                procedure_id=state["procedure_id"], step_id=state["step_id"])
            # Language-model output may explain or warn but cannot change procedure state.
            d = result["decision"]
            if (not result.get("parse_valid") or d["action"] in {"advance", "branch", "replan", "tool_request"}
                    or d.get("procedure_id") not in {None, state["procedure_id"]}
                    or d.get("step_id") not in {None, state["step_id"]}):
                d.update(action="clarify", spoken_text="I could not validate that answer against the current step. Please repeat or ask for the current step.",
                         procedure_id=state["procedure_id"], step_id=state["step_id"], tool_request=None)
            result["fast_path"] = False
        decision_ms = round((time.perf_counter() - start) * 1000)
        result["session"] = session_view(state)
        result["timing_ms"] = {**result.get("timing_ms", {}), "decision": decision_ms,
                               "first_text": decision_ms}
        result["tts_wav_base64"] = None
        result["audio_chunks_base64"] = []
        if speak and result["decision"].get("spoken_text"):
            t_tts = time.perf_counter()
            result["audio_chunks_base64"] = [
                base64.b64encode(data).decode("ascii")
                for data in tts.synthesize_wav_chunks(result["decision"]["spoken_text"])
            ]
            result["timing_ms"]["tts"] = round((time.perf_counter() - t_tts) * 1000)
        result["timing_ms"]["total"] = round((time.perf_counter() - start) * 1000)
        stored = dict(result)
        stored["audio_chunks_base64"] = []
        store.commit(state, turn_id, stored)
        return result


# ---------------------------------------------------------------------------
# Pipeline endpoints
# ---------------------------------------------------------------------------
@app.post("/transcribe")
def transcribe(audio: UploadFile = File(...)) -> dict:
    tmp = _save_upload(audio, suffix=".wav")
    norm = None
    try:
        norm = vad.prepare_audio(tmp)
        text = asr.transcribe(norm)
    finally:
        _cleanup(tmp)
        _cleanup(norm)
    return {"text": text}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    image_path = _decode_image(req.image_base64) if req.image_base64 else None
    try:
        if req.session_id:
            if not req.turn_id:
                raise HTTPException(status_code=400, detail="turn_id required with session_id.")
            return _session_turn(req.query, req.session_id, req.turn_id, req.speak, image_path)
        result = get_orchestrator().process_text(
            req.query, live_image_path=image_path, speak=req.speak
        )
    finally:
        _cleanup(image_path)
    return _with_audio(result)


@app.post("/converse")
def converse(
    audio: UploadFile = File(...),
    image: UploadFile | None = File(None),
    speak: bool = Form(True),
    session_id: str | None = Form(None),
    turn_id: str | None = Form(None),
) -> dict:
    request_start = time.perf_counter()
    asr_warm = asr.is_loaded()
    tmp = _save_upload(audio, suffix=".wav")
    image_path = None
    norm = None
    try:
        stage = time.perf_counter()
        norm = vad.prepare_audio(tmp)
        prep_ms = round((time.perf_counter() - stage) * 1000)
        stage = time.perf_counter()
        if not vad.detect_speech(norm):
            raise HTTPException(status_code=422, detail="No speech detected in audio.")
        vad_ms = round((time.perf_counter() - stage) * 1000)
        if image is not None:
            image_path = _save_upload(
                image, suffix=Path(image.filename or "img.png").suffix
            )
        stage = time.perf_counter()
        query = asr.transcribe(norm)
        asr_ms = round((time.perf_counter() - stage) * 1000)
        if not query:
            raise HTTPException(status_code=422, detail="No speech recognized in audio.")
        if session_id:
            if not turn_id:
                raise HTTPException(status_code=400, detail="turn_id required with session_id.")
            result = _session_turn(query, session_id, turn_id, speak, image_path)
        else:
            result = get_orchestrator().process_text(query, live_image_path=image_path, speak=speak)
        result["transcribed"] = True
        result.setdefault("timing_ms", {}).update(audio_prepare=prep_ms, vad=vad_ms, asr=asr_ms,
            first_audio=round((time.perf_counter() - request_start) * 1000) if speak else None,
            request_total=round((time.perf_counter() - request_start) * 1000))
        metrics.record("procedure_warm" if asr_warm and result.get("fast_path") else "voice_other",
                       {k: v for k, v in result["timing_ms"].items() if isinstance(v, (int, float))})
    finally:
        _cleanup(tmp)
        _cleanup(norm)
        _cleanup(image_path)
    return result if session_id else _with_audio(result)


@app.post("/converse/stream")
def converse_stream(
    audio: UploadFile = File(...),
    image: UploadFile | None = File(None),
    speak: bool = Form(True),
    session_id: str | None = Form(None),
    turn_id: str | None = Form(None),
):
    """Streaming sibling of /converse.

    Session turns emit query, validated audio_chunk events, then final or error.
    Their TTS is currently buffered before delivery. The legacy model-only path
    emits text deltas and a final result instead.
    """
    from fastapi.responses import StreamingResponse

    request_start = time.perf_counter()
    asr_warm = asr.is_loaded()
    tmp = _save_upload(audio, suffix=".wav")
    image_path = None
    norm = None
    try:
        stage = time.perf_counter()
        norm = vad.prepare_audio(tmp)
        prep_ms = round((time.perf_counter() - stage) * 1000)
        stage = time.perf_counter()
        if not vad.detect_speech(norm):
            raise HTTPException(status_code=422, detail="No speech detected in audio.")
        vad_ms = round((time.perf_counter() - stage) * 1000)
        if image is not None:
            image_path = _save_upload(
                image, suffix=Path(image.filename or "img.png").suffix
            )
        stage = time.perf_counter()
        query = asr.transcribe(norm)
        asr_ms = round((time.perf_counter() - stage) * 1000)
        if not query:
            raise HTTPException(status_code=422, detail="No speech recognized in audio.")
        if session_id and not turn_id:
            raise HTTPException(status_code=400, detail="turn_id required with session_id.")
    except BaseException:
        # The stream never starts, so drop the (still-unused) image upload here.
        _cleanup(image_path)
        raise
    finally:
        # Audio is only needed for transcription; the reasoning stream never reads it.
        _cleanup(tmp)
        _cleanup(norm)

    def _events():
        try:
            yield _ndjson({"type": "query", "text": query})
            if session_id:
                result = _session_turn(query, session_id, turn_id, speak, image_path)
                first_audio = None
                for chunk in result.get("audio_chunks_base64", []):
                    if first_audio is None:
                        first_audio = round((time.perf_counter() - request_start) * 1000)
                    yield _ndjson({"type": "audio_chunk", "wav_base64": chunk})
                final = dict(result)
                final.pop("audio_chunks_base64", None)
                final["transcribed"] = True
                final["timing_ms"] = {**result.get("timing_ms", {}), "audio_prepare": prep_ms,
                    "vad": vad_ms, "asr": asr_ms, "first_audio": first_audio,
                    "request_total": round((time.perf_counter() - request_start) * 1000)}
                metrics.record("procedure_warm" if asr_warm and result.get("fast_path") else "voice_other",
                    {k: v for k, v in final["timing_ms"].items() if isinstance(v, (int, float))})
                yield _ndjson({"type": "final", "result": final})
                return
            for kind, payload in get_orchestrator().stream_text(
                query, live_image_path=image_path, speak=speak
            ):
                if kind == "delta":
                    yield _ndjson({"type": "delta", "text": payload})
                else:  # "final"
                    final = _with_audio(payload)
                    final["transcribed"] = True
                    yield _ndjson({"type": "final", "result": final})
        except Exception as exc:  # keep the stream well-formed even on failure
            yield _ndjson({"type": "error", "detail": str(exc)})
        finally:
            _cleanup(image_path)

    return StreamingResponse(_events(), media_type="application/x-ndjson")


@app.post("/tts")
def tts(req: TTSRequest):
    from fastapi.responses import Response

    from ..models import tts as tts_engine

    data = tts_engine.synthesize_to_bytes(req.text)
    return Response(content=data, media_type="audio/wav")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_upload(upload: UploadFile, suffix: str) -> Path:
    fd, path = tempfile.mkstemp(suffix=suffix)
    size = 0
    try:
        with os.fdopen(fd, "wb") as f:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Upload too large.")
                f.write(chunk)
        if not size:
            raise HTTPException(status_code=400, detail="Empty upload.")
    except BaseException:
        _cleanup(path)
        raise
    return Path(path)


def _decode_image(b64: str) -> Path:
    if "," in b64:  # strip data URL prefix
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid base64 image.")
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large.")
    fd, path = tempfile.mkstemp(suffix=".png")
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return Path(path)


def _cleanup(path) -> None:
    """Best-effort delete of a temp file."""
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _with_audio(result: dict) -> dict:
    """Inline the TTS wav as base64 so the front-end can play it directly."""
    tts_path = result.get("tts_path")
    if tts_path and Path(tts_path).exists():
        result["tts_wav_base64"] = base64.b64encode(Path(tts_path).read_bytes()).decode()
        _cleanup(tts_path)
        result["tts_path"] = None
    else:
        result["tts_wav_base64"] = None
    return result


def _ndjson(obj: dict) -> str:
    """Serialize one newline-delimited JSON event for the streaming response."""
    return json.dumps(obj) + "\n"
