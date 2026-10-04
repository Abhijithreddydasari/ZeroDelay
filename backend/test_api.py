"""HTTP contract and stream failure tests without model downloads."""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

os.environ["ZD_WARMUP"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
from .agent.session_engine import SessionStore  # noqa: E402
from .api import server  # noqa: E402


def silent_wav() -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 16000)
    return out.getvalue()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous = server._sessions
        server._sessions = SessionStore(Path(self.temp.name) / "sessions.sqlite")
        self.client = TestClient(server.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        server._sessions = self.previous
        self.temp.cleanup()

    def test_session_api_and_replay(self):
        session = self.client.post("/sessions").json()
        sid = session["id"]
        for i, query in enumerate(("airlock procedure", "confirm procedure", "next step")):
            r = self.client.post("/ask", json={"query": query, "session_id": sid,
                "turn_id": f"t{i}", "speak": False})
            self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["session"]["step_id"], 2)
        replay = self.client.post("/ask", json={"query": "next step", "session_id": sid,
            "turn_id": "t2", "speak": False}).json()
        self.assertTrue(replay["replayed"])
        self.assertEqual(self.client.get(f"/sessions/{sid}").json()["step_id"], 2)

    def test_ready_reports_loaded_models(self):
        checks = self.client.get("/ready").json()
        self.assertIn("asr_loaded", checks)
        self.assertIn("piper_loaded", checks)
        self.assertIn("embedder_loaded", checks)
        self.assertIn("gemma_loaded", checks)
        self.assertEqual(checks["ready"], all(checks[key] for key in (
            "vector_index", "asr_loaded", "piper_loaded", "embedder_loaded", "gemma_loaded")))

    def test_stream_events_and_error(self):
        sid = self.client.post("/sessions").json()["id"]
        form = {"session_id": sid, "turn_id": "voice-1"}
        upload = {"audio": ("input.wav", silent_wav(), "audio/wav")}
        with patch.object(server.vad, "detect_speech", return_value=True), \
             patch.object(server.asr, "transcribe", return_value="airlock procedure"), \
             patch("backend.models.tts.synthesize_wav_chunks", return_value=iter([b"wav"])):
            r = self.client.post("/converse/stream", data=form, files=upload)
        self.assertEqual(r.status_code, 200)
        events = [json.loads(line) for line in r.text.splitlines()]
        self.assertEqual([e["type"] for e in events], ["query", "audio_chunk", "final"])
        self.assertEqual(events[-1]["result"]["session"]["pending_procedure_id"], "EVA-PREP-001")

        with patch.object(server.vad, "detect_speech", return_value=True), \
             patch.object(server.asr, "transcribe", return_value="next step"), \
             patch.object(server, "_session_turn", side_effect=RuntimeError("generation failed")):
            r = self.client.post("/converse/stream", data={"session_id": sid, "turn_id": "voice-2"}, files=upload)
        events = [json.loads(line) for line in r.text.splitlines()]
        self.assertEqual([e["type"] for e in events], ["query", "error"])
        self.assertEqual(self.client.get(f"/sessions/{sid}").json()["step_id"], None)

    def test_silence_does_not_create_a_turn(self):
        sid = self.client.post("/sessions").json()["id"]
        response = self.client.post("/converse/stream",
            data={"session_id": sid, "turn_id": "silent"},
            files={"audio": ("input.wav", silent_wav(), "audio/wav")})
        self.assertEqual(response.status_code, 422)
        self.assertIsNone(server.get_sessions().get_turn(sid, "silent"))

    def test_tts_failure_does_not_commit_advance(self):
        sid = self.client.post("/sessions").json()["id"]
        with patch("backend.models.tts.synthesize_wav_chunks", side_effect=RuntimeError("tts failed")):
            response = self.client.post("/ask", json={"query": "airlock procedure",
                "session_id": sid, "turn_id": "tts-failed", "speak": False})
            self.assertEqual(response.status_code, 200)
            with self.assertRaises(RuntimeError):
                self.client.post("/ask", json={"query": "confirm procedure",
                    "session_id": sid, "turn_id": "confirm-failed", "speak": True})
        self.assertIsNone(self.client.get(f"/sessions/{sid}").json()["step_id"])
        self.assertIsNone(server.get_sessions().get_turn(sid, "confirm-failed"))


if __name__ == "__main__":
    unittest.main()
