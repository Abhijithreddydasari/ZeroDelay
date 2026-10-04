"""A generation failure must close the stream and reach its caller."""
from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from .models.gemma import GemmaRuntime


class _Inputs(dict):
    def to(self, _device):
        return self


class _Model:
    device = "cpu"

    def generate(self, **_kwargs):
        raise RuntimeError("model failed")


class _Streamer:
    def __init__(self, *_args, **_kwargs):
        self.done = threading.Event()

    def end(self):
        self.done.set()

    def __iter__(self):
        if not self.done.wait(2):
            raise AssertionError("stream never ended")
        return iter(())


class GemmaStreamTests(unittest.TestCase):
    def test_worker_error_reaches_consumer(self):
        runtime = GemmaRuntime()
        runtime._model = _Model()
        runtime._processor = object()
        runtime._build_inputs = lambda *_args: _Inputs(input_ids=[[1]])
        with patch("transformers.TextIteratorStreamer", _Streamer):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                list(runtime._generate_stream([{"role": "user", "content": "test"}], 4))


if __name__ == "__main__":
    unittest.main()
