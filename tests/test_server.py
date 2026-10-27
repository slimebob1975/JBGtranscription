import base64
import os
import time
import unittest
from unittest import mock

import tests  # noqa: F401
import main
from fastapi.testclient import TestClient


RECEIVED = []
REAL_TRANSCRIBER = main.JBGtranscriber.JBGtranscriber


class FakeTranscriber(REAL_TRANSCRIBER):
    """The real class with no models: only what main.py asks of it is recorded."""

    def __init__(self, *args, **kwargs):
        pass

    def perform_transcription_steps(self, **kwargs):
        RECEIVED.append(kwargs)


class ServerTest(unittest.TestCase):
    def setUp(self):
        RECEIVED.clear()
        patcher = mock.patch.object(main.JBGtranscriber, "JBGtranscriber", FakeTranscriber)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(main.app)

    def tearDown(self):
        for folder in (main.UPLOAD_FOLDER, main.RESULTS_FOLDER):
            for path in folder.glob("*"):
                if path.is_file() and path.stat().st_mtime >= self.started:
                    path.unlink(missing_ok=True)

    started = time.time()

    def upload(self, name="a.mp3", **data):
        RECEIVED.clear()
        response = self.client.post("/upload/", files={"file": (name, b"x")}, data={"api_key": "sk", **data})
        return response, (RECEIVED[0] if RECEIVED else None)

    def test_config_and_prompts(self):
        config = self.client.get("/config").json()
        self.assertIn(config["default_model"], config["models"])
        self.assertIn(".m4a", config["audio_extensions"])
        prompts = self.client.get("/analysis_prompts").json()
        self.assertEqual(set(prompts), {"follow_up_questions", "speaker_diarization",
                                        "custom_analysis_preamble", "max_length"})
        options = self.client.get("/summary_prompts").json()["options"]
        self.assertEqual(len(options), 6)

    def test_formats_and_models(self):
        self.assertEqual(self.upload("video.mp4")[0].status_code, 400)
        self.assertEqual(self.upload(model="gpt-9")[0].status_code, 400)
        response, steps = self.upload("Möte.WAV")
        self.assertEqual(response.status_code, 200)

    def test_presentations_and_instructions(self):
        _, steps = self.upload(transcription_format="speakers", speakers_prompt="Ett möte.")
        self.assertTrue(steps["analyze_speakers"])
        self.assertEqual(steps["speaker_prompt"], "Ett möte.")
        _, steps = self.upload(transcription_format="raw", speakers="true")      # an old cached page
        self.assertEqual(steps["transcription_format"], "speakers")
        _, steps = self.upload(transcription_format="marked", speakers="true", speakers_prompt="X")
        self.assertFalse(steps["analyze_speakers"])
        self.assertEqual(steps["speaker_prompt"], "")
        response, _ = self.upload(custom_analysis="true", custom_prompt="   ")
        self.assertEqual(response.status_code, 400)
        _, steps = self.upload(custom_analysis="true", custom_prompt="Lista beslut.",
                               questions="true", questions_prompt="Egna frågor.")
        self.assertEqual((steps["custom_analysis_prompt"], steps["follow_up_prompt"]),
                         ("Lista beslut.", "Egna frågor."))

    def test_size_limit(self):
        with mock.patch.dict(os.environ, {"JBG_MAX_UPLOAD_MB": "1"}):
            response = self.client.post("/upload/", files={"file": ("big.wav", b"x" * (1024 * 1024 + 100))},
                                        data={"api_key": "sk"})
        self.assertEqual(response.status_code, 413)


class RetentionTest(unittest.TestCase):
    def test_old_files_are_removed_new_ones_kept(self):
        now = time.time()
        old = main.UPLOAD_FOLDER / "retention-test-old.mp3"
        new = main.UPLOAD_FOLDER / "retention-test-new.mp3"
        main.UPLOAD_FOLDER.mkdir(exist_ok=True)
        for path in (old, new):
            path.write_bytes(b"x")
        os.utime(old, (now - 25 * 3600, now - 25 * 3600))
        try:
            main.remove_stale_files(24 * 3600)
            self.assertFalse(old.exists())
            self.assertTrue(new.exists())
        finally:
            old.unlink(missing_ok=True)
            new.unlink(missing_ok=True)

    def test_retention_settings(self):
        for name, reader, cases in (
            ("JBG_RETENTION_HOURS", main.read_retention_hours, (("", 24), ("1", 3), ("abc", 24))),
            ("JBG_CACHE_RETENTION_HOURS", main.read_cache_retention_hours, (("", 24), ("0", 1), ("48", 48))),
        ):
            for raw, expected in cases:
                with mock.patch.dict(os.environ, {name: raw}):
                    self.assertEqual(reader(), expected, f"{name}={raw!r}")
