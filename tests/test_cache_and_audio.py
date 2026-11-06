import base64
import io
import os
import unittest
from pathlib import Path

import tests  # noqa: F401
import src.JBGtranscriber as J
from src.JBGSecureFileHandler import SecureFileHandler
from tests.support import browser_encrypt, in_temp_dir, make_transcriber, sine_wav_bytes


def with_fake_model(t, calls):
    t._model_candidates_for_accuracy = lambda: [J.TRANSCRIPTION_ACCURACY_MODELS[t.transcription_accuracy]]
    t._required_ram_for = lambda *a: 1
    t._enough_memory = lambda *a: True

    def fake_whisper(model_id, audio):
        calls.append(model_id)
        t.transcription_device = "cpu/int8"
        return "RESULT"
    t._transcribe_with_faster_whisper = fake_whisper
    t._postprocess_result = lambda r: ("Hej och välkommen.", "[00:00] Hej och välkommen.")
    return t


class TranscriptionCacheTest(unittest.TestCase):
    def test_write_hit_and_key(self):
        with in_temp_dir():
            Path("a.mp3").write_bytes(b"audio")
            calls = []
            first = with_fake_model(make_transcriber("", convert_path=Path("a.mp3")), calls)
            first.transcribe()
            files = list(J.TRANSCRIPTION_CACHE_DIR.glob("*.txt"))
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].name.endswith("_medel_faster-whisper.txt"))

            Path("renamed.mp3").write_bytes(b"audio")
            hit = with_fake_model(make_transcriber("", convert_path=Path("renamed.mp3")), calls)
            hit.transcribe()
            self.assertEqual(len(calls), 1)
            self.assertTrue(hit.transcription_from_cache)
            self.assertEqual(hit.transcriber_model_used, "KBLab/kb-whisper-small")
            self.assertIn("återanvändes", hit._transcription_provenance())

            higher = with_fake_model(make_transcriber("", convert_path=Path("a.mp3"),
                                                      transcription_accuracy="mycket_hog"), calls)
            higher.transcribe()
            self.assertEqual(len(calls), 2)

    def test_encrypted_runs_bypass_the_cache(self):
        with in_temp_dir():
            Path("a.mp3").write_bytes(b"audio")
            key = base64.b64encode(os.urandom(32)).decode()
            calls = []
            t = with_fake_model(make_transcriber("", convert_path=Path("a.mp3"),
                                                 secure_handler=SecureFileHandler(key),
                                                 audio_stream=io.BytesIO(b"x")), calls)
            try:
                import faster_whisper  # noqa: F401
            except ImportError:
                self.skipTest("faster-whisper not installed")
            # The stream is not real audio, so decoding fails before any cache use.
            with self.assertRaises(Exception):
                t.transcribe()
            self.assertFalse(J.TRANSCRIPTION_CACHE_DIR.exists() and any(J.TRANSCRIPTION_CACHE_DIR.iterdir()))


class EncryptedDecodingTest(unittest.TestCase):
    def test_wav_decodes_in_memory(self):
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            self.skipTest("faster-whisper not installed")
        with in_temp_dir():
            key = os.urandom(32)
            handler = SecureFileHandler(base64.b64encode(key).decode())
            path = Path("x.wav.encrypted")
            path.write_bytes(browser_encrypt(sine_wav_bytes(2.0), key))
            t = make_transcriber("", secure_handler=handler,
                                 audio_stream=handler.decrypt_file_to_memory(str(path)))
            t._model_candidates_for_accuracy = lambda: []
            with self.assertRaises(RuntimeError):      # no model: stops after decoding
                t.transcribe()
            self.assertAlmostEqual(len(t.audio_data) / 16000, 2.0, places=1)
            self.assertEqual(t.samplerate, 16000)

    def test_cli_finds_supported_files(self):
        with in_temp_dir():
            for name in ("a.mp3", "b.M4A", "c.wav", "d.txt", "e.mp4"):
                Path(name).write_bytes(b"x")
            found = [p.name for p in J.JBGtranscriber.find_audio_files(Path("."))]
            self.assertEqual(found, ["a.mp3", "b.M4A", "c.wav"])
