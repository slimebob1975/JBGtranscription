"""Shared helpers: a fake OpenAI client, a transcriber without models, test audio."""
import io
import math
import os
import struct
import tempfile
import wave
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from openai.types.chat import ChatCompletion

import src.JBGtranscriber as J


def completion(content, prompt_tokens=100, completion_tokens=20, cached_tokens=0, model="gpt-5.2"):
    """A response in the OpenAI SDK's own type."""
    return ChatCompletion.model_validate({
        "id": "test", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens,
                  "prompt_tokens_details": {"cached_tokens": cached_tokens}},
    })


class FakeOpenAI:
    """Records every request; answers through a function of (system, user)."""

    def __init__(self, answer=lambda system, user: "Resultat."):
        self.requests = []
        self.answer = answer
        fake = self

        class Client:
            def __init__(self, **kwargs):
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                system = kwargs["messages"][0]["content"]
                user = kwargs["messages"][1]["content"]
                fake.requests.append({"system": system, "user": user, "model": kwargs.get("model")})
                result = fake.answer(system, user)
                return result if isinstance(result, ChatCompletion) else completion(result)

        self.client_class = Client

    @contextmanager
    def installed(self):
        with mock.patch("openai.OpenAI", self.client_class):
            yield self


def make_transcriber(transcription="Välkomna till mötet. Tack.", **attrs):
    """A transcriber with no models loaded and segmentation reduced to one piece."""
    t = J.JBGtranscriber.__new__(J.JBGtranscriber)
    defaults = dict(
        api_key="sk-test", openai_model="gpt-5.2", model_calls=[], current_task="",
        prompt_policy=J.JBGtranscriber.load_policy_file(),
        transcription=transcription, transcription_w_timestamps="[00:00] " + transcription,
        transcriber_model_used="KBLab/kb-whisper-small", transcription_device="cpu/int8",
        transcription_accuracy="medel", transcription_engine=J.ENGINE_FASTER_WHISPER,
        transcription_from_cache=False, transcription_seconds=0.0, device="cpu",
        secure_handler=None, audio_stream=None, transcription_format="raw",
        summary="", follow_up_questions="", analyze_speakers="", marked_text="",
        custom_analysis="", analyses_on_marked_text=[], speaker_register={},
    )
    defaults.update(attrs)
    for key, value in defaults.items():
        setattr(t, key, value)
    t._get_encoder_for_segmentation = lambda: None
    t._split_into_segments = lambda text, enc, max_tokens=None, **k: [text]
    t._available_input_tokens = lambda *a, **k: 100000
    t._rewrite_segment_budget = lambda *a, **k: 100000
    t._resolve_rewrite_segment_tokens = lambda **k: 100000
    return t


def sine_wav_bytes(seconds=2.0, rate=44100):
    """A short stereo sine tone as wav, made without ffmpeg."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(int(seconds * rate)):
            v = int(8000 * math.sin(2 * math.pi * 440 * i / rate))
            frames += struct.pack("<hh", v, v)
        w.writeframes(bytes(frames))
    return buffer.getvalue()


def browser_encrypt(data, key):
    """The page's layout: a 12-byte IV followed by AES-GCM ciphertext and tag."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    iv = os.urandom(12)
    return iv + AESGCM(key).encrypt(iv, data, None)


@contextmanager
def in_temp_dir():
    """Run in an empty working directory (the cache folder is relative)."""
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            yield Path(d)
        finally:
            os.chdir(old)
