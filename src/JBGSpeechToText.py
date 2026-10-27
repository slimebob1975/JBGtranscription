"""
Speech to text: Whisper and faster-whisper, devices and memory, model
choice by accuracy, in-memory decoding of encrypted audio, and the
transcription cache.

Part of JBGtranscriber, which combines the mixins in src/JBGtranscriber.py.
"""

import hashlib
import os
import psutil
import time
import torch
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from src.JBGLogger import JBGLogger

logger = JBGLogger(level="INFO").logger


# Default markers
CACHE_MODEL_MARKER = "===MODEL==="
CACHE_TRANSCRIPTION_MARKER = "===TRANSCRIPTION==="
CACHE_TIMESTAMPED_MARKER = "===TIMESTAMPED==="
# Transcriptions of unencrypted uploads are kept here so that the same audio
# file uploaded again goes straight to the analysis steps. The files are
# clear text; main.py deletes them after JBG_CACHE_RETENTION_HOURS.
TRANSCRIPTION_CACHE_DIR = Path("cache")
TRANSCRIPTION_ACCURACY_LEVELS = ("mycket_hog", "hog", "medel", "bas", "liten")
DEFAULT_TRANSCRIPTION_ACCURACY = "medel"
TRANSCRIPTION_ACCURACY_MODELS = {
    "mycket_hog": "KBLab/kb-whisper-large",
    "hog": "KBLab/kb-whisper-medium",
    "medel": "KBLab/kb-whisper-small",
    "bas": "KBLab/kb-whisper-base",
    "liten": "KBLab/kb-whisper-tiny",
}
TRANSCRIPTION_ACCURACY_LABELS = {
    "mycket_hog": "mycket hög",
    "hog": "hög",
    "medel": "medel",
    "bas": "bas",
    "liten": "liten",
}
# --- Speaker identification --------------------------------------------------
#
# A long recording is diarized segment by segment. Labelled independently, the
# segments do not agree: "Intervjuobjekt 1" in one need not be the same person
# as in the next, because the model cannot see what it decided earlier. Each
# answer therefore ends with a register of the labels it used, which is parsed
# off, carried forward and given to the next segment.

# A rewriting task must give the whole text back, and normally gives back a
# little more: speaker labels or [FEL?] markers add words. A result that is
# materially shorter than its input means text was dropped, which no amount of
# prompt wording can rule out. Below the warning level it is reported; below
# the failure level the result is not trusted at all.
# Measured: a model reliably echoes back a transcription of around 5,000 tokens
# but gives up on 12,000, returning an excerpt instead. The output limit alone
# therefore allows segments that are too large in practice, so rewriting work
# gets a second, smaller cap. Override with JBG_REWRITE_SEGMENT_TOKENS.
# --- Transcription engine ---------------------------------------------------
#
# faster-whisper runs the same KBLab models through CTranslate2 with int8
# weights. Measured on a ten minute recording, both engines on CPU:
#
#   kb-whisper-small   3.57x realtime -> 7.26x   (2.03x)
#   kb-whisper-large   0.81x realtime -> 1.62x   (1.99x)
#
# The transcriptions differed by 4-6% at word level, and the differences were
# sentence boundaries, commas and dropped filler words rather than content.
# It is therefore the default, with the slower engine kept as a fallback.
# --- Device selection -------------------------------------------------------
#
# The two engines do not agree about what hardware is available, and cannot.
# Transformers asks torch, and the pinned build is torch==2.8.0+cpu, so it
# reports no GPU whatever the machine has. faster-whisper does not use torch at
# all: CTranslate2 has its own CUDA runtime and can use a card that torch
# cannot see. The device is therefore decided per engine rather than once for
# the process.
#
# JBG_DEVICE forces the answer: "cpu", "gpu" or "auto" (the default).
JBG_DEVICE_ENV = "JBG_DEVICE"
ENGINE_FASTER_WHISPER = "faster-whisper"
ENGINE_TRANSFORMERS = "transformers"
DEFAULT_TRANSCRIPTION_ENGINE = ENGINE_FASTER_WHISPER
# int8 weights are roughly a quarter the size of the float32 ones, so a model
# that does not fit under the other engine may well fit under this one.
FASTER_WHISPER_RAM_DIVISOR = 3.0
FASTER_WHISPER_COMPUTE_TYPE_CPU = "int8"
FASTER_WHISPER_COMPUTE_TYPE_GPU = "float16"
# --- Transcription speed, for experiments ----------------------------------
#
# Transformers has two long-form algorithms. Without chunk_length_s it uses the
# sequential one: a sliding window where each window conditions on the previous
# one. That is the more accurate of the two and is what runs today, but it is
# serial by construction and cannot be batched.
#
# With chunk_length_s set, the windows are independent and can be processed
# several at a time with batch_size. That is faster but slightly less accurate
# at the window boundaries, since each window loses the preceding context.
#
# Both are left unset so that behaviour does not change. Set the environment
# variables to measure the trade-off on a known recording:
#   JBG_CHUNK_LENGTH_S=30  JBG_BATCH_SIZE=8
JBG_CHUNK_LENGTH_S_ENV = "JBG_CHUNK_LENGTH_S"
JBG_BATCH_SIZE_ENV     = "JBG_BATCH_SIZE"
# Resampling target rate
RESAMPLING_TARGET_RATE = 16000
# Audio formats accepted for transcription, by file extension. PyAV and ffmpeg
# identify the actual format from the content; the list keeps out video and
# anything else they could open but that does not belong here.
SUPPORTED_AUDIO_EXTENSIONS = (".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg")


class SpeechToTextMixin:
    # Standard Settings
    TRANSCRIBER_MODEL_CANDIDATES = [
    "KBLab/kb-whisper-large",
    "KBLab/kb-whisper-medium",
    "KBLab/kb-whisper-small",
    "KBLab/kb-whisper-base",
    "KBLab/kb-whisper-tiny"
    ]
    TRANSCRIBER_MODEL_RAM_REQUIREMENTS = {
    "KBLab/kb-whisper-large": 6.0,
    "KBLab/kb-whisper-medium": 4.0,
    "KBLab/kb-whisper-small": 2.5,
    "KBLab/kb-whisper-base": 1.5,
    "KBLab/kb-whisper-tiny": 1.0,
    }
    TRANSCRIBER_MODEL_DEFAULT = TRANSCRIBER_MODEL_CANDIDATES[0]
    CACHE_DIR = "kb-whisper-cache"
    transcription_seconds = 0.0
    transcriber_model_used = ""
    transcription_accuracy = DEFAULT_TRANSCRIPTION_ACCURACY
    transcription_engine = DEFAULT_TRANSCRIPTION_ENGINE
    transcription_device = ""
    transcription_from_cache = False

    @staticmethod
    def do_nvidia_check(preferred_device):
        """"Check if we have GPU support or not and set data type accordingly """
        
        # Check if NVIDIA is supported
        if preferred_device == 'gpu':
            if torch.cuda.is_available():
                device = "cuda:0"
                torch_dtype = torch.float16
            else:
                device = "cpu"
                torch_dtype = torch.float32
        else:
            device = "cpu"
            torch_dtype = torch.float32

        # Set datatype and models        
        return device, torch_dtype

    @staticmethod
    def insert_newlines(string, n):
        """
        Insert a newline character into a string as close to every nth character as possible
        without breaking apart whole words.
        
        :param string: The string where newlines will be inserted
        :param n: The interval of characters where newlines should ideally be inserted
        :return: The modified string with newlines inserted
        """
        words = string.split()
        current_length = 0
        result = ""

        for word in words:
            # Check if adding the next word would exceed the desired line length
            if current_length + len(word) > n:
                # If so, add a newline character and reset the current line length
                result += "\n"
                current_length = 0

            # If adding a space would not overflow the line, add one before the word
            if current_length > 0:
                result += " "
                current_length += 1

            # Add the word to the result and increase the current line length
            result += word
            current_length += len(word)

        return result

    @staticmethod
    def find_audio_files(path):
        """Generates a list of supported audio files from a path"""

        return sorted(
            file for file in Path(path).rglob("*")
            if file.is_file() and file.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        )

    def get_transcription_cache_path(self):
        """Cache file for this audio content, accuracy level and engine.

        The key is the content, not the name, so a renamed copy still hits.
        Accuracy and engine are part of it so that a request for higher
        accuracy is never answered with an earlier, less accurate result.
        """
        with open(self.convert_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return TRANSCRIPTION_CACHE_DIR / (
            f"{file_hash}_{self.transcription_accuracy}_{self.transcription_engine}.txt"
        )

    def _read_transcription_cache(self, cache_file):
        """Load a cached transcription; returns False if there is none."""
        if not cache_file.exists():
            return False
        content = cache_file.read_text(encoding="utf-8")
        model_used, device = "", ""
        if content.startswith(CACHE_MODEL_MARKER):
            header, _, content = content.partition(CACHE_TRANSCRIPTION_MARKER)
            lines = header[len(CACHE_MODEL_MARKER):].strip().splitlines()
            model_used = lines[0].strip() if lines else ""
            device = lines[1].strip() if len(lines) > 1 else ""
        parts = content.split(CACHE_TIMESTAMPED_MARKER)
        self.transcription = parts[0].replace(CACHE_TRANSCRIPTION_MARKER, "").strip()
        self.transcription_w_timestamps = parts[1].strip() if len(parts) > 1 else ""
        self.transcriber_model_used = model_used
        self.transcription_device = device
        self.transcription_from_cache = True
        logger.info(f" Using cached transcription: {cache_file.name} (model {model_used or 'okänd'})")
        return True

    def _write_transcription_cache(self, cache_file):
        """Store the transcription, with the model that produced it."""
        TRANSCRIPTION_CACHE_DIR.mkdir(exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(f"{CACHE_MODEL_MARKER}\n")
            f.write(f"{self.transcriber_model_used}\n")
            f.write(f"{self.transcription_device}\n")
            f.write(f"{CACHE_TRANSCRIPTION_MARKER}\n")
            f.write(self.transcription + "\n")
            f.write(f"\n{CACHE_TIMESTAMPED_MARKER}\n")
            f.write(self.transcription_w_timestamps)
        logger.info(f" Transcription and timestamps cached as: {cache_file.name}")

    @staticmethod
    def _device_preference():
        """What the operator asked for: "cpu", "gpu" or "auto"."""
        raw = (os.getenv(JBG_DEVICE_ENV) or "auto").strip().lower()
        if raw in ("cpu", "gpu", "auto"):
            return raw
        logger.warning(f"Ignoring invalid {JBG_DEVICE_ENV} value: {raw!r}. Using 'auto'.")
        return "auto"

    @staticmethod
    def ctranslate2_gpu_count():
        """How many CUDA devices CTranslate2 can see.

        Asked of CTranslate2 rather than torch, because the two are independent:
        a CPU-only torch build says there is no GPU even when there is one.
        """
        try:
            import ctranslate2
            return int(ctranslate2.get_cuda_device_count())
        except Exception as e:
            logger.debug(f"Could not ask CTranslate2 about CUDA devices: {e}")
            return 0

    def _resolve_engine_device(self, engine):
        """Decide the device for one engine, honouring JBG_DEVICE."""
        preference = self._device_preference()

        if preference == "cpu":
            return "cpu"

        if engine == ENGINE_FASTER_WHISPER:
            available = self.ctranslate2_gpu_count() > 0
            source = "CTranslate2"
        else:
            available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
            source = "torch"

        if available:
            return "cuda"

        if preference == "gpu":
            logger.warning(
                f"{JBG_DEVICE_ENV}=gpu men {source} ser inget CUDA-stöd. Använder CPU."
            )
        return "cpu"

    def _faster_whisper_compute_type(self):
        return (FASTER_WHISPER_COMPUTE_TYPE_GPU if str(self.device).startswith("cuda")
                else FASTER_WHISPER_COMPUTE_TYPE_CPU)

    def _transcribe_with_faster_whisper(self, model_id, audio_input):
        """Transcribe through CTranslate2, in the shape _postprocess_result expects.

        The segment generator is lazy: nothing is decoded until it is consumed,
        so the loop below is where the work actually happens.
        """
        from faster_whisper import WhisperModel

        device = self._resolve_engine_device(ENGINE_FASTER_WHISPER)
        compute_type = (FASTER_WHISPER_COMPUTE_TYPE_GPU if device == "cuda"
                        else FASTER_WHISPER_COMPUTE_TYPE_CPU)
        logger.info(
            f"Använder faster-whisper ({compute_type}) på {device} för {model_id}."
        )

        try:
            model = WhisperModel(
                model_id,
                device=device,
                compute_type=compute_type,
                download_root=str(self.CACHE_DIR),
            )
        except Exception as e:
            if device != "cuda":
                raise
            # float16 on GPU needs cuDNN present and matching. Rather than fail
            # the run, fall back to the CPU and say so.
            logger.error(
                f"Kunde inte starta faster-whisper på GPU ({e}). Faller tillbaka på CPU."
            )
            device = "cpu"
            compute_type = FASTER_WHISPER_COMPUTE_TYPE_CPU
            model = WhisperModel(
                model_id,
                device=device,
                compute_type=compute_type,
                download_root=str(self.CACHE_DIR),
            )

        self.transcription_device = f"{device}/{compute_type}"

        segments, info = model.transcribe(
            audio_input,
            language="sv",
            # KBLab recommend this to reduce hallucinations when no prompt is used.
            condition_on_previous_text=False,
        )

        chunks, texts = [], []
        for segment in segments:
            text = (segment.text or "").strip()
            if not text:
                continue
            texts.append(text)
            chunks.append({"timestamp": (segment.start, segment.end), "text": text})

        logger.info(
            f"faster-whisper: språk {info.language} "
            f"({info.language_probability:.2f}), {len(chunks)} segment."
        )
        return {"text": " ".join(texts), "chunks": chunks}

    def _required_ram_for(self, model_id, engine):
        """RAM estimate for a model under a given engine.

        The table is written for the float32 weights the Transformers engine
        loads. CTranslate2 int8 weights are far smaller, so insisting on the
        same figure would refuse models that would in fact run.
        """
        required = self.TRANSCRIBER_MODEL_RAM_REQUIREMENTS.get(model_id)
        if required is None:
            return None
        if engine == ENGINE_FASTER_WHISPER:
            return round(required / FASTER_WHISPER_RAM_DIVISOR, 2)
        return required

    def _transcription_speed_options(self):
        """Optional chunking and batching settings, from the environment.

        Returns the kwargs to pass to the pipeline and a short text for the log,
        so that a run can be matched to the settings it used.
        """
        options = {}
        for env_name, key in ((JBG_CHUNK_LENGTH_S_ENV, "chunk_length_s"),
                              (JBG_BATCH_SIZE_ENV, "batch_size")):
            raw = os.getenv(env_name)
            if not raw:
                continue
            try:
                value = int(raw)
            except ValueError:
                logger.warning(f"Ignoring invalid {env_name} value: {raw!r}")
                continue
            if value > 0:
                options[key] = value

        if options:
            described = ", ".join(f"{k}={v}" for k, v in options.items())
            logger.info(f"Transkribering med {described} (sätt av miljövariabler).")
            return options, described

        return options, "sekventiell, utan batchning"

     # Some internal help functions
    def _model_candidates_for_accuracy(self):
        """Models to try, starting at the accuracy the user asked for.

        The chosen level is the ceiling: everything more accurate is dropped
        from the list, so a deliberately fast run is never silently upgraded.
        Everything below it is kept, so the existing RAM fallback still applies.
        """
        level = self.transcription_accuracy
        if level not in TRANSCRIPTION_ACCURACY_MODELS:
            logger.warning(
                f"Okänd noggrannhet '{level}'. Använder '{DEFAULT_TRANSCRIPTION_ACCURACY}'."
            )
            level = DEFAULT_TRANSCRIPTION_ACCURACY
            self.transcription_accuracy = level

        wanted = TRANSCRIPTION_ACCURACY_MODELS[level]
        candidates = list(self.TRANSCRIBER_MODEL_CANDIDATES)
        if wanted in candidates:
            candidates = candidates[candidates.index(wanted):]
        else:
            logger.warning(f"{wanted} finns inte bland kandidaterna. Använder hela listan.")

        logger.info(
            f"Vald noggrannhet: {TRANSCRIPTION_ACCURACY_LABELS.get(level, level)} "
            f"({wanted}). Kandidater: {', '.join(candidates)}"
        )
        return candidates

    def transcribe_default(self):
        """Put together the model of choice and do the transcription"""
    
        # Check cache
        cache_file = self.get_transcription_cache_path()
        if self._read_transcription_cache(cache_file):
            return
    
        # What model to use
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.transcriber_model_id, torch_dtype=self.torch_dtype, use_safetensors=True, cache_dir=self.CACHE_DIR
        )
        model.to(self.device)
        processor = AutoProcessor.from_pretrained(self.transcriber_model_id)

        # Define pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        generate_kwargs = {"task": "transcribe", "language": "sv"}

        # Do transcription
        result = pipe(str(self.convert_path), 
                chunk_length_s=30,
                generate_kwargs=generate_kwargs, 
                return_timestamps=True)
        
        self.transcription, self.transcription_w_timestamps = self._postprocess_result(result)
        self.transcriber_model_used = self.transcriber_model_id
        self._write_transcription_cache(cache_file)

    @staticmethod
    def _enough_memory(min_gb_required: float = 6.0) -> bool:
        """Return True if there seems to be enough free RAM to load a model.

        Logs both total and available RAM for debugging purposes.
        """
        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024 ** 3)
        available_gb = vm.available / (1024 ** 3)
        logger.debug(
            f"Memory check: required={min_gb_required:.2f} GB, "
            f"available={available_gb:.2f} GB, total={total_gb:.2f} GB"
        )
        return available_gb >= min_gb_required

    def transcribe(self):
        """Put together the model of choice and do the transcription"""

        # The option to use cached transcription is not available during encrypted mode
        if not self.secure_handler:
            if self._read_transcription_cache(self.get_transcription_cache_path()):
                return
        else:
            logger.warning(f" Cannot use cached transcriptions and encryption as the same time")
            if self.audio_stream:
                from faster_whisper import decode_audio

                self.audio_stream.seek(0)
                try:
                    # The decrypted audio is decoded inside this process with
                    # PyAV, straight from memory, into 16 kHz mono float32.
                    # pydub was used before, but it handed the bytes to ffmpeg
                    # and ffprobe through their cache: protocol, which buffers
                    # the input in a temporary file, so the decrypted audio
                    # reached the disk. PyAV also identifies the format from
                    # the content, so every supported format works.
                    self.audio_data = decode_audio(self.audio_stream, sampling_rate=RESAMPLING_TARGET_RATE)
                    self.samplerate = RESAMPLING_TARGET_RATE
                    logger.info(
                        f"Ljudet avkodat i minnet: "
                        f"{self._format_duration(len(self.audio_data) / RESAMPLING_TARGET_RATE)}"
                    )
                except Exception as e:
                    logger.error(f" Failed to load or process audio stream: {e}")
                    raise e
            else:
                raise ValueError("Ingen audio_stream tillgänglig")

        candidates = self._model_candidates_for_accuracy()
        transcription_started = time.time()

        for model_id in candidates:
            try:
                logger.info(f"Trying model: {model_id}")
                required_ram_gb = self._required_ram_for(model_id, self.transcription_engine)

                if required_ram_gb is None:
                    logger.warning(f"No RAM requirement configured for {model_id}, skipping.")
                    continue

                if not self._enough_memory(required_ram_gb):
                    # Särskilt fall: försök ändå med minsta modellen som sista utväg
                    if model_id == "KBLab/kb-whisper-tiny":
                        logger.warning(
                            "Available RAM verkar inte räcka för KBLab/kb-whisper-tiny, "
                            "men vi försöker ändå som sista utväg."
                        )
                    else:
                        logger.warning(
                            f"Skipping model {model_id} -- it is estimated that the available RAM is not enough."
                        )
                        continue

                logger.info(f"Estimated RAM requirement for model {model_id}: {required_ram_gb} GB")


                audio_input = self.audio_data if hasattr(self, "audio_data") else str(self.convert_path)

                if self.transcription_engine == ENGINE_FASTER_WHISPER:
                    speed_description = f"faster-whisper {self._faster_whisper_compute_type()}"
                    result = self._transcribe_with_faster_whisper(model_id, audio_input)
                else:
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        model_id, torch_dtype=self.torch_dtype, use_safetensors=True, cache_dir=self.CACHE_DIR
                    )
                    model.to(self.device)
                    processor = AutoProcessor.from_pretrained(model_id)

                    pipe = pipeline(
                        "automatic-speech-recognition",
                        model=model,
                        tokenizer=processor.tokenizer,
                        feature_extractor=processor.feature_extractor,
                        torch_dtype=self.torch_dtype,
                        device=self.device,
                    )

                    generate_kwargs = {"task": "transcribe", "language": "sv"}
                    self.transcription_device = str(self.device)

                    speed_options, speed_description = self._transcription_speed_options()
                    result = pipe(
                        audio_input,
                        generate_kwargs=generate_kwargs,
                        return_timestamps=True,
                        **speed_options
                    )

                self.transcription, self.transcription_w_timestamps = self._postprocess_result(result)
                self.transcription_seconds = time.time() - transcription_started
                word_count = f"{len((self.transcription or '').split()):,}".replace(",", " ")
                logger.info(
                    f" Transcription successful with model: {model_id} "
                    f"({word_count} ord, "
                    f"{self._format_duration(self.transcription_seconds)}, "
                    f"{speed_description}, {self.transcription_device})"
                )
                self.transcriber_model_used = model_id
                if model_id != TRANSCRIPTION_ACCURACY_MODELS.get(self.transcription_accuracy):
                    logger.warning(
                        f"Vald noggrannhet '{self.transcription_accuracy}' motsvarar "
                        f"{TRANSCRIPTION_ACCURACY_MODELS.get(self.transcription_accuracy)}, "
                        f"men {model_id} användes i stället."
                    )

                
                if not self.secure_handler:
                    self._write_transcription_cache(self.get_transcription_cache_path())
                return

            except Exception as e:
                logger.error(f"Model {model_id} failed: {e}")
                continue

        raise RuntimeError("All model options failed. Try reducing file size or increasing memory.")

    def _postprocess_result(self, result):
        """Post process result of call to transcription model"""
        
        transcription = result["text"]
        if self.insert_linebreaks:
            transcription = self.insert_newlines(transcription, 80)
        
        transcription_w_timestamps = ""
        for chunk in result["chunks"]:
            transcription_w_timestamps += str(chunk["timestamp"]) + ": " + str(chunk["text"]) + "\n"
        if self.insert_linebreaks:
            transcription_w_timestamps = self.insert_newlines(transcription_w_timestamps, 80)
            
        return transcription, transcription_w_timestamps
