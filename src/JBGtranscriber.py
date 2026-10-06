#!/usr/bin/env python3
try:
    import sys
    import hashlib
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import openai
    from pathlib import Path
    import json
    import tiktoken
    import time
    import psutil
    from pydub import AudioSegment
    import io
    import numpy as np
    import resampy
    import os
    import re
    from docx import Document
    from docx.shared import Pt
    from docx.enum.text import WD_COLOR_INDEX
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
except ModuleNotFoundError as ex:
    sys.exit("You probably need to install some missing modules:" + str(ex))
from src.JBGLogger import JBGLogger
    
# Default markers
CACHE_TRANSCRIPTION_MARKER = "===TRANSCRIPTION==="
CACHE_TIMESTAMPED_MARKER = "===TIMESTAMPED==="
MODEL_GPT_5_MARKER = "gpt-5"

# --- Segmentation of long transcriptions -------------------------------------
#
# The old implementation used a single hard-coded budget of 3500 tokens, which
# was sized for 4k-context models. Current models have far larger context
# windows, so a one-hour interview that used to be cut into 5-8 pieces normally
# fits in a single call today. Fewer segments means better summaries, so the
# budget is now resolved per model, can be overridden by configuration, and is
# automatically lowered if the API reports that the context window was exceeded.

# Used when the model is unknown. Deliberately conservative relative to a
# 128k-token context window, to leave room for instructions and the answer.
DEFAULT_MAX_INPUT_TOKENS = 100000

# Never segment into pieces smaller than this, however far the budget is lowered.
MIN_MAX_INPUT_TOKENS = 2000

# Reserved headroom for the model's own answer plus the chat scaffolding.
RESERVED_RESPONSE_TOKENS = 12000

# Longest matching prefix wins. Extend as models are added to the GUI.
MODEL_INPUT_TOKEN_BUDGETS = {
    "gpt-3.5": 12000,
    "gpt-4": 6000,
    "gpt-4-turbo": 100000,
    "gpt-4o": 100000,
    "gpt-4.1": 100000,
    "gpt-5": 100000,
}

# How many trailing sentences are repeated at the start of the next segment so
# that a thought split across a boundary is not lost.
SEGMENT_OVERLAP_SENTENCES = 2

# Pause between segment calls to stay clear of rate limits.
SEGMENT_PAUSE_SECONDS = 5

# How many times the budget may be halved in response to a context-length error.
MAX_BUDGET_DOWNSHIFTS = 4

# Kept for backwards compatibility with any external caller.
MAX_INPUT_TOKENS = DEFAULT_MAX_INPUT_TOKENS

# --- Transcription accuracy --------------------------------------------------
#
# The Whisper model was always the largest that fits in the available RAM. The
# user can now set a ceiling instead: the chosen level is used exactly, or the
# first level below it that fits. A choice is never upgraded past what was
# asked for, so a deliberately fast run stays fast.

STATISTICS_TABLE_HEADER = ["Moment", "Modell", "Anrop", "Tokens in", "Tokens ut", "Tid"]

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

DEFAULT_REWRITE_SEGMENT_TOKENS = 6000

# Speaker identification needs a smaller segment still. Measured on one
# recording: processed in a single call it reported 7, 6 and 6 speakers across
# three runs of the same audio; split into seven segments of about 2,100 tokens
# it reported the same five speakers twice. Carrying the register between
# segments appears to steady the result, so the segments are kept small enough
# that it is actually used. Override with JBG_DIARIZATION_SEGMENT_TOKENS.
DEFAULT_DIARIZATION_SEGMENT_TOKENS = 2500

# If the answer still comes back short, the segment size is cut and the step
# tried again rather than discarded outright.
MAX_REWRITE_RETRIES = 2
REWRITE_RETRY_DIVISOR = 3

REWRITE_MIN_WORD_RATIO_WARN = 0.90
REWRITE_MIN_WORD_RATIO_FAIL = 0.70

SPEAKER_REGISTER_SECTION_TITLE = "Identifierade talare:"

SPEAKER_REGISTER_HEADING = "### TALARREGISTER"

# The model does not always reproduce the heading exactly: it may drop the
# hashes, bold it, or change the case. A block that is not recognised would be
# left in the dialogue, so the match is deliberately loose.
_SPEAKER_REGISTER_LINE_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?[*_]*[ \t]*talarregister[ \t]*[*_:\-]*[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

SPEAKER_REGISTER_SUFFIX = (
    "AVSLUTA SVARET MED ETT TALARREGISTER:\n"
    "Skriv sist i svaret raden " + SPEAKER_REGISTER_HEADING + " och därefter en "
    "rad per talare du använt, på formen:\n"
    "Beteckning: kort kännetecken\n\n"
    "Kännetecknet ska vara sakligt och hjälpa till att känna igen samma person "
    "senare i samtalet, till exempel vilken roll personen har eller vilka frågor "
    "personen svarar på. Bedöm inte personen. Registret räknas inte som en del "
    "av dialogen.\n\n"
    "Skriv ingenting om talarna i själva dialogen: inga beskrivningar, "
    "sammanfattningar eller kommentarer om vem som är vem, vare sig före, under "
    "eller efter replikerna. Dialogen ska bara innehålla repliker med sin "
    "beteckning. Allt som beskriver talarna hör hemma i registret."
)

SPEAKER_REGISTER_PREAMBLE = (
    "TALARE SOM REDAN IDENTIFIERATS TIDIGARE I SAMTALET:\n"
    "{register}\n\n"
    "Använd samma beteckningar för dessa personer. Inför en ny beteckning bara "
    "för en röst som inte finns i listan."
)

SPEAKER_CONTEXT_PREAMBLE = (
    "SLUTET AV FÖREGÅENDE DEL, ENBART SOM SAMMANHANG:\n"
    "{context}\n\n"
    "Återge inte texten ovan i ditt svar. Den är redan behandlad. Börja ditt "
    "svar vid den nya texten."
)

# Sentences of the preceding segment passed as context. They are given as
# already-handled context rather than as overlap: the model is told not to
# repeat them, so a reply split across a boundary can be attributed correctly
# without the text appearing twice in the result.
SPEAKER_CONTEXT_SENTENCES = 3

# How the transcribed text is presented in the result document. Exactly one
# form is included; the three are mutually exclusive.
TRANSCRIPTION_FORMATS = ("raw", "timestamps", "marked")
DEFAULT_TRANSCRIPTION_FORMAT = "raw"

TRANSCRIPTION_FORMAT_HEADINGS = {
    "raw": "Transkribering",
    "timestamps": "Transkribering med tidsstämplar",
    "marked": "Transkribering med markerade misstänkta fel",
}

# Upper bound on a user-supplied summary instruction.
MAX_SUMMARY_PROMPT_CHARS = 20000

# --- Output token budget -----------------------------------------------------
#
# Two kinds of work are sent to the model, and they have opposite needs.
#
#   Condensing (summary, follow-up questions): the answer is far smaller than
#   the input, so a large input segment is fine.
#
#   Rewriting (suspicious phrases, speaker diarization): the model has to give
#   the whole text back, marked up. The answer is therefore as large as the
#   input, and the input segment must be small enough that the answer fits
#   inside the model's output limit. Otherwise the reply is cut off mid-text
#   and part of the transcription is silently lost.

DEFAULT_MAX_OUTPUT_TOKENS = 8000

# Longest matching prefix wins, as for the input budgets.
MODEL_OUTPUT_TOKEN_LIMITS = {
    "gpt-3.5": 4000,
    "gpt-4": 4000,
    "gpt-4o": 16000,
    "gpt-4.1": 32000,
    "gpt-5": 32000,
}

# A rewritten segment is longer than its input: speaker labels, [FEL?] markers
# and suggested corrections all add tokens. Leave room for that.
REWRITE_OUTPUT_HEADROOM = 1.4

# Which parameter name a model accepts for the output cap. Newer models reject
# max_tokens in favour of max_completion_tokens, and the only reliable way to
# find out is to try. The answer is cached per model so it is tried once.
_OUTPUT_TOKEN_PARAM_CACHE = {}

_UNSUPPORTED_PARAM_MARKERS = (
    "unsupported parameter",
    "unsupported_parameter",
    "unrecognized request argument",
    "is not supported with this model",
    "unknown parameter",
    "extra fields not permitted",
)

# Option ids used before the summary options were made configurable. Browsers
# still hold these in localStorage, and older clients still post them, so they
# are mapped onto their current equivalents rather than rejected.
LEGACY_SUMMARY_OPTION_IDS = {
    "short": "enkel",
    "extensive": "utforlig",
}

# Used only if the prompt policy cannot be read or contains no usable options.
FALLBACK_SUMMARY_OPTION_ID = "standard"
FALLBACK_SUMMARY_PROMPT = (
    "Sammanfatta följande transkribering på ett tydligt och sakligt sätt. "
    "Hitta inte på information. Om något är oklart, skriv att det är oklart."
)

_SPEAKER_LINE_RE = re.compile(r"^([^:\n]{2,40}?\s*\d*)\s*:\s+\S")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?:…])\s+")
_PARAGRAPH_BOUNDARY_RE = re.compile(r"\n\s*\n")

_CONTEXT_ERROR_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context length",
    "too many tokens",
    "reduce the length",
    "string too long",
)

# Appended to the user's own instruction when a transcription has to be
# processed in several passes.
MAP_STAGE_SUFFIX = (
    "ARBETSSÄTT FÖR DETTA STEG:\n"
    "Transkriberingen är för lång för att behandlas i ett svep och du får just nu "
    "bara en del av den. Följ instruktionerna ovan, men tillämpa dem enbart på det "
    "avsnitt du fått. Var utförlig och behåll detaljer, exempel och citat – detta är "
    "ett underlag som senare ska vävas ihop med underlag från övriga delar. "
    "Skriv ingen inledning och inga slutsatser om helheten."
)

REDUCE_STAGE_SUFFIX = (
    "ARBETSSÄTT FÖR DETTA STEG:\n"
    "Nedan följer flera underlag från samma transkribering, i kronologisk ordning. "
    "Väv ihop dem till en enda sammanhängande text som följer instruktionerna ovan. "
    "Slå ihop teman som återkommer i flera underlag i stället för att upprepa dem, "
    "men behåll detaljer, exempel och citat. Texten ska läsas som om den skrivits "
    "utifrån hela transkriberingen på en gång: nämn inte att materialet varit uppdelat "
    "och hänvisa inte till 'del 1', 'del 2' och så vidare."
)

# Temperature settings
DEFAULT_TEMPERATURE = 0.7
GPT_5_TEMPERATURE = 1.0

# Extra model options
EXTRA_MODEL_OPTIONS = {
    "gpt-5.1": {
        "reasoning_effort": "none",
    },
}

# Resampling target rate
RESAMPLING_TARGET_RATE = 16000

logger = JBGLogger(level="DEBUG").logger


class ApproximateEncoder:
    """Minimal stand-in for a tiktoken encoding.

    Used only when tiktoken cannot load its BPE data, for example when outbound
    network access is blocked. It splits on whitespace boundaries so that
    encode/decode round-trips preserve the text exactly, and deliberately
    over-estimates token counts so that segments stay inside the real budget.

    Swedish text tokenizes less efficiently than English, so roughly three
    tokens per whitespace-separated word is a safe upper bound.
    """

    APPROX_TOKENS_PER_WORD = 3

    def encode(self, text):
        if not text:
            return []
        # One "token" per unit of the estimate, carrying the text in the first.
        words = text.split()
        return [0] * max(1, len(words) * self.APPROX_TOKENS_PER_WORD)

    def decode(self, tokens):
        # Only used by the hard-split path for text without sentence
        # boundaries, which cannot be reconstructed from this estimate.
        raise NotImplementedError(
            "ApproximateEncoder cannot decode; token-level splitting is unavailable."
        )


class JBGtranscriber():
    
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
        
    def __init__(self, 
                 convert_path,
                 export_path = Path("."),
                 device = "cpu",
                 api_key = None,
                 openai_model = "gpt-4o",
                 secure_handler = None,
                 transcriber_model_id=TRANSCRIBER_MODEL_DEFAULT,
                 insert_linebreaks=False
                 ):
        self.convert_path = Path(convert_path)
        self.export_path = Path(export_path)
        self.api_key = api_key
        self.openai_model = openai_model
        self.secure_handler = secure_handler
        self.audio_stream = None    # Used with encryption mode on
        self.transcriber_model_id = transcriber_model_id
        self.insert_linebreaks = insert_linebreaks
        self.device, self.torch_dtype = JBGtranscriber.do_nvidia_check(device)
        
        self.transcription = ""
        self.transcription_w_timestamps = ""
        self.transcription_format = DEFAULT_TRANSCRIPTION_FORMAT
        self.speaker_register = {}
        self.transcription_accuracy = DEFAULT_TRANSCRIPTION_ACCURACY
        # Statistics only: counts and timings, never any text.
        self.model_calls = []
        self.transcription_seconds = 0.0
        self.current_task = ""
        # Filled in once a model has actually loaded and transcribed.
        self.transcriber_model_used = ""
        self.prompt_policy = self.load_prompt_policy()
        self.summary = ""
        self.marked_text = ""
        self.follow_up_questions = ""
        self.analyze_speakers = ""

    # Resolved from the package location rather than the current working
    # directory, so the policy is found regardless of how the app is started.
    POLICY_PATH = Path(__file__).resolve().parent.parent / "policy" / "prompt_policy.json"

    @staticmethod
    def load_policy_file():
        """Load the prompt policy from disk. Returns an empty dict on failure."""
        try:
            with open(JBGtranscriber.POLICY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f" Could not load prompt policy: {e}")
            return {}

    @staticmethod
    def policy_text(policy, key, default=""):
        """Return a policy entry as a single string.

        Entries are written either as a plain string or as a list of lines.
        Joining a string with "\\n".join() would insert a newline between every
        character, so the type has to be checked before joining.
        """
        value = policy.get(key, default)
        if isinstance(value, (list, tuple)):
            return "\n".join(str(line) for line in value)
        return str(value)

    @classmethod
    def summary_options(cls, policy=None):
        """Return the selectable summary options, in display order.

        Each option is a dict with id, label, description, default and prompt.
        The list is the single source of truth for both the GUI dropdown and
        the server-side validation of the chosen option.
        """
        policy = policy if policy is not None else cls.load_policy_file()
        raw = policy.get("summary_options") or []

        options = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            option_id = str(entry.get("id", "")).strip()
            prompt = cls.policy_text(entry, "prompt")
            if not option_id or not prompt.strip():
                continue
            options.append({
                "id": option_id,
                "label": str(entry.get("label") or option_id),
                "description": str(entry.get("description") or ""),
                "default": bool(entry.get("default")),
                "prompt": prompt,
            })

        if not options:
            logger.error(
                "No usable summary options in the prompt policy. "
                "Falling back to a single built-in instruction."
            )
            options = [{
                "id": FALLBACK_SUMMARY_OPTION_ID,
                "label": "Standard",
                "description": "",
                "default": True,
                "prompt": FALLBACK_SUMMARY_PROMPT,
            }]

        if not any(o["default"] for o in options):
            logger.warning(
                "No summary option is marked as default; using the first one."
            )
            options[0]["default"] = True

        return options

    @classmethod
    def resolve_summary_option_id(cls, option_id, options=None):
        """Map a requested option id onto an existing one.

        Ids that no longer exist fall back to the default option rather than
        failing, so a stale value stored in a browser does not break the run.
        """
        options = options if options is not None else cls.summary_options()
        known = {o["id"] for o in options}

        candidate = (option_id or "").strip()
        candidate = LEGACY_SUMMARY_OPTION_IDS.get(candidate, candidate)

        if candidate in known:
            return candidate

        default_id = next(o["id"] for o in options if o["default"])
        if candidate:
            logger.warning(
                f"Unknown summary option '{option_id}'; using default '{default_id}'."
            )
        return default_id

    # Set by call_openai: whether the last answer hit the length limit.
    last_answer_truncated = False

    # Class-level defaults so the document can be written even by an instance
    # that was not built through __init__.
    model_calls = []
    transcription_seconds = 0.0
    current_task = ""
    transcriber_model_used = ""
    transcription_accuracy = DEFAULT_TRANSCRIPTION_ACCURACY
    speaker_register = {}

    def load_prompt_policy(self):
        return JBGtranscriber.load_policy_file()

    
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
    def find_mp3_files(path):
        """Generates a list of mp3 files from a path"""
        
        return [file for file in Path(path).rglob("*.mp3")]
    
    @staticmethod
    def get_permitted_temperature(gpt_model, temperature):
        
        try:
            index = gpt_model.index(MODEL_GPT_5_MARKER)
            return GPT_5_TEMPERATURE
        except ValueError as ex:
            return temperature
        
    @staticmethod
    def get_model_specific_extra_arguments(gpt_model):
        return EXTRA_MODEL_OPTIONS.get(gpt_model, {})
    
    def _get_encoder_for_segmentation(self):
        try:
            return tiktoken.encoding_for_model(self.openai_model)
        except Exception:
            # Fallback for new / unknown models (e.g. gpt-5.*)
            logger.warning(
                f"tiktoken.encoding_for_model('{self.openai_model}') failed, "
                "falling back to o200k_base encoding."
            )

        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception as e:
            # tiktoken downloads its BPE files on first use. In a locked-down
            # network this fails, and without a fallback the whole summary step
            # would fail with it. An approximate encoder keeps segmentation
            # working; it only needs to be good enough to decide where to split.
            logger.warning(
                f"Could not load any tiktoken encoding ({e}). "
                "Falling back to an approximate character-based token estimate."
            )
            return ApproximateEncoder()

    def get_transcription_cache_path(self):
        """Generate a cache filename based on the audio file's full path (hashed)."""
        with open(self.convert_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        cache_file = Path("cache") / f"{file_hash}.txt"
        return cache_file
    
    @staticmethod
    def _is_unsupported_parameter_error(error):
        """True if the API rejected a parameter rather than the request itself."""
        message = str(error).lower()
        return any(marker in message for marker in _UNSUPPORTED_PARAM_MARKERS)

    def _create_completion(self, client, messages, max_output_tokens):
        """Make the call, capping the answer length where the model allows it.

        Models disagree about the parameter name: newer ones reject max_tokens
        and want max_completion_tokens. The working name is discovered once per
        model and cached; if neither is accepted the call is made without a cap
        rather than failing.
        """
        base = dict(
            model=self.openai_model,
            messages=messages,
            temperature=JBGtranscriber.get_permitted_temperature(self.openai_model, DEFAULT_TEMPERATURE),
            **JBGtranscriber.get_model_specific_extra_arguments(self.openai_model),
        )

        if not max_output_tokens:
            return client.chat.completions.create(**base)

        cached = _OUTPUT_TOKEN_PARAM_CACHE.get(self.openai_model, "unknown")
        candidates = [cached] if cached != "unknown" else ["max_completion_tokens", "max_tokens"]

        for param in candidates:
            if param is None:
                break
            try:
                completion = client.chat.completions.create(**base, **{param: max_output_tokens})
                _OUTPUT_TOKEN_PARAM_CACHE[self.openai_model] = param
                return completion
            except Exception as e:
                if JBGtranscriber._is_unsupported_parameter_error(e):
                    logger.info(f"Modellen {self.openai_model} accepterar inte '{param}'.")
                    continue
                raise

        _OUTPUT_TOKEN_PARAM_CACHE[self.openai_model] = None
        logger.warning(
            f"Modellen {self.openai_model} accepterar ingen gräns för svarslängd. "
            "Anropet görs utan begränsning."
        )
        return client.chat.completions.create(**base)

    def call_openai(self, instructions, input_message, max_output_tokens=None):
        """Anropar OpenAI:s GPT-modell med en given prompt."""
        # TODO: Consider adding reasoning_effort='none' for gpt-5.1 to get GPT-5.1-level intelligence with ultra-low latency.

        client = openai.OpenAI(api_key=self.api_key)
        started = time.time()

        completion = self._create_completion(
            client,
            [
                {"role": "system", "content": instructions},
                {"role": "user", "content": input_message},
            ],
            max_output_tokens,
        )

        usage = getattr(completion, "usage", None)
        self._record_model_call(
            task=self.current_task or "OpenAI-anrop",
            model=self.openai_model,
            sent=getattr(usage, "prompt_tokens", None),
            received=getattr(usage, "completion_tokens", None),
            seconds=time.time() - started,
        )

        choice = completion.choices[0]
        self.last_answer_truncated = getattr(choice, "finish_reason", None) == "length"
        if self.last_answer_truncated:
            logger.warning(
                "Modellens svar nådde längdgränsen och kan vara avklippt. "
                "Sänk segmentstorleken eller höj JBG_MAX_OUTPUT_TOKENS."
            )

        return choice.message

    def resolve_summary_instructions(self, style=None, custom_prompt=None):
        """Decide which instruction text to use for the summary.

        A non-empty custom prompt from the user always wins. Otherwise the
        instruction belonging to the chosen option is taken from the prompt
        policy, falling back to the default option.
        """
        candidate = (custom_prompt or "").strip()
        if candidate:
            if len(candidate) > MAX_SUMMARY_PROMPT_CHARS:
                logger.warning(
                    f"Summary instruction truncated from {len(candidate)} to "
                    f"{MAX_SUMMARY_PROMPT_CHARS} characters."
                )
                candidate = candidate[:MAX_SUMMARY_PROMPT_CHARS]
            logger.info(f"Using user-supplied summary instruction ({len(candidate)} characters).")
            return candidate

        options = JBGtranscriber.summary_options(self.prompt_policy)
        option_id = JBGtranscriber.resolve_summary_option_id(style, options)
        option = next(o for o in options if o["id"] == option_id)
        logger.info(f"Using default summary instruction for option '{option_id}'.")
        return option["prompt"]

    def generate_summary(self, style=None, custom_prompt=None):
        """Generate a summary using either a default or a user-edited instruction.

        The transcription is sent in one single call whenever it fits inside the
        model's input budget, since an undivided text gives the best summary.
        Segmentation is a fallback for genuinely long recordings only.
        """
        self.current_task = "Sammanfattning"
        instructions = self.resolve_summary_instructions(style=style, custom_prompt=custom_prompt)

        if not (self.transcription or "").strip():
            logger.warning("No transcription available to summarize.")
            self.summary = "Sammanfattning var inte tillgänglig"
            return

        try:
            self.summary = self._summarize(instructions)
        except Exception as e:
            logger.error(f" Summary generation failed: {e}")
            self.summary = "Sammanfattning var inte tillgänglig"

    def _summarize(self, instructions):
        """Run the summary, lowering the token budget if the API rejects the size."""

        enc = self._get_encoder_for_segmentation()
        budget = self._resolve_input_token_budget()
        downshifts = 0

        while True:
            available = self._available_input_tokens(instructions, enc, budget)
            segments = self._split_into_segments(self.transcription, enc, max_tokens=available)

            try:
                if len(segments) == 1:
                    logger.info(
                        "Transkriberingen ryms i ett anrop - sammanfattar hela texten på en gång."
                    )
                    return self.call_openai(
                        instructions=instructions,
                        input_message=self.transcription,
                        max_output_tokens=self._resolve_output_token_limit(),
                    ).content.strip()

                logger.info(
                    f"Transkriberingen delas i {len(segments)} segment "
                    f"(budget {available} tokens per segment)."
                )
                return self._summarize_in_segments(instructions, segments, enc, available)

            except Exception as e:
                if downshifts < MAX_BUDGET_DOWNSHIFTS and JBGtranscriber._is_context_length_error(e):
                    downshifts += 1
                    budget = max(MIN_MAX_INPUT_TOKENS, budget // 2)
                    logger.warning(
                        f"Modellen rapporterade att kontextfönstret överskreds. "
                        f"Sänker budgeten till {budget} tokens och försöker igen "
                        f"(försök {downshifts}/{MAX_BUDGET_DOWNSHIFTS})."
                    )
                    continue
                raise

    def _summarize_in_segments(self, instructions, segments, enc, available_tokens):
        """Summarize each segment, then merge the results into one coherent text."""

        map_instructions = instructions + "\n\n" + MAP_STAGE_SUFFIX
        partials = []

        for i, segment in enumerate(segments):
            if i > 0:
                time.sleep(SEGMENT_PAUSE_SECONDS)
            logger.info(f" Summarizing segment {i+1}/{len(segments)}...")
            context_note = (
                f"(Detta är del {i+1} av {len(segments)} av transkriberingen.)"
            )
            result = self.call_openai(
                instructions=map_instructions,
                input_message=context_note + "\n\n" + segment,
                max_output_tokens=self._resolve_output_token_limit(),
            )
            partials.append(result.content.strip())

        return self._reduce_partial_summaries(instructions, partials, enc, available_tokens)

    def _reduce_partial_summaries(self, instructions, partials, enc, available_tokens):
        """Merge partial summaries into one text, folding in several passes if needed."""

        reduce_instructions = instructions + "\n\n" + REDUCE_STAGE_SUFFIX
        round_number = 0

        while len(partials) > 1:
            round_number += 1
            groups = self._group_partials_to_fit(partials, enc, available_tokens)

            # Safety net: if a single partial cannot be grouped with anything
            # else, further folding will not converge. Concatenate instead of
            # looping forever.
            if len(groups) >= len(partials):
                logger.warning(
                    "Delunderlagen kan inte slås ihop ytterligare inom budgeten. "
                    "Sammanfogar dem direkt."
                )
                return "\n\n".join(partials)

            logger.info(
                f" Reduce-steg {round_number}: slår ihop {len(partials)} underlag "
                f"till {len(groups)}."
            )

            merged = []
            for i, group in enumerate(groups):
                if i > 0 or round_number > 1:
                    time.sleep(SEGMENT_PAUSE_SECONDS)
                body = "\n\n".join(
                    f"--- Underlag {j+1} av {len(group)} ---\n{part}"
                    for j, part in enumerate(group)
                )
                merged.append(
                    self.call_openai(
                        instructions=reduce_instructions,
                        input_message=body,
                        max_output_tokens=self._resolve_output_token_limit(),
                    ).content.strip()
                )
            partials = merged

        return partials[0] if partials else ""

    def _group_partials_to_fit(self, partials, enc, available_tokens):
        """Pack partial summaries into groups that each fit the input budget."""

        groups = []
        current = []
        current_tokens = 0

        for part in partials:
            part_tokens = len(self._tokenize(part, enc))
            if current and current_tokens + part_tokens > available_tokens:
                groups.append(current)
                current, current_tokens = [], 0
            current.append(part)
            current_tokens += part_tokens

        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _is_context_length_error(error):
        """Detect an API error caused by exceeding the model's context window."""
        message = str(error).lower()
        return any(marker in message for marker in _CONTEXT_ERROR_MARKERS)

    def _resolve_input_token_budget(self):
        """Resolve the input token budget for the configured OpenAI model."""

        override = os.getenv("JBG_MAX_INPUT_TOKENS")
        if override:
            try:
                resolved = max(MIN_MAX_INPUT_TOKENS, int(override))
                logger.info(f"Input token budget set to {resolved} by JBG_MAX_INPUT_TOKENS.")
                return resolved
            except ValueError:
                logger.warning(
                    f"Ignoring invalid JBG_MAX_INPUT_TOKENS value: {override!r}"
                )

        model = (self.openai_model or "").lower()
        best_budget = None
        best_prefix_length = -1
        for prefix, budget in MODEL_INPUT_TOKEN_BUDGETS.items():
            if model.startswith(prefix) and len(prefix) > best_prefix_length:
                best_budget, best_prefix_length = budget, len(prefix)

        if best_budget is None:
            logger.info(
                f"No token budget configured for model '{self.openai_model}'. "
                f"Using default {DEFAULT_MAX_INPUT_TOKENS}; it will be lowered "
                "automatically if the model rejects the request."
            )
            return DEFAULT_MAX_INPUT_TOKENS

        return best_budget

    def _condense_whole_text(self, instructions, text, task_name, reduce_suffix=None):
        """Run an instruction whose answer is much smaller than its input.

        One call when the text fits, otherwise map over segments and merge the
        partial answers into one.
        """
        if not (text or "").strip():
            return ""

        enc = self._get_encoder_for_segmentation()
        budget = self._resolve_input_token_budget()
        max_output = self._resolve_output_token_limit()
        available = self._available_input_tokens(instructions, enc, budget)
        segments = self._split_into_segments(text, enc, max_tokens=available)

        if len(segments) == 1:
            return (self.call_openai(
                instructions=instructions,
                input_message=text,
                max_output_tokens=max_output,
            ).content or "").strip()

        logger.info(f"{task_name}: {len(segments)} segment, slås ihop efteråt.")
        partials = []
        for i, segment in enumerate(segments):
            if i > 0:
                time.sleep(SEGMENT_PAUSE_SECONDS)
            logger.info(f" {task_name}: segment {i+1}/{len(segments)}...")
            partials.append((self.call_openai(
                instructions=instructions + "\n\n" + MAP_STAGE_SUFFIX,
                input_message=f"(Detta är del {i+1} av {len(segments)}.)\n\n" + segment,
                max_output_tokens=max_output,
            ).content or "").strip())

        body = "\n\n".join(
            f"--- Underlag {i+1} av {len(partials)} ---\n{part}"
            for i, part in enumerate(partials)
        )
        suffix = reduce_suffix if reduce_suffix else REDUCE_STAGE_SUFFIX
        return (self.call_openai(
            instructions=instructions + "\n\n" + suffix,
            input_message=body,
            max_output_tokens=max_output,
        ).content or "").strip()

    def _rewrite_whole_text(self, instructions, text, task_name):
        """Run an instruction that must give the whole text back, in segments.

        Used for work that transforms the transcription rather than condensing
        it. Segments carry no overlap, since the outputs are concatenated and an
        overlap would duplicate text.
        """
        enc = self._get_encoder_for_segmentation()
        max_output = self._resolve_output_token_limit()
        budget = min(
            self._rewrite_segment_budget(instructions, enc),
            self._resolve_rewrite_segment_tokens(),
        )

        for attempt in range(MAX_REWRITE_RETRIES + 1):
            result = self._rewrite_pass(instructions, text, task_name, enc, budget, max_output)
            if self._rewrite_kept_the_text(task_name, text, result):
                return result

            smaller = max(MIN_MAX_INPUT_TOKENS, budget // REWRITE_RETRY_DIVISOR)
            if attempt >= MAX_REWRITE_RETRIES or smaller >= budget:
                logger.error(
                    f"{task_name}: gav fortfarande ofullständig text. Steget används inte."
                )
                return ""

            budget = smaller
            logger.warning(
                f"{task_name}: försöker igen med mindre segment ({budget} tokens)."
            )

        return ""

    def _rewrite_pass(self, instructions, text, task_name, enc, budget, max_output):
        """One pass over the text at a given segment size."""
        segments = self._split_into_segments(text, enc, max_tokens=budget, overlap_sentences=0)
        logger.info(f"{task_name}: {len(segments)} segment (budget {budget} tokens).")

        parts, truncated_segments = [], 0
        for i, segment in enumerate(segments):
            if i > 0:
                time.sleep(SEGMENT_PAUSE_SECONDS)
            logger.info(f" {task_name}: segment {i+1}/{len(segments)}...")
            result = self.call_openai(
                instructions=instructions,
                input_message=segment,
                max_output_tokens=max_output,
            )
            if self.last_answer_truncated:
                truncated_segments += 1
            parts.append((result.content or "").strip())

        if truncated_segments:
            logger.error(
                f"{task_name}: {truncated_segments} av {len(segments)} segment kapades "
                "av modellens svarslängd. Resultatet kan sakna text."
            )

        return "\n\n".join(part for part in parts if part)

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

    def _resolve_rewrite_segment_tokens(self,
                                        env_name="JBG_REWRITE_SEGMENT_TOKENS",
                                        default=None):
        """Practical ceiling on how much text to ask a model to echo back.

        The output limit alone allows segments that are too large in practice,
        and the right size differs by task, so each one has its own ceiling.
        """
        override = os.getenv(env_name)
        if override:
            try:
                return max(MIN_MAX_INPUT_TOKENS, int(override))
            except ValueError:
                logger.warning(f"Ignoring invalid {env_name} value: {override!r}")
        return default if default is not None else DEFAULT_REWRITE_SEGMENT_TOKENS

    def _rewrite_kept_the_text(self, task_name, original, result):
        """Check that a rewriting step gave the whole text back.

        The model is asked to return the text marked up, but nothing forces it
        to: it may answer with only the passages it changed. That would replace
        the transcription in the document with an excerpt of it, silently.
        Comparing word counts catches it; a rewrite normally comes back
        slightly longer than it went in.
        """
        before = len((original or "").split())
        after = len((result or "").split())
        if not before:
            return True

        ratio = after / before
        if ratio < REWRITE_MIN_WORD_RATIO_FAIL:
            logger.error(
                f"{task_name}: svaret innehåller {after:,} ord mot {before:,} i "
                f"originalet ({ratio:.0%}). Så mycket text saknas att resultatet "
                "inte används.".replace(",", " ")
            )
            return False
        if ratio < REWRITE_MIN_WORD_RATIO_WARN:
            logger.warning(
                f"{task_name}: svaret innehåller {after:,} ord mot {before:,} i "
                f"originalet ({ratio:.0%}). Text kan saknas.".replace(",", " ")
            )
        else:
            logger.info(
                f"{task_name}: {after:,} ord tillbaka mot {before:,} in "
                f"({ratio:.0%}).".replace(",", " ")
            )
        return True

    def _resolve_output_token_limit(self):
        """Resolve how many tokens the model may produce in one answer."""
        override = os.getenv("JBG_MAX_OUTPUT_TOKENS")
        if override:
            try:
                return max(512, int(override))
            except ValueError:
                logger.warning(f"Ignoring invalid JBG_MAX_OUTPUT_TOKENS value: {override!r}")

        model = (self.openai_model or "").lower()
        best, best_len = None, -1
        for prefix, limit in MODEL_OUTPUT_TOKEN_LIMITS.items():
            if model.startswith(prefix) and len(prefix) > best_len:
                best, best_len = limit, len(prefix)
        return best if best is not None else DEFAULT_MAX_OUTPUT_TOKENS

    def _rewrite_segment_budget(self, instructions, enc):
        """Input budget for work where the model must return the whole text.

        Capped by the output limit rather than the context window, because the
        answer has to fit in the reply.
        """
        input_budget = self._available_input_tokens(
            instructions, enc, self._resolve_input_token_budget()
        )
        output_limit = self._resolve_output_token_limit()
        capped = int(output_limit / REWRITE_OUTPUT_HEADROOM)
        budget = max(MIN_MAX_INPUT_TOKENS, min(input_budget, capped))
        if capped < input_budget:
            logger.info(
                f"Segmentbudget begränsad av modellens svarslängd: {budget} tokens "
                f"(kontextfönstret hade tillåtit {input_budget})."
            )
        return budget

    def _available_input_tokens(self, instructions, enc, budget):
        """Tokens left for transcription content once instructions and answer are reserved."""
        instruction_tokens = len(self._tokenize(instructions, enc))
        available = budget - instruction_tokens - RESERVED_RESPONSE_TOKENS
        return max(MIN_MAX_INPUT_TOKENS, available)

    def find_suspicious_phrases(self):
        """Identifierar och markerar osannolika eller grammatiskt tveksamma ordkombinationer."""
        try:
            self.current_task = "Markering av misstänkta fel"
            instructions = JBGtranscriber.policy_text(self.prompt_policy, "suspicious_phrases")
            self.marked_text = self._rewrite_whole_text(
                instructions, self.transcription, "Markering av misstänkta fel"
            )
        except Exception as e:
            logger.error(f"An error occurred while finding suspicious phrases: {e}")
            # Left empty on purpose: when this is the chosen presentation, the
            # document falls back to the raw text rather than printing an error
            # message where the whole transcription should be.
            self.marked_text = ""

    def suggest_follow_up_questions(self):
        """Föreslår fem relevanta uppföljningsfrågor baserat på transkriberingen."""
        try:
            self.current_task = "Uppföljningsfrågor"
            instructions = JBGtranscriber.policy_text(
                self.prompt_policy, "follow_up_questions", "Generera uppföljningsfrågor:"
            )
            self.follow_up_questions = self._condense_whole_text(
                instructions,
                self.transcription,
                "Förslag på uppföljningsfrågor",
                reduce_suffix=(
                    "Nedan följer frågeförslag från flera delar av samma intervju. "
                    "Välj ut och formulera den slutliga uppsättningen frågor enligt "
                    "instruktionerna ovan. Slå ihop frågor som överlappar och "
                    "numrera dem. Nämn inte att materialet varit uppdelat."
                ),
            )
        except Exception as e:
            logger.error(f"An error occurred while generating follow-up questions: {e}")
            self.follow_up_questions = "Förslag till uppföljande frågor misslyckades"
            
    @staticmethod
    def _split_speaker_register(answer):
        """Separate the dialogue from the trailing speaker register.

        Returns (dialogue, register) where register maps a label to its short
        distinguishing note. A model that ignores the instruction simply yields
        an empty register, and the labels are recovered from the dialogue.
        """
        text = (answer or "").strip()
        if not text:
            return "", {}

        matches = list(_SPEAKER_REGISTER_LINE_RE.finditer(text))
        if not matches:
            return text, JBGtranscriber._labels_from_dialogue(text)

        # Everything from the last heading onwards is the register; any earlier
        # heading means the model repeated itself, so those blocks are dropped
        # from the dialogue too rather than left sitting in the middle of it.
        last = matches[-1]
        dialogue = text[:last.start()].strip()
        if len(matches) > 1:
            logger.warning(
                f"Talaranalys: {len(matches)} talarregister i ett svar. "
                "Endast det sista används."
            )
            keep = []
            cursor = 0
            for m in matches[:-1]:
                keep.append(text[cursor:m.start()])
                # Skip the block: up to the next blank line or the next heading.
                rest = text[m.end():]
                stop = rest.find("\n\n")
                cursor = m.end() + (stop if stop != -1 else len(rest))
            keep.append(text[cursor:last.start()])
            dialogue = "".join(keep).strip()

        register = {}
        for line in text[last.end():].splitlines():
            line = line.strip().lstrip("-").strip()
            if not line or ":" not in line:
                continue
            label, _, note = line.partition(":")
            label = label.strip()
            if label:
                register[label] = note.strip()

        if not register:
            register = JBGtranscriber._labels_from_dialogue(dialogue)
        return dialogue, register

    @staticmethod
    def _labels_from_dialogue(text):
        """Recover speaker labels from the dialogue itself.

        Used when the model did not produce a register. Only lines that begin
        with a short label followed by a colon are considered.
        """
        register = {}
        for line in (text or "").splitlines():
            match = _SPEAKER_LINE_RE.match(line.strip())
            if match:
                register.setdefault(match.group(1).strip(), "")
        return register

    @staticmethod
    def _append_speaker_register(dialogue, register):
        """Put the speaker register once, at the end of the section."""
        dialogue = (dialogue or "").strip()
        described = {
            label: note for label, note in (register or {}).items() if note.strip()
        }
        if not described:
            return dialogue

        lines = [SPEAKER_REGISTER_SECTION_TITLE]
        for label in sorted(described):
            lines.append(f"{label}: {described[label]}")
        return (dialogue + "\n\n" + "\n".join(lines)).strip()

    @staticmethod
    def _merge_speaker_registers(known, new):
        """Carry notes forward, keeping the first description of each speaker."""
        merged = dict(known)
        for label, note in new.items():
            if label not in merged or not merged[label]:
                merged[label] = note
        return merged

    @staticmethod
    def _format_speaker_register(register):
        return "\n".join(
            f"- {label}: {note}" if note else f"- {label}"
            for label, note in register.items()
        )

    def _speaker_context_tail(self, text, enc):
        """The last few sentences of a segment, for continuity across a boundary."""
        atoms = self._split_into_atoms(text)
        if not atoms:
            return ""
        return "".join(atoms[-SPEAKER_CONTEXT_SENTENCES:]).strip()

    def _diarize_segment(self, instructions, segment, part, total, register, context):
        """Diarize one segment, given what is known about the speakers so far."""
        prompt = instructions
        if register:
            prompt += "\n\n" + SPEAKER_REGISTER_PREAMBLE.format(
                register=JBGtranscriber._format_speaker_register(register)
            )
        prompt += "\n\n" + SPEAKER_REGISTER_SUFFIX

        body = ""
        if context:
            body += SPEAKER_CONTEXT_PREAMBLE.format(context=context) + "\n\n"
        if total > 1:
            body += f"(Detta är del {part} av {total}.)\n\n"
        body += segment

        answer = self.call_openai(
            instructions=prompt,
            input_message=body,
            max_output_tokens=self._resolve_output_token_limit(),
        ).content

        if self.last_answer_truncated:
            logger.error(
                f"Talaranalys: del {part} av {total} kapades av modellens svarslängd. "
                "Delar av texten kan saknas."
            )
        return JBGtranscriber._split_speaker_register(answer)

    def do_analyze_speakers(self):
        """Analysera transkriberingen med avseende på vilka talare som säger vad."""
        self.do_analyze_speakers_splitted()

    def do_analyze_speakers_splitted(self):
        """Label the transcription with who is speaking.

        The work rewrites the text rather than condensing it, so the segment
        size is capped by how much the model can answer. Segments are processed
        in order, each one told which speakers have already been identified and
        given the tail of the previous segment as context, so that the labels
        mean the same thing from beginning to end.
        """
        self.current_task = "Talaranalys"
        enc = self._get_encoder_for_segmentation()

        instructions = JBGtranscriber.policy_text(
            self.prompt_policy,
            "speaker_diarization",
            "Försök att identifiera olika röster i följande transkribering:",
        )
        # Capped the same way as other rewriting work, and more tightly: the
        # register is only useful if the text is actually split, and a whole
        # interview in one call gave an unstable speaker count.
        available = min(
            self._rewrite_segment_budget(instructions, enc),
            self._resolve_rewrite_segment_tokens(
                env_name="JBG_DIARIZATION_SEGMENT_TOKENS",
                default=DEFAULT_DIARIZATION_SEGMENT_TOKENS,
            ),
        )

        # No overlap: continuity is carried by the register and the context
        # tail instead, so that nothing is emitted twice.
        segments = self._split_into_segments(
            self.transcription, enc, max_tokens=available, overlap_sentences=0
        )

        logger.info(f"Talaranalys: {len(segments)} segment (budget {available} tokens).")

        register = {}
        parts = []
        try:
            for i, segment in enumerate(segments):
                if i > 0:
                    time.sleep(SEGMENT_PAUSE_SECONDS)
                    logger.info(f"Bearbetar segment {i+1}/{len(segments)}...")

                context = self._speaker_context_tail(segments[i - 1], enc) if i > 0 else ""
                dialogue, found = self._diarize_segment(
                    instructions, segment, i + 1, len(segments), register, context
                )
                register = JBGtranscriber._merge_speaker_registers(register, found)
                if dialogue:
                    parts.append(dialogue)

            if register:
                logger.info(
                    "Talaranalys identifierade: "
                    + ", ".join(sorted(register.keys()))
                )
            self.speaker_register = register

            # The register and the context tail keep the seams clean, so the
            # parts join directly. Measured over two runs of a seven-segment
            # recording: no turn repeated verbatim at any of the six seams.
            merged = "\n\n".join(parts)
            dialogue = merged

            if not self._rewrite_kept_the_text("Talaranalys", self.transcription, dialogue):
                self.analyze_speakers = "Försöket till identifiering av talare gav ofullständig text"
                self.speaker_register = {}
                return

            # One consolidated register at the end of the section, rather than
            # a description of the speakers after every segment.
            self.analyze_speakers = JBGtranscriber._append_speaker_register(dialogue, register)

        except Exception as e:
            logger.error(f"An error occurred while analyzing speakers: {e}")
            self.analyze_speakers = "Försöket till identifiering av talare misslyckades"
            self.speaker_register = {}

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

    def _tokenize(self, text, enc):
        return enc.encode(text)

    def _detokenize(self, tokens, enc):
        return enc.decode(tokens)
    
    def _split_into_atoms(self, text):
        """Split text into the smallest units a segment boundary may fall between.

        Paragraphs are preferred, sentences are used inside them. Trailing
        whitespace is carried on each atom so the text can be reassembled
        without loss.
        """
        atoms = []
        paragraphs = [p for p in _PARAGRAPH_BOUNDARY_RE.split(text) if p.strip()]

        for paragraph in paragraphs:
            sentences = [s for s in _SENTENCE_BOUNDARY_RE.split(paragraph.strip()) if s.strip()]
            if not sentences:
                continue
            for index, sentence in enumerate(sentences):
                is_last = index == len(sentences) - 1
                atoms.append(sentence.strip() + ("\n\n" if is_last else " "))

        return atoms

    def _enforce_atom_size(self, atoms, enc, max_tokens):
        """Hard-split any single atom that is larger than the budget on its own.

        Without this, a transcription with no sentence punctuation at all could
        produce an atom that never fits and the packing loop would not progress.
        """
        sized = []
        for atom in atoms:
            tokens = self._tokenize(atom, enc)
            if len(tokens) <= max_tokens:
                sized.append((atom, len(tokens)))
                continue

            logger.warning(
                f"Ett textblock på {len(tokens)} tokens saknar meningsgränser och "
                "delas på tokennivå."
            )
            try:
                for start in range(0, len(tokens), max_tokens):
                    piece_tokens = tokens[start:start + max_tokens]
                    sized.append((self._detokenize(piece_tokens, enc), len(piece_tokens)))
            except NotImplementedError:
                # The approximate encoder cannot reconstruct text from tokens,
                # so split on words instead and re-measure each piece.
                words = atom.split()
                if not words:
                    continue
                # One extra piece as margin, since a word-based split cannot
                # land exactly on the token budget.
                pieces = max(1, -(-len(tokens) // max_tokens)) + 1
                per_piece = max(1, -(-len(words) // pieces))
                for start in range(0, len(words), per_piece):
                    piece = " ".join(words[start:start + per_piece]) + " "
                    sized.append((piece, len(self._tokenize(piece, enc))))

        return sized

    def _split_into_segments(self, text, enc, max_tokens=None,
                             overlap_sentences=SEGMENT_OVERLAP_SENTENCES):
        """Split a transcription into segments that each fit the token budget.

        Returns the text unchanged as a single segment whenever it fits, which
        is the common case with current context windows. Boundaries are placed
        between sentences rather than at arbitrary token offsets, and a small
        sentence overlap preserves context across a boundary.
        """
        if max_tokens is None:
            max_tokens = self._resolve_input_token_budget()

        total_tokens = len(self._tokenize(text, enc))
        if total_tokens <= max_tokens:
            logger.info(
                f"Text has {total_tokens} tokens and fits within the budget of "
                f"{max_tokens} tokens - no segmentation needed."
            )
            return [text]

        sized_atoms = self._enforce_atom_size(self._split_into_atoms(text), enc, max_tokens)
        if not sized_atoms:
            return [text]

        segments = []
        current = []
        current_tokens = 0

        for atom, atom_tokens in sized_atoms:
            if current and current_tokens + atom_tokens > max_tokens:
                segments.append("".join(a for a, _ in current).strip())

                carry = current[-overlap_sentences:] if overlap_sentences > 0 else []
                carry_tokens = sum(t for _, t in carry)
                # Drop the overlap if it would leave no room for new content.
                if carry_tokens + atom_tokens > max_tokens:
                    carry, carry_tokens = [], 0
                current, current_tokens = list(carry), carry_tokens

            current.append((atom, atom_tokens))
            current_tokens += atom_tokens

        if current:
            segments.append("".join(a for a, _ in current).strip())

        logger.info(
            f"Text has {total_tokens} tokens and results in {len(segments)} segments "
            f"(budget {max_tokens} tokens, {overlap_sentences} sentences overlap)"
        )
        return segments

    
    def transcribe_default(self):
        """Put together the model of choice and do the transcription"""
    
        # Ensure cache directory exists
        Path("cache").mkdir(exist_ok=True)

        # Check cache
        cache_file = self.get_transcription_cache_path()
        if cache_file.exists():
            logger.info(f" Using cached transcription: {cache_file.name}")
            with open(cache_file, "r", encoding="utf-8") as f:
                content = f.read()
                parts = content.split(CACHE_TIMESTAMPED_MARKER)
                self.transcription = parts[0].replace(CACHE_TRANSCRIPTION_MARKER, "").strip()
                self.transcription_w_timestamps = parts[1].strip() if len(parts) > 1 else ""
            return
    
        # What model to use
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.transcriber_model_id, torch_dtype=self.torch_dtype, use_safetensors=True, cache_dir=JBGtranscriber.CACHE_DIR
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
        
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(f"{CACHE_TRANSCRIPTION_MARKER}\n")
            f.write(self.transcription + "\n")
            f.write(f"\n{CACHE_TIMESTAMPED_MARKER}\n")
            f.write(self.transcription_w_timestamps)
        
        logger.info(f" Transcription and timestamps cached as: {cache_file.name}")
    
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
            Path("cache").mkdir(exist_ok=True)
            cache_file = self.get_transcription_cache_path()
            if cache_file.exists():
                logger.info(f" Using cached transcription: {cache_file.name}")
                with open(cache_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    parts = content.split(CACHE_TIMESTAMPED_MARKER)
                    self.transcription = parts[0].replace(CACHE_TRANSCRIPTION_MARKER, "").strip()
                    self.transcription_w_timestamps = parts[1].strip() if len(parts) > 1 else ""
                return
        else:
            logger.warning(f" Cannot use cached transcriptions and encryption as the same time")
            if self.audio_stream:
                from pydub import AudioSegment
                import io

                self.audio_stream.seek(0)
                try:
                    # Läs MP3 från stream
                    audio = AudioSegment.from_file(io.BytesIO(self.audio_stream.read()), format="mp3")

                    # Konvertera till mono, 16-bit, 16kHz om det behövs
                    audio = audio.set_channels(1).set_sample_width(2)
                    original_rate = audio.frame_rate
                    if original_rate != RESAMPLING_TARGET_RATE:
                        logger.info(f"Resampling from {original_rate} Hz to {RESAMPLING_TARGET_RATE} Hz")
                        audio = audio.set_frame_rate(RESAMPLING_TARGET_RATE)
                    self.samplerate = audio.frame_rate

                    # Extrahera samples
                    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
                    self.audio_data = samples
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
                required_ram_gb = self.TRANSCRIBER_MODEL_RAM_REQUIREMENTS.get(model_id, None)

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


                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, torch_dtype=self.torch_dtype, use_safetensors=True, cache_dir=JBGtranscriber.CACHE_DIR
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

                audio_input = self.audio_data if hasattr(self, "audio_data") else str(self.convert_path)
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
                    f"{JBGtranscriber._format_duration(self.transcription_seconds)}, "
                    f"{speed_description})"
                )
                self.transcriber_model_used = model_id
                if model_id != TRANSCRIPTION_ACCURACY_MODELS.get(self.transcription_accuracy):
                    logger.warning(
                        f"Vald noggrannhet '{self.transcription_accuracy}' motsvarar "
                        f"{TRANSCRIPTION_ACCURACY_MODELS.get(self.transcription_accuracy)}, "
                        f"men {model_id} användes i stället."
                    )

                
                if not self.secure_handler:
                    cache_file = self.get_transcription_cache_path()
                    with open(cache_file, "w", encoding="utf-8") as f:
                        f.write(f"{CACHE_TRANSCRIPTION_MARKER}\n")
                        f.write(self.transcription + "\n")
                        f.write(f"\n{CACHE_TIMESTAMPED_MARKER}\n")
                        f.write(self.transcription_w_timestamps)

                    logger.info(f" Transcription and timestamps cached as: {cache_file.name}")
                return

            except Exception as e:
                logger.error(f"Model {model_id} failed: {e}")
                continue

        raise RuntimeError("All model options failed. Try reducing file size or increasing memory.")

    def _postprocess_result(self, result):
        """Post process result of call to transcription model"""
        
        transcription = result["text"]
        if self.insert_linebreaks:
            transcription = JBGtranscriber.insert_newlines(transcription, 80)
        
        transcription_w_timestamps = ""
        for chunk in result["chunks"]:
            transcription_w_timestamps += str(chunk["timestamp"]) + ": " + str(chunk["text"]) + "\n"
        if self.insert_linebreaks:
            transcription_w_timestamps = JBGtranscriber.insert_newlines(transcription_w_timestamps, 80)
            
        return transcription, transcription_w_timestamps

    @staticmethod
    def _configure_word_styles(document):
        """Apply a restrained, readable style hierarchy to generated Word files."""

        style_settings = {
            "Normal": (11, False, 0, 6),
            "Title": (22, True, 0, 12),
            "Heading 1": (16, True, 16, 8),
            "Heading 2": (14, True, 14, 6),
            "Heading 3": (12, True, 12, 4),
            "Heading 4": (11, True, 10, 3),
        }

        for style_name, (font_size, bold, before, after) in style_settings.items():
            if style_name not in document.styles:
                continue
            style = document.styles[style_name]
            style.font.name = "Aptos"
            style.font.size = Pt(font_size)
            style.font.bold = bold
            style.paragraph_format.space_before = Pt(before)
            style.paragraph_format.space_after = Pt(after)
            if style_name.startswith("Heading"):
                style.paragraph_format.keep_with_next = True

        document.styles["Normal"].paragraph_format.line_spacing = 1.08

    @staticmethod
    def _add_inline_word_markup(paragraph, text):
        """Render the small Markdown subset returned by the AI as native Word runs.

        Supported constructs are bold/italic Markdown and the application's
        [FEL?]...[/FEL?] markers. Control markers themselves are never written to
        the document.
        """

        bold = False
        italic = False
        code = False
        suspect = False
        buffer = []

        def flush_buffer():
            if not buffer:
                return
            run = paragraph.add_run("".join(buffer))
            run.bold = bold
            run.italic = italic
            if code:
                run.font.name = "Consolas"
            if suspect:
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
                run.bold = True
            buffer.clear()

        i = 0
        while i < len(text):
            if text.startswith("[FEL?]", i):
                flush_buffer()
                suspect = True
                i += len("[FEL?]")
                continue
            if text.startswith("[/FEL?]", i):
                flush_buffer()
                suspect = False
                i += len("[/FEL?]")
                continue
            if text.startswith("***", i):
                flush_buffer()
                bold = not bold
                italic = not italic
                i += 3
                continue
            if text.startswith("**", i):
                flush_buffer()
                bold = not bold
                i += 2
                continue
            if text[i] == "*":
                flush_buffer()
                italic = not italic
                i += 1
                continue
            if text[i] == "`":
                flush_buffer()
                code = not code
                i += 1
                continue

            buffer.append(text[i])
            i += 1

        flush_buffer()

    @staticmethod
    def _add_horizontal_rule(document):
        """Add a subtle Word paragraph border instead of a literal Markdown '---'."""

        paragraph = document.add_paragraph()
        p_pr = paragraph._p.get_or_add_pPr()
        p_bdr = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "4")
        bottom.set(qn("w:space"), "1")
        bottom.set(qn("w:color"), "B7C9E2")
        p_bdr.append(bottom)
        p_pr.append(p_bdr)
        paragraph.paragraph_format.space_before = Pt(3)
        paragraph.paragraph_format.space_after = Pt(6)

    @staticmethod
    def _add_plain_word_content(document, content, parse_inline=False, speaker_labels=False):
        """Write prose while preserving intentional paragraph and line breaks."""

        if not content:
            return

        blocks = re.split(r"\n\s*\n", content.strip())
        for block in blocks:
            if not block.strip():
                continue

            if speaker_labels:
                match = re.match(
                    r"^((?:Intervjuare|Intervjuobjekt)\s+\d+)\s*:\s*(.*)$",
                    block.strip(),
                    flags=re.DOTALL,
                )
                if match:
                    paragraph = document.add_paragraph()
                    label = paragraph.add_run(match.group(1) + ": ")
                    label.bold = True
                    JBGtranscriber._add_inline_word_markup(paragraph, match.group(2).strip())
                    paragraph.paragraph_format.space_after = Pt(4)
                    continue

            paragraph = document.add_paragraph()
            lines = block.splitlines()
            for line_index, line in enumerate(lines):
                if parse_inline:
                    JBGtranscriber._add_inline_word_markup(paragraph, line.rstrip())
                else:
                    paragraph.add_run(line.rstrip())
                if line_index < len(lines) - 1:
                    paragraph.add_run().add_break()

    @staticmethod
    def _add_structured_word_content(document, content):
        """Convert AI Markdown-like output to real Word headings, lists and runs."""

        if not content:
            return

        for raw_line in content.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                continue

            if re.fullmatch(r"-{2,}", stripped):
                JBGtranscriber._add_horizontal_rule(document)
                continue

            heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*#*$", stripped)
            if heading_match:
                hashes, heading_text = heading_match.groups()
                # AI summaries typically use ##/### for divisions that are nested
                # under the application's own Heading 1 section title.
                heading_level = 2 if len(hashes) <= 3 else min(4, len(hashes) - 1)
                document.add_heading(heading_text.rstrip(":").strip(), level=heading_level)
                continue

            bold_only = re.fullmatch(r"\*\*(.+?)\*\*\s*", stripped)
            if bold_only:
                heading_text = bold_only.group(1).strip()
                if re.match(r"^\d+[.)]\s+", heading_text):
                    heading_level = 3
                elif heading_text.endswith(":"):
                    heading_level = 4
                else:
                    heading_level = 3
                document.add_heading(heading_text.rstrip(":").strip(), level=heading_level)
                continue

            italic_label = re.match(r"^\*([^*]+?):\*\s*(.*)$", stripped)
            if italic_label:
                label, remainder = italic_label.groups()
                document.add_heading(label.strip(), level=4)
                if remainder.strip():
                    paragraph = document.add_paragraph()
                    JBGtranscriber._add_inline_word_markup(paragraph, remainder.strip())
                continue

            bullet_match = re.match(r"^(\s*)[-+*]\s+(.+)$", line)
            if bullet_match:
                indent, bullet_text = bullet_match.groups()
                level = min(2, max(0, len(indent.expandtabs(2)) // 2))
                style_name = "List Bullet" if level == 0 else f"List Bullet {level + 1}"
                paragraph = document.add_paragraph(style=style_name)
                JBGtranscriber._add_inline_word_markup(paragraph, bullet_text.strip())
                continue

            numbered_match = re.match(r"^(\s*)\d+[.)]\s+(.+)$", line)
            if numbered_match:
                indent, numbered_text = numbered_match.groups()
                level = min(2, max(0, len(indent.expandtabs(2)) // 2))
                style_name = "List Number" if level == 0 else f"List Number {level + 1}"
                paragraph = document.add_paragraph(style=style_name)
                JBGtranscriber._add_inline_word_markup(paragraph, numbered_text.strip())
                continue

            quote_match = re.match(r"^>\s?(.*)$", stripped)
            if quote_match:
                paragraph = document.add_paragraph(style="Quote")
                JBGtranscriber._add_inline_word_markup(paragraph, quote_match.group(1))
                continue

            paragraph = document.add_paragraph()
            JBGtranscriber._add_inline_word_markup(paragraph, stripped)

    def _record_model_call(self, task, model, sent, received, seconds):
        """Note one model call for the statistics table.

        Only counts and timings are kept - never any part of the text.
        """
        # Never append to the class-level default, which would be shared
        # between instances; give this instance its own list first.
        if "model_calls" not in self.__dict__:
            self.model_calls = []

        self.model_calls.append({
            "task": task,
            "model": model,
            "sent": sent or 0,
            "received": received or 0,
            "seconds": seconds,
        })

    def _model_statistics_rows(self):
        """Aggregate the recorded calls, one row per task and model."""
        rows = []
        order = []
        totals = {}
        for call in self.model_calls:
            key = (call["task"], call["model"])
            if key not in totals:
                totals[key] = {"calls": 0, "sent": 0, "received": 0, "seconds": 0.0}
                order.append(key)
            t = totals[key]
            t["calls"] += 1
            t["sent"] += call["sent"]
            t["received"] += call["received"]
            t["seconds"] += call["seconds"]

        for key in order:
            task, model = key
            t = totals[key]
            rows.append([
                task, model, str(t["calls"]),
                f"{t['sent']:,}".replace(",", " "),
                f"{t['received']:,}".replace(",", " "),
                JBGtranscriber._format_duration(t["seconds"]),
            ])

        if self.transcription_seconds:
            rows.insert(0, [
                "Transkribering",
                self.transcriber_model_used or "-",
                "1", "-", "-",
                JBGtranscriber._format_duration(self.transcription_seconds),
            ])

        if len(rows) > 1:
            rows.append([
                "Totalt", "",
                str(sum(t["calls"] for t in totals.values()) + (1 if self.transcription_seconds else 0)),
                f"{sum(t['sent'] for t in totals.values()):,}".replace(",", " "),
                f"{sum(t['received'] for t in totals.values()):,}".replace(",", " "),
                JBGtranscriber._format_duration(
                    sum(t["seconds"] for t in totals.values()) + (self.transcription_seconds or 0)
                ),
            ])
        return rows

    @staticmethod
    def _format_duration(seconds):
        seconds = int(round(seconds or 0))
        if seconds < 60:
            return f"{seconds} s"
        return f"{seconds // 60} min {seconds % 60} s"

    @staticmethod
    def _add_word_table(document, content):
        """Render the statistics as a real Word table."""
        header, rows = content[0], content[1:]
        if not rows:
            return
        table = document.add_table(rows=1, cols=len(header))
        table.style = "Light Grid Accent 1"
        for cell, title in zip(table.rows[0].cells, header):
            cell.text = title
            for run in cell.paragraphs[0].runs:
                run.bold = True
        for row in rows:
            cells = table.add_row().cells
            for cell, value in zip(cells, row):
                cell.text = str(value)

    def _transcription_provenance(self):
        """A short note on which model produced the transcription.

        The chosen accuracy is a ceiling, and the model actually used may be a
        smaller one if the available RAM did not allow the choice. That
        fallback is otherwise invisible, so it is recorded here: someone who
        picked a level deserves to know what they got.
        """
        level = self.transcription_accuracy
        wanted = TRANSCRIPTION_ACCURACY_MODELS.get(level, "")
        label = TRANSCRIPTION_ACCURACY_LABELS.get(level, level)
        used = self.transcriber_model_used

        if not used:
            return ""

        line = f"Vald noggrannhet: {label}. Modell som användes: {used}."
        if wanted and used != wanted:
            line += (
                f" Den valda nivån motsvarar {wanted}, men tillgängligt minne "
                "räckte inte, så en mindre modell användes i stället."
            )
        return line

    def _transcription_section(self):
        """Return the single transcription section for the result document.

        The three presentations are mutually exclusive. If the chosen one is
        unavailable - marking failed, or the model returned no timestamps - the
        raw text is used instead and the document says so, rather than leaving
        the reader with an error message where the transcription should be.
        """
        chosen = self.transcription_format if self.transcription_format in TRANSCRIPTION_FORMATS \
            else DEFAULT_TRANSCRIPTION_FORMAT

        content = {
            "raw": self.transcription,
            "timestamps": self.transcription_w_timestamps,
            "marked": self.marked_text,
        }.get(chosen, self.transcription)

        if not (content or "").strip() and chosen != "raw":
            logger.warning(
                f"Ingen text tillgänglig i formatet '{chosen}'. Använder rå text i stället."
            )
            note = {
                "timestamps": "[Tidsstämplar var inte tillgängliga. Rå text visas i stället.]",
                "marked": "[Markering av misstänkta fel kunde inte genomföras. Rå text visas i stället.]",
            }.get(chosen, "")
            body = (self.transcription or "").strip()
            return (
                TRANSCRIPTION_FORMAT_HEADINGS["raw"],
                (note + "\n\n" + body).strip() if note else body,
                "plain",
            )

        render_mode = "marked" if chosen == "marked" else "plain"
        return (TRANSCRIPTION_FORMAT_HEADINGS[chosen], content, render_mode)

    def write_to_output_file(self):
        """Write transcription and selected analyses to a Word document.

        When encryption is enabled, the DOCX package is created only in memory
        and AES-GCM encrypted before any result bytes are written to disk.
        """

        encrypted_output = self.secure_handler is not None
        if encrypted_output:
            if not self.export_path.name.lower().endswith(".docx.encrypted"):
                raise ValueError(
                    f"Encrypted output path must end with .docx.encrypted: {self.export_path}"
                )
        elif self.export_path.suffix.lower() != ".docx":
            raise ValueError(f"Output path must use the .docx extension: {self.export_path}")

        statistics = self._model_statistics_rows()
        sections = [
            self._transcription_section(),
            ("Sammanfattning", self.summary, "structured"),
            ("Uppföljningsfrågor", self.follow_up_questions, "structured"),
            ("Försök till identifiering av olika talare", self.analyze_speakers, "speakers"),
            # Provenance and statistics belong after the content, not between
            # the transcription and the analyses.
            ("Om transkriberingen", self._transcription_provenance(), "plain"),
            ("Statistik", [STATISTICS_TABLE_HEADER] + statistics if statistics else "", "table"),
        ]

        temp_path = self.export_path.with_name(self.export_path.name + ".tmp")
        try:
            self.export_path.parent.mkdir(parents=True, exist_ok=True)
            document = Document()
            JBGtranscriber._configure_word_styles(document)
            document.add_heading("Transkribering och analys", level=0)

            for heading, content, render_mode in sections:
                if not content:
                    continue
                document.add_heading(heading, level=1)
                if render_mode == "structured":
                    JBGtranscriber._add_structured_word_content(document, content)
                elif render_mode == "marked":
                    JBGtranscriber._add_plain_word_content(document, content, parse_inline=True)
                elif render_mode == "table":
                    JBGtranscriber._add_word_table(document, content)
                elif render_mode == "speakers":
                    JBGtranscriber._add_plain_word_content(document, content, parse_inline=True, speaker_labels=True)
                else:
                    JBGtranscriber._add_plain_word_content(document, content)

            # Build the complete DOCX ZIP package in memory. No plaintext DOCX
            # is written to the server filesystem in encrypted mode.
            docx_stream = io.BytesIO()
            document.save(docx_stream)
            docx_stream.seek(0)

            if encrypted_output:
                self.secure_handler.encrypt_bytesio(docx_stream, str(temp_path))
                os.replace(temp_path, self.export_path)
                logger.info(f"Krypterat DOCX-resultat sparat: {self.export_path}")
            else:
                with open(temp_path, "wb") as export_file:
                    export_file.write(docx_stream.getvalue())
                os.replace(temp_path, self.export_path)
                logger.info(f"DOCX-resultat sparat: {self.export_path}")
        except Exception as ex:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            logger.error(f"Misslyckades med att spara DOCX-resultat: {str(ex)}")
            raise

    def perform_transcription_steps(
        self,
        generate_summary=False,
        summary_style=None,
        summary_prompt=None,
        transcription_format=DEFAULT_TRANSCRIPTION_FORMAT,
        transcription_accuracy=DEFAULT_TRANSCRIPTION_ACCURACY,
        find_suspicious_phrases=None,
        suggest_follow_up_questions=False,
        analyze_speakers=False,
        progress_callback=None
    ):
        """Perform all transcription steps.

        transcription_format decides which single form of the transcribed text
        goes into the document. Marking suspected errors is the work behind the
        "marked" form, so it runs when that form is chosen; find_suspicious_phrases
        may still be passed explicitly to force it either way.
        """

        self.transcription_accuracy = (
            transcription_accuracy if transcription_accuracy in TRANSCRIPTION_ACCURACY_LEVELS
            else DEFAULT_TRANSCRIPTION_ACCURACY
        )
        self.transcription_format = (
            transcription_format if transcription_format in TRANSCRIPTION_FORMATS
            else DEFAULT_TRANSCRIPTION_FORMAT
        )
        if find_suspicious_phrases is None:
            find_suspicious_phrases = self.transcription_format == "marked"
        
        def report(msg):
            logger.info(msg)
            if progress_callback is not None:
                try:
                    progress_callback(msg)
                except Exception as e:
                    logger.error(f"Progress callback failed: {e}")

        # Transcribe the audio files
        report("Transkriberar ljudfilen...")
        self.transcribe()
        
        # The steps run in the order they appear in the GUI, which is also
        # the order the sections appear in the document. Marking suspected
        # errors produces the transcription section itself, so it is first.
        # Find and mark suspicious phrases if requested
        if find_suspicious_phrases:
            report(f"Letar misstänkta fel...")
            self.find_suspicious_phrases()
        
        # Generate a summary if requested
        if generate_summary:
            report(f"Genererar sammanfattning...")
            self.generate_summary(style=summary_style, custom_prompt=summary_prompt)
        
        # Suggest follow-up questions if requested
        if suggest_follow_up_questions:
            report(f"Föreslår uppföljningsfrågor...")
            self.suggest_follow_up_questions()
            
        # Analyze speakers if requested
        if analyze_speakers:
            report(f"Försöker identifiera olika talare...")
            #self.do_analyze_speakers()
            self.do_analyze_speakers_splitted()
        
        # Print a message confirming successful completion
        report(f"Alla steg i analysen genomfördes.")
            
        # Write the transcription results to an output file
        report(f"Spara resultatet till fil...")
        self.write_to_output_file()
        report(f"Transkribering avslutad.")

def check_script_arguments():
    """Check command-line arguments for the test script""" 
    if len(sys.argv) < 6:
        sys.exit("Usage: " + sys.argv[0] + " [path to .mp3 file or folder] [output folder] [device=gpu/cpu] [openai_api_key] (optional: model) (optional: summary style)")

    convert_path = Path(sys.argv[1])
    export_path = Path(sys.argv[2])
    device = sys.argv[3].lower()
    api_key = sys.argv[4]
    model = sys.argv[5] if len(sys.argv) > 5 else "gpt-4o"
    summary_options = JBGtranscriber.summary_options()
    valid_styles = [o["id"] for o in summary_options]
    default_style = next(o["id"] for o in summary_options if o["default"])
    summary_style = sys.argv[6] if len(sys.argv) > 6 else default_style

    if not convert_path.exists():
        sys.exit(f"{convert_path} is not a valid file or directory path")
    if not export_path.is_dir():
        sys.exit(f"{export_path} is not a valid directory path")
    if device not in ["cpu", "gpu"]:
        sys.exit("Device must be 'cpu' or 'gpu'")
    if summary_style not in valid_styles and summary_style not in LEGACY_SUMMARY_OPTION_IDS:
        sys.exit("Summary style must be one of: " + ", ".join(valid_styles))

    return convert_path, export_path, device, api_key, model, summary_style


def main():
    # Parse arguments
    convert_path, export_path, device, api_key, model, summary_style = check_script_arguments()

    # Gather files to process
    if convert_path.is_dir():
        convert_files = JBGtranscriber.find_mp3_files(convert_path)
    else:
        convert_files = [convert_path]

    for convert_file in convert_files:
        logger.info(f"Processing {convert_file.name}...")

        output_file = export_path / f"{convert_file.stem}.docx"

        # Instantiate transcriber with all required params
        transcriber = JBGtranscriber(
            convert_path=convert_file,
            export_path=output_file,
            device=device,
            api_key=api_key,
            openai_model=model
        )

        # Run all steps
        transcriber.perform_transcription_steps(
            generate_summary=True,
            summary_style=summary_style,
            transcription_format="marked",
            suggest_follow_up_questions=True,
            analyze_speakers=True
        )

    
# In case of commande line execution call
if __name__=="__main__":
    
    # Execute main program
    main()
    
    # End program successfully
    sys.exit(0)