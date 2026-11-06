"""
Calls to the language model: request parameters, retries when a model
rejects a parameter, output limits and the statistics of each call.

Part of JBGtranscriber, which combines the mixins in src/JBGtranscriber.py.
"""

import openai
import os
import time
from src.JBGLogger import JBGLogger

logger = JBGLogger(level="INFO").logger


MODEL_GPT_5_MARKER = "gpt-5"
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
_CONTEXT_ERROR_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context length",
    "too many tokens",
    "reduce the length",
    "string too long",
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


class LanguageModelMixin:
    # Set by call_openai: whether the last answer hit the length limit.
    last_answer_truncated = False
    # Class-level defaults so the document can be written even by an instance
    # that was not built through __init__.
    model_calls = []
    current_task = ""

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
            temperature=self.get_permitted_temperature(self.openai_model, DEFAULT_TEMPERATURE),
            **self.get_model_specific_extra_arguments(self.openai_model),
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
                if self._is_unsupported_parameter_error(e):
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
            # Part of prompt_tokens, but billed at the much lower cache price.
            cached=getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None),
        )

        choice = completion.choices[0]
        self.last_answer_truncated = getattr(choice, "finish_reason", None) == "length"
        if self.last_answer_truncated:
            logger.warning(
                "Modellens svar nådde längdgränsen och kan vara avklippt. "
                "Sänk segmentstorleken eller höj JBG_MAX_OUTPUT_TOKENS."
            )

        return choice.message

    @staticmethod
    def _is_context_length_error(error):
        """Detect an API error caused by exceeding the model's context window."""
        message = str(error).lower()
        return any(marker in message for marker in _CONTEXT_ERROR_MARKERS)

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

    def _record_model_call(self, task, model, sent, received, seconds, cached=None):
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
            "cached": cached or 0,
            "received": received or 0,
            "seconds": seconds,
        })
