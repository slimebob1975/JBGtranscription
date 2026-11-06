"""
Splitting text into segments that fit the model: tokens, sentence and
paragraph atoms, overlap, and the input and output token budgets.

Part of JBGtranscriber, which combines the mixins in src/JBGtranscriber.py.
"""

import os
import re
import tiktoken
from src.JBGLogger import JBGLogger

logger = JBGLogger(level="INFO").logger


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
# Kept for backwards compatibility with any external caller.
MAX_INPUT_TOKENS = DEFAULT_MAX_INPUT_TOKENS
DEFAULT_REWRITE_SEGMENT_TOKENS = 6000
# A rewritten segment is longer than its input: speaker labels, [FEL?] markers
# and suggested corrections all add tokens. Leave room for that.
REWRITE_OUTPUT_HEADROOM = 1.4
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?:…])\s+")
_PARAGRAPH_BOUNDARY_RE = re.compile(r"\n\s*\n")


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


class SegmentationMixin:
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
