"""
The analyses of the transcribed text: summary with map and reduce,
marking of suspected errors, follow-up questions and the custom analysis.

Part of JBGtranscriber, which combines the mixins in src/JBGtranscriber.py.
"""

import time
from src.JBGLogger import JBGLogger
from src.JBGPromptPolicy import DEFAULT_CUSTOM_ANALYSIS_PREAMBLE, MAX_SUMMARY_PROMPT_CHARS
from src.JBGSegmentation import MIN_MAX_INPUT_TOKENS

logger = JBGLogger(level="INFO").logger


# Pause between segment calls to stay clear of rate limits.
SEGMENT_PAUSE_SECONDS = 5
# How many times the budget may be halved in response to a context-length error.
MAX_BUDGET_DOWNSHIFTS = 4
# If the answer still comes back short, the segment size is cut and the step
# tried again rather than discarded outright.
MAX_REWRITE_RETRIES = 2
REWRITE_RETRY_DIVISOR = 3
REWRITE_MIN_WORD_RATIO_WARN = 0.90
REWRITE_MIN_WORD_RATIO_FAIL = 0.70
# Added to the instruction of an analysis that works from the marked text,
# and only then, so that instructions for unmarked text carry no rule about
# markers that are not there. The marked text holds both the original wording
# and the suggestion, so the model is asked to weigh them, not to trust the
# suggestion.
MARKED_TEXT_NOTE = (
    "OM TEXTENS MARKERINGAR:\n"
    "Texten har gåtts igenom efter misstänkta transkriberingsfel. Ett misstänkt "
    "fel står inom [FEL?] och [/FEL?], med den ursprungliga lydelsen först och "
    "ett förslag på trolig lydelse inom (Kanske: ...), till exempel: "
    "[FEL?]lufflarlagen (Kanske: LAS-lagen)[/FEL?]. Förslagen är gissningar som "
    "ingen har granskat. Använd i varje fall den lydelse, ursprunglig eller "
    "föreslagen, som är mest sannolik i sammanhanget. Är ett namn, en siffra "
    "eller en annan viktig detalj fortfarande osäker, återge den inte som säker. "
    "Återge inte markörerna och nämn inte att texten varit markerad."
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


class AnalysesMixin:
    custom_analysis = ""
    analyses_on_marked_text = []

    def _analysis_input(self, analysis_name):
        """The text a condensing analysis works from, and a note for its instruction.

        When suspected errors have been marked, the summary, follow-up
        questions and custom analysis work from the marked text, which holds
        both the original wording and the suggested correction. If marking was
        not chosen or failed, they work from the original transcription.
        Speaker identification is a presentation of its own and cannot be
        combined with marking.
        """
        marked = (self.__dict__.get("marked_text") or "").strip()
        if not marked:
            return self.transcription, ""
        if "analyses_on_marked_text" not in self.__dict__:
            self.analyses_on_marked_text = []
        if analysis_name not in self.analyses_on_marked_text:
            self.analyses_on_marked_text.append(analysis_name)
        logger.info(f"{analysis_name} bygger på texten med markerade misstänkta fel.")
        return marked, "\n\n" + MARKED_TEXT_NOTE

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
            text, note = self._analysis_input("sammanfattningen")
            self.summary = self._summarize(instructions + note, text)
        except Exception as e:
            logger.error(f" Summary generation failed: {e}")
            self.summary = "Sammanfattning var inte tillgänglig"

    def _summarize(self, instructions, text=None):
        """Run the summary, lowering the token budget if the API rejects the size."""

        text = self.transcription if text is None else text
        enc = self._get_encoder_for_segmentation()
        budget = self._resolve_input_token_budget()
        downshifts = 0

        while True:
            available = self._available_input_tokens(instructions, enc, budget)
            segments = self._split_into_segments(text, enc, max_tokens=available)

            try:
                if len(segments) == 1:
                    logger.info(
                        "Transkriberingen ryms i ett anrop - sammanfattar hela texten på en gång."
                    )
                    return self.call_openai(
                        instructions=instructions,
                        input_message=text,
                        max_output_tokens=self._resolve_output_token_limit(),
                    ).content.strip()

                logger.info(
                    f"Transkriberingen delas i {len(segments)} segment "
                    f"(budget {available} tokens per segment)."
                )
                return self._summarize_in_segments(instructions, segments, enc, available)

            except Exception as e:
                if downshifts < MAX_BUDGET_DOWNSHIFTS and self._is_context_length_error(e):
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

    def find_suspicious_phrases(self):
        """Identifierar och markerar osannolika eller grammatiskt tveksamma ordkombinationer."""
        try:
            self.current_task = "Markering av misstänkta fel"
            instructions = self.policy_text(self.prompt_policy, "suspicious_phrases")
            self.marked_text = self._rewrite_whole_text(
                instructions, self.transcription, "Markering av misstänkta fel"
            )
        except Exception as e:
            logger.error(f"An error occurred while finding suspicious phrases: {e}")
            # Left empty on purpose: when this is the chosen presentation, the
            # document falls back to the raw text rather than printing an error
            # message where the whole transcription should be.
            self.marked_text = ""

    @staticmethod
    def _user_instruction(text, what):
        """A user-edited instruction, trimmed and capped; empty means none.

        Only its length is logged, never its content, since it may hold
        case-specific context.
        """
        candidate = (text or "").strip()
        if not candidate:
            return ""
        if len(candidate) > MAX_SUMMARY_PROMPT_CHARS:
            logger.warning(
                f"{what}: instruktionen kortades från {len(candidate)} till "
                f"{MAX_SUMMARY_PROMPT_CHARS} tecken."
            )
            candidate = candidate[:MAX_SUMMARY_PROMPT_CHARS]
        logger.info(f"{what}: egen instruktion används ({len(candidate)} tecken).")
        return candidate

    def suggest_follow_up_questions(self, custom_prompt=None):
        """Föreslår uppföljningsfrågor baserat på transkriberingen."""
        try:
            self.current_task = "Uppföljningsfrågor"
            instructions = self._user_instruction(
                custom_prompt, "Uppföljningsfrågor"
            ) or self.policy_text(
                self.prompt_policy, "follow_up_questions", "Generera uppföljningsfrågor."
            )
            text, note = self._analysis_input("följdfrågorna")
            self.follow_up_questions = self._condense_whole_text(
                instructions + note,
                text,
                "Förslag på uppföljningsfrågor",
                reduce_suffix=(
                    "Nedan följer frågeförslag från flera delar av samma inspelning. "
                    "Välj ut och formulera den slutliga uppsättningen frågor enligt "
                    "instruktionerna ovan. Slå ihop frågor som överlappar och "
                    "numrera dem. Nämn inte att materialet varit uppdelat."
                ),
            )
        except Exception as e:
            logger.error(f"An error occurred while generating follow-up questions: {e}")
            self.follow_up_questions = "Förslag till uppföljande frågor misslyckades"

    def run_custom_analysis(self, custom_prompt):
        """The user's own analysis, after a fixed opening from the policy.

        The opening cannot be edited, so the instruction always starts the same
        way and the user only writes what is to be done. Long recordings are
        split and merged like a summary.
        """
        self.current_task = "Anpassad analys"
        addition = self._user_instruction(custom_prompt, "Anpassad analys")
        if not addition:
            return
        preamble = self.policy_text(
            self.prompt_policy, "custom_analysis_preamble", DEFAULT_CUSTOM_ANALYSIS_PREAMBLE
        )
        try:
            text, note = self._analysis_input("den anpassade analysen")
            self.custom_analysis = self._condense_whole_text(
                preamble + "\n" + addition + note,
                text,
                "Anpassad analys",
            )
        except Exception as e:
            logger.error(f"Den anpassade analysen misslyckades: {e}")
            self.custom_analysis = "Den anpassade analysen misslyckades"
