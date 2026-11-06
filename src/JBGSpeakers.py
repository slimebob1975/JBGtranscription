"""
Speaker identification: rewriting the text as a dialogue with speaker
labels, and the register of identified speakers. Kept in a module of its own
so that it can be removed as a unit (see BACKLOG).

Part of JBGtranscriber, which combines the mixins in src/JBGtranscriber.py.
"""

import re
import time
from src.JBGAnalyses import SEGMENT_PAUSE_SECONDS
from src.JBGLogger import JBGLogger

logger = JBGLogger(level="INFO").logger


# Speaker identification needs a smaller segment still. Measured on one
# recording: processed in a single call it reported 7, 6 and 6 speakers across
# three runs of the same audio; split into seven segments of about 2,100 tokens
# it reported the same five speakers twice. Carrying the register between
# segments appears to steady the result, so the segments are kept small enough
# that it is actually used. Override with JBG_DIARIZATION_SEGMENT_TOKENS.
DEFAULT_DIARIZATION_SEGMENT_TOKENS = 2500
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
# The format of the dialogue is fixed by the code, not by the editable
# instruction: the document is laid out by reading "Beteckning: replik" line by
# line, and a user who edits the instruction must not be able to break that.
# The editable text only describes the recording and which labels to use.
SPEAKER_FORMAT_SUFFIX = (
    "SVARETS FORM:\n"
    "Återge hela texten som en dialog. Varje replik står på en egen rad och börjar "
    "med talarens beteckning följd av kolon och mellanslag, till exempel:\n"
    "Talare 1: Kan du beskriva hur det går till?\n"
    "Talare 2: Ja, först får vi ett meddelande...\n"
    "Använd de beteckningar som instruktionen ovan anger. Gör en radbrytning mellan "
    "varje replik. Hitta inte på namn. Skriv inget utanför själva dialogen."
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
_SPEAKER_LINE_RE = re.compile(r"^([^:\n]{2,40}?\s*\d*)\s*:\s+\S")


class SpeakersMixin:
    speaker_register = {}

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
            return text, SpeakersMixin._labels_from_dialogue(text)

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
            register = SpeakersMixin._labels_from_dialogue(dialogue)
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
                register=self._format_speaker_register(register)
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
        return self._split_speaker_register(answer)

    def do_analyze_speakers(self, custom_prompt=None):
        """Analysera transkriberingen med avseende på vilka talare som säger vad."""
        self.do_analyze_speakers_splitted(custom_prompt=custom_prompt)

    def do_analyze_speakers_splitted(self, custom_prompt=None):
        """Label the transcription with who is speaking.

        The work rewrites the text rather than condensing it, so the segment
        size is capped by how much the model can answer. Segments are processed
        in order, each one told which speakers have already been identified and
        given the tail of the previous segment as context, so that the labels
        mean the same thing from beginning to end.
        """
        self.current_task = "Talaranalys"
        enc = self._get_encoder_for_segmentation()

        instructions = self._user_instruction(
            custom_prompt, "Talaranalys"
        ) or self.policy_text(
            self.prompt_policy,
            "speaker_diarization",
            "Försök att identifiera olika röster i följande transkribering:",
        )
        instructions += "\n\n" + SPEAKER_FORMAT_SUFFIX
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
                register = self._merge_speaker_registers(register, found)
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
                # Left empty, like a failed marking: as the chosen presentation
                # it would otherwise replace the whole transcription.
                self.analyze_speakers = ""
                self.speaker_register = {}
                return

            # One consolidated register at the end of the section, rather than
            # a description of the speakers after every segment.
            self.analyze_speakers = self._append_speaker_register(dialogue, register)

        except Exception as e:
            logger.error(f"An error occurred while analyzing speakers: {e}")
            self.analyze_speakers = ""
            self.speaker_register = {}
