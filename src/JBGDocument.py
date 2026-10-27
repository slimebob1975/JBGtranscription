"""
The result document in Word: sections, inline markup, the speaker layout,
tables, "Om transkriberingen", statistics and cost.

Part of JBGtranscriber, which combines the mixins in src/JBGtranscriber.py.
"""

import io
import os
import re
from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from src.JBGBuildInfo import get_build_info
from src.JBGLogger import JBGLogger
from src.JBGModelPrices import call_cost, format_cost, get_price_table
from src.JBGSpeakers import SPEAKER_REGISTER_SECTION_TITLE
from src.JBGSpeechToText import ENGINE_FASTER_WHISPER, TRANSCRIPTION_ACCURACY_LABELS, TRANSCRIPTION_ACCURACY_MODELS

logger = JBGLogger(level="INFO").logger


# --- Transcription accuracy --------------------------------------------------
#
# The Whisper model was always the largest that fits in the available RAM. The
# user can now set a ceiling instead: the chosen level is used exactly, or the
# first level below it that fits. A choice is never upgraded past what was
# asked for, so a deliberately fast run stays fast.

STATISTICS_TABLE_HEADER = ["Moment", "Modell", "Anrop", "Tokens in", "Tokens ut", "Tid"]
# How the transcribed text is presented in the result document. Exactly one
# form is included; the three are mutually exclusive.
TRANSCRIPTION_FORMATS = ("raw", "timestamps", "marked", "speakers")
DEFAULT_TRANSCRIPTION_FORMAT = "raw"
TRANSCRIPTION_FORMAT_HEADINGS = {
    "raw": "Transkribering",
    "timestamps": "Transkribering med tidsstämplar",
    "marked": "Transkribering med markerade misstänkta fel",
    "speakers": "Transkribering med talarangivelser",
}
# A reply in the speaker presentation: a label, a colon and the reply. The
# label can be anything the instruction asked for ("Intervjuare 1",
# "Ordförande", "Deltagare 2"), so it is recognised by its shape rather than
# by name: it starts with a capital letter, is at most 40 characters and has
# no sentence punctuation. Markdown bold around the label is tolerated.
_SPEAKER_TURN_RE = re.compile(
    r"^\*{0,2}([A-ZÅÄÖ][^:.!?\n*]{0,39}?)\*{0,2}\s*:\*{0,2}\s+(\S.*)$"
)


class DocumentMixin:
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
        known_labels = DocumentMixin._known_speaker_labels(blocks) if speaker_labels else set()
        for block in blocks:
            if not block.strip():
                continue

            if speaker_labels:
                DocumentMixin._add_speaker_block(document, block, known_labels)
                continue

            paragraph = document.add_paragraph()
            lines = block.splitlines()
            for line_index, line in enumerate(lines):
                if parse_inline:
                    DocumentMixin._add_inline_word_markup(paragraph, line.rstrip())
                else:
                    paragraph.add_run(line.rstrip())
                if line_index < len(lines) - 1:
                    paragraph.add_run().add_break()

    @staticmethod
    def _known_speaker_labels(blocks):
        """The labels to treat as speakers, or an empty set to accept any label shape.

        The list under "Identifierade talare:" names only speakers the model
        described, so a label also counts if it starts a line at least twice
        in the dialogue or ends with a number ("Deltagare 3"). A real speaker
        usually speaks more than once; a stray "Det var så:" seldom recurs.
        """
        listed, counts = set(), {}
        for block in blocks:
            lines = block.strip().splitlines()
            is_register = bool(lines) and lines[0].strip() == SPEAKER_REGISTER_SECTION_TITLE
            for line in lines[1:] if is_register else lines:
                match = _SPEAKER_TURN_RE.match(line.strip())
                if not match:
                    continue
                label = match.group(1).strip().casefold()
                if is_register:
                    listed.add(label)
                else:
                    counts[label] = counts.get(label, 0) + 1
        if not listed:
            return set()
        return listed | {
            label for label, n in counts.items() if n >= 2 or re.search(r"\d\s*$", label)
        }

    @staticmethod
    def _add_speaker_block(document, block, known_labels=frozenset()):
        """Write dialogue line by line, each reply its own paragraph with its label in bold.

        The speaker instruction asks for one reply per line without blank lines
        in between, so a block normally holds the whole dialogue. A line without
        a label continues the reply before it. The list of identified speakers
        at the end has the same "label: text" form and is written the same way,
        under its own title line.

        When the list is there, only the labels in it count, so that a line
        such as "Det var så: vi gick hem." inside a reply is not taken for a
        new speaker. Without the list, any label-shaped start counts.
        """
        current = None
        for line in block.splitlines():
            text = line.strip()
            if not text:
                continue
            match = _SPEAKER_TURN_RE.match(text)
            if match and known_labels and match.group(1).strip().casefold() not in known_labels:
                match = None
            if match:
                current = document.add_paragraph()
                label = current.add_run(match.group(1).strip() + ": ")
                label.bold = True
                DocumentMixin._add_inline_word_markup(current, match.group(2).strip())
                current.paragraph_format.space_after = Pt(4)
            elif current is not None:
                current.add_run().add_break()
                DocumentMixin._add_inline_word_markup(current, text)
            else:
                current = None
                DocumentMixin._add_inline_word_markup(document.add_paragraph(), text)

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
                DocumentMixin._add_horizontal_rule(document)
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
                    DocumentMixin._add_inline_word_markup(paragraph, remainder.strip())
                continue

            bullet_match = re.match(r"^(\s*)[-+*]\s+(.+)$", line)
            if bullet_match:
                indent, bullet_text = bullet_match.groups()
                level = min(2, max(0, len(indent.expandtabs(2)) // 2))
                style_name = "List Bullet" if level == 0 else f"List Bullet {level + 1}"
                paragraph = document.add_paragraph(style=style_name)
                DocumentMixin._add_inline_word_markup(paragraph, bullet_text.strip())
                continue

            numbered_match = re.match(r"^(\s*)\d+[.)]\s+(.+)$", line)
            if numbered_match:
                indent, numbered_text = numbered_match.groups()
                level = min(2, max(0, len(indent.expandtabs(2)) // 2))
                style_name = "List Number" if level == 0 else f"List Number {level + 1}"
                paragraph = document.add_paragraph(style=style_name)
                DocumentMixin._add_inline_word_markup(paragraph, numbered_text.strip())
                continue

            quote_match = re.match(r"^>\s?(.*)$", stripped)
            if quote_match:
                paragraph = document.add_paragraph(style="Quote")
                DocumentMixin._add_inline_word_markup(paragraph, quote_match.group(1))
                continue

            paragraph = document.add_paragraph()
            DocumentMixin._add_inline_word_markup(paragraph, stripped)

    def _model_statistics_rows(self):
        """Aggregate the recorded calls, one row per task and model."""
        rows = []
        order = []
        totals = {}
        prices = get_price_table()
        for call in self.model_calls:
            key = (call["task"], call["model"])
            if key not in totals:
                totals[key] = {"calls": 0, "sent": 0, "received": 0, "seconds": 0.0, "cost": 0.0}
                order.append(key)
            t = totals[key]
            t["calls"] += 1
            t["sent"] += call["sent"]
            t["received"] += call["received"]
            t["seconds"] += call["seconds"]
            cost = call_cost(call["model"], call["sent"], call.get("cached", 0), call["received"], prices)
            # One unpriced call makes the whole row unknown rather than too low.
            t["cost"] = None if cost is None or t["cost"] is None else t["cost"] + cost

        for key in order:
            task, model = key
            t = totals[key]
            rows.append([
                task, model, str(t["calls"]),
                f"{t['sent']:,}".replace(",", " "),
                f"{t['received']:,}".replace(",", " "),
                self._format_duration(t["seconds"]),
                format_cost(t["cost"]) if t["cost"] is not None else "-",
            ])

        if self.transcription_seconds:
            # Whisper runs locally; there is no charge to show.
            rows.insert(0, [
                "Transkribering",
                self.transcriber_model_used or "-",
                "1", "-", "-",
                self._format_duration(self.transcription_seconds),
                "-",
            ])

        if len(rows) > 1:
            rows.append([
                "Totalt", "",
                str(sum(t["calls"] for t in totals.values()) + (1 if self.transcription_seconds else 0)),
                f"{sum(t['sent'] for t in totals.values()):,}".replace(",", " "),
                f"{sum(t['received'] for t in totals.values()):,}".replace(",", " "),
                self._format_duration(
                    sum(t["seconds"] for t in totals.values()) + (self.transcription_seconds or 0)
                ),
                format_cost(total_cost) if (total_cost := self._openai_cost()[0]) is not None else "-",
            ])
        return rows

    def _openai_cost(self):
        """Total cost of the OpenAI calls and the models without a price.

        The total is None when there were no calls or any call lacks a price,
        so that a partial sum is never presented as the cost of the run.
        """
        calls = self.__dict__.get("model_calls", [])
        if not calls:
            return None, []
        prices = get_price_table()
        total, unpriced = 0.0, []
        for call in calls:
            cost = call_cost(call["model"], call["sent"], call.get("cached", 0), call["received"], prices)
            if cost is None:
                if call["model"] not in unpriced:
                    unpriced.append(call["model"])
            else:
                total += cost
        return (None if unpriced else total), unpriced

    def _marked_text_note(self):
        """Which analyses worked from the marked text, for "Om transkriberingen"."""
        used = self.__dict__.get("analyses_on_marked_text") or []
        if not used:
            return ""
        names = used[0] if len(used) == 1 else ", ".join(used[:-1]) + " och " + used[-1]
        note = (
            f"{names[0].upper() + names[1:]} bygger på texten med markerade misstänkta "
            "fel, där varje markering innehåller både den ursprungliga lydelsen och ett "
            "föreslaget alternativ."
        )
        return note

    def _cost_note(self):
        """One sentence for "Om transkriberingen" on what the OpenAI calls cost."""
        calls = self.__dict__.get("model_calls", [])
        if not calls:
            return ""
        total, unpriced = self._openai_cost()
        prices = get_price_table()
        if unpriced:
            return (
                "Kostnaden för OpenAI-anropen kunde inte uppskattas, eftersom prisfilen "
                f"saknar pris för {', '.join(unpriced)}."
            )
        checked = f"kontrollerade {prices['checked']}" if prices["checked"] else "utan angivet kontrolldatum"
        return (
            f"Uppskattad kostnad för OpenAI-anropen: {format_cost(total)} {prices['currency']}, "
            f"beräknad med listpriserna i prisfilen ({checked}). Cachad indata är räknad till "
            "cachepris. Fakturan från OpenAI kan avvika."
        )

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
        """A short note on which model and which program version produced it.

        The chosen accuracy is a ceiling, and the model actually used may be a
        smaller one if the available RAM did not allow the choice. That
        fallback is otherwise invisible, so it is recorded here: someone who
        picked a level deserves to know what they got. The program version
        lets every result document be traced to the code that produced it.
        """
        level = self.transcription_accuracy
        wanted = TRANSCRIPTION_ACCURACY_MODELS.get(level, "")
        label = TRANSCRIPTION_ACCURACY_LABELS.get(level, level)
        used = self.transcriber_model_used

        if not used:
            return ""

        line = f"Vald noggrannhet: {label}. Modell som användes: {used}."
        if self.transcription_device:
            engine = ("faster-whisper" if self.transcription_engine == ENGINE_FASTER_WHISPER
                      else "Transformers")
            line += f" Motor: {engine} på {self.transcription_device}."
        if wanted and used != wanted:
            line += (
                f" Den valda nivån motsvarar {wanted}, men tillgängligt minne "
                "räckte inte, så en mindre modell användes i stället."
            )
        if self.transcription_from_cache:
            line += " Transkriberingen återanvändes från en tidigare körning av samma ljudfil."
        parts = [line]
        marked_note = self._marked_text_note()
        if marked_note:
            parts.append(marked_note)
        cost_note = self._cost_note()
        if cost_note:
            parts.append(cost_note)
        parts.append(f"Programversion: {self._program_version()}.")
        return "\n\n".join(parts)

    @staticmethod
    def _program_version():
        """Version with commit and environment, e.g. "1.1.0 (a1b2c3d, prod)"."""
        build = get_build_info()
        details = [d for d in (build["commit_short"], build["environment"]) if d]
        return build["version"] + (f" ({', '.join(details)})" if details else "")

    def _transcription_section(self):
        """Return the single transcription section for the result document.

        The four presentations are mutually exclusive. If the chosen one is
        unavailable - marking or speaker identification failed, or the model
        returned no timestamps - the raw text is used instead and the document
        says so, rather than leaving the reader with an error message where the
        transcription should be.
        """
        chosen = self.transcription_format if self.transcription_format in TRANSCRIPTION_FORMATS \
            else DEFAULT_TRANSCRIPTION_FORMAT

        content = {
            "raw": self.transcription,
            "timestamps": self.transcription_w_timestamps,
            "marked": self.marked_text,
            "speakers": self.__dict__.get("analyze_speakers", ""),
        }.get(chosen, self.transcription)

        if not (content or "").strip() and chosen != "raw":
            logger.warning(
                f"Ingen text tillgänglig i formatet '{chosen}'. Använder rå text i stället."
            )
            note = {
                "timestamps": "[Tidsstämplar var inte tillgängliga. Rå text visas i stället.]",
                "marked": "[Markering av misstänkta fel kunde inte genomföras. Rå text visas i stället.]",
                "speakers": "[Talaridentifieringen kunde inte genomföras. Rå text visas i stället.]",
            }.get(chosen, "")
            body = (self.transcription or "").strip()
            return (
                TRANSCRIPTION_FORMAT_HEADINGS["raw"],
                (note + "\n\n" + body).strip() if note else body,
                "plain",
            )

        render_mode = {"marked": "marked", "speakers": "speakers"}.get(chosen, "plain")
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
            ("Anpassad analys", self.custom_analysis, "structured"),
            # Provenance and statistics belong after the content, not between
            # the transcription and the analyses.
            ("Om transkriberingen", self._transcription_provenance(), "plain"),
            ("Statistik",
             [STATISTICS_TABLE_HEADER + [f"Kostnad ({get_price_table()['currency']})"]] + statistics
             if statistics else "", "table"),
        ]

        temp_path = self.export_path.with_name(self.export_path.name + ".tmp")
        try:
            self.export_path.parent.mkdir(parents=True, exist_ok=True)
            document = Document()
            self._configure_word_styles(document)
            document.add_heading("Transkribering och analys", level=0)

            for heading, content, render_mode in sections:
                if not content:
                    continue
                document.add_heading(heading, level=1)
                if render_mode == "structured":
                    self._add_structured_word_content(document, content)
                elif render_mode == "marked":
                    self._add_plain_word_content(document, content, parse_inline=True)
                elif render_mode == "table":
                    self._add_word_table(document, content)
                elif render_mode == "speakers":
                    self._add_plain_word_content(document, content, parse_inline=True, speaker_labels=True)
                else:
                    self._add_plain_word_content(document, content)

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
