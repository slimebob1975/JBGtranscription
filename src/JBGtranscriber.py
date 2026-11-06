#!/usr/bin/env python3
"""
JBGtranscriber: transcription with KB-Whisper and analysis with OpenAI models.

The class is assembled from mixins, one module per responsibility:

  JBGPromptPolicy    PromptPolicyMixin
  JBGLanguageModel   LanguageModelMixin
  JBGSegmentation    SegmentationMixin
  JBGSpeechToText    SpeechToTextMixin
  JBGAnalyses        AnalysesMixin
  JBGSpeakers        SpeakersMixin
  JBGDocument        DocumentMixin

This module keeps the constructor, the step runner and the command-line
tool, and re-exports the module-level names that used to be defined here,
so that `src.JBGtranscriber.<name>` keeps working for main.py and others.
"""

try:
    import sys
    from pathlib import Path
    from src.JBGPromptPolicy import (
        DEFAULT_CUSTOM_ANALYSIS_PREAMBLE,
        FALLBACK_SUMMARY_OPTION_ID,
        FALLBACK_SUMMARY_PROMPT,
        LEGACY_SUMMARY_OPTION_IDS,
        MAX_SUMMARY_PROMPT_CHARS,
        PromptPolicyMixin,
    )
    from src.JBGLanguageModel import (
        DEFAULT_MAX_OUTPUT_TOKENS,
        DEFAULT_TEMPERATURE,
        EXTRA_MODEL_OPTIONS,
        GPT_5_TEMPERATURE,
        LanguageModelMixin,
        MODEL_GPT_5_MARKER,
        MODEL_OUTPUT_TOKEN_LIMITS,
        _CONTEXT_ERROR_MARKERS,
        _OUTPUT_TOKEN_PARAM_CACHE,
        _UNSUPPORTED_PARAM_MARKERS,
    )
    from src.JBGSegmentation import (
        ApproximateEncoder,
        DEFAULT_MAX_INPUT_TOKENS,
        DEFAULT_REWRITE_SEGMENT_TOKENS,
        MAX_INPUT_TOKENS,
        MIN_MAX_INPUT_TOKENS,
        MODEL_INPUT_TOKEN_BUDGETS,
        RESERVED_RESPONSE_TOKENS,
        REWRITE_OUTPUT_HEADROOM,
        SEGMENT_OVERLAP_SENTENCES,
        SegmentationMixin,
        _PARAGRAPH_BOUNDARY_RE,
        _SENTENCE_BOUNDARY_RE,
    )
    from src.JBGSpeechToText import (
        CACHE_MODEL_MARKER,
        CACHE_TIMESTAMPED_MARKER,
        CACHE_TRANSCRIPTION_MARKER,
        DEFAULT_TRANSCRIPTION_ACCURACY,
        DEFAULT_TRANSCRIPTION_ENGINE,
        ENGINE_FASTER_WHISPER,
        ENGINE_TRANSFORMERS,
        FASTER_WHISPER_COMPUTE_TYPE_CPU,
        FASTER_WHISPER_COMPUTE_TYPE_GPU,
        FASTER_WHISPER_RAM_DIVISOR,
        JBG_BATCH_SIZE_ENV,
        JBG_CHUNK_LENGTH_S_ENV,
        JBG_DEVICE_ENV,
        RESAMPLING_TARGET_RATE,
        SUPPORTED_AUDIO_EXTENSIONS,
        SpeechToTextMixin,
        TRANSCRIPTION_ACCURACY_LABELS,
        TRANSCRIPTION_ACCURACY_LEVELS,
        TRANSCRIPTION_ACCURACY_MODELS,
        TRANSCRIPTION_CACHE_DIR,
    )
    from src.JBGAnalyses import (
        AnalysesMixin,
        MAP_STAGE_SUFFIX,
        MARKED_TEXT_NOTE,
        MAX_BUDGET_DOWNSHIFTS,
        MAX_REWRITE_RETRIES,
        REDUCE_STAGE_SUFFIX,
        REWRITE_MIN_WORD_RATIO_FAIL,
        REWRITE_MIN_WORD_RATIO_WARN,
        REWRITE_RETRY_DIVISOR,
        SEGMENT_PAUSE_SECONDS,
    )
    from src.JBGSpeakers import (
        DEFAULT_DIARIZATION_SEGMENT_TOKENS,
        SPEAKER_CONTEXT_PREAMBLE,
        SPEAKER_CONTEXT_SENTENCES,
        SPEAKER_FORMAT_SUFFIX,
        SPEAKER_REGISTER_HEADING,
        SPEAKER_REGISTER_PREAMBLE,
        SPEAKER_REGISTER_SECTION_TITLE,
        SPEAKER_REGISTER_SUFFIX,
        SpeakersMixin,
        _SPEAKER_LINE_RE,
        _SPEAKER_REGISTER_LINE_RE,
    )
    from src.JBGDocument import (
        DEFAULT_TRANSCRIPTION_FORMAT,
        DocumentMixin,
        STATISTICS_TABLE_HEADER,
        TRANSCRIPTION_FORMATS,
        TRANSCRIPTION_FORMAT_HEADINGS,
        _SPEAKER_TURN_RE,
    )
except ModuleNotFoundError as ex:
    sys.exit("You probably need to install some missing modules:" + str(ex))
from src.JBGLogger import JBGLogger

logger = JBGLogger(level="INFO").logger

# Names this module provides, including those it re-exports from the mixins.
__all__ = [
    "AnalysesMixin",
    "ApproximateEncoder",
    "CACHE_MODEL_MARKER",
    "CACHE_TIMESTAMPED_MARKER",
    "CACHE_TRANSCRIPTION_MARKER",
    "DEFAULT_CUSTOM_ANALYSIS_PREAMBLE",
    "DEFAULT_DIARIZATION_SEGMENT_TOKENS",
    "DEFAULT_MAX_INPUT_TOKENS",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_REWRITE_SEGMENT_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TRANSCRIPTION_ACCURACY",
    "DEFAULT_TRANSCRIPTION_ENGINE",
    "DEFAULT_TRANSCRIPTION_FORMAT",
    "DocumentMixin",
    "ENGINE_FASTER_WHISPER",
    "ENGINE_TRANSFORMERS",
    "EXTRA_MODEL_OPTIONS",
    "FALLBACK_SUMMARY_OPTION_ID",
    "FALLBACK_SUMMARY_PROMPT",
    "FASTER_WHISPER_COMPUTE_TYPE_CPU",
    "FASTER_WHISPER_COMPUTE_TYPE_GPU",
    "FASTER_WHISPER_RAM_DIVISOR",
    "GPT_5_TEMPERATURE",
    "JBG_BATCH_SIZE_ENV",
    "JBG_CHUNK_LENGTH_S_ENV",
    "JBG_DEVICE_ENV",
    "LEGACY_SUMMARY_OPTION_IDS",
    "LanguageModelMixin",
    "MAP_STAGE_SUFFIX",
    "MARKED_TEXT_NOTE",
    "MAX_BUDGET_DOWNSHIFTS",
    "MAX_INPUT_TOKENS",
    "MAX_REWRITE_RETRIES",
    "MAX_SUMMARY_PROMPT_CHARS",
    "MIN_MAX_INPUT_TOKENS",
    "MODEL_GPT_5_MARKER",
    "MODEL_INPUT_TOKEN_BUDGETS",
    "MODEL_OUTPUT_TOKEN_LIMITS",
    "PromptPolicyMixin",
    "REDUCE_STAGE_SUFFIX",
    "RESAMPLING_TARGET_RATE",
    "RESERVED_RESPONSE_TOKENS",
    "REWRITE_MIN_WORD_RATIO_FAIL",
    "REWRITE_MIN_WORD_RATIO_WARN",
    "REWRITE_OUTPUT_HEADROOM",
    "REWRITE_RETRY_DIVISOR",
    "SEGMENT_OVERLAP_SENTENCES",
    "SEGMENT_PAUSE_SECONDS",
    "SPEAKER_CONTEXT_PREAMBLE",
    "SPEAKER_CONTEXT_SENTENCES",
    "SPEAKER_FORMAT_SUFFIX",
    "SPEAKER_REGISTER_HEADING",
    "SPEAKER_REGISTER_PREAMBLE",
    "SPEAKER_REGISTER_SECTION_TITLE",
    "SPEAKER_REGISTER_SUFFIX",
    "STATISTICS_TABLE_HEADER",
    "SUPPORTED_AUDIO_EXTENSIONS",
    "SegmentationMixin",
    "SpeakersMixin",
    "SpeechToTextMixin",
    "TRANSCRIPTION_ACCURACY_LABELS",
    "TRANSCRIPTION_ACCURACY_LEVELS",
    "TRANSCRIPTION_ACCURACY_MODELS",
    "TRANSCRIPTION_CACHE_DIR",
    "TRANSCRIPTION_FORMATS",
    "TRANSCRIPTION_FORMAT_HEADINGS",
    "_CONTEXT_ERROR_MARKERS",
    "_OUTPUT_TOKEN_PARAM_CACHE",
    "_PARAGRAPH_BOUNDARY_RE",
    "_SENTENCE_BOUNDARY_RE",
    "_SPEAKER_LINE_RE",
    "_SPEAKER_REGISTER_LINE_RE",
    "_SPEAKER_TURN_RE",
    "_UNSUPPORTED_PARAM_MARKERS",
    "JBGtranscriber",
]


class JBGtranscriber(
    PromptPolicyMixin,
    LanguageModelMixin,
    SegmentationMixin,
    SpeechToTextMixin,
    AnalysesMixin,
    SpeakersMixin,
    DocumentMixin,
):
    def __init__(self, 
                 convert_path,
                 export_path = Path("."),
                 device = "cpu",
                 api_key = None,
                 openai_model = "gpt-4o",
                 secure_handler = None,
                 transcriber_model_id=SpeechToTextMixin.TRANSCRIBER_MODEL_DEFAULT,
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
        self.device, self.torch_dtype = self.do_nvidia_check(device)
        
        self.transcription = ""
        self.transcription_w_timestamps = ""
        self.transcription_format = DEFAULT_TRANSCRIPTION_FORMAT
        self.speaker_register = {}
        self.transcription_accuracy = DEFAULT_TRANSCRIPTION_ACCURACY
        self.transcription_engine = DEFAULT_TRANSCRIPTION_ENGINE
        self.transcription_device = ""
        # Statistics only: counts and timings, never any text.
        self.model_calls = []
        self.transcription_seconds = 0.0
        self.current_task = ""
        # Filled in once a model has actually loaded and transcribed.
        self.transcriber_model_used = ""
        # True when the transcription was taken from the cache.
        self.transcription_from_cache = False
        self.prompt_policy = self.load_prompt_policy()
        self.summary = ""
        self.marked_text = ""
        self.follow_up_questions = ""
        self.custom_analysis = ""
        # Analyses that worked from the marked text, for "Om transkriberingen".
        self.analyses_on_marked_text = []
        self.analyze_speakers = ""

    def perform_transcription_steps(
        self,
        generate_summary=False,
        summary_style=None,
        summary_prompt=None,
        transcription_format=DEFAULT_TRANSCRIPTION_FORMAT,
        transcription_accuracy=DEFAULT_TRANSCRIPTION_ACCURACY,
        transcription_engine=DEFAULT_TRANSCRIPTION_ENGINE,
        find_suspicious_phrases=None,
        suggest_follow_up_questions=False,
        analyze_speakers=None,
        progress_callback=None,
        follow_up_prompt=None,
        speaker_prompt=None,
        custom_analysis_prompt=None,
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
        self.transcription_engine = (
            transcription_engine if transcription_engine in
            (ENGINE_FASTER_WHISPER, ENGINE_TRANSFORMERS)
            else DEFAULT_TRANSCRIPTION_ENGINE
        )
        self.transcription_format = (
            transcription_format if transcription_format in TRANSCRIPTION_FORMATS
            else DEFAULT_TRANSCRIPTION_FORMAT
        )
        if find_suspicious_phrases is None:
            find_suspicious_phrases = self.transcription_format == "marked"
        # Speaker identification is the work behind the "speakers" presentation.
        if analyze_speakers is None:
            analyze_speakers = self.transcription_format == "speakers"
        
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

        # Speaker identification likewise produces the transcription section
        if analyze_speakers:
            report(f"Försöker identifiera olika talare...")
            self.do_analyze_speakers_splitted(custom_prompt=speaker_prompt)
        
        # Generate a summary if requested
        if generate_summary:
            report(f"Genererar sammanfattning...")
            self.generate_summary(style=summary_style, custom_prompt=summary_prompt)
        
        # Suggest follow-up questions if requested
        if suggest_follow_up_questions:
            report(f"Föreslår uppföljningsfrågor...")
            self.suggest_follow_up_questions(custom_prompt=follow_up_prompt)
            

        # The user's own analysis, last as in the GUI
        if (custom_analysis_prompt or "").strip():
            report("Gör den anpassade analysen...")
            self.run_custom_analysis(custom_analysis_prompt)
        
        # Print a message confirming successful completion
        report(f"Alla steg i analysen genomfördes.")
            
        # Write the transcription results to an output file
        report(f"Spara resultatet till fil...")
        self.write_to_output_file()
        report(f"Transkribering avslutad.")


def check_script_arguments():
    """Check command-line arguments for the test script""" 
    if len(sys.argv) < 6:
        sys.exit("Usage: " + sys.argv[0] + " [path to audio file or folder] [output folder] [device=gpu/cpu] [openai_api_key] (optional: model) (optional: summary style)")

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
        convert_files = JBGtranscriber.find_audio_files(convert_path)
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
            # Speaker identification is a presentation of its own now, so it
            # cannot be added to the marked presentation here.
            transcription_format="marked",
            suggest_follow_up_questions=True,
        )

    
# In case of commande line execution call
if __name__=="__main__":
    
    # Execute main program
    main()
    
    # End program successfully
    sys.exit(0)