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
except ModuleNotFoundError as ex:
    sys.exit("You probably need to install some missing modules:" + str(ex))
from src.JBGLogger import JBGLogger
    
# Default markers
CACHE_TRANSCRIPTION_MARKER = "===TRANSCRIPTION==="
CACHE_TIMESTAMPED_MARKER = "===TIMESTAMPED==="
MODEL_GPT_5_MARKER = "gpt-5"

 # Max tokens per segment (input + output < 8000 tokens)
MAX_INPUT_TOKENS = 3500
OVERLAP_TOKENS = 300

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
        self.prompt_policy = self.load_prompt_policy()
        self.summary = ""
        self.marked_text = ""
        self.follow_up_questions = ""
        self.analyze_speakers = ""

    def load_prompt_policy(self):
        try:
            with open("policy/prompt_policy.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f" Could not load prompt policy: {e}")
            return {}
    
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
            return tiktoken.get_encoding("o200k_base")

    def get_transcription_cache_path(self):
        """Generate a cache filename based on the audio file's full path (hashed)."""
        with open(self.convert_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        cache_file = Path("cache") / f"{file_hash}.txt"
        return cache_file
    
    def call_openai_simple(self, prompt):
        """Anropar OpenAI:s GPT-modell med en given prompt."""
        # TODO: Consider adding reasoning_effort='none' for gpt-5.1 to get GPT-5.1-level intelligence with ultra-low latency.
        
        client = openai.OpenAI(api_key=self.api_key)

        extra_args = JBGtranscriber.get_model_specific_extra_arguments(self.openai_model)

        completion = client.chat.completions.create(
            model= self.openai_model,
            messages=[
                {"role": "developer", "content": "Du är en expert på transkriberingar av ljudfiler från exempelvis intervjuer."},
                {"role": "user", "content": prompt},
            ],
            temperature = JBGtranscriber.get_permitted_temperature(self.openai_model, DEFAULT_TEMPERATURE),
            **extra_args,
        )

        return completion.choices[0].message
    
    def call_openai(self, instructions, input_message):
        """Anropar OpenAI:s GPT-modell med en given prompt."""
        # TODO: Consider adding reasoning_effort='none' for gpt-5.1 to get GPT-5.1-level intelligence with ultra-low latency.
        
        client = openai.OpenAI(api_key=self.api_key)

        extra_args = JBGtranscriber.get_model_specific_extra_arguments(self.openai_model)
        logger.info(f"[INFO] Extra args for openai model {self.openai_model}: {str(extra_args)}")

        completion = client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": input_message},
            ],
            temperature = JBGtranscriber.get_permitted_temperature(self.openai_model, DEFAULT_TEMPERATURE),
             **extra_args,
        )

        return completion.choices[0].message

    def generate_summary(self, style="short"):
        
        """Genererar en sammanfattning baserat på vald stil ('short' eller 'extensive')."""
        if style == "extensive":
            self.generate_summary_splitted()
        else:
            system_message = self.prompt_policy.get("short_summary", "Sammanfatta detta:")
            try:
                self.summary = self.call_openai(
                    instructions=system_message,
                    input_message=self.transcription
                ).content
            except Exception as e:
                logger.error(f" Summary generation failed: {e}")
                self.summary = "Sammanfattning var inte tillgänglig"
                
    def generate_summary_splitted(self):

        logger.info(f"Starting segmented summary generation...")

        enc = self._get_encoder_for_segmentation()
        logger.debug(f"Investigated tiktoken encoding...")

        segments = self._split_into_segments(self.transcription, enc)
        logger.debug(f"Split text into segments...")

        prompt_lines = self.prompt_policy.get("extensive_summary", ["Sammanfatta detta:"])

        instructions = "\n".join(prompt_lines)
        logger.debug(f"Prompt instructions=",instructions)

        partial_summaries = []

        try:
            for i, segment in enumerate(segments):
                if i > 0:
                    time.sleep(5)  # prevent hitting rate limits
                logger.info(f" Summarizing segment {i+1}/{len(segments)}...")
                context_note = f"(Detta är del {i+1} av transkriptionen. Sammanfatta detta avsnitt noggrant.)"
                result = self.call_openai(instructions=instructions, input_message=context_note + "\n\n" + segment)
                partial_summaries.append(f"### Sammanfattning av del {i+1}:\n{result.content.strip()}")
        except Exception as e:
            logger.error(f" Failed to summarize segment: {e}")
            self.summary = "Sammanfattning misslyckades"
            return

        # Step 1: Seamless compilation
        compiled_summary = "\n\n".join(partial_summaries)

        # Step 2: Generate short wrap-up summary
        logger.info(f"[INFO] Generating short summary as concluding wrap-up...")
        try:
            short_prompt = "\n".join(self.prompt_policy.get("short_summary", ["Sammanfatta detta:"]))
            final_wrapup = self.call_openai(
                instructions=short_prompt,
                input_message=compiled_summary
            ).content.strip()
        except Exception as e:
            logger.warning(f" Short summary step failed: {e}")
            final_wrapup = "Kort sammanfattning kunde inte genereras."

        # Final result: seamless segments + wrap-up
        self.summary = compiled_summary + "\n\n### Kort sammanfattning:\n" + final_wrapup

    def find_suspicious_phrases(self):
        """Identifierar och markerar osannolika eller grammatiskt tveksamma ordkombinationer."""
        try:
            prompt = "\n".join(self.prompt_policy.get("suspicious_phrases", [])) + "\n" + self.transcription
            self.marked_text = self.call_openai_simple(prompt).content
        except Exception as e:
            logger.error(f"An error occurred while finding suspicious phrases: {e}")
            self.marked_text = "Markering av misstänkta fel i texten misslyckades"

    def suggest_follow_up_questions(self):
        """Föreslår fem relevanta uppföljningsfrågor baserat på transkriberingen."""
        try:
            prompt = self.prompt_policy.get("follow_up_questions", "Generera uppföljningsfrågor:") + "\n" + self.transcription
            self.follow_up_questions = self.call_openai_simple(prompt).content
        except Exception as e:
            logger.error(f"An error occurred while generating follow-up questions: {e}")
            self.follow_up_questions = "Förslag till uppföljande frågor misslyckades"
            
    def do_analyze_speakers(self):
        """Analysera transkriberingen med avseende på vilka talare som säger vad"""
        
        prompt = "\n".join(self.prompt_policy.get("speaker_diarization", [
            "Försök att identifiera olika röster i följande transkribering:"
        ]))
        
        input_message = f"""
            ------------------------
            {self.transcription}
            ------------------------
        """
        
        try:
            self.analyze_speakers = self.call_openai(instructions=prompt, input_message=input_message).content
        except Exception as e:
            logger.error(f"An error occurred while analyzing speakers: {e}")
            self.analyze_speakers = "Försöket till identifiering av talare misslyckades"

    def do_analyze_speakers_splitted(self):
        
        # Setup encoder for model
        enc = self._get_encoder_for_segmentation()
        
        segments = self._split_into_segments(self.transcription, enc)
        
        # No need to split text into segments
        if len(segments) == 1:
            self.do_analyze_speakers()
        else:
            # Text resulted in multiple segments
            diarized_segments = []
            prompt = "\n".join(self.prompt_policy.get("speaker_diarization", [
                "Försök att identifiera olika röster i följande transkribering:"
            ]))

            try:
                for i, segment in enumerate(segments):
                    if i>0: 
                        time.sleep(5)  # undvik rate limit for multiple API calls
                    logger.info(f"Bearbetar segment {i+1}/{len(segments)}...")
                    context = f"\nDetta är del {i+1}. Fortsätt numrera talare konsekvent.\n"
                    diarized = self.call_openai(instructions=prompt+context, input_message=segment).content
                    diarized_segments.append(diarized)
                self.analyze_speakers = ("\n\n".join(diarized_segments))
            except Exception as e:
                logger.error(f"An error occurred while analyzing speakers: {e}")
                self.analyze_speakers = "Försöket till identifiering av talare misslyckades"        
    
    # Some internal help functions
    def _tokenize(self, text, enc):
        return enc.encode(text)

    def _detokenize(self, tokens, enc):
        return enc.decode(tokens)
    
    def _split_into_segments(self, text, enc, max_tokens=MAX_INPUT_TOKENS, overlap=OVERLAP_TOKENS):
        tokens = self._tokenize(text, enc)
        segments = []
        i = 0
        while i < len(tokens):
            segment = tokens[i:i + max_tokens]
            segments.append(self._detokenize(segment, enc))
            i += max_tokens - overlap
        logger.info(f"Text has {len(tokens)} tokens and results in {len(segments)} segments")
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
    def _enough_memory(min_gb_required: float = 6.0, safety_margin_gb: float = 0.5) -> bool:
        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024 ** 3)
        available_gb = vm.available / (1024 ** 3)
        logger.debug(
            f"Memory check: required={min_gb_required:.2f} GB, "
            f"available={available_gb:.2f} GB, total={total_gb:.2f} GB"
        )
        return total_gb - min_gb_required >= safety_margin_gb

    
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

        for model_id in self.TRANSCRIBER_MODEL_CANDIDATES:
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
                result = pipe(
                    audio_input,
                    #chunk_length_s=30,
                    generate_kwargs=generate_kwargs,
                    return_timestamps=True
                )

                self.transcription, self.transcription_w_timestamps = self._postprocess_result(result)
                logger.info(f" Transcription successful with model: {model_id}")

                
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

    def write_to_output_file(self):
        """Write final result to the output file"""
        
        if self.secure_handler:
            logger.info("Sparar transkription krypterat med SecureFileHandler...")
            full_text = ""
            full_text += "### Rå transkribering:\n" + self.transcription + "\n\n"
            full_text += "### Transkribering med tidsstämplar:\n" + self.transcription_w_timestamps + "\n\n"
            if self.summary:
                full_text += "### Sammanfattning:\n" + self.summary + "\n\n"
            if self.marked_text:
                full_text += "### Transkription med markerade misstänkta fraser:\n" + self.marked_text + "\n\n"
            if self.follow_up_questions:
                full_text += "### Uppföljningsfrågor:\n" + self.follow_up_questions + "\n\n"
            if self.analyze_speakers:
                full_text += "### Försök till identifiering av olika talare:\n" + self.analyze_speakers + "\n"

            try:
                self.secure_handler.encrypt_text(full_text, str(self.export_path))
            except Exception as ex:
                logger.error(f"Kryptering av transkriptionsfil misslyckades: {str(ex)}")
        else:
            try:
                with open(self.export_path, "w", encoding="utf-8") as export_file:
                    export_file.write("### Rå transkribering:\n" + self.transcription + "\n\n")
                    export_file.write("### Transkribering med tidsstämplar:\n" + self.transcription_w_timestamps + "\n\n")
                    if self.summary:
                        export_file.write("### Sammanfattning:\n" + self.summary + "\n\n")
                    if self.marked_text:
                        export_file.write("### Transkription med markerade misstänkta fraser:\n" + self.marked_text + "\n\n")
                    if self.follow_up_questions:
                        export_file.write("### Uppföljningsfrågor:\n" + self.follow_up_questions + "\n\n")
                    if self.analyze_speakers:
                        export_file.write("### Försök till identifiering av olika talare:\n" + self.analyze_speakers + "\n")
            except Exception as ex:
                logger.error(f"Misslyckades med att spara transkriptionsfil: {str(ex)}")
            
    def perform_transcription_steps(
        self,
        generate_summary=False,
        summary_style="short",
        find_suspicious_phrases=False,
        suggest_follow_up_questions=False,
        analyze_speakers=False,
        progress_callback=None
    ):
        """Perform all transcription steps"""
        
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
        
        # Generate a summary if requested
        if generate_summary:
            report(f"Genererar sammanfattning...")
            self.generate_summary(style=summary_style)
        
        # Find and mark suspicious phrases if requested
        if find_suspicious_phrases:
            report(f"Letar misstänkta fel...")
            self.find_suspicious_phrases()
        
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
        sys.exit("Usage: " + sys.argv[0] + " [path to .mp3 file or folder] [output folder] [device=gpu/cpu] [openai_api_key] (optional: model)")

    convert_path = Path(sys.argv[1])
    export_path = Path(sys.argv[2])
    device = sys.argv[3].lower()
    api_key = sys.argv[4]
    model = sys.argv[5] if len(sys.argv) > 5 else "gpt-4o"
    summary_style = sys.argv[6] if len(sys.argv) > 6 else "short"

    if not convert_path.exists():
        sys.exit(f"{convert_path} is not a valid file or directory path")
    if not export_path.is_dir():
        sys.exit(f"{export_path} is not a valid directory path")
    if device not in ["cpu", "gpu"]:
        sys.exit("Device must be 'cpu' or 'gpu'")
    if summary_style not in ["short", "extensive"]:
        sys.exit("Summary style must be 'short' or 'extensive'")

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

        # Instantiate transcriber with all required params
        transcriber = JBGtranscriber(
            convert_path=convert_file,
            export_path=export_path,
            device=device,
            api_key=api_key,
            openai_model=model
        )

        # Run all steps
        transcriber.perform_transcription_steps(
            generate_summary=True,
            summary_style=summary_style,
            find_suspicious_phrases=True,
            suggest_follow_up_questions=True,
            analyze_speakers=True
        )

    
# In case of commande line execution call
if __name__=="__main__":
    
    # Execute main program
    main()
    
    # End program successfully
    sys.exit(0)