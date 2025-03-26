#!/usr/bin/env python3
try:
    import sys
    import os
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import openai
    from pathlib import Path
    import json
except ModuleNotFoundError as ex:
    sys.exit("You probably need to install some missing modules:" + str(ex))
    
class JBGtranscriber():
    
    # Standard Settings
    OPENAI_API_KEY_FILE = Path("./keys/openai_api_keys.json")
    MODEL_ID = "KBLab/kb-whisper-large"
    CACHE_DIR = ".cache"
    
    def __init__(self, 
                 convert_path,
                 export_path = Path("."),
                 device = "cpu",
                 model_id=MODEL_ID, 
                 openai_api_keys_file=OPENAI_API_KEY_FILE,
                 insert_linebreaks=False
                 ):
        self.convert_path = convert_path
        self.export_path = Path(str(export_path) + "\\" + os.path.basename(convert_path).split('/')[-1] + ".txt") 
        self.model_id = model_id
        self.openai_api_keys = JBGtranscriber.get_openai_api_keys(openai_api_keys_file)
        self.insert_linebreaks = insert_linebreaks
        self.device, self.torch_dtype = JBGtranscriber.do_nvidia_check(device)
        
        self.transcription = ""
        self.transcription_w_timestamps = ""
        self.summary = ""
        self.marked_text = ""
        self.follow_up_questions = ""
        self.analyze_speakers = ""

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
    def get_openai_api_keys(path_to_keys_file):
        
        """
        Reads the OpenAI API keys file and returns the keys as a dictionary with the following dictionary keys:
            OPENAI_API_KEY
            OPENAI_ORG_KEY
            OPENAI_GRP_KEY (optional)
        """
        
        with open(path_to_keys_file) as keys_file:
            data = keys_file.read()
        return json.loads(data)
    
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

    def call_openai(self, prompt):
        """Anropar OpenAI:s GPT-modell med en given prompt."""
        
        client = openai.OpenAI(api_key=self.openai_api_keys["OPENAI_API_KEY"], organization=self.openai_api_keys["OPENAI_ORG_KEY"])

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "developer", "content": "Du är en expert på transkriberingar av ljudfiler från exempelvis intervjuer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7
        )

        return completion.choices[0].message

    def generate_summary(self):
        """Skapar en sammanfattning av transkriberingen."""
        prompt = f"""Sammanfatta följande transkribering på ett koncist sätt med fokus på huvudpunkterna:
        {self.transcription}"""
        
        try:
            self.summary = self.call_openai(prompt).content
        except Exception as e:
            print(f"An error occurred while generating the summary: {e}")
            self.summary = "Sammanfattning var inte tillgängligt"

    def find_suspicious_phrases(self):
        """Identifierar och markerar osannolika eller grammatiskt tveksamma ordkombinationer."""
        prompt = f"""I följande transkribering, identifiera och markera tveksamma eller osannolika ordkombinationer 
        som kan bero på fel i transkriberingen. Markera misstänkta fel genom att omsluta dem med markörerna [FEL?] och [/FEL?]:
        {self.transcription}"""
        
        try:
            self.marked_text = self.call_openai(prompt).content
        except Exception as e:
            print(f"An error occurred while finding suspicious phrases: {e}")
            self.marked_text = "Markering av misstänkta fel i texten misslyckades"

    def suggest_follow_up_questions(self):
        """Föreslår fem relevanta uppföljningsfrågor baserat på transkriberingen."""
        prompt = f"""Baserat på följande transkribering, generera fem möjliga uppföljningsfrågor som en intervjuare skulle kunna ställa:
        {self.transcription}"""
        
        try:
            self.follow_up_questions = self.call_openai(prompt).content
        except Exception as e:
            print(f"An error occurred while generating follow-up questions: {e}")
            self.follow_up_questions = "Förslag till uppföljande frågor misslyckades"
            
    def do_analyze_speakers(self):
        """Analysera transkriberingen med avseende på vilka talare som säger vad"""
        prompt = f"""Du får en rå, odelad transkribering av en intervju med en eller flera intervjuare 
            och en eller flera intervjuobjekt (t.ex. handläggare på en myndighet). Transkriberingen saknar talarangivelser.

            Din uppgift är att:

            1. Identifiera antal distinkta röster i texten, både intervjuare och intervjuobjekt.
            2. Märka upp varje replik med en neutral beteckning som:
            - Intervjuare 1, Intervjuare 2, osv.
            - Intervjuobjekt 1, Intervjuobjekt 2, osv.
            3. Om en person tydligt återkommer längre fram (t.ex. fortsätter prata), använd samma beteckning som tidigare.
            4. Gör en tydlig radbrytning mellan varje replik, så att dialogen blir läsbar.

            Du får inte hitta på namn. Skriv inget utanför själva dialogen.

            Exempel på format:
            Intervjuare 1: Kan du beskriva hur processen ser ut från att ett ärende kommer in?
            Intervjuobjekt 1: Ja, först får vi ett meddelande...

            Här är transkriberingen: 
            ------------------------
            {self.transcription}
            ------------------------"""
        
        try:
            self.analyze_speakers = self.call_openai(prompt).content
        except Exception as e:
            print(f"An error occurred while analyzing speakers: {e}")
            self.analyze_speakers = "Försöket till talarsplitsning misslyckades"

    def transcribe(self):
        """Put together the model of choice and do the transcription"""
        
        # What model to use
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, use_safetensors=True, cache_dir=JBGtranscriber.CACHE_DIR
        )
        model.to(self.device)
        processor = AutoProcessor.from_pretrained(self.model_id)

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
        
        try:
            with open(self.export_path, "w") as export_file:
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
            print(f"Something went wrong on writing transcription results to the file {self.export_path}: {str(ex)}")
            
    def perform_transcription_steps(self, generate_summary=False, find_suspicious_phrases=False,\
        suggest_follow_up_questions=False, analyze_speakers=False):
        """Perform all transcription steps"""
        
        # Transcribe the audio files
        self.transcribe()
        
        # Generate a summary if requested
        if generate_summary:
            print("Generating summary")
            self.generate_summary()
            print("Summary generated successfully.")
        
        # Find and mark suspicious phrases if requested
        if find_suspicious_phrases:
            print("Finding suspicious phrases")
            self.find_suspicious_phrases()
            print("Suspicious phrases found successfully.")
        
        # Suggest follow-up questions if requested
        if suggest_follow_up_questions:
            print("Suggesting follow-up questions")
            self.suggest_follow_up_questions()
            print("Follow-up questions suggested successfully.")
            
        # Analyze speakers if requested
        if analyze_speakers:
            print("Analyzing speakers")
            self.do_analyze_speakers()
            print("Speakers analyzed successfully.")
        
        # Print a message confirming successful completion
        print("All transcription steps completed successfully.")
            
        # Write the transcription results to an output file
        self.write_to_output_file()

def check_script_arguments():
    """Check in arguments to test script""" 
    
    if len(sys.argv) < 4:
        sys.exit("Usage: " + (sys.argv)[0] + " [path to mp3 file(s) to convert] [path to preferred transcription directory] [device=gpu/cpu]")
    else:
        convert_path = os.path.abspath(sys.argv[1])
        if not (os.path.exists(convert_path) or os.path.isdir(convert_path)):
            sys.exit("{0} is not a valid file or directory path".format(convert_path))
        print("Convert file(s) path: ", convert_path)
        
        export_path = os.path.abspath(sys.argv[2])
        if not os.path.isdir(export_path):
            sys.exit("{0} is not a valid directory path".format(export_path))
        print("Export file path: ", export_path)    
        
        device = str(sys.argv[3])
        if device.lower() not in ['gpu', 'cpu']:
            sys.exit("Device must be set to 'gpu' or 'cpu', got '{0}'").format(device)
        
    return Path(convert_path), Path(export_path), device

def main():
    """
    The main program of this little script
    """
    
    # We need correct arguments passed to the script
    convert_path, export_path, device = check_script_arguments()
    
    # Generate a list of files to transcribe
    if os.path.isdir(convert_path):
        convert_files = JBGtranscriber.find_mp3_files(convert_path)
    elif os.path.exists(convert_path):
        convert_files = [convert_path]
    else:
        sys.exit("Could not find any files to transcribe at {0}".format(convert_path))
    
    # Perform transcription
    for convert_file in convert_files:
        
        print("Processing {0}".format(convert_file))
        
        # Create the transcription class
        jbg_transcriber = JBGtranscriber(convert_file, export_path, device)
        
        # Make transcription
        jbg_transcriber.transcribe()
        
        # Make calls to OpenAI API to generate extra information
        # Summary
        jbg_transcriber.generate_summary()

        # Mark suspicious phrases
        jbg_transcriber.find_suspicious_phrases()

        # Generate follow up questions
        jbg_transcriber.suggest_follow_up_questions()
        
         # Generate speaker analysis
        jbg_transcriber.analyze_speakers()

        # Print result to text file
        jbg_transcriber.write_to_output_file()
    
# In case of commande line execution call
if __name__=="__main__":
    
    # Execute main program
    main()
    
    # End program successfully
    sys.exit(0)