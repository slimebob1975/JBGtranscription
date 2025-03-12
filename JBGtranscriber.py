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
    OPENAI_API_KEY_FILE = Path(".\.openai_api_keys.json")
    MODEL_ID = "KBLab/kb-whisper-large"
    CACHE_DIR = ".cache"
    
    def __init__(self, 
                 convert_path,
                 export_path = Path("."),
                 device = "cpu",
                 model_id=MODEL_ID, 
                 openai_api_keys_file=OPENAI_API_KEY_FILE
                 ):
        self.convert_path = convert_path
        self.export_path = Path(str(export_path) + "\\" + os.path.basename(convert_path).split('/')[-1] + ".txt") 
        self.model_id = model_id
        self.openai_api_keys = JBGtranscriber.get_openai_api_keys(openai_api_keys_file)
        self.device, self.torch_dtype = JBGtranscriber.do_nvidia_check(device)
        
        self.transcription = ""
        self.transcription_w_timestamps = ""
        self.summary = ""
        self.marked_text = ""
        self.follow_up_questions = ""

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

    def call_openai(self, prompt, max_tokens=500):
        """Anropar OpenAI:s GPT-modell med en given prompt."""
        
        client = openai.OpenAI(api_key=self.openai_api_keys["OPENAI_API_KEY"], organization=self.openai_api_keys["OPENAI_ORG_KEY"])

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "developer", "content": "Du är en expert på transkriberingar av ljudfiler från exempelvis intervjuer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=max_tokens
        )

        return completion.choices[0].message

    def generate_summary(self):
        """Skapar en sammanfattning av transkriberingen."""
        prompt = f"""Sammanfatta följande transkribering på ett koncist sätt med fokus på huvudpunkterna:
        {self.transcription}"""
        
        try:
            self.summary = self.call_openai(prompt, max_tokens=300)
        except Exception as e:
            print(f"An error occurred while generating the summary: {e}")
            self.summary = "Sammanfattning var inte tillgängligt"

    def find_suspicious_phrases(self):
        """Identifierar och markerar osannolika eller grammatiskt tveksamma ordkombinationer."""
        prompt = f"""I följande transkribering, identifiera och markera tveksamma eller osannolika ordkombinationer 
        som kan bero på fel i transkriberingen. Markera dessa genom att omsluta dem med markörerna [FEL?] och [/FEL?]:
        {self.transcription}"""
        
        try:
            self.marked_text = self.call_openai(prompt, max_tokens=500)
        except Exception as e:
            print(f"An error occurred while finding suspicious phrases: {e}")
            self.marked_text = "Markering av misstänkta fel i texten misslyckades"

    def suggest_follow_up_questions(self):
        """Föreslår fem relevanta uppföljningsfrågor baserat på transkriberingen."""
        prompt = f"""Baserat på följande transkribering, generera fem möjliga uppföljningsfrågor som en intervjuare skulle kunna ställa:
        {self.transcription}"""
        
        try:
            self.follow_up_questions = self.call_openai(prompt, max_tokens=200)
        except Exception as e:
            print(f"An error occurred while generating follow-up questions: {e}")
            self.follow_up_questions = "Förslag till uppföljande frågor misslyckades"

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
        
        self.transcription, self.transcription_w_timestamps = JBGtranscriber._postprocess_result(result)

    @staticmethod
    def _postprocess_result(result):
        """Post process result of call to transcription model"""
        
        transcription = JBGtranscriber.insert_newlines(result["text"], 80)
        transcription_w_timestamps = ""
        for chunk in result["chunks"]:
            transcription_w_timestamps += str(chunk["timestamp"]) + ": " + \
                JBGtranscriber.insert_newlines(str(chunk["text"]), 80) + "\n"
            
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
                    export_file.write("### Uppföljningsfrågor:\n" + self.follow_up_questions + "\n")
        except Exception as ex:
            print(f"Something went wrong on writing transcription results to the file {self.export_path}: {str(ex)}")

def check_script_arguments():
    """Check in arguments to test script""" 
    
    if len(sys.argv) < 4:
        sys.exit("Usage: " + (sys.argv)[0] + " [path to mp3 file(s) to convert] [path to preferred transcription directory] [device=gpu/cpu]")
    else:
        convert_path = os.path.abspath(sys.argv[1])
        if not os.path.exists(convert_path) or not os.path.isdir(convert_path):
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

        # Print result to text file
        jbg_transcriber.write_to_output_file()
    
# In case of commande line execution call
if __name__=="__main__":
    
    # Execute main program
    main()
    
    # End program successfully
    sys.exit(0)