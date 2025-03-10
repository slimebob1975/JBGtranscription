#!/usr/bin/env python3
try:
    import sys
    import os
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import openai
    from pathlib import Path
    import json
except ModuleNotFoundError as ex:
    sys.exit("You probably need to install some missing modules:" + str(ex))
    
# Settings
OPENAI_API_KEY_FILE = Path(".\.openai_api_keys.json")
    
# Set model ID
MODEL_ID = "KBLab/kb-whisper-large"

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

def call_openai(prompt, max_tokens=500):
    """Anropar OpenAI:s GPT-modell med en given prompt."""
    
    keys = get_openai_api_keys(OPENAI_API_KEY_FILE)
    
    client = openai.OpenAI(api_key=keys["OPENAI_API_KEY"], organization=keys["OPENAI_ORG_KEY"])

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

def generate_summary(transcription_text):
    """Skapar en sammanfattning av transkriberingen."""
    prompt = f"""Sammanfatta följande transkribering på ett koncist sätt med fokus på huvudpunkterna:
    {transcription_text}"""
    
    return call_openai(prompt, max_tokens=300)

def find_suspicious_phrases(transcription_text):
    """Identifierar och markerar osannolika eller grammatiskt tveksamma ordkombinationer."""
    prompt = f"""I följande transkribering, identifiera och markera tveksamma eller osannolika ordkombinationer 
    som kan bero på fel i transkriberingen. Markera dessa genom att omsluta dem med markörerna [FEL?] och [/FEL?]:
    {transcription_text}"""
    
    return call_openai(prompt, max_tokens=500)

def suggest_follow_up_questions(transcription_text):
    """Föreslår fem relevanta uppföljningsfrågor baserat på transkriberingen."""
    prompt = f"""Baserat på följande transkribering, generera fem möjliga uppföljningsfrågor som en intervjuare skulle kunna ställa:
    {transcription_text}"""
    
    return call_openai(prompt, max_tokens=200)

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

def check_script_arguments():
    """Check in arguments to script""" 
    
    if len(sys.argv) < 2:
        sys.exit("Usage: " + (sys.argv)[0] + " [path to mp3 file to convert] [path to preferred transcription directory]")
    else:
        convert_path = os.path.abspath(sys.argv[1])
        if not os.path.exists(convert_path):
            sys.exit("{0} is not a valid file path".format(convert_path))
        print("Convert file path: ", convert_path)
        
        export_path = os.path.abspath(sys.argv[2])
        if not os.path.isdir(export_path):
            sys.exit("{0} is not a valid directory path".format(export_path))
        export_path = export_path + "\\" + os.path.basename(convert_path).split('/')[-1] + ".txt" 
        print("Export file path: ", export_path)    
        
    return Path(convert_path), Path(export_path)
        
def do_nvidia_check():
    """"Check if we have GPU support or not and set data type accordingly """
    
    # Check if NVIDIA is supported
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Set datatype and models
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    return device, torch_dtype

def get_model_and_transcribe(convert_path):
    """Put together the model of choice and do the transcription"""
    
    # We hope there is a GPU
    device, torch_dtype = do_nvidia_check()
    
    # What model to use
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, torch_dtype=torch_dtype, use_safetensors=True, cache_dir="cache"
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Define pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    generate_kwargs = {"task": "transcribe", "language": "sv"}

    # Do transcription
    result = pipe(str(convert_path), 
            chunk_length_s=30,
            generate_kwargs=generate_kwargs, 
            return_timestamps=True)
    
    return result

def postprocess_result(result):
    """Post process result of call to transcription model"""
    
    transcription = insert_newlines(result["text"], 80)
    transcript_w_timestamps = ""
    for chunk in result["chunks"]:
        transcript_w_timestamps += str(chunk["timestamp"]) + ": " + insert_newlines(str(chunk["text"]), 80) + "\n"
        
    return transcription, transcript_w_timestamps

def write_to_output_file(export_path, transcription, transcript_w_timestamps, summary, \
        marked_text, follow_up_questions):
    """Write final result to the output file"""
    
    try:
        with open(export_path, "w") as export_file:
            export_file.write("### Rå transkribering:\n" + transcription + "\n\n")
            export_file.write("### Transkribering med tidsstämplar:\n" + transcript_w_timestamps + "\n\n")
            export_file.write("### Sammanfattning:\n" + summary + "\n\n")
            export_file.write("### Transkription med markerade misstänkta fraser:\n" + marked_text + "\n\n")
            export_file.write("### Uppföljningsfrågor:\n" + follow_up_questions + "\n")
    except Exception as ex:
        print(f"Something went wrong on writing to the file {export_path}: {str(ex)}")

def main():
    """
    The main program of this little script
    """
    
    # We need correct arguments passed to the script
    convert_path, export_path = check_script_arguments()
    
    # Get the model and performed transcription
    result = get_model_and_transcribe(convert_path)
    
    # Postprocess transcription result which is a dictionary
    transcription, transcript_w_timestamps = postprocess_result(result)
        
    # Make calls to OpenAI API to generate extra information
    # Summary
    summary = generate_summary(transcription)

    # Mark suspicious phrases
    marked_text = find_suspicious_phrases(transcription)

    # Generate follow up questions
    follow_up_questions = suggest_follow_up_questions(transcription)

    # Print result to text file
    write_to_output_file(export_path, transcription, transcript_w_timestamps, summary, \
        marked_text, follow_up_questions)

    # End program successfully
    print("Script was successfully completed. Bye!")
    sys.exit(0)
    
if __name__=="__main__":
    main()