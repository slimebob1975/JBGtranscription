from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import sys
import shutil
import uuid
import src.JBGtranscriber as JBGtranscriber
from pathlib import Path
import torch

from fastapi import FastAPI

try:
    app = FastAPI()
except Exception as e:
    print("FastAPI ERROR:", e, file=sys.stderr)
    raise

UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
OPENAI_API_KEYS_FILE = Path.cwd() / Path("./src/keys/openai_api_keys.json")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.chmod(UPLOAD_FOLDER, 0o777)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.chmod(RESULTS_FOLDER, 0o777)

# Function to transcribe in the background
def transcribe_audio(file_path: str, result_path: str, device: str, api_key:str, openai_model: str, summarize: bool, \
    summary_style: str, suspicious: bool, questions: bool, speakers: bool):
    
    transcriber = JBGtranscriber.JBGtranscriber(Path(file_path), Path(result_path), device=device, \
        api_key=api_key, openai_model = openai_model)
    
    try:
       transcriber.perform_transcription_steps(
        generate_summary=summarize,
        summary_style=summary_style,
        find_suspicious_phrases=suspicious,
        suggest_follow_up_questions=questions,
        analyze_speakers=speakers
    )
    except Exception as e:
        return JSONResponse({"Transcription error": str(e)}, status_code=500)
    else:
        clean_up_files(Path(file_path), Path(result_path)) 
        return JSONResponse({"message": "Transcription completed successfully. Audio file deleted."}, status_code=200)
    
def clean_up_files(audio_file_path: Path, transcriptions_path: Path):
    
    # Remove last audio file
    Path.unlink(audio_file_path)
    
    # Remove all but the last transcription file (which is the transcription of the audio file)
    old_transcripts = sorted([file for file in transcriptions_path.glob("*.mp3.txt")], key=lambda x: x.stat().st_ctime)[:-1]
    for file in old_transcripts:
        Path.unlink(file)

# Entry point for uploading audio files
@app.post("/upload/")
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    api_key: str = Form(...),
    model: str = Form("gpt-4o"),
    summarize: bool = Form(False),
    summary_style: str = Form("short"),
    suspicious: bool = Form(False),
    questions: bool = Form(False),
    speakers: bool = Form(False)
):
    
    print(f"""
          OpenAI API key was provided: {api_key != "sk-..."}\n
          OpenAI model of choice: {model}\n
          OpenAI API tasks: \n
          \tSummary: {summarize} ({summary_style})\n
          \tMark suspicious: {suspicious} \n
          \tGenerate questions: {questions} \n 
          \tSpeaker detection: {speakers}
          """)
    
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_FOLDER, file_id + ".mp3")
    result_path = os.path.join(RESULTS_FOLDER, file_id + ".mp3.txt")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(
        transcribe_audio,
        file_path,
        result_path,
        DEVICE,
        api_key,
        model,
        summarize,
        summary_style,
        suspicious,
        questions,
        speakers
    )

    return JSONResponse({"message": "File uploaded, processing started.", "file_id": file_id})

# Endpoint to get transcription as JSON
@app.get("/transcription/{file_id}")
async def get_transcription(file_id: str):
    result_path = os.path.join(RESULTS_FOLDER, file_id + ".mp3.txt")
    
    if not os.path.exists(result_path):
        return JSONResponse({"error": "Transcription not ready yet."}, status_code=404)

    with open(result_path, "r", encoding="utf-8") as f:
        content = f.read()

    return JSONResponse({"transcription": content})

# Endpoint to download transcription as a file
@app.get("/download/{file_id}")
async def download_transcription(file_id: str):
    result_path = os.path.join(RESULTS_FOLDER, file_id + ".mp3.txt")

    if not os.path.exists(result_path):
        return JSONResponse({"error": "Transcription not found"}, status_code=404)

    return FileResponse(result_path, filename=f"{file_id}.txt", media_type="text/plain")

# ----------------------------------------------------------------
# Return the connection with the frontend
@app.get("/")
async def serve_home():
    return FileResponse("static/index.html")

# Ensure FastAPI serves static files (including index.html)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")