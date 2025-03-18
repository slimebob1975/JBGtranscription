from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import sys
import shutil
import uuid
import src.JBGtranscriber as JBGtranscriber
from pathlib import Path

from fastapi import FastAPI

try:
    app = FastAPI()
except Exception as e:
    print("FastAPI ERROR:", e, file=sys.stderr)
    raise

UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
DEVICE = "cpu"
OPENAI_API_KEYS_FILE = Path.cwd() / Path("./src/keys/openai_api_keys.json")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.chmod(UPLOAD_FOLDER, 0o777)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.chmod(RESULTS_FOLDER, 0o777)

# Function to transcribe in the background
def transcribe_audio(file_path: str, result_path: str, device: str):
    
    transcriber = JBGtranscriber.JBGtranscriber(Path(file_path), Path(result_path), device=device, \
        openai_api_keys_file=OPENAI_API_KEYS_FILE)
    
    try:
        transcriber.perform_transcription_steps(generate_summary=True, find_suspicious_phrases=True,\
            suggest_follow_up_questions=True)
    except Exception as e:
        return JSONResponse({"Transcription error": str(e)}, status_code=500)
    else:
        Path.unlink(Path(file_path))
        return JSONResponse({"message": "Transcription completed successfully. Audio file deleted."}, status_code=200)

@app.post("/upload/")
async def upload_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_FOLDER, file_id + ".mp3")
    result_path = os.path.join(RESULTS_FOLDER)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(transcribe_audio, file_path, result_path, DEVICE)

    return JSONResponse({"message": "File uploaded, processing started.", "file_id": file_id})

@app.get("/download/{file_id}")
async def get_transcription(file_id: str):
    result_path = os.path.join(RESULTS_FOLDER, file_id + ".mp3.txt")
    
    if not os.path.exists(result_path):
        return JSONResponse({"error": "Transcription not ready yet."}, status_code=404)

    with open(result_path, "r") as f:
        content = f.read()

    return JSONResponse({"transcription": content})

# Ensure FastAPI serves static files (including index.html)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")