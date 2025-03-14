from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid
import src.JBGtranscriber as JBGtranscriber
from pathlib import Path

app = FastAPI()

# Ensure FastAPI serves static files (including index.html)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
DEVICE = "cpu"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Function to transcribe in the background
def transcribe_audio(file_path: str, result_path: str, device: str):
    
    transcriber = JBGtranscriber(Path(file_path), Path(result_path), device=device)
    
    try:
        transcriber.perform_transcription_steps(generate_summary=True, find_suspicious_phrases=True,\
            suggest_follow_up_questions=True)
    except Exception as e:
        return JSONResponse({"Transcription error": str(e)}, status_code=500)

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_FOLDER, file_id + "_" + file.filename)
    result_path = os.path.join(RESULTS_FOLDER, file_id + ".txt")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(transcribe_audio, file_path, result_path, DEVICE)

    return JSONResponse({"message": "File uploaded, processing started.", "file_id": file_id})

@app.get("/download/{file_id}")
async def download_transcription(file_id: str):
    result_path = os.path.join(RESULTS_FOLDER, file_id + ".txt")
    
    if not os.path.exists(result_path):
        return JSONResponse({"error": "Transcription not ready yet."}, status_code=404)

    with open(result_path, "r") as f:
        content = f.read()

    return JSONResponse({"transcription": content})
