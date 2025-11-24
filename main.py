from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import os
from dotenv import load_dotenv
import sys
import shutil
import uuid
import src.JBGtranscriber as JBGtranscriber
from src.JBGSecureFileHandler import SecureFileHandler
from pathlib import Path
import torch
from src.JBGLogger import JBGLogger
from urllib.parse import unquote_plus
import base64
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import secrets
from io import BytesIO

logger = JBGLogger(level="DEBUG").logger

try:
    app = FastAPI()
except Exception as e:
    logger.error(f"FastAPI ERROR:", e, file=sys.stderr)
    raise

UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.chmod(UPLOAD_FOLDER, 0o777)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.chmod(RESULTS_FOLDER, 0o777)

# In-memory job status store: file_id -> dict
jobs = {}

# Function to transcribe in the background
def transcribe_audio(
    file_id: str,
    file_path: str,
    encryption_key: str,
    result_path: str,
    device: str,
    api_key: str,
    openai_model: str,
    summarize: bool,
    summary_style: str,
    suspicious: bool,
    questions: bool,
    speakers: bool,
):
    logger.info(f"Transcribe audio was called with encryption key: {encryption_key != ''}")
    if encryption_key:
        secure_handler = SecureFileHandler(encryption_key)
    else:
        secure_handler = None

    # Ensure there is a job entry
    jobs.setdefault(file_id, {
        "status": "Transkribering initieras...",
        "done": False,
        "error": None,
    })

    def progress_callback(message: str):
        job = jobs.get(file_id)
        if job is not None:
            job["status"] = message

    transcriber = JBGtranscriber.JBGtranscriber(
        Path(file_path),
        Path(result_path),
        device=device,
        api_key=api_key,
        openai_model=openai_model,
        secure_handler=secure_handler,
    )

    if secure_handler:
        audio_stream = secure_handler.decrypt_file_to_memory(file_path)
    else:
        audio_stream = None
    transcriber.audio_stream = audio_stream

    try:
        transcriber.perform_transcription_steps(
            generate_summary=summarize,
            summary_style=summary_style,
            find_suspicious_phrases=suspicious,
            suggest_follow_up_questions=questions,
            analyze_speakers=speakers,
            progress_callback=progress_callback,
        )

        # Radera krypterad mp3-fil från disk
        try:
            os.remove(file_path)
            logger.info(f"Uppladdad ljudfil raderad efter ev. kryptering och transkribering: {file_path}")
        except Exception as e:
            logger.warning(f"Misslyckades med att radera uppladdad ljudfil efter ev. kryptering och transkribering: {file_path}: {e}")

        # Städa gamla transkript
        clean_up_files(Path(file_path), Path(RESULTS_FOLDER))

        job = jobs.get(file_id)
        if job is not None:
            job["done"] = True
            # keep last progress message if set, otherwise set a default
            if not job.get("status"):
                job["status"] = "Transkribering avslutad."
            job["error"] = None

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        job = jobs.setdefault(file_id, {})
        job["done"] = True
        job["error"] = str(e)
        job["status"] = "Ett fel uppstod vid transkriberingen."
    
def clean_up_files(audio_file_path: Path, transcriptions_path: Path):
    
    # Remove last audio file
    if audio_file_path.exists():
        Path.unlink(audio_file_path)
    
    # Remove all but the last transcription file (which is the transcription of the audio file)
    old_transcripts = sorted([file for file in transcriptions_path.glob("*.mp3.txt")], key=lambda x: x.stat().st_ctime)[:-1]
    for file in old_transcripts:
        if file.exists():
            Path.unlink(file)
            
def decrypt_transcription_file_if_needed(result_path: str, encryption_key: str) -> str:

    if encryption_key:
        try:
            secure_handler = SecureFileHandler(encryption_key)
            decrypted_stream = secure_handler.decrypt_file_to_memory(result_path)
            return decrypted_stream.read().decode("utf-8")
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise HTTPException(status_code=500, detail="Could not decrypt transcription file.")
    else:
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Reading plaintext transcription failed: {e}")
            raise HTTPException(status_code=500, detail="Could not read transcription file.")

class FrameOptionsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        # Remove 'x-frame-options' if it exists (case-insensitive)
        if "x-frame-options" in response.headers:
            del response.headers["x-frame-options"]
        # Allow embedding from any origin (use with caution)
        response.headers["Content-Security-Policy"] = f"frame-ancestors {FRAME_ANCESTORS}"    
        return response
    
# Allow embedding via iframes
load_dotenv()
FRAME_ANCESTORS = os.getenv("FRAME_ANCESTORS", "*")
if not FRAME_ANCESTORS or FRAME_ANCESTORS == "*":
    logger.warning("⚠️ Warning: Using default FRAME_ANCESTORS='*'. Set in .env (localhost) or Azure App Settings (deployed).")
app.add_middleware(FrameOptionsMiddleware)

@app.get("/config")
def get_config():
    title = os.getenv("APP_TITLE", "JBG Transkribering")
    logger.info(f" Appens titel: {title}")
    encryption_is_optional = os.getenv("ENCRYPTION_IS_OPTIONAL", "1")
    logger.info(f" Kryptering är tillval: {encryption_is_optional}")
    return {"title": title, "encryption_is_optional": encryption_is_optional}

# To find and log the current user
@app.get("/me")
def get_user(request: Request):
    raw_user = request.headers.get("X-MS-CLIENT-PRINCIPAL-NAME", "okänd användare")
    user = unquote_plus(raw_user)
    logger.info(f" Användare inloggad: {user}")
    return {"user": user}

# Entry point for uploading audio files
@app.post("/upload/")
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    encryption_key: str = Form(""),
    api_key: str = Form(...),
    model: str = Form("gpt-4o"),
    summarize: bool = Form(False),
    summary_style: str = Form("short"),
    suspicious: bool = Form(False),
    questions: bool = Form(False),
    speakers: bool = Form(False)
):
    
    logger.info(f"Upload endpoint received encryption_key: {'✅ present' if encryption_key else '❌ missing or empty'}")
    logger.info(f"Length of encryption_key: {len(encryption_key)} characters")
    try:
        _ = base64.b64decode(encryption_key, validate=True)
        logger.info("encryption_key verkar vara giltig base64")
    except Exception:
        logger.warning("encryption_key är INTE giltig base64!")
    logger.info(f"""
          OpenAI API key was provided: {api_key != "sk-..."}\n
          OpenAI model of choice: {model}\n
          OpenAI API tasks: \n
          \tSummary: {summarize} ({summary_style})\n
          \tMark suspicious: {suspicious} \n
          \tGenerate questions: {questions} \n 
          \tSpeaker detection: {speakers}
          """)
    
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_FOLDER, file_id + (".mp3.encrypted" if encryption_key else ".mp3"))    
    result_path = os.path.join(RESULTS_FOLDER, file_id + ".mp3.txt")

     # Initiera jobbstatus
    jobs[file_id] = {
        "status": "Fil uppladdad. Väntar på att transkriberingen ska starta...",
        "done": False,
        "error": None,
    }

    if encryption_key:
        logger.info("Krypterad fil sparas...")
    else:
        logger.info("Ej krypterad fil sparas...")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(
        transcribe_audio,
        file_id,
        file_path,
        encryption_key,
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
async def get_transcription(file_id: str, encryption_key: str = ""):
    result_path = os.path.join(RESULTS_FOLDER, file_id + ".mp3.txt")
    job = jobs.get(file_id)

    # If file does not exist yet → still processing
    if not os.path.exists(result_path):
        # If job exists → show its status
        if job is not None:
            return JSONResponse(
                {
                    "done": False,
                    "status": job.get("status") or "Processar...",
                },
                status_code=202,
            )

        # If job does NOT exist → probably another instance is handling it
        return JSONResponse(
            {
                "done": False,
                "status": "Processar...",
            },
            status_code=202,
        )

    # If file exists, we treat it as finished.
    content = decrypt_transcription_file_if_needed(result_path, encryption_key)

    # If job exists → use its final message
    if job is not None:
        final_status = job.get("status") or "Transkribering avslutad."
    else:
        final_status = "Transkribering avslutad."

    return {
        "done": True,
        "status": final_status,
        "transcription": content,
    }

# Endpoint to download transcription as a file
@app.get("/download/{file_id}")
async def download_transcription(file_id: str, key: str = ""):
    result_path = os.path.join(RESULTS_FOLDER, file_id + ".mp3.txt")

    if not os.path.exists(result_path):
        return JSONResponse({"error": "Transcription not found"}, status_code=404)

    try:
        content = decrypt_transcription_file_if_needed(result_path, key)
        stream = BytesIO(content.encode("utf-8"))
        filename = f"{file_id}.txt"
        return StreamingResponse(
            stream,
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"Could not stream transcription result: {e}")
        return JSONResponse({"error": "Failed to serve transcription file."}, status_code=500)

# ----------------------------------------------------------------
# Return the connection with the frontend
@app.get("/")
async def serve_home():
    return FileResponse("static/index.html")

# Ensure FastAPI serves static files (including index.html)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")