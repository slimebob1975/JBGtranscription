from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form, Request, Response, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.background import BackgroundTask
import os
from dotenv import load_dotenv
import sys
import shutil
import uuid
import re
import src.JBGtranscriber as JBGtranscriber
from src.JBGSecureFileHandler import SecureFileHandler
from pathlib import Path
import torch
from src.JBGLogger import JBGLogger
from urllib.parse import unquote_plus
import base64
from io import BytesIO

logger = JBGLogger(level="DEBUG").logger

try:
    app = FastAPI()
except Exception as e:
    logger.error(f"FastAPI ERROR:", e, file=sys.stderr)
    raise

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULTS_FOLDER = BASE_DIR / "results"
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
UPLOAD_FOLDER.mkdir(exist_ok=True)
os.chmod(UPLOAD_FOLDER, 0o777)
RESULTS_FOLDER.mkdir(exist_ok=True)
os.chmod(RESULTS_FOLDER, 0o777)

# In-memory job status store: file_id -> dict
jobs = {}


def clean_up_audio_file(audio_file_path: Path):
    """Remove the temporary uploaded audio file after processing."""
    if not audio_file_path.exists():
        return

    try:
        audio_file_path.unlink()
        logger.info(f"Uppladdad ljudfil raderad efter transkribering: {audio_file_path}")
    except Exception as e:
        logger.warning(f"Misslyckades med att radera uppladdad ljudfil {audio_file_path}: {e}")


def make_result_filename(upload_filename: str, file_id: str) -> str:
    """Create a readable, Windows-safe DOCX filename with a collision-resistant suffix."""
    original_name = upload_filename or "transkribering.mp3"
    if original_name.lower().endswith(".enc"):
        original_name = original_name[:-4]

    source_stem = Path(original_name).stem
    safe_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", source_stem).strip(" .")
    safe_stem = (safe_stem or "transkribering")[:100]
    return f"{safe_stem}_{file_id}.docx"


def validate_file_id(file_id: str) -> str:
    """Accept only canonical UUID job identifiers before using them in glob patterns."""
    try:
        return str(uuid.UUID(file_id))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file id.")


def find_result_path(file_id: str):
    """Find an encrypted or plaintext result for a job without trusting user path input."""
    canonical_id = validate_file_id(file_id)
    patterns = (
        f"*_{canonical_id}.docx.encrypted",
        f"*_{canonical_id}.docx",
    )
    for pattern in patterns:
        matches = list(RESULTS_FOLDER.glob(pattern))
        if matches:
            return matches[0]
    return None


def download_name_from_storage_path(result_path: Path) -> str:
    """Return the client-facing .docx filename for a stored result."""
    name = result_path.name
    if name.lower().endswith(".encrypted"):
        name = name[:-len(".encrypted")]
    return name


def delete_result_file(result_path: Path):
    """Delete a result after its HTTP response has finished streaming."""
    try:
        result_path.unlink(missing_ok=True)
        logger.info(f"Resultatfil raderad efter avslutad nedladdningsström: {result_path}")
    except Exception as e:
        logger.warning(f"Kunde inte radera resultatfil efter nedladdning {result_path}: {e}")


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

        clean_up_audio_file(Path(file_path))

        job = jobs.get(file_id)
        if job is not None:
            job["done"] = True
            if not job.get("status"):
                job["status"] = "Transkribering avslutad."
            job["error"] = None

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        clean_up_audio_file(Path(file_path))
        job = jobs.setdefault(file_id, {})
        job["done"] = True
        job["error"] = str(e)
        job["status"] = "Ett fel uppstod vid transkriberingen."


class FrameOptionsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        # Remove 'x-frame-options' if it exists (case-insensitive)
        if "x-frame-options" in response.headers:
            del response.headers["x-frame-options"]
        # Allow embedding from configured origins.
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

    encryption_required = os.getenv("ENCRYPTION_IS_OPTIONAL", "1") != "1"
    if encryption_required and not encryption_key:
        raise HTTPException(status_code=400, detail="Encryption is required by server configuration.")

    if encryption_key:
        try:
            decoded_key = base64.b64decode(encryption_key, validate=True)
            if len(decoded_key) != 32:
                raise ValueError("AES-GCM key must be 32 bytes")
            logger.info("encryption_key verkar vara giltig base64 och 256 bitar lång")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid encryption key.")

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
    file_path = UPLOAD_FOLDER / (file_id + (".mp3.encrypted" if encryption_key else ".mp3"))
    download_filename = make_result_filename(file.filename, file_id)
    storage_filename = download_filename + (".encrypted" if encryption_key else "")
    result_path = RESULTS_FOLDER / storage_filename

    jobs[file_id] = {
        "status": "Fil uppladdad. Väntar på att transkriberingen ska starta...",
        "done": False,
        "error": None,
        "download_filename": download_filename,
        "encrypted": bool(encryption_key),
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
        str(file_path),
        encryption_key,
        str(result_path),
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


# Endpoint to get transcription job status.
# No result content or encryption key is sent through this endpoint.
@app.get("/transcription/{file_id}")
async def get_transcription_status(file_id: str):
    job = jobs.get(file_id)

    if job is None:
        result_path = find_result_path(file_id)
        if result_path is not None:
            return {
                "done": True,
                "status": "Transkribering avslutad. Startar nedladdning...",
                "download_filename": download_name_from_storage_path(result_path),
            }

        return JSONResponse(
            {"done": False, "status": "Processar..."},
            status_code=202,
        )

    if job.get("done") and job.get("error"):
        return JSONResponse(
            {
                "done": True,
                "status": job.get("status") or "Ett fel uppstod vid transkriberingen.",
                "error": job["error"],
            },
            status_code=500,
        )

    if not job.get("done"):
        return JSONResponse(
            {
                "done": False,
                "status": job.get("status") or "Processar...",
            },
            status_code=202,
        )

    return {
        "done": True,
        "status": "Transkribering avslutad. Startar nedladdning...",
        "download_filename": job.get("download_filename"),
    }


# Download the finished DOCX. If encrypted, decryption happens only in memory.
# The server-side result file is removed after the response has been fully streamed.
@app.post("/download/{file_id}")
async def download_transcription(file_id: str, encryption_key: str = Form("")):
    result_path = find_result_path(file_id)
    if result_path is None:
        return JSONResponse({"error": "Transcription not found"}, status_code=404)

    encrypted = result_path.name.lower().endswith(".encrypted")
    try:
        if encrypted:
            if not encryption_key:
                return JSONResponse(
                    {"error": "Encryption key is required"},
                    status_code=400,
                )
            secure_handler = SecureFileHandler(encryption_key)
            stream = secure_handler.decrypt_file_to_memory(str(result_path))
        else:
            stream = BytesIO(result_path.read_bytes())

        filename = download_name_from_storage_path(result_path)
        return StreamingResponse(
            stream,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            background=BackgroundTask(delete_result_file, result_path),
        )
    except Exception as e:
        logger.error(f"Could not stream DOCX result: {e}")
        return JSONResponse({"error": "Failed to serve DOCX result."}, status_code=500)


# ----------------------------------------------------------------
# Return the connection with the frontend
@app.get("/")
async def serve_home():
    return FileResponse(STATIC_DIR / "index.html")


# Ensure FastAPI serves static files (including index.html)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
