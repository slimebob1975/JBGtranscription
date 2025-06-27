from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
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

logger = JBGLogger(level="INFO").logger

try:
    app = FastAPI()
except Exception as e:
    logger.error(f"FastAPI ERROR:", e, file=sys.stderr)
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
def transcribe_audio(file_path: str, encryption_key: str, result_path: str, device: str, api_key:str, openai_model: str, \
    summarize: bool, summary_style: str, suspicious: bool, questions: bool, speakers: bool):
    
    logger.info(f"Transcribe audio was called with encryption key: {encryption_key != ''}")
    if encryption_key:
        secure_handler = SecureFileHandler(encryption_key)
    else:
        secure_handler = None

    transcriber = JBGtranscriber.JBGtranscriber(Path(file_path), Path(result_path), device=device, \
        api_key=api_key, openai_model = openai_model, secure_handler = secure_handler)
    
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
        analyze_speakers=speakers
    )
        # 🔥 Radera krypterad mp3-fil från disk
        try:
            os.remove(file_path)
            logger.info(f"Krypterad ljudfil raderad efter avkryptering och transkribering: {file_path}")
        except Exception as e:
            logger.warning(f"Misslyckades med att radera krypterad fil efter avkryptering och transkribering {file_path}: {e}")

    except Exception as e:
        return JSONResponse({"Transcription error": str(e)}, status_code=500)
    else:
        clean_up_files(Path(file_path), Path(result_path)) 
        return JSONResponse({"message": "Transcription completed successfully. Audio file deleted."}, status_code=200)
    
def clean_up_files(audio_file_path: Path, transcriptions_path: Path):
    
    # Remove last audio file
    if audio_file_path.exists():
        Path.unlink(audio_file_path)
    
    # Remove all but the last transcription file (which is the transcription of the audio file)
    old_transcripts = sorted([file for file in transcriptions_path.glob("*.mp3.txt")], key=lambda x: x.stat().st_ctime)[:-1]
    for file in old_transcripts:
        if file.exists():
            Path.unlink(file)

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

    if encryption_key:
        logger.info("Krypterad fil sparas...")
    else:
        logger.info("Ej krypterad fil sparas...")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(
        transcribe_audio,
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
async def get_transcription(file_id: str, encryption_key: str = None):
    result_path = Path(RESULTS_FOLDER) / f"{file_id}.mp3.txt"

    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Transkriberingsresultat hittades inte.")

    try:
        if encryption_key:
            logger.info("Dekrypterar transkriptionsfil med angiven nyckel...")
            secure_handler = SecureFileHandler(encryption_key)
            decrypted_stream = secure_handler.decrypt_file_to_memory(str(result_path))
            content = decrypted_stream.read().decode("utf-8")
        else:
            logger.info("Läser okrypterad transkriptionsfil...")
            with open(result_path, "r", encoding="utf-8") as f:
                content = f.read()

        return {"transcription": content}
    
    except Exception as e:
        logger.error(f"Fel vid inläsning av transkriptionsfil: {e}")
        raise HTTPException(status_code=500, detail="Fel vid läsning av transkriptionsfil.")

# Endpoint to download transcription as a file
@app.get("/download/{file_id}")
async def download_transcription(file_id: str, key: str = ""):
    result_path = os.path.join(RESULTS_FOLDER, file_id + ".mp3.txt")

    if not os.path.exists(result_path):
        return JSONResponse({"error": "Transcription not found"}, status_code=404)

    with open(result_path, "r", encoding="utf-8") as f:
        plain_text = f.read()

    if key:
        try:
            key_bytes = base64.b64decode(key)
            aesgcm = AESGCM(key_bytes)
            iv = secrets.token_bytes(12)
            encrypted = aesgcm.encrypt(iv, plain_text.encode("utf-8"), None)

            # Skapa en minnesfil att streama (IV + krypterad data)
            stream = BytesIO()
            stream.write(iv + encrypted)
            stream.seek(0)

            filename = f"{file_id}.encrypted.txt"
            return StreamingResponse(
                stream,
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        except Exception as e:
            logger.error(f"Kryptering av transkriptionsfil misslyckades: {e}")
            return JSONResponse({"error": "Kunde inte kryptera resultatfilen."}, status_code=500)

    else:
        # Fallback: returnera okrypterad textfil
        return FileResponse(result_path, filename=f"{file_id}.txt", media_type="text/plain")


# ----------------------------------------------------------------
# Return the connection with the frontend
@app.get("/")
async def serve_home():
    return FileResponse("static/index.html")

# Ensure FastAPI serves static files (including index.html)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")