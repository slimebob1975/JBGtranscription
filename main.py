from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form, Request, Response, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.background import BackgroundTask
import os
import asyncio
import time
from contextlib import asynccontextmanager, suppress
from dotenv import load_dotenv
import sys
import uuid
import re
import src.JBGtranscriber as JBGtranscriber
from src.JBGtranscriber import (
    MAX_SUMMARY_PROMPT_CHARS,
    LEGACY_SUMMARY_OPTION_IDS,
    TRANSCRIPTION_FORMATS,
    DEFAULT_TRANSCRIPTION_FORMAT,
    TRANSCRIPTION_ACCURACY_LEVELS,
    DEFAULT_TRANSCRIPTION_ACCURACY,
    ENGINE_FASTER_WHISPER,
    ENGINE_TRANSFORMERS,
    SUPPORTED_AUDIO_EXTENSIONS,
    TRANSCRIPTION_CACHE_DIR,
)
from src.JBGSecureFileHandler import SecureFileHandler
from src.JBGBuildInfo import get_build_info
from src.JBGModelPrices import get_price_table, price_for, validate_price_table
from pathlib import Path
import torch
from src.JBGLogger import JBGLogger
from urllib.parse import unquote_plus
import base64
from io import BytesIO

logger = JBGLogger(level="DEBUG").logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    # The price file is read once per process, so it is checked at every start.
    for warning in validate_price_table():
        logger.warning(f"Prisfilen: {warning}")
    # The cleanup loop is defined further down; it runs for as long as the app.
    cleanup_task = asyncio.create_task(cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task


try:
    app = FastAPI(lifespan=lifespan)
except Exception as e:
    logger.error(f"FastAPI ERROR:", e, file=sys.stderr)
    raise

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULTS_FOLDER = BASE_DIR / "results"
# What the Transformers engine can use. faster-whisper decides separately, since
# CTranslate2 can reach a GPU that a CPU-only torch build cannot see; both
# honour JBG_DEVICE.
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
if os.getenv("JBG_DEVICE", "auto").strip().lower() == "cpu":
    DEVICE = "cpu"
UPLOAD_FOLDER.mkdir(exist_ok=True)
os.chmod(UPLOAD_FOLDER, 0o777)
RESULTS_FOLDER.mkdir(exist_ok=True)
os.chmod(RESULTS_FOLDER, 0o777)

# In-memory job status store: file_id -> dict
jobs = {}

# Files are normally deleted when a job ends or a download completes. Anything
# left behind by a failed setup, a restart mid-job or a result nobody
# downloaded is removed here once it is older than the retention time. The
# rule is age alone, not "empty uploads/ at startup": the folders sit on disk
# that several instances can share, and a restarting instance must not delete
# an upload that another one is still transcribing.
DEFAULT_RETENTION_HOURS = 24
# Cached transcriptions of unencrypted uploads, in clear text, have a setting
# of their own. Deleting one early is harmless: the next upload of the same
# file is simply transcribed again.
DEFAULT_CACHE_RETENTION_HOURS = 24
MIN_CACHE_RETENTION_HOURS = 1
# Must exceed the longest job. startup.sh lets Gunicorn run one for up to two
# hours (--timeout 7200), so a lower setting is raised to this.
MIN_RETENTION_HOURS = 3
CLEANUP_INTERVAL_SECONDS = 3600


def _read_hours(name: str, default: int, minimum: int, reason: str) -> int:
    """Read a whole number of hours from the environment, with a default and a minimum."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        hours = int(raw)
    except ValueError:
        logger.warning(f"{name}={raw!r} är inte ett heltal. Använder {default} timmar.")
        return default
    if hours < minimum:
        logger.warning(f"{name}={hours} är {reason}. Använder {minimum} timmar.")
        return minimum
    return hours


def read_retention_hours() -> int:
    """Read JBG_RETENTION_HOURS, falling back to the default and enforcing the minimum."""
    return _read_hours("JBG_RETENTION_HOURS", DEFAULT_RETENTION_HOURS,
                       MIN_RETENTION_HOURS, "kortare än det längsta jobbet")


def read_cache_retention_hours() -> int:
    """Read JBG_CACHE_RETENTION_HOURS, falling back to the default and enforcing the minimum."""
    return _read_hours("JBG_CACHE_RETENTION_HOURS", DEFAULT_CACHE_RETENTION_HOURS,
                       MIN_CACHE_RETENTION_HOURS, "för kort")


def remove_stale_files(max_age_seconds: float, now: float = None, folders=None) -> int:
    """Delete files in the folders (uploads/ and results/ by default) older than max_age_seconds."""
    now = time.time() if now is None else now
    removed = 0
    for folder in folders or (UPLOAD_FOLDER, RESULTS_FOLDER):
        try:
            entries = list(folder.iterdir())
        except FileNotFoundError:
            continue
        for path in entries:
            try:
                if not path.is_file():
                    continue
                if now - path.stat().st_mtime <= max_age_seconds:
                    continue
                path.unlink(missing_ok=True)
                removed += 1
            except Exception as e:
                logger.warning(f"Kunde inte radera gammal fil {path}: {e}")
    return removed


def prune_jobs(max_age_seconds: float, now: float = None) -> int:
    """Drop job entries older than max_age_seconds from the in-memory store."""
    now = time.time() if now is None else now
    pruned = 0
    for file_id, job in list(jobs.items()):
        created = job.get("created")
        if created is not None and now - created > max_age_seconds:
            jobs.pop(file_id, None)
            pruned += 1
    return pruned


def run_cleanup(retention_hours: int, cache_retention_hours: int = DEFAULT_CACHE_RETENTION_HOURS):
    max_age_seconds = retention_hours * 3600
    removed = remove_stale_files(max_age_seconds)
    pruned = prune_jobs(max_age_seconds)
    cached = remove_stale_files(cache_retention_hours * 3600, folders=(TRANSCRIPTION_CACHE_DIR,))
    if removed or pruned:
        logger.info(
            f"Rensning: {removed} filer äldre än {retention_hours} timmar raderade, "
            f"{pruned} gamla jobb borttagna ur minnet."
        )
    if cached:
        logger.info(f"Rensning: {cached} cachade transkriberingar äldre än {cache_retention_hours} timmar raderade.")


async def cleanup_loop():
    """Run the cleanup at startup and then once per interval, off the event loop."""
    # Read here rather than at import, so that load_dotenv() has run.
    retention_hours = read_retention_hours()
    cache_retention_hours = read_cache_retention_hours()
    logger.info(f"Filer i uploads/ och results/ raderas när de är äldre än {retention_hours} timmar.")
    logger.info(f"Cachade transkriberingar raderas när de är äldre än {cache_retention_hours} timmar.")
    while True:
        try:
            await asyncio.to_thread(run_cleanup, retention_hours, cache_retention_hours)
        except Exception as e:
            logger.warning(f"Rensningen av gamla filer misslyckades: {e}")
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)


def clean_up_audio_file(audio_file_path: Path):
    """Remove the temporary uploaded audio file after processing."""
    if not audio_file_path.exists():
        return

    try:
        audio_file_path.unlink()
        logger.info(f"Uppladdad ljudfil raderad efter transkribering: {audio_file_path}")
    except Exception as e:
        logger.warning(f"Misslyckades med att radera uppladdad ljudfil {audio_file_path}: {e}")


DEFAULT_MAX_UPLOAD_MB = 1024


def read_max_upload_mb() -> int:
    """Read JBG_MAX_UPLOAD_MB, falling back to the default for a missing or bad value."""
    raw = os.getenv("JBG_MAX_UPLOAD_MB", "").strip()
    if not raw:
        return DEFAULT_MAX_UPLOAD_MB
    try:
        value = int(raw)
        if value > 0:
            return value
    except ValueError:
        pass
    logger.warning(f"JBG_MAX_UPLOAD_MB={raw!r} är inte ett positivt heltal. Använder {DEFAULT_MAX_UPLOAD_MB} MB.")
    return DEFAULT_MAX_UPLOAD_MB


def audio_extension(upload_filename: str):
    """Return the lower-case audio extension of an upload, or None if unsupported.

    An encrypted upload arrives as e.g. "intervju.m4a.enc"; the extension of
    the original file is the one that counts.
    """
    name = upload_filename or ""
    if name.lower().endswith(".enc"):
        name = name[:-4]
    extension = Path(name).suffix.lower()
    return extension if extension in SUPPORTED_AUDIO_EXTENSIONS else None


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
    summary_prompt: str,
    transcription_format: str,
    transcription_accuracy: str,
    transcription_engine: str,
    questions: bool,
    speakers: bool,
    questions_prompt: str = "",
    speakers_prompt: str = "",
    custom_prompt: str = "",
):
    logger.info(f"Transcribe audio was called with encryption key: {encryption_key != ''}")

    # Ensure there is a job entry
    jobs.setdefault(file_id, {
        "status": "Transkribering initieras...",
        "done": False,
        "error": None,
        "created": time.time(),
    })

    def progress_callback(message: str):
        job = jobs.get(file_id)
        if job is not None:
            job["status"] = message

    try:
        # Setup belongs inside the try as well. If the key handler, the
        # transcriber or the decryption fails, the upload must still be
        # deleted and the job ended, or the file stays on the server and the
        # client keeps polling a job that will never finish.
        if encryption_key:
            secure_handler = SecureFileHandler(encryption_key)
        else:
            secure_handler = None

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

        transcriber.perform_transcription_steps(
            generate_summary=summarize,
            summary_style=summary_style,
            summary_prompt=summary_prompt,
            transcription_format=transcription_format,
            transcription_accuracy=transcription_accuracy,
            transcription_engine=transcription_engine,
            suggest_follow_up_questions=questions,
            analyze_speakers=speakers,
            progress_callback=progress_callback,
            follow_up_prompt=questions_prompt,
            speaker_prompt=speakers_prompt,
            custom_analysis_prompt=custom_prompt,
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
        # Some exceptions carry no message, cryptography's InvalidTag among
        # them. An empty error would read as success in the status endpoint.
        job["error"] = str(e) or type(e).__name__
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
    return {
        "title": title,
        "encryption_is_optional": encryption_is_optional,
        "version": get_build_info(),
        "audio_extensions": list(SUPPORTED_AUDIO_EXTENSIONS),
        "models": get_price_table()["menu"],
        "default_model": get_price_table()["default"],
        # Price per million tokens for each menu model, shown in the menu.
        "model_prices": {m: price_for(m) for m in get_price_table()["menu"] if price_for(m)},
        "currency": get_price_table()["currency"],
        "max_upload_mb": read_max_upload_mb(),
    }


# Selectable, editable summary instructions for the GUI. Served from the prompt
# policy so that the file stays the single source of truth.
@app.get("/summary_prompts")
def get_summary_prompts():
    options = JBGtranscriber.JBGtranscriber.summary_options()
    return {
        "options": [
            {
                "id": o["id"],
                "label": o["label"],
                "description": o["description"],
                "default": o["default"],
                "prompt": o["prompt"],
            }
            for o in options
        ],
        "max_length": MAX_SUMMARY_PROMPT_CHARS,
    }


# Standard texts for the other editable analyses. The custom analysis has a
# fixed opening that the user cannot change; only what follows it is sent.
@app.get("/analysis_prompts")
def get_analysis_prompts():
    return {
        **JBGtranscriber.JBGtranscriber.analysis_prompt_defaults(),
        "max_length": MAX_SUMMARY_PROMPT_CHARS,
    }


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
    model: str = Form(""),
    summarize: bool = Form(False),
    summary_style: str = Form(""),
    summary_prompt: str = Form(""),
    transcription_format: str = Form(""),
    transcription_accuracy: str = Form(""),
    fast_engine: bool = Form(True),
    suspicious: bool = Form(False),   # retired; accepted as a synonym below
    questions: bool = Form(False),
    speakers: bool = Form(False),
    # Empty means the standard text from the policy file.
    questions_prompt: str = Form(""),
    speakers_prompt: str = Form(""),
    # The custom analysis: what the user wrote after the fixed opening.
    custom_analysis: bool = Form(False),
    custom_prompt: str = Form(""),
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

    # Only models offered in the menu, all of which have a price. Empty means
    # the default, for requests that do not come from the page.
    model = model.strip() or get_price_table()["default"]
    if model not in get_price_table()["menu"]:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    extension = audio_extension(file.filename)
    if extension is None:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format. Allowed: " + ", ".join(SUPPORTED_AUDIO_EXTENSIONS),
        )

    # The valid ids come from the prompt policy, so adding an option there is
    # enough. Ids retired in an earlier version are still accepted and mapped.
    summary_options = JBGtranscriber.JBGtranscriber.summary_options()
    valid_style_ids = {o["id"] for o in summary_options}
    if summary_style and summary_style not in valid_style_ids:
        if summary_style in LEGACY_SUMMARY_OPTION_IDS:
            summary_style = LEGACY_SUMMARY_OPTION_IDS[summary_style]
        else:
            raise HTTPException(status_code=400, detail="Invalid summary style.")

    # The presentations replaced separate checkboxes for marking suspected
    # errors and for identifying speakers. An old client that still posts
    # suspicious=true or speakers=true is read as asking for that
    # presentation, unless it chose another one, which cannot be combined.
    transcription_format = (transcription_format or "").strip().lower()
    if not transcription_format:
        transcription_format = "marked" if suspicious else DEFAULT_TRANSCRIPTION_FORMAT
    if speakers and transcription_format == DEFAULT_TRANSCRIPTION_FORMAT:
        transcription_format = "speakers"
    if transcription_format not in TRANSCRIPTION_FORMATS:
        raise HTTPException(status_code=400, detail="Invalid transcription format.")
    speakers = transcription_format == "speakers"

    # The accuracy choice is a ceiling on the Whisper model. An empty value
    # means the default; an unknown one is rejected rather than guessed at.
    transcription_accuracy = (transcription_accuracy or "").strip().lower()
    if not transcription_accuracy:
        transcription_accuracy = DEFAULT_TRANSCRIPTION_ACCURACY
    if transcription_accuracy not in TRANSCRIPTION_ACCURACY_LEVELS:
        raise HTTPException(status_code=400, detail="Invalid transcription accuracy.")

    # The GUI offers this as a checkbox rather than as two engine names, since
    # the choice is between the same models run two different ways.
    transcription_engine = ENGINE_FASTER_WHISPER if fast_engine else ENGINE_TRANSFORMERS

    summary_prompt = (summary_prompt or "").strip()
    if len(summary_prompt) > MAX_SUMMARY_PROMPT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Summary instruction exceeds {MAX_SUMMARY_PROMPT_CHARS} characters.",
        )
    if not summarize:
        summary_prompt = ""

    questions_prompt = (questions_prompt or "").strip() if questions else ""
    speakers_prompt = (speakers_prompt or "").strip() if speakers else ""
    custom_prompt = (custom_prompt or "").strip() if custom_analysis else ""
    for name, text in (("Follow-up question", questions_prompt),
                       ("Speaker", speakers_prompt),
                       ("Custom analysis", custom_prompt)):
        if len(text) > MAX_SUMMARY_PROMPT_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"{name} instruction exceeds {MAX_SUMMARY_PROMPT_CHARS} characters.",
            )
    if custom_analysis and not custom_prompt:
        raise HTTPException(
            status_code=400,
            detail="Custom analysis needs an instruction after the fixed opening.",
        )

    # The instruction may contain case-specific context, so only its size is
    # logged, never its content.
    summary_prompt_source = "egen text" if summary_prompt else "standardtext"

    def describe(text):
        return f"egen text, {len(text)} tecken" if text else "standardtext"

    logger.info(f"""
          OpenAI API key was provided: {api_key != "sk-..."}\n
          OpenAI model of choice: {model}\n
          OpenAI API tasks: \n
          \tSummary: {summarize} ({summary_style}, {summary_prompt_source}, {len(summary_prompt)} tecken)\n
          \tTranscription format: {transcription_format} \n          \tTranscription accuracy: {transcription_accuracy} \n          \tTranscription engine: {transcription_engine} \n
          \tGenerate questions: {questions} ({describe(questions_prompt)}) \n 
          \tSpeaker detection: {speakers} ({describe(speakers_prompt)})\n
          \tCustom analysis: {bool(custom_prompt)} ({len(custom_prompt)} tecken)
          """)

    file_id = str(uuid.uuid4())
    file_path = UPLOAD_FOLDER / (file_id + extension + (".encrypted" if encryption_key else ""))
    download_filename = make_result_filename(file.filename, file_id)
    storage_filename = download_filename + (".encrypted" if encryption_key else "")
    result_path = RESULTS_FOLDER / storage_filename

    jobs[file_id] = {
        "status": "Fil uppladdad. Väntar på att transkriberingen ska starta...",
        "done": False,
        "error": None,
        "download_filename": download_filename,
        "encrypted": bool(encryption_key),
        "created": time.time(),
    }

    if encryption_key:
        logger.info("Krypterad fil sparas...")
    else:
        logger.info("Ej krypterad fil sparas...")

    # The browser checks the size before uploading; this is the check for
    # requests that do not come from the page. Encryption adds 28 bytes.
    max_bytes = read_max_upload_mb() * 1024 * 1024 + 28
    written = 0
    with open(file_path, "wb") as buffer:
        while chunk := file.file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                break
            buffer.write(chunk)
    if written > max_bytes:
        file_path.unlink(missing_ok=True)
        jobs.pop(file_id, None)
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the maximum upload size of {read_max_upload_mb()} MB.",
        )

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
        summary_prompt,
        transcription_format,
        transcription_accuracy,
        transcription_engine,
        questions,
        speakers,
        questions_prompt,
        speakers_prompt,
        custom_prompt,
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
