import logging
import os
from pathlib import Path
from datetime import datetime

# Create the log directory if it doesn't exist
LOG_DIR = Path("log")
LOG_DIR.mkdir(exist_ok=True)

# Format: log/transcription_2024-05-15_14-30.log
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
log_filename = LOG_DIR / f"transcription_{timestamp}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
