import logging
import os
from pathlib import Path
from datetime import datetime

class JBGLogger:
    
    LOG_LEVELS =  {"CRITICAL": 50, "ERROR": 40, "WARNING": 30, "INFO": 20, "DEBUG": 10}
    
    # Logging is configured once per process. Without this, every further
    # JBGLogger left an empty log file behind: constructing a FileHandler opens
    # the file immediately, while basicConfig does nothing when the root logger
    # already has handlers, so the handler was created and then discarded. A
    # run produced one real log plus one empty file per extra logger created in
    # a different minute.
    _configured = False

    # The OpenAI client logs whole request bodies at DEBUG level, which means
    # the entire transcription would be written to the log in clear text. These
    # libraries are therefore kept at WARNING unless explicitly asked for.
    NOISY_LIBRARY_LOGGERS = ("openai", "httpx", "httpcore", "urllib3")

    def __init__(self, level="INFO", name="log"):

        if level not in self.LOG_LEVELS:
            raise TypeError(f"Log level must be to set to one of: {', '.join([key for key in self.LOG_LEVELS.keys()])}")
        else:
            self.level = self.LOG_LEVELS[level]

        if not JBGLogger._configured:
            # Create the log directory if it doesn't exist
            LOG_DIR = Path("log")
            LOG_DIR.mkdir(exist_ok=True)

            # Format: log/log_name_2024-05-15_14-30.log
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            log_filename = LOG_DIR / f"{name}_{timestamp}.log"

            # Configure logging
            logging.basicConfig(
                level=level,
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[
                    logging.FileHandler(log_filename, encoding="utf-8"),
                    logging.StreamHandler()
                ]
            )

            JBGLogger._quiet_libraries()
            JBGLogger._configured = True

        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _quiet_libraries():
        """Keep third-party request logging out of the log file.

        Set JBG_LOG_HTTP=1 to allow it back when debugging the API itself.
        Be aware that it puts the transcription in the log in clear text.
        """
        if os.getenv("JBG_LOG_HTTP") == "1":
            logging.getLogger(__name__).warning(
                "JBG_LOG_HTTP=1: bibliotekens anrop loggas, vilket innebär att "
                "transkriberingen hamnar i loggfilen i klartext."
            )
            return

        for name in JBGLogger.NOISY_LIBRARY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
