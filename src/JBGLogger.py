import logging
import os
from pathlib import Path
from datetime import datetime

class JBGLogger:
    
    LOG_LEVELS =  {"CRITICAL": 50, "ERROR": 40, "WARNING": 30, "INFO": 20, "DEBUG": 10}
    
    def __init__(self, level="INFO", name="log"):
        
        if level not in self.LOG_LEVELS:
            raise TypeError(f"Log level must be to set to one of: {', '.join([key for key in self.LOG_LEVELS.keys()])}")
        else:
            self.level = self.LOG_LEVELS[level]

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

        self.logger = logging.getLogger(__name__)
