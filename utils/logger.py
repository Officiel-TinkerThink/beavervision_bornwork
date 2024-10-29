import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from ..config.settings import Settings

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(Settings().LOG_LEVEL)
    
    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / f"{name}.log",
        maxBytes=10_000_000,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(Settings().LOG_FORMAT))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(Settings().LOG_FORMAT))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger