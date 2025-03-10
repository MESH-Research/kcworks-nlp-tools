import logging
from pathlib import Path
import config


def set_up_logging() -> None:
    # Define format string once
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Ensure logs directory exists
    logs_dir = Path(config.LOGS_PATH)
    logs_dir.mkdir(exist_ok=True)

    # Create full path for log file
    log_file_path = logs_dir / "kcworks_nlp_tools.log"

    root_logger = logging.getLogger()
    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set up file handler
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))

    # Configure root logger
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
