"""
==========================================================
Employee Attrition Machine Learning

logger.py

Author : Pramod Prakash Jadhav
==========================================================
"""

import logging
from pathlib import Path

from src.config import LOGS_DIR

# ==========================================================
# Log File Configuration
# ==========================================================

LOG_FILE = LOGS_DIR / "project.log"

# ==========================================================
# Logger Function
# ==========================================================

def get_logger(name: str = "EmployeeAttritionML") -> logging.Logger:
    """
    Creates and returns a configured logger.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # ------------------------------------------------------
    # Log Formatter
    # ------------------------------------------------------

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # ------------------------------------------------------
    # File Handler
    # ------------------------------------------------------

    file_handler = logging.FileHandler(LOG_FILE)

    file_handler.setLevel(logging.INFO)

    file_handler.setFormatter(formatter)

    # ------------------------------------------------------
    # Console Handler
    # ------------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)

    console_handler.setFormatter(formatter)

    # ------------------------------------------------------
    # Add Handlers
    # ------------------------------------------------------

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    logger.propagate = False

    return logger
