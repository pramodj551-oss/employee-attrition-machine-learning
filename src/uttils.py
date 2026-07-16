"""
==========================================================
Employee Attrition Machine Learning

utils.py

Author : Pramod Prakash Jadhav
==========================================================
"""

from pathlib import Path
from datetime import datetime

import joblib

from src.logger import get_logger

logger = get_logger()


# ==========================================================
# Create Directory
# ==========================================================

def create_directory(path):
    """
    Create a directory if it does not exist.
    """

    Path(path).mkdir(parents=True, exist_ok=True)

    logger.info(f"Directory ready: {path}")


# ==========================================================
# Save Model
# ==========================================================

def save_model(model, file_path):
    """
    Save trained machine learning model.
    """

    joblib.dump(model, file_path)

    logger.info(f"Model saved successfully: {file_path}")


# ==========================================================
# Load Model
# ==========================================================

def load_model(file_path):
    """
    Load saved machine learning model.
    """

    model = joblib.load(file_path)

    logger.info(f"Model loaded successfully: {file_path}")

    return model


# ==========================================================
# Current Timestamp
# ==========================================================

def current_timestamp():
    """
    Return current timestamp.
    """

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==========================================================
# Print Section Header
# ==========================================================

def print_header(title):
    """
    Print formatted section header.
    """

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ==========================================================
# Print Dataset Shape
# ==========================================================

def print_shape(df):
    """
    Print dataset shape.
    """

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")


# ==========================================================
# Separator
# ==========================================================

def separator():

    print("-" * 60)


# ==========================================================
# Execution Check
# ==========================================================

if __name__ == "__main__":

    print_header("Utility Module")

    print(current_timestamp())

    separator()

    logger.info("utils.py executed successfully.")
