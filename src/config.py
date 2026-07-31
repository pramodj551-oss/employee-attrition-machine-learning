"""
==========================================================
Employee Attrition Machine Learning

config.py

Author : Pramod Prakash Jadhav
==========================================================
"""

from pathlib import Path

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# Data Directories
# ==========================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ==========================================================
# Dataset
# ==========================================================

DATASET_PATH = RAW_DATA_DIR / "employee_attrition.csv"

# ==========================================================
# Model Directories
# ==========================================================

MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "best_model.pkl"

PIPELINE_PATH = MODELS_DIR / "pipeline.pkl"

FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"

SCALER_PATH = MODELS_DIR / "scaler.pkl"

LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

# ==========================================================
# Outputs
# ==========================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

REPORTS_DIR = OUTPUT_DIR / "reports"

LOGS_DIR = OUTPUT_DIR / "logs"

PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

# ==========================================================
# Reports
# ==========================================================

CLASSIFICATION_REPORT_PATH = (
    REPORTS_DIR / "classification_report.txt"
)

CONFUSION_MATRIX_PATH = (
    REPORTS_DIR / "confusion_matrix.png"
)

FEATURE_IMPORTANCE_PATH = (
    REPORTS_DIR / "feature_importance.csv"
)

PREDICTIONS_PATH = (
    PREDICTIONS_DIR / "predictions.csv"
)

# ==========================================================
# Random State
# ==========================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

# ==========================================================
# Logging
# ==========================================================

LOG_LEVEL = "INFO"

LOG_FILE = LOGS_DIR / "application.log"

# ==========================================================
# Create Required Directories
# ==========================================================

DIRECTORIES = [

    RAW_DATA_DIR,

    PROCESSED_DATA_DIR,

    MODELS_DIR,

    OUTPUT_DIR,

    REPORTS_DIR,

    LOGS_DIR,

    PREDICTIONS_DIR,

]

for directory in DIRECTORIES:

    directory.mkdir(
        parents=True,
        exist_ok=True,
)
