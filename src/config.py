"""
==========================================================
Employee Attrition Machine Learning

config.py

Author : Pramod Prakash Jadhav
==========================================================
"""

from pathlib import Path

# ==========================================================
# Project Root Directory
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

# ==========================================================
# Data Directories
# ==========================================================

DATA_DIR = ROOT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ==========================================================
# Dataset
# ==========================================================

DATASET_PATH = RAW_DATA_DIR / "employee_attrition.csv"

# ==========================================================
# Output Directories
# ==========================================================

OUTPUT_DIR = ROOT_DIR / "outputs"

CHARTS_DIR = OUTPUT_DIR / "charts"

REPORTS_DIR = OUTPUT_DIR / "reports"

METRICS_DIR = OUTPUT_DIR / "metrics"

LOGS_DIR = OUTPUT_DIR / "logs"

# ==========================================================
# Model Directory
# ==========================================================

MODEL_DIR = ROOT_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_model.pkl"

# ==========================================================
# Machine Learning Configuration
# ==========================================================

TARGET_COLUMN = "Attrition"

TEST_SIZE = 0.20

RANDOM_STATE = 42

# ==========================================================
# Model Names
# ==========================================================

MODEL_NAMES = [

    "Logistic Regression",

    "Decision Tree",

    "Random Forest",

    "Gradient Boosting",

    "K-Nearest Neighbors"

]

# ==========================================================
# Create Required Directories
# ==========================================================

DIRECTORIES = [

    PROCESSED_DATA_DIR,

    CHARTS_DIR,

    REPORTS_DIR,

    METRICS_DIR,

    LOGS_DIR,

    MODEL_DIR

]

for directory in DIRECTORIES:

    directory.mkdir(parents=True, exist_ok=True)
