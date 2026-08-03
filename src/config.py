"""
==========================================================
Employee Attrition Dashboard

config.py

Author : Pramod Prakash Jadhav
==========================================================

Central configuration file for the Employee Attrition
Dashboard project.
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
# Model Directory
# ==========================================================

SCALER_PATH = MODELS_DIR / "scaler.pkl" FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.joblib" LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

# ==========================================================
# Output Directories
# ==========================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHARTS_DIR = OUTPUT_DIR / "charts"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOGS_DIR = OUTPUT_DIR / "logs"

# ==========================================================
# Assets
# ==========================================================

ASSETS_DIR = PROJECT_ROOT / "assets"

LOGO_PATH = ASSETS_DIR / "logo.png"
STYLE_PATH = ASSETS_DIR / "style.css"

# ==========================================================
# Dashboard Configuration
# ==========================================================

APP_TITLE = "Employee Attrition Dashboard"

PAGE_ICON = "📊"

LAYOUT = "wide"

SIDEBAR_STATE = "expanded"

# ==========================================================
# Random State
# ==========================================================

RANDOM_STATE = 42

# ==========================================================
# Create Required Directories (safe: non-fatal on failure)
# ==========================================================

DIRECTORIES = [
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    CHARTS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
]

for directory in DIRECTORIES:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Environment may be read-only (e.g., certain hosting platforms).
        # Avoid failing import; directory creation can be retried/handled in a setup step.
        # If you have an app-level logger, log this event there.
        pass
    except Exception:
        # Catch-all to prevent import-time failures. Handle explicitly in setup if needed.
        pass

# ==========================================================
# Default Chart Size
# ==========================================================

CHART_WIDTH = 900

CHART_HEIGHT = 500

# ==========================================================
# Dashboard Theme Colors
# ==========================================================

PRIMARY_COLOR = "#1F77B4"

SUCCESS_COLOR = "#2CA02C"

WARNING_COLOR = "#FF7F0E"

DANGER_COLOR = "#D62728"

# ==========================================================
# Logging
# ==========================================================

LOG_FILE = LOGS_DIR / "application.log"

LOG_LEVEL = "INFO"
