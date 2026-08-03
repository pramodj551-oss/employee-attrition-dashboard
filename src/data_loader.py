"""
==========================================================
Employee Attrition Dashboard

data_loader.py

Author : Pramod Prakash Jadhav
==========================================================

Load and validate the Employee Attrition dataset.
"""


from pathlib import Path
import pandas as pd
import streamlit as st
from src.config import DATASET_PATH
from src.logger import logger

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    Load the employee attrition dataset (prefer parquet for speed & memory).
    Caches the loaded dataset for the session.
    Returns None on error so we don't cache empty DataFrames unintentionally.
    """
    try:
        logger.info("Loading dataset...")

        if not Path(DATASET_PATH).exists():
            raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

        # Prefer fast columnar format if available
        path = Path(DATASET_PATH)
        if path.suffix in (".parquet", ".pq", ".feather"):
            df = pd.read_parquet(path)
        else:
            # CSV fallback: tune parsing for performance
            # Provide dtype mapping if known. Example: dtype = {"col1": "int32", ...}
            df = pd.read_csv(path, low_memory=False)

        # Convert low-cardinality object columns to category to reduce memory
        for col in df.select_dtypes(include="object").columns:
            if df[col].nunique(dropna=False) / max(1, len(df)) < 0.5:
                df[col] = df[col].astype("category")

        logger.info("Dataset loaded successfully. Shape: %s", df.shape)
        return df

    except Exception as error:
        logger.exception("Failed to load dataset.")
        st.error(f"Error loading dataset: {error}")
        # Return None (not cached as a DataFrame) so UI can handle retry/notify
        return None


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Generate dataset summary. Not cached here to avoid hashing large DataFrame objects.
    """
    if df is None or df.empty:
        return {
            "Rows": 0,
            "Columns": 0,
            "Missing Values": 0,
            "Duplicate Rows": 0,
            "Numeric Features": 0,
            "Categorical Features": 0,
        }

    # Lightweight one-pass computations
    rows, cols = df.shape
    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    numeric = len(df.select_dtypes(include="number").columns)
    categorical = len(df.select_dtypes(exclude="number").columns)

    return {
        "Rows": rows,
        "Columns": cols,
        "Missing Values": missing,
        "Duplicate Rows": duplicates,
        "Numeric Features": numeric,
        "Categorical Features": categorical,
    }


def get_column_types(df: pd.DataFrame):
    """Return numeric and categorical columns. Not cached to avoid expensive hashing."""
    if df is None or df.empty:
        return [], []

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    # Prefer explicit object/category selection for categorical columns
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_columns, categorical_columns


def filter_dataset(df: pd.DataFrame, department=None, gender=None, attrition=None):
    """
    Filter dataset based on user selections.
    Avoid an unconditional df.copy() to save memory; return a view unless a copy is required.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    mask = pd.Series(True, index=df.index)

    if department:
        mask &= df["Department"] == department

    if gender:
        mask &= df["Gender"] == gender

    if attrition:
        mask &= df["Attrition"] == attrition

    # Return a view (no unnecessary copy). If downstream mutates the result, call .copy() there.
    return df.loc[mask]


def download_dataset(df: pd.DataFrame):
    """
    Convert dataframe to CSV bytes for download.
    """
    if df is None or df.empty:
        return "".encode("utf-8")
    return df.to_csv(index=False).encode("utf-8")
