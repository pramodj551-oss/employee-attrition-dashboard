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
    Load the employee attrition dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    try:

        logger.info("Loading dataset...")

        if not Path(DATASET_PATH).exists():
            raise FileNotFoundError(
                f"Dataset not found: {DATASET_PATH}"
            )

        df = pd.read_csv(DATASET_PATH)

        logger.info(
            "Dataset loaded successfully. Shape: %s",
            df.shape
        )

        return df

    except Exception as error:

        logger.exception("Failed to load dataset.")

        st.error(f"Error loading dataset: {error}")

        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Generate dataset summary.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
    """

    summary = {

        "Rows": df.shape[0],

        "Columns": df.shape[1],

        "Missing Values": int(df.isnull().sum().sum()),

        "Duplicate Rows": int(df.duplicated().sum()),

        "Numeric Features":
            len(df.select_dtypes(include="number").columns),

        "Categorical Features":
            len(df.select_dtypes(exclude="number").columns),
    }

    return summary


@st.cache_data(show_spinner=False)
def get_column_types(df: pd.DataFrame):
    """
    Return numeric and categorical columns.
    """

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    return numeric_columns, categorical_columns


@st.cache_data(show_spinner=False)
def filter_dataset(
    df: pd.DataFrame,
    department=None,
    gender=None,
    attrition=None,
):
    """
    Filter dataset based on user selections.
    """

    filtered_df = df.copy()

    if department:

        filtered_df = filtered_df[
            filtered_df["Department"] == department
        ]

    if gender:

        filtered_df = filtered_df[
            filtered_df["Gender"] == gender
        ]

    if attrition:

        filtered_df = filtered_df[
            filtered_df["Attrition"] == attrition
        ]

    return filtered_df


def download_dataset(df: pd.DataFrame):
    """
    Convert dataframe to CSV bytes for download.
    """

    return df.to_csv(index=False).encode("utf-8")
