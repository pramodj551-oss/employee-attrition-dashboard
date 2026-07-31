"""
==========================================================
Employee Attrition Dashboard

utils.py

Author : Pramod Prakash Jadhav
==========================================================

Common utility functions used across the dashboard.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import DATASET_PATH, MODEL_PATH


# ==========================================================
# File Helpers
# ==========================================================

def file_exists(file_path) -> bool:
    """
    Check whether a file exists.

    Parameters
    ----------
    file_path : str | Path

    Returns
    -------
    bool
    """
    return Path(file_path).exists()


def validate_project_files() -> dict:
    """
    Validate important project files.

    Returns
    -------
    dict
        Status of important files.
    """

    return {
        "Dataset": file_exists(DATASET_PATH),
        "Model": file_exists(MODEL_PATH),
    }


# ==========================================================
# Formatting Helpers
# ==========================================================

def format_number(value):
    """
    Format integer with comma separator.
    """

    try:
        return f"{int(value):,}"
    except Exception:
        return value


def format_percentage(value):
    """
    Convert decimal to percentage.
    """

    try:
        return f"{value * 100:.2f}%"
    except Exception:
        return value


# ==========================================================
# Dataset Helpers
# ==========================================================

def dataframe_to_csv(df: pd.DataFrame):
    """
    Convert dataframe into downloadable CSV.
    """

    return df.to_csv(index=False).encode("utf-8")


def dataset_overview(df: pd.DataFrame):
    """
    Return basic dataset information.
    """

    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
    }


# ==========================================================
# KPI Cards
# ==========================================================

def show_kpi_cards(df: pd.DataFrame):
    """
    Display dashboard KPI metrics.
    """

    overview = dataset_overview(df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rows",
            format_number(overview["Rows"])
        )

    with col2:
        st.metric(
            "Columns",
            overview["Columns"]
        )

    with col3:
        st.metric(
            "Missing",
            overview["Missing Values"]
        )

    with col4:
        st.metric(
            "Duplicates",
            overview["Duplicate Rows"]
        )


# ==========================================================
# Download Button
# ==========================================================

def download_dataframe(df: pd.DataFrame):
    """
    Display CSV download button.
    """

    st.download_button(
        label="⬇ Download CSV",
        data=dataframe_to_csv(df),
        file_name="employee_attrition.csv",
        mime="text/csv",
)
  # ==========================================================
# Prediction Helpers
# ==========================================================

import pandas as pd
import streamlit as st


def align_features(
    input_df: pd.DataFrame,
    feature_columns: list,
) -> pd.DataFrame:
    """
    Align prediction input with training features.

    Parameters
    ----------
    input_df : pd.DataFrame
        User input dataframe.
    feature_columns : list
        Feature columns used during model training.

    Returns
    -------
    pd.DataFrame
    """

    encoded = pd.get_dummies(
        input_df,
        drop_first=True
    )

    encoded = encoded.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    return encoded


# ==========================================================
# Prediction Formatting
# ==========================================================

def prediction_label(prediction: int) -> str:
    """
    Convert numeric prediction into readable label.
    """

    return "Yes" if prediction == 1 else "No"


def prediction_color(prediction: int) -> str:
    """
    Return color according to prediction.
    """

    return "red" if prediction == 1 else "green"


def format_confidence(confidence: float) -> str:
    """
    Format prediction confidence.
    """

    return f"{confidence:.2f}%"


# ==========================================================
# Model Information
# ==========================================================

def show_model_information(model):
    """
    Display model information.
    """

    st.subheader("Model Information")

    st.write(f"**Model:** {type(model).__name__}")

    st.write(
        f"**Predict Supported:** "
        f"{hasattr(model, 'predict')}"
    )

    st.write(
        f"**Predict Probability:** "
        f"{hasattr(model, 'predict_proba')}"
    )

    st.write(
        f"**Feature Importance:** "
        f"{hasattr(model, 'feature_importances_')}"
    )


# ==========================================================
# Dashboard Status
# ==========================================================

def show_project_status():
    """
    Display project status.
    """

    status = validate_project_files()

    st.subheader("Project Status")

    if status["Dataset"]:
        st.success("Dataset Available")
    else:
        st.error("Dataset Missing")

    if status["Model"]:
        st.success("Model Available")
    else:
        st.error("Model Missing")


# ==========================================================
# Alert Messages
# ==========================================================

def success_message(message: str):
    """
    Display success message.
    """
    st.success(message)


def warning_message(message: str):
    """
    Display warning message.
    """
    st.warning(message)


def error_message(message: str):
    """
    Display error message.
    """
    st.error(message)


def info_message(message: str):
    """
    Display information message.
    """
    st.info(message)
