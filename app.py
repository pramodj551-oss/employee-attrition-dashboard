"""
==========================================================
Employee Attrition Dashboard

app.py

Author : Pramod Prakash Jadhav
==========================================================
"""
import io
from typing import Optional

import pandas as pd
import streamlit as st

from src.data_loader import load_data
from src.dashboard import (
    show_home,
    show_dataset,
    show_eda,
    show_prediction,
    show_feature_importance,
    show_reports,
    show_about,
)


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Helpers for loading dataset (disk or uploaded)
# ---------------------------------------------------------
@st.cache_data
def parse_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    """Parse uploaded CSV bytes into a DataFrame and cache by bytes content."""
    return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)


def get_dataset() -> Optional[pd.DataFrame]:
    """
    Try to load dataset from disk using load_data().
    If not found, prompt user to upload a CSV via Streamlit file_uploader.
    Returns a DataFrame or None.
    """
    df = load_data()  # load_data may return None if file not found
    if df is not None:
        return df

    st.warning(
        "Dataset not found on disk. Please upload 'employee_attrition.csv' "
        "or place it at data/raw/employee_attrition.csv."
    )

    uploaded = st.file_uploader("Upload employee_attrition.csv", type=["csv"])
    if uploaded is not None:
        try:
            df = parse_uploaded_csv(uploaded.getvalue())
            st.success("Dataset uploaded successfully.")
            return df
        except Exception as e:
            st.error(f"Failed to parse uploaded CSV: {e}")
            return None

    # No dataset available
    return None


# ---------------------------------------------------------
# Load Dataset (disk first, then uploader)
# ---------------------------------------------------------
df = get_dataset()
if df is None:
    # If no dataset available, stop further rendering (user must upload or add file)
    st.stop()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("📊 Employee Attrition Dashboard")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Explorer",
        "EDA",
        "Prediction",
        "Feature Importance",
        "Reports",
        "About",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
Author:
Pramod Prakash Jadhav

AI/ML Developer | Security Analyst
"""
)


# ---------------------------------------------------------
# Navigation
# ---------------------------------------------------------
if menu == "Home":
    show_home(df)

elif menu == "Dataset Explorer":
    show_dataset(df)

elif menu == "EDA":
    show_eda(df)

elif menu == "Prediction":
    show_prediction(df)

elif menu == "Feature Importance":
    show_feature_importance()

elif menu == "Reports":
    show_reports()

elif menu == "About":
    show_about()
