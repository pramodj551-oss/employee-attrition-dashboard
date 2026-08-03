"""
==========================================================
Employee Attrition Dashboard

app.py

Author : Pramod Prakash Jadhav
==========================================================
"""

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
# Load Dataset
# ---------------------------------------------------------

# Replace current load_dataset / df = load_dataset() block in app.py with:

@st.cache_data
def load_dataset_from_file(file_bytes) -> pd.DataFrame:
    # called when user uploads a CSV; cached by the bytes
    import io
    return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)

# Attempt to load from disk first
df = load_dataset()

if df is None:
    st.warning("Dataset not found on disk. Please upload employee_attrition.csv to continue, or place it at data/raw/employee_attrition.csv.")
    uploaded_file = st.file_uploader("Upload employee_attrition.csv", type=["csv"])
    if uploaded_file is not None:
        # read uploaded file bytes and cache
        file_bytes = uploaded_file.getvalue()
        df = load_dataset_from_file(file_bytes)
    else:
        # show a friendly message and stop further rendering
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
