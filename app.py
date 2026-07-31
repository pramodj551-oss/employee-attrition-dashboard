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

df = load_data()

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
