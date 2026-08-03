"""
==========================================================
Employee Attrition Dashboard

dashboard.py

Author : Pramod Prakash Jadhav
==========================================================
"""

import streamlit as st

from src.charts import DashboardCharts

from src.data_loader import (
    load_data,
    get_dataset_summary,
    get_column_types,
    filter_dataset,
    download_dataset,
)

from src.predictor import Predictor


# ---------------------------
# Helpers and cached resources
# ---------------------------

@st.cache_resource
def get_predictor():
    """Create and cache a single Predictor instance (avoid repeated model loads)."""
    return Predictor()


def _df_fingerprint(df):
    """
    Lightweight fingerprint for a DataFrame used as a cache key:
    - shape and column names only (cheap, avoids hashing full data)
    """
    try:
        return (df.shape, tuple(df.columns))
    except Exception:
        # Fallback if df is None or not a DataFrame
        return (0, ())


def cached_chart(func_name: str, df, *args, **kwargs):
    """
    Cache generated charts in session_state using a lightweight key.
    func_name should be the DashboardCharts method name as string.
    """
    cache = st.session_state.setdefault("_chart_cache", {})
    key = (func_name, _df_fingerprint(df), str(args), str(sorted(kwargs.items())))
    if key not in cache:
        chart_func = getattr(DashboardCharts, func_name)
        cache[key] = chart_func(df, *args, **kwargs)
    return cache[key]


def _unique_sorted(df, col):
    """Return sorted unique values for a column (safe: returns [] on error)."""
    try:
        return sorted(df[col].dropna().unique().tolist())
    except Exception:
        return []


# ==========================================================
# Home Page
# ==========================================================

def show_home(df):
    """
    Display dashboard home page.
    """

    st.title("📊 Employee Attrition Dashboard")

    st.markdown(
        """
Welcome to the **Employee Attrition Dashboard**.

This dashboard helps HR professionals explore employee data,
analyze attrition trends, and predict whether an employee is
likely to leave the organization.
"""
    )

    summary = get_dataset_summary(df)

    st.subheader("📌 Dataset Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", summary["Rows"])
        st.metric("Columns", summary["Columns"])

    with col2:
        st.metric("Missing Values", summary["Missing Values"])
        st.metric("Duplicate Rows", summary["Duplicate Rows"])

    with col3:
        st.metric("Numeric Features", summary["Numeric Features"])
        st.metric("Categorical Features", summary["Categorical Features"])

    st.markdown("---")

    st.subheader("📈 Quick Insights")

    # Use cached chart generator to avoid recomputing heavy charts repeatedly
    chart = cached_chart("attrition_distribution", df)

    st.plotly_chart(chart, use_container_width=True)

    st.markdown("---")

    st.info(
        """
Use the navigation menu on the left to:

• Explore Dataset

• Perform EDA

• Predict Employee Attrition

• View Feature Importance

• Download Reports
"""
    )


# ==========================================================
# Dataset Explorer
# ==========================================================

def show_dataset(df):
    """
    Display dataset explorer.
    """

    st.title("📁 Dataset Explorer")

    # Show a preview by default to avoid sending very large dataframes to the browser
    st.markdown("#### Data Preview (first 100 rows)")
    if df is None:
        st.info("No dataset loaded.")
        return

    st.dataframe(df.head(100), use_container_width=True)

    if st.checkbox("Show full dataset (may be slow)", value=False):
        st.dataframe(df, use_container_width=True)

    st.markdown("---")

    numeric_cols, categorical_cols = get_column_types(df)

    # Precompute unique lists once to avoid repeated scans on reruns
    departments = _unique_sorted(df, "Department") if "Department" in df.columns else []
    genders = _unique_sorted(df, "Gender") if "Gender" in df.columns else []
    attritions = _unique_sorted(df, "Attrition") if "Attrition" in df.columns else []
    jobroles = _unique_sorted(df, "JobRole") if "JobRole" in df.columns else []
    overtimes = _unique_sorted(df, "OverTime") if "OverTime" in df.columns else []

    # Default selections
    department = None
    gender = None
    attrition = None

    if departments:
        department = st.selectbox("Department", ["All"] + departments)
        if department == "All":
            department = None

    if genders:
        gender = st.selectbox("Gender", ["All"] + genders)
        if gender == "All":
            gender = None

    if attritions:
        attrition = st.selectbox("Attrition", ["All"] + attritions)
        if attrition == "All":
            attrition = None

    filtered_df = filter_dataset(df, department, gender, attrition)

    st.markdown("### Filtered Dataset")

    # Show limited rows by default; allow user to expand to full if desired
    max_rows = 500
    if filtered_df is None or filtered_df.empty:
        st.info("No rows match the selected filters.")
    else:
        if filtered_df.shape[0] > max_rows:
            st.info(f"Filtered dataset has {filtered_df.shape[0]} rows; showing first {max_rows}.")
            st.dataframe(filtered_df.head(max_rows), use_container_width=True)
            if st.checkbox("Show all filtered rows"):
                st.dataframe(filtered_df, use_container_width=True)
        else:
            st.dataframe(filtered_df, use_container_width=True)

    st.markdown("---")

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Numeric Columns")
        st.write(numeric_cols)

    with col2:
        st.write("Categorical Columns")
        st.write(categorical_cols)

    st.markdown("---")

    csv = download_dataset(filtered_df)

    st.download_button(
        label="⬇ Download Filtered Dataset",
        data=csv,
        file_name="filtered_employee_attrition.csv",
        mime="text/csv",
    )


# ==========================================================
# EDA Page
# ==========================================================

def show_eda(df):
    """
    Display Exploratory Data Analysis dashboard.
    """

    st.title("📈 Exploratory Data Analysis")

    st.markdown("### Employee Attrition Analysis")

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            cached_chart("attrition_distribution", df),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            cached_chart("department_attrition", df),
            use_container_width=True,
        )

    # Row 2
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            cached_chart("gender_distribution", df),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            cached_chart("overtime_analysis", df),
            use_container_width=True,
        )

    # Row 3
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            cached_chart("monthly_income_distribution", df),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            cached_chart("age_distribution", df),
            use_container_width=True,
        )

    # Row 4
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            cached_chart("job_role_distribution", df),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            cached_chart("education_field_distribution", df),
            use_container_width=True,
        )

    st.markdown("---")

    st.subheader("Correlation Heatmap")

    st.plotly_chart(
        cached_chart("correlation_heatmap", df),
        use_container_width=True,
    )


# ==========================================================
# Prediction Page
# ==========================================================

def show_prediction(df):
    """
    Employee Attrition Prediction Page.
    """

    st.title("🤖 Employee Attrition Prediction")

    predictor = get_predictor()

    st.markdown("Enter employee details and click **Predict**.")

    # Precompute widget choices
    genders = _unique_sorted(df, "Gender") if "Gender" in df.columns else []
    departments = _unique_sorted(df, "Department") if "Department" in df.columns else []
    jobroles = _unique_sorted(df, "JobRole") if "JobRole" in df.columns else []
    overtimes = _unique_sorted(df, "OverTime") if "OverTime" in df.columns else []

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=60,
            value=30,
        )

        gender = st.selectbox(
            "Gender",
            genders if genders else ["Unknown"],
        )

        department = st.selectbox(
            "Department",
            departments if departments else ["Unknown"],
        )

        job_role = st.selectbox(
            "Job Role",
            jobroles if jobroles else ["Unknown"],
        )

    with col2:

        monthly_income = st.number_input(
            "Monthly Income",
            min_value=1000,
            value=5000,
        )

        years_at_company = st.number_input(
            "Years At Company",
            min_value=0,
            value=5,
        )

        overtime = st.selectbox(
            "OverTime",
            overtimes if overtimes else ["Unknown"],
        )

        job_satisfaction = st.slider(
            "Job Satisfaction",
            1,
            4,
            3,
        )

    if st.button("Predict Attrition"):

        import pandas as pd

        input_df = pd.DataFrame(
            {
                "Age": [age],
                "Gender": [gender],
                "Department": [department],
                "JobRole": [job_role],
                "MonthlyIncome": [monthly_income],
                "YearsAtCompany": [years_at_company],
                "OverTime": [overtime],
                "JobSatisfaction": [job_satisfaction],
            }
        )

        try:

            # Predictor instance is cached; ensure its internals do not re-load models repeatedly
            prediction = predictor.predict_label(input_df)
            confidence = predictor.predict_confidence(input_df)

            st.markdown("---")

            if prediction == "Yes":
                st.error("⚠ High Attrition Risk")
            else:
                st.success("✅ Low Attrition Risk")

            st.metric("Prediction", prediction)

            if confidence is not None:
                st.metric("Confidence", f"{confidence:.2f}%")

        except Exception as error:
            st.error(error)


# ==========================================================
# Feature Importance Page
# ==========================================================

def show_feature_importance():
    """
    Display feature importance.
    """

    st.title("⭐ Feature Importance")

    predictor = get_predictor()

    importance = predictor.get_feature_importance()

    if importance is None:
        st.warning("The loaded model does not support feature importance.")
        return

    try:
        feature_names = predictor.model.feature_names_in_
    except AttributeError:
        st.info("Feature names are not available in the saved model.")
        return

    chart = DashboardCharts.feature_importance(feature_names, importance)

    st.plotly_chart(chart, use_container_width=True)

    st.dataframe(
        {
            "Feature": feature_names,
            "Importance": importance,
        },
        use_container_width=True,
    )


# ==========================================================
# Reports Page
# ==========================================================

def show_reports():
    """
    Display generated reports.
    """

    st.title("📄 Reports")

    st.markdown(
        """
Download reports generated by the machine learning pipeline.
"""
    )

    from pathlib import Path

    report_dir = Path("outputs/reports")

    prediction_file = report_dir / "predictions.csv"
    classification_file = report_dir / "classification_report.txt"

    if prediction_file.exists():
        with open(prediction_file, "rb") as file:
            st.download_button(
                label="⬇ Download Predictions",
                data=file,
                file_name="predictions.csv",
                mime="text/csv",
            )
    else:
        st.info("Prediction report not found.")

    if classification_file.exists():
        with open(classification_file, "rb") as file:
            st.download_button(
                label="⬇ Download Classification Report",
                data=file,
                file_name="classification_report.txt",
                mime="text/plain",
            )
    else:
        st.info("Classification report not found.")


# ==========================================================
# About Page
# ==========================================================

def show_about():
    """
    Display project information.
    """

    st.title("ℹ️ About")

    st.markdown(
        """
## Employee Attrition Dashboard

A production-ready Human Resource Analytics Dashboard
developed using:

- Python
- Streamlit
- Scikit-Learn
- Plotly
- Pandas

This project predicts employee attrition and provides
interactive business insights.
"""
    )

    st.markdown("---")

    st.subheader("Developer")

    st.write("**Pramod Prakash Jadhav**")

    st.write("AI/ML Developer | Security Analyst")

    st.markdown("---")

    st.subheader("Technology Stack")

    st.markdown(
        """
- Python
- Streamlit
- Plotly
- Pandas
- NumPy
- Scikit-Learn
- Joblib
"""
    )

    st.markdown("---")

    st.subheader("Project Features")

    st.markdown(
        """
✅ Interactive Dashboard

✅ Employee Dataset Explorer

✅ Exploratory Data Analysis

✅ Employee Attrition Prediction

✅ Feature Importance

✅ Downloadable Reports

✅ Production-ready Architecture
"""
    )

    st.markdown("---")

    st.success("Thank you for exploring the Employee Attrition Dashboard!")


# ==========================================================
# App Entry Point
# ==========================================================

def main():
    """
    Configure the page, load data, render sidebar navigation,
    and dispatch to the selected page.
    """

    st.set_page_config(
        page_title="Employee Attrition Dashboard",
        page_icon="📊",
        layout="wide",
    )

    # Load dataset (use load_data from src.data_loader)
    df = load_data()

    st.sidebar.title("📊 Navigation")

    page = st.sidebar.radio(
        "Go to",
        [
            "Home",
            "Dataset Explorer",
            "EDA",
            "Predict Attrition",
            "Feature Importance",
            "Reports",
            "About",
        ],
    )

    if page == "Home":
        show_home(df)
    elif page == "Dataset Explorer":
        show_dataset(df)
    elif page == "EDA":
        show_eda(df)
    elif page == "Predict Attrition":
        show_prediction(df)
    elif page == "Feature Importance":
        show_feature_importance()
    elif page == "Reports":
        show_reports()
    elif page == "About":
        show_about()


if __name__ == "__main__":
    main()
