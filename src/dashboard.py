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
    get_dataset_summary,
    get_column_types,
    filter_dataset,
    download_dataset,
    load_dataset,  # NOTE: adjust this import to match your actual
                   # data_loader.py function name if it differs
                   # (e.g. read_data, load_data, get_dataframe, etc.)
)
from src.predictor import Predictor


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
        st.metric(
            "Missing Values",
            summary["Missing Values"]
        )
        st.metric(
            "Duplicate Rows",
            summary["Duplicate Rows"]
        )

    with col3:
        st.metric(
            "Numeric Features",
            summary["Numeric Features"]
        )
        st.metric(
            "Categorical Features",
            summary["Categorical Features"]
        )

    st.markdown("---")

    st.subheader("📈 Quick Insights")

    chart = DashboardCharts.attrition_distribution(df)

    st.plotly_chart(
        chart,
        use_container_width=True
    )

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

    st.write(df)

    st.markdown("---")

    numeric_cols, categorical_cols = get_column_types(df)

    department = None
    gender = None
    attrition = None

    if "Department" in df.columns:

        department = st.selectbox(
            "Department",
            ["All"] + sorted(
                df["Department"].unique().tolist()
            )
        )

        if department == "All":
            department = None

    if "Gender" in df.columns:

        gender = st.selectbox(
            "Gender",
            ["All"] + sorted(
                df["Gender"].unique().tolist()
            )
        )

        if gender == "All":
            gender = None

    if "Attrition" in df.columns:

        attrition = st.selectbox(
            "Attrition",
            ["All"] + sorted(
                df["Attrition"].unique().tolist()
            )
        )

        if attrition == "All":
            attrition = None

    filtered_df = filter_dataset(
        df,
        department,
        gender,
        attrition,
    )

    st.markdown("### Filtered Dataset")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

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
            DashboardCharts.attrition_distribution(df),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            DashboardCharts.department_attrition(df),
            use_container_width=True,
        )

    # Row 2
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            DashboardCharts.gender_distribution(df),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            DashboardCharts.overtime_analysis(df),
            use_container_width=True,
        )

    # Row 3
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            DashboardCharts.monthly_income_distribution(df),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            DashboardCharts.age_distribution(df),
            use_container_width=True,
        )

    # Row 4
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            DashboardCharts.job_role_distribution(df),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            DashboardCharts.education_field_distribution(df),
            use_container_width=True,
        )

    st.markdown("---")

    st.subheader("Correlation Heatmap")

    st.plotly_chart(
        DashboardCharts.correlation_heatmap(df),
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

    predictor = Predictor()

    st.markdown(
        "Enter employee details and click **Predict**."
    )

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
            sorted(df["Gender"].unique()),
        )

        department = st.selectbox(
            "Department",
            sorted(df["Department"].unique()),
        )

        job_role = st.selectbox(
            "Job Role",
            sorted(df["JobRole"].unique()),
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
            sorted(df["OverTime"].unique()),
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

            prediction = predictor.predict_label(input_df)

            confidence = predictor.predict_confidence(
                input_df
            )

            st.markdown("---")

            if prediction == "Yes":

                st.error(
                    "⚠ High Attrition Risk"
                )

            else:

                st.success(
                    "✅ Low Attrition Risk"
                )

            st.metric(
                "Prediction",
                prediction,
            )

            if confidence is not None:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%",
                )

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

    predictor = Predictor()

    importance = predictor.get_feature_importance()

    if importance is None:

        st.warning(
            "The loaded model does not support feature importance."
        )

        return

    try:

        feature_names = predictor.model.feature_names_in_

    except AttributeError:

        st.info(
            "Feature names are not available in the saved model."
        )

        return

    chart = DashboardCharts.feature_importance(
        feature_names,
        importance,
    )

    st.plotly_chart(
        chart,
        use_container_width=True,
    )

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
    classification_file = (
        report_dir / "classification_report.txt"
    )

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

    st.success(
        "Thank you for exploring the Employee Attrition Dashboard!"
    )


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

    # NOTE: load_dataset() is assumed to live in src/data_loader.py
    # and return a pandas DataFrame (e.g. reading a CSV from disk
    # or a data/ folder). Rename/adjust this call to match whatever
    # your actual loader function is called.
    df = load_dataset()

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
