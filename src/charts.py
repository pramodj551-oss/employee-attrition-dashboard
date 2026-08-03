"""
==========================================================
Employee Attrition Dashboard

charts.py

Author : Pramod Prakash Jadhav
==========================================================

Interactive Plotly chart utilities.
"""

import pandas as pd
import plotly.express as px
from typing import Optional

from src.logger import logger


class DashboardCharts:
    """
    Generate interactive charts for the dashboard.
    These functions avoid scanning the entire DataFrame unnecessarily and
    add lightweight sampling/limits for very large datasets.
    """

    # Thresholds for when to downsample large datasets for plotting
    _HIST_SAMPLE_THRESHOLD = 50_000
    _HIST_SAMPLE_SIZE = 50_000
    _CORR_MAX_COLUMNS = 25

    @staticmethod
    def _safe_sample_series(series: pd.Series, max_rows: int, random_state: int = 42) -> pd.Series:
        """Return a sampled series if length exceeds max_rows; otherwise return original (no copy)."""
        if series is None:
            return series
        series = series.dropna()
        if len(series) > max_rows:
            return series.sample(n=max_rows, random_state=random_state)
        return series

    @staticmethod
    def attrition_distribution(df: Optional[pd.DataFrame]):
        logger.info("Creating Attrition Distribution chart.")
        if df is None or df.empty:
            return px.pie(title="Employee Attrition Distribution (no data)")

        counts = df["Attrition"].value_counts(dropna=False).rename_axis("Attrition").reset_index(name="Count")
        return px.pie(
            counts,
            names="Attrition",
            values="Count",
            title="Employee Attrition Distribution",
            hole=0.45,
        )

    @staticmethod
    def department_attrition(df: Optional[pd.DataFrame]):
        logger.info("Creating Department chart.")
        if df is None or df.empty or "Department" not in df.columns:
            return px.bar(title="Employees by Department (no data)")

        data = df["Department"].value_counts(dropna=False).rename_axis("Department").reset_index(name="Employees")
        return px.bar(
            data,
            x="Department",
            y="Employees",
            title="Employees by Department",
        )

    @staticmethod
    def gender_distribution(df: Optional[pd.DataFrame]):
        logger.info("Creating Gender Distribution chart.")
        if df is None or df.empty or "Gender" not in df.columns:
            return px.bar(title="Gender Distribution (no data)")

        data = df["Gender"].value_counts(dropna=False).rename_axis("Gender").reset_index(name="Employees")
        return px.bar(
            data,
            x="Gender",
            y="Employees",
            title="Gender Distribution",
        )

    @staticmethod
    def overtime_analysis(df: Optional[pd.DataFrame]):
        logger.info("Creating Overtime Analysis chart.")
        if df is None or df.empty or "OverTime" not in df.columns:
            return px.bar(title="Overtime Analysis (no data)")

        data = df["OverTime"].value_counts(dropna=False).rename_axis("OverTime").reset_index(name="Employees")
        return px.bar(
            data,
            x="OverTime",
            y="Employees",
            title="Overtime Analysis",
        )

    @staticmethod
    def monthly_income_distribution(df: Optional[pd.DataFrame]):
        logger.info("Creating Monthly Income Distribution.")
        if df is None or df.empty or "MonthlyIncome" not in df.columns:
            return px.histogram(title="Monthly Income Distribution (no data)")

        series = df["MonthlyIncome"]
        series_sample = DashboardCharts._safe_sample_series(
            series, DashboardCharts._HIST_SAMPLE_THRESHOLD, random_state=42
        )
        data = pd.DataFrame({"MonthlyIncome": series_sample})
        # nbins stays small to reduce complexity on client
        return px.histogram(
            data,
            x="MonthlyIncome",
            nbins=30,
            title="Monthly Income Distribution",
        )

    @staticmethod
    def age_distribution(df: Optional[pd.DataFrame]):
        logger.info("Creating Age Distribution.")
        if df is None or df.empty or "Age" not in df.columns:
            return px.histogram(title="Age Distribution (no data)")

        series = df["Age"]
        series_sample = DashboardCharts._safe_sample_series(
            series, DashboardCharts._HIST_SAMPLE_THRESHOLD, random_state=42
        )
        data = pd.DataFrame({"Age": series_sample})
        return px.histogram(
            data,
            x="Age",
            nbins=20,
            title="Age Distribution",
        )

    @staticmethod
    def job_role_distribution(df: Optional[pd.DataFrame]):
        logger.info("Creating Job Role chart.")
        if df is None or df.empty or "JobRole" not in df.columns:
            return px.bar(title="Employees by Job Role (no data)")

        data = df["JobRole"].value_counts(dropna=False).rename_axis("JobRole").reset_index(name="Employees")
        return px.bar(
            data,
            x="JobRole",
            y="Employees",
            title="Employees by Job Role",
        )

    @staticmethod
    def education_field_distribution(df: Optional[pd.DataFrame]):
        logger.info("Creating Education Field chart.")
        if df is None or df.empty or "EducationField" not in df.columns:
            return px.bar(title="Education Field Distribution (no data)")

        data = df["EducationField"].value_counts(dropna=False).rename_axis("EducationField").reset_index(name="Employees")
        return px.bar(
            data,
            x="EducationField",
            y="Employees",
            title="Education Field Distribution",
        )

    @staticmethod
    def correlation_heatmap(df: Optional[pd.DataFrame]):
        logger.info("Creating Correlation Heatmap.")
        if df is None or df.empty:
            return px.imshow([], title="Correlation Heatmap (no data)")

        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] == 0:
            return px.imshow([], title="Correlation Heatmap (no numeric columns)")

        # If too many numeric columns, select top columns by variance to limit O(N^2) work
        if numeric.shape[1] > DashboardCharts._CORR_MAX_COLUMNS:
            variances = numeric.var(numeric_only=True).sort_values(ascending=False)
            top_cols = variances.head(DashboardCharts._CORR_MAX_COLUMNS).index.tolist()
            numeric = numeric[top_cols]

        correlation = numeric.corr()

        return px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Heatmap",
        )

    @staticmethod
    def feature_importance(feature_names, importance):
        logger.info("Creating Feature Importance chart.")
        if feature_names is None or importance is None:
            return px.bar(title="Feature Importance (no data)")

        data = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importance,
            }
        )

        data = data.sort_values(by="Importance", ascending=False).head(20)

        return px.bar(
            data,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 20 Feature Importance",
        )
