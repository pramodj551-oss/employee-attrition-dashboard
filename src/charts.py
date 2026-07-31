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

from src.logger import logger


class DashboardCharts:
    """
    Generate interactive charts for the dashboard.
    """

    @staticmethod
    def attrition_distribution(df: pd.DataFrame):

        logger.info("Creating Attrition Distribution chart.")

        counts = df["Attrition"].value_counts().reset_index()
        counts.columns = ["Attrition", "Count"]

        return px.pie(
            counts,
            names="Attrition",
            values="Count",
            title="Employee Attrition Distribution",
            hole=0.45,
        )

    @staticmethod
    def department_attrition(df: pd.DataFrame):

        logger.info("Creating Department chart.")

        data = (
            df.groupby("Department")["Attrition"]
            .count()
            .reset_index(name="Employees")
        )

        return px.bar(
            data,
            x="Department",
            y="Employees",
            title="Employees by Department",
        )

    @staticmethod
    def gender_distribution(df: pd.DataFrame):

        logger.info("Creating Gender Distribution chart.")

        data = (
            df.groupby("Gender")
            .size()
            .reset_index(name="Employees")
        )

        return px.bar(
            data,
            x="Gender",
            y="Employees",
            title="Gender Distribution",
        )

    @staticmethod
    def overtime_analysis(df: pd.DataFrame):

        logger.info("Creating Overtime Analysis chart.")

        data = (
            df.groupby("OverTime")
            .size()
            .reset_index(name="Employees")
        )

        return px.bar(
            data,
            x="OverTime",
            y="Employees",
            title="Overtime Analysis",
        )

    @staticmethod
    def monthly_income_distribution(df: pd.DataFrame):

        logger.info("Creating Monthly Income Distribution.")

        return px.histogram(
            df,
            x="MonthlyIncome",
            nbins=30,
            title="Monthly Income Distribution",
        )

    @staticmethod
    def age_distribution(df: pd.DataFrame):

        logger.info("Creating Age Distribution.")

        return px.histogram(
            df,
            x="Age",
            nbins=20,
            title="Age Distribution",
        )

    @staticmethod
    def job_role_distribution(df: pd.DataFrame):

        logger.info("Creating Job Role chart.")

        data = (
            df.groupby("JobRole")
            .size()
            .reset_index(name="Employees")
        )

        return px.bar(
            data,
            x="JobRole",
            y="Employees",
            title="Employees by Job Role",
        )

    @staticmethod
    def education_field_distribution(df: pd.DataFrame):

        logger.info("Creating Education Field chart.")

        data = (
            df.groupby("EducationField")
            .size()
            .reset_index(name="Employees")
        )

        return px.bar(
            data,
            x="EducationField",
            y="Employees",
            title="Education Field Distribution",
        )

    @staticmethod
    def correlation_heatmap(df: pd.DataFrame):

        logger.info("Creating Correlation Heatmap.")

        correlation = df.select_dtypes(include="number").corr()

        return px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Heatmap",
        )

    @staticmethod
    def feature_importance(feature_names, importance):

        logger.info("Creating Feature Importance chart.")

        data = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importance,
            }
        )

        data = data.sort_values(
            by="Importance",
            ascending=False
        ).head(20)

        return px.bar(
            data,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 20 Feature Importance",
      )
