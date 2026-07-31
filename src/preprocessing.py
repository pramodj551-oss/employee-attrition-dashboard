"""
==========================================================
Employee Attrition Dashboard

preprocessing.py

Author : Pramod Prakash Jadhav
==========================================================

Data preprocessing utilities for the Employee
Attrition Dashboard.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.logger import logger


class DataPreprocessor:
    """
    Data preprocessing class.
    """

    def __init__(self):

        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean dataset.
        """

        logger.info("Cleaning dataset...")

        data = df.copy()

        # Remove duplicate rows
        data = data.drop_duplicates()

        # Fill missing values
        for column in data.columns:

            if data[column].dtype == "object":

                mode = data[column].mode()[0]

                data[column].fillna(mode, inplace=True)

            else:

                median = data[column].median()

                data[column].fillna(median, inplace=True)

        logger.info("Data cleaning completed.")

        return data

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode Attrition column.
        """

        data = df.copy()

        if "Attrition" in data.columns:

            data["Attrition"] = self.label_encoder.fit_transform(
                data["Attrition"]
            )

        return data

    def prepare_features(self, df: pd.DataFrame):
        """
        Separate features and target.
        """

        data = self.encode_target(df)

        X = data.drop("Attrition", axis=1)

        y = data["Attrition"]

        X = pd.get_dummies(
            X,
            drop_first=True
        )

        return X, y

    def scale_features(
        self,
        X_train,
        X_test=None
    ):
        """
        Scale feature matrices.
        """

        X_train_scaled = self.scaler.fit_transform(X_train)

        if X_test is not None:

            X_test_scaled = self.scaler.transform(X_test)

            return X_train_scaled, X_test_scaled

        return X_train_scaled

    def prepare_prediction_input(
        self,
        input_df: pd.DataFrame,
        training_columns
    ):
        """
        Prepare user input for prediction.
        """

        encoded = pd.get_dummies(
            input_df,
            drop_first=True
        )

        encoded = encoded.reindex(
            columns=training_columns,
            fill_value=0
        )

        return encoded

    def process(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        """

        logger.info("Starting preprocessing pipeline...")

        data = self.clean_data(df)

        logger.info("Preprocessing completed.")

        return data
