"""
==========================================================
Employee Attrition Dashboard
predictor.py
Author : Pramod Prakash Jadhav
==========================================================
Machine Learning prediction module.
"""
from pathlib import Path
import joblib
import pandas as pd
from src.config import (
    MODEL_PATH,
    SCALER_PATH,
    FEATURE_COLUMNS_PATH,
    LABEL_ENCODER_PATH,
)
from src.logger import logger


class Predictor:
    """
    Prediction class for Employee Attrition Dashboard.
    """

    def __init__(self):
        self.model = self.load_artifact(MODEL_PATH)
        self.scaler = self.load_artifact(SCALER_PATH)
        self.feature_columns = self.load_artifact(FEATURE_COLUMNS_PATH)
        self.label_encoder = self.load_artifact(LABEL_ENCODER_PATH)

    def load_artifact(self, path):
        """
        Load a trained artifact (model, scaler, encoder, etc.) from disk.
        """
        try:
            if Path(path).exists():
                logger.info(f"Loaded: {path}")
                return joblib.load(path)
            logger.warning(f"Unable to load: {path}")
            return None
        except Exception as error:
            logger.exception(f"Unable to load artifact: {path}")
            raise error

    def predict(self, input_data: pd.DataFrame):
        """
        Predict employee attrition.
        """
        if self.feature_columns is not None:
            input_data = input_data.reindex(
                columns=self.feature_columns,
                fill_value=0
            )
        if self.scaler is not None:
            input_data = self.scaler.transform(input_data)

        prediction = self.model.predict(input_data)
        logger.info("Prediction completed.")
        return prediction

    def predict_probability(self, input_data: pd.DataFrame):
        """
        Predict probability of attrition.
        """
        if hasattr(self.model, "predict_proba"):
            probability = self.model.predict_proba(input_data)
            logger.info("Prediction probability generated.")
            return probability
        logger.warning("Model does not support predict_proba().")
        return None

    def predict_label(self, input_data: pd.DataFrame):
        """
        Return readable prediction label.
        """
        prediction = self.predict(input_data)[0]
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform([prediction])[0]
        return "Yes" if prediction == 1 else "No"

    def predict_confidence(self, input_data: pd.DataFrame):
        """
        Return prediction confidence score.
        """
        probability = self.predict_probability(input_data)
        if probability is None:
            return None
        confidence = probability.max(axis=1)[0]
        return round(confidence * 100, 2)

    def get_feature_importance(self):
        """
        Return feature importance if supported.
        """
        if hasattr(self.model, "feature_importances_"):
            logger.info("Feature importance extracted.")
            return self.model.feature_importances_
        if hasattr(self.model, "coef_"):
            logger.info("Feature importance extracted from coefficients.")
            return self.model.coef_[0]
        logger.warning("Current model does not provide feature importance.")
        return None

    def model_information(self):
        """
        Return model information.
        """
        return {
            "Model": type(self.model).__name__,
            "Supports Prediction": hasattr(self.model, "predict"),
            "Supports Probability": hasattr(self.model, "predict_proba"),
            "Supports Feature Importance": hasattr(
                self.model, "feature_importances_"
            ),
        }
