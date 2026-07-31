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

from src.config import MODEL_PATH
from src.logger import logger


class Predictor:
    """
    Prediction class for Employee Attrition Dashboard.
    """

    def __init__(self):

        self.model = self.load_model()

    def load_model(self):
        """
        Load trained machine learning model.
        """

        try:

            if not Path(MODEL_PATH).exists():
                raise FileNotFoundError(
                    f"Model not found: {MODEL_PATH}"
                )

            model = joblib.load(MODEL_PATH)

            logger.info("Model loaded successfully.")

            return model

        except Exception as error:

            logger.exception("Unable to load model.")

            raise error

    def predict(self, input_data: pd.DataFrame):
        """
        Predict employee attrition.
        """

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

        logger.warning(
            "Current model does not provide feature importance."
        )

        return None

    def model_information(self):
        """
        Return model information.
        """

        return {
            "Model": type(self.model).__name__,
            "Supports Prediction": hasattr(
                self.model,
                "predict"
            ),
            "Supports Probability": hasattr(
                self.model,
                "predict_proba"
            ),
            "Supports Feature Importance": hasattr(
                self.model,
                "feature_importances_"
            ),
          }
