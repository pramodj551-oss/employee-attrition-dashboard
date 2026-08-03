"""
==========================================================
Employee Attrition Dashboard
predictor.py
Author : Pramod Prakash Jadhav
==========================================================
Machine Learning prediction module.
"""
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
import streamlit as st

from src.config import (
    MODEL_PATH,
    SCALER_PATH,
    FEATURE_COLUMNS_PATH,
    LABEL_ENCODER_PATH,
)
from src.logger import logger


# Cached artifact loaders ----------------------------------------------------

@st.cache_resource
def _load_joblib_artifact(path: str) -> Optional[Any]:
    """
    Load an artifact using joblib and cache it as a Streamlit resource.
    Uses mmap_mode='r' to avoid extra memory copies for large numpy arrays
    when possible. Returns None if the path does not exist or load fails.
    """
    try:
        p = Path(path)
        if not p.exists():
            logger.warning("Artifact not found: %s", path)
            return None

        # joblib.load supports mmap_mode for numpy arrays inside pickles
        try:
            artifact = joblib.load(path, mmap_mode="r")
        except TypeError:
            # Older joblib versions might not accept mmap_mode; fallback
            artifact = joblib.load(path)
        logger.info("Loaded artifact: %s", path)
        return artifact
    except Exception as error:
        logger.exception("Failed to load artifact: %s", path)
        return None


# Predictor class -----------------------------------------------------------

class Predictor:
    """
    Prediction class for Employee Attrition Dashboard.

    Artifacts (model, scaler, feature columns, label encoder) are loaded via
    cached functions to avoid repeated disk I/O and to share them across
    session reruns.
    """

    def __init__(self):
        # Use cached loaders so repeated Predictor() instantiation is cheap
        self.model = _load_joblib_artifact(MODEL_PATH)
        self.scaler = _load_joblib_artifact(SCALER_PATH)
        self.feature_columns = _load_joblib_artifact(FEATURE_COLUMNS_PATH)
        self.label_encoder = _load_joblib_artifact(LABEL_ENCODER_PATH)

    # Internal helpers -----------------------------------------------------

    def _prepare_input(self, input_data: pd.DataFrame):
        """
        Reindex input_data to match feature_columns and apply scaler if available.

        Returns the prepared numpy array suitable for model.predict / predict_proba.

        Raises ValueError if model is not loaded.
        """
        if input_data is None or input_data.empty:
            raise ValueError("No input data provided for prediction.")

        df = input_data.copy()

        # Reindex to expected feature columns if provided
        if self.feature_columns is not None:
            try:
                df = df.reindex(columns=self.feature_columns, fill_value=0)
            except Exception as err:
                logger.exception("Failed to reindex input data to feature columns.")
                raise err

        # Apply scaler if present
        if self.scaler is not None:
            try:
                # Many scalers accept DataFrame or ndarray; use DataFrame if it works
                prepared = self.scaler.transform(df)
            except Exception:
                # Fallback to passing values
                prepared = self.scaler.transform(df.values)
        else:
            prepared = df.values

        return prepared

    # Prediction APIs -----------------------------------------------------

    def predict(self, input_data: pd.DataFrame):
        """
        Predict employee attrition labels (raw model output).

        input_data: pd.DataFrame with a single row or multiple rows.
        Returns: numpy array of predictions.
        """
        if self.model is None:
            raise ValueError("Model artifact is not loaded.")

        X = self._prepare_input(input_data)
        preds = self.model.predict(X)
        logger.info("Prediction completed for %d rows.", getattr(X, "shape", (0,))[0])
        return preds

    def predict_probability(self, input_data: pd.DataFrame):
        """
        Predict probability of attrition.

        Returns the probability array from model.predict_proba if available,
        otherwise returns None.
        """
        if self.model is None:
            logger.warning("Model artifact is not loaded; cannot compute probabilities.")
            return None

        if not hasattr(self.model, "predict_proba"):
            logger.warning("Model does not support predict_proba().")
            return None

        X = self._prepare_input(input_data)
        try:
            probability = self.model.predict_proba(X)
            logger.info("Prediction probability generated for %d rows.", getattr(X, "shape", (0,))[0])
            return probability
        except Exception:
            logger.exception("Failed to compute prediction probabilities.")
            return None

    def predict_label(self, input_data: pd.DataFrame):
        """
        Return readable prediction label for the first row (used by the Streamlit UI).
        If multiple rows are provided, returns the label for the first row.
        """
        preds = self.predict(input_data)
        if len(preds) == 0:
            raise ValueError("No prediction was returned by the model.")

        pred = preds[0]

        if self.label_encoder is not None:
            try:
                label = self.label_encoder.inverse_transform([pred])[0]
                return label
            except Exception:
                logger.exception("Label encoder failed to inverse_transform.")
                # Fallback to numeric mapping
        return "Yes" if int(pred) == 1 else "No"

    def predict_confidence(self, input_data: pd.DataFrame):
        """
        Return prediction confidence (percentage) for the first row if probabilities are available.
        """
        prob = self.predict_probability(input_data)
        if prob is None:
            return None
        # prob is array-like of shape (n_samples, n_classes)
        try:
            confidence = prob.max(axis=1)[0]
            return round(float(confidence) * 100, 2)
        except Exception:
            logger.exception("Failed to compute confidence from probabilities.")
            return None

    # Utility methods ----------------------------------------------------

    def get_feature_importance(self):
        """
        Return feature importance if supported by the model.
        """
        if self.model is None:
            logger.warning("Model artifact not loaded; cannot extract feature importance.")
            return None

        if hasattr(self.model, "feature_importances_"):
            logger.info("Feature importance extracted.")
            return getattr(self.model, "feature_importances_")
        if hasattr(self.model, "coef_"):
            logger.info("Feature importance extracted from coefficients.")
            coef = getattr(self.model, "coef_")
            # Handle multiclass coef arrays by reducing to first class if necessary
            if coef.ndim > 1:
                return coef[0]
            return coef
        logger.warning("Current model does not provide feature importance.")
        return None

    def model_information(self):
        """
        Return model information.
        """
        model_type = type(self.model).__name__ if self.model is not None else None
        return {
            "Model": model_type,
            "Supports Prediction": bool(self.model is not None and hasattr(self.model, "predict")),
            "Supports Probability": bool(self.model is not None and hasattr(self.model, "predict_proba")),
            "Supports Feature Importance": bool(self.model is not None and (
                hasattr(self.model, "feature_importances_") or hasattr(self.model, "coef_")
            )),
        }
