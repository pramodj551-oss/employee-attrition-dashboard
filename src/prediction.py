# src/prediction.py
import joblib
import pandas as pd

def load_model(path='models/best_model.pkl'):
    return joblib.load(path)

def predict_attrition(model, input_data):
    """input_data: dict or DataFrame with features"""
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data
    # Ensure preprocessing steps match training
    # (apply encoding, scaling, etc.)
    proba = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]
    return pred, proba
