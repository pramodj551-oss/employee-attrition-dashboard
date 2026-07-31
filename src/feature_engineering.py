# src/feature_engineering.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def create_features(df):
    """Create new features from raw data."""
    # Example: create age groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 30, 45, 60], labels=['Young', 'Middle', 'Senior'])
    # More transformations...
    return df

def encode_categorical(df, columns):
    """Label encode categorical columns."""
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])
    return df

def scale_features(df, columns):
    """Scale numerical features."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df
