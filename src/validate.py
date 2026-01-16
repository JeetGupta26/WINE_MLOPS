import pandas as pd
import logging
import os
import sys

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.ERROR
)

try:
    print("🔍 Loading dataset...")
    df = pd.read_csv("data/winequality-red.csv", sep=";")

    # -------- Dataset Integrity Checks (TC1) --------
    expected_columns = [
        'fixed acidity', 'volatile acidity', 'citric acid',
        'residual sugar', 'chlorides', 'free sulfur dioxide',
        'total sulfur dioxide', 'density', 'pH', 'sulphates',
        'alcohol', 'quality'
    ]

    # Column check
    if list(df.columns) != expected_columns:
        raise ValueError("Dataset schema mismatch")

    # Row count check
    if df.shape[0] < 1000:
        raise ValueError("Dataset has insufficient rows")

    # Null check
    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values")

    print("✅ Data validation passed")

except Exception as e:
    logging.error(f"Validation failed: {e}")
    print("❌ Pipeline stopped due to validation error")
    sys.exit(1)
