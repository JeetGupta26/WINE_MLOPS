import pandas as pd
import great_expectations as ge
import logging
import os

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.ERROR
)

try:
    df_pd = pd.read_csv("data/winequality-red.csv", sep=";")
    df = ge.from_pandas(df_pd)

    df.expect_column_to_exist("quality")
    df.expect_column_values_to_be_between("alcohol", 8, 15)
    df.expect_table_row_count_to_be_between(1000, 10000)

    results = df.validate()

    if not results["success"]:
        raise ValueError("Data validation failed")

    print("✅ Data validation passed")

except Exception as e:
    logging.error(f"Validation failed: {e}")
    raise SystemExit("❌ Pipeline stopped due to validation error")
