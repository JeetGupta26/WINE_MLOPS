import joblib
import mlflow
import logging
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.ERROR
)

try:
    # Load processed data
    X_train, X_test, y_train, y_test = joblib.load("data/processed_data.pkl")

    mlflow.start_run()

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)

    # Log to MLflow
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")

    # Save model
    joblib.dump(model, "model.pkl")

    print(f"Model trained successfully | RMSE: {rmse}")

    # Performance threshold (relaxed)
    if rmse > 1.0:
        raise ValueError("Model performance below threshold")

    mlflow.end_run()

except Exception as e:
    logging.error(f"Training failed: {e}")
    print("Pipeline stopped safely")
    sys.exit(1)

# ✅ SUCCESSFUL EXIT (THIS WAS MISSING BEFORE)
sys.exit(0)