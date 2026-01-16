import joblib
import mlflow
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.ERROR
)

try:
    X_train, X_test, y_train, y_test = joblib.load("data/processed_data.pkl")

    mlflow.start_run()

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)

    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, "model.pkl")

    if rmse > 1.0:
        raise ValueError("Model performance below threshold")

    mlflow.end_run()
    print("Model trained successfully | RMSE:", rmse)

except Exception as e:
    logging.error(f"Pipeline failed: {e}")
    raise SystemExit("Pipeline stopped safely")
