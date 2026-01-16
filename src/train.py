import joblib
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load processed data
X_train, X_test, y_train, y_test = joblib.load("data/processed_data.pkl")

# Start MLflow run
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

print(f"✅ Model trained successfully | RMSE: {rmse}")

mlflow.end_run()