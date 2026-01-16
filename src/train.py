import joblib
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = joblib.load("data/processed_data.pkl")

mlflow.start_run()

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5

mlflow.log_param("model", "RandomForestRegressor")
mlflow.log_metric("rmse", rmse)
mlflow.sklearn.log_model(model, "model")

joblib.dump(model, "model.pkl")

print(f"✅ Model trained successfully | RMSE: {rmse}")

mlflow.end_run()