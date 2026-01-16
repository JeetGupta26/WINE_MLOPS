import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("data/winequality-red.csv", sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Save preprocessed data
joblib.dump((X_train, X_test, y_train, y_test), "data/processed_data.pkl")
joblib.dump(scaler, "data/scaler.pkl")

print("Preprocessing completed successfully")
