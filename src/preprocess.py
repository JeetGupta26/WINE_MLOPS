import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Create directories for reports
os.makedirs("reports/preprocessing", exist_ok=True)

# Load dataset
df = pd.read_csv("data/winequality-red.csv", sep=";")

# ===============================
# VISUALIZATION 1: Quality distribution
# ===============================
plt.figure(figsize=(6,4))
sns.countplot(x="quality", data=df)
plt.title("Wine Quality Distribution")
plt.savefig("reports/preprocessing/quality_distribution.png")
plt.close()

# ===============================
# VISUALIZATION 2: Feature distributions
# ===============================
df.hist(figsize=(12,10))
plt.tight_layout()
plt.savefig("reports/preprocessing/feature_distributions.png")
plt.close()

# ===============================
# VISUALIZATION 3: Correlation heatmap
# ===============================
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.savefig("reports/preprocessing/correlation_heatmap.png")
plt.close()

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
