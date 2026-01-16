from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("data/scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    scaled_data = scaler.transform([data])
    prediction = model.predict(scaled_data)
    return jsonify({"predicted_quality": prediction[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
