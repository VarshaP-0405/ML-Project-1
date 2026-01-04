from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"

# ---------------------------------
# TRAIN MODEL FUNCTION
# ---------------------------------
def train_and_save_model():
    np.random.seed(42)

    baseline = np.random.uniform(2.5, 6.0, 800)
    measured = baseline * np.random.uniform(0.95, 1.35, 800)

    diff = measured - baseline
    ratio = measured / baseline

    df = pd.DataFrame({
        "baseline": baseline,
        "measured": measured,
        "diff": diff,
        "ratio": ratio
    })

    # ✔️ Well-calibrated abnormal condition
    df["label"] = (df["ratio"] > 1.20).astype(int)

    X = df[["baseline", "measured", "diff", "ratio"]]
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    # ✔️ Save BOTH files
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    print("✅ Model and scaler trained & saved")

# ---------------------------------
# ENSURE FILES EXIST
# ---------------------------------
if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    train_and_save_model()

# ---------------------------------
# LOAD MODEL & SCALER
# ---------------------------------
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

# ---------------------------------
# ROUTES
# ---------------------------------
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/predict", methods=["POST"])
def predict():
    baseline = float(request.form["baseline"])
    measured = float(request.form["corrected"])

    diff = measured - baseline
    ratio = measured / baseline

    X = np.array([[baseline, measured, diff, ratio]])
    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0][1]

    status = "Abnormal Dilation Pattern" if prob > 0.75 else "Normal Pupil Response"
    confidence = round(prob * 100, 2)

    return render_template(
        "result.html",
        baseline=baseline,
        measured=measured,
        ratio=round(ratio, 3),
        status=status,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
