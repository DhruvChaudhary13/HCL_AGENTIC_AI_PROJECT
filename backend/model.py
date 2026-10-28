"""Model training and prediction for the Patient Health Outcome Predictor.

The model is trained on SYNTHETIC data generated here, so the project works
out of the box. Replace `generate_data()` with a real, de-identified dataset
to use it for anything beyond a demo.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

TREATMENTS = ["Medication", "Surgery", "Physiotherapy", "Combined"]
GENDERS = ["Female", "Male"]

FEATURES = [
    "age", "gender", "bmi", "smoker", "diabetes", "hypertension",
    "heart_disease", "prior_admissions", "treatment", "duration_days",
]

LABELS = {
    "age": "Age",
    "gender": "Gender",
    "bmi": "BMI",
    "smoker": "Smoking",
    "diabetes": "Diabetes",
    "hypertension": "Hypertension",
    "heart_disease": "Heart disease",
    "prior_admissions": "Prior admissions",
    "treatment": "Treatment plan",
    "duration_days": "Treatment duration",
}


def generate_data(n=4000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 91, n),
        "gender": rng.choice(GENDERS, n),
        "bmi": np.round(rng.normal(27, 5, n).clip(15, 50), 1),
        "smoker": rng.integers(0, 2, n),
        "diabetes": rng.binomial(1, 0.2, n),
        "hypertension": rng.binomial(1, 0.3, n),
        "heart_disease": rng.binomial(1, 0.15, n),
        "prior_admissions": rng.poisson(1.0, n).clip(0, 10),
        "treatment": rng.choice(TREATMENTS, n),
        "duration_days": rng.integers(3, 61, n),
    })
    treat_effect = df["treatment"].map(
        {"Medication": 0.0, "Surgery": 0.3, "Physiotherapy": 0.2, "Combined": 0.7}
    )
    logit = (
        1.8
        - 0.03 * (df["age"] - 50)
        - 0.06 * (df["bmi"] - 25).clip(lower=0)
        - 0.6 * df["smoker"]
        - 0.5 * df["diabetes"]
        - 0.4 * df["hypertension"]
        - 0.8 * df["heart_disease"]
        - 0.3 * df["prior_admissions"]
        + treat_effect
        + 0.015 * (df["duration_days"].clip(upper=30) - 10)
        + rng.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["good_outcome"] = (rng.random(n) < prob).astype(int)
    return df


def encode(df):
    df = df.copy()
    df["gender"] = (df["gender"] == "Male").astype(int)
    for t in TREATMENTS:
        df[f"treatment_{t}"] = (df["treatment"] == t).astype(int)
    df = df.drop(columns=["treatment"])
    return df


def train():
    df = generate_data()
    X = encode(df.drop(columns=["good_outcome"]))
    y = df["good_outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )
    clf = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=1)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, proba > 0.5)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
        "train_rows": int(len(X_train)),
    }
    joblib.dump({"model": clf, "columns": list(X.columns), "metrics": metrics,
                 "means": X_train.mean().to_dict()}, MODEL_PATH)
    return metrics


_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            train()
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict(data):
    """data: dict with the FEATURES keys. Returns a result dict."""
    bundle = get_bundle()
    row = pd.DataFrame([data])[FEATURES]
    X = encode(row).reindex(columns=bundle["columns"], fill_value=0)
    clf = bundle["model"]
    p_good = float(clf.predict_proba(X)[0, 1])

    # Simple per-patient explanation: swap each feature for the training
    # average and see how much the probability moves.
    impacts = []
    for col in bundle["columns"]:
        alt = X.copy()
        alt[col] = bundle["means"][col]
        delta = p_good - float(clf.predict_proba(alt)[0, 1])
        impacts.append((col, delta))
    merged = {}
    for col, delta in impacts:
        key = "treatment" if col.startswith("treatment_") else col
        merged[key] = merged.get(key, 0.0) + delta
    factors = sorted(
        ({"feature": LABELS[k], "impact": round(v * 100, 1)} for k, v in merged.items()),
        key=lambda f: abs(f["impact"]), reverse=True,
    )[:5]

    if p_good >= 0.70:
        risk = "Low"
    elif p_good >= 0.45:
        risk = "Moderate"
    else:
        risk = "High"

    return {
        "good_outcome_probability": round(p_good * 100, 1),
        "poor_outcome_probability": round((1 - p_good) * 100, 1),
        "risk_level": risk,
        "top_factors": factors,
    }


if __name__ == "__main__":
    print("Training model on synthetic data...")
    print("Metrics:", train())
