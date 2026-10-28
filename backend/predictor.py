"""Loads the model trained in notebook/train_model.ipynb and makes predictions.

The notebook saves one file (a "bundle") that contains the fitted scikit-learn
pipeline plus a description of every input field. This module only reads it.
"""
import math
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(BASE_DIR, "model", "outcome_model.joblib")
)


class ModelNotReady(Exception):
    """Raised when the trained model file is missing or cannot be loaded."""


_bundle = None


def load_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise ModelNotReady(
                "No trained model found. Run notebook/train_model.ipynb to create "
                "backend/model/outcome_model.joblib, then restart the server."
            )
        try:
            bundle = joblib.load(MODEL_PATH)
        except Exception as exc:  # pickle / version problems
            raise ModelNotReady(f"The model file could not be loaded: {exc}") from exc
        trained_with = bundle.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            warnings.warn(
                f"Model was trained with scikit-learn {trained_with} but the server "
                f"has {sklearn.__version__}. Pin scikit-learn=={trained_with} in "
                "requirements.txt."
            )
        _bundle = bundle
    return _bundle


def get_schema():
    """Everything the frontend needs to build the form and show model info."""
    b = load_bundle()
    return {
        "outcome_label": b["outcome_label"],
        "dataset_name": b.get("dataset_name", ""),
        "model_name": b.get("model_name", ""),
        "features": b["features"],
        "metrics": b.get("metrics", {}),
        "base_rate": round(b.get("base_rate", 0.0) * 100, 1),
        "risk_bands": b.get("risk_bands", {}),
        "global_importance": b.get("global_importance", [])[:8],
    }


def validate(data):
    """Check the request body against the saved schema.

    Returns (row, warnings, error). `row` maps feature name -> clean value.
    """
    b = load_bundle()
    row, notes, problems = {}, [], []

    for field in b["features"]:
        name, label = field["name"], field["label"]
        if name not in data or data[name] in (None, ""):
            problems.append(f"'{label}' is required.")
            continue
        value = data[name]

        if field["type"] == "number":
            try:
                value = float(value)
            except (TypeError, ValueError):
                problems.append(f"'{label}' must be a number.")
                continue
            if not math.isfinite(value):
                problems.append(f"'{label}' must be a finite number.")
                continue
            if field.get("integer"):
                value = int(round(value))
            if value < field["min"] or value > field["max"]:
                notes.append(
                    f"{label} ({value:g}) is outside the range seen in training "
                    f"({field['min']:g} to {field['max']:g}); treat the result with caution."
                )
            row[name] = value
        else:
            value = str(value)
            if value not in field["options"]:
                problems.append(f"'{label}' must be one of: {', '.join(field['options'])}.")
                continue
            row[name] = value

    if problems:
        return None, [], " ".join(problems)
    return row, notes, None


def predict(row, notes=None):
    b = load_bundle()
    pipeline = b["pipeline"]
    features = b["features"]
    names = [f["name"] for f in features]

    # One batch: row 0 is the patient, rows 1..n replace one input with its
    # typical (default) value so we can see how much that input moved the risk.
    frames = [row]
    for f in features:
        alt = dict(row)
        alt[f["name"]] = f["default"]
        frames.append(alt)
    probs = pipeline.predict_proba(pd.DataFrame(frames)[names])[:, 1]
    p = float(probs[0])

    factors = []
    for f, p_typical in zip(features, probs[1:]):
        impact = (p - float(p_typical)) * 100
        if abs(impact) >= 0.1:
            factors.append({"feature": f["label"], "impact": round(impact, 1)})
    factors.sort(key=lambda x: abs(x["impact"]), reverse=True)

    bands = b["risk_bands"]
    if p >= bands["high_from"]:
        level = "High"
    elif p >= bands["moderate_from"]:
        level = "Moderate"
    else:
        level = "Low"

    quantiles = np.asarray(b["score_quantiles"])
    percentile = int(np.clip(np.searchsorted(quantiles, p, side="right") - 1, 0, 100))

    return {
        "outcome_label": b["outcome_label"],
        "probability": round(p * 100, 1),
        "risk_level": level,
        "percentile": percentile,
        "base_rate": round(b["base_rate"] * 100, 1),
        "top_factors": factors[:6],
        "warnings": notes or [],
    }
