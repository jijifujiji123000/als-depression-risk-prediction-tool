from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


MODEL_DIR = Path(__file__).resolve().parent / "models"

FEATURES = {
    "3_month": [
        "alsfrsr_total",
        "fss_total",
        "psqi_total",
        "检验_白蛋白_首次",
        "InBody_相位角_首次",
        "检验_淋巴细胞绝对值_首次",
        "log1p_NLR",
    ],
    "6_month": [
        "alsfrsr_total",
        "fss_total",
        "psqi_total",
        "log1p_NLR",
        "检验_白蛋白_首次",
        "检验_血红蛋白_首次",
        "检验_淋巴细胞绝对值_首次",
        "基线_年龄（岁）：",
    ],
}

DISPLAY = {
    "alsfrsr_total": "ALSFRS-R",
    "fss_total": "FSS",
    "psqi_total": "PSQI",
    "检验_白蛋白_首次": "Albumin",
    "InBody_相位角_首次": "Phase angle",
    "检验_淋巴细胞绝对值_首次": "Lymphocyte count",
    "log1p_NLR": "log1p(NLR)",
    "检验_血红蛋白_首次": "Hemoglobin",
    "基线_年龄（岁）：": "Age",
}

THRESHOLDS = {"3_month": 0.350, "6_month": 0.280}


@dataclass(frozen=True)
class PredictionResult:
    outcome: str
    probability: float
    threshold: float
    risk_label: str
    model_probabilities: tuple[float, ...]
    contributions: pd.DataFrame
    explanation_scale: str
    additivity_error: float | None


def _artifact_pattern(outcome: str) -> str:
    if outcome == "3_month":
        return "3month_logistic_MI*.pkl"
    if outcome == "6_month":
        return "6month_xgboost_MI*.pkl"
    raise ValueError(f"Unsupported outcome: {outcome}")


def load_locked_artifacts(outcome: str, model_dir: Path = MODEL_DIR) -> list[dict[str, Any]]:
    paths = sorted(model_dir.glob(_artifact_pattern(outcome)))
    if len(paths) != 20:
        raise RuntimeError(f"{outcome} requires exactly 20 locked artifacts; found {len(paths)}.")
    artifacts = [joblib.load(path) for path in paths]
    expected = FEATURES[outcome]
    for path, artifact in zip(paths, artifacts):
        if artifact.get("outcome") != outcome:
            raise RuntimeError(f"Outcome mismatch in {path.name}.")
        if list(artifact.get("selected_variables", [])) != expected:
            raise RuntimeError(f"Feature order mismatch in {path.name}.")
        model = artifact.get("model")
        if model is None or not hasattr(model, "predict_proba"):
            raise RuntimeError(f"Missing predict_proba pipeline in {path.name}.")
        if list(getattr(model, "classes_", [])) != [0, 1]:
            raise RuntimeError(f"Positive-class coding mismatch in {path.name}.")
    return artifacts


def build_model_frame(outcome: str, values: dict[str, float]) -> pd.DataFrame:
    lymphocytes = float(values["lymphocytes"])
    neutrophils = float(values["neutrophils"])
    if not np.isfinite(lymphocytes) or lymphocytes <= 0:
        raise ValueError("Lymphocyte count must be greater than 0.")
    if not np.isfinite(neutrophils) or neutrophils < 0:
        raise ValueError("Neutrophil count must be non-negative.")
    nlr = neutrophils / lymphocytes
    common = {
        "alsfrsr_total": float(values["alsfrsr"]),
        "fss_total": float(values["fss"]),
        "psqi_total": float(values["psqi"]),
        "检验_白蛋白_首次": float(values["albumin"]),
        "检验_淋巴细胞绝对值_首次": lymphocytes,
        "log1p_NLR": float(np.log1p(nlr)),
    }
    if outcome == "3_month":
        common["InBody_相位角_首次"] = float(values["phase_angle"])
    elif outcome == "6_month":
        common["检验_血红蛋白_首次"] = float(values["hemoglobin"])
        common["基线_年龄（岁）："] = float(values["age"])
    else:
        raise ValueError(f"Unsupported outcome: {outcome}")
    frame = pd.DataFrame([{name: common[name] for name in FEATURES[outcome]}])
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Inputs must be numeric, finite, and non-missing.")
    return frame


def _logistic_contributions(artifacts: list[dict[str, Any]], frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    rows = []
    errors = []
    for artifact in artifacts:
        pipeline = artifact["model"]
        transformed = np.asarray(pipeline.named_steps["pre"].transform(frame), dtype=float)[0]
        estimator = pipeline.named_steps["model"]
        contribution = transformed * estimator.coef_[0]
        margin = float(estimator.decision_function(transformed.reshape(1, -1))[0])
        reconstructed = float(estimator.intercept_[0] + contribution.sum())
        errors.append(abs(margin - reconstructed))
        rows.append(contribution)
    mean_values = np.mean(np.vstack(rows), axis=0)
    table = pd.DataFrame({
        "Feature": [DISPLAY[name] for name in FEATURES["3_month"]],
        "Contribution": mean_values,
    }).sort_values("Contribution", key=lambda s: s.abs(), ascending=False)
    return table.reset_index(drop=True), float(max(errors))


def _xgboost_contributions(artifacts: list[dict[str, Any]], frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    rows = []
    errors = []
    for artifact in artifacts:
        pipeline = artifact["model"]
        transformed = np.asarray(pipeline.named_steps["pre"].transform(frame), dtype=float)
        estimator = pipeline.named_steps["model"]
        matrix = xgb.DMatrix(transformed)
        values = estimator.get_booster().predict(matrix, pred_contribs=True)[0]
        margin = float(estimator.get_booster().predict(matrix, output_margin=True)[0])
        errors.append(abs(margin - float(values.sum())))
        rows.append(values[:-1])
    mean_values = np.mean(np.vstack(rows), axis=0)
    table = pd.DataFrame({
        "Feature": [DISPLAY[name] for name in FEATURES["6_month"]],
        "Contribution": mean_values,
    }).sort_values("Contribution", key=lambda s: s.abs(), ascending=False)
    return table.reset_index(drop=True), float(max(errors))


def predict_patient(
    outcome: str,
    values: dict[str, float],
    artifacts: list[dict[str, Any]] | None = None,
) -> PredictionResult:
    artifacts = artifacts or load_locked_artifacts(outcome)
    frame = build_model_frame(outcome, values)
    probabilities = tuple(float(item["model"].predict_proba(frame)[:, 1][0]) for item in artifacts)
    # Preserve the original formal pipeline's estimator-native stacking semantics:
    # Logistic outputs float64; XGBoost outputs float32.
    probability_dtype = np.float64 if outcome == "3_month" else np.float32
    probability = float(np.mean(np.asarray(probabilities, dtype=probability_dtype)))
    threshold = THRESHOLDS[outcome]
    risk_label = "Higher predicted risk" if probability >= threshold else "Lower predicted risk"
    if outcome == "3_month":
        contributions, error = _logistic_contributions(artifacts, frame)
        scale = "Standardized linear predictor (log-odds)"
    else:
        contributions, error = _xgboost_contributions(artifacts, frame)
        scale = "TreeSHAP raw log-odds margin"
    return PredictionResult(
        outcome=outcome,
        probability=probability,
        threshold=threshold,
        risk_label=risk_label,
        model_probabilities=probabilities,
        contributions=contributions,
        explanation_scale=scale,
        additivity_error=error,
    )
