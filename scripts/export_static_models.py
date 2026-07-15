from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis  # noqa: E402,F401 - required to unpickle the locked imputer class

MODEL_DIR = ROOT / "models"
OUTPUT = ROOT / "docs" / "model_data.json"


def scaler_payload(pipeline):
    scaler = pipeline.named_steps["pre"].named_transformers_["num"].named_steps["scale"]
    return {
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
    }


def export_logistic():
    models = []
    for path in sorted(MODEL_DIR.glob("3month_logistic_MI*.pkl")):
        pipeline = joblib.load(path)["model"]
        estimator = pipeline.named_steps["model"]
        item = scaler_payload(pipeline)
        item.update(
            coefficient=estimator.coef_[0].astype(float).tolist(),
            intercept=float(estimator.intercept_[0]),
        )
        models.append(item)
    if len(models) != 20:
        raise RuntimeError(f"Expected 20 three-month models, found {len(models)}")
    return models


def export_xgboost():
    models = []
    for path in sorted(MODEL_DIR.glob("6month_xgboost_MI*.pkl")):
        pipeline = joblib.load(path)["model"]
        booster = pipeline.named_steps["model"].get_booster()
        config = json.loads(booster.save_config())
        base_probability = float(
            config["learner"]["learner_model_param"]["base_score"].strip("[]")
        )
        item = scaler_payload(pipeline)
        item.update(
            base_margin=math.log(base_probability / (1.0 - base_probability)),
            trees=[json.loads(tree) for tree in booster.get_dump(dump_format="json")],
        )
        models.append(item)
    if len(models) != 20:
        raise RuntimeError(f"Expected 20 six-month models, found {len(models)}")
    return models


def walk_tree(node, values):
    if "leaf" in node:
        return float(node["leaf"])
    feature = int(node["split"][1:])
    next_id = node["yes"] if values[feature] < node["split_condition"] else node["no"]
    child = next(item for item in node["children"] if item["nodeid"] == next_id)
    return walk_tree(child, values)


def validate(payload):
    rng = np.random.default_rng(20260715)
    three_paths = sorted(MODEL_DIR.glob("3month_logistic_MI*.pkl"))
    six_paths = sorted(MODEL_DIR.glob("6month_xgboost_MI*.pkl"))
    max_error = 0.0
    for _ in range(100):
        x3 = rng.normal(size=7)
        for path, exported in zip(three_paths, payload["models"]["3_month"]):
            pipeline = joblib.load(path)["model"]
            raw = x3 * np.asarray(exported["scale"]) + np.asarray(exported["mean"])
            frame = pd.DataFrame([dict(zip(payload["features"]["3_month"], raw))])
            expected = float(pipeline.predict_proba(frame)[:, 1][0])
            margin = exported["intercept"] + float(np.dot(x3, exported["coefficient"]))
            actual = 1.0 / (1.0 + math.exp(-margin))
            max_error = max(max_error, abs(expected - actual))
        x6 = rng.normal(size=8)
        for path, exported in zip(six_paths, payload["models"]["6_month"]):
            pipeline = joblib.load(path)["model"]
            raw = x6 * np.asarray(exported["scale"]) + np.asarray(exported["mean"])
            frame = pd.DataFrame([dict(zip(payload["features"]["6_month"], raw))])
            expected = float(pipeline.predict_proba(frame)[:, 1][0])
            margin = exported["base_margin"] + sum(walk_tree(tree, x6) for tree in exported["trees"])
            actual = 1.0 / (1.0 + math.exp(-margin))
            max_error = max(max_error, abs(expected - actual))
    if max_error > 1e-6:
        raise RuntimeError(f"Static-model validation failed: max error={max_error:.3g}")
    return max_error


def main():
    first3 = joblib.load(sorted(MODEL_DIR.glob("3month_logistic_MI*.pkl"))[0])
    first6 = joblib.load(sorted(MODEL_DIR.glob("6month_xgboost_MI*.pkl"))[0])
    payload = {
        "version": "2026-07-15",
        "features": {
            "3_month": first3["selected_variables"],
            "6_month": first6["selected_variables"],
        },
        "thresholds": {"3_month": 0.35, "6_month": 0.28},
        "models": {
            "3_month": export_logistic(),
            "6_month": export_xgboost(),
        },
    }
    error = validate(payload)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")
    print(f"Maximum model-level probability error: {error:.3e}")


if __name__ == "__main__":
    main()
