from pathlib import Path

import pandas as pd
import pytest

from model_runtime import load_locked_artifacts, predict_patient


CASES = Path(__file__).with_name("prediction_test_cases.csv")


@pytest.fixture(scope="session")
def artifacts():
    return {
        "3_month": load_locked_artifacts("3_month"),
        "6_month": load_locked_artifacts("6_month"),
    }


def test_locked_artifact_count(artifacts):
    assert len(artifacts["3_month"]) == 20
    assert len(artifacts["6_month"]) == 20


def test_predictions_match_formal_pipeline(artifacts):
    cases = pd.read_csv(CASES)
    differences = []
    class_matches = []
    for row in cases.to_dict("records"):
        outcome = row.pop("outcome")
        expected_probability = float(row.pop("expected_probability"))
        expected_risk = row.pop("expected_risk")
        row.pop("case_id", None)
        values = {key: float(value) for key, value in row.items() if pd.notna(value)}
        result = predict_patient(outcome, values, artifacts[outcome])
        differences.append(abs(result.probability - expected_probability))
        class_matches.append(result.risk_label == expected_risk)
    assert max(differences) < 1e-8
    assert all(class_matches)


def test_invalid_lymphocyte_count_is_rejected(artifacts):
    values = {"alsfrsr": 38, "fss": 36, "psqi": 8, "albumin": 40, "phase_angle": 5, "lymphocytes": 0, "neutrophils": 3.8}
    with pytest.raises(ValueError, match="greater than 0"):
        predict_patient("3_month", values, artifacts["3_month"])


def test_xgboost_shap_additivity(artifacts):
    values = {"age": 55, "alsfrsr": 38, "fss": 36, "psqi": 8, "albumin": 40, "hemoglobin": 135, "lymphocytes": 1.6, "neutrophils": 3.8}
    result = predict_patient("6_month", values, artifacts["6_month"])
    assert result.additivity_error is not None
    assert result.additivity_error <= 1e-4
