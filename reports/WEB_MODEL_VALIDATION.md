# Web Model Validation

## Scope
Ten de-identified complete-input temporal-validation cases were selected: five per outcome, with one case from each probability quintile. Patient identifiers were not exported.

## Acceptance criteria
- Maximum absolute probability difference < 1e-8.
- Risk-stratum agreement = 100%.
- Exactly 20 locked artifacts loaded per outcome.
- XGBoost TreeSHAP raw-margin additivity error <= 1e-4.
- SHA-256 hashes match the formal model-artifact manifest.

## Results
- Test cases: 10
- Maximum absolute probability difference: 0.000e+00
- Mean absolute probability difference: 0.000e+00
- Risk-stratum agreement: 100.0%
- Model-artifact hashes checked: 40
- Model-artifact hash mismatches: 0
- Status: PASS

## Case-level audit
| case_id | outcome | formal_probability | web_backend_probability | absolute_difference | risk_match |
|---|---|---|---|---|---|
| 3M-01 | 3_month | 0.0344570024915 | 0.0344570024915 | 0 | True |
| 3M-02 | 3_month | 0.105173670842 | 0.105173670842 | 0 | True |
| 3M-03 | 3_month | 0.165413610254 | 0.165413610254 | 0 | True |
| 3M-04 | 3_month | 0.242864103865 | 0.242864103865 | 0 | True |
| 3M-05 | 3_month | 0.42518062237 | 0.42518062237 | 0 | True |
| 6M-01 | 6_month | 0.114921703935 | 0.114921703935 | 0 | True |
| 6M-02 | 6_month | 0.152507975698 | 0.152507975698 | 0 | True |
| 6M-03 | 6_month | 0.20383925736 | 0.20383925736 | 0 | True |
| 6M-04 | 6_month | 0.278034567833 | 0.278034567833 | 0 | True |
| 6M-05 | 6_month | 0.446913659573 | 0.446913659573 | 0 | True |

The web backend calls each saved pipeline's `predict_proba(X)[:, 1]` and averages 20 probabilities without refitting or changing preprocessing, thresholds, or model parameters.
