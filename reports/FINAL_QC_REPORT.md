# Final quality-control report

## Model integrity

- Locked 3-month artifacts loaded: 20 logistic regression pipelines.
- Locked 6-month artifacts loaded: 20 XGBoost pipelines.
- SHA-256 comparison against the formal manifest: 40 checked, 0 mismatches.
- No model was retrained, recalibrated, tuned, or replaced in the web project.
- Fixed thresholds: 0.350 at 3 months and 0.280 at 6 months.

## Prediction reproduction

- Deidentified temporal-validation cases: 10, with five cases per outcome.
- Maximum absolute probability difference: 0.000e+00.
- Mean absolute probability difference: 0.000e+00.
- Risk-stratum agreement: 100%.
- Automated tests: 4 passed.

## Explanation validation

- The 3-month logistic model uses standardized linear contributions.
- The 6-month XGBoost model uses TreeSHAP contributions on the raw-margin scale.
- Observed 6-month TreeSHAP additivity error in browser QA: 7.15e-07.
- Prespecified TreeSHAP acceptance threshold: 1.00e-04.

## Interface and privacy QA

- Desktop layout reviewed at 1280 x 720 pixels.
- Mobile layout reviewed at 390 x 844 pixels.
- Mobile horizontal overflow: none (scroll width equals client width).
- No overlapping or clipped interface text was identified in reviewed states.
- No patient identifiers are requested by the interface.
- No database persistence, patient-input file writes, analytics, or external patient-data API calls are implemented.

## Publication assets

- Figure 8 PNG exported at 300 dpi.
- Figure 8 PDF exported from the same real browser screenshots.
- Figure 8 legend supplied as a separate Word document.

Overall status: PASS.
