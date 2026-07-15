# Changelog

## 1.0.1 - 2026-07-15

- Converted all user-facing interface text, warnings, results, model explanations, documentation, and privacy statements to English.
- Converted the accompanying README and deployment instructions to English.
- Preserved all locked models, feature order, probability calculations, and thresholds without modification.
- Revalidated both prediction horizons and the responsive mobile layout.

## 1.0.0 - 2026-07-15

- Added the locked 3-month logistic regression ensemble with threshold 0.350.
- Added the locked 6-month XGBoost ensemble with threshold 0.280.
- Loaded exactly 20 imputation-specific artifacts for each prediction horizon.
- Added automatic NLR and log1p(NLR) calculation from neutrophil and lymphocyte counts.
- Added patient-level linear and TreeSHAP model explanations.
- Added privacy-first Chinese academic user interface and responsive mobile layout.
- Added deidentified prediction-reproduction tests and model validation report.
- Added Docker, Streamlit Cloud, and local deployment instructions.
