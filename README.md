# ALS Depressive-Symptom Risk Prediction Tool

This project is a research-oriented risk prediction tool accompanying an academic manuscript. The application loads only the formally locked models and does not retrain, select, recalibrate, or modify them.

## Locked models

- 3-month outcome: logistic regression, fixed threshold 0.350, and 20 imputation-specific models.
- 6-month outcome: XGBoost, fixed threshold 0.280, and 20 imputation-specific models.
- The final patient-level probability is the arithmetic mean of 20 values produced by `predict_proba(X)[:, 1]`.
- NLR is calculated automatically as neutrophil count divided by lymphocyte count; the models use `log1p(NLR)`.

## Local use

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://127.0.0.1:8501`.

## Tests

```bash
pytest -q
```

The tests use deidentified cases from the formal patient-level prediction files. Acceptance requires a maximum absolute probability difference below `1e-8` and 100% agreement in risk classification.

## Privacy design

- The interface does not collect names, identification numbers, hospital numbers, telephone numbers, addresses, or other identifying information.
- Predictions are calculated only in server-process memory and are not written to a database or file.
- Patient-input logs are not created, and no third-party patient-data API is called.
- Google Analytics is not used, and patient data are not transmitted to external analytics services.
- Inputs are not retained after the Streamlit session is refreshed or terminated.

## Important limitations

This tool is intended for research demonstration and risk stratification. It is not a clinical diagnosis and does not replace validated scales, clinical interviews, psychiatric assessment, or medical decision-making. The models underwent single-center temporal evaluation but have not yet undergone independent multicenter external validation.

See [deploy_instructions.md](deploy_instructions.md) and [reports/WEB_MODEL_VALIDATION.md](reports/WEB_MODEL_VALIDATION.md) for details.

## Manuscript assets

- `reports/Figure_8_Web_Tool.png`: 300-dpi composite created from screenshots of the running application.
- `reports/Figure_8_Web_Tool.pdf`: matching PDF version.
- `reports/Figure_8_Web_Tool_Legend.docx`: English title and complete figure legend.
- `reports/FINAL_QC_REPORT.md`: final model, prediction, explanation, interface, and privacy checks.
