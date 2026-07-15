# Deployment instructions

## Streamlit Community Cloud

1. Upload this complete directory to a GitHub repository. Keep all model files in `models/`.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and select **Create app**.
3. Select the repository and default branch, then set the main file path to `app.py`.
4. Select **Deploy** and allow the dependencies and 40 model artifacts to load.
5. Test both the 3- and 6-month modules and confirm that each result reports 20 locked models.
6. Review the layout, privacy statement, and model limitations on desktop and mobile devices.

This project does not require secrets, a database, or an external API. If the repository is public, confirm that it contains only the deidentified test data and model objects required for deployment.

## Docker

```bash
docker build -t als-depression-risk .
docker run --rm -p 8501:8501 als-depression-risk
```

Open `http://127.0.0.1:8501`.
