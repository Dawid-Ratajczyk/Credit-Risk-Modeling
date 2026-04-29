# Credit risk modelling

Streamlit web app for testing credit risk modeling models and params.

**Run it**

```bash
pip install -r requirements.txt
streamlit run web_app.py
```

Open the sidebar: language , pick your CSV or use `credit_data.csv`, set the target column (defaults to `default`), drop IDs if you want, then model + params. The chart is Altair;

Target should be 0/1 integers. 
