# Credit risk modelling

Small project for exploring **default-risk** style binary classification: loading tabular credit data, training **logistic regression**, **decision trees**, or **XGBoost**, and analysing how **probability thresholds** affect **accuracy**, **precision**, **recall**, and **F1** (plus **ROC-AUC**).

## Interactive web app (recommended)

A **Streamlit** dashboard lets you upload a CSV (or use the bundled sample), tune model hyperparameters, sweep classification cutoffs, and inspect confusion matrices and reports.

### Requirements

- Python 3.10+ recommended  
- Dependencies are listed in [`requirements.txt`](requirements.txt) (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `streamlit`, `altair` for the metrics chart).

### Install

```bash
pip install -r requirements.txt
```

### Run

From the project root (where `credit_data.csv` or your data lives):

```bash
streamlit run web_app.py
```

The app opens in your browser. Use the sidebar to:

- Choose **English** or **Polski** (strings live in [`locale_strings.py`](locale_strings.py)).
- Load data, set the **target** column (default name: `default`), and optional **columns to drop** (e.g. `client_id`).
- Pick a **model** and its parameters, then adjust the **threshold sweep** range and step.
- Read metrics vs threshold, best cutoffs for each metric on the grid, and a numeric table.

### Data format

- CSV with a **binary integer target** (0/1), e.g. `default`.
- Categorical columns are one-hot encoded (`drop_first=True`); non-numeric columns that are not encoded are dropped after preprocessing.
- The sample file [`credit_data.csv`](credit_data.csv) matches the original coursework-style layout (`client_id`, numeric features, `marital_status`, `property_type`, `default`).

## Legacy script

[`CreditRisk.py`](CreditRisk.py) is an earlier **script** that trains models on `credit_data.csv` and prints threshold-style analysis. It currently has **invalid imports** (`import plt`, `from numpy.dual import solve`) and will not run until those are fixed (e.g. `import matplotlib.pyplot as plt` and remove the unused `solve` import). Prefer **`web_app.py`** for day-to-day use.

## Project layout

| File | Purpose |
|------|--------|
| `web_app.py` | Streamlit UI, training, threshold curves, best-threshold summary |
| `locale_strings.py` | English / Polish UI strings |
| `credit_data.csv` | Example dataset |
| `requirements.txt` | Python dependencies |
| `CreditRisk.py` | Legacy training script (imports need repair) |

## Licence

No licence is specified in this repository; add one if you plan to share or publish the code.
