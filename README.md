# Credit risk modelling

Binary classification stuff — default vs not, that kind of thing. There’s a Streamlit app where you mess with thresholds and see what happens to accuracy / precision / recall / F1 and ROC-AUC. Models: logistic regression, decision tree, XGBoost.

**Run it**

```bash
pip install -r requirements.txt
streamlit run web_app.py
```

Open the sidebar: language (EN/PL text is in `locale_strings.py`), pick your CSV or use `credit_data.csv`, set the target column (defaults to `default`), drop IDs if you want, then model + params. The chart is Altair; no zoom on purpose.

Target should be 0/1 integers. Cat columns get one-hot’d; anything else weird/non-numeric tends to get dropped after preprocessing.

**Old file**

`CreditRisk.py` is leftover from before the app. Imports are broken (`plt`, `numpy.dual`) so don’t expect it to run unless you fix that. Use `web_app.py`.

**Files**

`web_app.py` — main app. `locale_strings.py` — translations. `requirements.txt` — deps. `credit_data.csv` — small example with `client_id`, some numbers, `marital_status`, `property_type`, `default`.

No licence file here; add one if you care about that.
