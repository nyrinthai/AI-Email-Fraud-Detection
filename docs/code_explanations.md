# Code Explanations

## model/preprocess.py

Contains all text preprocessing logic shared between training and inference.

- `scrub_pii(text)` — strips emails, phone numbers, URLs, and IP addresses using regex and replaces them with generic tokens (`<EMAIL>`, `<PHONE>`, `<URL>`, `<IP>`).
- `clean_text(text)` — calls `scrub_pii`, then lowercases and collapses whitespace.

**Important:** this function must remain identical between training and the UI. Any change shifts the feature distribution and will degrade accuracy.

---

## model/utils.py

Helper functions used across the project.

- `load_model()` — loads `model/artifacts/model.pkl` with joblib.
- `load_vectorizer()` — loads `model/artifacts/vectorizer.pkl` with joblib.
- `label_to_str(label)` — converts `0` → `"Legitimate"`, `1` → `"Phishing"`.

---

## model/predict.py

Handles inference. Used by the Streamlit app.

- `predict(email_text)` — runs the full pipeline: clean → vectorize → classify.
- Returns a dict: `{ label, verdict, confidence }`.
- Model and vectorizer are loaded once and cached in module-level variables.

---

## model/train_model.py

Trains the classifier. Run once by the Model Builder; output artifacts are committed to `model/artifacts/`.

1. Loads CSVs from `data/raw/`.
2. Applies `clean_text` to all rows.
3. Splits 80/20 stratified train/test.
4. Fits TF-IDF vectorizer on train set only.
5. Trains Logistic Regression and Random Forest, picks the higher F1.
6. Saves `model.pkl` and `vectorizer.pkl` to `model/artifacts/`.

Run with: `python -m model.train_model`

---

## app/streamlit_app.py

The user interface. Calls `predict()` from `model/predict.py` and displays the result.

- Text area for pasting email content.
- "Analyze" button triggers prediction.
- Shows green success banner for legitimate, red error banner for phishing, both with confidence %.

Run with: `streamlit run app/streamlit_app.py`

---

## How model and UI connect

```
streamlit_app.py
    └── predict(email_text)          [model/predict.py]
            ├── clean_text(text)     [model/preprocess.py]
            ├── vectorizer.transform [model/artifacts/vectorizer.pkl]
            └── model.predict        [model/artifacts/model.pkl]
```
