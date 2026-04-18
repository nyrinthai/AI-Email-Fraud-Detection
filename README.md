# Phishing Email Detection System

**CECS 458 — Team 10 — Phase 3**

A machine learning system that classifies email text as phishing or legitimate using a TF-IDF + Logistic Regression pipeline. Achieves **98.72% weighted F1** on a held-out test set of 16,622 emails.

---

## How It Works

```
User Input → PII Scrubbing → TF-IDF Vectorization → Classifier → Verdict + Confidence
```

1. User pastes email text into the UI
2. Regex strips personal info (emails, phones, URLs, IPs)
3. Cleaned text is converted to a 10,000-feature TF-IDF vector
4. Logistic Regression predicts phishing or legitimate with a confidence score
5. Result is displayed in the Streamlit UI

---

## Quick Start

**Requirements:** Python 3.9+

```bash
# 1. Clone the repo
git clone https://github.com/nyrinthai/AI-Email-Fraud-Detection.git
cd AI-Email-Fraud-Detection

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add model artifacts (get these from the Model Builder)
#    Place model.pkl and vectorizer.pkl in:  model/artifacts/

# 5. Run the app
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
phishing-detector/
├── app/
│   └── streamlit_app.py        ← Streamlit UI
├── model/
│   ├── preprocess.py           ← PII scrubbing + text cleaning
│   ├── predict.py              ← inference pipeline
│   ├── train_model.py          ← training script
│   ├── utils.py                ← shared helpers
│   └── artifacts/
│       ├── model.pkl           ← trained classifier (not in git)
│       └── vectorizer.pkl      ← fitted TF-IDF vectorizer (not in git)
├── data/
│   ├── raw/                    ← Kaggle source datasets (not in git)
│   └── processed/              ← cleaned/merged data (not in git)
├── docs/
│   ├── architecture.md         ← system design + pipeline diagram
│   └── code_explanations.md    ← what each file does
├── phishing_model_builder.ipynb ← full training notebook (Google Colab)
├── requirements.txt
└── README.md
```

---

## Model Performance

| Metric | Value |
|---|---|
| Selected model | Logistic Regression |
| Weighted F1 | **0.9872** |
| F1 on phishing class | **0.9876** |
| Test set size | 16,622 emails |
| Feature space | 10,000 TF-IDF terms (unigrams + bigrams) |

---

## Training Data

| Source | Rows |
|---|---|
| [naserabdullahalam/phishing-email-dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) | 82,486 |
| [francescogreco97/human-llm-generated-phishing-legitimate-emails](https://www.kaggle.com/datasets/francescogreco97/human-llm-generated-phishing-legitimate-emails) | 3,595 |
| **Total (after deduplication)** | **83,106** |

The second dataset was included specifically to cover modern LLM-generated phishing emails.

---

## Reproducing the Model

1. Open `phishing_model_builder.ipynb` in Google Colab
2. Set runtime to T4 GPU (optional)
3. Add your Kaggle API token when prompted
4. Run all cells top to bottom (~15 minutes)
5. Download `model.pkl` and `vectorizer.pkl` from `model/artifacts/`

All randomness is seeded at `random_state=42` — results are fully reproducible.

---

## Future Improvements

- SHAP explainability to highlight which words triggered the verdict
- BERT or other transformer-based models
- Expanded LLM-generated phishing data for better coverage of modern threats

---

## Team

| Role | Responsibility |
|---|---|
| Model Builder | Training pipeline, model artifacts, notebook |
| UI Integrator | Streamlit app, user input handling |
| Technical Writer | Architecture docs, code explanations |
| Project Manager | Progress reports, timeline, submission |
