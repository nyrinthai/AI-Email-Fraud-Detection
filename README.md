# Phishing Email Detection System

**CECS 458 — Team 10 — Final Project (Phase 4)**

An AI-powered system that classifies email text as **phishing** or **legitimate** using a TF-IDF + Logistic Regression pipeline. The model achieves **98.72% weighted F1-score** on a held-out test set of 16,622 emails.

---

## Overview

Phishing emails remain one of the most common cyber threats, especially with the rise of AI-generated attacks. This project provides a lightweight, accessible, and interpretable solution for detecting fraudulent emails.

Unlike enterprise-focused tools, this system is designed for:

* Students
* Individual users
* Small businesses

---

## How It Works

```
User Input → PII Scrubbing → TF-IDF Vectorization → Classifier → Verdict + Confidence
```

1. User inputs email text through the UI
2. Regex removes sensitive data (emails, phones, URLs, IPs)
3. Text is converted into a 10,000-feature TF-IDF vector
4. Logistic Regression predicts phishing or legitimate
5. Result is displayed with a confidence score

---

## Quick Start

**Requirements:** Python 3.9+

```bash
# 1. Clone the repo
git clone https://github.com/nyrinthai/AI-Email-Fraud-Detection.git
cd AI-Email-Fraud-Detection

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add model artifacts
# Place model.pkl and vectorizer.pkl in: model/artifacts/

# 5. Run the app
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
phishing-detector/
├── app/
│   └── streamlit_app.py
├── model/
│   ├── preprocess.py
│   ├── predict.py
│   ├── train_model.py
│   ├── utils.py
│   └── artifacts/
│       ├── model.pkl
│       └── vectorizer.pkl
├── data/
├── docs/
│   ├── architecture.md
│   └── code_explanations.md
├── phishing_model_builder.ipynb
├── requirements.txt
└── README.md
```

---

## Model Performance

| Metric              | Value               |
| ------------------- | ------------------- |
| Model               | Logistic Regression |
| Weighted F1         | **0.9872**          |
| F1 (Phishing Class) | **0.9876**          |
| Test Set Size       | 16,622 emails       |
| Feature Space       | 10,000 TF-IDF terms |

### Evaluation Summary

* High precision and recall for phishing detection
* Low false negative rate (important for security use cases)
* Balanced performance across both classes

---

## Training Data

| Source                        | Rows       |
| ----------------------------- | ---------- |
| Kaggle Phishing Email Dataset | 82,486     |
| Human + LLM-generated Dataset | 3,595      |
| **Total**                     | **83,106** |

The inclusion of LLM-generated phishing emails improves detection of modern, sophisticated attacks.

---

## Reproducing the Model

1. Open `phishing_model_builder.ipynb` in Google Colab
2. Set runtime (GPU optional)
3. Add Kaggle API token
4. Run all cells (~15 minutes)
5. Download:

   * `model.pkl`
   * `vectorizer.pkl`

All randomness is seeded (`random_state=42`) for reproducibility.

---

## Business Applicability

This system addresses key limitations in existing solutions:

| Feature      | Existing Tools | Our System                 |
| ------------ | -------------- | -------------------------- |
| Cost         | High           | Low                        |
| Target Users | Enterprises    | Students / Individuals     |
| Transparency | Low            | High (interpretable model) |
| Flexibility  | Limited        | Customizable               |

Potential deployment options:

* Browser extension
* Email filtering API
* Integration with existing email clients

---

## Future Improvements

* SHAP explainability for highlighting suspicious words
* Transformer-based models (e.g., BERT)
* Multi-language support
* Continuous retraining for evolving phishing tactics

---

## Team

| Member               | Contribution       |
| -------------------- | ------------------ |
| Aayush Roy           | Model Development  |
| Nyrin Thai           | UI & Integration   |
| Sean Luke Serranilla | Documentation      |
| Sheesh Dhawan        | Project Management |

---

## Conclusion

This project demonstrates how machine learning can be applied to improve email security in an accessible and transparent way. With strong performance and a lightweight design, the system provides a practical alternative to traditional phishing detection tools.

---
