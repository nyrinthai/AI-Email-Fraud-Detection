# System Architecture

## Overview

A machine learning pipeline that classifies email text as phishing or legitimate using TF-IDF vectorization and Logistic Regression.

## Pipeline

```
+----------------------+
|      User Input      |
|   Paste email text   |
+----------+-----------+
           |
           v
+----------------------+
|  Regex PII Scrubbing |
| Remove emails/phones |
+----------+-----------+
           |
           v
+----------------------+
|   TF-IDF Vectorizer  |
| Convert text to nums |
+----------+-----------+
           |
           v
+----------------------+
|   Classifier Model   |
|  Logistic Reg / RF   |
+----------+-----------+
           |
           v
+----------------------+
|    Output Display    |
|   Phishing / Legit   |
+----------------------+
```

## Data Flow

1. User pastes email text into the Streamlit UI.
2. PII scrubbing removes emails, phone numbers, URLs, and IP addresses (replaced with tokens like `<EMAIL>`, `<URL>`).
3. Cleaned text is transformed into a 10,000-feature TF-IDF vector (unigrams + bigrams).
4. The trained classifier predicts phishing (1) or legitimate (0) and returns a confidence score.
5. Result is displayed in the UI.

## Model Selection

| Model | Weighted F1 |
|---|---|
| **Logistic Regression** | **0.9872** (selected) |
| Random Forest | 0.9868 |

Logistic Regression was selected for marginally higher F1, faster inference, smaller artifact size, and interpretable coefficients.

## Training Data

| Source | Rows |
|---|---|
| naserabdullahalam/phishing-email-dataset (Kaggle) | 82,486 |
| francescogreco97/human-llm-generated-phishing-legitimate-emails (Kaggle) | 3,595 |
| **Total (after deduplication)** | **83,106** |
| Phishing | 42,895 (51.6%) |
| Legitimate | 40,211 (48.4%) |
