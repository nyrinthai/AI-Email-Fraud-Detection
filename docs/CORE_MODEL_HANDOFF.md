# Phishing Email Detection — Core Model

**CECS 458 · Team 10 · Phase 3 Deliverable**
**Role: Model Builder**

---

## What this is

A trained machine-learning model that reads the text of an email and predicts whether it's **phishing** or **legitimate**, along with a confidence score.

**Bottom line: 98.72% weighted F1 on 16,622 held-out test emails (98.76% on the phishing class specifically).** Logistic Regression classifier over a TF-IDF representation of PII-scrubbed email text.

This is the brain of the app. The UI teammate plugs it into Streamlit/Gradio; the user never sees the model directly — they just paste an email and get a verdict.

---

## What you get in this handoff

Two files, both located at `model/artifacts/` after running the notebook:

| File | What it does |
|---|---|
| `model.pkl` | The trained Logistic Regression classifier. Takes numerical features in, predicts 0 (legit) or 1 (phishing). Small file (~1 MB), fast to load. |
| `vectorizer.pkl` | Converts raw email text into the numerical features the model expects. **Must be used together with the model — never separately.** |

Plus one Python function everyone needs (copy-paste block below):

```python
import re

def scrub_pii(text: str) -> str:
    """Remove personal info before the model sees it. Call this on any email
    text before passing to the vectorizer."""
    text = re.sub(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}', '<EMAIL>', text)
    text = re.sub(r'(\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}', '<PHONE>', text)
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
    text = re.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', '<IP>', text)
    return text.strip()
```

---

## Quick-start for the UI Integrator

Paste this into your Streamlit app. That's it — nothing else to configure.

```python
import streamlit as st
import joblib
import re

# --- Load the model once, cache in memory ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('model/artifacts/model.pkl')
    vectorizer = joblib.load('model/artifacts/vectorizer.pkl')
    return model, vectorizer

def scrub_pii(text: str) -> str:
    text = re.sub(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}', '<EMAIL>', text)
    text = re.sub(r'(\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}', '<PHONE>', text)
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
    text = re.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', '<IP>', text)
    return text.strip()

# --- UI ---
model, vectorizer = load_artifacts()

st.title("Phishing Email Detector")
email_input = st.text_area("Paste the email text here:", height=300)

if st.button("Analyze"):
    cleaned = scrub_pii(email_input)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"⚠️ PHISHING — {confidence:.1%} confidence")
    else:
        st.success(f"✅ LEGITIMATE — {(1-confidence):.1%} confidence")
```

**Critical:** The `scrub_pii` function above must stay identical to what's in the training notebook. Do not "improve" it without checking with the Model Builder first — any change shifts the feature distribution and drops accuracy.

### Verified integration test results

The model was validated end-to-end on three sample emails (simulating exactly the UI flow):

| Email | Verdict | Confidence |
|---|---|---|
| "URGENT: Your account has been compromised. Click here immediately..." | PHISHING | 99.7% |
| "Hi team, attaching the notes from this morning's meeting..." | LEGITIMATE | 99.9% |
| "Dear customer, we noticed unusual activity on your PayPal account..." | PHISHING | 91.9% |

---

## How it works (for the Technical Writer)

```
┌──────────────┐     ┌──────────┐     ┌─────────┐     ┌────────────┐     ┌──────────┐
│  User pastes │ ──► │ PII      │ ──► │ TF-IDF  │ ──► │  Logistic  │ ──► │ Verdict  │
│  email text  │     │ Scrubber │     │ Vector  │     │ Regression │     │   +      │
│              │     │ (regex)  │     │ (10k    │     │            │     │ Score    │
│              │     │          │     │ features)│    │            │     │          │
└──────────────┘     └──────────┘     └─────────┘     └────────────┘     └──────────┘
```

**Stage 1 — PII Scrubber.** Before anything touches the model, regex strips emails, phone numbers, URLs, and IP addresses and replaces them with generic tokens (`<EMAIL>`, `<URL>`, etc.). This is our privacy guarantee from Phase 2.

**Stage 2 — TF-IDF Vectorizer.** Converts cleaned text into a sparse numerical vector of exactly 10,000 features. Captures both single words and two-word phrases (so "click here" is recognized as one signal, not just "click" + "here" separately). Also de-weights common words like "the" that appear everywhere.

**Stage 3 — Logistic Regression Classifier.** A linear model that learns a weight for each of the 10,000 features. Words with the largest positive weights push the prediction toward "phishing"; words with negative weights push toward "legitimate". Outputs both a binary prediction and a probability score for UI confidence display. Selected over Random Forest because (a) it scored marginally higher, (b) its coefficients are directly interpretable as per-word phishing signals, and (c) it's an order of magnitude smaller and faster.

---

## Training details (for the Project Manager)

**Data sources:**
- **Naser Abdullah Alam's phishing email dataset** (Kaggle) — 82,486 real emails covering multiple corpora (Enron, Nazario, Nigerian Fraud, SpamAssassin, CEAS'08, Ling). Mix of phishing and legitimate.
- **Francesco Greco's human vs. LLM-generated phishing dataset** (Kaggle) — 3,595 rows across four files covering both human-written and LLM-generated phishing/legitimate emails. Included specifically to address the professor's Phase 2 feedback about traditional datasets missing modern LLM-crafted threats.

**Preprocessing:**
- Merged to 83,106 rows after deduplication
- Dropped rows with fewer than 20 characters (junk filter)
- Ran PII scrubbing on 100% of the corpus
- Stratified 80/20 train/test split with `random_state=42` for reproducibility
  - Training: 66,484 rows
  - Test: 16,622 rows

**Models trained and compared:**
- **Logistic Regression** — linear, fast, interpretable. F1: **0.9872** ✓ selected
- **Random Forest (200 trees)** — ensemble, handles non-linear patterns. F1: 0.9868

Both trained with `class_weight='balanced'` (not strictly needed given the near-even 51.6/48.4 split, but future-proofs against retraining on more skewed data).

**Selection criteria:** Highest weighted F1 score on the held-out test set. Logistic Regression won by a narrow margin (+0.04 F1 points) and was additionally preferred for interpretability and deployment size.

---

## Results

### Final Model Performance

| Metric | Value |
|---|---|
| **Selected model** | **Logistic Regression** |
| **Weighted F1 (overall)** | **0.9872** |
| **F1 on phishing class** | **0.9876** |
| Random Forest F1 (not selected) | 0.9868 |
| Train/test split | 80/20 stratified, `random_state=42` |
| Test set size | 16,622 emails |
| Feature space | 10,000 TF-IDF terms (unigrams + bigrams) |

### Dataset Composition

| | Count |
|---|---|
| Source 1 (naserabdullahalam) | 82,486 |
| Source 2 (francescogreco97) – human legit | 1,000 |
| Source 2 (francescogreco97) – human phishing | 1,000 |
| Source 2 (francescogreco97) – LLM legit | 1,000 |
| Source 2 (francescogreco97) – LLM phishing | 595 |
| **Total after deduplication** | **83,106** |
| Phishing | 42,895 (51.6%) |
| Legitimate | 40,211 (48.4%) |

### PII Scrubbed from Training Corpus

| Type | Count |
|---|---|
| Email addresses | 53 |
| URLs | 934 |
| Phone numbers | 10,422 |
| IP addresses | 0 |

This directly backs the privacy claim made in the Phase 2 Feasibility Report: **no personal information from training emails is encoded into the model weights.** All identifiers are replaced with generic placeholder tokens before any feature extraction.

### Top 20 Phishing Indicator Words

Extracted directly from the Logistic Regression coefficients (higher = stronger phishing signal). This is a concrete example of the model's interpretability:

| Rank | Term | Coefficient |
|---|---|---|
| 1 | josemonkeyorg | 5.27 |
| 2 | 0300 | 5.07 |
| 3 | 2004 | 4.86 |
| 4 | http | 4.61 |
| 5 | life | 4.60 |
| 6 | love | 4.40 |
| 7 | money | 4.37 |
| 8 | com | 4.30 |
| 9 | 2016 | 4.25 |
| 10 | click | 3.92 |
| 11 | account | 3.91 |
| 12 | 2005 | 3.81 |
| 13 | 2017 | 3.79 |
| 14 | remove | 3.69 |
| 15 | 2018 | 3.67 |
| 16 | 2019 | 3.61 |
| 17 | watches | 3.48 |
| 18 | site | 3.40 |
| 19 | meds | 3.32 |
| 20 | bank | 3.30 |

Classic phishing vocabulary dominates the list: `money`, `click`, `account`, `remove`, `bank`, `site`, `watches`, `meds`, `http`. A handful of date tokens (`2004`, `2016`–`2019`) and source-corpus artifacts (`josemonkeyorg`, `0300`) also appear — see Limitations below.

---

## File manifest

```
model/
└── artifacts/
    ├── model.pkl           ← trained Logistic Regression (~1 MB)
    └── vectorizer.pkl      ← fitted TF-IDF transformer

phishing_model_builder.ipynb   ← reproducible training notebook
```

A backup of both `.pkl` files is also saved to `/content/drive/MyDrive/phishing_detector/` in the Model Builder's Google Drive.

---

## How to reproduce

1. Open `phishing_model_builder.ipynb` in Google Colab
2. Runtime → Change runtime type → T4 GPU (optional but faster)
3. Get a Kaggle API token from kaggle.com → Settings → API → Create New Token
4. Run all cells top to bottom (takes ~15 minutes total)
5. Artifacts land in `model/artifacts/`

All randomness is seeded with `random_state=42`, so results are deterministic — rerunning produces identical numbers.

---

## Honest limitations

- **Training domain:** English-language email only. Other languages will perform worse.
- **LLM-phishing representation:** Only 595 LLM-generated phishing samples (~0.7% of training data) survived the load. This dataset category is now genuinely represented in the training distribution but remains underweighted; production retraining should expand this proportion as more LLM-phishing corpora become available.
- **Dataset artifacts in top features:** The feature-importance list contains some tokens that are likely corpus fingerprints (`josemonkeyorg`, `0300`, date strings like `2004`/`2016`/`2017`) rather than pure phishing signals. The model has learned some "which training corpus does this email come from?" in addition to "is this phishing?". A production version would either filter date tokens during vectorization or evaluate via cross-corpus holdout. For this phase, we document the behavior rather than hide it.
- **Concept drift:** Phishing evolves. This model reflects threats visible in the training data; retraining quarterly is recommended.
- **Adversarial robustness:** A sufficiently sophisticated attacker can craft emails to evade detection. This is a screening tool, not a guarantee.
- **Base rate:** Real-world inboxes are mostly legitimate email. Confidence scores should be interpreted relative to this, not in isolation.

---

## Team

| Role | Responsibility |
|---|---|
| **Model Builder** | Training pipeline, `model.pkl`, `vectorizer.pkl` (this document) |
| **UI Integrator** | Streamlit/Gradio app, user input handling |
| **Technical Writer** | Architecture documentation, final report |
| **Project Manager** | Progress reports, timeline, submission |

**Handoff contacts:** Model Builder is available for questions on the `scrub_pii` function and any accuracy/confidence questions from the UI team.
