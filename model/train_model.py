"""
Run from the repo root:  python -m model.train_model
Requires datasets in data/raw/ (see docs/architecture.md for download instructions).
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

from model.preprocess import clean_text

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')


def load_datasets() -> pd.DataFrame:
    dfs = []

    phishing_csv = os.path.join(RAW_DIR, 'phishing_email.csv')
    if os.path.exists(phishing_csv):
        df = pd.read_csv(phishing_csv)
        df = df.rename(columns={'text_combined': 'text', 'Email Text': 'text'})
        dfs.append(df[['text', 'label']])

    for source in ['human-generated', 'llm-generated']:
        for fname, lbl in [('legit.csv', 0), ('phishing.csv', 1)]:
            path = os.path.join(RAW_DIR, source, fname)
            if os.path.exists(path):
                tmp = pd.read_csv(path, engine='python', on_bad_lines='skip')
                text_col = next((c for c in ['text', 'body', 'Email Text'] if c in tmp.columns), tmp.columns[0])
                dfs.append(pd.DataFrame({'text': tmp[text_col], 'label': lbl}))

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    df = df[df['text'].str.len() > 20].reset_index(drop=True)
    return df


def train():
    print("Loading datasets...")
    df = load_datasets()
    print(f"  {len(df)} rows — {df['label'].sum()} phishing, {(df['label']==0).sum()} legitimate")

    df['text_clean'] = df['text'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    vectorizer = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2), sublinear_tf=True, stop_words='english'
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced'),
    }

    best_name, best_model, best_f1 = None, None, -1
    for name, clf in models.items():
        print(f"Training {name}...")
        clf.fit(X_train_tfidf, y_train)
        preds = clf.predict(X_test_tfidf)
        f1 = f1_score(y_test, preds, average='weighted')
        print(f"  F1: {f1:.4f}")
        print(classification_report(y_test, preds, target_names=['Legitimate', 'Phishing']))
        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, clf

    print(f"\nBest model: {best_name} (F1={best_f1:.4f})")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(ARTIFACTS_DIR, 'model.pkl'))
    joblib.dump(vectorizer, os.path.join(ARTIFACTS_DIR, 'vectorizer.pkl'))
    print("Saved model.pkl and vectorizer.pkl to model/artifacts/")


if __name__ == '__main__':
    train()
