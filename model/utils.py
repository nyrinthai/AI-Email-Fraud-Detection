import joblib
import os


ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')


def load_model():
    return joblib.load(os.path.join(ARTIFACTS_DIR, 'model.pkl'))


def load_vectorizer():
    return joblib.load(os.path.join(ARTIFACTS_DIR, 'vectorizer.pkl'))


def label_to_str(label: int) -> str:
    return 'Phishing' if label == 1 else 'Legitimate'
