from model.preprocess import clean_text
from model.utils import load_model, load_vectorizer, label_to_str


_model = None
_vectorizer = None


def _load():
    global _model, _vectorizer
    if _model is None:
        _model = load_model()
        _vectorizer = load_vectorizer()


def predict(email_text: str) -> dict:
    _load()
    cleaned = clean_text(email_text)
    features = _vectorizer.transform([cleaned])
    label = int(_model.predict(features)[0])
    confidence = float(_model.predict_proba(features)[0][label])
    return {
        'label': label,
        'verdict': label_to_str(label),
        'confidence': confidence,
    }
