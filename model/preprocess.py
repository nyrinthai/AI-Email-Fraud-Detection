import re


def scrub_pii(text: str) -> str:
    text = re.sub(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}', '<EMAIL>', text)
    text = re.sub(r'(\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}', '<PHONE>', text)
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
    text = re.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', '<IP>', text)
    return text.strip()


def clean_text(text: str) -> str:
    text = scrub_pii(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
