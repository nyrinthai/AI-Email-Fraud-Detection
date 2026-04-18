import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
from model.predict import predict

st.set_page_config(page_title="Phishing Email Detector", page_icon="shield")

st.title("Phishing Email Detector")
st.caption("CECS 458 — Team 10 | Paste an email below to check if it's phishing or legitimate.")

email_input = st.text_area("Email text:", height=300, placeholder="Paste the full email text here...")

if st.button("Analyze", type="primary"):
    if not email_input.strip():
        st.warning("Please paste some email text first.")
    else:
        result = predict(email_input)
        if result['label'] == 1:
            st.error(f"PHISHING — {result['confidence']:.1%} confidence")
        else:
            st.success(f"LEGITIMATE — {result['confidence']:.1%} confidence")
