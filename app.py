import os
import pickle
from urllib.parse import urlparse
import numpy as np
import streamlit as st
from feature_extractor import extract_features, normalize_url, FEATURE_NAMES

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="URL Phishing Detector",
    page_icon="🛡️",
    layout="centered"
)

# ─── Load Model & Feature Count ──────────────────────────────────────────────
@st.cache_resource
def load_detector():
    if not os.path.exists('model.pkl'):
        return None, None

    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    expected_features = 20
    if os.path.exists('feature_count.pkl'):
        with open('feature_count.pkl', 'rb') as f:
            expected_features = pickle.load(f)

    return loaded_model, expected_features


model, expected_count = load_detector()

# ─── UI Header ───────────────────────────────────────────────────────────────
st.title("🛡️ URL Phishing Detection Tool")
st.markdown("Analyze any URL using machine learning feature extraction to detect phishing and malicious attempts.")
st.markdown("---")

if model is None:
    st.error("⚠️ `model.pkl` not found! Please run `python model.py` in the terminal to train the model first.")
    st.stop()

# ─── Input Form ──────────────────────────────────────────────────────────────
url_input = st.text_input(
    "Enter URL to inspect",
    placeholder="e.g. google.com, https://github.com/login, or http://paypa1-secure-login.xyz/update"
)

col_check, col_clear = st.columns([4, 1])
with col_check:
    check_btn = st.button("🔍 Check URL", use_container_width=True, type="primary")

if check_btn:
    if not url_input.strip():
        st.warning("Please enter a URL first.")
    else:
        raw_url = url_input.strip()
        normalized = normalize_url(raw_url)

        # Extract features
        features = extract_features(raw_url)
        features_array = np.array(features).reshape(1, -1)

        if features_array.shape[1] != model.n_features_in_:
            st.error(f"Feature count mismatch: model expects {model.n_features_in_} features, but extractor generated {features_array.shape[1]}. Please re-run `python model.py`.")
            st.stop()

        prediction = model.predict(features_array)[0]
        confidence = model.predict_proba(features_array)[0]

        st.markdown("---")

        # Result Banner
        if prediction == 1:
            conf_val = confidence[1] * 100
            st.error("🚨 **PHISHING URL DETECTED!** This URL shows high-risk characteristics.")
            st.markdown(f"### Phishing Risk Confidence: `{conf_val:.1f}%`")
            st.progress(conf_val / 100)
        else:
            conf_val = confidence[0] * 100
            st.success("✅ **URL APPEARS LEGITIMATE.** No malicious patterns detected.")
            st.markdown(f"### Safety Confidence: `{conf_val:.1f}%`")
            st.progress(conf_val / 100)

        # URL Parsing Breakdown
        st.markdown("---")
        st.subheader("🌐 URL Breakdown")
        parsed = urlparse(normalized)
        clean_host = (parsed.hostname or '').lower()
        if clean_host.startswith('www.'):
            clean_host = clean_host[4:]

        u_c1, u_c2, u_c3 = st.columns(3)
        u_c1.metric("Protocol", parsed.scheme.upper() if parsed.scheme else "HTTPS")
        u_c2.metric("Domain", clean_host if clean_host else "N/A")
        u_c3.metric("Path Length", len(parsed.path) if parsed.path else 0)

        # Detailed Feature Analysis
        st.markdown("---")
        st.subheader("📊 Feature Analysis")
        st.markdown("Detailed breakdown of the 20 structural, lexical, and behavioral metrics:")

        col1, col2 = st.columns(2)
        for i, (name, value) in enumerate(zip(FEATURE_NAMES, features)):
            # Format float values nicely
            display_val = f"{value:.2f}" if isinstance(value, float) else str(value)
            if i % 2 == 0:
                col1.metric(name, display_val)
            else:
                col2.metric(name, display_val)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
    This detector evaluates URLs using an **ensemble Random Forest classifier** trained on thousands of verified phishing and legitimate URLs.

    **Key Signals Inspected:**
    - 🔒 Protocol (HTTPS vs insecure HTTP)
    - 🏷️ Hostname vs Path keywords
    - 🌐 Domain entropy & randomness
    - 🔢 Digits and special characters
    - ⚠️ Suspicious vs trusted TLDs
    - 🧭 Subdomain and path depth
    - 🏛️ Verified top domain status
    """)

    st.markdown("---")
    st.header("💡 Try These Examples")

    st.markdown("**Legitimate URLs:**")
    st.code("https://www.google.com", language=None)
    st.code("https://github.com/login", language=None)
    st.code("https://apple.com/iphone", language=None)

    st.markdown("**Phishing URLs:**")
    st.code("http://paypa1-secure-login.xyz/update", language=None)
    st.code("http://192.168.1.1/verify-account", language=None)
    st.code("http://login-verify-account.xyz/secure", language=None)