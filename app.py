import os
from urllib.parse import urlparse
import joblib
import numpy as np
import streamlit as st
from feature_extractor import extract_features, normalize_url, FEATURE_NAMES
from explain import feature_risk

st.set_page_config(page_title="URL Phishing Detector", page_icon="🛡️", layout="centered")


@st.cache_resource
def load_detector():
    if not os.path.exists('model.joblib'):
        return None
    return joblib.load('model.joblib')   # only load files you trained yourself


bundle = load_detector()

st.title("🛡️ URL Phishing Detection Tool")
st.markdown("Analyze any URL using lexical feature extraction and a Random Forest classifier.")
st.markdown("---")

if bundle is None:
    st.error("⚠️ `model.joblib` not found. Run `python model.py dataset.csv` first.")
    st.stop()

model, baseline = bundle["model"], bundle["baseline"]

url_input = st.text_input(
    "Enter URL to inspect",
    placeholder="e.g. google.com, https://github.com/login, http://paypa1-secure-login.xyz/update",
)

if st.button("🔍 Check URL", use_container_width=True, type="primary"):
    raw_url = url_input.strip()
    if not raw_url:
        st.warning("Please enter a URL first.")
        st.stop()

    features = extract_features(raw_url)
    if len(features) != model.n_features_in_:
        st.error(f"Model expects {model.n_features_in_} features, extractor gave {len(features)}. Re-run model.py.")
        st.stop()

    p_phish, risk = feature_risk(model, features, baseline)
    st.markdown("---")
    if p_phish >= 0.5:
        st.error("🚨 **PHISHING URL DETECTED!** This URL shows high-risk characteristics.")
        st.markdown(f"### Phishing Risk: `{p_phish * 100:.1f}%`")
    else:
        st.success("✅ **URL APPEARS LEGITIMATE.**")
        st.markdown(f"### Safety Confidence: `{(1 - p_phish) * 100:.1f}%`")
    st.progress(float(p_phish))

    if not raw_url.lower().startswith(("http://", "https://")):
        st.caption("ℹ️ No scheme given, so HTTPS was assumed for the 'Has HTTPS' feature.")
    st.caption("Lexical model only: it cannot see page content, redirects, or domain age.")

    st.markdown("---")
    st.subheader("🌐 URL Breakdown")
    parsed = urlparse(normalize_url(raw_url))
    host = (parsed.hostname or '').lower()
    host = host[4:] if host.startswith('www.') else host
    c1, c2, c3 = st.columns(3)
    c1.metric("Protocol", parsed.scheme.upper())
    c2.metric("Domain", host or "N/A")
    c3.metric("Path Length", len(parsed.path or ''))

    st.markdown("---")
    st.subheader("📊 Per-Feature Risk Breakdown")
    st.markdown("How much each feature pushes the phishing probability **up (+)** or **down (−)**, "
                "measured by swapping it for a typical legitimate value.")
    order = np.argsort(-np.abs(risk))
    rows = [{
        "Feature": FEATURE_NAMES[i],
        "Value": round(float(features[i]), 3),
        "Typical legit": round(float(baseline[i]), 3),
        "Risk contribution (pp)": round(float(risk[i]) * 100, 1),
    } for i in order]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.bar_chart({FEATURE_NAMES[i]: float(risk[i]) * 100 for i in order[:8]})

with st.sidebar:
    st.header("About This Tool")
    st.markdown("An ensemble **Random Forest** trained on ~11k labelled URLs using 20 lexical features: "
                "protocol, hostname vs path keywords, entropy, digits, TLD risk, subdomain depth, "
                "and a known-top-domain flag.")
    st.markdown("---")
    st.header("💡 Try These")
    for u in ["https://www.google.com", "https://github.com/login",
              "http://paypa1-secure-login.xyz/update", "http://192.168.1.1/verify-account"]:
        st.code(u, language=None)
