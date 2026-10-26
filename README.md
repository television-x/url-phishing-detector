# URL Phishing Classifier

Classifies a URL as phishing or legitimate using only the URL string. It needs no page fetch and no network lookup. A Random Forest runs on 20 hand-built lexical features, and a Streamlit app shows the prediction with a per-feature risk breakdown.

## Results

Dataset: 11,430 labelled URLs (5,715 legitimate, 5,715 phishing). Only the `url` and `status` columns are used.

| Evaluation | Result |
|---|---|
| Random stratified 80/20 split | 88% accuracy |
| 5-fold stratified CV | 87.4% ± 0.4% accuracy |
| Domain-grouped split (no shared domains in train and test) | 85% accuracy, ROC-AUC 0.916 |
| 5-fold GroupKFold | 0.83 ± 0.04 macro-F1 |

The grouped split is the stricter estimate. On it the model catches 82% of phishing URLs (254 of 1,390 missed) with 121 false alarms.

Inference speed on a MacBook, excluding feature extraction:

- 13.6 ms per single `predict` call
- 10.5 µs per URL when batched

## How it works

```
URL → normalize → 20 lexical features → Random Forest → probability → Streamlit UI
```

- **Features:** URL, host, and path lengths; dot, hyphen, digit, and special-character counts; hostname entropy; subdomain depth; suspicious and trusted TLD flags; sensitive keywords in the host and in the path (kept separate); IP-address host; custom port; and a known-top-domain flag.
- **Normalization:** schemeless URLs are handled, `www.` is stripped, and the scheme check is case-insensitive.
- **Model:** `RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=2, class_weight='balanced')`. These values were chosen by hand, not grid-searched.
- **Explanations:** each feature is swapped for a typical legitimate value, and the drop in phishing probability is shown as that feature's risk contribution.

## Project structure

```
feature_extractor.py   URL → 20 features
model.py               train, evaluate (random + grouped splits), save model.joblib
explain.py             per-feature risk attribution
app.py                 Streamlit dashboard
dataset.csv            training data (needs `url` and `status` columns)
```

## Run it

```bash
pip3 install scikit-learn pandas numpy joblib streamlit
python3 model.py dataset.csv            # add --ablate for a drop-one-feature table
python3 -m streamlit run app.py
```

## Findings

- A random split over-estimates accuracy by about 3 to 4 points because the same domains appear in train and test, so the grouped split is the number to trust.
- The top signals describe URL shape (slash count, digit ratio, path length), and no single feature dominates.
- Hostname entropy showed no measurable effect on this dataset, which is mostly compromised and typosquatted sites rather than generated domains.
- The first version learned that a missing `www.` meant phishing. Normalizing it removed that shortcut.

## Limitations

- The model sees only the URL string, so it cannot inspect page content, redirects, or domain age.
- URL shorteners hide the real lexical features.
- The known-top-domain list is hardcoded (about 30 domains) and is a reputation prior, not something the model learned.
- Schemeless input is assumed to be HTTPS.
- Public-suffix handling uses a small hand-written table, and `tldextract` would be the production choice.

## Next steps

- WHOIS and DNS domain-age features.
- Edit-distance checks against popular brand names for typosquatting.
- Threshold tuning against a precision-recall curve.
- Exporting the forest to ONNX or compiled C for faster single-call inference.