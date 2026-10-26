"""
Train + honestly evaluate the phishing classifier.
Usage:  python model.py [dataset.csv] [--ablate]
"""
import sys
import time
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (GroupShuffleSplit, GroupKFold,
                                     StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from feature_extractor import extract_features, registered_domain, FEATURE_NAMES

URL_COLS = ['url', 'URL', 'Url']
LABEL_COLS = ['status', 'label', 'Label', 'Status', 'phishing']


def load(csv_path):
    df = pd.read_csv(csv_path)
    url_col = next((c for c in URL_COLS if c in df.columns), None)
    label_col = next((c for c in LABEL_COLS if c in df.columns), None)
    if url_col is None or label_col is None:   # fail loudly, never fake labels
        raise ValueError(f"Need URL and label columns, found: {list(df.columns)}")

    X, y, groups, skipped = [], [], [], []
    for url, lab in zip(df[url_col], df[label_col]):
        try:
            X.append(extract_features(url))
            groups.append(registered_domain(url))
            y.append(int(str(lab).lower() in ('phishing', '1')))
        except Exception as e:
            skipped.append((url, repr(e)))
    print(f"[*] Parsed {len(X)} URLs, skipped {len(skipped)}")
    for u, e in skipped[:5]:
        print(f"    skipped: {u!r} -> {e}")
    return np.array(X, dtype=np.float32), np.array(y), np.array(groups)


def make_model():
    return RandomForestClassifier(n_estimators=200, max_depth=14,
                                  min_samples_leaf=2, random_state=42, n_jobs=-1,
                                  class_weight='balanced')


def evaluate(X, y, groups, feature_idx=None):
    """Group-aware split: the same registered domain never appears in train and test."""
    cols = feature_idx if feature_idx is not None else list(range(X.shape[1]))
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                  .split(X, y, groups))
    m = make_model().fit(X[tr][:, cols], y[tr])
    p = m.predict_proba(X[te][:, cols])[:, 1]
    pred = (p >= 0.5).astype(int)
    return m, y[te], pred, p, tr, te


def main(csv_path, ablate):
    X, y, groups = load(csv_path)
    print(f"[*] Class balance: legit={np.sum(y == 0)} phishing={np.sum(y == 1)}")

    # (A) Plain stratified random split = the headline "test accuracy" on the resume.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42,
                                          stratify=y)
    rm = make_model().fit(Xtr, ytr)
    rpred = rm.predict(Xte)
    print("\n[*] (A) Random stratified 80/20 split")
    print(classification_report(yte, rpred, target_names=['Legitimate', 'Phishing']))
    cv_r = cross_val_score(make_model(), X, y,
                           cv=StratifiedKFold(5, shuffle=True, random_state=42),
                           scoring='accuracy')
    print(f"5-fold StratifiedKFold accuracy: {cv_r.mean():.4f} +/- {cv_r.std():.4f}")

    # (B) Domain-grouped split = stricter, leakage-free estimate.
    model, yt, pred, proba, tr, te = evaluate(X, y, groups)
    print("\n[*] (B) Group-split (no shared domains between train/test) report")
    print(classification_report(yt, pred, target_names=['Legitimate', 'Phishing']))
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(yt, pred))
    print(f"ROC-AUC: {roc_auc_score(yt, proba):.4f}")

    cv = cross_val_score(make_model(), X, y, groups=groups, cv=GroupKFold(5),
                         scoring='f1_macro')
    print(f"\n5-fold GroupKFold macro-F1: {cv.mean():.4f} +/- {cv.std():.4f}")

    print("\n[*] Feature importances (impurity-based)")
    for i in np.argsort(model.feature_importances_)[::-1]:
        print(f"  {FEATURE_NAMES[i]:<22s} {model.feature_importances_[i]:.3f}")

    if ablate:   # real ablation numbers instead of guessed ones
        print("\n[*] Drop-one-feature ablation (macro-F1 on the same split)")
        base = classification_report(yt, pred, output_dict=True)['macro avg']['f1-score']
        for i, name in enumerate(FEATURE_NAMES):
            keep = [j for j in range(X.shape[1]) if j != i]
            _, yt2, pr2, _, _, _ = evaluate(X, y, groups, keep)
            f1 = classification_report(yt2, pr2, output_dict=True)['macro avg']['f1-score']
            print(f"  without {name:<22s} F1={f1:.4f} (delta {f1 - base:+.4f})")

    # Honest latency: single-call vs batch
    sample = X[te][:1]
    t = time.perf_counter()
    for _ in range(200):
        model.predict(sample)
    single_ms = (time.perf_counter() - t) / 200 * 1000
    t = time.perf_counter()
    model.predict(X[te])
    batch_us = (time.perf_counter() - t) / len(te) * 1e6
    print(f"\nLatency: {single_ms:.2f} ms/URL single call, {batch_us:.1f} us/URL batched")

    # Per-class "typical legitimate" baseline, used by the app for risk attribution
    baseline = np.median(X[tr][y[tr] == 0], axis=0)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES,
                 "baseline": baseline,
                 "sklearn": sklearn.__version__, "python": platform.python_version()},
                "model.joblib", compress=3)
    print("[✓] Saved model.joblib (with version metadata)")


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    main(args[0] if args else 'dataset.csv', '--ablate' in sys.argv)