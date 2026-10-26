import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from feature_extractor import extract_features, FEATURE_NAMES


def train_from_kaggle_dataset(csv_path='dataset.csv'):
    """
    Train using URLs from Kaggle dataset, extracting our improved 20 features.
    """
    print(f"[*] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"[*] Dataset shape: {df.shape}")

    url_col = None
    for col in ['url', 'URL', 'Url', 'domain', 'Domain']:
        if col in df.columns:
            url_col = col
            break

    label_col = None
    for col in ['status', 'label', 'Label', 'Status', 'phishing']:
        if col in df.columns:
            label_col = col
            break

    if not url_col:
        print("[!] Could not find URL column. Available columns:", list(df.columns))
        print("[!] Falling back to demo model training...")
        return train_demo_model()

    print(f"[*] Using URL column: '{url_col}', Label column: '{label_col}'")
    print("[*] Extracting improved features from dataset URLs...")

    X = []
    y = []
    skipped = 0

    for _, row in df.iterrows():
        try:
            url = str(row[url_col])
            feats = extract_features(url)
            X.append(feats)

            if label_col:
                label = row[label_col]
                if isinstance(label, str):
                    y.append(1 if label.lower() == 'phishing' else 0)
                else:
                    y.append(int(label))
            else:
                y.append(0)
        except Exception:
            skipped += 1
            continue

    print(f"[*] Extracted features for {len(X)} URLs. Skipped: {skipped}")

    X = np.array(X)
    y = np.array(y)

    print(f"[*] Class distribution — Legitimate: {np.sum(y == 0)}, Phishing: {np.sum(y == 1)}")
    return _train_and_save(X, y)


def train_demo_model():
    """
    Train with realistic synthetic data matching the 20 features.
    """
    print("[*] Building realistic synthetic training data...")
    np.random.seed(42)
    n = 3000

    # Legitimate URL feature distributions
    legit = np.column_stack([
        np.random.randint(15, 45, n),    # 1. URL length
        np.random.randint(8, 20, n),     # 2. Hostname length
        np.random.randint(0, 15, n),     # 3. Path length
        np.random.randint(1, 3, n),      # 4. Dots in hostname
        np.zeros(n),                     # 5. Hyphens in hostname
        np.zeros(n),                     # 6. Has @
        np.ones(n),                      # 7. Has HTTPS
        np.random.randint(0, 2, n),      # 8. Subdomain count
        np.zeros(n),                     # 9. Has IP address
        np.zeros(n),                     # 10. Digits in hostname
        np.random.uniform(0.0, 0.05, n), # 11. Digit ratio in URL
        np.zeros(n),                     # 12. Keyword in hostname
        np.random.choice([0, 1], n, p=[0.9, 0.1]), # 13. Keyword in path
        np.zeros(n),                     # 14. Suspicious TLD
        np.ones(n),                      # 15. Trusted TLD
        np.random.randint(0, 2, n),      # 16. Special char count
        np.random.uniform(2.0, 3.2, n),  # 17. Hostname entropy
        np.random.randint(0, 3, n),      # 18. Slash count
        np.zeros(n),                     # 19. Has port
        np.random.choice([0, 1], n, p=[0.5, 0.5])  # 20. Is top domain
    ])

    # Phishing URL feature distributions
    phish = np.column_stack([
        np.random.randint(55, 140, n),   # 1. URL length
        np.random.randint(22, 55, n),    # 2. Hostname length
        np.random.randint(15, 60, n),    # 3. Path length
        np.random.randint(2, 6, n),      # 4. Dots in hostname
        np.random.randint(1, 5, n),      # 5. Hyphens in hostname
        np.random.choice([0, 1], n, p=[0.85, 0.15]), # 6. Has @
        np.random.choice([0, 1], n, p=[0.6, 0.4]),   # 7. Has HTTPS
        np.random.randint(1, 4, n),      # 8. Subdomain count
        np.random.choice([0, 1], n, p=[0.9, 0.1]),   # 9. Has IP address
        np.random.randint(1, 6, n),      # 10. Digits in hostname
        np.random.uniform(0.1, 0.4, n),  # 11. Digit ratio in URL
        np.random.choice([0, 1], n, p=[0.3, 0.7]),   # 12. Keyword in hostname
        np.random.choice([0, 1], n, p=[0.3, 0.7]),   # 13. Keyword in path
        np.random.choice([0, 1], n, p=[0.4, 0.6]),   # 14. Suspicious TLD
        np.random.choice([0, 1], n, p=[0.6, 0.4]),   # 15. Trusted TLD
        np.random.randint(3, 10, n),     # 16. Special char count
        np.random.uniform(3.5, 4.8, n),  # 17. Hostname entropy
        np.random.randint(3, 9, n),      # 18. Slash count
        np.random.choice([0, 1], n, p=[0.9, 0.1]),   # 19. Has port
        np.zeros(n)                      # 20. Is top domain (phishing is not real top domain)
    ])

    X = np.vstack([legit, phish])
    y = np.array([0] * n + [1] * n)
    return _train_and_save(X, y)


def _train_and_save(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n[*] Training regularized Random Forest on {len(X_train)} samples...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[✓] Test Accuracy: {acc * 100:.2f}%")
    print("\n[*] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

    # Save model and feature count
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('feature_count.pkl', 'wb') as f:
        pickle.dump(X.shape[1], f)

    print(f"[✓] Model saved -> model.pkl")
    print(f"[✓] Feature count saved -> feature_count.pkl ({X.shape[1]} features)")

    # Comprehensive verification check on known real-world URLs
    print("\n[*] Sanity check on real-world URLs:")
    test_urls = [
        ("https://www.google.com", "Legitimate"),
        ("google.com", "Legitimate"),
        ("https://github.com/login", "Legitimate"),
        ("https://github.com", "Legitimate"),
        ("https://apple.com/iphone", "Legitimate"),
        ("https://youtube.com", "Legitimate"),
        ("https://www.amazon.in/", "Legitimate"),
        ("http://paypa1-secure-login.xyz/update", "Phishing"),
        ("http://192.168.1.1/verify-account", "Phishing"),
        ("http://secure-banking-login-confirm.tk/update?id=12345", "Phishing"),
        ("http://login-verify-account.xyz/secure", "Phishing")
    ]

    for url, expected in test_urls:
        feats = np.array(extract_features(url)).reshape(1, -1)
        pred = model.predict(feats)[0]
        conf = model.predict_proba(feats)[0]
        result = "Phishing" if pred == 1 else "Legitimate"
        confidence = conf[pred] * 100
        match = "✓" if result == expected else "✗"
        print(f"  {match} [{expected:10s}] {result:<10s} ({confidence:.1f}%) — {url}")

    return model


if __name__ == '__main__':
    if os.path.exists('dataset.csv'):
        print("[*] Found dataset.csv — training with dataset URLs + improved feature extractor")
        train_from_kaggle_dataset()
    else:
        print("[!] dataset.csv not found — training with synthetic data")
        train_demo_model()