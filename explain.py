import numpy as np


def feature_risk(model, features, baseline):
    """
    Occlusion-based attribution: for each feature, replace it with the typical
    legitimate value (median of legit training URLs) and measure how much the
    phishing probability drops. Positive = this feature pushes toward phishing.
    """
    x = np.asarray(features, dtype=np.float32).reshape(1, -1)
    p_full = model.predict_proba(x)[0][1]
    # build all 20 perturbed rows and score them in ONE batched call
    X_pert = np.repeat(x, x.shape[1], axis=0)
    for i in range(x.shape[1]):
        X_pert[i, i] = baseline[i]
    p_pert = model.predict_proba(X_pert)[:, 1]
    return p_full, p_full - p_pert        # risk contribution per feature