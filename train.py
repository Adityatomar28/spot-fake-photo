import glob
import os
import pickle
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve

from features import extract_features, FEATURE_NAMES

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def load_paths(folder):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def build_dataset():
    real_paths = load_paths("real")
    screen_paths = load_paths("screen")

    if len(real_paths) == 0 or len(screen_paths) == 0:
        print("ERROR: put ~50 images in real/ and ~50 in screen/ first.")
        sys.exit(1)

    print(f"Found {len(real_paths)} real photos, {len(screen_paths)} screen photos")

    X, y, paths = [], [], []
    for p in real_paths:
        try:
            X.append(extract_features(p))
            y.append(0)
            paths.append(p)
        except Exception as e:
            print(f"  skip {p}: {e}")
    for p in screen_paths:
        try:
            X.append(extract_features(p))
            y.append(1)
            paths.append(p)
        except Exception as e:
            print(f"  skip {p}: {e}")

    return np.array(X), np.array(y), paths


def main():
    X, y, paths = build_dataset()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_splits = min(5, np.bincount(y).min())
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_score = -1.0
    best_params = None
    for C in [0.01, 0.1, 1, 10, 100]:
        for solver, l1_ratio, label in [
            ("lbfgs", 0.0, "l2"),
            ("liblinear", 1.0, "l1"),
        ]:
            clf_candidate = LogisticRegression(
                C=C,
                solver=solver,
                l1_ratio=l1_ratio,
                max_iter=2000,
                class_weight="balanced",
            )
            scores = cross_val_predict(clf_candidate, Xs, y, cv=skf)
            score = accuracy_score(y, scores)
            print(f"C={C}, penalty={label}, cv acc={score:.4f}")
            if score > best_score:
                best_score = score
                best_params = {
                    "C": C,
                    "solver": solver,
                    "l1_ratio": l1_ratio,
                    "label": label,
                }

    print(f"\nBest hyperparams: C={best_params['C']}, penalty={best_params['label']}, cv acc={best_score:.4f}")

    clf = LogisticRegression(
        C=best_params["C"],
        solver=best_params["solver"],
        l1_ratio=best_params["l1_ratio"],
        max_iter=2000,
        class_weight="balanced",
    )

    y_pred = cross_val_predict(clf, Xs, y, cv=skf)
    cv_acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    print(f"\n{n_splits}-fold cross-validated accuracy: {cv_acc:.3f}")
    print("Confusion matrix [rows=true 0/1, cols=pred 0/1]:")
    print(cm)

    clf.fit(Xs, y)

    probs_cv = clf.predict_proba(Xs)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, probs_cv)
    candidates = [(t, tp) for f, tp, t in zip(fpr, tpr, thresholds) if f <= 0.05]
    if candidates:
        best_threshold = max(candidates, key=lambda x: x[1])[0]
    else:
        best_threshold = 0.5
    best_threshold = float(np.clip(best_threshold, 0.05, 0.95))
    print(f"Suggested decision threshold (favoring precision): {best_threshold:.3f}")

    with open("model.pkl", "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "clf": clf,
            "feature_names": FEATURE_NAMES,
            "threshold": best_threshold,
            "cv_accuracy": cv_acc,
        }, f)

    print("\nSaved model.pkl")

    t0 = time.time()
    n_bench = min(20, len(paths))
    for p in paths[:n_bench]:
        extract_features(p)
    elapsed = (time.time() - t0) / n_bench * 1000
    print(f"Approx feature-extraction latency: {elapsed:.2f} ms/image (CPU)")


if __name__ == "__main__":
    main()
