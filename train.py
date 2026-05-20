"""
train.py
--------
Full training pipeline for the Smart Agriculture Decision Support System.

Execution order:
  1. Load + preprocess data
  2. Train Decision Tree  → crop recommendation
  3. Train KMeans         → soil segmentation
  4. Train Linear Reg.    → yield prediction
  5. Evaluate all models
  6. Save models to /models
  7. Save plots to /results
  8. Save metric reports to /results
"""

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")

# Make sure src/ is importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import load_and_preprocess
from models        import build_decision_tree, build_kmeans, build_pca, build_linear_regression
from utils         import (
    evaluate_classifier, evaluate_regression, evaluate_clustering,
    plot_feature_importance, plot_clusters, plot_residuals,
    save_accuracy_report, save_regression_metrics, save_clustering_metrics
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

N_CLUSTERS  = 5
RANDOM_SEED = 42


def save_model(model, name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[Train] Saved model: {path}")


def main():
    print("\n" + "=" * 65)
    print("  SMART AGRICULTURE DECISION SUPPORT SYSTEM – TRAINING")
    print("=" * 65 + "\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    (X_train, X_test,
     y_clf_train, y_clf_test,
     y_reg_train, y_reg_test,
     le, scaler, feature_names, df_raw) = load_and_preprocess()

    # ── 2. Decision Tree Classifier ───────────────────────────────────────────
    print("\n[Train] Training Decision Tree Classifier …")
    dt = build_decision_tree(random_state=RANDOM_SEED)
    dt.fit(X_train, y_clf_train)
    y_dt_pred = dt.predict(X_test)

    dt_metrics, dt_report = evaluate_classifier(y_clf_test, y_dt_pred,
                                                 label_names=list(le.classes_))
    print(f"        Accuracy : {dt_metrics['accuracy']:.4f}")
    print(f"        F1-Score : {dt_metrics['f1']:.4f}")

    plot_feature_importance(dt.feature_importances_, feature_names)
    save_accuracy_report(dt_metrics, dt_report, list(le.classes_))
    save_model(dt, "decision_tree")

    # ── 3. KMeans Clustering ──────────────────────────────────────────────────
    print("\n[Train] Training KMeans Clustering (Soil Segmentation) …")
    from sklearn.preprocessing import StandardScaler as SS
    X_all_scaled = X_train  # already scaled from preprocessing

    # Use ALL scaled data for clustering (unsupervised)
    import numpy as _np
    X_all = _np.vstack([X_train, X_test])

    km  = build_kmeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED)
    km.fit(X_all)
    cluster_labels = km.labels_

    km_metrics = evaluate_clustering(X_all, cluster_labels)
    print(f"        Silhouette Score: {km_metrics['silhouette_score']:.4f}")

    pca     = build_pca(n_components=2)
    X_pca   = pca.fit_transform(X_all)
    plot_clusters(X_pca, cluster_labels, N_CLUSTERS)
    save_clustering_metrics(km_metrics, N_CLUSTERS)
    save_model(km,  "knn")       # stored under 'knn.pkl' per spec
    save_model(pca, "pca")

    # ── 4. Linear Regression ──────────────────────────────────────────────────
    print("\n[Train] Training Linear Regression (Yield Prediction) …")
    lr = build_linear_regression()
    lr.fit(X_train, y_reg_train)
    y_lr_pred = lr.predict(X_test)

    lr_metrics = evaluate_regression(y_reg_test, y_lr_pred)
    print(f"        RMSE : {lr_metrics['RMSE']:.4f}")
    print(f"        MAE  : {lr_metrics['MAE']:.4f}")
    print(f"        R²   : {lr_metrics['R2']:.4f}")

    plot_residuals(y_reg_test, y_lr_pred)
    save_regression_metrics(lr_metrics)
    save_model(lr, "linear_regression")

    # ── 5. Save auxiliary objects (scaler + label encoder) ────────────────────
    save_model(le,     "label_encoder")
    save_model(scaler, "scaler")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print("=" * 65)
    print(f"\n  Decision Tree Accuracy   : {dt_metrics['accuracy']:.4f}")
    print(f"  Clustering Silhouette    : {km_metrics['silhouette_score']:.4f}")
    print(f"  Regression R²            : {lr_metrics['R2']:.4f}")
    print(f"\n  Models saved → {MODELS_DIR}")
    print(f"  Results saved → {os.path.join(BASE_DIR, 'results')}\n")


if __name__ == "__main__":
    main()
