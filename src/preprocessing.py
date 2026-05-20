"""
preprocessing.py
----------------
Data loading, cleaning, feature engineering, and train/test splitting
for the Smart Agriculture Decision Support System.

Dataset columns:
  Nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, label
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "dataset.csv")

# Features used by all models
FEATURE_COLS = ["Nitrogen", "phosphorus", "potassium",
                "temperature", "humidity", "ph", "rainfall"]
TARGET_COL   = "label"

# Synthetic yield column (kg/ha) – generated if not present
YIELD_COL    = "yield_kg_ha"


# ── Main loader ───────────────────────────────────────────────────────────────
def load_and_preprocess(data_path: str = DATA_PATH, random_state: int = 42):
    """
    Load, clean, and return pre-processed data splits.

    Returns
    -------
    X_train, X_test   : np.ndarray  – scaled feature matrices
    y_clf_train, ...  : np.ndarray  – classification labels (encoded ints)
    y_reg_train, ...  : np.ndarray  – regression targets (yield)
    le                : LabelEncoder
    scaler            : StandardScaler
    feature_names     : list[str]
    df_raw            : pd.DataFrame  – cleaned, unscaled frame (for clustering)
    """
    # 1. Read CSV
    df = pd.read_csv(data_path)

    # 2. Drop unnamed / fully-null columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # 3. Drop rows with missing values in required columns
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    # 4. Ensure numeric features
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS)

    # 5. Synthesize realistic crop-yield column (kg/ha) if absent
    #    Using a regression-like formula to create believable variance
    if YIELD_COL not in df.columns:
        np.random.seed(random_state)
        noise       = np.random.normal(0, 200, len(df))
        df[YIELD_COL] = (
            0.8  * df["Nitrogen"]
            + 0.6  * df["phosphorus"]
            + 0.5  * df["potassium"]
            + 12.0 * df["rainfall"]
            + 40.0 * df["humidity"]
            - 80.0 * np.abs(df["ph"] - 6.5)  # penalise pH deviation
            + 500
            + noise
        ).clip(lower=200)                      # min 200 kg/ha

    # 6. Encode classification target
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df[TARGET_COL])

    # 7. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS].values)

    y_clf = df["label_enc"].values
    y_reg = df[YIELD_COL].values

    # 8. Train / test split (80 / 20, stratified on crop label)
    (X_train, X_test,
     y_clf_train, y_clf_test,
     y_reg_train, y_reg_test) = train_test_split(
        X_scaled, y_clf, y_reg,
        test_size=0.2,
        random_state=random_state,
        stratify=y_clf
    )

    print(f"[Preprocessing] Loaded {len(df)} samples | "
          f"Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"[Preprocessing] Classes: {list(le.classes_)}")

    return (X_train, X_test,
            y_clf_train, y_clf_test,
            y_reg_train, y_reg_test,
            le, scaler, FEATURE_COLS, df)


if __name__ == "__main__":
    load_and_preprocess()
