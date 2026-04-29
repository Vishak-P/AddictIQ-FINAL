"""
train_model.py
==============
Social Media & Mobile Addiction Prediction - Model Training Script
Author  : ML/DevOps Engineer
Purpose : Train, evaluate, and persist the best classification model.
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1.  CONFIGURATION
# ──────────────────────────────────────────────
DATASET_PATH  = "dataset.csv"
MODEL_PATH    = "model.pkl"
METRICS_PATH  = "metrics.json"
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
FEATURES      = ["Age", "Daily_Usage_Hours", "Social_Media_Apps", "Screen_Time", "Sleep_Hours"]
TARGET        = "Addicted"


# ──────────────────────────────────────────────
# 2.  DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────
def load_and_preprocess(path: str):
    """Load CSV, handle missing values, encode target label."""
    print(f"[INFO] Loading dataset from '{path}' ...")
    df = pd.read_csv(path)

    print(f"[INFO] Raw shape: {df.shape}")
    print(df.head())

    # --- Handle missing values ---
    for col in FEATURES:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"[WARN] Filled missing values in '{col}' with median={median_val}")

    # Target is already 0/1 integer
    print(f"[INFO] Class distribution:\n{df[TARGET].value_counts()}\n")

    X = df[FEATURES].values
    y = df[TARGET].values
    return X, y, None


# ──────────────────────────────────────────────
# 3.  MODEL DEFINITIONS
# ──────────────────────────────────────────────
def build_models():
    """Return a dict of named sklearn Pipeline objects."""
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=500, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=3,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=4,
                random_state=RANDOM_STATE,
            )),
        ]),
    }
    return models


# ──────────────────────────────────────────────
# 4.  TRAINING & EVALUATION
# ──────────────────────────────────────────────
def train_and_evaluate(X, y):
    """Train all models, collect metrics, return best model + full report."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models   = build_models()
    results  = {}
    best_name, best_model, best_acc = None, None, -1
    print("=" * 60)
    print("  MODEL TRAINING & EVALUATION")
    print("=" * 60)
    for name, pipeline in models.items():
        print(f"\n[MODEL] {name}")
        # --- Train ---
        pipeline.fit(X_train, y_train)
        # --- Predict ---
        y_pred = pipeline.predict(X_test)
        # --- Metrics ---
        acc    = accuracy_score(y_test, y_pred)
        cm     = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True)
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

        print(f"  Accuracy        : {acc:.4f}")
        print(f"  CV Mean±Std     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Confusion Matrix:\n  {np.array(cm)}")
        print(f"  Classification Report:\n"
              f"{classification_report(y_test, y_pred, target_names=['Not Addicted','Addicted'])}")

        results[name] = {
            "accuracy":            round(acc, 4),
            "cv_mean":             round(float(cv_scores.mean()), 4),
            "cv_std":              round(float(cv_scores.std()), 4),
            "confusion_matrix":    cm,
            "classification_report": report,
        }

        if acc > best_acc:
            best_acc   = acc
            best_name  = name
            best_model = pipeline

    return best_name, best_model, results, best_acc


# ──────────────────────────────────────────────
# 5.  PERSIST ARTEFACTS
# ──────────────────────────────────────────────
def save_model(model, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n[SAVED] Model -> {path}")


def save_metrics(results: dict, best_name: str, path: str):
    output = {
        "best_model":    best_name,
        "model_results": results,
        "features":      FEATURES,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[SAVED] Metrics -> {path}")


# ──────────────────────────────────────────────
# 6.  MAIN
# ──────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  SOCIAL MEDIA ADDICTION PREDICTOR — TRAINING")
    print("=" * 60 + "\n")

    X, y, _ = load_and_preprocess(DATASET_PATH)
    best_name, best_model, results, best_acc = train_and_evaluate(X, y)

    print("\n" + "=" * 60)
    print(f"  BEST MODEL : {best_name}  (Accuracy: {best_acc:.4f})")
    print("=" * 60)

    save_model(best_model, MODEL_PATH)
    save_metrics(results, best_name, METRICS_PATH)

    print("\n[DONE] Training complete. Artefacts saved.\n")


if __name__ == "__main__":
    main()


# ──────────────────────────────────────────────
# BEST PRACTICES & IMPROVEMENT IDEAS
# ──────────────────────────────────────────────
"""
HOW TO IMPROVE ACCURACY
────────────────────────
1. Feature Engineering
   - usage_to_sleep_ratio   = daily_usage_time / (sleep_hours + 1)
   - app_intensity          = number_of_apps_used / screen_time
   - sleep_deficit          = max(0, 8 - sleep_hours)
   - weekend_vs_weekday flag if timestamps are available

2. Hyperparameter Tuning
   - Use GridSearchCV or RandomizedSearchCV on n_estimators, max_depth, etc.
   - Bayesian optimisation (Optuna) for faster convergence.

3. Imbalanced Data
   - Apply SMOTE (imbalanced-learn) if class ratio > 3:1.
   - Use class_weight='balanced' in models.

4. More Models
   - XGBoost, LightGBM, CatBoost for stronger tree ensembles.
   - Stacking / Voting ensembles for marginal gains.

5. Evaluation Tips
   - Always use stratified k-fold CV.
   - Report ROC-AUC, PR-AUC in addition to accuracy.
   - Watch for data leakage: scale INSIDE the pipeline.

6. Deployment Best Practices
   - Version your model artefacts (include training date, accuracy).
   - Monitor prediction distribution in production (data drift).
   - Retrain on schedule or when drift is detected.
"""
