"""
train_models.py
----------------
Trains 6 classification models on the Breast Cancer Wisconsin (Diagnostic) dataset,
evaluates them with Accuracy, AUC, Precision, Recall, F1, and MCC, saves the trained
models + scaler to the model/ folder, saves a held-out test split as test_data.csv,
and prints a comparison table you can paste into README.md.

Dataset: Breast Cancer Wisconsin (Diagnostic)
  - 569 instances, 30 numeric features, binary target (malignant=0 / benign=1)
  - Loaded from scikit-learn (no internet needed), originally from UCI ML Repository:
    https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

RANDOM_STATE = 42
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target  # 0 = malignant, 1 = benign
feature_names = list(X.columns)

print(f"Dataset shape: {X.shape[0]} instances, {X.shape[1]} features")
print(f"Class balance:\n{y.value_counts()}\n")

# ---------------------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Save the TEST split as test_data.csv (this is what you upload to the Streamlit app)
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print(f"Saved test_data.csv with {len(test_df)} rows -> used for Streamlit demo\n")

# ---------------------------------------------------------------------------
# 3. Scale features (needed for LR, KNN, SVM)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

# ---------------------------------------------------------------------------
# 4. Define models
#    (LR, KNN, SVM use scaled features; tree-based & NB use raw features)
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": (LogisticRegression(max_iter=5000, random_state=RANDOM_STATE), True),
    "Decision Tree":       (DecisionTreeClassifier(random_state=RANDOM_STATE), False),
    "kNN":                 (KNeighborsClassifier(n_neighbors=5), True),
    "Naive Bayes":         (GaussianNB(), False),
    "SVM":                 (SVC(probability=True, random_state=RANDOM_STATE), True),
    "Random Forest":       (RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE), False),
}

# ---------------------------------------------------------------------------
# 5. Train, evaluate, save
# ---------------------------------------------------------------------------
results = []
for name, (model, needs_scaling) in models.items():
    Xtr = X_train_scaled if needs_scaling else X_train
    Xte = X_test_scaled if needs_scaling else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)

    # Save model
    fname = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, os.path.join(MODEL_DIR, fname))
    print(f"Trained {name:22s} -> {metrics}")

# ---------------------------------------------------------------------------
# 6. Save comparison table
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(MODEL_DIR, "metrics_summary.csv"), index=False)

print("\n===== Comparison Table (paste into README.md) =====\n")
print(results_df.to_markdown(index=False))

with open(os.path.join(MODEL_DIR, "metrics_summary.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nAll models + scaler saved in the 'model/' folder.")
print("test_data.csv saved in project root.")
