"""
Streamlit App - Breast Cancer Classification Demo
---------------------------------------------------
Upload the test_data.csv (or any similarly-structured CSV with a 'target'
column), pick a trained model, and view its evaluation metrics, confusion
matrix, and classification report on your uploaded data.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": ("logistic_regression.pkl", True),
    "Decision Tree":       ("decision_tree.pkl", False),
    "kNN":                 ("knn.pkl", True),
    "Naive Bayes":         ("naive_bayes.pkl", False),
    "SVM":                 ("svm.pkl", True),
    "Random Forest":       ("random_forest.pkl", False),
}

st.set_page_config(page_title="ML Classifier Demo", layout="wide")


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    models = {}
    for name, (fname, _) in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return scaler, feature_names, models


scaler, feature_names, models = load_artifacts()

st.title("🔬 Breast Cancer Classification — Model Comparison Demo")
st.markdown(
    "This app lets you upload test data, pick a trained classification model, "
    "and inspect its performance. Target: **0 = malignant, 1 = benign**."
)

# ---------------------------------------------------------------------------
# 1. Dataset upload
# ---------------------------------------------------------------------------
st.header("1️⃣ Upload Test Data (CSV)")
uploaded_file = st.file_uploader(
    "Upload a CSV with the same 30 feature columns as the training data, plus a 'target' column.",
    type=["csv"],
)

if uploaded_file is None:
    st.info("👆 Upload `test_data.csv` from the repo to try the app, or use your own similarly-formatted CSV.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.write("Preview of uploaded data:", df.head())

if "target" not in df.columns:
    st.error("Uploaded CSV must contain a 'target' column with the true labels (0/1).")
    st.stop()

missing_cols = [c for c in feature_names if c not in df.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
    st.stop()

X_input = df[feature_names]
y_true = df["target"]

# ---------------------------------------------------------------------------
# 2. Model selection
# ---------------------------------------------------------------------------
st.header("2️⃣ Select a Model")
model_name = st.selectbox("Choose a classification model:", list(models.keys()))
model = models[model_name]
needs_scaling = MODEL_FILES[model_name][1]

X_eval = scaler.transform(X_input) if needs_scaling else X_input

y_pred = model.predict(X_eval)
y_proba = model.predict_proba(X_eval)[:, 1]

# ---------------------------------------------------------------------------
# 3. Evaluation metrics
# ---------------------------------------------------------------------------
st.header("3️⃣ Evaluation Metrics")

acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_proba)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{acc:.4f}")
col2.metric("AUC", f"{auc:.4f}")
col3.metric("Precision", f"{prec:.4f}")
col4.metric("Recall", f"{rec:.4f}")
col5.metric("F1 Score", f"{f1:.4f}")
col6.metric("MCC", f"{mcc:.4f}")

# ---------------------------------------------------------------------------
# 4. Confusion matrix + classification report
# ---------------------------------------------------------------------------
st.header("4️⃣ Confusion Matrix & Classification Report")

c1, c2 = st.columns(2)

with c1:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Malignant (0)", "Benign (1)"],
                yticklabels=["Malignant (0)", "Benign (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    st.pyplot(fig)

with c2:
    report = classification_report(y_true, y_pred, target_names=["Malignant", "Benign"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3))

# ---------------------------------------------------------------------------
# 5. All-models comparison (optional bonus view)
# ---------------------------------------------------------------------------
st.header("📊 Compare All Models on This Data")
if st.checkbox("Show comparison across all trained models"):
    rows = []
    for name, m in models.items():
        scale = MODEL_FILES[name][1]
        Xe = scaler.transform(X_input) if scale else X_input
        yp = m.predict(Xe)
        ypr = m.predict_proba(Xe)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_true, yp), 4),
            "AUC": round(roc_auc_score(y_true, ypr), 4),
            "Precision": round(precision_score(y_true, yp), 4),
            "Recall": round(recall_score(y_true, yp), 4),
            "F1": round(f1_score(y_true, yp), 4),
            "MCC": round(matthews_corrcoef(y_true, yp), 4),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"))

st.caption("Built for BITS Pilani WILP — ML Assignment 2")
