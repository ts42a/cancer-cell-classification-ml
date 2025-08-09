
import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import classification_report

# --------------
# Utility: train best model with CV + tuning (cached so it runs once)
# --------------
@st.cache_resource(show_spinner=True)
def train_best_pipeline(random_state: int = 42):
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=random_state, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    candidates = [
        (
            "gnb",
            Pipeline([("gnb", GaussianNB())]),
            {"gnb__var_smoothing": np.logspace(-12, -6, 7)},
        ),
        (
            "logreg",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000)),
                ]
            ),
            {
                "clf__C": np.logspace(-3, 3, 7),
                "clf__penalty": ["l2"],
                "clf__solver": ["lbfgs"],
            },
        ),
        (
            "svm",
            Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True))]),
            {
                "clf__kernel": ["rbf", "linear"],
                "clf__C": np.logspace(-2, 3, 6),
                "clf__gamma": ["scale", "auto"],
            },
        ),
    ]

    best_model = None
    best_name = None
    best_score = -1.0
    best_params = None

    for name, pipe, grid in candidates:
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1)
        gs.fit(X_train, y_train)
        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_model = gs.best_estimator_
            best_name = name
            best_params = gs.best_params_

    # final fit
    best_model.fit(X_train, y_train)
    test_acc = best_model.score(X_test, y_test)

    return {
        "pipeline": best_model,
        "name": best_name,
        "params": best_params,
        "test_acc": test_acc,
        "feature_names": feature_names,
        "target_names": data.target_names,
        "train_test_split": (X_train, X_test, y_train, y_test),
        "desc": "Tuned via 5-fold Stratified CV; best of GNB/LogReg/SVM."
    }

st.set_page_config(page_title="Breast Cancer Classifier Demo", page_icon="🩺", layout="centered")

st.title("🩺 Breast Cancer Classifier – Demo")
st.caption("Educational demo using the Breast Cancer Wisconsin (Diagnostic) dataset. **Not for medical use.**")

# Train or load the model
with st.spinner("Training models with cross-validation... (cached for future runs)"):
    state = train_best_pipeline()
pipe = state["pipeline"]
feature_names = list(state["feature_names"])
target_names = list(state["target_names"])

st.success(f"Loaded best model: **{state['name']}**  |  Test accuracy on hold-out: **{state['test_acc']*100:.2f}%**")

tab_img, tab_csv, tab_manual = st.tabs(["📷 Image upload", "📄 CSV features", "🧮 Manual features"])

# -------------
# Image tab (explanation)
# -------------
with tab_img:
    st.subheader("Upload an image (info)")
    img_file = st.file_uploader("Upload a mammogram or microscopy image", type=["png", "jpg", "jpeg"])
    if img_file is not None:
        st.image(img_file, caption="Uploaded image (preview)", use_container_width=True)
    st.warning(
        "This repository's model is trained on **tabular numeric features** (30 measurements) — "
        "not raw images. To make image uploads work properly, we'd need a separate computer vision "
        "model trained on medical images, which is out of scope here. "
        "Please switch to the **CSV features** or **Manual features** tab to get a prediction from this model."
    )
    st.info(
        "If you want an actual image-based workflow, I can scaffold a separate project using transfer learning "
        "(e.g., ResNet/EfficientNet) with medical disclaimers and evaluation protocols."
    )

# -------------
# CSV tab
# -------------
with tab_csv:
    st.subheader("Predict from CSV of 30 features")
    st.write("Expected columns:", ", ".join(feature_names))

    example = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
    st.download_button("Download CSV template", example.to_csv(index=False).encode("utf-8"), "features_template.csv", "text/csv")

    file = st.file_uploader("Upload a CSV with feature columns", type=["csv"], key="csv_uploader")
    threshold = st.slider("Decision threshold for 'malignant' (class 0) using predicted probability", 0.0, 1.0, 0.5, 0.01)
    if file is not None:
        try:
            df = pd.read_csv(file)
            missing = [c for c in feature_names if c not in df.columns]
            extra = [c for c in df.columns if c not in feature_names]
            if missing:
                st.error(f"Missing columns: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
            elif extra:
                st.warning(f"Extra columns will be ignored: {extra[:5]}{' ...' if len(extra) > 5 else ''}")
                X = df[feature_names].values
                probs = pipe.predict_proba(X)
                # malignant is index of target_names where name == 'malignant'
                malignant_index = target_names.index("malignant")
                malignant_prob = probs[:, malignant_index]
                preds = (malignant_prob >= threshold).astype(int)  # 1 if malignant per threshold
                # Map to labels: show both benign/malignant
                pred_labels = np.where(preds == 1, "malignant", "benign")
                out = df.copy()
                out["prob_malignant"] = malignant_prob
                out["prediction"] = pred_labels
                st.write("Predictions:")
                st.dataframe(out.head(50))
                st.download_button("Download predictions as CSV", out.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
            else:
                X = df[feature_names].values
                probs = pipe.predict_proba(X)
                malignant_index = target_names.index("malignant")
                malignant_prob = probs[:, malignant_index]
                preds = (malignant_prob >= threshold).astype(int)
                pred_labels = np.where(preds == 1, "malignant", "benign")
                out = df.copy()
                out["prob_malignant"] = malignant_prob
                out["prediction"] = pred_labels
                st.write("Predictions:")
                st.dataframe(out.head(50))
                st.download_button("Download predictions as CSV", out.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
        except Exception as e:
            st.exception(e)

# -------------
# Manual features tab
# -------------
with tab_manual:
    st.subheader("Enter features manually")
    data = load_breast_cancer()
    X = data.data
    means = X.mean(axis=0)
    stds = X.std(axis=0)

    with st.form("manual_form"):
        cols = st.columns(3)
        values = []
        for i, fname in enumerate(feature_names):
            with cols[i % 3]:
                default = float(means[i])
                val = st.number_input(fname, value=default, step=float(max(stds[i]*0.1, 1e-3)))
                values.append(val)
        threshold2 = st.slider("Decision threshold for 'malignant' (class 0)", 0.0, 1.0, 0.5, 0.01, key="thr2")
        submitted = st.form_submit_button("Predict")
    if submitted:
        X_one = np.array(values, dtype=float).reshape(1, -1)
        probs = pipe.predict_proba(X_one)[0]
        malignant_index = target_names.index("malignant")
        prob_malignant = float(probs[malignant_index])
        pred_label = "malignant" if prob_malignant >= threshold2 else "benign"
        st.metric("Prediction", pred_label.upper())
        st.write({ "prob_malignant": prob_malignant, "prob_benign": float(probs[1 - malignant_index]) })

st.divider()
st.caption("⚠️ **Disclaimer:** This app is for educational purposes only and is **not** a medical device. "
           "Do not use its output for diagnosis or treatment decisions.")
