
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

st.set_page_config(page_title="Breast Cancer Classifier Demo", page_icon="🩺", layout="centered")

# --------------
# Utility: train best model with CV + tuning (cached)
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

st.title("🩺 Breast Cancer Classifier – Demo")
st.caption("Educational demo using the Breast Cancer Wisconsin (Diagnostic) dataset. **Not for medical use.**")

with st.spinner("Training models with cross-validation... (cached for future runs)"):
    state = train_best_pipeline()
pipe = state["pipeline"]
feature_names = list(state["feature_names"])
target_names = list(state["target_names"])

st.success(f"Loaded best model: **{state['name']}**  |  Test accuracy on hold-out: **{state['test_acc']*100:.2f}%**")

tab_img, tab_csv, tab_manual = st.tabs(["📷 Image upload", "📄 CSV features", "🧮 Manual features"])

# Image tab (info only)
with tab_img:
    st.subheader("Upload an image (info)")
    img_file = st.file_uploader("Upload a mammogram or microscopy image", type=["png", "jpg", "jpeg"])
    if img_file is not None:
        st.image(img_file, caption="Uploaded image (preview)", use_container_width=True)
    st.warning(
        "This demo's model is trained on **tabular numeric features** (30 measurements) — not raw images. "
        "To support images, we'd build a separate computer-vision model trained on medical images."
    )
    st.info(
        "If you want an actual image-based workflow, I can scaffold a separate project using transfer learning "
        "(e.g., ResNet/EfficientNet) with medical disclaimers and evaluation protocols."
    )

# CSV tab
with tab_csv:
    st.subheader("Predict from CSV of 30 features")
    st.write("Expected columns:", ", ".join(feature_names))

    example = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
    st.download_button("Download CSV template", example.to_csv(index=False).encode("utf-8"), "features_template.csv", "text/csv")

    file = st.file_uploader("Upload a CSV with feature columns", type=["csv"], key="csv_uploader")
    threshold = st.slider("Decision threshold for 'malignant' (class 0)", 0.0, 1.0, 0.5, 0.01, key="csv_thr")
    if file is not None:
        try:
            df = pd.read_csv(file)
            missing = [c for c in feature_names if c not in df.columns]
            extra = [c for c in df.columns if c not in feature_names]
            if missing:
                st.error(f"Missing columns: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
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
                if extra:
                    st.caption(f"Note: extra columns ignored: {extra[:5]}{' ...' if len(extra) > 5 else ''}")
        except Exception as e:
            st.exception(e)

# Manual features tab
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
                step = float(max(stds[i]*0.1, 1e-3))
                val = st.number_input(fname, value=default, step=step, format="%.5f")
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

# New: Metrics & Thresholds expander (ROC/PR + downloadable report)
with st.expander("📉 Metrics & Thresholds"):
    st.write("Adjust the decision threshold for the positive class (**malignant = 0**) below when scoring *your inputs*.")
    thr_demo = st.slider("Global threshold for 'malignant' decisions", 0.0, 1.0, 0.5, 0.01, key="global_thr_demo")
    st.caption("Note: This threshold affects CSV/Manual predictions.")

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report, confusion_matrix

    X_train, X_test, y_train, y_test = state["train_test_split"]
    probs = state["pipeline"].predict_proba(X_test)[:, list(state["target_names"]).index("malignant")]
    preds = (probs >= thr_demo).astype(int)

    fpr, tpr, _ = roc_curve(y_test, probs, pos_label=list(state["target_names"]).index("malignant"))
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_test, probs, pos_label=list(state["target_names"]).index("malignant"))
    ap = average_precision_score(y_test, probs, pos_label=list(state["target_names"]).index("malignant"))

    fig1 = plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC – Malignant")
    plt.legend(loc="lower right")
    st.pyplot(fig1)

    fig2 = plt.figure()
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR – Malignant")
    plt.legend(loc="lower left")
    st.pyplot(fig2)

    # Confusion matrix at the chosen threshold
    cm = confusion_matrix(y_test, preds)
    st.write("Confusion Matrix at current threshold:")
    st.write(pd.DataFrame(cm, index=state["target_names"], columns=state["target_names"]))

    # Downloadable test-set report
    rep = classification_report(y_test, preds, target_names=state["target_names"], output_dict=True)
    rep_df = pd.DataFrame(rep).transpose().round(4)
    st.dataframe(rep_df)
    st.download_button("Download test-set report (CSV)", rep_df.to_csv().encode("utf-8"), file_name="test_classification_report.csv", mime="text/csv")

st.divider()
st.caption("⚠️ **Disclaimer:** This app is for educational purposes only and is **not** a medical device. Do not use its output for diagnosis or treatment decisions.")
