# 🩺 Breast Cancer Classification – Model Comparison with CV + Tuning

This repository classifies breast-cancer tumours as **malignant** or **benign** using Scikit‑learn’s built‑in **Breast Cancer Wisconsin (Diagnostic)** dataset — now upgraded to a **real‑world style workflow** with **cross‑validated hyperparameter tuning** and **model comparison**.

![Confusion Matrix](/images/confusion_matrix.png)

---

## 📌 What’s Inside
- Clean **end‑to‑end pipeline**: load → split → CV+tune → evaluate → visualize
- **Three models** out of the box: Gaussian Naive Bayes, Logistic Regression, SVM
- **StratifiedKFold CV** for robust estimates
- Clear **confusion matrix** and **classification report**

> Typical test accuracy on the default split is ~97–99% with tuned Logistic Regression or SVM (your exact number may vary with split).

---

## 📂 Dataset
- **Source:** [`sklearn.datasets.load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)  
- **Samples:** 569  
- **Features:** 30 numerical features describing cell nuclei  
- **Labels:** `0` = Malignant, `1` = Benign

![Class Distribution](/images/class_distribution.png)

---

## 🛠 Tech Stack
| Purpose            | Library         |
|--------------------|-----------------|
| Core ML            | scikit-learn    |
| Data handling      | pandas          |
| Visualisation      | matplotlib      |
| Utils              | numpy           |
| Language           | Python 3.10+    |

---

## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/your-username/cancer-cell-classification-ml.git
cd cancer-cell-classification-ml

# Install dependencies
pip install -r requirements.txt

# Run (compares GNB vs Logistic Regression vs SVM and picks best)
python cancer_classifier.py
```

**Example console output**
```
Cross-validated scores (mean ± std):
- gnb   : 0.94xx ± 0.0x   (best params: {...})
- logreg: 0.98xx ± 0.0x   (best params: {...})
- svm   : 0.98xx ± 0.0x   (best params: {...})

Selected model: svm
Test Accuracy: 98.25%
```

---

## 🧪 Reproducibility Tips
- We fix `random_state=42` in both the train/test split and CV shuffling.
- Results can vary slightly by scikit‑learn version and CPU/BLAS.

---

## ✨ Next Steps
- Add more models (Random Forest, Gradient Boosting, XGBoost/LightGBM)
- Use **Pipelines** with preprocessing (PCA, feature selection)
- Try **ROC‑AUC**, **precision/recall**, **PR curves** for imbalanced scenarios
- Package as a **Streamlit** demo app

---
