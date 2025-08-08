# Cancer Cell Classification using Scikit‑learn

A simple machine‑learning project that classifies breast‑cancer tumours as **malignant** or **benign** using Scikit‑learn’s built‑in **Breast Cancer Wisconsin (Diagnostic)** dataset.

![Confusion Matrix](/images/confusion_matrix.png)

## 📋 Project Overview
This project demonstrates an end‑to‑end ML workflow:

1. **Data Exploration**  
2. **Training** a Naive Bayes classifier  
3. **Evaluation** via accuracy score and confusion matrix  

## 📂 Dataset
* **Source:** [`sklearn.datasets.load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)  
* **Samples:** 569  
* **Features:** 30 numerical features describing cell nuclei  
* **Labels:** `0` = Malignant, `1` = Benign

![Class Distribution](/images/class_distribution.png)

## 🛠 Tech Stack
| Purpose | Library |
|---------|---------|
| Core ML | `scikit-learn` |
| Data wrangling | `pandas` |
| Visualisation | `matplotlib` |
| Language | Python 3.10 |

## 🚀 Quick Start
```bash
git clone https://github.com/your-username/cancer-cell-classification-ml.git
cd cancer-cell-classification-ml
pip install -r requirements.txt
python cancer_classifier.py
```

## 📈 Sample Output
```
Model Accuracy: 94.15%
```

The confusion‑matrix figure (above) shows correct vs. incorrect predictions; the current model achieves **94.15%** accuracy.

## ✨ Future Work
* Try other classifiers (Random Forest, SVM, Logistic Regression)
* Hyper‑parameter tuning with GridSearchCV
* Feature‑scaling & PCA for dimensionality reduction
* Deploy as an interactive Streamlit app
