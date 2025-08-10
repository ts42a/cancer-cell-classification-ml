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

## 📊 Data Balance (Corrected)
The dataset is **imbalanced** toward **benign** cases.

![Class Distribution](images/class_distribution_corrected.png)

- Malignant (0): **212** samples (~37.26%)
- Benign (1): **357** samples (~62.74%)

This matters because metrics like **accuracy** can look high even if the model is biased. We therefore also inspect precision/recall per class.

## 🧪 Current Best Model (Auto-selected, quick CV)
3-fold Stratified CV scores (mean ± std) on the training split:
- gnb: 0.9370 ± 0.0223
- logreg: 0.9685 ± 0.0064
- svm: 0.9738 ± 0.0134

Selected pipeline: **SVM**  
**Held-out test accuracy:** **98.94%**

![Confusion Matrix](images/confusion_matrix_tuned.png)

### Classification report (test set)
| class | precision | recall | f1-score | support |
|------:|----------:|-------:|---------:|--------:|
| malignant | 0.9857 | 0.9857 | 0.9857 | 70 |
| benign    | 0.9915    | 0.9915    | 0.9915    | 118 |
| **accuracy** |  |  | **0.9894** | **188** |
| macro avg | 0.9886 | 0.9886 | 0.9886 | 188 |
| weighted avg | 0.9894 | 0.9894 | 0.9894 | 188 |

> ⚠️ **Medical disclaimer:** This project is for **education only** and not a medical device.
