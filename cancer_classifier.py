"""
Breast Cancer Classification – model comparison with CV + tuning
Author: Tonmoy Sarker (ts42a)
Description:
    Loads the Breast Cancer Wisconsin dataset, compares multiple classifiers
    (GaussianNB, Logistic Regression, SVM) with cross-validated hyperparameter
    tuning, selects the best model, evaluates on a held-out test set,
    and visualizes the confusion matrix.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Hold-out split with stratification for class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y
    )

    # CV strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Candidate pipelines and parameter grids
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
            Pipeline([("scaler", StandardScaler()), ("clf", SVC())]),
            {
                "clf__kernel": ["rbf", "linear"],
                "clf__C": np.logspace(-2, 3, 6),
                "clf__gamma": ["scale", "auto"],
            },
        ),
    ]

    best_model = None
    best_name = None
    best_score = -1

    print("Cross-validated scores (mean ± std):")
    for name, pipe, grid in candidates:
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1)
        gs.fit(X_train, y_train)
        mean = gs.best_score_
        std = gs.cv_results_["std_test_score"][gs.best_index_]
        print(f"- {name:6s}: {mean:.4f} ± {std:.4f}  (best params: {gs.best_params_})")
        if mean > best_score:
            best_score = mean
            best_model = gs.best_estimator_
            best_name = name

    # Final train on the full training set with the selected model
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"\nSelected model: {best_name}")
    print(f"Test Accuracy: {acc*100:.2f}%")
    print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=data.target_names).plot(values_format="d")
    plt.title(f"{best_name.upper()} – Confusion Matrix (Acc: {acc*100:.2f}%)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
