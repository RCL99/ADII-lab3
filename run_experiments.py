"""Прогон всех трёх заданий лабораторной №3. Запуск: venv/bin/python run_experiments.py"""

import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import compare_classical_models, metrics, train_linear_regression

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path, target, sep=",", yes_no=None, stratify=False):
    """читает csv, масштабирует признаки по train"""
    df = pd.read_csv(path, sep=sep)
    if yes_no:
        df[yes_no] = (df[yes_no] == "Yes").astype(int)

    x = df.drop(columns=[target])
    y = df[target]
    if stratify:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x.columns)
    return x_train, y_train, x_test, y_test


def as_dict(m):
    """метрики в обычный словарь для json"""
    return {
        "roc_auc": m.roc_auc,
        "accuracy": m.accuracy,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
    }


def main():
    # задание 1 линейная регрессия
    x_train, y_train, x_test, y_test = load_data(
        "Student_Performance.csv",
        target="Performance Index",
        yes_no="Extracurricular Activities",
    )
    regression = train_linear_regression(RegressionInput(x_train, y_train, x_test, y_test))
    test_mse = float(mean_squared_error(y_test, regression.predictions))
    print(f"regression test MSE: {test_mse:.4f} (history: {len(regression.loss_history)} эпох)")

    # задания 2 и 3 четыре классификатора на одном разбиении
    x_train, y_train, x_test, y_test = load_data(
        "german.csv", target="Creditability", sep=";", stratify=True
    )
    comparison = compare_classical_models(ClassificationInput(x_train, y_train, x_test, y_test))

    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(x_train, y_train)
    logreg_proba = logreg.predict_proba(x_test)[:, 1]
    logreg_metrics = metrics(y_test, (logreg_proba >= 0.5).astype(int), logreg_proba)

    for name, m in [
        ("своя логистика", comparison.logistic_regression),
        ("готовая logreg", logreg_metrics),
        ("decision_tree", comparison.decision_tree),
        ("knn", comparison.knn),
    ]:
        print(
            f"{name:16} auc={m.roc_auc:.4f} acc={m.accuracy:.4f} "
            f"prec={m.precision:.4f} rec={m.recall:.4f} f1={m.f1:.4f}"
        )

    # experiment_results.json целиком
    results = {
        "linear_regression": {"test_mse": test_mse},
        "classification": {
            "own_logistic": as_dict(comparison.logistic_regression),
            "logistic_regression": as_dict(logreg_metrics),
            "decision_tree": as_dict(comparison.decision_tree),
            "knn": as_dict(comparison.knn),
        },
    }
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
