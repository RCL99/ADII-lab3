from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

import numpy as np

# y = X*w + b
# w = sum((x-mean(x)*(y-mean(y))/(sum(x-x**2))
# b = mean(y) - w * mean(x)
def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    x_train = np.asarray(data.train_features, float)
    y_train = np.asarray(data.train_target, float)
    x_test  = np.asarray(data.test_features, float)

    x_mean = np.mean(x_train)
    y_mean = np.mean(y_train)

    w = np.sum((x_train - x_mean) * (y_train - y_mean)) / np.sum((x_train - x_mean) ** 2)
    b = y_mean - w * x_mean

    # loss
    y_train_pred = x_train * w + b
    train_mse = np.mean((y_train - y_train_pred) ** 2)

    # test predictions
    predictions = x_test * w + b

    return RegressionResult(
        weights=w,
        bias=b,
        loss_history=[float(train_mse)],
        predictions=predictions,
    )

def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    raise NotImplementedError


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    raise NotImplementedError
