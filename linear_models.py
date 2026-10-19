import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# loss f
def BCE(y_true, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def MSE(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


# y = X@w + b
# loss = 1/2 * MSE = mean((pred-y)^2) / 2
# err = pred - y
# w -= lr * (1/n) * X^T @ err
# b -= lr * (1/n) * sum(err)
def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    x_train = np.asarray(data.train_features, float)
    y_train = np.asarray(data.train_target, float).ravel()
    x_test = np.asarray(data.test_features, float)

    n_samples, n_features = x_train.shape
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history: list[float] = []

    for _ in range(data.epochs):
        predictions = x_train @ weights + bias
        err = predictions - y_train
        loss_history.append(0.5 * MSE(y_train, predictions))
        grad_w = x_train.T @ err / n_samples
        grad_b = float(np.mean(err))
        weights -= data.learning_rate * grad_w
        bias -= data.learning_rate * grad_b

    predictions = x_test @ weights + bias

    return RegressionResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        predictions=predictions,
    )


# p = sigmoid(X@w + b)
# loss = bce = -mean(y*log(p) + (1-y)*log(1-p))
# err = p - y
# w -= lr * (1/n) * X^T @ err
# b -= lr * (1/n) * sum(err)
def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    x_train = np.asarray(data.train_features, float)
    y_train = np.asarray(data.train_target, float).ravel()
    x_test = np.asarray(data.test_features, float)

    n_samples, n_features = x_train.shape
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history: list[float] = []

    for _ in range(data.epochs):
        probs = sigmoid(x_train @ weights + bias)
        err = probs - y_train
        loss_history.append(BCE(y_train, probs))
        grad_w = x_train.T @ err / n_samples
        grad_b = float(np.mean(err))
        weights -= data.learning_rate * grad_w
        bias -= data.learning_rate * grad_b

    probs = sigmoid(x_test @ weights + bias)
    predictions = (probs >= 0.5).astype(int)

    return ClassificationResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        probabilities=probs,
        predictions=predictions,
    )


def metrics(y_true, y_pred, proba) -> ClassificationMetrics:
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, proba)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred)),
        recall=float(recall_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните вашу логистическую регрессию, Decision Tree и KNN на одном split.

    Формат experiment_results.json (его записывает run_experiments.py):
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    x_train = np.asarray(data.train_features, float)
    y_train = np.asarray(data.train_target, float).ravel()
    x_test = np.asarray(data.test_features, float)
    y_test = np.asarray(data.test_target, float).ravel()

    own = train_logistic_classifier(data)
    own_metrics = metrics(y_test, own.predictions, own.probabilities)

    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(x_train, y_train)
    tree_proba = tree.predict_proba(x_test)[:, 1]
    tree_metrics = metrics(y_test, (tree_proba >= 0.5).astype(int), tree_proba)

    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    knn_proba = knn.predict_proba(x_test)[:, 1]
    knn_metrics = metrics(y_test, (knn_proba >= 0.5).astype(int), knn_proba)

    return BaselineComparisonResult(
        logistic_regression=own_metrics,
        decision_tree=tree_metrics,
        knn=knn_metrics,
    )
