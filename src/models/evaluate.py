"""Evaluation helpers."""

from sklearn.metrics import accuracy_score, confusion_matrix


def basic_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
