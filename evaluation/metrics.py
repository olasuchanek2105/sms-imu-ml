import numpy as np

from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def evaluate_on_test(y_true, y_pred) -> dict:
    """
    Compute test-set evaluation metrics.
    """
    return {
        "report": classification_report(y_true, y_pred, digits=3),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def cross_validate_macro_f1(
    model,
    X,
    y,
    groups,
    n_splits: int = 5,
) -> float:
    """
    Perform GroupKFold cross-validation using macro-F1 score.
    """
    cv = GroupKFold(n_splits=n_splits)

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        groups=groups,
        scoring="f1_macro",
        n_jobs=-1,
    )

    return float(np.mean(scores))

