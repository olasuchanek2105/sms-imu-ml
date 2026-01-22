

"""
metrics.py
==========

Moduł pomocniczy zawierający funkcje do ewaluacji wytrenowanych modeli
klasyfikacyjnych na zbiorze testowym.

Celem modułu jest:
- ujednolicenie sposobu obliczania metryk jakości,
- zapewnienie czytelnego i powtarzalnego interfejsu ewaluacji modeli,
- uproszczenie kodu głównego (train.py, test_evaluation.py).

Ewaluacja oparta jest o klasyczne metryki stosowane w problemach
klasyfikacji z niezrównoważonymi klasami:
- macro-F1,
- accuracy,
- macierz pomyłek.
"""

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
