
"""
test_evaluation.py
==================

Moduł odpowiedzialny za końcową ocenę modeli klasyfikacyjnych
na niezależnym zbiorze testowym.

Celem modułu jest:
- przeprowadzenie uczciwej ewaluacji końcowej (final test),
- porównanie skuteczności różnych modeli przy identycznym
  podziale danych,
- wygenerowanie metryk wykorzystywanych w rozdziale
  „Wyniki” pracy inżynierskiej.

Moduł jest używany po zakończeniu:
- trenowania modeli,
- walidacji krzyżowej,
- strojenia hiperparametrów.
"""

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

def evaluate_on_test(
    models,
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Trenuje modele na pełnym zbiorze treningowym
    i ocenia je na zbiorze testowym.
    """
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

    return results
