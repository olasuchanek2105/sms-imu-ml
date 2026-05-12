
"""
baselines.py
============

Moduł zawierający implementację modeli jednoetapowych (baseline),
wykorzystywanych do porównania z zaproponowanym modelem dwuetapowym (stacking).

Modele bazowe:
- Random Forest uczony wyłącznie na cechach z sygnału surowego (RAW),
- Random Forest uczony wyłącznie na cechach z sygnału przefiltrowanego (FILT).

Celem modułu jest:
- zapewnienie punktu odniesienia (baseline) dla oceny jakości modelu dwuetapowego,
- uczciwe porównanie z zachowaniem izolacji danych testowych,
- wykorzystanie walidacji krzyżowej z grupowaniem po badanych (GroupKFold),
  co zapobiega przeciekowi informacji pomiędzy zbiorami.

Moduł NIE wykorzystuje mechanizmu stacking — każdy model działa
w trybie klasycznej, jednoetapowej klasyfikacji.
"""

from sklearn.ensemble import RandomForestClassifier

from evaluation.cross_validation import run_model_group_cv
from evaluation.metrics import evaluate_model


def get_single_stage_models(
    X_raw_train,
    X_raw_test,
    X_filt_train,
    X_filt_test,
    random_state=42
):
    return {
        "RF_single_RAW": (
            RandomForestClassifier(
                n_estimators=600,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            ),
            X_raw_train,
            X_raw_test
        ),
        "RF_single_FILT": (
            RandomForestClassifier(
                n_estimators=600,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            ),
            X_filt_train,
            X_filt_test
        ),
    }


def run_single_stage_cv(
    models,
    y_train,
    groups_train,
    n_splits=5
):
    results = {}

    for name, (model, X_tr, _) in models.items():
        results[name] = run_model_group_cv(
            model,
            X_tr,
            y_train,
            groups_train,
            n_splits
        )

    return results


def evaluate_single_stage_on_test(
    models,
    y_train,
    y_test
):
    results = {}

    for name, (model, X_tr, X_te) in models.items():
        model.fit(X_tr, y_train)
        results[name] = evaluate_model(model, X_te, y_test)

    return results
