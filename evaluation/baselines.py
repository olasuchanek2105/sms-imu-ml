
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
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


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
    cv = GroupKFold(n_splits=n_splits)
    results = {}

    for name, (model, X_tr, _) in models.items():
        scores_f1 = cross_val_score(
            model,
            X_tr,
            y_train,
            cv=cv,
            groups=groups_train,
            scoring="f1_macro",
            n_jobs=-1
        )

        scores_acc = cross_val_score(
            model,
            X_tr,
            y_train,
            cv=cv,
            groups=groups_train,
            scoring="accuracy",
            n_jobs=-1
        )

        results[name] = {
            "f1_mean": scores_f1.mean(),
            "f1_std": scores_f1.std(),
            "acc_mean": scores_acc.mean(),
            "acc_std": scores_acc.std(),
        }

    return results


def evaluate_single_stage_on_test(
    models,
    y_train,
    y_test
):
    results = {}

    for name, (model, X_tr, X_te) in models.items():
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        results[name] = {
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

    return results
