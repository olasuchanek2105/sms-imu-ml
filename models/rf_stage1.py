

"""
rf_stage1.py
============

Moduł odpowiedzialny za implementację pierwszego etapu
architektury dwuetapowej (stacking) wykorzystywanej w projekcie.

Etap 1 polega na trenowaniu klasyfikatora Random Forest
na cechach wyekstrahowanych z sygnału NIEPRZEFILTROWANEGO (RAW)
oraz generowaniu predykcji probabilistycznych wykorzystywanych
następnie jako dodatkowe cechy wejściowe w etapie drugim.

Funkcjonalność modułu obejmuje:
- definicję i konfigurację klasyfikatora Random Forest (Stage 1),
- obliczanie predykcji typu out-of-fold (OOF) z wykorzystaniem
  walidacji krzyżowej GroupKFold (podział po subjectach),
- trenowanie modelu etapu 1 na pełnym zbiorze treningowym
  oraz generowanie predykcji dla zbioru testowego,
- zabezpieczenie przed przeciekiem danych pomiędzy zbiorem
  treningowym i testowym.

Predykcje OOF zapewniają, że dla każdej próbki treningowej
prawdopodobieństwo klasy jest wyznaczane przez model,
który nie był uczony na tej próbce, co jest kluczowe
dla poprawnej implementacji algorytmu stacking.

Moduł jest wykorzystywany bezpośrednio w głównym skrypcie
trenującym modele oraz w dalszych etapach ewaluacji.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.base import clone

def build_rf_stage1(random_state=42):
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=14,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight={0: 1.0, 1: 1.0},
        random_state=random_state,
        n_jobs=-1
    )


def compute_oof_stage1(
    model,
    X_train,
    y_train,
    groups,
    n_splits=5
):
    """
    Oblicza predykcje OOF dla etapu 1 (RF na danych RAW).
    """
    X_ = X_train.reset_index(drop=True)
    y_ = y_train.reset_index(drop=True)
    groups_ = groups.reset_index(drop=True)

    oof = np.zeros(len(X_))
    gkf = GroupKFold(n_splits=n_splits)

    for fold, (tr, val) in enumerate(gkf.split(X_, y_, groups_)):
        print(f"Stage 1 OOF fold {fold + 1}")
        m = clone(model)
        m.fit(X_.iloc[tr], y_.iloc[tr])
        oof[val] = m.predict_proba(X_.iloc[val])[:, 1]

    return oof


def fit_stage1_full_and_predict(
    model,
    X_train,
    y_train,
    X_test
):
    """
    Trenuje model etapu 1 na pełnym zbiorze treningowym
    i generuje predykcje probabilistyczne dla zbioru testowego.
    """
    m = clone(model)
    m.fit(X_train, y_train)

    p_test = m.predict_proba(X_test)[:, 1]
    p_train_full = m.predict_proba(X_train)[:, 1]

    return m, p_train_full, p_test
