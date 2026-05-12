
"""
feature_importance.py
=====================

Moduł odpowiedzialny za analizę ważności cech (feature importance)
dla modeli wykorzystanych w projekcie.

Celem modułu jest:
- identyfikacja cech najbardziej istotnych dla procesu klasyfikacji,
- analiza wpływu cech surowych oraz cech stakowanych (stage1_p_high),
- interpretacja działania modeli jednoetapowych i dwuetapowych,
  co stanowi istotny element części analitycznej pracy inżynierskiej.

Zaimplementowane metody obejmują:
- ważność cech opartą o drzewa decyzyjne (Random Forest, AdaBoost),
- analizę bezwzględnych wartości współczynników regresji logistycznej.

UWAGA:
Wartości ważności cech nie są porównywalne pomiędzy różnymi typami modeli
(np. Random Forest vs Logistic Regression), lecz służą do analizy względnej
istotności cech w obrębie jednego modelu.
"""

import numpy as np
import pandas as pd

from models.stage2_models import (
    build_adaboost,
    build_logistic_regression,
    build_random_forest
)


def _importance_frame(X_train, importances):
    """Build a sorted feature-importance table."""
    return pd.DataFrame({
        "feature": X_train.columns,
        "importance": importances
    }).sort_values(by="importance", ascending=False)


def _rf_importance(X_train, y_train, random_state=42):
    """Fit a Random Forest and return its feature importances."""
    model = build_random_forest(random_state=random_state)
    model.fit(X_train, y_train)

    return _importance_frame(X_train, model.feature_importances_)


def rf_single_stage_importance(X_train, y_train, random_state=42):
    """Return Random Forest importances for single-stage RAW features."""
    return _rf_importance(X_train, y_train, random_state)


def rf_stage2_importance(X_train, y_train, random_state=42):
    """Return Random Forest importances for stage-two features."""
    return _rf_importance(X_train, y_train, random_state)


def logreg_stage2_importance(X_train, y_train):
    """Return absolute logistic regression coefficients as importances."""
    model = build_logistic_regression()
    model.fit(X_train, y_train)

    coef = model.named_steps["logisticregression"].coef_[0]

    return _importance_frame(X_train, np.abs(coef))


def adaboost_stage2_importance(X_train, y_train, random_state=42):
    """Return AdaBoost feature importances for stage-two features."""
    model = build_adaboost(random_state=random_state, max_depth=1)
    model.fit(X_train, y_train)

    return _importance_frame(X_train, model.feature_importances_)
