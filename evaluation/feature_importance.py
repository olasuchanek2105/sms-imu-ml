
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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def rf_single_stage_importance(X_train, y_train, random_state=42):
    model = RandomForestClassifier(
        n_estimators=600,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)


def rf_stage2_importance(X_train, y_train, random_state=42):
    model = RandomForestClassifier(
        n_estimators=600,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)


def logreg_stage2_importance(X_train, y_train):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs"
        )
    )
    model.fit(X_train, y_train)

    coef = model.named_steps["logisticregression"].coef_[0]

    return pd.DataFrame({
        "feature": X_train.columns,
        "importance": np.abs(coef)
    }).sort_values(by="importance", ascending=False)


def adaboost_stage2_importance(X_train, y_train, random_state=42):
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=200,
        learning_rate=0.7,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    return pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)
