import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_stacking_model(
    X_raw,
    X_filt,
    y,
    groups,
    train_idx,
    test_idx,
):
    # ===== STAGE 1 =====
    base_model = RandomForestClassifier(
        n_estimators=600,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight={0: 1.0, 1: 1.0, 2: 2.0},
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        max_depth=12,
    )

    n_classes = y.nunique()
    oof_proba = np.zeros((len(X_raw), n_classes))

    gkf = GroupKFold(n_splits=5)

    for tr_idx, val_idx in gkf.split(X_raw, y, groups):
        model = base_model.__class__(**base_model.get_params())
        model.fit(X_raw.iloc[tr_idx], y.iloc[tr_idx])
        oof_proba[val_idx] = model.predict_proba(X_raw.iloc[val_idx])

    base_model.fit(X_raw.loc[train_idx], y.loc[train_idx])
    test_proba = base_model.predict_proba(X_raw.loc[test_idx])

    proba_cols = [f"stage1_raw_p{i}" for i in range(n_classes)]
    proba_all = pd.DataFrame(oof_proba, columns=proba_cols, index=X_raw.index)
    proba_test = pd.DataFrame(test_proba, columns=proba_cols, index=test_idx)

    # ===== STAGE 2 =====
    X_stage2_train = pd.concat(
        [X_filt.loc[train_idx], proba_all.loc[train_idx]], axis=1
    )
    X_stage2_test = pd.concat(
        [X_filt.loc[test_idx], proba_test], axis=1
    )

    models = {
        "RF": RandomForestClassifier(n_estimators=600, random_state=42),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=200,
            learning_rate=0.7,
            random_state=42,
        ),
        "LogReg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced"),
        ),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_stage2_train, y.loc[train_idx])
        y_pred = model.predict(X_stage2_test)

        results[name] = {
            "model": model,
            "y_pred": y_pred,
        }

    return results
