from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier



def train_single_rf(
    X,
    y,
    groups,
    train_idx,
    test_idx,
):
    model = RandomForestClassifier(
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

    model.fit(X.loc[train_idx], y.loc[train_idx])
    y_pred = model.predict(X.loc[test_idx])

    return {
        "model": model,
        "y_pred": y_pred,
    }
