from sklearn.model_selection import GroupKFold, cross_val_score


def run_model_group_cv(
    model,
    X,
    y,
    groups,
    n_splits=5
):
    cv = GroupKFold(n_splits=n_splits)

    scores_f1 = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        groups=groups,
        scoring="f1_macro",
        n_jobs=-1
    )

    scores_acc = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        groups=groups,
        scoring="accuracy",
        n_jobs=-1
    )

    return {
        "f1_mean": scores_f1.mean(),
        "f1_std": scores_f1.std(),
        "acc_mean": scores_acc.mean(),
        "acc_std": scores_acc.std(),
    }


def run_group_cv(
    models,
    X,
    y,
    groups,
    n_splits=5
):
    """
    Uruchamia GroupKFold CV dla podanych modeli
    i zwraca średnie oraz odchylenia metryk.
    """
    results = {}

    for name, model in models.items():
        results[name] = run_model_group_cv(
            model,
            X,
            y,
            groups,
            n_splits
        )

    return results
