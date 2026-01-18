from utils.io import load_and_prepare
from models.stacking import train_stacking_model
from models.rf import train_single_rf
from evaluation.metrics import evaluate_on_test, cross_validate_macro_f1
from evaluation.plots import plot_confusion_matrix, plot_feature_importance

from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd


# ===== ŚCIEŻKI DO DANYCH =====
PATH_RAW = "data/features_5s_windows_not_filtered.csv"
PATH_FILT = "data/features_5s_windows_filtered.csv"


def main():
    # ===== LOAD DATA =====
    df_raw = load_and_prepare(PATH_RAW)
    df_filt = load_and_prepare(PATH_FILT)

    # ===== ALIGN RAW / FILT WINDOWS =====
    for df in (df_raw, df_filt):
        df["window_idx"] = df.groupby("file_base").cumcount()

    df_raw_idx = df_raw.set_index(["file_base", "window_idx"])
    df_filt_idx = df_filt.set_index(["file_base", "window_idx"])
    df_filt = df_filt_idx.loc[df_raw_idx.index].reset_index()
    df_raw = df_raw_idx.reset_index()

    # ===== TARGET & GROUPS =====
    y = df_raw["target"].astype(int)
    groups = df_raw["subject"]

    # ===== SUBJECT-WISE TRAIN / TEST SPLIT =====
    subject_labels = df_raw.groupby("subject")["target"].agg(
        lambda x: x.value_counts().idxmax()
    )

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_subj_idx, test_subj_idx = next(
        sss.split(subject_labels.index, subject_labels.values)
    )

    train_subjects = subject_labels.index[train_subj_idx]
    test_subjects = subject_labels.index[test_subj_idx]

    train_idx = df_raw[df_raw["subject"].isin(train_subjects)].index
    test_idx = df_raw[df_raw["subject"].isin(test_subjects)].index

    # ===== FEATURE MATRICES =====
    cols_to_drop = [
        "file",
        "file_base",
        "subject",
        "ankieta_score",
        "target",
        "window_idx",
    ]

    X_raw = df_raw.drop(columns=cols_to_drop, errors="ignore")
    X_filt = df_filt.drop(columns=cols_to_drop, errors="ignore")

    # ==========================================================
    # SINGLE-LEVEL RANDOM FOREST
    # ==========================================================
    rf_results = train_single_rf(
        X_raw,
        y,
        groups,
        train_idx,
        test_idx,
    )

    rf_metrics = evaluate_on_test(
        y.loc[test_idx],
        rf_results["y_pred"],
    )

    print("\nSINGLE RF macro-F1:", rf_metrics["macro_f1"])

    plot_confusion_matrix(
        rf_metrics["confusion_matrix"],
        title="Single-level RF (RAW)",
        save_path="figures/confusion_rf_raw.png",
    )

    # ==========================================================
    # STACKING MODEL
    # ==========================================================
    stacking_results = train_stacking_model(
        X_raw,
        X_filt,
        y,
        groups,
        train_idx,
        test_idx,
    )

    for name, result in stacking_results.items():
        metrics = evaluate_on_test(
            y.loc[test_idx],
            result["y_pred"],
        )

        print(f"\nSTACKING [{name}] macro-F1:", metrics["macro_f1"])

        plot_confusion_matrix(
            metrics["confusion_matrix"],
            title=f"Stacking – {name}",
            save_path=f"figures/confusion_stacking_{name}.png",
        )

        if "importances" in result:
            plot_feature_importance(
                result["importances"],
                title=f"Top features – {name}",
                save_path=f"figures/feature_importance_{name}.png",
            )


if __name__ == "__main__":
    main()
