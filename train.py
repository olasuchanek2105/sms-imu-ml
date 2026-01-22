"""
Główny skrypt treningowy realizujący kompletny potok
uczenia i ewaluacji modeli klasyfikacyjnych dla problemu
oceny ryzyka wystąpienia choroby kosmicznej (SMS) na podstawie
cech wyekstrahowanych z danych IMU.

Skrypt implementuje następujące etapy:
1. Wczytanie zestawów cech obliczonych na sygnale surowym (RAW)
   oraz przefiltrowanym (FILT).
2. Synchronizację okien czasowych pomiędzy danymi RAW i FILT.
3. Podział danych na zbiory treningowy i testowy z zachowaniem
   separacji na poziomie badanych (subject-level split).
4. Trening modelu pierwszego etapu (Random Forest) na cechach RAW
   oraz generowanie predykcji typu out-of-fold (OOF).
5. Konstrukcję cech stakowanych na podstawie predykcji etapu 1
   i połączenie ich z cechami FILT.
6. Trening i walidację modeli drugiego etapu (Random Forest,
   AdaBoost, Logistic Regression) z użyciem GroupKFold.
7. Ocenę końcową modeli na zbiorze testowym.
8. Porównanie z modelami jednoetapowymi (baseline) uczonymi
   wyłącznie na cechach RAW oraz FILT.
9. Analizę ważności cech dla wybranych modeli oraz wizualizację
   najbardziej informacyjnych zmiennych.

Skrypt stanowi centralny element eksperymentalny pracy i zapewnia
pełną kontrolę nad poprawnością walidacji, izolacją zbioru testowego
oraz wykrywaniem potencjalnego przecieku informacji (data leakage).
"""



from utils.io import load_and_prepare
from preprocessing.segment_window import add_window_index_and_align
from utils.validation import check_target_distribution
from utils.splits import subject_stratified_train_test_split
from utils.features import prepare_feature_sets
from models.rf_stage1 import build_rf_stage1, compute_oof_stage1
from models.stacking_features import build_stage1_feature
from utils.validation import check_oof_vs_full
from models.stacking_features import build_stage2_input
from models.stage2_models import get_stage2_models
from evaluation.cross_validation import run_group_cv
from evaluation.plots import plot_top_features
from evaluation.test_evaluation import evaluate_on_test
from models.rf_stage1 import (
    build_rf_stage1,
    compute_oof_stage1,
    fit_stage1_full_and_predict
)

 
path_filt = r"data\features_5s_windows_filtered_binary_problem.csv"
path_raw = r"data\features_5s_windows_not_filtered_binary_problem.csv"

df_raw  = load_and_prepare(path_raw)
df_filt = load_and_prepare(path_filt)

df_raw, df_filt = add_window_index_and_align(df_raw, df_filt)


y = df_raw["target"].astype(int)
groups = df_raw["subject"]

check_target_distribution(y)



train_idx, test_idx = subject_stratified_train_test_split(df_raw, y)


data = prepare_feature_sets(
    df_raw,
    df_filt,
    y,
    groups,
    train_idx,
    test_idx
)

X_raw_train  = data["X_raw_train"]
X_raw_test   = data["X_raw_test"]
X_filt_train = data["X_filt_train"]
X_filt_test  = data["X_filt_test"]
y_train      = data["y_train"]
y_test       = data["y_test"]
groups_train = data["groups_train"]







rf_stage1 = build_rf_stage1()

oof_p1_train = compute_oof_stage1(
    rf_stage1,
    X_raw_train,
    y_train,
    groups_train
)

rf_full, p1_train_full, p1_test = fit_stage1_full_and_predict(
    rf_stage1,
    X_raw_train,
    y_train,
    X_raw_test
)

check_oof_vs_full(oof_p1_train, p1_train_full)

proba_train, proba_test = build_stage1_feature(
    oof_p1_train,
    p1_test,
    X_filt_train.index,
    X_filt_test.index
)




X_stage2_train, X_stage2_test = build_stage2_input(
    X_filt_train,
    X_filt_test,
    proba_train,
    proba_test
)

models = get_stage2_models()



print("\n===== CV STAGE 2 =====")
cv_results = run_group_cv(
    models,
    X_stage2_train,
    y_train,
    groups_train
)

for name, r in cv_results.items():
    print(
        f"{name}: "
        f"macro-F1 = {r['f1_mean']:.4f} ± {r['f1_std']:.4f}, "
        f"accuracy = {r['acc_mean']:.4f} ± {r['acc_std']:.4f}"
    )

print("\n===== TEST =====")
test_results = evaluate_on_test(
    models,
    X_stage2_train,
    y_train,
    X_stage2_test,
    y_test
)

for name, r in test_results.items():
    print("\n", name)
    print("macro-F1:", r["macro_f1"])
    print("accuracy:", r["accuracy"])
    print("confusion matrix:\n", r["confusion_matrix"])


from evaluation.baselines import (
    get_single_stage_models,
    run_single_stage_cv,
    evaluate_single_stage_on_test
)

print("\n===== CV SINGLE-STAGE (RAW & FILT) =====")

single_stage_models = get_single_stage_models(
    X_raw_train,
    X_raw_test,
    X_filt_train,
    X_filt_test
)

cv_baseline = run_single_stage_cv(
    single_stage_models,
    y_train,
    groups_train
)

for name, r in cv_baseline.items():
    print(
        f"{name}: "
        f"macro-F1 = {r['f1_mean']:.4f} ± {r['f1_std']:.4f}, "
        f"accuracy = {r['acc_mean']:.4f} ± {r['acc_std']:.4f}"
    )

print("\n===== TEST SINGLE-STAGE (RAW & FILT) =====")

test_baseline = evaluate_single_stage_on_test(
    single_stage_models,
    y_train,
    y_test
)

for name, r in test_baseline.items():
    print("\n", name)
    print("macro-F1:", r["macro_f1"])
    print("accuracy:", r["accuracy"])
    print("confusion matrix:\n", r["confusion_matrix"])


from evaluation.feature_importance import (
    rf_single_stage_importance,
    rf_stage2_importance,
    logreg_stage2_importance,
    adaboost_stage2_importance
)


# RF single-stage (RAW)
fi_rf_single = rf_single_stage_importance(X_raw_train, y_train)
print(fi_rf_single.head(15))

# RF stage 2
fi_rf_stage2 = rf_stage2_importance(X_stage2_train, y_train)
print(fi_rf_stage2.head(10))

# LogReg stage 2
fi_logreg = logreg_stage2_importance(X_stage2_train, y_train)
print(fi_logreg.head(10))

# AdaBoost stage 2
fi_adaboost = adaboost_stage2_importance(X_stage2_train, y_train)
print(fi_adaboost.head(15))

plot_top_features(
    fi_adaboost,
    title="Najważniejsze cechy — AdaBoost (Stage 2)",
    xlabel="Ważność cechy"
)
