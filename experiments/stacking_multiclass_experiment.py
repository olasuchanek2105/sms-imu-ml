"""
EKSPERYMENTALNY SKRYPT — STACKING MULTICLASS (0/1/2)

UWAGA:
- plik NIE jest częścią finalnego pipeline'u,
- służy jako dokumentacja prób z klasyfikacją wieloklasową,
- kod zachowany w oryginalnej formie (research history).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler





# Funkcja pomocnicza do wczytania i przygotowania

def load_and_prepare(path):
    df = pd.read_csv(path, sep=';')

    # zamiana przecinków na kropki w kolumnach tekstowych
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.', regex=False)

    # konwersja wszystkiego co się da na liczby
    df = df.apply(pd.to_numeric, errors='ignore')

    # subject z nazwy pliku: np. "P_AD_BezGogli_cut.csv" -> "p_ad"
    df["subject"] = (
        df["file"]
        .astype(str)
        .str.lower()
        .str.replace(".csv", "", regex=False)
        .str.split("_")
        .str[:2]
        .str.join("_")
    )

    # wspólna baza nazwy pliku dla RAW i FILT
    # np. "P_AD_BezGogli_cut.csv"          -> "p_ad_bezgogli_cut"
    #     "P_AD_BezGogli_cut_filtered.csv" -> "p_ad_bezgogli_cut"
    df["file_base"] = (
        df["file"]
        .astype(str)
        .str.lower()
        .str.replace(".csv", "", regex=False)
        .str.replace("_filtered", "", regex=False)
    )

    return df


# Wczytanie dwóch wersji cech =

path_raw   = r"C:\Users\olasu\OneDrive\Desktop\pracaInz\projektInzynierski\praca_features\FEATURES\features_5s_windows_not_filtered.csv"
path_filt  = r"C:\Users\olasu\OneDrive\Desktop\pracaInz\projektInzynierski\praca_features\FEATURES\features_5s_windows_filtered.csv"

df_raw  = load_and_prepare(path_raw)   # dane z NIEprzefiltrowanego sygnału
df_filt = load_and_prepare(path_filt)  # dane z przefiltrowanego sygnału

#  indeksowanie okna w obrębie "file_base", żeby móc 1:1 dopasować raw <-> filt
for df in (df_raw, df_filt):
    df["window_idx"] = df.groupby("file_base").cumcount()

# sprwadzeie  ile okien na plik (po bazie nazwy)
print("\n Liczba okien na plik_base (RAW)")
print(df_raw.groupby("file_base")["window_idx"].max() + 1)

print("\n Liczba okien na plik_base (FILT) ")
print(df_filt.groupby("file_base")["window_idx"].max() + 1)

# Sprawdź, czy zestaw (file_base, window_idx) jest identyczny
keys_raw  = set(zip(df_raw["file_base"],  df_raw["window_idx"]))
keys_filt = set(zip(df_filt["file_base"], df_filt["window_idx"]))

if keys_raw != keys_filt:
    raise ValueError("Zestaw okien (file_base, window_idx) nie jest taki sam w RAW i FILT")

# Ustaw df_filt w dokładnie tej samej kolejności wierszy co df_raw
df_raw_indexed  = df_raw.set_index(["file_base", "window_idx"])
df_filt_indexed = df_filt.set_index(["file_base", "window_idx"])

df_filt_aligned = df_filt_indexed.loc[df_raw_indexed.index].reset_index()
df_raw          = df_raw_indexed.reset_index()
df_filt         = df_filt_aligned

# asserty 
assert len(df_raw) == len(df_filt), "Liczba wierszy się nie zgadza"
assert (df_raw["target"].values == df_filt["target"].values).all(), "Targety się różnią"

# kilka pierwszych wierszy czy wygląda sensownie
print("\nPodgląd po dopasowaniu RAW/FILT:")
print(df_raw[["file_base", "file", "window_idx"]].head())
print(df_filt[["file_base", "file", "window_idx"]].head())

# wspólne rzeczy
y = df_raw["target"].astype(int)
groups = df_raw["subject"]


# =========================
# SPLIT TRAIN / TEST (po SUBJECT)
# =========================

# etykieta subjecta = dominująca klasa
subject_labels = df_raw.groupby("subject")["target"].agg(lambda x: x.value_counts().idxmax())
unique_subjects = subject_labels.index.values
y_subject = subject_labels.values

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_subj_idx, test_subj_idx = next(sss.split(unique_subjects, y_subject))

train_subjects = unique_subjects[train_subj_idx]
test_subjects  = unique_subjects[test_subj_idx]

train_idx = df_raw[df_raw["subject"].isin(train_subjects)].index
test_idx  = df_raw[df_raw["subject"].isin(test_subjects)].index

print("\nTRAIN subjects:", sorted(train_subjects))
print("TEST subjects:",  sorted(test_subjects))

# sanity check
assert set(train_subjects).isdisjoint(set(test_subjects))


# =========================
# CECHY
# =========================

cols_to_drop = ["file", "file_base", "subject", "ankieta_score", "target", "window_idx"]

X_raw  = df_raw.drop(columns=cols_to_drop, errors="ignore")
X_filt = df_filt.drop(columns=cols_to_drop, errors="ignore")

X_raw_train  = X_raw.loc[train_idx]
X_raw_test   = X_raw.loc[test_idx]

X_filt_train = X_filt.loc[train_idx]
X_filt_test  = X_filt.loc[test_idx]

y_train = y.loc[train_idx]
y_test  = y.loc[test_idx]

groups_train = groups.loc[train_idx]


# =========================
# ETAP 1 — RANDOM FOREST (RAW)
# =========================

from sklearn.base import clone

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

n_classes = y_train.nunique()

# reset indeksów (WAŻNE!)
X_raw_train_ = X_raw_train.reset_index(drop=True)
y_train_ = y_train.reset_index(drop=True)
groups_train_ = groups_train.reset_index(drop=True)

oof_proba_train = np.zeros((len(X_raw_train_), n_classes))

gkf = GroupKFold(n_splits=5)

all_classes = np.sort(y_train_.unique())
n_classes = len(all_classes)

for fold, (tr, val) in enumerate(gkf.split(X_raw_train_, y_train_, groups_train_)):
    print(f"Stage 1 OOF fold {fold+1}")
    model_fold = clone(base_model)
    model_fold.fit(X_raw_train_.iloc[tr], y_train_.iloc[tr])
    proba = model_fold.predict_proba(X_raw_train_.iloc[val])

    # mapowanie do pełnego wektora klas
    proba_full = np.zeros((len(val), n_classes))

    for i, cls in enumerate(model_fold.classes_):
        cls_idx = np.where(all_classes == cls)[0][0]
        proba_full[:, cls_idx] = proba[:, i]

    oof_proba_train[val] = proba_full

############WERYFIKACJA
print("\n[CHECK 1] Subject leakage")

train_subjects_set = set(groups_train_)
test_subjects_set = set(groups.loc[test_idx])

assert train_subjects_set.isdisjoint(test_subjects_set), \
    "❌ PRZECIEK: subject jest jednocześnie w TRAIN i TEST"

print("✔ OK: brak wspólnych subjectów między TRAIN i TEST")


# model full (tylko TRAIN)
base_model_full = clone(base_model)
base_model_full.fit(X_raw_train, y_train)

proba_test = base_model_full.predict_proba(X_raw_test)

proba_cols = [f"stage1_raw_p{c}" for c in range(n_classes)]

proba_train = pd.DataFrame(
    oof_proba_train,
    columns=proba_cols,
    index=X_filt_train.index
)

proba_test = pd.DataFrame(
    proba_test,
    columns=proba_cols,
    index=X_filt_test.index
)

print("\n[CHECK 2] OOF vs full-model sanity check")

# predykcje z modelu uczonego na CAŁYM TRAIN
full_train_proba = base_model_full.predict_proba(X_raw_train)

# mapowanie do pełnego wektora klas
full_train_proba_mapped = np.zeros_like(oof_proba_train)

for i, cls in enumerate(base_model_full.classes_):
    cls_idx = np.where(all_classes == cls)[0][0]
    full_train_proba_mapped[:, cls_idx] = full_train_proba[:, i]

# średnia różnica
mean_diff = np.mean(np.abs(oof_proba_train - full_train_proba_mapped))
print(f"Średnia |OOF − full| = {mean_diff:.4f}")

assert mean_diff > 0.01, \
    "❌ PODEJRZENIE LEAKU: OOF zbyt podobne do predykcji modelu full"

print("✔ OK: OOF różnią się od predykcji modelu full (jak powinny)")


# =========================
# ETAP 2 — WEJŚCIE
# =========================

X_stage2_train = pd.concat([X_filt_train, proba_train], axis=1)
X_stage2_test  = pd.concat([X_filt_test,  proba_test], axis=1)


# =========================
# ETAP 2 — MODELE
# =========================

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=600,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=200,
        learning_rate=0.7,
        random_state=42
    ),
    "LogReg": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs"
        )
    ),
}

cv2 = GroupKFold(n_splits=5)

print("\n===== CV STAGE 2 (TRAIN ONLY) =====")

for name, model in models.items():
    scores = cross_val_score(
        model,
        X_stage2_train,
        y_train,
        cv=cv2,
        groups=groups_train,
        scoring="f1_macro",
        n_jobs=-1
    )
    print(f"{name}: macro-F1 = {scores.mean():.4f} ± {scores.std():.4f}")



# =========================
# TEST — FINALNE WYNIKI
# =========================

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

for name, model in models.items():
    print("\n==============================")
    print(f"TEST — STAGE 2 — {name}")
    print("==============================")

    model.fit(X_stage2_train, y_train)
    y_pred = model.predict(X_stage2_test)

    print("macro-F1:", f1_score(y_test, y_pred, average="macro"))
    print("accuracy:", accuracy_score(y_test, y_pred))
    print("confusion matrix:\n", confusion_matrix(y_test, y_pred))


print("\n[CHECK 3] Integralność OOF")

assert not np.isnan(oof_proba_train).any(), "❌ NaN w OOF"
assert np.all(oof_proba_train >= 0), "❌ Ujemne prawdopodobieństwa"
assert np.all(oof_proba_train <= 1), "❌ P > 1"

row_sums = oof_proba_train.sum(axis=1)

assert np.all(row_sums <= 1.0001), "❌ Suma prawdopodobieństw > 1"
print("✔ OK: OOF poprawne numerycznie")

print("\n[CHECK 4] Test isolation")

assert set(proba_train.index).isdisjoint(set(proba_test.index)), \
    "❌ PRZECIEK: wspólne indeksy TRAIN i TEST w cechach stakowanych"

print("✔ OK: test odizolowany od cech OOF")


print("\n[CHECK 5] Permutation test (leak detector)")

from sklearn.metrics import f1_score

rng = np.random.default_rng(42)
y_perm = rng.permutation(y_train_.values)

perm_scores = []

for tr, val in gkf.split(X_raw_train_, y_perm, groups_train_):
    m = clone(base_model)
    m.fit(X_raw_train_.iloc[tr], y_perm[tr])
    y_pred = m.predict(X_raw_train_.iloc[val])
    perm_scores.append(f1_score(y_perm[val], y_pred, average="macro"))

print("Permutation macro-F1:", np.mean(perm_scores))

assert np.mean(perm_scores) < 0.45, \
    "❌ PODEJRZENIE LEAKU: wysoki wynik na losowych etykietach"

print("✔ OK: brak sygnału przecieku (wynik ~ losowy)")



# # Podział train/test po SUBJECT (osobach)

# subject_labels = df_raw.groupby("subject")["target"].agg(lambda x: x.value_counts().idxmax())
# unique_subjects = subject_labels.index.values
# y_subject = subject_labels.values

# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
# train_subj_idx, test_subj_idx = next(sss.split(unique_subjects, y_subject))

# train_subjects = unique_subjects[train_subj_idx]
# test_subjects  = unique_subjects[test_subj_idx]

# train_idx = df_raw[df_raw["subject"].isin(train_subjects)].index
# test_idx  = df_raw[df_raw["subject"].isin(test_subjects)].index

# print("\nTRAIN subjects:", sorted(train_subjects))
# print("TEST subjects:",  sorted(test_subjects))


# # Tworzymy X z obu wersji cech 

# cols_to_drop = ["file", "file_base", "subject", "ankieta_score", "target", "window_idx"]

# X_raw  = df_raw.drop(columns=cols_to_drop, errors="ignore")
# X_filt = df_filt.drop(columns=cols_to_drop, errors="ignore")

# X_raw_train  = X_raw.loc[train_idx]
# X_raw_test   = X_raw.loc[test_idx]
# X_filt_train = X_filt.loc[train_idx]
# X_filt_test  = X_filt.loc[test_idx]

# y_train = y.loc[train_idx]
# y_test  = y.loc[test_idx]


# #ETAP 1: model na NIEprzefiltrowanych danych

# base_model = RandomForestClassifier(
#     n_estimators=600,
#     min_samples_split=5,
#     min_samples_leaf=2,
#     max_features="sqrt",
#     class_weight={0: 1.0, 1: 1.0, 2: 2.0},
#     bootstrap=True,
#     oob_score=True,
#     random_state=42,
#     n_jobs=-1,
#     max_depth=12,
# )

# n_classes = y.nunique()




# # Out-of-fold OOF predykcje do użycia jako cechy w etapie 2
# oof_proba = np.zeros((len(df_raw), n_classes))

# gkf = GroupKFold(n_splits=5)

# for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_raw, y, groups)):
#     print(f"\nFold {fold+1} (Stage 1, RAW)...")
#     model_fold = base_model.__class__(**base_model.get_params())  # klon
#     model_fold.fit(X_raw.iloc[tr_idx], y.iloc[tr_idx])
#     oof_proba[val_idx] = model_fold.predict_proba(X_raw.iloc[val_idx])

# # Model uczony tylko na train → predykcje na teście
# base_model_full = base_model.__class__(**base_model.get_params())
# base_model_full.fit(X_raw_train, y_train)

# test_proba = base_model_full.predict_proba(X_raw_test)

# proba_cols = [f"stage1_raw_p{c}" for c in range(n_classes)]
# proba_df_all  = pd.DataFrame(oof_proba, columns=proba_cols, index=df_raw.index)
# proba_df_test = pd.DataFrame(test_proba, columns=proba_cols, index=X_raw_test.index)

# proba_train = proba_df_all.loc[train_idx]
# proba_test  = proba_df_test.loc[test_idx]


# # ETAP 2: model na przefiltrowanych cechach + predykcje z ETAPU 1

# X_stage2_train = pd.concat([X_filt_train, proba_train], axis=1)
# X_stage2_test  = pd.concat([X_filt_test,  proba_test],  axis=1)

# models = {


#     "RandomForest_stage2": RandomForestClassifier(
#         n_estimators=600,
#         min_samples_split=5,
#         min_samples_leaf=2,
#         max_features="sqrt",
#         class_weight="balanced_subsample",
#         bootstrap=True,
#         oob_score=True,
#         random_state=42,
#         n_jobs=-1
#     ),
#     "AdaBoost_stage2": AdaBoostClassifier(
#         estimator=DecisionTreeClassifier(max_depth=2),
#         n_estimators=200,
#         learning_rate=0.7,
#         random_state=42
#     ),

#     "LogReg_stage2": make_pipeline(
#         StandardScaler(),
#         LogisticRegression(
#             class_weight="balanced",
#             max_iter=5000,      # więcej iteracji
#             solver="lbfgs",
#             n_jobs=-1
#         )
#     ),
    
    
# }










# for name, model in models.items():
#     print("\n==========================")
#     print(f"MODEL STAGE 2: {name}")
#     print("==========================")

#     model.fit(X_stage2_train, y_train)

#     # CV z grupami na stage2 (na CAŁYM zbiorze)
#     cv = GroupKFold(n_splits=5)
#     X_stage2_all = pd.concat([X_filt, proba_df_all], axis=1)

#     scores = cross_val_score(
#         model,
#         X_stage2_all,
#         y,
#         cv=cv,
#         groups=groups,
#         scoring="f1_macro",
#         n_jobs=-1
#     )
#     print("Średni macro-F1 (CV, stage2):", np.mean(scores))

#     y_pred = model.predict(X_stage2_test)

#     print("\n== Raport na TEST (stage2) ====================")
#     print(classification_report(y_test, y_pred, digits=3))

#     print("== Macierz Pomyłek (stage2) ===================")
#     print(confusion_matrix(y_test, y_pred))

#     if hasattr(model, "feature_importances_"):
#         importances = pd.Series(model.feature_importances_, index=X_stage2_train.columns).sort_values(ascending=False)
#         print("\nTop 10 cech (stage2):")
#         print(importances.head(10))


#     top10 = importances.head(10).sort_values()

#     plt.figure(figsize=(8, 6))
#     top10.plot(kind="barh")
#     plt.title(f"Top 10 najważniejszych cech – {name}")
#     plt.xlabel("Ważność cechy")
#     plt.tight_layout()

#     plt.savefig("feature_importance_adaboost_stage2_h.png", dpi=300)
#     plt.show()





# print("\n==========================")
# print("SINGLE-LEVEL RF — RAW")
# print("==========================")

# rf_raw = RandomForestClassifier(
#     n_estimators=600,
#     min_samples_split=5,
#     min_samples_leaf=2,
#     max_features="sqrt",
#     class_weight={0: 1.0, 1: 1.0, 2: 2.0},
#     bootstrap=True,
#     random_state=42,
#     n_jobs=-1,
#     max_depth=12,
# )

# # CV (GroupKFold) — dokładnie jak w stage2
# cv = GroupKFold(n_splits=5)
# scores_raw = cross_val_score(
#     rf_raw,
#     X_raw,
#     y,
#     cv=cv,
#     groups=groups,
#     scoring="f1_macro",
#     n_jobs=-1
# )

# print("Średni macro-F1 (CV, RAW):", scores_raw.mean())

# # Test
# rf_raw.fit(X_raw_train, y_train)
# y_pred_raw = rf_raw.predict(X_raw_test)

# print("\n== Raport TEST (RAW) ==")
# print(classification_report(y_test, y_pred_raw, digits=3))
# print("== Confusion matrix (RAW) ==")
# print(confusion_matrix(y_test, y_pred_raw))



# print("\n==========================")
# print("SINGLE-LEVEL RF — FILT")
# print("==========================")

# rf_filt = RandomForestClassifier(
#     n_estimators=600,
#     min_samples_split=5,
#     min_samples_leaf=2,
#     max_features="sqrt",
#     class_weight={0: 1.0, 1: 1.0, 2: 2.0},
#     bootstrap=True,
#     random_state=42,
#     n_jobs=-1,
#     max_depth=12,
# )

# scores_filt = cross_val_score(
#     rf_filt,
#     X_filt,
#     y,
#     cv=cv,
#     groups=groups,
#     scoring="f1_macro",
#     n_jobs=-1
# )

# print("Średni macro-F1 (CV, FILT):", scores_filt.mean())

# rf_filt.fit(X_filt_train, y_train)
# y_pred_filt = rf_filt.predict(X_filt_test)

# print("\n== Raport TEST (FILT) ==")
# print(classification_report(y_test, y_pred_filt, digits=3))
# print("== Confusion matrix (FILT) ==")
# print(confusion_matrix(y_test, y_pred_filt))
