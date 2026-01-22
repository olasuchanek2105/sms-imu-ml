

"""
stacking_features.py
====================

Moduł odpowiedzialny za budowę cech stakowanych oraz przygotowanie
wejścia do drugiego etapu architektury dwuetapowej (stacking).

Celem modułu jest:
- przekształcenie predykcji probabilistycznych z etapu 1
  w dodatkową cechę wejściową,
- zapewnienie ścisłej separacji zbioru treningowego i testowego,
- przygotowanie finalnych macierzy cech dla modeli etapu 2.

W pierwszym kroku predykcje typu out-of-fold (OOF) wygenerowane
przez model etapu 1 są mapowane do postaci jednej cechy
reprezentującej prawdopodobieństwo przynależności do klasy
wysokiego ryzyka.

Następnie cecha ta jest łączona z zestawem cech obliczonych
na sygnale przefiltrowanym (FILT), tworząc pełne wejście
dla modeli drugiego etapu.

Moduł zawiera zabezpieczenia przed przeciekiem danych
(data leakage), w szczególności poprzez weryfikację
rozłączności indeksów zbioru treningowego i testowego.

Implementacja jest zgodna z dobrymi praktykami stosowanymi
w algorytmach typu stacking oraz umożliwia jednoznaczną
interpretację wpływu predykcji etapu 1 na decyzje końcowe
modelu.
"""
import pandas as pd

def build_stage1_feature(oof_train, p_test, train_index, test_index):
    """
    Buduje cechę stakowaną na podstawie predykcji etapu 1.
    """
    X_train_feat = pd.DataFrame(
        {"stage1_p_high": oof_train},
        index=train_index
    )

    X_test_feat = pd.DataFrame(
        {"stage1_p_high": p_test},
        index=test_index
    )

    assert set(X_train_feat.index).isdisjoint(set(X_test_feat.index))

    return X_train_feat, X_test_feat




def build_stage2_input(
    X_filt_train,
    X_filt_test,
    proba_train,
    proba_test
):
    """
    Buduje macierze wejściowe dla etapu 2 (FILT + predykcje etapu 1).
    """
    X_stage2_train = pd.concat([X_filt_train, proba_train], axis=1)
    X_stage2_test  = pd.concat([X_filt_test,  proba_test], axis=1)

    return X_stage2_train, X_stage2_test
