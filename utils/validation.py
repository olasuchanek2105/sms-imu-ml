"""
Moduł zawierający funkcje walidacyjne i diagnostyczne
wspierające kontrolę jakości danych oraz poprawności
procesu uczenia modeli klasyfikacyjnych.

Funkcjonalności modułu obejmują:
- analizę rozkładu klas zmiennej docelowej (target),
  umożliwiającą wczesne wykrycie niezbalansowania danych,
- weryfikację poprawności predykcji typu out-of-fold (OOF)
  poprzez porównanie ich z predykcjami modelu uczonego
  na pełnym zbiorze treningowym,
- detekcję potencjalnego przecieku informacji (data leakage),
  który mógłby prowadzić do sztucznego zawyżenia wyników
  klasyfikacji.

Moduł ten pełni rolę narzędzia kontrolnego w potoku
uczenia maszynowego i jest wykorzystywany na etapie
przygotowania danych oraz konstrukcji modeli wieloetapowych
(stacking), zapewniając rzetelność i wiarygodność uzyskanych
rezultatów.
"""



import numpy as np

def check_target_distribution(y):
    print("UNIQUE TARGETS:", np.unique(y))
    print("CLASS COUNTS:\n", y.value_counts())


def check_oof_vs_full(oof, full_pred, threshold=0.01):
    diff = np.mean(np.abs(oof - full_pred))
    print("OOF vs FULL diff:", diff)
    assert diff > threshold
    return diff
