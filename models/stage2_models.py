

"""
stage2_models.py
================

Moduł definiujący zestaw modeli klasyfikacyjnych wykorzystywanych
w drugim etapie architektury dwuetapowej (stacking).

Etap 2 odpowiada za końcową klasyfikację ryzyka na podstawie:
- cech wyekstrahowanych z przefiltrowanego sygnału (FILT),
- dodatkowej cechy stakowanej pochodzącej z predykcji
  probabilistycznych modelu etapu 1.

W module zdefiniowano trzy różne klasyfikatory, reprezentujące
odmienne podejścia do uczenia maszynowego:
- Random Forest – model zespołowy oparty na drzewach decyzyjnych,
- AdaBoost – algorytm wzmacniający sekwencję słabych klasyfikatorów,
- Regresję logistyczną – model liniowy z normalizacją cech.

Zastosowanie kilku modeli w drugim etapie umożliwia:
- porównanie skuteczności algorytmów o różnej charakterystyce,
- analizę wpływu architektury modelu na końcowe decyzje,
- wybór najlepszego rozwiązania na podstawie wyników testowych.

Wszystkie modele są inicjalizowane z uwzględnieniem
niezrównoważenia klas (class_weight="balanced") oraz
parametrów zapewniających powtarzalność eksperymentów.
"""
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def get_stage2_models(random_state=42):
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=600,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=200,
            learning_rate=0.7,
            random_state=random_state
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
