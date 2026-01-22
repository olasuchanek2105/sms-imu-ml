

"""
plots.py
========

Moduł pomocniczy odpowiedzialny za wizualizację wyników analizy
ważności cech (feature importance).

Celem modułu jest:
- ułatwienie interpretacji modeli klasyfikacyjnych,
- graficzne przedstawienie najbardziej istotnych cech,
- zapewnienie spójnego stylu wykresów w całym projekcie.

Moduł wykorzystywany jest m.in. do prezentacji wyników:
- modeli jednoetapowych (baseline),
- modeli drugiego etapu w architekturze dwuetapowej (stacking).
"""

import matplotlib.pyplot as plt

def plot_top_features(df, title, xlabel, top_n=10):
    """
    Rysuje wykres ważności cech na podstawie DataFrame
    zawierającego kolumny: 'feature', 'importance'.
    """
    top_features = df.head(top_n)

    plt.figure(figsize=(6, 9))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.title(title, fontsize=18)
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.show()
