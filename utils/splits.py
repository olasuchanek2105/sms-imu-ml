
"""
Moduł odpowiedzialny za wykonywanie podziału zbioru danych
na część treningową i testową z zachowaniem integralności
na poziomie badanych (subject-level split).

Zaimplementowana metoda podziału:
- realizuje rozdział danych na poziomie unikalnych osób (subject),
  eliminując ryzyko przecieku informacji pomiędzy zbiorem treningowym
  i testowym,
- zachowuje stratyfikację klas na poziomie subject, poprzez przypisanie
  każdemu badanemu etykiety odpowiadającej dominującej klasie w jego danych,
- umożliwia kontrolę proporcji podziału oraz powtarzalność wyników
  dzięki parametrowi random_state.

Moduł ten jest kluczowym elementem potoku walidacyjnego, ponieważ zapewnia
realistyczną ocenę skuteczności modeli uczonych na danych sekwencyjnych
pochodzących od wielu badanych, zgodnie z dobrymi praktykami uczenia
maszynowego w analizie sygnałów biomedycznych.
"""

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def subject_stratified_train_test_split(
    df,
    y,
    test_size=0.3,
    random_state=42
):
    """
    Wykonuje podział train/test na poziomie subject
    z zachowaniem stratyfikacji klas (binarnie).
    """
    groups = df["subject"]

    subject_labels = (
        pd.DataFrame({"subject": groups, "y": y})
        .groupby("subject")["y"]
        .agg(lambda x: x.value_counts().idxmax())
    )

    subjects = subject_labels.index.values
    y_subject = subject_labels.values

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_s, test_s = next(sss.split(subjects, y_subject))

    train_subjects = subjects[train_s]
    test_subjects  = subjects[test_s]

    train_idx = df[df["subject"].isin(train_subjects)].index
    test_idx  = df[df["subject"].isin(test_subjects)].index

    assert set(train_subjects).isdisjoint(set(test_subjects))

    return train_idx, test_idx
