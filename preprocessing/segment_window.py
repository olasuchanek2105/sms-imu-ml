"""
window_alignment.py
===================

Moduł odpowiedzialny za synchronizację oraz wyrównanie okien
czasowych pomiędzy dwoma zestawami cech obliczonych na podstawie
sygnałów nieprzefiltrowanych (RAW) oraz przefiltrowanych (FILT).

Celem modułu jest:
- jednoznaczne dopasowanie okien czasowych pomiędzy zbiorami RAW i FILT,
- zapewnienie zgodności indeksów próbek wykorzystywanych w dalszych
  etapach przetwarzania danych,
- kontrola spójności etykiet klas pomiędzy zestawami cech.

Dla każdego pliku wejściowego generowany jest indeks okna
(window_idx), który identyfikuje kolejne okna czasowe w obrębie
tej samej sekwencji pomiarowej (file_base).

Następnie oba zbiory danych są wyrównywane na podstawie pary
(file_base, window_idx), co gwarantuje relację jeden-do-jednego
pomiędzy cechami RAW i FILT.

Moduł zawiera mechanizmy walidacyjne zapobiegające niespójnościom
danych, w tym:
- weryfikację zgodności liczby próbek,
- sprawdzenie identyczności etykiet klas po wyrównaniu.

Poprawne wyrównanie okien jest kluczowe dla implementacji
architektury dwuetapowej (stacking), w której cechy z obu
reprezentacji sygnału muszą odnosić się do tych samych fragmentów
czasowych.
"""


def add_window_index_and_align(df_raw, df_filt):
    for df in (df_raw, df_filt):
        df["window_idx"] = df.groupby("file_base").cumcount()

    df_raw_idx  = df_raw.set_index(["file_base", "window_idx"])
    df_filt_idx = df_filt.set_index(["file_base", "window_idx"])

    df_filt = df_filt_idx.loc[df_raw_idx.index].reset_index()
    df_raw  = df_raw_idx.reset_index()

    assert len(df_raw) == len(df_filt)
    assert (df_raw["target"].values == df_filt["target"].values).all()

    return df_raw, df_filt
