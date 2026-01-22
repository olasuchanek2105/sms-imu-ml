


def prepare_feature_sets(
    df_raw,
    df_filt,
    y,
    groups,
    train_idx,
    test_idx
):
    """
    Przygotowuje zestawy cech wejściowych dla modeli uczenia maszynowego
    na podstawie danych surowych (RAW) oraz przefiltrowanych (FILT).

    Funkcja:
    - usuwa kolumny nienależące do przestrzeni cech (np. metadane plików,
      identyfikatory badanych, etykiety klas),
    - rozdziela dane na podzbiory treningowe i testowe zgodnie
      z wcześniej wyznaczonymi indeksami,
    - zachowuje spójność indeksów pomiędzy cechami, etykietami oraz
      informacją o grupach (subjects).

    Parametry
    ---------
    df_raw : pandas.DataFrame
        Ramka danych zawierająca cechy wyekstrahowane z sygnału
        nieprzefiltrowanego (RAW).
    df_filt : pandas.DataFrame
        Ramka danych zawierająca cechy wyekstrahowane z sygnału
        przefiltrowanego (FILT).
    y : pandas.Series
        Wektor etykiet klas (target).
    groups : pandas.Series
        Wektor identyfikatorów grup (np. badanych), wykorzystywany
        w walidacji krzyżowej typu GroupKFold.
    train_idx : pandas.Index
        Indeksy próbek należących do zbioru treningowego.
    test_idx : pandas.Index
        Indeksy próbek należących do zbioru testowego.

    Zwraca
    -------
    dict
        Słownik zawierający:
        - X_raw_train : cechy RAW – zbiór treningowy,
        - X_raw_test  : cechy RAW – zbiór testowy,
        - X_filt_train : cechy FILT – zbiór treningowy,
        - X_filt_test  : cechy FILT – zbiór testowy,
        - y_train : etykiety klas dla zbioru treningowego,
        - y_test  : etykiety klas dla zbioru testowego,
        - groups_train : identyfikatory grup dla zbioru treningowego.
    """
    cols_drop = [
        "file", "file_base", "subject",
        "ankieta_score", "target", "window_idx"
    ]

    X_raw  = df_raw.drop(columns=cols_drop, errors="ignore")
    X_filt = df_filt.drop(columns=cols_drop, errors="ignore")

    return {
        "X_raw_train":  X_raw.loc[train_idx],
        "X_raw_test":   X_raw.loc[test_idx],
        "X_filt_train": X_filt.loc[train_idx],
        "X_filt_test":  X_filt.loc[test_idx],
        "y_train": y.loc[train_idx],
        "y_test":  y.loc[test_idx],
        "groups_train": groups.loc[train_idx],
    }
