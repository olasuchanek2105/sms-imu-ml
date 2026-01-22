import pandas as pd
import numpy as np
import re

"""
Moduł pomocniczy do wczytywania i wstępnego przygotowania danych
wykorzystywanych w projekcie oceny ryzyka wystąpienia choroby kosmicznej (SMS).

Plik zawiera funkcje odpowiedzialne za:
- wczytywanie plików CSV z wyekstrahowanymi cechami,
- wczytywanie surowych danych IMU z automatycznym wykrywaniem separatora
  oraz znaku dziesiętnego,
- wczytywanie plików z wynikami ankiet (ankieta_score) i etykietą docelową (target),
- normalizację nazw kolumn i konwersję danych do postaci numerycznej,
- wyodrębnienie identyfikatora badanego (subject) na podstawie nazw plików,
- tworzenie wspólnej nazwy bazowej pliku (file_base), umożliwiającej
  jednoznaczne dopasowanie danych RAW i FILT w dalszych etapach przetwarzania.

Moduł stanowi warstwę wejściową potoku przetwarzania danych i jest wykorzystywany
zarówno na etapie budowy zbiorów cech, jak i podczas trenowania oraz ewaluacji
modeli uczenia maszynowego.

Założenia:
- dane wejściowe mogą pochodzić z różnych źródeł i mieć różne formaty CSV,
- nazwy plików zawierają informację o badanym (subject),
- dane są przygotowywane do dalszego przetwarzania z użyciem pandas
  i bibliotek uczenia maszynowego.

Moduł nie zawiera logiki modelowania ani ewaluacji — odpowiada wyłącznie
za poprawne i spójne przygotowanie danych wejściowych.
"""


def load_and_prepare(path):
    df = pd.read_csv(path, sep=';')

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.', regex=False)

    df = df.apply(pd.to_numeric, errors='ignore')

    df["subject"] = (
        df["file"]
        .astype(str)
        .str.lower()
        .str.replace(".csv", "", regex=False)
        .str.split("_")
        .str[:2]
        .str.join("_")
    )

    df["file_base"] = (
        df["file"]
        .astype(str)
        .str.lower()
        .str.replace(".csv", "", regex=False)
        .str.replace("_filtered", "", regex=False)
    )

    return df




def read_imu_csv(full_path):
    """
    Heurystyczny odczyt plików IMU.
    Automatycznie wykrywa separator i znak dziesiętny.
    """
    with open(full_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
        first_line = f.readline()

    semis = first_line.count(';')
    commas = first_line.count(',')
    sep = ';' if semis > commas else ','

    decimal = None
    if re.search(r'\d,\d', first_line) and sep == ';':
        decimal = ','

    kw = dict(sep=sep, encoding='utf-8-sig', na_values=['', 'NaN', 'nan'])
    if decimal is not None and sep != decimal:
        kw['decimal'] = decimal

    df = pd.read_csv(full_path, **kw)
    df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip()
    return df


def read_targets(path):
    """
    Wczytuje plik z ankieta_score i target.
    Obsługuje polskie CSV oraz fallback do heurystyki IMU.
    """
    try:
        df = pd.read_csv(path, sep=';', decimal=',', encoding='utf-8-sig')
    except Exception:
        df = read_imu_csv(path)

    df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip()

    if "file" not in df.columns:
        raise ValueError(
            f"Brak kolumny 'file' w targetach. Mam: {list(df.columns)}"
        )

    df["file"] = df["file"].astype(str).str.strip()

    for c in df.columns:
        if c != "file":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df
