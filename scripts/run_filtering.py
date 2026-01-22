
"""
run_filtering.py
================

Skrypt realizujący etap wstępnego przetwarzania sygnałów
żyroskopowych poprzez filtrację dolnoprzepustową typu Butterwortha.

Działanie skryptu obejmuje:
1. Iteracyjne wczytanie surowych plików CSV z folderu danych wejściowych,
   zawierających sygnały IMU (żyroskop) po etapie przycięcia czasowego (CUT).
2. Automatyczne wyznaczenie częstotliwości próbkowania na podstawie
   wektora czasu zawartego w danych.
3. Oszacowanie optymalnej częstotliwości odcięcia filtra dolnoprzepustowego
   metodą analizy widmowej (Welcha) osobno dla każdego pliku.
4. Filtrację sygnałów żyroskopowych w osiach X, Y oraz Z z użyciem
   filtra Butterwortha bez przesunięcia fazowego (filtfilt).
5. Zapis przefiltrowanych sygnałów do nowego pliku CSV z zachowaniem
   oryginalnej struktury danych oraz dodaniem kolumn z sygnałem filtrowanym.

Wynikiem działania skryptu jest zestaw plików CSV zawierających
przefiltrowane sygnały IMU, wykorzystywane następnie w procesie
ekstrakcji cech oraz uczenia modeli klasyfikacyjnych.
"""
import os
import pandas as pd
from preprocessing.butterworth_filter import filter_gyro_dataframe


input_folder = "data/raw_cut"
output_folder = "data/filtered"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(input_folder, filename))
    df_filt, fs = filter_gyro_dataframe(df)

    out = os.path.join(
        output_folder,
        filename.replace(".csv", "_filtered.csv")
    )
    df_filt.to_csv(out, index=False)
