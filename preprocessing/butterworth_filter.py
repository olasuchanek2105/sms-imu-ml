
"""
butterworth_filter.py
=====================

Moduł odpowiedzialny za filtrację sygnałów żyroskopowych
z wykorzystaniem cyfrowego filtra dolnoprzepustowego Butterwortha.

Celem filtracji jest redukcja składowych wysokoczęstotliwościowych,
pochodzących m.in. od szumu pomiarowego oraz artefaktów dynamicznych,
przy jednoczesnym zachowaniu istotnych informacji o ruchu głowy.

W module zaimplementowano:
- projekt filtra dolnoprzepustowego Butterwortha,
- automatyczne oszacowanie częstotliwości odcięcia na podstawie
  analizy widmowej sygnału (metoda Welcha),
- filtrację sygnałów w osiach X, Y oraz Z z wykorzystaniem
  filtracji dwukierunkowej (filtfilt), eliminującej przesunięcie fazowe,
- obliczenie efektywnej częstotliwości próbkowania sygnału
  na podstawie wektora czasu.

Częstotliwość odcięcia filtra dobierana jest adaptacyjnie
jako minimum wartości bazowej oraz wielokrotności częstotliwości,
poniżej której zawarte jest 95% energii widma sygnału.
Takie podejście umożliwia dostosowanie parametrów filtracji
do charakterystyki konkretnego pomiaru.

Moduł stanowi element etapu wstępnego przetwarzania sygnałów
i wykorzystywany jest przed ekstrakcją cech oraz uczeniem modeli
klasyfikacyjnych.
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch

def butter_lowpass(fs, cutoff, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return b, a


def estimate_cutoff_welch(signal, fs, base_cutoff=5.0):
    f, Pxx = welch(signal, fs=fs, nperseg=min(2048, len(signal)))
    cum = np.cumsum(Pxx) / np.sum(Pxx)
    f95 = f[np.searchsorted(cum, 0.95)]
    return min(base_cutoff, 1.2 * f95)


def filter_gyro_dataframe(df, base_cutoff=5.0, order=4):
    time_col = [c for c in df.columns if "time" in c.lower()][0]
    time = df[time_col].to_numpy()
    time = time[~np.isnan(time)]

    fs = 1.0 / np.mean(np.diff(time))

    for axis in ["x", "y", "z"]:
        sig = df[f"Gyroscope {axis} (rad/s)"].to_numpy()
        sig = sig[~np.isnan(sig)]

        cutoff = estimate_cutoff_welch(sig, fs, base_cutoff)
        b, a = butter_lowpass(fs, cutoff, order)

        df[f"Gyro_{axis}_filtered"] = filtfilt(b, a, df[f"Gyroscope {axis} (rad/s)"])

    return df, fs
