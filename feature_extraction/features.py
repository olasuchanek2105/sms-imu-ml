
"""
features.py
===========

Moduł odpowiedzialny za ekstrakcję cech czasowych i widmowych
z jednowymiarowych sygnałów inercyjnych (żyroskopowych).

Cechy wyznaczane w tym module są obliczane na krótkich
oknach czasowych sygnału i stanowią wejście dla modeli
uczenia maszynowego wykorzystywanych w projekcie.

Zakres funkcjonalności:
- cechy czasowe (RMS, energia, liczba przejść przez zero),
- cechy widmowe oparte na dyskretnej transformacie Fouriera (FFT),
- odporność na przypadki degeneracyjne (np. sygnał zerowy).

Moduł wykorzystywany jest w etapie:
- segmentacji sygnału na okna,
- budowy zbioru cech do klasyfikacji ryzyka SMS.
"""
import numpy as np
from scipy.fft import rfft, rfftfreq


def zero_crossings(sig):
    return ((sig[:-1] * sig[1:]) < 0).sum()


def rms(sig):
    return float(np.sqrt(np.mean(sig**2)))


def energy(sig):
    return float(np.sum(sig**2))


def spectral_features(sig, fs):
    N = len(sig)
    yf = np.abs(rfft(sig))
    xf = rfftfreq(N, 1/fs)

    if np.all(yf == 0) or np.sum(yf) == 0:
        return {
            "mag_spec_entropy": 0.0,
            "mag_spec_centroid_hz": 0.0,
            "mag_spec_bandwidth_hz": 0.0,
            "mag_spec_dom_freq_hz": 0.0,
        }

    dom_idx = int(np.argmax(yf))
    dom_freq = float(xf[dom_idx])
    centroid = float(np.sum(xf * yf) / np.sum(yf))
    bandwidth = float(np.sqrt(np.sum(((xf - centroid) ** 2) * yf) / np.sum(yf)))

    p = yf / np.sum(yf)
    entropy = float(-np.sum(p * np.log(p + 1e-12)))

    return {
        "mag_spec_entropy": entropy,
        "mag_spec_centroid_hz": centroid,
        "mag_spec_bandwidth_hz": bandwidth,
        "mag_spec_dom_freq_hz": dom_freq,
    }
