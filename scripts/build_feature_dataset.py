"""
build_feature_dataset.py
========================

Skrypt odpowiedzialny za budowę zbioru cech wykorzystywanego
w procesie uczenia modeli klasyfikacyjnych ryzyka wystąpienia
choroby kosmicznej (Space Motion Sickness, SMS).

Działanie skryptu obejmuje następujące etapy:
1. Wczytanie surowych plików IMU  zawierających sygnały
   prędkości kątowej żyroskopu w osiach X, Y oraz Z.
2. Segmentację sygnałów na okna czasowe o długości 5 sekund
   bez nakładania.
3. Ekstrakcję cech czasowych oraz widmowych dla każdej osi
   oraz modułu wektora prędkości kątowej.
4. Dołączenie wartości ankieta_score oraz binarnej etykiety
   klasy (target) na podstawie zewnętrznego pliku z danymi
   referencyjnymi.
5. Zapis ujednoliconego zbioru cech do jednego pliku CSV,
   wykorzystywanego w dalszych etapach analizy i uczenia modeli.

Skrypt stanowi pierwszy etap potoku przetwarzania danych
i jest wykorzystywany zarówno dla danych nieprzefiltrowanych,
jak i przefiltrowanych w kolejnych wariantach eksperymentów.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from utils.io import read_imu_csv, read_targets
from utils.signal import (
    X_ALIASES, Y_ALIASES, Z_ALIASES,
    find_signal_column
)
from feature_extraction.features import (
    zero_crossings, rms, energy, spectral_features
)

# ================== KONFIGURACJA ==================
# odkomentuj jedną z opcji

# dla cech filtrowanych
FOLDER_DANYCH = r"data/cut_filtered"
PLIK_TARGETY = r"data/ankieta_score_and_target_filtered.csv"

# # #dla cech niefiltrowanych
# FOLDER_DANYCH = r"data/cut_to_same_length"
# PLIK_TARGETY = r"data/ankieta_score_and_target.csv"

FS = 100                  # Hz
WINDOW_SEC = 5            # sekundy
WINDOW = FS * WINDOW_SEC
OVERLAP = 0               # brak nakładania

WYJSCIE = "TESTfeatures_5s_windows_not_filtered_binary_problemTEST.csv"

APPLY_LOG = True
LOG_COLS = ["x_energy", "y_energy", "z_energy", "mag_energy"]

# ================== GŁÓWNY PRZEPŁYW ==================
def main():
    """Build the configured feature dataset from all source CSV files."""
    targets_df = read_targets(PLIK_TARGETY)
    validate_targets(targets_df)

    file2target = build_file_target_map(targets_df)
    print_known_inputs(file2target, FOLDER_DANYCH)

    all_rows = []
    counters = {
        "pliki_ok": 0,
        "pliki_puste": 0,
        "pliki_bez_targetu": 0,
        "okienek": 0,
    }

    step = get_window_step(WINDOW, OVERLAP)

    for filename in os.listdir(FOLDER_DANYCH):
        if not filename.endswith(".csv"):
            continue

        rows, status, window_count = process_file(filename, file2target, step)
        all_rows.extend(rows)
        counters[status] += 1
        counters["okienek"] += window_count

    save_dataset(all_rows, counters)


def validate_targets(targets_df):
    """Validate that the target table has required label columns."""
    if "target" not in targets_df.columns or "ankieta_score" not in targets_df.columns:
        raise ValueError(f"Brak kolumn 'target' lub 'ankieta_score' w {PLIK_TARGETY}. Mam: {list(targets_df.columns)}")


def build_file_target_map(targets_df):
    """Create a filename-to-target lookup from the target table."""
    file2target = {}
    for _, row in targets_df.iterrows():
        fname = str(row["file"]).strip()
        if pd.isna(row.get("target", np.nan)):
            continue
        file2target[fname] = {
            "ankieta_score": float(row.get("ankieta_score", np.nan)) if not pd.isna(row.get("ankieta_score", np.nan)) else np.nan,
                "target": int(row["target"])
        }

    return file2target


def print_known_inputs(file2target, folder_path):
    """Print target filenames and available CSV files for debugging."""
    print("\n=== ZAWARTOŚĆ KOLUMN 'file' Z CSV (PO POPRAWCE) ===")
    for k in file2target.keys():
        print(repr(k))

    print("\n=== PLIKI W FOLDERZE ===")
    for name in os.listdir(folder_path):
        if name.endswith(".csv"):
            print(repr(name))


def get_window_step(window, overlap):
    """Return the sample step used to move between consecutive windows."""
    return window - overlap if window - overlap > 0 else window


def find_target_match(filename, file2target):
    """Find a target entry matching a CSV filename case-insensitively."""
    for k in file2target.keys():
        if k.lower().replace(".csv", "") == filename.lower().replace(".csv", ""):
            return k
    return None


def process_file(filename, file2target, step):
    """Read one IMU file and return extracted feature rows with status."""
    print("Przetwarzam:", filename)

    match = find_target_match(filename, file2target)
    if not match:
        print("  ⚠ brak targetu dla", filename, "— pomijam")
        return [], "pliki_bez_targetu", 0

    target_info = file2target[match]
    print(f"  ✓ dopasowano do: {match}")

    full_path = os.path.join(FOLDER_DANYCH, filename)
    df = read_imu_csv(full_path)

    x_name = find_signal_column(df, X_ALIASES)
    y_name = find_signal_column(df, Y_ALIASES)
    z_name = find_signal_column(df, Z_ALIASES)

    if not x_name or not y_name or not z_name:
        print(f"  ❌ Nie znalazłam kolumn x/y/z. Mam: {list(df.columns)}")
        return [], "pliki_puste", 0

    for col in [x_name, y_name, z_name]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    imu_df = pd.DataFrame({
        "x": df[x_name],
        "y": df[y_name],
        "z": df[z_name]
    })

    before = len(imu_df)
    nnx, nny, nnz = imu_df["x"].notna().sum(), imu_df["y"].notna().sum(), imu_df["z"].notna().sum()
    print(f"  nie-NaN count: x={nnx}, y={nny}, z={nnz} (z {before})")

    imu_df = imu_df.dropna(subset=["x", "y", "z"])
    after = len(imu_df)
    if after < before:
        print(f"  usunięto {before - after} wierszy z NaN (pozostało {after})")

    if after == 0:
        print("  ❗ Po czyszczeniu brak danych (x/y/z całe NaN). Pomijam plik.")
        return [], "pliki_puste", 0

    x_sig = imu_df["x"].to_numpy(dtype=float)
    y_sig = imu_df["y"].to_numpy(dtype=float)
    z_sig = imu_df["z"].to_numpy(dtype=float)

    signal_length = len(x_sig)
    print(f"  długość sygnału po czyszczeniu: {signal_length}")

    if signal_length < WINDOW:
        print("  ℹ Za mało próbek na choć jedno okno — pomijam.")
        return [], "pliki_puste", 0

    rows = build_feature_rows(filename, x_sig, y_sig, z_sig, target_info, step)
    print(f"  ✔ wygenerowano okien: {len(rows)}")

    return rows, "pliki_ok", len(rows)


def build_feature_rows(filename, x_sig, y_sig, z_sig, target_info, step):
    """Slice signals into windows and build feature rows with labels."""
    rows = []
    start = 0

    while start + WINDOW <= len(x_sig):
        end = start + WINDOW

        row = extract_window_features(
            filename,
            x_sig[start:end],
            y_sig[start:end],
            z_sig[start:end]
        )
        row["ankieta_score"] = target_info["ankieta_score"]
        row["target"] = target_info["target"]
        rows.append(row)

        start += step

    return rows


def extract_window_features(filename, xw, yw, zw):
    """Extract time-domain and spectral features for one signal window."""
    mag = np.sqrt(xw**2 + yw**2 + zw**2)
    x_peaks, _ = find_peaks(xw)
    y_peaks, _ = find_peaks(yw)

    row = {
        "file": filename,

        "x_mean": float(np.mean(xw)),
        "x_std": float(np.std(xw)),
        "x_rms": rms(xw),
        "x_energy": energy(xw),
        "x_zero_crossings": int(zero_crossings(xw)),
        "x_count_peaks": int(len(x_peaks)),

        "y_mean": float(np.mean(yw)),
        "y_std": float(np.std(yw)),
        "y_rms": rms(yw),
        "y_energy": energy(yw),
        "y_zero_crossings": int(zero_crossings(yw)),
        "y_count_peaks": int(len(y_peaks)),

        "z_mean": float(np.mean(zw)),
        "z_std": float(np.std(zw)),
        "z_rms": rms(zw),
        "z_energy": energy(zw),

        "mag_rms": rms(mag),
        "mag_energy": energy(mag),
    }

    row.update(spectral_features(mag, FS))
    apply_log_transform(row)

    return row


def apply_log_transform(row):
    """Apply the configured log transform to selected energy features."""
    if not APPLY_LOG:
        return

    for c in LOG_COLS:
        if c in row:
            val = max(float(row[c]), 0.0)
            row[c] = float(np.log1p(val))


def save_dataset(all_rows, counters):
    """Write the feature dataset and print a processing summary."""
    big_df = pd.DataFrame(all_rows)
    big_df.to_csv(WYJSCIE, index=False, sep=';')
    print(f"\nZapisano {len(big_df)} wierszy do {WYJSCIE}")
    print(
        "Podsumowanie: "
        f"OK={counters['pliki_ok']}, "
        f"puste/bez danych={counters['pliki_puste']}, "
        f"bez targetu={counters['pliki_bez_targetu']}, "
        f"okien łącznie={counters['okienek']}"
    )


if __name__ == "__main__":
    main()
