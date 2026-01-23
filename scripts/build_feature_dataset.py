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

print(">>> FILE LOADED <<<")

# ================== IMPORTY STANDARDOWE ==================
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ================== IMPORTY Z PROJEKTU ==================
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
    # 1) Wczytaj targety
    targets_df = read_targets(PLIK_TARGETY)

    # walidacja kolumn docelowych
    if "target" not in targets_df.columns or "ankieta_score" not in targets_df.columns:
        raise ValueError(f"Brak kolumn 'target' lub 'ankieta_score' w {PLIK_TARGETY}. Mam: {list(targets_df.columns)}")

    # słownik file -> {ankieta_score, target}
    file2target = {}
    for _, row in targets_df.iterrows():
        fname = str(row["file"]).strip()
        if pd.isna(row.get("target", np.nan)):
            continue
        file2target[fname] = {
            "ankieta_score": float(row.get("ankieta_score", np.nan)) if not pd.isna(row.get("ankieta_score", np.nan)) else np.nan,
            "target": int(row["target"])
        }

    print("\n=== ZAWARTOŚĆ KOLUMN 'file' Z CSV (PO POPRAWCE) ===")
    for k in file2target.keys():
        print(repr(k))

    print("\n=== PLIKI W FOLDERZE ===")
    for name in os.listdir(FOLDER_DANYCH):
        if name.endswith(".csv"):
            print(repr(name))

    all_rows = []
    pliki_ok = 0
    pliki_puste = 0
    pliki_bez_targetu = 0
    okienek = 0

    step = WINDOW - OVERLAP if WINDOW - OVERLAP > 0 else WINDOW

    for filename in os.listdir(FOLDER_DANYCH):
        if not filename.endswith(".csv"):
            continue

        print("Przetwarzam:", filename)

        # dopasuj do targetów (case-insensitive, bez .csv)
        match = None
        for k in file2target.keys():
            if k.lower().replace(".csv", "") == filename.lower().replace(".csv", ""):
                match = k
                break

        if not match:
            print("  ⚠ brak targetu dla", filename, "— pomijam")
            pliki_bez_targetu += 1
            continue

        target_info = file2target[match]
        print(f"  ✓ dopasowano do: {match}")

        full_path = os.path.join(FOLDER_DANYCH, filename)
        df = read_imu_csv(full_path)

        # znajdź kolumny osi
        x_name = find_signal_column(df, X_ALIASES)
        y_name = find_signal_column(df, Y_ALIASES)
        z_name = find_signal_column(df, Z_ALIASES)

        if not x_name or not y_name or not z_name:
            print(f"  ❌ Nie znalazłam kolumn x/y/z. Mam: {list(df.columns)}")
            pliki_puste += 1
            continue

        # rzutuj na numeric
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
            pliki_puste += 1
            continue

        x_sig = imu_df["x"].to_numpy(dtype=float)
        y_sig = imu_df["y"].to_numpy(dtype=float)
        z_sig = imu_df["z"].to_numpy(dtype=float)

        N = len(x_sig)
        print(f"  długość sygnału po czyszczeniu: {N}")

        if N < WINDOW:
            print("  ℹ Za mało próbek na choć jedno okno — pomijam.")
            pliki_puste += 1
            continue

        # okienkowanie
        start = 0
        licznik_okien = 0
        while start + WINDOW <= N:
            end = start + WINDOW

            xw = x_sig[start:end]
            yw = y_sig[start:end]
            zw = z_sig[start:end]

            mag = np.sqrt(xw**2 + yw**2 + zw**2)
            x_peaks, _ = find_peaks(xw)
            y_peaks, _ = find_peaks(yw)

            spec = spectral_features(mag, FS)

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

            row.update(spec)

            if APPLY_LOG:
                for c in LOG_COLS:
                    if c in row:
                        # bezpiecznie dla zer/małych wartości
                        val = max(float(row[c]), 0.0)
                        row[c] = float(np.log1p(val))
                        
            row["ankieta_score"] = target_info["ankieta_score"]
            row["target"] = target_info["target"]

            all_rows.append(row)
            licznik_okien += 1
            okienek += 1

            start += (WINDOW - OVERLAP) if (WINDOW - OVERLAP) > 0 else WINDOW

        print(f"  ✔ wygenerowano okien: {licznik_okien}")
        pliki_ok += 1

    # 5) Zapis
    big_df = pd.DataFrame(all_rows)
    big_df.to_csv(WYJSCIE, index=False, sep=';')
    print(f"\nZapisano {len(big_df)} wierszy do {WYJSCIE}")
    print(f"Podsumowanie: OK={pliki_ok}, puste/bez danych={pliki_puste}, bez targetu={pliki_bez_targetu}, okien łącznie={okienek}")



if __name__ == "__main__":
    main()
