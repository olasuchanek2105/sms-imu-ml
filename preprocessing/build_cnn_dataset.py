import os
import numpy as np
import pandas as pd


def build_cnn_dataset(
    raw_folder_path,
    subject_target_mapping,
    window_size_sec=5,
    fs=100
):
    samples_per_window = window_size_sec * fs

    windows = []
    labels = []
    subjects = []

    skipped_no_target = 0
    skipped_too_short = 0

    for filename in os.listdir(raw_folder_path):
        if not filename.endswith(".csv"):
            continue

        subject_id = filename.replace(".csv", "").strip()

        # ✅ Skipujemy pliki, które nie mają targetu (np. pomiar fotela)
        target = subject_target_mapping.get(subject_id)
        if target is None:
            skipped_no_target += 1
            print(f"[SKIP] Brak targetu dla: {subject_id} ({filename})")
            continue

        df = pd.read_csv(os.path.join(raw_folder_path, filename))

        required_cols = [
            "Gyroscope x (rad/s)",
            "Gyroscope y (rad/s)",
            "Gyroscope z (rad/s)",
            "Absolute (rad/s)"
        ]

        # (opcjonalnie) jeśli kiedyś trafi się plik z innymi nagłówkami
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"[SKIP] Brak kolumn {missing} w pliku: {filename}")
            continue

        signal = df[required_cols].values

        n_windows = len(signal) // samples_per_window
        if n_windows == 0:
            skipped_too_short += 1
            print(f"[SKIP] Za krótki sygnał (<{samples_per_window} próbek): {filename}")
            continue

        for i in range(n_windows):
            start = i * samples_per_window
            end = start + samples_per_window

            window = signal[start:end]  # shape: (500, 4)

            windows.append(window)
            labels.append(int(target))
            person_id = extract_person_id(subject_id)
            subjects.append(person_id)

    X = np.array(windows, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    groups = np.array(subjects)

    print(f"Zbudowano dataset: X={X.shape}, y={y.shape}, subjects={len(np.unique(groups))}")
    print(f"Pominięto: bez targetu={skipped_no_target}, za krótkie={skipped_too_short}")

    return X, y, groups



def extract_person_id(subject_id: str) -> str:
    """
    Zamienia nazwę pliku/nagrania na ID osoby.
    Przykład: 'P_AD_BezGogli' -> 'P_AD'
    """
    parts = subject_id.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])  # 'P' + 'AD' => 'P_AD'
    return subject_id

def load_subject_target_mapping(target_csv_path):
    df = pd.read_csv(target_csv_path, sep=";")
    df.columns = df.columns.str.strip()

    # Usuwamy rozszerzenie .csv z kolumny file
    df["subject_id"] = df["file"].str.replace(".csv", "", regex=False)

    # Tworzymy słownik
    
    mapping = dict(zip(df["subject_id"], df["target"]))

    return mapping