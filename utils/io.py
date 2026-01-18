import pandas as pd

def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';')

    # zamiana przecinków na kropki w kolumnach tekstowych
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.', regex=False)

    # konwersja wszystkiego co się da na liczby
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass


    # subject z nazwy pliku: np. "P_AD_BezGogli_cut.csv" -> "p_ad"
    df["subject"] = (
        df["file"]
        .astype(str)
        .str.lower()
        .str.replace(".csv", "", regex=False)
        .str.split("_")
        .str[:2]
        .str.join("_")
    )

    # wspólna baza nazwy pliku dla RAW i FILT
    # np. "P_AD_BezGogli_cut.csv"          -> "p_ad_bezgogli_cut"
    #     "P_AD_BezGogli_cut_filtered.csv" -> "p_ad_bezgogli_cut"
    df["file_base"] = (
        df["file"]
        .astype(str)
        .str.lower()
        .str.replace(".csv", "", regex=False)
        .str.replace("_filtered", "", regex=False)
    )

    return df
