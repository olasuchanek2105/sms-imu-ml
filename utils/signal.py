"""
Moduł narzędziowy do identyfikacji i normalizacji sygnałów żyroskopowych
w danych IMU.

Plik zawiera zestaw funkcji i stałych umożliwiających:
- normalizację nazw kolumn (usuwanie znaków BOM, białych znaków, ujednolicenie zapisu),
- obsługę wielu wariantów nazw kolumn sygnałów żyroskopu w osiach X, Y oraz Z,
- automatyczne wyszukiwanie właściwej kolumny sygnału na podstawie list aliasów,
  niezależnie od formatu pliku wejściowego lub konwencji nazewniczej.

Zdefiniowane listy aliasów (X_ALIASES, Y_ALIASES, Z_ALIASES) pozwalają na
elastyczną obsługę danych pochodzących z różnych źródeł, aplikacji pomiarowych
oraz etapów przetwarzania (sygnał surowy i przefiltrowany).

Moduł ten stanowi warstwę adaptacyjną pomiędzy surowymi plikami pomiarowymi
a dalszym potokiem przetwarzania sygnałów i ekstrakcji cech, zapewniając
odporność pipeline’u na różnice w strukturze danych wejściowych.
"""



def normalize(name: str) -> str:
    return name.replace('\ufeff', '').strip().lower()


X_ALIASES = [
    "Gyro_x_filtered", "gyro_x_filtered",
    "Gyroscope x (rad/s)", "gyro_x", "gyroscope x",
    "x", "gyr_x", "gx", "gyro x", "gyro-x"
]

Y_ALIASES = [
    "Gyro_y_filtered", "gyro_y_filtered",
    "Gyroscope y (rad/s)", "gyro_y", "gyroscope y",
    "y", "gyr_y", "gy", "gyro y", "gyro-y"
]

Z_ALIASES = [
    "Gyro_z_filtered", "gyro_z_filtered",
    "Gyroscope z (rad/s)", "gyro_z", "gyroscope z",
    "z", "gyr_z", "gz", "gyro z", "gyro-z"
]


def find_signal_column(df, aliases):
    """
    Znajduje kolumnę sygnału na podstawie listy aliasów.
    """
    norm2orig = {normalize(c): c for c in df.columns}

    # pełne dopasowania
    for a in aliases:
        a_norm = normalize(a)
        if a_norm in norm2orig:
            return norm2orig[a_norm]

    # dopasowania częściowe
    for a in aliases:
        a_norm = normalize(a)
        for n, orig in norm2orig.items():
            if a_norm in n:
                return orig

    return None
