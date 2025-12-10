from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/raw/Base.xlsx")
PROCESSED_DIR = Path("data/processed")
OUT_PATH = PROCESSED_DIR / "data_clean.csv"


def main() -> None:
    """Carga el Excel crudo y genera un CSV limpio para el modelo."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(RAW_PATH)

    df.to_csv(OUT_PATH, index=False)

    print(f"Archivo procesado guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
