"""
EDA básico del proyecto.
"""

from pathlib import Path
from typing import Union

import pandas as pd


def run_eda(path: Union[str, Path]) -> pd.DataFrame:
    """
    Carga dataset y crea columna Fecha basada en Año + Semana.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {path}")

    df = pd.read_excel(path)

    # Crear fecha: Año + Semana
    if {"Año", "Semana"}.issubset(df.columns):
        df["Fecha"] = pd.to_datetime(
            df["Año"].astype(str) + "-W" + df["Semana"].astype(str) + "-1",
            format="%G-W%V-%u",
        )

    df = df.sort_values("Fecha").reset_index(drop=True)

    return df
    