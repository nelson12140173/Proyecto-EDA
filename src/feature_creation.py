"""
Creación de características y preprocesamiento.

Este módulo prepara X e y usando la columna 'Unidades'
como variable objetivo (target).
"""

from typing import Tuple
import pandas as pd


def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepara X e y para el modelo usando 'Unidades' como target.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con la columna objetivo 'Unidades'.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        X: variables predictoras
        y: variable objetivo.
    """

    target_col = "Unidades"

    if target_col not in df.columns:
        raise ValueError(f"No se encontró la columna '{target_col}' en el dataset.")

    df = df.copy()

    # Si tu notebook hacía limpieza o conversiones aquí, se agregan
    # (de momento no aplican porque 'Unidades' ya es numérica)

    y = df[target_col]
    X = df.drop(columns=[target_col])

    return X, y
    