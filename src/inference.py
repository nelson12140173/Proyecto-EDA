"""
Cálculo de inferencias.

Basado en 05_inference_calculation.ipynb:
- Carga pipeline final entrenado
- Carga Base_Test.xlsx
- Genera predicciones y guarda un CSV
"""

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


def run_inference(
    model: Optional[Pipeline],
    test_path: Path,
    output_path: Path,
) -> Path:
    """
    Genera predicciones usando el modelo final.

    Parameters
    ----------
    model : Pipeline or None
        Modelo ya entrenado. Si es None, intentará cargar
        'models/final_model_pipeline.pkl'.
    test_path : Path
        Ruta al archivo de testing (Base_Test.xlsx).
    output_path : Path
        Ruta donde se guardará el CSV con predicciones.

    Returns
    -------
    Path
        Ruta al archivo CSV generado.
    """
    if model is None:
        model_file = Path("models") / "final_model_pipeline.pkl"
        if not model_file.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en: {model_file}",
            )
        model = joblib.load(model_file)

    if not test_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset de test: {test_path}")

    df_test = pd.read_excel(test_path)

    # Asegurarse de que la columna objetivo no esté presente
    if "Valores" in df_test.columns:
        df_test = df_test.drop(columns=["Valores"])

    preds = model.predict(df_test)

    df_pred = df_test.copy()
    df_pred["Prediccion_Valores"] = preds

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(output_path, index=False, encoding="utf-8-sig")

    return output_path
