from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


DATA_PATH = Path("data/processed/data_clean.csv")
MODEL_PATH = Path("models/final_model_pipeline.pkl")
METRICS_PATH = Path("reports/metrics.json")


def main() -> None:
    """Evalúa el modelo final y guarda métricas en un JSON."""
    df = pd.read_csv(DATA_PATH)

    y = df["Unidades"]
    X = df.drop(columns=["Unidades"])

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X)

    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    metrics = {"rmse": rmse}

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"RMSE en todo el dataset: {rmse:.4f}")
    print(f"Métricas guardadas en {METRICS_PATH}")


if __name__ == "__main__":
    main()
