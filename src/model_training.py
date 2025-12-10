"""
Entrenamiento y tuning de modelos.

Basado en 04_model_training_n_tunning.ipynb:
- Define preprocesador (numérico + categórico)
- Evalúa varios modelos (LinearRegression, Ridge, Lasso,
  RandomForest, GradientBoosting) con GridSearchCV
- Selecciona el mejor por RMSE de validación
- Entrena modelo final con todos los datos
- Guarda el pipeline en models/final_model_pipeline.pkl
"""

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _build_preprocessor(
    X: pd.DataFrame,
) -> Tuple[ColumnTransformer, pd.Index, pd.Index]:
    """Construye el preprocesador (numérico + categórico)."""
    num_features = X.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X.select_dtypes(include=["object", "category"]).columns

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ],
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ],
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_features),
            ("cat", categorical_pipeline, cat_features),
        ],
    )

    return preprocessor, num_features, cat_features


def train_and_tune(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Entrena y ajusta varios modelos y selecciona el mejor.

    Parameters
    ----------
    X : pd.DataFrame
        Variables predictoras.
    y : pd.Series
        Variable objetivo.

    Returns
    -------
    Tuple[Pipeline, Dict[str, float]]
        Modelo campeón (pipeline completo) y métricas.
    """
    preprocessor, _, _ = _build_preprocessor(X)

    # Usamos una partición tipo "time series": primeros 80% entrenan, 20% validan
    split_index = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

    modelos = {
        "LinearRegression": {
            "model": LinearRegression(),
            "params": {},
        },
        "Ridge": {
            "model": Ridge(),
            "params": {"model__alpha": [0.1, 1, 10]},
        },
        "Lasso": {
            "model": Lasso(),
            "params": {"model__alpha": [0.01, 0.1, 1]},
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [5, 10, None],
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
            },
        },
    }

    resultados = []
    tscv = TimeSeriesSplit(n_splits=3)

    for nombre, config in modelos.items():
        print(f"\n=== Entrenando modelo: {nombre} ===")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", config["model"]),
            ],
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=config["params"],
            scoring="neg_root_mean_squared_error",
            cv=tscv,
            n_jobs=-1,
        )

        grid.fit(X_train, y_train)

        best_rmse = -grid.best_score_  
        val_pred = grid.predict(X_val)
        rmse_val = float(
            np.sqrt(mean_squared_error(y_val, val_pred)),
        )

        print(f"Mejor RMSE CV   : {best_rmse:.4f}")
        print(f"RMSE validación : {rmse_val:.4f}")

        resultados.append(
            {
                "modelo": nombre,
                "rmse_cv": float(best_rmse),
                "rmse_validacion": rmse_val,
                "best_estimator": grid.best_estimator_,
            },
        )

    # Seleccionar el modelo con menor RMSE de validación
    resultados_ordenados = sorted(
        resultados,
        key=lambda r: r["rmse_validacion"],
    )
    ganador = resultados_ordenados[0]
    best_model = ganador["best_estimator"]

    print("\n====================================")
    print(f"Modelo ganador: {ganador['modelo']}")
    print(f"RMSE validación: {ganador['rmse_validacion']}")
    print("====================================")

    # Entrenar modelo campeón con TODOS los datos
    best_model.fit(X, y)

    metrics = {
        "rmse_validacion": ganador["rmse_validacion"],
        "rmse_cv": ganador["rmse_cv"],
    }

    # Guardar el modelo final
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    output_path = models_dir / "final_model_pipeline.pkl"
    joblib.dump(best_model, output_path)

    print("\nModelo FINAL guardado en models/final_model_pipeline.pkl")

    return best_model, metrics


if __name__ == "__main__":
    """Punto de entrada del script de entrenamiento."""
    df = pd.read_csv("data/processed/data_clean.csv")

    y = df["Unidades"]
    X = df.drop(columns=["Unidades"])

    model, metrics = train_and_tune(X, y)

    print("\nEntrenamiento completado.")
    print("Métricas:", metrics)
    print("Modelo guardado en models/final_model_pipeline.pkl")
