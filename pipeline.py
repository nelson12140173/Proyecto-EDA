"""
Pipeline completo de procesamiento MLOps.
"""

import logging
from datetime import datetime
from pathlib import Path

from src.eda import run_eda
from src.feature_exploration import run_feature_exploration
from src.feature_creation import create_features
from src.model_training import train_and_tune
from src.inference import run_inference


def configure_logging() -> None:
    """Configura logging en archivo y consola."""
    logs_dir = Path("reports")
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
    )


def main() -> None:
    """Ejecuta el pipeline end-to-end."""
    logging.info("🚀 Inicio del pipeline MLOps")

    try:
        logging.info("1️⃣ EDA y carga de datos base")
        df_data = run_eda(Path("data") / "raw" / "Base.xlsx")

        logging.info("2️⃣ Exploración de características")
        df_data = run_feature_exploration(df_data)

        logging.info("3️⃣ Creación de features")
        features, target = create_features(df_data)

        logging.info("4️⃣ Entrenamiento y tuning del modelo")
        model, metrics = train_and_tune(features, target)
        logging.info("📊 Métricas finales: %s", metrics)

        logging.info("5️⃣ Inferencias sobre dataset de testing")
        preds_path = run_inference(
            model=model,
            test_path=Path("data") / "raw" / "Base_Test.xlsx",
            output_path=Path("reports") / "predicciones_testing.csv",
        )
        logging.info("Predicciones guardadas en: %s", preds_path)

        logging.info("🎉 Pipeline ejecutado exitosamente")

    except Exception as exc:  # pylint: disable=broad-except
        logging.error("❌ Error durante la ejecución del pipeline: %s", exc)
        raise exc


if __name__ == "__main__":
    configure_logging()
    main()
    