import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
from datetime import datetime
import pandas as pd
import os

# ==========================================
# CONFIGURACIÓN DEL MODELO
# ==========================================

# Ruta local del modelo Champion
MODEL_PATH = r"C:/Users/52358965/Proyecto/notebooks/mlruns/models/ChampionModel/version-1/artifacts/Model"


# Columnas esperadas por el modelo
EXPECTED_COLUMNS = [
    "Año",
    "Sucursal",
    "Mes",
    "Semana",
    "Producto",
    "Valores",
    "Unidades"
]

# Cargar modelo
print(">>> Cargando modelo campeón desde ruta local...")
model = mlflow.sklearn.load_model(MODEL_PATH)
print(">>> Modelo cargado correctamente.")

# Crear app Flask
app = Flask(__name__)


# ==========================================
# FUNCIÓN DE VALIDACIÓN GENERAL
# ==========================================

def validate_input(data):
    """
    Valida que el JSON contenga TODAS las columnas esperadas.
    Retorna: (True, None) si todo está bien
             (False, {detalle del error}) si falta algo
    """
    # Detectar si es registro único o batch
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        return False, "Formato inválido. Debe ser un JSON o lista de JSON."

    # Verificar columnas faltantes
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]

    if missing:
        return False, f"Faltan las siguientes columnas requeridas: {missing}"

    # Ordenar columnas en el orden correcto
    df = df[EXPECTED_COLUMNS]

    return True, df


# ==========================================
# ENDPOINT PREDICCIÓN INDIVIDUAL
# ==========================================

@app.route("/predict_single", methods=["POST"])
def predict_single():
    data = request.json

    valid, result = validate_input(data)
    if not valid:
        return jsonify({"error": result}), 400

    df = result
    prediction = model.predict(df)[0]

    response = {
        "prediction": float(prediction),
        "timestamp": datetime.now().isoformat(),
        "model_path": MODEL_PATH
    }

    return jsonify(response)


# ==========================================
# ENDPOINT PREDICCIÓN BATCH
# ==========================================

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.json

    valid, result = validate_input(data)
    if not valid:
        return jsonify({"error": result}), 400

    df = result
    predictions = model.predict(df).tolist()

    response = {
        "predictions": predictions,
        "cantidad_registros": len(predictions),
        "timestamp": datetime.now().isoformat(),
        "model_path": MODEL_PATH
    }

    return jsonify(response)


# ==========================================
# ENDPOINT HOME
# ==========================================

@app.route("/", methods=["GET"])
def home():
    return {
        "message": "API de modelo MLflow funcionando con validación",
        "columns_required": EXPECTED_COLUMNS
    }


# ==========================================
# EJECUTAR SERVIDOR
# ==========================================

if __name__ == "__main__":
    print(">>> API iniciada en http://127.0.0.1:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)