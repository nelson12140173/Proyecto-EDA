"""
Ejecución completa del pipeline MLOps:
- Carga de datos
- Entrenamiento de modelo
- Evaluación
Este script ejecuta el pipeline end-to-end usando DVC.
"""

import subprocess

def run_step(name, command):
    print(f"\n==========================")
    print(f"Ejecutando etapa: {name}")
    print(f"==========================")

    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"Error ejecutando la etapa {name}")
        exit(1)

    print(f"Etapa {name} completada.")

if __name__ == "__main__":
    print("\nEjecutando Pipeline MLOps End-to-End...\n")

    # Ejecutar pipeline DVC
    run_step("Pipeline DVC", "dvc repro")

    print("\nPipeline completado exitosamente.\n")
