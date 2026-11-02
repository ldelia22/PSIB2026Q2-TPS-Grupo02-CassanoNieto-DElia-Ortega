"""Entrena con los pacientes de entrenamiento y reporta sobre los de prueba.

Uso: python scripts/entrenar_y_evaluar.py
"""

from __future__ import annotations

import numpy as np

from estadios.clasificacion.modelo import ModeloEstadios
from estadios.comun.config import Config
from estadios.datos.carga import ids_disponibles
from estadios.datos.particion import particion_por_paciente
from estadios.evaluacion.metricas import reporte
from estadios.pipeline import obtener_tabla


def main() -> None:
    cfg = Config()
    ids_entrenamiento, ids_prueba = particion_por_paciente(ids_disponibles())
    print(f"Entrenamiento: {len(ids_entrenamiento)} pacientes. Prueba: {len(ids_prueba)}.")

    entrenamiento = [obtener_tabla(i, cfg) for i in ids_entrenamiento]
    modelo = ModeloEstadios(cfg.clasificacion).ajustar(entrenamiento)

    prueba = [obtener_tabla(i, cfg) for i in ids_prueba]
    y_real = np.concatenate([t.estadios for t in prueba])
    y_pred = np.concatenate([modelo.predecir(t) for t in prueba])

    resultado = reporte(y_real, y_pred)
    print(f"kappa = {resultado['kappa']:.3f}")
    print("F1 por estadio:", resultado["f1"])
    print("Matriz de confusión (filas: real):")
    print(resultado["confusion"])


if __name__ == "__main__":
    main()
