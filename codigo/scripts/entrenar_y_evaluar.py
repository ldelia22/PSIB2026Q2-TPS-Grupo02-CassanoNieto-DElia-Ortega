"""Entrena con los pacientes de entrenamiento y reporta sobre los de prueba.

Uso: python scripts/entrenar_y_evaluar.py [--toda-la-noche]

Por defecto clasifica y evalúa solo entre luces apagadas y encendidas (decisión 3).
"""

from __future__ import annotations

import argparse

import numpy as np

from estadios.clasificacion.modelo import ModeloEstadios
from estadios.comun.config import Config
from estadios.datos.carga import ids_disponibles
from estadios.datos.particion import particion_por_paciente
from estadios.evaluacion.metricas import reporte
from estadios.evaluacion.periodo import recortar_a_periodo_de_sueno
from estadios.pipeline import obtener_tabla


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toda-la-noche", action="store_true",
                        help="incluir las épocas antes de apagar y después de prender las luces")
    args = parser.parse_args()

    cfg = Config()
    ids_entrenamiento, ids_prueba = particion_por_paciente(ids_disponibles())
    print(f"Entrenamiento: {len(ids_entrenamiento)} pacientes. Prueba: {len(ids_prueba)}.")
    print("Toda la noche." if args.toda_la_noche else "Solo entre luces apagadas y encendidas.")

    def tabla(id_registro):
        t = obtener_tabla(id_registro, cfg)
        return t if args.toda_la_noche else recortar_a_periodo_de_sueno(t)

    entrenamiento = [tabla(i) for i in ids_entrenamiento]
    modelo = ModeloEstadios(cfg.clasificacion).ajustar(entrenamiento)

    prueba = [tabla(i) for i in ids_prueba]
    y_real = np.concatenate([t.estadios for t in prueba])
    y_pred = np.concatenate([modelo.predecir(t) for t in prueba])

    resultado = reporte(y_real, y_pred)
    print(f"kappa = {resultado['kappa']:.3f}")
    print("F1 por estadio:", resultado["f1"])
    print("Matriz de confusión (filas: real):")
    print(resultado["confusion"])


if __name__ == "__main__":
    main()
