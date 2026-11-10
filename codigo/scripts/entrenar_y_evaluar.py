"""Ajusta con los pacientes de entrenamiento y reporta el acuerdo con los técnicos.

Uso: python scripts/entrenar_y_evaluar.py [--evaluar-en entrenamiento|prueba] [--toda-la-noche]

Para elegir o comparar umbrales se usa SIEMPRE `--evaluar-en entrenamiento` (es el valor
por defecto): las reglas no se entrenan, así que medir sobre esos pacientes es válido y
no toca los de prueba. `--evaluar-en prueba` es para el resultado final, una sola vez,
con los umbrales ya fijados.

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
    parser.add_argument("--evaluar-en", choices=("entrenamiento", "prueba"), default="entrenamiento",
                        help="pacientes sobre los que se mide (por defecto, entrenamiento)")
    parser.add_argument("--toda-la-noche", action="store_true",
                        help="incluir las épocas antes de apagar y después de prender las luces")
    args = parser.parse_args()

    cfg = Config()
    ids_entrenamiento, ids_prueba = particion_por_paciente(ids_disponibles())
    print(f"Entrenamiento: {len(ids_entrenamiento)} pacientes. Prueba: {len(ids_prueba)}.")
    print("Toda la noche." if args.toda_la_noche else "Solo entre luces apagadas y encendidas.")
    if args.evaluar_en == "prueba":
        print("EVALUANDO SOBRE PRUEBA: usar solo para el resultado final, con los umbrales ya fijados.")
    else:
        print("Evaluando sobre los pacientes de entrenamiento.")

    def tabla(id_registro):
        t = obtener_tabla(id_registro, cfg)
        return t if args.toda_la_noche else recortar_a_periodo_de_sueno(t)

    entrenamiento = [tabla(i) for i in ids_entrenamiento]
    modelo = ModeloEstadios(cfg.clasificacion).ajustar(entrenamiento)

    evaluados = entrenamiento if args.evaluar_en == "entrenamiento" else [tabla(i) for i in ids_prueba]
    y_real = np.concatenate([t.estadios for t in evaluados])
    y_pred = np.concatenate([modelo.predecir(t) for t in evaluados])

    resultado = reporte(y_real, y_pred)
    print(f"kappa = {resultado['kappa']:.3f}")
    print("F1 por estadio:", resultado["f1"])
    print("Matriz de confusión (filas: real):")
    print(resultado["confusion"])


if __name__ == "__main__":
    main()
