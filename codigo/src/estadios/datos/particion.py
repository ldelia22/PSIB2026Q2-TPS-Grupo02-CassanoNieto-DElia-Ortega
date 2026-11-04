"""Selección de registros y partición entrenamiento/prueba.

Cada registro de la HMC es un paciente, así que partir por registro es partir por
paciente. Con la misma semilla, los tres obtenemos exactamente la misma partición.
"""

from __future__ import annotations

import numpy as np

SEMILLA = 42
N_REGISTROS = 70


def elegir_registros(
    ids: list[str], n: int = N_REGISTROS, semilla: int = SEMILLA
) -> list[str]:
    """Subconjunto al azar de `n` registros."""
    orden = np.random.default_rng(semilla).permutation(sorted(ids))
    return sorted(str(i) for i in orden[:n])


def particion_por_paciente(
    ids: list[str], fraccion_prueba: float = 0.2, semilla: int = SEMILLA
) -> tuple[list[str], list[str]]:
    """Devuelve (entrenamiento, prueba), sin pacientes en común."""
    orden = np.random.default_rng(semilla).permutation(sorted(ids))
    n_prueba = round(len(orden) * fraccion_prueba)
    prueba = sorted(str(i) for i in orden[:n_prueba])
    entrenamiento = sorted(str(i) for i in orden[n_prueba:])
    return entrenamiento, prueba
