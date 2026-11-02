"""Métricas de acuerdo con los técnicos. Todas ignoran las épocas SIN_ANOTAR."""

from __future__ import annotations

import numpy as np


def kappa(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """κ de Cohen: comparable con los técnicos (0,74) y con la red de referencia (0,79)."""
    raise NotImplementedError("Sección 3")


def f1_por_estadio(y_real: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """F1 de cada estadio, por nombre (ver NOMBRES_ESTADIOS)."""
    raise NotImplementedError("Sección 3")


def matriz_confusion(y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Matriz (n_estadios, n_estadios); las filas son el estadio real."""
    raise NotImplementedError("Sección 3")


def reporte(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    """Junta las tres métricas en un diccionario."""
    return {
        "kappa": kappa(y_real, y_pred),
        "f1": f1_por_estadio(y_real, y_pred),
        "confusion": matriz_confusion(y_real, y_pred),
    }
