"""Métricas de acuerdo con los técnicos. Todas ignoran las épocas SIN_ANOTAR."""

from __future__ import annotations

import numpy as np

from ..comun.constantes import N_ESTADIOS, NOMBRES_ESTADIOS, SIN_ANOTAR


def kappa(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """κ de Cohen: comparable con los técnicos (0,74) y con la red de referencia (0,79)."""
    matriz = matriz_confusion(y_real, y_pred).astype(float)
    total = matriz.sum()
    if total == 0:
        return float("nan")
    p_o = np.trace(matriz) / total
    p_e = (matriz.sum(axis=0) @ matriz.sum(axis=1)) / total**2
    return 1.0 if p_e == 1 else float((p_o - p_e) / (1 - p_e))


def f1_por_estadio(y_real: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """F1 de cada estadio, por nombre (ver NOMBRES_ESTADIOS)."""
    matriz = matriz_confusion(y_real, y_pred).astype(float)
    aciertos = np.diag(matriz)
    denominador = matriz.sum(axis=0) + matriz.sum(axis=1)  # Estimadas + reales.
    f1 = np.divide(2 * aciertos, denominador, out=np.zeros(N_ESTADIOS), where=denominador > 0)
    return dict(zip(NOMBRES_ESTADIOS, f1.tolist()))


def matriz_confusion(y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Matriz (n_estadios, n_estadios); las filas son el estadio real."""
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    validas = (y_real != SIN_ANOTAR) & (y_pred != SIN_ANOTAR)
    matriz = np.zeros((N_ESTADIOS, N_ESTADIOS), dtype=int)
    np.add.at(matriz, (y_real[validas], y_pred[validas]), 1)
    return matriz


def reporte(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    """Junta las tres métricas en un diccionario."""
    return {
        "kappa": kappa(y_real, y_pred),
        "f1": f1_por_estadio(y_real, y_pred),
        "confusion": matriz_confusion(y_real, y_pred),
    }
