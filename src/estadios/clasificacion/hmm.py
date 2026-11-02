"""Etapa 10: modelo oculto de Markov y algoritmo de Viterbi."""

from __future__ import annotations

import numpy as np


def estimar_transiciones(
    secuencias: list[np.ndarray], n_estados: int, suavizado: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Cuenta transiciones en los hipnogramas de entrenamiento.

    Ignora los pares con alguna época SIN_ANOTAR. El suavizado evita probabilidades
    nulas, que Viterbi nunca podría atravesar.

    Returns:
        Probabilidad inicial (n_estados,) y de transición (n_estados, n_estados),
        cuyas filas suman 1.
    """
    raise NotImplementedError("Etapa 10")


def viterbi(
    log_emisiones: np.ndarray, log_inicial: np.ndarray, log_transiciones: np.ndarray
) -> np.ndarray:
    """Secuencia de estados más probable de toda la noche.

    Args:
        log_emisiones: Forma (n_epocas, n_estados).

    Returns:
        Estado de cada época, de forma (n_epocas,).
    """
    raise NotImplementedError("Etapa 10")
