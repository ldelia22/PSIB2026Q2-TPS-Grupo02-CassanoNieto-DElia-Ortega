"""Etapa 10: análisis discriminante lineal."""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def entrenar_lda(X: np.ndarray, y: np.ndarray) -> LinearDiscriminantAnalysis:
    """Entrena con las épocas ya estandarizadas y anotadas."""
    raise NotImplementedError("Etapa 10")


def pesos_por_descriptor(
    lda: LinearDiscriminantAnalysis, nombres: list[str]
) -> dict[str, np.ndarray]:
    """Peso de cada descriptor en la decisión de cada estadio, para explicarla.

    Returns:
        Estadio (nombre, como en NOMBRES_ESTADIOS) -> peso de cada descriptor.
    """
    raise NotImplementedError("Etapa 10")
