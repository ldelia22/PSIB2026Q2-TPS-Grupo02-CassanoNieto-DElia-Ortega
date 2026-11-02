"""Etapa 4 (primera parte): detección de latidos. La HRV reutiliza este resultado."""

from __future__ import annotations

import numpy as np


def detectar_latidos(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Algoritmo de Pan-Tompkins.

    Returns:
        Índices de muestra de cada onda R, en orden creciente.
    """
    raise NotImplementedError("Etapa 4")
