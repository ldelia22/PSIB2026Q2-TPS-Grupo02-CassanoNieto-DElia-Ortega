"""Resume por época los eventos detectados (husos y ondas lentas)."""

from __future__ import annotations

import numpy as np

from ..comun.constantes import DURACION_EPOCA_S


def resumir_por_epoca(
    muestras: np.ndarray, valores: np.ndarray, fs: float, n_epocas: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cuenta los eventos de cada época y promedia un valor de ellos.

    Una época sin eventos vale 0, porque LDA no admite NaN. Los eventos posteriores a
    la última época completa se ignoran.

    Returns:
        Cantidad y promedio del valor, ambos de forma (n_epocas,).
    """
    epocas = muestras.astype(int) // int(DURACION_EPOCA_S * fs)
    dentro = epocas < n_epocas
    cantidad = np.bincount(epocas[dentro], minlength=n_epocas).astype(float)
    suma = np.bincount(epocas[dentro], weights=valores[dentro], minlength=n_epocas)
    return cantidad, np.divide(suma, cantidad, np.zeros(n_epocas), where=cantidad > 0)
