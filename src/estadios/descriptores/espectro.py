"""Etapa 6: espectro multitaper, Welch para comparar (H1) y descriptores por banda."""

from __future__ import annotations

import numpy as np

from ..comun.config import ConfigDescriptores
from ..comun.contratos import RegistroPreprocesado


def psd_multitaper(
    epocas: np.ndarray, fs: float, nw: float, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """PSD multitaper con ventanas DPSS (`scipy.signal.windows.dpss`).

    Args:
        epocas: Forma (n_epocas, n_muestras).

    Returns:
        Frecuencias y PSD, de forma (n_epocas, n_frecuencias).
    """
    raise NotImplementedError("Etapa 6")


def psd_welch(epocas: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """PSD de Welch, con las mismas formas que `psd_multitaper`."""
    raise NotImplementedError("Etapa 6")


def potencia_bandas(
    f: np.ndarray, psd: np.ndarray, bandas: dict[str, tuple[float, float]]
) -> dict[str, np.ndarray]:
    """Potencia absoluta de cada banda, por época."""
    raise NotImplementedError("Etapa 6")


def pendiente_espectral(f: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Pendiente de la recta log-potencia vs. log-frecuencia, por época."""
    raise NotImplementedError("Etapa 6")


def descriptores_espectrales(
    prep: RegistroPreprocesado, cfg: ConfigDescriptores
) -> tuple[np.ndarray, list[str]]:
    """Potencia absoluta y relativa, cocientes entre bandas y pendiente.

    Returns:
        Matriz (n_epocas, n_descriptores) y nombres con prefijo "espectro.".
    """
    raise NotImplementedError("Etapa 6")
