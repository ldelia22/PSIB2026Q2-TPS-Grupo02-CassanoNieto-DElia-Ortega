"""Etapa 9: variabilidad de la frecuencia cardíaca con el periodograma de Lomb."""

from __future__ import annotations

import numpy as np

from ..comun.config import ConfigDescriptores
from ..comun.contratos import RegistroPreprocesado


def descartar_ectopicos(
    ecg: np.ndarray, latidos: np.ndarray, fs: float, umbral_correlacion: float = 0.8
) -> np.ndarray:
    """Quita los latidos cuya forma se parece poco a la del latido promedio.

    Returns:
        Los latidos conservados.
    """
    raise NotImplementedError("Etapa 9")


def intervalos_rr(latidos: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Serie de intervalos entre latidos.

    Returns:
        Instante (s) de cada intervalo y su duración (s).
    """
    raise NotImplementedError("Etapa 9")


def potencias_lomb(t: np.ndarray, rr: np.ndarray) -> tuple[float, float]:
    """Potencia en LF y en HF de una ventana, con `scipy.signal.lombscargle`."""
    raise NotImplementedError("Etapa 9")


def descriptores_hrv(
    prep: RegistroPreprocesado, cfg: ConfigDescriptores
) -> tuple[np.ndarray, list[str]]:
    """HF y LF/HF en una ventana centrada en cada época. Nombres con prefijo "hrv."."""
    raise NotImplementedError("Etapa 9")
