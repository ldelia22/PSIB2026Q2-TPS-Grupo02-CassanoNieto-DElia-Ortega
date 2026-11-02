"""Etapa 9: descriptores de EOG y EMG."""

from __future__ import annotations

import numpy as np

from ..comun.config import ConfigDescriptores
from ..comun.contratos import RegistroPreprocesado


def energia_por_epoca(epocas: np.ndarray) -> np.ndarray:
    """Energía de cada época. En el EOG crece con los movimientos oculares de REM."""
    raise NotImplementedError("Etapa 9")


def amplitud_por_epoca(epocas: np.ndarray) -> np.ndarray:
    """Amplitud (por ejemplo, RMS) de cada época. En el EMG cae con la atonía."""
    raise NotImplementedError("Etapa 9")


def descriptores_perifericos(
    prep: RegistroPreprocesado, cfg: ConfigDescriptores
) -> tuple[np.ndarray, list[str]]:
    """Energía del EOG y amplitud del EMG. Nombres con prefijo "perifericos."."""
    raise NotImplementedError("Etapa 9")
