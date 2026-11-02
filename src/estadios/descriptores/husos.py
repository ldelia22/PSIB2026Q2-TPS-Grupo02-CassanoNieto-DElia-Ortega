"""Etapa 7: detección de husos de sueño en el EEG central."""

from __future__ import annotations

import numpy as np

from ..comun.config import ConfigDescriptores
from ..comun.contratos import RegistroPreprocesado


def detectar_husos(
    senal: np.ndarray,
    fs: float,
    banda_hz: tuple[float, float],
    duracion_s: tuple[float, float],
) -> np.ndarray:
    """Filtra en la banda, calcula la envolvente y busca tramos sobre un umbral.

    El umbral es relativo al propio registro.

    Returns:
        Forma (n_husos, 2): muestra de inicio y de fin de cada huso.
    """
    raise NotImplementedError("Etapa 7")


def descriptores_husos(
    prep: RegistroPreprocesado, cfg: ConfigDescriptores
) -> tuple[np.ndarray, list[str]]:
    """Cantidad y amplitud de husos por época. Nombres con prefijo "husos."."""
    raise NotImplementedError("Etapa 7")
