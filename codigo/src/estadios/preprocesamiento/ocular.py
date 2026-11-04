"""Etapa 3: remoción del artefacto ocular, con NLMS y Wiener para comparar.

Ojo con el costo: NLMS recorre unas 7 millones de muestras por canal y por noche.
Un bucle de Python puro tarda demasiado, conviene evaluar numba o submuestrear.
"""

from __future__ import annotations

import numpy as np

from ..comun.config import ConfigPreprocesamiento
from ..comun.contratos import Registro


def nlms(eeg: np.ndarray, referencia: np.ndarray, orden: int, mu: float) -> np.ndarray:
    """Cancelación adaptativa: devuelve el EEG sin lo que se explica por la referencia."""
    raise NotImplementedError("Etapa 3")


def wiener(eeg: np.ndarray, referencia: np.ndarray, orden: int) -> np.ndarray:
    """Igual que `nlms`, pero con una corrección fija calculada sobre toda la señal."""
    raise NotImplementedError("Etapa 3")


def remover_artefacto_ocular(registro: Registro, cfg: ConfigPreprocesamiento) -> Registro:
    """Limpia cada canal de EEG usando como referencia la diferencia E1 - E2."""
    raise NotImplementedError("Etapa 3")
