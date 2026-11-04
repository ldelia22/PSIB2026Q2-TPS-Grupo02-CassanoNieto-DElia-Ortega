"""Etapa 4 (segunda parte): remoción del artefacto cardíaco por plantilla."""

from __future__ import annotations

import numpy as np

from ..comun.config import ConfigPreprocesamiento
from ..comun.contratos import Registro


def restar_plantilla(
    senal: np.ndarray,
    latidos: np.ndarray,
    fs: float,
    n_latidos: int,
    ventana_s: tuple[float, float],
) -> np.ndarray:
    """Resta en cada latido el promedio de la señal alrededor de los últimos `n_latidos`."""
    raise NotImplementedError("Etapa 4")


def remover_artefacto_cardiaco(
    registro: Registro, latidos: np.ndarray, cfg: ConfigPreprocesamiento
) -> Registro:
    """Aplica `restar_plantilla` a cada canal de EEG."""
    raise NotImplementedError("Etapa 4")
