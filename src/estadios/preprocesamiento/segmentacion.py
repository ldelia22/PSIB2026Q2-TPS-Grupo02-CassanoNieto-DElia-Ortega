"""Etapa 5: segmentación en épocas y marcado de épocas no estacionarias."""

from __future__ import annotations

import numpy as np

from ..comun.config import ConfigPreprocesamiento
from ..comun.constantes import DURACION_EPOCA_S
from ..comun.contratos import Registro


def segmentar(senal: np.ndarray, fs: float, n_epocas: int) -> np.ndarray:
    """Corta la señal (o señales, en el último eje) en épocas de 30 s.

    Returns:
        Forma (n_epocas, n_muestras), o (n_epocas, n_canales, n_muestras) si la
        entrada es multicanal.
    """
    n_epoca = int(DURACION_EPOCA_S * fs)
    recortada = senal[..., : n_epocas * n_epoca]
    epocas = recortada.reshape(*senal.shape[:-1], n_epocas, n_epoca)
    return np.moveaxis(epocas, -2, 0)


def marcar_no_estacionarias(registro: Registro, cfg: ConfigPreprocesamiento) -> np.ndarray:
    """Segmentación adaptativa con modelo autorregresivo, la vista en clase.

    Returns:
        Booleano por época: True si la señal cambia sus propiedades dentro de la
        época (por ejemplo, por un despertar breve).
    """
    raise NotImplementedError("Etapa 5")
