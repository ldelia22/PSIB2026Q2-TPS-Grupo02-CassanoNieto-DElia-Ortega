"""Etapa 2: filtrado convencional, con diseños FIR e IIR para comparar."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..comun.config import ConfigPreprocesamiento
from ..comun.contratos import Registro


def filtrar_senal(
    senal: np.ndarray,
    fs: float,
    pasa_altos_hz: float | None,
    pasa_bajos_hz: float | None,
    notch_hz: float | None,
    tipo: Literal["iir", "fir"],
) -> np.ndarray:
    """Filtra sin desfase: el IIR se aplica hacia adelante y hacia atrás (filtfilt).

    Un corte en `None` omite ese filtro.
    """
    raise NotImplementedError("Etapa 2")


def filtrar_registro(registro: Registro, cfg: ConfigPreprocesamiento) -> Registro:
    """Filtra EEG, EOG y EMG. El ECG queda crudo.

    El EMG tiene su contenido útil en altas frecuencias: el pasa-bajos del EEG no
    se le aplica.
    """
    raise NotImplementedError("Etapa 2")


def respuesta_en_frecuencia(
    fs: float, cfg: ConfigPreprocesamiento
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Respuesta del filtro, para comparar FIR e IIR en el informe.

    Returns:
        Frecuencias, magnitud y retardo de grupo.
    """
    raise NotImplementedError("Etapa 2")
