"""Etapa 1: análisis exploratorio. Decide si se aplican las etapas 3 y 4."""

from __future__ import annotations

import numpy as np

from ..comun.contratos import Registro


def espectro_por_estadio(registro: Registro, canal: str) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """PSD promedio del canal en cada estadio.

    Returns:
        Frecuencias, y un diccionario estadio -> PSD.
    """
    raise NotImplementedError("Etapa 1")


def correlacion_eeg_eog(registro: Registro) -> dict[str, float]:
    """Correlación entre cada canal de EEG y la diferencia de EOG (artefacto ocular)."""
    raise NotImplementedError("Etapa 1")


def promedio_alrededor_de_latidos(
    senal: np.ndarray,
    latidos: np.ndarray,
    fs: float,
    ventana_s: tuple[float, float] = (-0.2, 0.6),
) -> tuple[np.ndarray, np.ndarray]:
    """Promedio de la señal sincronizado con los latidos.

    Si aparece una forma repetible, hay artefacto cardíaco. También mide H4: tras
    la remoción este promedio tiene que tender a cero.

    Returns:
        Tiempos (s) relativos al latido y promedio.
    """
    raise NotImplementedError("Etapa 1")
