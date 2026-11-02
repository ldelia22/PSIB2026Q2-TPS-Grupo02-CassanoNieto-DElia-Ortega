"""Etapa 7: promediado de Woody (H2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ResultadoWoody:
    plantilla: np.ndarray  # (n_muestras,)
    desplazamientos: np.ndarray  # Muestras que se movió cada realización.
    n_iteraciones: int


def promediado_woody(
    realizaciones: np.ndarray, max_desplazamiento: int, max_iteraciones: int = 20
) -> ResultadoWoody:
    """Alinea cada realización contra el promedio, recalcula y repite.

    Termina cuando los desplazamientos dejan de cambiar.

    Args:
        realizaciones: Forma (n_realizaciones, n_muestras).
        max_desplazamiento: Máximo corrimiento permitido, en muestras.
    """
    raise NotImplementedError("Etapa 7")


def similitud_con_plantilla(realizaciones: np.ndarray, plantilla: np.ndarray) -> np.ndarray:
    """Correlación de cada realización con la plantilla."""
    raise NotImplementedError("Etapa 7")
