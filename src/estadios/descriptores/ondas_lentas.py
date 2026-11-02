"""Etapa 7: detección de ondas lentas en el EEG frontal."""

from __future__ import annotations

import numpy as np

from ..comun.config import ConfigDescriptores
from ..comun.contratos import RegistroPreprocesado


def detectar_ondas_lentas(
    senal: np.ndarray,
    fs: float,
    banda_hz: tuple[float, float],
    pico_uv: float,
    pico_a_pico_uv: float,
) -> np.ndarray:
    """Criterio visto en clase: pico negativo menor a `pico_uv` y amplitud mínima.

    Returns:
        Índices de muestra del pico negativo de cada onda.
    """
    raise NotImplementedError("Etapa 7")


def descriptores_ondas_lentas(
    prep: RegistroPreprocesado, cfg: ConfigDescriptores
) -> tuple[np.ndarray, list[str]]:
    """Cantidad, amplitud y parecido con la forma típica del paciente, por época.

    La forma típica sale de `woody.promediado_woody`. Nombres con prefijo "ondas.".
    """
    raise NotImplementedError("Etapa 7")
