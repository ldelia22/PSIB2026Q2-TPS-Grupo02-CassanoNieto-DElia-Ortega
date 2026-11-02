"""Etapa 8: patrones espaciales comunes (H3).

Los pesos se ajustan SOLO con pacientes de entrenamiento. Por eso CSP no se calcula
al describir cada registro: el `ModeloEstadios` lo ajusta y aplica sobre las
covarianzas que quedan guardadas en la `TablaDescriptores`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FiltroCSP:
    par: tuple[int, int]  # Los dos estadios que separa.
    pesos: np.ndarray  # (n_filtros, n_canales). Cada fila es un filtro espacial.


def covarianzas_por_epoca(epocas_eeg: np.ndarray) -> np.ndarray:
    """Covarianza espacial de cada época.

    Args:
        epocas_eeg: Forma (n_epocas, n_canales, n_muestras).

    Returns:
        Forma (n_epocas, n_canales, n_canales).
    """
    raise NotImplementedError("Etapa 8")


def ajustar_csp(
    covarianzas: np.ndarray, estadios: np.ndarray, par: tuple[int, int], n_componentes: int
) -> FiltroCSP:
    """Autovalores generalizados de las covarianzas medias de los dos estadios.

    Se quedan los `n_componentes` filtros de cada extremo. Ignora las épocas de
    los demás estadios.
    """
    raise NotImplementedError("Etapa 8")


def aplicar_csp(covarianzas: np.ndarray, filtro: FiltroCSP) -> np.ndarray:
    """Log-potencia de la señal filtrada: log(diag(W C Wᵀ)), sin volver a la señal.

    Returns:
        Forma (n_epocas, n_filtros).
    """
    raise NotImplementedError("Etapa 8")
