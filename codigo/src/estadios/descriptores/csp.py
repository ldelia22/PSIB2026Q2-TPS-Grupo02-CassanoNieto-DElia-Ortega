"""Etapa 8: patrones espaciales comunes (H3).

Los pesos se ajustan SOLO con pacientes de entrenamiento. Por eso CSP no se calcula
al describir cada registro: el `ModeloEstadios` lo ajusta y aplica sobre las
covarianzas que quedan guardadas en la `TablaDescriptores`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg


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
    centradas = epocas_eeg - epocas_eeg.mean(axis=-1, keepdims=True)
    return centradas @ centradas.transpose(0, 2, 1) / (epocas_eeg.shape[-1] - 1)


def _covarianza_media(covarianzas: np.ndarray) -> np.ndarray:
    """Promedio con traza normalizada, para que no dominen las épocas de mayor amplitud."""
    trazas = np.trace(covarianzas, axis1=1, axis2=2)
    con_senal = trazas > 0  # Una época plana no aporta y dividiría por 0.
    return (covarianzas[con_senal] / trazas[con_senal, None, None]).mean(axis=0)


def ajustar_csp(
    covarianzas: np.ndarray, estadios: np.ndarray, par: tuple[int, int], n_componentes: int
) -> FiltroCSP:
    """Autovalores generalizados de las covarianzas medias de los dos estadios.

    Se quedan los `n_componentes` filtros de cada extremo. Ignora las épocas de
    los demás estadios.
    """
    media_a = _covarianza_media(covarianzas[estadios == par[0]])
    media_b = _covarianza_media(covarianzas[estadios == par[1]])
    _, vectores = linalg.eigh(media_a, media_a + media_b)  # Autovalores en orden creciente.
    extremos = np.r_[: n_componentes, -n_componentes:0]
    return FiltroCSP(par=par, pesos=vectores[:, extremos].T)


def aplicar_csp(covarianzas: np.ndarray, filtro: FiltroCSP) -> np.ndarray:
    """Log-potencia de la señal filtrada: log(diag(W C Wᵀ)), sin volver a la señal.

    Returns:
        Forma (n_epocas, n_filtros).
    """
    potencias = np.einsum("fc,ecd,fd->ef", filtro.pesos, covarianzas, filtro.pesos)
    return np.log(np.maximum(potencias, np.finfo(float).tiny))
