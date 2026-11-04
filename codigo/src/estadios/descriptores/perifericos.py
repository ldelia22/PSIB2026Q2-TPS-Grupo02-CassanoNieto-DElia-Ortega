"""Etapa 9: descriptores de EOG y EMG."""

from __future__ import annotations

import numpy as np

from ..comun.config import ConfigDescriptores
from ..comun.constantes import EMG, EOG_E1, EOG_E2
from ..comun.contratos import RegistroPreprocesado
from ..preprocesamiento.segmentacion import segmentar


def energia_por_epoca(epocas: np.ndarray) -> np.ndarray:
    """Energía de cada época. En el EOG crece con los movimientos oculares de REM."""
    return (epocas**2).sum(axis=-1)


def amplitud_por_epoca(epocas: np.ndarray) -> np.ndarray:
    """Amplitud (por ejemplo, RMS) de cada época. En el EMG cae con la atonía."""
    return np.sqrt((epocas**2).mean(axis=-1))


def descriptores_perifericos(
    prep: RegistroPreprocesado, cfg: ConfigDescriptores
) -> tuple[np.ndarray, list[str]]:
    """Energía del EOG y amplitud del EMG. Nombres con prefijo "perifericos.".

    Se usa E1 - E2 porque ahí los movimientos oculares se refuerzan y la actividad
    cerebral se cancela. Ambos van en escala logarítmica, como la potencia absoluta.
    """
    registro = prep.registro
    eog = registro.senales[EOG_E1] - registro.senales[EOG_E2]
    energia = energia_por_epoca(segmentar(eog, registro.fs, registro.n_epocas))
    amplitud = amplitud_por_epoca(segmentar(registro.senales[EMG], registro.fs, registro.n_epocas))

    matriz = np.column_stack((energia, amplitud))
    return (
        np.log10(np.maximum(matriz, np.finfo(float).tiny)),
        ["perifericos.energia_eog", "perifericos.amplitud_emg"],
    )
