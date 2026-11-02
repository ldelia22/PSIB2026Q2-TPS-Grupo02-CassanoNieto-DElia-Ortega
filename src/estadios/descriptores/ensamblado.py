"""Une los descriptores de cada etapa en una `TablaDescriptores`."""

from __future__ import annotations

import numpy as np

from ..comun.config import ConfigDescriptores
from ..comun.contratos import RegistroPreprocesado, TablaDescriptores
from ..preprocesamiento.segmentacion import segmentar
from . import csp, espectro, hrv, husos, ondas_lentas, perifericos


def describir_registro(prep: RegistroPreprocesado, cfg: ConfigDescriptores) -> TablaDescriptores:
    """Calcula todos los descriptores. La selección de grupos se hace al clasificar."""
    registro = prep.registro
    grupos = [
        espectro.descriptores_espectrales(prep, cfg),
        husos.descriptores_husos(prep, cfg),
        ondas_lentas.descriptores_ondas_lentas(prep, cfg),
        perifericos.descriptores_perifericos(prep, cfg),
        hrv.descriptores_hrv(prep, cfg),
    ]
    epocas_eeg = segmentar(registro.eeg(), registro.fs, registro.n_epocas)
    return TablaDescriptores(
        id=registro.id,
        matriz=np.hstack([matriz for matriz, _ in grupos]),
        nombres=[nombre for _, nombres in grupos for nombre in nombres],
        estadios=registro.estadios,
        covarianzas_eeg=csp.covarianzas_por_epoca(epocas_eeg),
        no_estacionaria=prep.no_estacionaria,
    )


def estandarizar_por_registro(matriz: np.ndarray) -> np.ndarray:
    """Media 0 y desvío 1 por descriptor, dentro de un mismo registro."""
    desvio = matriz.std(axis=0)
    desvio[desvio == 0] = 1
    return (matriz - matriz.mean(axis=0)) / desvio
