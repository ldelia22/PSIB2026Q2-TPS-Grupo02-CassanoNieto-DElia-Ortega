"""Etapa 4, segunda parte. Remoción del artefacto cardíaco del EEG por plantilla."""

from __future__ import annotations

import numpy as np
from scipy import signal

from ..comun.config import ConfigPreprocesamiento
from ..comun.constantes import CANALES_EEG
from ..comun.contratos import Registro

MAX_PICO_A_PICO_UV = 500.0  # Las ventanas con movimiento no entran en la plantilla.


def restar_plantilla(
    senal: np.ndarray,
    latidos: np.ndarray,
    fs: float,
    n_latidos: int,
    ventana_s: tuple[float, float],
) -> np.ndarray:
    """Resta en cada latido el promedio de la señal alrededor de los `n_latidos` anteriores.

    Al promediar, el EEG se cancela y queda el artefacto. Los primeros latidos, que no
    tienen historia, usan los siguientes.
    """
    desde, hasta = round(ventana_s[0] * fs), round(ventana_s[1] * fs)
    latidos = np.asarray(latidos, dtype=int)
    latidos = latidos[(latidos + desde >= 0) & (latidos + hasta < len(senal))]
    if len(latidos) == 0:
        return np.array(senal, dtype=float)

    indices = latidos[:, None] + np.arange(desde, hasta + 1)
    ventanas = senal[indices]
    validas = np.ptp(ventanas, axis=1) < MAX_PICO_A_PICO_UV

    # Con sumas acumuladas el promedio móvil no recorre cada ventana muchas veces.
    suma = np.vstack((np.zeros(ventanas.shape[1]), np.cumsum(ventanas * validas[:, None], axis=0)))
    cuenta = np.r_[0, np.cumsum(validas)]
    fin = np.maximum(np.arange(len(latidos)), n_latidos)
    fin = np.minimum(fin, len(latidos))
    inicio = np.maximum(fin - n_latidos, 0)
    cantidad = (cuenta[fin] - cuenta[inicio]).clip(min=1)
    plantillas = (suma[fin] - suma[inicio]) / cantidad[:, None]

    # El ahusado evita escalones en los bordes y las ventanas que se pisan se reparten.
    ahusado = signal.windows.tukey(ventanas.shape[1], 0.25)
    artefacto = np.zeros(len(senal))
    peso = np.zeros(len(senal))
    np.add.at(artefacto, indices.ravel(), (plantillas * ahusado).ravel())
    np.add.at(peso, indices.ravel(), np.tile(ahusado, len(latidos)))
    return senal - artefacto / np.maximum(peso, 1.0)


def remover_artefacto_cardiaco(
    registro: Registro, latidos: np.ndarray, cfg: ConfigPreprocesamiento
) -> Registro:
    """Aplica `restar_plantilla` a cada canal de EEG."""
    nuevas = {
        canal: restar_plantilla(
            registro.senales[canal], latidos, registro.fs, cfg.latidos_plantilla, cfg.ventana_plantilla_s
        )
        for canal in CANALES_EEG
    }
    return registro.reemplazar_senales(nuevas)
