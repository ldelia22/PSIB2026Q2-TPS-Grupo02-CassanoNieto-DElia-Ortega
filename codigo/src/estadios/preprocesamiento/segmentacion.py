"""Etapa 5. Segmentación en épocas y marcado de épocas no estacionarias."""

from __future__ import annotations

import numpy as np
from scipy import linalg, signal

from ..comun.config import ConfigPreprocesamiento
from ..comun.constantes import DURACION_EPOCA_S, EEG_C4
from ..comun.contratos import Registro

ORDEN_AR = 8
SUBVENTANA_S = 5.0
CAMBIO_MAXIMO = 4.0  # Veces que puede crecer el error de predicción dentro de una época.


def segmentar(senal: np.ndarray, fs: float, n_epocas: int) -> np.ndarray:
    """Corta la señal (o señales, en el último eje) en épocas de 30 s.

    La forma de salida es (n_epocas, n_muestras) o, si la entrada es multicanal,
    (n_epocas, n_canales, n_muestras).
    """
    n_epoca = int(DURACION_EPOCA_S * fs)
    recortada = senal[..., : n_epocas * n_epoca]
    epocas = recortada.reshape(*senal.shape[:-1], n_epocas, n_epoca)
    return np.moveaxis(epocas, -2, 0)


def _modelo_ar(x: np.ndarray) -> np.ndarray | None:
    """Filtro de error de predicción por Yule-Walker. Devuelve None si el tramo es plano."""
    x = x - x.mean()
    r = np.array([x[: len(x) - k] @ x[k:] for k in range(ORDEN_AR + 1)]) / len(x)
    if r[0] <= 0:
        return None
    return np.r_[1.0, -linalg.solve_toeplitz(r[:-1], r[1:])]


def _error_de_prediccion(x: np.ndarray, modelo: np.ndarray) -> float:
    """Potencia del error al aplicar el modelo sobre el tramo."""
    x = x - x.mean()
    return float(np.mean(signal.lfilter(modelo, 1.0, x)[ORDEN_AR:] ** 2))


def errores_por_tramo(epoca: np.ndarray, fs: float) -> np.ndarray | None:
    """Error de predicción de cada tramo de 5 s, dividido por el del primero.

    El modelo AR se ajusta en el primer tramo y se aplica sin cambios a los demás. Si
    la señal mantiene su espectro y su amplitud, el error se parece al del primer
    tramo. Devuelve None si el primer tramo es plano.
    """
    n_tramo = int(SUBVENTANA_S * fs)
    tramos = [epoca[j : j + n_tramo] for j in range(0, len(epoca) - n_tramo + 1, n_tramo)]
    modelo = _modelo_ar(tramos[0])
    if modelo is None:
        return None
    referencia = _error_de_prediccion(tramos[0], modelo)
    if referencia <= 0:
        return None
    return np.array([_error_de_prediccion(t, modelo) for t in tramos]) / referencia


def cambio_de_error(registro: Registro) -> np.ndarray:
    """Por época de C4, el mayor cociente de error entre los tramos 2 a 6 y el primero."""
    epocas = segmentar(registro.senales[EEG_C4], registro.fs, registro.n_epocas)
    cambio = np.zeros(registro.n_epocas)
    for i, epoca in enumerate(epocas):
        errores = errores_por_tramo(epoca, registro.fs)
        if errores is not None:
            cambio[i] = errores[1:].max()
    return cambio


def marcar_no_estacionarias(registro: Registro, cfg: ConfigPreprocesamiento) -> np.ndarray:
    """Devuelve un booleano por época, verdadero si la señal cambia dentro de ella.

    Si el modelo AR del primer tramo deja de describir la señal en algún tramo
    siguiente, hubo un cambio, por ejemplo un despertar breve o un movimiento.
    """
    return cambio_de_error(registro) > CAMBIO_MAXIMO
