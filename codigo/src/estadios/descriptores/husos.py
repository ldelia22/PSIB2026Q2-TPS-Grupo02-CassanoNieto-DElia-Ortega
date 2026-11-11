"""Etapa 7: detección de husos de sueño en el EEG central."""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from ..comun.config import ConfigDescriptores
from ..comun.contratos import RegistroPreprocesado
from .eventos import resumir_por_epoca

# Siguen a Mölle et al. (2002).
UMBRAL_DESVIOS = 1.5
SUAVIZADO_S = 0.2


def _envolvente(senal: np.ndarray, fs: float, banda_hz: tuple[float, float]) -> np.ndarray:
    """Envolvente de la señal en la banda, suavizada para no partir un huso en dos."""
    sos = signal.butter(4, banda_hz, "bandpass", fs=fs, output="sos")
    filtrada = signal.sosfiltfilt(sos, senal)
    envolvente = np.abs(signal.hilbert(filtrada))
    return ndimage.uniform_filter1d(envolvente, int(SUAVIZADO_S * fs))


def _buscar_tramos(
    envolvente: np.ndarray, fs: float, duracion_s: tuple[float, float]
) -> np.ndarray:
    """Tramos sobre el umbral, relativo al propio registro, con duración válida."""
    umbral = envolvente.mean() + UMBRAL_DESVIOS * envolvente.std()
    sobre = np.concatenate(([False], envolvente > umbral, [False]))
    cambios = np.flatnonzero(np.diff(sobre.astype(int)))
    tramos = cambios.reshape(-1, 2)
    duraciones = (tramos[:, 1] - tramos[:, 0]) / fs
    return tramos[(duraciones >= duracion_s[0]) & (duraciones <= duracion_s[1])]


def detectar_husos(
    senal: np.ndarray,
    fs: float,
    banda_hz: tuple[float, float],
    duracion_s: tuple[float, float],
) -> np.ndarray:
    """Filtra en la banda, calcula la envolvente y busca tramos sobre un umbral.

    El umbral es relativo al propio registro.

    Returns:
        Forma (n_husos, 2): muestra de inicio y de fin de cada huso.
    """
    return _buscar_tramos(_envolvente(senal, fs, banda_hz), fs, duracion_s)


def descriptores_husos(
    prep: RegistroPreprocesado, cfg: ConfigDescriptores
) -> tuple[np.ndarray, list[str]]:
    """Cantidad y amplitud de husos por época. Nombres con prefijo "husos."."""
    registro = prep.registro
    envolvente = _envolvente(registro.senales[cfg.canal_husos], registro.fs, cfg.husos_banda_hz)
    tramos = _buscar_tramos(envolvente, registro.fs, cfg.husos_duracion_s)

    # Cada huso cuenta en la época donde cae su centro.
    centros = tramos.mean(axis=1)
    amplitudes = np.array([envolvente[inicio:fin].max() for inicio, fin in tramos])
    cantidad, amplitud = resumir_por_epoca(centros, amplitudes, registro.fs, registro.n_epocas)
    return np.column_stack((cantidad, amplitud)), ["husos.cantidad", "husos.amplitud"]
