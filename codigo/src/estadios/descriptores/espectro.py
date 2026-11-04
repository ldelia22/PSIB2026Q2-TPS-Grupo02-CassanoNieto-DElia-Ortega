"""Etapa 6: espectro multitaper, Welch para comparar (H1) y descriptores por banda."""

from __future__ import annotations

import numpy as np
from scipy import signal

from ..comun.config import ConfigDescriptores
from ..comun.constantes import BANDAS_EEG, CANALES_EEG
from ..comun.contratos import RegistroPreprocesado
from ..preprocesamiento.segmentacion import segmentar

# Provisorios: el anteproyecto no los fija.
WELCH_VENTANA_S = 4.0  # Hann de 4 s: lóbulo principal de 1 Hz, como el multitaper.
BANDA_TOTAL_HZ = (0.5, 30.0)
BANDA_PENDIENTE_HZ = (1.0, 40.0)
COCIENTES = (("delta", "theta"), ("delta", "beta"), ("theta", "alfa"))


def psd_multitaper(
    epocas: np.ndarray, fs: float, nw: float, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """PSD multitaper con ventanas DPSS (`scipy.signal.windows.dpss`).

    Args:
        epocas: Forma (n_epocas, n_muestras).

    Returns:
        Frecuencias y PSD, de forma (n_epocas, n_frecuencias).
    """
    n = epocas.shape[-1]
    ventanas = signal.windows.dpss(n, nw, k, norm=2)
    f = np.fft.rfftfreq(n, 1 / fs)
    psd = np.empty((len(epocas), len(f)))
    # Época por época: todas juntas, las 29 ventanas no entran en memoria.
    for i, epoca in enumerate(epocas):
        transformadas = np.fft.rfft(ventanas * (epoca - epoca.mean()), axis=-1)
        psd[i] = (np.abs(transformadas) ** 2).mean(axis=0) / fs

    # Una sola cara: se dobla todo menos 0 Hz y, si existe, la frecuencia de Nyquist.
    psd[:, 1 : -1 if n % 2 == 0 else None] *= 2
    return f, psd


def psd_welch(epocas: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """PSD de Welch, con las mismas formas que `psd_multitaper`."""
    return signal.welch(epocas, fs, nperseg=int(WELCH_VENTANA_S * fs), axis=-1)


def potencia_bandas(
    f: np.ndarray, psd: np.ndarray, bandas: dict[str, tuple[float, float]]
) -> dict[str, np.ndarray]:
    """Potencia absoluta de cada banda, por época."""
    df = f[1] - f[0]
    return {
        nombre: psd[:, (f >= baja) & (f < alta)].sum(axis=1) * df
        for nombre, (baja, alta) in bandas.items()
    }


def pendiente_espectral(f: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Pendiente de la recta log-potencia vs. log-frecuencia, por época."""
    positivas = f > 0
    log_psd = _log(psd[:, positivas])
    return np.polyfit(np.log10(f[positivas]), log_psd.T, 1)[0]


def descriptores_espectrales(
    prep: RegistroPreprocesado, cfg: ConfigDescriptores
) -> tuple[np.ndarray, list[str]]:
    """Potencia absoluta y relativa, cocientes entre bandas y pendiente.

    Se calculan para cada canal de EEG. La potencia absoluta y los cocientes van en
    escala logarítmica, porque LDA supone descriptores con distribución normal.

    Returns:
        Matriz (n_epocas, n_descriptores) y nombres con prefijo "espectro.".
    """
    registro = prep.registro
    epocas = segmentar(registro.eeg(), registro.fs, registro.n_epocas)

    columnas: list[np.ndarray] = []
    nombres: list[str] = []
    for c, canal in enumerate(CANALES_EEG):
        if cfg.metodo_espectro == "multitaper":
            f, psd = psd_multitaper(
                epocas[:, c], registro.fs, cfg.multitaper_nw, cfg.multitaper_k
            )
        else:
            f, psd = psd_welch(epocas[:, c], registro.fs)

        potencias = potencia_bandas(f, psd, BANDAS_EEG)
        total = potencia_bandas(f, psd, {"total": BANDA_TOTAL_HZ})["total"]
        prefijo = f"espectro.{canal.split()[1].split('-')[0].lower()}"
        for banda, potencia in potencias.items():
            columnas.append(_log(potencia))
            nombres.append(f"{prefijo}_{banda}_abs")
            columnas.append(np.divide(potencia, total, np.zeros_like(potencia), where=total > 0))
            nombres.append(f"{prefijo}_{banda}_rel")
        for numerador, denominador in COCIENTES:
            columnas.append(_log(potencias[numerador]) - _log(potencias[denominador]))
            nombres.append(f"{prefijo}_{numerador}_{denominador}")

        rango = (f >= BANDA_PENDIENTE_HZ[0]) & (f <= BANDA_PENDIENTE_HZ[1])
        columnas.append(pendiente_espectral(f[rango], psd[:, rango]))
        nombres.append(f"{prefijo}_pendiente")

    return np.column_stack(columnas), nombres


def _log(x: np.ndarray) -> np.ndarray:
    """Logaritmo que da un valor finito aun en tramos planos, donde la potencia es 0."""
    return np.log10(np.maximum(x, np.finfo(float).tiny))
