"""Etapa 2. Filtrado convencional, con diseños FIR e IIR para comparar."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import signal

from ..comun.config import ConfigPreprocesamiento
from ..comun.constantes import CANALES_EEG, EMG, EOG_E1, EOG_E2
from ..comun.contratos import Registro

ORDEN_IIR = 4
Q_NOTCH = 30.0  # Ancho del notch de unos 1,7 Hz en 50 Hz.
ANCHO_NOTCH_FIR_HZ = 2.0  # Mitad del rechazo a cada lado de la frecuencia de línea.


def disenar_iir(
    fs: float, pasa_altos_hz: float | None, pasa_bajos_hz: float | None, notch_hz: float | None
) -> list[np.ndarray]:
    """Secciones de segundo orden de cada filtro pedido. Son estables aun con cortes muy bajos."""
    filtros = []
    if pasa_altos_hz is not None:
        filtros.append(signal.butter(ORDEN_IIR, pasa_altos_hz, "highpass", fs=fs, output="sos"))
    if pasa_bajos_hz is not None:
        filtros.append(signal.butter(ORDEN_IIR, pasa_bajos_hz, "lowpass", fs=fs, output="sos"))
    if notch_hz is not None:
        filtros.append(signal.tf2sos(*signal.iirnotch(notch_hz, Q_NOTCH, fs=fs)))
    return filtros


def _largo_fir(fs: float, transicion_hz: float) -> int:
    """Largo impar de un FIR con ventana de Hamming para ese ancho de transición."""
    return int(np.ceil(3.3 * fs / transicion_hz)) | 1


def disenar_fir(
    fs: float, pasa_altos_hz: float | None, pasa_bajos_hz: float | None, notch_hz: float | None
) -> list[np.ndarray]:
    """Coeficientes de fase lineal de cada filtro pedido."""
    filtros = []
    if pasa_altos_hz is not None:
        largo = _largo_fir(fs, pasa_altos_hz)
        filtros.append(signal.firwin(largo, pasa_altos_hz, pass_zero=False, fs=fs))
    if pasa_bajos_hz is not None:
        filtros.append(signal.firwin(_largo_fir(fs, 2.0), pasa_bajos_hz, fs=fs))
    if notch_hz is not None:
        banda = [notch_hz - ANCHO_NOTCH_FIR_HZ, notch_hz + ANCHO_NOTCH_FIR_HZ]
        filtros.append(signal.firwin(_largo_fir(fs, 2.0), banda, pass_zero="bandstop", fs=fs))
    return filtros


def filtrar_senal(
    senal: np.ndarray,
    fs: float,
    pasa_altos_hz: float | None,
    pasa_bajos_hz: float | None,
    notch_hz: float | None,
    tipo: Literal["iir", "fir"],
) -> np.ndarray:
    """Filtra sin desfase. Un corte en `None` omite ese filtro.

    El IIR se aplica hacia adelante y hacia atrás. El FIR tiene retardo constante y
    se compensa recortando la mitad de su largo a cada lado.
    """
    salida = np.asarray(senal, dtype=float)
    if tipo == "iir":
        cortes = [c for c in (pasa_altos_hz, pasa_bajos_hz, notch_hz) if c is not None]
        for sos, corte in zip(disenar_iir(fs, pasa_altos_hz, pasa_bajos_hz, notch_hz), cortes):
            # El relleno se ajusta al corte para que el transitorio no invada la señal.
            relleno = min(int(3 * fs / corte), len(salida) - 1)
            salida = signal.sosfiltfilt(sos, salida, padlen=relleno)
        return salida

    for h in disenar_fir(fs, pasa_altos_hz, pasa_bajos_hz, notch_hz):
        mitad = len(h) // 2
        extendida = np.pad(salida, mitad, mode="reflect")
        salida = signal.oaconvolve(extendida, h, mode="same")[mitad:-mitad]
    return salida


def filtrar_registro(registro: Registro, cfg: ConfigPreprocesamiento) -> Registro:
    """Filtra EEG, EOG y EMG. El ECG queda crudo porque Pan-Tompkins tiene su propio filtro.

    El EMG usa otro pasa-altos y no lleva pasa-bajos, porque su información útil está
    en las frecuencias altas.
    """

    def filtrar(senal: np.ndarray, pasa_altos: float, pasa_bajos: float | None) -> np.ndarray:
        return filtrar_senal(senal, registro.fs, pasa_altos, pasa_bajos, cfg.notch_hz, cfg.tipo_filtro)

    nuevas = {
        canal: filtrar(registro.senales[canal], cfg.pasa_altos_hz, cfg.pasa_bajos_hz)
        for canal in (*CANALES_EEG, EOG_E1, EOG_E2)
        if canal in registro.senales
    }
    if EMG in registro.senales:
        nuevas[EMG] = filtrar(registro.senales[EMG], cfg.emg_pasa_altos_hz, None)
    return registro.reemplazar_senales(nuevas)


def respuesta_en_frecuencia(
    fs: float, cfg: ConfigPreprocesamiento
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frecuencias, magnitud y retardo de grupo en segundos de una sola pasada del filtro del EEG."""
    n = 2**18  # Grilla fina para resolver el pasa-altos de décimas de Hz.
    h = np.ones(n, dtype=complex)
    f = np.linspace(0, fs / 2, n, endpoint=False)
    if cfg.tipo_filtro == "iir":
        for sos in disenar_iir(fs, cfg.pasa_altos_hz, cfg.pasa_bajos_hz, cfg.notch_hz):
            h *= signal.sosfreqz(sos, worN=n)[1]
    else:
        for coeficientes in disenar_fir(fs, cfg.pasa_altos_hz, cfg.pasa_bajos_hz, cfg.notch_hz):
            h *= signal.freqz(coeficientes, worN=n)[1]

    fase = np.unwrap(np.angle(h))
    retardo = -np.gradient(fase, 2 * np.pi * f)
    return f, np.abs(h), retardo
