"""Etapa 4, primera parte. Detección de latidos. Se usa para limpiar el EEG y para la HRV."""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

BANDA_QRS_HZ = (5.0, 15.0)
INTEGRACION_S = 0.150
REFRACTARIO_S = 0.200
VENTANA_ONDA_T_S = 0.360
APRENDIZAJE_S = 10.0
SIN_LATIDOS_S = 5.0  # Tras este tiempo sin latidos se reinician los umbrales.


def _senal_integrada(ecg: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pasa-banda, derivada, cuadrado e integración móvil, los cuatro pasos de Pan-Tompkins."""
    sos = signal.butter(2, BANDA_QRS_HZ, "bandpass", fs=fs, output="sos")
    filtrada = signal.sosfiltfilt(sos, ecg)
    derivada = np.convolve(filtrada, np.array([1, 2, 0, -2, -1]) * fs / 8, mode="same")
    integrada = ndimage.uniform_filter1d(derivada**2, max(int(INTEGRACION_S * fs), 1))
    return filtrada, derivada, integrada


def _umbrales_iniciales(alturas: np.ndarray, picos: np.ndarray, fs: float, desde: int) -> tuple[float, float]:
    """Estima el nivel de señal y de ruido con los picos de los próximos segundos."""
    tramo = alturas[(picos >= desde) & (picos < desde + APRENDIZAJE_S * fs)]
    if len(tramo) == 0:
        return float(np.median(alturas)), 0.0
    return float(np.percentile(tramo, 90)), float(np.median(tramo))


def detectar_latidos(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Algoritmo de Pan-Tompkins. Devuelve los índices de cada onda R en orden creciente."""
    filtrada, derivada, integrada = _senal_integrada(np.asarray(ecg, dtype=float), fs)
    picos, _ = signal.find_peaks(integrada, distance=max(int(REFRACTARIO_S * fs), 1))
    if len(picos) == 0:
        return np.array([], dtype=int)

    # Un artefacto enorme no debe inflar los umbrales durante el resto de la noche.
    alturas = np.minimum(integrada[picos], np.percentile(integrada[picos], 99))
    mitad_qrs = int(0.075 * fs)
    pendiente = lambda p: np.abs(derivada[max(p - mitad_qrs, 0) : p + mitad_qrs + 1]).max()

    spki, npki = _umbrales_iniciales(alturas, picos, fs, int(picos[0]))
    aceptados: list[int] = []
    pendiente_previa = 0.0
    rr_medio = fs  # Se arranca suponiendo 60 latidos por minuto.
    rechazados: list[tuple[int, float]] = []

    def aceptar(pico: int, altura: float, peso: float) -> None:
        nonlocal spki, rr_medio, pendiente_previa
        spki = (1 - peso) * spki + peso * altura
        if aceptados:
            rr_medio = 0.875 * rr_medio + 0.125 * (pico - aceptados[-1])
        aceptados.append(pico)
        # Promedio móvil de la pendiente, para no depender de un solo latido.
        actual = pendiente(pico)
        pendiente_previa = actual if pendiente_previa == 0 else 0.875 * pendiente_previa + 0.125 * actual

    for pico, altura in zip(picos.tolist(), alturas.tolist()):
        if aceptados and pico - aceptados[-1] > SIN_LATIDOS_S * fs:
            spki, npki = _umbrales_iniciales(alturas, picos, fs, pico)
            rechazados.clear()

        umbral = npki + 0.25 * (spki - npki)

        # Si pasó demasiado tiempo se revisan los picos descartados con un umbral menor.
        if aceptados and pico - aceptados[-1] > 1.66 * rr_medio and rechazados:
            mejor, altura_mejor = max(rechazados, key=lambda r: r[1])
            if altura_mejor > 0.5 * umbral:
                aceptar(mejor, altura_mejor, 0.25)
            rechazados.clear()

        es_onda_t = (
            aceptados
            and pico - aceptados[-1] < VENTANA_ONDA_T_S * fs
            and pendiente(pico) < 0.5 * pendiente_previa
        )
        if altura > umbral and not es_onda_t:
            aceptar(pico, altura, 0.125)
            rechazados.clear()
        else:
            npki = 0.875 * npki + 0.125 * altura
            if not es_onda_t:
                rechazados.append((pico, altura))

    if not aceptados:
        return np.array([], dtype=int)

    # La polaridad se decide una vez por noche, así un ECG invertido se detecta igual y
    # el punto de referencia es el mismo en todos los latidos.
    tramos = [filtrada[max(p - mitad_qrs, 0) : p + mitad_qrs + 1] for p in aceptados]
    signo = np.sign(np.median([t[np.argmax(np.abs(t))] for t in tramos])) or 1.0
    latidos = [
        max(p - mitad_qrs, 0) + int(np.argmax(signo * t)) for p, t in zip(aceptados, tramos)
    ]
    return np.unique(latidos).astype(int)
