"""Etapa 7: promediado de Woody (H2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ResultadoWoody:
    plantilla: np.ndarray  # (n_muestras,)
    desplazamientos: np.ndarray  # Muestras que se movió cada realización.
    n_iteraciones: int


def alinear(realizaciones: np.ndarray, desplazamientos: np.ndarray) -> np.ndarray:
    """Mueve cada realización a la derecha. Lo que queda sin dato se marca con NaN."""
    n = realizaciones.shape[1]
    indices = np.arange(n) - desplazamientos[:, None]
    validos = (indices >= 0) & (indices < n)
    movidas = np.take_along_axis(realizaciones, np.clip(indices, 0, n - 1), axis=1)
    return np.where(validos, movidas, np.nan)


def _promediar(alineadas: np.ndarray) -> np.ndarray:
    """Promedio de cada muestra entre las realizaciones que tienen dato."""
    validos = np.isfinite(alineadas)
    suma = np.where(validos, alineadas, 0.0).sum(axis=0)
    cuenta = validos.sum(axis=0)
    return np.divide(suma, cuenta, np.zeros_like(suma), where=cuenta > 0)


def promediado_woody(
    realizaciones: np.ndarray, max_desplazamiento: int, max_iteraciones: int = 20
) -> ResultadoWoody:
    """Alinea cada realización contra el promedio, recalcula y repite.

    Termina cuando los desplazamientos dejan de cambiar.

    Args:
        realizaciones: Forma (n_realizaciones, n_muestras).
        max_desplazamiento: Máximo corrimiento permitido, en muestras.
    """
    n_realizaciones, n = realizaciones.shape
    # Ordenados de menor a mayor corrimiento: ante un empate, gana el más chico.
    corrimientos = np.arange(-max_desplazamiento, max_desplazamiento + 1)
    corrimientos = corrimientos[np.argsort(np.abs(corrimientos), kind="stable")]
    # La FFT con relleno da la correlación de todos los corrimientos, sin dar la vuelta.
    largo = 2 * n
    centradas = realizaciones - realizaciones.mean(axis=1, keepdims=True)
    transformadas = np.conj(np.fft.rfft(centradas, largo, axis=1))

    desplazamientos = np.zeros(n_realizaciones, dtype=int)
    plantilla = realizaciones.mean(axis=0)
    for iteracion in range(1, max_iteraciones + 1):
        espectro_plantilla = np.fft.rfft(plantilla - plantilla.mean(), largo)
        correlacion = np.fft.irfft(transformadas * espectro_plantilla, largo, axis=1)
        mejores = np.argmax(correlacion[:, corrimientos % largo], axis=1)
        nuevos = corrimientos[mejores]

        convergio = np.array_equal(nuevos, desplazamientos)
        desplazamientos = nuevos
        plantilla = _promediar(alinear(realizaciones, desplazamientos))
        if convergio:
            break

    return ResultadoWoody(plantilla, desplazamientos, iteracion)


def similitud_con_plantilla(realizaciones: np.ndarray, plantilla: np.ndarray) -> np.ndarray:
    """Correlación de cada realización con la plantilla.

    Usa solo las muestras con dato, así que admite la salida de `alinear`. Una
    realización plana no se parece a nada: vale 0.
    """
    validos = np.isfinite(realizaciones)
    cuenta = validos.sum(axis=1, keepdims=True)
    x = np.where(validos, realizaciones, 0.0)
    y = np.where(validos, plantilla, 0.0)
    x = np.where(validos, x - x.sum(axis=1, keepdims=True) / cuenta, 0.0)
    y = np.where(validos, y - y.sum(axis=1, keepdims=True) / cuenta, 0.0)

    covarianza = (x * y).sum(axis=1)
    normas = np.sqrt((x**2).sum(axis=1) * (y**2).sum(axis=1))
    return np.divide(covarianza, normas, np.zeros_like(covarianza), where=normas > 0)
