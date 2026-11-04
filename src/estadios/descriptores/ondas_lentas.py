"""Etapa 7: detección de ondas lentas en el EEG frontal."""

from __future__ import annotations

import numpy as np
from scipy import signal

from ..comun.config import ConfigDescriptores
from ..comun.contratos import RegistroPreprocesado
from .eventos import resumir_por_epoca
from .woody import alinear, promediado_woody, similitud_con_plantilla

# Provisorios: el anteproyecto no los fija.
VENTANA_ONDA_S = 1.0  # Se recorta esto a cada lado del pico negativo.
MAX_DESPLAZAMIENTO_S = 0.25


def _filtrar(senal: np.ndarray, fs: float, banda_hz: tuple[float, float]) -> np.ndarray:
    sos = signal.butter(2, banda_hz, "bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, senal)


def _buscar_ondas(
    filtrada: np.ndarray, pico_uv: float, pico_a_pico_uv: float
) -> tuple[np.ndarray, np.ndarray]:
    """Recorre las ondas, de un cruce descendente por cero al siguiente.

    Returns:
        Muestra del pico negativo y amplitud pico a pico de las ondas que cumplen.
    """
    negativa = (filtrada < 0).astype(int)
    cambios = np.diff(negativa)
    bajadas = np.flatnonzero(cambios == 1) + 1
    subidas = np.flatnonzero(cambios == -1) + 1

    picos, amplitudes = [], []
    for inicio, fin in zip(bajadas[:-1], bajadas[1:]):
        subida = subidas[np.searchsorted(subidas, inicio)]
        pico = inicio + np.argmin(filtrada[inicio:subida])
        amplitud = filtrada[subida:fin].max() - filtrada[pico]
        if filtrada[pico] < pico_uv and amplitud >= pico_a_pico_uv:
            picos.append(pico)
            amplitudes.append(amplitud)
    return np.array(picos, dtype=int), np.array(amplitudes, dtype=float)


def detectar_ondas_lentas(
    senal: np.ndarray,
    fs: float,
    banda_hz: tuple[float, float],
    pico_uv: float,
    pico_a_pico_uv: float,
) -> np.ndarray:
    """Criterio visto en clase: pico negativo menor a `pico_uv` y amplitud mínima.

    Returns:
        Índices de muestra del pico negativo de cada onda.
    """
    return _buscar_ondas(_filtrar(senal, fs, banda_hz), pico_uv, pico_a_pico_uv)[0]


def descriptores_ondas_lentas(
    prep: RegistroPreprocesado, cfg: ConfigDescriptores
) -> tuple[np.ndarray, list[str]]:
    """Cantidad, amplitud y parecido con la forma típica del paciente, por época.

    La forma típica sale de `woody.promediado_woody`. Nombres con prefijo "ondas.".
    """
    registro = prep.registro
    fs = registro.fs
    filtrada = _filtrar(registro.senales[cfg.canal_ondas], fs, cfg.ondas_banda_hz)
    picos, amplitudes = _buscar_ondas(filtrada, cfg.ondas_pico_uv, cfg.ondas_pico_a_pico_uv)
    cantidad, amplitud = resumir_por_epoca(picos, amplitudes, fs, registro.n_epocas)

    # Las ondas pegadas a los bordes no tienen ventana completa: cuentan, pero sin similitud.
    mitad = int(VENTANA_ONDA_S * fs)
    recortables = picos[(picos >= mitad) & (picos + mitad < len(filtrada))]
    similitud = np.zeros(registro.n_epocas)
    if len(recortables) > 0:
        realizaciones = np.stack([filtrada[p - mitad : p + mitad + 1] for p in recortables])
        woody = promediado_woody(realizaciones, int(MAX_DESPLAZAMIENTO_S * fs))
        alineadas = alinear(realizaciones, woody.desplazamientos)
        similitudes = similitud_con_plantilla(alineadas, woody.plantilla)
        _, similitud = resumir_por_epoca(recortables, similitudes, fs, registro.n_epocas)

    return (
        np.column_stack((cantidad, amplitud, similitud)),
        ["ondas.cantidad", "ondas.amplitud", "ondas.similitud"],
    )
