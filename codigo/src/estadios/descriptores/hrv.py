"""Etapa 9. Variabilidad de la frecuencia cardíaca con el periodograma de Lomb."""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from ..comun.config import ConfigDescriptores
from ..comun.constantes import BANDA_HF, BANDA_LF, DURACION_EPOCA_S, ECG
from ..comun.contratos import RegistroPreprocesado

VENTANA_LATIDO_S = (-0.2, 0.4)  # Onda P, QRS y buena parte de la onda T.
UMBRAL_FORMA = 0.8  # Correlación mínima con el latido típico para conservar un latido.
RR_LIMITES_S = (0.3, 2.0)  # Fuera de este rango no es un intervalo fisiológico.
CAMBIO_RR_MAXIMO = 0.2  # Respecto de la mediana local. Detecta los latidos faltantes.
MUESTRAS_MEDIANA_RR = 11
MIN_INTERVALOS = 30  # Con menos intervalos, la estimación de LF no es confiable.
PASO_FRECUENCIA_HZ = 0.0025


def comparar_con_latido_tipico(
    ecg: np.ndarray, latidos: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compara la forma de cada latido con la del latido típico de la noche.

    El latido típico es la mediana, muestra a muestra, de todos los latidos. Los
    atípicos son minoría y casi no la mueven. La correlación mide parecido de forma
    sin importar la amplitud. Devuelve el latido típico, las ventanas, la correlación
    de cada una y qué latidos tienen ventana completa.
    """
    latidos = np.asarray(latidos, dtype=int)
    desde, hasta = round(VENTANA_LATIDO_S[0] * fs), round(VENTANA_LATIDO_S[1] * fs)
    completos = (latidos + desde >= 0) & (latidos + hasta < len(ecg))
    if not completos.any():
        return np.array([]), np.empty((0, hasta - desde + 1)), np.array([]), completos

    # El pasa-banda quita la deriva y el ruido para que solo se compare la forma.
    sos = signal.butter(2, (0.5, 40.0), "bandpass", fs=fs, output="sos")
    filtrada = signal.sosfiltfilt(sos, ecg)
    ventanas = filtrada[latidos[completos][:, None] + np.arange(desde, hasta + 1)]
    ventanas = ventanas - ventanas.mean(axis=1, keepdims=True)
    tipico = np.median(ventanas, axis=0)

    normas = np.linalg.norm(ventanas, axis=1) * np.linalg.norm(tipico)
    correlacion = np.divide(ventanas @ tipico, normas, np.zeros(len(ventanas)), where=normas > 0)
    return tipico, ventanas, correlacion, completos


def descartar_ectopicos(
    ecg: np.ndarray, latidos: np.ndarray, fs: float, umbral_correlacion: float = UMBRAL_FORMA
) -> np.ndarray:
    """Devuelve los latidos cuya forma se parece a la del latido típico de la noche."""
    latidos = np.asarray(latidos, dtype=int)
    _, _, correlacion, completos = comparar_con_latido_tipico(ecg, latidos, fs)
    conservar = np.ones(len(latidos), dtype=bool)  # Los de los bordes no se pueden evaluar.
    conservar[completos] = correlacion >= umbral_correlacion
    return latidos[conservar]


def intervalos_rr(latidos: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Instante en segundos de cada intervalo entre latidos y su duración en segundos."""
    return latidos[1:] / fs, np.diff(latidos) / fs


def filtrar_intervalos(t: np.ndarray, rr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quita los intervalos imposibles y los que abarcan un latido descartado o perdido."""
    if len(rr) == 0:
        return t, rr
    mediana = ndimage.median_filter(rr, size=MUESTRAS_MEDIANA_RR, mode="nearest")
    validos = (rr >= RR_LIMITES_S[0]) & (rr <= RR_LIMITES_S[1])
    validos &= np.abs(rr - mediana) <= CAMBIO_RR_MAXIMO * mediana
    return t[validos], rr[validos]


def potencias_lomb(t: np.ndarray, rr: np.ndarray) -> tuple[float, float]:
    """Potencia en LF y en HF de una ventana.

    Lomb no necesita muestras equiespaciadas, así que no hay que interpolar la serie
    de intervalos ni rellenar los latidos descartados.
    """
    frecuencias = np.arange(BANDA_LF[0], BANDA_HF[1] + PASO_FRECUENCIA_HZ, PASO_FRECUENCIA_HZ)
    periodograma = signal.lombscargle(t, rr - rr.mean(), 2 * np.pi * frecuencias)
    # Con esta escala, la integral del espectro equivale a la varianza de la serie.
    psd = 2 * (t[-1] - t[0]) / len(t) * periodograma

    def potencia(banda: tuple[float, float]) -> float:
        return float(psd[(frecuencias >= banda[0]) & (frecuencias < banda[1])].sum() * PASO_FRECUENCIA_HZ)

    return potencia(BANDA_LF), potencia(BANDA_HF)


def descriptores_hrv(
    prep: RegistroPreprocesado, cfg: ConfigDescriptores
) -> tuple[np.ndarray, list[str]]:
    """HF y LF/HF, en escala logarítmica, en una ventana centrada en cada época.

    Las épocas con pocos intervalos válidos toman la mediana de la noche, que no
    empuja a las reglas hacia ningún estadio.
    """
    registro = prep.registro
    n_epocas = registro.n_epocas
    conservados = descartar_ectopicos(registro.senales[ECG], prep.latidos, registro.fs)
    t, rr = filtrar_intervalos(*intervalos_rr(conservados, registro.fs))

    matriz = np.full((n_epocas, 2), np.nan)
    centros = (np.arange(n_epocas) + 0.5) * DURACION_EPOCA_S
    izquierda = np.searchsorted(t, centros - cfg.hrv_ventana_s / 2)
    derecha = np.searchsorted(t, centros + cfg.hrv_ventana_s / 2)
    for i, (a, b) in enumerate(zip(izquierda, derecha)):
        if b - a >= MIN_INTERVALOS:
            lf, hf = potencias_lomb(t[a:b], rr[a:b])
            if lf > 0 and hf > 0:
                matriz[i] = np.log10(hf), np.log10(lf / hf)

    faltantes = np.isnan(matriz)
    if faltantes.all():
        matriz[:] = 0.0
    elif faltantes.any():
        medianas = np.nanmedian(matriz, axis=0)
        matriz[faltantes] = np.take(medianas, np.nonzero(faltantes)[1])
    return matriz, ["hrv.hf", "hrv.lf_hf"]
