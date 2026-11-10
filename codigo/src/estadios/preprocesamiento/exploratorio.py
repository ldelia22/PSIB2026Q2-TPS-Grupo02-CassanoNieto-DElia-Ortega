"""Etapa 1. Análisis exploratorio del EEG. Define el filtrado y qué artefactos se remueven."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal

from ..comun.constantes import EOG_E1, EOG_E2, CANALES_EEG, SIN_ANOTAR
from ..comun.contratos import Registro
from .segmentacion import segmentar

VENTANA_WELCH_S = 4.0  # Resolución de 0,25 Hz, suficiente para aislar la línea eléctrica.

# Campos por canal del encabezado EDF, en orden, con su largo en bytes.
CAMPOS_EDF = (
    ("canal", 16), ("transductor", 80), ("unidad", 8), ("minimo", 8), ("maximo", 8),
    ("minimo_digital", 8), ("maximo_digital", 8), ("prefiltro", 80),
)


def encabezado_edf(ruta: Path) -> dict[str, dict[str, str | float]]:
    """Unidad, rango físico y filtros del equipo que declara el encabezado EDF de cada canal.

    El rango físico es el valor que corresponde al entero más grande que guarda el
    archivo, así que ninguna muestra puede superarlo.
    """
    with open(ruta, "rb") as archivo:
        n_canales = int(archivo.read(256)[252:256])
        texto = archivo.read(256 * n_canales).decode("latin-1")
    campos, posicion = {}, 0
    for nombre, largo in CAMPOS_EDF:
        campos[nombre] = [
            texto[posicion + k * largo : posicion + (k + 1) * largo].strip() for k in range(n_canales)
        ]
        posicion += n_canales * largo
    return {
        canal: {
            "unidad": campos["unidad"][k],
            "minimo": float(campos["minimo"][k]),
            "maximo": float(campos["maximo"][k]),
            "prefiltro": campos["prefiltro"][k],
        }
        for k, canal in enumerate(campos["canal"])
    }


def espectro_del_registro(senal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """PSD de toda la noche. La mediana evita que unos pocos movimientos dominen el resultado."""
    return signal.welch(senal, fs, nperseg=int(VENTANA_WELCH_S * fs), average="median")


def prominencia_de_linea(
    f: np.ndarray, psd: np.ndarray, candidatas: tuple[float, ...] = (50.0, 60.0)
) -> dict[float, float]:
    """Cuántos dB sobresale el espectro en cada frecuencia candidata respecto de sus vecinas."""
    resultado = {}
    for f0 in candidatas:
        distancia = np.abs(f - f0)
        pico = psd[distancia <= 0.5].max()
        fondo = np.median(psd[(distancia > 1.0) & (distancia <= 4.0)])
        resultado[f0] = float(10 * np.log10(pico / fondo)) if pico > 0 and fondo > 0 else 0.0
    return resultado


def espectro_por_estadio(registro: Registro, canal: str) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """PSD mediana del canal en cada estadio. Devuelve las frecuencias y un diccionario por estadio."""
    epocas = segmentar(registro.senales[canal], registro.fs, registro.n_epocas)
    f, psd = signal.welch(epocas, registro.fs, nperseg=int(VENTANA_WELCH_S * registro.fs), axis=-1)
    estadios = np.unique(registro.estadios[registro.estadios != SIN_ANOTAR])
    return f, {int(e): np.median(psd[registro.estadios == e], axis=0) for e in estadios}


def fraccion_cerca_del_limite(senal: np.ndarray, limite_uv: float) -> float:
    """Fracción de muestras cuyo valor absoluto alcanza `limite_uv`, para medir saturación."""
    return float(np.mean(np.abs(senal) >= limite_uv))


def correlacion_eeg_eog(registro: Registro, estadio: int | None = None) -> dict[str, float]:
    """Correlación de cada EEG con E1 - E2, opcionalmente dentro de un solo estadio.

    Antes se quita lo que está por debajo de 0,5 Hz, porque la deriva lenta de los
    electrodos dominaría la correlación y no dice nada sobre el artefacto ocular.
    """
    sos = signal.butter(2, 0.5, "highpass", fs=registro.fs, output="sos")
    eog = signal.sosfiltfilt(sos, registro.senales[EOG_E1] - registro.senales[EOG_E2])
    eog = segmentar(eog, registro.fs, registro.n_epocas)

    elegidas = np.ones(registro.n_epocas, dtype=bool)
    if estadio is not None:
        elegidas = registro.estadios == estadio

    resultado = {}
    for canal in CANALES_EEG:
        eeg = segmentar(signal.sosfiltfilt(sos, registro.senales[canal]), registro.fs, registro.n_epocas)
        resultado[canal] = float(np.corrcoef(eeg[elegidas].ravel(), eog[elegidas].ravel())[0, 1])
    return resultado


def promedio_alrededor_de_latidos(
    senal: np.ndarray,
    latidos: np.ndarray,
    fs: float,
    ventana_s: tuple[float, float] = (-0.2, 0.6),
) -> tuple[np.ndarray, np.ndarray]:
    """Promedio de la señal sincronizado con los latidos, y sus tiempos relativos en segundos.

    El EEG no está sincronizado con el corazón y se cancela al promediar. Si queda una
    forma repetible, es artefacto cardíaco.
    """
    desde, hasta = round(ventana_s[0] * fs), round(ventana_s[1] * fs)
    latidos = np.asarray(latidos, dtype=int)
    latidos = latidos[(latidos + desde >= 0) & (latidos + hasta < len(senal))]
    desplazamientos = np.arange(desde, hasta + 1)
    promedio = senal[latidos[:, None] + desplazamientos].mean(axis=0)
    return desplazamientos / fs, promedio
