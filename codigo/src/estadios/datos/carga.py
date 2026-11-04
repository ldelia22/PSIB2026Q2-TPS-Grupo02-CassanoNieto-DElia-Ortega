"""Lectura de registros de la HMC: `SNxxx.edf` y `SNxxx_sleepscoring.txt`."""

from __future__ import annotations

import os
from pathlib import Path

import mne
import numpy as np

from ..comun.constantes import CANALES, DURACION_EPOCA_S, ETIQUETA_A_ESTADIO, SIN_ANOTAR
from ..comun.contratos import Registro

# La base pesa unos 100 MB por registro: con ESTADIOS_DATOS se la puede tener en
# otro disco, fuera del repo y de carpetas sincronizadas con la nube.
DIR_DATOS = Path(os.environ.get("ESTADIOS_DATOS", "data"))
DIR_REGISTROS = DIR_DATOS / "recordings"


def ids_disponibles(directorio: Path = DIR_REGISTROS) -> list[str]:
    """Registros ya descargados, ordenados."""
    return sorted(p.stem for p in directorio.glob("SN???.edf"))


def leer_anotaciones(ruta: Path) -> tuple[np.ndarray, float | None, float | None]:
    """Lee el archivo de anotaciones de los técnicos.

    Returns:
        Estadio de cada época, e instantes (s) de luces apagadas y encendidas.
    """
    epocas: dict[int, int] = {}
    apagadas = encendidas = None
    # Formato: Date, Time, Recording onset, Duration, Annotation, Linked channel
    for linea in ruta.read_text().splitlines()[1:]:
        campos = [c.strip() for c in linea.split(",")]
        if len(campos) < 5:
            continue
        inicio, anotacion = float(campos[2]), campos[4]
        if anotacion in ETIQUETA_A_ESTADIO:
            epocas[round(inicio / DURACION_EPOCA_S)] = ETIQUETA_A_ESTADIO[anotacion]
        elif anotacion == "Lights off":
            apagadas = inicio
        elif anotacion == "Lights on":
            encendidas = inicio

    estadios = np.full(max(epocas) + 1 if epocas else 0, SIN_ANOTAR, dtype=int)
    for indice, estadio in epocas.items():
        estadios[indice] = estadio
    return estadios, apagadas, encendidas


def cargar_registro(id_registro: str, directorio: Path = DIR_REGISTROS) -> Registro:
    """Carga un registro con las señales en µV.

    Si señal y anotaciones no cubren la misma cantidad de épocas, se queda con las
    épocas que tienen ambas.
    """
    crudo = mne.io.read_raw_edf(directorio / f"{id_registro}.edf", preload=True, verbose="ERROR")
    fs = float(crudo.info["sfreq"])
    senales = dict(zip(CANALES, crudo.get_data(picks=list(CANALES), units="uV")))

    estadios, apagadas, encendidas = leer_anotaciones(
        directorio / f"{id_registro}_sleepscoring.txt"
    )
    epocas_completas = int(crudo.n_times // (fs * DURACION_EPOCA_S))
    return Registro(
        id=id_registro,
        fs=fs,
        senales=senales,
        estadios=estadios[:epocas_completas],
        luces_apagadas_s=apagadas,
        luces_encendidas_s=encendidas,
    )
