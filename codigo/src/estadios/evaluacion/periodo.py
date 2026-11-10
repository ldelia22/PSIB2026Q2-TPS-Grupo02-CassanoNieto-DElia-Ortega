"""Decisión 3: clasificar y evaluar solo entre luces apagadas y encendidas.

Las anotaciones incluyen vigilia antes de apagar y después de prender las luces. Esas
épocas son fáciles de clasificar e inflarían κ, y además corren los percentiles de la
noche hacia la vigilia. Por eso, por defecto, la tabla de cada paciente se recorta a
ese período antes de elegir umbrales, clasificar y evaluar.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ..comun.constantes import DURACION_EPOCA_S
from ..comun.contratos import TablaDescriptores
from ..datos.carga import DIR_REGISTROS, leer_anotaciones


def mascara_periodo_de_sueno(
    n_epocas: int, luces_apagadas_s: float | None, luces_encendidas_s: float | None
) -> np.ndarray:
    """Épocas entre luces apagadas y encendidas. Si falta alguna marca, no se recorta de ese lado.

    Las marcas se redondean a la época más cercana, igual que las anotaciones de estadio.
    """
    indice = np.arange(n_epocas)
    desde = round(luces_apagadas_s / DURACION_EPOCA_S) if luces_apagadas_s is not None else 0
    hasta = round(luces_encendidas_s / DURACION_EPOCA_S) if luces_encendidas_s is not None else n_epocas
    return (indice >= desde) & (indice < hasta)


def recortar_tabla(tabla: TablaDescriptores, mascara: np.ndarray) -> TablaDescriptores:
    """Copia de la tabla solo con las épocas marcadas."""
    return replace(
        tabla,
        matriz=tabla.matriz[mascara],
        estadios=tabla.estadios[mascara],
        covarianzas_eeg=tabla.covarianzas_eeg[mascara],
        no_estacionaria=tabla.no_estacionaria[mascara],
    )


def recortar_a_periodo_de_sueno(
    tabla: TablaDescriptores, dir_registros: Path = DIR_REGISTROS
) -> TablaDescriptores:
    """Recorta la tabla a las épocas entre luces apagadas y encendidas.

    Lee las marcas del archivo de anotaciones del registro. Si no existe (por ejemplo,
    con datos sintéticos), devuelve la tabla sin cambios.
    """
    ruta = Path(dir_registros) / f"{tabla.id}_sleepscoring.txt"
    if not ruta.exists():
        return tabla
    _, apagadas, encendidas = leer_anotaciones(ruta)
    return recortar_tabla(tabla, mascara_periodo_de_sueno(len(tabla.estadios), apagadas, encendidas))
