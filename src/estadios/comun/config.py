"""Parámetros del pipeline. Los valores por defecto salen del anteproyecto, salvo
`nlms_*`, `wiener_orden`, `ventana_plantilla_s` y `componentes_csp`, que son
provisorios: el anteproyecto no los fija y hay que ajustarlos.

Las hipótesis se prueban comparando configuraciones (con y sin cada etapa), por
eso todo lo que se prende, se apaga o se compara tiene que estar acá.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

from .constantes import EEG_C4, EEG_F4

GRUPOS_DESCRIPTORES = ("espectro", "husos", "ondas", "perifericos", "hrv", "csp")


@dataclass(frozen=True)
class ConfigPreprocesamiento:
    pasa_altos_hz: float = 0.5
    pasa_bajos_hz: float | None = None  # Se fija con el espectro de la etapa 1.
    notch_hz: float = 50.0
    tipo_filtro: Literal["iir", "fir"] = "iir"

    remover_ocular: bool = True
    metodo_ocular: Literal["nlms", "wiener"] = "nlms"
    nlms_orden: int = 5
    nlms_mu: float = 0.01
    wiener_orden: int = 5

    remover_cardiaco: bool = True
    latidos_plantilla: int = 30
    ventana_plantilla_s: tuple[float, float] = (-0.2, 0.6)


@dataclass(frozen=True)
class ConfigDescriptores:
    metodo_espectro: Literal["multitaper", "welch"] = "multitaper"
    multitaper_nw: float = 15.0  # Épocas de 30 s con 1 Hz de resolución.
    multitaper_k: int = 29

    canal_husos: str = EEG_C4
    husos_banda_hz: tuple[float, float] = (11.0, 16.0)
    husos_duracion_s: tuple[float, float] = (0.5, 2.0)

    canal_ondas: str = EEG_F4
    ondas_banda_hz: tuple[float, float] = (0.2, 4.0)
    ondas_pico_uv: float = -60.0
    ondas_pico_a_pico_uv: float = 75.0

    hrv_ventana_s: float = 300.0


@dataclass(frozen=True)
class ConfigClasificacion:
    grupos: tuple[str, ...] = GRUPOS_DESCRIPTORES
    pares_csp: tuple[tuple[int, int], ...] = ((1, 4),)  # (N1, R)
    componentes_csp: int = 2  # Por extremo: se usan 2 * componentes_csp filtros.
    usar_hmm: bool = True


@dataclass(frozen=True)
class Config:
    preprocesamiento: ConfigPreprocesamiento = field(default_factory=ConfigPreprocesamiento)
    descriptores: ConfigDescriptores = field(default_factory=ConfigDescriptores)
    clasificacion: ConfigClasificacion = field(default_factory=ConfigClasificacion)

    def huella_descriptores(self) -> str:
        """Identifica el caché: cambia solo si cambia cómo se calculan los descriptores."""
        contenido = {
            "preprocesamiento": asdict(self.preprocesamiento),
            "descriptores": asdict(self.descriptores),
        }
        texto = json.dumps(contenido, sort_keys=True)
        return hashlib.sha256(texto.encode()).hexdigest()[:12]
