"""Parámetros del pipeline. Los del preprocesamiento se eligieron con el análisis
exploratorio de `notebooks/01_frente_A_senales.ipynb`. Los umbrales de clasificación y
`componentes_csp` siguen siendo provisorios.

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
    pasa_altos_hz: float = 0.2  # Las ondas lentas se detectan desde 0,2 Hz.
    pasa_bajos_hz: float | None = 45.0  # Sobre 40 Hz ningún descriptor usa el EEG.
    notch_hz: float | None = 50.0  # Medido en el espectro crudo, no hay pico en 60 Hz.
    emg_pasa_altos_hz: float = 10.0  # Recomendación de la AASM para el EMG de mentón.
    tipo_filtro: Literal["iir", "fir"] = "iir"

    remover_ocular: bool = True
    metodo_ocular: Literal["nlms", "wiener"] = "wiener"  # NLMS agregó potencia delta en W y REM.
    nlms_orden: int = 5
    nlms_mu: float = 0.001
    wiener_orden: int = 5
    umbral_artefacto_uv: float = 500.0  # Por encima, el filtro ocular no se ajusta.
    ocular_umbral_relativo: float = 2.0  # Veces la mediana del valor eficaz del EOG.
    ocular_ventana_s: float = 2.0

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
    """Clasificador por reglas AASM (etapa 10). Reemplaza al LDA + HMM del anteproyecto,
    porque la cátedra no permite técnicas de IA en la clasificación del TP cuatrimestral.

    Cada umbral está en las unidades con que se interpreta el descriptor:

    - potencias relativas (δ, α): fracción de la potencia total, de 0 a 1;
    - husos y ondas lentas: cantidad de eventos detectados en la época de 30 s;
    - EMG, EOG y LF/HF: veces la mediana de la noche del paciente (1 = lo típico de
      esa noche), porque dependen de la amplitud y del electrodo.

    Los valores se eligen mirando los histogramas de
    `notebooks/03_frente_C_clasificacion.ipynb` con los pacientes de entrenamiento y se
    copian acá. Los actuales son provisorios.
    """

    # Grupos de descriptores que pueden usar las reglas. Sacar "hrv" apaga esa parte de
    # la regla de REM (H4). CSP está apagado por defecto: con datos reales el desempate
    # empeoraba la clasificación (H3 lo prende para compararlo).
    grupos: tuple[str, ...] = tuple(g for g in GRUPOS_DESCRIPTORES if g != "csp")

    # Canal de cada descriptor espectral (si la tabla no lo tiene, se usa el que haya).
    canal_alfa: str = "o2"
    canal_delta: str = "f4"

    # W: EMG alto y, además, ritmo α (ojos cerrados) o parpadeos (ojos abiertos).
    w_emg_min: float = 1.5  # veces la mediana
    w_alfa_min: float = 0.15  # fracción
    w_eog_min: float = 3.0  # veces la mediana
    # N3: δ relativa alta y ondas lentas.
    n3_delta_min: float = 0.7  # fracción
    n3_ondas_min: float = 3.0  # ondas por época
    # REM: atonía, movimientos oculares (o LF/HF alto si se usa la HRV) y sin husos.
    r_emg_max: float = 0.7  # veces la mediana
    r_eog_min: float = 2.0  # veces la mediana
    r_lf_hf_min: float = 1.5  # veces la mediana
    r_husos_max: float = 0.0  # husos por época
    # N2: al menos un huso.
    n2_husos_min: float = 1.0  # husos por época
    # Continuidad de N2 (AASM): una época sin rasgos propios que sigue a N2 sigue
    # siendo N2, en lugar de caer en N1 por descarte.
    continuidad_n2: bool = True

    # CSP: desempata N1/REM cuando la potencia CSP está lejos del punto medio entre los
    # dos estadios (en anchos típicos de cada estadio). Solo actúa si "csp" está en grupos.
    pares_csp: tuple[tuple[int, int], ...] = ((1, 4),)  # (N1, R)
    componentes_csp: int = 2  # Por extremo: se usan 2 * componentes_csp filtros.
    csp_margen: float = 1.0

    # Suavizado temporal (reemplaza al HMM).
    suavizado: Literal["reglas", "moda", "ninguno"] = "reglas"
    rem_min_epocas: int = 2
    ventana_moda: int = 5


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
