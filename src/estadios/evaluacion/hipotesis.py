"""Verificación de H1 a H4. Cada comparación es paciente por paciente (Wilcoxon).

H1 y H2 usan funciones del Frente B, H4 usa funciones del Frente A. Se escriben acá
para que el diseño experimental quede en un solo lugar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..comun.config import Config
from ..comun.contratos import Registro, TablaDescriptores


@dataclass
class Comparacion:
    """Resultado de comparar dos variantes sobre los mismos pacientes."""

    variante_a: np.ndarray  # Un valor por paciente.
    variante_b: np.ndarray
    p_valor: float


def comparar_por_paciente(a: np.ndarray, b: np.ndarray) -> Comparacion:
    """Prueba de Wilcoxon de muestras apareadas."""
    raise NotImplementedError("Sección 3")


def h1_variabilidad_espectral(registros: list[Registro]) -> Comparacion:
    """Variación de la potencia por banda entre épocas consecutivas de un mismo estadio.

    Compara multitaper (a) con Welch (b): H1 espera menor variación en a.
    """
    raise NotImplementedError("Sección 3")


def h2_recuperacion_de_forma(eeg_vigilia: list[np.ndarray], fs: float) -> Comparacion:
    """Inserta ondas lentas conocidas con corrimientos al azar en EEG real de vigilia.

    Compara el error de forma de Woody (a) con el del promedio ordinario (b).
    """
    raise NotImplementedError("Sección 3")


def h3_aporte_csp(
    entrenamiento: list[TablaDescriptores], prueba: list[TablaDescriptores], cfg: Config
) -> dict[str, Comparacion]:
    """F1 de N1 y de REM con CSP (a) y sin CSP (b). Claves "N1" y "R"."""
    raise NotImplementedError("Sección 3")


def h4_residuo_cardiaco(registros: list[Registro], cfg: Config) -> Comparacion:
    """Tamaño del promedio del EEG alrededor de cada latido, tras (a) y antes (b) de remover."""
    raise NotImplementedError("Sección 3")


def h4_aporte_al_acuerdo(
    con_etapas: tuple[list[TablaDescriptores], list[TablaDescriptores]],
    sin_etapas: tuple[list[TablaDescriptores], list[TablaDescriptores]],
    cfg: Config,
) -> Comparacion:
    """κ con (a) y sin (b) la remoción de artefactos y la HRV.

    Cada argumento es (entrenamiento, prueba), calculados con su configuración.
    """
    raise NotImplementedError("Sección 3")
