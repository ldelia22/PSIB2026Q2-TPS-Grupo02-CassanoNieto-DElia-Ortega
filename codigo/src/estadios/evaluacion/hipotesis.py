"""Verificación de H1 a H4. Cada comparación es paciente por paciente (Wilcoxon).

H1 y H2 usan funciones del Frente B, H4 usa funciones del Frente A. Se escriben acá
para que el diseño experimental quede en un solo lugar.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import wilcoxon

from ..clasificacion.modelo import ModeloEstadios
from ..comun.config import Config
from ..comun.contratos import Registro, TablaDescriptores


@dataclass
class Comparacion:
    """Resultado de comparar dos variantes sobre los mismos pacientes."""

    variante_a: np.ndarray  # Un valor por paciente.
    variante_b: np.ndarray
    p_valor: float


def comparar_por_paciente(a: np.ndarray, b: np.ndarray) -> Comparacion:
    """Prueba de Wilcoxon de muestras apareadas (bilateral)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.allclose(a, b):  # Sin diferencias, Wilcoxon no está definido.
        return Comparacion(a, b, 1.0)
    return Comparacion(a, b, float(wilcoxon(a, b).pvalue))


def _por_paciente(cfg_clasificacion, entrenamiento, prueba, metrica) -> np.ndarray:
    """Ajusta con `entrenamiento` y aplica `metrica(y_real, y_pred)` a cada paciente de prueba."""
    modelo = ModeloEstadios(cfg_clasificacion).ajustar(entrenamiento)
    return np.array([metrica(t.estadios, modelo.predecir(t)) for t in prueba])


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
    """F1 de N1 y de REM con CSP (a) y sin CSP (b). Claves "N1" y "R".

    Con CSP, el desempate N1/REM de `reglas.desempatar_con_csp` está activo.
    """
    from .metricas import f1_por_estadio

    con = replace(cfg.clasificacion, grupos=tuple({*cfg.clasificacion.grupos, "csp"}))
    sin = replace(cfg.clasificacion, grupos=tuple(g for g in cfg.clasificacion.grupos if g != "csp"))
    resultado = {}
    for estadio in ("N1", "R"):
        def metrica(y, p, e=estadio):
            return f1_por_estadio(y, p)[e]
        resultado[estadio] = comparar_por_paciente(
            _por_paciente(con, entrenamiento, prueba, metrica),
            _por_paciente(sin, entrenamiento, prueba, metrica),
        )
    return resultado


def h4_residuo_cardiaco(registros: list[Registro], cfg: Config) -> Comparacion:
    """Tamaño del promedio del EEG alrededor de cada latido, tras (a) y antes (b) de remover."""
    raise NotImplementedError("Sección 3")


def h4_aporte_al_acuerdo(
    con_etapas: tuple[list[TablaDescriptores], list[TablaDescriptores]],
    sin_etapas: tuple[list[TablaDescriptores], list[TablaDescriptores]],
    cfg: Config,
) -> Comparacion:
    """κ con (a) y sin (b) la remoción de artefactos y la HRV.

    Cada argumento es (entrenamiento, prueba), calculados con su configuración. Sin
    etapas, además, la regla de REM no puede usar la HRV.
    """
    from .metricas import kappa

    sin = replace(cfg.clasificacion, grupos=tuple(g for g in cfg.clasificacion.grupos if g != "hrv"))
    return comparar_por_paciente(
        _por_paciente(cfg.clasificacion, *con_etapas, kappa),
        _por_paciente(sin, *sin_etapas, kappa),
    )
