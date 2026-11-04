"""Verificación de H1 a H4. Cada comparación es paciente por paciente (Wilcoxon).

H1 y H2 usan funciones del Frente B, H4 usa funciones del Frente A. Se escriben acá
para que el diseño experimental quede en un solo lugar.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import wilcoxon

from ..clasificacion.modelo import ModeloEstadios
from ..comun.config import Config, ConfigDescriptores
from ..comun.constantes import BANDAS_EEG, EEG_C4, SIN_ANOTAR
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


def variacion_entre_epocas(log_potencias: np.ndarray, estadios: np.ndarray) -> float:
    """Diferencia absoluta media de log-potencia entre épocas consecutivas del mismo estadio.

    Args:
        log_potencias: Forma (n_epocas, n_bandas), en log10.
    """
    mismo = (estadios[:-1] == estadios[1:]) & (estadios[:-1] != SIN_ANOTAR)
    if not mismo.any():
        return float("nan")
    return float(np.abs(np.diff(log_potencias, axis=0))[mismo].mean())


def h1_variabilidad_espectral(
    registros: list[Registro],
    canal: str = EEG_C4,
    cfg: ConfigDescriptores | None = None,
    por_frecuencia: bool = False,
) -> Comparacion:
    """Variación de la potencia por banda entre épocas consecutivas de un mismo estadio.

    Compara multitaper (a) con Welch (b): H1 espera menor variación en a. Si el
    estadio no cambia, la potencia real tampoco debería cambiar mucho, así que lo que
    varía entre épocas vecinas es sobre todo ruido de estimación.

    Con `por_frecuencia`, compara la log-PSD de cada frecuencia entre 0,5 y 30 Hz en
    lugar de la potencia por banda. Integrar una banda ya promedia muchas frecuencias,
    lo que achica la diferencia entre métodos; frecuencia por frecuencia es donde el
    multitaper debería mostrar su ventaja.
    """
    from ..descriptores import espectro
    from ..preprocesamiento.segmentacion import segmentar

    cfg = cfg or ConfigDescriptores()
    a, b = [], []
    for registro in registros:
        epocas = segmentar(registro.senales[canal], registro.fs, registro.n_epocas)
        for metodo, destino in (("multitaper", a), ("welch", b)):
            if metodo == "multitaper":
                f, psd = espectro.psd_multitaper(epocas, registro.fs, cfg.multitaper_nw, cfg.multitaper_k)
            else:
                f, psd = espectro.psd_welch(epocas, registro.fs)
            if por_frecuencia:
                potencias = psd[:, (f >= 0.5) & (f <= 30.0)]
            else:
                bandas = espectro.potencia_bandas(f, psd, BANDAS_EEG)
                potencias = np.column_stack(list(bandas.values()))
            log_p = np.log10(np.maximum(potencias, np.finfo(float).tiny))
            destino.append(variacion_entre_epocas(log_p, registro.estadios))
    return comparar_por_paciente(np.array(a), np.array(b))


def onda_lenta_modelo(fs: float, duracion_s: float = 2.0, amplitud_uv: float = 80.0) -> np.ndarray:
    """Onda lenta de forma conocida para H2: pico negativo seguido de uno positivo menor.

    Coseno de 0,8 Hz con envolvente gaussiana: dura alrededor de 1 s y supera los
    75 µV pico a pico del criterio de detección.
    """
    t = np.arange(-duracion_s / 2, duracion_s / 2, 1 / fs)
    return -amplitud_uv * np.exp(-(t**2) / (2 * 0.3**2)) * np.cos(2 * np.pi * 0.8 * t)


def error_de_forma(estimada: np.ndarray, real: np.ndarray, max_corrimiento: int) -> float:
    """RMS del error relativo a la RMS de la onda real, tras el mejor corrimiento global.

    El corrimiento global se descuenta porque ningún método puede saber dónde está la
    onda "verdadera"; lo que se evalúa es la forma.
    """
    mejor = np.inf
    for c in range(-max_corrimiento, max_corrimiento + 1):
        error = np.roll(estimada, c) - real
        bordes = slice(max_corrimiento, len(real) - max_corrimiento)  # Sin lo que dio la vuelta.
        mejor = min(mejor, np.sqrt(np.mean(error[bordes] ** 2)))
    return float(mejor / np.sqrt(np.mean(real**2)))


def h2_recuperacion_de_forma(
    eeg_vigilia: list[np.ndarray],
    fs: float,
    n_ondas: int = 40,
    max_corrimiento_s: float = 0.25,
    semilla: int = 0,
) -> Comparacion:
    """Inserta ondas lentas conocidas con corrimientos al azar en EEG real de vigilia.

    Compara el error de forma de Woody (a) con el del promedio ordinario (b). Cada
    realización es una ventana centrada donde se "detectó" la onda, pero la onda real
    está corrida al azar hasta `max_corrimiento_s`, como pasa cuando el ruido mueve el
    pico detectado.
    """
    from ..descriptores.woody import promediado_woody

    rng = np.random.default_rng(semilla)
    modelo = onda_lenta_modelo(fs)
    largo = len(modelo)
    max_corrimiento = int(max_corrimiento_s * fs)
    ventana = largo + 2 * max_corrimiento
    real = np.zeros(ventana)
    real[max_corrimiento : max_corrimiento + largo] = modelo

    a, b = [], []
    for eeg in eeg_vigilia:
        n = min(n_ondas, len(eeg) // ventana)
        if n < 2:
            continue
        realizaciones = np.empty((n, ventana))
        for i in range(n):
            tramo = eeg[i * ventana : (i + 1) * ventana].astype(float)
            corrimiento = rng.integers(-max_corrimiento, max_corrimiento + 1)
            realizaciones[i] = tramo - tramo.mean() + np.roll(real, corrimiento)
        woody = promediado_woody(realizaciones, max_corrimiento).plantilla
        promedio = realizaciones.mean(axis=0)
        a.append(error_de_forma(woody, real, max_corrimiento))
        b.append(error_de_forma(promedio, real, max_corrimiento))
    return comparar_por_paciente(np.array(a), np.array(b))


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
    """Tamaño del promedio del EEG alrededor de cada latido, tras (a) y antes (b) de remover.

    Para aislar el efecto de la etapa 4, "antes" es la señal ya filtrada (y sin
    artefacto ocular, si esa etapa está activa) y "después", la misma señal tras restar
    la plantilla cardíaca. Por paciente se toma el máximo absoluto del promedio,
    promediado entre los canales de EEG: si la remoción funciona, tiende a cero.
    Usa funciones del Frente A (etapas 1, 2, 3 y 4).
    """
    from ..comun.constantes import CANALES_EEG, ECG
    from ..preprocesamiento import cardiaco, exploratorio, filtrado, ocular, qrs

    pre = cfg.preprocesamiento
    a, b = [], []
    for registro in registros:
        latidos = qrs.detectar_latidos(registro.senales[ECG], registro.fs)
        antes = filtrado.filtrar_registro(registro, pre)
        if pre.remover_ocular:
            antes = ocular.remover_artefacto_ocular(antes, pre)
        despues = cardiaco.remover_artefacto_cardiaco(antes, latidos, pre)

        def residuo(r: Registro) -> float:
            return float(np.mean([
                np.abs(exploratorio.promedio_alrededor_de_latidos(
                    r.senales[c], latidos, r.fs, pre.ventana_plantilla_s
                )[1]).max()
                for c in CANALES_EEG
            ]))

        a.append(residuo(despues))
        b.append(residuo(antes))
    return comparar_por_paciente(np.array(a), np.array(b))


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
