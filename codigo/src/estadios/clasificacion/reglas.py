"""Etapa 10: clasificación por reglas inspiradas en el manual AASM.

Reemplaza al LDA del anteproyecto: la cátedra no permite técnicas de inteligencia
artificial en la clasificación del TP cuatrimestral, y el LDA es un clasificador
supervisado que ajusta sus parámetros con épocas etiquetadas. Acá nada se entrena: cada
umbral lo elige el diseñador mirando los histogramas del exploratorio, igual que se
elige la frecuencia de corte de un filtro.

Las reglas se evalúan en orden y gana la primera que se cumple:

1. W   : EMG alto y potencia alfa relativa alta (occipital).
2. N3  : potencia delta relativa alta (frontal) y muchas ondas lentas.
3. REM : atonía (EMG bajo), movimientos oculares (energía de EOG alta, o LF/HF alto si
         se usa la HRV) y pocos husos.
4. N2  : husos.
5. N1  : lo que no cumplió ninguna regla (en la AASM, N1 se define en buena parte por
         descarte).

Después, CSP desempata entre N1 y REM cuando su evidencia es fuerte (H3): la época\ncambia de estadio si la potencia de la señal CSP queda claramente del lado del otro.

Los umbrales están en percentiles del propio registro, para compensar diferencias de
amplitud entre pacientes y montajes.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from ..comun.config import ConfigClasificacion
from ..comun.constantes import N1, N2, N3, R, W

# Nombre de cada regla, en el orden en que se evalúan. "csp" marca las épocas que
# cambió el desempate.
REGLAS = ("W", "N3", "R", "N2", "N1", "csp")


def percentil_por_registro(matriz: np.ndarray) -> np.ndarray:
    """Percentil (0 a 100) de cada valor dentro de su columna, es decir, de su noche.

    Returns:
        Siempre 2D, de forma (n_epocas, n_columnas).
    """
    matriz = np.asarray(matriz, dtype=float)
    if matriz.ndim == 1:
        matriz = matriz[:, None]
    n = matriz.shape[0]
    if n == 1:
        return np.full_like(matriz, 50.0, dtype=float)
    return (rankdata(matriz, axis=0) - 1) / (n - 1) * 100


def buscar_descriptor(nombres: list[str], grupo: str, base: str, canal: str | None = None) -> int | None:
    """Índice de "<grupo>.<canal>_<base>"; si no está, de "<grupo>.<base>" o de cualquier
    "<grupo>.*_<base>". None si el grupo no lo tiene."""
    candidatos = []
    if canal:
        candidatos.append(f"{grupo}.{canal}_{base}")
    candidatos.append(f"{grupo}.{base}")
    for candidato in candidatos:
        if candidato in nombres:
            return nombres.index(candidato)
    for i, nombre in enumerate(nombres):
        if nombre.startswith(f"{grupo}.") and nombre.endswith(f"_{base}"):
            return i
    return None


def descriptores_de_reglas(
    matriz: np.ndarray, nombres: list[str], cfg: ConfigClasificacion
) -> dict[str, np.ndarray]:
    """Percentil de cada descriptor que usan las reglas. Faltan los grupos apagados."""
    pedidos = {
        "emg": ("perifericos", "amplitud_emg", None),
        "eog": ("perifericos", "energia_eog", None),
        "alfa": ("espectro", "alfa_rel", cfg.canal_alfa),
        "delta": ("espectro", "delta_rel", cfg.canal_delta),
        "ondas": ("ondas", "cantidad", None),
        "husos": ("husos", "cantidad", None),
        "lf_hf": ("hrv", "lf_hf", None),
    }
    percentiles = {}
    for clave, (grupo, base, canal) in pedidos.items():
        if grupo not in cfg.grupos:
            continue
        i = buscar_descriptor(nombres, grupo, base, canal)
        if i is not None:
            percentiles[clave] = percentil_por_registro(matriz[:, i])[:, 0]
    return percentiles


def aplicar_reglas(
    p: dict[str, np.ndarray], cfg: ConfigClasificacion, n_epocas: int
) -> tuple[np.ndarray, np.ndarray]:
    """Árbol de decisión de la AASM sobre los percentiles.

    Una regla cuyo descriptor falta no se cumple nunca (salvo la de REM, que usa la HRV
    o el EOG, lo que haya).

    Returns:
        Estadio de cada época e índice en REGLAS de la regla que lo decidió.
    """
    falso = np.zeros(n_epocas, dtype=bool)

    def mayor(clave, umbral):
        return p[clave] >= umbral if clave in p else falso

    def menor(clave, umbral):
        return p[clave] <= umbral if clave in p else falso

    movimientos = mayor("eog", cfg.r_eog_min) | mayor("lf_hf", cfg.r_lf_hf_min)
    pocos_husos = menor("husos", cfg.r_husos_max) if "husos" in p else ~falso
    condiciones = [
        (W, mayor("emg", cfg.w_emg_min) & mayor("alfa", cfg.w_alfa_min)),
        (N3, mayor("delta", cfg.n3_delta_min) & mayor("ondas", cfg.n3_ondas_min)),
        (R, menor("emg", cfg.r_emg_max) & movimientos & pocos_husos),
        (N2, mayor("husos", cfg.n2_husos_min)),
    ]
    estadios = np.full(n_epocas, N1)
    regla = np.full(n_epocas, REGLAS.index("N1"))
    pendiente = np.ones(n_epocas, dtype=bool)
    for i, (estadio, cumple) in enumerate(condiciones):
        elegidas = pendiente & cumple
        estadios[elegidas] = estadio
        regla[elegidas] = i
        pendiente &= ~elegidas
    return estadios, regla


def puntaje_csp(log_potencias: np.ndarray) -> np.ndarray:
    """Cuánto se parece cada época al primer estadio del par: los últimos filtros de
    `ajustar_csp` maximizan su potencia y los primeros, la del segundo."""
    mitad = log_potencias.shape[1] // 2
    return log_potencias[:, mitad:].mean(axis=1) - log_potencias[:, :mitad].mean(axis=1)


def referencia_csp(puntaje: np.ndarray, estadios: np.ndarray, par: tuple[int, int]) -> tuple[float, float]:
    """Punto medio entre las medianas del puntaje CSP de los dos estadios y el ancho
    típico de cada uno (promedio de sus rangos intercuartiles). Se calcula con las mismas
    épocas de entrenamiento que los filtros CSP, como parte de la etapa 8.

    El ancho se mide dentro de cada estadio y no con los dos juntos: mezclados, el rango
    intercuartil queda del tamaño de la distancia entre los grupos y casi ninguna época
    superaría el margen.
    """
    a, b = puntaje[estadios == par[0]], puntaje[estadios == par[1]]
    centro = (np.median(a) + np.median(b)) / 2
    escala = np.mean([np.subtract(*np.percentile(x, [75, 25])) for x in (a, b)])
    return float(centro), float(max(escala, np.finfo(float).eps))


def desempatar_con_csp(
    estadios: np.ndarray,
    regla: np.ndarray,
    puntaje: np.ndarray,
    par: tuple[int, int],
    centro: float,
    escala: float,
    margen: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Entre las épocas asignadas a alguno de los dos estadios del par, cambia las que
    CSP ubica claramente del otro lado: a más de `margen` rangos intercuartiles del
    punto medio."""
    estadios, regla = estadios.copy(), regla.copy()
    z = (puntaje - centro) / escala
    del_par = np.isin(estadios, par)
    hacia_a = del_par & (estadios == par[1]) & (z >= margen)
    hacia_b = del_par & (estadios == par[0]) & (z <= -margen)
    estadios[hacia_a], estadios[hacia_b] = par[0], par[1]
    regla[hacia_a | hacia_b] = REGLAS.index("csp")
    return estadios, regla


def cruce_de_histogramas(a: np.ndarray, b: np.ndarray, bins: int = 50) -> float:
    """Punto donde se cruzan los histogramas normalizados de dos estadios.

    Es una ayuda para elegir un umbral a mano en el notebook, no un ajuste automático:
    el valor elegido se copia en `ConfigClasificacion`.
    """
    bordes = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), bins + 1)
    ha, _ = np.histogram(a, bordes, density=True)
    hb, _ = np.histogram(b, bordes, density=True)
    centros = (bordes[:-1] + bordes[1:]) / 2
    # Entre las medianas, el bin donde las densidades están más cerca.
    lo, hi = sorted((np.median(a), np.median(b)))
    entre = (centros >= lo) & (centros <= hi)
    if not entre.any():
        return float((lo + hi) / 2)
    i = np.flatnonzero(entre)[np.argmin(np.abs(ha - hb)[entre])]
    return float(centros[i])
