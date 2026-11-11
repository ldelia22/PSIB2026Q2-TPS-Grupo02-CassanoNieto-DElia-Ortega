"""Etapa 10: clasificación por reglas inspiradas en el manual AASM.

Reemplaza al LDA del anteproyecto: la cátedra no permite técnicas de inteligencia
artificial en la clasificación del TP cuatrimestral, y el LDA es un clasificador
supervisado que ajusta sus parámetros con épocas etiquetadas. Acá nada se entrena: cada
umbral lo elige el diseñador mirando los histogramas del exploratorio, igual que se
elige la frecuencia de corte de un filtro.

Las reglas se evalúan en orden y gana la primera que se cumple:

1. W   : EMG alto y, además, ritmo α occipital (ojos cerrados) o parpadeos (ojos
         abiertos).
2. N3  : potencia δ relativa alta (frontal) y ondas lentas.
3. REM : atonía (EMG bajo), movimientos oculares (energía de EOG alta, o LF/HF alto si
         se usa la HRV) y ningún huso.
4. N2  : al menos un huso.
5. N1  : lo que no cumplió ninguna regla (en la AASM, N1 se define en buena parte por
         descarte), salvo que siga a una época N2: por la regla de continuidad de la
         AASM, sigue siendo N2.

Si "csp" está en los grupos, CSP desempata entre N1 y REM cuando su evidencia es fuerte
(H3). Está apagado por defecto porque con datos reales empeoraba la clasificación.

Cada descriptor se usa en las unidades con que se interpreta (ver `ConfigClasificacion`).
Una primera versión expresaba todos como percentiles de la noche, pero con datos reales
eso forzaba una fracción fija de cada noche en cada regla, sin importar cuánto N3 o N2
tuviera el paciente. Ahora solo EMG, EOG y LF/HF, que dependen de la amplitud, se
normalizan por paciente, dividiendo por la mediana de su noche.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from ..comun.config import ConfigClasificacion
from ..comun.constantes import N1, N2, N3, R, W

# Nombre de cada regla, en el orden en que se evalúan. "csp" marca las épocas que
# cambió el desempate y "N2 cont.", las que la continuidad pasó de N1 a N2.
REGLAS = ("W", "N3", "R", "N2", "N1", "csp", "N2 cont.")

# Descriptores que dependen de la amplitud (están en log10) y se expresan en veces la
# mediana de la noche.
RELATIVOS_A_LA_NOCHE = ("emg", "eog", "lf_hf")


def percentil_por_registro(matriz: np.ndarray) -> np.ndarray:
    """Percentil (0 a 100) de cada valor dentro de su columna, es decir, de su noche.

    Returns:
        Siempre 2D, de forma (n_epocas, n_columnas).
    """
    matriz = np.asarray(matriz, dtype=float)
    if matriz.ndim == 1:
        matriz = matriz[:, None]
    salida = np.full(matriz.shape, np.nan)
    for j in range(matriz.shape[1]):
        # Una época sin valor (NaN) queda en NaN: no cumple ninguna regla que use ese
        # descriptor, y no corre el percentil de las demás.
        validos = ~np.isnan(matriz[:, j])
        n = int(validos.sum())
        if n == 1:
            salida[validos, j] = 50.0
        elif n > 1:
            salida[validos, j] = (rankdata(matriz[validos, j]) - 1) / (n - 1) * 100
    return salida


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


def veces_la_mediana(log10_valores: np.ndarray) -> np.ndarray:
    """Valor de cada época dividido por la mediana de la noche (1 = lo típico).

    Recibe el descriptor en log10, como lo entrega el Frente B. Las épocas sin valor
    (NaN) quedan en NaN y no cumplen ninguna regla.
    """
    x = np.asarray(log10_valores, dtype=float)
    if np.isnan(x).all():
        return x.copy()
    return 10 ** (x - np.nanmedian(x))


def descriptores_de_reglas(
    matriz: np.ndarray, nombres: list[str], cfg: ConfigClasificacion
) -> dict[str, np.ndarray]:
    """Cada descriptor que usan las reglas, en las unidades de `ConfigClasificacion`.

    Faltan los de grupos apagados o que la tabla no tiene.
    """
    pedidos = {
        "emg": ("perifericos", "amplitud_emg", None),
        "eog": ("perifericos", "energia_eog", None),
        "alfa": ("espectro", "alfa_rel", cfg.canal_alfa),
        "delta": ("espectro", "delta_rel", cfg.canal_delta),
        "ondas": ("ondas", "cantidad", None),
        "husos": ("husos", "cantidad", None),
        "lf_hf": ("hrv", "lf_hf", None),
    }
    valores = {}
    for clave, (grupo, base, canal) in pedidos.items():
        if grupo not in cfg.grupos:
            continue
        i = buscar_descriptor(nombres, grupo, base, canal)
        if i is None:
            continue
        columna = matriz[:, i].astype(float)
        valores[clave] = veces_la_mediana(columna) if clave in RELATIVOS_A_LA_NOCHE else columna
    return valores


def aplicar_reglas(
    v: dict[str, np.ndarray], cfg: ConfigClasificacion, n_epocas: int
) -> tuple[np.ndarray, np.ndarray]:
    """Árbol de decisión de la AASM sobre los descriptores.

    Una regla cuyo descriptor falta no se cumple nunca. Las alternativas (α o parpadeos
    en W; EOG o LF/HF en REM) usan la que haya.

    Returns:
        Estadio de cada época e índice en REGLAS de la regla que lo decidió.
    """
    falso = np.zeros(n_epocas, dtype=bool)

    def mayor(clave, umbral):
        return v[clave] >= umbral if clave in v else falso

    def menor(clave, umbral):
        return v[clave] <= umbral if clave in v else falso

    vigilia = mayor("alfa", cfg.w_alfa_min) | mayor("eog", cfg.w_eog_min)
    movimientos = mayor("eog", cfg.r_eog_min) | mayor("lf_hf", cfg.r_lf_hf_min)
    sin_husos = menor("husos", cfg.r_husos_max) if "husos" in v else ~falso
    condiciones = [
        (W, mayor("emg", cfg.w_emg_min) & vigilia),
        (N3, mayor("delta", cfg.n3_delta_min) & mayor("ondas", cfg.n3_ondas_min)),
        (R, menor("emg", cfg.r_emg_max) & movimientos & sin_husos),
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


def continuidad_n2(estadios: np.ndarray, regla: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Regla de continuidad de N2 de la AASM.

    Una época que quedó en N1 por descarte (no cumplió ninguna regla) y sigue a una
    época N2 se marca como N2: el sueño N2 continúa hasta que algo lo interrumpe (un
    despertar, movimientos oculares, ondas lentas), aunque la época no tenga husos.
    Se recorre en orden, así que se propaga por varias épocas seguidas.
    """
    estadios, regla = estadios.copy(), regla.copy()
    descarte = REGLAS.index("N1")
    for t in range(1, len(estadios)):
        if regla[t] == descarte and estadios[t - 1] == N2:
            estadios[t] = N2
            regla[t] = REGLAS.index("N2 cont.")
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
