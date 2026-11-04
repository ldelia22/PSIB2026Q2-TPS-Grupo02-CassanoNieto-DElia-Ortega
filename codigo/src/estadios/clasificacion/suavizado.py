"""Etapa 10: suavizado temporal por reglas. Reemplaza al HMM del anteproyecto.

El HMM aprendía las probabilidades de transición de los hipnogramas de entrenamiento,
por lo que también es una técnica de aprendizaje automático. Acá se usan reglas fijas
de continuidad, en el espíritu del manual AASM: el sueño cambia de estadio cada varios
minutos, así que una época aislada probablemente sea un error.
"""

from __future__ import annotations

import numpy as np

from ..comun.constantes import N_ESTADIOS, R


def _tramos(y: np.ndarray) -> list[tuple[int, int]]:
    """(inicio, fin) de cada tramo de épocas iguales consecutivas."""
    cortes = np.flatnonzero(np.diff(y)) + 1
    inicios = np.r_[0, cortes]
    fines = np.r_[cortes, len(y)]
    return list(zip(inicios.tolist(), fines.tolist()))


def corregir_aisladas(y: np.ndarray) -> np.ndarray:
    """Una época distinta entre dos vecinas iguales toma el estadio de las vecinas."""
    y = y.copy()
    for t in range(1, len(y) - 1):
        if y[t - 1] == y[t + 1] != y[t]:
            y[t] = y[t - 1]
    return y


def rem_minimo(y: np.ndarray, min_epocas: int) -> np.ndarray:
    """Un tramo de REM más corto que `min_epocas` toma el estadio anterior (o el
    siguiente, si está al principio)."""
    y = y.copy()
    for inicio, fin in _tramos(y):
        if y[inicio] == R and fin - inicio < min_epocas:
            if inicio > 0:
                y[inicio:fin] = y[inicio - 1]
            elif fin < len(y):
                y[inicio:fin] = y[fin]
    return y


def suavizar_por_reglas(y: np.ndarray, rem_min_epocas: int) -> np.ndarray:
    """Épocas aisladas y después REM mínimo."""
    return rem_minimo(corregir_aisladas(y), rem_min_epocas)


def filtro_de_moda(y: np.ndarray, ventana: int) -> np.ndarray:
    """Estadio más frecuente en una ventana centrada; en caso de empate queda el original."""
    if ventana < 2:
        return y.copy()
    medio = ventana // 2
    relleno = np.r_[np.full(medio, y[0]), y, np.full(medio, y[-1])]
    salida = y.copy()
    for t in range(len(y)):
        conteo = np.bincount(relleno[t : t + ventana], minlength=N_ESTADIOS)
        mejores = np.flatnonzero(conteo == conteo.max())
        if y[t] not in mejores:
            salida[t] = mejores[0]
    return salida
