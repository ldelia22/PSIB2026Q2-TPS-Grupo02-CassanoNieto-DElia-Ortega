"""Etapa 3. Remoción del artefacto ocular del EEG, con NLMS y Wiener para comparar."""

from __future__ import annotations

import numba
import numpy as np
from scipy import linalg, ndimage, signal

from ..comun.config import ConfigPreprocesamiento
from ..comun.constantes import CANALES_EEG, EOG_E1, EOG_E2
from ..comun.contratos import Registro


@numba.njit
def _nucleo_nlms(
    eeg: np.ndarray, referencia: np.ndarray, orden: int, mu: float, congelar: np.ndarray
) -> np.ndarray:
    """Recorre la señal una vez. Compilado con numba porque son millones de muestras."""
    pesos = np.zeros(orden)
    salida = eeg.copy()
    for i in range(orden - 1, len(eeg)):
        estimado = 0.0
        energia = 1e-6  # Evita dividir por cero si la referencia es plana.
        for k in range(orden):
            estimado += pesos[k] * referencia[i - k]
            energia += referencia[i - k] ** 2
        error = eeg[i] - estimado
        salida[i] = error
        if not congelar[i]:
            for k in range(orden):
                pesos[k] += mu * error * referencia[i - k] / energia
    return salida


def nlms(
    eeg: np.ndarray,
    referencia: np.ndarray,
    orden: int,
    mu: float,
    congelar: np.ndarray | None = None,
) -> np.ndarray:
    """Cancelación adaptativa. Devuelve el EEG sin lo que se explica por la referencia.

    Donde `congelar` es verdadero los pesos no se actualizan, para que un artefacto
    grande no desajuste el filtro.
    """
    if congelar is None:
        congelar = np.zeros(len(eeg), dtype=bool)
    return _nucleo_nlms(
        np.asarray(eeg, dtype=float), np.asarray(referencia, dtype=float), orden, float(mu), congelar
    )


def wiener(
    eeg: np.ndarray, referencia: np.ndarray, orden: int, excluir: np.ndarray | None = None
) -> np.ndarray:
    """Filtro óptimo con pesos fijos, resuelto con las ecuaciones de Wiener-Hopf sobre toda la señal."""
    eeg, referencia = np.asarray(eeg, dtype=float), np.asarray(referencia, dtype=float)
    validas = np.ones(len(eeg), dtype=bool) if excluir is None else ~excluir
    x = np.where(validas, referencia, 0.0)
    y = np.where(validas, eeg, 0.0)
    n = max(int(validas.sum()), 1)

    autocorrelacion = np.array([x[: len(x) - k] @ x[k:] for k in range(orden)]) / n
    cruzada = np.array([x[: len(x) - k] @ y[k:] for k in range(orden)]) / n
    autocorrelacion[0] += 1e-6  # Mantiene el sistema resoluble si la referencia es plana.
    pesos = linalg.solve_toeplitz(autocorrelacion, cruzada)
    return eeg - signal.lfilter(pesos, 1.0, referencia)


def actividad_ocular(
    referencia: np.ndarray, fs: float, cfg: ConfigPreprocesamiento
) -> np.ndarray:
    """Devuelve una ganancia entre 0 y 1 que indica dónde hay movimientos oculares.

    Es 1 donde el valor eficaz de la referencia supera varias veces su mediana de la
    noche. La transición se suaviza para no introducir escalones en el EEG.
    """
    largo = int(cfg.ocular_ventana_s * fs)
    eficaz = np.sqrt(ndimage.uniform_filter1d(referencia**2, largo))
    activa = (eficaz > cfg.ocular_umbral_relativo * np.median(eficaz)).astype(float)
    return ndimage.uniform_filter1d(activa, largo)


def remover_artefacto_ocular(registro: Registro, cfg: ConfigPreprocesamiento) -> Registro:
    """Limpia cada canal de EEG usando como referencia la diferencia E1 - E2.

    La referencia también contiene ondas lentas cerebrales. Si se corrigiera toda la
    noche se perderían justamente las ondas de N3, así que el filtro se ajusta y se
    aplica solo en los tramos con actividad ocular.
    """
    referencia = registro.senales[EOG_E1] - registro.senales[EOG_E2]
    ganancia = actividad_ocular(referencia, registro.fs, cfg)
    nuevas = {}
    for canal in CANALES_EEG:
        eeg = registro.senales[canal]
        artefacto = (np.abs(eeg) > cfg.umbral_artefacto_uv) | (np.abs(referencia) > cfg.umbral_artefacto_uv)
        fuera_del_ajuste = artefacto | (ganancia < 1.0)
        if cfg.metodo_ocular == "nlms":
            limpia = nlms(eeg, referencia, cfg.nlms_orden, cfg.nlms_mu, fuera_del_ajuste)
        else:
            limpia = wiener(eeg, referencia, cfg.wiener_orden, fuera_del_ajuste)
        nuevas[canal] = eeg + ganancia * (limpia - eeg)
    return registro.reemplazar_senales(nuevas)
