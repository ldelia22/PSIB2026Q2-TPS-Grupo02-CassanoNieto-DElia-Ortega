"""Datos sintéticos que respetan los contratos.

Permiten que cada frente avance sin esperar al anterior y sin bajar la base. Sirven
para probar que el código funciona, no para sacar conclusiones: las señales son
ruido filtrado por bandas con artefactos agregados, no sueño real.
"""

from __future__ import annotations

import numpy as np

from .constantes import (
    CANALES_EEG,
    DURACION_EPOCA_S,
    ECG,
    EMG,
    EOG_E1,
    EOG_E2,
    FS_HMC,
    N1,
    N2,
    N3,
    R,
    W,
)
from .contratos import Registro, RegistroPreprocesado, TablaDescriptores

_SECUENCIA = (W, N1, N2, N3, N3, N2, R, R, N2, W)

# Estadio -> bandas (Hz) y amplitud (µV) de la actividad cerebral.
_PERFIL_EEG = {
    W: [(8, 13, 15), (13, 30, 6)],
    N1: [(4, 8, 15), (8, 13, 5)],
    N2: [(0.5, 4, 15), (4, 8, 12), (11, 16, 10)],
    N3: [(0.5, 4, 60), (4, 8, 8)],
    R: [(4, 8, 12), (13, 30, 5)],
}
# Estadio -> ganancia de cada canal, en el orden de CANALES_EEG.
_GANANCIA_EEG = {
    W: (0.6, 0.8, 1.4, 0.8),
    N1: (0.9, 1.0, 1.0, 1.0),
    N2: (1.0, 1.2, 0.7, 1.2),
    N3: (1.4, 1.0, 0.6, 1.0),
    R: (0.8, 1.0, 1.2, 1.0),
}
_AMPLITUD_OCULAR = {W: 40, N1: 25, N2: 8, N3: 5, R: 60}
_AMPLITUD_EMG = {W: 30, N1: 20, N2: 15, N3: 12, R: 4}
# Cuánto del EOG se filtra a cada canal de EEG.
_FUGA_OCULAR = (0.30, 0.10, 0.02, 0.10)
_FUGA_CARDIACA = (2.0, 3.0, 4.0, 3.0)


def _ruido_de_banda(n: int, fs: float, baja: float, alta: float, rng) -> np.ndarray:
    """Ruido blanco limitado a una banda, con desvío 1."""
    espectro = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    espectro[(f < baja) | (f > alta)] = 0
    x = np.fft.irfft(espectro, n)
    return x / x.std()


def _pulsos(n: int, fs: float, instantes: np.ndarray, sigma_s: float) -> np.ndarray:
    """Suma de gaussianas de altura 1 centradas en esos instantes (en muestras)."""
    mitad = int(5 * sigma_s * fs)
    t = np.arange(-mitad, mitad + 1)
    pulso = np.exp(-0.5 * (t / (sigma_s * fs)) ** 2)
    impulsos = np.zeros(n)
    impulsos[instantes[(instantes >= 0) & (instantes < n)]] = 1
    return np.convolve(impulsos, pulso, mode="same")


def _generar(minutos: float, fs: float, semilla: int) -> tuple[Registro, np.ndarray]:
    rng = np.random.default_rng(semilla)
    n_epocas = int(minutos * 60 / DURACION_EPOCA_S)
    n_epoca = int(DURACION_EPOCA_S * fs)
    n = n_epocas * n_epoca

    # Cada estadio dura unas 4 épocas para que haya transiciones como en un hipnograma.
    estadios = np.array([_SECUENCIA[(i // 4) % len(_SECUENCIA)] for i in range(n_epocas)])

    eeg = np.zeros((len(CANALES_EEG), n))
    ocular = np.zeros(n)
    emg = np.zeros(n)
    for i, estadio in enumerate(estadios):
        tramo = slice(i * n_epoca, (i + 1) * n_epoca)
        for baja, alta, amplitud in _PERFIL_EEG[estadio]:
            comun = amplitud * _ruido_de_banda(n_epoca, fs, baja, alta, rng)
            for c, ganancia in enumerate(_GANANCIA_EEG[estadio]):
                propio = 0.3 * amplitud * _ruido_de_banda(n_epoca, fs, baja, alta, rng)
                eeg[c, tramo] += ganancia * comun + propio
        ocular[tramo] = _AMPLITUD_OCULAR[estadio] * _ruido_de_banda(n_epoca, fs, 0.3, 4, rng)
        emg[tramo] = _AMPLITUD_EMG[estadio] * _ruido_de_banda(n_epoca, fs, 10, 100, rng)

    # RR con oscilación respiratoria en 0,25 Hz: cae en la banda HF de la HRV.
    duracion = n / fs
    instantes = [0.5]
    while instantes[-1] < duracion:
        rr = 0.9 + 0.04 * np.sin(2 * np.pi * 0.25 * instantes[-1]) + rng.normal(0, 0.005)
        instantes.append(instantes[-1] + rr)
    latidos = (np.array(instantes[:-1]) * fs).astype(int)
    latidos = latidos[latidos < n]

    ecg = (
        800 * _pulsos(n, fs, latidos, 0.012)
        + 150 * _pulsos(n, fs, latidos + int(0.25 * fs), 0.05)
        + 10 * rng.standard_normal(n)
    )
    artefacto_cardiaco = _pulsos(n, fs, latidos + int(0.02 * fs), 0.025)
    for c in range(len(CANALES_EEG)):
        eeg[c] += _FUGA_OCULAR[c] * ocular + _FUGA_CARDIACA[c] * artefacto_cardiaco

    # El EOG se invierte entre E1 y E2; la actividad cerebral llega igual a ambos.
    cerebro_en_eog = 5 * _ruido_de_banda(n, fs, 0.5, 30, rng)
    senales = {nombre: eeg[c] for c, nombre in enumerate(CANALES_EEG)}
    senales[EOG_E1] = ocular / 2 + cerebro_en_eog
    senales[EOG_E2] = -ocular / 2 + cerebro_en_eog
    senales[EMG] = emg
    senales[ECG] = ecg
    registro = Registro(id=f"SINT{semilla:03d}", fs=fs, senales=senales, estadios=estadios)
    return registro, latidos


def registro_sintetico(minutos: float = 30, fs: float = FS_HMC, semilla: int = 0) -> Registro:
    """Registro crudo, con artefactos: entrada del Frente A."""
    return _generar(minutos, fs, semilla)[0]


def preprocesado_sintetico(
    minutos: float = 30, fs: float = FS_HMC, semilla: int = 0
) -> RegistroPreprocesado:
    """Registro con los latidos verdaderos: entrada del Frente B.

    Conserva los artefactos, así que no sirve para evaluar su remoción.
    """
    registro, latidos = _generar(minutos, fs, semilla)
    return RegistroPreprocesado(
        registro=registro,
        latidos=latidos,
        no_estacionaria=np.zeros(registro.n_epocas, dtype=bool),
    )


def tabla_sintetica(n_epocas: int = 600, semilla: int = 0) -> TablaDescriptores:
    """Descriptores gaussianos que dependen del estadio: entrada del Frente C."""
    rng = np.random.default_rng(semilla)
    # Bloques de duración variable: se parece a un hipnograma real, con estadios que duran varias épocas.
    estadios: list[int] = []
    while len(estadios) < n_epocas:
        estadio = _SECUENCIA[rng.integers(len(_SECUENCIA))]
        estadios.extend([estadio] * int(rng.integers(5, 30)))
    y = np.array(estadios[:n_epocas])

    nombres = [
        "espectro.delta_rel",
        "espectro.alfa_rel",
        "husos.cantidad",
        "ondas.cantidad",
        "perifericos.energia_eog",
        "perifericos.amplitud_emg",
        "hrv.hf",
        "hrv.lf_hf",
    ]
    # Valores en las mismas unidades que entrega el Frente B: potencias relativas como
    # fracción, husos y ondas como conteos por época, y EOG, EMG y HRV en log10.
    centros = np.array(
        [
            # delta alfa  husos ondas  eog   emg   hf    lf/hf
            [0.35, 0.25, 0.0, 0.1, 1.0, 0.6, -0.3, 0.4],  # W
            [0.50, 0.10, 0.2, 0.3, 0.5, 0.2, -0.1, 0.2],  # N1
            [0.65, 0.05, 2.5, 1.0, -0.3, 0.0, 0.2, -0.2],  # N2
            [0.85, 0.03, 0.5, 8.0, -0.5, -0.1, 0.4, -0.4],  # N3
            [0.50, 0.07, 0.0, 0.1, 1.2, -0.6, -0.2, 0.5],  # R
        ]
    )
    dispersion = np.array([0.07, 0.04, 0.0, 0.0, 0.3, 0.2, 0.2, 0.2])
    matriz = centros[y] + dispersion * rng.standard_normal((n_epocas, len(nombres)))
    matriz[:, :2] = matriz[:, :2].clip(0.0, 1.0)
    for j in (2, 3):  # Husos y ondas: conteos de Poisson.
        matriz[:, j] = rng.poisson(centros[y, j])

    # Covarianzas con varianza dominante en un canal distinto según el estadio.
    n_canales = len(CANALES_EEG)
    covarianzas = np.empty((n_epocas, n_canales, n_canales))
    dominante = {W: 2, N1: 1, N2: 3, N3: 0, R: 2}
    for i, estadio in enumerate(y):
        base = rng.standard_normal((n_canales, 4 * n_canales))
        base[dominante[int(estadio)]] *= 3
        covarianzas[i] = base @ base.T / base.shape[1]

    return TablaDescriptores(
        id=f"SINT{semilla:03d}",
        matriz=matriz,
        nombres=nombres,
        estadios=y,
        covarianzas_eeg=covarianzas,
        no_estacionaria=np.zeros(n_epocas, dtype=bool),
    )
