"""Constantes de la base HMC y de la clasificación AASM.

Los nombres de canal y la frecuencia de muestreo se verificaron contra los
headers EDF de los 151 registros de la HMC v1.1: son iguales en todos.
"""

FS_HMC = 256.0
DURACION_EPOCA_S = 30.0

# Estadios como enteros: es lo que circula por todo el código.
W, N1, N2, N3, R = 0, 1, 2, 3, 4
SIN_ANOTAR = -1
NOMBRES_ESTADIOS = ("W", "N1", "N2", "N3", "R")
N_ESTADIOS = len(NOMBRES_ESTADIOS)

ETIQUETA_A_ESTADIO = {
    "Sleep stage W": W,
    "Sleep stage N1": N1,
    "Sleep stage N2": N2,
    "Sleep stage N3": N3,
    "Sleep stage R": R,
}

EEG_F4 = "EEG F4-M1"
EEG_C4 = "EEG C4-M1"
EEG_O2 = "EEG O2-M1"
EEG_C3 = "EEG C3-M2"
EMG = "EMG chin"
EOG_E1 = "EOG E1-M2"
EOG_E2 = "EOG E2-M2"
ECG = "ECG"

CANALES_EEG = (EEG_F4, EEG_C4, EEG_O2, EEG_C3)
CANALES = (*CANALES_EEG, EMG, EOG_E1, EOG_E2, ECG)

BANDAS_EEG = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alfa": (8.0, 13.0),
    "sigma": (11.0, 16.0),
    "beta": (13.0, 30.0),
}
BANDA_LF = (0.04, 0.15)
BANDA_HF = (0.15, 0.4)
