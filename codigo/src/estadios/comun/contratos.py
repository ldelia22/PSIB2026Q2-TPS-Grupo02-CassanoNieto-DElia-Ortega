"""Estructuras que se pasan los frentes entre sí.

    Frente A  --- RegistroPreprocesado --->  Frente B  --- TablaDescriptores --->  Frente C

Mientras un frente implementa lo suyo, los demás pueden trabajar con los datos de
`estadios.comun.sintetico`, que respetan estos mismos contratos.

Convenciones: señales en µV, estadios como enteros (ver `constantes`), y latidos
como índices de muestra.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .constantes import CANALES_EEG, DURACION_EPOCA_S


@dataclass
class Registro:
    """Una PSG de un paciente."""

    id: str
    fs: float
    senales: dict[str, np.ndarray]  # Canal -> señal 1D en µV.
    estadios: np.ndarray  # Un entero por época; SIN_ANOTAR si no tiene etiqueta.
    luces_apagadas_s: float | None = None
    luces_encendidas_s: float | None = None

    def __post_init__(self) -> None:
        largos = {len(s) for s in self.senales.values()}
        if len(largos) != 1:
            raise ValueError(f"{self.id}: los canales tienen largos distintos: {largos}")
        epocas_completas = int(largos.pop() // (self.fs * DURACION_EPOCA_S))
        if len(self.estadios) > epocas_completas:
            raise ValueError(
                f"{self.id}: {len(self.estadios)} estadios para {epocas_completas} épocas"
            )

    @property
    def n_epocas(self) -> int:
        return len(self.estadios)

    def eeg(self) -> np.ndarray:
        """EEG con forma (n_canales, n_muestras), en el orden de CANALES_EEG."""
        return np.stack([self.senales[c] for c in CANALES_EEG])

    def reemplazar_senales(self, nuevas: dict[str, np.ndarray]) -> "Registro":
        """Copia del registro con esos canales cambiados y el resto intacto."""
        return replace(self, senales={**self.senales, **nuevas})


@dataclass
class RegistroPreprocesado:
    """Salida del Frente A: señal limpia, lista para calcular descriptores."""

    registro: Registro
    latidos: np.ndarray  # Índices de muestra de cada onda R.
    no_estacionaria: np.ndarray  # Booleano por época (etapa 5).
    intermedios: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    # Etapa -> canal -> señal. Solo se llena para la interfaz gráfica.


@dataclass
class TablaDescriptores:
    """Salida del Frente B: una fila por época. Es lo que se cachea en disco.

    Cada descriptor se llama "<grupo>.<nombre>", por ejemplo "espectro.delta_rel".
    Las covarianzas se guardan porque CSP se ajusta sobre los pacientes de
    entrenamiento: con ellas, la potencia de cada filtro se obtiene sin volver a
    leer las señales.
    """

    id: str
    matriz: np.ndarray  # (n_epocas, n_descriptores)
    nombres: list[str]
    estadios: np.ndarray  # (n_epocas,)
    covarianzas_eeg: np.ndarray  # (n_epocas, n_canales_eeg, n_canales_eeg)
    no_estacionaria: np.ndarray  # (n_epocas,)

    def seleccionar(self, grupos: tuple[str, ...]) -> "TablaDescriptores":
        """Copia solo con los descriptores de esos grupos."""
        columnas = [i for i, n in enumerate(self.nombres) if n.split(".")[0] in grupos]
        return replace(
            self,
            matriz=self.matriz[:, columnas],
            nombres=[self.nombres[i] for i in columnas],
        )

    def guardar(self, ruta: Path) -> None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ruta,
            id=self.id,
            matriz=self.matriz,
            nombres=np.array(self.nombres),
            estadios=self.estadios,
            covarianzas_eeg=self.covarianzas_eeg,
            no_estacionaria=self.no_estacionaria,
        )

    @classmethod
    def cargar(cls, ruta: Path) -> "TablaDescriptores":
        with np.load(ruta) as d:
            return cls(
                id=str(d["id"]),
                matriz=d["matriz"],
                nombres=[str(n) for n in d["nombres"]],
                estadios=d["estadios"],
                covarianzas_eeg=d["covarianzas_eeg"],
                no_estacionaria=d["no_estacionaria"],
            )
