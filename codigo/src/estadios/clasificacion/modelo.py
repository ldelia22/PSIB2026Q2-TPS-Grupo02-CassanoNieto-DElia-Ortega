"""El clasificador completo: reglas AASM + desempate CSP + suavizado temporal.

Reemplaza al LDA + HMM del anteproyecto (ver `reglas.py`). Lo único que se ajusta con
datos son los filtros CSP, que son filtrado espacial (etapa 8) y se calculan en
`ajustar` solo con pacientes de entrenamiento. Los umbrales de las reglas vienen de
`ConfigClasificacion`.
"""

from __future__ import annotations

import numpy as np

from ..comun.config import ConfigClasificacion
from ..comun.constantes import SIN_ANOTAR
from ..comun.contratos import TablaDescriptores
from ..descriptores import csp
from . import reglas, suavizado


class ModeloEstadios:
    def __init__(self, cfg: ConfigClasificacion):
        self.cfg = cfg
        self.filtros_csp: list[csp.FiltroCSP] = []
        self.referencias_csp: list[tuple[float, float]] = []  # (centro, escala) por filtro.

    def ajustar(self, tablas: list[TablaDescriptores]) -> "ModeloEstadios":
        """Solo calcula los filtros CSP y su punto medio; las reglas no se entrenan."""
        self.filtros_csp, self.referencias_csp = [], []
        if "csp" in self.cfg.grupos:
            estadios = np.concatenate([t.estadios for t in tablas])
            covarianzas = np.concatenate([t.covarianzas_eeg for t in tablas])
            anotadas = estadios != SIN_ANOTAR
            self.filtros_csp = [
                csp.ajustar_csp(
                    covarianzas[anotadas], estadios[anotadas], par, self.cfg.componentes_csp
                )
                for par in self.cfg.pares_csp
            ]
            self.referencias_csp = [
                reglas.referencia_csp(
                    reglas.puntaje_csp(csp.aplicar_csp(covarianzas[anotadas], filtro)),
                    estadios[anotadas],
                    filtro.par,
                )
                for filtro in self.filtros_csp
            ]
        return self

    def clasificar_epocas(self, tabla: TablaDescriptores) -> tuple[np.ndarray, np.ndarray]:
        """Reglas, desempate CSP y continuidad de N2, sin suavizado.

        Returns:
            Estadio de cada época e índice en `reglas.REGLAS` de la regla que lo decidió.
        """
        p = reglas.descriptores_de_reglas(tabla.matriz, tabla.nombres, self.cfg)
        estadios, regla = reglas.aplicar_reglas(p, self.cfg, len(tabla.estadios))
        for filtro, (centro, escala) in zip(self.filtros_csp, self.referencias_csp):
            puntaje = reglas.puntaje_csp(csp.aplicar_csp(tabla.covarianzas_eeg, filtro))
            estadios, regla = reglas.desempatar_con_csp(
                estadios, regla, puntaje, filtro.par, centro, escala, self.cfg.csp_margen
            )
        if self.cfg.continuidad_n2:
            estadios, regla = reglas.continuidad_n2(estadios, regla)
        return estadios, regla

    def suavizar(self, estadios: np.ndarray) -> np.ndarray:
        if self.cfg.suavizado == "reglas":
            return suavizado.suavizar_por_reglas(estadios, self.cfg.rem_min_epocas)
        if self.cfg.suavizado == "moda":
            return suavizado.filtro_de_moda(estadios, self.cfg.ventana_moda)
        return estadios.copy()

    def predecir(self, tabla: TablaDescriptores) -> np.ndarray:
        """Hipnograma estimado de un registro."""
        return self.suavizar(self.clasificar_epocas(tabla)[0])
