"""El clasificador completo: CSP + LDA + HMM.

Todo lo que se aprende de los datos (filtros CSP, LDA, transiciones) se ajusta en
`ajustar`, que solo debe recibir pacientes de entrenamiento.
"""

from __future__ import annotations

import numpy as np

from ..comun.config import ConfigClasificacion
from ..comun.constantes import N_ESTADIOS, SIN_ANOTAR
from ..comun.contratos import TablaDescriptores
from ..descriptores import csp
from ..descriptores.ensamblado import estandarizar_por_registro
from . import hmm, lda


class ModeloEstadios:
    def __init__(self, cfg: ConfigClasificacion):
        self.cfg = cfg
        self.filtros_csp: list[csp.FiltroCSP] = []
        self.lda = None
        self.log_inicial: np.ndarray | None = None
        self.log_transiciones: np.ndarray | None = None
        self.nombres: list[str] = []

    def _matriz(self, tabla: TablaDescriptores) -> np.ndarray:
        """Descriptores de los grupos elegidos más los de CSP, estandarizados."""
        elegidos = tabla.seleccionar(self.cfg.grupos)
        columnas = [elegidos.matriz]
        for filtro in self.filtros_csp:
            columnas.append(csp.aplicar_csp(tabla.covarianzas_eeg, filtro))
        return estandarizar_por_registro(np.hstack(columnas))

    def ajustar(self, tablas: list[TablaDescriptores]) -> "ModeloEstadios":
        estadios = np.concatenate([t.estadios for t in tablas])

        self.filtros_csp = []
        if "csp" in self.cfg.grupos:
            covarianzas = np.concatenate([t.covarianzas_eeg for t in tablas])
            anotadas = estadios != SIN_ANOTAR
            self.filtros_csp = [
                csp.ajustar_csp(
                    covarianzas[anotadas], estadios[anotadas], par, self.cfg.componentes_csp
                )
                for par in self.cfg.pares_csp
            ]

        X = np.vstack([self._matriz(t) for t in tablas])
        anotadas = estadios != SIN_ANOTAR
        self.lda = lda.entrenar_lda(X[anotadas], estadios[anotadas])

        self.nombres = list(tablas[0].seleccionar(self.cfg.grupos).nombres)
        for filtro in self.filtros_csp:
            self.nombres += [
                f"csp.{filtro.par[0]}_{filtro.par[1]}_{i}" for i in range(len(filtro.pesos))
            ]

        if self.cfg.usar_hmm:
            inicial, transiciones = hmm.estimar_transiciones(
                [t.estadios for t in tablas], N_ESTADIOS
            )
            self.log_inicial = np.log(inicial)
            self.log_transiciones = np.log(transiciones)
        return self

    def predecir(self, tabla: TablaDescriptores) -> np.ndarray:
        """Hipnograma estimado de un registro."""
        X = self._matriz(tabla)
        if not self.cfg.usar_hmm:
            return self.lda.predict(X)
        return hmm.viterbi(self.lda.predict_log_proba(X), self.log_inicial, self.log_transiciones)
