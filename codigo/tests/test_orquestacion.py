"""Verifica el cableado entre frentes reemplazando las etapas por funciones triviales."""

from dataclasses import replace

import numpy as np
import pytest

from estadios import pipeline
from estadios.clasificacion.modelo import ModeloEstadios
from estadios.comun.config import Config, ConfigClasificacion, ConfigDescriptores
from estadios.comun.constantes import N_ESTADIOS, SIN_ANOTAR
from estadios.comun.sintetico import preprocesado_sintetico, registro_sintetico, tabla_sintetica
from estadios.descriptores import csp, ensamblado, espectro, hrv, husos, ondas_lentas, perifericos
from estadios.preprocesamiento import cardiaco, filtrado, ocular, qrs, segmentacion


@pytest.fixture
def etapas_a(monkeypatch):
    """Reemplaza el Frente A y devuelve la lista de etapas que se llamaron."""
    llamadas = []

    def falsa(nombre, resultado=lambda registro, *_: registro):
        def funcion(*args):
            llamadas.append(nombre)
            return resultado(*args)

        return funcion

    monkeypatch.setattr(qrs, "detectar_latidos", falsa("qrs", lambda ecg, fs: np.array([10, 20])))
    monkeypatch.setattr(filtrado, "filtrar_registro", falsa("filtrado"))
    monkeypatch.setattr(ocular, "remover_artefacto_ocular", falsa("ocular"))
    monkeypatch.setattr(cardiaco, "remover_artefacto_cardiaco", falsa("cardiaco"))
    monkeypatch.setattr(
        segmentacion,
        "marcar_no_estacionarias",
        falsa("segmentacion", lambda registro, cfg: np.zeros(registro.n_epocas, dtype=bool)),
    )
    return llamadas


@pytest.fixture
def etapas_b(monkeypatch):
    """Reemplaza el Frente B con un descriptor constante por grupo."""

    def falsa(grupo):
        def funcion(prep, cfg):
            return np.ones((prep.registro.n_epocas, 1)), [f"{grupo}.x"]

        return funcion

    monkeypatch.setattr(espectro, "descriptores_espectrales", falsa("espectro"))
    monkeypatch.setattr(husos, "descriptores_husos", falsa("husos"))
    monkeypatch.setattr(ondas_lentas, "descriptores_ondas_lentas", falsa("ondas"))
    monkeypatch.setattr(perifericos, "descriptores_perifericos", falsa("perifericos"))
    monkeypatch.setattr(hrv, "descriptores_hrv", falsa("hrv"))
    monkeypatch.setattr(
        csp, "covarianzas_por_epoca", lambda e: np.einsum("eci,edi->ecd", e, e) / e.shape[-1]
    )


def test_preprocesar_respeta_orden_e_intermedios(etapas_a):
    prep = pipeline.preprocesar(
        registro_sintetico(minutos=2), Config().preprocesamiento, guardar_intermedios=True
    )
    assert etapas_a == ["qrs", "filtrado", "ocular", "cardiaco", "segmentacion"]
    assert list(prep.intermedios) == ["crudo", "filtrado", "sin_ocular", "sin_cardiaco"]


def test_sin_remocion_igual_se_detectan_latidos_para_la_hrv(etapas_a):
    cfg = replace(Config().preprocesamiento, remover_ocular=False, remover_cardiaco=False)
    prep = pipeline.preprocesar(registro_sintetico(minutos=2), cfg)
    assert etapas_a == ["qrs", "filtrado", "segmentacion"]
    assert prep.latidos.tolist() == [10, 20]
    assert prep.intermedios == {}


def test_describir_registro_arma_la_tabla(etapas_b):
    prep = preprocesado_sintetico(minutos=2)
    tabla = ensamblado.describir_registro(prep, ConfigDescriptores())
    assert tabla.nombres == ["espectro.x", "husos.x", "ondas.x", "perifericos.x", "hrv.x"]
    assert tabla.matriz.shape == (4, 5)
    assert tabla.covarianzas_eeg.shape == (4, 4, 4)


def test_obtener_tabla_usa_el_cache(etapas_a, etapas_b, monkeypatch, tmp_path):
    cargas = []

    def cargar(id_registro, directorio):
        cargas.append(id_registro)
        return registro_sintetico(minutos=2)

    monkeypatch.setattr(pipeline, "cargar_registro", cargar)
    primera = pipeline.obtener_tabla("SN001", Config(), tmp_path, tmp_path)
    segunda = pipeline.obtener_tabla("SN001", Config(), tmp_path, tmp_path)
    assert cargas == ["SN001"]
    np.testing.assert_array_equal(primera.matriz, segunda.matriz)


@pytest.fixture
def etapas_c(monkeypatch):
    """Reemplaza CSP por una versión mínima y registra con qué épocas se ajusta."""
    ajustes_csp = []

    def ajustar_csp(cov, estadios, par, n):
        ajustes_csp.append(estadios)
        return csp.FiltroCSP(par=par, pesos=np.eye(2 * n, cov.shape[1]))

    monkeypatch.setattr(csp, "ajustar_csp", ajustar_csp)
    return ajustes_csp


def test_modelo_ajusta_csp_sin_epocas_sin_anotar(etapas_c):
    tabla = tabla_sintetica(n_epocas=100)
    tabla.estadios[:5] = SIN_ANOTAR
    modelo = ModeloEstadios(ConfigClasificacion()).ajustar([tabla])

    assert SIN_ANOTAR not in etapas_c[0]
    assert len(modelo.filtros_csp) == len(modelo.referencias_csp) == 1
    prediccion = modelo.predecir(tabla)
    assert prediccion.shape == (100,)
    assert set(prediccion.tolist()) <= set(range(N_ESTADIOS))


def test_modelo_sin_csp_ni_suavizado(etapas_c):
    cfg = ConfigClasificacion(grupos=("espectro", "perifericos"), suavizado="ninguno")
    tabla = tabla_sintetica(n_epocas=100)
    modelo = ModeloEstadios(cfg).ajustar([tabla])

    assert etapas_c == []
    estadios, regla = modelo.clasificar_epocas(tabla)
    np.testing.assert_array_equal(modelo.predecir(tabla), estadios)
    # Sin husos ni ondas, N2 y N3 no pueden aparecer.
    assert not np.isin(estadios, [2, 3]).any()
