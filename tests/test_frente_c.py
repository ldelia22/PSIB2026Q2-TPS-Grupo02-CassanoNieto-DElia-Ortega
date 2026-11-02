"""Especificación del Frente C: clasificación y evaluación."""

import numpy as np
import pytest

from estadios.clasificacion import hmm, lda
from estadios.clasificacion.modelo import ModeloEstadios
from estadios.comun.config import ConfigClasificacion
from estadios.comun.constantes import N1, N3, N_ESTADIOS, NOMBRES_ESTADIOS, R, SIN_ANOTAR, W
from estadios.comun.sintetico import tabla_sintetica
from estadios.evaluacion import hipotesis, metricas


def test_kappa_perfecto_y_al_azar():
    rng = np.random.default_rng(0)
    y = rng.integers(0, N_ESTADIOS, 5000)
    assert metricas.kappa(y, y) == pytest.approx(1.0)
    assert abs(metricas.kappa(y, rng.integers(0, N_ESTADIOS, 5000))) < 0.05


def test_metricas_ignoran_las_epocas_sin_anotar():
    y_real = np.array([W, W, SIN_ANOTAR, N3])
    y_pred = np.array([W, W, R, N3])
    assert metricas.kappa(y_real, y_pred) == pytest.approx(1.0)
    assert metricas.matriz_confusion(y_real, y_pred).sum() == 3


def test_f1_detecta_un_estadio_mal_clasificado():
    # N1 es poco frecuente: la exactitud es alta, pero su F1 es 0.
    y_real = np.array([W] * 95 + [N1] * 5)
    y_pred = np.array([W] * 100)
    f1 = metricas.f1_por_estadio(y_real, y_pred)
    assert f1["N1"] == 0
    assert f1["W"] > 0.9
    assert set(f1) == set(NOMBRES_ESTADIOS)


def test_matriz_de_confusion_filas_reales():
    matriz = metricas.matriz_confusion(np.array([W, W, N3]), np.array([W, N3, N3]))
    assert matriz.shape == (N_ESTADIOS, N_ESTADIOS)
    assert matriz[W, N3] == 1
    assert matriz[N3, N3] == 1


def test_lda_separa_la_tabla_sintetica():
    tabla = tabla_sintetica(n_epocas=1000)
    modelo = lda.entrenar_lda(tabla.matriz, tabla.estadios)
    assert (modelo.predict(tabla.matriz) == tabla.estadios).mean() > 0.8
    pesos = lda.pesos_por_descriptor(modelo, tabla.nombres)
    assert set(pesos) == set(NOMBRES_ESTADIOS)
    assert all(len(p) == len(tabla.nombres) for p in pesos.values())


def test_transiciones_son_probabilidades_e_ignoran_sin_anotar():
    secuencia = np.array([W, W, N1, SIN_ANOTAR, N3, N3, N3])
    inicial, transiciones = hmm.estimar_transiciones([secuencia], N_ESTADIOS, suavizado=0.0)
    assert inicial.sum() == pytest.approx(1)
    assert transiciones.shape == (N_ESTADIOS, N_ESTADIOS)
    # N1 -> SIN_ANOTAR y SIN_ANOTAR -> N3 no cuentan.
    assert transiciones[W, N1] == pytest.approx(0.5)
    assert transiciones[N3, N3] == pytest.approx(1.0)


def test_transiciones_suavizadas_no_tienen_ceros():
    _, transiciones = hmm.estimar_transiciones([np.array([W, W, N1])], N_ESTADIOS)
    assert (transiciones > 0).all()
    assert transiciones.sum(axis=1) == pytest.approx(np.ones(N_ESTADIOS))


def test_viterbi_corrige_una_epoca_aislada():
    n = 21
    log_emisiones = np.log(np.full((n, N_ESTADIOS), 0.05))
    log_emisiones[:, N3] = np.log(0.8)
    # REM aislado, apenas más probable que N3 en esa época.
    log_emisiones[10, R] = np.log(0.5)
    log_emisiones[10, N3] = np.log(0.4)

    transiciones = np.full((N_ESTADIOS, N_ESTADIOS), 0.01)
    np.fill_diagonal(transiciones, 0.96)
    inicial = np.full(N_ESTADIOS, 1 / N_ESTADIOS)

    camino = hmm.viterbi(log_emisiones, np.log(inicial), np.log(transiciones))
    assert camino.tolist() == [N3] * n


def test_modelo_completo_sobre_datos_sinteticos():
    entrenamiento = [tabla_sintetica(semilla=s) for s in range(4)]
    prueba = tabla_sintetica(semilla=10)
    modelo = ModeloEstadios(ConfigClasificacion()).ajustar(entrenamiento)
    assert metricas.kappa(prueba.estadios, modelo.predecir(prueba)) > 0.7


def test_wilcoxon_detecta_una_mejora_consistente():
    rng = np.random.default_rng(0)
    b = rng.random(20)
    resultado = hipotesis.comparar_por_paciente(b + 0.1, b)
    assert resultado.p_valor < 0.01
