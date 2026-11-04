"""Especificación del Frente C: clasificación y evaluación."""

import numpy as np
import pytest

from estadios.clasificacion import reglas, suavizado
from estadios.clasificacion.modelo import ModeloEstadios
from estadios.comun.config import ConfigClasificacion
from estadios.comun.constantes import N1, N2, N3, N_ESTADIOS, NOMBRES_ESTADIOS, R, SIN_ANOTAR, W
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


def test_percentiles_por_registro():
    p = reglas.percentil_por_registro(np.array([10.0, 30.0, 20.0]))
    assert p.shape == (3, 1)
    assert p[:, 0].tolist() == [0.0, 100.0, 50.0]


def test_buscar_descriptor_prefiere_el_canal_y_si_no_usa_el_que_haya():
    nombres = ["espectro.f4_delta_rel", "espectro.c4_delta_rel", "husos.cantidad"]
    assert reglas.buscar_descriptor(nombres, "espectro", "delta_rel", "c4") == 1
    assert reglas.buscar_descriptor(nombres, "espectro", "delta_rel", "o2") == 0
    assert reglas.buscar_descriptor(nombres, "ondas", "cantidad") is None


def test_reglas_en_orden_y_n1_por_descarte():
    cfg = ConfigClasificacion()
    # Columnas: W, N3, REM, N2 y una época que no cumple nada.
    p = {
        "emg": np.array([90, 50, 10, 50, 50.0]),
        "alfa": np.array([90, 10, 10, 10, 50.0]),
        "delta": np.array([10, 95, 10, 30, 30.0]),
        "ondas": np.array([10, 95, 10, 30, 30.0]),
        "eog": np.array([50, 10, 95, 10, 50.0]),
        "husos": np.array([10, 10, 10, 90, 10.0]),
    }
    estadios, regla = reglas.aplicar_reglas(p, cfg, 5)
    assert estadios.tolist() == [W, N3, R, N2, N1]
    assert [reglas.REGLAS[i] for i in regla] == ["W", "N3", "R", "N2", "N1"]


def test_regla_sin_su_descriptor_no_se_cumple():
    cfg = ConfigClasificacion()
    p = {"emg": np.array([90.0]), "husos": np.array([10.0])}  # Falta alfa: no es W.
    estadios, _ = reglas.aplicar_reglas(p, cfg, 1)
    assert estadios.tolist() == [N1]


def test_rem_puede_usar_la_hrv_en_lugar_del_eog():
    cfg = ConfigClasificacion()
    p = {"emg": np.array([10.0]), "eog": np.array([10.0]), "lf_hf": np.array([95.0])}
    assert reglas.aplicar_reglas(p, cfg, 1)[0].tolist() == [R]


def test_csp_desempata_solo_con_evidencia_fuerte():
    estadios = np.array([N1, N1, R, R, N3])
    regla = np.zeros(5, dtype=int)
    puntaje = np.array([-3.0, 0.2, 3.0, -0.2, -3.0])  # Positivo: parecido a N1.
    nuevos, regla = reglas.desempatar_con_csp(estadios, regla, puntaje, (N1, R), 0.0, 1.0, 1.0)
    assert nuevos.tolist() == [R, N1, N1, R, N3]  # N3 no es del par: no cambia.
    assert (regla == reglas.REGLAS.index("csp")).sum() == 2


def test_cruce_de_histogramas_queda_entre_las_medianas():
    rng = np.random.default_rng(0)
    umbral = reglas.cruce_de_histogramas(rng.normal(0, 1, 2000), rng.normal(4, 1, 2000))
    assert 1.5 < umbral < 2.5


def test_suavizado_corrige_una_epoca_aislada():
    y = np.array([N3] * 10 + [R] + [N3] * 10)
    assert suavizado.corregir_aisladas(y).tolist() == [N3] * 21


def test_suavizado_rem_minimo():
    y = np.array([N2, N2, R, R, N2, N2, R, R, R, N2])
    assert suavizado.rem_minimo(y, 3).tolist() == [N2] * 6 + [R] * 3 + [N2]


def test_filtro_de_moda():
    y = np.array([W, W, N1, W, W, N2, N2, N2])
    assert suavizado.filtro_de_moda(y, 3).tolist() == [W, W, W, W, W, N2, N2, N2]


def test_modelo_completo_sobre_datos_sinteticos():
    entrenamiento = [tabla_sintetica(semilla=s) for s in range(4)]
    prueba = tabla_sintetica(semilla=10)
    modelo = ModeloEstadios(ConfigClasificacion()).ajustar(entrenamiento)
    assert metricas.kappa(prueba.estadios, modelo.predecir(prueba)) > 0.7


def test_modelo_no_aprende_de_las_etiquetas_salvo_csp():
    # Sin CSP, cambiar las etiquetas de entrenamiento no cambia la predicción.
    cfg = ConfigClasificacion(grupos=("espectro", "husos", "ondas", "perifericos", "hrv"))
    tabla = tabla_sintetica(semilla=10)
    entrenamiento = tabla_sintetica(semilla=0)
    a = ModeloEstadios(cfg).ajustar([entrenamiento]).predecir(tabla)
    entrenamiento.estadios[:] = W
    b = ModeloEstadios(cfg).ajustar([entrenamiento]).predecir(tabla)
    assert a.tolist() == b.tolist()


def test_wilcoxon_detecta_una_mejora_consistente():
    rng = np.random.default_rng(0)
    b = rng.random(20)
    resultado = hipotesis.comparar_por_paciente(b + 0.1, b)
    assert resultado.p_valor < 0.01


def test_h3_compara_con_y_sin_csp():
    from estadios.comun.config import Config

    entrenamiento = [tabla_sintetica(semilla=s) for s in range(3)]
    prueba = [tabla_sintetica(semilla=s) for s in range(10, 14)]
    resultado = hipotesis.h3_aporte_csp(entrenamiento, prueba, Config())
    assert set(resultado) == {"N1", "R"}
    assert len(resultado["N1"].variante_a) == 4
