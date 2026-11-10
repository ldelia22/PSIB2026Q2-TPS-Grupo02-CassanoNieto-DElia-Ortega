"""Especificación del Frente C: clasificación y evaluación."""

from dataclasses import replace

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


def _registros_ruido_blanco(n_registros=6, minutos=10, fs=128.0):
    from estadios.comun.constantes import EEG_C4
    from estadios.comun.contratos import Registro

    registros = []
    for semilla in range(n_registros):
        rng = np.random.default_rng(semilla)
        n_epocas = int(minutos * 2)
        senal = 20 * rng.standard_normal(int(n_epocas * 30 * fs))
        registros.append(Registro(f"RB{semilla}", fs, {EEG_C4: senal}, np.full(n_epocas, W)))
    return registros


def test_h1_multitaper_varia_menos_por_frecuencia_en_ruido_blanco():
    resultado = hipotesis.h1_variabilidad_espectral(_registros_ruido_blanco(), por_frecuencia=True)
    assert (resultado.variante_a < resultado.variante_b).all()


def test_h2_woody_recupera_mejor_la_forma():
    rng = np.random.default_rng(0)
    fs = 128.0
    eeg = [20 * rng.standard_normal(int(300 * fs)) for _ in range(6)]
    resultado = hipotesis.h2_recuperacion_de_forma(eeg, fs)
    assert len(resultado.variante_a) == 6
    assert (resultado.variante_a < resultado.variante_b).all()


def test_h4_residuo_compara_antes_y_despues(monkeypatch):
    """Con el Frente A reemplazado por versiones mínimas: verifica solo el diseño de H4."""
    from estadios.comun.config import Config
    from estadios.comun.constantes import CANALES_EEG
    from estadios.comun.sintetico import preprocesado_sintetico
    from estadios.preprocesamiento import cardiaco, exploratorio, filtrado, qrs

    preps = [preprocesado_sintetico(minutos=5, semilla=s) for s in range(5)]
    latidos = {p.registro.id: p.latidos for p in preps}

    def promedio(senal, lat, fs, ventana_s=(-0.2, 0.6)):
        desde, hasta = int(ventana_s[0] * fs), int(ventana_s[1] * fs)
        lat = lat[(lat + desde >= 0) & (lat + hasta < len(senal))]
        tramos = np.stack([senal[i + desde : i + hasta] for i in lat])
        return np.arange(desde, hasta) / fs, tramos.mean(axis=0)

    monkeypatch.setattr(qrs, "detectar_latidos", lambda ecg, fs: next(
        latidos[p.registro.id] for p in preps if p.registro.senales["ECG"] is ecg))
    monkeypatch.setattr(filtrado, "filtrar_registro", lambda r, cfg: r)
    monkeypatch.setattr(cardiaco, "remover_artefacto_cardiaco",
                        lambda r, lat, cfg: r.reemplazar_senales({c: 0 * r.senales[c] for c in CANALES_EEG}))
    monkeypatch.setattr(exploratorio, "promedio_alrededor_de_latidos", promedio)

    cfg = Config()
    cfg = replace(cfg, preprocesamiento=replace(cfg.preprocesamiento, remover_ocular=False))
    resultado = hipotesis.h4_residuo_cardiaco([p.registro for p in preps], cfg)
    assert (resultado.variante_a == 0).all()
    assert (resultado.variante_b > 0).all()


def test_referencia_csp_usa_el_ancho_de_cada_estadio():
    rng = np.random.default_rng(0)
    puntaje = np.r_[rng.normal(2, 0.5, 500), rng.normal(-2, 0.5, 500)]
    estadios = np.r_[np.full(500, N1), np.full(500, R)]
    centro, escala = reglas.referencia_csp(puntaje, estadios, (N1, R))
    assert abs(centro) < 0.2
    assert 0.5 < escala < 0.9  # Rango intercuartil de una normal de desvío 0,5: ~0,67.


def test_percentiles_ignoran_nan():
    p = reglas.percentil_por_registro(np.array([10.0, np.nan, 30.0, 20.0]))[:, 0]
    assert np.isnan(p[1])
    assert p[[0, 2, 3]].tolist() == [0.0, 100.0, 50.0]


def test_mascara_periodo_de_sueno():
    from estadios.evaluacion.periodo import mascara_periodo_de_sueno

    mascara = mascara_periodo_de_sueno(10, 60.0, 240.0)  # Épocas 2 a 7.
    assert np.flatnonzero(mascara).tolist() == [2, 3, 4, 5, 6, 7]
    assert mascara_periodo_de_sueno(5, None, None).all()


def test_recortar_a_periodo_lee_las_luces(tmp_path):
    from estadios.evaluacion.periodo import recortar_a_periodo_de_sueno

    tabla = tabla_sintetica(n_epocas=20)
    (tmp_path / f"{tabla.id}_sleepscoring.txt").write_text(
        "Date, Time, Recording onset, Duration, Annotation, Linked channel\n"
        "2001-01-01, 22:00:00, 90.0, 0.0, Lights off,\n"
        "2001-01-01, 06:00:00, 450.0, 0.0, Lights on,\n"
    )
    recortada = recortar_a_periodo_de_sueno(tabla, tmp_path)
    assert len(recortada.estadios) == 12  # Épocas 3 a 14.
    np.testing.assert_array_equal(recortada.estadios, tabla.estadios[3:15])
    assert recortada.covarianzas_eeg.shape[0] == recortada.matriz.shape[0] == 12
    # Sin archivo de anotaciones (datos sintéticos), no cambia.
    assert recortar_a_periodo_de_sueno(tabla, tmp_path / "otro") is tabla
