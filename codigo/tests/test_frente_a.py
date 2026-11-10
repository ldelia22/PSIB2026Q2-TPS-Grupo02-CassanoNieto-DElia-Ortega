"""Especificación del Frente A: preprocesamiento."""

import numpy as np
import pytest

from estadios.comun.config import ConfigDescriptores, ConfigPreprocesamiento
from estadios.comun.constantes import CANALES, CANALES_EEG, ECG, EEG_F4, EMG, EOG_E1, EOG_E2
from estadios.comun.contratos import Registro
from estadios.comun.sintetico import preprocesado_sintetico, registro_sintetico
from estadios.descriptores import hrv
from estadios.preprocesamiento import cardiaco, exploratorio, filtrado, ocular, qrs, segmentacion

FS = 256.0


def _amplitud_en(x, f0, fs=FS):
    """Amplitud de la componente de frecuencia f0."""
    t = np.arange(len(x)) / fs
    return 2 * abs(np.mean(x * np.exp(-2j * np.pi * f0 * t)))


@pytest.mark.parametrize("tipo", ["iir", "fir"])
def test_filtro_quita_deriva_y_red_y_conserva_ritmo(tipo):
    t = np.arange(int(120 * FS)) / FS
    ritmo = 20 * np.sin(2 * np.pi * 10 * t)
    x = ritmo + 50 * np.sin(2 * np.pi * 0.05 * t) + 30 * np.sin(2 * np.pi * 50 * t)
    y = filtrado.filtrar_senal(x, FS, 0.5, None, 50.0, tipo)

    centro = slice(int(30 * FS), int(90 * FS))
    assert _amplitud_en(y[centro], 50, FS) < 3  # Al menos 20 dB menos.
    assert _amplitud_en(y[centro], 0.05, FS) < 5
    assert _amplitud_en(y[centro], 10, FS) == pytest.approx(20, rel=0.05)
    # Sin desfase: la componente de 10 Hz sigue alineada con la original.
    assert np.corrcoef(y[centro], ritmo[centro])[0, 1] > 0.95


def test_filtrar_registro_no_toca_el_ecg():
    registro = registro_sintetico(minutos=2)
    filtrado_ = filtrado.filtrar_registro(registro, ConfigPreprocesamiento())
    np.testing.assert_array_equal(filtrado_.senales[ECG], registro.senales[ECG])


def _eeg_con_fuga(semilla=0, n=int(120 * FS)):
    rng = np.random.default_rng(semilla)
    # Amplitudes realistas: la referencia ronda los 30 µV y el EEG, los 5 µV.
    referencia = 30 * np.convolve(rng.standard_normal(n), np.ones(64) / 8, mode="same")
    cerebro = 5 * rng.standard_normal(n)
    return cerebro, referencia, cerebro + 0.5 * referencia


def _rms(x):
    return np.sqrt(np.mean(x**2))


def test_nlms_reduce_la_fuga():
    cerebro, referencia, eeg = _eeg_con_fuga()
    limpio = ocular.nlms(eeg, referencia, orden=5, mu=0.01)
    tramo = len(eeg) // 2  # Se ignora el tramo inicial de convergencia.
    assert _rms(limpio[tramo:] - cerebro[tramo:]) < 0.5 * _rms(0.5 * referencia[tramo:])


def test_wiener_reduce_la_fuga():
    cerebro, referencia, eeg = _eeg_con_fuga()
    limpio = ocular.wiener(eeg, referencia, orden=5)
    assert _rms(limpio - cerebro) < 0.5 * _rms(0.5 * referencia)


def test_pan_tompkins_encuentra_los_latidos():
    verdad = preprocesado_sintetico(minutos=2)
    detectados = qrs.detectar_latidos(verdad.registro.senales[ECG], FS)

    tolerancia = 0.05 * FS
    aciertos = sum(np.min(np.abs(detectados - v)) <= tolerancia for v in verdad.latidos)
    assert aciertos / len(verdad.latidos) > 0.95  # Sensibilidad.
    assert aciertos / len(detectados) > 0.95  # Valor predictivo positivo.


def test_restar_plantilla_cancela_el_artefacto():
    rng = np.random.default_rng(0)
    n = int(300 * FS)
    latidos = np.arange(int(FS), n - int(FS), int(0.9 * FS))
    t = np.arange(-int(0.2 * FS), int(0.6 * FS))
    forma = 20 * np.exp(-0.5 * (t / (0.03 * FS)) ** 2)

    senal = 5 * rng.standard_normal(n)
    for latido in latidos:
        senal[latido + t] += forma

    limpia = cardiaco.restar_plantilla(senal, latidos, FS, 30, (-0.2, 0.6))
    _, antes = exploratorio.promedio_alrededor_de_latidos(senal, latidos, FS)
    _, despues = exploratorio.promedio_alrededor_de_latidos(limpia, latidos, FS)
    assert np.abs(despues).max() < 0.2 * np.abs(antes).max()


def test_promedio_alrededor_de_latidos_recupera_la_forma():
    n = int(60 * FS)
    latidos = np.arange(int(FS), n - int(FS), int(FS))
    senal = np.zeros(n)
    senal[latidos] = 7.0
    t, promedio = exploratorio.promedio_alrededor_de_latidos(senal, latidos, FS)
    assert len(t) == len(promedio)
    assert promedio[np.argmin(np.abs(t))] == pytest.approx(7.0)


def test_prominencia_de_linea_distingue_50_de_60():
    x = np.random.default_rng(0).standard_normal(int(120 * FS)) + np.sin(
        2 * np.pi * 50 * np.arange(int(120 * FS)) / FS
    )
    f, psd = exploratorio.espectro_del_registro(x, FS)
    prominencia = exploratorio.prominencia_de_linea(f, psd)
    assert prominencia[50.0] > 10
    assert abs(prominencia[60.0]) < 3


def test_encabezado_edf_lee_rango_y_prefiltro(tmp_path):
    canales = [
        ("EEG C4-M1", "uV", "-800", "800", "HP:0.2Hz LP:35.0Hz"),
        ("ECG", "uV", "-2400", "2400", "HP:1.0Hz LP:150.0Hz"),
    ]
    valores = {
        "canal": [c[0] for c in canales], "transductor": ["", ""], "unidad": [c[1] for c in canales],
        "minimo": [c[2] for c in canales], "maximo": [c[3] for c in canales],
        "minimo_digital": ["-32768"] * 2, "maximo_digital": ["32767"] * 2,
        "prefiltro": [c[4] for c in canales],
    }
    general = (" " * 252 + "2".ljust(4)).encode("latin-1")
    por_canal = "".join(v.ljust(largo) for nombre, largo in exploratorio.CAMPOS_EDF for v in valores[nombre])
    ruta = tmp_path / "prueba.edf"
    ruta.write_bytes(general + por_canal.encode("latin-1") + b" " * (2 * (8 + 32)))

    encabezado = exploratorio.encabezado_edf(ruta)
    assert encabezado["EEG C4-M1"]["maximo"] == 800.0
    assert encabezado["ECG"]["minimo"] == -2400.0
    assert encabezado["EEG C4-M1"]["prefiltro"] == "HP:0.2Hz LP:35.0Hz"


def test_emg_conserva_las_altas_frecuencias_y_pierde_las_bajas():
    registro = registro_sintetico(minutos=2)
    t = np.arange(len(registro.senales[EMG])) / FS
    senal = 20 * np.sin(2 * np.pi * 80 * t) + 20 * np.sin(2 * np.pi * 2 * t)
    filtrado_ = filtrado.filtrar_registro(
        registro.reemplazar_senales({EMG: senal}), ConfigPreprocesamiento()
    )
    centro = filtrado_.senales[EMG][int(30 * FS) : int(90 * FS)]
    assert _amplitud_en(centro, 80) == pytest.approx(20, rel=0.1)
    assert _amplitud_en(centro, 2) < 1


def test_pan_tompkins_con_ecg_invertido():
    verdad = preprocesado_sintetico(minutos=2)
    detectados = qrs.detectar_latidos(-verdad.registro.senales[ECG], FS)
    aciertos = sum(np.min(np.abs(detectados - v)) <= 0.05 * FS for v in verdad.latidos)
    assert aciertos / len(verdad.latidos) > 0.95


def test_filtro_ocular_no_toca_los_tramos_sin_movimientos():
    n = int(120 * FS)
    rng = np.random.default_rng(0)
    referencia = rng.standard_normal(n)
    referencia[int(40 * FS) : int(60 * FS)] *= 30  # Único tramo con movimientos oculares.
    eeg = 5 * rng.standard_normal(n) + 0.5 * referencia
    senales = {canal: rng.standard_normal(n) for canal in CANALES}
    senales.update({canal: eeg.copy() for canal in CANALES_EEG})
    senales[EOG_E1], senales[EOG_E2] = referencia, np.zeros(n)
    registro = Registro("X", FS, senales, np.zeros(4, dtype=int))

    limpio = ocular.remover_artefacto_ocular(registro, ConfigPreprocesamiento())
    tranquilo = slice(int(5 * FS), int(30 * FS))
    movido = slice(int(45 * FS), int(55 * FS))
    np.testing.assert_allclose(limpio.senales[EEG_F4][tranquilo], eeg[tranquilo])
    assert _rms(limpio.senales[EEG_F4][movido]) < 0.5 * _rms(eeg[movido])


def test_marcar_no_estacionarias_devuelve_una_marca_por_epoca():
    registro = registro_sintetico(minutos=2)
    marcas = segmentacion.marcar_no_estacionarias(registro, ConfigPreprocesamiento())
    assert marcas.shape == (registro.n_epocas,)
    assert marcas.dtype == bool


def test_errores_por_tramo_detecta_un_cambio_de_amplitud():
    rng = np.random.default_rng(0)
    quieta = rng.standard_normal(int(30 * FS))
    cambia = quieta.copy()
    cambia[int(20 * FS) :] *= 3  # La potencia crece 9 veces en los últimos 10 s.
    assert segmentacion.errores_por_tramo(quieta, FS)[1:].max() < segmentacion.CAMBIO_MAXIMO
    assert segmentacion.errores_por_tramo(cambia, FS)[1:].max() > segmentacion.CAMBIO_MAXIMO
    assert segmentacion.errores_por_tramo(np.zeros(int(30 * FS)), FS) is None


# La HRV es del Frente A aunque el módulo esté en descriptores/: usa los latidos de qrs.py.


def test_intervalos_rr():
    latidos = np.array([0, 256, 512, 896])
    t, rr = hrv.intervalos_rr(latidos, FS)
    assert len(t) == len(rr) == 3
    assert rr == pytest.approx([1.0, 1.0, 1.5])


@pytest.mark.parametrize("f0, domina_hf", [(0.25, True), (0.08, False)])
def test_lomb_ubica_la_oscilacion_en_su_banda(f0, domina_hf):
    rng = np.random.default_rng(0)
    # Muestras irregulares: es justamente el caso que resuelve Lomb.
    t = np.cumsum(0.9 + 0.1 * rng.random(330))
    rr = 0.9 + 0.04 * np.sin(2 * np.pi * f0 * t)
    lf, hf = hrv.potencias_lomb(t, rr)
    assert (hf > lf) == domina_hf


def test_descartar_ectopicos_quita_el_latido_deforme():
    prep = preprocesado_sintetico(minutos=2)
    ecg = prep.registro.senales["ECG"].copy()
    latidos = prep.latidos
    # El latido 20 se invierte y se ensancha: pasa a tener otra forma.
    centro = latidos[20]
    ancho = int(0.2 * FS)
    t = np.arange(-ancho, ancho)
    ecg[centro - ancho : centro + ancho] = -500 * np.exp(-0.5 * (t / (0.06 * FS)) ** 2)

    conservados = hrv.descartar_ectopicos(ecg, latidos, FS)
    assert centro not in conservados
    assert len(conservados) >= len(latidos) - 2


def test_descriptores_hrv_respetan_el_contrato():
    prep = preprocesado_sintetico(minutos=10)
    matriz, nombres = hrv.descriptores_hrv(prep, ConfigDescriptores())
    assert matriz.shape == (prep.registro.n_epocas, len(nombres))
    assert all(n.startswith("hrv.") for n in nombres)
    assert np.isfinite(matriz).all()
