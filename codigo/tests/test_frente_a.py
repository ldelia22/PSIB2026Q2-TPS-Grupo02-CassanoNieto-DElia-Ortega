"""Especificación del Frente A: preprocesamiento."""

import numpy as np
import pytest

from estadios.comun.config import ConfigPreprocesamiento
from estadios.comun.constantes import ECG
from estadios.comun.sintetico import preprocesado_sintetico, registro_sintetico
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


def test_marcar_no_estacionarias_devuelve_una_marca_por_epoca():
    registro = registro_sintetico(minutos=2)
    marcas = segmentacion.marcar_no_estacionarias(registro, ConfigPreprocesamiento())
    assert marcas.shape == (registro.n_epocas,)
    assert marcas.dtype == bool
