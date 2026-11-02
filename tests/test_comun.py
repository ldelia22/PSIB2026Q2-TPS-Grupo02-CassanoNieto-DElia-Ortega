"""Contratos, configuración, carga, partición y datos sintéticos."""

from dataclasses import replace

import numpy as np
import pytest

from estadios.comun.config import Config, ConfigClasificacion
from estadios.comun.constantes import CANALES, N3, R, SIN_ANOTAR, W
from estadios.comun.contratos import Registro, TablaDescriptores
from estadios.comun.sintetico import preprocesado_sintetico, registro_sintetico, tabla_sintetica
from estadios.datos.carga import leer_anotaciones
from estadios.datos.particion import elegir_registros, particion_por_paciente
from estadios.preprocesamiento.segmentacion import segmentar

ANOTACIONES = """Date, Time, Recording onset, Duration, Annotation, Linked channel
01.01.01, 23.59.30, 0, 30, Sleep stage W,
02.01.01, 00.00.00, 30, 30, Sleep stage W,
02.01.01, 00.00.03.430, 33.43, 0, Lights off, EEG F4-A1
02.01.01, 00.00.30, 60, 30, Sleep stage N3,
02.01.01, 00.01.30, 120, 30, Sleep stage R,
02.01.01, 00.02.00, 150, 0, Lights on, EEG Fpz-Cz"""


def test_leer_anotaciones(tmp_path):
    ruta = tmp_path / "a.txt"
    ruta.write_text(ANOTACIONES)
    estadios, apagadas, encendidas = leer_anotaciones(ruta)
    # La época 3 (90 s) falta en el archivo: tiene que quedar marcada.
    assert estadios.tolist() == [W, W, N3, SIN_ANOTAR, R]
    assert apagadas == 33.43
    assert encendidas == 150


def test_registro_rechaza_canales_de_distinto_largo():
    with pytest.raises(ValueError):
        Registro("x", 256.0, {"a": np.zeros(10), "b": np.zeros(11)}, np.array([], dtype=int))


def test_registro_rechaza_mas_estadios_que_epocas():
    with pytest.raises(ValueError):
        Registro("x", 256.0, {"a": np.zeros(7680)}, np.zeros(2, dtype=int))


def test_particion_sin_pacientes_en_comun_y_reproducible():
    ids = [f"SN{i:03d}" for i in range(1, 71)]
    entrenamiento, prueba = particion_por_paciente(ids)
    assert not set(entrenamiento) & set(prueba)
    assert len(entrenamiento) == 56 and len(prueba) == 14
    assert (entrenamiento, prueba) == particion_por_paciente(list(reversed(ids)))


def test_elegir_registros_reproducible():
    ids = [f"SN{i:03d}" for i in range(1, 152)]
    assert elegir_registros(ids, 70) == elegir_registros(list(reversed(ids)), 70)
    assert len(elegir_registros(ids, 70)) == 70


def test_segmentar_monocanal_y_multicanal():
    fs = 256.0
    n = int(30 * fs)
    x = np.arange(3 * n + 10, dtype=float)
    epocas = segmentar(x, fs, 3)
    assert epocas.shape == (3, n)
    assert epocas[1, 0] == n

    multicanal = np.stack([x, -x])
    epocas = segmentar(multicanal, fs, 3)
    assert epocas.shape == (3, 2, n)
    assert epocas[2, 1, 0] == -2 * n


def test_tabla_ida_y_vuelta_por_disco(tmp_path):
    tabla = tabla_sintetica(n_epocas=20)
    ruta = tmp_path / "sub" / "t.npz"
    tabla.guardar(ruta)
    leida = TablaDescriptores.cargar(ruta)
    assert leida.id == tabla.id
    assert leida.nombres == tabla.nombres
    np.testing.assert_array_equal(leida.matriz, tabla.matriz)
    np.testing.assert_array_equal(leida.covarianzas_eeg, tabla.covarianzas_eeg)


def test_tabla_seleccionar_grupos():
    tabla = tabla_sintetica(n_epocas=20).seleccionar(("hrv",))
    assert tabla.nombres == ["hrv.hf", "hrv.lf_hf"]
    assert tabla.matriz.shape == (20, 2)


def test_huella_cambia_con_el_preprocesamiento_pero_no_con_el_clasificador():
    base = Config()
    sin_ocular = replace(
        base, preprocesamiento=replace(base.preprocesamiento, remover_ocular=False)
    )
    sin_hmm = replace(base, clasificacion=ConfigClasificacion(usar_hmm=False))
    assert base.huella_descriptores() != sin_ocular.huella_descriptores()
    assert base.huella_descriptores() == sin_hmm.huella_descriptores()


def test_sinteticos_respetan_los_contratos():
    registro = registro_sintetico(minutos=5)
    assert set(registro.senales) == set(CANALES)
    assert registro.n_epocas == 10
    assert registro.eeg().shape == (4, 10 * 7680)

    prep = preprocesado_sintetico(minutos=5)
    assert np.all(np.diff(prep.latidos) > 0)
    assert prep.no_estacionaria.shape == (10,)

    tabla = tabla_sintetica(n_epocas=50)
    assert tabla.matriz.shape == (50, len(tabla.nombres))
    assert tabla.covarianzas_eeg.shape == (50, 4, 4)
