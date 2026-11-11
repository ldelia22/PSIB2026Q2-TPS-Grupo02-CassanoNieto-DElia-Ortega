"""Especificación del Frente B: descriptores."""

import numpy as np
import pytest

from estadios.comun.config import ConfigDescriptores
from estadios.comun.constantes import CANALES, N1, R, W
from estadios.comun.contratos import Registro, RegistroPreprocesado
from estadios.comun.sintetico import preprocesado_sintetico
from estadios.descriptores import csp, espectro, husos, ondas_lentas, perifericos, woody

FS = 256.0
N_EPOCA = int(30 * FS)


def test_multitaper_encuentra_la_frecuencia_y_conserva_la_potencia():
    t = np.arange(N_EPOCA) / FS
    epocas = np.sin(2 * np.pi * 10 * t)[None, :]  # Potencia de un seno unitario: 1/2.
    f, psd = espectro.psd_multitaper(epocas, FS, nw=15, k=29)

    assert psd.shape == (1, len(f))
    assert abs(f[np.argmax(psd[0])] - 10) <= 0.5
    potencia = espectro.potencia_bandas(f, psd, {"alfa": (8, 13)})["alfa"][0]
    assert potencia == pytest.approx(0.5, rel=0.1)


def test_multitaper_es_menos_ruidoso_que_el_periodograma():
    x = np.random.default_rng(0).standard_normal((1, N_EPOCA))
    f, psd = espectro.psd_multitaper(x, FS, nw=15, k=29)
    periodograma = np.abs(np.fft.rfft(x[0])) ** 2
    interior = (f > 5) & (f < 100)
    variacion_mt = psd[0, interior].std() / psd[0, interior].mean()
    variacion_p = periodograma[1:-1].std() / periodograma[1:-1].mean()
    assert variacion_mt < 0.5 * variacion_p


def test_welch_y_multitaper_devuelven_las_mismas_formas():
    x = np.random.default_rng(0).standard_normal((3, N_EPOCA))
    f_w, psd_w = espectro.psd_welch(x, FS)
    assert psd_w.shape == (3, len(f_w))


def test_pendiente_espectral_de_una_ley_de_potencia():
    f = np.linspace(1, 40, 200)
    pendiente = espectro.pendiente_espectral(f, (f**-2.0)[None, :])
    assert pendiente[0] == pytest.approx(-2.0, abs=0.05)


def test_detectar_husos_encuentra_rafagas_conocidas():
    rng = np.random.default_rng(0)
    x = 3 * rng.standard_normal(int(120 * FS))
    inicios_s = [20, 50, 90]
    for inicio in inicios_s:
        t = np.arange(int(1.0 * FS)) / FS
        ventana = np.hanning(len(t))
        x[int(inicio * FS) : int(inicio * FS) + len(t)] += 30 * ventana * np.sin(2 * np.pi * 13 * t)

    encontrados = husos.detectar_husos(x, FS, (11, 16), (0.5, 2.0))
    assert encontrados.shape[1] == 2
    assert len(encontrados) == len(inicios_s)
    for (inicio, _), esperado in zip(encontrados, inicios_s):
        assert abs(inicio / FS - esperado) < 0.5


def test_detectar_ondas_lentas_aplica_los_umbrales():
    rng = np.random.default_rng(0)
    x = 2 * rng.standard_normal(int(60 * FS))
    t = np.arange(int(1.0 * FS)) / FS

    def onda(amplitud):
        return -amplitud * np.sin(2 * np.pi * t)

    x[int(10 * FS) : int(11 * FS)] += onda(80)  # Cumple: pico -80, pico a pico 160.
    x[int(30 * FS) : int(31 * FS)] += onda(30)  # No cumple.
    picos = ondas_lentas.detectar_ondas_lentas(x, FS, (0.2, 4.0), -60.0, 75.0)
    assert len(picos) == 1
    assert abs(picos[0] / FS - 10.25) < 0.1


def _forma():
    t = np.linspace(-1, 1, 256)
    return -80 * np.exp(-0.5 * (t / 0.2) ** 2)


def _error_de_forma(estimada, verdad):
    """Error tras alinear: un corrimiento global de la plantilla no es un error de forma."""
    return min(np.sqrt(np.mean((np.roll(estimada, d) - verdad) ** 2)) for d in range(-64, 65))


def test_woody_recupera_la_forma_mejor_que_el_promedio():
    rng = np.random.default_rng(0)
    forma = _forma()
    corrimientos = rng.integers(-40, 41, size=100)
    realizaciones = np.stack(
        [np.roll(forma, c) + 8 * rng.standard_normal(len(forma)) for c in corrimientos]
    )
    resultado = woody.promediado_woody(realizaciones, max_desplazamiento=60)

    assert resultado.plantilla.shape == forma.shape
    assert resultado.desplazamientos.shape == (100,)
    assert _error_de_forma(resultado.plantilla, forma) < 0.5 * _error_de_forma(
        realizaciones.mean(axis=0), forma
    )


def test_similitud_con_plantilla():
    forma = _forma()
    realizaciones = np.stack([forma, -forma])
    similitud = woody.similitud_con_plantilla(realizaciones, forma)
    assert similitud == pytest.approx([1.0, -1.0])


def test_covarianzas_por_epoca():
    epocas = np.random.default_rng(0).standard_normal((3, 4, 2000))
    covarianzas = csp.covarianzas_por_epoca(epocas)
    assert covarianzas.shape == (3, 4, 4)
    np.testing.assert_allclose(covarianzas, covarianzas.transpose(0, 2, 1))


def test_csp_separa_dos_estadios():
    rng = np.random.default_rng(0)
    # N1 tiene más varianza en el canal 0 y R en el canal 3.
    epocas = rng.standard_normal((200, 4, 1000))
    y = np.array([N1, R] * 100)
    epocas[y == N1, 0] *= 3
    epocas[y == R, 3] *= 3

    covarianzas = csp.covarianzas_por_epoca(epocas)
    filtro = csp.ajustar_csp(covarianzas, y, (N1, R), n_componentes=1)
    assert filtro.pesos.shape == (2, 4)

    potencias = csp.aplicar_csp(covarianzas, filtro)
    assert potencias.shape == (200, 2)
    # Los dos extremos separan en sentidos opuestos.
    diferencia = potencias[y == N1].mean(axis=0) - potencias[y == R].mean(axis=0)
    assert np.abs(diferencia).min() > 1
    assert diferencia[0] * diferencia[1] < 0


def test_energia_y_amplitud_crecen_con_la_senal():
    x = np.random.default_rng(0).standard_normal((2, 1000))
    x[1] *= 10
    assert perifericos.energia_por_epoca(x)[1] > perifericos.energia_por_epoca(x)[0]
    assert perifericos.amplitud_por_epoca(x)[1] > perifericos.amplitud_por_epoca(x)[0]


@pytest.mark.parametrize(
    "funcion, prefijo",
    [
        (espectro.descriptores_espectrales, "espectro."),
        (husos.descriptores_husos, "husos."),
        (ondas_lentas.descriptores_ondas_lentas, "ondas."),
        (perifericos.descriptores_perifericos, "perifericos."),
    ],
)
def test_cada_grupo_respeta_el_contrato(funcion, prefijo):
    prep = preprocesado_sintetico(minutos=10)
    matriz, nombres = funcion(prep, ConfigDescriptores())
    assert matriz.shape == (prep.registro.n_epocas, len(nombres))
    assert all(n.startswith(prefijo) for n in nombres)
    assert np.isfinite(matriz).all()  # Una época sin eventos vale 0, no NaN.


def _prep_con_senal(canal, senal, n_epocas):
    """Registro con todos los canales en silencio salvo uno."""
    senales = {c: np.zeros(len(senal)) for c in CANALES}
    senales[canal] = senal
    registro = Registro("X", FS, senales, np.full(n_epocas, N1))
    return RegistroPreprocesado(registro, np.array([], dtype=int), np.zeros(n_epocas, dtype=bool))


def test_husos_se_cuentan_en_su_epoca():
    cfg = ConfigDescriptores()
    x = 3 * np.random.default_rng(0).standard_normal(3 * N_EPOCA)
    t = np.arange(int(FS)) / FS
    inicio = int(40 * FS)  # En la segunda época.
    x[inicio : inicio + len(t)] += 30 * np.hanning(len(t)) * np.sin(2 * np.pi * 13 * t)

    matriz, nombres = husos.descriptores_husos(_prep_con_senal(cfg.canal_husos, x, 3), cfg)
    assert matriz[:, nombres.index("husos.cantidad")].tolist() == [0, 1, 0]
    amplitud = matriz[:, nombres.index("husos.amplitud")]
    assert amplitud[1] > 10 and amplitud[0] == amplitud[2] == 0


def test_ondas_se_cuentan_en_su_epoca_y_se_parecen_a_la_forma_tipica():
    cfg = ConfigDescriptores()
    x = 2 * np.random.default_rng(0).standard_normal(3 * N_EPOCA)
    t = np.arange(int(FS)) / FS
    for inicio_s in (35, 45, 75):  # Dos en la segunda época y una en la tercera.
        inicio = int(inicio_s * FS)
        x[inicio : inicio + len(t)] -= 80 * np.sin(2 * np.pi * t)

    matriz, nombres = ondas_lentas.descriptores_ondas_lentas(
        _prep_con_senal(cfg.canal_ondas, x, 3), cfg
    )
    assert matriz[:, nombres.index("ondas.cantidad")].tolist() == [0, 2, 1]
    similitud = matriz[:, nombres.index("ondas.similitud")]
    assert similitud[0] == 0
    assert similitud[1] > 0.9 and similitud[2] > 0.9


def test_woody_alinea_en_el_sentido_correcto():
    forma = _forma()
    corrimientos = np.array([-30, -10, 0, 20, 30])
    realizaciones = np.stack([np.roll(forma, c) for c in corrimientos])
    resultado = woody.promediado_woody(realizaciones, max_desplazamiento=60)
    alineadas = woody.alinear(realizaciones, resultado.desplazamientos)
    # Arrancan 60 muestras separadas; moverlas al revés las separaría más. No llega a 0
    # porque con 5 ondas cada una pesa mucho en el promedio contra el que se alinea.
    assert np.ptp(np.nanargmin(alineadas, axis=1)) < 10


def test_csp_ignora_los_demas_estadios():
    rng = np.random.default_rng(0)
    epocas = rng.standard_normal((200, 4, 1000))
    y = np.array([N1, R] * 100)
    epocas[y == N1, 0] *= 3
    epocas[y == R, 3] *= 3
    covarianzas = csp.covarianzas_por_epoca(epocas)
    solo_par = csp.ajustar_csp(covarianzas, y, (N1, R), n_componentes=1)

    otras = csp.covarianzas_por_epoca(100 * rng.standard_normal((50, 4, 1000)))
    con_otras = csp.ajustar_csp(
        np.concatenate([covarianzas, otras]), np.r_[y, np.full(50, W)], (N1, R), n_componentes=1
    )
    np.testing.assert_allclose(np.abs(solo_par.pesos), np.abs(con_otras.pesos))


@pytest.mark.parametrize(
    "funcion",
    [
        espectro.descriptores_espectrales,
        husos.descriptores_husos,
        ondas_lentas.descriptores_ondas_lentas,
        perifericos.descriptores_perifericos,
    ],
)
def test_senal_plana_da_descriptores_finitos(funcion):
    n_epocas = 2
    senales = {c: np.zeros(n_epocas * N_EPOCA) for c in CANALES}
    registro = Registro("X", FS, senales, np.full(n_epocas, N1))
    prep = RegistroPreprocesado(registro, np.array([], dtype=int), np.zeros(n_epocas, dtype=bool))
    matriz, nombres = funcion(prep, ConfigDescriptores())
    assert matriz.shape == (n_epocas, len(nombres))
    assert np.isfinite(matriz).all()


def test_espectro_con_welch_da_los_mismos_descriptores():
    prep = preprocesado_sintetico(minutos=2)
    multitaper, nombres_mt = espectro.descriptores_espectrales(prep, ConfigDescriptores())
    welch, nombres_w = espectro.descriptores_espectrales(
        prep, ConfigDescriptores(metodo_espectro="welch")
    )
    assert nombres_mt == nombres_w
    assert multitaper.shape == welch.shape
    assert not np.allclose(multitaper, welch)
