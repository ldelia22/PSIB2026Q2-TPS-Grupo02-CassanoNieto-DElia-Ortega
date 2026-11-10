"""Etapa 11. Interfaz gráfica. Correr desde codigo/ con `streamlit run app/streamlit_app.py`.

Cuatro vistas, y solo se dibuja la elegida.
- Registro. Los ocho canales crudos de una época, con el estadio de los técnicos.
- Preprocesamiento. Cada etapa del Frente A sobre la época elegida, con lo que la justifica.
- Latidos y HRV. Detección de latidos, descarte por forma, intervalos RR y periodograma de Lomb.
- Clasificación. Hipnograma estimado contra el de los técnicos, métricas y la regla de cada época.

Sin datos descargados funciona con el registro y los descriptores sintéticos.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

# Se trabaja desde codigo/, igual que los notebooks, para encontrar data/ sin importar desde dónde se corra.
CODIGO = Path(__file__).resolve().parents[1]
os.chdir(CODIGO)

# Si el paquete no está instalado con `pip install -e .`, se usa directamente codigo/src.
try:
    import estadios  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(CODIGO / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import signal

from estadios.clasificacion import reglas
from estadios.clasificacion.modelo import ModeloEstadios
from estadios.comun.config import GRUPOS_DESCRIPTORES, Config, ConfigDescriptores
from estadios.comun.constantes import (
    BANDA_HF,
    BANDA_LF,
    CANALES_EEG,
    DURACION_EPOCA_S,
    ECG,
    EEG_C4,
    EMG,
    EOG_E1,
    EOG_E2,
    N1,
    N2,
    N3,
    N_ESTADIOS,
    NOMBRES_ESTADIOS,
    SIN_ANOTAR,
    R,
    W,
)
from estadios.comun.contratos import Registro
from estadios.comun.sintetico import registro_sintetico, tabla_sintetica
from estadios.datos.carga import cargar_registro, ids_disponibles
from estadios.datos.particion import particion_por_paciente
from estadios.descriptores import hrv
from estadios.evaluacion import metricas
from estadios.pipeline import obtener_tabla, preprocesar
from estadios.preprocesamiento import exploratorio, filtrado, ocular, segmentacion
from estadios.preprocesamiento.segmentacion import segmentar

FUENTE_SINTETICA = "Sintético (sin datos)"
CFG = Config()
LIMITE_SATURACION_UV = 799.0

AZUL, NARANJA, AGUA, AMARILLO, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRIS, GRIS_CLARO = "#52514e", "#b9b8b2"
COLOR_ESTADIO = {W: AZUL, N1: NARANJA, N2: AGUA, N3: AMARILLO, R: MAGENTA}
ALTURA = {W: 4, R: 3, N1: 2, N2: 1, N3: 0}
ETIQUETAS_ALTURA = ["N3", "N2", "N1", "R", "W"]
NOMBRE_ETAPA = {
    "crudo": "Crudo",
    "filtrado": "Filtrado",
    "sin_ocular": "Sin artefacto ocular",
    "sin_cardiaco": "Sin artefacto cardíaco",
}

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.edgecolor": "#8a8981", "lines.linewidth": 1.0, "legend.frameon": False,
})


def nombre_estadio(estadio: int) -> str:
    return NOMBRES_ESTADIOS[estadio] if estadio != SIN_ANOTAR else "sin anotar"


def coma(valor: float, decimales: int = 1) -> str:
    return f"{valor:.{decimales}f}".replace(".", ",")


def corto(canal: str) -> str:
    partes = canal.split()
    return partes[1].split("-")[0] if len(partes) > 1 else canal


# ---------------------------------------------------------------- Cálculos (en caché)


@st.cache_resource(max_entries=2)
def cargar(fuente: str) -> Registro:
    if fuente == FUENTE_SINTETICA:
        return registro_sintetico()
    return cargar_registro(fuente)


@st.cache_resource(max_entries=2)
def preprocesado(fuente: str):
    """RegistroPreprocesado con todas las etapas intermedias."""
    return preprocesar(cargar(fuente), CFG.preprocesamiento, guardar_intermedios=True)


@st.cache_resource(max_entries=2)
def compuerta_ocular(fuente: str) -> np.ndarray:
    """Ganancia de la compuerta ocular en cada muestra, calculada sobre el EOG filtrado."""
    prep = preprocesado(fuente)
    filtradas = prep.intermedios["filtrado"]
    referencia = filtradas[EOG_E1] - filtradas[EOG_E2]
    return ocular.actividad_ocular(referencia, prep.registro.fs, CFG.preprocesamiento)


@st.cache_data(max_entries=16)
def promedio_cardiaco(fuente: str, canal: str) -> dict:
    """Promedio del canal alrededor de los latidos antes y después de la remoción, y en instantes al azar."""
    prep = preprocesado(fuente)
    fs = prep.registro.fs
    antes = prep.intermedios["sin_ocular"][canal]
    despues = prep.intermedios["sin_cardiaco"][canal]
    azar = np.sort(np.random.default_rng(0).integers(int(fs), len(antes) - int(fs), len(prep.latidos)))
    t, pa = exploratorio.promedio_alrededor_de_latidos(antes, prep.latidos, fs)
    _, pd_ = exploratorio.promedio_alrededor_de_latidos(despues, prep.latidos, fs)
    _, pz = exploratorio.promedio_alrededor_de_latidos(antes, azar, fs)
    return {"t": t, "antes": pa, "despues": pd_, "azar": pz}


@st.cache_data(max_entries=4)
def cambios_ar(fuente: str) -> np.ndarray:
    """Mayor cociente de error de predicción AR de cada época de C4."""
    return segmentacion.cambio_de_error(preprocesado(fuente).registro)


@st.cache_resource(max_entries=2)
def analisis_latidos(fuente: str) -> dict:
    """Latidos, comparación con el latido típico, intervalos RR y descriptores de HRV de la noche."""
    prep = preprocesado(fuente)
    fs = prep.registro.fs
    ecg = prep.registro.senales[ECG]
    tipico, ventanas, correlacion, completos = hrv.comparar_con_latido_tipico(ecg, prep.latidos, fs)
    conservados = hrv.descartar_ectopicos(ecg, prep.latidos, fs)
    t_todos, rr_todos = hrv.intervalos_rr(conservados, fs)
    t_rr, rr = hrv.filtrar_intervalos(t_todos, rr_todos)
    matriz, _ = hrv.descriptores_hrv(prep, ConfigDescriptores())
    return {
        "latidos": prep.latidos, "conservados": conservados, "tipico": tipico, "ventanas": ventanas,
        "correlacion": correlacion, "t_todos": t_todos, "rr_todos": rr_todos, "t_rr": t_rr, "rr": rr,
        "percentiles": reglas.percentil_por_registro(matriz),
    }


@st.cache_resource(max_entries=2)
def tabla_y_entrenamiento(fuente: str):
    """Descriptores del registro y tablas de entrenamiento.

    Con la fuente sintética, los descriptores salen de `tabla_sintetica` y no del
    registro que se muestra en las otras pestañas.
    """
    if fuente == FUENTE_SINTETICA:
        return tabla_sintetica(semilla=100), [tabla_sintetica(semilla=s) for s in range(8)]
    entrenamiento_ids, _ = particion_por_paciente(ids_disponibles())
    entrenamiento = [obtener_tabla(i, CFG) for i in entrenamiento_ids if i != fuente]
    return obtener_tabla(fuente, CFG), entrenamiento


@st.cache_resource(max_entries=4)
def clasificacion(fuente: str, cfg_clasificacion):
    """Estadio de cada época sin suavizar y suavizado, y la regla que lo decidió."""
    tabla, entrenamiento = tabla_y_entrenamiento(fuente)
    modelo = ModeloEstadios(cfg_clasificacion).ajustar(entrenamiento)
    sin_suavizar, regla = modelo.clasificar_epocas(tabla)
    return tabla, sin_suavizar, modelo.suavizar(sin_suavizar), regla


# ---------------------------------------------------------------- Gráficos


def mostrar(fig) -> None:
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def tramo_epoca(fs: float, epoca: int) -> slice:
    n = int(DURACION_EPOCA_S * fs)
    return slice(epoca * n, (epoca + 1) * n)


def dibujar_hipnograma(ax, y: np.ndarray, marca: int | None = None) -> None:
    horas = np.arange(len(y)) * DURACION_EPOCA_S / 3600
    paso = DURACION_EPOCA_S / 3600
    altura = np.array([ALTURA.get(int(e), np.nan) for e in y])
    ax.step(horas, altura, where="post", c=GRIS_CLARO, lw=0.6)
    for estadio, color in COLOR_ESTADIO.items():
        x = horas[y == estadio]
        ax.hlines(np.full(len(x), ALTURA[estadio]), x, x + paso, colors=color, lw=4)
    if marca is not None:
        ax.axvline(marca * paso, c="#d62728", lw=1)
    ax.set_yticks(range(5), ETIQUETAS_ALTURA)
    ax.set_ylim(-0.6, 4.6)
    ax.set_xlim(0, len(y) * paso)
    ax.grid(axis="y", visible=False)


def graficar_canales(senales: dict[str, np.ndarray], t: np.ndarray, colores: dict[str, str], titulo: str):
    fig, axs = plt.subplots(len(senales), 1, figsize=(12, 1.15 * len(senales) + 0.4), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, (nombre, x) in zip(axs, senales.items()):
        ax.plot(t, x, c=colores.get(nombre, AZUL), lw=0.6)
        ax.set_ylabel(nombre, rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0].set_title(titulo, loc="left")
    axs[-1].set_xlabel("Tiempo dentro de la época (s)")
    axs[-1].set_xlim(t[0], t[-1])
    fig.tight_layout()
    return fig


def graficar_confusion(matriz: np.ndarray):
    normalizada = matriz / matriz.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(normalizada, cmap="Blues", vmin=0, vmax=1)
    for i in range(N_ESTADIOS):
        for j in range(N_ESTADIOS):
            color = "white" if normalizada[i, j] > 0.6 else "black"
            ax.text(j, i, f"{normalizada[i, j]:.2f}\n({matriz[i, j]})", ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks(range(N_ESTADIOS), NOMBRES_ESTADIOS)
    ax.set_yticks(range(N_ESTADIOS), NOMBRES_ESTADIOS)
    ax.set(xlabel="Estimado", ylabel="Técnicos")
    ax.grid(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- Navegación


def mover_epoca(paso: int, n: int) -> None:
    st.session_state.epoca = int(np.clip(st.session_state.epoca + paso, 1, n))


def ir_a_estadio(estadios: np.ndarray) -> None:
    objetivo = NOMBRES_ESTADIOS.index(st.session_state.destino)
    candidatas = np.flatnonzero(estadios == objetivo)
    if len(candidatas) == 0:
        return
    siguientes = candidatas[candidatas > st.session_state.epoca - 1]
    st.session_state.epoca = int((siguientes[0] if len(siguientes) else candidatas[0]) + 1)


def barra_lateral() -> tuple[str, Registro, int, object]:
    st.sidebar.title("Estadios de sueño")
    fuente = st.sidebar.selectbox("Registro", [FUENTE_SINTETICA, *ids_disponibles()])
    registro = cargar(fuente)
    n = registro.n_epocas

    st.session_state.setdefault("epoca", 1)
    if st.session_state.epoca > n:
        st.session_state.epoca = 1
    st.sidebar.slider("Época", 1, n, key="epoca")
    izquierda, derecha = st.sidebar.columns(2)
    izquierda.button("Anterior", on_click=mover_epoca, args=(-1, n), width="stretch")
    derecha.button("Siguiente", on_click=mover_epoca, args=(1, n), width="stretch")
    izquierda, derecha = st.sidebar.columns([3, 2], vertical_alignment="bottom")
    izquierda.selectbox("Próxima época de", NOMBRES_ESTADIOS, key="destino")
    derecha.button("Ir", on_click=ir_a_estadio, args=(registro.estadios,), width="stretch")

    st.sidebar.divider()
    st.sidebar.subheader("Clasificador")
    suavizado = st.sidebar.radio("Suavizado", ["reglas", "moda", "ninguno"], horizontal=True)
    usar_csp = st.sidebar.checkbox("Desempate N1 y REM con CSP", value=True)
    grupos = tuple(g for g in GRUPOS_DESCRIPTORES if usar_csp or g != "csp")
    cfg_clasificacion = replace(CFG.clasificacion, suavizado=suavizado, grupos=grupos)
    return fuente, registro, st.session_state.epoca - 1, cfg_clasificacion


def encabezado(fuente: str, registro: Registro, epoca: int) -> None:
    st.title("Clasificación automática de estadios de sueño")
    st.caption(f"Registro {fuente}, {registro.n_epocas} épocas de 30 s ({coma(registro.n_epocas * DURACION_EPOCA_S / 3600)} h)")
    fig, ax = plt.subplots(figsize=(12, 1.7))
    dibujar_hipnograma(ax, registro.estadios, epoca)
    ax.set_xlabel("Tiempo (h)")
    fig.tight_layout()
    mostrar(fig)

    tramo = tramo_epoca(registro.fs, epoca)
    saturadas = sum(int((np.abs(registro.senales[c][tramo]) >= LIMITE_SATURACION_UV).sum()) for c in CANALES_EEG)
    marcada = bool(preprocesado(fuente).no_estacionaria[epoca])
    columnas = st.columns(4)
    columnas[0].metric("Época", f"{epoca + 1} de {registro.n_epocas}")
    columnas[1].metric("Estadio de los técnicos", nombre_estadio(int(registro.estadios[epoca])))
    columnas[2].metric("No estacionaria", "Sí" if marcada else "No")
    columnas[3].metric("Muestras saturadas en el EEG", saturadas)


# ---------------------------------------------------------------- Pestañas


def pestana_registro(registro: Registro, epoca: int) -> None:
    st.caption("Los ocho canales tal como vienen en el archivo, antes de cualquier procesamiento.")
    tramo = tramo_epoca(registro.fs, epoca)
    t = np.arange(tramo.stop - tramo.start) / registro.fs
    senales = {corto(c): x[tramo] for c, x in registro.senales.items()}
    colores = {corto(c): AZUL for c in CANALES_EEG} | {corto(EOG_E1): AGUA, corto(EOG_E2): AGUA,
                                                       corto(EMG): NARANJA, ECG: MAGENTA}
    titulo = f"Época {epoca + 1}, {nombre_estadio(int(registro.estadios[epoca]))} según los técnicos"
    mostrar(graficar_canales(senales, t, colores, titulo))


def seccion_cadena(prep, canal: str, epoca: int) -> None:
    st.caption("La misma época después de cada etapa. Cada etapa toma la salida de la anterior.")
    fs = prep.registro.fs
    tramo = tramo_epoca(fs, epoca)
    t = np.arange(tramo.stop - tramo.start) / fs
    senales = {NOMBRE_ETAPA.get(e, e): canales[canal][tramo] for e, canales in prep.intermedios.items()}
    colores = {nombre: (GRIS if nombre == "Crudo" else AZUL) for nombre in senales}
    mostrar(graficar_canales(senales, t, colores, f"{corto(canal)}, época {epoca + 1}"))


def seccion_filtrado(prep, canal: str, epoca: int) -> None:
    c = CFG.preprocesamiento
    st.caption(
        f"Pasa-altos de {c.pasa_altos_hz} Hz para conservar las ondas lentas, pasa-bajos de {c.pasa_bajos_hz:.0f} Hz "
        f"y notch de {c.notch_hz:.0f} Hz para la línea eléctrica. El IIR se aplica ida y vuelta, así que no desfasa."
    )
    fs = prep.registro.fs
    tramo = tramo_epoca(fs, epoca)
    izquierda, derecha = st.columns(2)
    with izquierda:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        visibles = []
        for etapa, color in (("crudo", GRIS_CLARO), ("filtrado", AZUL)):
            f, psd = signal.welch(prep.intermedios[etapa][canal][tramo], fs, nperseg=int(4 * fs))
            ax.semilogy(f, psd, c=color, label=NOMBRE_ETAPA[etapa])
            visibles.append(psd[(f >= 0.5) & (f <= 70)])
        for corte in (c.pasa_altos_hz, c.pasa_bajos_hz, c.notch_hz):
            ax.axvline(corte, c=NARANJA, ls="--", lw=0.8)
        visibles = np.concatenate(visibles)
        ax.set_ylim(max(visibles.min(), visibles.max() * 1e-8) / 2, visibles.max() * 3)
        ax.set(xlim=(0, 70), xlabel="Frecuencia (Hz)", ylabel="PSD (µV²/Hz)")
        ax.set_title(f"Espectro de {corto(canal)} en la época", loc="left")
        ax.legend()
        fig.tight_layout()
        mostrar(fig)
    with derecha:
        f, magnitud, _ = filtrado.respuesta_en_frecuencia(fs, c)
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.semilogx(f[1:], 40 * np.log10(np.maximum(magnitud[1:], 1e-12)), c=AZUL)
        ax.set(xlim=(0.05, fs / 2), ylim=(-80, 5), xlabel="Frecuencia (Hz)", ylabel="Ganancia (dB)")
        ax.set_title("Respuesta del filtro del EEG, ida y vuelta", loc="left")
        fig.tight_layout()
        mostrar(fig)


def seccion_ocular(fuente: str, prep, canal: str, epoca: int) -> None:
    st.caption(
        "La referencia es la diferencia E1 − E2. La compuerta vale 1 donde su valor eficaz en 2 s supera el doble "
        "de su mediana en la noche, y solo ahí se resta el artefacto. Así no se borran las ondas lentas de N3."
    )
    fs = prep.registro.fs
    tramo = tramo_epoca(fs, epoca)
    t = np.arange(tramo.stop - tramo.start) / fs
    ganancia = compuerta_ocular(fuente)
    filtradas = prep.intermedios["filtrado"]
    referencia = (filtradas[EOG_E1] - filtradas[EOG_E2])[tramo]

    columnas = st.columns(3)
    columnas[0].metric("Compuerta activa en la época", f"{100 * ganancia[tramo].mean():.0f} %")
    columnas[1].metric("Compuerta activa en la noche", f"{100 * (ganancia >= 1).mean():.0f} %")
    columnas[2].metric("Método", CFG.preprocesamiento.metodo_ocular.capitalize())

    fig, axs = plt.subplots(2, 1, figsize=(12, 4.2), sharex=True)
    axs[0].plot(t, referencia, c=AGUA, lw=0.6)
    techo = np.abs(referencia).max() * 1.05
    activa = ganancia[tramo] > 0.01
    if activa.any():
        axs[0].fill_between(t, -techo, techo, where=activa, color=NARANJA, alpha=0.15, lw=0, label="Compuerta activa")
        axs[0].legend(loc="upper right")
    axs[0].set(ylim=(-techo, techo), ylabel="E1 − E2 (µV)")
    axs[1].plot(t, filtradas[canal][tramo], c=GRIS_CLARO, lw=0.7, label="Antes")
    axs[1].plot(t, prep.intermedios["sin_ocular"][canal][tramo], c=AZUL, lw=0.6, label="Después")
    axs[1].set(ylabel=f"{corto(canal)} (µV)", xlabel="Tiempo dentro de la época (s)", xlim=(t[0], t[-1]))
    axs[1].legend(loc="upper right", ncols=2)
    fig.tight_layout()
    mostrar(fig)


def seccion_cardiaco(fuente: str, prep, canal: str, epoca: int) -> None:
    st.caption(
        "En cada latido se resta una plantilla igual al promedio de los 30 latidos anteriores. Abajo, el promedio "
        "de toda la noche alrededor de cada onda R. El EEG se cancela y queda el artefacto. El control con instantes "
        "al azar muestra cuánto EEG queda sin cancelarse."
    )
    fs = prep.registro.fs
    tramo = tramo_epoca(fs, epoca)
    t = np.arange(tramo.stop - tramo.start) / fs
    ecg = prep.registro.senales[ECG][tramo]
    latidos = prep.latidos[(prep.latidos >= tramo.start) & (prep.latidos < tramo.stop)] - tramo.start
    antes = prep.intermedios["sin_ocular"][canal][tramo]
    despues = prep.intermedios["sin_cardiaco"][canal][tramo]

    fig, axs = plt.subplots(3, 1, figsize=(12, 5.2), sharex=True)
    axs[0].plot(t, ecg, c=MAGENTA, lw=0.6)
    axs[0].plot(latidos / fs, ecg[latidos], "o", c=GRIS, ms=3)
    axs[0].set_ylabel("ECG (µV)")
    axs[1].plot(t, antes - despues, c=NARANJA, lw=0.7)
    axs[1].set_ylabel("Artefacto\nrestado (µV)")
    axs[2].plot(t, antes, c=GRIS_CLARO, lw=0.7, label="Antes")
    axs[2].plot(t, despues, c=AZUL, lw=0.6, label="Después")
    axs[2].set(ylabel=f"{corto(canal)} (µV)", xlabel="Tiempo dentro de la época (s)", xlim=(t[0], t[-1]))
    axs[2].legend(loc="upper right", ncols=2)
    fig.tight_layout()
    mostrar(fig)

    promedio = promedio_cardiaco(fuente, canal)
    izquierda, derecha = st.columns([2, 1])
    with izquierda:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(promedio["t"], promedio["antes"], c=NARANJA, label="Antes")
        ax.plot(promedio["t"], promedio["despues"], c=AZUL, label="Después")
        ax.plot(promedio["t"], promedio["azar"], c=GRIS, ls="--", lw=0.8, label="Instantes al azar")
        ax.set(xlabel="Tiempo desde la onda R (s)", ylabel=f"Promedio de {corto(canal)} (µV)")
        ax.set_title(f"Promedio de la noche alrededor de {len(prep.latidos)} latidos", loc="left")
        ax.legend()
        fig.tight_layout()
        mostrar(fig)
    with derecha:
        pico = {k: np.ptp(promedio[k]) for k in ("antes", "despues", "azar")}
        st.metric("Pico a pico antes", f"{coma(pico['antes'])} µV")
        st.metric("Pico a pico después", f"{coma(pico['despues'])} µV",
                  delta=f"{100 * (pico['despues'] / pico['antes'] - 1):.0f} %", delta_color="inverse")
        st.metric("Control al azar", f"{coma(pico['azar'])} µV")


def seccion_segmentacion(fuente: str, prep, epoca: int) -> None:
    st.caption(
        "Un modelo AR de orden 8 se ajusta en los primeros 5 s de la época de C4 y se aplica sin cambios a los "
        "cinco tramos siguientes. Si el error de predicción de algún tramo supera 4 veces al del primero, la señal "
        "cambió dentro de la época y se la marca como no estacionaria."
    )
    fs = prep.registro.fs
    epoca_c4 = segmentar(prep.registro.senales[EEG_C4], fs, prep.registro.n_epocas)[epoca]
    t = np.arange(len(epoca_c4)) / fs
    errores = segmentacion.errores_por_tramo(epoca_c4, fs)

    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.plot(t, epoca_c4, c=AZUL, lw=0.6)
    techo = max(np.abs(epoca_c4).max(), 1e-6) * 1.15
    if errores is not None:
        for k, e in enumerate(errores):
            ax.axvline(5 * k, c=GRIS, lw=0.6, ls=":")
            supera = e > segmentacion.CAMBIO_MAXIMO
            ax.text(5 * k + 2.5, techo, "ajuste" if k == 0 else f"×{coma(e)}", ha="center", va="bottom",
                    color=NARANJA if supera else GRIS, fontweight="bold" if supera else "normal")
    ax.set(ylim=(-techo, techo * 1.35), xlim=(0, t[-1]), ylabel="C4 (µV)", xlabel="Tiempo dentro de la época (s)")
    ax.set_title(f"Época {epoca + 1}, {'marcada como no estacionaria' if prep.no_estacionaria[epoca] else 'estacionaria'}", loc="left")
    fig.tight_layout()
    mostrar(fig)

    estadios = prep.registro.estadios
    marcas = prep.no_estacionaria
    izquierda, derecha = st.columns([3, 1])
    with izquierda:
        fig, axs = plt.subplots(2, 1, figsize=(9, 2.6), sharex=True, height_ratios=[3, 1])
        dibujar_hipnograma(axs[0], estadios, epoca)
        horas = np.flatnonzero(marcas) * DURACION_EPOCA_S / 3600
        axs[1].eventplot(horas, colors=NARANJA, lineoffsets=0.5, linelengths=0.8, linewidths=0.6)
        axs[1].set(yticks=[], ylabel="Marcadas", xlabel="Tiempo (h)")
        axs[1].grid(False)
        fig.tight_layout()
        mostrar(fig)
    with derecha:
        tabla = pd.DataFrame({
            "Épocas": [int((estadios == e).sum()) for e in COLOR_ESTADIO],
            "Marcadas (%)": [100 * marcas[estadios == e].mean() if (estadios == e).any() else np.nan for e in COLOR_ESTADIO],
        }, index=[NOMBRES_ESTADIOS[e] for e in COLOR_ESTADIO])
        st.dataframe(tabla.round(1), width="stretch")


def pestana_preprocesamiento(fuente: str, epoca: int) -> None:
    prep = preprocesado(fuente)
    secciones = {
        "Cadena completa": lambda canal: seccion_cadena(prep, canal, epoca),
        "Filtrado": lambda canal: seccion_filtrado(prep, canal, epoca),
        "Artefacto ocular": lambda canal: seccion_ocular(fuente, prep, canal, epoca),
        "Artefacto cardíaco": lambda canal: seccion_cardiaco(fuente, prep, canal, epoca),
        "Segmentación": lambda canal: seccion_segmentacion(fuente, prep, epoca),
    }
    izquierda, derecha = st.columns([4, 1], vertical_alignment="bottom")
    with izquierda:
        st.session_state.setdefault("seccion", "Cadena completa")
        seccion = st.segmented_control("Etapa", list(secciones), key="seccion") or "Cadena completa"
    with derecha:
        canal = st.selectbox("Canal de EEG", CANALES_EEG, format_func=corto, key="canal_prep",
                             disabled=seccion == "Segmentación")
    secciones[seccion](canal)


def pestana_hrv(fuente: str, epoca: int) -> None:
    prep = preprocesado(fuente)
    datos = analisis_latidos(fuente)
    fs = prep.registro.fs
    latidos, conservados = datos["latidos"], datos["conservados"]
    descartados = latidos[~np.isin(latidos, conservados)]

    columnas = st.columns(4)
    columnas[0].metric("Latidos detectados", f"{len(latidos):,}".replace(",", "."))
    columnas[1].metric("RR mediano", f"{coma(np.median(np.diff(latidos)) / fs, 2)} s")
    columnas[2].metric("Descartados por forma", f"{coma(100 * len(descartados) / max(len(latidos), 1))} %")
    columnas[3].metric("Intervalos válidos", f"{100 * len(datos['rr']) / max(len(datos['rr_todos']), 1):.0f} %")

    st.subheader("Latidos de la época")
    st.caption(
        "Pan-Tompkins filtra el ECG entre 5 y 15 Hz, lo deriva, lo eleva al cuadrado y lo integra en 150 ms. "
        "Los latidos cuya forma se parece poco a la del latido típico de la noche no se usan para la HRV."
    )
    tramo = tramo_epoca(fs, epoca)
    t = np.arange(tramo.stop - tramo.start) / fs
    ecg = prep.registro.senales[ECG][tramo]
    fig, ax = plt.subplots(figsize=(12, 2.4))
    ax.plot(t, ecg, c=MAGENTA, lw=0.6)
    for marcas, color, simbolo, etiqueta in ((conservados, AZUL, "o", "Conservado"), (descartados, NARANJA, "x", "Descartado")):
        en_epoca = marcas[(marcas >= tramo.start) & (marcas < tramo.stop)] - tramo.start
        ax.plot(en_epoca / fs, ecg[en_epoca], simbolo, c=color, ms=5, label=etiqueta)
    ax.set(xlim=(0, t[-1]), xlabel="Tiempo dentro de la época (s)", ylabel="ECG (µV)")
    ax.legend(loc="upper right", ncols=2)
    fig.tight_layout()
    mostrar(fig)

    st.subheader("Descarte por forma")
    st.caption(
        "El latido típico es la mediana, muestra a muestra, de todos los latidos de la noche. Se conservan los "
        f"latidos con correlación de {hrv.UMBRAL_FORMA} o más. La correlación no depende de la amplitud."
    )
    izquierda, derecha = st.columns(2)
    correlacion, ventanas = datos["correlacion"], datos["ventanas"]
    desde, hasta = (round(s * fs) for s in hrv.VENTANA_LATIDO_S)
    t_latido = np.arange(desde, hasta + 1) / fs
    rng = np.random.default_rng(0)
    with izquierda:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        for grupo, color in ((correlacion >= hrv.UMBRAL_FORMA, AZUL), (correlacion < hrv.UMBRAL_FORMA, NARANJA)):
            indices = np.flatnonzero(grupo)
            for i in rng.choice(indices, min(20, len(indices)), replace=False) if len(indices) else []:
                ax.plot(t_latido, ventanas[i], c=color, lw=0.5, alpha=0.35)
        ax.plot(t_latido, datos["tipico"], c="black", lw=1.6, label="Latido típico")
        ax.plot([], [], c=AZUL, label="Conservados")
        ax.plot([], [], c=NARANJA, label="Descartados")
        ax.set(xlabel="Tiempo desde la onda R (s)", ylabel="ECG filtrado (µV)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        mostrar(fig)
    with derecha:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.hist(correlacion, bins=np.linspace(-0.2, 1, 61), color=AZUL)
        ax.axvline(hrv.UMBRAL_FORMA, c=NARANJA, ls="--")
        ax.set(yscale="log", xlabel="Correlación con el latido típico", ylabel="Latidos")
        fig.tight_layout()
        mostrar(fig)

    st.subheader("Intervalos RR y HRV de la noche")
    st.caption(
        "Se descartan los intervalos fuera de 0,3 a 2 s y los que se alejan más de un 20 % de la mediana de sus "
        "vecinos. LF/HF se calcula en una ventana de 5 min centrada en cada época y se expresa como percentil "
        "dentro de la noche, igual que en las reglas."
    )
    marca_h = (epoca + 0.5) * DURACION_EPOCA_S / 3600
    fig, axs = plt.subplots(3, 1, figsize=(12, 5.4), sharex=True, height_ratios=[1.3, 2, 1.5])
    dibujar_hipnograma(axs[0], prep.registro.estadios, epoca)
    axs[1].plot(datos["t_todos"] / 3600, datos["rr_todos"], ".", c=GRIS_CLARO, ms=1.5, label="Descartados")
    axs[1].plot(datos["t_rr"] / 3600, datos["rr"], ".", c=AZUL, ms=1.5, label="Válidos")
    axs[1].set(ylabel="RR (s)", ylim=(0.2, 2.1))
    axs[1].legend(loc="upper right", ncols=2, markerscale=6)
    horas = (np.arange(len(datos["percentiles"])) + 0.5) * DURACION_EPOCA_S / 3600
    axs[2].plot(horas, datos["percentiles"][:, 1], c=AZUL, lw=0.7)
    axs[2].axhline(50, c=GRIS, lw=0.6, ls=":")
    axs[2].set(ylabel="LF/HF\n(percentil)", ylim=(0, 100), xlabel="Tiempo (h)")
    for ax in axs[1:]:
        ax.axvline(marca_h, c="#d62728", lw=1)
    fig.tight_layout()
    mostrar(fig)

    st.subheader("Periodograma de Lomb de la ventana")
    centro = (epoca + 0.5) * DURACION_EPOCA_S
    mitad = ConfigDescriptores().hrv_ventana_s / 2
    en_ventana = (datos["t_rr"] >= centro - mitad) & (datos["t_rr"] < centro + mitad)
    if en_ventana.sum() < hrv.MIN_INTERVALOS:
        st.info("En esta ventana hay muy pocos intervalos válidos. La época toma la mediana de la noche.")
        return
    t_v, rr_v = datos["t_rr"][en_ventana], datos["rr"][en_ventana]
    frecuencias, psd = hrv.periodograma_lomb(t_v, rr_v)
    lf, hf = hrv.potencias_lomb(t_v, rr_v)
    izquierda, derecha = st.columns([2, 1])
    with izquierda:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(frecuencias, psd * 1e6, c=AZUL)
        ax.axvspan(*BANDA_LF, color=NARANJA, alpha=0.12, lw=0, label="LF")
        ax.axvspan(*BANDA_HF, color=AGUA, alpha=0.12, lw=0, label="HF")
        ax.set(xlabel="Frecuencia (Hz)", ylabel="PSD (ms²/Hz)", xlim=(BANDA_LF[0], BANDA_HF[1]))
        ax.set_title(f"{int(en_ventana.sum())} intervalos en los 5 min alrededor de la época", loc="left")
        ax.legend(loc="upper right")
        fig.tight_layout()
        mostrar(fig)
    with derecha:
        st.metric("Potencia LF", f"{lf * 1e6:.0f} ms²")
        st.metric("Potencia HF", f"{hf * 1e6:.0f} ms²")
        st.metric("LF/HF", coma(lf / hf, 2) if hf > 0 else "sin dato")


def pestana_clasificacion(fuente: str, epoca: int, cfg_clasificacion) -> None:
    tabla, sin_suavizar, estimado, regla = clasificacion(fuente, cfg_clasificacion)
    if fuente == FUENTE_SINTETICA:
        st.caption("Sin datos reales se usan descriptores sintéticos, que no son los del registro de las otras pestañas.")
    epoca = min(epoca, len(tabla.estadios) - 1)
    reporte = metricas.reporte(tabla.estadios, estimado)

    columnas = st.columns(3)
    columnas[0].metric("κ de Cohen del registro", coma(reporte["kappa"], 3))
    columnas[1].metric("Técnicos de HMC", "0,74")
    columnas[2].metric("Red de referencia", "0,79")

    series = {"Técnicos": tabla.estadios, "Reglas": sin_suavizar, "Reglas y\nsuavizado": estimado}
    fig, axs = plt.subplots(len(series), 1, figsize=(12, 1.25 * len(series) + 0.5), sharex=True)
    for ax, (titulo, y) in zip(axs, series.items()):
        dibujar_hipnograma(ax, y, epoca)
        ax.set_ylabel(titulo, rotation=0, ha="right", va="center", fontsize=8)
    axs[-1].set_xlabel("Tiempo (h)")
    fig.tight_layout()
    mostrar(fig)

    st.subheader(f"Época {epoca + 1}")
    columnas = st.columns(4)
    columnas[0].metric("Técnicos", nombre_estadio(int(tabla.estadios[epoca])))
    columnas[1].metric("Reglas", nombre_estadio(int(sin_suavizar[epoca])))
    columnas[2].metric("Con suavizado", nombre_estadio(int(estimado[epoca])))
    columnas[3].metric("Regla que decidió", reglas.REGLAS[regla[epoca]])
    percentiles = reglas.descriptores_de_reglas(tabla.matriz, tabla.nombres, cfg_clasificacion)
    st.dataframe(
        pd.DataFrame({"Valor que usan las reglas": {k: round(float(v[epoca]), 2) for k, v in percentiles.items()}}).T,
        width="stretch",
    )

    izquierda, derecha = st.columns(2)
    with izquierda:
        mostrar(graficar_confusion(reporte["confusion"]))
    with derecha:
        st.write("F1 por estadio")
        st.dataframe(pd.DataFrame(reporte["f1"], index=["F1"]).round(3), width="stretch")
        conteo = pd.Series(np.bincount(regla, minlength=len(reglas.REGLAS)), index=reglas.REGLAS)
        st.write("Épocas decididas por cada regla")
        st.bar_chart(conteo, color=AZUL)


def main() -> None:
    st.set_page_config(page_title="Estadios de sueño", layout="wide")
    fuente, registro, epoca, cfg_clasificacion = barra_lateral()
    encabezado(fuente, registro, epoca)
    vistas = {
        "Registro": lambda: pestana_registro(registro, epoca),
        "Preprocesamiento": lambda: pestana_preprocesamiento(fuente, epoca),
        "Latidos y HRV": lambda: pestana_hrv(fuente, epoca),
        "Clasificación": lambda: pestana_clasificacion(fuente, epoca, cfg_clasificacion),
    }
    st.session_state.setdefault("vista", "Registro")
    vista = st.segmented_control("Vista", list(vistas), key="vista", label_visibility="collapsed") or "Registro"
    st.divider()
    vistas[vista]()


if __name__ == "__main__":
    from streamlit import runtime

    if runtime.exists():
        main()
    else:
        # Corrido con `python`, se relanza con `streamlit run` para que abra en el navegador.
        from streamlit.web import cli

        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
        sys.exit(cli.main())
