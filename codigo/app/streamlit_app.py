"""Etapa 11: interfaz gráfica. Correr desde codigo/ con: streamlit run app/streamlit_app.py

Tres pestañas:
- Registro: cada canal, época por época, con el estadio de los técnicos.
- Etapas: un mismo canal en cada etapa del preprocesamiento (Frente A).
- Hipnograma: estimado contra técnicos, métricas y la regla que decidió cada época.

Sin datos descargados funciona con el registro y los descriptores sintéticos.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

# Si el paquete no está instalado con `pip install -e .`, se usa directamente codigo/src.
try:
    import estadios  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from estadios.clasificacion import reglas
from estadios.clasificacion.modelo import ModeloEstadios
from estadios.comun.config import GRUPOS_DESCRIPTORES, Config
from estadios.comun.constantes import (
    DURACION_EPOCA_S,
    N1,
    N2,
    N3,
    N_ESTADIOS,
    NOMBRES_ESTADIOS,
    SIN_ANOTAR,
    R,
    W,
)
from estadios.comun.contratos import Registro, TablaDescriptores
from estadios.comun.sintetico import registro_sintetico, tabla_sintetica
from estadios.datos.carga import cargar_registro, ids_disponibles
from estadios.datos.particion import particion_por_paciente
from estadios.evaluacion import metricas
from estadios.pipeline import obtener_tabla, preprocesar

FUENTE_SINTETICA = "Sintético (sin datos)"
ALTURA = {W: 4, R: 3, N1: 2, N2: 1, N3: 0}  # Orden habitual del hipnograma.
ETIQUETAS_ALTURA = ["N3", "N2", "N1", "R", "W"]


def nombre_estadio(estadio: int) -> str:
    return NOMBRES_ESTADIOS[estadio] if estadio != SIN_ANOTAR else "sin anotar"


# ---------------------------------------------------------------- Datos (en caché)


@st.cache_resource
def cargar(fuente: str) -> Registro:
    if fuente == FUENTE_SINTETICA:
        return registro_sintetico()
    return cargar_registro(fuente)


@st.cache_resource
def preprocesado(fuente: str):
    """RegistroPreprocesado con las etapas intermedias, o el mensaje de lo que falta."""
    try:
        return preprocesar(cargar(fuente), Config().preprocesamiento, guardar_intermedios=True)
    except NotImplementedError as pendiente:
        return f"Pendiente del Frente A: {pendiente}."


@st.cache_resource
def tabla_y_entrenamiento(fuente: str):
    """Descriptores del registro y tablas de entrenamiento, o el mensaje de lo que falta.

    Con la fuente sintética, los descriptores salen de `tabla_sintetica` y no del
    registro que se muestra en las otras pestañas.
    """
    if fuente == FUENTE_SINTETICA:
        return tabla_sintetica(semilla=100), [tabla_sintetica(semilla=s) for s in range(8)]
    cfg = Config()
    try:
        entrenamiento_ids, _ = particion_por_paciente(ids_disponibles())
        entrenamiento = [obtener_tabla(i, cfg) for i in entrenamiento_ids if i != fuente]
        return obtener_tabla(fuente, cfg), entrenamiento
    except NotImplementedError as pendiente:
        return f"Faltan etapas para calcular los descriptores: {pendiente}."


# ---------------------------------------------------------------- Gráficos


def mostrar(fig) -> None:
    """Muestra la figura y la cierra, para no acumular figuras abiertas en memoria."""
    st.pyplot(fig)
    plt.close(fig)


def graficar_epoca(senales: dict[str, np.ndarray], fs: float, epoca: int, titulo: str):
    n = int(DURACION_EPOCA_S * fs)
    tramo = slice(epoca * n, (epoca + 1) * n)
    t = np.arange(n) / fs
    fig, axs = plt.subplots(len(senales), 1, figsize=(11, 1.3 * len(senales)), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, (canal, senal) in zip(axs, senales.items()):
        ax.plot(t, senal[tramo], lw=0.6)
        ax.set_ylabel(canal, rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0].set_title(titulo)
    axs[-1].set_xlabel("Tiempo dentro de la época (s)")
    fig.tight_layout()
    return fig


def graficar_hipnogramas(series: dict[str, np.ndarray], marca: int | None = None):
    fig, axs = plt.subplots(len(series), 1, figsize=(11, 1.6 * len(series)), sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (titulo, y) in zip(axs, series.items()):
        horas = np.arange(len(y)) * DURACION_EPOCA_S / 3600
        altura = np.array([ALTURA.get(int(e), np.nan) for e in y])
        ax.step(horas, altura, where="post", lw=0.8)
        ax.set_ylabel(titulo, fontsize=8)
        if marca is not None:
            ax.axvline(marca * DURACION_EPOCA_S / 3600, c="r", lw=0.8)
    axs[0].set_yticks(range(5), ETIQUETAS_ALTURA)
    axs[-1].set_xlabel("Tiempo (h)")
    fig.tight_layout()
    return fig


def graficar_confusion(matriz: np.ndarray):
    normalizada = matriz / matriz.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(normalizada, cmap="Blues", vmin=0, vmax=1)
    for i in range(N_ESTADIOS):
        for j in range(N_ESTADIOS):
            ax.text(j, i, f"{normalizada[i, j]:.2f}\n({matriz[i, j]})", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(N_ESTADIOS), NOMBRES_ESTADIOS)
    ax.set_yticks(range(N_ESTADIOS), NOMBRES_ESTADIOS)
    ax.set(xlabel="Estimado", ylabel="Técnicos")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- Pestañas


def pestana_registro(registro: Registro) -> None:
    columnas = st.columns(3)
    columnas[0].metric("Épocas", registro.n_epocas)
    columnas[1].metric("Horas", f"{registro.n_epocas * DURACION_EPOCA_S / 3600:.1f}")
    columnas[2].metric("Fs (Hz)", f"{registro.fs:.0f}")

    epoca = st.slider("Época", 1, registro.n_epocas, 1, key="epoca_registro") - 1
    titulo = f"Época {epoca + 1}: {nombre_estadio(int(registro.estadios[epoca]))} según los técnicos"
    mostrar(graficar_epoca(registro.senales, registro.fs, epoca, titulo))


def pestana_etapas(fuente: str, registro: Registro) -> None:
    prep = preprocesado(fuente)
    if isinstance(prep, str):
        st.info(prep)
        return
    canal = st.selectbox("Canal", list(prep.intermedios["crudo"]), key="canal_etapas")
    epoca = st.slider("Época", 1, registro.n_epocas, 1, key="epoca_etapas") - 1
    senales = {etapa: canales[canal] for etapa, canales in prep.intermedios.items()}
    mostrar(graficar_epoca(senales, registro.fs, epoca, f"{canal}, época {epoca + 1}"))


def pestana_hipnograma(fuente: str, cfg_clasificacion) -> None:
    datos = tabla_y_entrenamiento(fuente)
    if isinstance(datos, str):
        st.info(datos)
        return
    tabla, entrenamiento = datos
    if fuente == FUENTE_SINTETICA:
        st.caption("Sin datos reales: se usan descriptores sintéticos, no los del registro de las otras pestañas.")

    modelo = ModeloEstadios(cfg_clasificacion).ajustar(entrenamiento)
    sin_suavizar, regla = modelo.clasificar_epocas(tabla)
    estimado = modelo.suavizar(sin_suavizar)
    reporte = metricas.reporte(tabla.estadios, estimado)

    columnas = st.columns(3)
    columnas[0].metric("κ de Cohen", f"{reporte['kappa']:.3f}")
    columnas[1].metric("Técnicos HMC", "0,74")
    columnas[2].metric("Red de referencia", "0,79")

    epoca = st.slider("Época", 1, len(tabla.estadios), 1, key="epoca_hipnograma") - 1
    mostrar(graficar_hipnogramas(
        {"Técnicos": tabla.estadios, "Reglas": sin_suavizar, "Reglas +\nsuavizado": estimado}, epoca
    ))

    # Explicación de la época elegida: qué regla la decidió y con qué valores.
    st.subheader(f"Época {epoca + 1}")
    columnas = st.columns(4)
    columnas[0].metric("Técnicos", nombre_estadio(int(tabla.estadios[epoca])))
    columnas[1].metric("Reglas", nombre_estadio(int(sin_suavizar[epoca])))
    columnas[2].metric("Con suavizado", nombre_estadio(int(estimado[epoca])))
    columnas[3].metric("Regla que decidió", reglas.REGLAS[regla[epoca]])
    percentiles = reglas.descriptores_de_reglas(tabla.matriz, tabla.nombres, cfg_clasificacion)
    st.dataframe(
        pd.DataFrame({"Percentil en la noche": {k: round(float(v[epoca]), 1) for k, v in percentiles.items()}}).T
    )

    izquierda, derecha = st.columns(2)
    with izquierda:
        mostrar(graficar_confusion(reporte["confusion"]))
    with derecha:
        st.write("F1 por estadio")
        st.dataframe(pd.DataFrame(reporte["f1"], index=["F1"]).round(3))
        conteo = pd.Series(np.bincount(regla, minlength=len(reglas.REGLAS)), index=reglas.REGLAS)
        st.write("Épocas decididas por cada regla")
        st.bar_chart(conteo)


def main() -> None:
    st.set_page_config(page_title="Estadios de sueño", layout="wide")
    st.title("Clasificación automática de estadios de sueño")

    fuente = st.sidebar.selectbox("Registro", [FUENTE_SINTETICA, *ids_disponibles()])
    st.sidebar.header("Clasificador")
    suavizado = st.sidebar.radio("Suavizado", ["reglas", "moda", "ninguno"])
    usar_csp = st.sidebar.checkbox("Desempate N1/REM con CSP", value=True)
    base = Config().clasificacion
    grupos = tuple(g for g in GRUPOS_DESCRIPTORES if usar_csp or g != "csp")
    cfg_clasificacion = replace(base, suavizado=suavizado, grupos=grupos)

    registro = cargar(fuente)
    registro_tab, etapas_tab, hipnograma_tab = st.tabs(["Registro", "Etapas", "Hipnograma"])
    with registro_tab:
        pestana_registro(registro)
    with etapas_tab:
        pestana_etapas(fuente, registro)
    with hipnograma_tab:
        pestana_hipnograma(fuente, cfg_clasificacion)


if __name__ == "__main__":
    main()
