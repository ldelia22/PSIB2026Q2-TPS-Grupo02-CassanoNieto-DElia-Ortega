"""Etapa 11: interfaz gráfica. Correr con: streamlit run app/streamlit_app.py

Por ahora es solo la estructura de pantallas. Cada pestaña llama al pipeline y avisa
qué etapa falta mientras no esté implementada.
"""

from __future__ import annotations

import streamlit as st

from estadios.comun.config import Config
from estadios.comun.constantes import NOMBRES_ESTADIOS
from estadios.comun.contratos import Registro
from estadios.comun.sintetico import registro_sintetico
from estadios.datos.carga import cargar_registro, ids_disponibles
from estadios.pipeline import preprocesar

FUENTE_SINTETICA = "Sintético (sin datos)"


@st.cache_resource
def cargar(fuente: str) -> Registro:
    if fuente == FUENTE_SINTETICA:
        return registro_sintetico()
    return cargar_registro(fuente)


def pestana_registro(registro: Registro) -> None:
    columnas = st.columns(3)
    columnas[0].metric("Épocas", registro.n_epocas)
    columnas[1].metric("Horas", f"{registro.n_epocas * 30 / 3600:.1f}")
    columnas[2].metric("Fs (Hz)", f"{registro.fs:.0f}")
    st.write("Canales:", ", ".join(registro.senales))
    # TODO(Frente C): graficar cada canal con selector de época.


def pestana_etapas(registro: Registro, cfg: Config) -> None:
    try:
        prep = preprocesar(registro, cfg.preprocesamiento, guardar_intermedios=True)
    except NotImplementedError as pendiente:
        st.info(f"Pendiente del Frente A: {pendiente}.")
        return
    st.write("Etapas disponibles:", ", ".join(prep.intermedios))
    # TODO(Frente C): comparar un mismo canal entre etapas.


def pestana_hipnograma(registro: Registro) -> None:
    st.write("Estadios:", ", ".join(NOMBRES_ESTADIOS))
    st.info("Pendiente: cargar el modelo y comparar el hipnograma estimado con el de los técnicos.")
    # TODO(Frente C): hipnograma, matriz de confusión y κ del registro.


def main() -> None:
    st.set_page_config(page_title="Estadios de sueño", layout="wide")
    st.title("Clasificación automática de estadios de sueño")

    fuente = st.sidebar.selectbox("Registro", [FUENTE_SINTETICA, *ids_disponibles()])
    registro = cargar(fuente)
    cfg = Config()

    registro_tab, etapas_tab, hipnograma_tab = st.tabs(["Registro", "Etapas", "Hipnograma"])
    with registro_tab:
        pestana_registro(registro)
    with etapas_tab:
        pestana_etapas(registro, cfg)
    with hipnograma_tab:
        pestana_hipnograma(registro)


if __name__ == "__main__":
    main()
