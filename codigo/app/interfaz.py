"""Interfaz gráfica del TP. Se abre con `python app/interfaz.py` desde codigo/, o con el botón de Run.

1. Elegir un registro de la HMC o el registro sintético.
2. Procesar. Corre el pipeline en segundo plano, etapa por etapa, y muestra el avance.
3. Recorrer las épocas y ver cada etapa del preprocesamiento, los latidos y la HRV, y la
   clasificación comparada con los técnicos.

La interfaz no tiene lógica propia. Solo llama a las funciones de estadios/, así que lo que
se ve es exactamente lo que se evalúa. Los colores siguen el tema del sistema y el botón de
arriba a la derecha cambia de claro a oscuro.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

CODIGO = Path(__file__).resolve().parents[1]
os.chdir(CODIGO)  # Para encontrar data/ igual que los notebooks, sin importar desde dónde se corra.
try:
    import estadios  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(CODIGO / "src"))

from estadios.clasificacion import reglas  # noqa: E402
from estadios.clasificacion.modelo import ModeloEstadios  # noqa: E402
from estadios.comun.config import GRUPOS_DESCRIPTORES, Config, ConfigDescriptores  # noqa: E402
from estadios.comun.constantes import (  # noqa: E402
    BANDA_HF, BANDA_LF, CANALES_EEG, DURACION_EPOCA_S, ECG, EEG_C4, EEG_F4, EMG, EOG_E1, EOG_E2,
    N1, N2, N3, N_ESTADIOS, NOMBRES_ESTADIOS, SIN_ANOTAR, R, W,
)
from estadios.comun.contratos import RegistroPreprocesado  # noqa: E402
from estadios.comun.sintetico import registro_sintetico, tabla_sintetica  # noqa: E402
from estadios.datos.carga import cargar_registro, ids_disponibles  # noqa: E402
from estadios.datos.particion import particion_por_paciente  # noqa: E402
from estadios.descriptores import hrv  # noqa: E402
from estadios.evaluacion import metricas  # noqa: E402
from estadios.pipeline import obtener_tabla  # noqa: E402
from estadios.preprocesamiento import cardiaco, exploratorio, filtrado, ocular, qrs, segmentacion  # noqa: E402
from estadios.preprocesamiento.segmentacion import segmentar  # noqa: E402

CFG = Config()
FUENTE_SINTETICA = "Sintético (sin datos)"
TOTAL_PASOS = 7
EPOCA_H = DURACION_EPOCA_S / 3600
LIMITE_SATURACION_UV = 799.0

COLOR_ESTADIO = {W: "#2a78d6", N1: "#eb6834", N2: "#1baf7a", N3: "#eda100", R: "#e87ba4"}
ALTURA = {W: 4, R: 3, N1: 2, N2: 1, N3: 0}
ETIQUETAS_ALTURA = ["N3", "N2", "N1", "R", "W"]
NOMBRE_ETAPA = {"crudo": "Crudo", "filtrado": "Filtrado", "sin_ocular": "Sin artefacto ocular",
                "sin_cardiaco": "Sin artefacto cardíaco"}
NOMBRE_DESCRIPTOR = {"emg": "EMG", "eog": "Energía del EOG", "alfa": "Alfa en O2", "delta": "Delta en F4",
                    "ondas": "Ondas lentas en F4", "husos": "Husos en C4", "lf_hf": "LF/HF"}
VISTAS_PREP = ("Cadena completa", "Filtrado", "Artefacto ocular", "Artefacto cardíaco", "Segmentación")
VISTAS_HRV = ("Latidos de la época", "Descarte por forma", "Intervalos RR y HRV", "Periodograma de Lomb")
EXPLICACION_PREP = {
    "Cadena completa": "La misma época después de cada etapa. Cada etapa toma la salida de la anterior.",
    "Filtrado": "Pasa-altos de 0,2 Hz para conservar las ondas lentas, pasa-bajos de 45 Hz y notch de 50 Hz para la "
                "línea eléctrica. El IIR se aplica ida y vuelta, así que no desfasa la señal.",
    "Artefacto ocular": "La referencia es E1 − E2. La compuerta vale 1 donde su valor eficaz en 2 s supera el doble de "
                        "su mediana en la noche, y solo ahí se resta el artefacto. Así no se borran las ondas lentas de N3.",
    "Artefacto cardíaco": "En cada latido se resta una plantilla igual al promedio de los 30 latidos anteriores. Abajo, "
                          "el promedio de toda la noche alrededor de cada onda R. El EEG se cancela y queda el artefacto.",
    "Segmentación": "Un modelo AR de orden 8 se ajusta en los primeros 5 s de la época de C4 y se aplica a los cinco "
                    "tramos siguientes. Si el error de algún tramo supera 4 veces al del primero, la época se marca.",
}
EXPLICACION_HRV = {
    "Latidos de la época": "Pan-Tompkins filtra el ECG entre 5 y 15 Hz, lo deriva, lo eleva al cuadrado y lo integra "
                           "en 150 ms. Los latidos que no se parecen al latido típico no se usan para la HRV.",
    "Descarte por forma": "El latido típico es la mediana, muestra a muestra, de todos los latidos de la noche. Se "
                          "conservan los que tienen una correlación de 0,8 o más con él.",
    "Intervalos RR y HRV": "Se descartan los intervalos fuera de 0,3 a 2 s y los que se alejan más de un 20 % de sus "
                           "vecinos. LF/HF se mide en 5 min alrededor de cada época, en veces la mediana de la noche, como en la regla de REM.",
    "Periodograma de Lomb": "Lomb estima el espectro de los intervalos sin interpolarlos, aunque falten latidos. La "
                            "ventana es de 5 min centrada en la época elegida.",
}

TEMA = None  # None sigue al sistema operativo. "claro" u "oscuro" lo fuerzan.
PALETAS = {
    "claro": dict(fondo="#f3f2ee", superficie="#fbfbfa", panel="#e6e5df", texto="#1f1f1e",
                  texto2="#52514e", tenue="#a5a39b", borde="#d0cec6", acento="#2a78d6",
                  acento_activo="#2368bd", seleccion="#cfe0f6", grilla="#e4e3dd",
                  ok="#1a7f4b", aviso="#a86b00", error="#c0392b",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]),
    "oscuro": dict(fondo="#1a1a19", superficie="#232322", panel="#2e2e2c", texto="#f0efec",
                   texto2="#c3c2b7", tenue="#77766f", borde="#41413d", acento="#3987e5",
                   acento_activo="#4a95ec", seleccion="#28456b", grilla="#33332f",
                   ok="#4cc38a", aviso="#e0a33a", error="#ff6b5e",
                   series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d9739a"]),
}
PALETA = PALETAS["claro"]


# ------------------------------------------------------------ Cálculos

_ENTRENAMIENTO: dict[str, object] = {}


def numero(valor, decimales: int = 2) -> str:
    """Número con coma decimal, o una raya si no hay valor."""
    if valor is None or (isinstance(valor, float) and not np.isfinite(valor)):
        return "—"
    return f"{valor:.{decimales}f}".replace(".", ",")


def miles(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def corto(canal: str) -> str:
    partes = canal.split()
    return partes[1].split("-")[0] if len(partes) > 1 else canal


def nombre_estadio(estadio: int) -> str:
    return NOMBRES_ESTADIOS[estadio] if estadio != SIN_ANOTAR else "sin anotar"


def sistema_en_modo_oscuro() -> bool:
    """Averigua si Windows, macOS o GNOME usan el tema oscuro. Si no se puede saber, usa el claro."""
    try:
        if sys.platform == "win32":
            import winreg

            clave = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                   r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
            return winreg.QueryValueEx(clave, "AppsUseLightTheme")[0] == 0
        if sys.platform == "darwin":
            r = subprocess.run(["defaults", "read", "-g", "AppleInterfaceStyle"], capture_output=True, text=True, timeout=2)
            return r.stdout.strip().lower() == "dark"
        r = subprocess.run(["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
                           capture_output=True, text=True, timeout=2)
        return "dark" in r.stdout
    except Exception:
        return False


def promedio_cardiaco(antes: np.ndarray, despues: np.ndarray, latidos: np.ndarray, fs: float) -> dict:
    """Promedio alrededor de los latidos antes y después de la remoción, y en instantes al azar como control."""
    azar = np.sort(np.random.default_rng(0).integers(int(fs), len(antes) - int(fs), len(latidos)))
    t, pa = exploratorio.promedio_alrededor_de_latidos(antes, latidos, fs)
    _, pd_ = exploratorio.promedio_alrededor_de_latidos(despues, latidos, fs)
    _, pz = exploratorio.promedio_alrededor_de_latidos(antes, azar, fs)
    return {"t": t, "antes": pa, "despues": pd_, "azar": pz}


def analizar_hrv(final, latidos: np.ndarray, marcas: np.ndarray) -> dict:
    """Comparación con el latido típico, intervalos RR válidos y LF/HF en veces la mediana de la noche."""
    fs = final.fs
    ecg = final.senales[ECG]
    tipico, ventanas, correlacion, _ = hrv.comparar_con_latido_tipico(ecg, latidos, fs)
    conservados = hrv.descartar_ectopicos(ecg, latidos, fs)
    t_todos, rr_todos = hrv.intervalos_rr(conservados, fs)
    t_rr, rr = hrv.filtrar_intervalos(t_todos, rr_todos)
    matriz, _ = hrv.descriptores_hrv(RegistroPreprocesado(final, latidos, marcas), ConfigDescriptores())
    return {"conservados": conservados, "tipico": tipico, "ventanas": ventanas, "correlacion": correlacion,
            "t_todos": t_todos, "rr_todos": rr_todos, "t_rr": t_rr, "rr": rr,
            "lf_hf": reglas.veces_la_mediana(matriz[:, 1])}


def tablas_de_clasificacion(fuente: str):
    """Descriptores del registro y de los pacientes de entrenamiento, leídos del caché."""
    if fuente == FUENTE_SINTETICA:
        return tabla_sintetica(semilla=100), [tabla_sintetica(semilla=s) for s in range(8)]
    if not _ENTRENAMIENTO:
        for i in particion_por_paciente(ids_disponibles())[0]:
            _ENTRENAMIENTO[i] = obtener_tabla(i, CFG)
    entrenamiento = [t for i, t in _ENTRENAMIENTO.items() if i != fuente]
    tabla = _ENTRENAMIENTO.get(fuente) or obtener_tabla(fuente, CFG)
    return tabla, entrenamiento


def clasificar(tabla, entrenamiento, cfg_clasificacion) -> dict:
    """Reglas, desempate CSP y suavizado, con las métricas contra los técnicos."""
    modelo = ModeloEstadios(cfg_clasificacion).ajustar(entrenamiento)
    sin_suavizar, regla = modelo.clasificar_epocas(tabla)
    estimado = modelo.suavizar(sin_suavizar)
    return {"sin_suavizar": sin_suavizar, "estimado": estimado, "regla": regla,
            "reporte": metricas.reporte(tabla.estadios, estimado),
            "valores": reglas.descriptores_de_reglas(tabla.matriz, tabla.nombres, cfg_clasificacion)}


# ------------------------------------------------------------ Gráficos

def estilar(ax, grilla: bool = True) -> None:
    p = PALETA
    ax.set_facecolor(p["superficie"])
    ax.tick_params(colors=p["texto2"], labelsize=8)
    ax.xaxis.label.set_color(p["texto2"])
    ax.yaxis.label.set_color(p["texto2"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(p["borde"])
    if grilla:
        ax.grid(True, color=p["grilla"], linewidth=0.6)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)


def leyenda(ax, **opciones) -> None:
    ax.legend(frameon=False, labelcolor=PALETA["texto2"], fontsize=8, **opciones)


def titulo(ax, texto: str) -> None:
    ax.set_title(texto, color=PALETA["texto"], fontsize=10, loc="left")


def dibujar_hipnograma(ax, y: np.ndarray, marca: int | None = None) -> None:
    horas = np.arange(len(y)) * EPOCA_H
    altura = np.array([ALTURA.get(int(e), np.nan) for e in y])
    ax.step(horas, altura, where="post", c=PALETA["tenue"], lw=0.6)
    for estadio, color in COLOR_ESTADIO.items():
        x = horas[y == estadio]
        ax.hlines(np.full(len(x), ALTURA[estadio]), x, x + EPOCA_H, colors=color, lw=4)
    if marca is not None:
        ax.axvline((marca + 0.5) * EPOCA_H, c=PALETA["error"], lw=1.2)
    ax.set_yticks(range(5), ETIQUETAS_ALTURA)
    ax.set_ylim(-0.6, 4.6)
    ax.set_xlim(0, len(y) * EPOCA_H)
    estilar(ax)
    ax.grid(axis="y", visible=False)


class Grafico(ttk.Frame):
    """Una figura de matplotlib dentro de la ventana, con su barra de herramientas para zoom y guardar."""

    def __init__(self, padre, barra: bool = True, alto: int | None = None):
        super().__init__(padre)
        self.figura = Figure(layout="constrained")
        self.lienzo = FigureCanvasTkAgg(self.figura, master=self)
        self.con_barra = barra
        self.barra = None
        if barra:
            self.rehacer_barra()
        widget = self.lienzo.get_tk_widget()
        if alto:
            widget.configure(height=alto)
        widget.pack(fill="both", expand=True)
        self.limpiar()

    def rehacer_barra(self):
        """Matplotlib elige los íconos según el fondo, así que al cambiar de tema la barra se crea de nuevo."""
        if not self.con_barra:
            return
        vieja = self.barra
        if vieja is not None:
            for conexion in (vieja._id_press, vieja._id_release, vieja._id_drag):
                self.lienzo.mpl_disconnect(conexion)
            if self.lienzo.widgetlock.isowner(vieja):
                self.lienzo.widgetlock.release(vieja)
            vieja.destroy()
        self.barra = NavigationToolbar2Tk(self.lienzo, self, pack_toolbar=False)
        self.barra.update()
        widget = self.lienzo.get_tk_widget()
        self.barra.pack(side="bottom", fill="x", **({"before": widget} if widget.winfo_manager() else {}))

    def limpiar(self):
        self.figura.clear()
        self.figura.set_facecolor(PALETA["superficie"])
        self.lienzo.get_tk_widget().configure(background=PALETA["superficie"])

    def mensaje(self, texto: str):
        self.limpiar()
        self.figura.text(0.5, 0.5, texto, ha="center", va="center", color=PALETA["texto2"], fontsize=11)
        self.lienzo.draw_idle()


# ------------------------------------------------------------ Ventana principal

class Aplicacion(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Estadios de sueño · Clasificación automática")
        self.geometry(f"{min(1440, self.winfo_screenwidth() - 40)}x{min(900, self.winfo_screenheight() - 90)}")
        self.minsize(1100, 720)
        self._estilos()

        self.fuente = None
        self.registro = None
        self.res = None
        self.epoca = 0
        self.cola = queue.Queue()
        self.version = 0
        self.procesando = False
        self.sucias = set()
        self._pendiente = None

        self._armar_ventana()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.after(50, self._revisar_cola)
        self.bind_all("<Left>", lambda e: self._tecla(e, -1))
        self.bind_all("<Right>", lambda e: self._tecla(e, 1))
        self.elegir_registro()

    # ---------------- Aspecto

    def _estilos(self):
        base = tkfont.nametofont("TkDefaultFont")
        base.configure(size=12 if sys.platform == "darwin" else 10)
        self.familia = base.actual("family")
        self.estilo = ttk.Style(self)
        self.estilo.theme_use("clam")
        self._aplicar_tema(TEMA or ("oscuro" if sistema_en_modo_oscuro() else "claro"))

    def _aplicar_tema(self, nombre: str):
        global PALETA
        self.tema = nombre
        PALETA = p = PALETAS[nombre]
        for opcion, color in {"*Background": p["fondo"], "*Foreground": p["texto"],
                              "*activeBackground": p["panel"], "*activeForeground": p["texto"],
                              "*selectBackground": p["seleccion"], "*selectForeground": p["texto"],
                              "*selectColor": p["panel"], "*highlightBackground": p["fondo"],
                              "*highlightColor": p["acento"], "*insertBackground": p["texto"],
                              "*troughColor": p["grilla"], "*disabledForeground": p["tenue"],
                              "*TCombobox*Listbox.background": p["superficie"],
                              "*TCombobox*Listbox.foreground": p["texto"],
                              "*TCombobox*Listbox.selectBackground": p["seleccion"],
                              "*TCombobox*Listbox.selectForeground": p["texto"]}.items():
            self.option_add(opcion, color)
        self.configure(background=p["fondo"])

        estilo, familia = self.estilo, self.familia
        plano = lambda color: dict(lightcolor=color, darkcolor=color)  # noqa: E731
        activo = lambda normal, activo_: [("disabled", normal), ("pressed", activo_), ("active", activo_)]  # noqa: E731

        estilo.configure(".", background=p["fondo"], foreground=p["texto"], fieldbackground=p["superficie"],
                         bordercolor=p["borde"], troughcolor=p["grilla"], selectbackground=p["seleccion"],
                         selectforeground=p["texto"], insertcolor=p["texto"], arrowcolor=p["texto2"],
                         focuscolor=p["acento"], **plano(p["fondo"]))
        estilo.map(".", foreground=[("disabled", p["tenue"])])

        estilo.configure("Titulo.TLabel", font=(familia, 20, "bold"))
        estilo.configure("Sub.TLabel", foreground=p["texto2"])
        estilo.configure("Dato.TLabel", font=(familia, 12, "bold"))
        estilo.configure("Grande.TLabel", font=(familia, 18, "bold"))
        for clave in ("ok", "aviso", "error"):
            estilo.configure(f"Clase{clave}.TLabel", font=(familia, 15, "bold"), foreground=p[clave])
        estilo.configure("ClaseSin.TLabel", font=(familia, 12), foreground=p["texto2"])
        estilo.configure("TLabelframe", bordercolor=p["borde"], **plano(p["fondo"]))
        estilo.configure("TLabelframe.Label", foreground=p["texto"], font=(familia, 12, "bold"))

        estilo.configure("TButton", background=p["panel"], bordercolor=p["borde"], padding=(10, 5), **plano(p["panel"]))
        estilo.map("TButton", background=activo(p["fondo"], p["borde"]), lightcolor=activo(p["fondo"], p["borde"]),
                   darkcolor=activo(p["fondo"], p["borde"]), foreground=[("disabled", p["tenue"])])
        estilo.configure("Accion.TButton", font=(familia, 13, "bold"), padding=(10, 8), background=p["acento"],
                         foreground="#ffffff", bordercolor=p["acento"], **plano(p["acento"]))
        estilo.map("Accion.TButton", background=activo(p["panel"], p["acento_activo"]),
                   lightcolor=activo(p["panel"], p["acento_activo"]), darkcolor=activo(p["panel"], p["acento_activo"]),
                   bordercolor=[("disabled", p["borde"])], foreground=[("disabled", p["tenue"])])

        estilo.configure("TCombobox", background=p["panel"], bordercolor=p["borde"], **plano(p["superficie"]))
        estilo.map("TCombobox", fieldbackground=[("disabled", p["fondo"]), ("readonly", p["superficie"])],
                   foreground=[("disabled", p["tenue"]), ("readonly", p["texto"])],
                   selectbackground=[("readonly", p["superficie"])], selectforeground=[("readonly", p["texto"])])
        estilo.configure("TSpinbox", background=p["panel"], bordercolor=p["borde"], arrowcolor=p["texto2"],
                         **plano(p["superficie"]))
        estilo.configure("Horizontal.TScale", background=p["acento"], troughcolor=p["grilla"], bordercolor=p["borde"],
                         **plano(p["acento"]))
        for widget in ("TRadiobutton", "TCheckbutton"):
            estilo.configure(widget, background=p["fondo"], foreground=p["texto"], indicatorbackground=p["superficie"],
                             indicatorforeground=p["acento"], upperbordercolor=p["borde"], lowerbordercolor=p["borde"])
            estilo.map(widget, background=[("active", p["fondo"])],
                       indicatorbackground=[("selected", p["acento"]), ("active", p["panel"])],
                       indicatorforeground=[("selected", "#ffffff")])

        estilo.configure("TNotebook", bordercolor=p["borde"], tabmargins=(0, 0, 0, 0), **plano(p["fondo"]))
        estilo.configure("TNotebook.Tab", background=p["panel"], foreground=p["texto2"], bordercolor=p["borde"],
                         padding=(12, 5), **plano(p["panel"]))
        estilo.map("TNotebook.Tab", background=[("selected", p["fondo"]), ("active", p["borde"])],
                   lightcolor=[("selected", p["fondo"])], darkcolor=[("selected", p["fondo"])],
                   foreground=[("selected", p["texto"]), ("disabled", p["tenue"])])

        estilo.configure("Treeview", background=p["superficie"], fieldbackground=p["superficie"],
                         foreground=p["texto"], bordercolor=p["borde"], rowheight=26, **plano(p["superficie"]))
        estilo.map("Treeview", background=[("selected", p["seleccion"])], foreground=[("selected", p["texto"])])
        estilo.configure("Treeview.Heading", background=p["panel"], foreground=p["texto"], bordercolor=p["borde"],
                         font=(familia, 11, "bold"), **plano(p["panel"]))
        estilo.map("Treeview.Heading", background=[("active", p["borde"])])

        estilo.configure("Horizontal.TProgressbar", background=p["acento"], troughcolor=p["grilla"],
                         bordercolor=p["grilla"], **plano(p["acento"]))
        estilo.configure("TScrollbar", background=p["panel"], troughcolor=p["superficie"], bordercolor=p["borde"],
                         arrowcolor=p["texto2"], **plano(p["panel"]))
        estilo.map("TScrollbar", background=[("active", p["borde"])])

    def cambiar_tema(self):
        self._aplicar_tema("claro" if self.tema == "oscuro" else "oscuro")
        p = PALETA
        self.texto.configure(background=p["superficie"], foreground=p["texto"], insertbackground=p["texto"],
                             selectbackground=p["seleccion"], selectforeground=p["texto"])
        for etiqueta in ("ok", "aviso", "error"):
            self.texto.tag_configure(etiqueta, foreground=p[etiqueta])
        for grafico in self.graficos:
            grafico.rehacer_barra()
        self._texto_boton_tema()
        self._redibujar_todo()

    def _texto_boton_tema(self):
        self.btn_tema.configure(text="Cambiar a modo claro" if self.tema == "oscuro" else "Cambiar a modo oscuro")

    # ---------------- Armado de la ventana

    def _armar_ventana(self):
        cabecera = ttk.Frame(self, padding=(18, 14, 18, 6))
        cabecera.pack(fill="x")
        cabecera.columnconfigure(0, weight=1)
        titulos = ttk.Frame(cabecera)
        titulos.grid(row=0, column=0, sticky="w")
        ttk.Label(titulos, text="Estadios de sueño · Clasificación automática", style="Titulo.TLabel").pack(anchor="w")
        ttk.Label(titulos, text="Procesamiento de Señales e Imágenes Biomédicas · ITBA · "
                                "Herramienta académica, no es un dispositivo de diagnóstico",
                  style="Sub.TLabel").pack(anchor="w")
        self.btn_tema = ttk.Button(cabecera, command=self.cambiar_tema)
        self.btn_tema.grid(row=0, column=1, sticky="ne")
        self._texto_boton_tema()

        cuerpo = ttk.Frame(self, padding=(18, 8, 18, 18))
        cuerpo.pack(fill="both", expand=True)
        cuerpo.columnconfigure(1, weight=1)
        cuerpo.rowconfigure(0, weight=1)

        izquierda = ttk.Frame(cuerpo, width=360)
        izquierda.grid(row=0, column=0, sticky="ns", padx=(0, 14))
        derecha = ttk.Frame(cuerpo)
        derecha.grid(row=0, column=1, sticky="nsew")

        self._armar_paso_registro(izquierda)
        self._armar_paso_epoca(izquierda)
        self._armar_paso_clasificador(izquierda)
        self._armar_paso_procesamiento(izquierda)

        self.g_franja = Grafico(derecha, barra=False, alto=120)
        self.g_franja.pack(fill="x", pady=(0, 8))
        self.g_franja.lienzo.mpl_connect("button_press_event", self._click_hipnograma)
        self.pestanas = ttk.Notebook(derecha)
        self.pestanas.pack(fill="both", expand=True)
        self.pestanas.bind("<<NotebookTabChanged>>", lambda _: self._dibujar_visible())
        self._armar_avance()
        self._armar_senales()
        self._armar_preprocesamiento()
        self._armar_hrv()
        self._armar_clasificacion()
        self.graficos = (self.g_franja, self.g_senales, self.g_prep, self.g_hrv, self.g_clasif)

    def _armar_paso_registro(self, padre):
        marco = ttk.LabelFrame(padre, text=" 1 · Registro ", padding=12)
        marco.pack(fill="x")
        entrenamiento, prueba = particion_por_paciente(ids_disponibles()) if ids_disponibles() else ([], [])
        self.conjunto = {**{i: "entrenamiento" for i in entrenamiento}, **{i: "prueba" for i in prueba}}
        opciones = [*ids_disponibles(), FUENTE_SINTETICA]
        ttk.Label(marco, text="Polisomnografía de la HMC", style="Sub.TLabel").pack(anchor="w")
        self.var_fuente = tk.StringVar(value=opciones[0])
        cmb = ttk.Combobox(marco, textvariable=self.var_fuente, values=opciones, state="readonly")
        cmb.pack(fill="x", pady=(2, 8))
        cmb.bind("<<ComboboxSelected>>", lambda _: self.elegir_registro())
        self.lbl_registro = ttk.Label(marco, style="Dato.TLabel", wraplength=330)
        self.lbl_registro.pack(anchor="w")
        self.lbl_detalle = ttk.Label(marco, style="Sub.TLabel", wraplength=330, justify="left")
        self.lbl_detalle.pack(anchor="w")

    def _armar_paso_epoca(self, padre):
        marco = ttk.LabelFrame(padre, text=" 2 · Época ", padding=12)
        marco.pack(fill="x", pady=(10, 0))
        fila = ttk.Frame(marco)
        fila.pack(fill="x")
        ttk.Button(fila, text="◀", width=3, command=lambda: self.mover_epoca(-1)).pack(side="left")
        self.var_epoca = tk.StringVar(value="1")
        self.spin = ttk.Spinbox(fila, from_=1, to=1, textvariable=self.var_epoca, width=7, justify="center",
                                command=self._epoca_desde_spin)
        self.spin.pack(side="left", padx=6)
        self.spin.bind("<Return>", lambda _: self._epoca_desde_spin())
        ttk.Button(fila, text="▶", width=3, command=lambda: self.mover_epoca(1)).pack(side="left")
        self.lbl_de = ttk.Label(fila, style="Sub.TLabel")
        self.lbl_de.pack(side="left", padx=(8, 0))
        self.var_escala = tk.DoubleVar(value=1)
        self.escala = ttk.Scale(marco, from_=1, to=1, variable=self.var_escala, command=self._epoca_desde_escala)
        self.escala.pack(fill="x", pady=(8, 6))

        fila = ttk.Frame(marco)
        fila.pack(fill="x")
        ttk.Label(fila, text="Próxima época de", style="Sub.TLabel").pack(side="left")
        self.var_destino = tk.StringVar(value="N3")
        ttk.Combobox(fila, textvariable=self.var_destino, values=NOMBRES_ESTADIOS, state="readonly",
                     width=5).pack(side="left", padx=6)
        ttk.Button(fila, text="Ir", command=self.ir_a_estadio).pack(side="left")

        self.lbl_estadio = ttk.Label(marco, style="Grande.TLabel")
        self.lbl_estadio.pack(anchor="w", pady=(10, 0))
        self.lbl_info_epoca = ttk.Label(marco, style="Sub.TLabel", wraplength=330, justify="left")
        self.lbl_info_epoca.pack(anchor="w")

    def _armar_paso_clasificador(self, padre):
        marco = ttk.LabelFrame(padre, text=" 3 · Clasificador ", padding=12)
        marco.pack(fill="x", pady=(10, 0))
        ttk.Label(marco, text="Reglas AASM sobre los descriptores en sus unidades. Suavizado temporal", style="Sub.TLabel",
                  wraplength=330).pack(anchor="w")
        fila = ttk.Frame(marco)
        fila.pack(fill="x", pady=(4, 4))
        self.var_suavizado = tk.StringVar(value=CFG.clasificacion.suavizado)
        for valor, texto in (("reglas", "Reglas"), ("moda", "Moda"), ("ninguno", "Ninguno")):
            ttk.Radiobutton(fila, text=texto, value=valor, variable=self.var_suavizado,
                            command=self.reclasificar).pack(side="left", padx=(0, 10))
        self.var_csp = tk.BooleanVar(value="csp" in CFG.clasificacion.grupos)
        ttk.Checkbutton(marco, text="Desempate N1 y REM con CSP", variable=self.var_csp,
                        command=self.reclasificar).pack(anchor="w")

    def _armar_paso_procesamiento(self, padre):
        marco = ttk.LabelFrame(padre, text=" 4 · Procesamiento ", padding=12)
        marco.pack(fill="x", pady=(10, 0))
        self.btn_procesar = ttk.Button(marco, text="▶  Procesar registro", style="Accion.TButton", command=self.procesar)
        self.btn_procesar.pack(fill="x")
        self.barra = ttk.Progressbar(marco, maximum=TOTAL_PASOS)
        self.barra.pack(fill="x", pady=(8, 4))
        self.lbl_estado = ttk.Label(marco, style="Sub.TLabel", wraplength=330, justify="left")
        self.lbl_estado.pack(anchor="w")

    def _armar_avance(self):
        marco = ttk.Frame(self.pestanas, padding=10)
        self.pestanas.add(marco, text="Avance")
        p = PALETA
        self.texto = tk.Text(marco, wrap="word", state="disabled", relief="flat", padx=10, pady=8, highlightthickness=1,
                             font=("Menlo" if sys.platform == "darwin" else "Consolas", 11), background=p["superficie"],
                             foreground=p["texto"], selectbackground=p["seleccion"], selectforeground=p["texto"])
        barra = ttk.Scrollbar(marco, command=self.texto.yview)
        self.texto.configure(yscrollcommand=barra.set)
        barra.pack(side="right", fill="y")
        self.texto.pack(side="left", fill="both", expand=True)
        for etiqueta in ("ok", "aviso", "error"):
            self.texto.tag_configure(etiqueta, foreground=p[etiqueta])

    def _armar_senales(self):
        marco = ttk.Frame(self.pestanas, padding=10)
        self.pestanas.add(marco, text="Señales")
        ttk.Label(marco, text="Los ocho canales tal como vienen en el archivo, antes de cualquier procesamiento.",
                  style="Sub.TLabel").pack(anchor="w", pady=(0, 6))
        self.g_senales = Grafico(marco)
        self.g_senales.pack(fill="both", expand=True)

    def _armar_selector(self, marco, texto, valores, variable):
        ttk.Label(marco, text=texto, style="Sub.TLabel").pack(side="left")
        cmb = ttk.Combobox(marco, textvariable=variable, values=valores, state="readonly", width=20)
        cmb.pack(side="left", padx=(6, 16))
        cmb.bind("<<ComboboxSelected>>", lambda _: self._redibujar_visible())
        return cmb

    def _armar_preprocesamiento(self):
        marco = ttk.Frame(self.pestanas, padding=10)
        self.pestanas.add(marco, text="Preprocesamiento")
        fila = ttk.Frame(marco)
        fila.pack(fill="x")
        self.var_vista_prep = tk.StringVar(value=VISTAS_PREP[0])
        self._armar_selector(fila, "Etapa", VISTAS_PREP, self.var_vista_prep)
        self.var_canal = tk.StringVar(value=corto(EEG_F4))
        self.cmb_canal = self._armar_selector(fila, "Canal de EEG", [corto(c) for c in CANALES_EEG], self.var_canal)
        self.cmb_canal.configure(width=6)
        self.lbl_prep = ttk.Label(marco, style="Sub.TLabel", wraplength=980, justify="left")
        self.lbl_prep.pack(anchor="w", pady=(8, 0))
        self.lbl_prep_dato = ttk.Label(marco, style="Dato.TLabel")
        self.lbl_prep_dato.pack(anchor="w", pady=(4, 6))
        self.g_prep = Grafico(marco)
        self.g_prep.pack(fill="both", expand=True)

    def _armar_hrv(self):
        marco = ttk.Frame(self.pestanas, padding=10)
        self.pestanas.add(marco, text="Latidos y HRV")
        fila = ttk.Frame(marco)
        fila.pack(fill="x")
        self.var_vista_hrv = tk.StringVar(value=VISTAS_HRV[0])
        self._armar_selector(fila, "Vista", VISTAS_HRV, self.var_vista_hrv)
        self.lbl_hrv = ttk.Label(marco, style="Sub.TLabel", wraplength=980, justify="left")
        self.lbl_hrv.pack(anchor="w", pady=(8, 0))
        self.lbl_hrv_dato = ttk.Label(marco, style="Dato.TLabel")
        self.lbl_hrv_dato.pack(anchor="w", pady=(4, 6))
        self.g_hrv = Grafico(marco)
        self.g_hrv.pack(fill="both", expand=True)

    def _armar_clasificacion(self):
        marco = ttk.Frame(self.pestanas, padding=10)
        self.pestanas.add(marco, text="Clasificación")
        marco.columnconfigure(0, weight=1)
        marco.columnconfigure(1, weight=0, minsize=360)
        marco.rowconfigure(0, weight=1)
        self.g_clasif = Grafico(marco)
        self.g_clasif.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        derecha = ttk.Frame(marco)
        derecha.grid(row=0, column=1, sticky="nsew")
        self.lbl_clase = ttk.Label(derecha, style="ClaseSin.TLabel", wraplength=350, justify="left")
        self.lbl_clase.pack(anchor="w", pady=(0, 10))
        self.t_clasif = ttk.Treeview(derecha, columns=("Dato", "Valor"), show="headings", height=20, selectmode="none")
        for columna, ancho in (("Dato", 220), ("Valor", 120)):
            self.t_clasif.heading(columna, text=columna, anchor="w")
            self.t_clasif.column(columna, width=ancho, anchor="w", stretch=columna == "Dato")
        self.t_clasif.pack(fill="both", expand=True)

    # ---------------- Registro y época

    def elegir_registro(self):
        fuente = self.var_fuente.get()
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            registro = registro_sintetico() if fuente == FUENTE_SINTETICA else cargar_registro(fuente)
        finally:
            self.config(cursor="")
        self.fuente, self.registro = fuente, registro
        n = registro.n_epocas
        self.lbl_registro.configure(text=fuente)
        conjunto = self.conjunto.get(fuente)
        self.lbl_detalle.configure(text=f"{n} épocas de 30 s, {numero(n * EPOCA_H, 1)} h, {registro.fs:.0f} Hz"
                                        + (f"  ·  paciente de {conjunto}" if conjunto else ""))
        self.spin.configure(to=n)
        self.escala.configure(to=n)
        self.lbl_de.configure(text=f"de {n}")
        self.version += 1
        self.procesando = False
        self.res = None
        self.barra.configure(value=0)
        self.btn_procesar.state(["!disabled"])
        self.lbl_estado.configure(text="Listo para procesar.")
        self.epoca = 0
        self._actualizar_epoca()

    def mover_epoca(self, paso: int):
        self.ir_a_epoca(self.epoca + paso)

    def ir_a_epoca(self, epoca: int):
        if self.registro is None:
            return
        epoca = int(np.clip(epoca, 0, self.registro.n_epocas - 1))
        if epoca != self.epoca:
            self.epoca = epoca
            self._actualizar_epoca()

    def ir_a_estadio(self):
        objetivo = NOMBRES_ESTADIOS.index(self.var_destino.get())
        candidatas = np.flatnonzero(self.registro.estadios == objetivo)
        if len(candidatas):
            siguientes = candidatas[candidatas > self.epoca]
            self.ir_a_epoca(int(siguientes[0] if len(siguientes) else candidatas[0]))

    def _epoca_desde_spin(self):
        try:
            self.ir_a_epoca(int(self.var_epoca.get()) - 1)
        except ValueError:
            self.var_epoca.set(str(self.epoca + 1))

    def _epoca_desde_escala(self, valor):
        self.ir_a_epoca(round(float(valor)) - 1)

    def _tecla(self, evento, paso: int):
        if not isinstance(evento.widget, (ttk.Combobox, ttk.Spinbox, tk.Text, ttk.Entry)):
            self.mover_epoca(paso)

    def _click_hipnograma(self, evento):
        if evento.inaxes is not None and evento.xdata is not None:
            self.ir_a_epoca(int(evento.xdata / EPOCA_H))

    def _actualizar_epoca(self):
        """Actualiza los datos de la época y vuelve a dibujar, esperando un momento si la escala se sigue moviendo."""
        reg, e = self.registro, self.epoca
        self.var_epoca.set(str(e + 1))
        self.var_escala.set(e + 1)
        estadio = int(reg.estadios[e])
        self.lbl_estadio.configure(text=f"{nombre_estadio(estadio)} según los técnicos",
                                   foreground=COLOR_ESTADIO.get(estadio, PALETA["texto2"]))
        tramo = self._tramo(e)
        saturadas = sum(int((np.abs(reg.senales[c][tramo]) >= LIMITE_SATURACION_UV).sum()) for c in CANALES_EEG)
        partes = [f"Hora {numero(e * EPOCA_H, 2)}", f"{saturadas} muestras saturadas en el EEG"]
        if self.res is not None and self.res.get("marcas") is not None:
            partes.append("no estacionaria" if self.res["marcas"][e] else "estacionaria")
        self.lbl_info_epoca.configure(text="  ·  ".join(partes))
        self.sucias = {"franja", "senales", "prep", "hrv", "clasif"}
        if self._pendiente is not None:
            self.after_cancel(self._pendiente)
        self._pendiente = self.after(80, self._dibujar_visible)

    def _tramo(self, epoca: int) -> slice:
        n = int(DURACION_EPOCA_S * self.registro.fs)
        return slice(epoca * n, (epoca + 1) * n)

    # ---------------- Procesamiento en segundo plano

    def _cfg_clasificacion(self):
        grupos = tuple(g for g in GRUPOS_DESCRIPTORES if self.var_csp.get() or g != "csp")
        return replace(CFG.clasificacion, suavizado=self.var_suavizado.get(), grupos=grupos)

    def procesar(self):
        if self.procesando or self.registro is None:
            return
        self.version += 1
        self.procesando = True
        self.res = None
        self.btn_procesar.state(["disabled"])
        self.barra.configure(value=0)
        self.texto.configure(state="normal")
        self.texto.delete("1.0", "end")
        self.texto.configure(state="disabled")
        self.pestanas.select(0)
        self._escribir(f"{self.fuente}\n\n")
        datos = (self.version, self.fuente, self.registro, self._cfg_clasificacion())
        threading.Thread(target=self._trabajo, args=datos, daemon=True).start()

    def _trabajo(self, version, fuente, registro, cfg_clasificacion):
        """Corre cada etapa del pipeline en el mismo orden que pipeline.preprocesar. Solo habla con la ventana por la cola."""

        def avisar(*mensaje):
            self.cola.put((mensaje[0], version, *mensaje[1:]))

        def etapa(nombre, funcion, *args):
            try:
                return funcion(*args)
            except Exception as e:  # Una etapa rota no frena a las demás.
                avisar("texto", f"  ✖ {nombre}. {type(e).__name__}, {e}\n", "error")
            return None

        def paso(n, texto):
            avisar("paso", n, texto)
            avisar("texto", f"[{n}/{TOTAL_PASOS}] {texto}\n")

        cfg = CFG.preprocesamiento
        fs = registro.fs
        res = dict(intermedios={"crudo": dict(registro.senales)}, latidos=None, ganancia=None, marcas=None,
                   cambio=None, final=None, hrv=None, tabla=None, entrenamiento=None, clasif=None, promedios={})
        inicio = time.perf_counter()

        paso(1, "Detección de latidos con Pan-Tompkins")
        latidos = res["latidos"] = etapa("Latidos", qrs.detectar_latidos, registro.senales[ECG], fs)
        if latidos is None:
            avisar("fallo", "No se pudieron detectar los latidos.")
            return
        avisar("texto", f"  ✔ {miles(len(latidos))} latidos, RR mediano de {numero(np.median(np.diff(latidos)) / fs)} s\n", "ok")

        paso(2, "Filtrado")
        actual = etapa("Filtrado", filtrado.filtrar_registro, registro, cfg)
        if actual is None:
            avisar("fallo", "No se pudo filtrar el registro.")
            return
        res["intermedios"]["filtrado"] = dict(actual.senales)
        avisar("texto", f"  ✔ Pasa-altos de {numero(cfg.pasa_altos_hz, 1)} Hz, pasa-bajos de {cfg.pasa_bajos_hz:.0f} Hz "
                        f"y notch de {cfg.notch_hz:.0f} Hz, IIR ida y vuelta\n", "ok")

        paso(3, "Artefacto ocular")
        if cfg.remover_ocular:
            referencia = actual.senales[EOG_E1] - actual.senales[EOG_E2]
            res["ganancia"] = etapa("Ocular", ocular.actividad_ocular, referencia, fs, cfg)
            limpio = etapa("Ocular", ocular.remover_artefacto_ocular, actual, cfg)
            if limpio is not None:
                actual = limpio
                res["intermedios"]["sin_ocular"] = dict(actual.senales)
                activa = 100 * (res["ganancia"] >= 1).mean() if res["ganancia"] is not None else np.nan
                avisar("texto", f"  ✔ {cfg.metodo_ocular.capitalize()} con compuerta, activa en el "
                                f"{numero(activa, 0)} % de la noche\n", "ok")
        else:
            avisar("texto", "  ⚠ Desactivada en la configuración\n", "aviso")

        paso(4, "Artefacto cardíaco")
        if cfg.remover_cardiaco:
            antes = actual
            limpio = etapa("Cardíaco", cardiaco.remover_artefacto_cardiaco, actual, latidos, cfg)
            if limpio is not None:
                actual = limpio
                res["intermedios"]["sin_cardiaco"] = dict(actual.senales)
                prom = etapa("Cardíaco", promedio_cardiaco, antes.senales[EEG_F4], actual.senales[EEG_F4], latidos, fs)
                if prom is not None:
                    res["promedios"][EEG_F4] = prom
                    avisar("texto", f"  ✔ Promedio de F4 alrededor de los latidos de {numero(np.ptp(prom['antes']), 1)} "
                                    f"a {numero(np.ptp(prom['despues']), 1)} µV pico a pico\n", "ok")
        else:
            avisar("texto", "  ⚠ Desactivada en la configuración\n", "aviso")
        res["final"] = actual

        paso(5, "Segmentación y marca de no estacionariedad")
        res["cambio"] = etapa("Segmentación", segmentacion.cambio_de_error, actual)
        if res["cambio"] is not None:
            res["marcas"] = res["cambio"] > segmentacion.CAMBIO_MAXIMO
            avisar("texto", f"  ✔ {res['marcas'].sum()} épocas no estacionarias, el "
                            f"{numero(100 * res['marcas'].mean(), 1)} % de la noche\n", "ok")
        else:
            res["marcas"] = np.zeros(registro.n_epocas, dtype=bool)

        paso(6, "Variabilidad de la frecuencia cardíaca")
        res["hrv"] = etapa("HRV", analizar_hrv, actual, latidos, res["marcas"])
        if res["hrv"] is not None:
            h = res["hrv"]
            descartados = 100 * (1 - len(h["conservados"]) / max(len(latidos), 1))
            avisar("texto", f"  ✔ {numero(descartados, 1)} % de latidos descartados por forma, "
                            f"{miles(len(h['rr']))} intervalos válidos\n", "ok")

        paso(7, "Clasificación")
        datos = etapa("Clasificación", tablas_de_clasificacion, fuente)
        if datos is not None:
            res["tabla"], res["entrenamiento"] = datos
            res["clasif"] = etapa("Clasificación", clasificar, *datos, cfg_clasificacion)
        if res["clasif"] is not None:
            avisar("texto", f"  ✔ κ de Cohen de {numero(res['clasif']['reporte']['kappa'], 3)} contra los técnicos\n", "ok")
        avisar("listo", res, time.perf_counter() - inicio)

    def reclasificar(self):
        if self.res is None or self.res.get("tabla") is None:
            return
        version, tabla, entrenamiento = self.version, self.res["tabla"], self.res["entrenamiento"]
        cfg_clasificacion = self._cfg_clasificacion()

        def trabajo():
            try:
                self.cola.put(("clasif", version, clasificar(tabla, entrenamiento, cfg_clasificacion)))
            except Exception as e:
                self.cola.put(("texto", version, f"✖ Clasificación. {type(e).__name__}, {e}\n", "error"))

        threading.Thread(target=trabajo, daemon=True).start()

    def _revisar_cola(self):
        try:
            while True:
                tipo, version, *datos = self.cola.get_nowait()
                if version != self.version:
                    continue
                if tipo == "texto":
                    self._escribir(*datos)
                elif tipo == "paso":
                    self.barra.configure(value=datos[0] - 1)
                    self.lbl_estado.configure(text=f"Paso {datos[0]} de {TOTAL_PASOS}, {datos[1].lower()}")
                elif tipo == "listo":
                    self._terminar(*datos)
                elif tipo == "fallo":
                    self.procesando = False
                    self.btn_procesar.state(["!disabled"])
                    self.lbl_estado.configure(text=datos[0])
                elif tipo == "clasif" and self.res is not None:
                    self.res["clasif"] = datos[0]
                    self.sucias.add("clasif")
                    self._dibujar_visible()
        except queue.Empty:
            pass
        self.after(50, self._revisar_cola)

    def _escribir(self, texto, etiqueta=None):
        self.texto.configure(state="normal")
        self.texto.insert("end", texto, etiqueta or ())
        self.texto.configure(state="disabled")
        self.texto.see("end")

    def _terminar(self, res, segundos):
        self.procesando = False
        self.res = res
        self.btn_procesar.state(["!disabled"])
        self.barra.configure(value=TOTAL_PASOS)
        self.lbl_estado.configure(text=f"Listo, procesado en {numero(segundos, 1)} s.")
        self._escribir(f"\nTiempo total {numero(segundos, 1)} s\n")
        self._actualizar_epoca()
        self.pestanas.select(2)

    # ---------------- Dibujo

    def _redibujar_todo(self):
        self.sucias = {"franja", "senales", "prep", "hrv", "clasif"}
        self._dibujar_visible()

    def _redibujar_visible(self):
        self.sucias.add(self._pestana_visible())
        self._dibujar_visible()

    def _pestana_visible(self) -> str:
        return ("avance", "senales", "prep", "hrv", "clasif")[self.pestanas.index(self.pestanas.select())]

    def _dibujar_visible(self):
        self._pendiente = None
        if self.registro is None:
            return
        for nombre in ("franja", self._pestana_visible()):
            if nombre in self.sucias:
                self.sucias.discard(nombre)
                dibujar = {"franja": self._dibujar_franja, "senales": self._dibujar_senales, "prep": self._dibujar_prep,
                           "hrv": self._dibujar_hrv, "clasif": self._dibujar_clasificacion}.get(nombre)
                if dibujar is not None:
                    dibujar()

    def _dibujar_franja(self):
        g = self.g_franja
        g.limpiar()
        ax = g.figura.add_subplot()
        dibujar_hipnograma(ax, self.registro.estadios, self.epoca)
        ax.set_xlabel("Tiempo (h). Hacé click para ir a una época")
        g.lienzo.draw_idle()

    def _canales(self, g, senales: dict, colores: dict, texto: str):
        g.limpiar()
        t = np.arange(self._tramo(self.epoca).stop - self._tramo(self.epoca).start) / self.registro.fs
        axs = np.atleast_1d(g.figura.subplots(len(senales), 1, sharex=True))
        for ax, (nombre, x) in zip(axs, senales.items()):
            ax.plot(t, x, c=colores.get(nombre, PALETA["series"][0]), lw=0.6)
            ax.set_ylabel(nombre, rotation=0, ha="right", va="center", fontsize=8)
            estilar(ax)
        axs[-1].set_xlabel("Tiempo dentro de la época (s)")
        axs[-1].set_xlim(t[0], t[-1])
        titulo(axs[0], texto)
        g.lienzo.draw_idle()

    def _dibujar_senales(self):
        tramo = self._tramo(self.epoca)
        s = PALETA["series"]
        senales = {corto(c): x[tramo] for c, x in self.registro.senales.items()}
        colores = {corto(c): s[0] for c in CANALES_EEG} | {corto(EOG_E1): s[2], corto(EOG_E2): s[2],
                                                          corto(EMG): s[1], ECG: s[4]}
        self._canales(self.g_senales, senales, colores,
                      f"Época {self.epoca + 1}, {nombre_estadio(int(self.registro.estadios[self.epoca]))} según los técnicos")

    def _sin_procesar(self, g, etiqueta, dato):
        etiqueta.configure(text="")
        dato.configure(text="")
        g.mensaje("Procesá el registro para ver esta parte.")

    def _canal_elegido(self) -> str:
        return next(c for c in CANALES_EEG if corto(c) == self.var_canal.get())

    def _dibujar_prep(self):
        vista = self.var_vista_prep.get()
        self.cmb_canal.state(["disabled"] if vista == "Segmentación" else ["!disabled"])
        if self.res is None or self.res.get("final") is None:
            self._sin_procesar(self.g_prep, self.lbl_prep, self.lbl_prep_dato)
            return
        self.lbl_prep.configure(text=EXPLICACION_PREP[vista])
        self.lbl_prep_dato.configure(text="")
        {"Cadena completa": self._prep_cadena, "Filtrado": self._prep_filtrado, "Artefacto ocular": self._prep_ocular,
         "Artefacto cardíaco": self._prep_cardiaco, "Segmentación": self._prep_segmentacion}[vista]()
        self.g_prep.lienzo.draw_idle()

    def _prep_cadena(self):
        canal, tramo = self._canal_elegido(), self._tramo(self.epoca)
        p = PALETA
        senales = {NOMBRE_ETAPA[e]: canales[canal][tramo] for e, canales in self.res["intermedios"].items()}
        colores = {nombre: (p["texto2"] if nombre == "Crudo" else p["series"][0]) for nombre in senales}
        self._canales(self.g_prep, senales, colores, f"{corto(canal)}, época {self.epoca + 1}")

    def _prep_filtrado(self):
        g, p, c = self.g_prep, PALETA, CFG.preprocesamiento
        canal, tramo, fs = self._canal_elegido(), self._tramo(self.epoca), self.registro.fs
        g.limpiar()
        izquierda, derecha = g.figura.subplots(1, 2)
        visibles = []
        for etapa, color in (("crudo", p["tenue"]), ("filtrado", p["series"][0])):
            f, psd = signal.welch(self.res["intermedios"][etapa][canal][tramo], fs, nperseg=int(4 * fs))
            izquierda.semilogy(f, psd, c=color, label=NOMBRE_ETAPA[etapa])
            visibles.append(psd[(f >= 0.5) & (f <= 70)])
        for corte in (c.pasa_altos_hz, c.pasa_bajos_hz, c.notch_hz):
            izquierda.axvline(corte, c=p["series"][1], ls="--", lw=0.8)
        visibles = np.concatenate(visibles)
        izquierda.set_ylim(max(visibles.min(), visibles.max() * 1e-8) / 2, visibles.max() * 3)
        izquierda.set(xlim=(0, 70), xlabel="Frecuencia (Hz)", ylabel="PSD (µV²/Hz)")
        estilar(izquierda)
        leyenda(izquierda)
        titulo(izquierda, f"Espectro de {corto(canal)} en la época")
        f, magnitud, _ = filtrado.respuesta_en_frecuencia(fs, c)
        derecha.semilogx(f[1:], 40 * np.log10(np.maximum(magnitud[1:], 1e-12)), c=p["series"][0])
        derecha.set(xlim=(0.05, fs / 2), ylim=(-80, 5), xlabel="Frecuencia (Hz)", ylabel="Ganancia (dB)")
        estilar(derecha)
        titulo(derecha, "Respuesta del filtro del EEG, ida y vuelta")

    def _prep_ocular(self):
        g, p = self.g_prep, PALETA
        if "sin_ocular" not in self.res["intermedios"]:
            g.mensaje("La remoción ocular está desactivada en la configuración.")
            return
        canal, tramo, fs = self._canal_elegido(), self._tramo(self.epoca), self.registro.fs
        filtradas = self.res["intermedios"]["filtrado"]
        referencia = (filtradas[EOG_E1] - filtradas[EOG_E2])[tramo]
        ganancia = self.res["ganancia"][tramo]
        t = np.arange(len(referencia)) / fs
        self.lbl_prep_dato.configure(text=f"Compuerta activa en el {numero(100 * ganancia.mean(), 0)} % de la época y en el "
                                          f"{numero(100 * (self.res['ganancia'] >= 1).mean(), 0)} % de la noche")
        g.limpiar()
        arriba, abajo = g.figura.subplots(2, 1, sharex=True)
        arriba.plot(t, referencia, c=p["series"][2], lw=0.6)
        techo = max(np.abs(referencia).max(), 1e-6) * 1.05
        if (ganancia > 0.01).any():
            arriba.fill_between(t, -techo, techo, where=ganancia > 0.01, color=p["series"][1], alpha=0.15, lw=0,
                                label="Compuerta activa")
            leyenda(arriba, loc="upper right")
        arriba.set(ylim=(-techo, techo), ylabel="E1 − E2 (µV)")
        abajo.plot(t, filtradas[canal][tramo], c=p["tenue"], lw=0.7, label="Antes")
        abajo.plot(t, self.res["intermedios"]["sin_ocular"][canal][tramo], c=p["series"][0], lw=0.6, label="Después")
        abajo.set(ylabel=f"{corto(canal)} (µV)", xlabel="Tiempo dentro de la época (s)", xlim=(t[0], t[-1]))
        for ax in (arriba, abajo):
            estilar(ax)
        leyenda(abajo, loc="upper right", ncols=2)

    def _prep_cardiaco(self):
        g, p = self.g_prep, PALETA
        intermedios = self.res["intermedios"]
        if "sin_cardiaco" not in intermedios:
            g.mensaje("La remoción cardíaca está desactivada en la configuración.")
            return
        canal, tramo, fs = self._canal_elegido(), self._tramo(self.epoca), self.registro.fs
        anterior = intermedios["sin_ocular" if "sin_ocular" in intermedios else "filtrado"]
        if canal not in self.res["promedios"]:
            self.res["promedios"][canal] = promedio_cardiaco(anterior[canal], intermedios["sin_cardiaco"][canal],
                                                             self.res["latidos"], fs)
        prom = self.res["promedios"][canal]
        picos = {k: np.ptp(prom[k]) for k in ("antes", "despues", "azar")}
        self.lbl_prep_dato.configure(
            text=f"Promedio de la noche de {numero(picos['antes'], 1)} µV pico a pico antes y {numero(picos['despues'], 1)} µV "
                 f"después. Control al azar de {numero(picos['azar'], 1)} µV")

        ecg = self.registro.senales[ECG][tramo]
        latidos = self.res["latidos"]
        en_epoca = latidos[(latidos >= tramo.start) & (latidos < tramo.stop)] - tramo.start
        antes, despues = anterior[canal][tramo], intermedios["sin_cardiaco"][canal][tramo]
        t = np.arange(len(ecg)) / fs
        g.limpiar()
        arriba, abajo = g.figura.subfigures(2, 1, height_ratios=[1.6, 1])
        for sub in (arriba, abajo):
            sub.set_facecolor(p["superficie"])
        axs = arriba.subplots(3, 1, sharex=True)
        axs[0].plot(t, ecg, c=p["series"][4], lw=0.6)
        axs[0].plot(en_epoca / fs, ecg[en_epoca], "o", c=p["texto2"], ms=3)
        axs[0].set_ylabel("ECG (µV)")
        axs[1].plot(t, antes - despues, c=p["series"][1], lw=0.7)
        axs[1].set_ylabel("Restado (µV)")
        axs[2].plot(t, antes, c=p["tenue"], lw=0.7, label="Antes")
        axs[2].plot(t, despues, c=p["series"][0], lw=0.6, label="Después")
        axs[2].set(ylabel=f"{corto(canal)} (µV)", xlabel="Tiempo dentro de la época (s)", xlim=(t[0], t[-1]))
        for ax in axs:
            estilar(ax)
        leyenda(axs[2], loc="upper right", ncols=2)
        ax = abajo.subplots()
        ax.plot(prom["t"], prom["antes"], c=p["series"][1], label="Antes")
        ax.plot(prom["t"], prom["despues"], c=p["series"][0], label="Después")
        ax.plot(prom["t"], prom["azar"], c=p["texto2"], ls="--", lw=0.8, label="Instantes al azar")
        ax.set(xlabel="Tiempo desde la onda R (s)", ylabel="Promedio (µV)")
        estilar(ax)
        leyenda(ax, loc="lower right")
        titulo(ax, f"Promedio de {corto(canal)} alrededor de {miles(len(latidos))} latidos")

    def _prep_segmentacion(self):
        g, p = self.g_prep, PALETA
        final, fs, e = self.res["final"], self.registro.fs, self.epoca
        epoca_c4 = segmentar(final.senales[EEG_C4], fs, final.n_epocas)[e]
        errores = segmentacion.errores_por_tramo(epoca_c4, fs)
        marcas = self.res["marcas"]
        self.lbl_prep_dato.configure(text=f"Época {'marcada como no estacionaria' if marcas[e] else 'estacionaria'}. "
                                          f"En la noche hay {marcas.sum()} épocas marcadas, el "
                                          f"{numero(100 * marcas.mean(), 1)} %")
        t = np.arange(len(epoca_c4)) / fs
        g.limpiar()
        arriba, medio, abajo = g.figura.subplots(3, 1, height_ratios=[2.2, 1.4, 0.5])
        arriba.plot(t, epoca_c4, c=p["series"][0], lw=0.6)
        techo = max(np.abs(epoca_c4).max(), 1e-6) * 1.15
        if errores is not None:
            for k, error in enumerate(errores):
                arriba.axvline(5 * k, c=p["tenue"], lw=0.6, ls=":")
                supera = error > segmentacion.CAMBIO_MAXIMO
                arriba.text(5 * k + 2.5, techo, "ajuste" if k == 0 else f"×{numero(error, 1)}", ha="center", va="bottom",
                            color=p["series"][1] if supera else p["texto2"], fontweight="bold" if supera else "normal")
        arriba.set(ylim=(-techo, techo * 1.35), xlim=(0, t[-1]), ylabel="C4 (µV)", xlabel="Tiempo dentro de la época (s)")
        estilar(arriba)
        dibujar_hipnograma(medio, self.registro.estadios, e)
        medio.tick_params(labelbottom=False)
        abajo.eventplot(np.flatnonzero(marcas) * EPOCA_H, colors=p["series"][1], lineoffsets=0.5, linelengths=0.8,
                        linewidths=0.6)
        abajo.set(yticks=[], ylabel="Marcadas", xlabel="Tiempo (h)", xlim=medio.get_xlim())
        estilar(abajo, grilla=False)

    def _dibujar_hrv(self):
        vista = self.var_vista_hrv.get()
        if self.res is None or self.res.get("hrv") is None:
            self._sin_procesar(self.g_hrv, self.lbl_hrv, self.lbl_hrv_dato)
            return
        h, latidos, fs = self.res["hrv"], self.res["latidos"], self.registro.fs
        descartados = 100 * (1 - len(h["conservados"]) / max(len(latidos), 1))
        self.lbl_hrv.configure(text=EXPLICACION_HRV[vista])
        self.lbl_hrv_dato.configure(
            text=f"{miles(len(latidos))} latidos  ·  RR mediano {numero(np.median(np.diff(latidos)) / fs)} s  ·  "
                 f"{numero(descartados, 1)} % descartados por forma  ·  "
                 f"{numero(100 * len(h['rr']) / max(len(h['rr_todos']), 1), 0)} % de intervalos válidos")
        self.g_hrv.limpiar()
        {"Latidos de la época": self._hrv_latidos, "Descarte por forma": self._hrv_forma,
         "Intervalos RR y HRV": self._hrv_noche, "Periodograma de Lomb": self._hrv_lomb}[vista]()
        self.g_hrv.lienzo.draw_idle()

    def _hrv_latidos(self):
        p, h = PALETA, self.res["hrv"]
        tramo, fs = self._tramo(self.epoca), self.registro.fs
        ecg = self.registro.senales[ECG][tramo]
        latidos = self.res["latidos"]
        descartados = latidos[~np.isin(latidos, h["conservados"])]
        t = np.arange(len(ecg)) / fs
        ax = self.g_hrv.figura.add_subplot()
        ax.plot(t, ecg, c=p["series"][4], lw=0.6)
        for marcas, color, simbolo, etiqueta in ((h["conservados"], p["series"][0], "o", "Conservado"),
                                                 (descartados, p["series"][1], "x", "Descartado")):
            en_epoca = marcas[(marcas >= tramo.start) & (marcas < tramo.stop)] - tramo.start
            ax.plot(en_epoca / fs, ecg[en_epoca], simbolo, c=color, ms=6, label=etiqueta)
        ax.set(xlim=(0, t[-1]), xlabel="Tiempo dentro de la época (s)", ylabel="ECG (µV)")
        estilar(ax)
        leyenda(ax, loc="upper right", ncols=2)
        titulo(ax, f"ECG de la época {self.epoca + 1}")

    def _hrv_forma(self):
        p, h, fs = PALETA, self.res["hrv"], self.registro.fs
        izquierda, derecha = self.g_hrv.figura.subplots(1, 2)
        correlacion, ventanas = h["correlacion"], h["ventanas"]
        desde, hasta = (round(s * fs) for s in hrv.VENTANA_LATIDO_S)
        t = np.arange(desde, hasta + 1) / fs
        rng = np.random.default_rng(0)
        for grupo, color in ((correlacion >= hrv.UMBRAL_FORMA, p["series"][0]), (correlacion < hrv.UMBRAL_FORMA, p["series"][1])):
            indices = np.flatnonzero(grupo)
            for i in (rng.choice(indices, min(20, len(indices)), replace=False) if len(indices) else []):
                izquierda.plot(t, ventanas[i], c=color, lw=0.5, alpha=0.35)
        izquierda.plot(t, h["tipico"], c=p["texto"], lw=1.8, label="Latido típico")
        izquierda.plot([], [], c=p["series"][0], label="Conservados")
        izquierda.plot([], [], c=p["series"][1], label="Descartados")
        izquierda.set(xlabel="Tiempo desde la onda R (s)", ylabel="ECG filtrado (µV)")
        estilar(izquierda)
        leyenda(izquierda, loc="upper right")
        titulo(izquierda, "Latido típico con 20 conservados y 20 descartados")
        derecha.hist(correlacion, bins=np.linspace(-0.2, 1, 61), color=p["series"][0])
        derecha.axvline(hrv.UMBRAL_FORMA, c=p["series"][1], ls="--")
        derecha.set(yscale="log", xlabel="Correlación con el latido típico", ylabel="Latidos")
        estilar(derecha)
        titulo(derecha, f"Se conservan los latidos con {numero(hrv.UMBRAL_FORMA, 1)} o más")

    def _hrv_noche(self):
        p, h = PALETA, self.res["hrv"]
        axs = self.g_hrv.figura.subplots(3, 1, sharex=True, height_ratios=[1.2, 2, 1.4])
        dibujar_hipnograma(axs[0], self.registro.estadios, self.epoca)
        axs[1].plot(h["t_todos"] / 3600, h["rr_todos"], ".", c=p["tenue"], ms=1.5, label="Descartados")
        axs[1].plot(h["t_rr"] / 3600, h["rr"], ".", c=p["series"][0], ms=1.5, label="Válidos")
        axs[1].set(ylabel="RR (s)", ylim=(0.2, 2.1))
        horas = (np.arange(len(h["lf_hf"])) + 0.5) * EPOCA_H
        axs[2].semilogy(horas, h["lf_hf"], c=p["series"][0], lw=0.7)
        axs[2].axhline(1, c=p["texto2"], lw=0.6, ls=":")
        axs[2].axhline(CFG.clasificacion.r_lf_hf_min, c=p["series"][1], lw=0.8, ls="--")
        axs[2].set(ylabel="LF/HF (veces la mediana)", xlabel="Tiempo (h)")
        for ax in axs[1:]:
            estilar(ax)
            ax.axvline((self.epoca + 0.5) * EPOCA_H, c=p["error"], lw=1.2)
        leyenda(axs[1], loc="upper right", ncols=2, markerscale=6)

    def _hrv_lomb(self):
        p, h = PALETA, self.res["hrv"]
        centro = (self.epoca + 0.5) * DURACION_EPOCA_S
        mitad = ConfigDescriptores().hrv_ventana_s / 2
        en_ventana = (h["t_rr"] >= centro - mitad) & (h["t_rr"] < centro + mitad)
        if en_ventana.sum() < hrv.MIN_INTERVALOS:
            self.g_hrv.mensaje("En esta ventana hay muy pocos intervalos válidos. La época toma la mediana de la noche.")
            return
        t_v, rr_v = h["t_rr"][en_ventana], h["rr"][en_ventana]
        frecuencias, psd = hrv.periodograma_lomb(t_v, rr_v)
        lf, hf = hrv.potencias_lomb(t_v, rr_v)
        self.lbl_hrv_dato.configure(text=f"LF {numero(lf * 1e6, 0)} ms²  ·  HF {numero(hf * 1e6, 0)} ms²  ·  "
                                         f"LF/HF {numero(lf / hf) if hf > 0 else '—'}  ·  {int(en_ventana.sum())} intervalos")
        ax = self.g_hrv.figura.add_subplot()
        ax.plot(frecuencias, psd * 1e6, c=p["series"][0])
        ax.axvspan(*BANDA_LF, color=p["series"][1], alpha=0.12, lw=0, label="LF")
        ax.axvspan(*BANDA_HF, color=p["series"][2], alpha=0.12, lw=0, label="HF")
        ax.set(xlabel="Frecuencia (Hz)", ylabel="PSD (ms²/Hz)", xlim=(BANDA_LF[0], BANDA_HF[1]))
        estilar(ax)
        leyenda(ax, loc="upper right")
        titulo(ax, "Periodograma de Lomb de los 5 min alrededor de la época")

    def _dibujar_clasificacion(self):
        g, tabla_ui = self.g_clasif, self.t_clasif
        tabla_ui.delete(*tabla_ui.get_children())
        if self.res is None or self.res.get("clasif") is None:
            self.lbl_clase.configure(text="Procesá el registro para ver la clasificación.", style="ClaseSin.TLabel")
            g.mensaje("Procesá el registro para ver la clasificación.")
            return
        p, c, tabla = PALETA, self.res["clasif"], self.res["tabla"]
        e = min(self.epoca, len(tabla.estadios) - 1)
        tecnicos, estimado = int(tabla.estadios[e]), int(c["estimado"][e])
        coincide = tecnicos == estimado
        self.lbl_clase.configure(
            text=f"Época {e + 1}. Técnicos {nombre_estadio(tecnicos)}, clasificador {nombre_estadio(estimado)}",
            style=f"Clase{'ok' if coincide else 'error'}.TLabel")

        def fila(dato, valor=""):
            tabla_ui.insert("", "end", values=(dato, valor))

        fila("κ de Cohen del registro", numero(c["reporte"]["kappa"], 3))
        fila("κ entre técnicos de HMC", "0,74")
        fila("κ de la red de referencia", "0,79")
        fila("Regla que decidió la época", reglas.REGLAS[c["regla"][e]])
        fila("Reglas sin suavizar", nombre_estadio(int(c["sin_suavizar"][e])))
        for nombre, valores in c["valores"].items():
            fila(f"{NOMBRE_DESCRIPTOR.get(nombre, nombre)} (valor de la regla)", numero(float(valores[e]), 2))
        for estadio, f1 in c["reporte"]["f1"].items():
            fila(f"F1 de {estadio}", numero(f1, 3))

        g.limpiar()
        arriba, abajo = g.figura.subfigures(2, 1, height_ratios=[1.25, 1])
        for sub in (arriba, abajo):
            sub.set_facecolor(p["superficie"])
        series = {"Técnicos": tabla.estadios, "Reglas": c["sin_suavizar"], "Suavizado": c["estimado"]}
        axs = arriba.subplots(3, 1, sharex=True)
        for ax, (nombre, y) in zip(axs, series.items()):
            dibujar_hipnograma(ax, y, e)
            ax.set_ylabel(nombre, fontsize=8)
            ax.tick_params(labelbottom=False)
        axs[-1].tick_params(labelbottom=True)
        axs[-1].set_xlabel("Tiempo (h)")
        izquierda, derecha = abajo.subplots(1, 2)
        matriz = c["reporte"]["confusion"]
        normalizada = matriz / matriz.sum(axis=1, keepdims=True).clip(min=1)
        izquierda.imshow(normalizada, cmap="Blues", vmin=0, vmax=1)
        for i in range(N_ESTADIOS):
            for j in range(N_ESTADIOS):
                izquierda.text(j, i, f"{numero(normalizada[i, j])}\n({matriz[i, j]})", ha="center", va="center", fontsize=7,
                               color="white" if normalizada[i, j] > 0.6 else "black")
        izquierda.set_xticks(range(N_ESTADIOS), NOMBRES_ESTADIOS)
        izquierda.set_yticks(range(N_ESTADIOS), NOMBRES_ESTADIOS)
        izquierda.set(xlabel="Clasificador", ylabel="Técnicos")
        estilar(izquierda, grilla=False)
        titulo(izquierda, "Matriz de confusión por fila")
        conteo = np.bincount(c["regla"], minlength=len(reglas.REGLAS))
        derecha.bar(reglas.REGLAS, conteo, color=p["series"][0])
        derecha.set(ylabel="Épocas")
        estilar(derecha)
        titulo(derecha, "Épocas decididas por cada regla")
        g.lienzo.draw_idle()


def main():
    Aplicacion().mainloop()


if __name__ == "__main__":
    main()
