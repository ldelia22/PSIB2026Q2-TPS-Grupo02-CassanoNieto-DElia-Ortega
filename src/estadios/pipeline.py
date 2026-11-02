"""Conecta los frentes: registro crudo -> preprocesado -> descriptores (en caché)."""

from __future__ import annotations

from pathlib import Path

from .comun.config import Config, ConfigPreprocesamiento
from .comun.constantes import ECG
from .comun.contratos import Registro, RegistroPreprocesado, TablaDescriptores
from .datos.carga import DIR_DATOS, DIR_REGISTROS, cargar_registro
from .descriptores.ensamblado import describir_registro
from .preprocesamiento import cardiaco, filtrado, ocular, qrs, segmentacion

DIR_CACHE = DIR_DATOS / "descriptores"


def preprocesar(
    registro: Registro, cfg: ConfigPreprocesamiento, guardar_intermedios: bool = False
) -> RegistroPreprocesado:
    """Etapas 2 a 5. Con `guardar_intermedios`, conserva cada etapa para la interfaz."""
    intermedios = {}

    def anotar(etapa: str, actual: Registro) -> None:
        if guardar_intermedios:
            intermedios[etapa] = dict(actual.senales)

    anotar("crudo", registro)

    # Los latidos se detectan siempre, aunque no se remueva el artefacto: los usa la HRV.
    latidos = qrs.detectar_latidos(registro.senales[ECG], registro.fs)

    actual = filtrado.filtrar_registro(registro, cfg)
    anotar("filtrado", actual)

    if cfg.remover_ocular:
        actual = ocular.remover_artefacto_ocular(actual, cfg)
        anotar("sin_ocular", actual)

    if cfg.remover_cardiaco:
        actual = cardiaco.remover_artefacto_cardiaco(actual, latidos, cfg)
        anotar("sin_cardiaco", actual)

    return RegistroPreprocesado(
        registro=actual,
        latidos=latidos,
        no_estacionaria=segmentacion.marcar_no_estacionarias(actual, cfg),
        intermedios=intermedios,
    )


def ruta_cache(id_registro: str, cfg: Config, dir_cache: Path = DIR_CACHE) -> Path:
    """Cada configuración de preprocesamiento y descriptores tiene su propia carpeta."""
    return dir_cache / cfg.huella_descriptores() / f"{id_registro}.npz"


def obtener_tabla(
    id_registro: str,
    cfg: Config,
    dir_registros: Path = DIR_REGISTROS,
    dir_cache: Path = DIR_CACHE,
) -> TablaDescriptores:
    """Devuelve los descriptores del registro; los calcula solo si no están en caché."""
    ruta = ruta_cache(id_registro, cfg, dir_cache)
    if ruta.exists():
        return TablaDescriptores.cargar(ruta)
    registro = cargar_registro(id_registro, dir_registros)
    tabla = describir_registro(preprocesar(registro, cfg.preprocesamiento), cfg.descriptores)
    tabla.guardar(ruta)
    return tabla
