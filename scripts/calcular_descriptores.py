"""Calcula y cachea los descriptores de todos los registros descargados.

Uso: python scripts/calcular_descriptores.py
"""

from __future__ import annotations

from estadios.comun.config import Config
from estadios.datos.carga import ids_disponibles
from estadios.pipeline import obtener_tabla


def main() -> None:
    cfg = Config()
    ids = ids_disponibles()
    for i, id_registro in enumerate(ids, 1):
        print(f"[{i}/{len(ids)}] {id_registro}", flush=True)
        obtener_tabla(id_registro, cfg)


if __name__ == "__main__":
    main()
