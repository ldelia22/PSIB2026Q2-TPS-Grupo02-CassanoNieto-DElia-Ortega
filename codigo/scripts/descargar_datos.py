"""Descarga de PhysioNet el subconjunto de registros acordado (unos 100 MB cada uno).

Uso: python scripts/descargar_datos.py [--n 70] [--semilla 42]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

from estadios.datos.carga import DIR_REGISTROS
from estadios.datos.particion import N_REGISTROS, SEMILLA, elegir_registros

BASE = "https://physionet.org/files/hmc-sleep-staging/1.1"


def leer_url(ruta: str) -> str:
    with urllib.request.urlopen(f"{BASE}/{ruta}") as respuesta:
        return respuesta.read().decode()


def sha256(ruta: Path) -> str:
    resumen = hashlib.sha256()
    with ruta.open("rb") as archivo:
        while bloque := archivo.read(1 << 20):
            resumen.update(bloque)
    return resumen.hexdigest()


def descargar(nombre: str, esperado: str, destino: Path) -> None:
    ruta = destino / nombre
    if ruta.exists() and sha256(ruta) == esperado:
        return
    temporal = ruta.with_suffix(ruta.suffix + ".parcial")
    with urllib.request.urlopen(f"{BASE}/recordings/{nombre}") as respuesta, temporal.open("wb") as f:
        shutil.copyfileobj(respuesta, f)
    if sha256(temporal) != esperado:
        temporal.unlink()
        raise RuntimeError(f"{nombre}: el hash no coincide con el publicado")
    temporal.rename(ruta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=N_REGISTROS)
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    parser.add_argument("--destino", type=Path, default=DIR_REGISTROS)
    args = parser.parse_args()

    # Formato de cada línea: "<hash> recordings/SN001.edf"
    hashes = {}
    for linea in leer_url("SHA256SUMS.txt").splitlines():
        valor, ruta = linea.split()
        if ruta.startswith("recordings/"):
            hashes[ruta.removeprefix("recordings/")] = valor

    todos = [n.removesuffix(".edf") for n in hashes if not n.endswith("_sleepscoring.edf") and n.endswith(".edf")]
    elegidos = elegir_registros(todos, args.n, args.semilla)

    args.destino.mkdir(parents=True, exist_ok=True)
    for i, id_registro in enumerate(elegidos, 1):
        print(f"[{i}/{len(elegidos)}] {id_registro}", flush=True)
        for nombre in (f"{id_registro}.edf", f"{id_registro}_sleepscoring.txt"):
            descargar(nombre, hashes[nombre], args.destino)


if __name__ == "__main__":
    main()
