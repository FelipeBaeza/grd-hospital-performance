"""Comprobación de lo congelado en la Fase 4 (y de los planes B) contra su manifiesto SHA-256."""
from __future__ import annotations

import json
from pathlib import Path

from .config import sha256_archivo


def verificar_manifiesto(carpeta: Path, nombre: str = "manifiesto.json") -> dict:
    """Devuelve {ruta: (ok, huella_esperada, huella_actual)}. Un archivo que falta cuenta como distinto."""
    m = json.loads((carpeta / nombre).read_text(encoding="utf8"))
    res = {}
    for _tarea, arch in m["archivos"].items():
        for _k, d in arch.items():
            ruta = carpeta / d["ruta"]
            actual = sha256_archivo(ruta) if ruta.exists() else "no existe"
            res[d["ruta"]] = (actual == d["sha256"], d["sha256"], actual)
    return res


def exigir_intacto(carpeta: Path, nombre: str = "manifiesto.json") -> dict:
    r = verificar_manifiesto(carpeta, nombre)
    malos = [k for k, (ok, _, _) in r.items() if not ok]
    if malos:
        raise RuntimeError(f"archivos congelados distintos del manifiesto: {malos}")
    return r
