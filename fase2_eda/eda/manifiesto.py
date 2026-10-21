"""Manifiesto de reproducibilidad: huellas SHA-256 de insumos, configuración, código y salidas.

El manifiesto no contiene fechas, rutas absolutas ni nombres de usuario: dos corridas con los
mismos insumos producen el mismo manifiesto, byte a byte.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from . import rutas

PAQUETES = ("polars", "pyarrow", "numpy", "scipy", "matplotlib", "scikit-learn", "pytest")


def sha256_bytes(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def sha256_archivo(ruta: Path | str) -> str:
    with open(ruta, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def versiones() -> dict[str, str]:
    v = {"python": platform.python_version()}
    for p in PAQUETES:
        try:
            v[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            v[p] = "no instalado"
    return v


def _relativa(ruta: Path, bases: Iterable[Path]) -> str:
    ruta = ruta.resolve()
    for b in bases:
        try:
            return ruta.relative_to(b.resolve()).as_posix()
        except ValueError:
            continue
    return ruta.name


def huellas(archivos: Iterable[Path], bases: Iterable[Path] | None = None) -> dict[str, str]:
    """Mapa ruta relativa -> SHA-256, ordenado por ruta."""
    bases = list(bases) if bases is not None else [rutas.RAIZ_EDA, rutas.RAIZ_PROYECTO]
    out = {_relativa(Path(a), bases): sha256_archivo(a) for a in archivos if Path(a).is_file()}
    return dict(sorted(out.items()))


def archivos_de(carpeta: Path, patron: str = "*", excluir: Iterable[str] = ()) -> list[Path]:
    """Archivos de una carpeta (recursivo), sin caché de Python ni los nombres excluidos."""
    excluir = set(excluir)
    res = []
    for p in sorted(carpeta.rglob(patron)):
        if not p.is_file():
            continue
        if "__pycache__" in p.parts or p.name in excluir or p.suffix == ".pyc":
            continue
        res.append(p)
    return res


def construir_manifiesto(
    *,
    fuente: str,
    escaneos: Mapping[int, Any],
    semilla: int,
    archivos_config: Iterable[Path],
    archivos_codigo: Iterable[Path],
    archivos_salida: Iterable[Path],
    resumen_accesos: Mapping[str, Any] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Arma el manifiesto. ``escaneos`` es año -> EscaneoArchivo (de lector.py)."""
    datos = {
        str(anio): {
            "archivo": esc.nombre,
            "tamano_bytes": esc.tamano_bytes,
            "sha256": esc.sha256,
            "lineas_datos": esc.lineas_datos,
        }
        for anio, esc in sorted(escaneos.items())
    }
    return {
        "fuente": fuente,
        "semilla": int(semilla),
        "versiones": versiones(),
        "datos": datos,
        "config": huellas(archivos_config),
        "codigo": huellas(archivos_codigo),
        "salidas": huellas(archivos_salida),
        "accesos": dict(resumen_accesos or {}),
        "extras": dict(extras or {}),
    }


def a_json(manifiesto: Mapping[str, Any]) -> str:
    return json.dumps(manifiesto, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def escribir_manifiesto(manifiesto: Mapping[str, Any], destino: Path) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(a_json(manifiesto), encoding="utf8")
    return destino
