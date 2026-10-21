"""Rutas, reglas fijadas de antemano y constantes de la Fase 4."""
from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

RAIZ: Path = Path(__file__).resolve().parents[1]
DIR_CONFIG: Path = RAIZ / "config"
DIR_SALIDAS: Path = RAIZ / "salidas"
DIR_MODELOS: Path = RAIZ / "modelos"
RUTA_REGLAS: Path = DIR_CONFIG / "reglas_fase4.toml"
RUTA_DICCIONARIO: Path = DIR_CONFIG / "diccionario_oro_fase3.csv"


class Reglas:
    def __init__(self, datos: dict, ruta: Path):
        self.datos = datos
        self.ruta = ruta
        for k, v in datos.items():
            setattr(self, k, v)

    def huella(self) -> str:
        return sha256_archivo(self.ruta)


def cargar_reglas(ruta: Path | str | None = None) -> Reglas:
    ruta = Path(ruta) if ruta else RUTA_REGLAS
    with open(ruta, "rb") as f:
        return Reglas(tomllib.load(f), ruta)


def sha256_archivo(ruta: Path | str) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
