"""Rutas, protocolo pre-registrado y acceso a los paquetes de las Fases 3 y 4."""
from __future__ import annotations

import hashlib
import sys
import tomllib
from pathlib import Path

RAIZ: Path = Path(__file__).resolve().parents[1]
RAIZ_TESIS: Path = RAIZ.parent
FASE3: Path = RAIZ_TESIS / "fase3_preparacion"
FASE4: Path = RAIZ_TESIS / "fase4_modelado"
for p in (FASE4, FASE3):                       # los modelos de la Fase 4 se cargan con sus propias clases (Modelo, Platt, FactorUnico)
    if str(p) not in sys.path:
        sys.path.append(str(p))

RUTA_PROTOCOLO: Path = RAIZ / "config" / "protocolo_fase5.toml"
DIR_SALIDAS: Path = RAIZ / "salidas"
DIR_PLANES_B: Path = RAIZ / "planes_b"
REGISTRO_FASE3: Path = FASE3 / "salidas" / "tablas" / "registro_accesos.csv"
DIR_MODELOS_F4: Path = FASE4 / "modelos"


def sha256_archivo(ruta: Path | str) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Protocolo:
    def __init__(self, datos: dict, ruta: Path):
        self.datos, self.ruta = datos, ruta
        for k, v in datos.items():
            setattr(self, k, v)

    def huella(self) -> str:
        return sha256_archivo(self.ruta)


def cargar_protocolo(ruta: Path | str | None = None) -> Protocolo:
    ruta = Path(ruta) if ruta else RUTA_PROTOCOLO
    with open(ruta, "rb") as f:
        return Protocolo(tomllib.load(f), ruta)
