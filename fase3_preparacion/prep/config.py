"""Rutas y umbrales de la Fase 3."""
from __future__ import annotations

import tomllib
from pathlib import Path

RAIZ: Path = Path(__file__).resolve().parents[1]
RAIZ_PROYECTO: Path = RAIZ.parent
DIR_CONFIG: Path = RAIZ / "config"
DIR_SALIDAS: Path = RAIZ / "salidas"
ANIOS = (2019, 2020, 2021, 2022, 2023, 2024)
ANIOS_RESULTADO = (2019, 2020, 2021, 2022, 2023)   # años con desenlace libre
ANIO_SELLADO = 2024


class Umbrales:
    def __init__(self, datos: dict):
        self.datos = datos
        self.particion = datos["particion"]
        self.sensibles = datos["sensibles"]
        self.referencias = datos["referencias"]

    def __getitem__(self, nombre: str):
        return self.datos["umbral"][nombre]["valor"]


def cargar_umbrales(ruta: Path | str | None = None) -> Umbrales:
    with open(Path(ruta) if ruta else DIR_CONFIG / "umbrales_fase3.toml", "rb") as f:
        return Umbrales(tomllib.load(f))


def anio_split(anio: int, u: Umbrales | None = None) -> str:
    """Conjunto al que pertenece un año de egreso. Se decide solo por el año, nunca al azar."""
    u = u or cargar_umbrales()
    if anio in u.particion["anios_entrenamiento"]:
        return "entrenamiento"
    if anio == u.particion["anio_calibracion"]:
        return "calibracion"
    if anio == u.particion["anio_prueba"]:
        return "prueba"
    raise ValueError(f"año fuera de la partición: {anio}")
