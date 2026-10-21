"""Cortafuegos del año 2024 (regla R7).

El desenlace de 2024 (tipo de alta, fecha de alta y todo lo que se deriva de ellos) vive en un archivo aparte, en la carpeta ``sellado``.
Solo se puede leer con una autorización explícita y cada intento, concedido o no, queda en el registro de accesos.
"""
from __future__ import annotations

import csv
import datetime as dt
import os
from pathlib import Path

import polars as pl

from .config import ANIO_SELLADO, DIR_SALIDAS

VARIABLE_AUTORIZACION = "FASE3_AUTORIZA_DESENLACE_2024"
FRASE = "autorizo-fase-5"


class SelloRoto(PermissionError):
    """Se intentó leer el desenlace de 2024 sin autorización."""


def ruta_registro() -> Path:
    return Path(os.environ.get("FASE3_REGISTRO_ACCESOS", DIR_SALIDAS / "tablas" / "registro_accesos.csv"))


def registrar(evento: str, anio: int, detalle: str, concedido: bool) -> None:
    ruta = ruta_registro()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not ruta.exists()
    with open(ruta, "a", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["momento_utc", "evento", "anio", "detalle", "concedido"])
        w.writerow([dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), evento, anio, detalle, "si" if concedido else "no"])


def _autorizado() -> bool:
    return os.environ.get(VARIABLE_AUTORIZACION) == FRASE


def leer_desenlace(dir_sellado: Path | str, anio: int, columnas: list[str] | None = None, proposito: str = "") -> pl.DataFrame:
    """Lee el archivo de desenlace de un año. Para 2024 exige autorización y deja registro."""
    dir_sellado = Path(dir_sellado)
    if anio == ANIO_SELLADO:
        ok = _autorizado()
        registrar("lectura_desenlace", anio, proposito or "sin propósito declarado", ok)
        if not ok:
            raise SelloRoto("el desenlace de 2024 está sellado hasta la Fase 5")
        return pl.read_parquet(dir_sellado / f"desenlace_{anio}.parquet", columns=columnas)
    return pl.read_parquet(dir_sellado.parent / "plata" / f"desenlace_{anio}.parquet", columns=columnas)


def marca_cohorte_d_2024(dir_sellado: Path | str, ids_cohorte_c: pl.Series) -> pl.DataFrame:
    """Lectura enmascarada: devuelve SOLO ``ID_EPISODIO`` y ``ENTRA_A_D`` (desenlace conocido), nunca la mortalidad.

    Requiere la misma autorización y también queda en el registro. No se ejecuta en la corrida por defecto: depende de la aprobación explícita del profesor guía.
    """
    ok = _autorizado()
    registrar("lectura_enmascarada_cohorte", ANIO_SELLADO, "solo marca de entrada a la cohorte D", ok)
    if not ok:
        raise SelloRoto("la lectura enmascarada de 2024 no está autorizada")
    d = pl.read_parquet(Path(dir_sellado) / f"desenlace_{ANIO_SELLADO}.parquet", columns=["ID_EPISODIO", "ROL"])
    return d.filter(pl.col("ID_EPISODIO").is_in(ids_cohorte_c.implode())).select(
        "ID_EPISODIO", pl.col("ROL").is_in(["EVENTO", "NO_EVENTO"]).alias("ENTRA_A_D")
    )
