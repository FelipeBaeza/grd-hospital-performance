"""Ayudas de prueba de la Tarea 13: clonar una carpeta de salidas, registros hechos a mano y lanzadores falsos (no lanzan procesos)."""

from __future__ import annotations

import dataclasses
import shutil
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import polars as pl

from eda import cortafuegos, orquestador
from eda.contexto import Contexto

COLUMNAS_CIFRAS_INFORME = ("informe", "marcador", "csv", "columna", "filtro", "formato", "texto")


def clonar(ctx: Contexto, destino: Path | str, etiqueta: str | None = None) -> Contexto:
    """Copia la carpeta de salidas de ``ctx`` a ``<destino>/<etiqueta>`` y devuelve un contexto nuevo con los registros cargados de esa copia.

    Sirve para que una prueba dañe una salida sin tocar la de la sesión (que es de solo lectura). Con ``etiqueta="completo"`` la copia es de modo completo, pero sus tablas y métricas
    conservan la etiqueta con la que se escribieron.
    """
    destino = Path(destino)
    etiqueta = etiqueta or ctx.etiqueta
    shutil.copytree(ctx.dir_salida, destino / etiqueta)
    fuente = ctx.fuente if etiqueta == ctx.fuente.etiqueta else dataclasses.replace(ctx.fuente, etiqueta=etiqueta)
    nuevo = Contexto.crear(etiqueta, fuente=fuente, dir_salida_base=destino)
    orquestador.hidratar_registros(nuevo)
    return nuevo


def etiquetar_completo(ctx: Contexto) -> None:
    """Pone la etiqueta ``completo`` a todas las tablas y métricas registradas en memoria de ``ctx``."""
    for t in ctx._tablas.values():
        t["fuente"] = "completo"
    for m in ctx._metricas:
        m["nivel"] = "completo"


def registro(*filas: Mapping[str, object]) -> pl.DataFrame:
    """Registro de accesos del cortafuegos hecho a mano. Cada fila puede traer ``tarea``, ``anio``, ``proposito``, ``columnas_resultado``, ``reglas``, ``decision`` y ``motivo``."""
    base = {"tarea": "T03", "anio": 2019, "proposito": "estructura", "n_columnas": 1, "columnas_resultado": "", "reglas": "", "decision": cortafuegos.PERMITIDO, "motivo": "año no expuesto"}
    return pl.DataFrame([{**base, "secuencia": i, **f} for i, f in enumerate(filas, start=1)], schema=cortafuegos.ESQUEMA_REGISTRO)


def cifras_informe(*filas: Mapping[str, str]) -> pl.DataFrame:
    """Registro de cifras de los informes (todo texto) hecho a mano; las columnas que falten quedan vacías."""
    return pl.DataFrame([{c: f.get(c, "") for c in COLUMNAS_CIFRAS_INFORME} for f in filas], schema={c: pl.String for c in COLUMNAS_CIFRAS_INFORME})


class _GruposAlReves:
    """Lo que devuelve ``DataFrame.group_by`` cuando se reproduce el orden arbitrario de otra corrida: los grupos y las filas agregadas salen al revés."""

    def __init__(self, grupos) -> None:
        self._grupos = grupos

    def __iter__(self):
        return iter(reversed(list(self._grupos)))

    def __getattr__(self, nombre):
        atributo = getattr(self._grupos, nombre)
        if not callable(atributo):
            return atributo

        def envuelto(*a, **k):
            r = atributo(*a, **k)
            return r.reverse() if isinstance(r, pl.DataFrame) else r

        return envuelto


@contextmanager
def grupos_al_reves():
    """Mientras dura el bloque, ``DataFrame.group_by`` (sin ``maintain_order=True``) entrega los grupos en el orden contrario: reproduce el orden de otra corrida (DT11)."""
    original = pl.DataFrame.group_by

    def group_by(self, *by, maintain_order=False, **kw):
        g = original(self, *by, maintain_order=maintain_order, **kw)
        return g if maintain_order else _GruposAlReves(g)

    with mock.patch.object(pl.DataFrame, "group_by", group_by):
        yield


class LanzadorCorrida:
    """Lanzador falso de DT11: en cada llamada escribe los archivos pedidos bajo ``<base>/extracto`` y devuelve el código dado. No lanza ningún proceso."""

    def __init__(self, archivos: Sequence[Mapping[str, bytes]], codigos: Sequence[int] = (0, 0), falla_en: int | None = None) -> None:
        self.archivos, self.codigos, self.falla_en = list(archivos), list(codigos), falla_en
        self.bases: list[Path] = []

    def __call__(self, base: Path) -> int:
        i = len(self.bases)
        self.bases.append(Path(base))
        if self.falla_en == i:
            raise RuntimeError("el lanzador falló")
        for ruta, contenido in self.archivos[i].items():
            destino = Path(base) / "extracto" / ruta
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(contenido)
        return self.codigos[i]
