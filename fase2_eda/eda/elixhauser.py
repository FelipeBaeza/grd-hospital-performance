"""Índice de Elixhauser (31 categorías, Quan et al. 2005, CIE-10) calculado solo con diagnósticos secundarios (regla R3).

El mapeo vive en ``config/elixhauser_quan2005.csv`` (un prefijo por fila, sin punto, contrastado con dos fuentes públicas) y los
metadatos de las categorías (orden ELIX_01 a ELIX_31, pesos de van Walraven 2009, jerarquías y las cinco categorías sospechosas
de la Fase 1) en ``config/elixhauser_categorias.csv``.

Reglas de cálculo:
- Los códigos se normalizan (mayúsculas, sin espacios ni puntos) y se comparan por PREFIJO: ``I50`` abarca I500, I509, etc.
- Un código puede pertenecer a más de una categoría (por ejemplo I42.6 es insuficiencia cardíaca y abuso de alcohol).
- Jerarquías: la hipertensión complicada anula a la no complicada, la diabetes complicada a la no complicada y el cáncer
  metastásico al tumor sólido sin metástasis.
- No existe marcador de presente al ingreso (POA): los diagnósticos están codificados al egreso y algunas categorías pueden ser
  complicaciones de la hospitalización (limitación declarada, Fase 1 sección 5.1).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import polars as pl

from . import config, esquema, rutas

IDS: tuple[str, ...] = tuple(f"ELIX_{i:02d}" for i in range(1, 32))


@dataclass(frozen=True)
class Elixhauser:
    ids: tuple[str, ...]
    nombres_es: dict[str, str]
    nombres_en: dict[str, str]
    pesos: dict[str, int]
    domina: dict[str, str]
    sospechosas: frozenset[str]
    prefijos: dict[str, int]  # prefijo normalizado -> máscara de bits (bit i = categoría i, 0 = ELIX_01)
    largos: tuple[int, ...]  # largos de prefijo presentes en el mapeo
    solo_3c: bool = False

    def bit(self, elix_id: str) -> int:
        return 1 << self.ids.index(elix_id)


def cargar(
    ruta_mapeo: Path | str | None = None, ruta_categorias: Path | str | None = None, solo_3c: bool = False
) -> Elixhauser:
    """Carga el mapeo. Con ``solo_3c`` se conservan únicamente los prefijos de tres caracteres.

    Esa versión reproduce el mapeo actual del repositorio, que compara tres caracteres y por eso no reconoce las categorías
    definidas con cuatro (diabetes, entre otras).
    """
    ruta_mapeo = Path(ruta_mapeo) if ruta_mapeo else rutas.DIR_CONFIG / "elixhauser_quan2005.csv"
    ruta_categorias = Path(ruta_categorias) if ruta_categorias else rutas.DIR_CONFIG / "elixhauser_categorias.csv"
    cat = config.leer_csv_texto(ruta_categorias).sort(pl.col("orden").cast(pl.Int64))
    ids = tuple(cat["elix_id"].to_list())
    if ids != IDS:
        raise ValueError("elixhauser_categorias.csv debe listar ELIX_01 a ELIX_31 en orden")
    mapeo = config.leer_csv_texto(ruta_mapeo)
    prefijos: dict[str, int] = {}
    for fila in mapeo.iter_rows(named=True):
        p = fila["patron"].strip().upper()
        if solo_3c and len(p) != 3:
            continue
        prefijos[p] = prefijos.get(p, 0) | (1 << ids.index(fila["elix_id"]))
    return Elixhauser(
        ids=ids,
        nombres_es=dict(zip(cat["elix_id"], cat["nombre_es"])),
        nombres_en=dict(zip(cat["elix_id"], cat["nombre_en"])),
        pesos={r["elix_id"]: int(r["peso_van_walraven"]) for r in cat.iter_rows(named=True)},
        domina={r["elix_id"]: r["domina_a"] for r in cat.iter_rows(named=True) if r["domina_a"]},
        sospechosas=frozenset(r["elix_id"] for r in cat.iter_rows(named=True) if r["sospechosa_fase1"] == "si"),
        prefijos=prefijos,
        largos=tuple(sorted({len(p) for p in prefijos})),
        solo_3c=solo_3c,
    )


def normalizar(codigo: str) -> str:
    return codigo.strip().upper().replace(".", "")


def mascara_codigo(codigo: str, e: Elixhauser) -> int:
    """Máscara de categorías de un código CIE-10 (cualquier formato)."""
    n = normalizar(codigo)
    m = 0
    for largo in e.largos:
        if len(n) >= largo:
            m |= e.prefijos.get(n[:largo], 0)
    return m


def categorias_de(codigo: str, e: Elixhauser, jerarquia: bool = False) -> set[str]:
    """Categorías a las que pertenece un código. Con ``jerarquia`` se aplica la dominancia entre categorías."""
    m = mascara_codigo(codigo, e)
    ids = {i for k, i in enumerate(e.ids) if m & (1 << k)}
    return aplicar_jerarquia_ids(ids, e) if jerarquia else ids


def aplicar_jerarquia_ids(ids: set[str], e: Elixhauser) -> set[str]:
    out = set(ids)
    for dominante, dominada in e.domina.items():
        if dominante in out:
            out.discard(dominada)
    return out


def mascaras_de_codigos(codigos: Iterable[str], e: Elixhauser) -> dict[str, int]:
    """Máscara de cada código distinto (ya normalizado). Solo se guardan los que pertenecen a alguna categoría."""
    out = {}
    for c in codigos:
        m = mascara_codigo(c, e)
        if m:
            out[c] = m
    return out


LOTE_FILAS = 250_000


def _mascara_lote(df: pl.DataFrame, columnas: list[str], e: Elixhauser) -> pl.Series:
    norm = df.select([esquema.normalizar_cie10(pl.col(c)).alias(c) for c in columnas])
    distintos = pl.concat([norm[c] for c in columnas]).drop_nulls().unique().to_list()
    mapa = mascaras_de_codigos(distintos, e)
    if not mapa:
        return pl.Series("mascara", [0] * df.height, dtype=pl.Int64)
    partes = [norm[c].replace_strict(mapa, default=0, return_dtype=pl.Int64).fill_null(0) for c in columnas]
    return reduce(lambda a, b: a | b, partes).alias("mascara")


def mascara_por_episodio(df: pl.DataFrame, columnas: Sequence[str], e: Elixhauser, lote: int = LOTE_FILAS) -> pl.Series:
    """Máscara de bits por fila: unión de las categorías de todas las columnas de diagnóstico indicadas.

    Primero se mapean los códigos DISTINTOS y luego se une a las filas; es mucho más barato que mapear fila por fila.
    Se procesa por lotes de filas para acotar la memoria con los datos completos.
    """
    columnas = list(columnas)
    if not columnas or df.height == 0:
        return pl.Series("mascara", [0] * df.height, dtype=pl.Int64)
    partes = [_mascara_lote(df.slice(ini, lote), columnas, e) for ini in range(0, df.height, lote)]
    return (partes[0] if len(partes) == 1 else pl.concat(partes)).alias("mascara")


def flags_desde_mascara(mascara: pl.Series, e: Elixhauser, jerarquia: bool = True) -> pl.DataFrame:
    """Tabla con ELIX_01 a ELIX_31 (0/1), N_COMORB_ELIX y los puntajes de van Walraven (completo y sin las cinco sospechosas)."""
    cols = {i: ((mascara // (1 << k)) % 2).cast(pl.Int8).alias(i) for k, i in enumerate(e.ids)}
    d = pl.DataFrame(list(cols.values()))
    if jerarquia:
        for dominante, dominada in e.domina.items():
            d = d.with_columns(pl.when(pl.col(dominante) == 1).then(0).otherwise(pl.col(dominada)).cast(pl.Int8).alias(dominada))
    score = reduce(lambda a, b: a + b, [pl.col(i).cast(pl.Int32) * e.pesos[i] for i in e.ids])
    score5 = reduce(lambda a, b: a + b, [pl.col(i).cast(pl.Int32) * e.pesos[i] for i in e.ids if i not in e.sospechosas])
    return d.with_columns(
        pl.sum_horizontal([pl.col(i).cast(pl.Int32) for i in e.ids]).alias("N_COMORB_ELIX"),
        score.alias("SCORE_VANWALRAVEN"),
        score5.alias("PUNTAJE_VAN_WALRAVEN_SIN_5"),
    )


def calcular_flags(df: pl.DataFrame, columnas: Sequence[str], e: Elixhauser, jerarquia: bool = True) -> pl.DataFrame:
    """Conveniencia: máscara por episodio y tabla de indicadores."""
    return flags_desde_mascara(mascara_por_episodio(df, columnas, e), e, jerarquia)
