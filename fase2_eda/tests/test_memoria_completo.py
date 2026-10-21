"""Modo de baja memoria para la base completa: caché Parquet del lector y lecturas por bloques de columnas.

Lo que se prueba es que el camino de baja memoria da EXACTAMENTE los mismos datos y las mismas cifras que el camino original (texto completo en memoria):
- el Parquet de la caché tiene las mismas filas, columnas y valores que la lectura del texto, también con BOM, fechas DD-MM-AAAA, la columna 2 renombrada y
  líneas irregulares rechazadas;
- el enmascarado del año expuesto y el registro del cortafuegos funcionan igual;
- la caché se regenera si el archivo cambia y nunca deja un archivo a medias con el nombre definitivo;
- la descripción por bloques (Tarea 4) y el escaneo de RUT por bloques (Tarea 5) dan las mismas tablas.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest

from eda import calidad, descripcion, esquema, lector
from eda.contexto import Contexto
from eda.cortafuegos import Cortafuegos
from eda.lector import Fuente, Lector
from sintetico import crear_fuente_sintetica, escribir_archivo, generar_anio


@pytest.fixture(scope="module")
def fuente_con_irregulares(tmp_path_factory) -> Fuente:
    """Fuente sintética de seis años; 2020 trae dos líneas con un número de campos distinto del esperado."""
    raiz = tmp_path_factory.mktemp("memoria")
    f = crear_fuente_sintetica(raiz, n=600, semilla=11)
    df = generar_anio(2020, n=600, semilla=11)
    escribir_archivo(df, f.archivos[2020], lineas_extra=["campo|corto", "|".join(["x"] * 130)])
    return f


def _completo(f: Fuente) -> Fuente:
    return Fuente(etiqueta="completo", raiz=f.raiz, archivos=dict(f.archivos))


def test_carpeta_cache_por_defecto(fuente_con_irregulares, monkeypatch):
    monkeypatch.delenv("FASE2_CACHE_PARQUET", raising=False)
    assert lector.carpeta_cache(fuente_con_irregulares) is None  # el extracto no usa caché
    assert lector.carpeta_cache(_completo(fuente_con_irregulares)) == fuente_con_irregulares.raiz / "_cache_parquet_fase2"
    monkeypatch.setenv("FASE2_CACHE_PARQUET", "0")
    assert lector.carpeta_cache(_completo(fuente_con_irregulares)) is None
    monkeypatch.setenv("FASE2_CACHE_PARQUET", "/tmp/otra")
    assert lector.carpeta_cache(fuente_con_irregulares) == Path("/tmp/otra")


def test_parquet_igual_a_texto(fuente_con_irregulares, tmp_path):
    sin = Lector(fuente_con_irregulares, Cortafuegos(), cache=None)
    con = Lector(fuente_con_irregulares, Cortafuegos(), cache=tmp_path / "cache")
    for anio in fuente_con_irregulares.anios():
        a = sin.leer(anio, None, proposito="estructura")
        b = con.leer(anio, None, proposito="estructura")
        assert a.columns == b.columns == list(esquema.COLUMNAS)
        assert a.equals(b), anio
        assert con.contabilidad[anio] == sin.contabilidad[anio]
        cols = ["DIAGNOSTICO3", "CIP_ENCRIPTADO", "FECHA_INGRESO"]
        assert sin.leer(anio, cols, proposito="estructura").equals(con.leer(anio, cols, proposito="estructura"))
    assert con.escaneo(2020).rechazadas == 2


def test_enmascarado_y_registro_iguales(fuente_con_irregulares, tmp_path):
    cols_res = ["TIPOALTA", "FECHAALTA"]
    sin = Lector(fuente_con_irregulares, Cortafuegos(columnas_resultado=cols_res), cache=None)
    con = Lector(fuente_con_irregulares, Cortafuegos(columnas_resultado=cols_res), cache=tmp_path / "c")
    a = sin.leer(2024, ["COD_HOSPITAL", "TIPOALTA", "FECHAALTA"], proposito="estructura")
    b = con.leer(2024, ["COD_HOSPITAL", "TIPOALTA", "FECHAALTA"], proposito="estructura")
    assert a.equals(b)
    assert sin.cortafuegos.registro().drop("secuencia").equals(con.cortafuegos.registro().drop("secuencia"))


def test_cache_se_reutiliza_y_se_regenera(tmp_path):
    raiz = tmp_path / "datos"
    f = crear_fuente_sintetica(raiz, anios=(2019,), n=50, semilla=3)
    cache = tmp_path / "cache"
    l1 = Lector(f, Cortafuegos(), cache=cache)
    p1 = l1.parquet(2019)
    mtime = p1.stat().st_mtime_ns
    l2 = Lector(f, Cortafuegos(), cache=cache)
    assert l2.parquet(2019) == p1 and p1.stat().st_mtime_ns == mtime  # se reutiliza
    # el archivo cambia: otra huella, otro Parquet, con los datos nuevos
    escribir_archivo(generar_anio(2019, n=60, semilla=4), f.archivos[2019])
    l3 = Lector(f, Cortafuegos(), cache=cache)
    p3 = l3.parquet(2019)
    assert p3 != p1
    assert l3.leer(2019, ["COD_HOSPITAL"], proposito="estructura").height == 60
    assert not list(cache.glob("*.tmp"))


def test_parquet_danado_se_rehace(tmp_path):
    f = crear_fuente_sintetica(tmp_path / "d", anios=(2019,), n=40, semilla=5)
    cache = tmp_path / "cache"
    p = Lector(f, Cortafuegos(), cache=cache).parquet(2019)
    p.write_bytes(b"no es parquet")
    l = Lector(f, Cortafuegos(), cache=cache)
    assert l.leer(2019, ["COD_HOSPITAL"], proposito="estructura").height == 40


def test_leer_por_bloques_cubre_todas_las_columnas(fuente_con_irregulares, tmp_path):
    con = Lector(fuente_con_irregulares, Cortafuegos(), cache=tmp_path / "c")
    bloques = list(con.leer_por_bloques(2021, None, bloque=16, proposito="estructura"))
    assert [c for b in bloques for c in b.columns] == list(esquema.COLUMNAS)
    assert pl.concat(bloques, how="horizontal").equals(con.leer(2021, None, proposito="estructura"))


def test_descripcion_por_bloques_igual(fuente_con_irregulares, tmp_path):
    con = Lector(fuente_con_irregulares, Cortafuegos(columnas_resultado=["TIPOALTA", "FECHAALTA"]), cache=tmp_path / "c")
    for anio in (2019, 2024):
        prot = con.cortafuegos.columnas_protegidas(anio)
        acc_a, cont_a = descripcion.analizar_anio(con.leer(anio, None, proposito="estructura"), anio, prot)
        acc_b, cont_b = descripcion.analizar_anio_por_bloques(con, anio, prot, "T04")
        assert acc_a.perfil == acc_b.perfil
        assert acc_a.top == acc_b.top
        # value_counts no garantiza orden: la tarea ordena después, así que se compara como conjunto ordenado
        def orden(filas):
            return sorted(filas, key=lambda d: tuple(str(v) for v in d.values()))
        assert orden(acc_a.categorias) == orden(acc_b.categorias)
        assert orden(acc_a.hist_edad) == orden(acc_b.hist_edad) and orden(acc_a.hist_estancia) == orden(acc_b.hist_estancia)
        assert acc_a.pares == acc_b.pares
        assert cont_a == cont_b


def test_tareas_4_y_5_iguales_con_y_sin_cache(fuente_con_irregulares, tmp_path, monkeypatch):
    """Las tablas de las Tareas 4 y 5 son idénticas por el camino original y por el de baja memoria."""
    monkeypatch.delenv("FASE2_CACHE_PARQUET", raising=False)
    a = Contexto.crear("extracto", fuente=fuente_con_irregulares, dir_salida_base=tmp_path / "a")
    b = Contexto.crear("extracto", fuente=fuente_con_irregulares, dir_salida_base=tmp_path / "b")
    assert a.lector.cache is None
    b.lector.cache = tmp_path / "cache"
    for ctx in (a, b):
        descripcion.ejecutar(ctx)
        calidad.ejecutar(ctx)
    for nombre in sorted(p.name for p in a.dir_tablas.glob("*.csv")):
        assert (a.dir_tablas / nombre).read_bytes() == (b.dir_tablas / nombre).read_bytes(), nombre


def _solapes_ventanas_anidadas(frames: pl.DataFrame) -> pl.DataFrame:
    """Versión anterior de ``calidad.solapes`` (ventanas anidadas, casi cuadrática con millones de IDs): sirve de oráculo."""
    f = frames.filter(pl.col("ing").is_not_null() & pl.col("alta").is_not_null()).sort(["id", "ing", "alta"])
    f = f.with_columns(
        pl.col("alta").cum_max().over("id").shift(1).over("id").alias("_prev_max_alta"),
        ((pl.col("ing") == pl.col("ing").shift(1).over("id")) & (pl.col("alta") == pl.col("alta").shift(1).over("id"))).fill_null(False).alias("_dup"),
    )
    f = f.with_columns(((pl.col("ing") < pl.col("_prev_max_alta")) & ~pl.col("_dup")).fill_null(False).alias("_solapa"))
    return f.group_by("anio").agg(
        pl.col("_solapa").sum().alias("n_solapes"), pl.len().alias("n_evaluados"), pl.col("_dup").sum().alias("n_duplicados_exactos")
    ).sort("anio")


@pytest.mark.parametrize("semilla", [1, 2, 3])
def test_solapes_lineal_igual_a_la_version_anterior(semilla):
    import numpy as np

    rng = np.random.default_rng(semilla)
    n = 5000
    ing = np.datetime64("2019-01-01") + rng.integers(0, 900, n).astype("timedelta64[D]")
    alta = ing + rng.integers(0, 40, n).astype("timedelta64[D]")
    frames = pl.DataFrame({"id": rng.integers(0, 1500, n).astype(str), "ing": ing.astype("datetime64[D]"), "alta": alta.astype("datetime64[D]"),
                           "anio": rng.integers(2019, 2024, n)})
    frames = pl.concat([frames, frames.head(200)])  # duplicados exactos
    frames = frames.with_columns(pl.when(pl.int_range(pl.len()) % 97 == 0).then(None).otherwise(pl.col("alta")).alias("alta"))  # algunas altas nulas
    assert calidad.solapes(frames).equals(_solapes_ventanas_anidadas(frames))
