"""Catálogo de hospitales por año (E6). Solo códigos y agregados: ningún registro de paciente y ningún nombre de hospital (decisión 10).

Las celdas de conteo de 1 a 9 casos se suprimen (privacidad). El volumen mínimo de la decisión 6 (500 egresos y 10 muertes esperadas) se marca
solo como información: las muertes esperadas dependen del modelo (Fase 4), así que aquí se usa la cifra OBSERVADA de 2023 como aproximación.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from .config import ANIOS, ANIOS_RESULTADO


def suprimir(serie: pl.Expr, minimo: int = 10) -> pl.Expr:
    """Texto: el conteo, o «<10» si está entre 1 y 9."""
    return pl.when((serie > 0) & (serie < minimo)).then(pl.lit(f"<{minimo}")).otherwise(serie.cast(pl.String))


def construir_catalogo(anios: list[int], dir_plata: Path | str, dir_cohortes: Path | str, dir_oro: Path | str, ruta_hospitales: Path | str | None,
                       minimo_celda: int = 10) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Devuelve (catálogo hospital-año con conteos suprimidos, conciliación por año con el catálogo oficial)."""
    dir_plata, dir_cohortes, dir_oro = Path(dir_plata), Path(dir_cohortes), Path(dir_oro)
    filas = []
    for a in anios:
        p = pl.read_parquet(dir_plata / f"plata_{a}.parquet", columns=["COD_HOSPITAL"])
        c = pl.read_parquet(dir_cohortes / f"cohorte_{a}.parquet")
        p = p.with_columns(c["C"].alias("C"))
        agg = p.group_by("COD_HOSPITAL").agg(pl.len().cast(pl.Int64).alias("n_egresos"), pl.col("C").sum().cast(pl.Int64).alias("n_cohorte_c"))
        if a in ANIOS_RESULTADO:
            d = pl.read_parquet(dir_plata / f"desenlace_{a}.parquet", columns=["ROL"])
            p = p.with_columns(c["D"].alias("D"), (d["ROL"] == "EVENTO").alias("F"))
            agg = agg.join(p.group_by("COD_HOSPITAL").agg(pl.col("D").sum().cast(pl.Int64).alias("n_cohorte_d"), (pl.col("D") & pl.col("F")).sum().cast(pl.Int64).alias("n_fallecidos_d")), on="COD_HOSPITAL")
        else:
            agg = agg.with_columns(pl.lit(None, dtype=pl.Int64).alias("n_cohorte_d"), pl.lit(None, dtype=pl.Int64).alias("n_fallecidos_d"))
        filas.append(agg.with_columns(pl.lit(a).alias("anio")))
    cat = pl.concat(filas).select("anio", pl.col("COD_HOSPITAL").alias("cod_hospital"), "n_egresos", "n_cohorte_c", "n_cohorte_d", "n_fallecidos_d").sort(["anio", "cod_hospital"])
    oficial: set[str] = set()
    if ruta_hospitales and Path(ruta_hospitales).exists():
        h = pl.read_csv(ruta_hospitales, infer_schema=False, encoding="utf8")
        oficial = set(h[h.columns[0]].str.strip_chars().to_list())
    cat = cat.with_columns(
        pl.col("cod_hospital").is_in(sorted(oficial)).alias("en_catalogo_oficial"),
        ((pl.col("n_cohorte_d") >= 500) & (pl.col("n_fallecidos_d") >= 10)).alias("cumple_volumen_observado"),
    )
    concil = cat.group_by("anio").agg(
        pl.len().alias("hospitales_con_egresos"), pl.col("en_catalogo_oficial").sum().alias("en_catalogo_oficial"),
        (~pl.col("en_catalogo_oficial")).sum().alias("fuera_del_catalogo_oficial"),
        pl.col("cumple_volumen_observado").sum().alias("cumplen_volumen_observado"),
    ).sort("anio")
    cat_pub = cat.select(
        "anio", "cod_hospital", suprimir(pl.col("n_egresos"), minimo_celda).alias("n_egresos"), suprimir(pl.col("n_cohorte_c"), minimo_celda).alias("n_cohorte_c"),
        suprimir(pl.col("n_cohorte_d"), minimo_celda).alias("n_cohorte_d"), suprimir(pl.col("n_fallecidos_d"), minimo_celda).alias("n_fallecidos_d"),
        "en_catalogo_oficial", "cumple_volumen_observado",
    )
    return cat_pub, concil


def script_sql(cat: pl.DataFrame) -> str:
    """Script PostgreSQL del catálogo (tabla hospital-año). Solo agregados; no se ejecutó contra ninguna base en esta corrida."""
    filas = []
    for r in cat.iter_rows(named=True):
        def v(x):
            return "NULL" if x is None or x.startswith("<") else x
        filas.append("({anio}, '{cod}', {e}, {c}, {d}, {f}, {o}, {m})".format(
            anio=r["anio"], cod=r["cod_hospital"].replace("'", ""), e=v(r["n_egresos"]), c=v(r["n_cohorte_c"]), d=v(r["n_cohorte_d"]), f=v(r["n_fallecidos_d"]),
            o=str(bool(r["en_catalogo_oficial"])).upper(), m="NULL" if r["cumple_volumen_observado"] is None else str(bool(r["cumple_volumen_observado"])).upper()))
    cuerpo = ",\n".join(filas)
    return (
        "-- Catálogo de hospitales por año (Fase 3). Solo códigos y agregados; sin datos de pacientes ni nombres.\n"
        "-- Las celdas de 1 a 9 casos están suprimidas (NULL).\n"
        "CREATE TABLE IF NOT EXISTS catalogo_hospital_anio (\n  anio INTEGER NOT NULL,\n  cod_hospital TEXT NOT NULL,\n  n_egresos INTEGER,\n  n_cohorte_c INTEGER,\n"
        "  n_cohorte_d INTEGER,\n  n_fallecidos_d INTEGER,\n  en_catalogo_oficial BOOLEAN NOT NULL,\n  cumple_volumen_observado BOOLEAN,\n  PRIMARY KEY (anio, cod_hospital)\n);\n"
        "INSERT INTO catalogo_hospital_anio VALUES\n" + cuerpo + "\nON CONFLICT DO NOTHING;\n"
    )
