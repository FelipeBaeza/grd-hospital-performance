"""Capa Plata: datos limpios y estandarizados, con los tipos correctos y sin imputar.

Reglas:
- Una fecha o edad imposible se ANULA y el egreso se conserva (R5). Nada se imputa.
- Los diagnósticos se normalizan (mayúsculas, sin espacios ni puntos).
- Los traslados internos, los procedimientos y el bloque de recién nacidos no pasan a Plata como columnas: solo se cuenta cuántos hay.
- La severidad, el peso y el riesgo de mortalidad del GRD (fuga, R6) y los médicos encriptados no pasan a Plata.
- El desenlace (tipo de alta, fecha de alta, estadía) va a un archivo aparte; el de 2024 queda sellado.
- Garantía: cada fila de Bronce está en Plata (misma cantidad de filas, mismo orden).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from . import definiciones as dfn
from . import esquema
from . import transformaciones as tr
from .config import ANIO_SELLADO, Umbrales, cargar_umbrales
from . import sellado

COLS_LEER = [
    "ID_EPISODIO", "ANIO_ARCHIVO", "COD_HOSPITAL", "CIP_ENCRIPTADO", "SEXO", "FECHA_NACIMIENTO", "ETNIA", "PROVINCIA", "COMUNA", "NACIONALIDAD",
    "PREVISION", "SERVICIO_SALUD", "TIPO_PROCEDENCIA", "TIPO_INGRESO", "ESPECIALIDAD_MEDICA", "TIPO_ACTIVIDAD", "FECHA_INGRESO",
    "SERVICIOINGRESO", "HOSPPROCEDENCIA", "IR_29301_COD_GRD", *esquema.DIAG, *esquema.PROC, *esquema.FECHAS_TRASLADO,
    "FECHAALTA", "TIPOALTA",
]
COLS_FALTANTES = ["COD_HOSPITAL", "CIP_ENCRIPTADO", "SEXO", "FECHA_NACIMIENTO", "TIPO_PROCEDENCIA", "TIPO_INGRESO", "ESPECIALIDAD_MEDICA", "TIPO_ACTIVIDAD",
                  "FECHA_INGRESO", "SERVICIOINGRESO", "HOSPPROCEDENCIA", "IR_29301_COD_GRD", "DIAGNOSTICO1", "FECHAALTA", "TIPOALTA"]
SOLO_ESTRUCTURA_2024 = {"FECHAALTA", "TIPOALTA"}      # en el año sellado no se calcula nada sobre el desenlace


def _informada(c: str) -> pl.Expr:
    return pl.col(c).is_not_null() & (pl.col(c).str.strip_chars() != "")


def _faltante(c: str) -> pl.Expr:
    return (~_informada(c)) | esquema.es_centinela(pl.col(c))


def _contar_informadas(df: pl.DataFrame, columnas) -> pl.Series:
    return df.select(pl.sum_horizontal([_informada(c).cast(pl.Int32) for c in columnas]).cast(pl.Int32).alias("n"))["n"]


def mapa_roles(etiquetas: list[str], roles: pl.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    rol, canon = {}, {}
    for et in etiquetas:
        nombre, _, r = dfn.rol_de_etiqueta(et, roles)
        rol[et] = r or "SIN_DATO"
        canon[et] = nombre or "(sin rol)"
    return rol, canon


def transformar_lote(df: pl.DataFrame, anio: int, roles: pl.DataFrame, u: Umbrales) -> tuple[pl.DataFrame, pl.DataFrame, Counter, Counter]:
    """Un lote de Bronce -> (estructura, desenlace, faltantes, anulados). Función pura: no lee ni escribe archivos."""
    sellado_ = anio == ANIO_SELLADO
    nac = tr.a_fecha(pl.col("FECHA_NACIMIENTO"))
    ing = tr.a_fecha(pl.col("FECHA_INGRESO"))
    edad_b = tr.edad_anios(nac, ing)
    g = pl.col("IR_29301_COD_GRD").str.strip_chars()
    fmt6 = g.str.contains(r"^\d{6}$")
    idc = pl.col("CIP_ENCRIPTADO").str.strip_chars()
    id_valido = idc.str.contains(r"^\d+$") & (idc.str.len_chars() >= int(u["id_largo_minimo"]))
    base = df.select(
        "ID_EPISODIO",
        pl.col("ANIO_ARCHIVO").cast(pl.Int32).alias("ANIO_EGRESO"),
        pl.col("COD_HOSPITAL").str.strip_chars().fill_null(dfn.SIN_DATO).alias("COD_HOSPITAL"),
        pl.when(id_valido).then(idc.cast(pl.Int64, strict=False)).alias("ID_PACIENTE_NUM"),
        pl.when(pl.col("SEXO").is_in(["MUJER", "HOMBRE"])).then(pl.col("SEXO")).otherwise(pl.lit("SIN_DATO")).alias("SEXO"),
        nac.alias("FECHA_NACIMIENTO"),
        ing.alias("FECHA_INGRESO"),
        pl.when((edad_b >= u["edad_minima"]) & (edad_b <= u["edad_maxima"])).then(edad_b).cast(pl.Int32).alias("EDAD_ANIOS"),
        pl.col("TIPO_ACTIVIDAD").str.strip_chars().fill_null(dfn.SIN_DATO).alias("TIPO_ACTIVIDAD"),
        pl.when(pl.col("HOSPPROCEDENCIA").str.strip_chars() != "").then(pl.col("HOSPPROCEDENCIA").str.strip_chars()).alias("HOSPPROCEDENCIA"),
        g.alias("IR_29301_COD_GRD"),
        pl.when(g.is_null()).then(pl.lit("nulo"))
        .when(esquema.es_centinela(g)).then(pl.lit("centinela"))
        .when(~fmt6).then(pl.lit("formato_invalido"))
        .when(g.str.slice(0, 2) == "99").then(pl.lit("mdc_99"))
        .otherwise(pl.lit("agrupable")).alias("GRD_ESTADO"),
        pl.when(fmt6).then(g.str.slice(0, 2)).alias("MDC"),
        *[esquema.normalizar_cie10(pl.col(c)).alias(c) for c in esquema.DIAG],
    )
    cat = df.select([pl.col(c) for c in ("ETNIA", "PROVINCIA", "COMUNA", "NACIONALIDAD", "PREVISION", "SERVICIO_SALUD", "TIPO_PROCEDENCIA",
                                         "TIPO_INGRESO", "ESPECIALIDAD_MEDICA", "SERVICIOINGRESO")])
    extra = pl.DataFrame({c: dfn.limpiar_categorica(cat[c]) for c in cat.columns})
    cuentas = pl.DataFrame({
        "N_DX_SECUNDARIOS": _contar_informadas(df, esquema.DIAG_SEC),
        "N_PROCEDIMIENTOS": _contar_informadas(df, esquema.PROC),
        "N_TRASLADOS_INTERNOS": _contar_informadas(df, esquema.FECHAS_TRASLADO),
        "ES_TRASLADO_ENTRANTE": dfn.entrante_tipo_procedencia(df["TIPO_PROCEDENCIA"]).cast(pl.Int8),
    })
    estructura = pl.concat([base, extra, cuentas], how="horizontal")
    # ---- desenlace (aparte)
    alta = tr.a_fecha(pl.col("FECHAALTA"))
    etiquetas = df["TIPOALTA"].drop_nulls().unique().to_list()
    rol, canon = mapa_roles(etiquetas, roles) if etiquetas else ({}, {})
    est_b = tr.estancia_dias(ing, alta)
    des = df.select(
        "ID_EPISODIO",
        pl.col("TIPOALTA").replace_strict(rol, default="SIN_DATO", return_dtype=pl.String).fill_null("SIN_DATO").alias("ROL") if etiquetas else pl.lit("SIN_DATO").alias("ROL"),
        pl.col("TIPOALTA").replace_strict(canon, default="(sin rol)", return_dtype=pl.String).fill_null(dfn.SIN_DATO).alias("ETIQUETA_TIPOALTA") if etiquetas else pl.lit(dfn.SIN_DATO).alias("ETIQUETA_TIPOALTA"),
        alta.alias("FECHAALTA"),
        pl.when(est_b >= 0).then(est_b).cast(pl.Int32).alias("ESTANCIA_DIAS"),
    )
    etiquetas_salida = [l for l in roles["tipoalta"].to_list() if dfn.codigo_tipoalta(l) in dfn.CODIGOS_SALIDA_HOSPITAL]
    des = des.with_columns(pl.col("ETIQUETA_TIPOALTA").is_in(etiquetas_salida).cast(pl.Int8).alias("ES_TRASLADO_SALIENTE"))
    # ---- conteos de control
    falt: Counter = Counter()
    anul: Counter = Counter()
    for c in COLS_FALTANTES:
        if sellado_ and c in SOLO_ESTRUCTURA_2024:
            continue
        falt[c] = int(df.select(_faltante(c).sum()).item() or 0)
    anul["edad_fuera_de_rango"] = int(df.select(((edad_b < u["edad_minima"]) | (edad_b > u["edad_maxima"])).sum()).item() or 0)
    anul["fecha_nacimiento_ilegible"] = int(df.select((_informada("FECHA_NACIMIENTO") & nac.is_null()).sum()).item() or 0)
    anul["fecha_ingreso_ilegible"] = int(df.select((_informada("FECHA_INGRESO") & ing.is_null()).sum()).item() or 0)
    anul["id_paciente_atipico"] = int(df.select((_informada("CIP_ENCRIPTADO") & ~id_valido).sum()).item() or 0)
    if not sellado_:
        anul["fecha_alta_ilegible"] = int(df.select((_informada("FECHAALTA") & alta.is_null()).sum()).item() or 0)
        anul["estancia_negativa"] = int(df.select((est_b < 0).sum()).item() or 0)
    return estructura, des, falt, anul


def construir_plata(anio: int, dir_bronce: Path | str, dir_plata: Path | str, dir_sellado: Path | str, roles: pl.DataFrame,
                    u: Umbrales | None = None, lote: int | None = None) -> dict:
    u = u or cargar_umbrales()
    lote = int(lote or u["tamano_lote_filas"])
    dir_bronce, dir_plata, dir_sellado = Path(dir_bronce), Path(dir_plata), Path(dir_sellado)
    dir_plata.mkdir(parents=True, exist_ok=True)
    dir_sellado.mkdir(parents=True, exist_ok=True)
    ruta_b = dir_bronce / f"bronce_{anio}.parquet"
    destino_e = dir_plata / f"plata_{anio}.parquet"
    destino_d = (dir_sellado if anio == ANIO_SELLADO else dir_plata) / f"desenlace_{anio}.parquet"
    pf = pq.ParquetFile(ruta_b)
    esc_e = esc_d = None
    tmp_e, tmp_d = destino_e.with_name(destino_e.name + ".tmp"), destino_d.with_name(destino_d.name + ".tmp")
    falt: Counter = Counter()
    anul: Counter = Counter()
    n = 0
    try:
        for batch in pf.iter_batches(batch_size=lote, columns=COLS_LEER):
            df = pl.from_arrow(batch)
            est, des, f, a = transformar_lote(df, anio, roles, u)
            falt.update(f)
            anul.update(a)
            n += df.height
            te, td = est.to_arrow(), des.to_arrow()
            if esc_e is None:
                esc_e = pq.ParquetWriter(tmp_e, te.schema, compression="zstd")
                esc_d = pq.ParquetWriter(tmp_d, td.schema, compression="zstd")
            esc_e.write_table(te.cast(esc_e.schema))
            esc_d.write_table(td.cast(esc_d.schema))
        esc_e.close(); esc_d.close()
        tmp_e.replace(destino_e)
        tmp_d.replace(destino_d)
    finally:
        for t in (tmp_e, tmp_d):
            if t.exists():
                t.unlink()
    if anio == ANIO_SELLADO:
        sellado.registrar("escritura_sellada", anio, "el desenlace se escribió en el archivo sellado; no se calculó ningún agregado", True)
    # duplicados "mismo paciente y mismas fechas" (solo estructura: no usa la fecha de alta)
    dup = (
        pl.scan_parquet(destino_e).filter(pl.col("ID_PACIENTE_NUM").is_not_null() & pl.col("FECHA_INGRESO").is_not_null())
        .group_by(["COD_HOSPITAL", "ID_PACIENTE_NUM", "FECHA_INGRESO"]).len().filter(pl.col("len") > 1)
        .select((pl.col("len") - 1).sum().alias("n")).collect().item() or 0
    )
    filas_b = pq.ParquetFile(ruta_b).metadata.num_rows
    rep = {
        "anio": anio, "filas_bronce": filas_b, "filas_plata": n, "cuadra": filas_b == n == pq.ParquetFile(destino_e).metadata.num_rows,
        "columnas_plata": len(pq.ParquetFile(destino_e).schema_arrow.names), "faltantes": dict(falt), "anulados": dict(anul),
        "episodios_mismo_paciente_mismas_fechas": int(dup),
        "desenlace_sellado": anio == ANIO_SELLADO,
    }
    (dir_plata / f"reporte_plata_{anio}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf8")
    return rep
