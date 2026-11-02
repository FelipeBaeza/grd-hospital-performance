"""Segunda revisión técnica del profesor guía (02-11-2026, sobre la Fase 2 v7): análisis nuevos sin modelo (grupo B).

Como R1 a R5, son análisis **posteriores**, descriptivos, que no cambian ninguna decisión ya ejecutada, y siguen las reglas de la suite:

- el desenlace (tipo y fecha de alta) solo se lee en 2019 a 2023; 2024 se lee solo con propósito ``estructura``;
- ninguna celda de 1 a 9 casos se publica (supresión primaria y complementaria en ``Contexto.escribir_tabla``);
- no se publica ningún código de hospital: las cifras por hospital se resumen como distribución (percentiles entre hospitales) y por tipo de
  hospital del catálogo DEIS 2025.

Tablas ``s*``:

- S1: composición de EX06 (censurados) por tipo de alta y año.
- S2: proporción de censurados y de hospitalización domiciliaria por hospital (distribución entre hospitales y por tipo DEIS).
- S3: las definiciones de traslado entrante (TIPO_PROCEDENCIA con «OTROS HOSPITALES», PROCEDENCIA_AGR y HOSPPROCEDENCIA informado), su
  cruce y los traslados salientes.
- S4: tipo de ingreso por año (proporciones; los conteos van redondeados a la centena) y su cruce con MDC 14.
- S5: COVID-19 identificado por U07 en cualquier posición de diagnóstico: mortalidad y estadía por año, capítulo respiratorio con U07
  secundario y capítulo XV con su intervalo.
- S6: filas de identificadores con más de una fecha de nacimiento (Q13) o más de un sexo (Q14), por MDC y en la cohorte D.
- S7: filas duplicadas por clave natural (Q15) en la cohorte D y su concentración por hospital.
- S8: pacientes y días sobre el tope de estadía (p99) por hospital.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from . import catalogos, cohorte, elixhauser, esquema
from . import transformaciones as tr
from .contexto import Contexto
from .revision_profesor import ORDEN_GRUPOS, cargar_deis, leer_anio, tope_p99

TAREA = "S"
VOLUMEN_MINIMO = 500   # hospitales que entran en las distribuciones (decisión 6)
CLAVE_Q15 = ("CIP_ENCRIPTADO", "COD_HOSPITAL", "FECHA_INGRESO", "FECHAALTA", "DIAGNOSTICO1", "FECHA_NACIMIENTO", "SEXO")
ETIQUETA_ALTA = {
    "HOSP_DOMICILIARIA": "Hospitalización domiciliaria", "DERIV_HOSP_SERVICIO": "Derivación a otro hospital del servicio",
    "DERIV_HOSP_RED": "Derivación a otro hospital de la red nacional", "DERIV_OTROS_CENTROS": "Derivación a otros centros (cárcel, hogar)",
    "DERIV_PRIVADA_COMPRA": "Derivación a institución privada (compra)", "DERIV_PRIVADA_VOLUNTARIO": "Derivación a institución privada (voluntaria)",
    "NO_IDENTIFICADA": "No identificada", "DESCONOCIDO": "Desconocido",
}
OTRO_ALTA = "Otras o sin dato"
REGIMENES = ((2019, 2020), (2021, 2022, 2023, 2024))


def _grupo_mdc(mdc: pl.Expr, estado: pl.Expr) -> pl.Expr:
    return (pl.when(estado != "agrupable").then(pl.lit("Otras MDC o GRD no agrupable")).when(mdc == "14").then(pl.lit("MDC 14 (embarazo y parto)"))
            .when(mdc == "15").then(pl.lit("MDC 15 (recién nacidos)")).otherwise(pl.lit("Otras MDC o GRD no agrupable")))


# ------------------------------------------------------------------------------------------------------------------------------- lectura
def _extras(ctx: Contexto, anio: int, con_res: bool) -> pl.DataFrame:
    """Columnas crudas que ``leer_anio`` no trae: identificador, nacimiento, hospital de procedencia y (2019–2023) la clave de Q15."""
    cols = ["CIP_ENCRIPTADO", "FECHA_NACIMIENTO", "HOSPPROCEDENCIA"] + (list(CLAVE_Q15) if con_res else [])
    cols = list(dict.fromkeys(cols))
    df = ctx.lector.leer(anio, cols, proposito="resultado" if con_res else "estructura", tarea=TAREA)
    idc = pl.col("CIP_ENCRIPTADO")
    numerico = (idc.is_not_null() & ~esquema.es_centinela(idc) & idc.str.strip_chars().str.contains(r"^\d+$")).fill_null(False)
    hp = pl.col("HOSPPROCEDENCIA").str.strip_chars()
    out = [numerico.alias("id_ok"), idc.alias("id"), tr.a_fecha(pl.col("FECHA_NACIMIENTO")).alias("nac_txt"),
           (hp.is_not_null() & (hp != "")).fill_null(False).alias("hospproc_informado"),
           (hp.is_not_null() & (hp != "") & esquema.es_centinela(hp)).fill_null(False).alias("hospproc_centinela")]
    if con_res:
        dup = pl.struct(list(CLAVE_Q15)).is_first_distinct().not_()
        out.append((dup & numerico).alias("q15_sobrante"))
    return df.select(out)


def _u07(ctx: Contexto, anio: int) -> pl.Series:
    dx = ctx.lector.leer(anio, list(esquema.DIAG_SEC), proposito="estructura", tarea=TAREA)
    s = dx.select(pl.any_horizontal([esquema.normalizar_cie10(pl.col(c)).str.starts_with("U07").fill_null(False) for c in dx.columns]).alias("u"))["u"]
    del dx
    return s


def recolectar(ctx: Contexto, cat: catalogos.Catalogos | None = None) -> dict[str, Any]:
    cat = cat or catalogos.cargar_catalogos()
    e, e3 = elixhauser.cargar(), elixhauser.cargar(solo_3c=True)
    anios_res = set(ctx.umbrales.particion["anios_resultado"])
    from .panel import agrupar_procedencia

    filas, ids = [], []
    for anio in ctx.fuente.anios():
        con_res = anio in anios_res
        b = leer_anio(ctx, anio, cat, e, e3)
        x = _extras(ctx, anio, con_res)
        b = b.hstack(x)
        del x
        b = b.with_columns(_u07(ctx, anio).alias("u07_sec"),
                           pl.Series("proc_agr", agrupar_procedencia(b["procedencia"]).to_list()),
                           _grupo_mdc(pl.col("mdc"), pl.col("grd_estado")).alias("grupo_mdc"))
        keep = ["anio", "hospital", "tipo_ingreso", "mdc", "grupo_mdc", "capitulo", "cat3", "C", "entrante", "proc_agr",
                "hospproc_informado", "hospproc_centinela", "u07_sec"]
        if con_res:
            keep += ["D", "E", "evento", "rol", "codigo_alta", "saliente", "hosp_domiciliaria", "estancia", "q15_sobrante"]
        cat_cols = [c for c in ("hospital", "tipo_ingreso", "mdc", "grupo_mdc", "capitulo", "cat3", "proc_agr", "rol", "codigo_alta") if c in keep]
        filas.append(b.select(keep).with_columns([pl.col(c).cast(pl.Categorical) for c in cat_cols]))
        ids.append(b.filter(pl.col("id_ok")).select("anio", "id", "nac_txt", pl.col("sexo").cast(pl.Categorical),
                                                     pl.col("grupo_mdc").cast(pl.Categorical),
                                                     (pl.col("D") if con_res else pl.lit(False)).alias("D")))
        del b
    return {"b": pl.concat(filas, how="diagonal_relaxed"), "ids": pl.concat(ids)}


# --------------------------------------------------------------------------------------------------------------------------------- tablas
def _periodos(d: pl.DataFrame, anios: list[int], etiqueta: str) -> list[tuple[str, pl.DataFrame]]:
    return [(str(a), d.filter(pl.col("anio") == a)) for a in anios] + [(etiqueta, d.filter(pl.col("anio").is_in(anios)))]


def tabla_s1(b: pl.DataFrame, anios: list[int]) -> pl.DataFrame:
    c = b.filter(pl.col("C") & pl.col("anio").is_in(anios))
    cens = c.filter(~pl.col("D")).with_columns(pl.col("codigo_alta").cast(pl.String).replace_strict(ETIQUETA_ALTA, default=OTRO_ALTA).alias("tipo_alta"))
    filas = []
    for p, sub in _periodos(cens, anios, "2019-2023"):
        n_c = c.filter(pl.col("anio").is_in([int(p)] if p.isdigit() else anios)).height
        tot = sub.height
        for t, n in sub.group_by("tipo_alta").len().iter_rows():
            filas.append({"periodo": p, "tipo_alta": t, "n": int(n), "prop_ex06": n / tot, "prop_c": n / n_c, "n_ex06": tot, "n_c": n_c})
    orden = {v: i for i, v in enumerate([*ETIQUETA_ALTA.values(), OTRO_ALTA])}
    return pl.DataFrame(filas).with_columns(pl.col("tipo_alta").replace_strict(orden, return_dtype=pl.Int32).alias("_o")).sort(["periodo", "_o"]).drop("_o")


def _cuantiles(v: np.ndarray, pref: str) -> dict[str, float | None]:
    if v.size == 0:
        return {f"{pref}_{k}": None for k in ("p10", "p25", "mediana", "p75", "p90", "max")}
    q = np.percentile(v, [10, 25, 50, 75, 90, 100])
    return {f"{pref}_{k}": float(x) for k, x in zip(("p10", "p25", "mediana", "p75", "p90", "max"), q)}


def tabla_s2(b: pl.DataFrame, anios: list[int], deis: dict[str, str]) -> pl.DataFrame:
    c = b.filter(pl.col("C") & pl.col("anio").is_in(anios)).with_columns(
        pl.col("hospital").cast(pl.String).replace_strict(deis, default="Sin clasificar").alias("grupo"))
    h = c.group_by("hospital", "grupo").agg(pl.len().alias("n_c"), (~pl.col("D")).sum().alias("n_cens"), pl.col("hosp_domiciliaria").sum().alias("n_hd"))
    h = h.with_columns((pl.col("n_cens") / pl.col("n_c")).alias("p_cens"), (pl.col("n_hd") / pl.col("n_c")).alias("p_hd"),
                       (pl.col("n_hd") / pl.col("n_cens")).alias("p_hd_de_cens"))
    filas = []
    for g in ("Todos", *ORDEN_GRUPOS, "Sin clasificar"):
        sub = h if g == "Todos" else h.filter(pl.col("grupo") == g)
        if sub.is_empty():
            continue
        grandes = sub.filter(pl.col("n_c") >= VOLUMEN_MINIMO)
        f = {"grupo": g, "n_hospitales": sub.height, "n_hospitales_500": grandes.height, "n_c": int(sub["n_c"].sum()),
             "n_censurados": int(sub["n_cens"].sum()), "n_hosp_domiciliaria": int(sub["n_hd"].sum())}
        f["prop_censurados"] = f["n_censurados"] / f["n_c"]
        f["prop_hosp_domiciliaria"] = f["n_hosp_domiciliaria"] / f["n_c"]
        f["prop_hd_de_censurados"] = f["n_hosp_domiciliaria"] / f["n_censurados"] if f["n_censurados"] else None
        f |= _cuantiles(grandes["p_cens"].to_numpy(), "hosp_prop_censurados")
        f |= _cuantiles(grandes["p_hd"].to_numpy(), "hosp_prop_hd")
        f["hospitales_censura_sobre_10pct"] = int((grandes["p_cens"] > 0.10).sum())
        f["hospitales_censura_sobre_20pct"] = int((grandes["p_cens"] > 0.20).sum())
        alta = grandes.filter(pl.col("p_cens") > 0.20)
        f["mediana_prop_hd_de_censura_en_esos"] = float(alta["p_hd_de_cens"].median()) if alta.height else None
        f["hospitales_con_hd"] = int((grandes["n_hd"] > 0).sum())
        filas.append(f)
    return pl.DataFrame(filas)


def tabla_s3(b: pl.DataFrame, anios_res: list[int]) -> pl.DataFrame:
    t, h = pl.col("entrante"), pl.col("hospproc_informado")
    agr = (pl.col("proc_agr").cast(pl.String) == "DERIVADO_OTRO_HOSPITAL").fill_null(False)
    aggs = [pl.len().alias("n_total"), t.sum().alias("n_entrante_tipo"), h.sum().alias("n_hospproc"), (t & h).sum().alias("n_ambas"),
            (t & ~h).sum().alias("n_solo_tipo"), (~t & h).sum().alias("n_solo_hospproc"), (~t & ~h).sum().alias("n_ninguna"),
            pl.col("hospproc_centinela").sum().alias("n_hospproc_centinela"), agr.sum().alias("n_agr_derivado"),
            (agr & ~t).sum().alias("n_agr_no_tipo"), (t & ~agr).sum().alias("n_tipo_no_agr"), pl.col("saliente").fill_null(False).sum().alias("n_salientes")]
    partes = []
    for universo, filtro in (("todos los egresos", pl.lit(True)), ("cohorte C", pl.col("C"))):
        g = b.lazy().filter(filtro).group_by("anio").agg(aggs).collect().sort("anio")
        tot = g.filter(pl.col("anio").is_in(anios_res)).drop("anio").sum().with_columns(pl.lit("2019-2023").alias("periodo"))
        g = g.with_columns(pl.col("anio").cast(pl.String).alias("periodo")).drop("anio")
        partes.append(pl.concat([g, tot.select(g.columns)], how="vertical_relaxed").with_columns(pl.lit(universo).alias("universo")))
    out = pl.concat(partes).with_columns(
        pl.when(pl.col("periodo") == "2024").then(None).otherwise(pl.col("n_salientes")).alias("n_salientes"),
        (pl.col("n_ambas") / pl.col("n_hospproc")).alias("prop_hospproc_con_tipo"),
        (pl.col("n_ambas") / pl.col("n_entrante_tipo")).alias("prop_tipo_con_hospproc"))
    return out.select("universo", "periodo", *[c for c in out.columns if c not in ("universo", "periodo")])


def tabla_s4(b: pl.DataFrame) -> pl.DataFrame:
    """Proporciones por año y conteos redondeados a la centena (los tipos raros, de 1 a 9 casos, no se pueden despejar)."""
    principales = ("URGENCIA", "PROGRAMADA", "OBSTETRICA")
    ti = pl.col("tipo_ingreso").cast(pl.String)
    g = (b.lazy().select("anio", pl.when(ti.is_in(principales)).then(ti).otherwise(pl.lit("OTRA O SIN DATO")).alias("ti"),
                         (pl.col("mdc").cast(pl.String) == "14").fill_null(False).alias("m14"))
         .group_by("anio", "ti").agg(pl.len().alias("n"), pl.col("m14").sum().alias("n14")).collect())
    filas = []
    for anio in sorted(g["anio"].unique().to_list()):
        s = g.filter(pl.col("anio") == anio)
        tot, tot14 = int(s["n"].sum()), int(s["n14"].sum())
        d = {r["ti"]: (r["n"], r["n14"]) for r in s.iter_rows(named=True)}
        for t in (*principales, "OTRA O SIN DATO"):
            n, n14 = d.get(t, (0, 0))
            filas.append({"anio": anio, "tipo_ingreso": t, "centenas": round(n, -2) if n >= 100 else None,
                          "prop": round(n / tot, 4), "prop_de_mdc14": round(n14 / tot14, 4) if tot14 else None,
                          "prop_del_tipo_en_mdc14": round(n14 / n, 4) if n >= 100 else None})
    return pl.DataFrame(filas)


def tabla_s5(b: pl.DataFrame, anios: list[int], z: float, tope: int) -> pl.DataFrame:
    d = b.filter(pl.col("anio").is_in(anios) & pl.col("D")).select("anio", "cat3", "capitulo", "u07_sec", "D", "E", "evento", "estancia")
    u07p = pl.col("cat3") == "U07"
    grupos = (
        ("todos", "Todos los egresos", pl.lit(True)),
        ("u07_principal", "U07 como diagnóstico principal", u07p),
        ("u07_solo_secundario", "U07 solo en un diagnóstico secundario", ~u07p & pl.col("u07_sec")),
        ("sin_u07", "Sin U07 en ninguna posición", ~u07p & ~pl.col("u07_sec")),
        ("respiratorio", "Capítulo X (respiratorio)", pl.col("capitulo") == "X"),
        ("respiratorio_con_u07", "Capítulo X con U07 secundario", (pl.col("capitulo") == "X") & pl.col("u07_sec")),
        ("respiratorio_sin_u07", "Capítulo X sin U07", (pl.col("capitulo") == "X") & ~pl.col("u07_sec")),
        ("embarazo_xv", "Capítulo XV (embarazo, parto y puerperio)", pl.col("capitulo") == "XV"),
    )
    filas = []
    for g, et, f in grupos:
        f = f.fill_null(False)
        for anio in anios:
            s = d.filter((pl.col("anio") == anio) & f)
            sd, se = s.filter(pl.col("D")), s.filter(pl.col("E"))
            filas.append({"anio": anio, "grupo": g, "etiqueta": et, "n_d": sd.height, "n_fallecidos": int(sd["evento"].sum()),
                          "n_e": se.height, "estancia_media_tope": float(se["estancia"].clip(upper_bound=tope).mean()) if se.height else None})
    t = pl.DataFrame(filas).with_columns((pl.col("n_fallecidos") / pl.col("n_d")).alias("mortalidad"))
    t = cohorte.con_wilson(t, "n_fallecidos", "n_d", "", z)
    tot = t.filter(pl.col("grupo") == "todos").select("anio", pl.col("n_d").alias("_total_d"), pl.col("n_fallecidos").alias("_total_k"), pl.col("n_e").alias("_total_e"))
    res = t.filter(pl.col("grupo") == "respiratorio").select("anio", pl.col("n_d").alias("_resp_d"), pl.col("n_fallecidos").alias("_resp_k"), pl.col("n_e").alias("_resp_e"))
    return t.join(tot, on="anio").join(res, on="anio").select(
        "anio", "grupo", "etiqueta", "n_d", "n_fallecidos", "mortalidad", "ic_bajo", "ic_alto", "n_e", "estancia_media_tope",
        "_total_d", "_total_k", "_total_e", "_resp_d", "_resp_k", "_resp_e")


GRUPOS_S5B = (
    ("todos", "Todos los egresos", "pl.lit(True)"),
    ("u07_principal", "COVID-19 como diagnóstico principal (U07)", 'pl.col("cat3") == "U07"'),
    ("sin_u07_principal", "Sin U07 como diagnóstico principal", '(pl.col("cat3") != "U07").fill_null(True)'),
    ("u07_cualquiera", "U07 en cualquier posición", '(pl.col("cat3") == "U07").fill_null(False) | pl.col("u07_sec")'),
    ("sin_u07", "Sin U07 en ninguna posición", '~((pl.col("cat3") == "U07").fill_null(False) | pl.col("u07_sec"))'),
    ("respiratorio", "Capítulo X (respiratorio)", 'pl.col("capitulo") == "X"'),
    ("respiratorio_con_u07", "Capítulo X con U07 secundario", '(pl.col("capitulo") == "X") & pl.col("u07_sec")'),
    ("respiratorio_sin_u07", "Capítulo X sin U07", '(pl.col("capitulo") == "X") & ~pl.col("u07_sec")'),
    ("embarazo_xv", "Capítulo XV (embarazo, parto y puerperio)", 'pl.col("capitulo") == "XV"'),
)


def tabla_s5b(b: pl.DataFrame, z: float) -> pl.DataFrame:
    """Mortalidad agrupada por período (2019, 2022 y 2023 frente a 2020–2021) en la cohorte D, con su intervalo de Wilson."""
    d = b.filter(pl.col("D")).select("anio", "cat3", "capitulo", "u07_sec", "evento")
    filas = []
    for g, et, expr in GRUPOS_S5B:
        f = eval(expr).fill_null(False)
        for per, anios in (("referencia (2019, 2022 y 2023)", (2019, 2022, 2023)), ("pandemia (2020–2021)", (2020, 2021))):
            s = d.filter(pl.col("anio").is_in(list(anios)) & f)
            filas.append({"grupo": g, "etiqueta": et, "periodo": per, "n_d": s.height, "n_fallecidos": int(s["evento"].sum())})
    t = pl.DataFrame(filas).with_columns((pl.col("n_fallecidos") / pl.col("n_d")).alias("mortalidad"))
    return cohorte.con_wilson(t, "n_fallecidos", "n_d", "", z)


def razones_s5(t: pl.DataFrame) -> pl.DataFrame:
    """Razón entre el promedio de 2020–2021 y el de 2019, 2022 y 2023 (mortalidad y estadía), como en R1."""
    filas = []
    for g in t["grupo"].unique(maintain_order=True).to_list():
        s = t.filter(pl.col("grupo") == g)
        def m(col: str, anios: tuple[int, ...]) -> float | None:
            v = s.filter(pl.col("anio").is_in(list(anios)))[col].drop_nulls()
            return float(v.mean()) if v.len() == len(anios) else None
        r = {"grupo": g, "etiqueta": s["etiqueta"][0]}
        for col, nom in (("mortalidad", "mortalidad"), ("estancia_media_tope", "estancia")):
            a, ref = m(col, (2020, 2021)), m(col, (2019, 2022, 2023))
            r[f"{nom}_pandemia"], r[f"{nom}_referencia"] = a, ref
            r[f"razon_{nom}"] = a / ref if a is not None and ref else None
        filas.append(r)
    return pl.DataFrame(filas)


def tabla_s6(ids: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, tuple[int, int]]]:
    filas, control = [], {}
    sexo_ok = pl.col("sexo").is_in(["MUJER", "HOMBRE"])
    for reg in REGIMENES:
        f = ids.filter(pl.col("anio").is_in(list(reg)))
        etiqueta = f"régimen {reg[0]} a {reg[-1]}"
        g = f.group_by("id").agg(
            pl.len().alias("n_ep"), pl.col("nac_txt").drop_nulls().n_unique().alias("n_nac"),
            pl.col("sexo").filter(sexo_ok).n_unique().alias("n_sexo"), sexo_ok.sum().alias("n_sexo_ok"))
        q13 = g.filter((pl.col("n_ep") >= 2) & (pl.col("n_nac") > 1))["id"]
        q14 = g.filter((pl.col("n_sexo_ok") >= 2) & (pl.col("n_sexo") > 1))["id"]
        control[etiqueta] = (q13.len(), q14.len())
        f = f.with_columns(pl.col("id").is_in(q13.implode()).alias("q13"), pl.col("id").is_in(q14.implode()).alias("q14"))
        for grupo, sub in [*((m, f.filter(pl.col("grupo_mdc") == m)) for m in ("MDC 14 (embarazo y parto)", "MDC 15 (recién nacidos)", "Otras MDC o GRD no agrupable")),
                           ("Todas las filas", f), ("Cohorte D (2019 a 2023)", f.filter(pl.col("D")))]:
            n = sub.height
            filas.append({"regimen": etiqueta, "grupo": grupo, "n_filas": n, "n_filas_q13": int(sub["q13"].sum()), "n_filas_q14": int(sub["q14"].sum()),
                          "prop_q13": sub["q13"].sum() / n if n else None, "prop_q14": sub["q14"].sum() / n if n else None})
        tq = f.filter(pl.col("q13"))
        for fila in filas:
            if fila["regimen"] == etiqueta and fila["grupo"].startswith("MDC"):
                fila["parte_de_filas_q13"] = float((tq["grupo_mdc"] == fila["grupo"]).sum() / tq.height) if tq.height else None
    return pl.DataFrame(filas), control


def tabla_s7(b: pl.DataFrame, anios: list[int]) -> pl.DataFrame:
    d = b.filter(pl.col("anio").is_in(anios)).select("anio", "hospital", "q15_sobrante", "C", "D", "evento")
    filas = []
    for p, sub in _periodos(d, anios, "2019-2023"):
        dup = sub.filter(pl.col("q15_sobrante"))
        dupd = dup.filter(pl.col("D"))
        h = dupd.group_by("hospital").len().sort("len", descending=True)["len"].to_numpy()
        filas.append({"periodo": p, "n_sobrantes": dup.height, "n_sobrantes_c": int(dup["C"].sum()), "n_sobrantes_d": dupd.height,
                      "n_d": int(sub["D"].sum()), "prop_d": dupd.height / max(int(sub["D"].sum()), 1),
                      "hospitales_con_sobrantes_d": int(h.size), "parte_hospital_mayor": float(h[0] / h.sum()) if h.size else None,
                      "parte_5_hospitales_mayores": float(h[:5].sum() / h.sum()) if h.size else None,
                      "parte_fallecidos_en_sobrantes_d": float(dupd["evento"].mean()) if dupd.height else None})
    return pl.DataFrame(filas)


def tabla_s8(b: pl.DataFrame, anios: list[int], deis: dict[str, str], tope: int) -> pl.DataFrame:
    e = b.filter(pl.col("E") & pl.col("anio").is_in(anios)).with_columns(pl.col("hospital").cast(pl.String).replace_strict(deis, default="Sin clasificar").alias("grupo"))
    h = e.group_by("hospital", "grupo").agg(pl.len().alias("n_e"), (pl.col("estancia") > tope).sum().alias("n_sobre"),
                                            pl.col("estancia").sum().alias("dias"), (pl.col("estancia") - tope).clip(lower_bound=0).sum().alias("dias_sobre"))
    h = h.with_columns((pl.col("n_sobre") / pl.col("n_e")).alias("p_pac"), (pl.col("dias_sobre") / pl.col("dias")).alias("p_dias"))
    filas = []
    for g in ("Todos", *ORDEN_GRUPOS, "Sin clasificar"):
        sub = h if g == "Todos" else h.filter(pl.col("grupo") == g)
        if sub.is_empty():
            continue
        grandes = sub.filter(pl.col("n_e") >= VOLUMEN_MINIMO)
        f = {"grupo": g, "tope_dias": tope, "n_hospitales_500": grandes.height, "n_e": int(sub["n_e"].sum()), "n_sobre_tope": int(sub["n_sobre"].sum()),
             "dias_totales": int(sub["dias"].sum()), "dias_sobre_tope": int(sub["dias_sobre"].sum())}
        f["prop_pacientes"] = f["n_sobre_tope"] / f["n_e"]
        f["prop_dias"] = f["dias_sobre_tope"] / f["dias_totales"]
        f |= _cuantiles(grandes["p_pac"].to_numpy(), "hosp_prop_pacientes")
        f |= _cuantiles(grandes["p_dias"].to_numpy(), "hosp_prop_dias")
        filas.append(f)
    return pl.DataFrame(filas)


def tabla_s9(b: pl.DataFrame, tope: int) -> pl.DataFrame:
    """Índice de dispersión (varianza sobre media) de la estadía de la cohorte E, sin tope y con el tope p99 (observación 6)."""
    filas = []
    for p, anios in (("2019-2022", (2019, 2020, 2021, 2022)), ("2019-2023", (2019, 2020, 2021, 2022, 2023))):
        v = b.filter(pl.col("E") & pl.col("anio").is_in(list(anios)))["estancia"].drop_nulls().to_numpy().astype(float)
        for nombre, x in (("sin tope", v), (f"con tope de {tope} días", np.minimum(v, tope))):
            m, var = float(x.mean()), float(x.var(ddof=1))
            filas.append({"periodo": p, "estadia": nombre, "n": int(x.size), "media": m, "varianza": var, "indice_dispersion": var / m})
    return pl.DataFrame(filas)


# ----------------------------------------------------------------------------------------------------------------------------- ejecución
def ejecutar(ctx: Contexto, cat: catalogos.Catalogos | None = None, deis: dict[str, str] | None = None) -> dict[str, Any]:
    deis = cargar_deis() if deis is None else deis
    anios = [int(a) for a in ctx.umbrales.particion["anios_resultado"]]
    r = recolectar(ctx, cat)
    b = r["b"]
    est = b.filter(pl.col("E")).select("anio", "estancia")
    tope = tope_p99(est, ctx.umbrales.particion["anios_ranking"])
    z = float(ctx.umbrales["z_95"])
    w = ctx.escribir_tabla
    s1 = tabla_s1(b, anios)
    w("s1_ex06_composicion.csv", s1, tarea=TAREA, descripcion="Censurados (EX06) por tipo de alta, cohorte C", cols_n=("n",),
      dependientes=("prop_ex06", "prop_c"))
    w("s2_censura_hospital.csv", tabla_s2(b, anios, deis), tarea=TAREA, descripcion="Censura y hospitalización domiciliaria por hospital (distribución)",
      cols_n=("n_c", "n_censurados", "n_hosp_domiciliaria"), dependientes=("prop_censurados", "prop_hosp_domiciliaria", "prop_hd_de_censurados"))
    w("s3_traslados_definiciones.csv", tabla_s3(b, anios), tarea=TAREA, descripcion="Definiciones de traslado entrante y su cruce",
      cols_n=("n_total", "n_entrante_tipo", "n_hospproc", "n_ambas", "n_solo_tipo", "n_solo_hospproc", "n_ninguna", "n_hospproc_centinela",
              "n_agr_derivado", "n_agr_no_tipo", "n_tipo_no_agr", "n_salientes"),
      dependientes=("prop_hospproc_con_tipo", "prop_tipo_con_hospproc"))
    w("s4_tipo_ingreso.csv", tabla_s4(b), tarea=TAREA, descripcion="Tipo de ingreso por año (proporciones; conteos en centenas) y MDC 14")
    s5 = tabla_s5(b, anios, z, tope)
    w("s5_covid_u07.csv", s5, tarea=TAREA, descripcion="COVID-19 por U07 en cualquier posición: mortalidad (D) y estadía (E)",
      cols_n=("n_d", "n_fallecidos", "n_e"), dependientes=("mortalidad", "ic_bajo", "ic_alto", "estancia_media_tope"),
      auxiliares=("_total_d", "_total_k", "_total_e", "_resp_d", "_resp_k", "_resp_e"))
    publicada = pl.read_csv(ctx.dir_salida / "tablas" / "s5_covid_u07.csv")
    oculto = publicada.filter(pl.col("mortalidad").is_null() | pl.col("estancia_media_tope").is_null())
    s5r = razones_s5(publicada.filter(~pl.col("grupo").is_in(oculto["grupo"].unique().to_list())))
    w("s5_covid_resumen.csv", s5r, tarea=TAREA, descripcion="Razón 2020–2021 sobre 2019, 2022 y 2023 con U07 en cualquier posición")
    w("s5b_covid_periodos.csv", tabla_s5b(b, z), tarea=TAREA, descripcion="Mortalidad por período con U07 en cualquier posición (cohorte D)",
      cols_n=("n_d", "n_fallecidos"), dependientes=("mortalidad", "ic_bajo", "ic_alto"))
    s6, control = tabla_s6(r["ids"])
    w("s6_id_inconsistente_mdc.csv", s6, tarea=TAREA, descripcion="Filas de identificadores con Q13 o Q14 por MDC y en la cohorte D",
      cols_n=("n_filas", "n_filas_q13", "n_filas_q14"), dependientes=("prop_q13", "prop_q14"))
    w("s7_duplicados_cohorte.csv", tabla_s7(b, anios), tarea=TAREA, descripcion="Filas duplicadas (Q15) en las cohortes C y D",
      cols_n=("n_sobrantes", "n_sobrantes_c", "n_sobrantes_d", "n_d"), dependientes=("prop_d", "parte_fallecidos_en_sobrantes_d"))
    w("s8_tope_estadia_hospital.csv", tabla_s8(b, anios, deis, tope), tarea=TAREA, descripcion="Pacientes y días sobre el tope de estadía por hospital",
      cols_n=("n_e", "n_sobre_tope"), dependientes=("prop_pacientes",))
    w("s9_dispersion_estadia.csv", tabla_s9(b, tope), tarea=TAREA, descripcion="Índice de dispersión de la estadía, sin tope y con tope",
      cols_n=("n",), dependientes=("media", "varianza", "indice_dispersion"))
    print("control Q13/Q14 (ids):", control, "tope:", tope)
    return {"control": control, "tope": tope}
