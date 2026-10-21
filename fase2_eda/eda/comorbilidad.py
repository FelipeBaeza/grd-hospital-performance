"""CRISP-DM 2.3, parte 2: comorbilidad de Elixhauser y deriva de la codificación.

Las categorías se calculan con los diagnósticos secundarios (regla R3), sin mirar el desenlace: no se lee ninguna columna
de resultado, así que los seis años (incluido 2024) entran con propósito ``estructura``. La base principal son los
adultos hospitalizados (edad de ``edad_adulto`` años o más y actividad de hospitalización). Para poder contrastar con las
cifras documentadas, que provienen del mapeo del repositorio, también se cuentan todos los egresos y el mapeo de tres
caracteres.

Tablas:
- ``t23_elix_prevalencia``: prevalencia de las 31 categorías por año y período agrupado (adultos hospitalizados, mapeo completo).
- ``t23_elix_vs_3c``: mapeo completo frente al de tres caracteres, en adultos hospitalizados y en todos los egresos (supuesto S02).
- ``t23_elix_principal``: cuánto cambiaría cada categoría si se incluyera el diagnóstico principal (justifica la regla R3).
- ``t23_elix_phi``: coeficiente phi entre las 31 categorías (2019 a 2022 y 2023).
- ``t23_deriva_codificacion``: promedio de diagnósticos secundarios por año y hospital (supuesto S11).
Figuras: F08_01 a F08_03.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from . import catalogos, config, elixhauser, esquema, graficos
from . import transformaciones as tr
from .cohorte import ACTIVIDAD_HOSP, SIN_DATO, con_periodos, con_wilson, etiqueta_periodo
from .contexto import Contexto
from .informes import Informe, cifra

TAREA = "T08"
COLS = ("COD_HOSPITAL", "FECHA_NACIMIENTO", "FECHA_INGRESO", "TIPO_ACTIVIDAD", "DIAGNOSTICO1")
MIN_EPISODIOS_LINEA = 100  # un hospital se dibuja como línea solo si tiene al menos estos egresos en cada año (filtro de visualización)
TODOS = "TODOS"


# ------------------------------------------------------------------------------ lectura
def leer_anio(ctx: Contexto, anio: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(metadatos, diagnósticos secundarios) de todos los egresos de un año, en el mismo orden.

    Los metadatos traen ``hospital``, ``DIAGNOSTICO1`` y ``es_ah`` (adulto hospitalizado).
    """
    u = ctx.umbrales
    meta = ctx.lector.leer(anio, list(COLS), proposito="estructura", tarea=TAREA)
    edad_b = tr.edad_anios(tr.a_fecha(pl.col("FECHA_NACIMIENTO")), tr.a_fecha(pl.col("FECHA_INGRESO")))
    es_ah = (
        (edad_b >= int(u["edad_adulto"])) & (edad_b <= u["edad_maxima"]) & (pl.col("TIPO_ACTIVIDAD").str.strip_chars() == ACTIVIDAD_HOSP)
    ).fill_null(False)
    meta = meta.select(
        pl.col("COD_HOSPITAL").str.strip_chars().fill_null(SIN_DATO).alias("hospital"), pl.col("DIAGNOSTICO1"), es_ah.alias("es_ah")
    )
    dx = ctx.lector.leer(anio, list(esquema.DIAG_SEC), proposito="estructura", tarea=TAREA)
    if dx.height != meta.height:
        raise ValueError("los diagnósticos y los metadatos deben tener las mismas filas")
    return meta, dx


# ------------------------------------------------------------------- agregados de un año
def resumir_anio(meta: pl.DataFrame, dx: pl.DataFrame, e: elixhauser.Elixhauser, e3: elixhauser.Elixhauser) -> dict[str, Any]:
    """Conteos de un año: prevalencias (mapeo completo y de tres caracteres), diagnóstico principal, coocurrencia y codificación."""
    ids = list(e.ids)
    ah = meta["es_ah"]
    n_ah = int(ah.sum())
    sec = elixhauser.mascara_por_episodio(dx, esquema.DIAG_SEC, e)
    tres = elixhauser.mascara_por_episodio(dx, esquema.DIAG_SEC, e3)
    f_sec = elixhauser.flags_desde_mascara(sec, e)
    f_tres = elixhauser.flags_desde_mascara(tres, e3)

    def cuenta(f: pl.DataFrame, filtro: pl.Series | None = None) -> dict[str, int]:
        g = f if filtro is None else f.filter(filtro)
        return {i: int(g[i].sum()) for i in ids}

    meta_ah = meta.filter(ah)
    prin = elixhauser.mascara_por_episodio(meta_ah, ["DIAGNOSTICO1"], e)
    f_uni = elixhauser.flags_desde_mascara(sec.filter(ah) | prin, e)
    prin_bits = {i: int((((prin // (1 << k)) % 2) == 1).sum()) for k, i in enumerate(ids)}
    x = f_sec.filter(ah).select(ids).to_numpy().astype(np.float32)
    co = (x.T @ x).astype(np.int64) if n_ah else np.zeros((len(ids), len(ids)), dtype=np.int64)
    k_ah = dx.filter(ah).select(pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in esquema.DIAG_SEC]).alias("k"))["k"]
    por_hosp = (
        meta_ah.with_columns(k_ah.cast(pl.Int64).alias("k"))
        .group_by("hospital")
        .agg(pl.len().alias("n"), pl.col("k").sum().alias("suma"), (pl.col("k") ** 2).sum().alias("suma2"))
    )
    return {
        "n_todos": meta.height, "n": n_ah,
        "sec": cuenta(f_sec, ah), "tres": cuenta(f_tres, ah), "uni": cuenta(f_uni), "prin": prin_bits,
        "sec_todos": cuenta(f_sec), "tres_todos": cuenta(f_tres),
        "co": co, "suma": int(k_ah.cast(pl.Int64).sum() or 0), "suma2": int((k_ah.cast(pl.Int64) ** 2).sum() or 0), "hospitales": por_hosp,
    }


def largos_por_categoria(e: elixhauser.Elixhauser) -> dict[str, str]:
    """Largos de prefijo con los que se define cada categoría (3, 4 o ambos)."""
    largos: dict[str, set[int]] = {i: set() for i in e.ids}
    for prefijo, mascara in e.prefijos.items():
        for k, i in enumerate(e.ids):
            if mascara & (1 << k):
                largos[i].add(len(prefijo))
    return {i: " y ".join(str(x) for x in sorted(v)) for i, v in largos.items()}


# -------------------------------------------------------------------------------- tablas
def _frame_conteos(por_anio: dict[int, dict[str, Any]], clave: str, base: str, e: elixhauser.Elixhauser) -> pl.DataFrame:
    filas = [{"anio": a, "elix_id": i, "n": d[clave][i], "n_base": d[base]} for a, d in sorted(por_anio.items()) for i in e.ids]
    return pl.DataFrame(filas, schema={"anio": pl.Int64, "elix_id": pl.String, "n": pl.Int64, "n_base": pl.Int64})


def _periodos(anios: Sequence[int], u: config.Umbrales) -> dict[str, list[int]]:
    """Todos los años disponibles, el período de ranking y el de resultado, cuando están completos."""
    out: dict[str, list[int]] = {}
    if len(anios) > 1:
        out[etiqueta_periodo(anios)] = list(anios)
    for clave in ("anios_ranking", "anios_resultado"):
        lista = sorted(int(a) for a in u.particion[clave])
        if len(lista) > 1 and set(lista) <= set(anios):
            out[etiqueta_periodo(lista)] = lista
    return out


def tabla_prevalencia(por_anio: dict[int, dict[str, Any]], e: elixhauser.Elixhauser, u: config.Umbrales) -> pl.DataFrame:
    """Prevalencia por categoría, año y período agrupado en adultos hospitalizados con el mapeo completo."""
    # cada categoría tiene una fila por año con el mismo denominador anual: al sumar por categoría el denominador del período queda bien
    df = con_periodos(_frame_conteos(por_anio, "sec", "n", e), _periodos(sorted(por_anio), u), ["elix_id"], ["n", "n_base"])
    df = df.with_columns((pl.col("n") / pl.col("n_base")).alias("prevalencia"))
    df = con_wilson(df, "n", "n_base", "", float(u["z_95"]))
    # el rango desempata por categoría: con el orden de las filas (que sale de un group_by) los empates en n cambiaban de rango entre corridas
    df = df.sort(["periodo", "n", "elix_id"], descending=[False, True, False]).with_columns(
        pl.col("elix_id").replace_strict(e.nombres_es, default="", return_dtype=pl.String).alias("nombre_es"),
        pl.int_range(1, pl.len() + 1, dtype=pl.Int64).over("periodo").alias("rango"),
    )
    return df.select(["periodo", "elix_id", "nombre_es", "rango", "n", "n_base", "prevalencia", "ic_bajo", "ic_alto"]).sort(["periodo", "elix_id"])


def resumen_comorbilidad(r: dict[str, Any], u: config.Umbrales) -> pl.DataFrame:
    """Cifras derivadas: categorías que superan el umbral del mapeo de tres caracteres, pares redundantes y cambio entre hospitales."""
    v, ph, d = r["vs_3c"], r["phi"], r["deriva"]
    out: dict[str, Any] = {
        "n_categorias": len(r["e"].ids), "n_sobre_umbral_3c": int((v["supera_umbral"] == "si").sum()), "min_episodios_linea": MIN_EPISODIOS_LINEA,
    }
    dm = v.filter(pl.col("elix_id").is_in(["ELIX_11", "ELIX_12"]))
    for i in ("ELIX_11", "ELIX_12"):
        f = dm.filter(pl.col("elix_id") == i)
        out[f"prev_completo_{i}"] = f["prev_completo_ah"][0] if f.height else None
        out[f"prev_3c_{i}"] = f["prev_3c_ah"][0] if f.height else None
    rank = etiqueta_periodo(sorted(int(x) for x in u.particion["anios_ranking"]))
    celda = int(u["celda_minima"])
    pares = ph.filter(
        (pl.col("periodo") == rank) & (pl.col("elix_a") < pl.col("elix_b")) & pl.col("phi").is_not_null() & (pl.col("n_ab") >= celda)
    )
    if pares.height:
        top = pares.with_columns(pl.col("phi").abs().alias("_a")).sort("_a", descending=True).row(0, named=True)
        out.update({"phi_max": top["phi"], "phi_max_a": top["elix_a"], "phi_max_b": top["elix_b"], "periodo_phi": rank,
                    "n_pares_redundantes": int((pares["phi"].abs() >= float(u["redundancia_abs"])).sum()), "n_pares": pares.height})
    h = d.filter(pl.col("cod_hospital") != TODOS)
    anios = sorted(h["anio"].unique().to_list())
    if len(anios) > 1:
        a0, a1 = anios[0], anios[-1]
        ini = h.filter(pl.col("anio") == a0).select("cod_hospital", pl.col("n").alias("n0"))
        fin = h.filter(pl.col("anio") == a1).select("cod_hospital", pl.col("n").alias("n1"), "cambio_relativo")
        j = ini.join(fin, on="cod_hospital").filter((pl.col("n0") >= MIN_EPISODIOS_LINEA) & (pl.col("n1") >= MIN_EPISODIOS_LINEA) & pl.col("cambio_relativo").is_not_null())
        out["n_hospitales_cambio"] = j.height
        out["cambio_hosp_p10"] = float(j["cambio_relativo"].quantile(0.10)) if j.height else None
        out["cambio_hosp_p90"] = float(j["cambio_relativo"].quantile(0.90)) if j.height else None
        out["n_hospitales_con_alza"] = int((j["cambio_relativo"] > 0).sum()) if j.height else 0
    return pl.DataFrame([out])


def tabla_vs_3c(por_anio: dict[int, dict[str, Any]], e: elixhauser.Elixhauser, u: config.Umbrales) -> pl.DataFrame:
    """Mapeo completo frente al de tres caracteres, sobre todos los años (supuesto S02).

    La diferencia relativa se compara con ``cambio_relativo_relevante`` (20 %), el piso de relevancia práctica del protocolo.
    """
    umbral = float(u["cambio_relativo_relevante"])
    largos = largos_por_categoria(e)
    n_ah = sum(d["n"] for d in por_anio.values())
    n_todos = sum(d["n_todos"] for d in por_anio.values())
    suma = lambda clave, i: sum(d[clave][i] for d in por_anio.values())  # noqa: E731
    filas = []
    for i in e.ids:
        n_c, n_3 = suma("sec", i), suma("tres", i)
        n_ct, n_3t = suma("sec_todos", i), suma("tres_todos", i)
        pc = n_c / n_ah if n_ah else None
        p3 = n_3 / n_ah if n_ah else None
        dif_rel = ((p3 - pc) / pc) if (pc and p3 is not None) else None
        filas.append({
            "elix_id": i, "nombre_es": e.nombres_es[i], "largos_definicion": largos[i],
            "n_base_ah": n_ah, "n_completo_ah": n_c, "n_3c_ah": n_3, "prev_completo_ah": pc, "prev_3c_ah": p3,
            "dif_pp_ah": (100.0 * (p3 - pc)) if pc is not None else None, "dif_relativa_ah": dif_rel,
            "supera_umbral": ("si" if (dif_rel is not None and abs(dif_rel) > umbral) else "no") if pc else "no evaluable",
            "n_base_todos": n_todos, "n_completo_todos": n_ct, "n_3c_todos": n_3t,
            "prev_completo_todos": n_ct / n_todos if n_todos else None, "prev_3c_todos": n_3t / n_todos if n_todos else None,
        })
    return pl.DataFrame(filas, schema={
        "elix_id": pl.String, "nombre_es": pl.String, "largos_definicion": pl.String, "n_base_ah": pl.Int64, "n_completo_ah": pl.Int64, "n_3c_ah": pl.Int64,
        "prev_completo_ah": pl.Float64, "prev_3c_ah": pl.Float64, "dif_pp_ah": pl.Float64, "dif_relativa_ah": pl.Float64, "supera_umbral": pl.String,
        "n_base_todos": pl.Int64, "n_completo_todos": pl.Int64, "n_3c_todos": pl.Int64, "prev_completo_todos": pl.Float64, "prev_3c_todos": pl.Float64,
    })


def tabla_principal(por_anio: dict[int, dict[str, Any]], e: elixhauser.Elixhauser) -> pl.DataFrame:
    """Efecto de incluir el diagnóstico principal en el cálculo de cada categoría (regla R3: solo secundarios)."""
    n_base = sum(d["n"] for d in por_anio.values())
    filas = []
    for i in e.ids:
        n_s = sum(d["sec"][i] for d in por_anio.values())
        n_u = sum(d["uni"][i] for d in por_anio.values())
        n_p = sum(d["prin"][i] for d in por_anio.values())
        filas.append({
            "elix_id": i, "nombre_es": e.nombres_es[i], "n_base": n_base, "n_secundario": n_s, "n_con_principal": n_u, "n_principal": n_p,
            "prev_secundario": n_s / n_base if n_base else None, "prev_con_principal": n_u / n_base if n_base else None,
            "aumento_relativo": ((n_u - n_s) / n_s) if n_s else None,
        })
    return pl.DataFrame(filas, schema={
        "elix_id": pl.String, "nombre_es": pl.String, "n_base": pl.Int64, "n_secundario": pl.Int64, "n_con_principal": pl.Int64, "n_principal": pl.Int64,
        "prev_secundario": pl.Float64, "prev_con_principal": pl.Float64, "aumento_relativo": pl.Float64,
    })


def phi_desde_conteos(co: np.ndarray, n: int) -> np.ndarray:
    """Matriz phi a partir de la matriz de coocurrencias (la diagonal son las marginales). NaN donde una variable es constante."""
    m = np.diag(co).astype(float)
    num = n * co.astype(float) - np.outer(m, m)
    varianza = m * (n - m)
    den = np.sqrt(np.outer(varianza, varianza))
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(den > 0, num / den, np.nan)
    np.fill_diagonal(phi, np.where(varianza > 0, 1.0, np.nan))
    return phi


def tabla_phi(por_anio: dict[int, dict[str, Any]], e: elixhauser.Elixhauser, u: config.Umbrales) -> pl.DataFrame:
    """Phi entre pares de categorías en 2019 a 2022 (ranking) y en 2023 (confirmación), sobre adultos hospitalizados."""
    conf = int(u.particion["anio_confirmacion"])
    rank_cfg = sorted(int(x) for x in u.particion["anios_ranking"])
    periodos: dict[str, list[int]] = {}
    if set(rank_cfg) <= set(por_anio):
        periodos[etiqueta_periodo(rank_cfg)] = rank_cfg
    if conf in por_anio:
        periodos[str(conf)] = [conf]
    ids = list(e.ids)
    filas = []
    for nombre, lista in periodos.items():
        co = sum(por_anio[a]["co"] for a in lista)
        n = sum(por_anio[a]["n"] for a in lista)
        phi = phi_desde_conteos(co, n)
        for r, a in enumerate(ids):
            for c, b in enumerate(ids):
                filas.append({
                    "periodo": nombre, "elix_a": a, "elix_b": b, "nombre_a": e.nombres_es[a], "nombre_b": e.nombres_es[b], "n_base": n,
                    "n_a": int(co[r, r]), "n_b": int(co[c, c]), "n_ab": int(co[r, c]), "phi": None if np.isnan(phi[r, c]) else float(phi[r, c]),
                })
    return pl.DataFrame(filas, schema={
        "periodo": pl.String, "elix_a": pl.String, "elix_b": pl.String, "nombre_a": pl.String, "nombre_b": pl.String, "n_base": pl.Int64,
        "n_a": pl.Int64, "n_b": pl.Int64, "n_ab": pl.Int64, "phi": pl.Float64,
    })


def tabla_deriva(por_anio: dict[int, dict[str, Any]], u: config.Umbrales) -> pl.DataFrame:
    """Promedio de diagnósticos secundarios por año y hospital, con IC y cambio relativo contra el primer año (supuesto S11)."""
    z = float(u["z_95"])
    partes = []
    for a, d in sorted(por_anio.items()):
        h = d["hospitales"].rename({"hospital": "cod_hospital"}).select(["cod_hospital", "n", "suma", "suma2"]).sort("cod_hospital")
        tot = pl.DataFrame({"cod_hospital": [TODOS], "n": [d["n"]], "suma": [d["suma"]], "suma2": [d["suma2"]]},
                           schema={"cod_hospital": pl.String, "n": pl.UInt32, "suma": pl.Int64, "suma2": pl.Int64})
        partes.append(pl.concat([tot, h]).with_columns(pl.lit(a, dtype=pl.Int64).alias("anio")))
    df = pl.concat(partes).with_columns(pl.col("n").cast(pl.Int64))
    df = df.with_columns(
        (pl.col("suma") / pl.col("n")).alias("media"),
        ((pl.col("suma2") - pl.col("suma") ** 2 / pl.col("n")) / (pl.col("n") - 1)).alias("varianza"),
    ).with_columns(
        pl.col("varianza").clip(lower_bound=0).sqrt().alias("desv"),
        (z * (pl.col("varianza").clip(lower_bound=0) / pl.col("n")).sqrt()).alias("_mitad"),
    ).with_columns((pl.col("media") - pl.col("_mitad")).alias("ic_bajo"), (pl.col("media") + pl.col("_mitad")).alias("ic_alto"))
    primero = min(por_anio)
    ref = df.filter(pl.col("anio") == primero).select(
        "cod_hospital", pl.col("media").alias("_m0"), pl.col("varianza").alias("_v0"), pl.col("n").alias("_n0")
    )
    df = df.join(ref, on="cod_hospital", how="left")
    ok = (pl.col("media") > 0) & (pl.col("_m0") > 0) & (pl.col("anio") != primero)
    se_log = ((pl.col("varianza") / (pl.col("n") * pl.col("media") ** 2)) + (pl.col("_v0") / (pl.col("_n0") * pl.col("_m0") ** 2))).sqrt()
    razon = pl.col("media") / pl.col("_m0")
    df = df.with_columns(
        pl.when(ok).then(razon - 1.0).alias("cambio_relativo"),
        pl.when(ok).then(razon * (-z * se_log).exp() - 1.0).alias("cambio_bajo"),
        pl.when(ok).then(razon * (z * se_log).exp() - 1.0).alias("cambio_alto"),
    )
    df = df.with_columns((pl.col("cod_hospital") != TODOS).alias("_h"))
    return df.sort(["_h", "cod_hospital", "anio"]).select(
        ["anio", "cod_hospital", "n", "suma", "media", "desv", "ic_bajo", "ic_alto", "cambio_relativo", "cambio_bajo", "cambio_alto"]
    )


# ----------------------------------------------------------------------------- ejecución
def calcular(ctx: Contexto, cat: catalogos.Catalogos | None = None) -> dict[str, Any]:
    e, e3 = elixhauser.cargar(), elixhauser.cargar(solo_3c=True)
    por_anio: dict[int, dict[str, Any]] = {}
    for anio in ctx.fuente.anios():
        meta, dx = leer_anio(ctx, anio)
        por_anio[anio] = resumir_anio(meta, dx, e, e3)
        del meta, dx
    u = ctx.umbrales
    r = {
        "por_anio": por_anio, "e": e,
        "prevalencia": tabla_prevalencia(por_anio, e, u), "vs_3c": tabla_vs_3c(por_anio, e, u), "principal": tabla_principal(por_anio, e),
        "phi": tabla_phi(por_anio, e, u), "deriva": tabla_deriva(por_anio, u),
    }
    r["resumen"] = resumen_comorbilidad(r, u)
    return r


def _escribir(ctx: Contexto, r: dict[str, Any]) -> None:
    TA = TAREA
    ctx.escribir_tabla("t23_elix_prevalencia.csv", r["prevalencia"], tarea=TA, descripcion="Prevalencia de las 31 categorías de Elixhauser en adultos hospitalizados",
                       cols_n=["n", "n_base"], dependientes=["prevalencia", "ic_bajo", "ic_alto"])
    ctx.escribir_tabla("t23_elix_vs_3c.csv", r["vs_3c"], tarea=TA, descripcion="Mapeo completo frente al de tres caracteres (supuesto S02)",
                       cols_n=["n_completo_ah", "n_3c_ah", "n_completo_todos", "n_3c_todos"],
                       dependientes=["prev_completo_ah", "prev_3c_ah", "dif_pp_ah", "dif_relativa_ah", "supera_umbral", "prev_completo_todos", "prev_3c_todos"])
    ctx.escribir_tabla("t23_elix_principal.csv", r["principal"], tarea=TA, descripcion="Efecto de incluir el diagnóstico principal en cada categoría (regla R3)",
                       cols_n=["n_secundario", "n_con_principal", "n_principal"], dependientes=["prev_secundario", "prev_con_principal", "aumento_relativo"])
    ctx.escribir_tabla("t23_elix_phi.csv", r["phi"], tarea=TA, descripcion="Coeficiente phi entre las 31 categorías de Elixhauser",
                       cols_n=["n_a", "n_b", "n_ab"], dependientes=["phi"])
    ctx.escribir_tabla("t23_deriva_codificacion.csv", r["deriva"], tarea=TA,
                       descripcion="Promedio de diagnósticos secundarios en adultos hospitalizados por año y hospital",
                       cols_n=["n"], dependientes=["suma", "media", "desv", "ic_bajo", "ic_alto", "cambio_relativo", "cambio_bajo", "cambio_alto"])
    ctx.escribir_tabla("t23_resumen_comorbilidad.csv", r["resumen"], tarea=TA, descripcion="Cifras derivadas de la comorbilidad y la codificación")


def registrar_metricas(ctx: Contexto, r: dict[str, Any]) -> None:
    por_anio, e = r["por_anio"], r["e"]
    anios = sorted(por_anio)
    pool = etiqueta_periodo(anios)
    m = ctx.registrar_metrica
    v = r["vs_3c"]
    for row in v.iter_rows(named=True):
        i = row["elix_id"]
        m(f"elix.prev_pct.{i}", None if row["prev_completo_ah"] is None else 100.0 * row["prev_completo_ah"], tarea=TAREA, tabla="t23_elix_vs_3c.csv", alcance=pool,
          nota="adultos hospitalizados, mapeo completo, solo diagnósticos secundarios")
        m(f"elix.prev_pct_todos.{i}", None if row["prev_completo_todos"] is None else 100.0 * row["prev_completo_todos"], tarea=TAREA, tabla="t23_elix_vs_3c.csv",
          alcance=pool, nota="todos los egresos, mapeo completo")
        m(f"elix.prev_pct_3c_todos.{i}", None if row["prev_3c_todos"] is None else 100.0 * row["prev_3c_todos"], tarea=TAREA, tabla="t23_elix_vs_3c.csv",
          alcance=pool, nota="todos los egresos, mapeo de tres caracteres (método del repositorio)")
    dia = v.filter(pl.col("elix_id").is_in(["ELIX_11", "ELIX_12"]))
    if dia.height == 2 and dia["n_base_ah"][0]:
        m("elix.prev_pct.diabetes", 100.0 * (dia["n_completo_ah"].sum()) / dia["n_base_ah"][0], tarea=TAREA, tabla="t23_elix_vs_3c.csv", alcance=pool,
          nota="diabetes complicada más no complicada, adultos hospitalizados, mapeo completo")
        m("elix.prev_pct_3c_todos.diabetes", 100.0 * (dia["n_3c_todos"].sum()) / dia["n_base_todos"][0], tarea=TAREA, tabla="t23_elix_vs_3c.csv", alcance=pool,
          nota="diabetes complicada más no complicada, todos los egresos, mapeo de tres caracteres")
    m("elix.n_categorias", len(e.ids), tarea=TAREA, tabla="t23_elix_prevalencia.csv", alcance="mapeo", nota="categorías del mapeo implementado")
    m("elix.n_categorias_sobre_umbral_3c", int((v["supera_umbral"] == "si").sum()), tarea=TAREA, tabla="t23_elix_vs_3c.csv", alcance=pool,
      nota="categorías cuya prevalencia con tres caracteres difiere del mapeo completo más que el umbral de cambio relativo")
    d = r["deriva"].filter(pl.col("cod_hospital") == TODOS)
    for row in d.iter_rows(named=True):
        m(f"codificacion.media_dx_sec_adultos_hosp.{row['anio']}", row["media"], tarea=TAREA, tabla="t23_deriva_codificacion.csv", ic_bajo=row["ic_bajo"],
          ic_alto=row["ic_alto"])
    if d.height > 1:
        a0, a1 = d.row(0, named=True), d.row(d.height - 1, named=True)
        m("codificacion.aumento_pct_2019_2024", 100.0 * a1["cambio_relativo"] if a1["cambio_relativo"] is not None else None, tarea=TAREA,
          tabla="t23_deriva_codificacion.csv", ic_bajo=None if a1["cambio_bajo"] is None else 100.0 * a1["cambio_bajo"],
          ic_alto=None if a1["cambio_alto"] is None else 100.0 * a1["cambio_alto"], alcance=etiqueta_periodo([a0["anio"], a1["anio"]]),
          nota="cambio relativo del promedio entre el primer y el último año")


def ejecutar(ctx: Contexto, cat: catalogos.Catalogos | None = None) -> dict[str, Any]:
    r = calcular(ctx, cat)
    _escribir(ctx, r)
    registrar_metricas(ctx, r)
    figura_prevalencia(ctx)
    figura_codificacion(ctx)
    figura_phi(ctx)
    return r


# ---------------------------------------------------------------------------------- figuras
def _arr(valores, escala: float = 1.0) -> np.ndarray:
    return np.array([np.nan if v is None else escala * float(v) for v in valores], dtype=float)


def figura_prevalencia(ctx: Contexto) -> None:
    t = ctx.leer_tabla("t23_elix_prevalencia.csv", columnas_texto=["periodo", "elix_id", "nombre_es"])
    anios = sorted(p for p in t["periodo"].unique().to_list() if p.isdigit())
    if not anios:
        return
    pooled = t.filter(~pl.col("periodo").is_in(anios)).filter(pl.col("periodo") == etiqueta_periodo([int(a) for a in anios]))
    orden = pooled.sort("prevalencia", descending=True, nulls_last=True)["elix_id"].to_list() if pooled.height else sorted(t["elix_id"].unique().to_list())
    nombres = dict(zip(t["elix_id"], t["nombre_es"]))
    t = t.with_columns((pl.col("prevalencia") * 100).alias("pct"))
    m, filas, cols = graficos.tabla_a_matriz(t.filter(pl.col("periodo").is_in(anios)), "elix_id", "periodo", "pct", orden_filas=orden, orden_columnas=anios)
    fig, ax = graficos.nueva_figura(figsize=(8.2, 9.6))
    graficos.mapa_calor(fig, ax, m, [f"{f[-2:]}  {nombres.get(f, f)}" for f in filas], cols, vmin=0, etiqueta_barra="% de los adultos hospitalizados", fmt="{:.1f}")
    ax.set_xlabel("Año del archivo")
    ctx.figuras.guardar_figura(
        fig, id_figura="F08_01", tarea=8,
        titulo="Prevalencia de las 31 categorías de Elixhauser por año",
        descripcion_alt="Mapa de calor con el porcentaje de adultos hospitalizados que tiene cada una de las 31 categorías de Elixhauser en cada año, "
                        "ordenadas de mayor a menor prevalencia.",
        csv="t23_elix_prevalencia.csv", anios=f"{anios[0]} a {anios[-1]}", decision="8",
        extra_pie="Mapeo completo de Quan 2005 sobre diagnósticos secundarios; adultos hospitalizados.",
    )


def figura_codificacion(ctx: Contexto) -> None:
    t = ctx.leer_tabla("t23_deriva_codificacion.csv", columnas_texto=["cod_hospital"])
    nac = t.filter(pl.col("cod_hospital") == TODOS).sort("anio")
    if nac.height == 0:
        return
    fig, (a1, a2) = graficos.nueva_figura_multiple(1, 2, figsize=(11.0, 4.8))
    x = nac["anio"].to_numpy()
    y, lo, hi = _arr(nac["media"]), _arr(nac["ic_bajo"]), _arr(nac["ic_alto"])
    a1.errorbar(x, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], marker="o", capsize=3, lw=1.5, color=graficos.OKABE_ITO[0])
    for xi, yi, hi_i in zip(x, y, hi):
        if not np.isnan(yi):
            a1.annotate(f"{yi:.2f}".replace(".", ","), (xi, hi_i), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7,
                        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.85})
    a1.set_xticks(x)
    a1.set_ylabel("Diagnósticos secundarios por egreso (promedio)")
    a1.set_xlabel("Año del archivo")
    a1.set_title("Total nacional", fontsize=9)
    a1.set_ylim(bottom=0)
    hosp = t.filter(pl.col("cod_hospital") != TODOS)
    anios = sorted(nac["anio"].to_list())
    dibujados = 0
    for cod, g in hosp.group_by("cod_hospital", maintain_order=True):
        g = g.sort("anio")
        if g.height != len(anios) or g["n"].null_count() or int(g["n"].min() or 0) < MIN_EPISODIOS_LINEA or g["media"].null_count():
            continue
        a2.plot(g["anio"].to_numpy(), _arr(g["media"]), color="#9A9A9A", lw=0.8, alpha=0.8)
        dibujados += 1
    a2.plot(x, y, color=graficos.OKABE_ITO[3], lw=2.2, label="Total nacional")
    a2.set_xticks(x)
    a2.set_xlabel("Año del archivo")
    a2.set_ylabel("Diagnósticos secundarios por egreso (promedio)")
    a2.set_title(f"Hospitales con al menos {MIN_EPISODIOS_LINEA} egresos cada año ({dibujados} líneas)", fontsize=9)
    a2.set_ylim(bottom=0)
    a2.legend(loc="lower right")
    ctx.figuras.guardar_figura(
        fig, id_figura="F08_02", tarea=8,
        titulo="Deriva de la codificación: diagnósticos secundarios por egreso",
        descripcion_alt="Dos paneles por año: a la izquierda el promedio nacional de diagnósticos secundarios por egreso de adultos hospitalizados con su intervalo "
                        "de confianza; a la derecha una línea gris por hospital y el total nacional destacado.",
        csv="t23_deriva_codificacion.csv", anios=f"{anios[0]} a {anios[-1]}", decision="17", uso_interno=True,
        extra_pie="Panel derecho: solo hospitales con egresos suficientes cada año.",
    )


def figura_phi(ctx: Contexto) -> None:
    t = ctx.leer_tabla("t23_elix_phi.csv", columnas_texto=["periodo", "elix_a", "elix_b"])
    periodos = [p for p in t["periodo"].unique().to_list() if "-" in p]
    if not periodos:
        return
    periodo = periodos[0]
    s = t.filter(pl.col("periodo") == periodo)
    ids = sorted(s["elix_a"].unique().to_list())
    m, filas, cols = graficos.tabla_a_matriz(s, "elix_a", "elix_b", "phi", orden_filas=ids, orden_columnas=ids)
    fuera = np.abs(m[~np.eye(len(ids), dtype=bool)])
    tope = float(np.nanmax(fuera)) if np.isfinite(fuera).any() else 0.5
    tope = max(0.1, round(tope + 0.049, 1))
    fig, ax = graficos.nueva_figura(figsize=(8.2, 7.4))
    etiquetas = [i[-2:] for i in ids]
    graficos.mapa_calor(fig, ax, m, etiquetas, etiquetas, cmap=graficos.CMAP_DIVERGENTE, vmin=-tope, vmax=tope, etiqueta_barra="Coeficiente phi", anotar=False)
    ax.set_xlabel("Categoría de Elixhauser (número)")
    ax.set_ylabel("Categoría de Elixhauser (número)")
    ctx.figuras.guardar_figura(
        fig, id_figura="F08_03", tarea=8,
        titulo=f"Matriz phi entre las 31 categorías de Elixhauser ({periodo})",
        descripcion_alt="Mapa de calor cuadrado con el coeficiente phi entre cada par de las 31 categorías de Elixhauser en adultos hospitalizados; la diagonal vale uno.",
        csv="t23_elix_phi.csv", anios=periodo, decision="8",
        extra_pie="Las categorías se numeran como en ELIX_01 a ELIX_31; el rango de color es simétrico y excluye la diagonal.",
    )


# ------------------------------------------------------------------------------------ informe
def agregar_secciones(inf: Informe, ctx: Contexto, n0: int = 1) -> None:
    """Agrega al informe de exploración (CRISP-DM 2.3) las secciones de comorbilidad y deriva de la codificación."""
    from .descripcion import Cifra_umbral

    u = ctx.umbrales
    anios = sorted(int(x) for x in ctx.leer_tabla("t23_deriva_codificacion.csv", columnas_texto=["cod_hospital"])["anio"].unique().to_list())
    pool = etiqueta_periodo(anios)
    rs = lambda col: cifra(ctx, "t23_resumen_comorbilidad.csv", col, columnas_texto=["phi_max_a", "phi_max_b", "periodo_phi"])  # noqa: E731
    res = ctx.leer_tabla("t23_resumen_comorbilidad.csv", columnas_texto=["phi_max_a", "phi_max_b", "periodo_phi"]).row(0, named=True)

    # --- prevalencia ----------------------------------------------------------------------------------
    inf.seccion(f"{n0}. Prevalencia de las categorías de Elixhauser")
    top = lambda col, rango: cifra(ctx, "t23_elix_prevalencia.csv", col, columnas_texto=["periodo", "elix_id", "nombre_es"], periodo=pool, rango=rango)  # noqa: E731
    inf.parrafo(
        "Las {n_cat} categorías se calculan con el mapeo de Quan et al., contrastado con una segunda fuente pública, sobre los diagnósticos secundarios (regla R3) y con las "
        "jerarquías habituales: la hipertensión complicada anula a la no complicada, la diabetes complicada a la no complicada y el cáncer metastásico al tumor sólido. "
        "La base son los adultos hospitalizados. Considerando todos los años, la categoría más frecuente es {nom1|texto} ({p1|prop1}), seguida de {nom2|texto} ({p2|prop1}) y {nom3|texto} ({p3|prop1}).",
        n_cat=rs("n_categorias"), nom1=top("nombre_es", 1), p1=top("prevalencia", 1), nom2=top("nombre_es", 2), p2=top("prevalencia", 2),
        nom3=top("nombre_es", 3), p3=top("prevalencia", 3),
    )
    inf.figura("F08_01")
    tp = ctx.leer_tabla("t23_elix_prevalencia.csv", columnas_texto=["periodo", "elix_id", "nombre_es"]).filter((pl.col("periodo") == pool) & (pl.col("rango") <= 12)).sort("rango")
    inf.tabla(
        "t23_elix_prevalencia.csv", df=tp, columnas=["rango", "elix_id", "nombre_es", "n", "prevalencia", "ic_bajo", "ic_alto"],
        encabezados={"rango": "Rango", "elix_id": "Categoría", "nombre_es": "Nombre", "n": "Egresos", "prevalencia": "Prevalencia", "ic_bajo": "IC bajo", "ic_alto": "IC alto"},
        formatos={"prevalencia": "prop1", "ic_bajo": "prop1", "ic_alto": "prop1"}, leyenda="Las doce categorías más frecuentes, todos los años",
    )

    # --- mapeo completo frente a tres caracteres ----------------------------------------------------------
    inf.seccion(f"{n0 + 1}. Mapeo completo frente al de tres caracteres")
    v = lambda col, i: cifra(ctx, "t23_elix_vs_3c.csv", col, columnas_texto=["elix_id"], elix_id=i)  # noqa: E731
    sobre = int(res["n_sobre_umbral_3c"])
    veredicto = (
        "el supuesto S02 se rechaza: el mapeo de tres caracteres no equivale al completo."
        if sobre > 0 else "el supuesto S02 se acepta: el mapeo de tres caracteres equivale al completo."
    )
    inf.parrafo(
        "El mapeo del repositorio compara solo tres caracteres del código, así que no reconoce las categorías que Quan define con cuatro. La diabetes no complicada pasa de "
        "{dm_c|prop1} con el mapeo completo a {dm_3|prop1} con tres caracteres, y la complicada de {dmc_c|prop1} a {dmc_3|prop1}. De las {n_cat} categorías, {sobre} difieren "
        "en más de {umbral|prop1} de su prevalencia relativa; " + veredicto,
        dm_c=v("prev_completo_ah", "ELIX_11"), dm_3=v("prev_3c_ah", "ELIX_11"), dmc_c=v("prev_completo_ah", "ELIX_12"), dmc_3=v("prev_3c_ah", "ELIX_12"),
        n_cat=rs("n_categorias"), sobre=rs("n_sobre_umbral_3c"), umbral=Cifra_umbral(ctx, "cambio_relativo_relevante"),
    )
    tv = ctx.leer_tabla("t23_elix_vs_3c.csv", columnas_texto=["elix_id", "nombre_es", "largos_definicion", "supera_umbral"]).filter(pl.col("supera_umbral") == "si").sort("dif_relativa_ah")
    inf.tabla(
        "t23_elix_vs_3c.csv", df=tv, columnas=["elix_id", "nombre_es", "largos_definicion", "prev_completo_ah", "prev_3c_ah", "dif_relativa_ah"],
        encabezados={"elix_id": "Categoría", "nombre_es": "Nombre", "largos_definicion": "Largos de prefijo", "prev_completo_ah": "Prevalencia completa",
                     "prev_3c_ah": "Prevalencia con 3 caracteres", "dif_relativa_ah": "Diferencia relativa"},
        formatos={"prev_completo_ah": "prop2", "prev_3c_ah": "prop2", "dif_relativa_ah": "prop1"}, leyenda="Categorías que superan el umbral de diferencia relativa",
    )

    # --- diagnóstico principal y regla R3 ----------------------------------------------------------------------
    inf.seccion(f"{n0 + 2}. Diagnóstico principal y regla R3")
    tpr = ctx.leer_tabla("t23_elix_principal.csv", columnas_texto=["elix_id", "nombre_es"]).filter(pl.col("aumento_relativo").is_not_null())
    mayor = tpr.sort("aumento_relativo", descending=True).row(0, named=True) if tpr.height else None
    if mayor is not None:
        pr = lambda col: cifra(ctx, "t23_elix_principal.csv", col, columnas_texto=["elix_id", "nombre_es"], elix_id=mayor["elix_id"])  # noqa: E731
        inf.parrafo(
            "La regla R3 calcula las categorías solo con diagnósticos secundarios, porque el diagnóstico principal es el motivo del ingreso y no una comorbilidad previa. "
            "Si se incluyera, la categoría con mayor aumento sería {nombre|texto}, que pasaría de {prev_s|prop1} a {prev_p|prop1} de los adultos hospitalizados ({aumento|prop1} más).",
            nombre=pr("nombre_es"), prev_s=pr("prev_secundario"), prev_p=pr("prev_con_principal"), aumento=pr("aumento_relativo"),
        )
    tt = tpr.sort("aumento_relativo", descending=True).head(8)
    inf.tabla(
        "t23_elix_principal.csv", df=tt, columnas=["elix_id", "nombre_es", "n_secundario", "n_principal", "prev_secundario", "prev_con_principal", "aumento_relativo"],
        encabezados={"elix_id": "Categoría", "nombre_es": "Nombre", "n_secundario": "Con la categoría (secundarios)", "n_principal": "Con la categoría (principal)",
                     "prev_secundario": "Prevalencia (secundarios)", "prev_con_principal": "Prevalencia (con principal)", "aumento_relativo": "Aumento relativo"},
        formatos={"prev_secundario": "prop2", "prev_con_principal": "prop2", "aumento_relativo": "prop1"}, leyenda="Las categorías que más cambiarían al incluir el diagnóstico principal",
    )

    # --- asociación entre categorías --------------------------------------------------------------------------
    inf.seccion(f"{n0 + 3}. Asociación entre categorías")
    if res.get("phi_max_a"):
        inf.parrafo(
            "El mayor coeficiente phi entre dos categorías distintas en {periodo|texto} es {phi|dec2}, entre {a|texto} y {b|texto}. {n_red} de {n_pares} pares con casos suficientes alcanzan "
            "el umbral de redundancia de {umbral|dec2}; las categorías de Elixhauser se mantienen como indicadores separados y la agrupación formal de variables redundantes "
            "se hace en la selección de variables.",
            periodo=rs("periodo_phi"), phi=rs("phi_max"), a=rs("phi_max_a"), b=rs("phi_max_b"), n_red=rs("n_pares_redundantes"), n_pares=rs("n_pares"),
            umbral=Cifra_umbral(ctx, "redundancia_abs"),
        )
    inf.figura("F08_03")
    tph = ctx.leer_tabla("t23_elix_phi.csv", columnas_texto=["periodo", "elix_a", "elix_b", "nombre_a", "nombre_b"])
    rank = res.get("periodo_phi")
    if rank:
        tph = tph.filter((pl.col("periodo") == rank) & (pl.col("elix_a") < pl.col("elix_b")) & pl.col("phi").is_not_null()).with_columns(pl.col("phi").abs().alias("_a"))
        inf.tabla(
            "t23_elix_phi.csv", df=tph.sort("_a", descending=True).head(10), columnas=["elix_a", "nombre_a", "elix_b", "nombre_b", "n_ab", "phi"],
            encabezados={"elix_a": "Categoría A", "nombre_a": "Nombre A", "elix_b": "Categoría B", "nombre_b": "Nombre B", "n_ab": "Egresos con ambas", "phi": "Phi"},
            formatos={"phi": "dec2"}, leyenda="Los diez pares con mayor asociación",
        )

    # --- deriva de la codificación ---------------------------------------------------------------------------------
    inf.seccion(f"{n0 + 4}. Deriva de la codificación")
    d = lambda col, a: cifra(ctx, "t23_deriva_codificacion.csv", col, columnas_texto=["cod_hospital"], cod_hospital=TODOS, anio=a)  # noqa: E731
    inf.parrafo(
        "El promedio de diagnósticos secundarios por egreso de adultos hospitalizados pasa de {m0|dec2} en {a0|texto} a {m1|dec2} en {a1|texto}, un cambio relativo de {cambio|prop1} "
        "(IC {lo|prop1} a {hi|prop1}). Si la codificación crece sin que cambien los pacientes, el puntaje de comorbilidad sube y un modelo entrenado en los años antiguos "
        "subestima el riesgo de los nuevos; es evidencia para la decisión 17.",
        m0=d("media", anios[0]), a0=d("anio", anios[0]), m1=d("media", anios[-1]), a1=d("anio", anios[-1]),
        cambio=d("cambio_relativo", anios[-1]), lo=d("cambio_bajo", anios[-1]), hi=d("cambio_alto", anios[-1]),
    )
    inf.figura("F08_02")
    if res.get("n_hospitales_cambio"):
        inf.parrafo(
            "Entre los {n_h} hospitales con al menos {min_h} egresos en el primer y el último año, el cambio relativo del promedio va de {p10|prop1} (percentil diez) a {p90|prop1} "
            "(percentil noventa), y {n_alza} aumentan. La codificación no cambia igual en todos los hospitales, por eso el cambio pesa sobre la comparación entre ellos. "
            "Esta figura es de uso interno: muestra códigos de hospital y no nombres.",
            n_h=rs("n_hospitales_cambio"), min_h=rs("min_episodios_linea"), p10=rs("cambio_hosp_p10"), p90=rs("cambio_hosp_p90"), n_alza=rs("n_hospitales_con_alza"),
        )
    td = ctx.leer_tabla("t23_deriva_codificacion.csv", columnas_texto=["cod_hospital"]).filter(pl.col("cod_hospital") == TODOS)
    inf.tabla(
        "t23_deriva_codificacion.csv", df=td, columnas=["anio", "n", "media", "ic_bajo", "ic_alto", "cambio_relativo", "cambio_bajo", "cambio_alto"],
        encabezados={"anio": "Año", "n": "Adultos hospitalizados", "media": "Promedio", "ic_bajo": "IC bajo", "ic_alto": "IC alto", "cambio_relativo": "Cambio relativo",
                     "cambio_bajo": "IC bajo del cambio", "cambio_alto": "IC alto del cambio"},
        formatos={"media": "dec2", "ic_bajo": "dec2", "ic_alto": "dec2", "cambio_relativo": "prop1", "cambio_bajo": "prop1", "cambio_alto": "prop1"},
        leyenda="Promedio de diagnósticos secundarios, total nacional",
    )
    limites = [
        "- Las categorías no distinguen comorbilidades presentes al ingreso de complicaciones de la hospitalización: los diagnósticos están codificados al egreso y no hay marcador de presente al ingreso.",
        "- Los intervalos ignoran el agrupamiento por hospital y son indicativos.",
    ]
    if ctx.etiqueta == "extracto":                  # las frases sobre el extracto solo van si la corrida usó extractos
        limites.append("- En el extracto los conteos absolutos no se comparan con los datos completos.")
    inf.texto_libre_sin_cifras("\n".join(limites))

