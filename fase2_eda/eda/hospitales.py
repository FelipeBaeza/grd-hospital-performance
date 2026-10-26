"""CRISP-DM 2.3, parte 3: hospitales, identificador del paciente, historial y traslados.

Qué mide (decisiones 1, 2, 5, 6, 13, 16 y 17):
- Perfil de cada hospital por año: volumen, proporción de menores (hospitales pediátricos o monográficos) y presencia en el catálogo.
- Volumen elegible y muertes esperadas por estandarización indirecta por edad y sexo, con la cuadrícula de umbrales de la decisión 6.
- Embudo crudo de mortalidad con límites de Spiegelhalter (2005) y sobredispersión winsorizada.
- Factibilidad del historial de 12 meses: identificador válido y ventana completa dentro de un mismo espacio de identificadores.
- Enlace de los traslados salientes con un ingreso posterior en otro hospital, y de los entrantes con un traslado previo.
- Autorreferencia de HOSPPROCEDENCIA (el hospital de procedencia es el mismo que el del egreso) y mortalidad de los entrantes.

Partición temporal: el año expuesto solo aporta estructura (perfil de hospitales y procedencia). El historial, los enlaces y todo lo que usa el
desenlace se calculan con 2019 a 2023, y los enlaces nunca buscan ingresos en 2024: los traslados cercanos al fin del período quedan fuera
del denominador (no son elegibles) en lugar de contarse como no enlazados.

Los identificadores solo existen en memoria: las salidas son conteos y tasas. Los hospitales se identifican por código, nunca por nombre.

En un extracto (fracción no aleatoria de los egresos) los enlaces y el historial se subestiman por construcción, porque el ingreso
siguiente de un paciente casi nunca está en el mismo extracto: esas cifras solo prueban la mecánica y se rotulan como no interpretables.
"""

from __future__ import annotations

import datetime as dt
import difflib
import re
import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from . import calidad, catalogos, config, elixhauser, estadistica, graficos
from .cohorte import (
    ACTIVIDAD_HOSP,
    CODIGO_TIPOALTA,
    _grupo_edad_expr,
    con_periodos,
    con_wilson,
    etiqueta_periodo,
    leer_base,
)
from .contexto import Contexto
from .informes import Informe, cifra

TAREA = "T09"
TODAS = "(todas)"
TEXTO_ENTRANTE = "OTROS HOSPITALES"  # TIPO_PROCEDENCIA que contiene este texto = traslado entrante
CODIGOS_SALIDA_HOSPITAL = ("DERIV_HOSP_SERVICIO", "DERIV_HOSP_RED")
ETIQUETAS_SALIDA = tuple(e for e, c in CODIGO_TIPOALTA.items() if c in CODIGOS_SALIDA_HOSPITAL)
SIMILITUD_MINIMA = 0.90  # dos nombres de hospital con esta similitud se consideran el mismo establecimiento (autorreferencia aproximada)
MIN_EPISODIOS_EMBUDO = 100  # filtro de visualización: solo los hospitales fuera de los límites con al menos estos egresos llevan etiqueta


# ----------------------------------------------------------------------------- nombres
def normalizar_nombre(nombre: str) -> str:
    """Mayúsculas, sin tildes ni signos y con espacios colapsados (para comparar nombres de hospital)."""
    sin = "".join(c for c in unicodedata.normalize("NFD", nombre.upper()) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^A-Z0-9]+", " ", sin).strip()


def similitud_nombres(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def mapa_nombres(cat: catalogos.Catalogos) -> dict[str, str]:
    """Código de hospital -> nombre normalizado del catálogo (solo en memoria: los nombres no se escriben en las salidas)."""
    return {c: normalizar_nombre(n) for c, n in zip(cat.hospitales["cod_hospital"], cat.hospitales["nombre"])}


# ------------------------------------------------------------------ agregados de un año
def _ah(edad_adulto: int) -> pl.Expr:
    return ((pl.col("edad") >= edad_adulto) & (pl.col("actividad") == ACTIVIDAD_HOSP)).fill_null(False)


def _entrante() -> pl.Expr:
    return pl.col("procedencia").str.to_uppercase().str.contains(TEXTO_ENTRANTE).fill_null(False)


def procesar_anio(base: pl.DataFrame, edad_adulto: int) -> dict[str, Any]:
    """Agregados de un año a partir de la base tipada con extras. Con desenlace (``rol``) agrega mortalidad y los episodios mínimos."""
    c = pl.col
    con_res = "rol" in base.columns
    adulto = (c("edad") >= edad_adulto).fill_null(False)
    out: dict[str, Any] = {"n": base.height}
    out["hospitales"] = base.group_by("hospital").agg(
        pl.len().alias("n_egresos"),
        adulto.sum().alias("n_adultos"),
        (c("edad") < edad_adulto).fill_null(False).sum().alias("n_menores"),
        c("edad").is_null().sum().alias("n_edad_desconocida"),
    )
    out["procedencia_total"] = base.group_by("procedencia").agg(pl.len().alias("n"))
    out["procedencia_nombres"] = base.filter(c("hospproc").is_not_null()).group_by(["procedencia", "hospital", "hospproc"]).agg(pl.len().alias("n"))
    out["id_validos"] = base["id_num"].drop_nulls().unique()
    if not con_res:
        return out
    d = base.filter(c("D"))
    out["hospitales_d"] = d.group_by("hospital").agg(pl.len().alias("n_D"), (c("rol") == "EVENTO").sum().alias("k_D"))
    out["estratos_d"] = d.group_by(["hospital", _grupo_edad_expr().alias("grupo_edad"), "sexo"]).agg(
        pl.len().alias("n"), (c("rol") == "EVENTO").sum().alias("k")
    )
    out["entrantes"] = base.filter(c("B") & c("conocido")).group_by(_entrante().alias("entrante")).agg(
        pl.len().alias("n"), (c("rol") == "EVENTO").sum().alias("k")
    )
    out["episodios"] = base.select(
        c("anio").cast(pl.Int32),
        c("id_num"),
        c("ing"),
        c("alta"),
        c("hospital"),
        _ah(edad_adulto).alias("es_ah"),
        (_ah(edad_adulto) & c("etiqueta_roles").is_in(list(ETIQUETAS_SALIDA))).fill_null(False).alias("salida"),
        (_ah(edad_adulto) & _entrante()).fill_null(False).alias("entrante_ah"),
    )
    return out


# --------------------------------------------------------------------------- hospitales
def tabla_hospitales_anio(por_anio: dict[int, dict[str, Any]], cat: catalogos.Catalogos, u: config.Umbrales) -> pl.DataFrame:
    """Perfil de cada hospital y año: volumen, menores, catálogo y marca de hospital pediátrico o monográfico (decisión 5)."""
    umbral = float(u["hospital_pediatrico_prop"])
    partes = [d["hospitales"].with_columns(pl.lit(a, dtype=pl.Int64).alias("anio")) for a, d in sorted(por_anio.items())]
    df = pl.concat(partes).rename({"hospital": "cod_hospital"})
    df = df.with_columns(*[pl.col(c).cast(pl.Int64) for c in ("n_egresos", "n_adultos", "n_menores", "n_edad_desconocida")])
    df = df.with_columns(
        (pl.col("n_menores") / pl.col("n_egresos")).alias("prop_menores"),
        pl.when(pl.col("cod_hospital").is_in(sorted(cat.hospitales_set))).then(pl.lit("si")).otherwise(pl.lit("no")).alias("en_catalogo"),
    ).with_columns(pl.when(pl.col("prop_menores") > umbral).then(pl.lit("si")).otherwise(pl.lit("no")).alias("pediatrico"))
    return df.select(
        ["anio", "cod_hospital", "en_catalogo", "n_egresos", "n_adultos", "n_menores", "n_edad_desconocida", "prop_menores", "pediatrico"]
    ).sort(["anio", "cod_hospital"])


def esperadas_por_hospital(estratos: pl.DataFrame) -> pl.DataFrame:
    """Muertes esperadas por estandarización indirecta: tasas nacionales por edad y sexo aplicadas a la composición de cada hospital."""
    tasa = (
        estratos.group_by(["grupo_edad", "sexo"])
        .agg(pl.col("n").sum().alias("N"), pl.col("k").sum().alias("K"))
        .with_columns((pl.col("K") / pl.col("N")).alias("tasa"))
        .select("grupo_edad", "sexo", "tasa")
    )
    j = estratos.join(tasa, on=["grupo_edad", "sexo"], how="left")
    return j.group_by("hospital").agg((pl.col("n") * pl.col("tasa")).sum().alias("esperadas"))


def tabla_hospitales_volumen(por_anio: dict[int, dict[str, Any]], u: config.Umbrales) -> pl.DataFrame:
    """Egresos elegibles (cohorte D) y muertes esperadas por hospital y año, con el cumplimiento del volumen mínimo propuesto (decisión 6)."""
    vol_n, vol_m = int(u["volumen_minimo_egresos"]), float(u["muertes_esperadas_minimas"])
    partes = []
    for a, d in sorted(por_anio.items()):
        if "hospitales_d" not in d:
            continue
        esp = esperadas_por_hospital(d["estratos_d"])
        h = d["hospitales_d"].join(esp, on="hospital", how="left").with_columns(pl.lit(a, dtype=pl.Int64).alias("anio"))
        partes.append(h)
    if not partes:
        return pl.DataFrame(schema={"anio": pl.Int64, "cod_hospital": pl.String, "n_D": pl.Int64, "esperados_D": pl.Float64})
    df = pl.concat(partes).rename({"hospital": "cod_hospital", "esperadas": "esperados_D"})
    df = df.with_columns(pl.col("n_D").cast(pl.Int64), pl.col("k_D").cast(pl.Int64))
    df = df.with_columns(
        (pl.col("n_D") >= vol_n).alias("_a"),
        (pl.col("esperados_D") >= vol_m).alias("_b"),
    ).with_columns(
        pl.when(pl.col("_a")).then(pl.lit("si")).otherwise(pl.lit("no")).alias("cumple_egresos"),
        pl.when(pl.col("_b")).then(pl.lit("si")).otherwise(pl.lit("no")).alias("cumple_muertes"),
        pl.when(pl.col("_a") & pl.col("_b")).then(pl.lit("si")).otherwise(pl.lit("no")).alias("cumple_volumen"),
    )
    return df.select(["anio", "cod_hospital", "n_D", "k_D", "esperados_D", "cumple_egresos", "cumple_muertes", "cumple_volumen"]).sort(["anio", "cod_hospital"])


def tabla_volumen_umbrales(vol: pl.DataFrame, hosp: pl.DataFrame, u: config.Umbrales) -> pl.DataFrame:
    """Cuántos hospitales cumplen cada combinación de egresos mínimos y muertes esperadas mínimas, por año (decisión 6)."""
    filas = []
    ped = hosp.filter(pl.col("pediatrico") == "si").select("anio", "cod_hospital").with_columns(pl.lit(True).alias("_ped"))
    v = vol.join(ped, on=["anio", "cod_hospital"], how="left").with_columns(pl.col("_ped").fill_null(False))
    for a in sorted(v["anio"].unique().to_list()):
        s = v.filter(pl.col("anio") == a)
        for n_min in u["volumen_grid_egresos"]:
            for m_min in u["volumen_grid_muertes"]:
                ok_n, ok_m = s["n_D"] >= n_min, s["esperados_D"].fill_null(0.0) >= m_min
                ambos = ok_n & ok_m
                filas.append({
                    "anio": a, "egresos_min": int(n_min), "muertes_esperadas_min": float(m_min), "n_hospitales": s.height, "n_cumplen_egresos": int(ok_n.sum()),
                    "n_cumplen_muertes": int(ok_m.sum()), "n_cumplen_ambos": int(ambos.sum()), "n_cumplen_ambos_no_pediatricos": int((ambos & ~s["_ped"]).sum()),
                    "prop_cumplen_ambos": float(ambos.sum()) / s.height if s.height else None,
                })
    return pl.DataFrame(filas, schema={
        "anio": pl.Int64, "egresos_min": pl.Int64, "muertes_esperadas_min": pl.Float64, "n_hospitales": pl.Int64, "n_cumplen_egresos": pl.Int64,
        "n_cumplen_muertes": pl.Int64, "n_cumplen_ambos": pl.Int64, "n_cumplen_ambos_no_pediatricos": pl.Int64, "prop_cumplen_ambos": pl.Float64,
    })


def tabla_hospitales_resumen(hosp: pl.DataFrame, vol: pl.DataFrame, u: config.Umbrales) -> pl.DataFrame:
    """Una fila por año: hospitales, fuera de catálogo, pediátricos, cumplen el volumen propuesto y proporción de celdas hospital-año bajo el mínimo."""
    celda = int(u["celda_minima"])
    filas = []
    for a in sorted(hosp["anio"].unique().to_list()):
        h = hosp.filter(pl.col("anio") == a)
        v = vol.filter(pl.col("anio") == a)
        filas.append({
            "anio": a, "n_hospitales": h.height, "n_fuera_catalogo": int((h["en_catalogo"] == "no").sum()), "n_pediatricos": int((h["pediatrico"] == "si").sum()),
            "n_hospitales_con_cohorte": v.height if v.height else None,
            "n_cumplen_volumen": int((v["cumple_volumen"] == "si").sum()) if v.height else None,
            "prop_celdas_bajo_minimo": float((v["n_D"] < celda).sum()) / v.height if v.height else None,
        })
    return pl.DataFrame(filas, schema={
        "anio": pl.Int64, "n_hospitales": pl.Int64, "n_fuera_catalogo": pl.Int64, "n_pediatricos": pl.Int64, "n_hospitales_con_cohorte": pl.Int64,
        "n_cumplen_volumen": pl.Int64, "prop_celdas_bajo_minimo": pl.Float64,
    })


# ------------------------------------------------------------------------------ embudo
def _res_anios(por_anio: dict[int, dict[str, Any]]) -> list[int]:
    return sorted(a for a, d in por_anio.items() if "hospitales_d" in d)


def tabla_embudo(por_anio: dict[int, dict[str, Any]], u: config.Umbrales) -> pl.DataFrame:
    """Embudo crudo de mortalidad por hospital en la cohorte D, por año y para el período de resultado (sin ajuste por riesgo)."""
    celda = int(u["celda_minima"])
    z95, z998, winsor = float(u["z_95"]), float(u["z_998"]), float(u["winsor"])
    res = _res_anios(por_anio)
    periodos = [(str(a), [a]) for a in res] + ([(etiqueta_periodo(res), res)] if len(res) > 1 else [])
    filas = []
    for nombre, lista in periodos:
        h = pl.concat([por_anio[a]["hospitales_d"] for a in lista]).group_by("hospital").agg(pl.col("n_D").sum().alias("n"), pl.col("k_D").sum().alias("k")).sort("hospital")
        total_n, total_k = int(h["n"].sum()), int(h["k"].sum())
        if total_n == 0:
            continue
        p0 = total_k / total_n
        n, k = h["n"].to_numpy().astype(float), h["k"].to_numpy().astype(float)
        z = estadistica.puntaje_z_proporcion(k, n, p0)
        valido = (n >= celda) & np.isfinite(z)
        phi = estadistica.phi_spiegelhalter(z[valido], winsor) if valido.any() else 1.0
        l95 = estadistica.limites_proporcion(p0, n, z95, phi)
        l998 = estadistica.limites_proporcion(p0, n, z998, phi)
        for i, cod in enumerate(h["hospital"].to_list()):
            p = k[i] / n[i] if n[i] else None
            def fuera(lim):
                if not valido[i] or p is None:
                    return "no evaluable"
                return "alto" if p > lim[1][i] else ("bajo" if p < lim[0][i] else "no")
            filas.append({
                "periodo": nombre, "cod_hospital": cod, "n": int(n[i]), "k": int(k[i]), "p": p, "p0": p0, "phi": phi,
                "z": float(z[i]) if np.isfinite(z[i]) else None, "z_ajustado": float(z[i] / np.sqrt(phi)) if np.isfinite(z[i]) else None,
                "lim95_bajo": float(l95[0][i]), "lim95_alto": float(l95[1][i]), "lim998_bajo": float(l998[0][i]), "lim998_alto": float(l998[1][i]),
                "fuera_95": fuera(l95), "fuera_998": fuera(l998),
            })
    return pl.DataFrame(filas, schema={
        "periodo": pl.String, "cod_hospital": pl.String, "n": pl.Int64, "k": pl.Int64, "p": pl.Float64, "p0": pl.Float64, "phi": pl.Float64, "z": pl.Float64,
        "z_ajustado": pl.Float64, "lim95_bajo": pl.Float64, "lim95_alto": pl.Float64, "lim998_bajo": pl.Float64, "lim998_alto": pl.Float64,
        "fuera_95": pl.String, "fuera_998": pl.String,
    })


# ----------------------------------------------------------------- historial y enlaces
def regimenes_de(ids_por_anio: dict[int, pl.Series], u: config.Umbrales) -> list[list[int]]:
    """Espacios de identificadores (años consecutivos que comparten pacientes), con la regla de quiebre de la Tarea 5."""
    anios = sorted(ids_por_anio)
    if len(anios) < 2:
        return [anios] if anios else []
    traslape = calidad.traslape_ids(ids_por_anio)
    consecutivos = {(r["anio_a"], r["anio_b"]): r["prop_min"] for r in traslape.filter(pl.col("consecutivo") == "si").iter_rows(named=True)}
    return calidad.regimenes(anios, calidad.detectar_quiebres(consecutivos, float(u["id_quiebre_ratio"])))


def historial_regimen(sub: pl.DataFrame, inicio: dt.date, ventana: int) -> pl.DataFrame:
    """Factibilidad del historial de un espacio de identificadores: conteos por año.

    ``sub`` trae los episodios del régimen con ``idx``, ``anio``, ``id_num``, ``ing`` y ``alta``. Un episodio es viable si su identificador es válido y su ventana de
    ``ventana`` días queda dentro del régimen. El egreso previo se busca estrictamente antes del ingreso y dentro de la ventana (no cuenta el reingreso del mismo día).
    """
    c = pl.col
    v = sub.with_columns(
        c("id_num").is_not_null().alias("id_ok"),
        (c("ing").is_not_null() & (c("ing").dt.offset_by(f"-{ventana}d") >= pl.lit(inicio))).fill_null(False).alias("vent_ok"),
    )
    izq = v.filter(c("id_ok") & c("ing").is_not_null()).select("idx", "id_num", "ing").sort("ing")
    der = (
        v.filter(c("id_ok") & c("ing").is_not_null() & c("alta").is_not_null() & (c("alta") >= c("ing")))
        .select("id_num", c("alta").alias("alta_prev"))
        .sort("alta_prev")
    )
    if izq.height and der.height:
        j = izq.join_asof(
            der, left_on="ing", right_on="alta_prev", by="id_num", strategy="backward", allow_exact_matches=False, check_sortedness=False
        )  # ambas tablas ya vienen ordenadas por fecha
        j = j.select("idx", ((c("ing") - c("alta_prev")).dt.total_days() <= ventana).fill_null(False).alias("con_previo"))
        v = v.join(j, on="idx", how="left").with_columns(c("con_previo").fill_null(False))
    else:
        v = v.with_columns(pl.lit(False).alias("con_previo"))
    v = v.with_columns((c("id_ok") & c("vent_ok")).alias("viable"))
    return v.group_by("anio").agg(
        pl.len().alias("n_episodios"), c("id_ok").sum().alias("n_id_valido"), c("vent_ok").sum().alias("n_ventana_completa"),
        c("viable").sum().alias("n_viable"), (c("viable") & c("con_previo")).sum().alias("n_con_previo"),
    ).sort("anio")


def enlaces_regimen(sub: pl.DataFrame, inicio: dt.date, fin: dt.date, ventanas: Sequence[int]) -> pl.DataFrame:
    """Enlace de traslados salientes con un ingreso posterior en otro hospital y de entrantes con un traslado previo, dentro de un espacio de identificadores.

    Elegibilidad: un traslado saliente solo cuenta para una ventana W si ``alta + W`` no pasa de ``fin`` (último día observado del régimen), y un entrante si
    ``ingreso - W`` no es anterior a ``inicio``. Lo que cae fuera no se cuenta como no enlazado: se excluye del denominador.
    """
    c = pl.col
    wmax = int(max(ventanas))
    sal = sub.filter(c("salida")).select("idx", "anio", "id_num", "alta", "hospital")
    dest = sub.filter(c("id_num").is_not_null() & c("ing").is_not_null()).select(
        c("idx").alias("idx_d"), "id_num", c("ing").alias("ing_d"), c("hospital").alias("hosp_d"), c("entrante_ah").alias("ent_d")
    )
    pares = (
        sal.filter(c("id_num").is_not_null() & c("alta").is_not_null())
        .join(dest, on="id_num")
        .with_columns((c("ing_d") - c("alta")).dt.total_days().alias("delta"))
        .filter((c("delta") >= 0) & (c("delta") <= wmax) & (c("hospital") != c("hosp_d")))
    )
    min_s = pares.group_by("idx").agg(c("delta").min().alias("delta_min"))
    min_d = pares.filter(c("ent_d")).group_by("idx_d").agg(c("delta").min().alias("delta_min"))
    s2 = sal.join(min_s, on="idx", how="left")
    ent = sub.filter(c("entrante_ah")).select(c("idx").alias("idx_d"), "anio", "id_num", "ing").join(min_d, on="idx_d", how="left")
    filas = []
    for w in sorted(int(x) for x in ventanas):
        elegible_s = c("id_num").is_not_null() & c("alta").is_not_null() & (c("alta").dt.offset_by(f"{w}d") <= pl.lit(fin))
        g = s2.group_by("anio").agg(
            pl.len().alias("n_total"), c("id_num").is_null().sum().alias("n_id_invalido"), elegible_s.sum().alias("n_elegibles"),
            (elegible_s & (c("delta_min") <= w)).sum().alias("n_enlazados"),
        )
        filas += [{"direccion": "saliente", "ventana_dias": w, **r} for r in g.iter_rows(named=True)]
        elegible_e = c("id_num").is_not_null() & c("ing").is_not_null() & (c("ing").dt.offset_by(f"-{w}d") >= pl.lit(inicio))
        g = ent.group_by("anio").agg(
            pl.len().alias("n_total"), c("id_num").is_null().sum().alias("n_id_invalido"), elegible_e.sum().alias("n_elegibles"),
            (elegible_e & (c("delta_min") <= w)).sum().alias("n_enlazados"),
        )
        filas += [{"direccion": "entrante", "ventana_dias": w, **r} for r in g.iter_rows(named=True)]
    return pl.DataFrame(filas, schema={
        "direccion": pl.String, "ventana_dias": pl.Int64, "anio": pl.Int32, "n_total": pl.Int64, "n_id_invalido": pl.Int64, "n_elegibles": pl.Int64, "n_enlazados": pl.Int64,
    })


def historial_y_enlaces(episodios: pl.DataFrame, regimenes: list[list[int]], u: config.Umbrales) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Aplica historial y enlaces a cada régimen (solo años con desenlace) y devuelve las tablas por año y régimen."""
    ventana = int(u["ventana_historial_dias"])
    ventanas = [int(x) for x in u["ventanas_traslado_dias"]]
    hist, enl = [], []
    for reg in regimenes:
        if not reg:
            continue
        sub = episodios.filter(pl.col("anio").is_in(reg)).with_row_index("idx")
        if sub.height == 0:
            continue
        inicio, fin = dt.date(min(reg), 1, 1), dt.date(max(reg), 12, 31)
        etiqueta = etiqueta_periodo(reg)
        hist.append(historial_regimen(sub, inicio, ventana).with_columns(pl.lit(etiqueta).alias("regimen"), pl.col("anio").cast(pl.Int64)))
        enl.append(enlaces_regimen(sub, inicio, fin, ventanas).with_columns(pl.lit(etiqueta).alias("regimen"), pl.col("anio").cast(pl.Int64)))
    esq_h = {"anio": pl.Int64, "n_episodios": pl.UInt32, "n_id_valido": pl.UInt32, "n_ventana_completa": pl.UInt32, "n_viable": pl.UInt32, "n_con_previo": pl.UInt32, "regimen": pl.String}
    h = pl.concat(hist) if hist else pl.DataFrame(schema=esq_h)
    e = pl.concat(enl) if enl else pl.DataFrame(schema={"direccion": pl.String, "ventana_dias": pl.Int64, "anio": pl.Int64, "n_total": pl.Int64, "n_id_invalido": pl.Int64,
                                                          "n_elegibles": pl.Int64, "n_enlazados": pl.Int64, "regimen": pl.String})
    return h, e


def tabla_historial(hist: pl.DataFrame, u: config.Umbrales) -> pl.DataFrame:
    minimo = float(u["historial_factibilidad_minima"])
    if hist.height == 0:
        return hist
    df = hist.with_columns(*[pl.col(c).cast(pl.Int64) for c in ("n_episodios", "n_id_valido", "n_ventana_completa", "n_viable", "n_con_previo")])
    df = df.with_columns(
        (pl.col("n_id_valido") / pl.col("n_episodios")).alias("prop_id_valido"),
        (pl.col("n_ventana_completa") / pl.col("n_episodios")).alias("prop_ventana_completa"),
        (pl.col("n_viable") / pl.col("n_episodios")).alias("factibilidad"),
        (pl.col("n_con_previo") / pl.col("n_viable")).alias("prop_con_previo"),
    ).with_columns(pl.when(pl.col("factibilidad") >= minimo).then(pl.lit("si")).otherwise(pl.lit("no")).alias("cumple_minimo"))
    return df.select([
        "anio", "regimen", "n_episodios", "n_id_valido", "n_ventana_completa", "n_viable", "n_con_previo", "prop_id_valido", "prop_ventana_completa",
        "factibilidad", "prop_con_previo", "cumple_minimo",
    ]).sort("anio")


def tabla_enlaces(enl: pl.DataFrame, u: config.Umbrales) -> pl.DataFrame:
    """Tasa de enlace por dirección, ventana y año (y para todos los años juntos), con IC de Wilson."""
    if enl.height == 0:
        return enl
    z = float(u["z_95"])
    cols = ["n_total", "n_id_invalido", "n_elegibles", "n_enlazados"]
    por_anio = enl.select(["direccion", "ventana_dias", "anio", *cols]).with_columns(pl.col("anio").cast(pl.Int64))
    anios = sorted(por_anio["anio"].unique().to_list())
    df = con_periodos(por_anio, {etiqueta_periodo(anios): anios} if len(anios) > 1 else {}, ["direccion", "ventana_dias"], cols)
    df = df.with_columns((pl.col("n_enlazados") / pl.col("n_elegibles")).alias("tasa_enlace"))
    df = con_wilson(df, "n_enlazados", "n_elegibles", "", z)
    return df.select(["periodo", "direccion", "ventana_dias", *cols, "tasa_enlace", "ic_bajo", "ic_alto"]).sort(["direccion", "ventana_dias", "periodo"])


# -------------------------------------------------------------------- procedencia y entrantes
def tabla_autorref(por_anio: dict[int, dict[str, Any]], cat: catalogos.Catalogos) -> pl.DataFrame:
    """Hospital de procedencia frente al hospital del egreso, por año y tipo de procedencia (los nombres se comparan en memoria)."""
    nombres = mapa_nombres(cat)
    cache: dict[tuple[str, str], tuple[bool, bool]] = {}
    filas = []
    for a, d in sorted(por_anio.items()):
        total = dict(d["procedencia_total"].iter_rows())
        cuentas: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])   # con hospital de procedencia, evaluables, exacta, aproximada
        for proc, hosp, hp, n in d["procedencia_nombres"].iter_rows():
            fila = cuentas[proc]
            fila[0] += n
            nc = nombres.get(hosp)
            if nc is None:
                continue
            clave = (nc, hp)
            if clave not in cache:
                nh = normalizar_nombre(hp)
                cache[clave] = (nh == nc, nh == nc or similitud_nombres(nh, nc) >= SIMILITUD_MINIMA)
            exacta, aprox = cache[clave]
            fila[1] += n
            fila[2] += n if exacta else 0
            fila[3] += n if aprox else 0
        todo = [sum(v[i] for v in cuentas.values()) for i in range(4)]
        for proc in sorted(set(total) | set(cuentas)):
            v = cuentas.get(proc, [0, 0, 0, 0])
            filas.append({"anio": a, "tipo_procedencia": proc, "n_episodios": int(total.get(proc, 0)), "n_con_hospproc": v[0], "n_evaluables": v[1],
                          "n_autorref_exacta": v[2], "n_autorref_aprox": v[3]})
        filas.append({"anio": a, "tipo_procedencia": TODAS, "n_episodios": int(sum(total.values())), "n_con_hospproc": todo[0], "n_evaluables": todo[1],
                      "n_autorref_exacta": todo[2], "n_autorref_aprox": todo[3]})
    df = pl.DataFrame(filas, schema={
        "anio": pl.Int64, "tipo_procedencia": pl.String, "n_episodios": pl.Int64, "n_con_hospproc": pl.Int64, "n_evaluables": pl.Int64,
        "n_autorref_exacta": pl.Int64, "n_autorref_aprox": pl.Int64,
    })
    return df.with_columns(
        (pl.col("n_con_hospproc") / pl.col("n_episodios")).alias("prop_con_hospproc"),
        (pl.col("n_autorref_exacta") / pl.col("n_evaluables")).alias("prop_autorref_exacta"),
        (pl.col("n_autorref_aprox") / pl.col("n_evaluables")).alias("prop_autorref_aprox"),
    ).sort(["anio", "tipo_procedencia"])


def tabla_entrantes(por_anio: dict[int, dict[str, Any]], u: config.Umbrales) -> pl.DataFrame:
    """Mortalidad de los traslados entrantes y de los demás ingresos en adultos de la cohorte dura con desenlace conocido."""
    z = float(u["z_95"])
    res = _res_anios(por_anio)
    filas = []
    periodos = [(str(a), [a]) for a in res] + ([(etiqueta_periodo(res), res)] if len(res) > 1 else [])
    for nombre, lista in periodos:
        cuenta = {True: [0, 0], False: [0, 0]}
        for a in lista:
            for ent, n, k in por_anio[a]["entrantes"].iter_rows():
                cuenta[bool(ent)][0] += int(n)
                cuenta[bool(ent)][1] += int(k)
        rr, lo, hi = estadistica.razon_de_proporciones(cuenta[True][1], cuenta[True][0], cuenta[False][1], cuenta[False][0], z)
        for ent in (True, False):
            n, k = cuenta[ent]
            filas.append({"periodo": nombre, "entrante": "si" if ent else "no", "n": n, "n_fallecidos": k, "rr": rr if ent else None, "rr_bajo": lo if ent else None,
                          "rr_alto": hi if ent else None})
    df = pl.DataFrame(filas, schema={"periodo": pl.String, "entrante": pl.String, "n": pl.Int64, "n_fallecidos": pl.Int64, "rr": pl.Float64, "rr_bajo": pl.Float64,
                                     "rr_alto": pl.Float64})
    df = df.with_columns((pl.col("n_fallecidos") / pl.col("n")).alias("mortalidad"))
    df = con_wilson(df, "n_fallecidos", "n", "", z)
    return df.select(["periodo", "entrante", "n", "n_fallecidos", "mortalidad", "ic_bajo", "ic_alto", "rr", "rr_bajo", "rr_alto"])


# ---------------------------------------------------------------------------- ejecución
def calcular(ctx: Contexto, cat: catalogos.Catalogos | None = None, regimenes: list[list[int]] | None = None) -> dict[str, Any]:
    """Recorre los años una vez. ``regimenes`` permite pasar los espacios de identificadores de la Tarea 5; si no, se calculan con 2019 a 2023."""
    cat = cat or catalogos.cargar_catalogos()
    e, e3 = elixhauser.cargar(), elixhauser.cargar(solo_3c=True)
    u = ctx.umbrales
    edad_adulto = int(u["edad_adulto"])
    por_anio: dict[int, dict[str, Any]] = {}
    for anio in ctx.fuente.anios():
        base = leer_base(ctx, anio, cat, e, e3, extras=True)
        por_anio[anio] = procesar_anio(base, edad_adulto)
        del base
    res = _res_anios(por_anio)
    r: dict[str, Any] = {"por_anio": por_anio}
    r["hospitales"] = tabla_hospitales_anio(por_anio, cat, u)
    r["autorref"] = tabla_autorref(por_anio, cat)
    if not res:
        return r
    r["volumen"] = tabla_hospitales_volumen(por_anio, u)
    r["umbrales"] = tabla_volumen_umbrales(r["volumen"], r["hospitales"], u)
    r["resumen_anio"] = tabla_hospitales_resumen(r["hospitales"], r["volumen"], u)
    r["embudo"] = tabla_embudo(por_anio, u)
    r["entrantes"] = tabla_entrantes(por_anio, u)
    if regimenes is None:
        regimenes = regimenes_de({a: por_anio[a]["id_validos"] for a in res}, u)
    regimenes = [[a for a in reg if a in res] for reg in regimenes]
    episodios = pl.concat([por_anio[a]["episodios"] for a in res])
    hist, enl = historial_y_enlaces(episodios, [reg for reg in regimenes if reg], u)
    r["regimenes"] = [reg for reg in regimenes if reg]
    r["historial"] = tabla_historial(hist, u)
    r["enlaces"] = tabla_enlaces(enl, u)
    r["resumen"] = resumen(r, u)
    return r


def resumen(r: dict[str, Any], u: config.Umbrales) -> pl.DataFrame:
    """Veredictos de los criterios de las decisiones 1, 2 y 13 (una fila)."""
    out: dict[str, Any] = {
        "n_regimenes": len(r.get("regimenes", [])), "regimenes": "; ".join(etiqueta_periodo(x) for x in r.get("regimenes", [])),
    }
    a = r["autorref"].filter(pl.col("tipo_procedencia") == TODAS)
    n_eval, n_ex, n_ap = int(a["n_evaluables"].sum()), int(a["n_autorref_exacta"].sum()), int(a["n_autorref_aprox"].sum())
    out["autorref_prop_exacta"] = n_ex / n_eval if n_eval else None
    out["autorref_prop_aprox"] = n_ap / n_eval if n_eval else None
    maximo = float(u["autorreferencia_procedencia_max"])
    out["autorref_supera_maximo"] = ("si" if (out["autorref_prop_aprox"] or 0) > maximo else "no") if n_eval else "no evaluable"
    h = r["historial"]
    if h.height:
        ult = h.sort("anio").row(h.height - 1, named=True)
        out["historial_anio"] = int(ult["anio"])
        out["historial_factibilidad"] = ult["factibilidad"]
        out["historial_cumple_minimo"] = ult["cumple_minimo"]
        out["historial_prop_con_previo"] = ult["prop_con_previo"]
    e = r["enlaces"]
    if e.height:
        w = int(max(u["ventanas_traslado_dias"]))
        anios = sorted(int(p) for p in e["periodo"].unique().to_list() if p.isdigit())
        if anios:
            f = e.filter((pl.col("direccion") == "saliente") & (pl.col("ventana_dias") == w) & (pl.col("periodo") == str(anios[-1])))
            if f.height:
                out["enlace_anio"] = anios[-1]
                out["enlace_ventana_dias"] = w
                out["enlace_tasa"] = f["tasa_enlace"][0]
                out["enlace_cumple_minimo"] = "si" if (f["tasa_enlace"][0] or 0) >= float(u["enlace_traslado_minimo"]) else "no"
                out["enlace_n_elegibles"] = int(f["n_elegibles"][0])
    emb = r["embudo"]
    res = _res_anios(r["por_anio"])
    if emb.height and len(res) > 1:
        periodo = etiqueta_periodo(res)
        s = emb.filter(pl.col("periodo") == periodo)
        ev = s.filter(pl.col("fuera_95") != "no evaluable")
        out.update({
            "embudo_periodo": periodo, "embudo_p0": float(s["p0"][0]), "embudo_phi": float(s["phi"][0]), "embudo_n_evaluados": ev.height,
            "embudo_n_fuera_95": int((ev["fuera_95"] != "no").sum()), "embudo_n_fuera_998": int((ev["fuera_998"] != "no").sum()),
        })
    return pl.DataFrame([out])


def _escribir(ctx: Contexto, r: dict[str, Any]) -> None:
    TA = TAREA
    ctx.escribir_tabla("t23_hospitales_anio.csv", r["hospitales"], tarea=TA, descripcion="Perfil de cada hospital por año: volumen, menores y catálogo (por código)",
                       cols_n=["n_egresos", "n_adultos", "n_menores", "n_edad_desconocida"], dependientes=["prop_menores", "pediatrico"])
    # _columna y _clave unen n_episodios con la distribución de TIPO_PROCEDENCIA de la Tarea 4 (mismas celdas): hereda lo que esa tabla ocultó
    ctx.escribir_tabla("t23_procedencia_autorref.csv", r["autorref"].with_columns(pl.lit("TIPO_PROCEDENCIA").alias("_columna"), pl.col("tipo_procedencia").alias("_clave")),
                       tarea=TA, descripcion="Hospital de procedencia frente al hospital del egreso, por año y tipo de procedencia",
                       cols_n=["n_episodios", "n_con_hospproc", "n_evaluables", "n_autorref_exacta", "n_autorref_aprox"],
                       dependientes=["prop_con_hospproc", "prop_autorref_exacta", "prop_autorref_aprox"], auxiliares=["_columna", "_clave"])
    if "volumen" not in r:
        return
    ctx.escribir_tabla("t23_hospitales_volumen.csv", r["volumen"].drop("k_D"), tarea=TA,
                       descripcion="Egresos elegibles y muertes esperadas por hospital y año (decisión 6)",
                       cols_n=["n_D"], dependientes=["esperados_D", "cumple_egresos", "cumple_muertes", "cumple_volumen"])
    ctx.escribir_tabla("t23_volumen_umbrales.csv", r["umbrales"], tarea=TA, descripcion="Hospitales que cumplen cada combinación de volumen mínimo (decisión 6)")
    ctx.escribir_tabla("t23_hospitales_resumen.csv", r["resumen_anio"], tarea=TA, descripcion="Hospitales por año: catálogo, pediátricos y volumen")
    ctx.escribir_tabla("t23_embudo.csv", r["embudo"], tarea=TA, descripcion="Embudo crudo exploratorio de mortalidad por hospital (cohorte D)",
                       cols_n=["n", "k"], dependientes=["p", "z", "z_ajustado", "lim95_bajo", "lim95_alto", "lim998_bajo", "lim998_alto", "fuera_95", "fuera_998"])
    ctx.escribir_tabla("t23_resultado_entrantes.csv", r["entrantes"], tarea=TA, descripcion="Mortalidad de los traslados entrantes y de los demás ingresos",
                       cols_n=["n", "n_fallecidos"], dependientes=["mortalidad", "ic_bajo", "ic_alto", "rr", "rr_bajo", "rr_alto"])
    ctx.escribir_tabla("t23_historial_factibilidad.csv", r["historial"], tarea=TA, descripcion="Factibilidad del historial de 12 meses por año (decisión 1)",
                       cols_n=["n_episodios", "n_id_valido", "n_ventana_completa", "n_viable", "n_con_previo"],
                       dependientes=["prop_id_valido", "prop_ventana_completa", "factibilidad", "prop_con_previo", "cumple_minimo"])
    ctx.escribir_tabla("t23_enlaces_traslado.csv", r["enlaces"], tarea=TA, descripcion="Enlace de traslados salientes y entrantes por ventana y año (decisiones 13 y 16)",
                       cols_n=["n_total", "n_id_invalido", "n_elegibles", "n_enlazados"], dependientes=["tasa_enlace", "ic_bajo", "ic_alto"])
    # el resumen y las métricas copian celdas de los enlaces: se leen de la tabla ya protegida (supresión complementaria, cambio 1.6)
    pub = ctx.leer_tabla("t23_enlaces_traslado.csv", columnas_texto=["periodo", "direccion"])
    r["enlaces_publicado"] = pub
    res = r["resumen"]
    if "enlace_anio" in res.columns and res["enlace_anio"][0] is not None:
        f = pub.filter((pl.col("direccion") == "saliente") & (pl.col("ventana_dias") == res["enlace_ventana_dias"][0])
                       & (pl.col("periodo") == str(res["enlace_anio"][0])))
        if f.height:
            res = res.with_columns(pl.lit(f["tasa_enlace"][0], dtype=pl.Float64).alias("enlace_tasa"),
                                   pl.lit(f["n_elegibles"][0], dtype=pl.Int64).alias("enlace_n_elegibles"))
    r["resumen"] = res
    ctx.escribir_tabla("t23_resumen_hospitales.csv", res, tarea=TA, descripcion="Veredictos de los criterios de hospitales, historial y traslados",
                       cols_n=["enlace_n_elegibles"])


def registrar_metricas(ctx: Contexto, r: dict[str, Any]) -> None:
    m = ctx.registrar_metrica
    extracto = ctx.etiqueta == "extracto"
    aviso = " Extracto: el ingreso siguiente de un paciente casi nunca está en el mismo extracto; no interpretable." if extracto else ""
    for row in r["hospitales"].group_by("anio").agg(pl.col("pediatrico").eq("si").sum().alias("n_ped")).iter_rows(named=True):
        m(f"hospitales.n_pediatricos.{row['anio']}", row["n_ped"], tarea=TAREA, tabla="t23_hospitales_anio.csv", alcance=str(row["anio"]),
          nota="hospitales con más de la mitad de sus egresos en menores")
    from .proteccion import suma_publicada

    a = ctx.leer_tabla("t23_procedencia_autorref.csv", columnas_texto=["tipo_procedencia"]).filter(pl.col("tipo_procedencia") == TODAS)
    n_eval = suma_publicada(a["n_evaluables"])
    if n_eval:
        ex, ap = suma_publicada(a["n_autorref_exacta"]), suma_publicada(a["n_autorref_aprox"])
        m("procedencia.autorref_pct_exacta", None if ex is None else 100.0 * ex / n_eval, tarea=TAREA, tabla="t23_procedencia_autorref.csv",
          nota="HOSPPROCEDENCIA igual al hospital del egreso (nombres normalizados iguales)")
        m("procedencia.autorref_pct_aprox", None if ap is None else 100.0 * ap / n_eval, tarea=TAREA, tabla="t23_procedencia_autorref.csv",
          nota="como la anterior, aceptando nombres muy parecidos")
    if "volumen" not in r:
        return
    for row in r["resumen_anio"].iter_rows(named=True):
        if row["n_cumplen_volumen"] is not None:
            m(f"hospitales.n_cumplen_volumen.{row['anio']}", row["n_cumplen_volumen"], tarea=TAREA, tabla="t23_hospitales_resumen.csv", alcance=str(row["anio"]),
              nota="hospitales con al menos los egresos y las muertes esperadas mínimos propuestos")
    for row in r["historial"].iter_rows(named=True):
        m(f"historial.factibilidad_pct.{row['anio']}", 100.0 * row["factibilidad"], tarea=TAREA, tabla="t23_historial_factibilidad.csv", alcance=str(row["anio"]),
          nota="identificador válido y ventana completa dentro del espacio de identificadores" + aviso)
    w = int(max(ctx.umbrales["ventanas_traslado_dias"]))
    enl = r.get("enlaces_publicado", r["enlaces"])          # la tabla publicada: una celda oculta como complemento no sale por las métricas
    for row in enl.filter(pl.col("periodo").str.contains(r"^\d{4}$")).iter_rows(named=True):
        anio = row["periodo"]
        if row["direccion"] == "saliente" and row["ventana_dias"] == w:
            m(f"traslados.n_salientes_adultos_hosp.{anio}", row["n_total"], tarea=TAREA, tabla="t23_enlaces_traslado.csv", conteo=True, alcance=anio,
              nota="derivaciones a otro hospital de adultos hospitalizados")
            if row["tasa_enlace"] is not None:
                m(f"traslados.enlace_pct_30d.{anio}", 100.0 * row["tasa_enlace"], tarea=TAREA, tabla="t23_enlaces_traslado.csv", alcance=anio,
                  ic_bajo=100.0 * row["ic_bajo"], ic_alto=100.0 * row["ic_alto"], nota=f"{w} días, solo ingresos de 2019 a 2023 y dentro del espacio de identificadores." + aviso)
        if row["direccion"] == "entrante" and row["ventana_dias"] == w and row["tasa_enlace"] is not None:
            m(f"traslados.enlace_entrante_pct_30d.{anio}", 100.0 * row["tasa_enlace"], tarea=TAREA, tabla="t23_enlaces_traslado.csv", alcance=anio,
              nota="entrantes de adultos hospitalizados con un traslado saliente previo." + aviso)
    for row in r["entrantes"].iter_rows(named=True):
        if row["mortalidad"] is not None:
            m(f"entrantes.mortalidad_pct.{row['entrante']}.{row['periodo'].replace('-', '_')}", 100.0 * row["mortalidad"], tarea=TAREA, tabla="t23_resultado_entrantes.csv",
              ic_bajo=100.0 * row["ic_bajo"], ic_alto=100.0 * row["ic_alto"], alcance=row["periodo"], nota="adultos de la cohorte dura con desenlace conocido")


def ejecutar(ctx: Contexto, cat: catalogos.Catalogos | None = None, regimenes: list[list[int]] | None = None) -> dict[str, Any]:
    r = calcular(ctx, cat, regimenes)
    _escribir(ctx, r)
    registrar_metricas(ctx, r)
    if "embudo" in r:
        figura_embudo(ctx)
        figura_volumen(ctx)
        figura_historial(ctx)
    return r


# ---------------------------------------------------------------------------------- figuras
def _arr(valores, escala: float = 1.0) -> np.ndarray:
    return np.array([np.nan if v is None else escala * float(v) for v in valores], dtype=float)


def figura_embudo(ctx: Contexto) -> None:
    t = ctx.leer_tabla("t23_embudo.csv", columnas_texto=["periodo", "cod_hospital", "fuera_95", "fuera_998"])
    periodos = [p for p in t["periodo"].unique().to_list() if "-" in p] + [max(p for p in t["periodo"].unique().to_list() if p.isdigit())]
    periodos = list(dict.fromkeys(periodos))
    z95, z998 = float(ctx.umbrales["z_95"]), float(ctx.umbrales["z_998"])
    fig, ejes = graficos.nueva_figura_multiple(1, len(periodos), figsize=(6.2 * len(periodos), 5.0), squeeze=False)
    for ax, periodo in zip(ejes[0], periodos):
        s = t.filter(pl.col("periodo") == periodo)
        p0, phi = float(s["p0"][0]), float(s["phi"][0])
        con = s.filter(pl.col("p").is_not_null())
        n_max = float(max(con["n"].max() or 100, 100)) * 1.15 if con.height else 1000.0
        xs = np.geomspace(10, n_max, 200)
        for z, estilo, etiqueta in ((z95, "--", "Límite 95 %"), (z998, "-", "Límite 99,8 %")):
            lo, hi = estadistica.limites_proporcion(p0, xs, z, phi)
            ax.plot(xs, 100 * lo, estilo, color="#555555", lw=1.0, label=etiqueta)
            ax.plot(xs, 100 * hi, estilo, color="#555555", lw=1.0)
        ax.axhline(100 * p0, color=graficos.OKABE_ITO[0], lw=1.0, alpha=0.8, label="Promedio nacional")
        dentro = con.filter(pl.col("fuera_998") == "no")
        f95 = con.filter((pl.col("fuera_998") == "no") & (pl.col("fuera_95") != "no"))
        dentro = dentro.filter(pl.col("fuera_95") == "no")
        f998 = con.filter(~pl.col("fuera_998").is_in(["no", "no evaluable"]))
        ax.scatter(dentro["n"], _arr(dentro["p"], 100), s=14, color=graficos.OKABE_ITO[0], alpha=0.8, label="Dentro de los límites")
        ax.scatter(f95["n"], _arr(f95["p"], 100), s=22, color=graficos.OKABE_ITO[1], label="Fuera del 95 %")
        ax.scatter(f998["n"], _arr(f998["p"], 100), s=26, color=graficos.OKABE_ITO[3], label="Fuera del 99,8 %")
        for fila in f998.iter_rows(named=True):
            if fila["n"] >= MIN_EPISODIOS_EMBUDO:
                ax.annotate(fila["cod_hospital"], (fila["n"], 100 * fila["p"]), textcoords="offset points", xytext=(4, 3), fontsize=6)
        ax.set_xscale("log")
        graficos.formato_miles(ax, "x")
        ax.set_xlabel("Egresos de la cohorte del modelo de mortalidad (escala logarítmica)")
        ax.set_ylabel("Mortalidad cruda (%)")
        ax.set_title(f"{periodo}: sobredispersión φ = {phi:.2f}".replace(".", ","), fontsize=9)
        ax.set_ylim(bottom=0)
    ejes[0][0].legend(fontsize=6.5, loc="upper right")
    ctx.figuras.guardar_figura(
        fig, id_figura="F09_01", tarea=9,
        titulo="Embudo crudo exploratorio de la mortalidad por hospital",
        descripcion_alt="Gráficos de embudo: cada punto es un hospital con su mortalidad cruda y su número de egresos de la cohorte del modelo; las líneas son los límites de "
                        "control al 95 % y al 99,8 % alrededor del promedio nacional, ajustados por sobredispersión.",
        csv="t23_embudo.csv", anios=" y ".join(periodos), decision="6", uso_interno=True,
        extra_pie="Sin ajuste por riesgo; límites con aproximación normal y sobredispersión winsorizada (Spiegelhalter 2005). Los hospitales con casos suprimidos no se dibujan.",
    )


def figura_volumen(ctx: Contexto) -> None:
    t = ctx.leer_tabla("t23_hospitales_volumen.csv", columnas_texto=["cod_hospital", "cumple_egresos"])
    anios = sorted(t["anio"].unique().to_list())
    if not anios:
        return
    fig, ax = graficos.nueva_figura(figsize=(8.0, 4.9))
    rng = np.random.default_rng(int(ctx.semilla))
    for k, a in enumerate(anios):
        s = t.filter((pl.col("anio") == a) & pl.col("n_D").is_not_null() & (pl.col("n_D") > 0))
        x = a + rng.uniform(-0.18, 0.18, size=s.height)
        ax.scatter(x, s["n_D"], s=14, color=graficos.OKABE_ITO[0], alpha=0.75)
        ok = int((s["cumple_egresos"] == "si").sum())
        ax.text(a, 1.02, f"{ok} de {s.height}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7)
    for umbral in ctx.umbrales["volumen_grid_egresos"]:
        ax.axhline(umbral, color="#555555", lw=0.8, ls="--" if umbral != ctx.umbrales["volumen_minimo_egresos"] else "-")
        ax.text(anios[-1] + 0.75, umbral * 1.03, f"{int(umbral):,}".replace(",", "."), fontsize=7, va="bottom", ha="right")
    ax.set_yscale("log")
    graficos.formato_miles(ax, "y")
    ax.set_xticks(anios)
    ax.set_xlim(anios[0] - 0.5, anios[-1] + 0.8)
    ax.set_xlabel("Año del archivo")
    ax.set_ylabel("Egresos de la cohorte del modelo (escala logarítmica)")
    ctx.figuras.guardar_figura(
        fig, id_figura="F09_02", tarea=9,
        titulo="Distribución del volumen de egresos por hospital",
        descripcion_alt="Puntos por año, uno por hospital, con el número de egresos de la cohorte del modelo de mortalidad en escala logarítmica; las líneas horizontales marcan "
                        "los volúmenes mínimos candidatos.",
        csv="t23_hospitales_volumen.csv", anios=f"{anios[0]} a {anios[-1]}", decision="6", uso_interno=True,
        extra_pie="Sobre cada año: hospitales que superan el volumen mínimo propuesto sobre el total de hospitales dibujados. Los hospitales con menos de diez egresos elegibles no se dibujan.",
    )


def figura_historial(ctx: Contexto) -> None:
    t = ctx.leer_tabla("t23_historial_factibilidad.csv", columnas_texto=["regimen", "cumple_minimo"]).sort("anio")
    if t.height == 0:
        return
    anios = t["anio"].to_list()
    x = np.arange(len(anios))
    ancho = 0.26
    fig, ax = graficos.nueva_figura(figsize=(8.0, 4.8))
    for k, (col, etiqueta) in enumerate((("prop_id_valido", "Identificador válido"), ("prop_ventana_completa", "Ventana de doce meses completa"), ("factibilidad", "Ambas (viable)"))):
        valores = _arr(t[col], 100)
        pos = x + (k - 1) * ancho
        ax.bar(pos, valores, width=ancho, color=graficos.OKABE_ITO[[0, 4, 2][k]], label=etiqueta)
        for xi, v in zip(pos, valores):
            if np.isfinite(v) and v == 0:
                ax.text(xi, 1.5, "0 %", ha="center", va="bottom", fontsize=7)
    ax.plot(x, _arr(t["prop_con_previo"], 100), "D", color=graficos.OKABE_ITO[3], label="Con egreso previo en la ventana (entre los viables)")
    minimo = 100 * float(ctx.umbrales["historial_factibilidad_minima"])
    ax.axhline(minimo, color="#555555", ls="--", lw=1.0, label=f"Mínimo de factibilidad ({minimo:.0f} %)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in anios])
    ax.set_ylim(0, 108)
    ax.set_xlabel("Año del archivo")
    ax.set_ylabel("% de los egresos del año")
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ctx.figuras.guardar_figura(
        fig, id_figura="F09_03", tarea=9,
        titulo="Factibilidad del historial de doce meses por año",
        descripcion_alt="Barras por año con el porcentaje de egresos con identificador válido, con ventana de doce meses completa dentro del espacio de identificadores y con ambas "
                        "condiciones; un rombo marca el porcentaje con egreso previo y una línea el mínimo de factibilidad.",
        csv="t23_historial_factibilidad.csv", anios=f"{anios[0]} a {anios[-1]}", decision="1",
        extra_pie="El primer año de cada espacio de identificadores no tiene ventana completa por construcción."
                  + (" En un extracto el egreso previo se subestima." if ctx.etiqueta == "extracto" else ""),
    )


# ------------------------------------------------------------------------------------ informe
def agregar_secciones(inf: Informe, ctx: Contexto, n0: int = 1) -> None:
    """Agrega al informe de exploración (CRISP-DM 2.3) las secciones de hospitales, historial, traslados y procedencia."""
    from .descripcion import Cifra_umbral

    u = ctx.umbrales
    res_hosp = ctx.leer_tabla("t23_hospitales_resumen.csv")
    res = ctx.leer_tabla("t23_resumen_hospitales.csv", columnas_texto=["autorref_supera_maximo", "historial_cumple_minimo", "enlace_cumple_minimo", "embudo_periodo", "regimenes"]).row(0, named=True)
    anios_res = sorted(res_hosp.filter(pl.col("n_hospitales_con_cohorte").is_not_null())["anio"].to_list())
    ultimo = anios_res[-1]
    extracto = ctx.etiqueta == "extracto"
    rh = lambda col, a=ultimo: cifra(ctx, "t23_hospitales_resumen.csv", col, anio=a)  # noqa: E731
    rs = lambda col: cifra(ctx, "t23_resumen_hospitales.csv", col, columnas_texto=["autorref_supera_maximo", "historial_cumple_minimo", "enlace_cumple_minimo", "embudo_periodo", "regimenes"])  # noqa: E731

    # --- hospitales ----------------------------------------------------------------------------------
    inf.seccion(f"{n0}. Hospitales: perfil y volumen")
    inf.parrafo(
        "En el último año con desenlace ({anio|texto}) hay {n_h} hospitales con egresos, {n_ped} de ellos pediátricos o monográficos (más de {prop|prop1} de sus egresos son de "
        "menores) y {n_fuera} fuera del catálogo. Con el volumen mínimo propuesto ({egresos|entero} egresos elegibles y {muertes|entero} muertes esperadas) cumplen {n_vol}. "
        "Las muertes esperadas salen de estandarizar por edad y sexo con las tasas nacionales del mismo año. En {anio|texto}, {celdas|prop1} de los hospitales tienen menos de "
        "{celda|entero} egresos elegibles y quedan con la celda suprimida.",
        anio=rh("anio"), n_h=rh("n_hospitales"), n_ped=rh("n_pediatricos"), prop=Cifra_umbral(ctx, "hospital_pediatrico_prop"), n_fuera=rh("n_fuera_catalogo"),
        egresos=Cifra_umbral(ctx, "volumen_minimo_egresos"), muertes=Cifra_umbral(ctx, "muertes_esperadas_minimas"), n_vol=rh("n_cumplen_volumen"),
        celdas=rh("prop_celdas_bajo_minimo"), celda=Cifra_umbral(ctx, "celda_minima"),
    )
    inf.tabla(
        "t23_hospitales_resumen.csv", df=res_hosp,
        encabezados={"anio": "Año", "n_hospitales": "Hospitales", "n_fuera_catalogo": "Fuera del catálogo", "n_pediatricos": "Pediátricos o monográficos",
                     "n_hospitales_con_cohorte": "Con cohorte", "n_cumplen_volumen": "Cumplen el volumen", "prop_celdas_bajo_minimo": "Celdas bajo el mínimo"},
        formatos={"prop_celdas_bajo_minimo": "prop1"}, leyenda="Hospitales por año",
    )
    grid = ctx.leer_tabla("t23_volumen_umbrales.csv").filter(pl.col("anio") == ultimo)
    inf.tabla(
        "t23_volumen_umbrales.csv", df=grid,
        columnas=["egresos_min", "muertes_esperadas_min", "n_hospitales", "n_cumplen_egresos", "n_cumplen_muertes", "n_cumplen_ambos", "n_cumplen_ambos_no_pediatricos"],
        encabezados={"egresos_min": "Egresos mínimos", "muertes_esperadas_min": "Muertes esperadas mínimas", "n_hospitales": "Hospitales", "n_cumplen_egresos": "Cumplen egresos",
                     "n_cumplen_muertes": "Cumplen muertes", "n_cumplen_ambos": "Cumplen ambos", "n_cumplen_ambos_no_pediatricos": "Ambos, sin pediátricos"},
        formatos={"egresos_min": "entero", "muertes_esperadas_min": "entero"}, leyenda="Cuadrícula de volumen mínimo en el último año con desenlace",
    )
    inf.figura("F09_02")

    # --- embudo ------------------------------------------------------------------------------------------
    if res.get("embudo_periodo"):
        inf.seccion(f"{n0 + 1}. Embudo crudo exploratorio")
        inf.parrafo(
            "El embudo del período {periodo|texto} compara la mortalidad cruda de cada hospital con el promedio nacional de la cohorte del modelo ({p0|prop2}) y dibuja los límites "
            "de control al 95 % y al 99,8 % que corresponden a su volumen. Se estima un factor de sobredispersión de {phi|dec2} con los puntajes z winsorizados (método de "
            "Spiegelhalter). De {n_ev} hospitales evaluables, {n95} quedan fuera del límite de 95 % y {n998} fuera del de 99,8 %. Es una vista exploratoria sin ajuste por riesgo: "
            "no es el indicador. La figura incluye también el último año solo, con su propio factor.",
            periodo=rs("embudo_periodo"), p0=rs("embudo_p0"), phi=rs("embudo_phi"), n_ev=rs("embudo_n_evaluados"), n95=rs("embudo_n_fuera_95"), n998=rs("embudo_n_fuera_998"),
        )
        inf.figura("F09_01")

    # --- identificador, historial y traslados ----------------------------------------------------------------
    inf.seccion(f"{n0 + 2}. Identificador, historial y traslados")
    inicio_reg = (
        "El identificador del paciente forma un solo espacio ({regimenes|texto})."
        if int(res["n_regimenes"]) == 1
        else "El identificador del paciente forma {n_reg} espacios sin pacientes en común ({regimenes|texto})."
    )
    inf.parrafo(
        inicio_reg + " El historial de doce meses solo es posible dentro de un espacio, y el primer año de cada uno no tiene ventana completa por construcción. "
        "Un egreso es viable si su identificador es válido y su ventana queda dentro del espacio.",
        n_reg=rs("n_regimenes"), regimenes=rs("regimenes"),
    )
    inf.tabla(
        "t23_historial_factibilidad.csv", df=ctx.leer_tabla("t23_historial_factibilidad.csv", columnas_texto=["regimen", "cumple_minimo"]),
        columnas=["anio", "regimen", "n_episodios", "prop_id_valido", "prop_ventana_completa", "factibilidad", "prop_con_previo", "cumple_minimo"],
        encabezados={"anio": "Año", "regimen": "Espacio de identificadores", "n_episodios": "Egresos", "prop_id_valido": "Identificador válido",
                     "prop_ventana_completa": "Ventana completa", "factibilidad": "Viable", "prop_con_previo": "Con egreso previo", "cumple_minimo": "Cumple el mínimo"},
        formatos={"prop_id_valido": "prop1", "prop_ventana_completa": "prop1", "factibilidad": "prop1", "prop_con_previo": "prop1"}, leyenda="Factibilidad del historial",
    )
    inf.figura("F09_03")
    if res.get("historial_anio") is not None:
        cumple = res.get("historial_cumple_minimo") == "si"
        inf.parrafo(
            "En {anio|texto} la factibilidad del historial es {fact|prop1}, " + ("igual o superior" if cumple else "inferior") + " al mínimo de {minimo|prop1} de la decisión 1. "
            "El porcentaje de egresos con un egreso previo dentro de la ventana es {previo|prop1} entre los viables.",
            anio=rs("historial_anio"), fact=rs("historial_factibilidad"), minimo=Cifra_umbral(ctx, "historial_factibilidad_minima"), previo=rs("historial_prop_con_previo"),
        )
    enl = ctx.leer_tabla("t23_enlaces_traslado.csv", columnas_texto=["periodo", "direccion"])
    w = int(max(u["ventanas_traslado_dias"]))
    mostrar = [str(ultimo), etiqueta_periodo(anios_res)]
    inf.tabla(
        "t23_enlaces_traslado.csv", df=enl.filter(pl.col("periodo").is_in(mostrar)).sort(["direccion", "ventana_dias", "periodo"]),
        columnas=["periodo", "direccion", "ventana_dias", "n_total", "n_id_invalido", "n_elegibles", "n_enlazados", "tasa_enlace", "ic_bajo", "ic_alto"],
        encabezados={"periodo": "Período", "direccion": "Dirección", "ventana_dias": "Ventana (días)", "n_total": "Traslados", "n_id_invalido": "Sin identificador válido",
                     "n_elegibles": "Elegibles", "n_enlazados": "Enlazados", "tasa_enlace": "Tasa de enlace", "ic_bajo": "IC bajo", "ic_alto": "IC alto"},
        formatos={"ventana_dias": "entero", "tasa_enlace": "prop1", "ic_bajo": "prop1", "ic_alto": "prop1"}, leyenda="Enlace de traslados salientes y entrantes",
    )
    if res.get("enlace_tasa") is not None:
        cumple = res.get("enlace_cumple_minimo") == "si"
        inf.parrafo(
            "En {anio|texto}, {tasa|prop1} de los {n_el} traslados salientes elegibles se enlaza con un ingreso en otro hospital dentro de {w|entero} días. Esa tasa "
            + ("alcanza" if cumple else "no alcanza") + " el mínimo de {minimo|prop1} que justifica el rastreo (decisión 13). Un traslado cuyo plazo de enlace se sale del último día "
            "observado no se cuenta como no enlazado: se excluye del denominador. Los ingresos de 2024 no se buscan.",
            anio=rs("enlace_anio"), tasa=rs("enlace_tasa"), n_el=rs("enlace_n_elegibles"), w=rs("enlace_ventana_dias"), minimo=Cifra_umbral(ctx, "enlace_traslado_minimo"),
        )
    if extracto:
        inf.parrafo(
            "En un extracto, el ingreso siguiente de un paciente casi nunca está en el mismo extracto, de modo que la factibilidad del historial, el egreso previo y el enlace de "
            "traslados se subestiman por construcción. Estas cifras solo prueban la mecánica y no sirven para decidir: quedan no evaluables hasta correr con los datos completos."
        )

    # --- procedencia -----------------------------------------------------------------------------------------
    inf.seccion(f"{n0 + 3}. Hospital de procedencia")
    supera = res.get("autorref_supera_maximo") == "si"
    inf.parrafo(
        "HOSPPROCEDENCIA trae el nombre del hospital de procedencia. En {aprox|prop1} de los egresos que lo informan, el nombre coincide con el del hospital del egreso "
        "({exacta|prop1} con nombres idénticos tras normalizar). El criterio de la decisión 2 acepta hasta {maximo|prop1}: " + ("se supera, así que HOSPPROCEDENCIA no sirve para "
        "definir el traslado entrante y se propone usar TIPO_PROCEDENCIA." if supera else "no se supera, así que HOSPPROCEDENCIA puede definir el traslado entrante."),
        aprox=rs("autorref_prop_aprox"), exacta=rs("autorref_prop_exacta"), maximo=Cifra_umbral(ctx, "autorreferencia_procedencia_max"),
    )
    ar = ctx.leer_tabla("t23_procedencia_autorref.csv", columnas_texto=["tipo_procedencia"])
    ult = ar.filter(pl.col("anio") == ultimo).sort("n_con_hospproc", descending=True, nulls_last=True).head(8)
    inf.tabla(
        "t23_procedencia_autorref.csv", df=ult,
        columnas=["tipo_procedencia", "n_episodios", "n_con_hospproc", "n_autorref_aprox", "prop_autorref_aprox"],
        encabezados={"tipo_procedencia": "Tipo de procedencia", "n_episodios": "Egresos", "n_con_hospproc": "Con hospital de procedencia", "n_autorref_aprox": "Del mismo hospital",
                     "prop_autorref_aprox": "% del mismo hospital"}, formatos={"prop_autorref_aprox": "prop1"}, leyenda="Hospital de procedencia por tipo de procedencia, último año con desenlace",
    )

    # --- entrantes ------------------------------------------------------------------------------------------
    inf.seccion(f"{n0 + 4}. Mortalidad de los traslados entrantes")
    ent = ctx.leer_tabla("t23_resultado_entrantes.csv", columnas_texto=["periodo", "entrante"])
    pool = etiqueta_periodo(anios_res)
    fila = ent.filter((pl.col("periodo") == pool) & (pl.col("entrante") == "si"))
    if fila.height == 1:
        e = lambda col, ent_="si": cifra(ctx, "t23_resultado_entrantes.csv", col, columnas_texto=["periodo", "entrante"], periodo=pool, entrante=ent_)  # noqa: E731
        inf.parrafo(
            "Se llama entrante al ingreso cuyo TIPO_PROCEDENCIA es otro hospital de la red. En adultos de la cohorte dura con desenlace conocido, la mortalidad de los entrantes es "
            "{m_si|prop2} y la de los demás ingresos {m_no|prop2}; la razón es {rr|dec2} (IC {lo|dec2} a {hi|dec2}). Es evidencia para la decisión 16, que decide si se excluyen "
            "los entrantes o solo los enlazados con un traslado saliente.",
            m_si=e("mortalidad"), m_no=e("mortalidad", "no"), rr=e("rr"), lo=e("rr_bajo"), hi=e("rr_alto"),
        )
    inf.tabla(
        "t23_resultado_entrantes.csv", df=ent, columnas=["periodo", "entrante", "n", "n_fallecidos", "mortalidad", "ic_bajo", "ic_alto", "rr"],
        encabezados={"periodo": "Período", "entrante": "Entrante", "n": "Egresos", "n_fallecidos": "Fallecidos", "mortalidad": "Mortalidad", "ic_bajo": "IC bajo",
                     "ic_alto": "IC alto", "rr": "Razón vs no entrantes"}, formatos={"mortalidad": "prop2", "ic_bajo": "prop2", "ic_alto": "prop2", "rr": "dec2"},
        leyenda="Mortalidad cruda de entrantes y no entrantes",
    )
    inf.seccion(f"{n0 + 5}. Límites")
    inf.texto_libre_sin_cifras(
        "- Los intervalos y el embudo ignoran el agrupamiento por paciente y por hospital.\n"
        "- El embudo es crudo: sin ajuste por riesgo, no es el indicador de desempeño.\n"
        "- El factor de sobredispersión multiplicativo crece con el volumen de cada hospital, por lo que no es comparable entre períodos de distinta duración: el período agrupado da un factor mayor que un año solo.\n"
        "- La comparación de nombres de hospital es aproximada: un nombre escrito distinto puede quedar como no coincidente.\n"
        "- Los enlaces y el historial dependen de que el identificador sea consistente; las inconsistencias de nacimiento y sexo por identificador están en el informe de calidad.\n"
        "- Las figuras por hospital son de uso interno y usan códigos, no nombres, mientras la decisión 10 siga abierta."
    )
