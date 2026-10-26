"""CRISP-DM 2.4: calidad contra catálogos y consistencia cruzada (lote 2 de reglas).

Tablas:
- ``t24_validez_codigos.csv``: validez de CIE-10, CIE-9, GRD, severidad, mortalidad, pabellón y hospital por año.
- ``t24_formato_cie10.csv``: formato de los diagnósticos (punto, minúsculas, espacios, tildes) por año.
- ``t24_capitulos_cie10.csv``: capítulo CIE-10 del diagnóstico principal por año.
- ``t24_hospitales_catalogo.csv`` y ``t24_hospitales_resumen.csv``: códigos de hospital frente a Hospitales.csv (solo códigos).
- ``t24_catalogo_grd_comparacion.csv``: GRDs_T10901.xlsx frente a IR-GRD.csv.
- ``t24_grd_tipo_actividad.csv``: tipo de GRD (tercer dígito) frente a TIPO_ACTIVIDAD.
- ``t24_tipoalta_etiquetas.csv``: etiquetas de TIPOALTA frente a tipoalta_roles.csv (2019 a 2023).
- ``t24_consistencia_cruzada.csv``: violaciones de las reglas Q20 a Q32.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from . import calidad, catalogos, config, esquema, graficos
from . import transformaciones as tr
from .calidad import Regla, _fila_regla, ESQUEMA_REGLAS
from .contexto import Contexto
from .informes import Informe, cifra

TAREA = "T06"
ANIOS_TXT = "2019 a 2024"

FAMILIAS: dict[str, tuple[str, tuple[str, ...]]] = {
    "cie10_principal": ("CIE-10 diagnóstico principal", ("DIAGNOSTICO1",)),
    "cie10_secundario": ("CIE-10 diagnósticos secundarios", esquema.DIAG_SEC),
    "cie9": ("CIE-9-MC procedimientos", esquema.PROC),
    "grd": ("GRD (código)", ("IR_29301_COD_GRD",)),
    "severidad": ("Severidad del GRD", ("IR_29301_SEVERIDAD",)),
    "mortalidad": ("Riesgo de mortalidad del GRD", ("IR_29301_MORTALIDAD",)),
    "pabellon": ("Uso de pabellón", ("USOSPABELLON",)),
}
ESTADOS_ORDEN = ["valido", "solo_categoria", "solo_en_ir_grd_csv", "no_agrupable_98_99", "invalido", "no_en_catalogo", "formato_invalido", "centinela"]


# ---------------------------------------------------------------------------- conteos por clase
def contar_clases(df: pl.DataFrame, columnas: tuple[str, ...], clasificador) -> Counter:
    """Cuenta valores informados por clase usando solo los valores distintos de cada columna."""
    total: Counter = Counter()
    for col in columnas:
        if col not in df.columns:
            continue
        vc = df[col].drop_nulls().value_counts()
        for v, c in vc.iter_rows():
            total[clasificador(v)] += int(c)
    return total


def clasificadores(cat: catalogos.Catalogos) -> dict[str, Any]:
    return {
        "cie10_principal": lambda v: catalogos.clasificar_cie10(v, cat.cie10_exacto, cat.cie10_cat3),
        "cie10_secundario": lambda v: catalogos.clasificar_cie10(v, cat.cie10_exacto, cat.cie10_cat3),
        "cie9": lambda v: catalogos.clasificar_cie9(v, cat.cie9_exacto, cat.cie9_cat2),
        "grd": lambda v: catalogos.clasificar_grd(v, cat.grd_xlsx_set, cat.grd_csv_set),
        "severidad": lambda v: catalogos.clasificar_en_conjunto(v, cat.severidad),
        "mortalidad": lambda v: catalogos.clasificar_en_conjunto(v, cat.mortalidad),
        "pabellon": lambda v: catalogos.clasificar_en_conjunto(v, cat.pabellon),
    }


# Qué clases cuentan como violación y cuáles forman el denominador de cada regla de validez.
VALIDEZ_REGLAS: dict[str, tuple[str, set[str], set[str]]] = {
    # tipo de regla: (familia, clases que violan, clases evaluadas)
    "cie10_principal_invalido": ("cie10_principal", {"invalido"}, {"valido", "solo_categoria", "invalido"}),
    "cie9_invalido": ("cie9", {"invalido"}, {"valido", "solo_categoria", "invalido"}),
    "grd_fuera_catalogo": ("grd", {"no_en_catalogo", "formato_invalido"}, {"valido", "solo_en_ir_grd_csv", "no_en_catalogo", "formato_invalido"}),
    "usospabellon_invalido": ("pabellon", {"invalido"}, {"valido", "invalido"}),
}


# -------------------------------------------------------------------- formato de diagnósticos
def formato_cie10(df: pl.DataFrame, columnas: tuple[str, ...] = esquema.DIAG) -> dict[str, int]:
    """Cuenta valores de diagnóstico por rasgo de formato. Un valor puede tener varios rasgos."""
    out: Counter = Counter()
    for col in columnas:
        vc = df[col].drop_nulls().value_counts()
        for v, c in vc.iter_rows():
            c = int(c)
            out["valores"] += c
            if catalogos._es_centinela_txt(v):
                out["centinela"] += c
                continue
            out["con_punto" if "." in v else "sin_punto"] += c
            if v != v.upper():
                out["con_minuscula"] += c
            if v != v.strip():
                out["con_espacio_extremo"] += c
            if not v.isascii():
                out["con_caracter_no_ascii"] += c
    return dict(out)


# --------------------------------------------------------------------------- GRD y actividad
def grupo_grd(tipo3: str | None) -> str:
    if tipo3 is None:
        return "sin_tipo"
    if tipo3 in catalogos.TIPO_GRD_HOSPITALIZADO:
        return "hospitalizado"
    if tipo3 in catalogos.TIPO_GRD_AMBULATORIO:
        return "ambulatorio"
    return "otro"


def preparar_cruzadas(df: pl.DataFrame) -> pl.DataFrame:
    """Columnas auxiliares (prefijo ``_``) para las reglas cruzadas."""
    cols = set(df.columns)
    e: list[pl.Expr] = []
    if "IR_29301_COD_GRD" in cols:
        g = pl.col("IR_29301_COD_GRD").str.strip_chars()
        e += [
            g.str.contains(r"^\d{6}$").alias("_grd6"),
            pl.when(g.str.contains(r"^\d{6}$")).then(g.str.slice(0, 2)).otherwise(None).alias("_mdc"),
            pl.when(g.str.contains(r"^\d{6}$")).then(g.str.slice(2, 1)).otherwise(None).alias("_tipo3"),
        ]
    if "DIAGNOSTICO1" in cols:
        e.append(esquema.normalizar_cie10(pl.col("DIAGNOSTICO1")).alias("_dx1"))
    if "FECHA_NACIMIENTO" in cols and "FECHA_INGRESO" in cols:
        e.append(tr.edad_anios(tr.a_fecha(pl.col("FECHA_NACIMIENTO")), tr.a_fecha(pl.col("FECHA_INGRESO"))).alias("_edad"))
    return df.with_columns(e) if e else df


def _sumas(df: pl.DataFrame, viol: pl.Expr, evalu: pl.Expr) -> tuple[int, int]:
    v, e = df.select(viol.fill_null(False).sum().alias("v"), evalu.fill_null(False).sum().alias("e")).row(0)
    return int(v), int(e)


def _sexo_ok() -> pl.Expr:
    return pl.col("SEXO").is_in(["MUJER", "HOMBRE"])


def eval_sexo_dx(df: pl.DataFrame, r: Regla, u) -> tuple[int, int]:
    dx = pl.col("_dx1")
    femenino = dx.str.starts_with("O") | dx.str.contains(r"^N(?:7\d|8\d|9[0-8])")
    masculino = dx.str.contains(r"^N(?:4\d|5[0-3])") | dx.str.contains(r"^C6[0-3]")
    viol = ((pl.col("SEXO") == "HOMBRE") & femenino) | ((pl.col("SEXO") == "MUJER") & masculino)
    return _sumas(df, viol, _sexo_ok() & dx.is_not_null())


def eval_mdc15_edad(df: pl.DataFrame, r: Regla, u) -> tuple[int, int]:
    base = pl.col("_grd6") & (pl.col("_mdc") == "15") & pl.col("_edad").is_not_null()
    return _sumas(df, base & (pl.col("_edad") > u[r.umbral]), base)


def eval_mdc14_hombre(df: pl.DataFrame, r: Regla, u) -> tuple[int, int]:
    base = pl.col("_grd6") & (pl.col("_mdc") == "14") & _sexo_ok()
    return _sumas(df, base & (pl.col("SEXO") == "HOMBRE"), base)


def eval_grd_ambulatorio_en_hosp(df: pl.DataFrame, r: Regla, u) -> tuple[int, int]:
    t = pl.col("_tipo3")
    base = pl.col("_grd6") & t.is_in(sorted(catalogos.TIPO_GRD_HOSPITALIZADO | catalogos.TIPO_GRD_AMBULATORIO)) & pl.col("TIPO_ACTIVIDAD").is_not_null()
    viol = base & t.is_in(sorted(catalogos.TIPO_GRD_AMBULATORIO)) & pl.col("TIPO_ACTIVIDAD").is_in(sorted(catalogos.ACTIVIDAD_HOSPITALIZACION))
    return _sumas(df, viol, base)


def eval_grd_hosp_en_ambulatoria(df: pl.DataFrame, r: Regla, u) -> tuple[int, int]:
    t = pl.col("_tipo3")
    base = pl.col("_grd6") & t.is_in(sorted(catalogos.TIPO_GRD_HOSPITALIZADO | catalogos.TIPO_GRD_AMBULATORIO)) & pl.col("TIPO_ACTIVIDAD").is_not_null()
    viol = base & t.is_in(sorted(catalogos.TIPO_GRD_HOSPITALIZADO)) & pl.col("TIPO_ACTIVIDAD").is_in(sorted(catalogos.ACTIVIDAD_AMBULATORIA))
    return _sumas(df, viol, base)


def eval_fallecido_riesgo_cero(df: pl.DataFrame, r: Regla, u) -> tuple[int, int]:
    fallecido = pl.col("TIPOALTA") == "FALLECIDO"
    clase_ok = pl.col("IR_29301_MORTALIDAD").is_in(["0", "1", "2", "3"])
    return _sumas(df, fallecido & (pl.col("IR_29301_MORTALIDAD") == "0"), fallecido & clase_ok)


def eval_tipoalta_sin_rol(df: pl.DataFrame, roles: pl.DataFrame) -> tuple[int, int, Counter]:
    vc = df["TIPOALTA"].drop_nulls().value_counts()
    viol = ev = 0
    metodos: Counter = Counter()
    for et, c in vc.iter_rows():
        _, metodo, _ = catalogos.rol_de_etiqueta(et, roles)
        ev += int(c)
        metodos[metodo] += int(c)
        if metodo == "sin_rol":
            viol += int(c)
    return viol, ev, metodos


# ------------------------------------------------------------------------------ ejecución
def columnas_a_leer(permitidas: list[Regla]) -> list[str]:
    # el hospital y los diagnósticos se perfilan siempre (no son columnas de resultado)
    cols: set[str] = {"COD_HOSPITAL", *esquema.DIAG}
    for r in permitidas:
        cols |= set(r.columnas)
    return [c for c in esquema.COLUMNAS if c in cols]


def calcular(ctx: Contexto, cat: catalogos.Catalogos | None = None) -> dict[str, Any]:
    cat = cat or catalogos.cargar_catalogos()
    u = ctx.umbrales
    cf = ctx.cortafuegos
    reglas = calidad.reglas_lote2()
    anios = ctx.fuente.anios()
    clasif = clasificadores(cat)

    filas_val: list[dict] = []
    filas_fmt: list[dict] = []
    filas_cap: list[dict] = []
    filas_hosp: list[dict] = []
    filas_reglas: list[dict] = []
    filas_act: list[dict] = []
    conteo_tipoalta: Counter = Counter()
    metodos_tipoalta: Counter = Counter()
    n_tipoalta_con_valor = 0
    anios_resultado = set(u.particion["anios_resultado"])

    for anio in anios:
        permitidas = [r for r in reglas if calidad.regla_permitida(r, anio, cf)]
        con_resultado = [r.id for r in permitidas if set(r.columnas) & cf.columnas_resultado]
        cols = columnas_a_leer(permitidas)
        df = ctx.lector.leer(anio, cols, proposito="calidad", tarea=TAREA, reglas=con_resultado)
        dfp = preparar_cruzadas(df)
        ids_permitidas = {r.id for r in permitidas}

        # --- validez por familia ---
        clases_por_fam: dict[str, Counter] = {}
        for fam, (nombre, columnas) in FAMILIAS.items():
            if any(c not in df.columns for c in columnas):
                continue  # familia no disponible este año (columna de resultado enmascarada o no pedida)
            clases = contar_clases(df, columnas, clasif[fam])
            clases_por_fam[fam] = clases
            total = sum(clases.values())
            for estado, n in sorted(clases.items(), key=lambda kv: ESTADOS_ORDEN.index(kv[0]) if kv[0] in ESTADOS_ORDEN else 99):
                filas_val.append({"anio": anio, "familia": nombre, "estado": estado, "n": n, "n_total_familia": total, "prop": n / total if total else None})

        # --- formato de diagnósticos y capítulos ---
        if all(c in df.columns for c in esquema.DIAG):
            fm = formato_cie10(df)
            for rasgo in ("valores", "con_punto", "sin_punto", "con_minuscula", "con_espacio_extremo", "con_caracter_no_ascii", "centinela"):
                v = fm.get(rasgo, 0)
                filas_fmt.append({"anio": anio, "rasgo": rasgo, "n": v, "prop": v / fm["valores"] if fm.get("valores") else None})
        if "DIAGNOSTICO1" in df.columns:
            cap: Counter = Counter()
            vc = df["DIAGNOSTICO1"].drop_nulls().value_counts()
            for v, c in vc.iter_rows():
                if catalogos._es_centinela_txt(v):
                    cap["centinela"] += int(c)
                else:
                    cap[cat.cie10_capitulo_por_cat3.get(catalogos.normalizar_cie(v)[:3], "sin capítulo")] += int(c)
            tot = sum(cap.values())
            for k, n in sorted(cap.items()):
                filas_cap.append({"anio": anio, "capitulo": k, "n": n, "prop": n / tot if tot else None})

        # --- hospitales ---
        if "COD_HOSPITAL" in df.columns:
            vc = df["COD_HOSPITAL"].drop_nulls().value_counts()
            for cod, n in vc.sort(["count", "COD_HOSPITAL"], descending=[True, False]).iter_rows():
                filas_hosp.append({"anio": anio, "cod_hospital": cod, "estado": catalogos.clasificar_hospital(cod, cat.hospitales), "n_episodios": int(n)})

        # --- reglas de validez (desde los conteos) ---
        for r in reglas:
            if r.id not in ids_permitidas:
                filas_reglas.append(_fila_regla(r, str(anio), None, None, "BLOQUEADA_2024", "el cortafuegos no autoriza la regla"))
                continue
            if r.tipo in VALIDEZ_REGLAS:
                fam, viol, evalu = VALIDEZ_REGLAS[r.tipo]
                c = clases_por_fam.get(fam, Counter())
                filas_reglas.append(_fila_regla(r, str(anio), sum(c[k] for k in viol), sum(c[k] for k in evalu)))
            elif r.tipo == "hospital_fuera_catalogo":
                h = [x for x in filas_hosp if x["anio"] == anio]
                filas_reglas.append(_fila_regla(r, str(anio), sum(x["n_episodios"] for x in h if x["estado"] == "fuera_de_catalogo"), sum(x["n_episodios"] for x in h)))
            elif r.tipo == "severidad_mortalidad_invalida":
                cs, cm = clases_por_fam.get("severidad", Counter()), clases_por_fam.get("mortalidad", Counter())
                filas_reglas.append(_fila_regla(r, str(anio), cs["invalido"] + cm["invalido"], cs["valido"] + cs["invalido"] + cm["valido"] + cm["invalido"]))
            elif r.tipo == "tipoalta_sin_rol":
                v, e, met = eval_tipoalta_sin_rol(df, cat.roles_tipoalta)
                filas_reglas.append(_fila_regla(r, str(anio), v, e))
                for et, c in df["TIPOALTA"].drop_nulls().value_counts().iter_rows():
                    conteo_tipoalta[et] += int(c)
                metodos_tipoalta.update(met)
                n_tipoalta_con_valor += e
            else:
                f = {
                    "sexo_dx_incompatible": eval_sexo_dx,
                    "mdc15_edad_mayor": eval_mdc15_edad,
                    "mdc14_hombre": eval_mdc14_hombre,
                    "grd_ambulatorio_en_hospitalizacion": eval_grd_ambulatorio_en_hosp,
                    "grd_hospitalizado_en_ambulatoria": eval_grd_hosp_en_ambulatoria,
                    "fallecido_riesgo_grd_cero": eval_fallecido_riesgo_cero,
                }[r.tipo]
                v, e = f(dfp, r, u)
                filas_reglas.append(_fila_regla(r, str(anio), v, e))

        # --- tipo de GRD frente a la actividad ---
        if {"_tipo3", "TIPO_ACTIVIDAD"} <= set(dfp.columns):
            sub = dfp.filter(pl.col("_grd6") & pl.col("TIPO_ACTIVIDAD").is_not_null()).group_by(["_tipo3", "TIPO_ACTIVIDAD"]).len()
            for t3, act, n in sub.iter_rows():
                filas_act.append({"anio": anio, "tipo_grd": t3, "grupo_grd": grupo_grd(t3), "tipo_actividad": act, "n": int(n)})
        del df, dfp

    # totales de las reglas por años evaluados
    reglas_df = pl.DataFrame(filas_reglas, schema=ESQUEMA_REGLAS)
    tot = []
    for r in reglas:
        sub = reglas_df.filter((pl.col("regla_id") == r.id) & (pl.col("ambito").str.contains(r"^\d{4}$")))
        ok = sub.filter(pl.col("estado") == "EVALUADA")
        tot.append(_fila_regla(r, "total", int(ok["n_violaciones"].sum()), int(ok["n_evaluados"].sum()),
                               nota=f"suma de {ok.height} años evaluados" + ("" if ok.height == sub.height else "; faltan años bloqueados")))
    reglas_df = pl.concat([reglas_df, pl.DataFrame(tot, schema=ESQUEMA_REGLAS)]).sort(["regla_id", "ambito"])

    # etiquetas de TIPOALTA contra el archivo de roles
    filas_et = []
    etiquetas_roles = cat.roles_tipoalta["tipoalta"].to_list()
    vistas = set()
    for et, n in sorted(conteo_tipoalta.items(), key=lambda kv: -kv[1]):
        nombre, metodo, rol = catalogos.rol_de_etiqueta(et, cat.roles_tipoalta)
        vistas.add(nombre)
        filas_et.append({"etiqueta": et, "longitud": len(et), "n": n, "etiqueta_en_roles": nombre, "metodo": metodo, "rol_mortalidad": rol,
                         "union_exacta_pierde": "si" if metodo != "exacta" else "no"})
    for nombre in etiquetas_roles:
        if nombre not in vistas:
            filas_et.append({"etiqueta": "(no aparece en los datos)", "longitud": 0, "n": 0, "etiqueta_en_roles": nombre, "metodo": "sin_datos", "rol_mortalidad": None, "union_exacta_pierde": "no"})
    et_df = pl.DataFrame(filas_et, schema={"etiqueta": pl.String, "longitud": pl.Int64, "n": pl.Int64, "etiqueta_en_roles": pl.String, "metodo": pl.String,
                                          "rol_mortalidad": pl.String, "union_exacta_pierde": pl.String})
    perdidos = int(et_df.filter(pl.col("union_exacta_pierde") == "si")["n"].sum())

    comp = comparar_catalogos_grd(cat)
    return {
        "validez": pl.DataFrame(filas_val, infer_schema_length=None).sort(["familia", "anio"]) if filas_val else pl.DataFrame(),
        "formato": pl.DataFrame(filas_fmt), "capitulos": pl.DataFrame(filas_cap).sort(["anio", "capitulo"]) if filas_cap else pl.DataFrame(),
        "hospitales": pl.DataFrame(filas_hosp).sort(["anio", "cod_hospital"]),
        "reglas": reglas_df, "actividad": pl.DataFrame(filas_act).sort(["anio", "tipo_grd", "tipo_actividad"]) if filas_act else pl.DataFrame(),
        "etiquetas": et_df, "pct_perdido_union_exacta": 100.0 * perdidos / n_tipoalta_con_valor if n_tipoalta_con_valor else None,
        "n_tipoalta": n_tipoalta_con_valor, "comparacion_grd": comp, "lista_reglas": reglas,
    }


def comparar_catalogos_grd(cat: catalogos.Catalogos) -> pl.DataFrame:
    """GRDs_T10901.xlsx frente a IR-GRD.csv (con ceros restituidos)."""
    solo_x = cat.grd_xlsx.filter(~pl.col("cod_grd").is_in(list(cat.grd_csv_set))).select(["cod_grd", "etiqueta"]).with_columns(pl.lit("solo_en_xlsx").alias("donde"))
    solo_c = cat.grd_csv.filter(~pl.col("cod_grd").is_in(list(cat.grd_xlsx_set))).select(["cod_grd", "etiqueta"]).with_columns(pl.lit("solo_en_ir_grd_csv").alias("donde"))
    return pl.concat([solo_x, solo_c]).sort(["donde", "cod_grd"]).select(["donde", "cod_grd", "etiqueta"])


def ejecutar(ctx: Contexto, cat: catalogos.Catalogos | None = None) -> dict[str, Any]:
    cat = cat or catalogos.cargar_catalogos()
    r = calcular(ctx, cat)
    TA = TAREA
    ctx.escribir_tabla("t24_validez_codigos.csv", r["validez"], tarea=TA, descripcion="Validez de códigos contra catálogos",
                       cols_n=["n"], dependientes=["prop"])
    ctx.escribir_tabla("t24_formato_cie10.csv", r["formato"], tarea=TA, descripcion="Formato de los diagnósticos", cols_n=["n"], dependientes=["prop"])
    ctx.escribir_tabla("t24_capitulos_cie10.csv", r["capitulos"], tarea=TA, descripcion="Capítulo CIE-10 del diagnóstico principal", cols_n=["n"], dependientes=["prop"])
    ctx.escribir_tabla("t24_hospitales_catalogo.csv", r["hospitales"], tarea=TA, descripcion="Códigos de hospital frente a Hospitales.csv (solo códigos)", cols_n=["n_episodios"])
    resumen_h = (
        r["hospitales"].group_by(["anio", "estado"]).agg(pl.len().alias("n_codigos"), pl.col("n_episodios").sum().alias("n_episodios")).sort(["anio", "estado"])
    )
    ctx.escribir_tabla("t24_hospitales_resumen.csv", resumen_h, tarea=TA, descripcion="Resumen de hospitales frente al catálogo", cols_n=["n_episodios"])
    ctx.escribir_tabla("t24_catalogo_grd_comparacion.csv", r["comparacion_grd"], tarea=TA, descripcion="Códigos GRD que están solo en uno de los dos catálogos")
    ctx.escribir_tabla("t24_grd_tipo_actividad.csv", r["actividad"], tarea=TA, descripcion="Tipo de GRD frente a TIPO_ACTIVIDAD", cols_n=["n"])
    ctx.escribir_tabla("t24_tipoalta_etiquetas.csv", r["etiquetas"], tarea=TA, descripcion="Etiquetas de TIPOALTA frente a los roles de la Fase 1", cols_n=["n"])
    ctx.escribir_tabla("t24_consistencia_cruzada.csv", r["reglas"], tarea=TA, descripcion="Violaciones de reglas de catálogos y consistencia cruzada",
                       cols_n=["n_violaciones"], dependientes=["prop_violaciones"])
    resumen = pl.DataFrame([_resumen(ctx, r, cat)])
    ctx.escribir_tabla("t24_resumen_catalogos.csv", resumen, tarea=TA, descripcion="Cifras globales de catálogos y consistencia cruzada")
    _registrar_metricas(ctx, r, cat)
    figura_validez(ctx)
    figura_inconsistencias(ctx)
    return r


def _resumen(ctx: Contexto, r: dict, cat: catalogos.Catalogos) -> dict[str, Any]:
    h = r["hospitales"]
    fuera = h.filter(pl.col("estado") == "fuera_de_catalogo")["cod_hospital"].n_unique() if h.height else 0
    priv = h.filter(pl.col("estado") == "en_catalogo_privado_probable")["cod_hospital"].n_unique() if h.height else 0
    et = r["etiquetas"]
    trunc = et.filter(pl.col("metodo") == "por_prefijo")
    comp = r["comparacion_grd"]
    return {
        "n_reglas_cruzadas": len(r["lista_reglas"]),
        "n_codigos_hospital_distintos": h["cod_hospital"].n_unique() if h.height else 0,
        "n_codigos_hospital_fuera_catalogo": fuera,
        "n_codigos_hospital_privado_probable": priv,
        "n_hospitales_catalogo": cat.hospitales.height,
        "n_etiquetas_tipoalta": et.filter(pl.col("n") > 0).height,
        "n_etiquetas_tipoalta_por_prefijo": trunc.height,
        "n_etiquetas_tipoalta_sin_rol": et.filter(pl.col("metodo") == "sin_rol").height,
        "pct_perdido_union_exacta": r["pct_perdido_union_exacta"],
        "n_grd_xlsx": cat.grd_xlsx.height,
        "n_grd_ir_csv": cat.grd_csv.height,
        "n_grd_solo_xlsx": comp.filter(pl.col("donde") == "solo_en_xlsx").height,
        "n_grd_solo_csv": comp.filter(pl.col("donde") == "solo_en_ir_grd_csv").height,
        "n_cie10_catalogo": cat.cie10.height,
        "n_cie9_catalogo": cat.cie9.height,
    }


def _total(r: dict, rid: str) -> int | None:
    t = r["reglas"].filter((pl.col("regla_id") == rid) & (pl.col("ambito") == "total"))
    return int(t["n_violaciones"][0]) if t.height and t["n_violaciones"][0] is not None else None


def _registrar_metricas(ctx: Contexto, r: dict, cat: catalogos.Catalogos) -> None:
    from .proteccion import suma_publicada

    # las cifras de tablas protegidas se toman de la tabla publicada (cambio 1.6): una celda oculta no vuelve a salir por el ledger
    r = {**r, "validez": ctx.leer_tabla("t24_validez_codigos.csv", columnas_texto=["familia", "estado"]),
         "capitulos": ctx.leer_tabla("t24_capitulos_cie10.csv", columnas_texto=["capitulo"]),
         "reglas": ctx.leer_tabla("t24_consistencia_cruzada.csv", columnas_texto=["regla_id", "nombre", "severidad", "accion", "ambito", "estado", "nota"])}
    h = r["hospitales"]
    for anio, sub in h.group_by("anio"):
        ctx.registrar_metrica(f"hospitales.n_distintos.{anio[0]}", sub["cod_hospital"].n_unique(), tarea=TAREA, tabla="t24_hospitales_catalogo.csv")
    if h.height:
        n_max = max(sub["cod_hospital"].n_unique() for _, sub in h.group_by("anio"))
        ctx.registrar_metrica("hospitales.n_distintos.maximo", n_max, tarea=TAREA, tabla="t24_hospitales_catalogo.csv")
        ctx.registrar_metrica("hospitales.n_distintos.total", h["cod_hospital"].n_unique(), tarea=TAREA, tabla="t24_hospitales_catalogo.csv")
    v = r["validez"]
    grd = v.filter((pl.col("familia") == "GRD (código)"))
    for estado, clave in (("centinela", "calidad.grd_centinela.total"), ("no_agrupable_98_99", "calidad.grd_99xxxx.total")):
        ctx.registrar_metrica(clave, suma_publicada(grd.filter(pl.col("estado") == estado)["n"]), tarea=TAREA, tabla="t24_validez_codigos.csv", conteo=True)
    pr = v.filter(pl.col("familia") == "CIE-10 diagnóstico principal")
    if pr.height:
        tot = suma_publicada(pr.filter(pl.col("estado").is_in(["valido", "solo_categoria", "invalido"]))["n"])
        val = suma_publicada(pr.filter(pl.col("estado").is_in(["valido", "solo_categoria"]))["n"])
        ctx.registrar_metrica("calidad.cie10_principal_valido_pct", 100.0 * val / tot if (tot and val is not None) else None, tarea=TAREA, tabla="t24_validez_codigos.csv")
    f = r["formato"]
    if f.height:
        s = f.filter(pl.col("rasgo") == "con_punto")["n"].sum()
        t = f.filter(pl.col("rasgo") == "valores")["n"].sum()
        sc = f.filter(pl.col("rasgo") == "centinela")["n"].sum()
        ctx.registrar_metrica("calidad.diagnostico_con_punto_pct", 100.0 * s / (t - sc) if (t - sc) else None, tarea=TAREA, tabla="t24_formato_cie10.csv",
                              nota="sobre todas las columnas de diagnóstico y no solo los diez primeros campos")
        for rasgo in ("con_minuscula", "con_espacio_extremo", "con_caracter_no_ascii"):
            ctx.registrar_metrica(f"calidad.diagnostico_{rasgo}", int(f.filter(pl.col("rasgo") == rasgo)["n"].sum()), tarea=TAREA, tabla="t24_formato_cie10.csv", conteo=True)
    cap = r["capitulos"]
    if cap.height:
        tot = suma_publicada(cap.filter(~pl.col("capitulo").is_in(["centinela", "sin capítulo"]))["n"])
        for cap_id, g in cap.group_by("capitulo"):
            n_cap = suma_publicada(g["n"])
            ctx.registrar_metrica(f"cie10.capitulo_pct.{cap_id[0]}", 100.0 * n_cap / tot if (tot and n_cap is not None) else None, tarea=TAREA, tabla="t24_capitulos_cie10.csv")
    ctx.registrar_metrica("tipoalta.pct_perdido_union_exacta", r["pct_perdido_union_exacta"], tarea=TAREA, tabla="t24_tipoalta_etiquetas.csv",
                          nota="% de egresos 2019 a 2023 cuya etiqueta de TIPOALTA no coincide exactamente con tipoalta_roles.csv")
    for rid, clave in (("Q25", "calidad.sexo_dx_incompatible"), ("Q26", "calidad.mdc15_edad_mayor"), ("Q27", "calidad.mdc14_hombre"),
                       ("Q28", "calidad.grd_ambulatorio_en_hospitalizacion"), ("Q29", "calidad.grd_hospitalizado_en_ambulatoria"),
                       ("Q22", "calidad.grd_fuera_catalogo"), ("Q23", "calidad.hospital_fuera_catalogo"), ("Q20", "calidad.cie10_principal_invalido"),
                       ("Q31", "calidad.usospabellon_invalido")):
        ctx.registrar_metrica(f"{clave}.total", _total(r, rid), tarea=TAREA, tabla="t24_consistencia_cruzada.csv", conteo=True)


# --------------------------------------------------------------------------------- figuras
def figura_validez(ctx: Contexto) -> None:
    t = ctx.leer_tabla("t24_validez_codigos.csv", columnas_texto=["familia", "estado"])
    familias = [n for n, _ in FAMILIAS.values() if n in set(t["familia"].to_list())]
    estados = ["valido", "solo_categoria", "solo_en_ir_grd_csv", "no_agrupable_98_99", "invalido", "no_en_catalogo", "formato_invalido", "centinela"]
    nombres = {"valido": "válido", "solo_categoria": "solo categoría", "solo_en_ir_grd_csv": "solo en IR-GRD.csv", "no_agrupable_98_99": "98 o 99 (no agrupable)",
               "invalido": "inválido", "no_en_catalogo": "no está en el catálogo", "formato_invalido": "formato inválido", "centinela": "centinela"}
    usados = [e for e in estados if e in set(t["estado"].to_list())]
    m = np.full((len(familias), len(usados)), np.nan)
    for i, fam in enumerate(familias):
        sub = t.filter(pl.col("familia") == fam)
        total = float(sub.group_by("anio").agg(pl.col("n_total_familia").first()).get_column("n_total_familia").sum())
        for j, est in enumerate(usados):
            s = sub.filter(pl.col("estado") == est)
            if s.height and total:
                m[i, j] = 100.0 * float(s["n"].drop_nulls().sum()) / total
    logm = np.where(np.isnan(m), np.nan, np.log10(m + 0.0001))
    fig, ax = graficos.nueva_figura(figsize=(10.5, 4.6))
    graficos.mapa_calor(fig, ax, logm, familias, [nombres[e] for e in usados], vmin=-4, vmax=2, etiqueta_barra="log10(% de los valores informados)", anotar=False, rotar_x=20)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            txt = "n/d" if np.isnan(m[i, j]) else (f"{m[i, j]:.3f}".replace(".", ","))
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color="white" if (not np.isnan(logm[i, j]) and logm[i, j] < -0.5) else "black")
    ctx.figuras.guardar_figura(
        fig, id_figura="F06_01", tarea=6,
        titulo="Validez de los códigos contra los catálogos de referencia",
        descripcion_alt="Mapa de calor con el porcentaje de valores informados de cada familia de códigos (CIE-10, CIE-9, GRD, severidad, mortalidad y pabellón) en cada estado de validez.",
        csv="t24_validez_codigos.csv", anios=ANIOS_TXT,
        extra_pie="Porcentaje acumulado de los años evaluados; gris: sin casos o suprimido.",
    )


def figura_inconsistencias(ctx: Contexto) -> None:
    t = ctx.leer_tabla("t24_consistencia_cruzada.csv", columnas_texto=["ambito"]).filter(pl.col("ambito").str.contains(r"^\d{4}$"))
    orden_sev = {"alta": 0, "media": 1, "baja": 2}
    reglas = t.select(["regla_id", "nombre", "severidad"]).unique().with_columns(pl.col("severidad").replace_strict(orden_sev).alias("o")).sort(["o", "regla_id"])
    etiquetas = [f"[{r['severidad']}] {r['regla_id']} {r['nombre']}" for r in reglas.iter_rows(named=True)]
    anios = sorted(t["ambito"].unique().to_list())
    m = np.full((reglas.height, len(anios)), np.nan)
    for i, rid in enumerate(reglas["regla_id"].to_list()):
        for j, a in enumerate(anios):
            s = t.filter((pl.col("regla_id") == rid) & (pl.col("ambito") == a))
            if s.height and s["prop_violaciones"][0] is not None:
                m[i, j] = float(s["prop_violaciones"][0]) * 100000.0
    logm = np.where(np.isnan(m), np.nan, np.log10(m + 1.0))
    fig, ax = graficos.nueva_figura(figsize=(11.5, 6.2))
    graficos.mapa_calor(fig, ax, logm, etiquetas, anios, vmin=0, vmax=5.0, etiqueta_barra="log10(violaciones por 100.000 + 1)", anotar=False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            txt = "n/d" if np.isnan(m[i, j]) else f"{m[i, j]:,.0f}".replace(",", ".")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6, color="white" if (not np.isnan(logm[i, j]) and logm[i, j] < 2.5) else "black")
    ax.set_xlabel("Año del archivo")
    ctx.figuras.guardar_figura(
        fig, id_figura="F06_02", tarea=6,
        titulo="Inconsistencias de catálogos y reglas cruzadas por año",
        descripcion_alt="Mapa de calor con las violaciones por cada 100.000 filas de las reglas de catálogos y consistencia cruzada, por año y severidad.",
        csv="t24_consistencia_cruzada.csv", anios=ANIOS_TXT, decision="15",
        extra_pie="Gris: celda suprimida o regla que el cortafuegos no autoriza en 2024.",
    )


# ------------------------------------------------------------------------------------ informe
def agregar_secciones(inf: Informe, ctx: Contexto) -> None:
    """Agrega al informe 2.4 las secciones de catálogos y consistencia cruzada."""
    rs = lambda col: cifra(ctx, "t24_resumen_catalogos.csv", col)  # noqa: E731
    inf.seccion("6. Validez contra los catálogos de referencia")
    inf.parrafo(
        "Los códigos de CIE-10, CIE-9-MC, GRD, severidad, mortalidad y pabellón se contrastaron con los catálogos de la raíz del proyecto. "
        "Un código es válido si existe en el catálogo; si solo existe su categoría, se informa aparte. Los centinelas no cuentan como inválidos."
    )
    inf.figura("F06_01")
    inf.seccion("7. Catálogo de GRD: planilla del oficio frente a IR-GRD.csv")
    inf.parrafo(
        "La planilla del oficio tiene {n_xlsx} códigos y IR-GRD.csv tiene {n_csv}, tras restituir los ceros a la izquierda que ese archivo pierde. "
        "{n_solo_x} códigos están solo en la planilla y {n_solo_c} solo en el archivo CSV.",
        n_xlsx=rs("n_grd_xlsx"), n_csv=rs("n_grd_ir_csv"), n_solo_x=rs("n_grd_solo_xlsx"), n_solo_c=rs("n_grd_solo_csv"),
    )
    inf.tabla("t24_catalogo_grd_comparacion.csv", max_filas=12, columnas_texto=["cod_grd", "etiqueta", "donde"],
              encabezados={"donde": "Dónde está", "cod_grd": "Código GRD", "etiqueta": "Etiqueta"}, leyenda="Códigos presentes en uno solo de los catálogos")
    inf.seccion("8. Hospitales frente a Hospitales.csv")
    inf.parrafo(
        "En los archivos aparecen {n_cod} códigos de hospital distintos; {n_fuera} no figuran en Hospitales.csv y {n_priv} figuran con nombre de clínica o sanatorio "
        "(tipo inferido por el nombre, porque el catálogo no trae el tipo ni el nivel de complejidad). Las tablas por hospital usan códigos y no nombres.",
        n_cod=rs("n_codigos_hospital_distintos"), n_fuera=rs("n_codigos_hospital_fuera_catalogo"), n_priv=rs("n_codigos_hospital_privado_probable"),
    )
    inf.tabla("t24_hospitales_resumen.csv", columnas_texto=["estado"], encabezados={"anio": "Año", "estado": "Estado", "n_codigos": "Códigos", "n_episodios": "Egresos"},
              leyenda="Códigos de hospital por año frente al catálogo")
    inf.seccion("9. Etiquetas de TIPOALTA frente a los roles de la Fase 1")
    inf.parrafo(
        "Se observaron {n_et} etiquetas de TIPOALTA en 2019 a 2023, de las cuales {n_pref} no coinciden exactamente con el archivo de roles por estar cortadas; "
        "se asignan por prefijo. {n_sin} etiquetas quedan sin rol. Una unión exacta pierde {perd|pct2} de los egresos de ese período.",
        n_et=rs("n_etiquetas_tipoalta"), n_pref=rs("n_etiquetas_tipoalta_por_prefijo"), n_sin=rs("n_etiquetas_tipoalta_sin_rol"), perd=rs("pct_perdido_union_exacta"),
    )
    inf.tabla("t24_tipoalta_etiquetas.csv", columnas_texto=["etiqueta", "etiqueta_en_roles", "metodo", "rol_mortalidad", "union_exacta_pierde"],
              encabezados={"etiqueta": "Etiqueta en los datos", "longitud": "Largo", "n": "Egresos", "etiqueta_en_roles": "Etiqueta en roles", "metodo": "Método",
                           "rol_mortalidad": "Rol", "union_exacta_pierde": "Unión exacta pierde"},
              leyenda="Etiquetas de TIPOALTA y su rol")
    inf.seccion("10. Reglas cruzadas")
    inf.parrafo(
        "Las reglas cruzadas verifican la coherencia entre columnas: sexo y diagnóstico, edad y MDC 15, tipo de GRD y actividad, y riesgo de mortalidad del GRD y desenlace. "
        "La regla que cruza el desenlace no se evalúa en 2024."
    )
    inf.figura("F06_02")
    reglas = ctx.leer_tabla("t24_consistencia_cruzada.csv", columnas_texto=["ambito"])
    inf.tabla("t24_consistencia_cruzada.csv", df=reglas.filter(pl.col("ambito") == "total").select(["regla_id", "nombre", "severidad", "accion", "n_violaciones", "n_evaluados", "prop_violaciones"]),
              encabezados={"regla_id": "Regla", "nombre": "Nombre", "severidad": "Severidad", "accion": "Acción", "n_violaciones": "Violaciones",
                           "n_evaluados": "Evaluados", "prop_violaciones": "Proporción"}, formatos={"prop_violaciones": "prop2"},
              leyenda="Violaciones totales de las reglas de catálogos y consistencia cruzada")
    inf.tabla("t24_grd_tipo_actividad.csv", df=ctx.leer_tabla("t24_grd_tipo_actividad.csv", columnas_texto=["tipo_grd", "grupo_grd", "tipo_actividad"]).group_by(["grupo_grd", "tipo_actividad"]).agg(
                  pl.when(pl.col("n").null_count() > 0).then(None).otherwise(pl.col("n").cast(pl.Int64, strict=False).sum()).alias("n")).sort(["grupo_grd", "tipo_actividad"]),  # con un año oculto la suma no se publica
              columnas_texto=["grupo_grd", "tipo_actividad"], encabezados={"grupo_grd": "Tipo de GRD", "tipo_actividad": "Actividad", "n": "Egresos"},
              leyenda="Tipo de GRD frente a TIPO_ACTIVIDAD (años acumulados)")
