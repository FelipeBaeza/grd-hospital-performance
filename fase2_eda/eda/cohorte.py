"""CRISP-DM 2.3, parte 1: cohorte, resultado y estadía.

Aporta evidencia a las decisiones 5 (población), 7 (estadías largas), 9 (dispersión de la estadía), 15 (estadía de 0 días
y actividad ambulatoria) y 17 (ventana de entrenamiento). Solo entrega agregados.

Reglas de partición (Fase 2, sección 4.2):
- Todo lo que usa el desenlace (TIPOALTA) o la fecha de alta se calcula solo para 2019 a 2023.
- 2024 se usa únicamente con propósito ``estructura`` y sin columnas de resultado: aporta el tamaño de la cohorte antes
  de mirar el desenlace (exclusiones EX01 a EX05), nunca mortalidad ni estadía. Un intento de leer el desenlace de 2024
  se hace como autoprueba, debe ser rechazado y queda en el registro de accesos.

Definición de la cascada (todas las exclusiones se aplican en secuencia, por eso suman exactamente el total):
  EX01 GRD no agrupable (sin código, centinela, formato inválido o MDC 99)
  EX02 recién nacidos (MDC 15)
  EX03 MDC 14 sin comorbilidades de Elixhauser (solo diagnósticos secundarios, sin mirar el desenlace)
  A = cohorte dura; B = A y adultos; C = B y actividad de hospitalización; D = C sin censurados (modelo de mortalidad);
  E = D, solo sobrevivientes y con estadía mayor que cero (modelo de estadía).
La definición del pipeline (EX03 con "sobrevive" y mapeo de tres caracteres) se reproduce de forma aproximada solo para
contrastar con las cifras documentadas; el código original del pipeline no está en el repositorio revisado.

Tablas: t23_cascada, t23_cascada_pipeline, t23_cascada_variantes, t23_tipoalta, t23_resultado_anio, t23_resultado_estratos,
t23_resultado_edad_sexo, t23_serie_mensual, t23_estancia_cero, t23_estancia_hist, t23_estancia_dispersion,
t23_cola_estancia y t23_resumen. Figuras: F07_01 a F07_05.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import polars as pl

from . import catalogos, config, elixhauser, esquema, estadistica, graficos
from . import transformaciones as tr
from .contexto import Contexto
from .cortafuegos import AccesoBloqueado
from .informes import Informe, cifra

TAREA = "T07"
ACTIVIDAD_HOSP = "HOSPITALIZACIÓN"
SIN_DATO = "(sin dato)"
ROLES = ("EVENTO", "NO_EVENTO", "CENSURADO", "SIN_DATO")

COLS_BASE = (
    "SEXO",
    "FECHA_NACIMIENTO",
    "FECHA_INGRESO",
    "TIPO_ACTIVIDAD",
    "TIPO_INGRESO",
    "TIPO_PROCEDENCIA",
    "IR_29301_COD_GRD",
    "DIAGNOSTICO1",
)
COLS_RESULTADO = ("TIPOALTA", "FECHAALTA")
COLS_EXTRA = ("COD_HOSPITAL", "CIP_ENCRIPTADO", "HOSPPROCEDENCIA")

GRUPOS_EDAD = (
    ("0 a 4", 0, 4), ("5 a 14", 5, 14), ("15 a 24", 15, 24), ("25 a 34", 25, 34), ("35 a 44", 35, 44),
    ("45 a 54", 45, 54), ("55 a 64", 55, 64), ("65 a 74", 65, 74), ("75 a 84", 75, 84), ("85 o más", 85, 200),
)
EDAD_DESCONOCIDA = "(edad desconocida)"
TRAMOS_ESTANCIA = (
    ("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4), ("5", 5, 5), ("6", 6, 6), ("7", 7, 7),
    ("8 a 9", 8, 9), ("10 a 14", 10, 14), ("15 a 29", 15, 29), ("30 a 59", 30, 59), ("60 a 89", 60, 89),
    ("90 a 179", 90, 179), ("180 a 364", 180, 364), ("365 o más", 365, 10**9),
)
CODIGO_TIPOALTA = {
    "DOMICILIO": "DOMICILIO",
    "FALLECIDO": "FALLECIDO",
    "HOSPITALIZACIÓN DOMICILIARIA": "HOSP_DOMICILIARIA",
    "DERIVACIÓN OTRO HOSPITAL DEL SERVICIO": "DERIV_HOSP_SERVICIO",
    "ALTA VOLUNTARIA": "ALTA_VOLUNTARIA",
    "DERIVACIÓN OTRO HOSPITAL DE LA RED NACIONAL": "DERIV_HOSP_RED",
    "DERIVACIÓN A OTROS CENTROS (CÁRCEL, HOGAR)": "DERIV_OTROS_CENTROS",
    "FUGA DEL PACIENTE": "FUGA",
    "DERIVACIÓN INST. PRIVADA (COMPRA DE SERVICIOS)": "DERIV_PRIVADA_COMPRA",
    "DERIVACIÓN INST. PRIVADA (VOLUNTARIO)": "DERIV_PRIVADA_VOLUNTARIO",
    "NO IDENTIFICADA": "NO_IDENTIFICADA",
    "DESCONOCIDO": "DESCONOCIDO",
}
DESCRIPCION_PASO = {
    "TOTAL": ("total", "Egresos leídos", "no"),
    "EX01": ("exclusion", "GRD no agrupable: sin código, centinela, formato inválido o MDC 99", "no"),
    "EX02": ("exclusion", "Recién nacidos (MDC 15)", "no"),
    "EX03": ("exclusion", "Obstétricos sin comorbilidades (MDC 14 sin categorías de Elixhauser)", "no"),
    "A": ("cohorte", "Variante A: cohorte dura", "no"),
    "EX04": ("exclusion", "Menores de 18 años o con edad desconocida", "no"),
    "B": ("cohorte", "Variante B: adultos", "no"),
    "EX05": ("exclusion", "Actividad distinta de hospitalización", "no"),
    "C": ("cohorte", "Variante C: adultos hospitalizados", "no"),
    "EX06": ("exclusion", "Censurados por tipo de alta o sin dato de desenlace", "si"),
    "D": ("cohorte", "Variante D: cohorte del modelo de mortalidad", "si"),
    "EX07": ("exclusion", "Fallecidos (salen del modelo de estadía)", "si"),
    "EX08": ("exclusion", "Sobrevivientes con estadía de 0 días o desconocida", "si"),
    "E": ("cohorte", "Variante E: cohorte del modelo de estadía", "si"),
}
ORDEN_PASOS = tuple(DESCRIPCION_PASO)
PASOS_ESTRUCTURA = ("TOTAL", "EX01", "EX02", "EX03", "A", "EX04", "B", "EX05", "C")
DESCRIPCION_VARIANTE = {
    "A": "Cohorte dura: sin GRD no agrupable, recién nacidos ni obstétricos sin comorbilidades",
    "B": "A y solo adultos",
    "C": "B y solo actividad de hospitalización",
    "D": "C sin censurados: cohorte del modelo de mortalidad",
    "E": "D, solo sobrevivientes y con estadía mayor que cero: cohorte del modelo de estadía",
}


# ---------------------------------------------------------------------------- utilidades
def etiqueta_periodo(anios: Sequence[int]) -> str:
    a = sorted(anios)
    return str(a[0]) if len(a) == 1 else f"{a[0]}-{a[-1]}"


def _slug(texto: str) -> str:
    sin = "".join(c for c in unicodedata.normalize("NFD", texto.upper()) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^A-Z0-9]+", "_", sin).strip("_")


def codigo_tipoalta(etiqueta_en_roles: str) -> str:
    return CODIGO_TIPOALTA.get(etiqueta_en_roles) or _slug(etiqueta_en_roles)


def mapas_rol(etiquetas: Iterable[str], roles: pl.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    """(etiqueta -> rol de mortalidad, etiqueta -> etiqueta del archivo de roles) para las etiquetas observadas."""
    rol: dict[str, str] = {}
    canon: dict[str, str] = {}
    for et in etiquetas:
        nombre, _, r = catalogos.rol_de_etiqueta(et, roles)
        rol[et] = r or "SIN_DATO"
        canon[et] = nombre or "(sin rol)"
    return rol, canon


def _sumar(dicts: Iterable[dict[str, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + int(v)
    return out


def _grupo_edad_expr() -> pl.Expr:
    e = pl.col("edad")
    expr = pl.when(e.is_null()).then(pl.lit(EDAD_DESCONOCIDA))
    for nombre, lo, hi in GRUPOS_EDAD:
        expr = expr.when((e >= lo) & (e <= hi)).then(pl.lit(nombre))
    return expr.otherwise(pl.lit(EDAD_DESCONOCIDA))


def _texto(col: str) -> pl.Expr:
    return pl.col(col).str.strip_chars().fill_null(SIN_DATO)


# ---------------------------------------------------------------------- base tipada por año
def tipar(df: pl.DataFrame, anio: int, cat: catalogos.Catalogos, u: config.Umbrales) -> pl.DataFrame:
    """Convierte las columnas de texto del lector en la base tipada de un año (sin diagnósticos secundarios).

    Edades fuera de rango (negativas o sobre el máximo) y estadías negativas quedan nulas: son errores de registro que la
    regla R5 manda anular. Las estadías largas se conservan (la cola se estudia en la decisión 7).
    """
    con_res = "TIPOALTA" in df.columns
    nac = tr.a_fecha(pl.col("FECHA_NACIMIENTO"))
    ing = tr.a_fecha(pl.col("FECHA_INGRESO"))
    edad_b = tr.edad_anios(nac, ing)
    g = pl.col("IR_29301_COD_GRD").str.strip_chars()
    fmt6 = g.str.contains(r"^\d{6}$")
    cat3 = esquema.normalizar_cie10(pl.col("DIAGNOSTICO1")).str.slice(0, 3)
    capitulo = pl.when(cat3.is_null()).then(pl.lit(SIN_DATO)).otherwise(
        cat3.replace_strict(cat.cie10_capitulo_por_cat3, default="sin capítulo", return_dtype=pl.String)
    )
    exprs: list[pl.Expr] = [
        pl.lit(int(anio), dtype=pl.Int32).alias("anio"),
        pl.when(pl.col("SEXO").is_in(["MUJER", "HOMBRE"])).then(pl.col("SEXO")).otherwise(pl.lit("SIN_DATO")).alias("sexo"),
        pl.when((edad_b >= u["edad_minima"]) & (edad_b <= u["edad_maxima"])).then(edad_b).alias("edad"),
        _texto("TIPO_ACTIVIDAD").alias("actividad"),
        _texto("TIPO_INGRESO").alias("tipo_ingreso"),
        _texto("TIPO_PROCEDENCIA").alias("procedencia"),
        capitulo.alias("capitulo"),
        pl.when(fmt6).then(g.str.slice(0, 2)).alias("mdc"),
        pl.when(g.is_null()).then(pl.lit("nulo"))
        .when(esquema.es_centinela(g)).then(pl.lit("centinela"))
        .when(~fmt6).then(pl.lit("formato_invalido"))
        .when(g.str.slice(0, 2) == "99").then(pl.lit("mdc_99"))
        .otherwise(pl.lit("agrupable"))
        .alias("grd_estado"),
    ]
    # columnas opcionales para hospitales, historial y traslados (Tarea 9)
    if "COD_HOSPITAL" in df.columns:
        exprs.append(pl.col("COD_HOSPITAL").str.strip_chars().fill_null(SIN_DATO).alias("hospital"))
    if "CIP_ENCRIPTADO" in df.columns:
        idc = pl.col("CIP_ENCRIPTADO").str.strip_chars()
        valido = idc.str.contains(r"^\d+$") & (idc.str.len_chars() >= int(u["id_largo_minimo"]))
        exprs += [pl.when(valido).then(idc.cast(pl.Int64, strict=False)).alias("id_num"), ing.alias("ing")]
    if "HOSPPROCEDENCIA" in df.columns:
        exprs.append(pl.col("HOSPPROCEDENCIA").str.strip_chars().alias("hospproc"))
    if con_res:
        alta = tr.a_fecha(pl.col("FECHAALTA"))
        est_b = tr.estancia_dias(ing, alta)
        if "CIP_ENCRIPTADO" in df.columns:
            exprs.append(alta.alias("alta"))
        etiquetas = df["TIPOALTA"].drop_nulls().unique().to_list()
        rol, canon = mapas_rol(etiquetas, cat.roles_tipoalta)
        if etiquetas:
            exprs.append(pl.col("TIPOALTA").replace_strict(rol, default="SIN_DATO", return_dtype=pl.String).fill_null("SIN_DATO").alias("rol"))
            exprs.append(pl.col("TIPOALTA").replace_strict(canon, default="(sin rol)", return_dtype=pl.String).fill_null(SIN_DATO).alias("etiqueta_roles"))
        else:
            exprs.append(pl.lit("SIN_DATO").alias("rol"))
            exprs.append(pl.lit(SIN_DATO).alias("etiqueta_roles"))
        exprs.append(pl.when(est_b >= 0).then(est_b).alias("estancia"))
        exprs.append(pl.when(alta.dt.year() == int(anio)).then(alta.dt.month().cast(pl.Int32)).alias("mes_alta"))
    return df.select(exprs)


def agregar_comorbilidad(
    base: pl.DataFrame, dx: pl.DataFrame, e: elixhauser.Elixhauser, e3: elixhauser.Elixhauser
) -> pl.DataFrame:
    """Agrega ``n_comorb`` y ``n_comorb_3c`` (categorías de Elixhauser en diagnósticos secundarios) solo para MDC 14 agrupable.

    El resto de las filas queda nulo: la comorbilidad solo se necesita para la exclusión EX03.
    """
    if dx.height != base.height:
        raise ValueError("la base y los diagnósticos deben tener las mismas filas")
    n = base.height
    m = base.select(((pl.col("mdc") == "14") & (pl.col("grd_estado") == "agrupable")).fill_null(False).alias("m"))["m"]
    a = np.full(n, -1, dtype=np.int32)
    a3 = np.full(n, -1, dtype=np.int32)
    if n and m.any():
        sub = dx.filter(m)
        pos = np.flatnonzero(m.to_numpy())
        a[pos] = elixhauser.calcular_flags(sub, esquema.DIAG_SEC, e)["N_COMORB_ELIX"].to_numpy()
        a3[pos] = elixhauser.calcular_flags(sub, esquema.DIAG_SEC, e3)["N_COMORB_ELIX"].to_numpy()
    return base.with_columns(pl.Series("n_comorb", a), pl.Series("n_comorb_3c", a3)).with_columns(
        pl.when(pl.col("n_comorb") >= 0).then(pl.col("n_comorb")).alias("n_comorb"),
        pl.when(pl.col("n_comorb_3c") >= 0).then(pl.col("n_comorb_3c")).alias("n_comorb_3c"),
    )


def marcar(base: pl.DataFrame, edad_adulto: int = 18) -> pl.DataFrame:
    """Indicadores de exclusión y de variante de cohorte (A a E). Los que usan el desenlace solo existen si hay ``rol``."""
    c = pl.col
    con_res = "rol" in base.columns
    f = lambda e: e.fill_null(False)  # noqa: E731
    b = base.with_columns((c("grd_estado") != "agrupable").alias("ex01"))
    b = b.with_columns(
        f(~c("ex01") & (c("mdc") == "15")).alias("ex02"),
        f(~c("ex01") & (c("mdc") == "14") & (c("n_comorb") == 0)).alias("ex03"),
        f(~c("ex01") & (c("mdc") == "14") & (c("n_comorb_3c") == 0)).alias("ex03_pipe_sd"),
        f((c("actividad") != ACTIVIDAD_HOSP) & (c("actividad") != SIN_DATO)).alias("ambulatoria"),
    )
    b = b.with_columns((~(c("ex01") | c("ex02") | c("ex03"))).alias("A"))
    b = b.with_columns(f(c("A") & (c("edad") >= edad_adulto)).alias("B"))
    b = b.with_columns(f(c("B") & (c("actividad") == ACTIVIDAD_HOSP)).alias("C"))
    if con_res:
        b = b.with_columns(c("rol").is_in(["EVENTO", "NO_EVENTO"]).alias("conocido"))
        b = b.with_columns(
            (c("C") & c("conocido")).alias("D"),
            f(c("ex03_pipe_sd") & (c("rol") == "NO_EVENTO")).alias("ex03_pipe"),
        )
        b = b.with_columns(
            f(c("D") & (c("rol") == "NO_EVENTO") & (c("estancia") > 0)).alias("E"),
            (~(c("ex01") | c("ex02") | c("ex03_pipe"))).alias("dura_pipe"),
        )
        b = b.with_columns(
            f(c("dura_pipe") & (c("estancia") > 0) & ~c("ambulatoria") & (c("rol") != "EVENTO")).alias("est_pipe"),
        )
        # lectura alternativa de la cohorte de estadía del pipeline: además se excluyen los censurados
        b = b.with_columns(f(c("est_pipe") & (c("rol") != "CENSURADO")).alias("est_pipe_sc"))
    return b


def leer_base(
    ctx: Contexto, anio: int, cat: catalogos.Catalogos, e: elixhauser.Elixhauser, e3: elixhauser.Elixhauser, extras: bool = False
) -> pl.DataFrame:
    """Lee un año con el cortafuegos y devuelve la base tipada con los indicadores de cohorte.

    En 2019 a 2023 se leen el desenlace y la fecha de alta; en el año expuesto solo columnas de estructura.
    Con ``extras`` se agregan hospital, identificador numérico, fechas y hospital de procedencia (Tarea 9); ninguna de
    esas columnas es de resultado, así que también se pueden leer en el año expuesto.
    """
    con_res = anio in set(ctx.umbrales.particion["anios_resultado"])
    cols = list(COLS_BASE) + (list(COLS_EXTRA) if extras else []) + (list(COLS_RESULTADO) if con_res else [])
    df = ctx.lector.leer(anio, cols, proposito="resultado" if con_res else "estructura", tarea=TAREA)
    base = tipar(df, anio, cat, ctx.umbrales)
    del df
    dx = ctx.lector.leer(anio, list(esquema.DIAG_SEC), proposito="estructura", tarea=TAREA)
    base = agregar_comorbilidad(base, dx, e, e3)
    del dx
    return marcar(base, int(ctx.umbrales["edad_adulto"]))


# ------------------------------------------------------------------ agregados de un año
def _conteo_roles(base: pl.DataFrame, filtro: pl.Expr) -> dict[str, int]:
    d = {r: 0 for r in ROLES}
    for rol, n in base.filter(filtro).group_by("rol").len().iter_rows():
        d[rol] = int(n)
    d["n"] = sum(d[r] for r in ROLES)
    return d


def _dias(base: pl.DataFrame, filtro: pl.Expr) -> dict[int, int]:
    return {int(d): int(n) for d, n in base.filter(filtro).group_by("estancia").len().iter_rows() if d is not None}


def resumir_anio(base: pl.DataFrame) -> dict[str, Any]:
    """Todos los agregados de un año (sin suprimir). Si no hay ``rol`` (año expuesto) solo entrega la parte de estructura."""
    c = pl.col
    con_res = "rol" in base.columns
    llaves = ["ex01", "ex02", "ex03", "ex03_pipe_sd", "A", "B", "C", "ambulatoria"]
    if con_res:
        llaves += ["D", "E", "ex03_pipe", "dura_pipe", "est_pipe", "est_pipe_sc"]
    fila = base.select(pl.len().alias("total"), *[c(k).sum().alias(k) for k in llaves]).row(0, named=True)
    cas = {k: int(v) for k, v in fila.items()}
    out: dict[str, Any] = {"cas": cas}
    if not con_res:
        return out
    filtros = {
        "todos": pl.lit(True), "A": c("A"), "B": c("B"), "C": c("C"), "D": c("D"), "E": c("E"),
        "Bp": c("B") & c("conocido"), "Ap": c("A") & c("conocido"),
    }
    out["roles"] = {nombre: _conteo_roles(base, f) for nombre, f in filtros.items()}
    todos = out["roles"]["todos"]
    cas["fallecidos"], cas["censurados"], cas["sin_dato"] = todos["EVENTO"], todos["CENSURADO"], todos["SIN_DATO"]
    cas["D_fallecidos"] = out["roles"]["D"]["EVENTO"]
    cas["cero"] = int(base.select((c("estancia") == 0).sum()).item() or 0)
    out["tipoalta"] = base.group_by(["etiqueta_roles", "rol"]).len().rename({"len": "n"})
    # estratos de mortalidad: adultos de la cohorte dura con desenlace conocido (B sin censurados)
    bp = base.filter(c("B") & c("conocido"))
    variables = {
        "actividad_grupo": pl.when(c("actividad") == ACTIVIDAD_HOSP).then(pl.lit("hospitalización")).otherwise(pl.lit("ambulatoria")),
        "tipo_actividad": c("actividad"),
        "estancia_cero": pl.when(c("estancia").is_null()).then(pl.lit(SIN_DATO)).when(c("estancia") == 0).then(pl.lit("0 días")).otherwise(pl.lit("1 día o más")),
        "tipo_ingreso": c("tipo_ingreso"),
        "tipo_procedencia": c("procedencia"),
        "capitulo_cie10": c("capitulo"),
    }
    partes = []
    for nombre, expr in variables.items():
        g = bp.group_by(expr.alias("categoria")).agg(pl.len().alias("n"), (c("rol") == "EVENTO").sum().alias("n_fallecidos"))
        partes.append(g.with_columns(pl.lit(nombre).alias("variable")))
    out["estratos"] = pl.concat(partes).select(["variable", "categoria", "n", "n_fallecidos"])
    # edad y sexo: cohorte dura con desenlace conocido (A sin censurados)
    ap = base.filter(c("A") & c("conocido"))
    out["edad_sexo"] = ap.group_by([_grupo_edad_expr().alias("grupo_edad"), "sexo"]).agg(
        pl.len().alias("n"),
        (c("rol") == "EVENTO").sum().alias("n_fallecidos"),
        ((c("rol") == "NO_EVENTO") & c("estancia").is_not_null()).sum().alias("n_estancia"),
        pl.when(c("rol") == "NO_EVENTO").then(c("estancia").cast(pl.Int64)).sum().alias("suma_estancia"),
    )
    out["mensual"] = (
        base.filter(c("mes_alta").is_not_null())
        .group_by("mes_alta")
        .agg(pl.len().alias("n_egresos"), (c("rol") == "EVENTO").sum().alias("n_fallecidos"), (c("rol") == "CENSURADO").sum().alias("n_censurados"))
        .rename({"mes_alta": "mes"})
    )
    out["cero"] = base.group_by(["actividad", "rol"]).agg(
        pl.len().alias("n"), (c("estancia") == 0).sum().alias("n_cero"), c("estancia").is_not_null().sum().alias("n_estancia_valida")
    )
    out["dias_E"] = _dias(base, c("E"))
    out["dias_D"] = _dias(base, c("D") & (c("rol") == "NO_EVENTO") & c("estancia").is_not_null())
    return out


# ------------------------------------------------------------- períodos y cascadas
CLAVES_ESTRUCTURA = ("total", "ex01", "ex02", "ex03", "ex03_pipe_sd", "A", "B", "C", "ambulatoria")


def periodos_agregados(u: config.Umbrales, anios: Sequence[int]) -> dict[str, list[int]]:
    """Períodos agrupados que se pueden formar con los años disponibles (ranking y resultado)."""
    disponibles = set(anios)
    out: dict[str, list[int]] = {}
    for clave in ("anios_ranking", "anios_resultado"):
        lista = sorted(int(a) for a in u.particion[clave])
        if len(lista) > 1 and set(lista) <= disponibles:
            out[etiqueta_periodo(lista)] = lista
    return out


def filas_cascada(periodo: str, d: dict[str, int], con_res: bool) -> list[dict[str, Any]]:
    """Filas secuenciales de la cascada de un período. Lanza error si los pasos no cuadran con el total."""
    t = d["total"]
    valores: dict[str, int] = {
        "TOTAL": t, "EX01": d["ex01"], "EX02": d["ex02"], "EX03": d["ex03"], "A": d["A"],
        "EX04": d["A"] - d["B"], "B": d["B"], "EX05": d["B"] - d["C"], "C": d["C"],
    }
    if t - d["ex01"] - d["ex02"] - d["ex03"] != d["A"]:
        raise ValueError(f"la cascada de {periodo} no cuadra: el total menos las exclusiones es distinto de la cohorte A")
    if con_res:
        valores.update({
            "EX06": d["C"] - d["D"], "D": d["D"], "EX07": d["D_fallecidos"],
            "EX08": d["D"] - d["D_fallecidos"] - d["E"], "E": d["E"],
        })
        if min(valores["EX06"], valores["EX07"], valores["EX08"]) < 0:
            raise ValueError(f"la cascada de {periodo} tiene pasos negativos")
    filas = []
    for i, paso in enumerate(ORDEN_PASOS):
        if paso not in valores:
            continue
        tipo, desc, usa = DESCRIPCION_PASO[paso]
        filas.append({
            "periodo": periodo, "orden": i, "paso": paso, "tipo": tipo, "descripcion": desc, "n": int(valores[paso]),
            "prop_del_total": valores[paso] / t if t else None, "usa_desenlace": usa,
            "alcance": "resultado" if usa == "si" else "estructura", "cuadra": "si",
        })
    return filas


def filas_cascada_pipeline(periodo: str, d: dict[str, int], con_res: bool) -> list[dict[str, Any]]:
    """Cascada con la definición aproximada del pipeline (EX03 con sobrevivientes y mapeo de tres caracteres)."""
    t = d["total"]
    pasos: list[tuple[str, str, int, str]] = [
        ("TOTAL", "Egresos leídos", t, "estructura"),
        ("EX01", "GRD no agrupable", d["ex01"], "estructura"),
        ("EX02", "Recién nacidos (MDC 15)", d["ex02"], "estructura"),
        ("EX03_SIN_DESENLACE", "MDC 14 sin comorbilidades con mapeo de tres caracteres, sin mirar el desenlace (cota superior de EX03)", d["ex03_pipe_sd"], "estructura"),
        ("AMBULATORIA", "Actividad distinta de hospitalización (se solapa con las demás)", d["ambulatoria"], "estructura"),
    ]
    if con_res:
        if t - d["ex01"] - d["ex02"] - d["ex03_pipe"] != d["dura_pipe"]:
            raise ValueError(f"la cascada del pipeline de {periodo} no cuadra")
        pasos += [
            ("EX03_PIPELINE", "Obstétricos sin comorbilidades, solo sobrevivientes (mapeo de tres caracteres)", d["ex03_pipe"], "resultado"),
            ("COHORTE_DURA_PIPELINE", "Cohorte dura con la definición del pipeline", d["dura_pipe"], "resultado"),
            ("ESTANCIA_CERO", "Estadía de 0 días (se solapa con las demás)", d["cero"], "resultado"),
            ("FALLECIDOS", "Fallecidos (se solapan con las demás)", d["fallecidos"], "resultado"),
            ("CENSURADOS", "Censurados por tipo de alta", d["censurados"], "resultado"),
            ("COHORTE_ESTANCIA_PIPELINE", "Cohorte dura del pipeline sin estadía de 0 días, actividad ambulatoria ni fallecidos", d["est_pipe"], "resultado"),
            ("COHORTE_ESTANCIA_PIPELINE_SIN_CENSURADOS", "Lectura alternativa: la cohorte anterior sin censurados", d["est_pipe_sc"], "resultado"),
        ]
    return [
        {"periodo": periodo, "paso": p, "descripcion": desc, "n": int(n), "prop_del_total": (n / t if t else None), "alcance": al}
        for p, desc, n, al in pasos
    ]


def _periodos(por_anio: dict[int, dict[str, Any]], u: config.Umbrales) -> list[tuple[str, list[int], bool]]:
    """(etiqueta, años, ¿con desenlace?) de cada año y de los períodos agregados."""
    anios = sorted(por_anio)
    res_anios = [a for a in anios if "roles" in por_anio[a]]
    out = [(str(a), [a], a in res_anios) for a in anios]
    cfg_res = sorted(int(x) for x in u.particion["anios_resultado"])
    if len(cfg_res) > 1 and set(cfg_res) <= set(res_anios):
        out.append((etiqueta_periodo(cfg_res), cfg_res, True))
    if len(anios) > 1 and set(anios) - set(res_anios):
        out.append((etiqueta_periodo(anios), anios, False))
    return out


ESQUEMA_CASCADA = {
    "periodo": pl.String, "orden": pl.Int64, "paso": pl.String, "tipo": pl.String, "descripcion": pl.String, "n": pl.Int64,
    "prop_del_total": pl.Float64, "usa_desenlace": pl.String, "alcance": pl.String, "cuadra": pl.String,
}
ESQUEMA_PIPELINE = {
    "periodo": pl.String, "paso": pl.String, "descripcion": pl.String, "n": pl.Int64, "prop_del_total": pl.Float64, "alcance": pl.String,
}
ESQUEMA_VARIANTES = {
    "periodo": pl.String, "variante": pl.String, "descripcion": pl.String, "n": pl.Int64, "n_fallecidos": pl.Int64, "n_no_evento": pl.Int64,
    "n_censurados": pl.Int64, "n_sin_dato": pl.Int64, "prevalencia": pl.Float64, "prevalencia_conocida": pl.Float64, "dif_prevalencia_pp": pl.Float64,
}


def con_wilson(df: pl.DataFrame, k: str, n: str, sufijo: str, z: float) -> pl.DataFrame:
    """Agrega ``ic_bajo<sufijo>`` e ``ic_alto<sufijo>`` (Wilson) de la proporción ``k/n``; nulos donde ``k`` o ``n`` son nulos."""
    ok = df[k].is_not_null() & df[n].is_not_null()
    lo, hi = estadistica.wilson(df[k].fill_null(0).to_numpy(), df[n].fill_null(0).to_numpy(), z)
    return df.with_columns(
        pl.when(ok).then(pl.Series(lo)).alias(f"ic_bajo{sufijo}"), pl.when(ok).then(pl.Series(hi)).alias(f"ic_alto{sufijo}")
    )


def construir_cascadas(por_anio: dict[int, dict[str, Any]], u: config.Umbrales) -> dict[str, pl.DataFrame]:
    """Tablas de cascada, de cascada del pipeline y de variantes para cada año y para los períodos agregados."""
    periodos = _periodos(por_anio, u)
    filas_cas: list[dict] = []
    filas_pipe: list[dict] = []
    filas_var: list[dict] = []
    cfg_res = etiqueta_periodo(sorted(int(x) for x in u.particion["anios_resultado"]))
    for nombre, lista, con_res in periodos:
        if con_res:
            d = _sumar([por_anio[a]["cas"] for a in lista])
        else:
            d = _sumar([{k: v for k, v in por_anio[a]["cas"].items() if k in CLAVES_ESTRUCTURA} for a in lista])
        filas_cas += filas_cascada(nombre, d, con_res)
        filas_pipe += filas_cascada_pipeline(nombre, d, con_res)
        # variantes: solo años sueltos y el período de resultado
        if len(lista) == 1 or nombre == cfg_res:
            previa: float | None = None
            for v in "ABCDE":
                if con_res:
                    r = _sumar([por_anio[a]["roles"][v] for a in lista])
                    k, nn, cen, sd, n = r["EVENTO"], r["NO_EVENTO"], r["CENSURADO"], r["SIN_DATO"], r["n"]
                    conocidos = k + nn
                    # la variante E solo tiene sobrevivientes: su prevalencia no está definida
                    prev_c = k / conocidos if (conocidos and v != "E") else None
                    filas_var.append({
                        "periodo": nombre, "variante": v, "descripcion": DESCRIPCION_VARIANTE[v], "n": n, "n_fallecidos": k,
                        "n_no_evento": nn, "n_censurados": cen, "n_sin_dato": sd,
                        "prevalencia": (k / n) if (n and v != "E") else None,
                        "prevalencia_conocida": prev_c,
                        "dif_prevalencia_pp": (100.0 * (prev_c - previa)) if (prev_c is not None and previa is not None) else None,
                    })
                    if prev_c is not None:
                        previa = prev_c
                elif v in "ABC":  # año sin desenlace: solo el tamaño
                    filas_var.append({
                        "periodo": nombre, "variante": v, "descripcion": DESCRIPCION_VARIANTE[v], "n": d[v], "n_fallecidos": None,
                        "n_no_evento": None, "n_censurados": None, "n_sin_dato": None, "prevalencia": None, "prevalencia_conocida": None,
                        "dif_prevalencia_pp": None,
                    })
    var = pl.DataFrame(filas_var, schema=ESQUEMA_VARIANTES)
    k_nn = (var["n_fallecidos"] + var["n_no_evento"])
    var = var.with_columns(k_nn.alias("_conocidos"))
    var = con_wilson(var, "n_fallecidos", "_conocidos", "_conocida", u["z_95"])
    var = con_wilson(var, "n_fallecidos", "n", "", u["z_95"]).drop("_conocidos")
    # la variante E solo tiene sobrevivientes: no hay prevalencia ni intervalo que informar
    var = var.with_columns(
        [pl.when(pl.col("variante") == "E").then(None).otherwise(pl.col(c)).alias(c) for c in ("ic_bajo", "ic_alto", "ic_bajo_conocida", "ic_alto_conocida")]
    )
    var = var.select(["periodo", "variante", "descripcion", "n", "n_fallecidos", "n_no_evento", "n_censurados", "n_sin_dato", "prevalencia", "ic_bajo",
                      "ic_alto", "prevalencia_conocida", "ic_bajo_conocida", "ic_alto_conocida", "dif_prevalencia_pp"])
    return {
        "cascada": pl.DataFrame(filas_cas, schema=ESQUEMA_CASCADA),
        "cascada_pipeline": pl.DataFrame(filas_pipe, schema=ESQUEMA_PIPELINE),
        "variantes": var,
    }


# --------------------------------------------------------------- resultado y estratos
def _con_anio(frames: dict[int, pl.DataFrame]) -> pl.DataFrame:
    return pl.concat([f.with_columns(pl.lit(int(a), dtype=pl.Int64).alias("anio")) for a, f in sorted(frames.items())])


def con_periodos(df: pl.DataFrame, agregados: dict[str, list[int]], claves: Sequence[str], sumas: Sequence[str]) -> pl.DataFrame:
    """Suma los conteos anuales en los períodos agrupados. Devuelve ``periodo`` (año o período) más ``claves`` y ``sumas``."""
    base = df.select(["anio", *claves, *sumas])
    partes = [base.with_columns(pl.col("anio").cast(pl.String).alias("periodo")).drop("anio")]
    presentes = set(base["anio"].unique().to_list())
    for nombre, anios in agregados.items():
        if not set(anios) <= presentes:
            continue
        g = (
            base.filter(pl.col("anio").is_in(anios))
            .group_by(list(claves))
            .agg([pl.col(c).sum() for c in sumas])
            .with_columns(pl.lit(nombre).alias("periodo"))
        )
        partes.append(g.select(partes[0].columns))
    return pl.concat(partes)


def _anios_resultado(por_anio: dict[int, dict[str, Any]]) -> list[int]:
    return sorted(a for a in por_anio if "roles" in por_anio[a])


def _umbral_cambio(u: config.Umbrales) -> float:
    return float(u["cambio_relativo_relevante"])


def _clasificar_cambio(rr: float, bajo: float, alto: float, umbral: float) -> str:
    """``alza`` o ``baja`` si el IC de la razón excluye 1 y el cambio relativo alcanza el umbral; ``no`` si no."""
    if any(np.isnan(x) for x in (rr, bajo, alto)):
        return "no evaluable"
    if (bajo > 1.0 or alto < 1.0) and abs(rr - 1.0) >= umbral:
        return "alza" if rr > 1.0 else "baja"
    return "no"


def tabla_resultado_anio(por_anio: dict[int, dict[str, Any]], u: config.Umbrales) -> pl.DataFrame:
    """Mortalidad por año y período para todos los egresos y para la cohorte del modelo (D), con razón contra el año previo y contra el primero."""
    z = float(u["z_95"])
    res = _anios_resultado(por_anio)
    agreg = periodos_agregados(u, res)
    post = [a for a in sorted(int(x) for x in u.particion["anios_ranking"]) if res and a != res[0] and a in res]
    if len(post) > 1:
        agreg[etiqueta_periodo(post)] = post
    periodos = [(str(a), [a]) for a in res] + list(agreg.items())
    filas = []
    for nombre, lista in periodos:
        for base in ("todos", "D"):
            r = _sumar([por_anio[a]["roles"][base] for a in lista])
            filas.append({
                "periodo": nombre, "base": base, "n": r["n"], "n_fallecidos": r["EVENTO"], "n_no_evento": r["NO_EVENTO"],
                "n_censurados": r["CENSURADO"], "n_sin_dato": r["SIN_DATO"],
            })
    esq = {"periodo": pl.String, "base": pl.String, "n": pl.Int64, "n_fallecidos": pl.Int64, "n_no_evento": pl.Int64, "n_censurados": pl.Int64, "n_sin_dato": pl.Int64}
    df = pl.DataFrame(filas, schema=esq)
    df = df.with_columns(
        (pl.col("n_fallecidos") / pl.col("n")).alias("mortalidad"),
        (pl.col("n_fallecidos") / (pl.col("n_fallecidos") + pl.col("n_no_evento"))).alias("mortalidad_conocida"),
        (pl.col("n_fallecidos") + pl.col("n_no_evento")).alias("_conocidos"),
    )
    df = con_wilson(df, "n_fallecidos", "n", "", z)
    df = con_wilson(df, "n_fallecidos", "_conocidos", "_conocida", z).drop("_conocidos")

    kn = {(r["periodo"], r["base"]): (r["n_fallecidos"], r["n"]) for r in df.iter_rows(named=True)}
    umbral = _umbral_cambio(u)
    cols: dict[str, list] = {c: [] for c in ("rr_previo", "rr_previo_bajo", "rr_previo_alto", "quiebre_anual", "rr_primero", "rr_primero_bajo", "rr_primero_alto")}
    primero = str(res[0]) if res else ""
    for r in df.iter_rows(named=True):
        p, b = r["periodo"], r["base"]
        k1, n1 = kn[(p, b)]
        previo = str(int(p) - 1) if p.isdigit() else None
        if previo is not None and (previo, b) in kn:
            rr, lo, hi = estadistica.razon_de_proporciones(k1, n1, *kn[(previo, b)], z)
            cols["rr_previo"].append(rr); cols["rr_previo_bajo"].append(lo); cols["rr_previo_alto"].append(hi)
            cols["quiebre_anual"].append(_clasificar_cambio(rr, lo, hi, umbral))
        else:
            for c in ("rr_previo", "rr_previo_bajo", "rr_previo_alto"):
                cols[c].append(None)
            cols["quiebre_anual"].append(None)
        if p != primero and (primero, b) in kn:
            rr, lo, hi = estadistica.razon_de_proporciones(k1, n1, *kn[(primero, b)], z)
            cols["rr_primero"].append(rr); cols["rr_primero_bajo"].append(lo); cols["rr_primero_alto"].append(hi)
        else:
            for c in ("rr_primero", "rr_primero_bajo", "rr_primero_alto"):
                cols[c].append(None)
    schema_cols = {c: (pl.String if c == "quiebre_anual" else pl.Float64) for c in cols}
    return df.with_columns([pl.Series(c, v, dtype=schema_cols[c]) for c, v in cols.items()])


def tabla_tipoalta(por_anio: dict[int, dict[str, Any]], u: config.Umbrales) -> pl.DataFrame:
    """Distribución de las etiquetas de TIPOALTA (con su rol) por año y período, sobre todos los egresos."""
    res = _anios_resultado(por_anio)
    frames = {a: por_anio[a]["tipoalta"] for a in res}
    df = con_periodos(_con_anio(frames), periodos_agregados(u, res), ["etiqueta_roles", "rol"], ["n"])
    df = df.with_columns(
        (pl.col("n") / pl.col("n").sum().over("periodo")).alias("prop"),
        pl.col("etiqueta_roles").map_elements(codigo_tipoalta, return_dtype=pl.String).alias("codigo"),
    )
    return df.select(["periodo", "etiqueta_roles", "codigo", "rol", "n", "prop"]).rename({"etiqueta_roles": "etiqueta"}).sort(
        ["periodo", "n", "etiqueta", "rol"], descending=[False, True, False, False]
    )


VARIABLES_DECISION_15 = {"actividad_grupo": "ambulatoria", "estancia_cero": "0 días"}


def tabla_estratos(por_anio: dict[int, dict[str, Any]], u: config.Umbrales) -> pl.DataFrame:
    """Mortalidad por estrato (actividad, estadía cero, tipo de ingreso, procedencia y capítulo) en B sin censurados."""
    z = float(u["z_95"])
    res = _anios_resultado(por_anio)
    frames = {a: por_anio[a]["estratos"] for a in res}
    df = con_periodos(_con_anio(frames), periodos_agregados(u, res), ["variable", "categoria"], ["n", "n_fallecidos"])
    tot = df.group_by(["periodo", "variable"]).agg(pl.col("n").sum().alias("N"), pl.col("n_fallecidos").sum().alias("K"))
    df = df.join(tot, on=["periodo", "variable"], how="left")
    umbral = float(u["estrato_mortalidad_relativa_max"])
    rr, lo, hi, resto, flag = [], [], [], [], []
    for r in df.iter_rows(named=True):
        n_r, k_r = r["N"] - r["n"], r["K"] - r["n_fallecidos"]
        resto.append(k_r / n_r if n_r > 0 else None)
        q, a, b = estadistica.razon_de_proporciones(r["n_fallecidos"], r["n"], k_r, n_r, z)
        rr.append(q); lo.append(a); hi.append(b)
        if VARIABLES_DECISION_15.get(r["variable"]) == r["categoria"]:
            flag.append("si" if (q == q and q < umbral) else "no")
        else:
            flag.append("")
    df = df.with_columns(
        (pl.col("n") / pl.col("N")).alias("participacion"),
        (pl.col("n_fallecidos") / pl.col("n")).alias("mortalidad"),
        pl.Series("mortalidad_resto", resto, dtype=pl.Float64),
        pl.Series("razon_vs_resto", rr, dtype=pl.Float64),
        pl.Series("razon_bajo", lo, dtype=pl.Float64),
        pl.Series("razon_alto", hi, dtype=pl.Float64),
        pl.Series("bajo_umbral", flag, dtype=pl.String),
    )
    df = con_wilson(df, "n_fallecidos", "n", "", z)
    return df.select([
        "periodo", "variable", "categoria", "n", "n_fallecidos", "participacion", "mortalidad", "ic_bajo", "ic_alto", "mortalidad_resto",
        "razon_vs_resto", "razon_bajo", "razon_alto", "bajo_umbral",
    ]).sort(["variable", "periodo", "n", "categoria"], descending=[False, False, True, False])


def tabla_edad_sexo(por_anio: dict[int, dict[str, Any]], u: config.Umbrales) -> pl.DataFrame:
    """Mortalidad y estadía de sobrevivientes por grupo de edad y sexo en A sin censurados."""
    z = float(u["z_95"])
    res = _anios_resultado(por_anio)
    frames = {a: por_anio[a]["edad_sexo"] for a in res}
    df = con_periodos(_con_anio(frames), periodos_agregados(u, res), ["grupo_edad", "sexo"], ["n", "n_fallecidos", "n_estancia", "suma_estancia"])
    orden = {nombre: i for i, (nombre, _, _) in enumerate(GRUPOS_EDAD)}
    df = df.with_columns(
        pl.col("grupo_edad").replace_strict(orden, default=99, return_dtype=pl.Int64).alias("orden"),
        (pl.col("n_fallecidos") / pl.col("n")).alias("mortalidad"),
        (pl.col("suma_estancia") / pl.col("n_estancia")).alias("estancia_media"),
    )
    df = con_wilson(df, "n_fallecidos", "n", "", z)
    return df.select(["periodo", "orden", "grupo_edad", "sexo", "n", "n_fallecidos", "mortalidad", "ic_bajo", "ic_alto", "n_estancia", "estancia_media"]).sort(["periodo", "orden", "sexo"])


def tabla_estancia_cero(por_anio: dict[int, dict[str, Any]], u: config.Umbrales) -> pl.DataFrame:
    """Composición de las estadías de 0 días por actividad y desenlace, sobre todos los egresos."""
    res = _anios_resultado(por_anio)
    frames = {a: por_anio[a]["cero"] for a in res}
    df = con_periodos(_con_anio(frames), periodos_agregados(u, res), ["actividad", "rol"], ["n", "n_cero", "n_estancia_valida"])
    return df.select(["periodo", "actividad", "rol", "n", "n_cero", "n_estancia_valida"]).sort(["periodo", "actividad", "rol"])


# ------------------------------------------------------------------------ serie mensual
def detectar_quiebres(
    anios: Sequence[int], meses: Sequence[int], n: Sequence[int], k: Sequence[int], z: float, umbral: float
) -> list[dict[str, Any]]:
    """Compara la mortalidad de cada mes con la del MISMO mes del año anterior (razón de proporciones de Katz).

    Es un cambio anual, como lo define ``cambio_relativo_relevante``, y controla la estacionalidad: comparar con los meses
    inmediatamente anteriores marca como cambio el ciclo invernal. Un mes se marca ``alza`` o ``baja`` si el IC de la
    razón excluye 1 y el cambio relativo alcanza ``umbral``. Los meses del primer año no tienen referencia. Los IC no se
    ajustan por comparaciones múltiples: son indicativos.
    """
    datos = {(int(a), int(m)): (int(nn), int(kk)) for a, m, nn, kk in zip(anios, meses, n, k)}
    out = []
    for a, m in zip(anios, meses):
        ref = datos.get((int(a) - 1, int(m)))
        if ref is None:
            out.append({"rr": None, "bajo": None, "alto": None, "cambio": None, "quiebre": None})
            continue
        n1, k1 = datos[(int(a), int(m))]
        rr, lo, hi = estadistica.razon_de_proporciones(k1, n1, ref[1], ref[0], z)
        out.append({
            "rr": rr if rr == rr else None, "bajo": lo if lo == lo else None, "alto": hi if hi == hi else None,
            "cambio": (rr - 1.0) if rr == rr else None, "quiebre": _clasificar_cambio(rr, lo, hi, umbral),
        })
    return out


def tabla_serie_mensual(por_anio: dict[int, dict[str, Any]], u: config.Umbrales) -> pl.DataFrame:
    z = float(u["z_95"])
    res = _anios_resultado(por_anio)
    df = _con_anio({a: por_anio[a]["mensual"] for a in res}).sort(["anio", "mes"])
    q = detectar_quiebres(df["anio"].to_list(), df["mes"].to_list(), df["n_egresos"].to_list(), df["n_fallecidos"].to_list(), z, _umbral_cambio(u))
    df = df.with_columns(
        (pl.col("anio").cast(pl.String) + "-" + pl.col("mes").cast(pl.String).str.zfill(2)).alias("mes_id"),
        (pl.col("n_fallecidos") / pl.col("n_egresos")).alias("mortalidad"),
        pl.Series("rr_mismo_mes", [x["rr"] for x in q], dtype=pl.Float64),
        pl.Series("rr_bajo", [x["bajo"] for x in q], dtype=pl.Float64),
        pl.Series("rr_alto", [x["alto"] for x in q], dtype=pl.Float64),
        pl.Series("cambio_relativo", [x["cambio"] for x in q], dtype=pl.Float64),
        pl.Series("quiebre", [x["quiebre"] for x in q], dtype=pl.String),
    )
    df = con_wilson(df, "n_fallecidos", "n_egresos", "", z)
    return df.select([
        "anio", "mes", "mes_id", "n_egresos", "n_fallecidos", "n_censurados", "mortalidad", "ic_bajo", "ic_alto",
        "rr_mismo_mes", "rr_bajo", "rr_alto", "cambio_relativo", "quiebre",
    ])


# --------------------------------------------------------------------------------- estadía
def _sumar_dias(por_anio: dict[int, dict[str, Any]], clave: str, anios: Sequence[int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for a in anios:
        for d, n in por_anio[a][clave].items():
            out[d] = out.get(d, 0) + n
    return out


def estadisticas_dias(dias: dict[int, int]) -> dict[str, float | int | None]:
    """Resumen de una distribución de estadías dada como {días: episodios}."""
    if not dias:
        return {"n": 0, "media": None, "varianza": None, "indice_dispersion": None, "cv": None, "mediana": None, "p90": None, "p99": None, "maximo": None}
    v, c = list(dias), list(dias.values())
    n, media, var = estadistica.momentos_de_conteos(v, c)
    ok = var == var and media and media > 0
    return {
        "n": n, "media": media, "varianza": var if var == var else None,
        "indice_dispersion": (var / media) if ok else None, "cv": (float(np.sqrt(var)) / media) if ok else None,
        "mediana": estadistica.percentil_de_conteos(v, c, 0.5), "p90": estadistica.percentil_de_conteos(v, c, 0.9),
        "p99": estadistica.percentil_de_conteos(v, c, 0.99), "maximo": max(v),
    }


BASES_ESTANCIA = {"E": ("dias_E", "E: sobrevivientes con estadía mayor que cero"), "D_sobrevivientes": ("dias_D", "Sobrevivientes de D (incluye estadía cero)")}


def tabla_dispersion(por_anio: dict[int, dict[str, Any]], u: config.Umbrales) -> pl.DataFrame:
    res = _anios_resultado(por_anio)
    periodos = [(str(a), [a]) for a in res] + list(periodos_agregados(u, res).items())
    filas = []
    for base, (clave, _) in BASES_ESTANCIA.items():
        for nombre, lista in periodos:
            filas.append({"periodo": nombre, "base": base, **estadisticas_dias(_sumar_dias(por_anio, clave, lista))})
    esq = {"periodo": pl.String, "base": pl.String, "n": pl.Int64, "media": pl.Float64, "varianza": pl.Float64, "indice_dispersion": pl.Float64,
           "cv": pl.Float64, "mediana": pl.Float64, "p90": pl.Float64, "p99": pl.Float64, "maximo": pl.Float64}
    return pl.DataFrame(filas, schema=esq)


def histograma_estancia(dias: dict[int, int]) -> list[tuple[int, str, int]]:
    out = []
    for i, (nombre, lo, hi) in enumerate(TRAMOS_ESTANCIA):
        out.append((i, nombre, sum(n for d, n in dias.items() if lo <= d <= hi)))
    return out


def tabla_hist_estancia(por_anio: dict[int, dict[str, Any]], u: config.Umbrales) -> pl.DataFrame:
    res = _anios_resultado(por_anio)
    periodos = [(str(a), [a]) for a in res] + list(periodos_agregados(u, res).items())
    filas = []
    for nombre, lista in periodos:
        h = histograma_estancia(_sumar_dias(por_anio, "dias_D", lista))
        total = sum(n for _, _, n in h)
        for orden, tramo, n in h:
            filas.append({"periodo": nombre, "orden": orden, "tramo": tramo, "n": n, "prop": n / total if total else None})
    return pl.DataFrame(filas, schema={"periodo": pl.String, "orden": pl.Int64, "tramo": pl.String, "n": pl.Int64, "prop": pl.Float64})


def _n_sobre(dias: dict[int, int], corte: float) -> int:
    return sum(n for d, n in dias.items() if d > corte)


ESQUEMA_COLA = {
    "base": pl.String, "regla": pl.String, "corte_dias": pl.Int64, "n_def": pl.Int64, "n_afectados_def": pl.Int64, "prop_afectados_def": pl.Float64,
    "cumple_def": pl.String, "n_conf": pl.Int64, "n_afectados_conf": pl.Int64, "prop_afectados_conf": pl.Float64, "cumple_conf": pl.String,
    "propuesta": pl.String, "maximo_def": pl.Int64,
}


def tabla_cola_estancia(por_anio: dict[int, dict[str, Any]], u: config.Umbrales) -> pl.DataFrame:
    """Reglas de corte de estadías largas definidas con 2019 a 2022 y confirmadas en 2023 (decisión 7).

    Para cada base de sobrevivientes se toman los percentiles candidatos de 2019 a 2022; se informa a cuántos
    sobrevivientes afectaría cada corte (estadía mayor que el corte) en ese período y en 2023. La propuesta es el
    percentil más bajo que no supere ``cola_estadia_perdida_max`` en 2019 a 2022.
    """
    res = _anios_resultado(por_anio)
    rank = [a for a in sorted(int(x) for x in u.particion["anios_ranking"]) if a in res]
    conf = int(u.particion["anio_confirmacion"])
    if not rank or conf not in res:
        return pl.DataFrame(schema=ESQUEMA_COLA)
    tope = float(u["cola_estadia_perdida_max"])
    reglas = [(f"p{str(round(q * 100, 2)).rstrip('0').rstrip('.').replace('.', ',')}", q) for q in u["cola_estadia_percentiles"]]
    reglas.append((f"{int(u['estancia_maxima_dias'])} días", None))
    filas = []
    for base, (clave, _) in BASES_ESTANCIA.items():
        d_def, d_conf = _sumar_dias(por_anio, clave, rank), _sumar_dias(por_anio, clave, [conf])
        n_def, n_conf = sum(d_def.values()), sum(d_conf.values())
        maximo = max(d_def) if d_def else None
        elegido = False
        for nombre, q in reglas:
            if q is None:
                corte = int(u["estancia_maxima_dias"])
            else:
                corte = int(estadistica.percentil_de_conteos(list(d_def), list(d_def.values()), q))
            a_def, a_conf = _n_sobre(d_def, corte), _n_sobre(d_conf, corte)
            p_def = a_def / n_def if n_def else None
            p_conf = a_conf / n_conf if n_conf else None
            cumple = p_def is not None and p_def <= tope
            propuesta = "no"
            if q is not None and cumple and not elegido:
                propuesta, elegido = "si", True
            filas.append({
                "base": base, "regla": nombre, "corte_dias": corte, "n_def": n_def, "n_afectados_def": a_def, "prop_afectados_def": p_def,
                "cumple_def": "si" if cumple else "no", "n_conf": n_conf, "n_afectados_conf": a_conf, "prop_afectados_conf": p_conf,
                "cumple_conf": "si" if (p_conf is not None and p_conf <= tope) else "no", "propuesta": propuesta, "maximo_def": maximo,
            })
    return pl.DataFrame(filas, schema=ESQUEMA_COLA)


# ---------------------------------------------------------------------------- ejecución
def autoprueba_2024(ctx: Contexto) -> str:
    """Intenta leer el desenlace del año expuesto; el cortafuegos debe rechazarlo y dejarlo en el registro de accesos."""
    anio = int(ctx.umbrales.particion["anio_expuesto"])
    if anio not in ctx.fuente.archivos:
        return "no aplica"
    try:
        ctx.lector.leer(anio, ["TIPOALTA"], proposito="resultado", tarea=TAREA)
    except AccesoBloqueado:
        return "rechazado"
    raise RuntimeError(f"el cortafuegos permitió leer el desenlace de {anio}: la partición temporal está rota")


def _resumen(r: dict[str, Any], u: config.Umbrales) -> dict[str, Any]:
    """Cifras derivadas que usan los informes y que no están en una sola fila de otra tabla."""
    por_anio = r["por_anio"]
    res = _anios_resultado(por_anio)
    if not res:
        return {}
    cero = _sumar([{f"{a}|{b}": int(n) for a, b, n in por_anio[x]["cero"].select(["actividad", "rol", "n_cero"]).iter_rows()} for x in res])
    tot_act = _sumar([{f"{a}": int(n) for a, n in por_anio[x]["cero"].group_by("actividad").agg(pl.col("n").sum()).iter_rows()} for x in res])
    n_cero = sum(cero.values())
    amb = lambda k: k.split("|")[0] != ACTIVIDAD_HOSP  # noqa: E731
    n_cero_amb = sum(v for k, v in cero.items() if amb(k))
    por_rol = {rol: sum(v for k, v in cero.items() if k.split("|")[1] == rol) for rol in ROLES}
    n_amb = sum(v for k, v in tot_act.items() if k not in (ACTIVIDAD_HOSP, SIN_DATO))
    conocidos_cero = por_rol["EVENTO"] + por_rol["NO_EVENTO"]
    mens = r["serie_mensual"]
    evaluados = mens.filter(pl.col("quiebre").is_in(["no", "alza", "baja"]))
    return {
        "n_cero_total": n_cero, "n_cero_ambulatoria": n_cero_amb, "n_cero_fallecidos": por_rol["EVENTO"],
        "n_cero_censurados": por_rol["CENSURADO"], "n_cero_sobrevivientes": por_rol["NO_EVENTO"],
        "prop_cero_ambulatoria": n_cero_amb / n_cero if n_cero else None,
        "prop_cero_fallecidos": por_rol["EVENTO"] / n_cero if n_cero else None,
        "prop_cero_censurados": por_rol["CENSURADO"] / n_cero if n_cero else None,
        "mortalidad_cero_conocida": por_rol["EVENTO"] / conocidos_cero if conocidos_cero else None,
        "n_ambulatoria": n_amb, "n_ambulatoria_cero": n_cero_amb,
        "prop_ambulatoria_cero": n_cero_amb / n_amb if n_amb else None,
        "n_meses_evaluados": evaluados.height,
        "n_meses_alza": int((mens["quiebre"] == "alza").sum()),
        "n_meses_baja": int((mens["quiebre"] == "baja").sum()),
        "intento_2024": r["intento_2024"],
    }


def calcular(ctx: Contexto, cat: catalogos.Catalogos | None = None) -> dict[str, Any]:
    """Recorre los años una sola vez y devuelve todas las tablas (sin suprimir)."""
    cat = cat or catalogos.cargar_catalogos()
    e, e3 = elixhauser.cargar(), elixhauser.cargar(solo_3c=True)
    por_anio: dict[int, dict[str, Any]] = {}
    for anio in ctx.fuente.anios():
        base = leer_base(ctx, anio, cat, e, e3)
        por_anio[anio] = resumir_anio(base)
        del base
    u = ctx.umbrales
    out: dict[str, Any] = {"por_anio": por_anio}
    out.update(construir_cascadas(por_anio, u))
    if _anios_resultado(por_anio):
        out["tipoalta"] = tabla_tipoalta(por_anio, u)
        out["resultado_anio"] = tabla_resultado_anio(por_anio, u)
        out["estratos"] = tabla_estratos(por_anio, u)
        out["edad_sexo"] = tabla_edad_sexo(por_anio, u)
        out["serie_mensual"] = tabla_serie_mensual(por_anio, u)
        out["estancia_cero"] = tabla_estancia_cero(por_anio, u)
        out["hist_estancia"] = tabla_hist_estancia(por_anio, u)
        out["dispersion"] = tabla_dispersion(por_anio, u)
        out["cola"] = tabla_cola_estancia(por_anio, u)
    out["intento_2024"] = autoprueba_2024(ctx)
    if _anios_resultado(por_anio):
        out["resumen"] = pl.DataFrame([_resumen(out, u)])
    return out


CONTEOS_VARIANTE = ("n", "n_fallecidos", "n_no_evento", "n_censurados", "n_sin_dato")
SIGUIENTE_VARIANTE = {"A": "B", "B": "C", "C": "D", "D": "E"}


def con_variante_siguiente(var: pl.DataFrame) -> pl.DataFrame:
    """Agrega ``_hijo_<conteo>``: el valor de la variante siguiente del mismo período (columnas auxiliares de la supresión complementaria).

    Cada variante es la anterior menos un paso de la cascada; la resta de dos variantes vecinas es el conteo de ese paso y también se protege.
    """
    hijo = var.select("periodo", pl.col("variante").alias("_v"), *[pl.col(c).alias(f"_hijo_{c}") for c in CONTEOS_VARIANTE])
    padre = var.with_columns(pl.col("variante").replace_strict(SIGUIENTE_VARIANTE, default=None, return_dtype=pl.String).alias("_v"))
    return padre.join(hijo, on=["periodo", "_v"], how="left", maintain_order="left").drop("_v")


def con_total_secreto(est: pl.DataFrame, resultado_anio: pl.DataFrame, minimo: int) -> pl.DataFrame:
    """Agrega ``_T_oculto``: en qué períodos el total de fallecidos de la base de los estratos no puede deducirse.

    La base de los estratos (adultos de la cohorte dura con desenlace conocido) es la cohorte D más la actividad ambulatoria. Los fallecidos
    de D son públicos (``t23_resultado_anio.csv``), así que el total de la base menos los de D deja los fallecidos en actividad ambulatoria.
    Donde ese resto tiene de 1 a 9 casos, el total es secreto. Los períodos agregados suman sus años, así que se decide con la misma
    supresión complementaria sobre una tabla de totales (el período 2019-2022 queda secreto si el 2023 lo es y 2019-2023 se publica).
    """
    from . import proteccion as pr

    tot = est.filter(pl.col("variable") == "actividad_grupo").group_by("periodo").agg(pl.col("n_fallecidos").sum().alias("T"))
    d = resultado_anio.filter(pl.col("base") == "D").select("periodo", pl.col("n_fallecidos").alias("D"))
    t = tot.join(d, on="periodo", how="left").sort("periodo")
    a22, a23 = ("2019", "2020", "2021", "2022"), ("2019", "2020", "2021", "2022", "2023")
    prot = pr.Proteccion(
        grupos=(pr.Grupo((), donde=pr.donde(periodo=("2019-2022", *a22)), columnas=("T",)), pr.Grupo((), donde=pr.donde(periodo=("2019-2023", *a23)), columnas=("T",)),
                pr.Grupo((), donde=pr.donde(periodo=("2019-2022", "2019-2023", "2023")), columnas=("T",))),
        filas=(pr.Fila(("T", "D"), resto_de="T"),),
    )
    oculto = pr.complementar(t, ["T"], prot, minimo).ocultas["T"]
    secretos = {p for p, h in zip(t["periodo"].to_list(), oculto) if h}
    return est.with_columns(pl.col("periodo").is_in(list(secretos)).alias("_T_oculto"))


def _escribir(ctx: Contexto, r: dict[str, Any]) -> None:
    TA = TAREA
    ctx.escribir_tabla("t23_cascada.csv", r["cascada"], tarea=TA, descripcion="Cascada secuencial de exclusiones y variantes de cohorte A a E",
                       cols_n=["n"], dependientes=["prop_del_total"])
    ctx.escribir_tabla("t23_cascada_pipeline.csv", r["cascada_pipeline"], tarea=TA,
                       descripcion="Cascada con la definición aproximada del pipeline (EX03 con sobrevivientes y mapeo de tres caracteres)",
                       cols_n=["n"], dependientes=["prop_del_total"])
    ctx.escribir_tabla("t23_cascada_variantes.csv", con_variante_siguiente(r["variantes"]), tarea=TA, descripcion="Tamaño, fallecidos y prevalencia de las variantes de cohorte A a E",
                       cols_n=["n", "n_fallecidos", "n_no_evento", "n_censurados", "n_sin_dato"],
                       dependientes=["prevalencia", "ic_bajo", "ic_alto", "prevalencia_conocida", "ic_bajo_conocida", "ic_alto_conocida", "dif_prevalencia_pp"],
                       auxiliares=[f"_hijo_{c}" for c in CONTEOS_VARIANTE])
    if "resultado_anio" not in r:
        return
    # _columna y _clave unen esta tabla con la distribución de TIPOALTA de la Tarea 4 (mismas celdas): hereda lo que esa tabla ocultó
    ctx.escribir_tabla("t23_tipoalta.csv", r["tipoalta"].with_columns(pl.lit("TIPOALTA").alias("_columna"), pl.col("etiqueta").alias("_clave")), tarea=TA,
                       descripcion="Distribución del tipo de alta con su rol, por año y período", cols_n=["n"], dependientes=["prop"], auxiliares=["_columna", "_clave"])
    ctx.escribir_tabla("t23_resultado_anio.csv", r["resultado_anio"], tarea=TA,
                       descripcion="Mortalidad por año y período con IC de Wilson y razón contra el año previo y contra el primer año",
                       cols_n=["n", "n_fallecidos", "n_no_evento", "n_censurados", "n_sin_dato"],
                       dependientes=["mortalidad", "mortalidad_conocida", "ic_bajo", "ic_alto", "ic_bajo_conocida", "ic_alto_conocida", "rr_previo",
                                     "rr_previo_bajo", "rr_previo_alto", "quiebre_anual", "rr_primero", "rr_primero_bajo", "rr_primero_alto"])
    ctx.escribir_tabla("t23_resultado_estratos.csv", con_total_secreto(r["estratos"], r["resultado_anio"], ctx.celda_minima), tarea=TA,
                       descripcion="Mortalidad por estrato en adultos de la cohorte dura con desenlace conocido (decisión 15)",
                       cols_n=["n", "n_fallecidos"],
                       dependientes=["participacion", "mortalidad", "ic_bajo", "ic_alto", "razon_vs_resto", "razon_bajo", "razon_alto", "bajo_umbral"],
                       auxiliares=["_T_oculto"])
    ctx.escribir_tabla("t23_resultado_edad_sexo.csv", r["edad_sexo"], tarea=TA,
                       descripcion="Mortalidad y estadía de sobrevivientes por grupo de edad y sexo en la cohorte dura con desenlace conocido",
                       cols_n=["n", "n_fallecidos", "n_estancia"], dependientes=["mortalidad", "ic_bajo", "ic_alto", "estancia_media"])
    ctx.escribir_tabla("t23_serie_mensual.csv", r["serie_mensual"], tarea=TA, descripcion="Serie mensual de egresos y mortalidad por mes de alta, con detección de cambios",
                       cols_n=["n_egresos", "n_fallecidos", "n_censurados"],
                       dependientes=["mortalidad", "ic_bajo", "ic_alto", "rr_mismo_mes", "rr_bajo", "rr_alto", "cambio_relativo", "quiebre"])
    ctx.escribir_tabla("t23_estancia_cero.csv", r["estancia_cero"], tarea=TA, descripcion="Estadías de 0 días por actividad y desenlace",
                       cols_n=["n", "n_cero", "n_estancia_valida"])
    ctx.escribir_tabla("t23_estancia_hist.csv", r["hist_estancia"], tarea=TA, descripcion="Histograma de la estadía de los sobrevivientes de la cohorte del modelo de mortalidad",
                       cols_n=["n"], dependientes=["prop"])
    ctx.escribir_tabla("t23_estancia_dispersion.csv", r["dispersion"], tarea=TA, descripcion="Media, varianza e índice de dispersión de la estadía de los sobrevivientes",
                       cols_n=["n"], dependientes=["media", "varianza", "indice_dispersion", "cv", "mediana", "p90", "p99", "maximo"])
    ctx.escribir_tabla("t23_cola_estancia.csv", r["cola"], tarea=TA, descripcion="Reglas de corte de estadías largas definidas en 2019 a 2022 y confirmadas en 2023",
                       cols_n=["n_def", "n_afectados_def", "n_conf", "n_afectados_conf"], dependientes=["prop_afectados_def", "prop_afectados_conf"])
    ctx.escribir_tabla("t23_resumen.csv", r["resumen"], tarea=TA, descripcion="Cifras derivadas de la cohorte, el desenlace y la estadía",
                       cols_n=["n_cero_total", "n_cero_ambulatoria", "n_cero_fallecidos", "n_cero_censurados", "n_cero_sobrevivientes", "n_ambulatoria",
                               "n_ambulatoria_cero"],
                       dependientes=["prop_cero_ambulatoria", "prop_cero_fallecidos", "prop_cero_censurados", "mortalidad_cero_conocida", "prop_ambulatoria_cero"])


def _n_paso(df: pl.DataFrame, periodo: str, paso: str, col: str = "paso") -> int | None:
    s = df.filter((pl.col("periodo") == periodo) & (pl.col(col) == paso))
    return int(s["n"][0]) if s.height else None


def registrar_metricas(ctx: Contexto, r: dict[str, Any]) -> None:
    """Cifras clave para el ledger (cada una con el alcance de años que cubre)."""
    por_anio = r["por_anio"]
    res = _anios_resultado(por_anio)
    anios = sorted(por_anio)
    pipe = r["cascada_pipeline"]
    a_todos = etiqueta_periodo(anios)
    a_res = etiqueta_periodo(res) if res else ""
    periodo_estructura = a_todos if pipe.filter(pl.col("periodo") == a_todos).height else a_res
    m = ctx.registrar_metrica

    for clave, paso in (("cascada.n_ex01", "EX01"), ("cascada.n_ex02", "EX02"), ("cascada.n_ambulatoria", "AMBULATORIA")):
        m(clave, _n_paso(pipe, periodo_estructura, paso), tarea=TAREA, tabla="t23_cascada_pipeline.csv", conteo=True, alcance=periodo_estructura)
    t_est = _n_paso(pipe, periodo_estructura, "TOTAL")
    n_amb = _n_paso(pipe, periodo_estructura, "AMBULATORIA")
    if t_est:
        m("actividad.pct_no_hospitalizacion", 100.0 * n_amb / t_est, tarea=TAREA, tabla="t23_cascada_pipeline.csv", alcance=periodo_estructura)
    exp = int(ctx.umbrales.particion["anio_expuesto"])
    if exp in por_anio:
        m("cascada.n_ex03_pipeline.cota_sup_2024", _n_paso(pipe, str(exp), "EX03_SIN_DESENLACE"), tarea=TAREA, tabla="t23_cascada_pipeline.csv", conteo=True,
          alcance=str(exp), nota="MDC 14 sin comorbilidades según el mapeo de tres caracteres, sin mirar el desenlace: cota superior de EX03 del pipeline en el año expuesto")
    if not res:
        return
    nota_ap = "definición aproximada del pipeline: mapeo de tres caracteres; el código original no está en el repositorio revisado"
    for clave, paso in (("cascada.n_ex03_pipeline", "EX03_PIPELINE"), ("cascada.n_cohorte_dura_pipeline", "COHORTE_DURA_PIPELINE"),
                        ("cascada.n_estancia_cero", "ESTANCIA_CERO"), ("cascada.n_fallecidos", "FALLECIDOS"),
                        ("cascada.n_censurados", "CENSURADOS"), ("cascada.n_cohorte_estancia", "COHORTE_ESTANCIA_PIPELINE"),
                        ("cascada.n_cohorte_estancia.sin_censurados", "COHORTE_ESTANCIA_PIPELINE_SIN_CENSURADOS")):
        m(clave, _n_paso(pipe, a_res, paso), tarea=TAREA, tabla="t23_cascada_pipeline.csv", conteo=True, alcance=a_res,
          nota=nota_ap if paso in ("EX03_PIPELINE", "COHORTE_DURA_PIPELINE", "COHORTE_ESTANCIA_PIPELINE", "COHORTE_ESTANCIA_PIPELINE_SIN_CENSURADOS") else "")
    t_res = _n_paso(pipe, a_res, "TOTAL")
    if t_res:
        m("estancia.pct_cero_dias", 100.0 * _n_paso(pipe, a_res, "ESTANCIA_CERO") / t_res, tarea=TAREA, tabla="t23_cascada_pipeline.csv", alcance=a_res)
        m("cohorte.pct_estancia_total", 100.0 * _n_paso(pipe, a_res, "COHORTE_ESTANCIA_PIPELINE") / t_res, tarea=TAREA, tabla="t23_cascada_pipeline.csv",
          alcance=a_res, nota=nota_ap)
        m("cohorte.pct_estancia_total.sin_censurados", 100.0 * _n_paso(pipe, a_res, "COHORTE_ESTANCIA_PIPELINE_SIN_CENSURADOS") / t_res, tarea=TAREA,
          tabla="t23_cascada_pipeline.csv", alcance=a_res, nota="lectura alternativa: además se excluyen los censurados")
    # mortalidad
    ra = r["resultado_anio"].filter(pl.col("base") == "todos")
    for row in ra.iter_rows(named=True):
        clave = f"resultado.mortalidad_pct.{row['periodo']}" if row["periodo"].isdigit() else (
            "resultado.mortalidad_pct.total" if row["periodo"] == a_res else f"resultado.mortalidad_pct.{row['periodo'].replace('-', '_')}")
        m(clave, 100.0 * row["mortalidad"], tarea=TAREA, tabla="t23_resultado_anio.csv", ic_bajo=100.0 * row["ic_bajo"], ic_alto=100.0 * row["ic_alto"],
          alcance=row["periodo"] if not row["periodo"].isdigit() else "")
    por = {row["periodo"]: row["mortalidad"] for row in ra.iter_rows(named=True)}
    if por.get("2021") and por.get("2023"):
        m("resultado.caida_mortalidad_2021_2023_pct", 100.0 * (por["2021"] - por["2023"]) / por["2021"], tarea=TAREA, tabla="t23_resultado_anio.csv", alcance="2021-2023")
    # tipo de alta
    from .proteccion import suma_publicada

    tot = r["tipoalta"].filter(pl.col("periodo") == a_res)["n"].sum()           # el total del período es público (filas leídas)
    ta = ctx.leer_tabla("t23_tipoalta.csv", columnas_texto=["periodo", "etiqueta", "codigo", "rol"]).filter(pl.col("periodo") == a_res)
    if tot:
        for row in ta.iter_rows(named=True):
            m(f"tipoalta.pct.{row['codigo']}", None if row["n"] is None else 100.0 * row["n"] / tot, tarea=TAREA, tabla="t23_tipoalta.csv", alcance=a_res)

        def suma(f: pl.Expr) -> float | None:
            s = suma_publicada(ta.filter(f)["n"])
            return None if s is None else 100.0 * s / tot
        m("tipoalta.pct.sobrevivientes", suma(pl.col("rol") == "NO_EVENTO"), tarea=TAREA, tabla="t23_tipoalta.csv", alcance=a_res)
        m("tipoalta.pct.censurados", suma(pl.col("rol") == "CENSURADO"), tarea=TAREA, tabla="t23_tipoalta.csv", alcance=a_res)
        m("tipoalta.pct.derivaciones_hospital", suma(pl.col("codigo").is_in(["DERIV_HOSP_SERVICIO", "DERIV_HOSP_RED"])), tarea=TAREA, tabla="t23_tipoalta.csv", alcance=a_res)
    # variantes
    for row in ctx.leer_tabla("t23_cascada_variantes.csv", columnas_texto=["periodo", "variante", "descripcion"]).iter_rows(named=True):
        per = row["periodo"].replace("-", "_")
        m(f"cohorte.n.{row['variante']}.{per}", row["n"], tarea=TAREA, tabla="t23_cascada_variantes.csv", conteo=True, alcance=row["periodo"])
        if row["prevalencia"] is not None:
            m(f"cohorte.prevalencia_pct.{row['variante']}.{per}", 100.0 * row["prevalencia"], tarea=TAREA, tabla="t23_cascada_variantes.csv",
              ic_bajo=100.0 * row["ic_bajo"], ic_alto=100.0 * row["ic_alto"], alcance=row["periodo"], nota="fallecidos sobre todos los egresos de la variante")
            m(f"cohorte.prevalencia_conocida_pct.{row['variante']}.{per}", 100.0 * row["prevalencia_conocida"], tarea=TAREA, tabla="t23_cascada_variantes.csv",
              ic_bajo=100.0 * row["ic_bajo_conocida"], ic_alto=100.0 * row["ic_alto_conocida"], alcance=row["periodo"],
              nota="fallecidos sobre fallecidos más sobrevivientes (sin censurados)")
    # estadía de la cohorte del modelo de estadía
    dis = r["dispersion"].filter((pl.col("periodo") == a_res) & (pl.col("base") == "E"))
    if dis.height:
        m("estancia.mediana_dias", dis["mediana"][0], tarea=TAREA, tabla="t23_estancia_dispersion.csv", alcance=a_res, nota="sobrevivientes de la variante E")
        m("estancia.media_dias", dis["media"][0], tarea=TAREA, tabla="t23_estancia_dispersion.csv", alcance=a_res, nota="sobrevivientes de la variante E")
        m("estancia.indice_dispersion", dis["indice_dispersion"][0], tarea=TAREA, tabla="t23_estancia_dispersion.csv", alcance=a_res, nota="varianza sobre media, variante E")
    m("cortafuegos.intento_2024_rechazado", 1.0 if r["intento_2024"] == "rechazado" else 0.0, tarea=TAREA, tabla="t23_resumen.csv",
      nota="1 si el intento de leer el desenlace del año expuesto fue rechazado por el cortafuegos")


def ejecutar(ctx: Contexto, cat: catalogos.Catalogos | None = None) -> dict[str, Any]:
    """Calcula, escribe tablas y figuras y registra las métricas de la cohorte, el resultado y la estadía."""
    r = calcular(ctx, cat)
    _escribir(ctx, r)
    registrar_metricas(ctx, r)
    if "resultado_anio" in r:
        figura_cascada(ctx)
        figura_mortalidad_anual(ctx)
        figura_estancia(ctx)
        figura_serie_mensual(ctx)
        figura_edad_sexo(ctx)
    return r


# ---------------------------------------------------------------------------------- figuras
ETIQUETA_CORTA = {
    "TOTAL": "Total de egresos", "EX01": "EX01 GRD no agrupable", "EX02": "EX02 Recién nacidos (MDC 15)",
    "EX03": "EX03 Obstétricos sin comorbilidad", "A": "A  Cohorte dura", "EX04": "EX04 Menores o edad desconocida", "B": "B  Adultos",
    "EX05": "EX05 Actividad no hospitalaria", "C": "C  Adultos hospitalizados", "EX06": "EX06 Censurados", "D": "D  Modelo de mortalidad",
    "EX07": "EX07 Fallecidos", "EX08": "EX08 Estadía de 0 días o desconocida", "E": "E  Modelo de estadía",
}


def _fmt_n(v: int | float | None) -> str:
    return "n<10" if v is None else f"{int(v):,}".replace(",", ".")


def _pct_txt(v: float | None) -> str:
    return "" if v is None else f"{f'{100.0 * v:.1f}'.replace('.', ',')} %"


def _arr(valores, escala: float = 1.0) -> np.ndarray:
    return np.array([np.nan if v is None else escala * float(v) for v in valores], dtype=float)


def _periodo_resultado(ctx: Contexto) -> str:
    return etiqueta_periodo(sorted(int(x) for x in ctx.umbrales.particion["anios_resultado"]))


def figura_cascada(ctx: Contexto) -> None:
    periodo = _periodo_resultado(ctx)
    t = ctx.leer_tabla("t23_cascada.csv", columnas_texto=["periodo", "paso"]).filter(pl.col("periodo") == periodo).sort("orden")
    if t.height == 0:
        return
    filas = t.to_dicts()
    total = filas[0]["n"] or 1
    fig, ax = graficos.nueva_figura(figsize=(10.5, 6.4))
    corrida = total
    colores = {"total": "#6B6B6B", "cohorte": graficos.OKABE_ITO[0], "exclusion": graficos.OKABE_ITO[3]}
    for i, f in enumerate(filas):
        y = len(filas) - 1 - i
        n = f["n"]
        n0 = 0 if n is None else n
        if f["tipo"] == "exclusion":
            ax.barh(y, n0, left=corrida - n0, color=colores["exclusion"], height=0.62)
            corrida -= n0
            x_txt = corrida + n0
            txt = "n<10" if n is None else f"−{_fmt_n(n)} ({_pct_txt(f['prop_del_total'])})"
        else:
            ax.barh(y, n0, color=colores[f["tipo"]], height=0.62)
            corrida = n0 if f["tipo"] == "cohorte" else corrida
            x_txt = n0
            txt = f"{_fmt_n(n)} ({_pct_txt(f['prop_del_total'])})"
        ax.text(x_txt + total * 0.01, y, txt, va="center", ha="left", fontsize=7.5)
    ax.set_yticks(range(len(filas)))
    ax.set_yticklabels([ETIQUETA_CORTA[f["paso"]] for f in reversed(filas)], fontsize=8)
    ax.set_xlim(0, total * 1.32)
    ax.set_xlabel("Egresos")
    graficos.formato_miles(ax, "x")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=colores["total"], label="Total"), Patch(color=colores["cohorte"], label="Cohorte resultante"),
                       Patch(color=colores["exclusion"], label="Egresos excluidos en el paso")], loc="lower right", fontsize=7.5)
    ctx.figuras.guardar_figura(
        fig, id_figura="F07_01", tarea=7,
        titulo=f"Cascada de exclusiones y variantes de cohorte ({periodo})",
        descripcion_alt="Barras horizontales en cascada: del total de egresos se restan los GRD no agrupables, los recién nacidos y los obstétricos sin comorbilidad "
                        "para llegar a la cohorte dura A; luego los menores, la actividad no hospitalaria, los censurados, los fallecidos y las estadías de 0 días "
                        "para llegar a las variantes B, C, D y E.",
        csv="t23_cascada.csv", anios=periodo, decision="5; 15",
        extra_pie="Los pasos que usan el desenlace solo existen para 2019 a 2023.",
    )


def figura_mortalidad_anual(ctx: Contexto) -> None:
    t = ctx.leer_tabla("t23_resultado_anio.csv", columnas_texto=["periodo", "base"]).filter(pl.col("periodo").str.contains(r"^\d{4}$"))
    if t.height == 0:
        return
    fig, ax = graficos.nueva_figura(figsize=(8.0, 4.8))
    anios = sorted({int(p) for p in t["periodo"]})
    for k, (base, etiqueta) in enumerate((("todos", "Todos los egresos"), ("D", "Cohorte del modelo de mortalidad (D)"))):
        s = t.filter(pl.col("base") == base).sort("periodo")
        if s.height == 0:
            continue
        x = np.array([int(p) for p in s["periodo"]], dtype=float) + (k - 0.5) * 0.08
        y, lo, hi = _arr(s["mortalidad"], 100), _arr(s["ic_bajo"], 100), _arr(s["ic_alto"], 100)
        ax.errorbar(x, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], marker="o", capsize=3, lw=1.4, color=graficos.OKABE_ITO[k * 2], label=etiqueta)
        for xi, yi, hi_i in zip(x, y, hi):
            if not np.isnan(yi):
                ax.annotate(f"{yi:.1f}".replace(".", ","), (xi, hi_i), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7,
                            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.85})
    ax.set_xticks(anios)
    ax.set_xlabel("Año del archivo")
    ax.set_ylabel("Mortalidad cruda (%)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    ctx.figuras.guardar_figura(
        fig, id_figura="F07_02", tarea=7,
        titulo="Mortalidad cruda por año con intervalo de confianza",
        descripcion_alt="Dos series de puntos con barras de error por año: mortalidad sobre todos los egresos y sobre la cohorte del modelo de mortalidad.",
        csv="t23_resultado_anio.csv", anios=etiqueta_periodo(anios), decision="17",
        extra_pie="Intervalos de Wilson al 95 %; ignoran el agrupamiento por paciente y hospital.",
    )


def figura_estancia(ctx: Contexto) -> None:
    t = ctx.leer_tabla("t23_estancia_hist.csv", columnas_texto=["periodo", "tramo"])
    series = [(p, c) for p, c in (("2019-2022", graficos.OKABE_ITO[0]), ("2023", graficos.OKABE_ITO[1])) if t.filter(pl.col("periodo") == p).height]
    if not series:
        return
    fig, ax = graficos.nueva_figura(figsize=(9.0, 4.8))
    tramos = [x[0] for x in TRAMOS_ESTANCIA]
    xs = np.arange(len(tramos))
    ancho = 0.38
    for k, (periodo, color) in enumerate(series):
        s = t.filter(pl.col("periodo") == periodo).sort("orden")
        y = _arr(s["prop"], 100)
        y = np.where(y > 0, y, np.nan)
        ax.bar(xs + (k - 0.5) * ancho, y, width=ancho, color=color, label=f"{periodo}")
    ax.set_yscale("log")
    graficos.formato_decimal(ax, "y")
    ax.set_xticks(xs)
    ax.set_xticklabels(tramos, rotation=45, ha="right")
    ax.set_xlabel("Estadía (días)")
    ax.set_ylabel("% de los sobrevivientes (escala logarítmica)")
    ax.legend(title="Período", loc="upper right")
    ctx.figuras.guardar_figura(
        fig, id_figura="F07_03", tarea=7,
        titulo="Histograma logarítmico de la estadía de los sobrevivientes",
        descripcion_alt="Barras por tramo de días de estadía, con escala logarítmica en el eje vertical, para los sobrevivientes de la cohorte del modelo de mortalidad "
                        "en 2019 a 2022 y en 2023.",
        csv="t23_estancia_hist.csv", anios=_periodo_resultado(ctx), decision="7; 9",
        extra_pie="Los tramos con menos de diez casos no se dibujan.",
    )


def figura_serie_mensual(ctx: Contexto) -> None:
    t = ctx.leer_tabla("t23_serie_mensual.csv", columnas_texto=["mes_id", "quiebre"]).sort(["anio", "mes"])
    if t.height == 0:
        return
    x = np.arange(t.height)
    fig, (a1, a2) = graficos.nueva_figura_multiple(2, 1, figsize=(10.5, 6.2), sharex=True)
    a1.plot(x, _arr(t["n_egresos"]), color=graficos.OKABE_ITO[0], lw=1.3)
    a1.set_ylabel("Egresos por mes")
    graficos.formato_miles(a1, "y")
    a1.set_ylim(bottom=0)
    m, lo, hi = _arr(t["mortalidad"], 100), _arr(t["ic_bajo"], 100), _arr(t["ic_alto"], 100)
    a2.fill_between(x, lo, hi, color=graficos.OKABE_ITO[4], alpha=0.35, lw=0)
    a2.plot(x, m, color=graficos.OKABE_ITO[2], lw=1.3, label="Mortalidad mensual")
    q = np.array(t["quiebre"].fill_null("").to_list())
    for etiqueta, marca in (("alza", "^"), ("baja", "v")):
        sel = q == etiqueta
        if sel.any():
            a2.scatter(x[sel], m[sel], marker=marca, color=graficos.OKABE_ITO[3], s=36, zorder=3, label=f"Cambio relevante ({etiqueta})")
    a2.set_ylabel("Mortalidad (%)")
    a2.set_ylim(bottom=0)
    inicios = [i for i, mes in enumerate(t["mes"].to_list()) if mes == 1]
    a2.set_xticks(inicios)
    a2.set_xticklabels([str(t["anio"][i]) for i in inicios])
    a2.set_xlabel("Mes de alta (marca en enero de cada año)")
    a2.legend(loc="upper right", fontsize=7.5)
    ctx.figuras.guardar_figura(
        fig, id_figura="F07_04", tarea=7,
        titulo="Egresos y mortalidad por mes con cambios relevantes",
        descripcion_alt="Dos paneles con el mismo eje de meses: arriba los egresos por mes y abajo la mortalidad mensual con su intervalo de confianza; "
                        "los triángulos marcan los meses cuya mortalidad cambia de forma relevante frente al mismo mes del año anterior.",
        csv="t23_serie_mensual.csv", anios=_periodo_resultado(ctx), decision="17",
        extra_pie="Cambio relevante: IC de la razón contra el mismo mes del año anterior sin incluir 1 y cambio de al menos el umbral."
                  + (" En el extracto el volumen anual es fijo: solo la mortalidad es comparable entre años." if ctx.etiqueta == "extracto" else ""),
    )


def figura_edad_sexo(ctx: Contexto) -> None:
    t = ctx.leer_tabla("t23_resultado_edad_sexo.csv", columnas_texto=["periodo", "grupo_edad", "sexo"])
    periodos = [p for p in ("2019-2022", "2023") if t.filter(pl.col("periodo") == p).height]
    if not periodos:
        return
    fig, (a1, a2) = graficos.nueva_figura_multiple(1, 2, figsize=(11.5, 4.9))
    etiquetas = [g[0] for g in GRUPOS_EDAD]
    xs = np.arange(len(etiquetas))
    colores = {"MUJER": graficos.OKABE_ITO[5], "HOMBRE": graficos.OKABE_ITO[0]}
    hay_positivos = False
    for sexo, color in colores.items():
        for periodo, estilo in zip(periodos, ("-", "--")):
            s = t.filter((pl.col("periodo") == periodo) & (pl.col("sexo") == sexo) & (pl.col("grupo_edad") != EDAD_DESCONOCIDA)).sort("orden")
            if s.height == 0:
                continue
            pos = s["orden"].to_numpy()
            m, lo, hi = _arr(s["mortalidad"], 100), _arr(s["ic_bajo"], 100), _arr(s["ic_alto"], 100)
            m = np.where(m > 0, m, np.nan)
            hay_positivos = hay_positivos or bool(np.isfinite(m).any())
            a1.plot(pos, m, estilo, marker="o", ms=3.5, color=color, label=f"{sexo.capitalize()} {periodo}")
            if periodo == periodos[0]:
                a1.fill_between(pos, np.where(lo > 0, lo, np.nan), hi, color=color, alpha=0.15, lw=0)
            a2.plot(pos, _arr(s["estancia_media"]), estilo, marker="o", ms=3.5, color=color, label=f"{sexo.capitalize()} {periodo}")
    if hay_positivos:                      # con todas las celdas suprimidas (extractos chicos) no hay nada que poner en escala logarítmica
        a1.set_yscale("log")
    graficos.formato_decimal(a1, "y")
    a1.set_ylabel("Mortalidad (%, escala logarítmica)")
    a2.set_ylabel("Estadía media de los sobrevivientes (días)")
    a2.set_ylim(bottom=0)
    for ax in (a1, a2):
        ax.set_xticks(xs)
        ax.set_xticklabels(etiquetas, rotation=45, ha="right")
        ax.set_xlabel("Grupo de edad (años)")
    a1.legend(fontsize=7)
    ctx.figuras.guardar_figura(
        fig, id_figura="F07_05", tarea=7,
        titulo="Mortalidad y estadía por grupo de edad y sexo",
        descripcion_alt="Dos paneles por grupo de edad, separados por sexo y período: a la izquierda la mortalidad con escala logarítmica y a la derecha la estadía "
                        "media de los sobrevivientes.",
        csv="t23_resultado_edad_sexo.csv", anios=_periodo_resultado(ctx), decision="5",
        extra_pie="Cohorte dura con desenlace conocido; la banda es el IC de 2019 a 2022.",
    )


# ------------------------------------------------------------------------------------ informe
def agregar_secciones(inf: Informe, ctx: Contexto, n0: int = 1) -> None:
    """Agrega al informe de exploración (CRISP-DM 2.3) las secciones de cohorte, resultado y estadía.

    ``n0`` es el número de la primera sección. Todas las cifras salen de los CSV de ``tablas``.
    """
    from .descripcion import Cifra_umbral

    u = ctx.umbrales
    pr = _periodo_resultado(ctx)
    rank = etiqueta_periodo(sorted(int(x) for x in u.particion["anios_ranking"]))
    conf = str(int(u.particion["anio_confirmacion"]))
    anios_res = sorted(int(x) for x in u.particion["anios_resultado"])

    def cas(col: str, paso: str, periodo: str = pr):
        return cifra(ctx, "t23_cascada.csv", col, columnas_texto=["periodo", "paso"], periodo=periodo, paso=paso)

    def var(col: str, v: str, periodo: str = pr):
        return cifra(ctx, "t23_cascada_variantes.csv", col, columnas_texto=["periodo", "variante"], periodo=periodo, variante=v)

    def res(col: str, periodo: str, base: str = "todos"):
        return cifra(ctx, "t23_resultado_anio.csv", col, columnas_texto=["periodo", "base", "quiebre_anual"], periodo=periodo, base=base)

    def rs(col: str):
        return cifra(ctx, "t23_resumen.csv", col, columnas_texto=["intento_2024"])

    def disp(col: str, periodo: str, base: str = "E"):
        return cifra(ctx, "t23_estancia_dispersion.csv", col, columnas_texto=["periodo", "base"], periodo=periodo, base=base)

    # --- cohorte y cascada ---------------------------------------------------------------
    inf.seccion(f"{n0}. Cohorte y cascada de exclusiones")
    inf.parrafo(
        "La cascada resta, en orden, los egresos que quedan fuera de cada variante de cohorte. Cada paso excluye solo lo que queda del anterior, "
        "así que la cohorte final y las exclusiones suman el total en cada año. Los pasos EX01 a EX03 no miran el desenlace: la cohorte dura A "
        "no depende de quién falleció. Entre 2019 y 2023 se midieron {total} egresos.",
        total=cas("n", "TOTAL"),
    )
    inf.figura("F07_01")
    tc = ctx.leer_tabla("t23_cascada.csv", columnas_texto=["periodo", "paso", "tipo", "descripcion", "usa_desenlace", "alcance", "cuadra"]).filter(pl.col("periodo") == pr).sort("orden")
    inf.tabla(
        "t23_cascada.csv", df=tc, columnas=["paso", "descripcion", "n", "prop_del_total", "usa_desenlace"],
        encabezados={"paso": "Paso", "descripcion": "Descripción", "n": "Egresos", "prop_del_total": "% del total", "usa_desenlace": "Usa el desenlace"},
        formatos={"prop_del_total": "prop2"}, leyenda="Cascada de 2019 a 2023",
    )
    inf.parrafo(
        "Los GRD no agrupables (EX01) suman {n1} egresos ({p1|prop2}); los recién nacidos (EX02), {n2} ({p2|prop1}); y los obstétricos sin comorbilidades (EX03), "
        "{n3} ({p3|prop1}). La cohorte dura A queda con {na} egresos ({pa|prop1}). Los menores de {edad_adulto} años o con edad desconocida suman {n4}, "
        "la actividad distinta de hospitalización {n5} y los censurados por tipo de alta {n6}.",
        n1=cas("n", "EX01"), p1=cas("prop_del_total", "EX01"), n2=cas("n", "EX02"), p2=cas("prop_del_total", "EX02"),
        n3=cas("n", "EX03"), p3=cas("prop_del_total", "EX03"), na=cas("n", "A"), pa=cas("prop_del_total", "A"),
        edad_adulto=Cifra_umbral(ctx, "edad_adulto"), n4=cas("n", "EX04"), n5=cas("n", "EX05"), n6=cas("n", "EX06"),
    )
    inf.parrafo(
        "Las exclusiones que dependen del desenlace (EX06 a EX08) solo existen para 2019 a 2023, y el año expuesto 2024 aporta únicamente los pasos que no lo usan. "
        "El intento de leer el desenlace de 2024 fue {intento|texto} por el cortafuegos y quedó en el registro de accesos.",
        intento=rs("intento_2024"),
    )

    # --- prevalencia por variante (decisión 5) -----------------------------------------------
    inf.seccion(f"{n0 + 1}. Prevalencia de la mortalidad por variante de cohorte")
    inf.parrafo(
        "La decisión 5 limita la población a adultos y la decisión 15 a la actividad de hospitalización. La prevalencia se informa de dos formas: "
        "sobre todos los egresos de la variante y sobre los egresos con desenlace conocido (fallecidos más sobrevivientes, sin censurados). "
        "Al pasar de A a B la prevalencia con desenlace conocido cambia de {pa|prop2} a {pb|prop2}, y llega a {pc|prop2} en la variante C. "
        "En 2023 la variante C tiene {pc_conf|prop2} sobre todos sus egresos y {pc_conf_c|prop2} con desenlace conocido.",
        pa=var("prevalencia_conocida", "A"), pb=var("prevalencia_conocida", "B"), pc=var("prevalencia_conocida", "C"),
        pc_conf=var("prevalencia", "C", conf), pc_conf_c=var("prevalencia_conocida", "C", conf),
    )
    tv = ctx.leer_tabla("t23_cascada_variantes.csv", columnas_texto=["periodo", "variante", "descripcion"]).filter(pl.col("periodo").is_in([pr, conf])).sort(["periodo", "variante"])
    inf.tabla(
        "t23_cascada_variantes.csv", df=tv,
        columnas=["periodo", "variante", "n", "n_fallecidos", "n_censurados", "prevalencia", "prevalencia_conocida", "ic_bajo_conocida", "ic_alto_conocida", "dif_prevalencia_pp"],
        encabezados={"periodo": "Período", "variante": "Variante", "n": "Egresos", "n_fallecidos": "Fallecidos", "n_censurados": "Censurados",
                     "prevalencia": "Prevalencia (todos)", "prevalencia_conocida": "Prevalencia (conocido)", "ic_bajo_conocida": "IC bajo",
                     "ic_alto_conocida": "IC alto", "dif_prevalencia_pp": "Cambio (pp)"},
        formatos={"prevalencia": "prop2", "prevalencia_conocida": "prop2", "ic_bajo_conocida": "prop2", "ic_alto_conocida": "prop2", "dif_prevalencia_pp": "dec2"},
        leyenda="Variantes de cohorte; intervalos de Wilson indicativos",
    )
    inf.parrafo(
        "La definición del pipeline se reproduce de forma aproximada (mapeo de comorbilidades de tres caracteres y exclusión de obstétricos solo si sobreviven), "
        "porque el código original no está en el repositorio revisado. Su cascada se compara con las cifras documentadas en el cierre."
    )
    tp = ctx.leer_tabla("t23_cascada_pipeline.csv", columnas_texto=["periodo", "paso", "descripcion", "alcance"]).filter(pl.col("periodo") == pr)
    inf.tabla(
        "t23_cascada_pipeline.csv", df=tp, columnas=["paso", "descripcion", "n", "prop_del_total"],
        encabezados={"paso": "Paso", "descripcion": "Descripción", "n": "Egresos", "prop_del_total": "% del total"},
        formatos={"prop_del_total": "prop2"}, leyenda="Cascada con la definición aproximada del pipeline",
    )

    # --- resultado: tipo de alta y mortalidad ------------------------------------------------------
    inf.seccion(f"{n0 + 2}. Desenlace: tipo de alta y mortalidad")
    inf.parrafo(
        "El rol de cada tipo de alta viene de la decisión de la Fase 1 (archivo de roles): los fallecidos son el evento, el alta a domicilio, la voluntaria y la fuga son "
        "no evento, y los traslados y la hospitalización domiciliaria son censurados. Las etiquetas cortadas se asignan por prefijo."
    )
    tt = ctx.leer_tabla("t23_tipoalta.csv", columnas_texto=["periodo", "etiqueta", "codigo", "rol"]).filter(pl.col("periodo") == pr)
    inf.tabla(
        "t23_tipoalta.csv", df=tt, columnas=["etiqueta", "rol", "n", "prop"],
        encabezados={"etiqueta": "Tipo de alta", "rol": "Rol", "n": "Egresos", "prop": "% del total"}, formatos={"prop": "prop2"},
        leyenda="Tipo de alta de 2019 a 2023",
    )
    items = ", ".join(f"{{m{a}|prop2}} en {a}" for a in anios_res)
    cif = {f"m{a}": res("mortalidad", str(a)) for a in anios_res}
    post = [a for a in sorted(int(x) for x in u.particion["anios_ranking"]) if a != anios_res[0]]
    post_et = etiqueta_periodo(post)
    inf.parrafo(
        f"La mortalidad cruda sobre todos los egresos fue {items}. La mortalidad agrupada de {post_et} es {{rr_total|dec2}} veces la de {anios_res[0]} "
        "(IC {rr_bajo|dec2} a {rr_alto|dec2}), evidencia para la decisión 17. Un año se marca como alza o baja cuando el intervalo de la razón contra el año previo "
        "excluye la igualdad y el cambio relativo alcanza {umbral|prop1}.",
        rr_total=res("rr_primero", post_et), rr_bajo=res("rr_primero_bajo", post_et), rr_alto=res("rr_primero_alto", post_et),
        umbral=Cifra_umbral(ctx, "cambio_relativo_relevante"), **cif,
    )
    inf.figura("F07_02")
    ta = ctx.leer_tabla("t23_resultado_anio.csv", columnas_texto=["periodo", "base", "quiebre_anual"]).filter(pl.col("periodo").str.contains(r"^\d{4}$"))
    inf.tabla(
        "t23_resultado_anio.csv", df=ta,
        columnas=["periodo", "base", "n", "n_fallecidos", "mortalidad", "ic_bajo", "ic_alto", "rr_previo", "quiebre_anual"],
        encabezados={"periodo": "Año", "base": "Base", "n": "Egresos", "n_fallecidos": "Fallecidos", "mortalidad": "Mortalidad", "ic_bajo": "IC bajo",
                     "ic_alto": "IC alto", "rr_previo": "Razón vs año previo", "quiebre_anual": "Cambio anual"},
        formatos={"mortalidad": "prop2", "ic_bajo": "prop2", "ic_alto": "prop2", "rr_previo": "dec2"},
        leyenda="Mortalidad cruda por año; base D es la cohorte del modelo de mortalidad",
    )
    inf.parrafo(
        "La serie mensual compara cada mes con el mismo mes del año anterior, lo que controla la estacionalidad. De los {n_eval} meses con referencia, {n_alza} muestran un "
        "alza relevante y {n_baja} una baja relevante. Los intervalos no se ajustan por comparaciones múltiples y son indicativos.",
        n_eval=rs("n_meses_evaluados"), n_alza=rs("n_meses_alza"), n_baja=rs("n_meses_baja"),
    )
    inf.figura("F07_04")
    inf.parrafo("La mortalidad y la estadía también cambian con la edad y el sexo, y la cohorte dura incluye a los menores hasta que la decisión 5 los saque.")
    inf.figura("F07_05")

    # --- estadía ------------------------------------------------------------------------------------
    inf.seccion(f"{n0 + 3}. Estadía de los sobrevivientes")
    inf.parrafo(
        "En 2019 a 2022 la estadía de los sobrevivientes de la variante E tiene mediana {mediana|entero} días, media {media|dec1} y varianza {varianza|dec1}. El índice de "
        "dispersión (varianza sobre media) es {indice|dec1}; un valor mayor que uno indica sobredispersión frente a la distribución de Poisson, evidencia para elegir "
        "entre Poisson y Tweedie en la decisión 9. En 2023 el índice es {indice_conf|dec1}.",
        mediana=disp("mediana", rank), media=disp("media", rank), varianza=disp("varianza", rank), indice=disp("indice_dispersion", rank),
        indice_conf=disp("indice_dispersion", conf),
    )
    inf.figura("F07_03")
    td = ctx.leer_tabla("t23_estancia_dispersion.csv", columnas_texto=["periodo", "base"]).filter(pl.col("base") == "E")
    inf.tabla(
        "t23_estancia_dispersion.csv", df=td,
        columnas=["periodo", "n", "media", "varianza", "indice_dispersion", "mediana", "p90", "p99", "maximo"],
        encabezados={"periodo": "Período", "n": "Sobrevivientes", "media": "Media", "varianza": "Varianza", "indice_dispersion": "Índice de dispersión",
                     "mediana": "Mediana", "p90": "p90", "p99": "p99", "maximo": "Máximo"},
        formatos={"media": "dec1", "varianza": "dec1", "indice_dispersion": "dec1", "mediana": "entero", "p90": "entero", "p99": "entero", "maximo": "entero"},
        leyenda="Estadía en días de los sobrevivientes de la variante E",
    )
    tcola = ctx.leer_tabla("t23_cola_estancia.csv", columnas_texto=["base", "regla", "cumple_def", "cumple_conf", "propuesta"]).filter(pl.col("base") == "E")
    prop_f = tcola.filter(pl.col("propuesta") == "si")
    if prop_f.height == 1:
        regla = prop_f["regla"][0]
        c_cola = lambda col: cifra(ctx, "t23_cola_estancia.csv", col, columnas_texto=["base", "regla"], base="E", regla=regla)  # noqa: E731
        inf.parrafo(
            "La decisión 7 se evalúa con 2019 a 2022 y se confirma con 2023. El corte más bajo que afecta a no más de {tope|prop1} de los sobrevivientes es la regla "
            "{regla|texto}, con {corte|entero} días: afectaría a {pdef|prop2} de los sobrevivientes de 2019 a 2022 y a {pconf|prop2} de los de 2023. "
            "El máximo observado en 2019 a 2022 fue de {maximo|entero} días.",
            tope=Cifra_umbral(ctx, "cola_estadia_perdida_max"), regla=c_cola("regla"), corte=c_cola("corte_dias"),
            pdef=c_cola("prop_afectados_def"), pconf=c_cola("prop_afectados_conf"), maximo=c_cola("maximo_def"),
        )
    inf.tabla(
        "t23_cola_estancia.csv", df=tcola,
        columnas=["regla", "corte_dias", "n_afectados_def", "prop_afectados_def", "cumple_def", "n_afectados_conf", "prop_afectados_conf", "cumple_conf", "propuesta"],
        encabezados={"regla": "Regla", "corte_dias": "Corte (días)", "n_afectados_def": "Afectados 2019-2022", "prop_afectados_def": "% 2019-2022",
                     "cumple_def": "Cumple", "n_afectados_conf": "Afectados 2023", "prop_afectados_conf": "% 2023", "cumple_conf": "Cumple 2023", "propuesta": "Propuesta"},
        formatos={"prop_afectados_def": "prop2", "prop_afectados_conf": "prop2", "corte_dias": "entero"},
        leyenda="Reglas de corte de estadías largas, base E",
    )

    # --- estadía de 0 días y actividad (decisión 15) ----------------------------------------------------
    inf.seccion(f"{n0 + 4}. Estadía cero y actividad ambulatoria")
    inf.parrafo(
        "De los {n_cero} egresos con estadía cero en 2019 a 2023, {p_amb|prop1} son de actividad ambulatoria, {p_fall|prop2} fallecieron el mismo día y {p_cens|prop2} son "
        "censurados. Entre los {n_amb} egresos de actividad ambulatoria, {p_amb_cero|prop1} tienen estadía cero. Entre los egresos con estadía cero y desenlace "
        "conocido, la mortalidad es {m_cero|prop2}.",
        n_cero=rs("n_cero_total"), p_amb=rs("prop_cero_ambulatoria"), p_fall=rs("prop_cero_fallecidos"), p_cens=rs("prop_cero_censurados"),
        n_amb=rs("n_ambulatoria"), p_amb_cero=rs("prop_ambulatoria_cero"), m_cero=rs("mortalidad_cero_conocida"),
    )
    te = ctx.leer_tabla("t23_estancia_cero.csv", columnas_texto=["periodo", "actividad", "rol"]).filter(pl.col("periodo") == pr)
    inf.tabla(
        "t23_estancia_cero.csv", df=te, columnas=["actividad", "rol", "n", "n_cero"],
        encabezados={"actividad": "Actividad", "rol": "Desenlace", "n": "Egresos", "n_cero": "Con estadía de 0 días"},
        leyenda="Estadías cero por actividad y desenlace, 2019 a 2023",
    )
    inf.parrafo(
        "Para la decisión 15 se compara la mortalidad de cada estrato con la del resto en adultos de la cohorte dura con desenlace conocido. Si la razón es menor que "
        "{umbral|dec2}, el estrato diluye el cociente observado sobre esperado y se propone sacarlo del modelo de mortalidad. La decisión final es humana.",
        umbral=Cifra_umbral(ctx, "estrato_mortalidad_relativa_max"),
    )
    tes = ctx.leer_tabla("t23_resultado_estratos.csv", columnas_texto=["periodo", "variable", "categoria", "bajo_umbral"]).filter(
        pl.col("variable").is_in(list(VARIABLES_DECISION_15)) & pl.col("periodo").is_in([rank, conf]) & (pl.col("bajo_umbral") != "")
    )
    inf.tabla(
        "t23_resultado_estratos.csv", df=tes,
        columnas=["periodo", "variable", "categoria", "n", "n_fallecidos", "mortalidad", "razon_vs_resto", "razon_bajo", "razon_alto", "bajo_umbral"],
        encabezados={"periodo": "Período", "variable": "Variable", "categoria": "Estrato", "n": "Egresos", "n_fallecidos": "Fallecidos", "mortalidad": "Mortalidad",
                     "razon_vs_resto": "Razón vs resto", "razon_bajo": "IC bajo", "razon_alto": "IC alto", "bajo_umbral": "Bajo el umbral"},
        formatos={"mortalidad": "prop2", "razon_vs_resto": "dec2", "razon_bajo": "dec2", "razon_alto": "dec2"},
        leyenda="Estratos de la decisión 15",
    )

    # --- límites ---------------------------------------------------------------------------------------
    inf.seccion(f"{n0 + 5}. Límites")
    limites = [
        "- Los intervalos de confianza ignoran el agrupamiento por paciente y por hospital.",
        "- La cascada del pipeline es una reproducción aproximada de su definición.",
        "- El desenlace y la fecha de alta de 2024 no se leen: el cortafuegos lo impide y deja registro del intento.",
    ]
    if ctx.etiqueta == "extracto":                  # las frases sobre el extracto solo van si la corrida usó extractos
        limites.insert(
            0, "- Los extractos no son una muestra aleatoria simple: las prevalencias y las razones son indicativas y los conteos absolutos no se comparan con los datos completos."
        )
    inf.texto_libre_sin_cifras("\n".join(limites))

