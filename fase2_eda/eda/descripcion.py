"""CRISP-DM 2.2: descripción de los datos.

Tablas principales:
- ``t22_perfil_columnas.csv``: nulos, centinelas, cardinalidad, Chao1, tipo inferido, rangos y patrones por columna y año.
- ``t22_top_valores.csv``: valores más frecuentes (n >= 10) de columnas categóricas no sensibles.
- ``t22_cardinalidades.csv``: cardinalidad observada y estimada (Chao1) contra la documentada.
- ``t22_deriva_categorias.csv`` y ``t22_psi.csv``: deriva de seis variables categóricas.
- ``t22_hist_edad.csv`` y ``t22_hist_estancia.csv``: histogramas desde conteos (la estadía solo 2019 a 2023).
- ``t22_pares_traslado.csv`` y ``t22_bloque_rn.csv``: pares fecha-servicio de traslados y bloque de recién nacidos.
- ``t22_etiquetas_truncadas.csv``: etiquetas categóricas que parecen cortadas.
- ``diccionario_medido.csv`` y ``t22_conciliacion.csv``: conciliación fila a fila con el diccionario de la Fase 2.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats as sps

from . import config, esquema, estadistica, graficos, rutas
from . import transformaciones as tr
from .contexto import Contexto
from .informes import Informe, cifra

TAREA = "T04"
ANIOS_TXT = "2019 a 2024"

# Columnas cuyos valores nunca se listan (identificadores, fechas, valores con estructura de RUT).
SIN_VALORES = frozenset(
    {
        "CIP_ENCRIPTADO",
        "MEDICOINTERV1_ENCRIPTADO",
        "MEDICOALTA_ENCRIPTADO",
        "FECHAPROCEDIMIENTO1",
        *esquema.FECHAS,
    }
)
CATEGORICAS_DERIVA = ("TIPO_ACTIVIDAD", "ETNIA", "PREVISION", "TIPO_PROCEDENCIA", "TIPOALTA", "TIPO_INGRESO")
COLS_CARDINALIDAD = (
    "COD_HOSPITAL",
    "SERVICIO_SALUD",
    "ESPECIALIDAD_MEDICA",
    "PREVISION",
    "TIPO_PROCEDENCIA",
    "TIPOALTA",
    "TIPO_ACTIVIDAD",
    "TIPO_INGRESO",
    "ETNIA",
    "NACIONALIDAD",
    "PROVINCIA",
    "COMUNA",
    "SERVICIOINGRESO",
    "DIAGNOSTICO1",
    "IR_29301_COD_GRD",
    "HOSPPROCEDENCIA",
)
MAX_DISTINTOS_TOP = 400
K_TOP = 15

_ISO = r"^\d{4}-\d{2}-\d{2}$"
_DMY = r"^\d{2}-\d{2}-\d{4}$"
_ENTERO = r"^-?\d+$"
_DECIMAL = r"^-?\d*[.,]?\d+$"  # números con o sin parte decimal (coma o punto; incluye ",5")


# ---------------------------------------------------------------------------- utilidades
def etiquetas_truncadas(etiquetas: list[str]) -> list[str]:
    """Etiquetas que parecen cortadas por un límite de largo.

    Criterio: paréntesis abiertos sin cerrar y un largo igual o apenas menor al máximo observado entre
    esas etiquetas (el corte puede dejar un espacio final que luego se recorta). Las etiquetas
    completas con paréntesis balanceados no se marcan.
    """
    candidatas = [e for e in etiquetas if e.count("(") > e.count(")")]
    if not candidatas:
        return []
    tope = max(len(e.rstrip()) for e in candidatas)
    return sorted(e for e in candidatas if len(e) >= tope - 1)


def inferir_tipo(s: pl.Series) -> str:
    """Tipo lógico de una columna de texto: fecha_iso, fecha_dmy, entero, decimal, texto o vacia."""
    nn = s.drop_nulls()
    if nn.len() == 0:
        return "vacia"
    cent = pl.DataFrame({"v": nn}).select(esquema.es_centinela(pl.col("v")).alias("c"))["c"]
    nn = nn.filter(~cent)
    if nn.len() == 0:
        return "texto"
    n = nn.len()
    for tipo, pat in (("fecha_iso", _ISO), ("fecha_dmy", _DMY), ("entero", _ENTERO), ("decimal", _DECIMAL)):
        if nn.str.contains(pat).sum() / n >= 0.99:
            return tipo
    return "texto"


PERFIL_SCHEMA = {
    "columna": pl.String,
    "anio": pl.Int64,
    "n_filas": pl.Int64,
    "n_nulos": pl.Int64,
    "n_centinela": pl.Int64,
    "n_informados": pl.Int64,
    "n_distintos": pl.Int64,
    "chao1": pl.Float64,
    "tipo_inferido": pl.String,
    "longitud_min": pl.Int64,
    "longitud_max": pl.Int64,
    "p01": pl.Float64,
    "mediana": pl.Float64,
    "p99": pl.Float64,
    "anio_min": pl.Int64,
    "anio_max": pl.Int64,
    "patron_moda": pl.String,
    "n_patron_moda": pl.Int64,
    "prop_patron_moda": pl.Float64,
    "n_patrones": pl.Int64,
    "valor_modal": pl.String,
    "n_modal": pl.Int64,
    "prop_modal": pl.Float64,
    "enmascarada": pl.String,
}


@dataclass
class AcumuladoAnio:
    perfil: list[dict] = field(default_factory=list)
    top: list[dict] = field(default_factory=list)
    categorias: list[dict] = field(default_factory=list)
    hist_edad: list[dict] = field(default_factory=list)
    hist_estancia: list[dict] = field(default_factory=list)
    pares: list[dict] = field(default_factory=list)


def _tramo_estancia(dias: pl.Expr) -> pl.Expr:
    return (
        pl.when(dias < 0).then(pl.lit("negativa"))
        .when(dias <= 60).then(dias.cast(pl.String))
        .when(dias <= 90).then(pl.lit("61-90"))
        .when(dias <= 180).then(pl.lit("91-180"))
        .when(dias <= 365).then(pl.lit("181-365"))
        .otherwise(pl.lit(">365"))
    )


def _orden_tramo(t: str) -> int:
    if t == "negativa":
        return -1
    if t.isdigit():
        return int(t)
    return {"61-90": 61, "91-180": 91, "181-365": 181, ">365": 366}[t]


def perfil_columna(df: pl.DataFrame, col: str, anio: int, protegida: bool) -> tuple[dict, list[dict], Counter | None]:
    """Perfil de una columna en un año. Devuelve (fila de perfil, filas top, conteo de valores)."""
    s = df[col]
    n = df.height
    n_nulos = s.null_count()
    fila: dict[str, Any] = {
        "columna": col,
        "anio": anio,
        "n_filas": n,
        "n_nulos": n_nulos,
        "n_centinela": None,
        "n_informados": n - n_nulos,
        "n_distintos": None,
        "chao1": None,
        "tipo_inferido": None,
        "longitud_min": None,
        "longitud_max": None,
        "p01": None,
        "mediana": None,
        "p99": None,
        "anio_min": None,
        "anio_max": None,
        "patron_moda": None,
        "n_patron_moda": None,
        "prop_patron_moda": None,
        "n_patrones": None,
        "valor_modal": None,
        "n_modal": None,
        "prop_modal": None,
        "enmascarada": "si" if protegida else "no",
    }
    nn = s.drop_nulls()
    if nn.len() == 0:
        fila["tipo_inferido"] = "vacia"
        fila["n_centinela"] = 0
        fila["n_distintos"] = 0
        return fila, [], None

    # patrón de caracteres (válido también para columnas enmascaradas)
    pat = pl.DataFrame({"v": nn}).select(esquema.mascara_patron(pl.col("v")).alias("p"))["p"].value_counts().sort(
        ["count", "p"], descending=[True, False]
    )
    fila["n_patrones"] = pat.height
    fila["patron_moda"] = pat["p"][0]
    fila["n_patron_moda"] = int(pat["count"][0])
    fila["prop_patron_moda"] = pat["count"][0] / nn.len()
    if protegida:
        return fila, [], None

    cent = int(pl.DataFrame({"v": nn}).select(esquema.es_centinela(pl.col("v")).sum()).item())
    fila["n_centinela"] = cent
    vc = nn.value_counts()
    fila["n_distintos"] = vc.height
    fila["tipo_inferido"] = inferir_tipo(s)
    largos = nn.str.len_chars()
    fila["longitud_min"] = int(largos.min())
    fila["longitud_max"] = int(largos.max())
    tipo = fila["tipo_inferido"]
    if tipo in ("entero", "decimal"):
        x = pl.DataFrame({"v": nn}).select(tr.a_decimal(pl.col("v")).alias("x"))["x"].drop_nulls()
        if x.len():
            fila["p01"], fila["mediana"], fila["p99"] = (
                float(x.quantile(0.01)),
                float(x.median()),
                float(x.quantile(0.99)),
            )
    elif tipo in ("fecha_iso", "fecha_dmy"):
        f = pl.DataFrame({"v": nn}).select(tr.a_fecha(pl.col("v")).dt.year().alias("y"))["y"].drop_nulls()
        if f.len():
            fila["anio_min"], fila["anio_max"] = int(f.min()), int(f.max())
    if col not in SIN_VALORES and vc.height <= 200:
        mejor = vc.sort(["count", col], descending=[True, False]).row(0)
        fila["valor_modal"] = mejor[0]
        fila["n_modal"] = int(mejor[1])
        fila["prop_modal"] = mejor[1] / nn.len()
    if vc.height > 0:
        fila["chao1"] = estadistica.chao1(vc["count"].to_list()) if col not in SIN_VALORES else None

    top: list[dict] = []
    elegible = col not in SIN_VALORES and (vc.height <= MAX_DISTINTOS_TOP or col in ("DIAGNOSTICO1", "PROCEDIMIENTO1"))
    if elegible and tipo in ("texto", "entero", "decimal"):
        t = vc.sort(["count", col], descending=[True, False]).head(K_TOP)
        for v, c in t.iter_rows():
            top.append(
                {
                    "columna": col,
                    "anio": anio,
                    "valor": v,
                    "n": c,
                    "prop": c / nn.len(),
                    "espacios_extremos": "si" if v != v.strip() else "no",
                }
            )
    conteo = Counter(dict(vc.iter_rows())) if (col not in SIN_VALORES and vc.height <= 20000) else None
    return fila, top, conteo


def analizar_anio(df: pl.DataFrame, anio: int, protegidas: frozenset[str]) -> tuple[AcumuladoAnio, dict[str, Counter]]:
    """Perfila las 129 columnas de un año y calcula conteos auxiliares (sin guardar filas)."""
    acc = AcumuladoAnio()
    conteos: dict[str, Counter] = {}
    perfilar_columnas(df, anio, protegidas, acc, conteos)
    analizar_resto(df, anio, protegidas, acc)
    return acc, conteos


COLS_RESTO = (
    CATEGORICAS_DERIVA
    + ("FECHA_NACIMIENTO", "FECHA_INGRESO", "FECHAALTA")
    + tuple(f"{t}{k}" for k in range(1, 10) for t in ("FECHATRASLADO", "SERVICIOTRASLADO"))
)
"""Columnas que necesita ``analizar_resto`` (deriva de categorías, histogramas y pares de traslado)."""


def perfilar_columnas(df: pl.DataFrame, anio: int, protegidas: frozenset[str], acc: "AcumuladoAnio", conteos: dict[str, Counter]) -> None:
    """Perfil de cada columna de ``df`` (puede ser un bloque de columnas): agrega al acumulado y a los conteos."""
    for col in df.columns:
        fila, top, conteo = perfil_columna(df, col, anio, col in protegidas)
        acc.perfil.append(fila)
        acc.top.extend(top)
        if conteo is not None:
            conteos[col] = conteo


def analizar_anio_por_bloques(lector, anio: int, protegidas: frozenset[str], tarea: str, bloque: int = 16) -> tuple["AcumuladoAnio", dict[str, Counter]]:
    """Igual que ``analizar_anio`` pero sin tener las 129 columnas a la vez: perfila por bloques de columnas y lee aparte
    solo las columnas que necesitan la deriva, los histogramas y los pares. Da exactamente las mismas cifras."""
    acc = AcumuladoAnio()
    conteos: dict[str, Counter] = {}
    for df in lector.leer_por_bloques(anio, None, bloque=bloque, proposito="estructura", tarea=tarea):
        perfilar_columnas(df, anio, protegidas, acc, conteos)
        del df
    df = lector.leer(anio, list(COLS_RESTO), proposito="estructura", tarea=tarea)
    analizar_resto(df, anio, protegidas, acc)
    return acc, conteos


def analizar_resto(df: pl.DataFrame, anio: int, protegidas: frozenset[str], acc: "AcumuladoAnio") -> None:
    """Deriva de categorías, histogramas de edad y estadía y pares fecha-servicio de traslados de un año."""
    # categorías para la deriva (la columna de resultado del año expuesto no se tabula)
    for col in CATEGORICAS_DERIVA:
        if col in protegidas:
            continue
        vc = df[col].drop_nulls().value_counts()
        total = int(vc["count"].sum()) if vc.height else 0
        for v, c in vc.iter_rows():
            acc.categorias.append({"columna": col, "anio": anio, "categoria": v, "n": c, "prop": c / total if total else None})

    # histogramas desde conteos
    nac = tr.a_fecha(pl.col("FECHA_NACIMIENTO"))
    ing = tr.a_fecha(pl.col("FECHA_INGRESO"))
    edad = df.select(tr.edad_anios(nac, ing).alias("edad"))["edad"].drop_nulls()
    h = edad.filter((edad >= 0) & (edad <= 110)).value_counts().sort("edad")
    for e, c in h.iter_rows():
        acc.hist_edad.append({"anio": anio, "edad": e, "n": c})
    if "FECHAALTA" not in protegidas:
        dias = df.select(tr.estancia_dias(ing, tr.a_fecha(pl.col("FECHAALTA"))).alias("d"))["d"].drop_nulls()
        tramos = pl.DataFrame({"d": dias}).select(_tramo_estancia(pl.col("d")).alias("t"))["t"].value_counts()
        for t, c in tramos.iter_rows():
            acc.hist_estancia.append({"anio": anio, "tramo": t, "orden": _orden_tramo(t), "n": c})

    # pares fecha-servicio de los traslados internos
    for k in range(1, 10):
        f, s = df[f"FECHATRASLADO{k}"], df[f"SERVICIOTRASLADO{k}"]
        acc.pares.append(
            {
                "anio": anio,
                "traslado": k,
                "n_fecha": f.len() - f.null_count(),
                "n_servicio": s.len() - s.null_count(),
                "n_solo_fecha": int((f.is_not_null() & s.is_null()).sum()),
                "n_solo_servicio": int((f.is_null() & s.is_not_null()).sum()),
            }
        )


# --------------------------------------------------------------------------- diccionario
def _entero_es(txt: str) -> int:
    return int(str(txt).replace(".", "").replace(",", "").strip() or 0)


def cargar_diccionario(ruta: Path | str | None = None) -> pl.DataFrame:
    """Diccionario de la Fase 2 con los conteos como enteros."""
    ruta = Path(ruta) if ruta else rutas.DIR_FASE2 / "Fase_2__Diccionario_de_Base_de_Datos.csv"
    d = config.leer_csv_texto(ruta)
    return pl.DataFrame(
        {
            "columna": d["Variable"],
            "que_representa": d["Qué representa"],
            "tipo_doc": d["Tipo de dato"],
            "con_datos_doc": [_entero_es(x) for x in d["Registros con datos"]],
            "sin_datos_doc": [_entero_es(x) for x in d["Registros sin datos"]],
            "uso": d["Uso en el proyecto"],
            "categoria_fase1": d["Categoría Fase 1"],
            "nota": d["Nota de la categoría"],
        }
    )


def conciliar(
    con_doc: int,
    sin_doc: int,
    sin_med: int,
    n_med: int,
    modo: str,
    z_umbral: float = 3.0,
) -> tuple[str, float | None]:
    """Compara el conteo de vacíos medido con el documentado. Devuelve (estado, z).

    ``completo``: coincidencia exacta de los conteos.
    ``extracto``: contraste de proporciones con error estándar binomial; los eventos raros (menos de
    cinco casos esperados) no se evalúan salvo que el diccionario afirme algo determinista.
    """
    n_doc = con_doc + sin_doc
    if n_doc <= 0:
        return "NO EVALUABLE", None
    if modo == "completo":
        return ("COINCIDE" if (sin_med == sin_doc and n_med == n_doc) else "DISCREPANCIA"), None
    p = sin_doc / n_doc
    if p in (0.0, 1.0):  # afirmación determinista: todo vacío o todo con dato
        esperado = p * n_med
        return ("COINCIDE" if sin_med == esperado else "DISCREPANCIA"), None
    esperado = p * n_med
    if min(esperado, n_med - esperado) < 5:
        # evento raro: Poisson sobre el lado raro
        lado = sin_med if esperado < n_med - esperado else n_med - sin_med
        lam = min(esperado, n_med - esperado)
        pv = float(min(sps.poisson.sf(lado - 1, lam), sps.poisson.cdf(lado, lam))) if lado >= 0 else 1.0
        return ("DISCREPANCIA" if pv < 0.001 else "NO EVALUABLE"), None
    se = np.sqrt(n_med * p * (1 - p))
    z = float((sin_med - esperado) / se)
    return ("DISCREPANCIA" if abs(z) > z_umbral else "COINCIDE"), z


def conciliar_estratificado(
    con_doc: int,
    sin_doc: int,
    estratos: list[tuple[int, int, float]],
    z_umbral: float = 3.0,
) -> tuple[str, float | None, float | None]:
    """Contraste del porcentaje sin dato con post-estratificación por año.

    ``estratos`` es una lista de (filas del extracto del año, sin dato del año, peso poblacional del año).
    Cada año del extracto es una muestra de tamaño parecido de poblaciones de tamaños distintos; el porcentaje
    poblacional se estima ponderando cada año por su volumen: p = sum(W_h p_h), con varianza
    sum(W_h^2 p_h (1 - p_h) / n_h). Devuelve (estado, z, p estimado).
    """
    n_doc = con_doc + sin_doc
    n_tot = sum(n for n, _, _ in estratos)
    sin_tot = sum(s for _, s, _ in estratos)
    if n_doc <= 0 or n_tot <= 0:
        return "NO EVALUABLE", None, None
    p_doc = sin_doc / n_doc
    w = np.array([max(float(x), 0.0) for _, _, x in estratos])
    w = w / w.sum() if w.sum() > 0 else np.full(len(estratos), 1.0 / len(estratos))
    nh = np.array([n for n, _, _ in estratos], dtype=float)
    ph = np.array([s / n if n else 0.0 for n, s, _ in estratos])
    p_hat = float((w * ph).sum())
    if p_doc in (0.0, 1.0):  # afirmación determinista
        return ("COINCIDE" if abs(p_hat - p_doc) < 1e-12 else "DISCREPANCIA"), None, p_hat
    esperado = p_doc * n_tot
    if min(esperado, n_tot - esperado) < 5:
        lado = sin_tot if esperado < n_tot - esperado else n_tot - sin_tot
        lam = min(esperado, n_tot - esperado)
        pv = float(min(sps.poisson.sf(lado - 1, lam), sps.poisson.cdf(lado, lam))) if lado >= 0 else 1.0
        return ("DISCREPANCIA" if pv < 0.001 else "NO EVALUABLE"), None, p_hat
    var = float((w**2 * ph * (1 - ph) / nh).sum())
    if var <= 0:
        return ("COINCIDE" if abs(p_hat - p_doc) < 1e-12 else "DISCREPANCIA"), None, p_hat
    z = (p_hat - p_doc) / np.sqrt(var)
    return ("DISCREPANCIA" if abs(z) > z_umbral else "COINCIDE"), float(z), p_hat


def magnitud_discrepancia(estado: str, dif_pp: float | None, dif_relevante_pp: float) -> str:
    """Rotula la magnitud de una discrepancia por su diferencia en puntos porcentuales.

    ``no aplica`` si el estado no es DISCREPANCIA; ``relevante`` si la diferencia llega al umbral
    ``conciliacion_dif_relevante_pp`` (la igualdad cuenta como relevante); ``despreciable`` en otro caso
    o si no hay diferencia calculable. Lo usan la conciliación del diccionario (Tarea 4) y el ledger (Tarea 12).
    """
    if estado != "DISCREPANCIA":
        return "no aplica"
    return "relevante" if (dif_pp is not None and dif_pp >= dif_relevante_pp) else "despreciable"


def tabla_conciliacion(
    dic: pl.DataFrame,
    perfil: pl.DataFrame,
    modo: str,
    z_umbral: float,
    pesos_anio: dict[int, float] | None = None,
    dif_relevante_pp: float = 0.5,
) -> pl.DataFrame:
    """Una fila por columna del diccionario con el conteo medido sumado en todos los años.

    En modo extracto, si se entregan ``pesos_anio`` (volumen poblacional de cada año) el contraste se
    hace con post-estratificación por año; sin pesos, con una proporción simple.
    """
    por_anio: dict[str, list[tuple[int, int, float]]] = {}
    if modo == "extracto" and pesos_anio:
        for r in perfil.select(["columna", "anio", "n_filas", "n_nulos"]).iter_rows(named=True):
            por_anio.setdefault(r["columna"], []).append(
                (int(r["n_filas"]), int(r["n_nulos"]), float(pesos_anio.get(int(r["anio"]), 0.0)))
            )
    agg = perfil.group_by("columna").agg(
        pl.col("n_filas").sum().alias("n_filas_med"),
        pl.col("n_nulos").sum().alias("n_sin_med"),
        pl.col("n_centinela").sum().alias("n_centinela_med"),
        pl.col("n_distintos").max().alias("distintos_max"),
        pl.col("tipo_inferido").drop_nulls().mode().sort().first().alias("tipo_medido"),
    )
    filas = []
    por_col = {r["columna"]: r for r in agg.iter_rows(named=True)}
    for d in dic.iter_rows(named=True):
        m = por_col.get(d["columna"])
        if m is None:
            filas.append({**d, "n_filas_med": None, "n_con_med": None, "n_sin_med": None, "n_centinela_med": None,
                          "pct_sin_doc": None, "pct_sin_med": None, "dif_pp": None, "z": None, "estado": "NO EVALUABLE",
                          "magnitud": "no aplica", "tipo_medido": None, "observacion": "la columna no está en los archivos"})
            continue
        n_doc = d["con_datos_doc"] + d["sin_datos_doc"]
        if d["columna"] in por_anio:
            estado, z, _ = conciliar_estratificado(d["con_datos_doc"], d["sin_datos_doc"], por_anio[d["columna"]], z_umbral)
        else:
            estado, z = conciliar(d["con_datos_doc"], d["sin_datos_doc"], int(m["n_sin_med"]), int(m["n_filas_med"]), modo, z_umbral)
        obs = []
        t_doc, t_med = d["tipo_doc"], m["tipo_medido"]
        if t_doc == "Fecha" and t_med not in ("fecha_iso", "fecha_dmy", "vacia", None):
            obs.append(f"documentada como fecha pero el contenido es {t_med}")
        if t_doc == "Número" and t_med not in ("entero", "decimal", "vacia", None):
            obs.append(f"documentada como número pero el contenido es {t_med}")
        if m["n_centinela_med"] and m["n_centinela_med"] > 0:
            obs.append("tiene valores centinela que el diccionario no cuenta como faltantes")
        pct_doc = 100.0 * d["sin_datos_doc"] / n_doc if n_doc else None
        pct_med = 100.0 * m["n_sin_med"] / m["n_filas_med"] if m["n_filas_med"] else None
        dif = abs(pct_med - pct_doc) if (pct_doc is not None and pct_med is not None) else None
        filas.append(
            {
                **d,
                "n_filas_med": int(m["n_filas_med"]),
                "n_con_med": int(m["n_filas_med"] - m["n_sin_med"]),
                "n_sin_med": int(m["n_sin_med"]),
                "n_centinela_med": int(m["n_centinela_med"] or 0),
                "pct_sin_doc": pct_doc,
                "pct_sin_med": pct_med,
                "dif_pp": dif,
                "z": z,
                "estado": estado,
                "magnitud": magnitud_discrepancia(estado, dif, dif_relevante_pp),
                "tipo_medido": t_med,
                "observacion": "; ".join(obs),
            }
        )
    return pl.DataFrame(filas)


# ------------------------------------------------------------------------------ ejecución
def _periodo_conteos(
    conteos_por_anio: dict[int, dict[str, Counter]], anios: list[int], col: str
) -> Counter | None:
    tot: Counter = Counter()
    hay = False
    for a in anios:
        c = conteos_por_anio.get(a, {}).get(col)
        if c is not None:
            tot.update(c)
            hay = True
    return tot if hay else None


def calcular(ctx: Contexto) -> dict[str, pl.DataFrame]:
    """Recorre los años una vez y devuelve todas las tablas de la descripción (sin suprimir)."""
    u = ctx.umbrales
    acc_perfil: list[dict] = []
    acc_top: list[dict] = []
    acc_cat: list[dict] = []
    acc_edad: list[dict] = []
    acc_est: list[dict] = []
    acc_par: list[dict] = []
    conteos_por_anio: dict[int, dict[str, Counter]] = {}
    anios = ctx.fuente.anios()
    for anio in anios:
        protegidas = ctx.cortafuegos.columnas_protegidas(anio)
        if ctx.lector.cache is not None:
            # base completa: por bloques de columnas, para no tener las 129 columnas en memoria a la vez
            df = None
            acc, conteos = analizar_anio_por_bloques(ctx.lector, anio, protegidas, TAREA)
        else:
            df = ctx.lector.leer(anio, None, proposito="estructura", tarea=TAREA)
            acc, conteos = analizar_anio(df, anio, protegidas)
        acc_perfil += acc.perfil
        acc_top += acc.top
        acc_cat += acc.categorias
        acc_edad += acc.hist_edad
        acc_est += acc.hist_estancia
        acc_par += acc.pares
        conteos_por_anio[anio] = conteos
        del df

    perfil = pl.DataFrame(acc_perfil, schema=PERFIL_SCHEMA).sort(["columna", "anio"])
    # --- cardinalidades (por año y acumuladas) ---
    permitidos = [a for a in anios if a in u.particion["anios_resultado"]]
    filas_card = []
    for col in COLS_CARDINALIDAD:
        for p in perfil.filter(pl.col("columna") == col).iter_rows(named=True):
            if p["n_distintos"] is None:
                continue
            filas_card.append({"columna": col, "periodo": str(p["anio"]), "n_observado": p["n_distintos"], "chao1": p["chao1"]})
        for nombre, lista in (("2019-2023", permitidos), ("2019-2024", anios)):
            tot = _periodo_conteos(conteos_por_anio, lista, col)
            if tot is None or (nombre == "2019-2024" and col in ctx.cortafuegos.columnas_resultado):
                continue
            # un período acumulado solo vale si todos sus años tienen conteos
            if not all(col in conteos_por_anio.get(a, {}) for a in lista):
                continue
            filas_card.append(
                {"columna": col, "periodo": nombre, "n_observado": len(tot), "chao1": estadistica.chao1(tot.values())}
            )
    card = pl.DataFrame(filas_card, infer_schema_length=None)

    # --- etiquetas truncadas sobre conteos acumulados ---
    filas_trunc = []
    cols_con_conteo = sorted({c for d in conteos_por_anio.values() for c in d})
    for col in cols_con_conteo:
        tot = _periodo_conteos(conteos_por_anio, anios, col)
        if tot is None or len(tot) > MAX_DISTINTOS_TOP:
            continue
        for et in etiquetas_truncadas(list(tot)):
            filas_trunc.append({"columna": col, "etiqueta": et, "longitud": len(et), "n": tot[et]})
    trunc = pl.DataFrame(filas_trunc, schema={"columna": pl.String, "etiqueta": pl.String, "longitud": pl.Int64, "n": pl.Int64})

    # --- deriva de categorías y PSI ---
    cat = pl.DataFrame(acc_cat, infer_schema_length=None).sort(["columna", "anio", "categoria"]) if acc_cat else pl.DataFrame()
    filas_psi = []
    for col in CATEGORICAS_DERIVA:
        sub = cat.filter(pl.col("columna") == col)
        anios_col = sorted(sub["anio"].unique().to_list())
        dist = {a: dict(sub.filter(pl.col("anio") == a).select(["categoria", "n"]).iter_rows()) for a in anios_col}
        for a0, a1 in zip(anios_col[:-1], anios_col[1:]):
            v = estadistica.psi(dist[a0], dist[a1])
            filas_psi.append({"columna": col, "anio_base": a0, "anio_comparado": a1, "psi": v,
                              "clasificacion": estadistica.clasificar_psi(v, u["psi_moderado"], u["psi_alto"]),
                              "n_categorias": len(set(dist[a0]) | set(dist[a1]))})
        if len(anios_col) >= 2:
            a0 = anios_col[0]
            for a1 in anios_col[1:]:
                if a1 == anios_col[1]:
                    continue  # ya está como par consecutivo
                v = estadistica.psi(dist[a0], dist[a1])
                filas_psi.append({"columna": col, "anio_base": a0, "anio_comparado": a1, "psi": v,
                                  "clasificacion": estadistica.clasificar_psi(v, u["psi_moderado"], u["psi_alto"]),
                                  "n_categorias": len(set(dist[a0]) | set(dist[a1]))})
    psi_df = pl.DataFrame(filas_psi, infer_schema_length=None)

    # --- pooled RN y pares ---
    rn_cols = list(esquema.BLOQUE_RN)
    filas_rn = []
    for col in rn_cols:
        tot = _periodo_conteos(conteos_por_anio, anios, col)
        pf = perfil.filter(pl.col("columna") == col)
        n_inf = int(pf["n_informados"].sum())
        if tot:
            modal, nm = max(tot.items(), key=lambda kv: (kv[1], kv[0]))
            filas_rn.append({"columna": col, "n_informados": n_inf, "n_distintos": len(tot), "valor_modal": modal,
                             "n_modal": nm, "prop_modal": nm / n_inf if n_inf else None})
        else:
            filas_rn.append({"columna": col, "n_informados": n_inf, "n_distintos": 0, "valor_modal": None,
                             "n_modal": 0, "prop_modal": None})
    rn = pl.DataFrame(filas_rn, schema={"columna": pl.String, "n_informados": pl.Int64, "n_distintos": pl.Int64,
                                         "valor_modal": pl.String, "n_modal": pl.Int64, "prop_modal": pl.Float64})
    pares = pl.DataFrame(acc_par)
    pares_tot = (
        pares.group_by("traslado")
        .agg(pl.col("n_fecha").sum(), pl.col("n_servicio").sum(), pl.col("n_solo_fecha").sum(), pl.col("n_solo_servicio").sum())
        .with_columns((pl.col("n_servicio") - pl.col("n_fecha")).alias("diferencia"))
        .sort("traslado")
    )
    top = pl.DataFrame(acc_top, infer_schema_length=None).sort(["columna", "anio", "n", "valor"], descending=[False, False, True, False]) if acc_top else pl.DataFrame()
    edad = pl.DataFrame(acc_edad).sort(["anio", "edad"]) if acc_edad else pl.DataFrame()
    est = pl.DataFrame(acc_est).sort(["anio", "orden"]) if acc_est else pl.DataFrame()
    return {
        "perfil": perfil, "top": top, "card": card, "trunc": trunc, "cat": cat, "psi": psi_df,
        "rn": rn, "pares": pares_tot, "pares_anio": pares, "edad": edad, "est": est, "conteos_etnia": Counter(),
        "_conteos": conteos_por_anio,
    }


def ejecutar(ctx: Contexto, ruta_diccionario: Path | None = None) -> dict[str, Path]:
    u = ctx.umbrales
    r = calcular(ctx)
    perfil, cat = r["perfil"], r["cat"]
    TA = TAREA

    # perfil: protegemos los conteos de casos
    ctx.escribir_tabla(
        "t22_perfil_columnas.csv",
        perfil.with_columns(
            (100.0 * pl.col("n_nulos") / pl.col("n_filas")).alias("pct_nulos"),
            (100.0 * pl.col("n_centinela").fill_null(0) / pl.col("n_filas")).alias("pct_centinela"),
            (100.0 * (pl.col("n_nulos") + pl.col("n_centinela").fill_null(0)) / pl.col("n_filas")).alias("pct_faltante"),
        ),
        tarea=TA, descripcion="Perfil de las columnas por año",
        cols_n=["n_nulos", "n_centinela", "n_informados", "n_modal", "n_patron_moda"],
        dependientes=["pct_nulos", "pct_centinela", "pct_faltante", "prop_modal", "prop_patron_moda", "chao1"],
    )
    ctx.escribir_tabla("t22_top_valores.csv", r["top"], tarea=TA, descripcion="Valores más frecuentes de columnas categóricas",
                       cols_n=["n"], dependientes=["prop"])
    # documentado vs observado
    doc = config.leer_csv_texto(rutas.DIR_CONFIG / "cifras_documentadas.csv")
    doc_card = {row["clave_eda"].split(".", 1)[1]: float(row["valor"]) for row in doc.iter_rows(named=True)
                if row["clave_eda"].startswith("cardinalidad.")}
    card = r["card"].with_columns(
        pl.col("columna").replace_strict(doc_card, default=None, return_dtype=pl.Float64).alias("documentado")
    )
    ctx.escribir_tabla("t22_cardinalidades.csv", card, tarea=TA, descripcion="Cardinalidad observada, Chao1 y documentada")
    ctx.escribir_tabla("t22_deriva_categorias.csv", cat, tarea=TA, descripcion="Distribución de seis categóricas por año",
                       cols_n=["n"], dependientes=["prop"])
    ctx.escribir_tabla("t22_psi.csv", r["psi"], tarea=TA, descripcion="PSI entre años de seis categóricas")
    ctx.escribir_tabla("t22_hist_edad.csv", r["edad"], tarea=TA, descripcion="Histograma de edad al ingreso", cols_n=["n"])
    ctx.escribir_tabla("t22_hist_estancia.csv", r["est"], tarea=TA, descripcion="Histograma de estadía (solo 2019 a 2023)", cols_n=["n"])
    ctx.escribir_tabla("t22_pares_traslado.csv", r["pares"], tarea=TA, descripcion="Pares fecha-servicio de traslados internos",
                       cols_n=["n_fecha", "n_servicio", "n_solo_fecha", "n_solo_servicio"])
    ctx.escribir_tabla("t22_bloque_rn.csv", r["rn"], tarea=TA, descripcion="Bloque de recién nacidos: valor modal",
                       cols_n=["n_informados", "n_modal"], dependientes=["prop_modal"])
    ctx.escribir_tabla("t22_etiquetas_truncadas.csv", r["trunc"], tarea=TA, descripcion="Etiquetas categóricas que parecen cortadas",
                       cols_n=["n"])

    # --- conciliación con el diccionario ---
    dic = cargar_diccionario(ruta_diccionario)
    pesos = {int(row["clave_eda"].rsplit(".", 1)[1]): float(row["valor"]) for row in doc.iter_rows(named=True)
             if row["clave_eda"].startswith("volumen.filas_leidas.") and row["clave_eda"].rsplit(".", 1)[1].isdigit()}
    conc = tabla_conciliacion(
        dic, perfil, ctx.etiqueta, float(u["conciliacion_z"]), pesos_anio=pesos,
        dif_relevante_pp=float(u["conciliacion_dif_relevante_pp"]),
    )
    ctx.escribir_tabla("t22_conciliacion.csv", conc.select(
        ["columna", "tipo_doc", "tipo_medido", "con_datos_doc", "sin_datos_doc", "n_filas_med", "n_con_med", "n_sin_med",
         "n_centinela_med", "pct_sin_doc", "pct_sin_med", "dif_pp", "z", "estado", "magnitud", "categoria_fase1", "observacion"]),
        tarea=TA, descripcion="Conciliación fila a fila con el diccionario de la Fase 2",
        cols_n=["n_con_med", "n_sin_med", "n_centinela_med"])
    ctx.escribir_tabla("diccionario_medido.csv", conc.select(
        ["columna", "que_representa", "tipo_doc", "tipo_medido", "categoria_fase1", "n_filas_med", "n_con_med", "n_sin_med",
         "n_centinela_med", "pct_sin_doc", "pct_sin_med", "dif_pp", "estado", "magnitud", "observacion"]),
        tarea=TA, descripcion="Diccionario medido: documentado frente a medido",
        cols_n=["n_con_med", "n_sin_med", "n_centinela_med"])

    # --- métricas para el ledger ---
    _registrar_metricas(ctx, r, dic, conc)
    resumen = pl.DataFrame([_resumen(ctx, r, conc, dic)])
    ctx.escribir_tabla("t22_resumen.csv", resumen, tarea=TA, descripcion="Cifras globales de la descripción")

    figura_nulos(ctx)
    figura_deriva_categorias(ctx)
    figura_edad_estancia(ctx)
    informe = construir_informe(ctx)
    return {"informe": informe}


def _resumen(ctx: Contexto, r: dict, conc: pl.DataFrame, dic: pl.DataFrame) -> dict:
    perfil = r["perfil"]
    por_col = perfil.group_by("columna").agg(
        pl.col("n_informados").sum().alias("inf"), pl.col("n_centinela").sum().alias("cent")
    )
    estados = conc["estado"].value_counts()
    ne = {k: v for k, v in estados.iter_rows()}
    rn = r["rn"].filter(pl.col("columna").is_in(["RN2ESTADO", "RN3ESTADO", "RN4ESTADO"]))
    return {
        "n_columnas_diccionario": dic.height,
        "n_columnas_perfiladas": perfil["columna"].n_unique(),
        "n_columnas_vacias": int((por_col["inf"] == 0).sum()),
        "n_columnas_con_centinelas": int((por_col["cent"].fill_null(0) > 0).sum()),
        "n_coincide": int(ne.get("COINCIDE", 0)),
        "n_discrepancia": int(ne.get("DISCREPANCIA", 0)),
        "n_discrepancia_relevante": int((conc["magnitud"] == "relevante").sum()),
        "n_no_evaluable": int(ne.get("NO EVALUABLE", 0)),
        "n_etiquetas_truncadas": r["trunc"].height,
        "rn_prop_modal_minima": float(rn["prop_modal"].drop_nulls().min()) if rn["prop_modal"].drop_nulls().len() else None,
        "psi_maximo": float(r["psi"]["psi"].max()) if r["psi"].height else None,
        "modo_conciliacion": "exacta (datos completos)" if ctx.etiqueta == "completo" else "por proporción ponderada por año, con error estándar (extracto)",
    }


def _registrar_metricas(ctx: Contexto, r: dict, dic: pl.DataFrame, conc: pl.DataFrame) -> None:
    perfil = r["perfil"]
    por_col = perfil.group_by("columna").agg(
        pl.col("n_filas").sum().alias("n"), pl.col("n_nulos").sum().alias("nulos"), pl.col("n_informados").sum().alias("inf")
    )
    for row in por_col.iter_rows(named=True):
        c = row["columna"]
        ctx.registrar_metrica(f"faltantes.n_nulos.{c}", row["nulos"], tarea=TAREA, tabla="t22_perfil_columnas.csv", conteo=True)
        ctx.registrar_metrica(f"faltantes.pct_nulos.{c}", 100.0 * row["nulos"] / row["n"], tarea=TAREA, tabla="t22_perfil_columnas.csv")
    hp = por_col.filter(pl.col("columna") == "HOSPPROCEDENCIA")
    if hp.height:
        ctx.registrar_metrica("procedencia.n_con_hospproc", int(hp["inf"][0]), tarea=TAREA, tabla="t22_perfil_columnas.csv", conteo=True)
        ctx.registrar_metrica("procedencia.pct_con_hospproc", 100.0 * hp["inf"][0] / hp["n"][0], tarea=TAREA, tabla="t22_perfil_columnas.csv")
    # cardinalidades acumuladas 2019-2024 (o 2019-2023 para columnas de resultado)
    for row in r["card"].iter_rows(named=True):
        if row["periodo"] == "2019-2024" or (row["periodo"] == "2019-2023" and row["columna"] in ctx.cortafuegos.columnas_resultado):
            ctx.registrar_metrica(f"cardinalidad.{row['columna']}", row["n_observado"], tarea=TAREA, tabla="t22_cardinalidades.csv")
            ctx.registrar_metrica(f"cardinalidad.{row['columna']}.chao1", row["chao1"], tarea=TAREA, tabla="t22_cardinalidades.csv")
    if r["card"].filter((pl.col("columna") == "TIPOALTA") & (pl.col("periodo") == "2019-2023")).height:
        n = r["card"].filter((pl.col("columna") == "TIPOALTA") & (pl.col("periodo") == "2019-2023"))["n_observado"][0]
        ctx.registrar_metrica("tipoalta.n_valores_distintos", n, tarea=TAREA, tabla="t22_cardinalidades.csv",
                              nota="valores distintos observados en 2019 a 2023")
    # categorías del diccionario
    for cat, n in dic.group_by("categoria_fase1").len().iter_rows():
        ctx.registrar_metrica(f"diccionario.n_columnas_categoria.{cat}", n, tarea=TAREA, tabla="diccionario_medido.csv")
    # ETNIA 'OTRO' con y sin espacio final, y sexo
    et = Counter()
    sexo = Counter()
    for a, d in r["_conteos"].items():
        et.update(d.get("ETNIA", {}))
        sexo.update(d.get("SEXO", {}))
    if et:
        ctx.registrar_metrica("calidad.etnia_otro_sin_espacio", et.get("OTRO", 0), tarea=TAREA, tabla="t22_top_valores.csv", conteo=True)
        ctx.registrar_metrica("calidad.etnia_otro_con_espacio", et.get("OTRO ", 0), tarea=TAREA, tabla="t22_top_valores.csv", conteo=True)
    m, h = sexo.get("MUJER", 0), sexo.get("HOMBRE", 0)
    if m + h:
        ctx.registrar_metrica("poblacion.pct_mujeres", 100.0 * m / (m + h), tarea=TAREA, tabla="t22_perfil_columnas.csv")
    pc = por_col.to_dict(as_series=False)
    inf = dict(zip(pc["columna"], pc["inf"]))
    for clave, col in (("calidad.fechatraslado2_con_dato", "FECHATRASLADO2"), ("calidad.serviciotraslado2_con_dato", "SERVICIOTRASLADO2"),
                       ("calidad.rn1estado_con_dato", "RN1ESTADO"), ("calidad.rn2estado_con_dato", "RN2ESTADO"),
                       ("calidad.condicionneonato2_con_dato", "CONDICIONDEALTANEONATO2")):
        ctx.registrar_metrica(clave, inf.get(col), tarea=TAREA, tabla="t22_perfil_columnas.csv", conteo=True)
    ctx.registrar_metrica("calidad.id_vacio.total", int(por_col.filter(pl.col("columna") == "CIP_ENCRIPTADO")["nulos"][0]),
                          tarea=TAREA, tabla="t22_perfil_columnas.csv", nota="valores nulos; los centinelas se cuentan aparte", conteo=True)
    ctx.registrar_metrica("calidad.alta_vacia.total", int(por_col.filter(pl.col("columna") == "FECHAALTA")["nulos"][0]),
                          tarea=TAREA, tabla="t22_perfil_columnas.csv", conteo=True)




# ---------------------------------------------------------------------------------- figuras
def figura_nulos(ctx: Contexto) -> None:
    t = ctx.leer_tabla("t22_perfil_columnas.csv")
    filas = list(esquema.COLUMNAS)
    m, f, c = graficos.tabla_a_matriz(t, "columna", "anio", "pct_faltante", orden_filas=filas)
    fig, ax = graficos.nueva_figura(figsize=(8.5, 19.0))
    graficos.mapa_calor(fig, ax, m, f, c, vmin=0, vmax=100, anotar=False, etiqueta_barra="% de nulos y centinelas")
    ax.tick_params(axis="y", labelsize=5.2)
    ax.set_xlabel("Año del archivo")
    ctx.figuras.guardar_figura(
        fig, id_figura="F04_01", tarea=4,
        titulo="Faltantes y centinelas por columna y año",
        descripcion_alt="Mapa de calor con el porcentaje de valores nulos o centinela de cada una de las 129 columnas en cada año.",
        csv="t22_perfil_columnas.csv", anios=ANIOS_TXT,
        extra_pie="En 2024 las columnas de resultado solo informan nulos.",
    )


def _acortar(txt: str, n: int = 30) -> str:
    """Etiqueta de leyenda: hace visibles los espacios en los extremos y acorta las muy largas."""
    visible = txt.strip() + " (con espacio extremo)" if txt != txt.strip() else txt
    return visible if len(visible) <= n + 10 else visible[: n + 9] + "…"


def figura_deriva_categorias(ctx: Contexto) -> None:
    t = ctx.leer_tabla("t22_deriva_categorias.csv", columnas_texto=["categoria"])
    fig, ejes = graficos.nueva_figura_multiple(2, 3, figsize=(13.0, 7.6))
    for ax, col in zip(ejes.ravel(), CATEGORICAS_DERIVA):
        sub = t.filter(pl.col("columna") == col)
        if sub.height == 0:
            ax.set_visible(False)
            continue
        anios = sorted(sub["anio"].unique().to_list())
        top = (
            sub.group_by("categoria").agg(pl.col("n").sum().alias("n")).sort(["n", "categoria"], descending=[True, False]).head(7)["categoria"].to_list()
        )
        base = np.zeros(len(anios))
        for k, cat in enumerate(top + ["(otras)"]):
            if cat == "(otras)":
                vals = []
                for a in anios:
                    s = sub.filter(pl.col("anio") == a)
                    resto = s.filter(~pl.col("categoria").is_in(top))["prop"].drop_nulls().sum()
                    vals.append(float(resto) * 100)
            else:
                vals = []
                for a in anios:
                    s = sub.filter((pl.col("anio") == a) & (pl.col("categoria") == cat))
                    v = s["prop"][0] if s.height else 0.0
                    vals.append(0.0 if v is None else float(v) * 100)
            ax.bar([str(a) for a in anios], vals, bottom=base, label=_acortar(cat), color=graficos.OKABE_ITO[k % 8], width=0.75)
            base += np.array(vals)
        ax.set_title(col, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% de los valores informados")
        ax.legend(fontsize=5.5, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=False)
        ax.grid(axis="x", visible=False)
    ctx.figuras.guardar_figura(
        fig, id_figura="F04_02", tarea=4,
        titulo="Deriva de seis variables categóricas por año",
        descripcion_alt="Barras apiladas con la distribución porcentual de actividad, etnia, previsión, procedencia, tipo de alta e ingreso en cada año.",
        csv="t22_deriva_categorias.csv", anios=ANIOS_TXT, decision="4; 15",
        extra_pie="Tipo de alta solo hasta 2023.",
    )


def figura_edad_estancia(ctx: Contexto) -> None:
    e = ctx.leer_tabla("t22_hist_edad.csv")
    h = ctx.leer_tabla("t22_hist_estancia.csv", columnas_texto=["tramo"])
    fig, (ax1, ax2) = graficos.nueva_figura_multiple(1, 2, figsize=(11.0, 4.8))
    for k, a in enumerate(sorted(e["anio"].unique().to_list())):
        s = e.filter(pl.col("anio") == a).sort("edad")
        total = e.filter(pl.col("anio") == a)["n"].sum()
        ax1.plot(s["edad"].to_list(), [(v or 0) * 100.0 / total for v in s["n"].to_list()], label=str(a), color=graficos.OKABE_ITO[k % 8], lw=1.2)
    ax1.set_xlabel("Edad al ingreso (años)")
    ax1.set_ylabel("% de los egresos del año")
    ax1.set_title("Edad", fontsize=9)
    ax1.legend(ncol=3, fontsize=7)
    for k, a in enumerate(sorted(h["anio"].unique().to_list())):
        s = h.filter((pl.col("anio") == a) & (pl.col("orden") >= 0) & (pl.col("orden") <= 30)).sort("orden")
        total = h.filter(pl.col("anio") == a)["n"].sum()
        ax2.plot(s["orden"].to_list(), [(v or 0) * 100.0 / total for v in s["n"].to_list()], label=str(a), color=graficos.OKABE_ITO[k % 8], lw=1.2)
    ax2.set_xlabel("Estadía (días, hasta 30)")
    ax2.set_ylabel("% de los egresos del año")
    ax2.set_title("Estadía (2019 a 2023)", fontsize=9)
    ax2.legend(ncol=3, fontsize=7)
    ctx.figuras.guardar_figura(
        fig, id_figura="F04_03", tarea=4,
        titulo="Distribución de la edad al ingreso y de la estadía",
        descripcion_alt="Dos paneles con líneas por año: porcentaje de egresos por edad y porcentaje de egresos por días de estadía, construidos desde conteos.",
        csv="t22_hist_edad.csv", anios=ANIOS_TXT, decision="5; 7",
        extra_pie="La estadía de 2024 no se analiza (cortafuegos).",
    )


# ------------------------------------------------------------------------------------ informe
def construir_informe(ctx: Contexto) -> Path:
    inf = Informe(ctx, "t22_informe.md", "Informe de descripción de los datos (CRISP-DM 2.2)")
    rs = lambda col: cifra(ctx, "t22_resumen.csv", col)  # noqa: E731
    inf.seccion("1. Alcance y método")
    inf.parrafo(
        "Se perfilaron las {n_cols} columnas de los archivos anuales: nulos, valores centinela, cardinalidad con el estimador Chao1, "
        "tipo inferido, rangos robustos y patrones de caracteres. Los conteos con menos de diez casos se suprimen. "
        "Las columnas de resultado de 2024 se entregan enmascaradas y solo informan nulos y patrón.",
        n_cols=rs("n_columnas_perfiladas"),
    )
    inf.seccion("2. Faltantes y valores centinela")
    inf.parrafo(
        "{n_vacias} columnas están vacías en todos los años y {n_cent} columnas tienen valores centinela (por ejemplo DESCONOCIDO) "
        "que el diccionario no cuenta como faltantes.",
        n_vacias=rs("n_columnas_vacias"),
        n_cent=rs("n_columnas_con_centinelas"),
    )
    inf.figura("F04_01")
    inf.seccion("3. Cardinalidad observada y estimada")
    inf.parrafo(
        "En un extracto la cardinalidad observada subestima la de los datos completos; el estimador Chao1 corrige parte de ese sesgo "
        "si el muestreo fuera aleatorio, algo que los extractos no garantizan. La tabla compara con las cifras del diccionario."
    )
    inf.tabla(
        "t22_cardinalidades.csv",
        df=ctx.leer_tabla("t22_cardinalidades.csv", columnas_texto=["periodo"]).filter(pl.col("periodo").str.contains("-")).sort(["columna", "periodo"]),
        columnas=["columna", "periodo", "n_observado", "chao1", "documentado"],
        encabezados={"columna": "Columna", "periodo": "Período", "n_observado": "Observado", "chao1": "Chao1", "documentado": "Documentado"},
        formatos={"chao1": "entero", "documentado": "entero"},
        leyenda="Cardinalidad acumulada por período",
    )
    inf.seccion("4. Deriva de categorías")
    inf.parrafo(
        "Las seis variables categóricas cambian entre años. El PSI máximo entre años consecutivos o contra el primer año fue {psi_max|dec2}; "
        "los valores de {moderado|dec2} o más indican cambio moderado y los de {alto|dec2} o más, cambio grande.",
        psi_max=rs("psi_maximo"),
        moderado=Cifra_umbral(ctx, "psi_moderado"),
        alto=Cifra_umbral(ctx, "psi_alto"),
    )
    inf.figura("F04_02")
    inf.tabla(
        "t22_psi.csv",
        df=ctx.leer_tabla("t22_psi.csv").sort("psi", descending=True),
        max_filas=15,
        encabezados={"columna": "Columna", "anio_base": "Año base", "anio_comparado": "Año comparado", "psi": "PSI",
                     "clasificacion": "Clasificación", "n_categorias": "Categorías"},
        formatos={"psi": "dec3"},
        leyenda="Mayores PSI entre años",
    )
    inf.seccion("5. Edad y estadía")
    inf.figura("F04_03")
    inf.seccion("6. Conciliación con el diccionario")
    inf.parrafo(
        "La conciliación compara, columna por columna, los registros sin dato del diccionario con lo medido ({modo|texto}). "
        "Resultado: {n_coincide} columnas coinciden, {n_disc} discrepan y {n_ne} no son evaluables por tratarse de eventos raros. "
        "De las que discrepan, {n_rel} difieren en {dif_min|dec1} puntos porcentuales o más (magnitud relevante); "
        "en un extracto no aleatorio, las demás son diferencias despreciables aunque superen el umbral de z.",
        modo=rs("modo_conciliacion"),
        n_coincide=rs("n_coincide"),
        n_disc=rs("n_discrepancia"),
        n_ne=rs("n_no_evaluable"),
        n_rel=rs("n_discrepancia_relevante"),
        dif_min=Cifra_umbral(ctx, "conciliacion_dif_relevante_pp"),
    )
    conc = ctx.leer_tabla("t22_conciliacion.csv", columnas_texto=["estado", "magnitud"]).filter(
        (pl.col("estado") == "DISCREPANCIA") & (pl.col("magnitud") == "relevante")
    ).sort("dif_pp", descending=True)
    inf.tabla(
        "t22_conciliacion.csv",
        df=conc,
        columnas=["columna", "pct_sin_doc", "pct_sin_med", "dif_pp", "z", "observacion"],
        encabezados={"columna": "Columna", "pct_sin_doc": "% sin dato (diccionario)", "pct_sin_med": "% sin dato (medido)",
                     "dif_pp": "Diferencia (pp)", "z": "z", "observacion": "Observación"},
        formatos={"pct_sin_doc": "pct2", "pct_sin_med": "pct2", "dif_pp": "dec2", "z": "dec1"},
        max_filas=40,
        leyenda="Columnas con discrepancia relevante",
    )
    inf.seccion("7. Bloque de recién nacidos y pares de traslados")
    inf.parrafo(
        "En RN2ESTADO, RN3ESTADO y RN4ESTADO, el valor más frecuente concentra al menos {rn_min|prop1} de los valores informados, lo que indica relleno constante "
        "más que información clínica.",
        rn_min=rs("rn_prop_modal_minima"),
    )
    inf.tabla("t22_bloque_rn.csv", encabezados={"columna": "Columna", "n_informados": "Informados", "n_distintos": "Distintos", "valor_modal": "Valor modal", "n_modal": "N modal", "prop_modal": "Proporción modal"},
              formatos={"prop_modal": "prop1"}, columnas_texto=["valor_modal"], leyenda="Bloque de recién nacidos")
    inf.tabla("t22_pares_traslado.csv", encabezados={"traslado": "Traslado", "n_fecha": "Con fecha", "n_servicio": "Con servicio", "n_solo_fecha": "Solo fecha", "n_solo_servicio": "Solo servicio", "diferencia": "Diferencia"},
              formatos={"traslado": "texto"}, leyenda="Pares fecha-servicio de traslados internos")
    inf.seccion("8. Etiquetas truncadas")
    inf.parrafo(
        "Se detectaron {n_trunc} etiquetas categóricas que parecen cortadas por un límite de largo (paréntesis sin cerrar). "
        "Una unión exacta con los roles de tipo de alta pierde esos casos.",
        n_trunc=rs("n_etiquetas_truncadas"),
    )
    inf.tabla("t22_etiquetas_truncadas.csv", encabezados={"columna": "Columna", "etiqueta": "Etiqueta", "longitud": "Largo", "n": "N"},
              columnas_texto=["etiqueta"], leyenda="Etiquetas que parecen cortadas")
    inf.seccion("9. Límites")
    inf.texto_libre_sin_cifras(
        "- Los extractos no son una muestra aleatoria simple: las comparaciones de proporciones son indicativas.\n"
        "- Los intervalos ignoran el agrupamiento por paciente y por hospital.\n"
        "- La estadía de 2024 no se analiza por el cortafuegos."
    )
    return inf.guardar()


def Cifra_umbral(ctx: Contexto, nombre: str):
    """Cifra que proviene de umbrales.toml (registrada en una tabla de umbrales)."""
    from .informes import Cifra
    import json as _json

    if "umbrales.csv" not in ctx._tablas:
        ctx.escribir_tabla("umbrales.csv", ctx.umbrales.a_tabla(), tarea="T01", descripcion="Umbrales congelados del protocolo")
    return Cifra(nombre, float(ctx.umbrales[nombre]), "umbrales.csv", "valor", _json.dumps({"umbral": nombre}))
