"""CRISP-DM 2.4: calidad de los datos, lote 1 (fechas, identificador del paciente y duplicados).

Las reglas se declaran en ``config/reglas_calidad.toml``. Cada regla devuelve solo conteos de violaciones
(nunca valores individuales) y, si lo declara ``permite_expuesto``, puede leer las columnas de resultado de 2024
sin entregar nunca una distribución de resultado.

Tablas:
- ``t24_reglas.csv``: violaciones por regla, año (o régimen de IDs) y severidad.
- ``t24_id_largo.csv``: distribución del largo del identificador por año.
- ``t24_id_traslape.csv``: traslape de identificadores entre pares de años.
- ``t24_id_regimenes.csv``: regímenes de identificadores (años que comparten espacio de IDs).
- ``t24_id_consistencia.csv``: identificadores con más de una fecha de nacimiento o sexo, por régimen.
- ``t24_duplicados.csv``: filas duplicadas y episodios solapados por año.
- ``t24_rut_por_columna.csv``: conteo de valores con estructura de RUT por columna y año.
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from . import config, esquema, graficos, rutas
from . import transformaciones as tr
from .contexto import Contexto
from .informes import Informe, cifra

TAREA = "T05"
ANIOS_TXT = "2019 a 2024"
SEVERIDADES = ("alta", "media", "baja")
ACCIONES = ("anular", "recodificar", "excluir", "informar_fonasa", "informar")
ALCANCES = ("anio", "regimen")
COLUMNAS_BASE = (
    "CIP_ENCRIPTADO",
    "FECHA_NACIMIENTO",
    "SEXO",
    "FECHA_INGRESO",
    "FECHAALTA",
    "COD_HOSPITAL",
    "DIAGNOSTICO1",
)
RE_RUT_POLARS = r"^\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]$"


# ------------------------------------------------------------------------------ reglas
@dataclass(frozen=True)
class Regla:
    id: str
    nombre: str
    tipo: str
    columnas: tuple[str, ...]
    severidad: str
    accion: str
    permite_expuesto: bool
    alcance: str
    referencia: str
    descripcion: str
    umbral: str = ""


GRUPOS_COLUMNAS = {"DIAG": esquema.DIAG, "DIAG_SEC": esquema.DIAG_SEC, "PROC": esquema.PROC}
# Tipos de regla del lote 2 (catálogos y consistencia cruzada); se evalúan en eda/calidad_catalogos.py.
TIPOS_LOTE2 = (
    "cie10_principal_invalido",
    "cie9_invalido",
    "grd_fuera_catalogo",
    "hospital_fuera_catalogo",
    "fallecido_riesgo_grd_cero",
    "sexo_dx_incompatible",
    "mdc15_edad_mayor",
    "mdc14_hombre",
    "grd_ambulatorio_en_hospitalizacion",
    "grd_hospitalizado_en_ambulatoria",
    "tipoalta_sin_rol",
    "usospabellon_invalido",
    "severidad_mortalidad_invalida",
)


def cargar_reglas(ruta: Path | str | None = None) -> list[Regla]:
    """Todas las reglas de ``config/reglas_calidad.toml`` (lotes 1 y 2)."""
    ruta = Path(ruta) if ruta else rutas.DIR_CONFIG / "reglas_calidad.toml"
    with open(ruta, "rb") as f:
        datos = tomllib.load(f)
    return [
        Regla(
            id=r["id"],
            nombre=r["nombre"],
            tipo=r["tipo"],
            columnas=tuple(r.get("columnas", [])) + tuple(c for g in r.get("columnas_grupo", []) for c in GRUPOS_COLUMNAS[g]),
            severidad=r["severidad"],
            accion=r["accion"],
            permite_expuesto=bool(r.get("permite_expuesto", False)),
            alcance=r.get("alcance", "anio"),
            referencia=r.get("referencia", ""),
            descripcion=r.get("descripcion", ""),
            umbral=r.get("umbral", ""),
        )
        for r in datos.get("regla", [])
    ]


def reglas_lote1(ruta: Path | str | None = None) -> list[Regla]:
    return [r for r in cargar_reglas(ruta) if r.tipo not in TIPOS_LOTE2]


def reglas_lote2(ruta: Path | str | None = None) -> list[Regla]:
    return [r for r in cargar_reglas(ruta) if r.tipo in TIPOS_LOTE2]


def ids_autorizados_expuesto(reglas: list[Regla]) -> list[str]:
    """Reglas que pueden leer columnas de resultado del año expuesto (solo devuelven violaciones)."""
    return [r.id for r in reglas if r.permite_expuesto]


def validar_reglas(reglas: list[Regla]) -> list[str]:
    problemas: list[str] = []
    ids = [r.id for r in reglas]
    if len(ids) != len(set(ids)):
        problemas.append("hay ids de regla repetidos")
    for r in reglas:
        if r.tipo not in TIPOS and r.tipo not in TIPOS_ESPECIALES and r.tipo not in TIPOS_LOTE2:
            problemas.append(f"{r.id}: tipo desconocido {r.tipo!r}")
        if r.severidad not in SEVERIDADES:
            problemas.append(f"{r.id}: severidad inválida {r.severidad!r}")
        if r.accion not in ACCIONES:
            problemas.append(f"{r.id}: acción inválida {r.accion!r}")
        if r.alcance not in ALCANCES:
            problemas.append(f"{r.id}: alcance inválido {r.alcance!r}")
        if not r.descripcion.strip():
            problemas.append(f"{r.id}: sin descripción")
        if r.permite_expuesto and r.accion == "":
            problemas.append(f"{r.id}: acción vacía")
    return problemas


# ---------------------------------------------------------------------------- preparación
def preparar(df: pl.DataFrame) -> pl.DataFrame:
    """Agrega columnas tipadas (prefijo ``_``) que usan las reglas. No modifica las columnas originales."""
    cols = set(df.columns)
    e = []
    if "FECHA_NACIMIENTO" in cols:
        e.append(tr.a_fecha(pl.col("FECHA_NACIMIENTO")).alias("_nac"))
    if "FECHA_INGRESO" in cols:
        e.append(tr.a_fecha(pl.col("FECHA_INGRESO")).alias("_ing"))
    if "FECHAALTA" in cols:
        e.append(tr.a_fecha(pl.col("FECHAALTA")).alias("_alta"))
    if e:
        df = df.with_columns(e)
    if {"_nac", "_ing"} <= set(df.columns):
        df = df.with_columns(tr.edad_anios(pl.col("_nac"), pl.col("_ing")).alias("_edad"))
    if {"_ing", "_alta"} <= set(df.columns):
        df = df.with_columns(tr.estancia_dias(pl.col("_ing"), pl.col("_alta")).alias("_est"))
    if "CIP_ENCRIPTADO" in cols:
        idc = pl.col("CIP_ENCRIPTADO")
        df = df.with_columns(
            pl.when(idc.is_null())
            .then(pl.lit("vacio"))
            .when(esquema.es_centinela(idc))
            .then(pl.lit("centinela"))
            .when(idc.str.strip_chars().str.contains(r"^\d+$"))
            .then(pl.lit("numerico"))
            .otherwise(pl.lit("no_numerico"))
            .alias("_id_clase"),
            idc.str.strip_chars().str.len_chars().alias("_id_largo"),
        )
    return df


# --------------------------------------------------------------- tipos de regla por año
Evaluacion = tuple[int, int]  # (violaciones, evaluados)


def _t_vacio(df: pl.DataFrame, r: Regla, u, anio: int) -> Evaluacion:
    col = r.columnas[0]
    return int(df[col].null_count()), df.height


def _t_fecha_no_convertible(df: pl.DataFrame, r: Regla, u, anio: int) -> Evaluacion:
    viol = ev = 0
    for col in r.columnas:
        raw = pl.col(col)
        parseada = tr.a_fecha(raw)
        informado = raw.is_not_null() & ~esquema.es_centinela(raw)
        res = df.select(
            (informado & parseada.is_null()).sum().alias("v"),
            informado.sum().alias("e"),
        ).row(0)
        viol += int(res[0])
        ev += int(res[1])
    return viol, ev


def _t_alta_antes_ingreso(df: pl.DataFrame, r: Regla, u, anio: int) -> Evaluacion:
    ok = pl.col("_ing").is_not_null() & pl.col("_alta").is_not_null()
    res = df.select(((pl.col("_alta") < pl.col("_ing")) & ok).sum().alias("v"), ok.sum().alias("e")).row(0)
    return int(res[0]), int(res[1])


def _t_estancia_maxima(df: pl.DataFrame, r: Regla, u, anio: int) -> Evaluacion:
    res = df.select((pl.col("_est") > u[r.umbral]).fill_null(False).sum().alias("v"), pl.col("_est").is_not_null().sum().alias("e")).row(0)
    return int(res[0]), int(res[1])


def _t_edad_negativa(df: pl.DataFrame, r: Regla, u, anio: int) -> Evaluacion:
    res = df.select((pl.col("_edad") < u[r.umbral]).fill_null(False).sum().alias("v"), pl.col("_edad").is_not_null().sum().alias("e")).row(0)
    return int(res[0]), int(res[1])


def _t_edad_mayor(df: pl.DataFrame, r: Regla, u, anio: int) -> Evaluacion:
    res = df.select((pl.col("_edad") > u[r.umbral]).fill_null(False).sum().alias("v"), pl.col("_edad").is_not_null().sum().alias("e")).row(0)
    return int(res[0]), int(res[1])


def _t_id_clase(clase: str) -> Callable[..., Evaluacion]:
    def f(df: pl.DataFrame, r: Regla, u, anio: int) -> Evaluacion:
        return int((df["_id_clase"] == clase).sum()), df.height

    return f


def _t_id_corto(df: pl.DataFrame, r: Regla, u, anio: int) -> Evaluacion:
    num = pl.col("_id_clase") == "numerico"
    res = df.select((num & (pl.col("_id_largo") < u[r.umbral])).sum().alias("v"), num.sum().alias("e")).row(0)
    return int(res[0]), int(res[1])


def _t_duplicado_clave(df: pl.DataFrame, r: Regla, u, anio: int) -> Evaluacion:
    sub = df.filter(pl.col("_id_clase") == "numerico").select(list(r.columnas))
    return int(sub.height - sub.unique().height), sub.height


def _t_anio_alta_distinto(df: pl.DataFrame, r: Regla, u, anio: int) -> Evaluacion:
    ok = pl.col("_alta").is_not_null()
    res = df.select(((pl.col("_alta").dt.year() != anio) & ok).sum().alias("v"), ok.sum().alias("e")).row(0)
    return int(res[0]), int(res[1])


def _t_anio_ingreso_distinto(df: pl.DataFrame, r: Regla, u, anio: int) -> Evaluacion:
    ok = pl.col("_ing").is_not_null()
    res = df.select(((pl.col("_ing").dt.year() != anio) & ok).sum().alias("v"), ok.sum().alias("e")).row(0)
    return int(res[0]), int(res[1])


TIPOS: dict[str, Callable[..., Evaluacion]] = {
    "vacio": _t_vacio,
    "fecha_no_convertible": _t_fecha_no_convertible,
    "alta_antes_ingreso": _t_alta_antes_ingreso,
    "estancia_maxima": _t_estancia_maxima,
    "edad_negativa": _t_edad_negativa,
    "edad_mayor": _t_edad_mayor,
    "id_vacio": _t_id_clase("vacio"),
    "id_centinela": _t_id_clase("centinela"),
    "id_no_numerico": _t_id_clase("no_numerico"),
    "id_corto": _t_id_corto,
    "duplicado_clave": _t_duplicado_clave,
    "anio_alta_distinto": _t_anio_alta_distinto,
    "anio_ingreso_distinto": _t_anio_ingreso_distinto,
}
# Reglas que no se evalúan con la preparación de un solo año.
TIPOS_ESPECIALES = ("id_un_nacimiento", "id_un_sexo", "solape_episodios", "rut_en_columna")


def evaluar_regla(df_prep: pl.DataFrame, r: Regla, u, anio: int) -> Evaluacion:
    return TIPOS[r.tipo](df_prep, r, u, anio)


def regla_permitida(r: Regla, anio: int, cf) -> bool:
    """¿Se puede evaluar esta regla en este año según el cortafuegos?"""
    if anio != cf.anio_expuesto:
        return True
    if set(r.columnas) & cf.columnas_resultado:
        return r.id in cf.reglas_autorizadas
    return True


# ------------------------------------------------------------- régimen de identificadores
def traslape_ids(ids_por_anio: dict[int, pl.Series]) -> pl.DataFrame:
    """Traslape de identificadores distintos entre todos los pares de años."""
    anios = sorted(ids_por_anio)
    filas = []
    for i, a in enumerate(anios):
        for b in anios[i + 1 :]:
            sa, sb = ids_por_anio[a], ids_por_anio[b]
            inter = int(sa.is_in(sb).sum())
            menor = min(sa.len(), sb.len())
            filas.append(
                {
                    "anio_a": a,
                    "anio_b": b,
                    "consecutivo": "si" if b == a + 1 else "no",
                    "n_ids_a": sa.len(),
                    "n_ids_b": sb.len(),
                    "n_interseccion": inter,
                    "prop_min": inter / menor if menor else None,
                    "prop_a": inter / sa.len() if sa.len() else None,
                    "prop_b": inter / sb.len() if sb.len() else None,
                }
            )
    return pl.DataFrame(
        filas,
        schema={
            "anio_a": pl.Int64, "anio_b": pl.Int64, "consecutivo": pl.String, "n_ids_a": pl.Int64, "n_ids_b": pl.Int64,
            "n_interseccion": pl.Int64, "prop_min": pl.Float64, "prop_a": pl.Float64, "prop_b": pl.Float64,
        },
    )


def detectar_quiebres(consecutivos: dict[tuple[int, int], float], ratio: float) -> list[tuple[int, int]]:
    """Pares consecutivos cuyo traslape es menor a ``ratio`` veces la mediana del traslape de los demás pares."""
    quiebres = []
    for par, v in sorted(consecutivos.items()):
        otros = [x for p, x in consecutivos.items() if p != par and x is not None]
        if not otros or v is None:
            continue
        med = float(np.median(otros))
        if med > 0 and v < ratio * med:
            quiebres.append(par)
    return quiebres


def regimenes(anios: list[int], quiebres: list[tuple[int, int]]) -> list[list[int]]:
    """Agrupa años contiguos que comparten espacio de IDs; un quiebre separa dos regímenes."""
    cortes = {b for _, b in quiebres}
    grupos: list[list[int]] = []
    for a in sorted(anios):
        if not grupos or a in cortes:
            grupos.append([a])
        else:
            grupos[-1].append(a)
    return grupos


def etiqueta_regimen(anios: list[int]) -> str:
    return f"{anios[0]}" if len(anios) == 1 else f"{anios[0]} a {anios[-1]}"


# ------------------------------------------------------- reglas con alcance de régimen
def _consistencia_id(frames: pl.DataFrame) -> tuple[int, int, int, int]:
    """(ids con >=2 episodios, con más de un nacimiento, con más de un sexo, ids con sexo conocido en >=2 episodios)."""
    sexo_ok = pl.col("sexo").is_in(["MUJER", "HOMBRE"])
    g = frames.group_by("id").agg(
        pl.len().alias("n_ep"),
        pl.col("nac").drop_nulls().n_unique().alias("n_nac"),
        pl.col("sexo").filter(sexo_ok).n_unique().alias("n_sexo"),
        sexo_ok.sum().alias("n_sexo_ok"),
    )
    multi = g.filter(pl.col("n_ep") >= 2)
    multi_sexo = g.filter(pl.col("n_sexo_ok") >= 2)
    return (
        multi.height,
        int(multi.filter(pl.col("n_nac") > 1).height),
        int(multi_sexo.filter(pl.col("n_sexo") > 1).height),
        multi_sexo.height,
    )


def solapes(frames: pl.DataFrame) -> pl.DataFrame:
    """Episodios cuyo ingreso es anterior al alta de un episodio previo del mismo ID (duplicados exactos excluidos).

    Devuelve por año del episodio posterior: (anio, n_solapes, n_evaluados, n_duplicados_exactos).
    """
    f = frames.filter(pl.col("ing").is_not_null() & pl.col("alta").is_not_null()).sort(["id", "ing", "alta"])
    if f.height == 0:
        return pl.DataFrame(schema={"anio": pl.Int64, "n_solapes": pl.Int64, "n_evaluados": pl.Int64, "n_duplicados_exactos": pl.Int64})
    # Las filas ya vienen ordenadas por id: «el episodio anterior del mismo ID» es la fila anterior si comparte id. Así se evitan las ventanas anidadas
    # (``.over("id")`` dentro de ``.over("id")``), que con millones de IDs distintos crecen de forma casi cuadrática. Mismo resultado, en tiempo lineal.
    mismo_id = (pl.col("id") == pl.col("id").shift(1)).fill_null(False)
    f = f.with_columns(pl.col("alta").cum_max().over("id").alias("_cum_max_alta"))
    f = f.with_columns(
        pl.when(mismo_id).then(pl.col("_cum_max_alta").shift(1)).otherwise(None).alias("_prev_max_alta"),
        (mismo_id & (pl.col("ing") == pl.col("ing").shift(1)) & (pl.col("alta") == pl.col("alta").shift(1))).fill_null(False).alias("_dup"),
    )
    f = f.with_columns(((pl.col("ing") < pl.col("_prev_max_alta")) & ~pl.col("_dup")).fill_null(False).alias("_solapa"))
    return (
        f.group_by("anio")
        .agg(pl.col("_solapa").sum().alias("n_solapes"), pl.len().alias("n_evaluados"), pl.col("_dup").sum().alias("n_duplicados_exactos"))
        .sort("anio")
    )


# ----------------------------------------------------------------------------- ejecución
def _fila_regla(r: Regla, ambito: str, viol: int | None, ev: int | None, estado: str = "EVALUADA", nota: str = "") -> dict[str, Any]:
    return {
        "regla_id": r.id,
        "nombre": r.nombre,
        "severidad": r.severidad,
        "accion": r.accion,
        "ambito": ambito,
        "n_violaciones": viol,
        "n_evaluados": ev,
        "prop_violaciones": (viol / ev) if (viol is not None and ev) else None,
        "estado": estado,
        "nota": nota,
    }


ESQUEMA_REGLAS = {
    "regla_id": pl.String, "nombre": pl.String, "severidad": pl.String, "accion": pl.String, "ambito": pl.String,
    "n_violaciones": pl.Int64, "n_evaluados": pl.Int64, "prop_violaciones": pl.Float64, "estado": pl.String, "nota": pl.String,
}


def escanear_rut(ctx: Contexto, regla: Regla) -> tuple[pl.DataFrame, dict[int, tuple[int, int]]]:
    """Cuenta, por columna y año, valores con estructura de RUT. Nunca guarda el valor."""
    filas = []
    por_anio: dict[int, tuple[int, int]] = {}
    for anio in ctx.fuente.anios():
        usa_regla = regla_permitida(regla, anio, ctx.cortafuegos)
        reglas_uso = [regla.id] if usa_regla else []
        if ctx.lector.cache is not None:
            # base completa: las 129 columnas por bloques, para no tenerlas todas en memoria a la vez
            bloques = ctx.lector.leer_por_bloques(anio, None, proposito="calidad", tarea=TAREA, reglas=reglas_uso)
        else:
            bloques = iter([ctx.lector.leer(anio, None, proposito="calidad", tarea=TAREA, reglas=reglas_uso)])
        total = 0
        evaluados = 0
        for df in bloques:
            for col in df.columns:
                s = df[col]
                evaluados += s.len() - s.null_count()
                n = int(df.select(pl.col(col).str.contains(RE_RUT_POLARS).sum()).item() or 0)
                if n:
                    filas.append({"columna": col, "anio": anio, "n": n})
                    total += n
            del df
        por_anio[anio] = (total, evaluados)
    t = pl.DataFrame(filas, schema={"columna": pl.String, "anio": pl.Int64, "n": pl.Int64}).sort(["columna", "anio"])
    return t, por_anio


def calcular(ctx: Contexto) -> dict[str, Any]:
    """Evalúa todas las reglas y construye las tablas del lote 1 (sin suprimir)."""
    u = ctx.umbrales
    cf = ctx.cortafuegos
    reglas = reglas_lote1()
    problemas = validar_reglas(cargar_reglas())
    if problemas:
        raise ValueError(f"reglas inválidas: {problemas}")
    por_anio = [r for r in reglas if r.alcance == "anio" and r.tipo in TIPOS]
    anios = ctx.fuente.anios()

    filas_reglas: list[dict] = []
    filas_largo: list[dict] = []
    almacen: dict[int, pl.DataFrame] = {}
    ids_por_anio: dict[int, pl.Series] = {}
    for anio in anios:
        permitidas = [r for r in por_anio if regla_permitida(r, anio, cf)]
        con_resultado = [r.id for r in permitidas if set(r.columnas) & cf.columnas_resultado]
        df = ctx.lector.leer(anio, list(COLUMNAS_BASE), proposito="calidad", tarea=TAREA, reglas=con_resultado)
        dfp = preparar(df)
        for r in por_anio:
            if r in permitidas:
                v, e = evaluar_regla(dfp, r, u, anio)
                filas_reglas.append(_fila_regla(r, str(anio), v, e))
            else:
                filas_reglas.append(_fila_regla(r, str(anio), None, None, "BLOQUEADA_2024", "el cortafuegos no autoriza la regla"))
        # distribución del largo del identificador
        total = dfp.height
        cl = dict(dfp.group_by("_id_clase").len().iter_rows())
        largos = (
            dfp.filter(pl.col("_id_clase") == "numerico").group_by("_id_largo").len().sort("_id_largo")
        )
        for largo, n in largos.iter_rows():
            filas_largo.append({"anio": anio, "categoria": str(largo), "orden": int(largo), "n": int(n), "prop": n / total})
        for clase, orden in (("vacio", 100), ("centinela", 101), ("no_numerico", 102)):
            filas_largo.append({"anio": anio, "categoria": clase, "orden": orden, "n": int(cl.get(clase, 0)), "prop": cl.get(clase, 0) / total})
        validos = dfp.filter(pl.col("_id_clase") == "numerico")
        ids_por_anio[anio] = validos["CIP_ENCRIPTADO"].unique()
        almacen[anio] = validos.select(
            pl.col("CIP_ENCRIPTADO").alias("id"), pl.col("_nac").alias("nac"), pl.col("SEXO").alias("sexo"),
            pl.col("_ing").alias("ing"), pl.col("_alta").alias("alta"), pl.lit(anio, dtype=pl.Int64).alias("anio"),
            pl.col("_id_largo").alias("id_largo"),
        )
        del df, dfp

    # --- régimen de identificadores ---
    tras = traslape_ids(ids_por_anio)
    consecutivos = {
        (r["anio_a"], r["anio_b"]): r["prop_min"] for r in tras.filter(pl.col("consecutivo") == "si").iter_rows(named=True)
    }
    quiebres = detectar_quiebres(consecutivos, float(u["id_quiebre_ratio"]))
    grupos = regimenes(anios, quiebres)
    filas_reg = []
    for k, g in enumerate(grupos, start=1):
        filas_reg.append({"regimen": k, "anio_inicio": g[0], "anio_fin": g[-1], "anios": ", ".join(str(a) for a in g), "etiqueta": etiqueta_regimen(g)})
    reg_df = pl.DataFrame(filas_reg, schema={"regimen": pl.Int64, "anio_inicio": pl.Int64, "anio_fin": pl.Int64, "anios": pl.String, "etiqueta": pl.String})

    # --- reglas con alcance de régimen ---
    por_regimen = {r.tipo: r for r in reglas if r.alcance == "regimen"}
    filas_cons: list[dict] = []
    filas_dup: list[dict] = []
    for g in grupos:
        frames = pl.concat([almacen[a] for a in g])
        etiqueta = f"régimen {etiqueta_regimen(g)}"
        minimo = int(u["id_largo_minimo"])
        # los IDs muy cortos pueden ser colisiones entre pacientes distintos: se reportan por separado
        grupos_id = (
            ("todos los IDs", frames),
            (f"IDs con {minimo} o más dígitos", frames.filter(pl.col("id_largo") >= minimo)),
            (f"IDs con menos de {minimo} dígitos", frames.filter(pl.col("id_largo") < minimo)),
        )
        for nombre_grupo, sub in grupos_id:
            n_multi_g, n_nac_g, n_sexo_g, n_multi_sexo_g = _consistencia_id(sub)
            filas_cons.append(
                {"regimen": etiqueta, "grupo_id": nombre_grupo, "n_ids_con_dos_o_mas_episodios": n_multi_g,
                 "n_ids_nacimiento_inconsistente": n_nac_g, "n_ids_con_sexo_en_dos_o_mas_episodios": n_multi_sexo_g,
                 "n_ids_sexo_inconsistente": n_sexo_g}
            )
        n_multi, n_nac, n_sexo, n_multi_sexo = _consistencia_id(frames)
        if "id_un_nacimiento" in por_regimen:
            filas_reglas.append(_fila_regla(por_regimen["id_un_nacimiento"], etiqueta, n_nac, n_multi))
        if "id_un_sexo" in por_regimen:
            filas_reglas.append(_fila_regla(por_regimen["id_un_sexo"], etiqueta, n_sexo, n_multi_sexo))
        s = solapes(frames)
        for row in s.iter_rows(named=True):
            filas_dup.append({"anio": row["anio"], "n_solapes": row["n_solapes"], "n_evaluados_solape": row["n_evaluados"],
                              "n_duplicados_exactos_por_id_y_fechas": row["n_duplicados_exactos"]})
            if "solape_episodios" in por_regimen:
                filas_reglas.append(_fila_regla(por_regimen["solape_episodios"], str(row["anio"]), row["n_solapes"], row["n_evaluados"]))
    cons = pl.DataFrame(filas_cons)

    # duplicados por clave natural (regla de alcance anual) junto con los solapes (regla de régimen)
    id_dup = next((r.id for r in reglas if r.tipo == "duplicado_clave"), "")
    esq_solape = {"anio": pl.Int64, "n_solapes": pl.Int64, "n_evaluados_solape": pl.Int64, "n_duplicados_exactos_por_id_y_fechas": pl.Int64}
    dup_df = pl.DataFrame(filas_dup, schema=esq_solape)
    dup_clave = pl.DataFrame(
        [
            {"anio": int(f["ambito"]), "n_filas_duplicadas_clave": f["n_violaciones"], "n_filas_con_id_valido": f["n_evaluados"]}
            for f in filas_reglas
            if f["regla_id"] == id_dup and f["ambito"].isdigit()
        ],
        schema={"anio": pl.Int64, "n_filas_duplicadas_clave": pl.Int64, "n_filas_con_id_valido": pl.Int64},
    )
    dup_all = dup_clave.join(dup_df, on="anio", how="full", coalesce=True).sort("anio")

    # --- RUT ---
    rut_regla = next((r for r in reglas if r.tipo == "rut_en_columna"), None)
    rut_df = pl.DataFrame(schema={"columna": pl.String, "anio": pl.Int64, "n": pl.Int64})
    if rut_regla is not None:
        rut_df, rut_anio = escanear_rut(ctx, rut_regla)
        for anio, (v, e) in sorted(rut_anio.items()):
            filas_reglas.append(_fila_regla(rut_regla, str(anio), v, e))

    reglas_df = pl.DataFrame(filas_reglas, schema=ESQUEMA_REGLAS)
    # totales por regla de alcance anual (solo años evaluados)
    tot = []
    ids_anuales = {r.id for r in reglas if r.alcance == "anio" or r.tipo == "solape_episodios"}
    for rid, sub in reglas_df.filter(pl.col("ambito").str.contains(r"^\d{4}$")).group_by("regla_id", maintain_order=True):
        rid = rid[0] if isinstance(rid, tuple) else rid
        if rid not in ids_anuales:
            continue
        ok = sub.filter(pl.col("estado") == "EVALUADA")
        r = next(x for x in reglas if x.id == rid)
        tot.append(
            _fila_regla(r, "total", int(ok["n_violaciones"].sum()), int(ok["n_evaluados"].sum()),
                        nota=f"suma de {ok.height} años evaluados" + ("" if ok.height == sub.height else "; faltan años bloqueados"))
        )
    reglas_df = pl.concat([reglas_df, pl.DataFrame(tot, schema=ESQUEMA_REGLAS)]).sort(["regla_id", "ambito"])
    largo_df = pl.DataFrame(filas_largo).sort(["anio", "orden"])
    return {
        "reglas": reglas_df, "largo": largo_df, "traslape": tras, "regimenes": reg_df, "consistencia": cons,
        "duplicados": dup_all, "rut": rut_df, "quiebres": quiebres, "lista_reglas": reglas,
    }


def ejecutar(ctx: Contexto) -> dict[str, Any]:
    r = calcular(ctx)
    TA = TAREA
    ctx.escribir_tabla("t24_reglas.csv", r["reglas"], tarea=TA, descripcion="Violaciones de reglas de calidad por regla y año",
                       cols_n=["n_violaciones"], dependientes=["prop_violaciones"])
    ctx.escribir_tabla("t24_id_largo.csv", r["largo"], tarea=TA, descripcion="Largo del identificador por año", cols_n=["n"], dependientes=["prop"])
    ctx.escribir_tabla("t24_id_traslape.csv", r["traslape"], tarea=TA, descripcion="Traslape de identificadores entre años",
                       cols_n=["n_interseccion"], dependientes=["prop_min", "prop_a", "prop_b"])
    ctx.escribir_tabla("t24_id_regimenes.csv", r["regimenes"], tarea=TA, descripcion="Regímenes de identificadores")
    ctx.escribir_tabla("t24_id_consistencia.csv", r["consistencia"], tarea=TA, descripcion="Consistencia de nacimiento y sexo por identificador",
                       cols_n=["n_ids_nacimiento_inconsistente", "n_ids_sexo_inconsistente"])
    ctx.registrar_metrica(
        "id.consistencia.nacimiento_inconsistente_prop",
        _prop_inconsistente(r["consistencia"], "n_ids_nacimiento_inconsistente", "n_ids_con_dos_o_mas_episodios"),
        tarea=TA, tabla="t24_id_consistencia.csv", nota="IDs de cualquier largo, todos los regímenes",
    )
    ctx.registrar_metrica(
        "id.consistencia.sexo_inconsistente_prop",
        _prop_inconsistente(r["consistencia"], "n_ids_sexo_inconsistente", "n_ids_con_sexo_en_dos_o_mas_episodios"),
        tarea=TA, tabla="t24_id_consistencia.csv", nota="IDs de cualquier largo, todos los regímenes",
    )
    ctx.escribir_tabla("t24_duplicados.csv", r["duplicados"], tarea=TA, descripcion="Duplicados por clave natural y episodios solapados",
                       cols_n=["n_solapes", "n_duplicados_exactos_por_id_y_fechas", "n_filas_duplicadas_clave"])
    ctx.escribir_tabla("t24_rut_por_columna.csv", r["rut"], tarea=TA, descripcion="Valores con estructura de RUT por columna y año", cols_n=["n"])
    resumen = pl.DataFrame([_resumen(ctx, r)])
    ctx.escribir_tabla("t24_resumen.csv", resumen, tarea=TA, descripcion="Cifras globales de la calidad, lote 1")
    _registrar_metricas(ctx, r)
    figura_violaciones(ctx)
    figura_traslape(ctx)
    figura_largo_id(ctx)
    return r


def _prop_inconsistente(cons: pl.DataFrame, col_inc: str, col_den: str) -> float | None:
    """Proporción de IDs inconsistentes entre todos los regímenes, para el grupo 'todos los IDs'."""
    t = cons.filter(pl.col("grupo_id") == "todos los IDs")
    den = int(t[col_den].sum())
    return float(t[col_inc].sum()) / den if den else None


def _total(r: dict, regla_id: str) -> int | None:
    t = r["reglas"].filter((pl.col("regla_id") == regla_id) & (pl.col("ambito") == "total"))
    return int(t["n_violaciones"][0]) if t.height and t["n_violaciones"][0] is not None else None


def _resumen(ctx: Contexto, r: dict) -> dict[str, Any]:
    reglas = r["reglas"]
    anuales = reglas.filter(pl.col("ambito").str.contains(r"^\d{4}$"))
    return {
        "n_reglas": len(r["lista_reglas"]),
        "n_reglas_alta": sum(1 for x in r["lista_reglas"] if x.severidad == "alta"),
        "n_regimenes": r["regimenes"].height,
        "quiebres": ", ".join(f"{a} a {b}" for a, b in r["quiebres"]) if r["quiebres"] else "ninguno",
        "n_quiebres": len(r["quiebres"]),
        "regimenes": "; ".join(r["regimenes"]["etiqueta"].to_list()),
        "n_bloqueadas": int((anuales["estado"] == "BLOQUEADA_2024").sum()),
        "ratio_quiebre": float(ctx.umbrales["id_quiebre_ratio"]),
    }


def _registrar_metricas(ctx: Contexto, r: dict) -> None:
    mapa = {
        "Q05": "calidad.alta_antes_ingreso", "Q03": "calidad.alta_vacia", "Q07": "calidad.edad_negativa",
        "Q08": "calidad.edad_mayor_110", "Q09": "calidad.id_vacio", "Q17": "calidad.valores_rut",
        "Q01": "calidad.fecha_nacimiento_vacia", "Q02": "calidad.fecha_ingreso_vacia", "Q10": "calidad.id_centinela",
    }
    for rid, clave in mapa.items():
        v = _total(r, rid)
        ctx.registrar_metrica(f"{clave}.total", v, tarea=TAREA, tabla="t24_reglas.csv", conteo=True)
        sub = r["reglas"].filter((pl.col("regla_id") == rid) & pl.col("ambito").str.contains(r"^\d{4}$"))
        for row in sub.iter_rows(named=True):
            ctx.registrar_metrica(f"{clave}.{row['ambito']}", row["n_violaciones"], tarea=TAREA, tabla="t24_reglas.csv", conteo=True)
    ed = [_total(r, "Q07"), _total(r, "Q08")]
    if all(x is not None for x in ed):
        ctx.registrar_metrica("calidad.edad_imposible.total", sum(ed), tarea=TAREA, tabla="t24_reglas.csv", conteo=True)
    ctx.registrar_metrica("calidad.n_quiebres_id", len(r["quiebres"]), tarea=TAREA, tabla="t24_id_regimenes.csv")
    ctx.registrar_metrica("calidad.n_regimenes_id", r["regimenes"].height, tarea=TAREA, tabla="t24_id_regimenes.csv")
    if r["quiebres"]:
        ctx.registrar_metrica("calidad.primer_quiebre_id_anio_posterior", min(b for _, b in r["quiebres"]), tarea=TAREA, tabla="t24_id_regimenes.csv")
    # traslape exacto entre años consecutivos
    for row in r["traslape"].filter(pl.col("consecutivo") == "si").iter_rows(named=True):
        ctx.registrar_metrica(f"id.traslape_prop_min.{row['anio_a']}_{row['anio_b']}", row["prop_min"], tarea=TAREA, tabla="t24_id_traslape.csv")


# ----------------------------------------------------------------------------------- figuras
def figura_violaciones(ctx: Contexto) -> None:
    t = ctx.leer_tabla("t24_reglas.csv", columnas_texto=["ambito"]).filter(pl.col("ambito").str.contains(r"^\d{4}$"))
    orden_sev = {"alta": 0, "media": 1, "baja": 2}
    reglas = (
        t.select(["regla_id", "nombre", "severidad"]).unique().with_columns(pl.col("severidad").replace_strict(orden_sev).alias("o"))
        .sort(["o", "regla_id"])
    )
    etiquetas = [f"[{r['severidad']}] {r['regla_id']} {r['nombre']}" for r in reglas.iter_rows(named=True)]
    ids = reglas["regla_id"].to_list()
    anios = sorted(t["ambito"].unique().to_list())
    m = np.full((len(ids), len(anios)), np.nan)
    for i, rid in enumerate(ids):
        for j, a in enumerate(anios):
            s = t.filter((pl.col("regla_id") == rid) & (pl.col("ambito") == a))
            if s.height and s["prop_violaciones"][0] is not None:
                m[i, j] = float(s["prop_violaciones"][0]) * 100000.0
    logm = np.where(np.isnan(m), np.nan, np.log10(m + 1.0))
    fig, ax = graficos.nueva_figura(figsize=(10.5, 7.0))
    graficos.mapa_calor(fig, ax, logm, etiquetas, anios, vmin=0, vmax=max(5.0, float(np.nanmax(logm)) if np.isfinite(logm).any() else 5.0),
                        etiqueta_barra="log10(violaciones por 100.000 + 1)", anotar=False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            txt = "n/d" if np.isnan(m[i, j]) else f"{m[i, j]:,.0f}".replace(",", ".")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6, color="white" if (not np.isnan(logm[i, j]) and logm[i, j] < 2.5) else "black")
    ax.set_xlabel("Año del archivo")
    ctx.figuras.guardar_figura(
        fig, id_figura="F05_01", tarea=5,
        titulo="Violaciones de reglas de calidad por año y severidad",
        descripcion_alt="Mapa de calor con las violaciones por cada 100.000 filas de cada regla de calidad y cada año, ordenado por severidad.",
        csv="t24_reglas.csv", anios=ANIOS_TXT, decision="11",
        extra_pie="Gris: celda suprimida o regla bloqueada por el cortafuegos.",
    )


def figura_traslape(ctx: Contexto) -> None:
    t = ctx.leer_tabla("t24_id_traslape.csv")
    anios = sorted(set(t["anio_a"].to_list()) | set(t["anio_b"].to_list()))
    m = np.full((len(anios), len(anios)), np.nan)
    for row in t.iter_rows(named=True):
        i, j = anios.index(row["anio_a"]), anios.index(row["anio_b"])
        v = row["prop_min"]
        if v is not None:
            m[i, j] = m[j, i] = float(v) * 100.0
    fig, ax = graficos.nueva_figura(figsize=(7.2, 5.6))
    graficos.mapa_calor(fig, ax, m, [str(a) for a in anios], [str(a) for a in anios], vmin=0, etiqueta_barra="% de los IDs del año con menos IDs",
                        fmt="{:.2f}", ocultar_diagonal=True)
    ax.set_xlabel("Año")
    ax.set_ylabel("Año")
    ctx.figuras.guardar_figura(
        fig, id_figura="F05_02", tarea=5,
        titulo="Traslape de identificadores entre años",
        descripcion_alt="Matriz simétrica con el porcentaje de identificadores de paciente compartidos entre cada par de años, relativo al año con menos identificadores.",
        csv="t24_id_traslape.csv", anios=ANIOS_TXT, decision="1; 13",
        extra_pie="La diagonal no se calcula.",
    )


def figura_largo_id(ctx: Contexto) -> None:
    t = ctx.leer_tabla("t24_id_largo.csv", columnas_texto=["categoria"])
    cats = t.select(["categoria", "orden"]).unique().sort("orden")["categoria"].to_list()
    anios = sorted(t["anio"].unique().to_list())
    fig, ax = graficos.nueva_figura(figsize=(8.6, 5.0))
    base = np.zeros(len(anios))
    for k, c in enumerate(cats):
        vals = []
        for a in anios:
            s = t.filter((pl.col("anio") == a) & (pl.col("categoria") == c))
            v = s["prop"][0] if s.height else 0.0
            vals.append(0.0 if v is None else float(v) * 100)
        if sum(vals) == 0:
            continue
        etiqueta = c if c.isdigit() else {"vacio": "vacío", "centinela": "centinela (texto)", "no_numerico": "no numérico"}[c]
        ax.bar([str(a) for a in anios], vals, bottom=base, label=(f"{etiqueta} dígitos" if c.isdigit() else etiqueta),
               color=graficos.OKABE_ITO[k % 8], width=0.75)
        base += np.array(vals)
    ax.set_ylabel("% de las filas del año")
    ax.set_xlabel("Año del archivo")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.grid(axis="x", visible=False)
    ctx.figuras.guardar_figura(
        fig, id_figura="F05_03", tarea=5,
        titulo="Largo del identificador del paciente por año",
        descripcion_alt="Barras apiladas con la distribución del número de dígitos del identificador y de los identificadores vacíos o con texto, por año.",
        csv="t24_id_largo.csv", anios=ANIOS_TXT, decision="1",
    )


# ------------------------------------------------------------------------------------ informe
def agregar_secciones(inf: Informe, ctx: Contexto) -> None:
    """Agrega al informe 2.4 las secciones del lote 1 (fechas, identificador y duplicados)."""
    rs = lambda col: cifra(ctx, "t24_resumen.csv", col, columnas_texto=["quiebres", "regimenes"])  # noqa: E731
    inf.seccion("1. Reglas de calidad: fechas, identificador y duplicados")
    inf.parrafo(
        "Se evaluaron {n_reglas} reglas declarativas ({n_alta} de severidad alta). Cada regla devuelve solo conteos; "
        "las que leen columnas de resultado de 2024 lo hacen con autorización del cortafuegos y entregan únicamente violaciones.",
        n_reglas=rs("n_reglas"),
        n_alta=rs("n_reglas_alta"),
    )
    inf.figura("F05_01")
    reglas = ctx.leer_tabla("t24_reglas.csv", columnas_texto=["ambito"])
    inf.tabla(
        "t24_reglas.csv",
        df=reglas.filter(pl.col("ambito") == "total").select(["regla_id", "nombre", "severidad", "accion", "n_violaciones", "n_evaluados", "prop_violaciones"]),
        encabezados={"regla_id": "Regla", "nombre": "Nombre", "severidad": "Severidad", "accion": "Acción", "n_violaciones": "Violaciones",
                     "n_evaluados": "Evaluados", "prop_violaciones": "Proporción"},
        formatos={"prop_violaciones": "prop2"},
        leyenda="Violaciones totales por regla (años evaluados)",
    )
    inf.seccion("2. Identificador del paciente: regímenes y traslape")
    inf.parrafo(
        "El traslape de identificadores entre años consecutivos permite detectar si cambió el espacio de IDs. "
        "Un par de años tiene quiebre si su traslape es menor a {ratio|dec2} veces la mediana de los demás pares. "
        "Quiebres encontrados: {quiebres|texto}. Regímenes de IDs: {regimenes|texto}.",
        ratio=rs("ratio_quiebre"),
        quiebres=rs("quiebres"),
        regimenes=rs("regimenes"),
    )
    inf.figura("F05_02")
    inf.figura("F05_03")
    inf.tabla("t24_id_regimenes.csv", encabezados={"regimen": "Régimen", "anio_inicio": "Desde", "anio_fin": "Hasta", "anios": "Años", "etiqueta": "Etiqueta"},
              columnas_texto=["anios", "etiqueta"], leyenda="Regímenes de identificadores")
    inf.seccion("3. Consistencia del identificador dentro de cada régimen")
    inf.tabla("t24_id_consistencia.csv", leyenda="Identificadores con más de una fecha de nacimiento o de sexo",
              columnas_texto=["regimen", "grupo_id"],
              encabezados={"regimen": "Régimen", "grupo_id": "Grupo de IDs", "n_ids_con_dos_o_mas_episodios": "IDs con 2 o más episodios",
                           "n_ids_nacimiento_inconsistente": "Nacimiento inconsistente",
                           "n_ids_con_sexo_en_dos_o_mas_episodios": "IDs con sexo conocido en 2 o más episodios",
                           "n_ids_sexo_inconsistente": "Sexo inconsistente"})
    inf.seccion("4. Duplicados y episodios solapados")
    inf.tabla("t24_duplicados.csv", leyenda="Duplicados por clave natural y solapes",
              encabezados={"anio": "Año", "n_solapes": "Solapes", "n_evaluados_solape": "Evaluados", "n_duplicados_exactos_por_id_y_fechas": "Duplicados exactos (ID y fechas)",
                           "n_filas_duplicadas_clave": "Filas duplicadas por clave", "n_filas_con_id_valido": "Filas con ID válido"})
    inf.seccion("5. Valores con estructura de RUT")
    inf.parrafo(
        "Se buscó por patrón, en las columnas de los archivos, cualquier valor con forma de RUT. Solo se guardan conteos por columna y año; "
        "nunca el valor. Cualquier hallazgo se informa a FONASA por un canal formal (decisión 11)."
    )
    inf.tabla("t24_rut_por_columna.csv", max_filas=20, encabezados={"columna": "Columna", "anio": "Año", "n": "N"}, leyenda="Valores con estructura de RUT")
