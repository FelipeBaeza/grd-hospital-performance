"""Tarea 11, Elixhauser: odds ratio de las 31 categorías contra la mortalidad (H08) y sensibilidad de la decisión 8 a la duración de la estadía.

``tabla_or_elix`` entrega, por período (2019-2022 y 2023), el cuadro de 2 por 2 de cada categoría, su OR crudo con el intervalo de Woolf y su OR ajustado por edad (lineal y al
cuadrado) y sexo con el error de Wald de una regresión logística sin penalización (``modelos.RIDGE_WALD``); los valores q de Benjamini y Hochberg se calculan sobre las 31
categorías, por separado en cada período y tipo de OR. ``tabla_sensibilidad`` ordena a los sobrevivientes de la ventana de ranking (cohorte E) en cuatro tramos de estadía,
disjuntos y exhaustivos, y mide por categoría cuánto crece su prevalencia del primer al último tramo y su OR de estadía larga; con ambas arma el índice de complicación probable,
que ordena las categorías (la lista final queda pendiente de revisión clínica, decisión 12: la tabla informa y no decide).

Una separación solo anula el OR ajustado de la categoría cuyo propio coeficiente está marcado (``separacion_por_coef``): la de una columna ajena, como SEXO "(sin dato)", no anula
nada. Las funciones ``tabla_*`` solo CALCULAN la tabla real (sin suprimir); la escribe ``seleccion.ejecutar`` con las listas ``BLOQUES_*`` de este módulo, las mismas con que
``calcular`` arma la versión publicable. No lee datos: recibe el panel.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from scipy import stats

from . import elixhauser, graficos, modelos
from . import estadistica as est
from . import panel as modulo_panel
from .contexto import Contexto
from .panel import V_CONF, V_RANK

TAREA = "T11"
LISTA_REPOSITORIO = ("ELIX_02", "ELIX_22", "ELIX_24", "ELIX_25", "ELIX_26")
"""Categorías de la lista del repositorio: arritmias, coagulopatía, desnutrición, electrolitos y hemorragia aguda. La correspondencia por nombre (desnutrición con pérdida de
peso y hemorragia aguda con anemia por pérdida de sangre) es una interpretación PENDIENTE DE REVISIÓN CLÍNICA (decisión 12)."""
DISENO_AJUSTE = modelos.BASE_EDAD_SEXO
"""Edad estandarizada, su cuadrado y sexo (HOMBRE y "(sin dato)" contra MUJER): el diseño que acompaña a la categoría en los dos OR ajustados."""
TOP = 5

ESQUEMA_OR = {
    "periodo": pl.String, "elix_id": pl.String, "nombre_es": pl.String, "sospechosa_fase1": pl.String, "n_expuestos": pl.Int64, "n_eventos_expuestos": pl.Int64,
    "n_no_expuestos": pl.Int64, "n_eventos_no_expuestos": pl.Int64, "prevalencia": pl.Float64, "or_crudo": pl.Float64, "or_crudo_bajo": pl.Float64, "or_crudo_alto": pl.Float64,
    "p_crudo": pl.Float64, "q_crudo": pl.Float64, "or_ajustado": pl.Float64, "or_ajustado_bajo": pl.Float64, "or_ajustado_alto": pl.Float64, "p_ajustado": pl.Float64,
    "q_ajustado": pl.Float64, "or_relevante_crudo": pl.String, "or_relevante_ajustado": pl.String, "correccion_haldane": pl.String, "convergio": pl.String, "estado": pl.String,
}
ESQUEMA_SENSIBILIDAD = {
    "elix_id": pl.String, "nombre_es": pl.String, "corte_p25": pl.Float64, "corte_p50": pl.Float64, "corte_p75": pl.Float64,
    **{f"n_t{k}": pl.Int64 for k in range(1, 5)}, **{f"k_t{k}": pl.Int64 for k in range(1, 5)}, **{f"prev_t{k}": pl.Float64 for k in range(1, 5)},
    "razon_prev_t4_t1": pl.Float64, "razon_bajo": pl.Float64, "razon_alto": pl.Float64, "or_larga": pl.Float64, "or_larga_bajo": pl.Float64, "or_larga_alto": pl.Float64,
    "p_larga": pl.Float64, "q_larga": pl.Float64, "indice_complicacion": pl.Float64, "rango_indice": pl.Int64, "sospechosa_fase1": pl.String, "en_lista_repositorio": pl.String,
    "en_top5": pl.String, "estado": pl.String,
}
COLUMNAS_OR_ELIX = list(ESQUEMA_OR)
COLUMNAS_SENSIBILIDAD = list(ESQUEMA_SENSIBILIDAD)
_TEXTO_OR = ["periodo", "elix_id", "nombre_es", "sospechosa_fase1", "or_relevante_crudo", "or_relevante_ajustado", "correccion_haldane", "convergio", "estado"]

# Supresión (DIS 11): por los conteos del cuadro de 2 por 2 en el OR; por los conteos de tramo en la sensibilidad. Las razones, el OR de estadía larga y el índice dependen de
# los conteos de varios tramos de la misma fila: anularlos cuando cualquiera es pequeño es lo correcto. Los cortes de estadía (percentiles de toda la cohorte, repetidos en las
# 31 filas) no derivan de los conteos de una categoría y no se anulan; tampoco las marcas ni ``estado``.
COLS_N_OR = ["n_expuestos", "n_eventos_expuestos", "n_no_expuestos", "n_eventos_no_expuestos"]
DEPENDIENTES_OR = [
    "prevalencia", "or_crudo", "or_crudo_bajo", "or_crudo_alto", "p_crudo", "q_crudo", "or_ajustado", "or_ajustado_bajo", "or_ajustado_alto", "p_ajustado", "q_ajustado",
    "or_relevante_crudo", "or_relevante_ajustado", "correccion_haldane", "convergio",
]
BLOQUES_OR_ELIX = [(COLS_N_OR, DEPENDIENTES_OR)]
COLS_N_SENSIBILIDAD = [f"n_t{k}" for k in range(1, 5)] + [f"k_t{k}" for k in range(1, 5)]
DEPENDIENTES_SENSIBILIDAD = [
    *[f"prev_t{k}" for k in range(1, 5)], "razon_prev_t4_t1", "razon_bajo", "razon_alto", "or_larga", "or_larga_bajo", "or_larga_alto", "p_larga", "q_larga", "indice_complicacion",
    "rango_indice",
]
BLOQUES_SENSIBILIDAD = [(COLS_N_SENSIBILIDAD, DEPENDIENTES_SENSIBILIDAD)]


def _finito(x: Any) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def _si_no(x: bool | None) -> str | None:
    return None if x is None else ("si" if x else "no")


# ============================================================================================================================== OR
def _ajuste_categoria(x: np.ndarray, y: np.ndarray, base: np.ndarray, z: float) -> dict[str, Any]:
    """OR ajustado de una categoría: regresión logística sin penalización de ``y`` sobre la categoría y el diseño de edad y sexo (``base``).

    El coeficiente de la categoría es el 1 (la intercepción es el 0). Devuelve el OR con su IC de Wald, ``p``, ``convergio`` y ``estado``: "separación" si el coeficiente DE LA
    CATEGORÍA está marcado (valor absoluto sobre ``SEPARACION_COEF_MAX`` o error estándar sobre ``SEPARACION_SE_MAX``), "no convergió" si no hubo convergencia y "ok" en otro caso.
    La bandera de una columna ajena no interviene.
    """
    res = est.regresion_logistica(np.column_stack([x.astype(np.float64), base]), y, ridge=modelos.RIDGE_WALD)
    salida: dict[str, Any] = {"convergio": bool(res.convergio), "or": None, "bajo": None, "alto": None, "p": None, "estado": "ok"}
    if bool(res.separacion_por_coef[1]):
        salida["estado"] = "separación"
    elif not res.convergio:
        salida["estado"] = "no convergió"
    else:
        b, se = float(res.coef[1]), float(res.se[1])
        with np.errstate(over="ignore", invalid="ignore"):
            o, bajo, alto = float(np.exp(b)), float(np.exp(b - z * se)), float(np.exp(b + z * se))
        if not (np.isfinite(se) and np.isfinite([o, bajo, alto]).all()):
            salida["estado"] = "separación"                       # un límite de Wald infinito es información degenerada: el coeficiente no se estima
        else:
            salida.update({"or": o, "bajo": bajo, "alto": alto, "p": _finito(est.p_wald(b, se))})
    return salida


def _diseno_edad_sexo(panel: modulo_panel.Panel, anios: Sequence[int], semilla: int) -> np.ndarray:
    """Columnas de edad estandarizada, su cuadrado y sexo (las constantes se omiten) de las filas de los años dados, densas: el diseño es pequeño."""
    espec = modelos.especificar_diseno(panel, DISENO_AJUSTE, anios, semilla, omitir_nivel_base=True)
    X, _, _ = modelos.aplicar_diseno(espec, panel, anios)
    return X.toarray()


def _periodo(ctx: Contexto, panel: modulo_panel.Panel, etiqueta: str, anios: Sequence[int], e: elixhauser.Elixhauser) -> list[dict[str, Any]]:
    u = ctx.umbrales
    z = float(u["z_95"])
    d = panel.ventana(anios)
    y = d["y"].to_numpy().astype(np.int8)
    n = int(y.size)
    base = _diseno_edad_sexo(panel, anios, ctx.semilla)
    con_ambos = 0 < int(y.sum()) < n
    filas: list[dict[str, Any]] = []
    for elix_id in e.ids:
        x = d[elix_id].to_numpy().astype(np.int8)
        a, b = int(((x == 1) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
        c, dd = int(((x == 0) & (y == 1)).sum()), int(((x == 0) & (y == 0)).sum())
        fila: dict[str, Any] = {
            "periodo": etiqueta, "elix_id": elix_id, "nombre_es": e.nombres_es[elix_id], "sospechosa_fase1": "si" if elix_id in e.sospechosas else "no",
            "n_expuestos": a + b, "n_eventos_expuestos": a, "n_no_expuestos": c + dd, "n_eventos_no_expuestos": c, "prevalencia": (a + b) / n if n else None,
            "or_crudo": None, "or_crudo_bajo": None, "or_crudo_alto": None, "p_crudo": None, "q_crudo": None, "or_ajustado": None, "or_ajustado_bajo": None,
            "or_ajustado_alto": None, "p_ajustado": None, "q_ajustado": None, "or_relevante_crudo": None, "or_relevante_ajustado": None, "correccion_haldane": None,
            "convergio": None, "estado": "ok",
        }
        if not con_ambos:
            fila["estado"] = "sin ambos desenlaces"
        elif a + b == 0 or c + dd == 0:
            fila["estado"] = "sin expuestos"
        else:
            o, bajo, alto, corregido = est.or_woolf(a, b, c, dd, z)
            with np.errstate(divide="ignore", invalid="ignore"):
                se = (np.log(alto) - np.log(bajo)) / (2.0 * z)
            fila.update({
                "or_crudo": _finito(o), "or_crudo_bajo": _finito(bajo), "or_crudo_alto": _finito(alto), "p_crudo": _finito(est.p_wald(np.log(o), se)),
                "correccion_haldane": _si_no(bool(corregido)), "or_relevante_crudo": modelos.or_es_relevante(_finito(o), _finito(bajo), _finito(alto), float(u["or_relevante"])),
            })
            aj = _ajuste_categoria(x, y, base, z)
            fila.update({
                "or_ajustado": aj["or"], "or_ajustado_bajo": aj["bajo"], "or_ajustado_alto": aj["alto"], "p_ajustado": aj["p"], "convergio": _si_no(aj["convergio"]),
                "estado": aj["estado"], "or_relevante_ajustado": modelos.or_es_relevante(aj["or"], aj["bajo"], aj["alto"], float(u["or_relevante"])),
            })
        filas.append(fila)
    for p_col, q_col in (("p_crudo", "q_crudo"), ("p_ajustado", "q_ajustado")):                      # Benjamini y Hochberg sobre las 31 categorías con p finito, por período
        p = np.array([np.nan if f[p_col] is None else f[p_col] for f in filas], dtype=np.float64)
        q = est.benjamini_hochberg(p)
        for f, qi in zip(filas, q):
            f[q_col] = _finito(qi)
    return filas


def tabla_or_elix(ctx: Contexto, panel: modulo_panel.Panel) -> pl.DataFrame:
    """``t23_or_elix.csv`` real (sin suprimir): 31 filas por período ("2019-2022" y, si hay confirmación, "2023").

    Por categoría: los cuatro conteos del cuadro de 2 por 2 contra la mortalidad, la prevalencia, el OR crudo con el IC de Woolf (con la corrección de Haldane y Anscombe si alguna celda
    es 0, marcada en ``correccion_haldane``), el OR ajustado con el IC de Wald (ridge cero, con la categoría, la edad estandarizada, su cuadrado y las columnas de SEXO contra
    MUJER), sus valores p y q (Benjamini y Hochberg por período y tipo de OR) y las marcas ``or_relevante_*`` (regla de ``or_relevante`` con IC que excluye 1). ``estado``:
    "ok", "sin expuestos" (sin expuestos o sin no expuestos: cifras nulas), "sin ambos desenlaces" (el período no tiene eventos o no tiene sobrevivientes), "separación" (el coeficiente
    de la categoría está marcado: el OR ajustado y todo lo que de él depende queda nulo) o "no convergió"; el OR crudo se publica en los dos últimos.
    """
    e = elixhauser.cargar()
    periodos = [(V_RANK, panel.anios_ranking)] + ([(V_CONF, [panel.anio_confirmacion])] if panel.anio_confirmacion is not None else [])
    filas: list[dict[str, Any]] = []
    for etiqueta, anios in periodos:
        filas += _periodo(ctx, panel, etiqueta, anios, e)
    return pl.DataFrame(filas, schema=ESQUEMA_OR)


# ============================================================================================================== sensibilidad de estadía
def asignar_tramos(estancia: Sequence[float], p25: float, p50: float, p75: float) -> np.ndarray:
    """Tramo de estadía (1 a 4) de cada valor, disjunto y exhaustivo por construcción: se asigna desde la cola.

    ``tramo = 4`` si la estadía es al menos el p75 (la "estadía larga"), ``3`` si es mayor que el p50 (y menor que el p75), ``2`` si es mayor que el p25 (y a lo más el p50) y ``1`` en
    otro caso. Si los cortes empatan, un valor que cumpliría la condición de dos tramos queda en el de mayor número y un tramo puede quedar vacío (la suma de los cuatro conteos
    sigue siendo el total).
    """
    est_ = np.asarray(estancia, dtype=np.float64)
    return np.where(est_ >= p75, 4, np.where(est_ > p50, 3, np.where(est_ > p25, 2, 1))).astype(np.int64)


def indice_complicacion(
    razon_bajo: Sequence[float | None], or_larga_bajo: Sequence[float | None], elix_id: Sequence[str], elegible: Sequence[bool],
) -> dict[str, list[Any]]:
    """Índice de complicación probable y su orden total (función pura, para probar empates sin panel).

    Solo las categorías elegibles con las dos cifras finitas tienen índice. Es el promedio de dos rangos, el de ``razon_bajo`` y el de ``or_larga_bajo`` (de mayor a menor, con rango
    promedio en los empates), reescalado como ``1 - (promedio - 1) / (m - 1)`` con ``m`` categorías con índice. ``rango_indice`` es la posición de 1 a ``m`` al ordenar por (índice
    descendente, rango de ``or_larga_bajo`` ascendente, rango de ``razon_bajo`` ascendente, ``elix_id`` ascendente): un orden total, así que ``en_top5`` marca EXACTAMENTE
    ``min(5, m)`` categorías ("si"; "no" en las demás con índice; ``None`` sin índice). Con ``m < 2`` no hay escala y ``indice``, ``rango`` y ``en_top5`` quedan nulos sin lanzar
    excepción. Devuelve las listas ``indice``, ``rango`` y ``en_top5`` alineadas con la entrada.
    """
    rb = np.array([np.nan if v is None else v for v in razon_bajo], dtype=np.float64)
    ob = np.array([np.nan if v is None else v for v in or_larga_bajo], dtype=np.float64)
    ids = list(elix_id)
    if not (len(rb) == len(ob) == len(ids) == len(elegible)):
        raise ValueError("indice_complicacion: las entradas tienen largos distintos")
    n = len(ids)
    con = np.flatnonzero(np.asarray(elegible, dtype=bool) & np.isfinite(rb) & np.isfinite(ob))
    indice: list[float | None] = [None] * n
    rango: list[int | None] = [None] * n
    top: list[str | None] = [None] * n
    m = int(con.size)
    if m < 2:
        return {"indice": indice, "rango": rango, "en_top5": top}
    r_razon = stats.rankdata(-rb[con], method="average")
    r_or = stats.rankdata(-ob[con], method="average")
    promedio = (r_razon + r_or) / 2.0
    valor = 1.0 - (promedio - 1.0) / (m - 1.0)
    orden = sorted(range(m), key=lambda i: (promedio[i], r_or[i], r_razon[i], ids[con[i]]))          # promedio ascendente = índice descendente, sin ruido de punto flotante
    for pos, i in enumerate(orden, start=1):
        rango[con[i]] = pos
        top[con[i]] = "si" if pos <= min(TOP, m) else "no"
        indice[con[i]] = float(valor[i])
    return {"indice": indice, "rango": rango, "en_top5": top}


def _fila_sensibilidad(
    elix_id: str, nombre: str, sospechosa: bool, cortes: tuple[float | None, float | None, float | None], n_t: list[int],
) -> dict[str, Any]:
    return {
        "elix_id": elix_id, "nombre_es": nombre, "corte_p25": cortes[0], "corte_p50": cortes[1], "corte_p75": cortes[2],
        **{f"n_t{k + 1}": n_t[k] for k in range(4)}, **{f"k_t{k + 1}": None for k in range(4)}, **{f"prev_t{k + 1}": None for k in range(4)},
        "razon_prev_t4_t1": None, "razon_bajo": None, "razon_alto": None, "or_larga": None, "or_larga_bajo": None, "or_larga_alto": None, "p_larga": None, "q_larga": None,
        "indice_complicacion": None, "rango_indice": None, "sospechosa_fase1": "si" if sospechosa else "no",
        "en_lista_repositorio": "si" if elix_id in LISTA_REPOSITORIO else "no", "en_top5": None, "estado": "ok",
    }


def tabla_sensibilidad(ctx: Contexto, panel: modulo_panel.Panel, or_elix: pl.DataFrame | None = None) -> pl.DataFrame:
    """``t23_sensibilidad_elix.csv`` real (sin suprimir): una fila por categoría sobre la cohorte E (``ES_COHORTE_E = 1``) de la ventana de ranking.

    Los cortes ``corte_p25``, ``corte_p50`` y ``corte_p75`` son percentiles de ``ESTANCIA_DIAS`` con ``percentil_de_conteos`` y se repiten en las 31 filas; los tramos salen de
    ``asignar_tramos``. Por categoría: ``n_t1`` a ``n_t4`` (sobrevivientes del tramo, iguales en las 31 filas), ``k_t1`` a ``k_t4`` (con la categoría), las prevalencias, la razón de
    prevalencias T4 contra T1 con el IC de Katz y el OR de estadía larga (T4 contra el resto) con el IC de Wald de la regresión sin penalización sobre la categoría, la edad
    estandarizada, su cuadrado y SEXO. El índice de complicación probable (``indice_complicacion``, ``rango_indice``, ``en_top5``) se calcula solo con las categorías que tienen al
    menos ``celda_minima`` expuestos en T1 y en T4 y ``estado = "ok"``. ``estado``: "ok", "sin expuestos" (la categoría no tiene expuestos o no tiene no expuestos en la cohorte E),
    "separación" o "no convergió" (estos dos solo anulan el OR de estadía larga, su IC, ``p_larga``, ``q_larga`` y por lo tanto el índice) y "sin variación de estadía" (todos los
    sobrevivientes caen en el mismo tramo: no hay estadía larga que comparar). ``or_elix`` es la tabla de OR: de ella salen el nombre y la marca de la Fase 1 de cada categoría
    (si no se pasa, del catálogo).
    """
    e = elixhauser.cargar()
    u = ctx.umbrales
    z, celda = float(u["z_95"]), ctx.celda_minima
    anios = panel.anios_ranking
    d = panel.ventana(anios)
    en_e = d["ES_COHORTE_E"].to_numpy() == 1
    estancia = d["ESTANCIA_DIAS"].cast(pl.Float64).to_numpy()[en_e]
    nombres = dict(e.nombres_es)
    sospechosas = set(e.sospechosas)
    if or_elix is not None and or_elix.height:
        previa = or_elix.filter(pl.col("periodo") == V_RANK)
        nombres.update(dict(zip(previa["elix_id"], previa["nombre_es"])))
        sospechosas = set(previa.filter(pl.col("sospechosa_fase1") == "si")["elix_id"].to_list())
    if estancia.size == 0:
        filas = []
        for elix_id in e.ids:
            f = _fila_sensibilidad(elix_id, nombres[elix_id], elix_id in sospechosas, (None, None, None), [0, 0, 0, 0])
            f.update({"estado": "sin expuestos", **{f"k_t{k}": 0 for k in range(1, 5)}})
            filas.append(f)
        return pl.DataFrame(filas, schema=ESQUEMA_SENSIBILIDAD)
    valores, conteos = np.unique(estancia, return_counts=True)
    cortes = tuple(est.percentil_de_conteos(valores, conteos, q) for q in (0.25, 0.50, 0.75))
    tramo = asignar_tramos(estancia, *cortes)
    n_t = [int((tramo == k).sum()) for k in range(1, 5)]
    base = _diseno_edad_sexo(panel, anios, ctx.semilla)[en_e]
    larga = (tramo == 4).astype(np.int8)
    filas = []
    for elix_id in e.ids:
        x = d[elix_id].to_numpy().astype(np.int8)[en_e]
        f = _fila_sensibilidad(elix_id, nombres[elix_id], elix_id in sospechosas, cortes, n_t)
        k_t = [int(((tramo == k) & (x == 1)).sum()) for k in range(1, 5)]
        for k in range(4):
            f[f"k_t{k + 1}"] = k_t[k]
            f[f"prev_t{k + 1}"] = k_t[k] / n_t[k] if n_t[k] > 0 else None
        razon, rbajo, ralto = est.razon_de_proporciones(k_t[3], n_t[3], k_t[0], n_t[0], z)
        f.update({"razon_prev_t4_t1": _finito(razon), "razon_bajo": _finito(rbajo), "razon_alto": _finito(ralto)})
        expuestos, no_expuestos = int((x == 1).sum()), int((x == 0).sum())
        if expuestos == 0 or no_expuestos == 0:
            f["estado"] = "sin expuestos"
        elif not (0 < int(larga.sum()) < larga.size):
            f["estado"] = "sin variación de estadía"
        else:
            aj = _ajuste_categoria(x, larga, base, z)
            f.update({"or_larga": aj["or"], "or_larga_bajo": aj["bajo"], "or_larga_alto": aj["alto"], "p_larga": aj["p"], "estado": aj["estado"]})
        filas.append(f)
    q = est.benjamini_hochberg(np.array([np.nan if f["p_larga"] is None else f["p_larga"] for f in filas], dtype=np.float64))
    for f, qi in zip(filas, q):
        f["q_larga"] = _finito(qi)
    elegible = [f["estado"] == "ok" and f["k_t1"] >= celda and f["k_t4"] >= celda for f in filas]
    idx = indice_complicacion([f["razon_bajo"] for f in filas], [f["or_larga_bajo"] for f in filas], [f["elix_id"] for f in filas], elegible)
    for f, i, r, t in zip(filas, idx["indice"], idx["rango"], idx["en_top5"]):
        f["indice_complicacion"], f["rango_indice"], f["en_top5"] = i, r, t
    return pl.DataFrame(filas, schema=ESQUEMA_SENSIBILIDAD)


def resumen_elixhauser(or_elix: pl.DataFrame, sensibilidad: pl.DataFrame) -> dict[str, Any]:
    """Cifras de Elixhauser para el resumen: las cinco primeras por índice (``elix_top5``, ids en orden de rango separados por "; "), cuántas coinciden con las cinco de la Fase 1 y
    con las de la lista del repositorio, y cuántas categorías tienen OR ajustado relevante en la ventana de ranking."""
    top = sensibilidad.filter(pl.col("en_top5") == "si").sort("rango_indice", maintain_order=True)["elix_id"].to_list()
    fase1 = set(sensibilidad.filter(pl.col("sospechosa_fase1") == "si")["elix_id"].to_list())
    repo = set(sensibilidad.filter(pl.col("en_lista_repositorio") == "si")["elix_id"].to_list())
    relevantes = or_elix.filter((pl.col("periodo") == V_RANK) & (pl.col("or_relevante_ajustado") == "si")).height
    return {
        "elix_top5": "; ".join(top), "elix_n_coinciden_fase1": len(set(top) & fase1), "elix_n_coinciden_repo": len(set(top) & repo), "elix_n_or_relevante_ajustado": relevantes,
        "elix_n_con_indice": int(sensibilidad.filter(pl.col("rango_indice").is_not_null()).height),
    }


# =========================================================================================================================== figura
def _num(serie: pl.Series) -> pl.Series:
    return serie.cast(pl.Float64, strict=False)


def figura_or_elix(ctx: Contexto) -> bool:
    """F11_04: bosque de odds ratio de las 31 categorías, crudo y ajustado, en 2019-2022. Se arma leyendo SOLO ``t23_or_elix.csv``.

    Categorías ordenadas por OR ajustado (mayor arriba; las sin OR ajustado, abajo por OR crudo); dos marcadores con IC por fila (crudo y ajustado, desplazados verticalmente);
    escala logarítmica con coma decimal; línea en 1 y banda entre el inverso de ``or_relevante`` y ``or_relevante``; las cinco de la Fase 1 (columna ``sospechosa_fase1`` de la
    misma tabla) con asterisco en la etiqueta. Una categoría sin ninguna cifra (sin expuestos o con la fila suprimida por tener menos de diez casos en una celda) se rotula "n/d"
    y no se dibuja. Devuelve ``False`` si la tabla no tiene filas.
    """
    t = ctx.leer_tabla("t23_or_elix.csv", columnas_texto=_TEXTO_OR)
    cols = ["or_crudo", "or_crudo_bajo", "or_crudo_alto", "or_ajustado", "or_ajustado_bajo", "or_ajustado_alto"]
    t = t.filter(pl.col("periodo") == V_RANK).with_columns([_num(pl.col(c)) for c in cols])
    if t.height == 0:
        return False
    t = t.sort(["or_ajustado", "or_crudo", "elix_id"], descending=[True, True, False], nulls_last=True, maintain_order=True)
    n = t.height
    umbral = float(ctx.umbrales["or_relevante"])
    alto_leyenda = 0.8
    altura = graficos.altura_filas(n, por_fila=0.28, minimo=5.0, extra=1.6)
    fig, (ax, hueco_leyenda) = graficos.nueva_figura_multiple(2, 1, figsize=(10.0, altura + alto_leyenda), gridspec_kw={"height_ratios": [altura, alto_leyenda]})
    hueco_leyenda.axis("off")
    color_crudo, color_ajustado = graficos.OKABE_ITO[4], graficos.OKABE_ITO[3]
    etiquetas, valores = [], []
    for i, f in enumerate(t.iter_rows(named=True)):
        etiquetas.append(f"{f['elix_id']}{'*' if f['sospechosa_fase1'] == 'si' else ''} {f['nombre_es']}")
        hay = False
        for clave, marcador, color, desplazamiento, relleno in (
            ("or_crudo", "o", color_crudo, -0.16, "white"), ("or_ajustado", "D", color_ajustado, 0.16, color_ajustado),
        ):
            o, bajo, alto = f[clave], f[f"{clave}_bajo"], f[f"{clave}_alto"]
            if o is None:
                continue
            hay = True
            valores += [x for x in (o, bajo, alto) if x is not None]
            err = None if bajo is None or alto is None else [[max(o - bajo, 0.0)], [max(alto - o, 0.0)]]
            ax.errorbar(o, i + desplazamiento, xerr=err, fmt=marcador, color=color, markerfacecolor=relleno, markersize=4.2, elinewidth=0.9, capsize=1.6, zorder=3)
        if not hay:
            ax.text(umbral * 1.08, i, "n/d", ha="left", va="center", fontsize=6.5, color="#555555")           # a la derecha de la banda: la línea en 1 no la cruza
    ax.set_xscale("log")
    inferior = max(min(valores) / 1.25, 0.02) if valores else 0.5
    superior = min(max(valores) * 1.25, 200.0) if valores else 2.0
    ax.set_xlim(min(inferior, 1.0 / (umbral * 1.1)), max(superior, umbral * 1.1))
    ax.axvspan(1.0 / umbral, umbral, color="#E6E6E6", alpha=0.6, zorder=0)
    ax.axvline(1.0, color="#333333", linewidth=1.0, zorder=1)
    marcas = [v for v in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0) if ax.get_xlim()[0] <= v <= ax.get_xlim()[1]]
    ax.set_xticks(marcas)
    ax.xaxis.set_minor_formatter(NullFormatter())
    graficos.formato_decimal(ax, "x")
    ax.set_yticks(range(n))
    ax.set_yticklabels(etiquetas, fontsize=6.8)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlabel("Odds ratio de mortalidad (escala logarítmica; 1 = sin asociación)")
    ax.grid(axis="y", visible=False)
    manijas = [
        Line2D([], [], marker="o", color=color_crudo, markerfacecolor="white", linestyle="", markersize=5, label="OR crudo con IC"),
        Line2D([], [], marker="D", color=color_ajustado, linestyle="", markersize=5, label="OR ajustado por edad y sexo con IC"),
        Line2D([], [], color="#BDBDBD", linewidth=6, label="Sin relevancia práctica (entre el inverso del umbral y el umbral)"),
    ]
    leyenda = hueco_leyenda.legend(handles=manijas, loc="center", bbox_to_anchor=(0.5, 0.5), ncol=2, frameon=False, fontsize=7.5)
    leyenda.set_in_layout(False)
    ctx.figuras.guardar_figura(
        fig, id_figura="F11_04", tarea=11, titulo="Odds ratio de las categorías de Elixhauser, crudo y ajustado (2019 a 2022)",
        descripcion_alt="Bosque horizontal con las 31 categorías de Elixhauser ordenadas por su odds ratio ajustado de mortalidad: para cada una, un círculo con el odds ratio crudo y un rombo "
                        "con el ajustado por edad y sexo, ambos con intervalo de confianza, en escala logarítmica; una banda gris marca el rango sin relevancia práctica y un "
                        "asterisco señala las cinco categorías que la Fase 1 sospecha.",
        csv="t23_or_elix.csv", anios="2019 a 2022", decision="8", uso_interno=False,
        extra_pie="* Categoría sospechosa en la Fase 1. n/d: sin expuestos o fila suprimida por tener menos de diez casos en una celda. El odds ratio ajustado no se publica "
                  "si el coeficiente de la categoría tiene separación o el ajuste no converge.",
    )
    return True


# FIN-sensibilidad_elix
