"""Tarea 11, sospechas: proxy del hospital (H02a, H02b, H03) y fuga (H14, S09), con las figuras F11_02 y F11_06.

``tabla_proxy`` mide, para cada variable candidata (también las de equidad y las condicionales), su asociación con el hospital con la MISMA medida de la Tarea 10
(``asociacion.medida_par`` contra COD_HOSPITAL) en la ventana de ranking y en 2023, y agrega la variación entre hospitales de los ingresos a cuidados especiales (H02a) y la
cardinalidad de ESPECIALIDAD_MEDICA por hospital (H02b), que el panel calcula sobre el texto sin tope de niveles. ``tabla_fuga`` mide el AUC univariado de las variables que solo
existen al alta o durante la hospitalización y de las condicionales cercanas a ser fuga, con el mismo puntaje y la misma dirección que el ranking, y lo compara con dos modelos de
referencia (edad y sexo; las 31 categorías de Elixhauser). IR_29301_MORTALIDAD queda prohibida siempre (R6) aunque su señal sea baja.

Las funciones ``tabla_*`` solo CALCULAN y devuelven la tabla real (sin suprimir): quien escribe es ``seleccion.ejecutar`` con ``panel.escribir_por_bloques`` y las listas
``BLOQUES_*`` de este módulo, las mismas que usa ``calcular`` para la versión publicable. Las figuras se arman leyendo solo su CSV. No lee datos: recibe el panel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl
from matplotlib.lines import Line2D

from . import asociacion, decision, graficos, modelos
from . import estadistica as est
from . import panel as modulo_panel
from .contexto import Contexto
from .panel import MIN_FILAS_HOSPITAL, V_CONF, V_RANK

TAREA = "T11"
ETIQUETA_V_RESULTADO = 0.10
"""Constante de visualización de F11_02: se rotulan los puntos con asociación con el resultado de al menos este valor (el tamaño pequeño de la escala de Cohen que cita el
protocolo para ``proxy_hospital_moderado``). No es un umbral del protocolo ni de decisión."""

CONDICIONALES_FUGA = ("SERVICIOINGRESO", "ESPECIALIDAD_MEDICA", "NIVEL_CUIDADO_INGRESO", "DERIVADO_OTRO_HOSPITAL")
"""Candidatas de rol CONDICIONAL que podrían codificar lo que pasa después del ingreso: se miden con el AUC de fuga junto a las variables de solo fuga."""
NOTA_R6 = "R6: prohibida bajo cualquier circunstancia, aunque la señal sea baja"
MOTIVO_SUPRIMIDO = "[suprimido: menos de diez casos en la fila de la tabla fuente]"
SUPRIMIDA = "suprimida"

_TEXTO_PROXY = ["tipo_fila", "variable", "tipo", "rol_fase1", "grupo", "medida", "clase_proxy", "analiza_resultado", "estado"]
_TEXTO_FUGA = [
    "tipo_fila", "variable", "categoria", "tipo", "rol_fase1", "prohibida_siempre", "direccion", "auc_optimista", "medida_resultado", "referencia_maxima", "supera_a_basales",
    "supera_umbral_fuga", "coincide_mayoria", "estado", "nota",
]

ESQUEMA_PROXY = {
    "tipo_fila": pl.String, "variable": pl.String, "tipo": pl.String, "rol_fase1": pl.String, "grupo": pl.String, "medida": pl.String, "n": pl.Int64, "n_2023": pl.Int64,
    "asociacion_hospital": pl.Float64, "asociacion_hospital_2023": pl.Float64, "clase_proxy": pl.String, "analiza_resultado": pl.String, "v_resultado": pl.Float64,
    "auc": pl.Float64, "estado": pl.String, "n_hospitales": pl.Int64, "prop_global": pl.Float64, "prop_minima": pl.Float64, "prop_p10": pl.Float64, "prop_mediana": pl.Float64,
    "prop_p90": pl.Float64, "prop_maxima": pl.Float64, "razon_p90_p10": pl.Float64, "phi_sobredispersion": pl.Float64, "card_minima": pl.Int64, "card_mediana": pl.Float64,
    "card_maxima": pl.Int64, "n_niveles_totales": pl.Int64, "n_niveles_en_un_hospital": pl.Int64,
}
ESQUEMA_FUGA = {
    "tipo_fila": pl.String, "variable": pl.String, "categoria": pl.String, "tipo": pl.String, "rol_fase1": pl.String, "prohibida_siempre": pl.String, "n": pl.Int64,
    "n_eventos": pl.Int64, "n_2023": pl.Int64, "n_eventos_2023": pl.Int64, "auc_ranking": pl.Float64, "auc_ranking_bajo": pl.Float64, "auc_ranking_alto": pl.Float64,
    "auc_confirmacion": pl.Float64, "auc_confirmacion_bajo": pl.Float64, "auc_confirmacion_alto": pl.Float64, "direccion": pl.String, "auc_optimista": pl.String,
    "v_resultado": pl.Float64, "medida_resultado": pl.String, "auc_ref_edad_sexo": pl.Float64, "auc_ref_elix31": pl.Float64, "referencia_maxima": pl.String,
    "supera_a_basales": pl.String, "supera_umbral_fuga": pl.String, "n_sin_traslado": pl.Int64, "n_iguales_sin_traslado": pl.Int64, "n_con_traslado": pl.Int64,
    "n_iguales_con_traslado": pl.Int64, "prop_igualdad": pl.Float64, "ic_bajo": pl.Float64, "ic_alto": pl.Float64, "coincide_mayoria": pl.String, "estado": pl.String,
    "nota": pl.String,
}
COLUMNAS_PROXY = list(ESQUEMA_PROXY)
COLUMNAS_FUGA = list(ESQUEMA_FUGA)

# Supresión por bloques (DIS 11): lo usan ``calcular`` (versión publicable) y ``seleccion.ejecutar`` (escritura). Los conteos de hospitales no son conteos de casos y no van a ``cols_n``.
BLOQUES_PROXY = [(["n"], ["asociacion_hospital", "clase_proxy", "v_resultado", "auc"]), (["n_2023"], ["asociacion_hospital_2023"])]
BLOQUES_FUGA = [
    (["n", "n_eventos"], ["auc_ranking", "auc_ranking_bajo", "auc_ranking_alto", "v_resultado"]),
    (
        ["n_2023", "n_eventos_2023"],
        ["auc_confirmacion", "auc_confirmacion_bajo", "auc_confirmacion_alto", "auc_ref_edad_sexo", "auc_ref_elix31", "supera_a_basales", "supera_umbral_fuga"],
    ),
    (["n_sin_traslado", "n_iguales_sin_traslado", "n_con_traslado", "n_iguales_con_traslado"], ["prop_igualdad", "ic_bajo", "ic_alto", "coincide_mayoria"]),
]


def _finito(x: Any) -> float | None:
    """Número finito como ``float`` o ``None`` (NaN, infinito y nulo pasan a ``None``)."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def _fila(esquema: Mapping[str, Any], **valores: Any) -> dict[str, Any]:
    """Fila con todas las columnas del esquema (nulas salvo las dadas); una clave que no es columna es un error de tipeo."""
    sobran = set(valores) - set(esquema)
    if sobran:
        raise KeyError(f"sospechas: columnas desconocidas {sorted(sobran)}")
    return {c: valores.get(c) for c in esquema}


# ============================================================================================================================ proxy
def clasificar_proxy(valor: float | None, u) -> str:
    """"alto", "moderado", "bajo" o "no evaluable" con el valor redondeado a los decimales publicados (``proxy_hospital_alto`` y ``proxy_hospital_moderado`` con igualdad)."""
    if valor is None or not np.isfinite(valor):
        return "no evaluable"
    v = float(est.redondear_publicado(valor))
    if v >= float(u["proxy_hospital_alto"]):
        return "alto"
    if v >= float(u["proxy_hospital_moderado"]):
        return "moderado"
    return "bajo"


def variacion_por_hospital(hospital: Sequence[Any], marca: Sequence[float], min_filas: int = MIN_FILAS_HOSPITAL, winsor: float = 0.10) -> dict[str, Any]:
    """Variación entre hospitales de una proporción (evidencia de H02a: ingresos a cuidados especiales).

    Entran los hospitales con al menos ``min_filas`` filas con valor (``marca`` es 0, 1 o NaN). Devuelve ``n_hospitales``; ``prop_global`` (la proporción agrupada de esos hospitales,
    que es el ``p0`` de los puntajes z); ``prop_minima``, ``prop_p10``, ``prop_mediana``, ``prop_p90`` y ``prop_maxima`` (cuantiles lineales de ``np.quantile`` de la proporción de
    cada hospital); ``razon_p90_p10`` (nula si el p10 es 0); y ``phi_sobredispersion``: el factor de Spiegelhalter de los puntajes ``puntaje_z_proporcion`` de cada hospital con la
    winsorización de la Tarea 9. Sin hospitales elegibles todo queda nulo salvo ``n_hospitales = 0``.
    """
    df = pl.DataFrame({"h": pl.Series(list(hospital), dtype=pl.String), "m": pl.Series([None if v is None or np.isnan(v) else float(v) for v in marca], dtype=pl.Float64)}).drop_nulls()
    por = df.group_by("h", maintain_order=True).agg(pl.len().alias("n"), pl.col("m").sum().alias("k")).filter(pl.col("n") >= int(min_filas)).sort("h")
    vacio = {k: None for k in ("prop_global", "prop_minima", "prop_p10", "prop_mediana", "prop_p90", "prop_maxima", "razon_p90_p10", "phi_sobredispersion")}
    if por.height == 0:
        return {"n_hospitales": 0, **vacio}
    n, k = por["n"].to_numpy().astype(np.float64), por["k"].to_numpy().astype(np.float64)
    prop = k / n
    p0 = float(k.sum() / n.sum())
    minima, p10, mediana, p90, maxima = (float(q) for q in np.quantile(prop, [0.0, 0.1, 0.5, 0.9, 1.0], method="linear"))
    z = est.puntaje_z_proporcion(k, n, p0)
    return {
        "n_hospitales": int(por.height), "prop_global": p0, "prop_minima": minima, "prop_p10": p10, "prop_mediana": mediana, "prop_p90": p90, "prop_maxima": maxima,
        "razon_p90_p10": (p90 / p10) if p10 > 0 else None, "phi_sobredispersion": _finito(est.phi_spiegelhalter(z, winsor)),
    }


def tabla_proxy(ctx: Contexto, panel: modulo_panel.Panel, ranking: pl.DataFrame) -> pl.DataFrame:
    """``t23_proxy_hospital.csv`` real (sin suprimir): una fila ``variable`` por candidata, la fila ``variacion_uci`` y la fila ``cardinalidad_especialidad``.

    La asociación con el hospital es ``asociacion.medida_par`` contra COD_HOSPITAL en la ventana de ranking (``asociacion_hospital``) y en 2023 (``asociacion_hospital_2023``): la
    misma medida de la Tarea 10 (V de Cramér corregida para categóricas y binarias, raíz de eta al cuadrado para numéricas). ETNIA y PUEBLO_ORIGINARIO llevan la asociación con el
    hospital pero ninguna cifra contra el resultado (decisión 4). ``v_resultado`` y ``auc`` salen de la fila de 2019-2022 del ranking (``auc`` es el ``auc_senal``). La fila de
    cardinalidad COPIA los agregados que el panel calculó sobre el texto sin tope de niveles (MEDIUM 1): no los recalcula desde la columna recortada a 60 niveles.
    """
    u = ctx.umbrales
    analizadas = asociacion.nombres_analizados(panel)
    referencias = [n for n in analizadas if panel.variables[n].es_referencia]
    if not referencias:
        raise ValueError("tabla_proxy: el panel no tiene la referencia COD_HOSPITAL")
    hospital = referencias[0]
    nombres = [n for n in analizadas if n != hospital]
    cols_rank = asociacion.preparar_columnas(panel, panel.anios_ranking)
    cols_conf = asociacion.preparar_columnas(panel, [panel.anio_confirmacion]) if panel.anio_confirmacion is not None else None
    del_ranking: dict[str, tuple[Any, Any]] = {}
    if ranking is not None and ranking.height:
        for f in ranking.filter(pl.col("ventana") == V_RANK).select("variable", "v_resultado", "auc_senal").iter_rows():
            del_ranking[f[0]] = (f[1], f[2])
    filas = []
    for n in nombres:
        v = panel.variables[n]
        p = asociacion.medida_par(cols_rank[n], cols_rank[hospital])
        p23 = asociacion.medida_par(cols_conf[n], cols_conf[hospital]) if cols_conf is not None else None
        v_res, auc = del_ranking.get(n, (None, None)) if v.analiza_resultado else (None, None)
        filas.append(_fila(
            ESQUEMA_PROXY, tipo_fila="variable", variable=n, tipo=v.tipo, rol_fase1=v.rol_fase1, grupo=v.grupo, medida=p.medida, n=p.n, n_2023=p23.n if p23 else None,
            asociacion_hospital=_finito(p.valor), asociacion_hospital_2023=_finito(p23.valor) if p23 else None, clase_proxy=clasificar_proxy(_finito(p.valor), u),
            analiza_resultado="si" if v.analiza_resultado else "no", v_resultado=_finito(v_res), auc=_finito(auc), estado=p.estado,
        ))
    if "NIVEL_CUIDADO_INGRESO" in panel.datos.columns:
        d = panel.ventana(panel.anios_ranking)
        marca = d["NIVEL_CUIDADO_INGRESO"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
        var = variacion_por_hospital(d["COD_HOSPITAL"].to_list(), marca, MIN_FILAS_HOSPITAL, float(u["winsor"]))
        filas.append(_fila(ESQUEMA_PROXY, tipo_fila="variacion_uci", variable="NIVEL_CUIDADO_INGRESO", tipo="bin", **var))
    card = panel.cardinalidad_especialidad
    if card:
        filas.append(_fila(
            ESQUEMA_PROXY, tipo_fila="cardinalidad_especialidad", variable="ESPECIALIDAD_MEDICA", tipo="cat", n_hospitales=card.get("n_hospitales"),
            card_minima=card.get("card_minima"), card_mediana=card.get("card_mediana"), card_maxima=card.get("card_maxima"), n_niveles_totales=card.get("n_niveles_totales"),
            n_niveles_en_un_hospital=card.get("n_niveles_en_un_hospital"),
        ))
    return pl.DataFrame(filas, schema=ESQUEMA_PROXY)


# ============================================================================================================================= fuga
def concordancia(igual: Sequence[float], traslados: Sequence[float]) -> list[dict[str, Any]]:
    """Concordancia del servicio de ingreso con el de alta (S09): dos filas, entre egresos sin traslado interno y con traslado.

    ``igual`` es SERVICIO_INGRESO_IGUAL_ALTA (1, 0 o NaN si algún servicio es nulo o centinela) y ``traslados`` N_TRASLADOS_INTERNOS (NaN si no se conoce). Los denominadores
    cuentan solo las filas con el indicador no nulo (contar las nulas como "no iguales" subestimaría la proporción) y solo las filas con el conteo de traslados no nulo se
    clasifican. ``prop_igualdad`` lleva el IC de Wilson y ``coincide_mayoria`` es "si" si la proporción supera 0,5; sin filas el estado es "sin datos".
    """
    ig = np.asarray(igual, dtype=np.float64)
    tr = np.asarray(traslados, dtype=np.float64)
    if ig.shape != tr.shape:
        raise ValueError(f"concordancia: 'igual' y 'traslados' tienen largos distintos ({ig.size} y {tr.size})")
    ok = ~np.isnan(ig) & ~np.isnan(tr)
    salida = []
    for tipo_fila, mascara, c_n, c_k in (
        ("concordancia", ok & (tr == 0), "n_sin_traslado", "n_iguales_sin_traslado"),
        ("concordancia_con_traslado", ok & (tr > 0), "n_con_traslado", "n_iguales_con_traslado"),
    ):
        n, k = int(mascara.sum()), int((mascara & (ig == 1)).sum())
        fila: dict[str, Any] = {"tipo_fila": tipo_fila, c_n: n, c_k: k, "estado": "ok" if n > 0 else "sin datos", "prop_igualdad": None, "ic_bajo": None, "ic_alto": None, "coincide_mayoria": None}
        if n > 0:
            bajo, alto = est.wilson(k, n)
            fila.update({"prop_igualdad": k / n, "ic_bajo": float(bajo), "ic_alto": float(alto), "coincide_mayoria": "si" if k / n > 0.5 else "no"})
        salida.append(fila)
    return salida


def marcas_de_fuga(auc_ranking: float | None, auc_confirmacion: float | None, auc_edad_sexo: float | None, auc_elix31: float | None, umbral_fuga: float) -> dict[str, str | None]:
    """``supera_a_basales``, ``referencia_maxima`` y ``supera_umbral_fuga`` de una variable, comparando valores redondeados a los decimales publicados.

    ``supera_a_basales``: el ``auc_confirmacion`` (honesto) es estrictamente mayor que el máximo de los AUC de 2023 de las dos referencias. ``referencia_maxima`` nombra la
    que fija ese máximo (en empate, MODELO_EDAD_SEXO). ``supera_umbral_fuga``: el mayor entre ``auc_ranking`` y ``auc_confirmacion`` alcanza el umbral, con igualdad.
    """
    def r(x: float | None) -> float | None:
        return None if x is None or not np.isfinite(x) else float(est.redondear_publicado(x))

    refs = [(r(auc_edad_sexo), "MODELO_EDAD_SEXO"), (r(auc_elix31), "MODELO_ELIX31")]
    refs = [(v, nombre) for v, nombre in refs if v is not None]
    maximo, nombre = (None, None)
    if refs:
        maximo, nombre = max(refs, key=lambda t: t[0])                         # ``max`` conserva el primero en un empate
    conf, rank = r(auc_confirmacion), r(auc_ranking)
    supera = None if conf is None or maximo is None else ("si" if conf > maximo else "no")
    propios = [v for v in (rank, conf) if v is not None]
    return {
        "supera_a_basales": supera, "referencia_maxima": nombre,
        "supera_umbral_fuga": None if not propios else ("si" if max(propios) >= float(umbral_fuga) else "no"),
    }


def tabla_fuga(ctx: Contexto, panel: modulo_panel.Panel, ranking: pl.DataFrame, referencias: Mapping[str, Mapping[str, Any]]) -> pl.DataFrame:
    """``t23_fuga.csv`` real (sin suprimir): variables de solo fuga y condicionales, los dos modelos de referencia y la concordancia de servicios.

    Las filas ``variable`` usan el AUC univariado compartido de ``modelos`` (mismo puntaje y misma dirección que el ranking): ``auc_ranking`` (en muestra en las categóricas),
    ``auc_confirmacion`` (honesto) y su IC de Hanley y McNeil. ``estado`` describe la medición de 2019-2022 con los valores del ranking; si no es "ok" el AUC queda nulo. La
    asociación con el resultado de una variable que ya está en el ranking se copia de él (es la misma cuenta). IR_29301_MORTALIDAD queda prohibida siempre (R6).
    """
    u = ctx.umbrales
    conf = panel.anio_confirmacion
    umbral = float(u["auc_sospecha_fuga"])
    edad_sexo = referencias.get("MODELO_EDAD_SEXO", {})
    elix31 = referencias.get("MODELO_ELIX31", {})
    ref_edad, ref_elix = _finito(edad_sexo.get("auc_2023")), _finito(elix31.get("auc_2023"))
    en_ranking: dict[str, tuple[Any, Any]] = {}
    if ranking is not None and ranking.height:
        for f in ranking.filter(pl.col("ventana") == V_RANK).select("variable", "v_resultado", "medida_resultado").iter_rows():
            en_ranking[f[0]] = (f[1], f[2])
    solo_fuga = [n for n, v in panel.variables.items() if v.uso == "solo_fuga" and n in panel.datos.columns]
    condicionales = [n for n in CONDICIONALES_FUGA if n in panel.variables and n in panel.datos.columns]
    y_rank = modelos.desenlace(panel, panel.anios_ranking)
    filas = []
    for categoria, nombres in (("solo_fuga", solo_fuga), ("condicional", condicionales)):
        for n in nombres:
            v = panel.variables[n]
            ajuste = modelos.ajustar_univariado(panel, n)
            estado, r_rank = modelos.medir_ventana(panel, ajuste, panel.anios_ranking)
            r_conf = modelos.medir_ventana(panel, ajuste, [conf])[1] if conf is not None else None
            if n in en_ranking:
                v_res, medida = en_ranking[n]
            else:
                v_res, medida, _ = asociacion.con_resultado(modelos.columna_del_panel(panel, n, panel.anios_ranking), y_rank)
            marcas = marcas_de_fuga(_finito(r_rank.auc), _finito(r_conf.auc) if r_conf else None, ref_edad, ref_elix, umbral)
            filas.append(_fila(
                ESQUEMA_FUGA, tipo_fila="variable", variable=n, categoria=categoria, tipo=v.tipo, rol_fase1=v.rol_fase1, prohibida_siempre="si" if n == "IR_29301_MORTALIDAD" else "no",
                n=r_rank.n, n_eventos=r_rank.n_eventos, n_2023=r_conf.n if r_conf else None, n_eventos_2023=r_conf.n_eventos if r_conf else None,
                auc_ranking=_finito(r_rank.auc), auc_ranking_bajo=_finito(r_rank.ic_bajo), auc_ranking_alto=_finito(r_rank.ic_alto),
                auc_confirmacion=_finito(r_conf.auc) if r_conf else None, auc_confirmacion_bajo=_finito(r_conf.ic_bajo) if r_conf else None,
                auc_confirmacion_alto=_finito(r_conf.ic_alto) if r_conf else None, direccion=ajuste.direccion or None, auc_optimista="si" if v.tipo == "cat" else "no",
                v_resultado=_finito(v_res), medida_resultado=medida, auc_ref_edad_sexo=ref_edad, auc_ref_elix31=ref_elix, estado=estado,
                nota=NOTA_R6 if n == "IR_29301_MORTALIDAD" else None, **marcas,
            ))
    for nombre in ("MODELO_EDAD_SEXO", "MODELO_ELIX31"):
        if nombre not in referencias:
            continue
        ref = referencias[nombre]
        n_ent, ev_ent, n_23, ev_23 = ref.get("n_entrenamiento"), ref.get("n_eventos_entrenamiento"), ref.get("n_2023"), ref.get("n_eventos_2023")
        auc_r, auc_c = _finito(ref.get("auc_ranking")), _finito(ref.get("auc_2023"))
        ic_r = est.ic_hanley_mcneil(auc_r, ev_ent, n_ent - ev_ent) if auc_r is not None and n_ent else (None, None)
        ic_c = est.ic_hanley_mcneil(auc_c, ev_23, n_23 - ev_23) if auc_c is not None and n_23 else (None, None)
        filas.append(_fila(
            ESQUEMA_FUGA, tipo_fila="referencia", variable=nombre, categoria="referencia", tipo="modelo", n=n_ent, n_eventos=ev_ent, n_2023=n_23 if conf is not None else None,
            n_eventos_2023=ev_23 if conf is not None else None, auc_ranking=auc_r, auc_ranking_bajo=_finito(ic_r[0]), auc_ranking_alto=_finito(ic_r[1]), auc_confirmacion=auc_c,
            auc_confirmacion_bajo=_finito(ic_c[0]), auc_confirmacion_alto=_finito(ic_c[1]), estado=ref.get("estado"),
        ))
    if {"SERVICIO_INGRESO_IGUAL_ALTA", "N_TRASLADOS_INTERNOS"} <= set(panel.datos.columns):
        d = panel.ventana(panel.anios_ranking)
        igual = d["SERVICIO_INGRESO_IGUAL_ALTA"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
        traslados = d["N_TRASLADOS_INTERNOS"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
        for fila in concordancia(igual, traslados):
            filas.append(_fila(ESQUEMA_FUGA, variable="SERVICIO_INGRESO_IGUAL_ALTA", categoria="concordancia", tipo="bin", **fila))
    return pl.DataFrame(filas, schema=ESQUEMA_FUGA)


# ========================================================================================================================= métricas
def registrar_estadistica(
    ctx: Contexto, clave: str, real: Any, publicado: Any, *, tabla: str, alcance: str, ic_bajo: Any = None, ic_alto: Any = None, nota: str = "", tarea: str = TAREA,
) -> None:
    """Registra una métrica con valor estadístico desde la versión publicable de su tabla fuente.

    Si el valor real existe y el publicado quedó nulo (la fila fuente se suprimió por celda pequeña), la métrica se registra con valor nulo y la nota
    "[suprimido: menos de diez casos en la fila de la tabla fuente]": una cifra que una tabla anuló no se vuelve a publicar en otro lado. Nunca es un conteo de casos
    (``conteo=False``) y siempre lleva alcance.
    """
    real, publicado = _finito(real), _finito(publicado)
    if real is not None and publicado is None:
        nota = f"{nota} {MOTIVO_SUPRIMIDO}".strip()
        ic_bajo = ic_alto = None
    ctx.registrar_metrica(
        clave, publicado, tarea=tarea, tabla=tabla, ic_bajo=_finito(ic_bajo) if publicado is not None else None, ic_alto=_finito(ic_alto) if publicado is not None else None,
        nota=nota, conteo=False, alcance=alcance,
    )


def registrar_metricas(ctx: Contexto, proxy: pl.DataFrame, proxy_pub: pl.DataFrame, fuga: pl.DataFrame, fuga_pub: pl.DataFrame) -> None:
    """Métricas ``proxy.*`` y ``fuga.*`` (tarea T11): valores desde las versiones publicables, con alcance explícito y ninguna que cubra 2024.

    ``proxy.v.<variable>`` (2019-2022) y ``proxy.v_2023.<variable>`` (2023) son la asociación con el hospital; ``proxy.n_alto`` y ``proxy.n_moderado`` cuentan variables (no casos);
    ``proxy.uci_phi`` es la sobredispersión de los ingresos a cuidados especiales. ``fuga.auc.<variable>`` (2019-2022) y ``fuga.auc_2023.<variable>`` (2023) son los AUC de las
    filas de la tabla de fuga; ``fuga.n_supera_umbral`` cuenta variables sobre el umbral y ``fuga.concordancia_pct_sin_traslado`` es el porcentaje de igualdad sin traslado.
    """
    variables = proxy.filter(pl.col("tipo_fila") == "variable")
    nota_conf = "confirmación no disponible" if variables.height and variables["n_2023"].null_count() == variables.height else ""
    for real, pub in zip(variables.iter_rows(named=True), proxy_pub.filter(pl.col("tipo_fila") == "variable").iter_rows(named=True)):
        n = real["variable"]
        registrar_estadistica(ctx, f"proxy.v.{n}", real["asociacion_hospital"], pub["asociacion_hospital"], tabla="t23_proxy_hospital.csv", alcance=V_RANK)
        registrar_estadistica(ctx, f"proxy.v_2023.{n}", real["asociacion_hospital_2023"], pub["asociacion_hospital_2023"], tabla="t23_proxy_hospital.csv", alcance=V_CONF, nota=nota_conf)
    for clave, clase in (("proxy.n_alto", "alto"), ("proxy.n_moderado", "moderado")):
        ctx.registrar_metrica(clave, int((variables["clase_proxy"] == clase).sum()), tarea=TAREA, tabla="t23_proxy_hospital.csv", conteo=False, alcance=V_RANK, nota="variables, no casos")
    uci_real = proxy.filter(pl.col("tipo_fila") == "variacion_uci")
    uci_pub = proxy_pub.filter(pl.col("tipo_fila") == "variacion_uci")
    if uci_real.height:
        registrar_estadistica(ctx, "proxy.uci_phi", uci_real["phi_sobredispersion"][0], uci_pub["phi_sobredispersion"][0], tabla="t23_proxy_hospital.csv", alcance=V_RANK)
    fv = fuga.filter(pl.col("tipo_fila") == "variable")
    fv_pub = fuga_pub.filter(pl.col("tipo_fila") == "variable")
    for real, pub in zip(fv.iter_rows(named=True), fv_pub.iter_rows(named=True)):
        n = real["variable"]
        registrar_estadistica(
            ctx, f"fuga.auc.{n}", real["auc_ranking"], pub["auc_ranking"], tabla="t23_fuga.csv", alcance=V_RANK, ic_bajo=pub["auc_ranking_bajo"], ic_alto=pub["auc_ranking_alto"],
            nota="categórica: optimista" if real["auc_optimista"] == "si" else "",
        )
        registrar_estadistica(
            ctx, f"fuga.auc_2023.{n}", real["auc_confirmacion"], pub["auc_confirmacion"], tabla="t23_fuga.csv", alcance=V_CONF, ic_bajo=pub["auc_confirmacion_bajo"],
            ic_alto=pub["auc_confirmacion_alto"], nota="" if real["n_2023"] is not None else "confirmación no disponible",
        )
    ctx.registrar_metrica(
        "fuga.n_supera_umbral", int((fv["supera_umbral_fuga"] == "si").sum()), tarea=TAREA, tabla="t23_fuga.csv", conteo=False, alcance="2019-2023",
        nota="variables, no casos; ranking 2019 a 2022 y confirmación 2023",
    )
    conc = fuga.filter(pl.col("tipo_fila") == "concordancia")
    conc_pub = fuga_pub.filter(pl.col("tipo_fila") == "concordancia")
    if conc.height:
        real_pct = None if conc["prop_igualdad"][0] is None else 100.0 * conc["prop_igualdad"][0]
        pub_pct = None if conc_pub["prop_igualdad"][0] is None else 100.0 * conc_pub["prop_igualdad"][0]
        registrar_estadistica(ctx, "fuga.concordancia_pct_sin_traslado", real_pct, pub_pct, tabla="t23_fuga.csv", alcance=V_RANK, nota="porcentaje de egresos sin traslado interno")


# ========================================================================================================================= figuras
def _num(serie: pl.Series) -> pl.Series:
    return serie.cast(pl.Float64, strict=False)


def figura_proxy(ctx: Contexto) -> bool:
    """F11_02: mapa de proxies, con la asociación con el hospital en x y con el resultado en y. Se arma leyendo SOLO ``t23_proxy_hospital.csv``.

    Líneas verticales en ``proxy_hospital_moderado`` y ``proxy_hospital_alto`` con las zonas rotuladas; color por situación en la Fase 1 (excluida, condicional o permitida,
    según ``rol_fase1``). Las etiquetas (variables con asociación con el hospital en o sobre el nivel moderado, o con el resultado en o sobre ``ETIQUETA_V_RESULTADO``,
    en orden de mayor asociación con el hospital) se colocan con ``etiquetas_sin_solape`` dentro del gancho ``antes_de_guardar``, sobre el diseño final con título y pie.
    ETNIA y PUEBLO_ORIGINARIO no se dibujan (no se analizan frente al resultado) y el pie lo dice. Devuelve ``False`` si no hay nada que dibujar.
    """
    t = ctx.leer_tabla("t23_proxy_hospital.csv", columnas_texto=_TEXTO_PROXY)
    v = t.filter(pl.col("tipo_fila") == "variable").with_columns(_num(pl.col("asociacion_hospital")), _num(pl.col("v_resultado")))
    puntos = v.filter(pl.col("asociacion_hospital").is_not_null() & pl.col("v_resultado").is_not_null())
    if puntos.height == 0:
        return False
    moderado, alto = float(ctx.umbrales["proxy_hospital_moderado"]), float(ctx.umbrales["proxy_hospital_alto"])
    situaciones = {"excluida": (graficos.OKABE_ITO[3], "o", "Excluida por la Fase 1"), "condicional": (graficos.OKABE_ITO[1], "s", "Condicional en la Fase 1"),
                   "permitida": (graficos.OKABE_ITO[0], "^", "Permitida por la Fase 1")}

    def situacion(rol: str) -> str:
        return "excluida" if rol in decision.ROLES_EXCLUYENTES else ("condicional" if rol == "CONDICIONAL" else "permitida")

    puntos = puntos.with_columns(pl.col("rol_fase1").map_elements(situacion, return_dtype=pl.String).alias("situacion"))
    x_max = max(float(puntos["asociacion_hospital"].max()) * 1.08, 0.45)
    y_max = max(float(puntos["v_resultado"].max()) * 1.15, 0.15)
    fig, ax = graficos.nueva_figura(figsize=(10.0, 7.0))
    for clave, (color, marcador, etiqueta) in situaciones.items():
        sub = puntos.filter(pl.col("situacion") == clave)
        if sub.height:
            ax.scatter(sub["asociacion_hospital"], sub["v_resultado"], s=26, color=color, marker=marcador, edgecolors="white", linewidths=0.4, label=etiqueta, zorder=3)
    for corte in (moderado, alto):
        ax.axvline(corte, color="#555555", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xlim(-0.01, x_max)
    ax.set_ylim(-0.005, y_max)
    ax.text((moderado + alto) / 2.0, 1.0, "moderado", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.5, color="#333333")
    ax.text((alto + x_max) / 2.0, 1.0, "alto", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.5, color="#333333")
    ax.set_xlabel("Asociación con el hospital (0 = ninguna, 1 = el hospital la determina)")
    ax.set_ylabel("Asociación con la mortalidad (0 = ninguna)")
    graficos.formato_decimal(ax, "x")
    graficos.formato_decimal(ax, "y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, frameon=False, fontsize=8)
    rotuladas = puntos.filter((pl.col("asociacion_hospital") >= moderado) | (pl.col("v_resultado") >= ETIQUETA_V_RESULTADO)).sort(
        ["asociacion_hospital", "variable"], descending=[True, False], maintain_order=True,
    )
    xs, ys, textos = rotuladas["asociacion_hospital"].to_list(), rotuladas["v_resultado"].to_list(), rotuladas["variable"].to_list()

    def etiquetar(f) -> None:
        graficos.etiquetas_sin_solape(f.axes[0], xs, ys, textos)

    ctx.figuras.guardar_figura(
        fig, id_figura="F11_02", tarea=11, titulo="Mapa de proxies: asociación con el resultado y con el hospital (2019 a 2022)",
        descripcion_alt="Gráfico de dispersión de las variables candidatas: el eje horizontal es su asociación con el hospital de egreso y el vertical su asociación con la mortalidad; "
                        "dos líneas verticales marcan el nivel moderado y el alto de proxy del hospital y el color indica si la Fase 1 la excluye, la deja condicional o la permite.",
        csv="t23_proxy_hospital.csv", anios="2019 a 2022", decision="2; 3", uso_interno=False,
        extra_pie="Se rotulan las variables con asociación moderada o alta con el hospital o con asociación pequeña o mayor con la mortalidad. ETNIA y PUEBLO_ORIGINARIO no se "
                  "dibujan: no se analizan frente a resultados.",
        antes_de_guardar=etiquetar,
    )
    return True


def figura_fuga(ctx: Contexto) -> bool:
    """F11_06: evidencia de fuga de las variables de solo fuga y condicionales frente a las referencias basales. Se arma leyendo SOLO ``t23_fuga.csv``.

    Un punto redondo (ranking, en muestra en las categóricas) y un rombo (2023, confirmación) con su IC por variable, ordenadas por el AUC de confirmación; líneas verticales en el AUC
    de 2023 de las dos referencias y en el umbral de fuga; IR_29301_MORTALIDAD rotulada "prohibida siempre (R6)". Una variable sin AUC (constante) no se dibuja. Devuelve ``False``
    si no hay variables con AUC.
    """
    t = ctx.leer_tabla("t23_fuga.csv", columnas_texto=_TEXTO_FUGA)
    cols = ["auc_ranking", "auc_ranking_bajo", "auc_ranking_alto", "auc_confirmacion", "auc_confirmacion_bajo", "auc_confirmacion_alto", "auc_ref_edad_sexo", "auc_ref_elix31"]
    v = t.filter(pl.col("tipo_fila") == "variable").with_columns([_num(pl.col(c)) for c in cols])
    v = v.filter(pl.col("auc_ranking").is_not_null() | pl.col("auc_confirmacion").is_not_null())
    if v.height == 0:
        return False
    v = v.sort(["auc_confirmacion", "auc_ranking", "variable"], descending=[True, True, False], nulls_last=True, maintain_order=True)
    umbral = float(ctx.umbrales["auc_sospecha_fuga"])
    refs = t.filter(pl.col("tipo_fila") == "referencia").with_columns(_num(pl.col("auc_confirmacion")))
    ref_edad = refs.filter(pl.col("variable") == "MODELO_EDAD_SEXO")["auc_confirmacion"].drop_nulls().to_list()
    ref_elix = refs.filter(pl.col("variable") == "MODELO_ELIX31")["auc_confirmacion"].drop_nulls().to_list()
    n = v.height
    alto_leyenda = 0.95
    # Dos filas: el gráfico y una banda angosta propia para la leyenda, fuera de los datos y sin chocar con el pie que agrega ``guardar_figura``.
    fig, (ax, hueco_leyenda) = graficos.nueva_figura_multiple(
        2, 1, figsize=(10.0, graficos.altura_filas(n, por_fila=0.45, minimo=4.5, extra=1.8) + alto_leyenda),
        gridspec_kw={"height_ratios": [graficos.altura_filas(n, por_fila=0.45, minimo=4.5, extra=1.8), alto_leyenda]},
    )
    hueco_leyenda.axis("off")
    color_rank, color_conf = graficos.OKABE_ITO[0], graficos.OKABE_ITO[3]
    etiquetas = []
    for i, f in enumerate(v.iter_rows(named=True)):
        for clave, marcador, color, desplazamiento in (("auc_ranking", "o", color_rank, -0.14), ("auc_confirmacion", "D", color_conf, 0.14)):
            valor = f[clave]
            if valor is None:
                continue
            bajo, alto = f[f"{clave}_bajo"], f[f"{clave}_alto"]
            err = None if bajo is None or alto is None else [[max(valor - bajo, 0.0)], [max(alto - valor, 0.0)]]
            ax.errorbar(valor, i + desplazamiento, xerr=err, fmt=marcador, color=color, markersize=4.5, elinewidth=0.9, capsize=2.0, zorder=3)
        sufijo = " (condicional)" if f["categoria"] == "condicional" else ""
        etiquetas.append(f"{f['variable']}{sufijo}" + ("  [prohibida siempre (R6)]" if f["prohibida_siempre"] == "si" else ""))
    ax.set_yticks(range(n))
    ax.set_yticklabels(etiquetas, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_ylim(n - 0.5, -0.5)
    manijas = [
        Line2D([], [], marker="o", color=color_rank, linestyle="", markersize=5, label="AUC 2019 a 2022 (en muestra en las categóricas)"),
        Line2D([], [], marker="D", color=color_conf, linestyle="", markersize=5, label="AUC 2023 (confirmación)"),
        Line2D([], [], color="#555555", linestyle="-.", linewidth=1.0, label="Umbral de sospecha de fuga"),
    ]
    for valores, estilo, nombre in ((ref_edad, "--", "AUC 2023 del modelo de edad y sexo"), (ref_elix, ":", "AUC 2023 del modelo de las 31 categorías")):
        if valores:
            ax.axvline(valores[0], color=graficos.OKABE_ITO[2], linestyle=estilo, linewidth=1.2, zorder=1)
            manijas.append(Line2D([], [], color=graficos.OKABE_ITO[2], linestyle=estilo, linewidth=1.2, label=nombre))
    ax.axvline(umbral, color="#555555", linestyle="-.", linewidth=1.0, zorder=1)
    # El eje x llega hasta donde llegan los intervalos (un IC de 2023 con pocos casos puede bajar de 0,45) para que ninguna barra quede cortada en el borde.
    limites = [x for c in ("auc_ranking_bajo", "auc_confirmacion_bajo", "auc_ranking", "auc_confirmacion") for x in v[c].drop_nulls().to_list()]
    ax.set_xlim(max(0.0, min(0.45, np.floor((min(limites) - 0.02) * 20.0) / 20.0)) if limites else 0.45, 1.02)
    ax.set_xticks([t for t in np.arange(0.0, 1.0001, 0.1) if ax.get_xlim()[0] - 1e-9 <= t <= 1.0 + 1e-9])
    graficos.formato_decimal(ax, "x")
    ax.set_xlabel("AUC univariado frente a la mortalidad (0,5 = sin señal)")
    ax.grid(axis="y", visible=False)
    leyenda = hueco_leyenda.legend(handles=manijas, loc="center", bbox_to_anchor=(0.5, 0.5), ncol=2, frameon=False, fontsize=7.5)
    leyenda.set_in_layout(False)
    ctx.figuras.guardar_figura(
        fig, id_figura="F11_06", tarea=11, titulo="Evidencia de fuga frente a las referencias basales (2019 a 2023)",
        descripcion_alt="Gráfico de puntos con intervalo por variable de solo fuga y condicional: un círculo para el AUC de 2019 a 2022 y un rombo para el de 2023; líneas verticales "
                        "marcan el AUC de 2023 de los modelos de edad y sexo y de las categorías de Elixhauser y el umbral de sospecha de fuga.",
        csv="t23_fuga.csv", anios="2019 a 2023", decision="2; 14", uso_interno=False,
        extra_pie="IR_29301_MORTALIDAD es una variable prohibida siempre por la regla R6, aunque su señal fuera baja. Las categóricas son optimistas en 2019 a 2022.",
    )
    return True


# FIN-sospechas
