"""Tarea 11: selección estadística de variables (CRISP-DM 2.3). Ranking, estabilidad, importancia, parámetros y figuras; orquestación (``calcular`` y ``ejecutar``), resumen,
métricas e informe.

Se rankea con 2019 a 2022 y se confirma con 2023; 2024 nunca se lee. Este módulo solo lee datos cuando construye el panel (``panel.construir_panel``, el único punto de lectura)
y no llama a ``ctx.lector``: todo lo demás recibe el panel.

- ``tabla_ranking``: señal univariada de cada candidata que analiza el resultado, en formato largo (una fila por variable y ventana). Una categórica puntúa con la tasa suavizada
  de cada nivel estimada en 2019-2022 (optimista en esa ventana, honesta en 2023); ``auc_senal`` es el AUC por validación cruzada dejando un año fuera cuando existe.
- ``tabla_estabilidad``: AUC por año con el puntaje y la dirección fijos del ranking, cambio de signo y PSI (contra 2023 y 2019 contra 2020-2022).
- ``tabla_importancia``: incremento de AUC sobre edad y sexo para toda candidata que analiza el resultado, y modelo L1 con importancia por permutación en 2023 para las que la
  Fase 1 no excluye.
- ``tabla_parametros``: las constantes técnicas de la tarea, leídas de los módulos que las definen.
- ``calcular``: encadena todo lo anterior con ``sospechas``, ``sensibilidad_elix`` y ``decision`` y devuelve cada tabla real y su versión publicable (clave ``_pub``). ``tabla_resumen``
  arma ``t23_resumen_seleccion.csv``; ``ejecutar`` escribe las tablas, registra las métricas (``registrar_metricas``) y dibuja las seis figuras; ``agregar_secciones`` agrega al informe
  las nueve secciones de la tarea (la octava es de ``decision``).

Las funciones ``tabla_*`` solo CALCULAN la tabla real (sin suprimir). La escritura (``panel.escribir_por_bloques`` con las listas ``BLOQUES_*`` de este módulo) la hace ``ejecutar``,
y ``calcular`` aplica las mismas listas para la versión publicable que alimenta el resumen, las métricas y la matriz: lo que una tabla suprimió no se vuelve a publicar en otro lado.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from . import asociacion, decision, graficos, modelos, sensibilidad_elix, sospechas, supresion
from . import estadistica as est
from . import panel as modulo_panel
from .contexto import Contexto
from .informes import cifra
from .panel import SIN_DATO, V_CONF, V_RANK

TAREA = "T11"
N_SECCIONES = 9

ESQUEMA_RANKING = {
    "variable": pl.String, "tipo": pl.String, "rol_fase1": pl.String, "ventana": pl.String, "n_filas_ventana": pl.Int64, "n": pl.Int64, "n_eventos": pl.Int64,
    "n_no_eventos": pl.Int64, "n_faltante": pl.Int64, "prop_faltante": pl.Float64, "clase_faltante": pl.String, "auc": pl.Float64, "auc_bajo": pl.Float64, "auc_alto": pl.Float64,
    "direccion": pl.String, "auc_optimista": pl.String, "auc_cv": pl.Float64, "auc_senal": pl.Float64, "auc_senal_bajo": pl.Float64, "auc_senal_alto": pl.Float64,
    "clase_senal": pl.String, "posicion": pl.Int64, "v_resultado": pl.Float64, "medida_resultado": pl.String, "info_mutua_norm": pl.Float64, "n_expuestos": pl.Int64,
    "n_eventos_expuestos": pl.Int64, "n_no_expuestos": pl.Int64, "n_eventos_no_expuestos": pl.Int64, "or_crudo": pl.Float64, "or_bajo": pl.Float64, "or_alto": pl.Float64,
    "or_relevante": pl.String, "or_faltante": pl.Float64, "or_faltante_bajo": pl.Float64, "or_faltante_alto": pl.Float64, "faltante_informativo": pl.String, "estado": pl.String,
}
ESQUEMA_ESTABILIDAD = {
    "tipo_fila": pl.String, "variable": pl.String, "tipo": pl.String, "posicion": pl.Int64, "anio": pl.Int64, "comparacion": pl.String, "n": pl.Int64, "n_eventos": pl.Int64,
    "n_base": pl.Int64, "n_nuevo": pl.Int64, "auc": pl.Float64, "auc_bajo": pl.Float64, "auc_alto": pl.Float64, "auc_optimista": pl.String, "efecto": pl.Float64, "signo": pl.String,
    "efecto_suficiente": pl.String, "cambio_signo": pl.String, "n_anios_efecto_suficiente": pl.Int64, "psi": pl.Float64, "clase_psi": pl.String,
}
ESQUEMA_IMPORTANCIA = {
    "tipo_fila": pl.String, "variable": pl.String, "tipo": pl.String, "clase_proxy": pl.String, "n_columnas_diseno": pl.Int64, "n_entrenamiento": pl.Int64,
    "n_eventos_entrenamiento": pl.Int64, "n_2023": pl.Int64, "n_eventos_2023": pl.Int64, "auc_base_2023": pl.Float64, "auc_con_variable_2023": pl.Float64,
    "incremento_auc": pl.Float64, "incremento_estado": pl.String, "en_modelo_l1": pl.String, "l1_sobrevive": pl.String, "l1_n_coef_no_cero": pl.Int64,
    "l1_coef_max_abs": pl.Float64, "importancia_permutacion": pl.Float64, "importancia_sd": pl.Float64, "importa": pl.String, "auc_2023": pl.Float64, "auc_ranking": pl.Float64,
    "l1_c": pl.Float64, "l1_convergio": pl.String,
}
ESQUEMA_PARAMETROS = {"modulo": pl.String, "parametro": pl.String, "valor": pl.String, "para_que_sirve": pl.String, "afecta": pl.String}
COLUMNAS_RANKING = list(ESQUEMA_RANKING)
COLUMNAS_ESTABILIDAD = list(ESQUEMA_ESTABILIDAD)
COLUMNAS_IMPORTANCIA = list(ESQUEMA_IMPORTANCIA)
COLUMNAS_PARAMETROS = list(ESQUEMA_PARAMETROS)
_TEXTO_RANKING = ["variable", "tipo", "rol_fase1", "ventana", "clase_faltante", "direccion", "auc_optimista", "clase_senal", "medida_resultado", "or_relevante", "faltante_informativo", "estado"]
_TEXTO_ESTABILIDAD = ["tipo_fila", "variable", "tipo", "comparacion", "auc_optimista", "signo", "efecto_suficiente", "cambio_signo", "clase_psi"]
_TEXTO_IMPORTANCIA = ["tipo_fila", "variable", "tipo", "clase_proxy", "incremento_estado", "en_modelo_l1", "l1_sobrevive", "importa", "l1_convergio"]

# Supresión por tabla (DIS 11). ``calcular`` (versión publicable) y ``ejecutar`` (escritura) usan las mismas listas.
COLS_N_RANKING = ["n_filas_ventana", "n", "n_eventos", "n_no_eventos", "n_expuestos", "n_eventos_expuestos", "n_no_expuestos", "n_eventos_no_expuestos"]
DEPENDIENTES_RANKING = [
    "n_faltante", "prop_faltante", "clase_faltante", "auc", "auc_bajo", "auc_alto", "auc_cv", "auc_senal", "auc_senal_bajo", "auc_senal_alto", "clase_senal", "posicion", "v_resultado",
    "info_mutua_norm", "or_crudo", "or_bajo", "or_alto", "or_relevante", "or_faltante", "or_faltante_bajo", "or_faltante_alto", "faltante_informativo",
]
# El conteo de faltantes tiene su propio bloque (NIT 7): con 1 a 9 faltantes se anulan ``n``, la proporción de faltantes y todo el bloque de faltante informativo, para que
# ``n_filas_ventana - n`` no revele el conteo; un conteo ajeno no pierde por eso la fila entera.
BLOQUES_RANKING = [
    (COLS_N_RANKING, DEPENDIENTES_RANKING),
    (["n_faltante"], ["n", "prop_faltante", "clase_faltante", "faltante_informativo", "or_faltante", "or_faltante_bajo", "or_faltante_alto"]),
]
BLOQUES_ESTABILIDAD = [(["n", "n_eventos", "n_base", "n_nuevo"], ["auc", "auc_bajo", "auc_alto", "efecto", "signo", "efecto_suficiente", "psi", "clase_psi"])]
BLOQUES_IMPORTANCIA = [
    (["n_entrenamiento", "n_eventos_entrenamiento"], ["l1_sobrevive", "l1_n_coef_no_cero", "l1_coef_max_abs", "auc_ranking", "l1_c"]),
    (
        ["n_2023", "n_eventos_2023"],
        ["auc_base_2023", "auc_con_variable_2023", "incremento_auc", "importancia_permutacion", "importancia_sd", "importa", "auc_2023"],
    ),
]


def _finito(x: Any) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def _pub(x: Any) -> float | None:
    """Valor redondeado a los decimales con que se publican las tablas (``None`` si falta): contra él se comparan los umbrales."""
    x = _finito(x)
    return None if x is None else float(est.redondear_publicado(x))


def clase_senal(auc_senal: float | None, u) -> str | None:
    """"alta" desde ``auc_senal_alta``, "media" desde ``auc_senal_media``, "débil" desde ``auc_senal_baja`` (el tramo sin nombre en el protocolo) y "baja" bajo eso."""
    v = _pub(auc_senal)
    if v is None:
        return None
    if v >= float(u["auc_senal_alta"]):
        return "alta"
    if v >= float(u["auc_senal_media"]):
        return "media"
    if v >= float(u["auc_senal_baja"]):
        return "débil"
    return "baja"


def clase_faltante(prop: float | None, u) -> str:
    """"alto" desde ``faltante_alto``, "alerta" desde ``faltante_alerta``, "bajo" en otro caso y "no evaluable" si la proporción falta. Informativa: la regla CALIDAD usa solo ``faltante_alto``."""
    v = _pub(prop)
    if v is None:
        return "no evaluable"
    if v >= float(u["faltante_alto"]):
        return "alto"
    if v >= float(u["faltante_alerta"]):
        return "alerta"
    return "bajo"


# ==================================================================================================================== ranking
def _faltantes(panel: modulo_panel.Panel, nombre: str, anios: Sequence[int]) -> np.ndarray:
    """Marca de faltante por fila de la ventana: nulo en una numérica o binaria; nivel "(sin dato)" en una categórica (definición del panel)."""
    tipo, x = modelos.valores_del_panel(panel, nombre, anios)
    if tipo == "cat":
        niveles = panel.niveles[nombre]
        return (x == niveles.index(SIN_DATO)) if SIN_DATO in niveles else np.zeros(x.size, dtype=bool)
    return np.isnan(x)


def _codigos_info_mutua(panel: modulo_panel.Panel, nombre: str, anios: Sequence[int]) -> np.ndarray:
    """Códigos para la información mutua: deciles de la ventana en las numéricas (el nulo es un bin propio), el valor en las binarias (el nulo es un nivel propio) y los niveles."""
    tipo, x = modelos.valores_del_panel(panel, nombre, anios)
    if tipo == "num":
        return est.asignar_bins(x, est.bordes_deciles(x))
    if tipo == "bin":
        return np.where(np.isnan(x), 2, x).astype(np.int64)
    return x.astype(np.int64)


def _cuadro_binaria(panel: modulo_panel.Panel, nombre: str, anios: Sequence[int], y: np.ndarray) -> tuple[int, int, int, int]:
    """(a, b, c, d) = expuestos con evento, expuestos sin evento, no expuestos con evento y no expuestos sin evento, entre las filas con valor."""
    _, x = modelos.valores_del_panel(panel, nombre, anios)
    ok = ~np.isnan(x)
    a, b = int(((x == 1) & (y == 1) & ok).sum()), int(((x == 1) & (y == 0) & ok).sum())
    c, d = int(((x == 0) & (y == 1) & ok).sum()), int(((x == 0) & (y == 0) & ok).sum())
    return a, b, c, d


def tabla_ranking(ctx: Contexto, panel: modulo_panel.Panel) -> pl.DataFrame:
    """``t23_ranking_variables.csv`` real (sin suprimir), formato largo: una fila por variable y ventana ("2019-2022" y, si hay confirmación, "2023").

    Variables: ``uso == "candidata"``, ``analiza_resultado`` y ``evaluable != "no"`` (ni ETNIA ni PUEBLO_ORIGINARIO). Puntaje: el valor mismo en numéricas y binarias; en una
    categórica, la tasa suavizada por nivel estimada con la ventana de ranking. ``direccion`` es "+" si el AUC bruto de la ventana de ranking es al menos 0,5 y "-" si no; ``auc`` es el
    AUC orientado con su IC de Hanley y McNeil (en 2023 con la dirección del ranking: puede quedar bajo 0,5, y eso es una señal de inestabilidad, no un error). En una categórica
    ``auc`` es optimista (``auc_optimista = "si"``) y ``auc_senal`` es el AUC por validación cruzada ``auc_cv``, con su propio IC (``auc_senal_bajo`` y ``auc_senal_alto``); sin ``auc_cv``
    (un solo año de ranking) repite ``auc``. ``prop_faltante`` es una fracción de 0 a 1. La fila de 2023 trae solo los conteos, el faltante, el AUC con su IC y ``v_resultado``.
    ``estado``: "ok", "constante", "sin ambos desenlaces" o "no evaluable en la ventana"; si no es "ok" no hay cifras de AUC ni asociación.
    """
    u = ctx.umbrales
    z = float(u["z_95"])
    ranking, conf = list(panel.anios_ranking), panel.anio_confirmacion
    columnas_rank = asociacion.preparar_columnas(panel, ranking)
    columnas_conf = asociacion.preparar_columnas(panel, [conf]) if conf is not None else None
    y_rank = modelos.desenlace(panel, ranking)
    y_conf = modelos.desenlace(panel, [conf]) if conf is not None else None
    min_falt = float(u["faltante_informativo_min_prop"])
    filas: list[dict[str, Any]] = []
    for nombre in modelos.variables_ranking(panel):
        v = panel.variables[nombre]
        ajuste = modelos.ajustar_univariado(panel, nombre)
        cv = modelos.auc_cv_univariado(panel, ajuste, z)
        estado, r = modelos.medir_ventana(panel, ajuste, ranking, z)
        falta = _faltantes(panel, nombre, ranking)
        n_filas, n_falt = int(falta.size), int(falta.sum())
        prop = n_falt / n_filas if n_filas else None
        ok = estado == "ok"
        auc, bajo, alto = (_finito(r.auc), _finito(r.ic_bajo), _finito(r.ic_alto)) if ok else (None, None, None)
        optimista = "si" if v.tipo == "cat" else "no"
        auc_cv = _finito(cv.auc) if (ok and cv is not None) else (auc if v.tipo != "cat" else None)
        usa_cv = ok and v.tipo == "cat" and cv is not None and auc_cv is not None
        auc_senal = auc_cv if auc_cv is not None else auc
        senal_bajo, senal_alto = ((_finito(cv.ic_bajo), _finito(cv.ic_alto)) if usa_cv else (bajo, alto))
        v_res, medida, _ = asociacion.con_resultado(columnas_rank[nombre], y_rank)
        fila: dict[str, Any] = {c: None for c in ESQUEMA_RANKING}
        fila.update({
            "variable": nombre, "tipo": v.tipo, "rol_fase1": v.rol_fase1, "ventana": V_RANK, "n_filas_ventana": n_filas, "n": r.n, "n_eventos": r.n_eventos, "n_no_eventos": r.n_no_eventos,
            "n_faltante": n_falt, "prop_faltante": prop, "clase_faltante": clase_faltante(prop, u), "auc": auc, "auc_bajo": bajo, "auc_alto": alto,
            "direccion": ajuste.direccion or None, "auc_optimista": optimista, "auc_cv": auc_cv, "auc_senal": auc_senal, "auc_senal_bajo": senal_bajo, "auc_senal_alto": senal_alto,
            "clase_senal": clase_senal(auc_senal, u), "v_resultado": _finito(v_res) if ok else None, "medida_resultado": medida, "estado": estado,
            "info_mutua_norm": _finito(est.informacion_mutua_normalizada(_codigos_info_mutua(panel, nombre, ranking), y_rank)) if ok else None,
        })
        if v.tipo == "bin":
            a, b, c, d = _cuadro_binaria(panel, nombre, ranking, y_rank)
            fila.update({"n_expuestos": a + b, "n_eventos_expuestos": a, "n_no_expuestos": c + d, "n_eventos_no_expuestos": c})
            if ok and a + b > 0 and c + d > 0:
                o, ob, oa, _ = est.or_woolf(a, b, c, d, z)
                fila.update({"or_crudo": _finito(o), "or_bajo": _finito(ob), "or_alto": _finito(oa), "or_relevante": modelos.or_es_relevante(_finito(o), _finito(ob), _finito(oa), float(u["or_relevante"]))})
        # faltante informativo: el cuadro faltante contra desenlace, solo con proporción y celdas suficientes
        celdas = [int((falta & (y_rank == 1)).sum()), int((falta & (y_rank == 0)).sum()), int((~falta & (y_rank == 1)).sum()), int((~falta & (y_rank == 0)).sum())]
        if prop is not None and _pub(prop) >= min_falt and min(celdas) >= ctx.celda_minima:
            o, ob, oa, _ = est.or_woolf(*celdas, z)
            fila.update({
                "or_faltante": _finito(o), "or_faltante_bajo": _finito(ob), "or_faltante_alto": _finito(oa),
                "faltante_informativo": "si" if modelos.or_es_relevante(_finito(o), _finito(ob), _finito(oa), float(u["faltante_informativo_or"])) == "si" else "no",
            })
        else:
            fila["faltante_informativo"] = "no evaluable"
        filas.append(fila)
        if conf is not None:
            estado23, r23 = modelos.medir_ventana(panel, ajuste, [conf], z)
            falta23 = _faltantes(panel, nombre, [conf])
            n23, f23 = int(falta23.size), int(falta23.sum())
            ok23 = estado23 == "ok"
            v23, medida23, _ = asociacion.con_resultado(columnas_conf[nombre], y_conf)
            f = {c: None for c in ESQUEMA_RANKING}
            f.update({
                "variable": nombre, "tipo": v.tipo, "rol_fase1": v.rol_fase1, "ventana": V_CONF, "n_filas_ventana": n23, "n": r23.n, "n_eventos": r23.n_eventos,
                "n_no_eventos": r23.n_no_eventos, "n_faltante": f23, "prop_faltante": f23 / n23 if n23 else None, "auc": _finito(r23.auc) if ok23 else None,
                "auc_bajo": _finito(r23.ic_bajo) if ok23 else None, "auc_alto": _finito(r23.ic_alto) if ok23 else None, "v_resultado": _finito(v23) if ok23 else None,
                "medida_resultado": medida23, "estado": estado23,
            })
            filas.append(f)
    # posición: orden por auc_senal publicado (descendente) y, en empate, por nombre
    ordenadas = sorted((f for f in filas if f["ventana"] == V_RANK and f["auc_senal"] is not None), key=lambda f: (-_pub(f["auc_senal"]), f["variable"]))
    for k, f in enumerate(ordenadas, start=1):
        f["posicion"] = k
    return pl.DataFrame(filas, schema=ESQUEMA_RANKING)


# ================================================================================================================== estabilidad
def _codigos_psi(panel: modulo_panel.Panel, nombre: str) -> tuple[np.ndarray, int]:
    """Códigos de bin de TODAS las filas del panel y número de bins: deciles de la ventana de ranking en las numéricas (el nulo es un bin propio), niveles en las categóricas y
    valores 0, 1 y nulo en las binarias. Son los mismos bins para las dos comparaciones."""
    todos = sorted(set(panel.datos["anio"].to_list()))
    tipo, x = modelos.valores_del_panel(panel, nombre, todos)
    if tipo == "num":
        bordes = est.bordes_deciles(modelos.valores_del_panel(panel, nombre, panel.anios_ranking)[1])
        return est.asignar_bins(x, bordes), int(bordes.size) + 2
    if tipo == "bin":
        return np.where(np.isnan(x), 2, x).astype(np.int32), 3
    return x.astype(np.int32), len(panel.niveles[nombre])


def tabla_estabilidad(ctx: Contexto, panel: modulo_panel.Panel, ranking: pl.DataFrame) -> pl.DataFrame:
    """``t23_estabilidad.csv`` real (sin suprimir), formato largo por ``tipo_fila``.

    ``auc_anual``: una fila por variable del ranking y año del panel (2019 a 2023) con el puntaje y la dirección que fijó la ventana de ranking (``n``, ``n_eventos``, ``auc`` y su IC,
    ``auc_optimista``: "si" en las categóricas de 2019 a 2022, cuyo puntaje se estimó con esos años, "no" en 2023 y en numéricas y binarias; ``efecto = |auc - 0,5|``, ``signo`` y
    ``efecto_suficiente`` con ``estabilidad_efecto_minimo``). ``resumen``: ``cambio_signo`` ("si" si entre los años con efecto suficiente hay signos "+" y "-", "no" si hay al menos dos
    años y todos coinciden, "no evaluable" si hay menos de dos), ``n_anios_efecto_suficiente`` y la ``posicion`` del ranking. ``psi``: dos filas por variable candidata (también las
    que no analizan el resultado): "2019-2022 contra 2023" y "2019 contra 2020-2022" (evidencia de H17a), con deciles de la ventana de ranking y los mismos bins en ambas.
    """
    u = ctx.umbrales
    z = float(u["z_95"])
    minimo = float(u["estabilidad_efecto_minimo"])
    anios = sorted(set(panel.datos["anio"].to_list()))
    posicion: dict[str, int | None] = {}
    if ranking is not None and ranking.height:
        posicion = {f[0]: f[1] for f in ranking.filter(pl.col("ventana") == V_RANK).select("variable", "posicion").iter_rows()}
    vacia = {c: None for c in ESQUEMA_ESTABILIDAD}
    anuales: list[dict[str, Any]] = []
    resumenes: list[dict[str, Any]] = []
    for nombre in modelos.variables_ranking(panel):
        v = panel.variables[nombre]
        ajuste = modelos.ajustar_univariado(panel, nombre)
        con_efecto: list[str] = []
        for anio in anios:
            estado, r = modelos.medir_ventana(panel, ajuste, [anio], z)
            ok = estado == "ok"
            auc = _pub(r.auc) if ok else None
            efecto = _pub(abs(auc - 0.5)) if auc is not None else None
            signo = None if auc is None else ("+" if auc > 0.5 else ("-" if auc < 0.5 else "0"))
            suficiente = None if efecto is None else ("si" if efecto >= minimo else "no")
            if suficiente == "si":
                con_efecto.append(signo)
            fila = dict(vacia)
            fila.update({
                "tipo_fila": "auc_anual", "variable": nombre, "tipo": v.tipo, "anio": int(anio), "n": r.n, "n_eventos": r.n_eventos, "auc": _finito(r.auc) if ok else None,
                "auc_bajo": _finito(r.ic_bajo) if ok else None, "auc_alto": _finito(r.ic_alto) if ok else None,
                "auc_optimista": "si" if (v.tipo == "cat" and anio in panel.anios_ranking) else "no", "efecto": efecto, "signo": signo, "efecto_suficiente": suficiente,
            })
            anuales.append(fila)
        if "+" in con_efecto and "-" in con_efecto:
            cambio = "si"
        elif len(con_efecto) >= 2:
            cambio = "no"
        else:
            cambio = "no evaluable"
        fila = dict(vacia)
        fila.update({"tipo_fila": "resumen", "variable": nombre, "tipo": v.tipo, "posicion": posicion.get(nombre), "cambio_signo": cambio, "n_anios_efecto_suficiente": len(con_efecto)})
        resumenes.append(fila)
    psi_filas: list[dict[str, Any]] = []
    anio_fila = panel.datos["anio"].to_numpy()
    primero = min(panel.anios_ranking) if panel.anios_ranking else None
    resto = [a for a in panel.anios_ranking if a != primero]
    conf = panel.anio_confirmacion
    for nombre, v in panel.variables.items():
        if v.uso != "candidata" or v.evaluable == "no" or nombre not in panel.datos.columns:
            continue
        codigos, n_codigos = _codigos_psi(panel, nombre)
        comparaciones = (
            ("2019-2022 contra 2023", np.isin(anio_fila, panel.anios_ranking), (anio_fila == conf) if conf is not None else np.zeros(anio_fila.size, dtype=bool)),
            ("2019 contra 2020-2022", (anio_fila == primero) if primero is not None else np.zeros(anio_fila.size, dtype=bool), np.isin(anio_fila, resto)),
        )
        for etiqueta, base, nuevo in comparaciones:
            n_base, n_nuevo = int(base.sum()), int(nuevo.sum())
            valor = _finito(est.psi_de_codigos(codigos[base], codigos[nuevo], n_codigos)) if n_base and n_nuevo else None
            fila = dict(vacia)
            fila.update({
                "tipo_fila": "psi", "variable": nombre, "tipo": v.tipo, "comparacion": etiqueta, "n_base": n_base, "n_nuevo": n_nuevo, "psi": valor,
                "clase_psi": est.clasificar_psi(float("nan") if valor is None else _pub(valor), float(u["psi_moderado"]), float(u["psi_alto"])),
            })
            psi_filas.append(fila)
    return pl.DataFrame(anuales + resumenes + psi_filas, schema=ESQUEMA_ESTABILIDAD)


# ================================================================================================================== importancia
def tabla_importancia(ctx: Contexto, panel: modulo_panel.Panel, referencias: Mapping[str, Mapping[str, Any]], clase_proxy: Mapping[str, str]) -> pl.DataFrame:
    """``t23_importancia.csv`` real (sin suprimir): una fila ``variable`` por candidata que analiza el resultado y una fila ``referencia`` por modelo.

    El incremento de AUC en 2023 sobre edad y sexo se calcula para TODA candidata que analiza el resultado, incluidas las que la Fase 1 excluye (llevan ``en_modelo_l1 = "no"``),
    porque es la única forma de evaluar la regla BAJA_SENAL. El modelo L1 (con todos los niveles de las categóricas) y la importancia por permutación en 2023 cubren solo las de
    ``en_modelo_l1 = "si"``: las que la Fase 1 no excluye y que tienen alguna columna en el diseño. ``importa`` es "si" si el valor redondeado alcanza ``importancia_permutacion_min``.
    ``clase_proxy`` es la de ``t23_proxy_hospital.csv``: advierte qué variables del L1 están cerca de ser un proxy del hospital. Las filas de referencia son MODELO_EDAD_SEXO,
    MODELO_ELIX31 y MODELO_L1 (con ``l1_c`` y ``l1_convergio``, marca que no se anula).
    """
    u = ctx.umbrales
    semilla = ctx.semilla
    ranking, conf = list(panel.anios_ranking), panel.anio_confirmacion
    base = modelos.ajuste_base(panel, semilla)
    candidatas = modelos.variables_ranking(panel)
    en_l1 = modelos.variables_modelo(panel)
    espec = modelos.especificar_diseno(panel, en_l1, ranking, semilla, omitir_nivel_base=False)
    X, y, grupos = modelos.aplicar_diseno(espec, panel, ranking, entrenamiento=True)
    l1 = modelos.ajustar_l1(X, y, espec.columnas, grupos, semilla)
    auc_l1_rank = modelos.auc_l1(l1, X, y)
    auc_l1_2023, n_2023, ev_2023, perm = None, 0, 0, pl.DataFrame(schema=modelos.ESQUEMA_PERMUTACION)
    if conf is not None:
        X2, y2, g2 = modelos.aplicar_diseno(espec, panel, [conf])
        n_2023, ev_2023 = int(y2.size), int(y2.sum())
        auc_l1_2023 = modelos.auc_l1(l1, X2, y2)
        perm = modelos.importancia_permutacion(X2, y2, l1, g2, semilla=semilla)
    importancia = {f["variable"]: f for f in perm.iter_rows(named=True)}
    minimo = float(u["importancia_permutacion_min"])
    filas: list[dict[str, Any]] = []
    for nombre in candidatas:
        v = panel.variables[nombre]
        inc = modelos.incremento_auc(panel, nombre, semilla, base=base)
        columnas = grupos.get(nombre, [])
        dentro = nombre in en_l1 and len(columnas) > 0
        fila = {c: None for c in ESQUEMA_IMPORTANCIA}
        fila.update({
            "tipo_fila": "variable", "variable": nombre, "tipo": v.tipo, "clase_proxy": clase_proxy.get(nombre), "n_columnas_diseno": inc["n_columnas_diseno"],
            "n_entrenamiento": inc["n_entrenamiento"], "n_eventos_entrenamiento": inc["n_eventos_entrenamiento"], "n_2023": inc["n_2023"], "n_eventos_2023": inc["n_eventos_2023"],
            "auc_base_2023": _finito(inc["auc_base_2023"]), "auc_con_variable_2023": _finito(inc["auc_con_variable_2023"]), "incremento_auc": _finito(inc["incremento_auc"]),
            "incremento_estado": inc["incremento_estado"], "en_modelo_l1": "si" if dentro else "no",
        })
        if dentro:
            if l1.estado != "ok":
                fila["l1_sobrevive"] = l1.estado
            else:
                coef = l1.coef[columnas]
                fila.update({"l1_sobrevive": "si" if l1.sobrevive_por_variable[nombre] else "no", "l1_n_coef_no_cero": int(np.count_nonzero(coef)), "l1_coef_max_abs": float(np.abs(coef).max())})
                if nombre in importancia:
                    valor = importancia[nombre]
                    fila.update({
                        "importancia_permutacion": _finito(valor["importancia"]), "importancia_sd": _finito(valor["desviacion"]),
                        "importa": "si" if (_pub(valor["importancia"]) is not None and _pub(valor["importancia"]) >= minimo) else "no",
                    })
        filas.append(fila)
    for nombre in ("MODELO_EDAD_SEXO", "MODELO_ELIX31"):
        if nombre not in referencias:
            continue
        r = referencias[nombre]
        fila = {c: None for c in ESQUEMA_IMPORTANCIA}
        fila.update({
            "tipo_fila": "referencia", "variable": nombre, "tipo": "modelo", "n_columnas_diseno": r.get("n_columnas"), "n_entrenamiento": r.get("n_entrenamiento"),
            "n_eventos_entrenamiento": r.get("n_eventos_entrenamiento"), "n_2023": r.get("n_2023") if conf is not None else None,
            "n_eventos_2023": r.get("n_eventos_2023") if conf is not None else None, "auc_2023": _finito(r.get("auc_2023")), "auc_ranking": _finito(r.get("auc_ranking")),
        })
        filas.append(fila)
    fila = {c: None for c in ESQUEMA_IMPORTANCIA}
    nz = int(np.count_nonzero(l1.coef)) if l1.estado == "ok" else None
    fila.update({
        "tipo_fila": "referencia", "variable": "MODELO_L1", "tipo": "modelo", "n_columnas_diseno": len(espec.columnas), "n_entrenamiento": int(y.size), "n_eventos_entrenamiento": int(y.sum()),
        "n_2023": n_2023 if conf is not None else None, "n_eventos_2023": ev_2023 if conf is not None else None, "l1_sobrevive": l1.estado if l1.estado != "ok" else None,
        "l1_n_coef_no_cero": nz, "l1_coef_max_abs": float(np.abs(l1.coef).max()) if l1.estado == "ok" and l1.coef.size else None, "auc_2023": _finito(auc_l1_2023),
        "auc_ranking": _finito(auc_l1_rank), "l1_c": _finito(l1.c_elegido), "l1_convergio": ("si" if l1.convergio else "no") if l1.estado == "ok" else None,
    })
    filas.append(fila)
    return pl.DataFrame(filas, schema=ESQUEMA_IMPORTANCIA)


# =================================================================================================================== parámetros
def _texto_valor(valor: Any) -> str:
    """Valor en texto sin notación científica y con coma decimal: cuatro cifras significativas en los decimales, enteros tal cual."""
    if isinstance(valor, (bool, np.bool_)):
        return "si" if valor else "no"
    if isinstance(valor, (int, np.integer)):
        return str(int(valor))
    x = float(f"{float(valor):.4g}")
    return np.format_float_positional(x, trim="-", unique=True).replace(".", ",")


def tabla_parametros() -> pl.DataFrame:
    """``t23_parametros_seleccion.csv``: una fila por constante técnica de la Tarea 11 (más ``relleno_nulos_modelos``), leyendo su valor de los módulos que las definen.

    Ninguna es un umbral del protocolo (``config/umbrales.toml`` no cambia). No incluye la semilla (un número de ocho dígitos en una columna de texto activaría el escáner de
    identificadores). ``valor`` va en texto sin notación científica y con coma decimal; las listas se separan con "; ".
    """
    filas = [
        ("estadistica", "SUAVIZADO_M", est.SUAVIZADO_M, "Pseudoegresos con la tasa global que se suman a cada nivel de una categórica para estimar su tasa", "auc_cv, auc_senal, el representante de cada cluster, BAJA_SENAL y FUGA para las categóricas"),
        ("estadistica", "EPS_COLINEALIDAD", est.EPS_COLINEALIDAD, "Criterio de colinealidad exacta: una variable es combinación lineal de las demás si su R cuadrado supera uno menos este valor", "VIF nulo con colinealidad exacta"),
        ("estadistica", "DECIMALES_PUBLICADOS", est.DECIMALES_PUBLICADOS, "Decimales con que se escriben los flotantes y con que se comparan los umbrales", "Todas las reglas del árbol de decisión y las marcas de las tablas"),
        ("estadistica", "SEPARACION_COEF_MAX", est.SEPARACION_COEF_MAX, "Un coeficiente de la regresión logística con valor absoluto mayor se declara con separación", "Solo anula el OR de ese coeficiente, nunca un AUC"),
        ("estadistica", "SEPARACION_SE_MAX", est.SEPARACION_SE_MAX, "Un coeficiente con error estándar mayor se declara con separación", "Solo anula el OR de ese coeficiente, nunca un AUC"),
        ("modelos", "RIDGE_AUC", modelos.RIDGE_AUC, "Ridge de los diseños que entregan un AUC (equivale a C igual a uno de scikit-learn)", "Incremento de AUC, referencias y modelo base"),
        ("modelos", "RIDGE_WALD", modelos.RIDGE_WALD, "Ridge de los diseños que entregan un OR con error de Wald (cero: el OR no tiene sesgo de penalización)", "OR ajustado y OR de estadía larga de Elixhauser"),
        ("modelos", "TOPE_FILAS_MODELOS", "sin tope" if modelos.TOPE_FILAS_MODELOS is None else modelos.TOPE_FILAS_MODELOS, "Tope de filas de entrenamiento de los modelos (muestra reproducible si la ventana lo supera)", "Referencias, incremento de AUC y modelo L1"),
        ("modelos", "TOPE_FILAS_L1", "sin tope" if modelos.TOPE_FILAS_L1 is None else modelos.TOPE_FILAS_L1, "Filas con que se ajusta el modelo L1 (muestra reproducible); su AUC e importancia se evalúan con todas las filas", "Modelo L1"),
        ("modelos", "CS_L1", "; ".join(_texto_valor(c) for c in modelos.CS_L1), "Valores de C (inverso de la penalización) que explora el modelo L1", "Variables que sobreviven en el modelo L1 y su importancia"),
        ("modelos", "PLIEGUES_CV", modelos.PLIEGUES_CV, "Pliegues de la validación cruzada del modelo L1 (el valor del encargo)", "C elegido y variables que sobreviven en el modelo L1"),
        ("modelos", "REPETICIONES_PERMUTACION", modelos.REPETICIONES_PERMUTACION, "Repeticiones de la importancia por permutación de cada variable", "Importancia por permutación y su desviación"),
        ("sospechas", "MIN_FILAS_HOSPITAL", sospechas.MIN_FILAS_HOSPITAL, "Un hospital entra a las medidas de variación entre hospitales solo con al menos este número de filas válidas", "Variación de los ingresos a cuidados especiales y cardinalidad de ESPECIALIDAD_MEDICA por hospital"),
        ("sospechas", "ETIQUETA_V_RESULTADO", sospechas.ETIQUETA_V_RESULTADO, "Umbral de visualización de F11_02: se rotulan los puntos con asociación con el resultado de al menos este valor", "Solo las etiquetas de la figura F11_02"),
    ]
    filas = [{"modulo": m, "parametro": p, "valor": v if isinstance(v, str) else _texto_valor(v), "para_que_sirve": s, "afecta": a} for m, p, v, s, a in filas]
    filas.append({
        "modulo": "modelos", "parametro": "relleno_nulos_modelos", "valor": "mediana del entrenamiento más indicador de nulos",
        "para_que_sirve": "Los nulos de numéricas y binarias se rellenan con la mediana de entrenamiento (cero en binarias) más una columna indicadora, solo dentro de los modelos "
                          "exploratorios; la regla R5 de la Fase 1 lo admite para las líneas base y el panel conserva los nulos",
        "afecta": "Modelo base, referencias, incremento de AUC y modelo L1",
    })
    return pl.DataFrame(filas, schema=ESQUEMA_PARAMETROS)


# ===================================================================================================================== figuras
def _num(serie: pl.Series) -> pl.Series:
    return serie.cast(pl.Float64, strict=False)


def _ent(serie: pl.Series) -> pl.Series:
    """Entero (o nulo) de una columna que ``leer_tabla`` pudo inferir como texto por estar vacía."""
    return serie.cast(pl.Int64, strict=False)


def _coma(x: float, decimales: int = 2) -> str:
    return f"{x:.{decimales}f}".replace(".", ",")


COLOR_TIPO = {"num": graficos.OKABE_ITO[0], "bin": graficos.OKABE_ITO[1], "cat": graficos.OKABE_ITO[2]}
NOMBRE_TIPO = {"num": "Numérica", "bin": "Binaria", "cat": "Categórica"}
COLOR_PSI = {"estable": graficos.OKABE_ITO[2], "moderado": graficos.OKABE_ITO[1], "alto": graficos.OKABE_ITO[3]}


def figura_ranking(ctx: Contexto) -> bool:
    """F11_01: señal univariada frente a la mortalidad. Se arma leyendo SOLO ``t23_ranking_variables.csv``.

    Barras horizontales desde 0,5 hasta ``auc_senal`` ordenadas por ``posicion``, con barras de error finas de ``auc_senal_bajo`` a ``auc_senal_alto`` (el IC del número que la barra
    dibuja; en las categóricas no es el de ``auc``, que es en muestra), un rombo con el AUC de 2023 donde no esté suprimido, color por tipo y trama diagonal en las variables que la Fase
    1 excluye. Líneas verticales en los umbrales de señal débil, media y alta y de sospecha de fuga con su rótulo arriba. Devuelve ``False`` si ninguna variable tiene señal.
    """
    t = ctx.leer_tabla("t23_ranking_variables.csv", columnas_texto=_TEXTO_RANKING)
    t = t.with_columns([_num(pl.col(c)) for c in ("auc", "auc_cv", "auc_senal", "auc_senal_bajo", "auc_senal_alto", "posicion")])
    rank = t.filter((pl.col("ventana") == V_RANK) & pl.col("auc_senal").is_not_null()).sort(["posicion", "variable"], maintain_order=True)
    if rank.height == 0:
        return False
    u = ctx.umbrales
    confirmacion = {f["variable"]: f["auc"] for f in t.filter(pl.col("ventana") == V_CONF).iter_rows(named=True) if f["auc"] is not None}
    n = rank.height
    alto_leyenda = 0.95
    altura = graficos.altura_filas(n, por_fila=0.17, minimo=4.5, extra=1.8)
    fig, (ax, hueco) = graficos.nueva_figura_multiple(2, 1, figsize=(10.0, altura + alto_leyenda), gridspec_kw={"height_ratios": [altura, alto_leyenda]})
    hueco.axis("off")
    etiquetas = []
    for i, f in enumerate(rank.iter_rows(named=True)):
        excluida = f["rol_fase1"] in decision.ROLES_EXCLUYENTES
        ax.barh(i, f["auc_senal"] - 0.5, left=0.5, height=0.72, color=COLOR_TIPO.get(f["tipo"], "#888888"), edgecolor="white" if excluida else "none", hatch="///" if excluida else None, linewidth=0.0, zorder=2)
        if f["auc_senal_bajo"] is not None and f["auc_senal_alto"] is not None:
            ax.errorbar(f["auc_senal"], i, xerr=[[max(f["auc_senal"] - f["auc_senal_bajo"], 0.0)], [max(f["auc_senal_alto"] - f["auc_senal"], 0.0)]], fmt="none", ecolor="#222222", elinewidth=0.7, capsize=1.4, zorder=3)
        if f["variable"] in confirmacion:
            ax.plot(confirmacion[f["variable"]], i, marker="D", color="black", markersize=3.4, linestyle="none", zorder=4)
        etiquetas.append(f"{f['variable']}†" if (f["auc_optimista"] == "si" and f["auc_cv"] is None) else f["variable"])
    ax.set_yticks(range(n))
    ax.set_yticklabels(etiquetas, fontsize=6.5)
    ax.set_ylim(n - 0.5, -0.5)
    valores = rank["auc_senal_bajo"].drop_nulls().to_list() + list(confirmacion.values())
    minimo = min([0.5, *valores]) if valores else 0.5
    ax.set_xlim(min(0.45, np.floor((minimo - 0.01) * 20.0) / 20.0), 1.01)
    ax.axvline(0.5, color="black", linewidth=0.9, zorder=1)
    for clave in ("auc_senal_baja", "auc_senal_media", "auc_senal_alta", "auc_sospecha_fuga"):
        x = float(u[clave])
        ax.axvline(x, color="#666666", linestyle=":", linewidth=0.9, zorder=1)
        ax.text(x, 1.0, _coma(x), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color="#333333")
    graficos.formato_decimal(ax, "x")
    ax.set_xlabel("AUC (0,5 = sin señal)")
    ax.grid(axis="y", visible=False)
    manijas = [Patch(facecolor=COLOR_TIPO[k], label=NOMBRE_TIPO[k]) for k in ("num", "bin", "cat")]
    manijas += [
        Patch(facecolor="#BBBBBB", edgecolor="white", hatch="///", label="Excluida por la Fase 1"),
        Line2D([], [], marker="D", color="black", linestyle="none", markersize=4, label="AUC de 2023 (confirmación)"),
        Line2D([], [], color="#222222", linewidth=0.8, marker="|", label="Intervalo de confianza del AUC de la barra"),
    ]
    leyenda = hueco.legend(handles=manijas, loc="center", bbox_to_anchor=(0.5, 0.5), ncol=3, frameon=False, fontsize=7.5)
    leyenda.set_in_layout(False)
    pie = ("Líneas punteadas: umbrales de señal débil, media y alta y de sospecha de fuga. En las categóricas la barra es el AUC por validación cruzada dejando un año fuera. "
           "† categórica sin validación cruzada (AUC en muestra, optimista).")
    ctx.figuras.guardar_figura(
        fig, id_figura="F11_01", tarea=11, titulo="Señal univariada frente a la mortalidad (2019 a 2022 y 2023)",
        descripcion_alt="Barras horizontales con el AUC univariado de cada variable frente a la mortalidad en 2019 a 2022, desde el valor 0,5 que indica ausencia de señal, ordenadas de mayor a "
                        "menor, con su intervalo de confianza y un rombo con el AUC de 2023; el color indica el tipo de variable, la trama las excluidas por la Fase 1 y líneas "
                        "verticales los umbrales de señal y de sospecha de fuga.",
        csv="t23_ranking_variables.csv", anios="2019 a 2022 y 2023", decision="2; 14", uso_interno=False, extra_pie=pie,
    )
    return True


def figura_estabilidad(ctx: Contexto) -> bool:
    """F11_03: estabilidad por año de la señal de cada variable. Se arma leyendo SOLO ``t23_estabilidad.csv``.

    Izquierda: mapa de calor de ``auc - 0,5`` por variable (ordenadas por la posición del ranking) y año, con la escala divergente simétrica; las celdas suprimidas o sin dato, en
    gris; un asterisco en la etiqueta de las variables con cambio de signo y un punto pequeño en las celdas ``auc_optimista = "si"`` (categóricas de 2019 a 2022, cuyo puntaje se estimó en
    esos años). Derecha: dos columnas de PSI coloreadas por clase (estable, moderado o alto) con el valor anotado. Devuelve ``False`` si la tabla no tiene filas de AUC por año.
    """
    t = ctx.leer_tabla("t23_estabilidad.csv", columnas_texto=_TEXTO_ESTABILIDAD)
    t = t.with_columns([_num(pl.col(c)) for c in ("posicion", "anio", "auc", "psi")])
    anual = t.filter(pl.col("tipo_fila") == "auc_anual")
    if anual.height == 0:
        return False
    resumen = t.filter(pl.col("tipo_fila") == "resumen")
    orden = resumen.sort(["posicion", "variable"], nulls_last=True, maintain_order=True)["variable"].to_list()
    for nombre in sorted(set(anual["variable"].to_list()) - set(orden)):
        orden.append(nombre)
    anios = sorted(int(a) for a in anual["anio"].unique().to_list())
    cambio = {f["variable"]: f["cambio_signo"] for f in resumen.iter_rows(named=True)}
    fila_de, columna_de = {v: i for i, v in enumerate(orden)}, {a: j for j, a in enumerate(anios)}
    valores = np.full((len(orden), len(anios)), np.nan)
    optimista = np.zeros(valores.shape, dtype=bool)
    for f in anual.iter_rows(named=True):
        i, j = fila_de[f["variable"]], columna_de[int(f["anio"])]
        if f["auc"] is not None:
            valores[i, j] = f["auc"] - 0.5
        optimista[i, j] = f["auc_optimista"] == "si"
    vmax = max(0.05, float(np.ceil(np.nanmax(np.abs(valores)) * 20.0) / 20.0)) if np.isfinite(valores).any() else 0.05
    n = len(orden)
    fig, (ax, ax_psi) = graficos.nueva_figura_multiple(1, 2, figsize=(11.0, graficos.altura_filas(n, por_fila=0.17, minimo=4.5, extra=2.2)), width_ratios=[5, 2.2])
    etiquetas = [f"{v}*" if cambio.get(v) == "si" else v for v in orden]
    barra = graficos.mapa_calor(
        fig, ax, valores, etiquetas, [str(a) for a in anios], cmap=graficos.CMAP_DIVERGENTE, vmin=-vmax, vmax=vmax, etiqueta_barra="AUC menos 0,5 (con la dirección del ranking)",
        anotar=True, fmt="{:.2f}",
    )
    barra.ax.tick_params(labelsize=7)
    graficos.formato_barra_decimal(barra)
    ax.tick_params(axis="y", labelsize=6.5, length=2)
    ax.tick_params(axis="x", labelsize=8)
    for i, j in zip(*np.nonzero(optimista)):
        ax.plot(j - 0.42, i, marker="o", color="black", markersize=1.8, linestyle="none", zorder=5)
    ax.set_xlabel("Año")
    # panel de PSI: dos columnas, una celda por variable y comparación, coloreadas por clase
    psi = t.filter(pl.col("tipo_fila") == "psi")
    comparaciones = ["2019-2022 contra 2023", "2019 contra 2020-2022"]
    celdas = {(f["variable"], f["comparacion"]): f for f in psi.iter_rows(named=True)}
    for i, nombre in enumerate(orden):
        for j, comp in enumerate(comparaciones):
            f = celdas.get((nombre, comp))
            clase = f["clase_psi"] if f is not None else None
            color = COLOR_PSI.get(clase, graficos.GRIS_SUPRIMIDO)
            ax_psi.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, facecolor=color, edgecolor="white", linewidth=0.6))
            if f is not None and f["psi"] is not None:
                r, g, b = (int(color[k:k + 2], 16) / 255.0 for k in (1, 3, 5))
                ax_psi.text(j, i, _coma(f["psi"]), ha="center", va="center", fontsize=6, color="white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black")
            else:
                ax_psi.text(j, i, "n/d", ha="center", va="center", fontsize=6, color="#555555")
    ax_psi.set_xlim(-0.5, 1.5)
    ax_psi.set_ylim(n - 0.5, -0.5)
    ax_psi.set_xticks([0, 1])
    ax_psi.set_xticklabels(["Ranking contra\n2023", "2019 contra\n2020-2022"], fontsize=7)
    ax_psi.set_yticks([])
    ax_psi.set_xlabel("Índice de estabilidad (PSI)")
    ax_psi.grid(False)
    ax_psi.spines[:].set_visible(False)
    ax_psi.xaxis.set_ticks_position("none")
    manijas = [Patch(facecolor=COLOR_PSI[k], label=k.capitalize()) for k in ("estable", "moderado", "alto")]
    ax_psi.legend(handles=manijas, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False, fontsize=7, title="Clase de PSI", title_fontsize=7)
    pie = ("Gris: sin dato o celda suprimida. Punto negro: categórica de 2019 a 2022, cuyo puntaje se estimó con esos mismos años (optimista). Asterisco: la señal cambia de signo "
           "entre años con efecto suficiente.")
    ctx.figuras.guardar_figura(
        fig, id_figura="F11_03", tarea=11, titulo="Estabilidad por año de la señal de cada variable (2019 a 2023)",
        descripcion_alt="A la izquierda, mapa de calor con el AUC menos 0,5 de cada variable en cada año de 2019 a 2023, con una escala que va del azul al naranja; a la derecha, dos columnas "
                        "con el índice de estabilidad poblacional de cada variable contra 2023 y de 2019 contra 2020 a 2022, coloreadas como estable, moderado o alto.",
        csv="t23_estabilidad.csv", anios="2019 a 2023", decision="17", uso_interno=False, extra_pie=pie,
    )
    return True


def figura_importancia(ctx: Contexto) -> bool:
    """F11_05: importancia por permutación de las variables del modelo L1 en 2023. Se arma leyendo SOLO ``t23_importancia.csv``.

    Barras horizontales de la caída media de AUC con una barra de error de una desviación, línea vertical en ``importancia_permutacion_min``, en gris las variables sin coeficientes
    en el L1, solo variables con ``en_modelo_l1 = "si"`` y trama diagonal en las de ``clase_proxy`` moderado o alto (sus coeficientes pueden codificar al hospital y condicionan a las
    demás). Devuelve ``False`` si ninguna variable del L1 tiene importancia publicada.
    """
    t = ctx.leer_tabla("t23_importancia.csv", columnas_texto=_TEXTO_IMPORTANCIA)
    t = t.with_columns([_num(pl.col(c)) for c in ("importancia_permutacion", "importancia_sd", "auc_2023")])
    v = t.filter((pl.col("tipo_fila") == "variable") & (pl.col("en_modelo_l1") == "si") & pl.col("importancia_permutacion").is_not_null())
    if v.height == 0:
        return False
    v = v.sort(["importancia_permutacion", "variable"], descending=[True, False], maintain_order=True)
    minimo = float(ctx.umbrales["importancia_permutacion_min"])
    n = v.height
    alto_leyenda = 0.8
    altura = graficos.altura_filas(n, por_fila=0.24, minimo=4.0, extra=1.8)
    fig, (ax, hueco) = graficos.nueva_figura_multiple(2, 1, figsize=(9.5, altura + alto_leyenda), gridspec_kw={"height_ratios": [altura, alto_leyenda]})
    hueco.axis("off")
    azul, gris = graficos.OKABE_ITO[0], "#A6A6A6"
    for i, f in enumerate(v.iter_rows(named=True)):
        sin_coef = f["l1_sobrevive"] != "si"
        cerca_de_proxy = f["clase_proxy"] in ("moderado", "alto")
        ax.barh(i, f["importancia_permutacion"], height=0.72, color=gris if sin_coef else azul, edgecolor="white" if cerca_de_proxy else "none", hatch="///" if cerca_de_proxy else None, linewidth=0.0, zorder=2)
        if sin_coef:
            ax.plot(f["importancia_permutacion"], i, marker="o", color=gris, markersize=3.2, linestyle="none", zorder=3)         # sin coeficientes la caída es exactamente cero: la barra no se ve, el punto sí
        elif f["importancia_sd"] is not None:
            ax.errorbar(f["importancia_permutacion"], i, xerr=f["importancia_sd"], fmt="none", ecolor="#222222", elinewidth=0.8, capsize=1.8, zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels(v["variable"].to_list(), fontsize=7)
    ax.set_ylim(n - 0.5, -0.5)
    ax.axvline(minimo, color="#555555", linestyle="--", linewidth=1.0, zorder=1)
    ax.axvline(0.0, color="black", linewidth=0.8, zorder=1)
    ax.text(minimo, 1.0, "umbral de importancia", transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=7, color="#333333")
    graficos.formato_decimal(ax, "x")
    ax.set_xlabel("Caída media del AUC en 2023 al permutar la variable")
    ax.grid(axis="y", visible=False)
    manijas = [
        Patch(facecolor=azul, label="Con coeficientes en el modelo L1"), Line2D([], [], marker="o", color=gris, linestyle="none", markersize=4, label="Sin coeficientes en el modelo L1 (caída cero)"),
        Patch(facecolor="#BBBBBB", edgecolor="white", hatch="///", label="Cerca de ser proxy del hospital (moderado o alto)"),
        Line2D([], [], color="#555555", linestyle="--", linewidth=1.0, label="Umbral de importancia"),
    ]
    leyenda = hueco.legend(handles=manijas, loc="center", bbox_to_anchor=(0.5, 0.5), ncol=2, frameon=False, fontsize=7.5)
    leyenda.set_in_layout(False)
    modelo = t.filter(pl.col("variable") == "MODELO_L1")
    auc_modelo = modelo["auc_2023"][0] if modelo.height else None
    pie = "Barra de error: una desviación de las repeticiones de la permutación. " + (f"AUC del modelo L1 en 2023: {_coma(auc_modelo)}. " if auc_modelo is not None else "")
    pie += "Las variables con trama pueden codificar al hospital y condicionan los coeficientes de las demás."
    ctx.figuras.guardar_figura(
        fig, id_figura="F11_05", tarea=11, titulo="Importancia por permutación de las variables en 2023",
        descripcion_alt="Barras horizontales con la caída media del AUC en 2023 al permutar cada variable del modelo de regresión con penalización L1, con barra de error de una desviación; una "
                        "línea vertical marca el umbral de importancia, las variables sin coeficientes están en gris y las cercanas a ser proxy del hospital llevan trama.",
        csv="t23_importancia.csv", anios="2023", decision="", uso_interno=False, extra_pie=pie,
    )
    return True


# ==================================================================================================================== resumen
IDS_FIGURAS = ("F11_01", "F11_02", "F11_03", "F11_04", "F11_05", "F11_06")
ALCANCE_TOTAL = "2019-2023"
NOTA_TOTAL = "ranking 2019 a 2022 y confirmación 2023"
_COLUMNAS_MATRIZ_RESUMEN = [
    "n_matriz", *[f"n_recomendacion_{g}" for g in decision.BINES_RECOMENDACION.values()], *[f"n_contraste_{e}" for e in decision.ETIQUETAS_CONTRASTE], "n_restriccion_r3",
    "n_valor_suprimido", "n_sin_medicion",
]
COLUMNAS_RESUMEN = [
    "n_universo", "n_crudas", "n_derivadas", "n_evaluables", "n_parciales", "n_no_evaluables",
    "n_candidatas_ranking", "mejor_variable", "mejor_auc_senal", "n_senal_alta", "n_senal_media", "n_senal_debil", "n_senal_baja", "n_faltante_alerta", "n_faltante_alto",
    "n_faltante_informativo", "n_filas_ranking_suprimidas",
    "n_cambio_signo", "n_psi_alto_2023", "n_psi_moderado_2023", "h17a_n_clave", "h17a_n_psi_alto", "h17a_prop_psi_alto", "h17a_mayoria_psi_alto",
    "n_proxy_alto", "n_proxy_moderado", "uci_phi_sobredispersion", "uci_razon_p90_p10", "espec_n_niveles_en_un_hospital", "espec_n_niveles_totales",
    "elix_top5", "elix_n_coinciden_fase1", "elix_n_coinciden_repo", "elix_n_or_relevante_ajustado", "elix_n_con_indice",
    "auc_ref_edad_sexo_2023", "auc_ref_elix31_2023", "auc_modelo_l1_2023", "l1_c", "l1_n_variables_sobreviven", "n_importan",
    "fuga_n_supera_umbral", "concordancia_prop_sin_traslado", "concordancia_prop_con_traslado",
    *_COLUMNAS_MATRIZ_RESUMEN,
    "confirmacion_disponible", "t22_disponible", "historial_evaluable", "figuras_omitidas",
]
"""Columnas de ``t23_resumen_seleccion.csv`` en el orden de DIS 10.10 (las de la matriz salen de ``decision.resumen_matriz``). Se agregan ``elix_n_con_indice`` y ``n_sin_medicion``, que los
textos del informe necesitan y el resumen del diseño no trae."""
_TEXTO_RESUMEN = {"mejor_variable", "h17a_mayoria_psi_alto", "elix_top5", "confirmacion_disponible", "t22_disponible", "historial_evaluable", "figuras_omitidas"}
_REAL_RESUMEN = {
    "mejor_auc_senal", "h17a_prop_psi_alto", "uci_phi_sobredispersion", "uci_razon_p90_p10", "auc_ref_edad_sexo_2023", "auc_ref_elix31_2023", "auc_modelo_l1_2023", "l1_c",
    "concordancia_prop_sin_traslado", "concordancia_prop_con_traslado",
}
ESQUEMA_RESUMEN = {c: pl.String if c in _TEXTO_RESUMEN else pl.Float64 if c in _REAL_RESUMEN else pl.Int64 for c in COLUMNAS_RESUMEN}


def figuras_omitidas(
    ranking_pub: pl.DataFrame, proxy_pub: pl.DataFrame, estabilidad_pub: pl.DataFrame, or_elix_pub: pl.DataFrame, importancia_pub: pl.DataFrame, fuga_pub: pl.DataFrame,
) -> list[str]:
    """Ids de las figuras que no se dibujan porque su tabla publicable no tiene nada que mostrar: las mismas condiciones con que cada ``figura_*`` devuelve ``False`` al leer su CSV."""
    sin: list[str] = []
    if ranking_pub.filter((pl.col("ventana") == V_RANK) & pl.col("auc_senal").is_not_null()).height == 0:
        sin.append("F11_01")
    if proxy_pub.filter((pl.col("tipo_fila") == "variable") & pl.col("asociacion_hospital").is_not_null() & pl.col("v_resultado").is_not_null()).height == 0:
        sin.append("F11_02")
    if estabilidad_pub.filter(pl.col("tipo_fila") == "auc_anual").height == 0:
        sin.append("F11_03")
    if or_elix_pub.filter(pl.col("periodo") == V_RANK).height == 0:
        sin.append("F11_04")
    if importancia_pub.filter((pl.col("tipo_fila") == "variable") & (pl.col("en_modelo_l1") == "si") & pl.col("importancia_permutacion").is_not_null()).height == 0:
        sin.append("F11_05")
    if fuga_pub.filter((pl.col("tipo_fila") == "variable") & (pl.col("auc_ranking").is_not_null() | pl.col("auc_confirmacion").is_not_null())).height == 0:
        sin.append("F11_06")
    return sin


def _cuenta(t: pl.DataFrame, columna: str, valor: str) -> int:
    return int((t[columna] == valor).sum())


def tabla_resumen(ctx: Contexto, r: Mapping[str, Any]) -> pl.DataFrame:
    """``t23_resumen_seleccion.csv`` (una fila, sin ``cols_n``: ningún campo es un conteo de casos). ``r`` es el diccionario de ``calcular`` sin la clave ``resumen``.

    Regla de procedencia (DIS 10.10, hallazgo 17): los conteos de variables por clase, recomendación o etiqueta salen de las tablas reales (no son conteos de casos); las cifras
    estadísticas (``mejor_auc_senal`` y la variable que la lleva, ``auc_ref_*``, ``auc_modelo_l1_2023``, ``l1_c``, ``uci_*``, ``concordancia_*``, ``h17a_prop_psi_alto`` y las cuatro
    cifras de Elixhauser) salen de las versiones publicables, de modo que una fila suprimida no aporta su valor. ``n_derivadas`` incluye los cuatro nombres que solo menciona
    ``reglas_variables.csv`` (``n_crudas + n_derivadas = n_universo``). Las "variables clave" de H17a son las candidatas evaluables cuyo rol no es excluyente; ``h17a_n_clave`` y
    ``h17a_n_psi_alto`` cuentan las que tienen PSI de 2019 contra 2020-2022 (y cuántas llegan a ``psi_alto``), ``h17a_prop_psi_alto`` es la proporción publicable y
    ``h17a_mayoria_psi_alto`` dice "si" si es mayor que la mitad.
    """
    panel, u = r["panel"], ctx.umbrales
    d = r["disponibilidad"]
    rk, rkp = r["ranking"].filter(pl.col("ventana") == V_RANK), r["ranking_pub"].filter(pl.col("ventana") == V_RANK)
    out: dict[str, Any] = {c: None for c in COLUMNAS_RESUMEN}
    out.update(
        n_universo=d.height, n_crudas=_cuenta(d, "origen", "cruda"), n_derivadas=d.height - _cuenta(d, "origen", "cruda"), n_evaluables=_cuenta(d, "evaluable", "si"),
        n_parciales=_cuenta(d, "evaluable", "parcial"), n_no_evaluables=_cuenta(d, "evaluable", "no"),
    )
    # --- ranking
    publicadas = sorted((f for f in rkp.iter_rows(named=True) if f["auc_senal"] is not None), key=lambda f: (-_pub(f["auc_senal"]), f["variable"]))
    out.update(
        n_candidatas_ranking=rk.height, mejor_variable=publicadas[0]["variable"] if publicadas else None, mejor_auc_senal=_finito(publicadas[0]["auc_senal"]) if publicadas else None,
        n_senal_alta=_cuenta(rk, "clase_senal", "alta"), n_senal_media=_cuenta(rk, "clase_senal", "media"), n_senal_debil=_cuenta(rk, "clase_senal", "débil"),
        n_senal_baja=_cuenta(rk, "clase_senal", "baja"), n_faltante_alerta=_cuenta(rk, "clase_faltante", "alerta"), n_faltante_alto=_cuenta(rk, "clase_faltante", "alto"),
        n_faltante_informativo=_cuenta(rk, "faltante_informativo", "si"),
        n_filas_ranking_suprimidas=sum(1 for a, b in zip(rk["auc_senal"].to_list(), rkp["auc_senal"].to_list()) if a is not None and b is None),
    )
    # --- estabilidad
    es, esp = r["estabilidad"], r["estabilidad_pub"]
    psi23 = es.filter((pl.col("tipo_fila") == "psi") & (pl.col("comparacion") == decision.COMP_CONFIRMACION))
    clave = {n for n, v in panel.variables.items() if v.uso == "candidata" and v.evaluable != "no" and v.rol_fase1 not in decision.ROLES_EXCLUYENTES}

    def psi_clave(t: pl.DataFrame) -> pl.DataFrame:
        return t.filter((pl.col("tipo_fila") == "psi") & (pl.col("comparacion") == decision.COMP_RESTO) & pl.col("variable").is_in(sorted(clave)) & pl.col("psi").is_not_null())

    real_h, pub_h = psi_clave(es), psi_clave(esp)
    prop = _cuenta(pub_h, "clase_psi", "alto") / pub_h.height if pub_h.height else None
    out.update(
        n_cambio_signo=_cuenta(es.filter(pl.col("tipo_fila") == "resumen"), "cambio_signo", "si"), n_psi_alto_2023=_cuenta(psi23, "clase_psi", "alto"),
        n_psi_moderado_2023=_cuenta(psi23, "clase_psi", "moderado"), h17a_n_clave=real_h.height, h17a_n_psi_alto=_cuenta(real_h, "clase_psi", "alto"), h17a_prop_psi_alto=prop,
        h17a_mayoria_psi_alto=None if prop is None else ("si" if prop > 0.5 else "no"),
    )
    # --- proxy
    pv = r["proxy"].filter(pl.col("tipo_fila") == "variable")
    uci = r["proxy_pub"].filter(pl.col("tipo_fila") == "variacion_uci")
    card = panel.cardinalidad_especialidad or {}
    out.update(
        n_proxy_alto=_cuenta(pv, "clase_proxy", "alto"), n_proxy_moderado=_cuenta(pv, "clase_proxy", "moderado"),
        uci_phi_sobredispersion=_finito(uci["phi_sobredispersion"][0]) if uci.height else None, uci_razon_p90_p10=_finito(uci["razon_p90_p10"][0]) if uci.height else None,
        espec_n_niveles_en_un_hospital=card.get("n_niveles_en_un_hospital"), espec_n_niveles_totales=card.get("n_niveles_totales"),
    )
    # --- Elixhauser (versiones publicables: el índice y las marcas dependen de conteos que pueden estar suprimidos)
    elix = sensibilidad_elix.resumen_elixhauser(r["or_elix_pub"], r["sensibilidad_pub"])
    out.update({k: elix[k] for k in ("elix_top5", "elix_n_coinciden_fase1", "elix_n_coinciden_repo", "elix_n_or_relevante_ajustado", "elix_n_con_indice")})
    # --- modelos
    imp, impp = r["importancia"], r["importancia_pub"]
    refs = {f["variable"]: f for f in impp.filter(pl.col("tipo_fila") == "referencia").iter_rows(named=True)}
    vars_imp = imp.filter(pl.col("tipo_fila") == "variable")
    out.update(
        auc_ref_edad_sexo_2023=_finito(refs.get("MODELO_EDAD_SEXO", {}).get("auc_2023")), auc_ref_elix31_2023=_finito(refs.get("MODELO_ELIX31", {}).get("auc_2023")),
        auc_modelo_l1_2023=_finito(refs.get("MODELO_L1", {}).get("auc_2023")), l1_c=_finito(refs.get("MODELO_L1", {}).get("l1_c")),
        l1_n_variables_sobreviven=_cuenta(vars_imp, "l1_sobrevive", "si"), n_importan=_cuenta(vars_imp, "importa", "si"),
    )
    # --- fuga
    fv = r["fuga"].filter(pl.col("tipo_fila") == "variable")
    sin_t = r["fuga_pub"].filter(pl.col("tipo_fila") == "concordancia")
    con_t = r["fuga_pub"].filter(pl.col("tipo_fila") == "concordancia_con_traslado")
    out.update(
        fuga_n_supera_umbral=_cuenta(fv, "supera_umbral_fuga", "si"),
        concordancia_prop_sin_traslado=_finito(sin_t["prop_igualdad"][0]) if sin_t.height else None,
        concordancia_prop_con_traslado=_finito(con_t["prop_igualdad"][0]) if con_t.height else None,
    )
    # --- matriz y estado
    out.update(decision.resumen_matriz(r["matriz"], r["matriz_pub"]))
    matriz = r["matriz"]
    hist = matriz.filter(pl.col("variable").is_in(list(decision.HISTORIALES)))
    omitidas = r["figuras_omitidas"]
    out.update(
        confirmacion_disponible="si" if panel.anio_confirmacion is not None else "no", t22_disponible="si" if "t22" in set(d["fuente_faltante"].to_list()) else "no",
        historial_evaluable="si" if (hist.height and _cuenta(hist, "evaluable", "no") < hist.height) else "no", figuras_omitidas=";".join(omitidas) if omitidas else "ninguna",
    )
    return pl.DataFrame([out], schema=ESQUEMA_RESUMEN)


# ==================================================================================================================== calcular
def calcular(ctx: Contexto, panel: modulo_panel.Panel | None = None, asoc: Mapping[str, Any] | None = None, cat=None) -> dict[str, Any]:
    """Calcula toda la Tarea 11 sin escribir nada y devuelve cada tabla en su versión real y en su versión publicable (clave terminada en ``_pub``).

    Si ``panel`` es ``None`` se toma el de ``asoc`` o se construye con ``panel.construir_panel(ctx, cat, tarea="T11")``; si ``asoc`` es ``None`` se llama a
    ``asociacion.calcular(ctx, panel)`` (la Tarea 12 pasará los resultados de la Tarea 10 para no recalcular). Orden: ranking, estabilidad, proxy (necesita el ranking), referencias e
    importancia (necesita la clase de proxy de cada variable), fuga (necesita las referencias), OR y sensibilidad de Elixhauser, disponibilidad y matriz (``decision.armar_matriz``),
    resumen y parámetros. La matriz decide con las versiones reales y se publica con las ``_pub``: lo que una tabla fuente anuló por celda pequeña no se vuelve a publicar en otro lado.
    Claves: ``panel``, ``asoc``, ``ranking``, ``estabilidad``, ``proxy``, ``referencias``, ``importancia``, ``fuga``, ``or_elix``, ``sensibilidad``, sus versiones ``_pub``,
    ``disponibilidad``, ``matriz``, ``matriz_pub``, ``evidencia`` (lo que recibe ``decision``), ``figuras_omitidas``, ``parametros`` y ``resumen``.
    """
    if panel is None:
        panel = asoc["panel"] if asoc is not None else modulo_panel.construir_panel(ctx, cat, tarea=TAREA)
    if asoc is None:
        asoc = asociacion.calcular(ctx, panel, cat)

    def publicable(t: pl.DataFrame, bloques: Sequence[tuple[Sequence[str], Sequence[str]]]) -> pl.DataFrame:
        return supresion.suprimir_bloques(t, bloques, minimo=ctx.celda_minima)[0]

    ranking = tabla_ranking(ctx, panel)
    estabilidad = tabla_estabilidad(ctx, panel, ranking)
    proxy = sospechas.tabla_proxy(ctx, panel, ranking)
    clase_proxy = {f["variable"]: f["clase_proxy"] for f in proxy.filter(pl.col("tipo_fila") == "variable").iter_rows(named=True)}
    referencias = modelos.referencias(panel, ctx.semilla)
    importancia = tabla_importancia(ctx, panel, referencias, clase_proxy)
    fuga = sospechas.tabla_fuga(ctx, panel, ranking, referencias)
    or_elix = sensibilidad_elix.tabla_or_elix(ctx, panel)
    sensibilidad = sensibilidad_elix.tabla_sensibilidad(ctx, panel, or_elix)
    pub = {
        "ranking_pub": publicable(ranking, BLOQUES_RANKING), "estabilidad_pub": publicable(estabilidad, BLOQUES_ESTABILIDAD), "proxy_pub": publicable(proxy, sospechas.BLOQUES_PROXY),
        "importancia_pub": publicable(importancia, BLOQUES_IMPORTANCIA), "fuga_pub": publicable(fuga, sospechas.BLOQUES_FUGA),
        "or_elix_pub": publicable(or_elix, sensibilidad_elix.BLOQUES_OR_ELIX), "sensibilidad_pub": publicable(sensibilidad, sensibilidad_elix.BLOQUES_SENSIBILIDAD),
    }
    evidencia = {
        "ranking": ranking, "estabilidad": estabilidad, "proxy": proxy, "importancia": importancia, "fuga": fuga, "clusters": asoc["clusters"],
        **{k: pub[k] for k in ("ranking_pub", "estabilidad_pub", "proxy_pub", "importancia_pub", "fuga_pub")},
    }
    armado = decision.armar_matriz(ctx, panel, evidencia)
    r: dict[str, Any] = {
        "panel": panel, "asoc": asoc, "ranking": ranking, "estabilidad": estabilidad, "proxy": proxy, "referencias": referencias, "importancia": importancia, "fuga": fuga,
        "or_elix": or_elix, "sensibilidad": sensibilidad, **pub, "disponibilidad": armado["disponibilidad"], "matriz": armado["matriz"], "matriz_pub": armado["matriz_pub"],
        "evidencia": evidencia, "parametros": tabla_parametros(),
    }
    r["figuras_omitidas"] = figuras_omitidas(pub["ranking_pub"], pub["proxy_pub"], pub["estabilidad_pub"], pub["or_elix_pub"], pub["importancia_pub"], pub["fuga_pub"])
    r["resumen"] = tabla_resumen(ctx, r)
    return r


# ==================================================================================================================== ejecutar
def _escribir(ctx: Contexto, r: Mapping[str, Any]) -> None:
    """Escribe las tablas de la Tarea 11 (las de la matriz las escribe ``decision.ejecutar``). Los conteos de bases distintas en una misma fila van por bloques de supresión."""
    modulo_panel.escribir_por_bloques(
        ctx, "t23_ranking_variables.csv", r["ranking"], BLOQUES_RANKING, tarea=TAREA,
        descripcion="Señal univariada de cada candidata frente a la mortalidad en 2019 a 2022 y en 2023 (formato largo, un AUC por ventana y su intervalo)",
    )
    cols, deps = BLOQUES_ESTABILIDAD[0]
    ctx.escribir_tabla(
        "t23_estabilidad.csv", r["estabilidad"], tarea=TAREA, cols_n=cols, dependientes=deps,
        descripcion="AUC por año con el puntaje y la dirección del ranking, cambio de signo de la señal y PSI contra 2023 y de 2019 contra 2020 a 2022 (formato largo)",
    )
    modulo_panel.escribir_por_bloques(
        ctx, "t23_proxy_hospital.csv", r["proxy"], sospechas.BLOQUES_PROXY, tarea=TAREA,
        descripcion="Asociación de cada candidata con el hospital de egreso, variación entre hospitales de los ingresos a cuidados especiales y cardinalidad de ESPECIALIDAD_MEDICA por hospital",
    )
    cols, deps = sensibilidad_elix.BLOQUES_OR_ELIX[0]
    ctx.escribir_tabla(
        "t23_or_elix.csv", r["or_elix"], tarea=TAREA, cols_n=cols, dependientes=deps,
        descripcion="Odds ratio crudo y ajustado por edad y sexo de las 31 categorías de Elixhauser contra mortalidad, por período (formato largo)",
    )
    cols, deps = sensibilidad_elix.BLOQUES_SENSIBILIDAD[0]
    ctx.escribir_tabla(
        "t23_sensibilidad_elix.csv", r["sensibilidad"], tarea=TAREA, cols_n=cols, dependientes=deps,
        descripcion="Gradiente de prevalencia de cada categoría de Elixhauser por tramo de estadía de los sobrevivientes, OR de estadía larga e índice de complicación probable",
    )
    modulo_panel.escribir_por_bloques(
        ctx, "t23_importancia.csv", r["importancia"], BLOQUES_IMPORTANCIA, tarea=TAREA,
        descripcion="Incremento de AUC sobre edad y sexo de cada candidata, modelo con penalización L1 e importancia por permutación en 2023, y los modelos de referencia",
    )
    modulo_panel.escribir_por_bloques(
        ctx, "t23_fuga.csv", r["fuga"], sospechas.BLOQUES_FUGA, tarea=TAREA,
        descripcion="AUC de las variables de solo fuga y de las condicionales frente a dos modelos de referencia y concordancia de SERVICIOINGRESO con SERVICIOALTA",
    )
    ctx.escribir_tabla("t23_resumen_seleccion.csv", r["resumen"], tarea=TAREA, descripcion="Cifras de la Tarea 11 para los informes y para la Tarea 12 (una fila)")
    ctx.escribir_tabla("t23_parametros_seleccion.csv", r["parametros"], tarea=TAREA, descripcion="Constantes técnicas de la selección de variables: valor, para qué sirven y qué resultados afectan")


def registrar_metricas(ctx: Contexto, r: Mapping[str, Any]) -> None:
    """Registra las métricas de la Tarea 11 (``tarea = "T11"``, ``tabla`` y ``alcance`` explícitos, ``conteo = False``) desde las versiones publicables.

    ``seleccion.auc.<variable>`` es ``auc_senal`` con el IC de ``auc_senal_bajo`` y ``auc_senal_alto`` (nota "categórica: validación cruzada" u "optimista"); ``seleccion.auc_2023.*``,
    ``seleccion.incremento_auc.*`` y ``seleccion.importancia.*`` son de 2023. Si la fila fuente quedó suprimida el valor es nulo y la nota lo dice (``sospechas.registrar_estadistica``).
    Se agregan los conteos de la matriz (``seleccion.n_*``, de variables y no de casos), las cifras de Elixhauser y de los modelos de referencia, y las ``proxy.*`` y ``fuga.*``.
    Ninguna cubre 2024.
    """
    est_ = sospechas.registrar_estadistica
    sin_conf = r["panel"].anio_confirmacion is None
    nota_conf = "confirmación no disponible" if sin_conf else ""
    for real, pub in zip(r["ranking"].iter_rows(named=True), r["ranking_pub"].iter_rows(named=True)):
        n, fuera = real["variable"], real["estado"] if real["estado"] != "ok" else ""
        if real["ventana"] == V_RANK:
            nota = ("categórica: validación cruzada" if real["auc_cv"] is not None else "categórica: optimista") if real["tipo"] == "cat" else ""
            est_(
                ctx, f"seleccion.auc.{n}", real["auc_senal"], pub["auc_senal"], tabla="t23_ranking_variables.csv", alcance=V_RANK, ic_bajo=pub["auc_senal_bajo"],
                ic_alto=pub["auc_senal_alto"], nota=nota or (f"estado: {fuera}" if fuera else ""),
            )
            if sin_conf:                                                                       # sin 2023 el ranking no trae fila de confirmación: la métrica existe y queda nula
                est_(ctx, f"seleccion.auc_2023.{n}", None, None, tabla="t23_ranking_variables.csv", alcance=V_CONF, nota=nota_conf)
        else:
            est_(
                ctx, f"seleccion.auc_2023.{n}", real["auc"], pub["auc"], tabla="t23_ranking_variables.csv", alcance=V_CONF, ic_bajo=pub["auc_bajo"], ic_alto=pub["auc_alto"],
                nota=f"estado: {fuera}" if fuera else "",
            )
    for real, pub in zip(r["importancia"].iter_rows(named=True), r["importancia_pub"].iter_rows(named=True)):
        if real["tipo_fila"] != "variable":
            continue
        n = real["variable"]
        estado = real["incremento_estado"] if real["incremento_estado"] != "ok" else ""
        est_(ctx, f"seleccion.incremento_auc.{n}", real["incremento_auc"], pub["incremento_auc"], tabla="t23_importancia.csv", alcance=V_CONF, nota=nota_conf or estado)
        if real["en_modelo_l1"] == "si":
            est_(ctx, f"seleccion.importancia.{n}", real["importancia_permutacion"], pub["importancia_permutacion"], tabla="t23_importancia.csv", alcance=V_CONF, nota=nota_conf)
    res = r["resumen"].row(0, named=True)
    for clave, valor in (
        ("n_total", res["n_matriz"]), ("n_mantener", res["n_recomendacion_mantener"]), ("n_mantener_clinico", res["n_recomendacion_mantener_clinico"]),
        ("n_excluir_fase1", res["n_recomendacion_excluir_fase1"]), ("n_excluir_fuga", res["n_recomendacion_excluir_fuga"]), ("n_excluir_proxy", res["n_recomendacion_excluir_proxy"]),
        ("n_condicional", res["n_recomendacion_condicional_proxy"] + res["n_recomendacion_condicional_calidad"]), ("n_redundante", res["n_recomendacion_redundante"]),
        ("n_baja_senal", res["n_recomendacion_baja_senal"]), ("n_no_evaluable", res["n_recomendacion_no_evaluable"]),
    ):
        ctx.registrar_metrica(f"seleccion.{clave}", valor, tarea=TAREA, tabla="matriz_decision_variables.csv", conteo=False, alcance=ALCANCE_TOTAL, nota=NOTA_TOTAL)
    for clave, columna in (("seleccion.elix.n_coinciden_fase1", "elix_n_coinciden_fase1"), ("seleccion.elix.n_coinciden_repo", "elix_n_coinciden_repo")):
        ctx.registrar_metrica(clave, res[columna], tarea=TAREA, tabla="t23_sensibilidad_elix.csv", conteo=False, alcance=V_RANK, nota="categorías, no casos")
    real_ref = {f["variable"]: f for f in r["importancia"].filter(pl.col("tipo_fila") == "referencia").iter_rows(named=True)}
    pub_ref = {f["variable"]: f for f in r["importancia_pub"].filter(pl.col("tipo_fila") == "referencia").iter_rows(named=True)}
    for clave, modelo in (("seleccion.auc_ref_edad_sexo", "MODELO_EDAD_SEXO"), ("seleccion.auc_ref_elix31", "MODELO_ELIX31"), ("seleccion.auc_modelo_l1", "MODELO_L1")):
        if modelo in real_ref:
            est_(ctx, clave, real_ref[modelo]["auc_2023"], pub_ref[modelo]["auc_2023"], tabla="t23_importancia.csv", alcance=V_CONF, nota=nota_conf)
    sospechas.registrar_metricas(ctx, r["proxy"], r["proxy_pub"], r["fuga"], r["fuga_pub"])


def ejecutar(ctx: Contexto, panel: modulo_panel.Panel | None = None, asoc: Mapping[str, Any] | None = None, cat=None) -> dict[str, Any]:
    """Calcula, escribe las tablas de la Tarea 11 (más las dos del panel y las dos de la matriz), registra las métricas, dibuja F11_01 a F11_06 y devuelve el diccionario de ``calcular``.

    Una figura se omite solo si su tabla publicable está vacía: sin error y anotado en ``figuras_omitidas`` del resumen, que se decide antes de escribir.
    """
    r = calcular(ctx, panel, asoc, cat)
    modulo_panel.escribir_tablas(ctx, r["panel"], tarea=TAREA)
    _escribir(ctx, r)
    decision.ejecutar(ctx, r["panel"], r["evidencia"])
    registrar_metricas(ctx, r)
    omitidas = set(r["figuras_omitidas"])
    for id_figura, dibujar in (
        ("F11_01", figura_ranking), ("F11_02", sospechas.figura_proxy), ("F11_03", figura_estabilidad), ("F11_04", sensibilidad_elix.figura_or_elix), ("F11_05", figura_importancia),
        ("F11_06", sospechas.figura_fuga),
    ):
        if id_figura not in omitidas:
            dibujar(ctx)
    return r


# ===================================================================================================================== informe
_TEXTO_DISPONIBILIDAD = [c for c, t in decision.ESQUEMA_DISPONIBILIDAD_BASE.items() if t == pl.String] + ["fuente_faltante"]
_TEXTO_PROXY_LISTA = list(sospechas._TEXTO_PROXY)
_TEXTO_FUGA_LISTA = list(sospechas._TEXTO_FUGA)
_TEXTO_SENSIBILIDAD = ["elix_id", "nombre_es", "sospechosa_fase1", "en_lista_repositorio", "en_top5", "estado"]


def agregar_secciones(inf, ctx: Contexto, n0: int = 1) -> None:
    """Agrega al informe de exploración (CRISP-DM 2.3) las nueve secciones de la Tarea 11 (``N_SECCIONES = 9``); la octava la agrega ``decision.agregar_secciones``.

    Las cifras salen de ``t23_resumen_seleccion.csv`` con ``cifra(...)`` y los umbrales de ``descripcion.Cifra_umbral``; las tablas se incrustan con ``inf.tabla(df=...)`` desde los
    CSV de la tarea (``prop_faltante`` con ``prop1``) y las figuras con ``inf.figura`` solo si se generaron en esta corrida (nunca una figura omitida). La prosa son plantillas con
    marcadores: sin dígitos sueltos. Las frases que mencionan el extracto solo se agregan si la etiqueta del contexto es "extracto".
    """
    from .descripcion import Cifra_umbral

    def rs(col: str):
        return cifra(ctx, "t23_resumen_seleccion.csv", col, columnas_texto=sorted(_TEXTO_RESUMEN))

    umbral = lambda nombre: Cifra_umbral(ctx, nombre)  # noqa: E731
    res = ctx.leer_tabla("t23_resumen_seleccion.csv", columnas_texto=sorted(_TEXTO_RESUMEN)).row(0, named=True)
    figuras = set(ctx.figuras.ids())
    extracto = ctx.etiqueta == "extracto"

    def figura(id_figura: str) -> None:
        if id_figura in figuras:
            inf.figura(id_figura)

    # --- 1. universo y disponibilidad ----------------------------------------------------------------------------------------------------------------
    inf.seccion(f"{n0}. Universo de variables y disponibilidad")
    inf.parrafo(
        "El universo que cubre la matriz de decisión reúne las {n_universo} variables de la documentación: {n_crudas} columnas crudas y {n_derivadas} derivadas o nombradas por las "
        "reglas de la Fase 1 (`reglas_variables.csv`). Por evaluabilidad: evaluables en el panel, {n_eval}; parciales (ETNIA y PUEBLO_ORIGINARIO se describen, pero no se analizan "
        "frente a los resultados, decisión 4), {n_parc}; no evaluables (el panel las resume en otra variable, no las construye o no pertenecen a la cohorte), {n_no}. "
        "Cada una lleva su motivo, su rol en la Fase 1 y su faltante por año en `disponibilidad_variables.csv`.",
        n_universo=rs("n_universo"), n_crudas=rs("n_crudas"), n_derivadas=rs("n_derivadas"), n_eval=rs("n_evaluables"), n_parc=rs("n_parciales"), n_no=rs("n_no_evaluables"),
    )
    disp = ctx.leer_tabla("disponibilidad_variables.csv", columnas_texto=_TEXTO_DISPONIBILIDAD)
    motivos = (
        disp.filter(pl.col("evaluable") == "no")
        .group_by("motivo_no_evaluable")
        .agg(pl.len().alias("n_variables"), pl.col("variable").sort().head(3).str.join(", ").alias("ejemplos"))
        .sort(["n_variables", "motivo_no_evaluable"], descending=[True, False])
    )
    if motivos.height:
        inf.tabla(
            "disponibilidad_variables.csv", df=motivos, columnas=["motivo_no_evaluable", "n_variables", "ejemplos"], max_filas=30,
            encabezados={"motivo_no_evaluable": "Motivo", "n_variables": "Variables", "ejemplos": "Ejemplos"}, leyenda="Variables no evaluables por motivo",
        )
    if res["historial_evaluable"] == "si":
        inf.parrafo("Las tres variables de historial (`N_EGRESOS_12M`, `DIAS_DESDE_EGRESO_PREVIO` y `HISTORIA_DISPONIBLE`) entran al panel con los datos completos y se evalúan con la hipótesis H01c.")
    elif extracto:
        inf.parrafo(
            "En el extracto el ingreso previo de un paciente casi nunca está en el mismo extracto, de modo que `N_EGRESOS_12M`, `DIAS_DESDE_EGRESO_PREVIO` y `HISTORIA_DISPONIBLE` no se "
            "pueden evaluar y quedan como «NO EVALUABLE en extracto»; con los datos completos se evalúan (hipótesis H01c)."
        )
    else:
        inf.parrafo("Las variables de historial no son evaluables en esta corrida y quedan como «NO EVALUABLE» en la matriz (hipótesis H01c).")

    # --- 2. señal univariada -------------------------------------------------------------------------------------------------------------------------
    ranking = ctx.leer_tabla("t23_ranking_variables.csv", columnas_texto=_TEXTO_RANKING).with_columns(
        *[_num(pl.col(c)) for c in ("auc", "auc_senal", "auc_senal_bajo", "auc_senal_alto", "prop_faltante")], _ent(pl.col("posicion"))
    )
    inf.seccion(f"{n0 + 1}. Señal univariada")
    inf.parrafo(
        "El AUC univariado mide cuánto separa una variable los egresos con muerte de los que sobreviven: un AUC de un medio es la ausencia de señal. Se calcula en 2019-2022 y se confirma "
        "en 2023 con la dirección fijada en el ranking, de modo que un cambio de signo aparece como un AUC de 2023 por debajo de ese valor. Entran al ranking {n_cand} candidatas y la de "
        "mayor señal es {mejor|texto}, con {auc|dec3}. Por clase de señal: alta (desde {alta|dec2}), {n_alta}; media (desde {media|dec2}), {n_media}; débil (desde {debil|dec2}), "
        "{n_debil}; baja, {n_baja}. Las categóricas puntúan con la tasa por nivel estimada en la misma ventana, así que su AUC en muestra es optimista: se ordenan por validación "
        "cruzada dejando un año fuera. Variables con faltante en alerta (desde {falerta|prop1}): {n_fa}; en nivel alto (desde {falto|prop1}): {n_fh}; con faltante informativo del "
        "desenlace: {n_fi}. Filas del ranking con cifras anuladas por tener pocos casos: {n_sup}.",
        n_cand=rs("n_candidatas_ranking"), n_alta=rs("n_senal_alta"), alta=umbral("auc_senal_alta"), n_media=rs("n_senal_media"),
        media=umbral("auc_senal_media"), n_debil=rs("n_senal_debil"), debil=umbral("auc_senal_baja"), n_baja=rs("n_senal_baja"), mejor=rs("mejor_variable"), auc=rs("mejor_auc_senal"),
        n_fa=rs("n_faltante_alerta"), falerta=umbral("faltante_alerta"), n_fh=rs("n_faltante_alto"), falto=umbral("faltante_alto"), n_fi=rs("n_faltante_informativo"),
        n_sup=rs("n_filas_ranking_suprimidas"),
    )
    figura("F11_01")
    conf = ranking.filter(pl.col("ventana") == V_CONF).select("variable", pl.col("auc").alias("auc_2023"))
    mejores = ranking.filter((pl.col("ventana") == V_RANK) & pl.col("auc_senal").is_not_null()).sort(["posicion", "variable"], maintain_order=True).head(12).join(conf, on="variable", how="left")
    if mejores.height:
        inf.tabla(
            "t23_ranking_variables.csv", df=mejores, columnas=["posicion", "variable", "tipo", "auc_senal", "auc_senal_bajo", "auc_senal_alto", "auc_2023", "clase_senal", "prop_faltante"],
            encabezados={"posicion": "Posición", "variable": "Variable", "tipo": "Tipo", "auc_senal": "AUC 2019-2022", "auc_senal_bajo": "Límite inferior", "auc_senal_alto": "Límite superior",
                         "auc_2023": "AUC 2023", "clase_senal": "Señal", "prop_faltante": "Faltante"},
            formatos={"auc_senal": "dec3", "auc_senal_bajo": "dec3", "auc_senal_alto": "dec3", "auc_2023": "dec3", "prop_faltante": "prop1"},
            leyenda="Las doce variables con mayor señal univariada en 2019 a 2022, con su AUC de 2023",
        )

    # --- 3. estabilidad y deriva ---------------------------------------------------------------------------------------------------------------------
    est_t = ctx.leer_tabla("t23_estabilidad.csv", columnas_texto=_TEXTO_ESTABILIDAD).with_columns(_num(pl.col("psi")), _ent(pl.col("posicion")), _ent(pl.col("n_anios_efecto_suficiente")))
    inf.seccion(f"{n0 + 2}. Estabilidad y deriva")
    mayoria = res["h17a_mayoria_psi_alto"]
    frase_h17a = {"si": "la mayoría de las variables clave supera ese nivel", "no": "la mayoría de las variables clave no supera ese nivel"}.get(mayoria, "la proporción no se puede evaluar")
    inf.parrafo(
        "La estabilidad repite el AUC año por año con el puntaje y la dirección fijos del ranking. Una variable cambia de signo si, entre los años con efecto suficiente (al menos "
        "{efecto|dec2} de distancia al valor sin señal), hay AUC por encima y por debajo de él. Variables con cambio de signo: {n_cs}. El índice de estabilidad poblacional (PSI) "
        "compara la distribución de cada variable en 2019-2022 con la de 2023; variables con PSI alto (desde {psi_alto|dec2}): {n_alto}; moderado (desde {psi_mod|dec2}): {n_mod}. "
        "Evidencia de la hipótesis H17a (2019 frente a 2020-2022): variables clave, es decir candidatas evaluables cuyo rol en la Fase 1 no es excluyente, con PSI medido: {h_n}; con "
        "PSI alto: {h_alto}, que son {h_prop|prop1}; " + frase_h17a + ". La decisión sobre la ventana de entrenamiento es humana.",
        efecto=umbral("estabilidad_efecto_minimo"), n_cs=rs("n_cambio_signo"), n_alto=rs("n_psi_alto_2023"), psi_alto=umbral("psi_alto"), n_mod=rs("n_psi_moderado_2023"),
        psi_mod=umbral("psi_moderado"), h_n=rs("h17a_n_clave"), h_alto=rs("h17a_n_psi_alto"), h_prop=rs("h17a_prop_psi_alto"),
    )
    figura("F11_03")
    cambios = est_t.filter((pl.col("tipo_fila") == "resumen") & (pl.col("cambio_signo") == "si")).sort(["posicion", "variable"], nulls_last=True, maintain_order=True).head(15)
    if cambios.height:
        inf.tabla(
            "t23_estabilidad.csv", df=cambios, columnas=["variable", "tipo", "posicion", "n_anios_efecto_suficiente"],
            encabezados={"variable": "Variable", "tipo": "Tipo", "posicion": "Posición en el ranking", "n_anios_efecto_suficiente": "Años con efecto suficiente"},
            leyenda="Variables con cambio de signo de la señal entre años (las quince de mayor señal)",
        )
    psi23 = est_t.filter((pl.col("tipo_fila") == "psi") & (pl.col("comparacion") == decision.COMP_CONFIRMACION)).select("variable", "tipo", pl.col("psi").alias("psi_2023"), pl.col("clase_psi").alias("clase_2023"))
    psi_h = est_t.filter((pl.col("tipo_fila") == "psi") & (pl.col("comparacion") == decision.COMP_RESTO)).select("variable", pl.col("psi").alias("psi_h17a"), pl.col("clase_psi").alias("clase_h17a"))
    mayor_psi = psi23.join(psi_h, on="variable", how="left").filter(pl.col("psi_2023").is_not_null()).sort(["psi_2023", "variable"], descending=[True, False], maintain_order=True).head(10)
    if mayor_psi.height:
        inf.tabla(
            "t23_estabilidad.csv", df=mayor_psi, columnas=["variable", "tipo", "psi_2023", "clase_2023", "psi_h17a", "clase_h17a"],
            encabezados={"variable": "Variable", "tipo": "Tipo", "psi_2023": "PSI contra 2023", "clase_2023": "Clase contra 2023", "psi_h17a": "PSI de 2019 contra 2020-2022",
                         "clase_h17a": "Clase de 2019 contra 2020-2022"},
            formatos={"psi_2023": "dec3", "psi_h17a": "dec3"}, leyenda="Las diez variables con mayor PSI contra 2023",
        )

    # --- 4. proxy del hospital -------------------------------------------------------------------------------------------------------------------------
    proxy = ctx.leer_tabla("t23_proxy_hospital.csv", columnas_texto=_TEXTO_PROXY_LISTA).with_columns([_num(pl.col(c)) for c in ("asociacion_hospital", "asociacion_hospital_2023", "v_resultado")])
    inf.seccion(f"{n0 + 3}. Proxy del hospital")
    inf.parrafo(
        "Una variable es un proxy del hospital si identifica el establecimiento de egreso: un modelo que la use podría estar aprendiendo qué hospital atendió al paciente y no su estado. "
        "Se mide la asociación de cada candidata con el hospital con la misma medida de la Tarea 10 (V de Cramér corregida o razón de correlación). Variables con asociación alta "
        "(desde {alto|dec2}): {n_alto}; moderada (desde {mod|dec2}): {n_mod}. Hipótesis H02a: la proporción de ingresos a cuidados especiales varía entre hospitales, con una "
        "sobredispersión de {phi|dec2} y una razón entre los percentiles noventa y diez de {razon|dec2}. Hipótesis H02b: niveles de ESPECIALIDAD_MEDICA, {n_tot}; de ellos, usados por "
        "un solo hospital y que por sí solos identifican al establecimiento, {n_uno}.",
        n_alto=rs("n_proxy_alto"), alto=umbral("proxy_hospital_alto"), n_mod=rs("n_proxy_moderado"), mod=umbral("proxy_hospital_moderado"), phi=rs("uci_phi_sobredispersion"),
        razon=rs("uci_razon_p90_p10"), n_tot=rs("espec_n_niveles_totales"), n_uno=rs("espec_n_niveles_en_un_hospital"),
    )
    figura("F11_02")
    mas_proxy = proxy.filter((pl.col("tipo_fila") == "variable") & pl.col("asociacion_hospital").is_not_null()).sort(["asociacion_hospital", "variable"], descending=[True, False], maintain_order=True).head(12)
    if mas_proxy.height:
        inf.tabla(
            "t23_proxy_hospital.csv", df=mas_proxy, columnas=["variable", "rol_fase1", "asociacion_hospital", "asociacion_hospital_2023", "clase_proxy", "v_resultado"],
            encabezados={"variable": "Variable", "rol_fase1": "Rol en la Fase 1", "asociacion_hospital": "Asociación 2019-2022", "asociacion_hospital_2023": "Asociación 2023",
                         "clase_proxy": "Clase de proxy", "v_resultado": "Asociación con el resultado"},
            formatos={"asociacion_hospital": "dec3", "asociacion_hospital_2023": "dec3", "v_resultado": "dec3"}, leyenda="Las doce variables más asociadas con el hospital",
        )

    # --- 5. comorbilidad de Elixhauser ---------------------------------------------------------------------------------------------------------------------
    sensibilidad = ctx.leer_tabla("t23_sensibilidad_elix.csv", columnas_texto=_TEXTO_SENSIBILIDAD).with_columns(
        *[_num(pl.col(c)) for c in ("razon_prev_t4_t1", "or_larga", "indice_complicacion")], _ent(pl.col("rango_indice"))
    )
    inf.seccion(f"{n0 + 4}. Comorbilidad de Elixhauser")
    inf.parrafo(
        "Para cada categoría de Elixhauser (ELIX_01 a ELIX_31) se estima el odds ratio crudo y el ajustado por edad y sexo contra la mortalidad. Categorías con odds ratio ajustado "
        "relevante (desde {or_rel|dec2}, con un intervalo de confianza que deja fuera la ausencia de efecto): {n_rel}. La sensibilidad de estadía (decisión 8) ordena las categorías "
        "con un índice de complicación probable: el promedio de dos rangos, el de la razón de prevalencias entre el tramo de mayor estadía y el de menor, y el del odds ratio de "
        "estadía larga, ambos con el límite inferior de su intervalo. Categorías que cumplen las condiciones de tamaño y tienen índice: {n_idx}. Las cinco primeras son {top5|texto}. "
        "Coincidencias de esas cinco con las que sospecha la Fase 1: {n_f1}; con la lista del repositorio: {n_repo}. Esa correspondencia, la lista final y el sentido clínico de cada "
        "categoría quedan pendientes de revisión clínica (decisión 12): la tabla informa y no decide.",
        n_rel=rs("elix_n_or_relevante_ajustado"), or_rel=umbral("or_relevante"), n_idx=rs("elix_n_con_indice"), top5=rs("elix_top5"), n_f1=rs("elix_n_coinciden_fase1"),
        n_repo=rs("elix_n_coinciden_repo"),
    )
    figura("F11_04")
    primeras = sensibilidad.filter(pl.col("rango_indice").is_not_null()).sort(["rango_indice", "elix_id"], maintain_order=True).head(10)
    if primeras.height:
        inf.tabla(
            "t23_sensibilidad_elix.csv", df=primeras,
            columnas=["rango_indice", "elix_id", "nombre_es", "razon_prev_t4_t1", "or_larga", "indice_complicacion", "sospechosa_fase1", "en_lista_repositorio"],
            encabezados={"rango_indice": "Rango", "elix_id": "Categoría", "nombre_es": "Nombre", "razon_prev_t4_t1": "Razón de prevalencias", "or_larga": "OR de estadía larga",
                         "indice_complicacion": "Índice", "sospechosa_fase1": "Sospechosa en la Fase 1", "en_lista_repositorio": "En la lista del repositorio"},
            formatos={"razon_prev_t4_t1": "dec2", "or_larga": "dec2", "indice_complicacion": "dec3"}, leyenda="Las diez categorías con mayor índice de complicación probable",
        )

    # --- 6. valor incremental e importancia -------------------------------------------------------------------------------------------------------------
    imp = ctx.leer_tabla("t23_importancia.csv", columnas_texto=_TEXTO_IMPORTANCIA).with_columns(
        [_num(pl.col(c)) for c in ("incremento_auc", "importancia_permutacion", "importancia_sd")]
    )
    inf.seccion(f"{n0 + 5}. Valor incremental e importancia")
    inf.parrafo(
        "El valor incremental es la mejora del AUC de 2023 al agregar una variable a un modelo base con edad y sexo ajustado en 2019-2022. Como referencias se usan dos modelos: el de "
        "edad y sexo, con un AUC de {a|dec3} en 2023, y el de las categorías de Elixhauser solas (ELIX_01 a ELIX_31, sin edad ni sexo), con {b|dec3}. El modelo con penalización L1 "
        "reúne las variables que la Fase 1 no excluye y elige su penalización por validación cruzada (C igual a {c|dec3}); su AUC en 2023 es {l1|dec3}. Variables que sobreviven en "
        "él: {n_sob}. Variables que importan al permutarlas (caída del AUC de al menos {minimo|dec3}): {n_imp}. Los coeficientes y las importancias de las demás variables están "
        "condicionados a la presencia de las condicionales cercanas a ser proxy del hospital (las de trama en la figura), cuyos coeficientes pueden codificar el hospital.",
        a=rs("auc_ref_edad_sexo_2023"), b=rs("auc_ref_elix31_2023"), c=rs("l1_c"), l1=rs("auc_modelo_l1_2023"), n_sob=rs("l1_n_variables_sobreviven"), n_imp=rs("n_importan"),
        minimo=umbral("importancia_permutacion_min"),
    )
    figura("F11_05")
    importantes = imp.filter((pl.col("tipo_fila") == "variable") & (pl.col("en_modelo_l1") == "si") & pl.col("importancia_permutacion").is_not_null()).sort(
        ["importancia_permutacion", "variable"], descending=[True, False], maintain_order=True
    ).head(12)
    if importantes.height:
        inf.tabla(
            "t23_importancia.csv", df=importantes, columnas=["variable", "clase_proxy", "incremento_auc", "l1_sobrevive", "importancia_permutacion", "importancia_sd", "importa"],
            encabezados={"variable": "Variable", "clase_proxy": "Clase de proxy", "incremento_auc": "Incremento de AUC", "l1_sobrevive": "Sobrevive en L1",
                         "importancia_permutacion": "Caída del AUC", "importancia_sd": "Desviación", "importa": "Importa"},
            formatos={"incremento_auc": "dec3", "importancia_permutacion": "dec4", "importancia_sd": "dec4"}, leyenda="Las doce variables del modelo L1 con mayor importancia por permutación en 2023",
        )

    # --- 7. fuga ----------------------------------------------------------------------------------------------------------------------------------------
    fuga = ctx.leer_tabla("t23_fuga.csv", columnas_texto=_TEXTO_FUGA_LISTA).with_columns([_num(pl.col(c)) for c in ("auc_ranking", "auc_confirmacion")])
    inf.seccion(f"{n0 + 6}. Fuga")
    inf.parrafo(
        "Las variables que solo existen al alta o durante la hospitalización, prohibidas por la Fase 1, y las condicionales que podrían codificar lo que ocurre después del ingreso se "
        "miden con el mismo AUC univariado y se comparan con dos modelos basales y con el umbral de sospecha de fuga ({umbral|dec2}). Variables que alcanzan el umbral en el ranking "
        "o en la confirmación: {n}. `IR_29301_MORTALIDAD` queda prohibida siempre (R6), aunque su señal sea baja. La concordancia entre SERVICIOINGRESO y SERVICIOALTA (S09) es de "
        "{sin|prop1} en los egresos sin traslado interno y de {con|prop1} en los que tienen traslado.",
        umbral=umbral("auc_sospecha_fuga"), n=rs("fuga_n_supera_umbral"), sin=rs("concordancia_prop_sin_traslado"), con=rs("concordancia_prop_con_traslado"),
    )
    figura("F11_06")
    vf = fuga.filter(pl.col("tipo_fila") == "variable").sort(["auc_confirmacion", "auc_ranking", "variable"], descending=[True, True, False], nulls_last=True, maintain_order=True)
    if vf.height:
        inf.tabla(
            "t23_fuga.csv", df=vf, columnas=["variable", "categoria", "auc_ranking", "auc_confirmacion", "referencia_maxima", "supera_a_basales", "supera_umbral_fuga"],
            encabezados={"variable": "Variable", "categoria": "Categoría", "auc_ranking": "AUC 2019-2022", "auc_confirmacion": "AUC 2023", "referencia_maxima": "Referencia mayor",
                         "supera_a_basales": "Supera a los basales", "supera_umbral_fuga": "Supera el umbral"},
            formatos={"auc_ranking": "dec3", "auc_confirmacion": "dec3"}, leyenda="AUC de las variables de solo fuga y de las condicionales frente a los modelos basales",
        )

    # --- 8. matriz de decisión (decision) -------------------------------------------------------------------------------------------------------------
    decision.agregar_secciones(inf, ctx, n0 + 7)

    # --- 9. advertencias y límites ---------------------------------------------------------------------------------------------------------------------
    inf.seccion(f"{n0 + 8}. Advertencias y límites")
    lineas = []
    if extracto:
        lineas.append(
            "- Los datos son un extracto y no una muestra aleatoria simple: nada de lo anterior es concluyente. Las cifras prueban la mecánica y no sirven para decidir qué "
            "variables se mantienen."
        )
    lineas += [
        "- Los intervalos de confianza ignoran el agrupamiento por paciente y por hospital, y los de la selección de variables tampoco lo consideran.",
        "- Las categóricas puntuadas con la tasa de la misma ventana son optimistas; por eso se ordenan por validación cruzada y se marcan en las figuras.",
    ]
    if res["historial_evaluable"] != "si":
        lineas.append("- El historial del paciente no es evaluable en esta corrida: depende de los datos completos y de la decisión 1.")
    lineas += [
        "- ETNIA y PUEBLO_ORIGINARIO no se analizan frente a los resultados (decisión 4); solo se describen y se mide su asociación con el hospital.",
        "- El año 2023 sirve a la vez de confirmación de la selección y de calibración del modelo final: seleccionar con él puede volver levemente optimista la calibración. El año 2024 "
        "no se lee.",
        "- Los odds ratio y la importancia por permutación pueden subestimar variables correlacionadas entre sí (los clusters de la Tarea 10).",
        "- El criterio clínico está pendiente de la decisión 12: solo las fichas «verificada en fuente» activan la regla de mantener por criterio clínico; el resto queda pendiente de "
        "revisión clínica y la lista de pendientes está en el diseño.",
        "- Los coeficientes y las importancias de las demás variables están condicionados a la presencia de las condicionales cercanas a proxy del hospital (las de clase de proxy "
        "moderada o alta), cuyos coeficientes pueden codificar el hospital.",
        "- El relleno de los nulos con la mediana del entrenamiento más un indicador de nulos solo ocurre dentro de los modelos exploratorios; la regla R5 de la Fase 1 lo admite para "
        "las líneas base y el panel conserva los nulos.",
        "- Las tablas de proxy, importancia y fuga llevan conteos de dos ventanas (2019-2022 y 2023) en la misma fila; por eso se escriben por bloques de supresión en lugar de separarlas "
        "en dos tablas, que es la regla general de privacidad del proyecto. Cada bloque suprime solo sus propias cifras y todos los conteos quedan registrados para el escáner de "
        "privacidad.",
    ]
    inf.texto_libre_sin_cifras("\n".join(lineas))
    inf.parrafo("Las constantes técnicas de la tarea, con su valor y los resultados que afectan, están en la tabla siguiente. Ninguna es un umbral del protocolo.")
    inf.tabla(
        "t23_parametros_seleccion.csv", df=ctx.leer_tabla("t23_parametros_seleccion.csv", columnas_texto=COLUMNAS_PARAMETROS), columnas=COLUMNAS_PARAMETROS,
        encabezados={"modulo": "Módulo", "parametro": "Parámetro", "valor": "Valor", "para_que_sirve": "Para qué sirve", "afecta": "Qué afecta"},
        leyenda="Parámetros técnicos de la selección de variables",
    )


# FIN-seleccion
