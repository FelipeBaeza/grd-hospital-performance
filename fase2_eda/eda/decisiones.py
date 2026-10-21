"""Tarea 12 (parte 2): evidencia para las 17 decisiones pendientes de la Fase 1 y verificación de los supuestos S01 a S15.

La evidencia RECOMIENDA y no decide: cada decisión queda "pendiente de decisión humana". Cada hipótesis de ``config/hipotesis.csv`` se evalúa con una función pura sobre una
``FuenteEvidencia`` (tablas ya publicadas, umbrales y métricas), de modo que las reglas se prueban con DataFrames hechos a mano; ``FuenteContexto`` envuelve un ``Contexto``.
Una tabla ausente o una celda nula (suprimida) deja la hipótesis ``no_evaluable`` con su motivo, nunca una excepción.

Resultado de una hipótesis: ``respalda`` (el criterio se cumple en el sentido de la propuesta de la Fase 1), ``matiza`` (se cumple en parte, es condicional o solo describe),
``contradice`` o ``no_evaluable``. Clase: ``decisiva`` (el criterio trae una regla o un umbral) o ``informativa`` (solo "se informa"). Una hipótesis que requiere los datos
completos (el historial, los enlaces y el volumen por hospital se subestiman por construcción en un extracto) queda ``no_evaluable`` en un extracto sin leer su tabla.

El semáforo de cada decisión (``agregar_semaforo``) usa solo las hipótesis primarias de la decisión: gris si la decisión es institucional o no depende de datos o si una
hipótesis que requiere los datos completos no es evaluable; rojo si alguna decisiva evaluable contradice; verde si todas las decisivas respaldan y ninguna quedó sin evaluar;
amarillo si hay evidencia que matiza o solo describe; gris en otro caso. La Tarea 12 no lee datos: solo CSV publicados y métricas.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import polars as pl

from . import config, graficos, ledger, protocolo
from .contexto import Contexto
from .informes import Informe, cifra, fmt_dec, fmt_entero
from .sospechas import clasificar_proxy

TAREA = "T12"
ANIOS_TXT = "2019 a 2024"
ESTADO_DECISION = "pendiente de decisión humana"
SEMAFOROS = ("verde", "amarillo", "rojo", "gris")
RESULTADOS = ("respalda", "matiza", "contradice", "no_evaluable")
TIPOS_EVIDENCIA = ("cuantitativa", "parcial", "institucional", "no depende de datos")
INSTITUCIONALES = frozenset({10, 12})
"""Decisiones cuyo criterio es "Decisión institucional" en ``hipotesis.csv`` (una prueba lo verifica); la 4 no depende de datos pero es metodológica."""
VARIABLES_EQUIDAD = ("PREVISION", "NACIONALIDAD", "ES_EXTRANJERO", "PROVINCIA", "COMUNA")
VARIABLES_HISTORIAL = ("N_EGRESOS_12M", "DIAS_DESDE_EGRESO_PREVIO")
CIERRE_EXTRACTO = "las cifras del extracto son indicativas: repetir con los datos completos antes de decidir"
MOTIVO_REQUIERE_COMPLETO = "requiere datos completos: en un extracto esta evidencia se subestima por construcción"
MOTIVO_SIN_DATOS = "no depende de datos"
"""Motivo de las hipótesis que ninguna corrida resuelve (decisiones metodológicas o institucionales: H04, H10 y H12)."""
SIN_FALTANTE = "nada pendiente de datos; falta la decisión humana"

FALTA_CON_DATOS_COMPLETOS: dict[str, str] = {
    "H01a": "el traslape de identificadores entre años consecutivos con todos los egresos",
    "H01b": "la factibilidad del historial de doce meses con el identificador y la ventana completa de cada egreso",
    "H01c": "la señal, el valor incremental y el proxy del hospital de las variables de historial, que dependen del ingreso previo del paciente",
    "H02a": "la asociación del servicio de ingreso con el hospital y su señal con todos los egresos",
    "H02b": "la asociación de la especialidad médica con el hospital y su cardinalidad por hospital con todos los egresos",
    "H02c": "la autorreferencia del hospital de procedencia con todos los egresos",
    "H03": "la asociación de las variables de equidad con el hospital con todos los egresos",
    "H04": "decisión metodológica de la Fase 1: la Fase 2 solo documenta la calidad de la etnia",
    "H05a": "el efecto de limitar la cohorte a adultos sobre la prevalencia con todos los egresos",
    "H05b": "la lista de hospitales pediátricos o monográficos con todos los egresos",
    "H06a": "cuántos hospitales cumplen el volumen mínimo de egresos y de muertes esperadas",
    "H06b": "la proporción de celdas por hospital bajo el mínimo publicable",
    "H07": "la cola de la distribución de estadías con todos los sobrevivientes",
    "H08": "el contraste de las categorías de Elixhauser con las complicaciones de la hospitalización con todos los egresos",
    "H09": "la dispersión de la estadía de los sobrevivientes con todos los egresos",
    "H10": "decisión institucional sobre si el prototipo muestra los nombres de los hospitales",
    "H11": "el conteo de valores con estructura de RUT en todas las columnas y años",
    "H12": "decisión institucional sobre quién revisa las decisiones del dominio",
    "H13": "la tasa de enlace de los traslados salientes con un ingreso posterior en un espacio de identificadores consistente",
    "H14": "la señal de la severidad del GRD frente a las variables basales con todos los egresos",
    "H15a": "la composición de las estadías de cero días con todos los egresos",
    "H15b": "la mortalidad de los estratos ambulatorio y de estadía cero con todos los egresos",
    "H16a": "la proporción de traslados entrantes que se enlaza con un traslado saliente previo",
    "H16b": "la mortalidad de los entrantes y de los demás ingresos con todos los egresos",
    "H17a": "la deriva de las variables clave entre 2019 y 2020 a 2022 con todos los egresos",
    "H17b": "los quiebres de la serie mensual con todos los egresos",
}


# ============================================================================================================================ fuente de evidencia
class FuenteEvidencia(Protocol):
    """Lo que necesitan las evaluaciones: tablas publicadas, umbrales, métricas y las cifras documentadas. Todo método de lectura devuelve ``None`` si no existe."""

    etiqueta: str

    def tabla(self, nombre: str, columnas_texto: Sequence[str] = ()) -> pl.DataFrame | None: ...

    def umbral(self, nombre: str) -> Any: ...

    def metrica(self, clave: str) -> Mapping[str, Any] | None: ...

    def cifra_documentada(self, id_cifra: str) -> float | None: ...


class FuenteContexto:
    """``FuenteEvidencia`` sobre un ``Contexto``: lee las tablas de ``ctx.dir_tablas``, los umbrales del protocolo y las métricas de la corrida (memoria y disco)."""

    def __init__(self, ctx: Contexto) -> None:
        self._ctx = ctx
        self.etiqueta = ctx.etiqueta
        self._metricas = ledger.metricas_disponibles(ctx)
        self._cifras: dict[str, float] | None = None

    def tabla(self, nombre: str, columnas_texto: Sequence[str] = ()) -> pl.DataFrame | None:
        try:
            return self._ctx.leer_tabla(nombre, columnas_texto=columnas_texto)
        except FileNotFoundError:
            return None

    def umbral(self, nombre: str) -> Any:
        u = self._ctx.umbrales
        return u[nombre] if nombre in u.nombres() else u.particion[nombre]

    def metrica(self, clave: str) -> Mapping[str, Any] | None:
        return self._metricas.get(clave)

    def cifra_documentada(self, id_cifra: str) -> float | None:
        if self._cifras is None:
            self._cifras = {f["id"]: float(f["valor"]) for f in ledger.cargar_cifras().iter_rows(named=True)}
        return self._cifras.get(id_cifra)


class _UmbralesDe:
    """Adaptador para las funciones de otros módulos que indexan los umbrales como ``u["nombre"]``."""

    def __init__(self, fuente: FuenteEvidencia) -> None:
        self._f = fuente

    def __getitem__(self, nombre: str) -> Any:
        return self._f.umbral(nombre)


class _SinDato(Exception):
    """La evidencia no se puede leer: falta la tabla, la fila o la celda (por ejemplo, está suprimida)."""


@dataclass(frozen=True)
class Resultado:
    resultado: str
    veredicto: str
    tabla: str
    cifras_clave: str
    motivo: str = ""


@dataclass(frozen=True)
class Hipotesis:
    id: str
    decision: int
    clase: str
    requiere_completo: bool
    evaluador: Callable[[FuenteEvidencia, Mapping[str, Resultado]], Resultado]
    tablas: tuple[str, ...]


# ============================================================================================================================ lectura y formato
def _tabla(f: FuenteEvidencia, nombre: str, columnas_texto: Sequence[str] = ()) -> pl.DataFrame:
    t = f.tabla(nombre, columnas_texto)
    if t is None:
        raise _SinDato(f"falta la tabla {nombre}")
    return t


def _fila(t: pl.DataFrame, **filtro: Any) -> dict[str, Any]:
    """La única fila que cumple el filtro (comparación como texto); si no hay exactamente una, no hay dato."""
    sel = t
    for k, v in filtro.items():
        if k not in sel.columns:
            raise _SinDato(f"la tabla no tiene la columna {k}")
        sel = sel.filter(pl.col(k).cast(pl.String) == str(v))
    if sel.height != 1:
        raise _SinDato(f"la tabla no trae una sola fila para {', '.join(f'{k} = {v}' for k, v in filtro.items())}" if filtro else "la tabla no trae una sola fila")
    return sel.row(0, named=True)


def _cel(fila: Mapping[str, Any], col: str) -> Any:
    if col not in fila:
        raise _SinDato(f"la tabla no tiene la columna {col}")
    v = fila[col]
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        raise _SinDato(f"{col} no tiene valor (suprimido o vacío)")
    return v


def _si_no(fila: Mapping[str, Any], col: str) -> bool:
    v = str(_cel(fila, col)).strip().lower()
    if v not in ("si", "no"):
        raise _SinDato(f"{col} no es si ni no")
    return v == "si"


def _ent(v: Any) -> str:
    return fmt_entero(v)


def _d(v: Any, n: int = 2) -> str:
    return fmt_dec(v, n)


def _p(v: Any, n: int = 1) -> str:
    """Proporción (0 a 1) como porcentaje con coma decimal."""
    return f"{fmt_dec(float(v) * 100.0, n)} %"


def _periodo_agregado(f: FuenteEvidencia) -> str:
    return ledger.etiqueta_alcance([int(a) for a in f.umbral("anios_resultado")])


def _clase_proxy(f: FuenteEvidencia, valor: Any) -> str:
    clase = clasificar_proxy(None if valor is None else float(valor), _UmbralesDe(f))
    if clase == "no evaluable":
        raise _SinDato("la asociación con el hospital no tiene valor")
    return clase


def _r(resultado: str, veredicto: str, tablas: str, cifras: str, motivo: str = "") -> Resultado:
    return Resultado(resultado, veredicto, tablas, cifras, motivo)


# ============================================================================================================================ decisión 1
def _h01a(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t24_resumen.csv"
    r = _fila(_tabla(f, t, ["quiebres", "regimenes"]))
    n = int(_cel(r, "n_quiebres"))
    cifras = f"quiebres del identificador entre años consecutivos = {_ent(n)}" + (f" ({r['quiebres']})" if n > 0 else "") + f"; espacios de identificadores = {r['regimenes']}"
    if n == 0:
        return _r("respalda", "el identificador es continuo entre años consecutivos", t, cifras)
    return _r("matiza", f"el identificador tiene {_ent(n)} quiebre entre años consecutivos: el historial no cruza el quiebre", t, cifras)


def _h01b(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t23_historial_factibilidad.csv"
    d = _tabla(f, t, ["regimen", "cumple_minimo"])
    filas = d.to_dicts()
    primero: dict[str, int] = {}
    for r in filas:
        primero[r["regimen"]] = min(primero.get(r["regimen"], int(r["anio"])), int(r["anio"]))
    evaluables = [r for r in filas if int(r["anio"]) != primero[r["regimen"]]]            # el primer año de cada espacio no tiene ventana completa por construcción
    if not evaluables:
        raise _SinDato("ningún año tiene ventana de doce meses completa dentro de su espacio de identificadores")
    cumplen = [_si_no(r, "cumple_minimo") for r in evaluables]
    detalle = "; ".join(f"{r['anio']}: {_p(_cel(r, 'factibilidad'))}" for r in evaluables)
    cifras = f"años evaluables = {_ent(len(evaluables))}; cumplen el mínimo = {_ent(sum(cumplen))} ({detalle}); mínimo = {_p(f.umbral('historial_factibilidad_minima'), 0)}"
    if all(cumplen):
        return _r("respalda", "el historial de doce meses es viable en todos los años evaluables", t, cifras)
    if not any(cumplen):
        return _r("contradice", "el historial de doce meses no es viable en ningún año evaluable", t, cifras)
    return _r("matiza", "el historial de doce meses es viable solo en algunos años", t, cifras)


def _h01c(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "matriz_decision_variables.csv"
    h01b = previos.get("H01b")
    if h01b is None or h01b.resultado == "no_evaluable":
        raise _SinDato("la factibilidad del historial (H01b) no es evaluable")
    m = _tabla(f, t, ["variable", "medicion_completa"])
    inc_min, prox = float(f.umbral("incremento_auc_minimo")), float(f.umbral("proxy_hospital_moderado"))
    cumplen, partes = [], []
    for v in VARIABLES_HISTORIAL:
        r = _fila(m, variable=v)
        if str(r.get("medicion_completa")) != "si":
            raise _SinDato(f"{v} no tiene medición completa")
        inc, asoc = float(_cel(r, "incremento_auc")), float(_cel(r, "asociacion_hospital"))
        cumplen.append(inc > inc_min and asoc < prox)
        partes.append(f"{v}: incremento de AUC = {_d(inc, 3)}, asociación con el hospital = {_d(asoc, 3)}")
    cifras = "; ".join(partes) + f"; incremento mínimo = {_d(inc_min, 3)}; proxy moderado desde {_d(prox, 2)}"
    if all(cumplen) and h01b.resultado == "respalda":
        return _r("respalda", "las dos variables de historial aportan señal sin ser proxy del hospital y la factibilidad se cumple", t, cifras)
    if not any(cumplen):
        return _r("contradice", "ninguna variable de historial aporta señal sin ser proxy del hospital", t, cifras)
    return _r("matiza", "las variables de historial cumplen en parte el criterio de señal, proxy y factibilidad", t, cifras)


# ============================================================================================================================ decisión 2
def _proxy_variable(f: FuenteEvidencia, variable: str) -> tuple[float, str]:
    p = _fila(_tabla(f, "t23_proxy_hospital.csv", ["tipo_fila", "variable", "clase_proxy"]), tipo_fila="variable", variable=variable)
    asoc = float(_cel(p, "asociacion_hospital"))
    return asoc, _clase_proxy(f, asoc)


def _h02a(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    tablas = "t23_proxy_hospital.csv, t23_fuga.csv"
    asoc, clase = _proxy_variable(f, "SERVICIOINGRESO")
    g = _fila(_tabla(f, "t23_fuga.csv", ["tipo_fila", "variable", "supera_umbral_fuga"]), tipo_fila="variable", variable="SERVICIOINGRESO")
    fuga = _si_no(g, "supera_umbral_fuga")
    cifras = (f"SERVICIOINGRESO: asociación con el hospital = {_d(asoc, 3)} ({clase}); "
              f"{'supera' if fuga else 'no supera'} el umbral de fuga de {_d(f.umbral('auc_sospecha_fuga'), 2)}")
    if clase == "alto" or fuga:
        return _r("contradice", "el servicio de ingreso se parece a un proxy del hospital o a una fuga: se propone excluirlo", tablas, cifras)
    if clase == "moderado":
        return _r("matiza", "el servicio de ingreso queda condicional por su asociación con el hospital", tablas, cifras)
    return _r("respalda", "el servicio de ingreso no es proxy del hospital ni fuga", tablas, cifras)


def _h02b(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t23_proxy_hospital.csv"
    asoc, clase = _proxy_variable(f, "ESPECIALIDAD_MEDICA")
    cifras = f"ESPECIALIDAD_MEDICA: asociación con el hospital = {_d(asoc, 3)} ({clase})"
    try:
        c = _fila(_tabla(f, t, ["tipo_fila", "variable"]), tipo_fila="cardinalidad_especialidad", variable="ESPECIALIDAD_MEDICA")
        cifras += f"; niveles en total = {_ent(_cel(c, 'n_niveles_totales'))}; niveles usados por un solo hospital = {_ent(_cel(c, 'n_niveles_en_un_hospital'))}"
    except _SinDato:
        pass                                                                                 # la cardinalidad solo se informa
    if clase == "alto":
        return _r("contradice", "la especialidad médica identifica al hospital: se propone excluirla", t, cifras)
    if clase == "moderado":
        return _r("matiza", "la especialidad médica queda condicional por su asociación con el hospital", t, cifras)
    return _r("respalda", "la especialidad médica no identifica al hospital", t, cifras)


def _h02c(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t23_resumen_hospitales.csv"
    r = _fila(_tabla(f, t, ["autorref_supera_maximo"]))
    supera = _si_no(r, "autorref_supera_maximo")
    cifras = f"HOSPPROCEDENCIA con el hospital del egreso = {_p(_cel(r, 'autorref_prop_exacta'))} (máximo aceptado {_p(f.umbral('autorreferencia_procedencia_max'), 0)})"
    if supera:
        return _r("matiza", "HOSPPROCEDENCIA no define el traslado entrante: se propone usar TIPO_PROCEDENCIA", t, cifras)
    return _r("respalda", "HOSPPROCEDENCIA puede definir el traslado entrante", t, cifras)


# ============================================================================================================================ decisión 3
def _h03(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t23_proxy_hospital.csv"
    d = _tabla(f, t, ["tipo_fila", "variable", "clase_proxy"])
    partes, clases = [], []
    for v in VARIABLES_EQUIDAD:
        try:
            asoc, clase = _proxy_variable_en(f, d, v)
        except _SinDato:
            continue
        partes.append(f"{v} {_d(asoc, 3)} ({clase})")
        clases.append(clase)
    if not clases:
        raise _SinDato("ninguna variable de equidad tiene asociación con el hospital")
    cifras = "asociación con el hospital: " + "; ".join(partes)
    if any(c in ("alto", "moderado") for c in clases):
        return _r("respalda", "alguna variable de equidad es proxy del hospital: conviene mantenerlas fuera del modelo principal", t, cifras)
    return _r("matiza", "ninguna variable de equidad es proxy del hospital: la reserva se justifica por equidad y no por proxy", t, cifras)


def _proxy_variable_en(f: FuenteEvidencia, d: pl.DataFrame, variable: str) -> tuple[float, str]:
    p = _fila(d, tipo_fila="variable", variable=variable)
    asoc = float(_cel(p, "asociacion_hospital"))
    return asoc, _clase_proxy(f, asoc)


# ============================================================================================================================ decisión 4
def _h04(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t22_deriva_categorias.csv"
    cifras = ""
    d = f.tabla(t, ["columna", "categoria"])
    if d is not None and "columna" in d.columns:
        et = d.filter(pl.col("columna") == "ETNIA")
        if et.height:
            cifras = f"categorías distintas de ETNIA en los archivos = {_ent(et['categoria'].n_unique())}"
    return _r("no_evaluable", "decisión metodológica de la Fase 1: la Fase 2 solo documenta la calidad", t, cifras, MOTIVO_SIN_DATOS)


# ============================================================================================================================ decisión 5
def _h05a(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t23_cascada_variantes.csv"
    per = _periodo_agregado(f)
    d = _tabla(f, t, ["periodo", "variante"])
    a = _fila(d, periodo=per, variante="A")
    b = _fila(d, periodo=per, variante="B")
    # ``dif_prevalencia_pp`` de la variante B se calcula sobre ``prevalencia_conocida`` (fallecidos entre los egresos con desenlace conocido): las dos prevalencias que se citan
    # usan esa misma base, porque con la de ``prevalencia`` (sobre todos los egresos) la resta no concilia con la diferencia publicada
    pa, pb = float(_cel(a, "prevalencia_conocida")), float(_cel(b, "prevalencia_conocida"))
    dif = _cel(b, "dif_prevalencia_pp")
    cifras = (f"{per}: prevalencia entre los egresos con desenlace conocido de la cohorte dura (A) = {_p(pa, 2)}; solo adultos (B) = {_p(pb, 2)}; "
              f"diferencia = {_d(dif, 2)} puntos porcentuales (calculada con los valores sin redondear)")
    return _r("matiza", "limitar la cohorte a adultos cambia la prevalencia; el efecto se informa y la decisión es humana", t, cifras)


def _h05b(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t23_hospitales_resumen.csv"
    d = _tabla(f, t)
    con = d.filter(pl.col("n_hospitales_con_cohorte").is_not_null()) if "n_hospitales_con_cohorte" in d.columns else d
    if con.height == 0:
        raise _SinDato("ningún año tiene hospitales con cohorte")
    r = con.sort("anio").row(-1, named=True)
    cifras = f"{r['anio']}: hospitales pediátricos o monográficos = {_ent(_cel(r, 'n_pediatricos'))} de {_ent(_cel(r, 'n_hospitales'))}"
    return _r("matiza", "hay hospitales pediátricos o monográficos que la restricción a adultos deja fuera; el efecto se informa", t, cifras)


# ============================================================================================================================ decisión 6
def _h06a(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t23_volumen_umbrales.csv"
    d = _tabla(f, t)
    eg, mu = int(f.umbral("volumen_minimo_egresos")), int(f.umbral("muertes_esperadas_minimas"))
    sel = d.filter((pl.col("egresos_min") == eg) & (pl.col("muertes_esperadas_min") == mu)).sort("anio")
    if sel.height == 0:
        raise _SinDato("la tabla no trae la combinación de volumen propuesta")
    partes = [f"{r['anio']}: {_ent(_cel(r, 'n_cumplen_ambos'))} de {_ent(_cel(r, 'n_hospitales'))}" for r in sel.iter_rows(named=True)]
    cifras = f"hospitales que cumplen {_ent(eg)} egresos y {_ent(mu)} muertes esperadas: " + "; ".join(partes)
    return _r("matiza", "se informa cuántos hospitales cumplen el volumen mínimo propuesto en cada año", t, cifras)


def _h06b(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t23_hospitales_resumen.csv"
    d = _tabla(f, t).filter(pl.col("prop_celdas_bajo_minimo").is_not_null()).sort("anio")
    if d.height == 0:
        raise _SinDato("ningún año trae la proporción de celdas bajo el mínimo")
    cifras = "celdas de hospital bajo el mínimo publicable: " + "; ".join(f"{r['anio']}: {_p(r['prop_celdas_bajo_minimo'])}" for r in d.iter_rows(named=True))
    return _r("matiza", "se informa la proporción de celdas por hospital que queda suprimida", t, cifras)


# ============================================================================================================================ decisión 7
def _h07(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t23_cola_estancia.csv"
    d = _tabla(f, t, ["base", "regla", "cumple_def", "cumple_conf", "propuesta"])
    prop = d.filter(pl.col("propuesta") == "si")
    if prop.height == 0:
        return _r("contradice", "ninguna regla de corte afecta a no más de la proporción máxima de sobrevivientes",
                  t, f"reglas candidatas = {_ent(d.height)}; ninguna cumple el máximo de {_p(f.umbral('cola_estadia_perdida_max'), 1)}")
    filas = prop.to_dicts()
    partes = [f"regla {r['regla']} (base {r['base']}): corte en {_ent(_cel(r, 'corte_dias'))} días, definición {r['cumple_def']}, confirmación {r['cumple_conf']}" for r in filas]
    cifras = "; ".join(partes) + f"; pérdida máxima = {_p(f.umbral('cola_estadia_perdida_max'), 1)}"
    if any(_si_no(r, "cumple_def") and _si_no(r, "cumple_conf") for r in filas):
        return _r("respalda", "la regla propuesta se cumple en la definición y se confirma en el año de confirmación", t, cifras)
    return _r("matiza", "la regla propuesta se cumple en la definición pero no se confirma en el año de confirmación", t, cifras)


# ============================================================================================================================ decisión 8
def _h08(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t23_resumen_seleccion.csv"
    r = _fila(_tabla(f, t, ["elix_top5"]))
    n = int(_cel(r, "elix_n_coinciden_fase1"))
    repo = int(_cel(r, "elix_n_coinciden_repo"))
    cifras = f"cinco categorías con mayor índice de complicación = {r['elix_top5']}; coinciden con la lista de la Fase 1 = {_ent(n)}; con la del repositorio = {_ent(repo)}"
    if n >= 5:
        return _r("respalda", "las cinco categorías más propensas a ser complicaciones coinciden con la lista de la Fase 1", t, cifras)
    if n == 0:
        return _r("contradice", "ninguna de las cinco categorías más propensas coincide con la lista de la Fase 1", t, cifras)
    return _r("matiza", "la lista de la Fase 1 coincide solo en parte con las categorías más propensas a ser complicaciones; la lista final queda pendiente de revisión clínica", t, cifras)


# ============================================================================================================================ decisión 9
def _o(fila: Mapping[str, Any], col: str, n: int = 3) -> str:
    """Cifra opcional de una fila (``sin dato`` si no existe o está vacía)."""
    v = fila.get(col)
    return "sin dato" if v is None else _d(v, n)


def _h09(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t23_estancia_dispersion.csv"
    per = _periodo_agregado(f)
    r = _fila(_tabla(f, t, ["periodo", "base"]), periodo=per, base="E")
    idx = float(_cel(r, "indice_dispersion"))
    cifras = f"{per}: índice de dispersión de la estadía de los sobrevivientes (varianza sobre media) = {_d(idx, 2)}; media = {_o(r, 'media', 2)} días"
    return _r("matiza", "la dispersión de la estadía es evidencia complementaria para elegir entre las distribuciones Poisson y Tweedie", t, cifras)


# ============================================================================================================================ decisiones 10 y 12
def _h10(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    return _r("no_evaluable", "decisión institucional: se aplica la salvaguarda de códigos y aviso de uso interno en los gráficos por hospital", "figuras_indice.csv", "", MOTIVO_SIN_DATOS)


def _h12(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    return _r("no_evaluable", "decisión institucional: lo no verificable queda marcado como pendiente de revisión clínica", "conocimiento_clinico.csv", "", MOTIVO_SIN_DATOS)


# ============================================================================================================================ decisión 11
def _h11(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t24_rut_por_columna.csv"
    m = f.metrica("calidad.valores_rut.total")
    if m is None:
        raise _SinDato("falta la métrica calidad.valores_rut.total")
    v = m.get("valor")
    if v is None:
        if "[suprimido" in str(m.get("nota") or ""):
            return _r("respalda", "hay valores con estructura de RUT, pero menos de diez: el conteo no se publica", t, "valores con estructura de RUT = entre 1 y 9 (suprimido)")
        raise _SinDato("la métrica calidad.valores_rut.total no tiene valor")
    if float(v) > 0:
        return _r("respalda", "hay valores con estructura de RUT que informar a FONASA por el canal formal", t, f"valores con estructura de RUT = {_ent(v)}")
    return _r("matiza", "no se encontraron valores con estructura de RUT en esta fuente", t, f"valores con estructura de RUT = {_ent(v)}")


# ============================================================================================================================ decisión 13
def _h13(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t23_resumen_hospitales.csv"
    r = _fila(_tabla(f, t, ["enlace_cumple_minimo"]))
    cumple = _si_no(r, "enlace_cumple_minimo")
    cifras = (f"tasa de enlace de los traslados salientes a {_ent(_cel(r, 'enlace_ventana_dias'))} días en {_cel(r, 'enlace_anio')} = {_p(_cel(r, 'enlace_tasa'))}; "
              f"mínimo = {_p(f.umbral('enlace_traslado_minimo'), 0)}")
    if cumple:
        return _r("respalda", "el rastreo de traslados salientes se justifica: la tasa de enlace alcanza el mínimo", t, cifras)
    return _r("contradice", "el rastreo de traslados salientes no se justifica: la tasa de enlace no alcanza el mínimo", t, cifras)


# ============================================================================================================================ decisión 14
def _h14(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t23_fuga.csv"
    d = _tabla(f, t, ["tipo_fila", "variable", "supera_a_basales", "supera_umbral_fuga", "prohibida_siempre"])
    s = _fila(d, tipo_fila="variable", variable="IR_29301_SEVERIDAD")
    m = _fila(d, tipo_fila="variable", variable="IR_29301_MORTALIDAD")
    basales, fuga = _si_no(s, "supera_a_basales"), _si_no(s, "supera_umbral_fuga")
    umbral = float(f.umbral("auc_sospecha_fuga"))
    ventanas = [txt for txt, col in (("el ranking de 2019 a 2022", "auc_ranking"), ("la confirmación de 2023", "auc_confirmacion")) if s.get(col) is not None and float(s[col]) >= umbral]
    donde = f" en {' y '.join(ventanas)}" if fuga and ventanas else ""                      # la marca de la tabla mira las dos ventanas: se dice cuál alcanza el umbral
    cifras = (f"IR_29301_SEVERIDAD: AUC de ranking = {_o(s, 'auc_ranking')} y de confirmación = {_o(s, 'auc_confirmacion')}; {'supera' if basales else 'no supera'} a las variables basales; "
              f"{'supera' if fuga else 'no supera'} el umbral de fuga de {_d(umbral, 2)}{donde}; "
              f"IR_29301_MORTALIDAD (prohibida siempre por R6): AUC de ranking = {_o(m, 'auc_ranking')} y de confirmación = {_o(m, 'auc_confirmacion')}")
    if basales and fuga:
        return _r("respalda", "la severidad del GRD revela el desenlace: corresponde a una sensibilidad aparte", t, cifras)
    return _r("matiza", "la severidad del GRD no supera a la vez a las variables basales y el umbral de fuga", t, cifras)


# ============================================================================================================================ decisión 15
def _h15a(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t23_resumen.csv"
    r = _fila(_tabla(f, t, ["intento_2024"]))
    partes = [f"estadías de cero días = {_ent(_cel(r, 'n_cero_total'))}"]
    for col, texto in (("prop_cero_ambulatoria", "de actividad ambulatoria"), ("prop_cero_fallecidos", "con fallecimiento"), ("prop_cero_censurados", "censuradas")):
        if r.get(col) is not None:
            partes.append(f"{texto} = {_p(r[col])}")
    return _r("matiza", "se informa la composición de las estadías de cero días; la decisión es humana", t, "; ".join(partes))


def _h15b(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t23_resultado_estratos.csv"
    per = _periodo_agregado(f)
    d = _tabla(f, t, ["periodo", "variable", "categoria", "bajo_umbral"])
    amb = _fila(d, periodo=per, variable="actividad_grupo", categoria="ambulatoria")
    cero = _fila(d, periodo=per, variable="estancia_cero", categoria="0 días")
    b1, b2 = _si_no(amb, "bajo_umbral"), _si_no(cero, "bajo_umbral")
    cifras = (f"{per}: mortalidad del estrato ambulatorio = {_d(_cel(amb, 'razon_vs_resto'), 2)} veces la del resto; "
              f"del estrato de estadía cero = {_d(_cel(cero, 'razon_vs_resto'), 2)} veces; umbral = {_d(f.umbral('estrato_mortalidad_relativa_max'), 2)} veces")
    if not b1 and not b2:
        return _r("respalda", "ningún estrato diluye la mortalidad: no hace falta sacarlos del modelo", t, cifras)
    return _r("matiza", "algún estrato tiene una mortalidad relativa bajo el umbral: se propone sacarlo del modelo de mortalidad (variante C)", t, cifras)


# ============================================================================================================================ decisión 16
def _h16a(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t23_enlaces_traslado.csv"
    per = _periodo_agregado(f)
    w = int(max(f.umbral("ventanas_traslado_dias")))
    r = _fila(_tabla(f, t, ["periodo", "direccion"]), periodo=per, direccion="entrante", ventana_dias=w)
    cifras = f"{per}: entrantes enlazados con un traslado saliente previo a {_ent(w)} días = {_p(_cel(r, 'tasa_enlace'))} de {_ent(_cel(r, 'n_elegibles'))} elegibles"
    return _r("matiza", "se informa la proporción de entrantes que se enlaza con un traslado saliente previo", t, cifras)


def _h16b(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t23_resultado_entrantes.csv"
    per = _periodo_agregado(f)
    d = _tabla(f, t, ["periodo", "entrante"])
    si, no = _fila(d, periodo=per, entrante="si"), _fila(d, periodo=per, entrante="no")
    cifras = (f"{per}: mortalidad de los entrantes = {_p(_cel(si, 'mortalidad'), 2)}; de los demás ingresos = {_p(_cel(no, 'mortalidad'), 2)}; "
              f"razón = {_d(_cel(si, 'rr'), 2)} (intervalo de {_d(_cel(si, 'rr_bajo'), 2)} a {_d(_cel(si, 'rr_alto'), 2)})")
    return _r("matiza", "se informa la diferencia de mortalidad de los entrantes con su intervalo; la decisión es humana", t, cifras)


# ============================================================================================================================ decisión 17
def _h17a(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t23_resumen_seleccion.csv"
    r = _fila(_tabla(f, t))
    n_alto, n_clave = int(_cel(r, "h17a_n_psi_alto")), int(_cel(r, "h17a_n_clave"))
    if n_clave <= 0:
        raise _SinDato("no hay variables clave para comparar")
    mayoria = 2 * n_alto > n_clave                                                     # más de la mitad: la mitad exacta no es mayoría
    h01a = previos.get("H01a")
    quiebre = h01a is not None and h01a.resultado == "matiza"
    cifras = (f"variables clave con PSI alto contra 2023 = {_ent(n_alto)} de {_ent(n_clave)}; "
              f"quiebre del identificador = {'sí' if quiebre else 'no'}; PSI alto desde {_d(f.umbral('psi_alto'), 2)}")
    if mayoria or quiebre:
        return _r("matiza", "hay deriva entre 2019 y los años siguientes o un quiebre del identificador: se propone la ventana de 2020 a 2022", t, cifras)
    return _r("respalda", "no hay deriva ni quiebre del identificador que justifique acortar la ventana de entrenamiento", t, cifras)


def _h17b(f: FuenteEvidencia, previos: Mapping[str, Resultado]) -> Resultado:
    t = "t23_resumen.csv, t23_serie_mensual.csv"
    try:
        r = _fila(_tabla(f, "t23_resumen.csv", ["intento_2024"]))
        alza, baja = int(_cel(r, "n_meses_alza")), int(_cel(r, "n_meses_baja"))
        evaluados = r.get("n_meses_evaluados")
    except _SinDato:
        s = _tabla(f, "t23_serie_mensual.csv", ["quiebre"])
        alza, baja, evaluados = int((s["quiebre"] == "alza").sum()), int((s["quiebre"] == "baja").sum()), None
    cifras = f"meses con quiebre al alza = {_ent(alza)}; a la baja = {_ent(baja)}" + (f" de {_ent(evaluados)} meses evaluados" if evaluados is not None else "")
    if alza + baja > 0:
        return _r("matiza", "la serie mensual tiene quiebres: se evalúa si justifican cambiar la ventana de entrenamiento", t, cifras)
    return _r("respalda", "la serie mensual no tiene quiebres que justifiquen cambiar la ventana de entrenamiento", t, cifras)


# ============================================================================================================================ registro de hipótesis
HIPOTESIS: tuple[Hipotesis, ...] = (
    Hipotesis("H01a", 1, "decisiva", False, _h01a, ("t24_resumen.csv",)),
    Hipotesis("H01b", 1, "decisiva", True, _h01b, ("t23_historial_factibilidad.csv",)),
    Hipotesis("H01c", 1, "decisiva", True, _h01c, ("matriz_decision_variables.csv",)),
    Hipotesis("H02a", 2, "decisiva", False, _h02a, ("t23_proxy_hospital.csv", "t23_fuga.csv")),
    Hipotesis("H02b", 2, "decisiva", False, _h02b, ("t23_proxy_hospital.csv",)),
    Hipotesis("H02c", 2, "decisiva", False, _h02c, ("t23_resumen_hospitales.csv",)),
    Hipotesis("H03", 3, "decisiva", False, _h03, ("t23_proxy_hospital.csv",)),
    Hipotesis("H04", 4, "informativa", False, _h04, ("t22_deriva_categorias.csv",)),
    Hipotesis("H05a", 5, "informativa", False, _h05a, ("t23_cascada_variantes.csv",)),
    Hipotesis("H05b", 5, "informativa", False, _h05b, ("t23_hospitales_resumen.csv",)),
    Hipotesis("H06a", 6, "informativa", True, _h06a, ("t23_volumen_umbrales.csv",)),
    Hipotesis("H06b", 6, "informativa", True, _h06b, ("t23_hospitales_resumen.csv",)),
    Hipotesis("H07", 7, "decisiva", False, _h07, ("t23_cola_estancia.csv",)),
    Hipotesis("H08", 8, "decisiva", False, _h08, ("t23_resumen_seleccion.csv",)),
    Hipotesis("H09", 9, "informativa", False, _h09, ("t23_estancia_dispersion.csv",)),
    Hipotesis("H10", 10, "informativa", False, _h10, ("figuras_indice.csv",)),
    Hipotesis("H11", 11, "decisiva", False, _h11, ("t24_rut_por_columna.csv",)),
    Hipotesis("H12", 12, "informativa", False, _h12, ("conocimiento_clinico.csv",)),
    Hipotesis("H13", 13, "decisiva", True, _h13, ("t23_resumen_hospitales.csv",)),
    Hipotesis("H14", 14, "decisiva", False, _h14, ("t23_fuga.csv",)),
    Hipotesis("H15a", 15, "informativa", False, _h15a, ("t23_resumen.csv",)),
    Hipotesis("H15b", 15, "decisiva", False, _h15b, ("t23_resultado_estratos.csv",)),
    Hipotesis("H16a", 16, "informativa", True, _h16a, ("t23_enlaces_traslado.csv",)),
    Hipotesis("H16b", 16, "informativa", False, _h16b, ("t23_resultado_entrantes.csv",)),
    Hipotesis("H17a", 17, "decisiva", False, _h17a, ("t23_resumen_seleccion.csv",)),
    Hipotesis("H17b", 17, "decisiva", False, _h17b, ("t23_resumen.csv", "t23_serie_mensual.csv")),
)
POR_ID: dict[str, Hipotesis] = {h.id: h for h in HIPOTESIS}


def validar_registro(hipotesis: pl.DataFrame, decisiones: pl.DataFrame) -> list[str]:
    """Problemas del registro frente a ``config/hipotesis.csv`` y a las 17 decisiones (lista vacía si concuerdan)."""
    problemas: list[str] = []
    h_cfg = {r["id"]: r for r in hipotesis.iter_rows(named=True) if r["id"].startswith("H")}
    if set(h_cfg) != set(POR_ID):
        problemas.append(f"las hipótesis del registro no son las de hipotesis.csv: faltan {sorted(set(h_cfg) - set(POR_ID))} y sobran {sorted(set(POR_ID) - set(h_cfg))}")
    nums = {int(n) for n in decisiones["n"].to_list()}
    for h in HIPOTESIS:
        cfg = h_cfg.get(h.id)
        if cfg is not None and str(h.decision) != cfg["decision"].strip():
            problemas.append(f"{h.id}: la decisión del registro ({h.decision}) no es la de hipotesis.csv ({cfg['decision']})")
        if h.decision not in nums:
            problemas.append(f"{h.id}: la decisión {h.decision} no existe en decisiones_pendientes.csv")
        if h.clase not in ("decisiva", "informativa"):
            problemas.append(f"{h.id}: clase inválida {h.clase!r}")
    return problemas


# ============================================================================================================================ evaluación
def evaluar_hipotesis(fuente: FuenteEvidencia, hip: Hipotesis, previos: Mapping[str, Resultado] | None = None) -> Resultado:
    """Evalúa una hipótesis. Una hipótesis que requiere los datos completos queda ``no_evaluable`` en un extracto sin leer su tabla; una tabla ausente o una celda nula
    también la dejan ``no_evaluable`` con su motivo."""
    tablas = ", ".join(hip.tablas)
    if hip.requiere_completo and fuente.etiqueta != "completo":
        return Resultado("no_evaluable", "no evaluable en un extracto", tablas, "", MOTIVO_REQUIERE_COMPLETO)
    try:
        return hip.evaluador(fuente, previos or {})
    except _SinDato as e:
        return Resultado("no_evaluable", f"no evaluable: {e}", tablas, "", str(e))
    except (pl.exceptions.ColumnNotFoundError, KeyError) as e:
        motivo = f"la tabla no trae una columna esperada: {e}"
        return Resultado("no_evaluable", f"no evaluable: {motivo}", tablas, "", motivo)


def evaluar_todas(fuente: FuenteEvidencia) -> dict[str, Resultado]:
    """Las 26 hipótesis en el orden del registro (cada una ve los resultados de las anteriores)."""
    out: dict[str, Resultado] = {}
    for h in HIPOTESIS:
        out[h.id] = evaluar_hipotesis(fuente, h, out)
    return out


ESQUEMA_HIPOTESIS = {
    "id_hipotesis": pl.String, "decision": pl.Int64, "clase": pl.String, "requiere_completo": pl.String, "resultado": pl.String, "veredicto": pl.String, "tabla": pl.String,
    "cifras_clave": pl.String, "motivo": pl.String,
}


def construir_hipotesis(resultados: Mapping[str, Resultado]) -> pl.DataFrame:
    """``t25_hipotesis_evaluadas.csv``: una fila por hipótesis."""
    filas = [
        {"id_hipotesis": h.id, "decision": h.decision, "clase": h.clase, "requiere_completo": "si" if h.requiere_completo else "no", "resultado": resultados[h.id].resultado,
         "veredicto": resultados[h.id].veredicto, "tabla": resultados[h.id].tabla, "cifras_clave": resultados[h.id].cifras_clave, "motivo": resultados[h.id].motivo}
        for h in HIPOTESIS
    ]
    return pl.DataFrame(filas, schema=ESQUEMA_HIPOTESIS)


# ============================================================================================================================ cascada (S10 y DT05)
IDENTIDADES_CASCADA: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("A", "TOTAL", ("EX01", "EX02", "EX03")),
    ("B", "A", ("EX04",)),
    ("C", "B", ("EX05",)),
    ("D", "C", ("EX06",)),
    ("E", "D", ("EX07", "EX08")),
)
"""``(resultado, base, restas)``: el paso ``resultado`` es la ``base`` menos la suma de las ``restas`` en cada período de la cascada secuencial."""


def revisar_cascada(df: pl.DataFrame) -> tuple[list[str], int]:
    """Revisa las identidades de la cascada secuencial en cada período. Devuelve los problemas y cuántas identidades se pudieron verificar.

    Una identidad con algún paso nulo (suprimido) o ausente no se verifica y no cuenta como problema. ``df`` trae ``periodo``, ``paso`` y ``n``.
    """
    problemas: list[str] = []
    verificadas = 0
    por_periodo: dict[str, dict[str, Any]] = {}
    for r in df.select(["periodo", "paso", "n"]).iter_rows():
        por_periodo.setdefault(str(r[0]), {})[r[1]] = r[2]
    for periodo in sorted(por_periodo):
        pasos = por_periodo[periodo]
        for resultado, base, restas in IDENTIDADES_CASCADA:
            necesarios = [resultado, base, *restas]
            if any(pasos.get(p) is None for p in necesarios):
                continue
            verificadas += 1
            esperado = int(pasos[base]) - sum(int(pasos[p]) for p in restas)
            if int(pasos[resultado]) != esperado:
                problemas.append(f"{periodo}: {resultado} es {_ent(pasos[resultado])} y {base} menos {' y '.join(restas)} da {_ent(esperado)}")
    return problemas, verificadas


def problemas_cascada(df: pl.DataFrame) -> list[str]:
    """Identidades de la cascada que no cuadran (lista vacía si cuadra). La usan el supuesto S10 y el verificador DT05."""
    return revisar_cascada(df)[0]


# ============================================================================================================================ semáforo y decisiones
def tipo_evidencia(n: int, hipotesis: pl.DataFrame) -> str:
    """``cuantitativa`` si alguna hipótesis primaria depende de datos; ``parcial`` si no y alguna depende en parte; si todas son ``no``, ``institucional`` para las decisiones
    10 y 12 y ``no depende de datos`` para la 4."""
    deps = [r["depende_de_datos"] for r in hipotesis.iter_rows(named=True) if r["decision"].strip() == str(n)]
    if "si" in deps:
        return "cuantitativa"
    if "parcial" in deps:
        return "parcial"
    return "institucional" if n in INSTITUCIONALES else "no depende de datos"


def agregar_semaforo(pares: Sequence[tuple[Hipotesis, Resultado]], tipo: str) -> str:
    """Semáforo de una decisión a partir de sus hipótesis primarias y su tipo de evidencia (las siete reglas del módulo, en orden)."""
    if tipo in ("institucional", "no depende de datos"):
        return "gris"
    if any(h.requiere_completo and r.resultado == "no_evaluable" for h, r in pares):
        return "gris"
    decisivas = [r for h, r in pares if h.clase == "decisiva"]
    decisivas_ev = [r for r in decisivas if r.resultado != "no_evaluable"]
    if any(r.resultado == "contradice" for r in decisivas_ev):
        return "rojo"
    if decisivas_ev and len(decisivas_ev) == len(decisivas) and all(r.resultado == "respalda" for r in decisivas_ev):
        return "verde"
    if decisivas_ev:
        return "amarillo"
    if any(h.clase == "informativa" and r.resultado != "no_evaluable" for h, r in pares):
        return "amarillo"
    return "gris"


CORRECCIONES_TEXTO_FASE1: tuple[tuple[str, str], ...] = (
    (r"(?<=\d);(?=\d)", ","),
    (r"\bultimos\b", "últimos"), (r"\bdias\b", "días"), (r"\bmedica\b", "médica"), (r"\bAnalisis\b", "Análisis"), (r"\banalisis\b", "análisis"),
    (r"\bprevision\b", "previsión"), (r"\bsocioeconomico\b", "socioeconómico"), (r"\bPoblacion\b", "Población"), (r"\banos\b", "años"), (r"\bmas\b", "más"),
    (r"\bminimo\b", "mínimo"), (r"\bpequenas\b", "pequeñas"), (r"\bEstadias\b", "Estadías"), (r"\bEstadia\b", "Estadía"), (r"\bestadia\b", "estadía"),
    (r"\bcategorias\b", "categorías"), (r"\bacademico\b", "académico"), (r"\bcomo informar\b", "cómo informar"), (r"\bRevision\b", "Revisión"), (r"\bclinica\b", "clínica"),
    (r"\bquien revisa\b", "quién revisa"), (r"\bque traslado$", "que trasladó"), (r"^Que traslados\b", "Qué traslados"),
)
"""Reparaciones del texto de ``decisiones_pendientes.csv`` (``docs/``, solo lectura): el archivo viene sin tildes y escribe el decimal con punto y coma ("0;95"), porque el
separador de columnas es la coma. La tabla es explícita y solo toca las palabras que ese archivo trae."""


def normalizar_texto_fase1(texto: str) -> str:
    """El texto de una decisión de la Fase 1 con tildes y con el decimal escrito con coma (la fuente trae ``0;95``)."""
    for patron, reemplazo in CORRECCIONES_TEXTO_FASE1:
        texto = re.sub(patron, reemplazo, texto)
    return texto


def _afectadas_de(texto: str) -> set[int]:
    return {int(p.strip()) for p in (texto or "").replace(",", ";").split(";") if p.strip().isdigit()}


def _sin_repetir(items: Sequence[str]) -> list[str]:
    visto: set[str] = set()
    return [x for x in items if x and not (x in visto or visto.add(x))]


ESQUEMA_DECISIONES = {
    "n": pl.Int64, "tema": pl.String, "propuesta_fase1": pl.String, "hipotesis": pl.String, "tablas_evidencia": pl.String, "evidencia": pl.String, "veredicto": pl.String,
    "tipo_evidencia": pl.String, "semaforo": pl.String, "que_falta_con_datos_completos": pl.String, "estado": pl.String,
}
COLUMNAS_TEXTO_DECISIONES = [c for c, t in ESQUEMA_DECISIONES.items() if t == pl.String]


def _texto_evidencia(n: int, resultados: Mapping[str, Resultado], hipotesis: pl.DataFrame, supuestos: pl.DataFrame | None) -> str:
    partes: list[str] = []
    for h in HIPOTESIS:
        r = resultados[h.id]
        if h.decision == n and r.cifras_clave:
            partes.append(f"{h.id}: {r.cifras_clave}")
    sup = {s["id"]: s for s in supuestos.iter_rows(named=True)} if supuestos is not None else {}
    for fila in hipotesis.iter_rows(named=True):
        if fila["decision"].strip() == str(n) or n not in _afectadas_de(fila.get("decision_afectada", "")):
            continue
        hid = fila["id"]
        if hid in POR_ID:
            r = resultados[hid]
            if r.cifras_clave and r.resultado != "no_evaluable":
                partes.append(f"{hid} (también afecta a esta decisión): {r.cifras_clave}")
        elif hid in sup:
            s = sup[hid]
            partes.append(f"{hid} {s['estado']}" + (f": {s['cifra']}" if s["cifra"] else ""))
    return " | ".join(partes) if partes else "sin cifras: la decisión no depende de datos"


def _texto_veredicto(n: int, resultados: Mapping[str, Resultado]) -> str:
    prims = [h for h in HIPOTESIS if h.decision == n]
    cuenta = {k: sum(resultados[h.id].resultado == k for h in prims) for k in RESULTADOS}
    cab = f"{cuenta['respalda']} respalda, {cuenta['matiza']} matiza, {cuenta['contradice']} contradice, {cuenta['no_evaluable']} no evaluable"
    frases = [f"{h.id}{'' if h.clase == 'decisiva' else ' (solo informa)'}: {resultados[h.id].veredicto}" for h in prims]
    return cab + (" | " + " | ".join(frases) if frases else "")


def _texto_falta(n: int, resultados: Mapping[str, Resultado], etiqueta: str) -> str:
    frases = [FALTA_CON_DATOS_COMPLETOS.get(h.id, f"la evidencia de {h.id} no se pudo leer en esta corrida")
              for h in HIPOTESIS if h.decision == n and resultados[h.id].resultado == "no_evaluable"]
    if etiqueta != "completo":
        frases.append(CIERRE_EXTRACTO)
    frases = _sin_repetir(frases)
    return "; ".join(frases) if frases else SIN_FALTANTE


def construir_decisiones(
    etiqueta: str, hipotesis: pl.DataFrame, decisiones: pl.DataFrame, resultados: Mapping[str, Resultado], supuestos: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """``decisiones_evidencia.csv``: una fila por decisión de ``decisiones_pendientes.csv`` con su evidencia leída de los CSV, su veredicto y su semáforo.

    Todo el texto de evidencia sale de las celdas de las tablas (``Resultado.cifras_clave``); el estado es siempre "pendiente de decisión humana": la evidencia recomienda y no decide.
    """
    cobertura = {int(r["decision"]): r["hipotesis"] for r in protocolo.tabla_cobertura(hipotesis, decisiones).iter_rows(named=True)}
    filas = []
    for d in sorted(decisiones.iter_rows(named=True), key=lambda r: int(r["n"])):
        n = int(d["n"])
        prims = [h for h in HIPOTESIS if h.decision == n]
        tipo = tipo_evidencia(n, hipotesis)
        pares = [(h, resultados[h.id]) for h in prims]
        filas.append(
            {
                "n": n, "tema": normalizar_texto_fase1(d["tema"]), "propuesta_fase1": normalizar_texto_fase1(d["propuesta"]), "hipotesis": cobertura.get(n, ""),
                "tablas_evidencia": "; ".join(_sin_repetir([t.strip() for h in prims for t in resultados[h.id].tabla.split(",")])),
                "evidencia": _texto_evidencia(n, resultados, hipotesis, supuestos), "veredicto": _texto_veredicto(n, resultados), "tipo_evidencia": tipo,
                "semaforo": agregar_semaforo(pares, tipo), "que_falta_con_datos_completos": _texto_falta(n, resultados, etiqueta), "estado": ESTADO_DECISION,
            }
        )
    return pl.DataFrame(filas, schema=ESQUEMA_DECISIONES)


# ============================================================================================================================ supuestos S01 a S15
@dataclass(frozen=True)
class ResultadoSupuesto:
    estado: str
    cifra: str
    tabla: str
    motivo: str = ""


ESTADOS_SUPUESTO = ("ACEPTADO", "RECHAZADO", "NO EVALUABLE")
UMBRAL_S01 = 0.01
"""El texto del criterio de S01 dice "menores a 1 %" (una prueba lo ata a ``hipotesis.csv``)."""
ANIO_S07 = 2019
"""El texto de S07 dice "cambia de nombre solo en 2019"."""
REGLA_S03 = "regla aplicada: se acepta solo con el 100 % exacto de códigos válidos, sin tolerancia"
REGLA_S12 = "regla aplicada: una columna es relleno constante solo si su valor más frecuente es el 100 %; cualquier proporción menor cuenta como información"
REGLA_S13 = "regla aplicada: se acepta si el diccionario sigue el conteo sin centinelas en más de la mitad de las columnas con centinelas (mayoría simple)"
"""Las reglas operativas de los supuestos cuyo criterio de ``hipotesis.csv`` no trae una regla numérica o la trae sin tolerancia: se declaran en el motivo de cada fila."""
COLUMNAS_S05 = ("ESPECIALIDAD_MEDICA", "PREVISION", "TIPO_PROCEDENCIA", "TIPOALTA", "DIAGNOSTICO1")
COLUMNAS_S12 = ("RN2ESTADO", "RN3ESTADO", "RN4ESTADO")
METRICAS_S14 = ("calidad.alta_antes_ingreso.total", "calidad.edad_negativa.total", "calidad.edad_mayor_110.total")


def _metrica_valor(f: FuenteEvidencia, clave: str) -> float | None:
    m = f.metrica(clave)
    return None if m is None or m.get("valor") is None else float(m["valor"])


def _ok(cifra: str, tabla: str) -> ResultadoSupuesto:
    return ResultadoSupuesto("ACEPTADO", cifra, tabla)


def _no(cifra: str, tabla: str) -> ResultadoSupuesto:
    return ResultadoSupuesto("RECHAZADO", cifra, tabla)


def _base_s01(f: FuenteEvidencia) -> str:
    """Base de las dos proporciones de S01: los IDs con dos o más episodios (en el sexo, los que además lo tienen conocido), sumados en todos los regímenes. Vacío si la tabla
    no está o no trae los denominadores."""
    d = f.tabla("t24_id_consistencia.csv", ["regimen", "grupo_id"])
    cols = ("n_ids_con_dos_o_mas_episodios", "n_ids_con_sexo_en_dos_o_mas_episodios")
    if d is None or "grupo_id" not in d.columns or any(c not in d.columns for c in cols):
        return ""
    d = d.filter(pl.col("grupo_id") == "todos los IDs")                                 # el mismo grupo que usa la métrica de la Tarea 5
    n_nac, n_sexo = (int(d[c].drop_nulls().sum()) for c in cols)
    return f"; base: IDs con dos o más episodios = {_ent(n_nac)} (nacimiento) y {_ent(n_sexo)} (sexo conocido en dos o más episodios)"


def _s01(f: FuenteEvidencia, cascada: pl.DataFrame | None) -> ResultadoSupuesto:
    t = "t24_id_consistencia.csv"
    vn = _metrica_valor(f, "id.consistencia.nacimiento_inconsistente_prop")
    vs = _metrica_valor(f, "id.consistencia.sexo_inconsistente_prop")
    if vn is None and vs is None:
        raise _SinDato("faltan las proporciones de inconsistencia del identificador")
    texto = lambda v: "sin dato" if v is None else _p(v, 2)  # noqa: E731
    cifra = f"IDs con fecha de nacimiento inconsistente = {texto(vn)}; con sexo inconsistente = {texto(vs)}; máximo aceptado = {_p(UMBRAL_S01, 0)}{_base_s01(f)}"
    if any(v is not None and v >= UMBRAL_S01 for v in (vn, vs)):
        return _no(cifra, t)
    if vn is None or vs is None:
        raise _SinDato("falta una de las dos proporciones de inconsistencia")
    return _ok(cifra, t)


def _s02(f: FuenteEvidencia, cascada: pl.DataFrame | None) -> ResultadoSupuesto:
    t = "t23_elix_vs_3c.csv"
    d = f.tabla(t, ["elix_id", "supera_umbral"])
    if d is not None and "supera_umbral" in d.columns:
        vals = [str(x).strip().lower() for x in d["supera_umbral"].to_list() if x is not None]
        n_si = vals.count("si")
        if n_si + vals.count("no") == 0:
            raise _SinDato("ninguna categoría de Elixhauser tiene prevalencia evaluable con los dos mapeos")
        cifra = f"categorías cuya prevalencia con tres caracteres difiere más del umbral = {_ent(n_si)} de {_ent(len(vals))}"
    else:
        v = _metrica_valor(f, "elix.n_categorias_sobre_umbral_3c")                    # sin la tabla, la métrica que registró la Tarea 8
        if v is None:
            raise _SinDato("falta la tabla t23_elix_vs_3c.csv y la métrica elix.n_categorias_sobre_umbral_3c")
        n_si = int(v)
        cifra = f"categorías cuya prevalencia con tres caracteres difiere más del umbral = {_ent(n_si)}"
    return _no(cifra, t) if n_si >= 1 else _ok(cifra, t)


def _s03(f: FuenteEvidencia, cascada: pl.DataFrame | None) -> ResultadoSupuesto:
    t = "t24_validez_codigos.csv"
    v = _metrica_valor(f, "calidad.cie10_principal_valido_pct")
    if v is None:
        raise _SinDato("falta la métrica calidad.cie10_principal_valido_pct")
    cifra = f"diagnósticos principales normalizados que son válidos en CIE-10 = {fmt_dec(v, 4)} %"
    if v == 100.0:
        return ResultadoSupuesto("ACEPTADO", cifra, t, REGLA_S03)
    return ResultadoSupuesto("RECHAZADO", cifra, t, f"{REGLA_S03}; la métrica cubre el diagnóstico principal")


def _s04(f: FuenteEvidencia, cascada: pl.DataFrame | None) -> ResultadoSupuesto:
    t = "t24_tipoalta_etiquetas.csv"
    d = _tabla(f, t, ["etiqueta", "metodo"])
    minimo = int(f.umbral("celda_minima"))
    con_casos = d.filter(pl.col("n") >= minimo)                                          # la regla solo mira las etiquetas con casos suficientes
    sin_rol = con_casos.filter(pl.col("metodo") == "sin_rol")
    fuera = d.height - con_casos.height                                                  # etiquetas con menos casos (suprimidas) o que no aparecen en los datos
    cifra = (f"etiquetas de TIPOALTA con {_ent(minimo)} casos o más y sin rol asignado = {_ent(sin_rol.height)} de {_ent(con_casos.height)} con esos casos; "
             f"no cuentan {_ent(fuera)} etiquetas con menos casos, suprimidas o ausentes de los datos")
    return _no(cifra, t) if sin_rol.height else _ok(cifra, t)


def _s05(f: FuenteEvidencia, cascada: pl.DataFrame | None) -> ResultadoSupuesto:
    t = "t22_cardinalidades.csv"
    d = _tabla(f, t, ["columna", "periodo"])
    completo = f.etiqueta == "completo"
    estados, partes = [], []
    for col in COLUMNAS_S05:
        agregados = d.filter((pl.col("columna") == col) & pl.col("periodo").str.contains("-"))
        if agregados.height == 0:
            estados.append("NO EVALUABLE")
            partes.append(f"{col}: sin período agregado")
            continue
        r = agregados.sort("periodo").row(-1, named=True)
        obs, doc, chao = r.get("n_observado"), r.get("documentado"), r.get("chao1")
        if obs is None or doc is None:
            estados.append("NO EVALUABLE")
            partes.append(f"{col}: sin dato")
            continue
        partes.append(f"{col}: observados {_ent(obs)}, documentados {_ent(doc)}" + ("" if completo or chao is None else f", Chao1 {_ent(chao)}"))
        if completo:
            estados.append("ACEPTADO" if obs == doc else "RECHAZADO")
        elif obs > doc:
            estados.append("RECHAZADO")
        elif chao is not None and doc <= chao:
            estados.append("ACEPTADO")
        else:
            estados.append("NO EVALUABLE")
    cifra = "; ".join(partes)
    if "RECHAZADO" in estados:
        return _no(cifra, t)
    if "NO EVALUABLE" in estados:
        return ResultadoSupuesto("NO EVALUABLE", cifra, t, "el extracto no permite confirmar la cardinalidad documentada: el estimador Chao1 queda por debajo de ella")
    return _ok(cifra, t)


def _s06(f: FuenteEvidencia, cascada: pl.DataFrame | None) -> ResultadoSupuesto:
    t = "t21_deriva_formato.csv"
    d = _tabla(f, t, ["columna", "clase"])
    presentes: dict[tuple[Any, str], set[str]] = {}
    for r in d.filter(pl.col("clase").is_in(["ISO", "DD-MM-AAAA"])).iter_rows(named=True):
        n = r.get("n_clase")
        if n is None or n > 0:                                                      # un conteo suprimido (de 1 a 9) también cuenta como presente
            presentes.setdefault((r["anio"], r["columna"]), set()).add(r["clase"])
    mezclan = sorted(k for k, v in presentes.items() if len(v) > 1)
    cifra = f"combinaciones de año y columna de fecha que mezclan formatos = {_ent(len(mezclan))} de {_ent(len(presentes))} con fechas"
    return _no(cifra, t) if mezclan else _ok(cifra, t)


def _s07(f: FuenteEvidencia, cascada: pl.DataFrame | None) -> ResultadoSupuesto:
    t = "t21_esquema_diferencias.csv"
    n = _metrica_valor(f, "estructura.n_anios_id_beneficiario")
    primero = _metrica_valor(f, "estructura.primer_anio_id_beneficiario")
    if n is None:
        raise _SinDato("falta la métrica estructura.n_anios_id_beneficiario")
    cifra = f"años con la columna del identificador llamada ID_BENEFICIARIO = {_ent(n)}" + (f" (el primero es {int(primero)})" if primero is not None else "")
    if int(n) == 1 and primero is not None and int(primero) == ANIO_S07:
        return _ok(cifra, t)
    return _no(cifra, t)


def _s08(f: FuenteEvidencia, cascada: pl.DataFrame | None) -> ResultadoSupuesto:
    t = "t24_resumen_catalogos.csv"
    maximo = _metrica_valor(f, "hospitales.n_distintos.maximo")
    c77, c78 = f.cifra_documentada("C077"), f.cifra_documentada("C078")
    if maximo is None or c77 is None or c78 is None:
        raise _SinDato("falta el máximo de hospitales distintos o las cifras documentadas C077 y C078")
    fuera = int(_cel(_fila(_tabla(f, t)), "n_codigos_hospital_fuera_catalogo"))
    lo, hi = min(c77, c78), max(c77, c78)
    cifra = (f"máximo de hospitales distintos en un año = {_ent(maximo)}; documentados = {_ent(lo)} y {_ent(hi)}; "
             f"códigos de hospital fuera del catálogo = {_ent(fuera)}")
    if fuera > 0 or maximo > hi:
        return _no(cifra, t)
    if maximo >= lo:
        return _ok(cifra, t)
    if f.etiqueta == "completo":
        return _no(cifra, t)
    return ResultadoSupuesto("NO EVALUABLE", cifra, t, "el extracto puede no contener todos los hospitales")


MOTIVO_S09_ACEPTADO = ("coincide con el servicio de alta en la mayoría de los egresos sin traslado interno, lo esperable por construcción y que no distingue el ingreso de la estadía, "
                       "y en la minoría de los que tienen traslado interno: la variable registra el servicio de ingreso; aceptarlo no levanta las sospechas de proxy del hospital ni de "
                       "fuga que la Tarea 11 mantiene sobre SERVICIOINGRESO")
MOTIVO_S09_RECHAZADO = "coincide con el servicio de alta en la mayoría de los egresos, también en los que tienen traslado interno: la variable refleja el servicio de la estadía"
MOTIVO_S09_GRUPO_PEQUENO = ("algún grupo (egresos sin traslado interno o con traslado interno) tiene menos egresos que el mínimo publicable, está suprimido o falta en la tabla: sin los "
                            "dos grupos no se puede contrastar el servicio de ingreso con el de alta")
MOTIVO_S09_INCONSISTENTE = ("el servicio de ingreso no coincide con el de alta en la mayoría de los egresos sin traslado interno, donde la igualdad es esperable por construcción: "
                            "la medición es inconsistente y no permite concluir")


@dataclass(frozen=True)
class _GrupoS09:
    """Un grupo de la concordancia de S09: egresos con los dos servicios conocidos, los que coinciden y la cifra tal como se cita."""

    n: int
    k: int
    texto: str


def _grupo_s09(d: pl.DataFrame, tipo_fila: str, col_n: str, col_k: str, nombre: str) -> _GrupoS09 | None:
    """El grupo de una fila de concordancia de ``t23_fuga.csv``, o ``None`` si la fila falta o sus cifras están suprimidas (menos de diez casos)."""
    try:
        fila = _fila(d, tipo_fila=tipo_fila)
        n, k = int(_cel(fila, col_n)), int(_cel(fila, col_k))
        texto = f"{_p(_cel(fila, 'prop_igualdad'))} de los egresos {nombre} (intervalo de {_p(_cel(fila, 'ic_bajo'))} a {_p(_cel(fila, 'ic_alto'))})"
    except _SinDato:
        return None
    return _GrupoS09(n, k, texto)


def _s09(f: FuenteEvidencia, cascada: pl.DataFrame | None) -> ResultadoSupuesto:
    """S09 contrasta los egresos sin traslado interno con los que tienen traslado interno (cambio 1.5 del protocolo). Sin traslado la coincidencia entre SERVICIOINGRESO y
    SERVICIOALTA es esperable por construcción y no discrimina; la prueba informativa es el grupo con traslado. Mayoría es estrictamente más de la mitad, calculada con los
    conteos, y cada grupo debe tener al menos ``celda_minima`` egresos. ACEPTADO si coincide en la mayoría sin traslado y en la minoría con traslado (la variable registra el
    ingreso); RECHAZADO si coincide en la mayoría también con traslado (refleja la estadía); NO EVALUABLE si algún grupo es menor al mínimo, está suprimido o falta, o si no
    coincide en la mayoría sin traslado (inconsistente). Aceptarlo no levanta las sospechas de proxy ni de fuga de la Tarea 11."""
    t = "t23_fuga.csv"
    d = _tabla(f, t, ["tipo_fila", "variable", "coincide_mayoria"])
    minimo = int(f.umbral("celda_minima"))
    sin = _grupo_s09(d, "concordancia", "n_sin_traslado", "n_iguales_sin_traslado", "sin traslado interno")
    con = _grupo_s09(d, "concordancia_con_traslado", "n_con_traslado", "n_iguales_con_traslado", "con traslado interno")
    citados = [g.texto for g in (sin, con) if g is not None]
    cifra = "el servicio de ingreso coincide con el de alta en " + " y en ".join(citados) if citados else ""
    if sin is None or con is None or min(sin.n, con.n) < minimo:
        return ResultadoSupuesto("NO EVALUABLE", cifra, t, MOTIVO_S09_GRUPO_PEQUENO)
    if 2 * sin.k <= sin.n:                                                                 # la mitad exacta no es mayoría
        return ResultadoSupuesto("NO EVALUABLE", cifra, t, MOTIVO_S09_INCONSISTENTE)
    if 2 * con.k > con.n:
        return ResultadoSupuesto("RECHAZADO", cifra, t, MOTIVO_S09_RECHAZADO)
    return ResultadoSupuesto("ACEPTADO", cifra, t, MOTIVO_S09_ACEPTADO)


def _s10(f: FuenteEvidencia, cascada: pl.DataFrame | None) -> ResultadoSupuesto:
    t = "t23_cascada.csv"
    df = cascada if cascada is not None else _tabla(f, t, ["periodo", "paso"])
    problemas, verificadas = revisar_cascada(df)
    if verificadas == 0:
        raise _SinDato("ninguna identidad de la cascada se puede verificar con las celdas publicadas")
    cifra = f"identidades de la cascada verificadas = {_ent(verificadas)}; que no cuadran = {_ent(len(problemas))}"
    if problemas:
        return ResultadoSupuesto("RECHAZADO", cifra, t, "; ".join(problemas[:3]))
    return _ok(cifra, t)


def _s11(f: FuenteEvidencia, cascada: pl.DataFrame | None) -> ResultadoSupuesto:
    t = "t23_deriva_codificacion.csv"
    m = f.metrica("codificacion.aumento_pct_2019_2024")
    if m is None or m.get("valor") is None or m.get("ic_bajo") is None or m.get("ic_alto") is None:
        raise _SinDato("falta la métrica codificacion.aumento_pct_2019_2024 con su intervalo")
    lo, hi = float(m["ic_bajo"]), float(m["ic_alto"])
    cifra = f"aumento relativo del promedio de diagnósticos secundarios entre el primer y el último año = {fmt_dec(m['valor'], 1)} % (intervalo de {fmt_dec(lo, 1)} % a {fmt_dec(hi, 1)} %)"
    if lo > 0:
        return _ok(cifra, t)
    if hi < 0:
        return _no(cifra, t)
    return ResultadoSupuesto("NO EVALUABLE", cifra, t, "el intervalo incluye el cero")


def _s12(f: FuenteEvidencia, cascada: pl.DataFrame | None) -> ResultadoSupuesto:
    t = "t22_bloque_rn.csv"
    d = _tabla(f, t, ["columna"])
    informadas, partes = [], []
    for col in COLUMNAS_S12:
        r = _fila(d, columna=col)
        n = r.get("n_informados")
        if n is not None and n <= 0:
            partes.append(f"{col}: sin valores")
            continue
        informadas.append((col, r.get("prop_modal")))                                # un conteo suprimido es una columna con valores, de proporción desconocida
        if n is None:
            partes.append(f"{col}: pocos valores (suprimido)")
        else:
            partes.append(f"{col}: {_ent(n)} valores, el más frecuente es el {_p(r['prop_modal'], 2)}" if r.get("prop_modal") is not None else f"{col}: {_ent(n)} valores")
    cifra = "; ".join(partes)
    if not informadas:
        raise _SinDato("ninguna de las tres columnas trae valores")
    if any(p is not None and p < 1.0 for _, p in informadas):
        return ResultadoSupuesto("RECHAZADO", cifra, t, f"{REGLA_S12}; alguna columna trae información y no solo relleno constante")
    if any(p is None for _, p in informadas):
        raise _SinDato("la proporción del valor más frecuente está suprimida")
    return ResultadoSupuesto("ACEPTADO", cifra, t, REGLA_S12)


def _s13(f: FuenteEvidencia, cascada: pl.DataFrame | None) -> ResultadoSupuesto:
    t = "t22_conciliacion.csv"
    d = _tabla(f, t, ["columna", "estado"])
    n_ev = n_sigue = 0
    for r in d.iter_rows(named=True):
        if r.get("estado") not in ("COINCIDE", "DISCREPANCIA") or not r.get("n_centinela_med") or r["n_centinela_med"] <= 0:
            continue
        doc, med, n_fil = r.get("pct_sin_doc"), r.get("pct_sin_med"), r.get("n_filas_med")
        if doc is None or med is None or not n_fil or r.get("n_sin_med") is None:
            continue
        con_centinela = 100.0 * (r["n_sin_med"] + r["n_centinela_med"]) / n_fil
        n_ev += 1
        n_sigue += abs(doc - med) <= abs(doc - con_centinela)
    if n_ev == 0:
        raise _SinDato("ninguna columna con centinelas tiene conciliación evaluable")
    cifra = f"columnas con centinelas donde el diccionario sigue el conteo sin centinelas = {_ent(n_sigue)} de {_ent(n_ev)}"
    return ResultadoSupuesto("ACEPTADO" if n_sigue * 2 > n_ev else "RECHAZADO", cifra, t, REGLA_S13)


def _s14(f: FuenteEvidencia, cascada: pl.DataFrame | None) -> ResultadoSupuesto:
    t = "t24_reglas.csv"
    valores = [_metrica_valor(f, k) for k in METRICAS_S14]
    cifra = (f"fechas y edades imposibles en total (alta antes del ingreso, edad negativa y edad sobre el máximo) = {_ent(sum(valores))}" if all(v is not None for v in valores)
             else "fechas y edades imposibles: alguna cifra está suprimida por tener menos de diez casos")
    return ResultadoSupuesto("NO EVALUABLE", cifra, t, "el criterio dice pocas y no trae una regla numérica")


def _s15(f: FuenteEvidencia, cascada: pl.DataFrame | None) -> ResultadoSupuesto:
    t = "t24_duplicados.csv"
    d = _tabla(f, t)
    columnas = [c for c in ("n_filas_duplicadas_clave", "n_solapes", "n_duplicados_exactos_por_id_y_fechas") if c in d.columns]
    if not columnas:
        raise _SinDato("la tabla no trae conteos de duplicados ni de solapes")
    celdas = [v for c in columnas for v in d[c].to_list()]
    suprimidas = sum(v is None for v in celdas)
    total = sum(v for v in celdas if v is not None)
    cifra = (f"suma de las celdas visibles de {_ent(len(columnas))} medidas de duplicados y solapes, que pueden contar los mismos episodios más de una vez, = {_ent(total)}; "
             f"celdas suprimidas por tener menos de diez casos = {_ent(suprimidas)}")
    if total == 0 and suprimidas == 0:
        return _ok(cifra, t)
    return ResultadoSupuesto("NO EVALUABLE", cifra, t, "relevantes no tiene una regla numérica: se informa el conteo")


SUPUESTOS: dict[str, Callable[[FuenteEvidencia, pl.DataFrame | None], ResultadoSupuesto]] = {
    "S01": _s01, "S02": _s02, "S03": _s03, "S04": _s04, "S05": _s05, "S06": _s06, "S07": _s07, "S08": _s08, "S09": _s09, "S10": _s10, "S11": _s11, "S12": _s12, "S13": _s13,
    "S14": _s14, "S15": _s15,
}


def evaluar_supuesto(fuente: FuenteEvidencia, id_supuesto: str, cascada: pl.DataFrame | None = None) -> ResultadoSupuesto:
    """Evalúa un supuesto; una tabla ausente o una celda nula lo dejan NO EVALUABLE con su motivo."""
    try:
        return SUPUESTOS[id_supuesto](fuente, cascada)
    except _SinDato as e:
        return ResultadoSupuesto("NO EVALUABLE", "", "", str(e))
    except (pl.exceptions.ColumnNotFoundError, KeyError) as e:
        return ResultadoSupuesto("NO EVALUABLE", "", "", f"la tabla no trae una columna esperada: {e}")


ESQUEMA_SUPUESTOS = {"id": pl.String, "supuesto": pl.String, "criterio": pl.String, "estado": pl.String, "cifra": pl.String, "tabla": pl.String, "nivel": pl.String, "motivo": pl.String}


def construir_supuestos(fuente: FuenteEvidencia, hipotesis: pl.DataFrame, cascada: pl.DataFrame | None = None) -> pl.DataFrame:
    """``supuestos_verificados.csv``: S01 a S15 con su estado, la cifra que lo respalda y la tabla de donde sale."""
    cfg = {r["id"]: r for r in hipotesis.iter_rows(named=True) if r["id"].startswith("S")}
    filas = []
    for sid in SUPUESTOS:
        r = evaluar_supuesto(fuente, sid, cascada)
        filas.append({"id": sid, "supuesto": cfg[sid]["pregunta"], "criterio": cfg[sid]["criterio_decision"], "estado": r.estado, "cifra": r.cifra,
                      "tabla": r.tabla or cfg[sid]["tablas_clave"], "nivel": fuente.etiqueta, "motivo": r.motivo})
    return pl.DataFrame(filas, schema=ESQUEMA_SUPUESTOS)


# ============================================================================================================================ resumen y métricas
def resumen_decisiones(decisiones: pl.DataFrame, supuestos: pl.DataFrame, hipotesis: pl.DataFrame | None = None) -> dict[str, int]:
    """Conteos del semáforo, de los supuestos y de las hipótesis para el resumen de evidencia."""
    out = {"decisiones_n_total": decisiones.height}
    for s in SEMAFOROS:
        out[f"decisiones_n_{s}"] = int((decisiones["semaforo"] == s).sum())
    out["decisiones_n_supuestos"] = supuestos.height
    out["decisiones_n_supuestos_aceptados"] = int((supuestos["estado"] == "ACEPTADO").sum())
    out["decisiones_n_supuestos_rechazados"] = int((supuestos["estado"] == "RECHAZADO").sum())
    out["decisiones_n_supuestos_no_evaluables"] = int((supuestos["estado"] == "NO EVALUABLE").sum())
    if hipotesis is not None:
        out["decisiones_n_hipotesis"] = hipotesis.height
        for r in RESULTADOS:
            out[f"decisiones_n_hipotesis_{r}"] = int((hipotesis["resultado"] == r).sum())
    return out


def registrar_metricas(ctx: Contexto, decisiones: pl.DataFrame, supuestos: pl.DataFrame) -> None:
    """Métricas ``decisiones.*`` del cierre (decisiones y supuestos, no casos)."""
    r = resumen_decisiones(decisiones, supuestos)
    nota = "decisiones o supuestos, no casos"
    for s in SEMAFOROS:
        ctx.registrar_metrica(f"decisiones.n_{s}", r[f"decisiones_n_{s}"], tarea=TAREA, tabla="decisiones_evidencia.csv", nota=nota)
    for clave, col in (("n_supuestos_aceptados", "decisiones_n_supuestos_aceptados"), ("n_supuestos_rechazados", "decisiones_n_supuestos_rechazados"),
                       ("n_supuestos_no_evaluables", "decisiones_n_supuestos_no_evaluables")):
        ctx.registrar_metrica(f"decisiones.{clave}", r[col], tarea=TAREA, tabla="supuestos_verificados.csv", nota=nota)


# ============================================================================================================================ figura F12_01
COLORES_SEMAFORO = {"verde": "#009E73", "amarillo": "#F0E442", "rojo": "#D55E00", "gris": "#999999"}
TEXTURAS_SEMAFORO = {"verde": "", "amarillo": "..", "rojo": "xx", "gris": "//"}
ETIQUETAS_TIPO = {"cuantitativa": "Cuantitativa", "parcial": "Parcial", "institucional": "Institucional", "no depende de datos": "No depende\nde datos"}


def figura_semaforo(ctx: Contexto) -> None:
    """F12_01: semáforo de las decisiones por tipo de evidencia (barras apiladas con el número de decisiones escrito en cada segmento y una textura por color)."""
    from matplotlib.ticker import MaxNLocator

    t = ctx.leer_tabla("decisiones_evidencia.csv", columnas_texto=COLUMNAS_TEXTO_DECISIONES)
    tipos = [x for x in TIPOS_EVIDENCIA if x in set(t["tipo_evidencia"].to_list())]
    x = np.arange(len(tipos))
    fondo = np.zeros(len(tipos))
    fig, ax = graficos.nueva_figura(figsize=(8.0, 4.8))
    for s in SEMAFOROS:
        valores = np.array([int(((t["tipo_evidencia"] == tipo) & (t["semaforo"] == s)).sum()) for tipo in tipos], dtype=float)
        ax.bar(x, valores, bottom=fondo, width=0.55, color=COLORES_SEMAFORO[s], hatch=TEXTURAS_SEMAFORO[s], edgecolor="black", linewidth=0.5, label=s.capitalize())
        for xi, v, b in zip(x, valores, fondo):
            if v > 0:
                ax.text(xi, b + v / 2.0, str(int(v)), ha="center", va="center", fontsize=9, fontweight="bold", bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.9})
        fondo = fondo + valores
    ax.set_xticks(x)
    ax.set_xticklabels([ETIQUETAS_TIPO[x_] for x_ in tipos])
    ax.set_ylim(0, max(1.0, float(fondo.max())) * 1.12)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Tipo de evidencia de la decisión")
    ax.set_ylabel("Decisiones pendientes")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, title=None)
    ctx.figuras.guardar_figura(
        fig, id_figura="F12_01", tarea=12,
        titulo="Semáforo de las decisiones pendientes según el tipo de evidencia",
        descripcion_alt="Barras apiladas por tipo de evidencia (cuantitativa, parcial, institucional y no depende de datos) con el número de decisiones en verde, amarillo, rojo y gris; "
                        "el número va escrito en cada segmento y cada color tiene su textura.",
        csv="decisiones_evidencia.csv", anios=ANIOS_TXT, decision="",
        extra_pie="El semáforo recomienda y no decide: toda decisión sigue pendiente de decisión humana."
                  + (" Con extractos la evidencia es indicativa." if ctx.etiqueta == "extracto" else ""),
    )


# ============================================================================================================================ ejecución
def ejecutar(ctx: Contexto, *, cascada: pl.DataFrame | None = None) -> dict[str, pl.DataFrame]:
    """Evalúa las 26 hipótesis y los 15 supuestos, escribe ``t25_hipotesis_evaluadas.csv``, ``decisiones_evidencia.csv`` y ``supuestos_verificados.csv``, registra las
    métricas ``decisiones.*``, agrega los conteos al resumen de evidencia y dibuja F12_01.

    ``cascada`` es la cascada de la Tarea 7 SIN suprimir (para el supuesto S10); sin ella, S10 usa ``t23_cascada.csv``. No lee datos.
    """
    hip_cfg, dec_cfg = config.cargar_hipotesis(), config.cargar_decisiones_fase1()
    problemas = protocolo.validar_hipotesis(hip_cfg, dec_cfg) + validar_registro(hip_cfg, dec_cfg)
    if problemas:
        raise ValueError("el protocolo de hipótesis no es válido: " + "; ".join(problemas[:5]))
    fuente = FuenteContexto(ctx)
    resultados = evaluar_todas(fuente)
    hip_df = construir_hipotesis(resultados)
    sup_df = construir_supuestos(fuente, hip_cfg, cascada)
    dec_df = construir_decisiones(ctx.etiqueta, hip_cfg, dec_cfg, resultados, sup_df)
    ctx.escribir_tabla("t25_hipotesis_evaluadas.csv", hip_df, tarea=TAREA, descripcion="Resultado de cada una de las 26 hipótesis de la Fase 2 según su criterio de decisión")
    ctx.escribir_tabla("decisiones_evidencia.csv", dec_df, tarea=TAREA,
                       descripcion="Evidencia, veredicto y semáforo de cada una de las 17 decisiones pendientes de la Fase 1; la evidencia recomienda y no decide")
    ctx.escribir_tabla("supuestos_verificados.csv", sup_df, tarea=TAREA, descripcion="Estado de los 15 supuestos de datos con la cifra y la tabla que lo respaldan")
    registrar_metricas(ctx, dec_df, sup_df)
    ledger.fusionar_resumen(ctx, resumen_decisiones(dec_df, sup_df, hip_df))
    figura_semaforo(ctx)
    return {"hipotesis": hip_df, "decisiones": dec_df, "supuestos": sup_df}


# ============================================================================================================================ informe
N_SECCIONES = 3
COLUMNAS_TEXTO_SUPUESTOS = ["id", "supuesto", "criterio", "estado", "cifra", "tabla", "nivel", "motivo"]


def agregar_secciones(inf: Informe, ctx: Contexto, n0: int = 1) -> None:
    """Agrega al informe de evidencia (``N_SECCIONES = 3``): el semáforo con F12_01, el detalle de cada decisión y los supuestos verificados.

    Necesita ``decisiones_evidencia.csv``, ``supuestos_verificados.csv`` y los conteos de ``t25_resumen_evidencia.csv`` (los escribe ``ejecutar``). Las cifras de cada decisión
    viajan como marcadores ``{evidencia|texto}`` y ``{veredicto|texto}`` leídos del CSV.
    """
    r = lambda col: cifra(ctx, "t25_resumen_evidencia.csv", col)  # noqa: E731
    dec = ctx.leer_tabla("decisiones_evidencia.csv", columnas_texto=COLUMNAS_TEXTO_DECISIONES)
    sup = ctx.leer_tabla("supuestos_verificados.csv", columnas_texto=COLUMNAS_TEXTO_SUPUESTOS)

    inf.seccion(f"{n0}. Evidencia por decisión")
    inf.parrafo(
        "Cada una de las {n_dec} decisiones pendientes de la Fase 1 recibe un semáforo que resume la evidencia de esta corrida: verde cuando la evidencia respalda la propuesta, "
        "amarillo cuando la matiza o es condicional, rojo cuando la contradice y gris cuando no se puede evaluar con un extracto, depende de los datos completos o es una decisión "
        "institucional. Hay {n_verde} decisiones en verde, {n_amarillo} en amarillo, {n_rojo} en rojo y {n_gris} en gris. El semáforo recomienda y no decide: las {n_dec} "
        "decisiones siguen pendientes de decisión humana.",
        n_dec=r("decisiones_n_total"), n_verde=r("decisiones_n_verde"), n_amarillo=r("decisiones_n_amarillo"), n_rojo=r("decisiones_n_rojo"), n_gris=r("decisiones_n_gris"),
    )
    if ctx.etiqueta == "extracto":
        inf.parrafo(
            "Con extractos la evidencia es indicativa. El historial del paciente, los enlaces de traslados y el volumen por hospital se subestiman por construcción, así que esas "
            "decisiones quedan en gris hasta correr con los datos completos."
        )
    inf.tabla(
        "decisiones_evidencia.csv", df=dec, columnas=["n", "tema", "tipo_evidencia", "semaforo"],
        encabezados={"n": "Decisión", "tema": "Tema", "tipo_evidencia": "Tipo de evidencia", "semaforo": "Semáforo"}, leyenda="Semáforo de las decisiones pendientes",
    )
    inf.figura("F12_01")

    inf.seccion(f"{n0 + 1}. Detalle por decisión")
    inf.parrafo(
        "Para cada decisión se muestran la propuesta de la Fase 1, el veredicto que sale del criterio de cada hipótesis, las cifras leídas de las tablas y lo que falta con los "
        "datos completos. Una hipótesis informativa solo describe: no tiene un umbral de decisión."
    )
    for n in dec["n"].to_list():
        e = lambda col, n_=n: cifra(ctx, "decisiones_evidencia.csv", col, columnas_texto=COLUMNAS_TEXTO_DECISIONES, n=n_)  # noqa: E731
        inf.parrafo(
            "Decisión {n|texto}, {tema|texto}. Propuesta de la Fase 1: {propuesta|texto}. Tipo de evidencia: {tipo|texto}; semáforo: {semaforo|texto}. "
            "Veredicto: {veredicto|texto}. Evidencia: {evidencia|texto}. Falta con los datos completos: {falta|texto}. Estado: {estado|texto}.",
            n=e("n"), tema=e("tema"), propuesta=e("propuesta_fase1"), tipo=e("tipo_evidencia"), semaforo=e("semaforo"), veredicto=e("veredicto"), evidencia=e("evidencia"),
            falta=e("que_falta_con_datos_completos"), estado=e("estado"),
        )

    inf.seccion(f"{n0 + 2}. Supuestos verificados")
    inf.parrafo(
        "De los {n_sup} supuestos de datos, {n_ac} se aceptan, {n_re} se rechazan y {n_ne} no se pueden evaluar con esta fuente. Un supuesto rechazado señala una afirmación de las "
        "fases anteriores que los datos no sostienen y alimenta los cambios sugeridos; un supuesto cuyo criterio no trae una regla numérica queda no evaluable.",
        n_sup=r("decisiones_n_supuestos"), n_ac=r("decisiones_n_supuestos_aceptados"), n_re=r("decisiones_n_supuestos_rechazados"), n_ne=r("decisiones_n_supuestos_no_evaluables"),
    )
    inf.tabla(
        "supuestos_verificados.csv", df=sup, columnas=["id", "supuesto", "estado", "cifra", "tabla"],
        encabezados={"id": "Supuesto", "supuesto": "Afirmación", "estado": "Estado", "cifra": "Cifra", "tabla": "Tabla"}, leyenda="Supuestos de datos verificados",
    )


# FIN-decisiones
