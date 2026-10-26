"""Tarea 12 (parte 3): cambios sugeridos a las Fases 1 y 2, cambios al protocolo, resumen de la evidencia y cierre.

``cambios_fase1.csv`` y ``cambios_fase2.csv`` listan cambios SUGERIDOS a los documentos (``docs/`` nunca se edita). Cada cambio se emite solo si su condición se cumple en los CSV de
la corrida y toda cifra de su texto sale de una tabla publicada (``fmt_dec`` y ``fmt_entero``); las frases fijas citan los documentos y no resultados. Salen de tres fuentes:

(a) cada cifra del ledger que DISCREPA y no es despreciable (en modo completo, toda discrepancia), sin repetir lo que ya cubre un hallazgo del catálogo;
(b) cada supuesto RECHAZADO que no cubre un hallazgo ni una cifra del ledger ya emitidos; y
(c) el catálogo ``HALLAZGOS``: una función por hallazgo con su condición, su documento, su ubicación y su evidencia.

``tipo``: ``corrección`` cuando el modo es completo o la evidencia es estructural o algebraica (colinealidad exacta, diferencia entre documentos); ``aclaración`` cuando solo cambia una
definición, una lista o una procedencia; ``pendiente de datos completos`` cuando depende de una estadística del extracto o de un conteo de población. ``prioridad``: ``alta`` si toca R6,
si una decisión afectada queda en rojo, si un estimado de la Fase 1 DISCREPA o si un supuesto RECHAZADO afecta una decisión; ``media`` para el resto de las discrepancias relevantes;
``baja`` para las despreciables y las aclaraciones.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from . import config, decisiones, ledger
from .contexto import Contexto
from .informes import Informe, cifra, fmt_dec, fmt_entero

TAREA = "T12"
PRIORIDADES = ("alta", "media", "baja")
TIPOS = ("corrección", "aclaración", "pendiente de datos completos")
DOCUMENTOS = ("Fase 1", "Fase 2")
COLUMNAS_CAMBIOS = ("id", "documento", "ubicacion", "texto_actual", "hallazgo", "texto_propuesto", "evidencia_tabla", "evidencia_clave", "prioridad", "tipo")
ESQUEMA_CAMBIOS = {c: pl.String for c in COLUMNAS_CAMBIOS}
COLUMNAS_TEXTO_CAMBIOS = list(COLUMNAS_CAMBIOS)
ESQUEMA_PROTOCOLO = {"id": pl.String, "descripcion": pl.String}
PREFIJO_ID = {"Fase 1": "CF1", "Fase 2": "CF2"}
ARCHIVO_CAMBIOS = {"Fase 1": "cambios_fase1.csv", "Fase 2": "cambios_fase2.csv"}

TIPO_ESTADISTICA = "estadistica"        # depende de una estadística del extracto o de un conteo de población: pendiente con extractos, corrección con datos completos
TIPO_LISTA = "lista"                    # una lista distinta: pendiente con extractos, aclaración con datos completos
TIPO_ESTRUCTURAL = "estructural"        # evidencia estructural o algebraica: siempre corrección
TIPO_DEFINICION = "definicion"          # solo cambia una definición o una procedencia: siempre aclaración
TIPO_SUPUESTO: dict[str, str] = {
    "S01": TIPO_ESTADISTICA, "S02": TIPO_ESTRUCTURAL, "S03": TIPO_ESTADISTICA, "S04": TIPO_ESTRUCTURAL, "S05": TIPO_ESTADISTICA, "S06": TIPO_ESTRUCTURAL, "S07": TIPO_ESTRUCTURAL,
    "S08": TIPO_ESTRUCTURAL, "S09": TIPO_ESTADISTICA, "S10": TIPO_ESTRUCTURAL, "S11": TIPO_ESTADISTICA, "S12": TIPO_ESTADISTICA, "S13": TIPO_ESTADISTICA, "S14": TIPO_ESTADISTICA,
    "S15": TIPO_ESTADISTICA,
}
CONCILIADAS_EN_EL_DOCUMENTO: dict[frozenset[str], str] = {
    frozenset({"C030", "C100"}): "la sección 4.1 de la Fase 2 ya las reconcilia: la primera se mide en cinco años y la segunda en seis, con definiciones distintas",
}
"""Pares de cifras con la misma clave y el mismo alcance en ``cifras_documentadas.csv`` que el propio documento reconcilia (leído con el texto del PDF de la Fase 2): no son una
contradicción, sino un alcance que el cuadro de datos no declara."""
UBICACION_CORREGIDA: dict[str, str] = {"C078": "9.1"}
"""La columna ``ubicacion`` de ``cifras_documentadas.csv`` (de la que solo se corrige ``clave_eda``) cita la sección 8 para C078, pero los 68 hospitales de la Fase 1 están en la
sección 9.1 (verificado con el texto del PDF); el cambio sugerido cita la sección correcta."""
SUPUESTO_LEDGER: dict[str, tuple[str, ...]] = {"S07": ("C018",), "S08": ("C077", "C078")}
"""Supuestos cuya cifra ya está en el ledger: si la cifra del ledger se emite en (a), el supuesto no se repite en (b)."""


# ============================================================================================================================ evidencia y cambios
@dataclass(frozen=True)
class Evidencia:
    """Todo lo que leen los hallazgos: la fuente (tablas publicadas, umbrales, métricas y cifras documentadas) y las tablas de la propia Tarea 12."""

    fuente: decisiones.FuenteEvidencia
    ledger: pl.DataFrame
    supuestos: pl.DataFrame
    decisiones: pl.DataFrame
    cifras: pl.DataFrame
    hipotesis: pl.DataFrame

    @property
    def completo(self) -> bool:
        return self.fuente.etiqueta == "completo"


@dataclass(frozen=True)
class Cambio:
    documento: str
    ubicacion: str
    texto_actual: str
    hallazgo: str
    texto_propuesto: str
    evidencia_tabla: str
    evidencia_clave: str
    prioridad: str
    tipo: str


@dataclass(frozen=True)
class Hallazgo:
    id: str
    funcion: Callable[[Evidencia], list[Cambio]]
    ledger_ids: frozenset[str] = field(default_factory=frozenset)
    supuestos: frozenset[str] = field(default_factory=frozenset)


# ============================================================================================================================ ayudas de lectura
def _ent(v: Any) -> str:
    return fmt_entero(v)


def _d(v: Any, n: int = 2) -> str:
    return fmt_dec(v, n)


def _p(v: Any, n: int = 1) -> str:
    """Proporción (0 a 1) como porcentaje."""
    return f"{fmt_dec(float(v) * 100.0, n)} %"


def _pp(v: Any, n: int = 1) -> str:
    """Valor que ya está en porcentaje."""
    return f"{fmt_dec(float(v), n)} %"


def _lista(items: Sequence[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " y " + items[-1]


def _l(ev: Evidencia, cid: str) -> dict[str, Any] | None:
    """La fila del ledger de una cifra (``None`` si no está)."""
    f = ev.ledger.filter(pl.col("id") == cid)
    return f.row(0, named=True) if f.height == 1 else None


def _doc(ev: Evidencia, cid: str) -> float | None:
    r = _l(ev, cid)
    return None if r is None or r["valor_doc"] is None else float(r["valor_doc"])


def _estado_supuesto(ev: Evidencia, sid: str) -> str | None:
    f = ev.supuestos.filter(pl.col("id") == sid)
    return f["estado"][0] if f.height == 1 else None


def _semaforo(ev: Evidencia, n: int) -> str | None:
    f = ev.decisiones.filter(pl.col("n") == n)
    return f["semaforo"][0] if f.height == 1 else None


def _tabla(ev: Evidencia, nombre: str, textos: Sequence[str] = ()) -> pl.DataFrame | None:
    return ev.fuente.tabla(nombre, textos)


def _fila(t: pl.DataFrame | None, **filtro: Any) -> dict[str, Any] | None:
    """La única fila de la tabla que cumple el filtro (comparación como texto); ``None`` si no hay exactamente una."""
    if t is None:
        return None
    sel = t
    for k, v in filtro.items():
        if k not in sel.columns:
            return None
        sel = sel.filter(pl.col(k).cast(pl.String) == str(v))
    return sel.row(0, named=True) if sel.height == 1 else None


def _valores_o_none(fila: Mapping[str, Any] | None, *cols: str) -> list[Any] | None:
    """Los valores de las columnas, o ``None`` si falta la fila, alguna columna o alguna celda está vacía (suprimida)."""
    if fila is None or any(c not in fila or fila[c] is None for c in cols):
        return None
    return [fila[c] for c in cols]


def _periodo(ev: Evidencia) -> str:
    return ledger.etiqueta_alcance([int(a) for a in ev.fuente.umbral("anios_resultado")])


def _rotulo(r: Mapping[str, Any]) -> str:
    """Descripción de una cifra del ledger sin el prefijo de la lista."""
    for prefijo in ("Estimado de la Fase 1: ", "Cascada: "):
        if str(r["descripcion"]).startswith(prefijo):
            return str(r["descripcion"])[len(prefijo):]
    return str(r["descripcion"])


def _valor_txt(unidad: str, decimales: int, v: float) -> str:
    """Un valor del ledger con su unidad (año sin separador, porcentaje con signo, el resto con los decimales documentados); la regla es la de ``ledger.valor_con_unidad``."""
    return ledger.valor_con_unidad(unidad, int(decimales), v)


def _tipo(ev: Evidencia, base: str) -> str:
    if base == TIPO_ESTRUCTURAL:
        return "corrección"
    if base == TIPO_DEFINICION:
        return "aclaración"
    if base == TIPO_LISTA:
        return "aclaración" if ev.completo else "pendiente de datos completos"
    return "corrección" if ev.completo else "pendiente de datos completos"


def _prioridad(base: str, *, alta: bool = False) -> str:
    return "alta" if alta else base


def _hay_rojo(ev: Evidencia, *decisiones_afectadas: int) -> bool:
    return any(_semaforo(ev, n) == "rojo" for n in decisiones_afectadas)


# ============================================================================================================================ hallazgo 1
ETIQUETAS_CASCADA_FASE1 = {"C112": "GRD no asignable", "C113": "recién nacidos sanos (MDC 15)", "C114": "partos normales sin complicación (MDC 14)"}
"""Rótulos de la tabla de la Fase 1, sección 9.3. C115 no se cita: la Fase 1 lo da redondeado ("unos" y un número en miles) y la Fase 2 lo repite con otra cifra."""


def _h_estimados_fase1(ev: Evidencia) -> list[Cambio]:
    filas = [r for r in (_l(ev, c) for c in ("C112", "C113", "C114", "C115")) if r is not None]
    comparables = [r for r in filas if r["estado"] in ("CONFIRMA", "DISCREPA")]
    if not comparables:
        return []
    citados = [f"{ETIQUETAS_CASCADA_FASE1[r['id']]}, unos {_ent(r['valor_doc'])} casos" for r in filas if r["id"] in ETIQUETAS_CASCADA_FASE1 and r["valor_doc"] is not None]
    actual = "Cascada de exclusiones, volumen estimado" + (": " + "; ".join(citados) if citados else "")
    partes, k = [], 0
    for r in comparables:
        doc, med = _ent(r["valor_doc"]), _ent(r["valor_medido"])
        if r["estado"] == "DISCREPA":
            partes.append(f"{_rotulo(r)}: el estimado {doc} difiere del conteo medido {med}")
        else:
            k += 1
            if r["requiere_resultado"] == "si":
                partes.append(f"{_rotulo(r)}: el estimado {doc} es compatible con el conteo medido de {r['alcance_medido']} ({med}) más la cota del año expuesto")
            else:
                partes.append(f"{_rotulo(r)}: el estimado {doc} es igual al conteo medido {med}")
    hallazgo = (f"Bajo la hipótesis de que los estimados de la Fase 1 provienen de un extracto, coinciden {_ent(k)} de {_ent(len(comparables))} comparables: " + "; ".join(partes))
    propuesto = "Reemplazar los estimados por los valores medidos con los datos completos (Fase 2, sección 5.2)"
    if k:
        propuesto += " y dejar constancia de que algunos estimados provienen de un extracto"
    alta = any(r["estado"] == "DISCREPA" for r in comparables)
    return [Cambio("Fase 1", "9.3", actual, hallazgo, propuesto, "ledger_cifras.csv", "C112; C113; C114; C115", _prioridad("baja", alta=alta), _tipo(ev, TIPO_ESTADISTICA))]


# ============================================================================================================================ hallazgo 2
def _h_entre_10_y_20(ev: Evidencia) -> list[Cambio]:
    c128, c129 = _l(ev, "C128"), _l(ev, "C129")
    if not any(r is not None and r["estado"] == "DISCREPA" for r in (c128, c129)):
        return []
    pares = ledger.razones_fase2_fase1(ev.cifras)
    if not pares:
        return []
    desc = {f["id"]: f["descripcion"] for f in ev.cifras.iter_rows(named=True)}
    rot = lambda p: str(desc[p["id_fase2"]]).replace("Cascada: ", "")  # noqa: E731
    lo = float(c128["valor_doc"]) if c128 and c128["valor_doc"] is not None else 10.0
    hi = float(c129["valor_doc"]) if c129 and c129["valor_doc"] is not None else 20.0
    dentro = [p for p in pares if lo <= p["razon"] <= hi]
    hallazgo = ("Razón de lo medido sobre el estimado de la Fase 1: " + "; ".join(f"{rot(p)} {_d(p['razon'], 2)}" for p in pares)
                + f". Caen entre {_ent(lo)} y {_ent(hi)} veces: " + (_lista([rot(p) for p in dentro]) if dentro else "ninguna"))
    metrica = ev.fuente.metrica("volumen.filas_leidas.total")
    total = ev.fuente.cifra_documentada("C007")
    if not ev.completo and metrica and metrica.get("valor") and total:
        hallazgo += f". El inverso de la fracción de muestreo del extracto es {_d(float(total) / float(metrica['valor']), 2)}"
    coinciden = [r for r in (_l(ev, c) for c in ("C113", "C114")) if r is not None and r["estado"] == "CONFIRMA"]
    if coinciden and not ev.completo:
        nombres = _lista([_rotulo(r) for r in coinciden])
        hallazgo += f". {nombres[:1].upper()}{nombres[1:]}: el estimado de la Fase 1 es compatible con el conteo del extracto, de modo que la razón refleja el muestreo"
    propuesto = "Reemplazar la frase por las razones medidas paso a paso o aclarar que los estimados de la Fase 1 provienen de un extracto"
    return [Cambio("Fase 2", "5.2", "Los estimados de la Fase 1 estaban entre 10 y 20 veces más bajos que lo medido.", hallazgo, propuesto, "ledger_cifras.csv", "C128; C129",
                   "media", _tipo(ev, TIPO_ESTRUCTURAL))]


# ============================================================================================================================ hallazgo 3
TEXTO_CINCO_FASE1 = ("Cinco son las más sospechosas: trastornos de líquidos y electrolitos, coagulopatía, anemia por pérdida de sangre, pérdida de peso y trastornos de la circulación "
                     "pulmonar.")


def _h_cinco_categorias(ev: Evidencia) -> list[Cambio]:
    r = _fila(_tabla(ev, "t23_resumen_seleccion.csv", ["elix_top5"]))
    v = _valores_o_none(r, "elix_top5", "elix_n_coinciden_fase1", "elix_n_coinciden_repo")
    if v is None or int(v[1]) >= 5:
        return []
    top5, n_f1, n_repo = str(v[0]).replace("; ", ", "), v[1], v[2]
    hallazgo = (f"El índice de complicación probable ordena {top5} como las cinco categorías más propensas a ser complicaciones de la hospitalización; "
                f"coinciden {_ent(n_f1)} con la lista de la Fase 1 y {_ent(n_repo)} con la del repositorio (Fase 2, sección 6.1)")
    propuesto = f"Dejar una sola lista de cinco categorías para la sensibilidad, a decidir con revisión clínica (decisión 12); la medición propone {top5}"
    return [Cambio("Fase 1", "8.2 y decisión 8", TEXTO_CINCO_FASE1, hallazgo, propuesto, "t23_resumen_seleccion.csv", "elix_top5; elix_n_coinciden_fase1; elix_n_coinciden_repo",
                   _prioridad("media", alta=_hay_rojo(ev, 8)), _tipo(ev, TIPO_LISTA))]


# ============================================================================================================================ hallazgo 4
def _h_28_categorias(ev: Evidencia) -> list[Cambio]:
    m = ev.fuente.metrica("elix.n_categorias")
    if m is None or m.get("valor") is None or int(round(float(m["valor"]))) == 28:
        return []
    n = int(round(float(m["valor"])))
    hallazgo = (f"La suite implementa {_ent(n)} categorías de Elixhauser con el mapeo de Quan; la cifra de 28 categorías del repositorio no se puede medir aquí porque su código "
                "no forma parte de la suite")
    propuesto = "Indicar qué categorías implementa el repositorio y cuáles omite, o corregir la cifra, antes de dejar una sola lista (sección 6.1)"
    return [Cambio("Fase 2", "6.1", "El repositorio usa 28 y su sensibilidad quita arritmias, coagulopatía, desnutrición, electrolitos y hemorragia aguda.", hallazgo, propuesto,
                   "t23_elix_prevalencia.csv", "elix.n_categorias", "baja", _tipo(ev, TIPO_DEFINICION))]


# ============================================================================================================================ hallazgo 5
def _h_hospproc(ev: Evidencia) -> list[Cambio]:
    r = _fila(_tabla(ev, "t23_resumen_hospitales.csv", ["autorref_supera_maximo"]))
    v = _valores_o_none(r, "autorref_supera_maximo", "autorref_prop_exacta")
    if v is None or str(v[0]) != "si":
        return []
    n_proc, pct_proc = _doc(ev, "C120"), _doc(ev, "C121")
    actual = ("Entrantes. " + (f"Hay {_ent(n_proc)} episodios con hospital de procedencia ({_d(pct_proc, 1)} %). " if n_proc is not None and pct_proc is not None else "")
              + "Hay una contradicción que resolver: la Fase 1 excluye los traslados entrantes de los modelos, pero DERIVADO_OTRO_HOSPITAL figura como predictor condicional.")
    hallazgo = (f"HOSPPROCEDENCIA trae el nombre del hospital y coincide con el hospital del egreso en {_p(v[1])} de los egresos que lo informan, sobre el máximo aceptado de "
                f"{_p(ev.fuente.umbral('autorreferencia_procedencia_max'), 0)}: no sirve para definir el traslado entrante")
    propuesto = "Definir el traslado entrante con TIPO_PROCEDENCIA (otro hospital de la red) y no con HOSPPROCEDENCIA"
    return [Cambio("Fase 2", "5.4", actual, hallazgo, propuesto, "t23_resumen_hospitales.csv; t23_procedencia_autorref.csv", "autorref_prop_exacta; autorref_supera_maximo",
                   _prioridad("media", alta=_hay_rojo(ev, 2, 16)), _tipo(ev, TIPO_ESTADISTICA))]


# ============================================================================================================================ hallazgo 6
def _h_cola_estancia(ev: Evidencia) -> list[Cambio]:
    t = _tabla(ev, "t23_cola_estancia.csv", ["base", "regla", "cumple_def", "cumple_conf", "propuesta"])
    if t is None or "propuesta" not in t.columns:
        return []
    prop = t.filter(pl.col("propuesta") == "si")
    if prop.height == 0:
        return []
    r = prop.sort(pl.col("base") != "E", maintain_order=True).row(0, named=True)           # la base E (cohorte del modelo de estadía) primero
    v = _valores_o_none(r, "regla", "corte_dias", "cumple_conf")
    if v is None:
        return []
    regla, corte, conf = v
    d_, c_ = r.get("prop_afectados_def"), r.get("prop_afectados_conf")
    afecta = (f"afecta a {_p(d_, 2)} de los sobrevivientes de 2019 a 2022" if d_ is not None else "afecta a una fracción de los sobrevivientes de 2019 a 2022")
    afecta += f" y a {_p(c_, 2)} en 2023" if c_ is not None else ""
    se_confirma = str(conf) == "si"
    hallazgo = f"La regla {regla} (corte en {_ent(corte)} días, base {r['base']}) {afecta}; " + ("se confirma en 2023" if se_confirma else "no se confirma en 2023")
    propuesto = f"Fijar la regla de estadías largas en {regla} ({_ent(corte)} días)" + (", confirmada en 2023" if se_confirma else " y confirmarla con los datos completos")
    tipo = "pendiente de datos completos" if (not se_confirma or not ev.completo) else "corrección"
    return [Cambio("Fase 1", "decisión 7", "Estadías muy largas. Definir la regla en la Fase 2 con los datos de 2019 a 2022 solamente.", hallazgo, propuesto, "t23_cola_estancia.csv",
                   "regla; corte_dias; propuesta; cumple_conf", _prioridad("media", alta=_hay_rojo(ev, 7)), tipo)]


# ============================================================================================================================ hallazgo 7
def _h_mapeo_3c(ev: Evidencia) -> list[Cambio]:
    t = _tabla(ev, "t23_elix_vs_3c.csv", ["elix_id", "supera_umbral"])
    if t is None or "supera_umbral" not in t.columns:
        return []
    sobre = [str(r["elix_id"]) for r in t.iter_rows(named=True) if str(r["supera_umbral"]).strip().lower() == "si"]
    s02 = _estado_supuesto(ev, "S02")
    if s02 != "RECHAZADO" and not sobre:
        return []
    partes = []
    for elix, nombre in (("ELIX_11", "diabetes no complicada"), ("ELIX_12", "diabetes complicada")):
        v = _valores_o_none(_fila(t, elix_id=elix), "prev_completo_ah", "prev_3c_ah")
        if v is not None:
            partes.append(f"{nombre} ({elix}) {_p(v[0])} con el mapeo completo y {_p(v[1])} con tres caracteres")
    hallazgo = (f"Con el mapeo de tres caracteres en vez del mapeo completo, {_ent(len(sobre))} de {_ent(t.height)} categorías difieren más del umbral de la prevalencia relativa "
                "en adultos hospitalizados" + (f" ({_lista(sobre)})" if sobre else "") + ("; " + "; ".join(partes) if partes else ""))
    propuesto = "Usar el mapeo completo de Quan (códigos de tres y de cuatro caracteres sin puntos) en la suite y en el repositorio"
    return [Cambio("Fase 2", "3.5", "Todas las categorías con prevalencia visible en la figura están definidas con 3 caracteres. f2_02 (chequeo 13) lo comprueba. "
                   "Mientras no se resuelva, ELIX_11 y ELIX_12 no se pueden usar.",
                   hallazgo, propuesto, "t23_elix_vs_3c.csv; t23_resumen_comorbilidad.csv", "supera_umbral; prev_completo_ah; prev_3c_ah", _prioridad("media", alta=s02 == "RECHAZADO"),
                   _tipo(ev, TIPO_ESTRUCTURAL))]


# ============================================================================================================================ hallazgo 8
def _h_colinealidad(ev: Evidencia) -> list[Cambio]:
    t = _tabla(ev, "t23_vif.csv", ["conjunto", "variable", "es_agregado", "colinealidad_exacta"])
    if t is None or "colinealidad_exacta" not in t.columns:
        return []
    exactas = t.filter((pl.col("conjunto") == "con_agregados") & (pl.col("colinealidad_exacta") == "si"))
    agregadas = [str(v) for v in exactas.filter(pl.col("es_agregado") == "si")["variable"].to_list()]
    if not agregadas:
        return []
    n_cat = exactas.filter(pl.col("es_agregado") == "no").height
    hallazgo = (f"Con las categorías de Elixhauser en el mismo modelo, {_lista(agregadas)} son combinaciones lineales exactas de ellas: la colinealidad exacta alcanza a {_ent(n_cat)} "
                "categorías del conjunto con agregados")
    propuesto = "Usar el conteo de comorbilidades o el puntaje de van Walraven solo en la línea base simple y nunca junto con las 31 categorías en el mismo modelo"
    return [Cambio("Fase 1", "8.3", "Mortalidad. Regresión logística con edad, sexo, grupo clínico y las 31 categorías.", hallazgo, propuesto, "t23_vif.csv",
                   "conjunto = con_agregados; colinealidad_exacta = si", "media", _tipo(ev, TIPO_ESTRUCTURAL))]


# ============================================================================================================================ hallazgo 9
def _h_auc_l1(ev: Evidencia) -> list[Cambio]:
    r = _fila(_tabla(ev, "t23_resumen_seleccion.csv", ["elix_top5"]))
    v = _valores_o_none(r, "auc_modelo_l1_2023", "auc_ref_edad_sexo_2023", "auc_ref_elix31_2023")
    imp = _tabla(ev, "t23_importancia.csv", ["tipo_fila", "variable", "clase_proxy", "en_modelo_l1", "importa"])
    if v is None or imp is None or "importa" not in imp.columns:
        return []
    cercanas = imp.filter((pl.col("tipo_fila") == "variable") & (pl.col("en_modelo_l1") == "si") & (pl.col("importa") == "si") & pl.col("clase_proxy").is_in(["moderado", "alto"]))
    nombres = [str(x) for x in cercanas["variable"].to_list()]
    if not nombres:
        return []
    hallazgo = (f"El modelo L1 llega a un AUC de {_d(v[0], 3)} en 2023, frente a {_d(v[1], 3)} del modelo de edad y sexo y {_d(v[2], 3)} del de las 31 categorías. Incluye variables con "
                f"asociación moderada o alta con el hospital que importan al permutarlas ({_lista(nombres)}): están cerca de ser proxy o fuga y no hay un L1 sin ellas, de modo que "
                "ese AUC no es un desempeño esperable")
    propuesto = "No presentar el AUC del L1 como desempeño esperable y medir un L1 sin las variables cercanas a proxy o fuga antes de citarlo (pendiente de la Tarea 11)"
    return [Cambio("Fase 1", "Requisito: control de variables prohibidas en el modelo", "Este control automático previene el error más grave que puede cometerse en un modelo "
                   "predictivo: usar información del futuro para predecir el presente.", hallazgo, propuesto, "t23_resumen_seleccion.csv; t23_importancia.csv",
                   "auc_modelo_l1_2023; en_modelo_l1; importa; clase_proxy", "media", _tipo(ev, TIPO_ESTADISTICA))]


# ============================================================================================================================ hallazgo 10
def _h_fuga(ev: Evidencia) -> list[Cambio]:
    t = _tabla(ev, "t23_fuga.csv", ["tipo_fila", "variable", "supera_umbral_fuga", "prohibida_siempre"])
    if t is None or "supera_umbral_fuga" not in t.columns:
        return []
    umbral = float(ev.fuente.umbral("auc_sospecha_fuga"))
    filas = [r for r in t.filter(pl.col("tipo_fila") == "variable").iter_rows(named=True) if str(r.get("supera_umbral_fuga")) == "si"]
    if not filas:
        return []
    partes, r6 = [], False
    for r in filas:
        ar, ac = r.get("auc_ranking"), r.get("auc_confirmacion")
        ventanas = [n for n, a in (("el ranking de 2019 a 2022", ar), ("la confirmación de 2023", ac)) if a is not None and float(a) >= umbral]
        partes.append(f"{r['variable']}: AUC de {_d(ar, 3) if ar is not None else 'sin dato'} en el ranking y de {_d(ac, 3) if ac is not None else 'sin dato'} en la confirmación, "
                      f"sobre {_d(umbral, 2)} en {_lista(ventanas) or 'ninguna ventana'}")
        r6 = r6 or str(r.get("prohibida_siempre")) == "si"
    hallazgo = "Variables que superan el umbral de fuga: " + "; ".join(partes)
    propuesto = "Mantener estas variables fuera de los modelos; IR_29301_MORTALIDAD y la severidad del GRD quedan prohibidas siempre, también en los análisis de sensibilidad (R6 y decisión 14)"
    actual = ("R6: la columna IR_29301_MORTALIDAD que entrega el agrupador GRD está prohibida como variable del modelo bajo cualquier circunstancia, tampoco en los análisis de "
              "sensibilidad. Decisión 14, severidad del GRD: usarla solo en un análisis de sensibilidad aparte, nunca en el modelo principal.")
    return [Cambio("Fase 1", "R6 y decisión 14", actual, hallazgo, propuesto, "t23_fuga.csv", "supera_umbral_fuga; auc_ranking; auc_confirmacion", _prioridad("media", alta=r6),
                   _tipo(ev, TIPO_ESTADISTICA))]


# ============================================================================================================================ hallazgo 11
TEXTO_DECISION2_FASE1 = ("Variables condicionales. Decidir sobre servicio de ingreso, especialidad médica y paciente derivado después de revisar si reflejan gravedad o capacidad del "
                         "hospital. Si el servicio de ingreso registra la unidad donde estuvo el paciente durante la hospitalización, se descarta.")


def _h_servicio_ingreso(ev: Evidencia) -> list[Cambio]:
    """Sin traslado interno la igualdad entre SERVICIOINGRESO y SERVICIOALTA es esperable por construcción y no distingue el ingreso de la estadía: la prueba informativa es la
    fila con traslado interno (cambio 1.5 del protocolo, que amplía el criterio de S09). La frase causal ("refleja la estadía") se emite únicamente si las dos filas coinciden en
    la mayoría (S09 rechazado). Con la fila con traslado en contra (S09 aceptado) el cambio compara las dos proporciones, deja constancia de que la variable registra el ingreso
    sin levantar las sospechas de proxy ni de fuga de la Tarea 11 y pide confirmar con los datos completos; sin esa fila no concluye."""
    t = _tabla(ev, "t23_fuga.csv", ["tipo_fila", "variable", "coincide_mayoria"])
    cols = ("prop_igualdad", "ic_bajo", "ic_alto", "coincide_mayoria")
    sin = _valores_o_none(_fila(t, tipo_fila="concordancia"), *cols)
    if sin is None or str(sin[3]) != "si":
        return []
    con = _valores_o_none(_fila(t, tipo_fila="concordancia_con_traslado"), *cols)
    base = f"SERVICIOINGRESO coincide con SERVICIOALTA en {_p(sin[0])} de los egresos sin traslado interno (intervalo de {_p(sin[1])} a {_p(sin[2])})"
    clave = "tipo_fila = concordancia; prop_igualdad; coincide_mayoria"
    if con is None:
        hallazgo = base + ": esa igualdad es esperable por construcción y no distingue el ingreso de la estadía; falta la fila con traslado interno, que es la prueba informativa"
        propuesto = "Medir también la coincidencia en los egresos con traslado interno antes de aplicar la regla de la propia decisión sobre SERVICIOINGRESO"
    else:
        hallazgo = base + f" y en {_p(con[0])} de los egresos con traslado interno (intervalo de {_p(con[1])} a {_p(con[2])})"
        clave = "tipo_fila = concordancia; tipo_fila = concordancia_con_traslado; prop_igualdad; coincide_mayoria"
        if str(con[3]) == "si":
            hallazgo += ": en la práctica refleja el servicio de la estadía y no el del ingreso"
            propuesto = ("Aplicar la regla de la propia decisión: SERVICIOINGRESO coincide en su mayoría con el servicio de alta, también en los egresos con traslado interno, "
                         "de modo que se descartaría del modelo principal; confirmar qué registra antes de usarla")
        else:
            hallazgo += (": sin traslado interno la igualdad es esperable por construcción y no distingue el ingreso de la estadía, y con traslado interno no coincide en la "
                         "mayoría, lo que se espera si registra el ingreso")
            propuesto = ("Dejar constancia de que, con esta evidencia, SERVICIOINGRESO registra el servicio de ingreso y no el de la estadía, de modo que la regla de descarte de la "
                         "decisión 2 no se activa por esa razón; esto no levanta las sospechas de proxy del hospital ni de fuga de la Tarea 11 sobre la variable, que la matriz de "
                         "decisión sigue evaluando; confirmar con los datos completos")
    return [Cambio("Fase 1", "decisión 2", TEXTO_DECISION2_FASE1, hallazgo, propuesto, "t23_fuga.csv", clave,
                   _prioridad("media", alta=_estado_supuesto(ev, "S09") == "RECHAZADO" or _hay_rojo(ev, 2)), _tipo(ev, TIPO_ESTADISTICA))]


# ============================================================================================================================ hallazgo 12
def _h_diagnosticos_secundarios(ev: Evidencia) -> list[Cambio]:
    filas = {c: _l(ev, c) for c in ("C070", "C071", "C072", "C073", "C074", "C075", "C076")}
    if any(r is None or r["valor_doc"] is None or r["valor_medido"] is None for r in filas.values()):
        return []
    if not any(r["estado"] == "DISCREPA" for r in filas.values()):
        return []
    docs = [_d(filas[c]["valor_doc"], 2) for c in ("C070", "C071", "C072", "C073", "C074", "C075")]
    actual = (f"El repositorio reporta que el promedio de diagnósticos secundarios en adultos hospitalizados sube de {docs[0]} en 2019 a {', '.join(docs[1:-1])} y {docs[-1]} en 2024, "
              f"un {_ent(filas['C076']['valor_doc'])} % más.")
    n_dif = sum(r["estado"] == "DISCREPA" for r in filas.values())
    ic = f" (intervalo de {_d(filas['C076']['ic_bajo'], 1)} % a {_d(filas['C076']['ic_alto'], 1)} %)" if filas["C076"]["ic_bajo"] is not None and filas["C076"]["ic_alto"] is not None else ""
    hallazgo = (f"Medido: de {_d(filas['C070']['valor_medido'], 2)} en 2019 a {_d(filas['C075']['valor_medido'], 2)} en 2024, un {_d(filas['C076']['valor_medido'], 1)} % más{ic}; "
                f"{_ent(n_dif)} de las siete cifras difieren de lo documentado por más de tres errores estándar")
    tendencia = _estado_supuesto(ev, "S11") == "ACEPTADO"
    propuesto = "Reemplazar por los valores medidos con los datos completos" + ("; la tendencia creciente se confirma" if tendencia else "")
    return [Cambio("Fase 2", "3.6", actual, hallazgo, propuesto, "t23_deriva_codificacion.csv", "codificacion.media_dx_sec_adultos_hosp.2019; codificacion.media_dx_sec_adultos_hosp.2024; "
                   "codificacion.aumento_pct_2019_2024", "media" if filas["C076"]["magnitud"] == "relevante" else "baja", _tipo(ev, TIPO_ESTADISTICA))]


# ============================================================================================================================ hallazgo 13
def _h_cohorte_estadia(ev: Evidencia) -> list[Cambio]:
    literal = ev.fuente.metrica("cohorte.pct_estancia_total")
    sin_cens = ev.fuente.metrica("cohorte.pct_estancia_total.sin_censurados")
    doc, n_doc = _doc(ev, "C051"), _doc(ev, "C110")
    if not literal or literal.get("valor") is None or not sin_cens or sin_cens.get("valor") is None or doc is None:
        return []
    margen = float(ev.fuente.umbral("conciliacion_dif_relevante_pp"))
    if min(abs(float(literal["valor"]) - doc), abs(float(sin_cens["valor"]) - doc)) < margen:
        return []                                                                                      # alguna lectura coincide con lo documentado: no hay nada que corregir
    alcance = literal.get("alcance") or _periodo(ev)
    actual = (f"Cohorte de estadía: {_ent(n_doc)} episodios, {_d(doc, 2)} % del total." if n_doc is not None else f"Cohorte de estadía: {_d(doc, 2)} % del total.")
    hallazgo = (f"En {alcance} la definición literal del pipeline (mapeo de tres caracteres) da {_pp(literal['valor'], 2)} del total y la que además excluye a los censurados da "
                f"{_pp(sin_cens['valor'], 2)}; ninguna coincide con el {_d(doc, 2)} % documentado, que abarca 2019 a 2024 (el desenlace de 2024 no se lee)")
    propuesto = "Indicar si la cifra incluye o no a los censurados y recalcularla con los datos completos"
    return [Cambio("Fase 2", "3.3 y 5.2", actual, hallazgo, propuesto, "t23_cascada_pipeline.csv", "cohorte.pct_estancia_total; cohorte.pct_estancia_total.sin_censurados", "media",
                   _tipo(ev, TIPO_ESTADISTICA))]


# ============================================================================================================================ hallazgo 14
def _h_prevalencia_fase1(ev: Evidencia) -> list[Cambio]:
    per = _periodo(ev)
    t = _tabla(ev, "t23_cascada_variantes.csv", ["periodo", "variante"])
    if t is None:
        return []
    valores = {v: _fila(t, periodo=per, variante=v) for v in ("A", "B", "C", "D")}
    prev = {v: (r or {}).get("prevalencia") for v, r in valores.items()}
    if prev["A"] is None:
        return []
    c116 = _doc(ev, "C116")
    c117 = _doc(ev, "C117")
    actual = ((f"El {_d(c116, 2)} % de mortalidad corresponde a la base completa. " if c116 is not None else "La mortalidad corresponde a la base completa. ")
              + "Después de las exclusiones el valor cambia, así que se recalcula en la Fase 2.")
    partes = [f"{v} {_p(p, 2)}" for v, p in prev.items() if p is not None]
    hallazgo = f"Prevalencia de la mortalidad (fallecidos sobre todos los egresos de la variante) en {per} según la variante de cohorte: " + ", ".join(partes)
    if c117 is not None:
        hallazgo += f". La cifra de {_d(c117, 2)} % de adultos hospitalizados de 2023 usa otra definición de cohorte y no es comparable (C116 y C117)"
    propuesto = "Reemplazar la cifra por la prevalencia de la variante de cohorte que se decida (decisiones 5 y 15)"
    return [Cambio("Fase 1", "9.5", actual, hallazgo, propuesto, "t23_cascada_variantes.csv; ledger_cifras.csv", "periodo; variante; prevalencia; C116; C117", "media",
                   _tipo(ev, TIPO_ESTADISTICA))]


# ============================================================================================================================ hallazgo 15
def _h_tipos_de_alta(ev: Evidencia) -> list[Cambio]:
    r = _fila(_tabla(ev, "t24_resumen_catalogos.csv"))
    v = _valores_o_none(r, "n_etiquetas_tipoalta", "n_etiquetas_tipoalta_por_prefijo", "pct_perdido_union_exacta")
    if v is None or int(v[1]) <= 0:
        return []
    hallazgo = (f"{_ent(v[1])} de las {_ent(v[0])} etiquetas de TIPOALTA solo se reconocen por prefijo porque el archivo las trae cortadas; una unión exacta con el archivo de roles "
                f"dejaría sin rol a {_pp(v[2], 2)} de los egresos")
    propuesto = "Listar los valores de TIPOALTA con su rol, indicar cuáles vienen cortados y fijar la regla de prefijo para asignarlos"
    return [Cambio("Fase 1", "9.2", "Otros tipos que aparezcan: en la Fase 2 se listan todos los valores de TIPOALTA y se decide cada uno.", hallazgo, propuesto,
                   "t24_resumen_catalogos.csv; t24_tipoalta_etiquetas.csv", "n_etiquetas_tipoalta_por_prefijo; pct_perdido_union_exacta", "media", _tipo(ev, TIPO_DEFINICION))]


# ============================================================================================================================ hallazgo 16
def _h_contradicciones(ev: Evidencia) -> list[Cambio]:
    out = []
    for par in ledger.contradicciones_documentales(ev.ledger):
        a, b = _l(ev, par["id_a"]), _l(ev, par["id_b"])
        if a is None or b is None:
            continue
        ub = a["ubicacion"] if a["ubicacion"] == b["ubicacion"] else f"{a['ubicacion']} y {b['ubicacion']}"
        actual = (f"{a['descripcion']}: {_valor_txt(a['unidad'], a['decimales'], par['valor_a'])} (sección {a['ubicacion']}); "
                  f"{b['descripcion']}: {_valor_txt(b['unidad'], b['decimales'], par['valor_b'])} (sección {b['ubicacion']}).")
        explicacion = CONCILIADAS_EN_EL_DOCUMENTO.get(frozenset({par["id_a"], par["id_b"]}))
        if explicacion:                                    # el propio documento las reconcilia: no es una contradicción sino un alcance mal declarado en el cuadro
            hallazgo = f"Las dos cifras se miden con la misma columna ({par['clave']}) y el cuadro de datos las declara con el mismo alcance ({par['alcance']}), pero {explicacion}"
            propuesto = "Declarar en cada cuadro el alcance y la definición de la cifra"
            tipo = _tipo(ev, TIPO_DEFINICION)
        else:
            hallazgo = f"Las dos cifras se miden con la misma columna ({par['clave']}) y el mismo alcance ({par['alcance']}) pero se documentan distintas"
            propuesto = "Unificar la definición y la cifra, o aclarar por qué difieren"
            tipo = _tipo(ev, TIPO_ESTRUCTURAL)
        out.append(Cambio(par["documento"], ub, actual, hallazgo, propuesto, "ledger_cifras.csv", f"{par['id_a']}; {par['id_b']}", "media", tipo))
    return out


# ============================================================================================================================ hallazgo 17
def _h_diabetes_cero(ev: Evidencia) -> list[Cambio]:
    c068 = _l(ev, "C068")
    r = _fila(_tabla(ev, "t23_resumen_comorbilidad.csv"))
    v = _valores_o_none(r, "prev_completo_ELIX_11", "prev_completo_ELIX_12")
    if c068 is None or c068["estado"] != "CONFIRMA" or c068["valor_doc"] is None or c068["valor_medido"] is None or v is None or float(v[0]) <= 0:
        return []
    actual = f"Pero muestra diabetes complicada y no complicada con {_d(c068['valor_doc'], 1)} %, algo muy improbable en pacientes hospitalizados."
    hallazgo = (f"El mapeo de tres caracteres reproduce la cifra de la figura ({_d(c068['valor_medido'], 1)} %), mientras el mapeo completo da {_p(v[0])} de diabetes no complicada y "
                f"{_p(v[1])} de diabetes complicada en adultos hospitalizados")
    propuesto = "Confirmar que la causa es el mapeo de tres caracteres y corregirlo con el mapeo completo antes de usar ELIX_11 y ELIX_12"
    return [Cambio("Fase 2", "3.5", actual, hallazgo, propuesto, "t23_resumen_comorbilidad.csv; ledger_cifras.csv", "prev_completo_ELIX_11; prev_completo_ELIX_12; C068",
                   _prioridad("media", alta=_estado_supuesto(ev, "S02") == "RECHAZADO"), _tipo(ev, TIPO_ESTRUCTURAL))]


# ============================================================================================================================ catálogo
HALLAZGOS: tuple[Hallazgo, ...] = (
    Hallazgo("estimados_fase1", _h_estimados_fase1, frozenset({"C112", "C113", "C114", "C115"})),
    Hallazgo("entre_10_y_20", _h_entre_10_y_20, frozenset({"C128", "C129"})),
    Hallazgo("cinco_categorias", _h_cinco_categorias),
    Hallazgo("categorias_del_repositorio", _h_28_categorias),
    Hallazgo("hospproc", _h_hospproc),
    Hallazgo("cola_estancia", _h_cola_estancia),
    Hallazgo("mapeo_tres_caracteres", _h_mapeo_3c, frozenset({f"C{i:03d}" for i in range(62, 69)}), frozenset({"S02"})),
    Hallazgo("colinealidad_exacta", _h_colinealidad),
    Hallazgo("auc_l1", _h_auc_l1),
    Hallazgo("fuga", _h_fuga),
    Hallazgo("servicio_ingreso", _h_servicio_ingreso, frozenset(), frozenset({"S09"})),
    Hallazgo("diagnosticos_secundarios", _h_diagnosticos_secundarios, frozenset({f"C{i:03d}" for i in range(70, 77)}), frozenset({"S11"})),
    Hallazgo("cohorte_estadia", _h_cohorte_estadia, frozenset({"C051"})),
    Hallazgo("prevalencia_fase1", _h_prevalencia_fase1, frozenset({"C116", "C117"})),
    Hallazgo("tipos_de_alta", _h_tipos_de_alta),
    Hallazgo("contradicciones_documentales", _h_contradicciones, frozenset({"C030", "C100"})),
    Hallazgo("diabetes_cero", _h_diabetes_cero, frozenset({"C068"})),
)


# ============================================================================================================================ fuentes (a) y (b)
def _cambio_de_ledger(ev: Evidencia, r: Mapping[str, Any]) -> Cambio:
    unidad, dec = str(r["unidad"]), int(r["decimales"])
    doc, med = _valor_txt(unidad, dec, r["valor_doc"]), _valor_txt(unidad, dec, r["valor_medido"])
    cota = (not ev.completo) and ledger.es_clave_de_distintos(str(r["clave_usada"] or ""))
    hallazgo = f"La suite midió {med} (alcance {r['alcance_medido']}) en {r['tabla_origen']}"
    if cota:
        hallazgo += "; el extracto es una cota inferior de la población"
    elif not ev.completo and r["tipo"] == "proporcion":
        hallazgo += "; con un extracto el resultado es indicativo"
    if r["tipo"] == "estructura":
        propuesto = f"Corregir {doc} por {med}"
    elif cota:
        propuesto = f"Reemplazar {doc} por al menos {med}"
    else:
        propuesto = f"Reemplazar {doc} por {med}" + ("" if ev.completo else " tras repetir la medición con los datos completos")
    if r["documento"] == "Fase 1":
        prioridad = "alta"
    elif r["magnitud"] == "despreciable":
        prioridad = "baja"
    else:
        prioridad = "media"
    tipo = "corrección" if (ev.completo or r["tipo"] == "estructura" or cota) else "pendiente de datos completos"
    return Cambio(str(r["documento"]), UBICACION_CORREGIDA.get(str(r["id"]), str(r["ubicacion"])), f"{r['descripcion']}: {doc}", hallazgo, propuesto,
                  str(r["tabla_origen"] or "ledger_cifras.csv"), str(r["clave_usada"] or r["id"]), prioridad, tipo)


def _cambios_ledger(ev: Evidencia, cubiertas: set[str]) -> tuple[list[Cambio], set[str]]:
    """(a) Una cifra del ledger que DISCREPA y no es despreciable (en modo completo, toda discrepancia), sin repetir lo que ya cubre un hallazgo del catálogo."""
    cambios, ids = [], set()
    for r in ev.ledger.sort("id").iter_rows(named=True):
        if r["estado"] != "DISCREPA" or r["id"] in cubiertas or r["valor_doc"] is None or r["valor_medido"] is None:
            continue
        if not ev.completo and r["magnitud"] == "despreciable":
            continue
        cambios.append(_cambio_de_ledger(ev, r))
        ids.add(str(r["id"]))
    return cambios, ids


def _afecta_decision(ev: Evidencia, sid: str) -> bool:
    f = ev.hipotesis.filter(pl.col("id") == sid)
    return f.height == 1 and bool(str(f["decision_afectada"][0]).strip())


def _cambios_supuestos(ev: Evidencia, cubiertos: set[str], ledger_emitido: set[str]) -> list[Cambio]:
    """(b) Un supuesto RECHAZADO que no cubre un hallazgo del catálogo ni una cifra del ledger ya emitida."""
    cambios = []
    for s in ev.supuestos.sort("id").iter_rows(named=True):
        sid = str(s["id"])
        if s["estado"] != "RECHAZADO" or sid in cubiertos or any(c in ledger_emitido for c in SUPUESTO_LEDGER.get(sid, ())):
            continue
        medido = str(s["cifra"] or "")
        hallazgo = "; ".join(x for x in (medido, str(s["motivo"] or "")) if x) or "El criterio del supuesto no se cumple"
        propuesto = (f"Reemplazar la afirmación del documento por lo medido ({medido}) e indicar la regla aplicada; criterio del supuesto: {s['criterio']}" if medido
                     else f"Aplicar el criterio del supuesto: {s['criterio']}")
        cambios.append(Cambio("Fase 2", f"Supuesto {sid}", str(s["supuesto"]), hallazgo, propuesto, str(s["tabla"]), sid,
                              "alta" if _afecta_decision(ev, sid) else "media", _tipo(ev, TIPO_SUPUESTO.get(sid, TIPO_ESTADISTICA))))
    return cambios


def prioridad_final(c: Cambio) -> str:
    """Una aclaración no pasa de ``baja`` salvo que otra regla la haya hecho ``alta`` (R6, una decisión en rojo, un estimado de la Fase 1 o un supuesto rechazado)."""
    return "baja" if c.tipo == "aclaración" and c.prioridad == "media" else c.prioridad


def _a_tabla(documento: str, cambios: Sequence[Cambio]) -> pl.DataFrame:
    ordenados = sorted(enumerate(cambios), key=lambda p: (PRIORIDADES.index(prioridad_final(p[1])), p[0]))
    filas = [
        {"id": f"{PREFIJO_ID[documento]}-{i:03d}", "documento": c.documento, "ubicacion": c.ubicacion, "texto_actual": c.texto_actual, "hallazgo": c.hallazgo,
         "texto_propuesto": c.texto_propuesto, "evidencia_tabla": c.evidencia_tabla, "evidencia_clave": c.evidencia_clave, "prioridad": prioridad_final(c), "tipo": c.tipo}
        for i, (_, c) in enumerate(ordenados, start=1)
    ]
    return pl.DataFrame(filas, schema=ESQUEMA_CAMBIOS)


def construir_cambios(ev: Evidencia) -> dict[str, pl.DataFrame]:
    """Los cambios sugeridos a cada documento (``"Fase 1"`` y ``"Fase 2"``), ordenados por prioridad. Cada hallazgo solo se emite si su condición se cumple."""
    emitidos: list[Cambio] = []
    ledger_cubierto: set[str] = set()
    supuestos_cubiertos: set[str] = set()
    for h in HALLAZGOS:
        cs = h.funcion(ev)
        if cs:
            emitidos.extend(cs)
            ledger_cubierto |= h.ledger_ids
            supuestos_cubiertos |= h.supuestos
    desde_ledger, ids_ledger = _cambios_ledger(ev, ledger_cubierto)
    emitidos.extend(desde_ledger)
    emitidos.extend(_cambios_supuestos(ev, supuestos_cubiertos, ledger_cubierto | ids_ledger))
    return {doc: _a_tabla(doc, [c for c in emitidos if c.documento == doc]) for doc in DOCUMENTOS}


# ============================================================================================================================ protocolo, resumen y ejecución
def tabla_cambios_protocolo(ctx: Contexto) -> pl.DataFrame:
    """``cambios_protocolo.csv``: los cambios hechos al protocolo después de ver resultados (``cambios_protocolo`` de ``config/umbrales.toml``)."""
    filas = []
    for i, texto in enumerate(ctx.umbrales.meta.get("cambios_protocolo", []), start=1):
        id_, sep, resto = str(texto).partition(":")
        filas.append({"id": id_.strip(), "descripcion": resto.strip()} if sep else {"id": str(i), "descripcion": str(texto).strip()})
    return pl.DataFrame(filas, schema=ESQUEMA_PROTOCOLO)


def resumen_cambios(fase1: pl.DataFrame, fase2: pl.DataFrame, protocolo: pl.DataFrame, version: str) -> dict[str, Any]:
    """Conteos de los cambios sugeridos y del protocolo para el resumen de evidencia."""
    todos = pl.concat([fase1, fase2])
    c = lambda col, valor: int((todos[col] == valor).sum())  # noqa: E731
    return {
        "cambios_n_total": todos.height, "cambios_n_fase1": fase1.height, "cambios_n_fase2": fase2.height,
        "cambios_n_alta": c("prioridad", "alta"), "cambios_n_media": c("prioridad", "media"), "cambios_n_baja": c("prioridad", "baja"),
        "cambios_n_correccion": c("tipo", "corrección"), "cambios_n_aclaracion": c("tipo", "aclaración"), "cambios_n_pendiente": c("tipo", "pendiente de datos completos"),
        "cambios_n_protocolo": protocolo.height, "cambios_protocolo_version": f"v{version}",
    }


def armar_evidencia(ctx: Contexto) -> Evidencia:
    """Lee las tablas ya publicadas de la Tarea 12 (la versión con supresión) y arma la ``Evidencia`` de los hallazgos."""
    return Evidencia(
        fuente=decisiones.FuenteContexto(ctx),
        ledger=ctx.leer_tabla("ledger_cifras.csv", columnas_texto=list(ledger.COLUMNAS_TEXTO_LEDGER)),
        supuestos=ctx.leer_tabla("supuestos_verificados.csv", columnas_texto=decisiones.COLUMNAS_TEXTO_SUPUESTOS),
        decisiones=ctx.leer_tabla("decisiones_evidencia.csv", columnas_texto=decisiones.COLUMNAS_TEXTO_DECISIONES),
        cifras=ledger.cargar_cifras(),
        hipotesis=config.cargar_hipotesis(),
    )


def ejecutar(ctx: Contexto, *, cascada: pl.DataFrame | None = None) -> dict[str, pl.DataFrame]:
    """Corre la Tarea 12 completa: el ledger, la evidencia por decisión y los supuestos, y escribe los cambios sugeridos a las Fases 1 y 2, los cambios al protocolo y el resumen.

    Tablas de la tarea: ``ledger_cifras.csv`` (más ``ledger_motivos.csv``), ``t25_hipotesis_evaluadas.csv``, ``decisiones_evidencia.csv``, ``supuestos_verificados.csv``,
    ``cambios_fase1.csv``, ``cambios_fase2.csv``, ``cambios_protocolo.csv`` y ``t25_resumen_evidencia.csv``; figuras F12_01 y F12_02. No lee datos. ``cascada`` es la cascada de la
    Tarea 7 sin suprimir (para el supuesto S10).
    """
    led = ledger.ejecutar(ctx)
    dec = decisiones.ejecutar(ctx, cascada=cascada)
    cambios = construir_cambios(armar_evidencia(ctx))
    for doc in DOCUMENTOS:
        ctx.escribir_tabla(ARCHIVO_CAMBIOS[doc], cambios[doc], tarea=TAREA,
                           descripcion=f"Cambios sugeridos al documento de la {doc} (no se edita ningún documento) con su hallazgo, su evidencia y su prioridad")
    protocolo = tabla_cambios_protocolo(ctx)
    ctx.escribir_tabla("cambios_protocolo.csv", protocolo, tarea=TAREA, descripcion="Cambios hechos al protocolo congelado después de ver resultados")
    ledger.fusionar_resumen(ctx, resumen_cambios(cambios["Fase 1"], cambios["Fase 2"], protocolo, str(ctx.umbrales.meta.get("version", ""))))
    return {"ledger": led, **dec, "cambios_fase1": cambios["Fase 1"], "cambios_fase2": cambios["Fase 2"], "cambios_protocolo": protocolo}


# ============================================================================================================================ informe
N_SECCIONES_CAMBIOS = 1
N_SECCIONES = 3
COLUMNAS_TEXTO_HIPOTESIS = list(decisiones.ESQUEMA_HIPOTESIS)
ENCABEZADOS_CAMBIOS = {"id": "Cambio", "ubicacion": "Ubicación", "prioridad": "Prioridad", "tipo": "Tipo", "hallazgo": "Hallazgo", "texto_propuesto": "Texto propuesto"}


def agregar_secciones_cambios(inf: Informe, ctx: Contexto, n0: int = 1) -> None:
    """Agrega al informe de evidencia (``N_SECCIONES_CAMBIOS = 1``) los cambios sugeridos a los documentos. Necesita ``cambios_fase1.csv``, ``cambios_fase2.csv`` y el resumen."""
    r = lambda col: cifra(ctx, "t25_resumen_evidencia.csv", col)  # noqa: E731
    inf.seccion(f"{n0}. Cambios sugeridos a los documentos de las fases anteriores")
    inf.parrafo(
        "Se sugieren {n_total} cambios a los documentos: {n_f1} a la Fase 1 y {n_f2} a la Fase 2. Por prioridad hay {n_alta} altos, {n_media} medios y {n_baja} bajos; por tipo, "
        "{n_corr} correcciones, {n_acl} aclaraciones y {n_pend} pendientes de datos completos. Son sugerencias: esta suite no edita los documentos de las fases, y cada una trae "
        "su hallazgo, la tabla y la clave que la respaldan.",
        n_total=r("cambios_n_total"), n_f1=r("cambios_n_fase1"), n_f2=r("cambios_n_fase2"), n_alta=r("cambios_n_alta"), n_media=r("cambios_n_media"), n_baja=r("cambios_n_baja"),
        n_corr=r("cambios_n_correccion"), n_acl=r("cambios_n_aclaracion"), n_pend=r("cambios_n_pendiente"),
    )
    for doc in DOCUMENTOS:
        archivo = ARCHIVO_CAMBIOS[doc]
        t = ctx.leer_tabla(archivo, columnas_texto=COLUMNAS_TEXTO_CAMBIOS)
        if t.height == 0:
            inf.texto_libre_sin_cifras(f"Con esta evidencia no hay cambios sugeridos a la {doc}.")
            continue
        inf.tabla(archivo, df=t, columnas=list(ENCABEZADOS_CAMBIOS), encabezados=ENCABEZADOS_CAMBIOS, max_filas=40, leyenda=f"Cambios sugeridos a la {doc}")


def agregar_secciones(inf: Informe, ctx: Contexto, n0: int = 1) -> None:
    """Agrega al informe de cierre (``N_SECCIONES = 3``): los cambios al protocolo, las limitaciones y lo que falta con los datos completos."""
    r = lambda col: cifra(ctx, "t25_resumen_evidencia.csv", col, columnas_texto=["cambios_protocolo_version"])  # noqa: E731
    extracto = ctx.etiqueta == "extracto"

    inf.seccion(f"{n0}. Cambios al protocolo")
    inf.parrafo(
        "Después de ver resultados se hicieron {n_cambios} cambios al protocolo congelado en la Tarea 1, que quedó en la versión {version|texto}. La tabla dice qué cambió y por qué.",
        n_cambios=r("cambios_n_protocolo"), version=r("cambios_protocolo_version"),
    )
    inf.tabla("cambios_protocolo.csv", df=ctx.leer_tabla("cambios_protocolo.csv", columnas_texto=["id", "descripcion"]), encabezados={"id": "Cambio", "descripcion": "Descripción"},
              leyenda="Cambios al protocolo después de ver resultados")

    inf.seccion(f"{n0 + 1}. Limitaciones")
    limites = []
    if extracto:
        limites.append("- Con extractos las cifras son indicativas: el extracto es una fracción menor de los datos y no es una muestra aleatoria simple. El modo completo solo se probó con datos sintéticos.")
    limites += [
        "- El desenlace de 2024 nunca se leyó. Las cifras que lo requieren se verifican solo con 2019 a 2023 o con las cotas de la Tarea 7; el resto queda no verificable.",
        "- Los intervalos de confianza ignoran el agrupamiento por paciente y por hospital.",
        "- El ledger compara cada cifra con una sola medición: si el documento y la suite usan definiciones distintas, la cifra queda no verificable y no se declara una discrepancia.",
        "- El mapeo de tres caracteres del pipeline es una reconstrucción aproximada: el código original no está en el repositorio revisado.",
        "- El AUC del modelo L1 no es un desempeño esperable: incluye variables cercanas a proxy o fuga y la Tarea 11 no midió un L1 sin ellas.",
        "- El conocimiento clínico que no proviene de las cuatro referencias verificadas en fuente queda pendiente de revisión clínica (decisión 12).",
        "- La evidencia recomienda y no decide: toda decisión sigue pendiente de decisión humana.",
    ]
    inf.texto_libre_sin_cifras("\n".join(limites))

    inf.seccion(f"{n0 + 2}. Qué falta con los datos completos")
    inf.parrafo(
        "{n_noev} de las {n_hip} hipótesis quedan no evaluables, {n_nv} de las {n_cifras} cifras documentadas quedan no verificables y {n_sup_ne} de los {n_sup} supuestos no se pueden "
        "evaluar. Entre las hipótesis, unas se repiten con los datos completos antes de decidir (la corrida usa `run_fase2_eda.py` con la fuente completa y la carpeta de datos) y "
        "otras no dependen de datos: son decisiones metodológicas o institucionales que ninguna corrida resuelve.",
        n_noev=r("decisiones_n_hipotesis_no_evaluable"), n_hip=r("decisiones_n_hipotesis"), n_nv=r("ledger_n_no_verificable"), n_cifras=r("ledger_n_cifras"),
        n_sup_ne=r("decisiones_n_supuestos_no_evaluables"), n_sup=r("decisiones_n_supuestos"),
    )
    hip = ctx.leer_tabla("t25_hipotesis_evaluadas.csv", columnas_texto=COLUMNAS_TEXTO_HIPOTESIS)
    pend = hip.filter(pl.col("resultado") == "no_evaluable")
    encabezados = {"id_hipotesis": "Hipótesis", "decision": "Decisión", "clase": "Clase", "motivo": "Motivo"}
    columnas = ["id_hipotesis", "decision", "clase", "motivo"]
    con_datos = pend.filter(pl.col("motivo") != decisiones.MOTIVO_SIN_DATOS)                  # se repiten con los datos completos (o no se pudieron leer en esta corrida)
    sin_datos = pend.filter(pl.col("motivo") == decisiones.MOTIVO_SIN_DATOS)                  # decisiones metodológicas o institucionales: los datos no las resuelven
    if con_datos.height:
        inf.tabla("t25_hipotesis_evaluadas.csv", df=con_datos, columnas=columnas, encabezados=encabezados, leyenda="Hipótesis que se repiten con los datos completos")
    if sin_datos.height:
        inf.tabla("t25_hipotesis_evaluadas.csv", df=sin_datos, columnas=columnas, encabezados=encabezados, leyenda="Hipótesis que no dependen de datos")


# FIN-cierre
