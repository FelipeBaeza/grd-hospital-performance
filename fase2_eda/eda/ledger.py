"""Tarea 12 (parte 1): ledger de cifras documentadas frente a lo medido por la suite.

Cada una de las 129 cifras de ``config/cifras_documentadas.csv`` (Fases 1 y 2) se compara con la métrica que la suite midió con la misma ``clave_eda`` y recibe un
``estado`` en {CONFIRMA, DISCREPA, NO VERIFICABLE} con su ``motivo``. La lógica de comparación de proporciones es la de la Tarea 4 (``descripcion.conciliar`` y
``descripcion.magnitud_discrepancia``): aquí no se duplica.

Reglas, en orden (la primera que calza decide; ver ``evaluar_cifra``):

1. ``tipo = aritmetica`` (C128 y C129): se recalcula con las propias cifras documentadas, sin métrica (razón entre la medida de la Fase 2 y el estimado de la Fase 1).
2. ``NO_COMPARABLES``: la propia Fase 2 dice que las definiciones difieren; NO VERIFICABLE con el motivo fijo.
3. Sin métrica o sin valor: NO VERIFICABLE. Si la cifra requiere el desenlace y su alcance incluye el año expuesto, el motivo es el del año expuesto.
4. Privacidad: un conteo documentado o medido de 1 a 9 casos no se publica ni se compara.
5. ``SUBESTIMADAS_EN_EXTRACTO``: la métrica se subestima por construcción en un extracto.
6. Año expuesto: si la cifra requiere el desenlace y su alcance incluye 2024, el desenlace de 2024 no se lee jamás. Un conteo absoluto se compara con el intervalo
   [conteo de 2019 a 2023, ese conteo más la cota que dejó la Tarea 7] (inclusivo); sin cota queda NO VERIFICABLE. Si los alcances (documentado y medido) son
   períodos distintos, NO VERIFICABLE.
7. Por tipo y nivel: en modo completo, igualdad al redondeo de ``decimales``; en extracto un conteo de la población completa no es comparable (salvo la hipótesis de
   que la cifra de la Fase 1 proviene del extracto, la cota inferior de los conteos de valores distintos y los casos especiales) y una proporción se contrasta con un
   error estándar (el del intervalo de la métrica o el binomial) contra ``conciliacion_z``. En extracto el resultado es indicativo.

Todo estado queda justificado con el valor medido, su tabla (``tabla_origen``) y un ``motivo``. La Tarea 12 no lee datos: usa las métricas de la corrida.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from . import config, descripcion, graficos, rutas
from .contexto import Contexto
from .informes import Informe, cifra, fmt_dec, fmt_entero

TAREA = "T12"
ANIOS_TXT = "2019 a 2024"
ESTADOS = ("CONFIRMA", "DISCREPA", "NO VERIFICABLE")
TIPOS = ("absoluto", "proporcion", "estructura", "aritmetica")
N_CIFRAS_ESPERADAS = 129
SEPARADOR_DETALLE = " | "

COLUMNAS_LEDGER = (
    "id", "documento", "ubicacion", "descripcion", "tipo", "unidad", "alcance_doc", "requiere_resultado", "valor_doc", "decimales", "clave_eda", "clave_usada",
    "valor_medido", "ic_bajo", "ic_alto", "nivel", "alcance_medido", "tabla_origen", "n_doc", "n_medido", "dif_abs", "dif_rel", "z", "magnitud", "estado", "motivo", "nota",
)
ESQUEMA_LEDGER: dict[str, Any] = {
    "id": pl.String, "documento": pl.String, "ubicacion": pl.String, "descripcion": pl.String, "tipo": pl.String, "unidad": pl.String, "alcance_doc": pl.String,
    "requiere_resultado": pl.String, "valor_doc": pl.Float64, "decimales": pl.Int64, "clave_eda": pl.String, "clave_usada": pl.String, "valor_medido": pl.Float64,
    "ic_bajo": pl.Float64, "ic_alto": pl.Float64, "nivel": pl.String, "alcance_medido": pl.String, "tabla_origen": pl.String, "n_doc": pl.Int64, "n_medido": pl.Int64,
    "dif_abs": pl.Float64, "dif_rel": pl.Float64, "z": pl.Float64, "magnitud": pl.String, "estado": pl.String, "motivo": pl.String, "nota": pl.String,
}
COLUMNAS_TEXTO_LEDGER = ("id", "documento", "ubicacion", "descripcion", "tipo", "unidad", "alcance_doc", "requiere_resultado", "clave_eda", "clave_usada", "nivel",
                         "alcance_medido", "tabla_origen", "magnitud", "estado", "motivo", "nota")
COLS_N_LEDGER = ("n_doc", "n_medido")
DEPENDIENTES_LEDGER = ("valor_doc", "valor_medido", "ic_bajo", "ic_alto", "dif_abs", "dif_rel", "z")

# Motivos fijos (sin cifras): el detalle numérico viaja en las columnas de valores y, cuando hace falta, tras ``SEPARADOR_DETALLE``.
MOTIVOS: dict[str, str] = {
    "sin_clave": "la cifra no tiene clave de medición en la suite",
    "sin_metrica": "la corrida no produjo la métrica",
    "suprimida": "valor suprimido por tener menos de diez casos",
    "nula": "la métrica no trae valor",
    "pocos_casos": "cifra con menos de diez casos: no se publica",
    "sin_cota": "el alcance documentado incluye 2024; el desenlace de 2024 no se lee y no hay cota",
    "poblacion_completa": "conteo de la población completa: el extracto no es comparable",
    "alcances_distintos": "alcances distintos: la suite midió otro período que el documentado",
    "sin_error_estandar": "la métrica no trae error estándar",
    "sin_tamano": "no hay tamaño de muestra para el alcance medido",
    "evento_raro": "evento raro en el extracto: la comparación no es evaluable",
    "valores_distintos": "el extracto puede no contener todos los valores distintos: el observado es una cota inferior",
    "mas_de_extracto": "el observado en el extracto es una cota inferior de la población: no alcanza para confirmar ni para contradecir",
    "aproximada_poblacion": "comparación aproximada de un conteo de la población completa: el extracto no es comparable",
    "sin_pares": "no hay pares de cifras de la Fase 2 y de la Fase 1 para recalcular la razón",
    "clave_aritmetica": "la clave aritmética no es mínimo ni máximo de las razones",
    "tipo_desconocido": "el tipo de cifra no tiene regla de comparación",
    "unidad_sin_regla": "la unidad de la cifra no tiene regla de comparación en extracto",
}

# Constantes por id (cada una se prueba contra ``config/cifras_documentadas.csv``).
NO_COMPARABLES: dict[str, str] = {
    "C116": "la Fase 2 compara tres definiciones de cohorte y de prevalencia y dice que la cifra de la Fase 1 no coincide con ninguna cohorte medida; la variante de la "
            "suite no es la cohorte de los scripts exploratorios",
    "C117": "definiciones distintas de la cohorte de adultos hospitalizados: la prevalencia de los scripts exploratorios no usa la cohorte de la variante C de la suite",
    "C089": "la métrica de la suite cubre todas las columnas de diagnóstico y el documento solo los diez primeros campos",
}
SUBESTIMADAS_EN_EXTRACTO: dict[str, str] = {
    "C119": "en un extracto el ingreso siguiente de un paciente casi nunca está en el mismo archivo: la tasa de enlace se subestima por construcción y la propia métrica la "
            "declara no interpretable",
}
COMPARACION_ESPECIAL: dict[str, str] = {"C015": "aproximado", "C125": "mas_de", "C126": "aproximado", "C127": "aproximado"}
HIPOTESIS_ORIGEN_EXTRACTO: frozenset[str] = frozenset({"C112", "C113", "C114", "C115"})
PREFIJOS_DISTINTOS: tuple[str, ...] = ("cardinalidad.", "tipoalta.n_valores_distintos", "hospitales.n_distintos.")
CLAVES_SIN_ERROR_ESTANDAR: frozenset[str] = frozenset({"resultado.caida_mortalidad_2021_2023_pct"})
ARITMETICA: dict[str, str] = {"doc.razon_minima_cascada": "min", "doc.razon_maxima_cascada": "max"}

VALOR_PEQUENO = "<10"
"""Valor de ``cifras_documentadas.csv`` para un conteo documentado de 1 a 9 casos: el registro no copia la cifra del documento (cambio 1.6) y la fila queda NO
VERIFICABLE por pocos casos, igual que si se hubiera copiado."""

_RE_ALCANCE = re.compile(r"^\s*(?:(\d{4})\s*[-_]\s*(\d{4})|(\d{4}))\s*$")


# ---------------------------------------------------------------------------------------------------------------------------------- redondeo y alcances
def redondear(x: float, decimales: int) -> float:
    """Redondeo de mitad hacia arriba (lejos de cero) sobre la representación decimal más corta de ``x`` (2,575 con dos decimales da 2,58)."""
    q = Decimal(1).scaleb(-int(decimales))
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def cifras_significativas(x: float) -> int:
    """Cifras significativas de un valor documentado sin contar los ceros finales (970000 tiene dos)."""
    digitos = Decimal(repr(float(abs(x)))).normalize().as_tuple().digits
    return max(1, len(digitos))


def redondear_cifras(x: float, cifras: int) -> float:
    """Redondea ``x`` a ``cifras`` cifras significativas (mitad hacia arriba)."""
    if x == 0:
        return 0.0
    exponente = math.floor(math.log10(abs(x))) - int(cifras) + 1
    return redondear(x, -exponente)


def anios_de_alcance(texto: str | None) -> tuple[int, int] | None:
    """``(primer año, último año)`` de un alcance numérico (``2023`` o ``2019-2023``); ``None`` si no es numérico (``diccionario``, ``documental``...)."""
    m = _RE_ALCANCE.match(texto or "")
    if m is None:
        return None
    if m.group(3):
        a = int(m.group(3))
        return a, a
    lo, hi = int(m.group(1)), int(m.group(2))
    return (lo, hi) if lo <= hi else None


def etiqueta_alcance(anios: Sequence[int]) -> str:
    """``2019`` si hay un solo año o ``2019-2023`` con el primero y el último."""
    a = sorted(int(x) for x in anios)
    return str(a[0]) if len(a) == 1 or a[0] == a[-1] else f"{a[0]}-{a[-1]}"


def incluye_anio(alcance: str | None, anio: int) -> bool:
    r = anios_de_alcance(alcance)
    return r is not None and r[0] <= int(anio) <= r[1]


def n_del_alcance(alcance: str | None, filas_por_anio: Mapping[int, int | float | None]) -> int | None:
    """Suma de las filas leídas de los años del alcance (cota superior del ``n`` real); ``None`` si el alcance no es numérico o falta algún año."""
    r = anios_de_alcance(alcance)
    if r is None:
        return None
    total = 0
    for a in range(r[0], r[1] + 1):
        v = filas_por_anio.get(a)
        if v is None:
            return None
        total += int(v)
    return total


def alternativas_de(clave_eda: str | None) -> list[str]:
    """Alternativas de ``clave_eda`` separadas por ``;`` (la primera es la preferida)."""
    return [k.strip() for k in (clave_eda or "").split(";") if k.strip()]


def es_clave_de_distintos(clave: str) -> bool:
    return any(clave.startswith(p) for p in PREFIJOS_DISTINTOS) and not clave.endswith(".chao1")


def _pequena(v: float | None, minimo: int) -> bool:
    return v is not None and 0 < round(float(v)) < int(minimo)


def _txt(v: float, decimales: int = 0) -> str:
    return fmt_entero(v) if int(decimales) <= 0 else fmt_dec(v, int(decimales))


def _vt(fila: Mapping[str, str], v: float, decimales: int = 0) -> str:
    """Valor para un motivo: un año se escribe sin separador de miles (2024 y no 2.024)."""
    return str(int(round(v))) if fila["unidad"] == "anio" else _txt(v, decimales)


def valor_con_unidad(unidad: str, decimales: int, v: float) -> str:
    """Una cifra del ledger con el formato de su unidad: un año sin separador de miles, un porcentaje con su signo y los decimales documentados, un conteo como entero y el resto
    (días, diagnósticos, veces) con los decimales documentados."""
    if unidad == "anio":
        return str(int(round(v)))
    if unidad == "porcentaje":
        return f"{fmt_dec(float(v), max(int(decimales), 0))} %"
    return _txt(v, decimales)


# ---------------------------------------------------------------------------------------------------------------------------------------- reglas del contexto
@dataclass(frozen=True)
class Reglas:
    """Umbrales que usa el ledger (todos de ``config/umbrales.toml``) y el nivel de la corrida."""

    z_umbral: float
    dif_relevante_pp: float
    z_95: float
    celda_minima: int
    anio_expuesto: int
    anios_resultado: tuple[int, ...]
    nivel: str

    @classmethod
    def desde_contexto(cls, ctx: Contexto) -> "Reglas":
        u = ctx.umbrales
        return cls(
            z_umbral=float(u["conciliacion_z"]),
            dif_relevante_pp=float(u["conciliacion_dif_relevante_pp"]),
            z_95=float(u["z_95"]),
            celda_minima=int(u["celda_minima"]),
            anio_expuesto=int(u.particion["anio_expuesto"]),
            anios_resultado=tuple(int(a) for a in u.particion["anios_resultado"]),
            nivel=ctx.etiqueta,
        )


# -------------------------------------------------------------------------------------------------------------------------------------------- carga y validación
def cargar_cifras(ruta: Path | str | None = None) -> pl.DataFrame:
    """Las cifras documentadas, todo como texto."""
    return config.leer_csv_texto(Path(ruta) if ruta else rutas.DIR_CONFIG / "cifras_documentadas.csv")


def validar_cifras(df: pl.DataFrame, esperadas: int | None = N_CIFRAS_ESPERADAS) -> list[str]:
    """Problemas de ``cifras_documentadas.csv`` (lista vacía si está bien): ids, tipos, números y claves; también las constantes por id."""
    obligatorias = {"id", "documento", "ubicacion", "descripcion", "valor", "unidad", "decimales", "tipo", "alcance", "requiere_resultado", "clave_eda", "nota"}
    faltan = obligatorias - set(df.columns)
    if faltan:
        return [f"faltan columnas en cifras_documentadas.csv: {sorted(faltan)}"]
    problemas: list[str] = []
    if esperadas is not None and df.height != esperadas:
        problemas.append(f"hay {df.height} cifras y se esperaban {esperadas}")
    ids = df["id"].to_list()
    if len(ids) != len(set(ids)):
        problemas.append("hay ids de cifras repetidos")
    for f in df.iter_rows(named=True):
        cid = f["id"]
        if f["tipo"] not in TIPOS:
            problemas.append(f"{cid}: tipo inválido {f['tipo']!r}")
        if f["requiere_resultado"] not in ("si", "no"):
            problemas.append(f"{cid}: requiere_resultado debe ser si o no ({f['requiere_resultado']!r})")
        for campo in ("valor", "decimales"):
            if campo == "valor" and f[campo] == VALOR_PEQUENO:
                if not _es_conteo_absoluto(f):
                    problemas.append(f"{cid}: {VALOR_PEQUENO} solo vale para un conteo absoluto")
                continue
            try:
                float(f[campo])
            except (TypeError, ValueError):
                problemas.append(f"{cid}: {campo} no es numérico ({f[campo]!r})")
        for k in (f["clave_eda"] or "").split(";"):
            if k != k.strip() or re.search(r"\s", k) or (k == "" and f["clave_eda"]):
                problemas.append(f"{cid}: alternativa de clave_eda mal formada {k!r}")
    por_id = {f["id"]: f for f in df.iter_rows(named=True)}
    for nombre, ids_const in (("NO_COMPARABLES", NO_COMPARABLES), ("SUBESTIMADAS_EN_EXTRACTO", SUBESTIMADAS_EN_EXTRACTO), ("COMPARACION_ESPECIAL", COMPARACION_ESPECIAL),
                              ("HIPOTESIS_ORIGEN_EXTRACTO", HIPOTESIS_ORIGEN_EXTRACTO)):
        for cid in ids_const:
            if cid not in por_id:
                problemas.append(f"{nombre}: la cifra {cid} no existe en cifras_documentadas.csv")
    for cid in HIPOTESIS_ORIGEN_EXTRACTO:
        if cid in por_id and por_id[cid]["documento"] != "Fase 1":
            problemas.append(f"HIPOTESIS_ORIGEN_EXTRACTO: {cid} debe ser una cifra de la Fase 1")
    for cid in COMPARACION_ESPECIAL:
        if cid in por_id and por_id[cid]["tipo"] not in ("absoluto", "estructura"):
            problemas.append(f"COMPARACION_ESPECIAL: {cid} debe ser un conteo (absoluto o estructura)")
    return problemas


def indexar_metricas(df: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """Diccionario ``clave -> métrica`` (valor, ic_bajo, ic_alto, tabla, nivel, alcance, nota) con los textos nulos como cadena vacía."""
    out: dict[str, dict[str, Any]] = {}
    for r in df.iter_rows(named=True):
        out[r["clave"]] = {
            "valor": None if r["valor"] is None else float(r["valor"]),
            "ic_bajo": None if r.get("ic_bajo") is None else float(r["ic_bajo"]),
            "ic_alto": None if r.get("ic_alto") is None else float(r["ic_alto"]),
            "tabla": r.get("tabla") or "",
            "nivel": r.get("nivel") or "",
            "alcance": r.get("alcance") or "",
            "nota": r.get("nota") or "",
        }
    return out


def metricas_disponibles(ctx: Contexto) -> dict[str, dict[str, Any]]:
    """Métricas de la corrida: las de memoria y, para lo que falte (por ejemplo con ``--solo T12``), las de ``tablas/metricas_eda.csv``. La memoria manda."""
    en_disco: dict[str, dict[str, Any]] = {}
    ruta = ctx.dir_tablas / "metricas_eda.csv"
    if ruta.is_file():
        textos = {c: pl.String for c in ("clave", "tarea", "tabla", "nivel", "alcance", "nota")}
        en_disco = indexar_metricas(pl.read_csv(ruta, infer_schema_length=100000, schema_overrides=textos))
    return {**en_disco, **indexar_metricas(ctx.metricas())}


def filas_leidas_por_anio(metricas: Mapping[str, Mapping[str, Any]]) -> dict[int, int]:
    """``volumen.filas_leidas.<año>`` de las métricas (tamaño de cada archivo leído)."""
    out: dict[int, int] = {}
    for clave, m in metricas.items():
        mt = re.fullmatch(r"volumen\.filas_leidas\.(\d{4})", clave)
        if mt and m.get("valor") is not None:
            out[int(mt.group(1))] = int(round(float(m["valor"])))
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------- una cifra
def _fila_base(fila: Mapping[str, str], reglas: Reglas) -> dict[str, Any]:
    out: dict[str, Any] = {c: None for c in COLUMNAS_LEDGER}
    out.update(
        id=fila["id"], documento=fila["documento"], ubicacion=fila["ubicacion"], descripcion=fila["descripcion"], tipo=fila["tipo"], unidad=fila["unidad"],
        alcance_doc=fila["alcance"], requiere_resultado=fila["requiere_resultado"], valor_doc=_valor_doc(fila), decimales=int(float(fila["decimales"])),
        clave_eda=fila["clave_eda"], nivel=reglas.nivel, magnitud="no aplica", nota=fila.get("nota") or "",
    )
    if _es_conteo_absoluto(fila) and out["valor_doc"] is not None:      # el conteo documentado queda protegido aunque la fila no llegue a compararse
        out["n_doc"] = int(round(out["valor_doc"]))
    return out


def _valor_doc(fila: Mapping[str, str]) -> float | None:
    return None if fila["valor"] == VALOR_PEQUENO else float(fila["valor"])


def _es_conteo_absoluto(fila: Mapping[str, str]) -> bool:
    return fila["tipo"] == "absoluto" and fila["unidad"] == "conteo"


def _cerrar(out: dict[str, Any], estado: str, motivo: str) -> dict[str, Any]:
    out["estado"] = estado
    out["motivo"] = motivo
    if estado not in ("CONFIRMA", "DISCREPA"):
        out["magnitud"] = "no aplica"
    return out


def _diferencias(out: dict[str, Any]) -> None:
    doc, med = out["valor_doc"], out["valor_medido"]
    out["dif_abs"] = abs(med - doc)
    out["dif_rel"] = None if doc == 0 else abs(med - doc) / abs(doc)


def _alcance_medido(alcance_metrica: str, alcance_doc: str, requiere_resultado: str, reglas: Reglas) -> str:
    """Alcance de la medición: el de la métrica; si viene vacío, el documentado, salvo que la cifra requiera el desenlace y su alcance incluya el año expuesto
    (entonces los años con desenlace, porque el cortafuegos garantiza que el año expuesto no aportó el desenlace)."""
    if alcance_metrica:
        return alcance_metrica
    if requiere_resultado == "si" and incluye_anio(alcance_doc, reglas.anio_expuesto):
        return etiqueta_alcance(reglas.anios_resultado)
    return alcance_doc


def _z_por_ic(medido: float, doc: float, ic_bajo: float | None, ic_alto: float | None, z_95: float) -> float | None:
    """z con el error estándar que sale del intervalo de la métrica: ``(ic_alto - ic_bajo) / (2 z_95)``; ``None`` si no hay intervalo o su ancho no es positivo."""
    if ic_bajo is None or ic_alto is None:
        return None
    se = (float(ic_alto) - float(ic_bajo)) / (2.0 * z_95)
    return (medido - doc) / se if se > 0 else None


def _estado_de_conciliacion(estado: str) -> str:
    return {"COINCIDE": "CONFIRMA", "DISCREPANCIA": "DISCREPA"}.get(estado, "NO VERIFICABLE")


def _lecturas(fila: Mapping[str, str], metricas: Mapping[str, Mapping[str, Any]], excepto: str | None = None) -> str:
    """Texto con el valor de las demás alternativas de ``clave_eda`` que tienen valor (para no perder una lectura alternativa)."""
    d = int(float(fila["decimales"]))
    partes = [f"{k} = {_txt(m['valor'], d)}" for k in alternativas_de(fila["clave_eda"]) if k != excepto and (m := metricas.get(k)) and m["valor"] is not None]
    return ("otras lecturas: " + "; ".join(partes)) if partes else ""


def _marcar_apoyo(out: dict[str, Any], fila: Mapping[str, str], clave: str, m: Mapping[str, Any], reglas: Reglas) -> None:
    """Rellena el valor medido de apoyo (clave, valor, intervalo, tabla y alcance) sin comparar."""
    out["clave_usada"] = clave
    out["valor_medido"] = m["valor"]
    out["ic_bajo"] = m["ic_bajo"]
    out["ic_alto"] = m["ic_alto"]
    out["tabla_origen"] = m["tabla"]
    out["alcance_medido"] = _alcance_medido(m["alcance"], fila["alcance"], fila["requiere_resultado"], reglas)
    if _es_conteo_absoluto(fila) and m["valor"] is not None:
        out["n_medido"] = int(round(m["valor"]))


def _por_cota(out: dict[str, Any], fila: Mapping[str, str], clave: str, cota: float, reglas: Reglas) -> dict[str, Any]:
    """Compatibilidad de un conteo con el intervalo [conteo de los años con desenlace, ese conteo más la cota del año expuesto] (bordes incluidos)."""
    lo, hi, doc = float(out["valor_medido"]), float(out["valor_medido"]) + float(cota), float(out["valor_doc"])
    dentro = lo <= doc <= hi
    _diferencias(out)                    # la diferencia es siempre |medido - documentado|; la compatibilidad se juzga contra el intervalo, que va en el motivo
    detalle = (f"intervalo compatible de {fmt_entero(lo)} (alcance {out['alcance_medido']}) a {fmt_entero(hi)} (más la cota de {reglas.anio_expuesto}); "
               f"la cifra documentada {'cae dentro' if dentro else 'cae fuera'}")
    if reglas.nivel == "extracto":
        detalle += "; hipótesis: la cifra de la Fase 1 proviene del extracto"
    return _cerrar(out, "CONFIRMA" if dentro else "DISCREPA", detalle)


def _comparar_conteo(out: dict[str, Any], fila: Mapping[str, str], clave: str, reglas: Reglas) -> dict[str, Any]:
    """Regla 7 para ``absoluto`` y ``estructura``."""
    cid = fila["id"]
    doc, med, d = float(out["valor_doc"]), float(out["valor_medido"]), int(out["decimales"])
    especial = COMPARACION_ESPECIAL.get(cid)
    if reglas.nivel == "completo":
        if especial == "mas_de":
            _diferencias(out)
            return _cerrar(out, "CONFIRMA" if med > doc else "DISCREPA", f"la cifra documentada dice más de {_txt(doc, d)} y se midió {_txt(med, d)}")
        if especial == "aproximado":
            cs = cifras_significativas(doc)
            _diferencias(out)
            igual = redondear_cifras(med, cs) == redondear_cifras(doc, cs)
            return _cerrar(out, "CONFIRMA" if igual else "DISCREPA", f"comparación aproximada a {cs} cifras significativas: se midió {_txt(med, d)}")
        _diferencias(out)
        igual = redondear(med, d) == redondear(doc, d)
        return _cerrar(out, "CONFIRMA" if igual else "DISCREPA", f"igualdad al redondeo de {d} decimales con los datos completos: se midió {_vt(fila, med, d)}")
    # extracto
    if especial == "mas_de":
        if med > doc:
            _diferencias(out)
            return _cerrar(out, "CONFIRMA", f"el extracto ya tiene más de {_txt(doc, d)} valores distintos ({_txt(med, d)}): la población los tiene")
        return _cerrar(out, "NO VERIFICABLE", MOTIVOS["mas_de_extracto"])
    if especial == "aproximado":
        return _cerrar(out, "NO VERIFICABLE", MOTIVOS["aproximada_poblacion"])
    if cid in HIPOTESIS_ORIGEN_EXTRACTO:
        _diferencias(out)
        igual = redondear(med, d) == redondear(doc, d)
        return _cerrar(out, "CONFIRMA" if igual else "DISCREPA",
                       f"hipótesis: la cifra de la Fase 1 proviene del extracto; el extracto tiene {_txt(med, d)}")
    if es_clave_de_distintos(clave):
        if med > doc:
            _diferencias(out)
            return _cerrar(out, "DISCREPA", f"el extracto ya tiene más valores distintos que los documentados ({_txt(med, d)}) y es una cota inferior de la población")
        if med == doc:
            _diferencias(out)
            return _cerrar(out, "CONFIRMA", "el extracto tiene los mismos valores distintos que los documentados; es una cota inferior de la población")
        return _cerrar(out, "NO VERIFICABLE", MOTIVOS["valores_distintos"])
    if fila["tipo"] == "estructura":
        _diferencias(out)
        igual = redondear(med, d) == redondear(doc, d)
        return _cerrar(out, "CONFIRMA" if igual else "DISCREPA", f"igualdad con lo que midió la suite de la estructura de los archivos o del diccionario: {_vt(fila, med, d)}")
    return _cerrar(out, "NO VERIFICABLE", MOTIVOS["poblacion_completa"])


def _nota_redondeo(estado: str, z: float | None, med: float, doc: float, decimales: int, z_umbral: float) -> str:
    """Nota para una DISCREPA que la tolerancia de redondeo de la cifra documentada absorbería (el estado no cambia: solo se dice).

    La cifra documentada está redondeada a ``decimales``, de modo que su valor verdadero puede estar a media unidad del último decimal. Con el mismo error estándar, la
    diferencia baja a ``|med - doc| - media unidad`` y z se reescala en esa proporción; si aun así queda dentro del umbral, el motivo lo dice. Vacío si no hay z (afirmación
    determinista) o si la discrepancia sobrevive a la tolerancia.
    """
    if estado != "DISCREPA" or z is None:
        return ""
    dif = abs(med - doc)
    if dif <= 0:
        return ""
    z_tol = abs(z) * max(dif - 0.5 * 10.0 ** (-int(decimales)), 0.0) / dif
    if z_tol > z_umbral:
        return ""
    return f"; con la tolerancia de redondeo de la cifra documentada (media unidad del último decimal) z = {fmt_dec(z_tol, 2)} y no discreparía"


def _comparar_proporcion(out: dict[str, Any], fila: Mapping[str, str], clave: str, reglas: Reglas, filas_por_anio: Mapping[int, int]) -> dict[str, Any]:
    """Regla 7 para ``proporcion``."""
    doc, med, d, unidad = float(out["valor_doc"]), float(out["valor_medido"]), int(out["decimales"]), fila["unidad"]
    pp = unidad == "porcentaje"

    def magnitud(estado: str) -> str:
        dif = abs(med - doc) if pp else None
        return descripcion.magnitud_discrepancia("DISCREPANCIA" if estado == "DISCREPA" else "COINCIDE", dif, reglas.dif_relevante_pp) if pp else "no aplica"

    if reglas.nivel == "completo":
        _diferencias(out)
        igual = redondear(med, d) == redondear(doc, d)
        estado = "CONFIRMA" if igual else "DISCREPA"
        out["magnitud"] = magnitud(estado)
        return _cerrar(out, estado, f"igualdad al redondeo de {d} decimales con los datos completos: se midió {_txt(med, max(d, 2))}")
    # extracto
    if clave in CLAVES_SIN_ERROR_ESTANDAR:
        return _cerrar(out, "NO VERIFICABLE", MOTIVOS["sin_error_estandar"] + f"{SEPARADOR_DETALLE}es una magnitud derivada de dos tasas")
    z = _z_por_ic(med, doc, out["ic_bajo"], out["ic_alto"], reglas.z_95)
    indicativo = "; extracto no aleatorio: resultado indicativo"
    if z is not None:
        out["z"] = z
        _diferencias(out)
        estado = "DISCREPA" if abs(z) > reglas.z_umbral else "CONFIRMA"
        out["magnitud"] = magnitud(estado)
        redondeo = _nota_redondeo(estado, z, med, doc, d, reglas.z_umbral)
        return _cerrar(out, estado, f"z = {fmt_dec(z, 2)} con el error estándar del intervalo de la métrica (umbral {fmt_dec(reglas.z_umbral, 1)}){redondeo}{indicativo}")
    if not pp:
        return _cerrar(out, "NO VERIFICABLE", MOTIVOS["sin_error_estandar"])
    n_med = n_del_alcance(out["alcance_medido"], filas_por_anio)
    if n_med is None or n_med <= 0:
        return _cerrar(out, "NO VERIFICABLE", MOTIVOS["sin_tamano"])
    n_doc = 10 ** (d + 2)                                 # conteos equivalentes: sin_doc / n_doc recupera la proporción documentada
    sin_doc = int(round(doc * 10**d))
    sin_med = int(round(med / 100.0 * n_med))
    estado_t4, z_t4 = descripcion.conciliar(n_doc - sin_doc, sin_doc, sin_med, n_med, "extracto", reglas.z_umbral)
    estado = _estado_de_conciliacion(estado_t4)
    if estado == "NO VERIFICABLE":
        return _cerrar(out, "NO VERIFICABLE", MOTIVOS["evento_raro"])
    out["z"] = z_t4
    _diferencias(out)
    out["magnitud"] = magnitud(estado)
    zt = f"z = {fmt_dec(z_t4, 2)}; " if z_t4 is not None else "afirmación determinista; "
    redondeo = _nota_redondeo(estado, z_t4, med, doc, d, reglas.z_umbral)
    return _cerrar(out, estado, f"{zt}error estándar binomial con n = {fmt_entero(n_med)}, cota superior del n real{redondeo}{indicativo}")


def _evaluar_con_metrica(
    base: dict[str, Any], fila: Mapping[str, str], clave: str, m: Mapping[str, Any], metricas: Mapping[str, Mapping[str, Any]], reglas: Reglas,
    filas_por_anio: Mapping[int, int],
) -> dict[str, Any]:
    out = dict(base)
    _marcar_apoyo(out, fila, clave, m, reglas)
    cid, tipo = fila["id"], fila["tipo"]
    doc, med = float(out["valor_doc"]), float(m["valor"])
    # privacidad: un conteo de 1 a 9 casos no se publica ni se compara
    if _es_conteo_absoluto(fila) and (_pequena(doc, reglas.celda_minima) or _pequena(med, reglas.celda_minima)):
        return _cerrar(out, "NO VERIFICABLE", MOTIVOS["pocos_casos"])
    # métricas que en un extracto se subestiman por construcción
    if reglas.nivel == "extracto" and cid in SUBESTIMADAS_EN_EXTRACTO:
        return _cerrar(out, "NO VERIFICABLE", SUBESTIMADAS_EN_EXTRACTO[cid])
    # año expuesto
    req = fila["requiere_resultado"] == "si"
    if req and incluye_anio(fila["alcance"], reglas.anio_expuesto):
        cota = metricas.get(f"{clave}.cota_sup_2024")
        cota_v = None if cota is None else cota["valor"]
        if tipo == "absoluto" and cota_v is not None:
            if reglas.nivel == "completo" or cid in HIPOTESIS_ORIGEN_EXTRACTO:
                return _por_cota(out, fila, clave, float(cota_v), reglas)
            return _cerrar(out, "NO VERIFICABLE", MOTIVOS["poblacion_completa"])
        return _cerrar(out, "NO VERIFICABLE", MOTIVOS["sin_cota"])
    # alcances distintos
    r_doc, r_med = anios_de_alcance(fila["alcance"]), anios_de_alcance(out["alcance_medido"])
    if r_doc is not None and r_med is not None and r_doc != r_med:
        return _cerrar(out, "NO VERIFICABLE", MOTIVOS["alcances_distintos"] + f"{SEPARADOR_DETALLE}se midió {out['alcance_medido']} y el documento dice {fila['alcance']}")
    if tipo in ("absoluto", "estructura"):
        return _comparar_conteo(out, fila, clave, reglas)
    if tipo == "proporcion":
        return _comparar_proporcion(out, fila, clave, reglas, filas_por_anio)
    return _cerrar(out, "NO VERIFICABLE", MOTIVOS["tipo_desconocido"])


def evaluar_cifra(
    fila: Mapping[str, str], metricas: Mapping[str, Mapping[str, Any]], reglas: Reglas, filas_por_anio: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Una fila del ledger para una cifra no aritmética. Función pura: ``fila`` es una fila de ``cifras_documentadas.csv`` (texto), ``metricas`` el diccionario de
    ``indexar_metricas`` y ``filas_por_anio`` el tamaño de cada archivo leído (para el error estándar binomial)."""
    filas_por_anio = filas_por_anio or {}
    base = _fila_base(fila, reglas)
    cid = fila["id"]
    alternativas = alternativas_de(fila["clave_eda"])
    existentes = [(k, metricas[k]) for k in alternativas if k in metricas]
    con_valor = [(k, m) for k, m in existentes if m["valor"] is not None]
    if fila["valor"] == VALOR_PEQUENO:                                  # conteo documentado de menos de diez casos: no se compara ni se publica
        if con_valor or existentes:
            _marcar_apoyo(base, fila, *(con_valor or existentes)[0], reglas)
            base.update(valor_medido=None, ic_bajo=None, ic_alto=None, n_medido=None)   # como antes: la fila no publica lo medido
        return _cerrar(base, "NO VERIFICABLE", MOTIVOS["pocos_casos"])
    if cid in NO_COMPARABLES:
        if existentes:
            _marcar_apoyo(base, fila, *(con_valor or existentes)[0], reglas)
        return _cerrar(base, "NO VERIFICABLE", NO_COMPARABLES[cid])
    if not con_valor:
        if existentes:
            k, m = existentes[0]
            _marcar_apoyo(base, fila, k, m, reglas)
            if "[suprimido" in m["nota"]:
                return _cerrar(base, "NO VERIFICABLE", MOTIVOS["suprimida"])
            return _cerrar(base, "NO VERIFICABLE", MOTIVOS["nula"] + (f"{SEPARADOR_DETALLE}{m['nota']}" if m["nota"] else ""))
        if fila["requiere_resultado"] == "si" and incluye_anio(fila["alcance"], reglas.anio_expuesto):
            return _cerrar(base, "NO VERIFICABLE", MOTIVOS["sin_cota"])
        return _cerrar(base, "NO VERIFICABLE", MOTIVOS["sin_clave"] if not alternativas else MOTIVOS["sin_metrica"])
    resultados = []
    for k, m in con_valor:
        r = _evaluar_con_metrica(base, fila, k, m, metricas, reglas, filas_por_anio)
        resultados.append(r)
        if r["estado"] == "CONFIRMA":
            break
    elegido = next((r for r in resultados if r["estado"] == "CONFIRMA"), None) or next((r for r in resultados if r["estado"] == "DISCREPA"), None) or resultados[0]
    otras = _lecturas(fila, metricas, excepto=elegido["clave_usada"])
    if otras:
        elegido["motivo"] = f"{elegido['motivo']}{SEPARADOR_DETALLE}{otras}"
    return elegido


# ------------------------------------------------------------------------------------------------------------------------------------------------ aritmética
def razones_fase2_fase1(cifras: pl.DataFrame) -> list[dict[str, Any]]:
    """Pares de cifras que comparten ``clave_eda`` con una medida de la Fase 2 y un estimado de la Fase 1: la razón es el valor de la Fase 2 sobre el de la Fase 1."""
    por_clave: dict[str, list[Mapping[str, str]]] = {}
    for r in cifras.iter_rows(named=True):
        if r["tipo"] != "aritmetica" and r["clave_eda"] and r["valor"] != VALOR_PEQUENO:
            por_clave.setdefault(r["clave_eda"], []).append(r)
    pares = []
    for clave in sorted(por_clave):
        f2 = [f for f in por_clave[clave] if f["documento"] == "Fase 2"]
        f1 = [f for f in por_clave[clave] if f["documento"] == "Fase 1"]
        if len(f2) == 1 and len(f1) == 1 and float(f1[0]["valor"]) > 0:
            v2, v1 = float(f2[0]["valor"]), float(f1[0]["valor"])
            pares.append({"clave": clave, "id_fase2": f2[0]["id"], "id_fase1": f1[0]["id"], "valor_fase2": v2, "valor_fase1": v1, "razon": v2 / v1})
    return pares


def evaluar_aritmetica(fila: Mapping[str, str], razones: Sequence[Mapping[str, Any]], reglas: Reglas, factor_muestreo: float | None = None) -> dict[str, Any]:
    """C128 y C129: el mínimo y el máximo de las razones recalculadas con las cifras documentadas, comparados con el valor documentado al redondeo de ``decimales``."""
    out = _fila_base(fila, reglas)
    out.update(nivel="documental", alcance_medido="documental", tabla_origen="cifras_documentadas.csv", clave_usada=fila["clave_eda"])
    extremo = ARITMETICA.get(fila["clave_eda"])
    if extremo is None:
        return _cerrar(out, "NO VERIFICABLE", MOTIVOS["clave_aritmetica"])
    if not razones:
        return _cerrar(out, "NO VERIFICABLE", MOTIVOS["sin_pares"])
    valores = [float(r["razon"]) for r in razones]
    medido = min(valores) if extremo == "min" else max(valores)
    out["valor_medido"] = medido
    _diferencias(out)
    igual = redondear(medido, out["decimales"]) == redondear(out["valor_doc"], out["decimales"])
    detalle = "razones de la Fase 2 sobre la Fase 1: " + "; ".join(f"{r['clave']} = {fmt_dec(float(r['razon']), 2)}" for r in razones)
    if factor_muestreo is not None:
        detalle += f"{SEPARADOR_DETALLE}inverso de la fracción de muestreo del extracto: {fmt_dec(factor_muestreo, 2)}"
    return _cerrar(out, "CONFIRMA" if igual else "DISCREPA", detalle)


# ------------------------------------------------------------------------------------------------------------------------------------------------ tabla completa
def factor_muestreo(cifras: pl.DataFrame, metricas: Mapping[str, Mapping[str, Any]]) -> float | None:
    """Inverso de la fracción de muestreo del extracto: episodios documentados de los seis años sobre las filas leídas (``volumen.filas_leidas.total``)."""
    m = metricas.get("volumen.filas_leidas.total")
    doc = cifras.filter((pl.col("clave_eda") == "volumen.filas_leidas.total") & (pl.col("documento") == "Fase 2"))
    if m is None or not m.get("valor") or doc.height != 1 or doc["valor"][0] == VALOR_PEQUENO:
        return None
    return float(doc["valor"][0]) / float(m["valor"])


def calcular(ctx: Contexto, cifras: pl.DataFrame | None = None, metricas: Mapping[str, Mapping[str, Any]] | None = None) -> pl.DataFrame:
    """El ledger completo (sin suprimir): una fila por cifra documentada, ordenadas por ``id``."""
    cifras = cargar_cifras() if cifras is None else cifras
    metricas = metricas_disponibles(ctx) if metricas is None else metricas
    reglas = Reglas.desde_contexto(ctx)
    filas = filas_leidas_por_anio(metricas)
    razones = razones_fase2_fase1(cifras)
    factor = factor_muestreo(cifras, metricas) if reglas.nivel == "extracto" else None
    resultado = []
    for f in cifras.sort("id").iter_rows(named=True):
        resultado.append(evaluar_aritmetica(f, razones, reglas, factor) if f["tipo"] == "aritmetica" else evaluar_cifra(f, metricas, reglas, filas))
    return pl.DataFrame(resultado, schema=ESQUEMA_LEDGER)


def motivo_base(motivo: str | None) -> str:
    """Parte fija del motivo (antes del detalle)."""
    return (motivo or "").split(SEPARADOR_DETALLE)[0]


def tabla_motivos(df: pl.DataFrame) -> pl.DataFrame:
    """Cifras NO VERIFICABLE agrupadas por la parte fija del motivo: ``motivo``, ``n_cifras`` e ``ids`` (ordenados)."""
    nv = df.filter(pl.col("estado") == "NO VERIFICABLE")
    grupos: dict[str, list[str]] = {}
    for cid, mot in zip(nv["id"].to_list(), nv["motivo"].to_list()):
        grupos.setdefault(motivo_base(mot), []).append(cid)
    filas = [{"motivo": m, "n_cifras": len(ids), "ids": "; ".join(sorted(ids))} for m, ids in grupos.items()]
    filas.sort(key=lambda r: (-r["n_cifras"], r["motivo"]))
    return pl.DataFrame(filas, schema={"motivo": pl.String, "n_cifras": pl.Int64, "ids": pl.String})


def resumen_ledger(df: pl.DataFrame) -> dict[str, int]:
    """Conteos del ledger para el resumen de evidencia."""
    c = lambda estado: int((df["estado"] == estado).sum())  # noqa: E731
    return {
        "ledger_n_cifras": df.height,
        "ledger_n_confirma": c("CONFIRMA"),
        "ledger_n_discrepa": c("DISCREPA"),
        "ledger_n_no_verificable": c("NO VERIFICABLE"),
        "ledger_n_discrepa_relevante": int(((df["estado"] == "DISCREPA") & (df["magnitud"] == "relevante")).sum()),
    }


def fusionar_resumen(ctx: Contexto, valores: Mapping[str, Any]) -> None:
    """Agrega columnas a la fila única de ``t25_resumen_evidencia.csv`` (la base de las cifras de los informes de evidencia y de cierre). Cada módulo de la Tarea 12 aporta
    las suyas y la tabla se reescribe completa."""
    previo: dict[str, Any] = {}
    if (ctx.dir_tablas / "t25_resumen_evidencia.csv").is_file():
        previo = ctx.leer_tabla("t25_resumen_evidencia.csv").row(0, named=True)
    previo.update(valores)
    ctx.escribir_tabla(
        "t25_resumen_evidencia.csv", pl.DataFrame([previo]), tarea=TAREA,
        descripcion="Resumen de la evidencia para decidir: conteos del ledger, del semáforo, de los supuestos y de los cambios sugeridos",
    )


def contradicciones_documentales(df: pl.DataFrame) -> list[dict[str, Any]]:
    """Pares de cifras del MISMO documento que se miden con la misma clave y el mismo alcance pero traen valores documentados distintos."""
    util = df.filter(pl.col("clave_usada").is_not_null() & (pl.col("clave_usada") != "") & pl.col("valor_doc").is_not_null())
    grupos: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for r in util.iter_rows(named=True):
        grupos.setdefault((r["clave_usada"], r["alcance_doc"], r["documento"]), []).append(r)
    out = []
    for (clave, alcance, documento), filas in sorted(grupos.items()):
        filas = sorted(filas, key=lambda r: r["id"])
        for i in range(len(filas)):
            for j in range(i + 1, len(filas)):
                if filas[i]["valor_doc"] != filas[j]["valor_doc"]:
                    out.append({"clave": clave, "alcance": alcance, "documento": documento, "id_a": filas[i]["id"], "id_b": filas[j]["id"],
                                "valor_a": filas[i]["valor_doc"], "valor_b": filas[j]["valor_doc"]})
    return out


def registrar_metricas(ctx: Contexto, df: pl.DataFrame) -> None:
    """Métricas ``ledger.*`` del cierre (cifras, no casos)."""
    r = resumen_ledger(df)
    for clave, valor in (("ledger.n_cifras", r["ledger_n_cifras"]), ("ledger.n_confirma", r["ledger_n_confirma"]), ("ledger.n_discrepa", r["ledger_n_discrepa"]),
                         ("ledger.n_no_verificable", r["ledger_n_no_verificable"]), ("ledger.n_discrepa_relevante", r["ledger_n_discrepa_relevante"])):
        ctx.registrar_metrica(clave, valor, tarea=TAREA, tabla="ledger_cifras.csv", nota="cifras o variables, no casos")


# ----------------------------------------------------------------------------------------------------------------------------------------------------- figura
COLORES_ESTADO = {"CONFIRMA": "#009E73", "DISCREPA": "#D55E00", "NO VERIFICABLE": "#999999"}
TEXTURAS_ESTADO = {"CONFIRMA": "", "DISCREPA": "//", "NO VERIFICABLE": ".."}
ETIQUETAS_ESTADO = {"CONFIRMA": "Confirma", "DISCREPA": "Discrepa", "NO VERIFICABLE": "No verificable"}


def figura_ledger(ctx: Contexto) -> None:
    """F12_02: conteo de los tres estados del ledger por documento de origen (barras agrupadas, el conteo escrito sobre cada barra y una textura por estado)."""
    t = ctx.leer_tabla("ledger_cifras.csv", columnas_texto=list(COLUMNAS_TEXTO_LEDGER))
    documentos = sorted(t["documento"].unique().to_list())
    x = np.arange(len(documentos))
    ancho = 0.26
    fig, ax = graficos.nueva_figura(figsize=(8.0, 4.8))
    maximo = 1
    for k, estado in enumerate(ESTADOS):
        valores = [int(((t["documento"] == d) & (t["estado"] == estado)).sum()) for d in documentos]
        maximo = max(maximo, max(valores))
        pos = x + (k - 1) * ancho
        ax.bar(pos, valores, width=ancho, color=COLORES_ESTADO[estado], hatch=TEXTURAS_ESTADO[estado], edgecolor="black", linewidth=0.4, label=ETIQUETAS_ESTADO[estado])
        for xi, v in zip(pos, valores):
            ax.text(xi, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(documentos)
    ax.set_ylim(0, maximo * 1.15)
    ax.set_xlabel("Documento de origen de la cifra")
    ax.set_ylabel("Cifras documentadas")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    nivel = "extracto" if ctx.etiqueta == "extracto" else "completo"
    ctx.figuras.guardar_figura(
        fig, id_figura="F12_02", tarea=12,
        titulo="Estado de las cifras documentadas frente a lo medido, por documento",
        descripcion_alt="Barras agrupadas por documento de origen (Fase 1 y Fase 2) con el número de cifras confirmadas, con discrepancia y no verificables; el número va escrito "
                        "sobre cada barra y cada estado tiene su color y su textura.",
        csv="ledger_cifras.csv", anios=ANIOS_TXT, decision="",
        extra_pie=("En un extracto, las comparaciones de proporciones son indicativas y los conteos de la población completa no son comparables." if nivel == "extracto" else ""),
    )


# ----------------------------------------------------------------------------------------------------------------------------------------------------- ejecución
def ejecutar(ctx: Contexto) -> pl.DataFrame:
    """Calcula y escribe ``ledger_cifras.csv`` (con la supresión de conteos de 1 a 9 casos) y ``ledger_motivos.csv``, registra las métricas ``ledger.*``, agrega los conteos
    al resumen de evidencia y dibuja F12_02. Devuelve el ledger sin suprimir (como los demás módulos)."""
    cifras = cargar_cifras()
    problemas = validar_cifras(cifras)
    if problemas:
        raise ValueError("cifras_documentadas.csv no es válido: " + "; ".join(problemas[:5]))
    df = calcular(ctx, cifras)
    ctx.escribir_tabla(
        "ledger_cifras.csv", df, tarea=TAREA, cols_n=COLS_N_LEDGER, dependientes=DEPENDIENTES_LEDGER,
        descripcion="Cada cifra documentada de las Fases 1 y 2 frente a lo medido por la suite, con su estado y su motivo",
    )
    ctx.escribir_tabla("ledger_motivos.csv", tabla_motivos(df), tarea=TAREA, descripcion="Cifras no verificables agrupadas por la parte fija de su motivo")
    registrar_metricas(ctx, df)
    fusionar_resumen(ctx, resumen_ledger(df))
    figura_ledger(ctx)
    return df


# ------------------------------------------------------------------------------------------------------------------------------------------------------ informe
N_SECCIONES = 2


def agregar_secciones(inf: Informe, ctx: Contexto, n0: int = 1) -> None:
    """Agrega al informe de evidencia (``N_SECCIONES = 2``): las reglas del ledger con F12_02 y las discrepancias con las cifras no verificables por motivo.

    Necesita ``ledger_cifras.csv``, ``ledger_motivos.csv`` y los conteos de ``t25_resumen_evidencia.csv`` (los escribe ``ejecutar``).
    """
    from .descripcion import Cifra_umbral

    r = lambda col: cifra(ctx, "t25_resumen_evidencia.csv", col)  # noqa: E731
    extracto = ctx.etiqueta == "extracto"
    led = ctx.leer_tabla("ledger_cifras.csv", columnas_texto=list(COLUMNAS_TEXTO_LEDGER))

    inf.seccion(f"{n0}. Cifras documentadas frente a lo medido")
    inf.parrafo(
        "El ledger compara {n_cifras} cifras de los informes de la Fase 1 y de la Fase 2 con lo que midió esta corrida. {n_confirma} quedan confirmadas, {n_discrepa} discrepan "
        "({n_rel} con una diferencia relevante) y {n_nover} no se pueden verificar con esta fuente. Una cifra solo se confirma o se contradice cuando la suite midió lo mismo, "
        "en el mismo alcance de años y con la misma definición; en cualquier otro caso queda no verificable y se dice por qué.",
        n_cifras=r("ledger_n_cifras"), n_confirma=r("ledger_n_confirma"), n_discrepa=r("ledger_n_discrepa"), n_rel=r("ledger_n_discrepa_relevante"),
        n_nover=r("ledger_n_no_verificable"),
    )
    inf.lista(
        [
            "Proporciones: la diferencia se contrasta con un error estándar (el del intervalo de la métrica o el binomial de la Tarea 4) y discrepa si supera {z|dec1} errores "
            "estándar; la magnitud se rotula relevante desde {pp|dec1} puntos porcentuales. La cifra documentada está redondeada: si media unidad de su último decimal bastara para "
            "que la discrepancia desapareciera, el motivo lo dice y el estado no cambia.",
            "Conteos absolutos: con los datos completos se exige igualdad al redondeo documentado. En un extracto un conteo de la población completa no es comparable, salvo la "
            "hipótesis de que la cifra de la Fase 1 proviene del extracto y la cota inferior de los conteos de valores distintos.",
            "Cifras que requieren el desenlace y abarcan 2024: el desenlace de 2024 nunca se lee. Un conteo se compara con el intervalo entre el conteo de 2019 a 2023 y ese "
            "conteo más la cota del año expuesto; sin cota queda no verificable. La diferencia que se publica es siempre la que hay entre lo medido y lo documentado, aunque la "
            "compatibilidad se juzgue con el intervalo.",
            "Conteos de uno a nueve casos: no se publican ni se comparan.",
        ],
        z=Cifra_umbral(ctx, "conciliacion_z"), pp=Cifra_umbral(ctx, "conciliacion_dif_relevante_pp"),
    )
    if extracto:
        inf.parrafo(
            "Con extractos el resultado de las proporciones es indicativo: el extracto es una fracción menor de los datos y no es una muestra aleatoria simple. Una "
            "discrepancia pequeña pero estadísticamente distinta de cero es esperable con cientos de miles de filas."
        )
    inf.figura("F12_02")

    inf.seccion(f"{n0 + 1}. Discrepancias y cifras no verificables")
    disc = led.filter(pl.col("estado") == "DISCREPA").with_columns(
        pl.when(pl.col("magnitud") == "relevante").then(0).otherwise(1).alias("_orden")
    ).sort(["_orden", "id"]).drop("_orden")
    # cada cifra con el formato de su unidad (un año sin separador, un conteo entero, un porcentaje con sus decimales documentados); las columnas conservan su nombre
    disc = disc.with_columns(*(
        pl.Series(col, [None if v is None else valor_con_unidad(u, d, v) for u, d, v in zip(disc["unidad"], disc["decimales"], disc[col])], dtype=pl.String)
        for col in ("valor_doc", "valor_medido")
    ))
    inf.parrafo(
        "Se listan las discrepancias, primero las de magnitud relevante. Una diferencia de tamaño despreciable puede ser estadísticamente distinta de cero sin importar en la "
        "práctica; una discrepancia de un conteo contra un extracto solo se afirma cuando el extracto es una cota inferior de la población."
    )
    inf.tabla(
        "ledger_cifras.csv", df=disc, columnas=["id", "documento", "descripcion", "valor_doc", "valor_medido", "dif_rel", "z", "magnitud", "motivo"],
        encabezados={"id": "Cifra", "documento": "Documento", "descripcion": "Descripción", "valor_doc": "Documentado", "valor_medido": "Medido", "dif_rel": "Diferencia relativa",
                     "z": "z", "magnitud": "Magnitud", "motivo": "Motivo"},
        formatos={"dif_rel": "prop1"}, max_filas=40, leyenda="Cifras documentadas con discrepancia",
    )
    inf.tabla(
        "ledger_motivos.csv", df=ctx.leer_tabla("ledger_motivos.csv", columnas_texto=["motivo", "ids"]), columnas=["motivo", "n_cifras", "ids"],
        encabezados={"motivo": "Motivo", "n_cifras": "Cifras", "ids": "Cifras afectadas"}, leyenda="Cifras no verificables por motivo",
    )


# FIN-ledger
