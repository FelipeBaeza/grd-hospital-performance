"""Definiciones reutilizadas de la Fase 2 (eda/panel.py, eda/hospitales.py y eda/catalogos.py), sin cambios de lógica.

Todo lo que aquí se llama «PENDIENTE DE REVISIÓN CLÍNICA» sigue dependiendo de la contraparte clínica (decisión 12).
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import polars as pl

from . import esquema

SIN_DATO = "(sin dato)"
SIN_MAPEO = "(sin mapeo)"
TEXTO_ENTRANTE = "OTROS HOSPITALES"   # un TIPO_PROCEDENCIA que contiene este texto es un traslado entrante
ACTIVIDAD_HOSP = "HOSPITALIZACIÓN"
ROMANOS = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X", 11: "XI", 12: "XII", 13: "XIII",
           14: "XIV", 15: "XV", 16: "XVI", 17: "XVII", 18: "XVIII", 19: "XIX", 20: "XX", 21: "XXI", 22: "XXII"}
CODIGOS_SALIDA_HOSPITAL = ("DERIV_HOSP_SERVICIO", "DERIV_HOSP_RED")
CODIGO_TIPOALTA = {
    "DOMICILIO": "DOMICILIO", "FALLECIDO": "FALLECIDO", "HOSPITALIZACIÓN DOMICILIARIA": "HOSP_DOMICILIARIA",
    "DERIVACIÓN OTRO HOSPITAL DEL SERVICIO": "DERIV_HOSP_SERVICIO", "ALTA VOLUNTARIA": "ALTA_VOLUNTARIA",
    "DERIVACIÓN OTRO HOSPITAL DE LA RED NACIONAL": "DERIV_HOSP_RED", "DERIVACIÓN A OTROS CENTROS (CÁRCEL, HOGAR)": "DERIV_OTROS_CENTROS",
    "FUGA DEL PACIENTE": "FUGA", "DERIVACIÓN INST. PRIVADA (COMPRA DE SERVICIOS)": "DERIV_PRIVADA_COMPRA",
    "DERIVACIÓN INST. PRIVADA (VOLUNTARIO)": "DERIV_PRIVADA_VOLUNTARIO", "NO IDENTIFICADA": "NO_IDENTIFICADA", "DESCONOCIDO": "DESCONOCIDO",
}
_TILDES = str.maketrans("ÓÉÍÁÚ", "OEIAU")


def normalizar_nombre(nombre: str) -> str:
    """Mayúsculas, sin tildes ni signos y con espacios colapsados."""
    sin = "".join(c for c in unicodedata.normalize("NFD", nombre.upper()) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^A-Z0-9]+", " ", sin).strip()


def _slug(texto: str) -> str:
    sin = "".join(c for c in unicodedata.normalize("NFD", texto.upper()) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^A-Z0-9]+", "_", sin).strip("_")


def codigo_tipoalta(etiqueta: str) -> str:
    return CODIGO_TIPOALTA.get(etiqueta) or _slug(etiqueta)


def _es_centinela_texto(texto: str) -> bool:
    return texto.strip().upper().translate(_TILDES) in esquema.CENTINELAS


def _vacio_o_centinela(texto: str) -> bool:
    return texto.strip() == "" or _es_centinela_texto(texto)


def por_distintos(valores: pl.Series, funcion: Callable[[str], Any], dtype: pl.DataType) -> pl.Series:
    """Aplica ``funcion`` solo a los valores distintos no nulos (mucho más barato que fila por fila)."""
    s = valores.cast(pl.String)
    mapa = {d: r for d in s.drop_nulls().unique().to_list() if (r := funcion(d)) is not None}
    if not mapa:
        return pl.Series(valores.name, [None] * len(s), dtype=dtype)
    return s.replace_strict(mapa, default=None, return_dtype=dtype).alias(valores.name)


def limpiar_categorica(valores: pl.Series, *, extra_sin_dato: Sequence[str] = ()) -> pl.Series:
    """Sin espacios extremos; nulo, vacío y centinela pasan al nivel «(sin dato)». Nunca se imputa."""
    s = valores.cast(pl.String).str.strip_chars()
    extra = {x.strip() for x in extra_sin_dato}
    malos = [d for d in s.drop_nulls().unique().to_list() if d == "" or _es_centinela_texto(d) or d in extra]
    if malos:
        s = s.to_frame("v").select(pl.when(pl.col("v").is_in(malos)).then(pl.lit(SIN_DATO)).otherwise(pl.col("v")).alias("v"))["v"]
    return s.fill_null(SIN_DATO).alias(valores.name)


_PROCEDENCIA_CRUDA: dict[str, str] = {
    "SERVICIO EMERGENCIA (DOMICILIO)": "URGENCIA", "APS URGENCIA (SAPU, SUR, SUC)": "URGENCIA",
    "OTROS HOSPITALES DE LA RED": "DERIVADO_OTRO_HOSPITAL", "OTROS HOSPITALES RED NACIONAL": "DERIVADO_OTRO_HOSPITAL",
    "CENTRO ESPECIALIDADES (CDT, CRS, CONSULTORIO ADOS. ESP)": "PROGRAMADO", "APS CONSULTORIO (CESFAM)": "PROGRAMADO",
    "CONSULTA PRIVADA": "PROGRAMADO", "PLAN DE RESOLUCIÓN LE": "PROGRAMADO", "CIRUGÍA MAYOR AMBULATORIA (CMA)": "PROGRAMADO",
    "CARDIOCIRUGÍA PAGO GRD": "PROGRAMADO",
}
MAPA_PROCEDENCIA_AGR = {normalizar_nombre(k): v for k, v in _PROCEDENCIA_CRUDA.items()}


def agrupar_procedencia(valores: pl.Series) -> pl.Series:
    """PROCEDENCIA_AGR: URGENCIA, DERIVADO_OTRO_HOSPITAL o PROGRAMADO; lo no listado, «(sin mapeo)»; nulo, vacío o centinela, «(sin dato)». PENDIENTE DE REVISIÓN CLÍNICA."""
    def grupo(t: str) -> str:
        return SIN_DATO if _vacio_o_centinela(t) else MAPA_PROCEDENCIA_AGR.get(normalizar_nombre(t), SIN_MAPEO)
    return por_distintos(valores, grupo, pl.String).fill_null(SIN_DATO)


def entrante_tipo_procedencia(valores: pl.Series) -> pl.Series:
    """1 si el TIPO_PROCEDENCIA indica traslado entrante (otro hospital); 0 en cualquier otro caso (nunca nulo)."""
    return por_distintos(valores, lambda t: int(TEXTO_ENTRANTE in normalizar_nombre(t)), pl.Int8).fill_null(0)


_RE_CUIDADO = tuple(re.compile(p) for p in (r"\bUCI\b", r"\bUTI\b", r"CUIDADOS? INTENSIVOS?", r"TRATAMIENTOS? INTERMEDIOS?"))


def nivel_cuidado_ingreso(valores: pl.Series) -> pl.Series:
    """1 si el servicio de ingreso es de cuidados especiales, 0 si está informado y no coincide, nulo si falta. PENDIENTE DE REVISIÓN CLÍNICA."""
    def marca(t: str) -> int | None:
        if _vacio_o_centinela(t):
            return None
        n = normalizar_nombre(t)
        return int(any(p.search(n) for p in _RE_CUIDADO))
    return por_distintos(valores, marca, pl.Int8)


def prefijo_comun(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def rol_de_etiqueta(etiqueta: str, roles: pl.DataFrame, fraccion_min: float = 0.9, ventaja_min: int = 5) -> tuple[str | None, str, str | None]:
    """Rol de mortalidad de una etiqueta de TIPOALTA (las etiquetas cortadas en el origen se asignan por prefijo)."""
    et = etiqueta.strip().upper()
    etiquetas = [(r["tipoalta"], r["rol_mortalidad"]) for r in roles.iter_rows(named=True)]
    for nombre, rol in etiquetas:
        if nombre.strip().upper() == et:
            return nombre, "exacta", rol
    puntajes = sorted(((prefijo_comun(et, n.strip().upper()), n, rol) for n, rol in etiquetas), reverse=True)
    if puntajes and puntajes[0][0] / max(len(et), 1) >= fraccion_min:
        segundo = puntajes[1][0] if len(puntajes) > 1 else 0
        if puntajes[0][0] - segundo >= ventaja_min:
            return puntajes[0][1], "por_prefijo", puntajes[0][2]
    return None, "sin_rol", None


def cargar_capitulos_cie10(ruta: str) -> dict[str, str]:
    """Categoría de 3 caracteres -> capítulo CIE-10 en romano (es el GRUPO_CLINICO)."""
    d = pl.read_csv(ruta, columns=["Código", "Capítulo"], infer_schema=False, encoding="utf8")
    d = d.with_columns(
        pl.col("Código").str.strip_chars().str.to_uppercase().str.replace_all(".", "", literal=True).alias("norm"),
        pl.col("Capítulo").str.extract(r"Cap\.(\d+)", 1).cast(pl.Int64, strict=False).alias("cap_num"),
    ).with_columns(
        pl.col("norm").str.slice(0, 3).alias("cat3"),
        pl.col("cap_num").replace_strict(ROMANOS, default=None, return_dtype=pl.String).alias("capitulo"),
    )
    return dict(d.filter(pl.col("capitulo").is_not_null()).select(["cat3", "capitulo"]).unique(subset=["cat3"], keep="first", maintain_order=True).iter_rows())
