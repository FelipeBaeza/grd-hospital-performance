"""Catálogos de referencia de la raíz del proyecto y su lectura segura.

- ``CIE-10.csv`` (códigos CIE-10 con capítulo), ``CIE-9 .csv`` (procedimientos CIE-9-MC; el nombre lleva un espacio).
- ``GRDs_T10901.xlsx`` (hoja ``GRD Glosa``, encabezado en la fila 2) e ``IR-GRD.csv`` (pierde los ceros a la izquierda).
- ``Hospitales.csv``, ``Severidad GRD.csv``, ``Mortalidad GRD.csv`` y ``Tipo de Pabellon.csv`` (dos columnas con coma final).
Los códigos siempre se leen como TEXTO. Del archivo de CIE-10 (grande) solo se leen las columnas necesarias.
"""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from . import config, rutas

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
ROMANOS = {
    1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X", 11: "XI",
    12: "XII", 13: "XIII", 14: "XIV", 15: "XV", 16: "XVI", 17: "XVII", 18: "XVIII", 19: "XIX", 20: "XX",
    21: "XXI", 22: "XXII",
}
# Tipos de GRD según el tercer dígito (oficio FONASA 23364/2026): 0 error, 1 procedimientos hospitalizados,
# 2 y 3 procedimientos ambulatorios, 4 médicos hospitalizados, 5 médicos ambulatorios, 6 partos hospitalizados,
# 7 partos ambulatorios, 8 recién nacidos hospitalizados, 9 recién nacidos ambulatorios.
TIPO_GRD_HOSPITALIZADO = frozenset("1468")
TIPO_GRD_AMBULATORIO = frozenset("23579")
ACTIVIDAD_HOSPITALIZACION = frozenset({"HOSPITALIZACIÓN"})
ACTIVIDAD_AMBULATORIA = frozenset({"CIRUGÍA MAYOR AMBULATORIA (CMA)", "HOSPITALIZACIÓN DIURNA"})
PRIVADO_POR_NOMBRE = ("clínica", "clinica", "sanatorio")


def normalizar_cie(codigo: str) -> str:
    """Mayúsculas, sin espacios extremos y sin puntos."""
    return codigo.strip().upper().replace(".", "")


# ------------------------------------------------------------------------------ xlsx
def leer_hoja_xlsx(ruta: Path | str, indice_hoja: int = 1) -> list[dict[str, str | None]]:
    """Lee una hoja de un .xlsx con la biblioteca estándar. Devuelve una lista de filas {columna: valor}."""
    with zipfile.ZipFile(ruta) as z:
        compartidas: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
                compartidas.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
        hoja = ET.fromstring(z.read(f"xl/worksheets/sheet{indice_hoja}.xml"))
    filas = []
    for r in hoja.find("m:sheetData", _NS).findall("m:row", _NS):
        d: dict[str, str | None] = {}
        for c in r.findall("m:c", _NS):
            col = re.match(r"[A-Z]+", c.get("r")).group()
            v = c.find("m:v", _NS)
            tipo = c.get("t")
            if tipo == "inlineStr":
                is_ = c.find("m:is", _NS)
                d[col] = "".join(t.text or "" for t in is_.iter(f"{{{_NS['m']}}}t")) if is_ is not None else None
            elif v is None:
                d[col] = None
            elif tipo == "s":
                d[col] = compartidas[int(v.text)]
            else:
                d[col] = v.text
        filas.append(d)
    return filas


def _pad_grd(valor: str | None) -> str | None:
    if valor is None:
        return None
    v = str(valor).strip()
    if v.endswith(".0"):
        v = v[:-2]
    return v.zfill(6) if v.isdigit() else v


def cargar_grd_xlsx(ruta: Path | str, fila_encabezado: int = 2) -> pl.DataFrame:
    """Catálogo GRD (GRDs_T10901.xlsx): código de 6 dígitos como texto y glosa."""
    filas = leer_hoja_xlsx(ruta)
    datos = filas[fila_encabezado:]  # las filas 1 y 2 son título y encabezado
    cod = [_pad_grd(f.get("A")) for f in datos]
    et = [f.get("B") for f in datos]
    df = pl.DataFrame({"cod_grd": cod, "etiqueta": et}, schema={"cod_grd": pl.String, "etiqueta": pl.String}).filter(pl.col("cod_grd").is_not_null())
    return _con_estructura_grd(df)


def _con_estructura_grd(df: pl.DataFrame) -> pl.DataFrame:
    c = pl.col("cod_grd")
    return df.with_columns(
        c.str.slice(0, 2).alias("mdc"),
        c.str.slice(2, 1).alias("tipo_grd"),
        c.str.slice(0, 5).alias("grd_base"),
        c.str.slice(5, 1).alias("severidad_digito"),
        pl.col("etiqueta").str.slice(0, 2).alias("prefijo_etiqueta"),
    )


def cargar_dos_columnas(ruta: Path | str) -> pl.DataFrame:
    """CSV de dos columnas con encabezado y coma final (Hospitales, Severidad, Mortalidad, Pabellón, IR-GRD)."""
    filas = []
    with open(ruta, encoding="utf-8", newline="") as f:
        lector = csv.reader(f)
        next(lector, None)
        for fila in lector:
            if fila and fila[0].strip():
                filas.append((fila[0].strip(), fila[1].strip() if len(fila) > 1 else ""))
    return pl.DataFrame(filas, schema={"codigo": pl.String, "etiqueta": pl.String}, orient="row")


def cargar_ir_grd_csv(ruta: Path | str) -> pl.DataFrame:
    """IR-GRD.csv pierde los ceros a la izquierda: se rellena a seis dígitos."""
    d = cargar_dos_columnas(ruta).rename({"codigo": "cod_grd"})
    d = d.with_columns(pl.col("cod_grd").map_elements(_pad_grd, return_dtype=pl.String))
    return _con_estructura_grd(d)


def cargar_hospitales(ruta: Path | str) -> pl.DataFrame:
    """Hospitales.csv: código, nombre y tipo inferido del nombre (no hay nivel de complejidad)."""
    d = cargar_dos_columnas(ruta).rename({"codigo": "cod_hospital", "etiqueta": "nombre"})
    return d.with_columns(
        pl.col("nombre").map_elements(lambda n: "privado_probable" if n.lower().startswith(PRIVADO_POR_NOMBRE) else "publico_probable", return_dtype=pl.String).alias("tipo_por_nombre")
    )


def cargar_cie10(ruta: Path | str) -> pl.DataFrame:
    """CIE-10: código normalizado (sin punto), categoría de 3 caracteres y capítulo en romano."""
    d = pl.read_csv(ruta, columns=["Código", "Capítulo"], infer_schema=False, encoding="utf8")
    d = d.rename({"Código": "codigo", "Capítulo": "capitulo_txt"})
    return d.with_columns(
        pl.col("codigo").str.strip_chars().str.to_uppercase().str.replace_all(".", "", literal=True).alias("norm"),
        pl.col("capitulo_txt").str.extract(r"Cap\.(\d+)", 1).cast(pl.Int64, strict=False).alias("cap_num"),
    ).with_columns(
        pl.col("norm").str.slice(0, 3).alias("cat3"),
        pl.col("cap_num").replace_strict(ROMANOS, default=None, return_dtype=pl.String).alias("capitulo"),
    )


def cargar_cie9(ruta: Path | str) -> pl.DataFrame:
    d = pl.read_csv(ruta, columns=["Código", "Descripción"], infer_schema=False, encoding="utf8")
    d = d.rename({"Código": "codigo", "Descripción": "descripcion"})
    return d.with_columns(
        pl.col("codigo").str.strip_chars().str.replace_all(".", "", literal=True).alias("norm"),
    ).with_columns(pl.col("norm").str.slice(0, 2).alias("cat2"))


@dataclass
class Catalogos:
    cie10: pl.DataFrame
    cie10_exacto: frozenset[str]
    cie10_cat3: frozenset[str]
    cie10_capitulo_por_cat3: dict[str, str]
    cie9: pl.DataFrame
    cie9_exacto: frozenset[str]
    cie9_cat2: frozenset[str]
    grd_xlsx: pl.DataFrame
    grd_xlsx_set: frozenset[str]
    grd_csv: pl.DataFrame
    grd_csv_set: frozenset[str]
    hospitales: pl.DataFrame
    hospitales_set: frozenset[str]
    pabellon: frozenset[str]
    severidad: frozenset[str]
    mortalidad: frozenset[str]
    roles_tipoalta: pl.DataFrame = field(default_factory=pl.DataFrame)


def cargar_catalogos(raiz: Path | None = None) -> Catalogos:
    raiz = raiz or rutas.RAIZ_PROYECTO
    cie10 = cargar_cie10(raiz / "CIE-10.csv")
    cap = dict(cie10.filter(pl.col("capitulo").is_not_null()).select(["cat3", "capitulo"]).unique(subset=["cat3"], keep="first", maintain_order=True).iter_rows())
    cie9 = cargar_cie9(raiz / "CIE-9 .csv")
    xlsx = cargar_grd_xlsx(raiz / "GRDs_T10901.xlsx")
    ircsv = cargar_ir_grd_csv(raiz / "IR-GRD.csv")
    hosp = cargar_hospitales(raiz / "Hospitales.csv")
    return Catalogos(
        cie10=cie10,
        cie10_exacto=frozenset(cie10["norm"].to_list()),
        cie10_cat3=frozenset(cie10["cat3"].to_list()),
        cie10_capitulo_por_cat3=cap,
        cie9=cie9,
        cie9_exacto=frozenset(cie9["norm"].to_list()),
        cie9_cat2=frozenset(cie9["cat2"].to_list()),
        grd_xlsx=xlsx,
        grd_xlsx_set=frozenset(xlsx["cod_grd"].to_list()),
        grd_csv=ircsv,
        grd_csv_set=frozenset(ircsv["cod_grd"].to_list()),
        hospitales=hosp,
        hospitales_set=frozenset(hosp["cod_hospital"].to_list()),
        pabellon=frozenset(cargar_dos_columnas(raiz / "Tipo de Pabellon.csv")["codigo"].to_list()),
        severidad=frozenset(cargar_dos_columnas(raiz / "Severidad GRD.csv")["codigo"].to_list()),
        mortalidad=frozenset(cargar_dos_columnas(raiz / "Mortalidad GRD.csv")["codigo"].to_list()),
        roles_tipoalta=config.cargar_tipoalta_roles(),
    )


# ------------------------------------------------------------------- clasificadores de valores
def _es_centinela_txt(v: str) -> bool:
    import unicodedata

    from .esquema import CENTINELAS

    sin_tilde = "".join(c for c in unicodedata.normalize("NFD", v.strip().upper()) if unicodedata.category(c) != "Mn")
    return sin_tilde in CENTINELAS


def clasificar_cie10(valor: str, exacto: frozenset[str], cat3: frozenset[str]) -> str:
    if _es_centinela_txt(valor):
        return "centinela"
    n = normalizar_cie(valor)
    if n in exacto:
        return "valido"
    if n[:3] in cat3:
        return "solo_categoria"
    return "invalido"


def clasificar_cie9(valor: str, exacto: frozenset[str], cat2: frozenset[str]) -> str:
    if _es_centinela_txt(valor):
        return "centinela"
    n = normalizar_cie(valor)
    if n in exacto:
        return "valido"
    if n[:2] in cat2:
        return "solo_categoria"
    return "invalido"


def clasificar_grd(valor: str, xlsx: frozenset[str], ircsv: frozenset[str]) -> str:
    v = valor.strip()
    if _es_centinela_txt(v):
        return "centinela"
    if not re.fullmatch(r"\d{6}", v):
        return "formato_invalido"
    if v[:2] in ("98", "99"):
        return "no_agrupable_98_99"
    if v in xlsx:
        return "valido"
    if v in ircsv:
        return "solo_en_ir_grd_csv"
    return "no_en_catalogo"


def clasificar_en_conjunto(valor: str, validos: frozenset[str]) -> str:
    v = valor.strip()
    if _es_centinela_txt(v):
        return "centinela"
    return "valido" if v in validos else "invalido"


def clasificar_hospital(valor: str, hospitales: pl.DataFrame) -> str:
    v = valor.strip()
    fila = hospitales.filter(pl.col("cod_hospital") == v)
    if fila.height == 0:
        return "fuera_de_catalogo"
    return "en_catalogo_" + fila["tipo_por_nombre"][0]


def prefijo_comun(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def rol_de_etiqueta(etiqueta: str, roles: pl.DataFrame, fraccion_min: float = 0.9, ventaja_min: int = 5) -> tuple[str | None, str, str | None]:
    """Rol de mortalidad y estadía de una etiqueta de TIPOALTA.

    Devuelve (etiqueta del archivo de roles o None, método, rol_mortalidad). Método: ``exacta``, ``por_prefijo`` (etiqueta cortada)
    o ``sin_rol``. La etiqueta cortada coincide con una del archivo de roles en al menos ``fraccion_min`` de su largo y con ventaja
    clara sobre la segunda candidata.
    """
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
