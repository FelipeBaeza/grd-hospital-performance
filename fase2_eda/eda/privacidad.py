"""Escaneo automático de las salidas: ni identificadores, ni fechas completas, ni celdas pequeñas.

Chequeos (cada uno devuelve un conteo de violaciones; el cierre exige cero):
- ``encabezado_sensible``: una tabla tiene una columna llamada como un identificador o la fecha de nacimiento.
- ``fecha_completa``: aparece una fecha AAAA-MM-DD o DD-MM-AAAA real en tablas, informes o manifiesto.
- ``rut``: aparece un valor con estructura de RUT.
- ``numero_largo_en_texto``: una columna de texto contiene una secuencia de 8 o más dígitos (posible identificador).
- ``celda_pequena``: una columna de conteo registrada en el índice de tablas tiene valores entre 1 y 9.
Las máscaras de patrón (``9999-99-99``) no son fechas reales y no se marcan.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import polars as pl

from . import supresion

RE_FECHA_ISO = re.compile(r"(?<!\d)(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?!\d)")
RE_FECHA_DMY = re.compile(r"(?<!\d)(?:0[1-9]|[12]\d|3[01])-(?:0[1-9]|1[0-2])-(?:19|20)\d{2}(?!\d)")
# Se excluye la máscara de patrón compuesta solo por nueves (99999999-9), que no es un dato.
RE_RUT = re.compile(r"(?<![\dA-Za-z])(?!9{1,2}\.?9{3}\.?9{3}-9)\d{1,2}\.?\d{3}\.?\d{3}-[\dkK](?![\dA-Za-z])")
# Secuencias de 8 o más dígitos que no sean solo nueves (las máscaras de patrón, p. ej. 99999999, no son datos).
RE_NUM_LARGO = re.compile(r"(?<![\w.,])(?!9+(?![\w.,]))\d{8,}(?![\w.,])")

ENCABEZADOS_SENSIBLES = frozenset(
    {
        "CIP_ENCRIPTADO",
        "ID_BENEFICIARIO",
        "FECHA_NACIMIENTO",
        "MEDICOINTERV1_ENCRIPTADO",
        "MEDICOALTA_ENCRIPTADO",
        "ID_EPISODIO",
        "RUT",
    }
)


def _cuenta(patron: re.Pattern[str], texto: str) -> int:
    return len(patron.findall(texto))


def escanear_texto(texto: str) -> dict[str, int]:
    """Conteo de fechas completas, RUT y números largos en un texto libre (informe, manifiesto)."""
    return {
        "fecha_completa": _cuenta(RE_FECHA_ISO, texto) + _cuenta(RE_FECHA_DMY, texto),
        "rut": _cuenta(RE_RUT, texto),
        "numero_largo_en_texto": _cuenta(RE_NUM_LARGO, texto),
    }


def _cadenas_json(objeto) -> Iterator[str]:
    """Las claves y los valores de texto de un JSON ya leído (los números, booleanos y nulos no son texto)."""
    if isinstance(objeto, str):
        yield objeto
    elif isinstance(objeto, dict):
        for clave, valor in objeto.items():
            yield str(clave)
            yield from _cadenas_json(valor)
    elif isinstance(objeto, (list, tuple)):
        for valor in objeto:
            yield from _cadenas_json(valor)


def escanear_manifiesto(texto: str) -> dict[str, int]:
    """Conteo de fechas completas, RUT y números largos en las CADENAS de un manifiesto JSON.

    Un tamaño en bytes o un conteo de líneas (``"tamano_bytes": 24503085``) es un número del manifiesto y no un identificador: escaneado como texto plano, el último campo
    de cada objeto (sin coma que lo siga) se marcaba como número largo con los archivos reales de 8 o más dígitos. Si el texto no es JSON válido se escanea como texto.
    """
    try:
        datos = json.loads(texto)
    except json.JSONDecodeError:
        return escanear_texto(texto)
    total = {"fecha_completa": 0, "rut": 0, "numero_largo_en_texto": 0}
    for cadena in _cadenas_json(datos):
        for chequeo, n in escanear_texto(cadena).items():
            total[chequeo] += n
    return total


def escanear_tabla(ruta: Path, columnas_conteo: list[str], minimo: int = 10) -> dict[str, int]:
    """Escanea un CSV de tablas. Las columnas numéricas no se revisan en busca de números largos."""
    df = pl.read_csv(ruta, infer_schema_length=100000, truncate_ragged_lines=False)
    res = {
        "encabezado_sensible": sum(1 for c in df.columns if c.upper() in ENCABEZADOS_SENSIBLES),
        "fecha_completa": 0,
        "rut": 0,
        "numero_largo_en_texto": 0,
        "celda_pequena": supresion.contar_pequenas(df, columnas_conteo, minimo) if columnas_conteo else 0,
    }
    for col, tipo in df.schema.items():
        if tipo != pl.String:
            continue
        # Se recorre en Python porque las expresiones usan búsquedas hacia atrás (no soportadas por polars).
        for v in df[col].drop_nulls().to_list():
            conteo = escanear_texto(v)
            for k, n in conteo.items():
                res[k] += n
    return res


def escanear_salidas(dir_salida: Path, indice_tablas: pl.DataFrame, minimo: int = 10) -> pl.DataFrame:
    """Escanea tablas, informes y manifiesto de una carpeta de salidas. Devuelve una fila por chequeo y archivo."""
    cols_por_tabla: dict[str, list[str]] = {}
    if indice_tablas.height:
        for r in indice_tablas.iter_rows(named=True):
            cols = [c for c in (r["columnas_conteo"] or "").split(";") if c]
            cols_por_tabla[r["nombre"]] = cols
    filas = []
    for ruta in sorted((dir_salida / "tablas").glob("*.csv")):
        res = escanear_tabla(ruta, cols_por_tabla.get(ruta.name, []), minimo)
        for chequeo, n in res.items():
            filas.append({"archivo": f"tablas/{ruta.name}", "chequeo": chequeo, "n_violaciones": int(n)})
    for ruta in sorted((dir_salida / "informes").glob("*.md")):
        for chequeo, n in escanear_texto(ruta.read_text(encoding="utf8")).items():
            filas.append({"archivo": f"informes/{ruta.name}", "chequeo": chequeo, "n_violaciones": int(n)})
    manifiesto = dir_salida / "manifiesto.json"
    if manifiesto.is_file():
        for chequeo, n in escanear_manifiesto(manifiesto.read_text(encoding="utf8")).items():
            filas.append({"archivo": "manifiesto.json", "chequeo": chequeo, "n_violaciones": int(n)})
    return pl.DataFrame(
        filas, schema={"archivo": pl.String, "chequeo": pl.String, "n_violaciones": pl.Int64}
    )


def total_violaciones(escaneo: pl.DataFrame) -> int:
    return int(escaneo["n_violaciones"].sum()) if escaneo.height else 0
