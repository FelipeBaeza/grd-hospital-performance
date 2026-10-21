"""Escáner de privacidad: sin valores con forma de RUT, sin fechas completas y sin conteos de 1 a 9 casos (ni reconstruibles desde otras columnas).
Es el mismo criterio de la Fase 5; se usa para validar los datos que entran a la base y los registros del servidor."""
from __future__ import annotations

import csv
import re
from pathlib import Path

RE_RUT = re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]\b")
RE_FECHA = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4})\b")
EXTENSIONES = {".csv", ".json", ".md", ".sql", ".txt"}
COLUMNAS_CONTEO = re.compile(r"^(n|n_.*|.*_n|casos|conteo|filas.*|hospitales.*)$", re.I)


def escanear_texto(texto: str) -> dict[str, int]:
    return {"rut": len(RE_RUT.findall(texto)), "fechas": len(RE_FECHA.findall(texto))}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def celdas_pequenas_csv(ruta: Path, minimo: int = 10) -> int:
    """Conteos de 1 a ``minimo``-1 en columnas de conteo, observados ocultos con O/E publicado y deciles de mortalidad con menos de ``minimo`` muertes."""
    n = 0
    with open(ruta, newline="", encoding="utf8") as f:
        filas = list(csv.DictReader(f))
    if not filas:
        return 0
    cab = list(filas[0].keys())
    for fila in filas:
        for c in cab:
            if COLUMNAS_CONTEO.match(c) and (v := _num(fila[c])) is not None and v == int(v) and 1 <= int(v) < minimo:
                n += 1
        if "observado" in fila and "oe" in fila:
            obs = _num(fila["observado"])
            if obs is None and fila["oe"] not in ("", None):                   # observado oculto pero O/E visible: se reconstruye observado = O/E × esperado
                n += 1
            elif obs is not None and 0 < obs < minimo and "esperado" in fila and ruta.name.startswith(("indicador", "oe_hospital", "subgrupo")) and "mortalidad" in str(fila.get("tarea", ruta.name)):
                n += 1
        if {"decil", "n", "observado"} <= set(cab) and str(fila.get("tarea", ruta.name)).startswith("mortalidad"):
            o, m = _num(fila["observado"]), _num(fila["n"])
            if o is not None and m is not None and round(o * m) < minimo:
                n += 1
    return n


def escanear_carpeta(raiz: Path | str, minimo: int = 10) -> list[dict]:
    hallazgos = []
    for ruta in sorted(Path(raiz).rglob("*")):
        if not ruta.is_file() or ruta.suffix.lower() not in EXTENSIONES:
            continue
        texto = ruta.read_text(encoding="utf8", errors="replace")
        r = escanear_texto(texto)
        pequenas = celdas_pequenas_csv(ruta, minimo) if ruta.suffix.lower() == ".csv" else 0
        if r["rut"] or r["fechas"] or pequenas:
            hallazgos.append({"archivo": str(ruta.relative_to(raiz)), "rut": r["rut"], "fechas": r["fechas"], "celdas_pequenas": pequenas})
    return hallazgos
