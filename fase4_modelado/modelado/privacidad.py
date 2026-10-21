"""Escaneo de privacidad de las salidas (criterio DT04 de la Fase 2): sin valores con forma de RUT, sin fechas completas y sin celdas de 1 a 9 casos."""
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


def celdas_pequenas_csv(ruta: Path, minimo: int = 10, excluir: tuple[str, ...] = ()) -> int:
    """Cuenta celdas enteras entre 1 y minimo-1 en las columnas de conteo de un CSV (se excluyen las columnas que no son de personas)."""
    n = 0
    with open(ruta, newline="", encoding="utf8") as f:
        lector = csv.reader(f)
        try:
            cab = next(lector)
        except StopIteration:
            return 0
        idx = [i for i, c in enumerate(cab) if COLUMNAS_CONTEO.match(c) and c not in excluir]
        for fila in lector:
            for i in idx:
                if i < len(fila) and fila[i].isdigit() and 1 <= int(fila[i]) < minimo:
                    n += 1
    return n


def escanear_carpeta(raiz: Path | str, minimo: int = 10, excluir_archivos: tuple[str, ...] = ()) -> list[dict]:
    """Una fila por archivo con hallazgos de RUT, fechas completas o celdas pequeñas."""
    hallazgos = []
    for ruta in sorted(Path(raiz).rglob("*")):
        if not ruta.is_file() or ruta.suffix.lower() not in EXTENSIONES or ruta.name in excluir_archivos:
            continue
        texto = ruta.read_text(encoding="utf8", errors="replace")
        r = escanear_texto(texto)
        pequenas = celdas_pequenas_csv(ruta, minimo) if ruta.suffix.lower() == ".csv" else 0
        if r["rut"] or r["fechas"] or pequenas:
            hallazgos.append({"archivo": str(ruta.relative_to(raiz)), "rut": r["rut"], "fechas": r["fechas"], "celdas_pequenas": pequenas})
    return hallazgos
