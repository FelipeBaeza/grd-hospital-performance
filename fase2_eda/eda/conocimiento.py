"""Tabla de conocimiento clínico (config/conocimiento_clinico.csv): qué se espera de cada variable y con qué respaldo.

Regla de la Fase 2: el conocimiento clínico solo se cita de fuentes reconocidas (Elixhauser 1998, Quan 2005, van Walraven 2009
y Spiegelhalter 2005). Todo lo demás queda marcado como ``pendiente de revisión clínica (decisión 12)`` y se acompaña de la
tabla de la exploración que muestra lo observado en los datos. La tabla es una primera versión para el revisor clínico: no
sustituye su criterio.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from . import config, elixhauser, rutas

COLUMNAS = ("variable", "grupo", "racional", "direccion_esperada", "referencia", "certeza", "validar_con", "evidencia_eda")
CERTEZA_VERIFICADA = "verificada en fuente"
CERTEZA_PENDIENTE = "pendiente de revisión clínica (decisión 12)"
CERTEZAS = (CERTEZA_VERIFICADA, CERTEZA_PENDIENTE)
SIN_REFERENCIA = "sin referencia verificada"
FUENTES_RECONOCIDAS = (
    "Elixhauser et al. 1998",
    "Quan et al. 2005",
    "van Walraven et al. 2009",
    "Spiegelhalter 2005",
)


def cargar(ruta: Path | str | None = None) -> pl.DataFrame:
    return config.leer_csv_texto(Path(ruta) if ruta else rutas.DIR_CONFIG / "conocimiento_clinico.csv")


def referencias_de(texto: str) -> list[str]:
    return [t.strip() for t in texto.split(";") if t.strip()]


def validar(df: pl.DataFrame) -> list[str]:
    """Problemas de la tabla (lista vacía si cumple las reglas)."""
    problemas: list[str] = []
    if tuple(df.columns) != COLUMNAS:
        return [f"las columnas deben ser {COLUMNAS}"]
    if df["variable"].n_unique() != df.height or (df["variable"] == "").any():
        problemas.append("la variable debe ser única y no vacía")
    faltan = sorted(set(elixhauser.IDS) - set(df["variable"].to_list()))
    if faltan:
        problemas.append(f"faltan categorías de Elixhauser: {faltan}")
    for fila in df.iter_rows(named=True):
        v = fila["variable"]
        for campo in ("grupo", "racional", "direccion_esperada", "validar_con"):
            if not fila[campo].strip():
                problemas.append(f"{v}: {campo} vacío")
        if fila["certeza"] not in CERTEZAS:
            problemas.append(f"{v}: certeza inválida {fila['certeza']!r}")
        refs = referencias_de(fila["referencia"])
        if not refs:
            problemas.append(f"{v}: referencia vacía (use {SIN_REFERENCIA!r})")
        elif refs == [SIN_REFERENCIA]:
            if fila["certeza"] != CERTEZA_PENDIENTE:
                problemas.append(f"{v}: sin referencia verificada exige certeza pendiente de revisión clínica")
        else:
            ajenas = [r for r in refs if r not in FUENTES_RECONOCIDAS]
            if ajenas:
                problemas.append(f"{v}: fuente no reconocida {ajenas}")
    return problemas
