"""Rutas de la suite. Todo es relativo a la ubicación de este paquete."""

from __future__ import annotations

from pathlib import Path

RAIZ_EDA: Path = Path(__file__).resolve().parents[1]
RAIZ_PROYECTO: Path = RAIZ_EDA.parent
DIR_CONFIG: Path = RAIZ_EDA / "config"
DIR_SALIDAS: Path = RAIZ_EDA / "salidas"
DIR_DOCS: Path = RAIZ_PROYECTO / "docs"
DIR_FASE1: Path = DIR_DOCS / "fase1_comprension_dominio"
DIR_FASE2: Path = DIR_DOCS / "fase2_comprension_datos"

FUENTES_VALIDAS = ("extracto", "completo")


def dir_salida(fuente: str, base: Path | None = None) -> Path:
    """Carpeta de salidas de una fuente (``extracto`` o ``completo``)."""
    if fuente not in FUENTES_VALIDAS:
        raise ValueError(f"fuente debe ser una de {FUENTES_VALIDAS}, no {fuente!r}")
    return (base or DIR_SALIDAS) / fuente


def subdirs_salida(raiz: Path) -> dict[str, Path]:
    """Subcarpetas estándar de una carpeta de salidas (se crean si no existen)."""
    sub = {n: raiz / n for n in ("tablas", "figuras", "informes")}
    for p in sub.values():
        p.mkdir(parents=True, exist_ok=True)
    return sub
