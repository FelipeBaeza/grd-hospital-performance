"""Conversión de tipos a partir de las columnas de texto del lector."""

from __future__ import annotations

import polars as pl

_FORMATO_ISO = "%Y-%m-%d"
_FORMATO_DMY = "%d-%m-%Y"


def a_fecha(expr: pl.Expr) -> pl.Expr:
    """Texto a fecha aceptando AAAA-MM-DD y DD-MM-AAAA. Lo que no se pueda convertir queda nulo."""
    s = expr.str.strip_chars()
    return pl.coalesce(
        s.str.strptime(pl.Date, _FORMATO_ISO, strict=False, exact=True),
        s.str.strptime(pl.Date, _FORMATO_DMY, strict=False, exact=True),
    )


def edad_anios(nacimiento: pl.Expr, ingreso: pl.Expr) -> pl.Expr:
    """Edad en años cumplidos al ingreso (entera). Puede resultar negativa si el nacimiento es posterior."""
    # mes y día son Int8: se convierten a Int32 antes de multiplicar para evitar desbordes
    md_ing = ingreso.dt.month().cast(pl.Int32) * 100 + ingreso.dt.day().cast(pl.Int32)
    md_nac = nacimiento.dt.month().cast(pl.Int32) * 100 + nacimiento.dt.day().cast(pl.Int32)
    ant = md_ing < md_nac
    return (ingreso.dt.year().cast(pl.Int32) - nacimiento.dt.year().cast(pl.Int32) - ant.cast(pl.Int32)).cast(pl.Int32)


def estancia_dias(ingreso: pl.Expr, alta: pl.Expr) -> pl.Expr:
    """Días entre el ingreso y el alta (puede ser negativo si hay un error de registro)."""
    return (alta - ingreso).dt.total_days().cast(pl.Int32)


def a_decimal(expr: pl.Expr) -> pl.Expr:
    """Texto con coma o punto decimal (incluye ',5' sin cero inicial) a Float64."""
    return expr.str.strip_chars().str.replace(",", ".", literal=True).cast(pl.Float64, strict=False)


def a_entero(expr: pl.Expr) -> pl.Expr:
    return expr.str.strip_chars().cast(pl.Int64, strict=False)
