"""Versión publicable de las tablas de una evaluación que convive con otra del mismo año (enmienda 03; incidente I-12).

El protocolo v1.0 publica sus tablas por hospital y por subgrupo con conteos exactos (egresos, muertes observadas, esperados y
límites). La enmienda 03 evalúa el mismo año con una cohorte que quita algunas filas (hospitalización domiciliaria, filas
repetidas, traslados entrantes enlazados y todo el MDC 14). Si las dos versiones se publicaran exactas, la resta diría cuántos
egresos o muertes quitó la regla en cada hospital, a menudo de 1 a 9.

Por eso las tablas de la versión nueva se publican así:

- los conteos de personas (``n``, ``n_egresos``) van redondeados a la decena;
- se quitan el observado, el esperado y los límites del embudo (del límite se despeja el esperado, y con el O/E, el observado);
- el O/E, sus intervalos, las tasas y los índices van con tres decimales.

Las tablas completas quedan en la carpeta de trabajo (``<trabajo>/<carpeta>/tablas_completas``), fuera del repositorio. Las
puertas del cierre las leen de ahí. Los criterios, la calibración por deciles y el resumen no cambian: son nacionales.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

DECIMALES = 3
PASO = 10
QUITAR = ("observado", "esperado", "lim_inf_95", "lim_sup_95", "lim_inf_998", "lim_sup_998")
REDONDEAR_N = ("n", "n_egresos")
TABLAS = ("oe_hospital_mortalidad", "oe_hospital_estadia", "subgrupos_mortalidad", "subgrupos_estadia", "indice_exhaustividad_hospital",
          "tasa_traslado_hospital", "traslado_entrante_hospital")


def a_la_decena(s: pd.Series, minimo: int = 10) -> pd.Series:
    """Conteo redondeado a la decena (mitades hacia arriba); «<10» si está entre 1 y 9."""
    v = pd.to_numeric(s, errors="coerce")
    r = ((v + PASO // 2) // PASO * PASO).astype("Int64").astype(str)
    return r.where(~((v > 0) & (v < minimo)), f"<{minimo}").where(v.notna(), s.astype(str))


def reducir_tabla(d: pd.DataFrame, minimo: int = 10) -> pd.DataFrame:
    d = d.drop(columns=[c for c in QUITAR if c in d.columns]).copy()
    for c in d.columns:
        if c in REDONDEAR_N:
            d[c] = a_la_decena(d[c], minimo)
        elif c not in ("cod_hospital", "variable", "grupo", "zona", "tarea") and pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(DECIMALES)
    return d


def tablas_completas(trabajo: Path, nombre_carpeta: str) -> Path:
    return Path(trabajo) / nombre_carpeta / "tablas_completas"


def publicar(destino: Path, trabajo: Path, minimo: int = 10) -> list[str]:
    """Guarda las tablas completas de ``destino/tablas`` en la carpeta de trabajo (si no estaban) y deja en ``destino/tablas`` la
    versión publicable. Se puede repetir: siempre parte de las completas."""
    destino = Path(destino)
    completas = tablas_completas(trabajo, destino.name)
    completas.mkdir(parents=True, exist_ok=True)
    hechas = []
    for t in TABLAS:
        pub, comp = destino / "tablas" / f"{t}.csv", completas / f"{t}.csv"
        if not comp.exists():
            if not pub.exists():
                continue
            shutil.copy2(pub, comp)
        reducir_tabla(pd.read_csv(comp, dtype={"cod_hospital": str}), minimo).to_csv(pub, index=False)
        hechas.append(t)
    return hechas


def leer(destino: Path, trabajo: Path, tabla: str) -> pd.DataFrame:
    """La tabla completa si existe en la carpeta de trabajo; si no, la publicada (protocolo v1.0)."""
    comp = tablas_completas(trabajo, Path(destino).name) / f"{tabla}.csv"
    return pd.read_csv(comp if comp.exists() else Path(destino) / "tablas" / f"{tabla}.csv")
