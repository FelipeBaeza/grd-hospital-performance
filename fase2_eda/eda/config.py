"""Carga de configuración: umbrales (TOML), hipótesis, reglas de la Fase 1 y fuentes."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from . import rutas


def leer_csv_texto(ruta: Path | str) -> pl.DataFrame:
    """Lee un CSV de configuración con todas las columnas como texto.

    Las celdas vacías quedan como cadena vacía para no confundirlas con ausencia
    de columna.
    """
    return pl.read_csv(
        ruta,
        infer_schema=False,
        encoding="utf8",
        empty_string_is_null=False,
    )


@dataclass(frozen=True)
class Umbrales:
    """Umbrales congelados del protocolo (config/umbrales.toml)."""

    meta: dict[str, Any]
    particion: dict[str, Any]
    cortafuegos: dict[str, Any]
    _umbrales: dict[str, dict[str, Any]]

    def __getitem__(self, nombre: str) -> Any:
        return self._umbrales[nombre]["valor"]

    def get(self, nombre: str, defecto: Any = None) -> Any:
        if nombre in self._umbrales:
            return self._umbrales[nombre]["valor"]
        return defecto

    def justificacion(self, nombre: str) -> str:
        return str(self._umbrales[nombre].get("justificacion", ""))

    def nombres(self) -> list[str]:
        return sorted(self._umbrales)

    def a_tabla(self) -> pl.DataFrame:
        filas = [
            {
                "umbral": n,
                "valor": json.dumps(self._umbrales[n]["valor"], ensure_ascii=False),
                "justificacion": self.justificacion(n),
            }
            for n in self.nombres()
        ]
        return pl.DataFrame(filas, schema={"umbral": pl.String, "valor": pl.String, "justificacion": pl.String})


def cargar_umbrales(ruta: Path | str | None = None) -> Umbrales:
    ruta = Path(ruta) if ruta else rutas.DIR_CONFIG / "umbrales.toml"
    with open(ruta, "rb") as f:
        datos = tomllib.load(f)
    return Umbrales(
        meta=dict(datos.get("meta", {})),
        particion=dict(datos.get("particion", {})),
        cortafuegos=dict(datos.get("cortafuegos", {})),
        _umbrales={k: dict(v) for k, v in datos.get("umbral", {}).items()},
    )


def validar_umbrales(u: Umbrales) -> list[str]:
    """Devuelve la lista de problemas del archivo de umbrales (vacía si está bien)."""
    problemas: list[str] = []
    if not u.nombres():
        problemas.append("no hay umbrales definidos")
    for n in u.nombres():
        d = u._umbrales[n]
        if "valor" not in d:
            problemas.append(f"{n}: falta 'valor'")
        if not str(d.get("justificacion", "")).strip():
            problemas.append(f"{n}: falta 'justificacion'")
    return problemas


def cargar_hipotesis(ruta: Path | str | None = None) -> pl.DataFrame:
    return leer_csv_texto(Path(ruta) if ruta else rutas.DIR_CONFIG / "hipotesis.csv")


def cargar_plan_figuras(ruta: Path | str | None = None) -> pl.DataFrame:
    return leer_csv_texto(Path(ruta) if ruta else rutas.DIR_CONFIG / "plan_figuras.csv")


def cargar_definicion_terminado(ruta: Path | str | None = None) -> pl.DataFrame:
    return leer_csv_texto(Path(ruta) if ruta else rutas.DIR_CONFIG / "definicion_terminado.csv")


def cargar_arbol_decision(ruta: Path | str | None = None) -> pl.DataFrame:
    return leer_csv_texto(Path(ruta) if ruta else rutas.DIR_CONFIG / "arbol_decision.csv")


def cargar_decisiones_fase1(ruta: Path | str | None = None) -> pl.DataFrame:
    """Las 17 decisiones pendientes de la Fase 1 (n, tema, propuesta, estado)."""
    return leer_csv_texto(Path(ruta) if ruta else rutas.DIR_FASE1 / "decisiones_pendientes.csv")


def cargar_reglas_variables(ruta: Path | str | None = None) -> pl.DataFrame:
    """Rol de cada variable según la Fase 1 (variable, categoria, nota)."""
    return leer_csv_texto(Path(ruta) if ruta else rutas.DIR_FASE1 / "reglas_variables.csv")


def cargar_tipoalta_roles(ruta: Path | str | None = None) -> pl.DataFrame:
    return leer_csv_texto(Path(ruta) if ruta else rutas.DIR_FASE1 / "tipoalta_roles.csv")


def cargar_spec_fase1(ruta: Path | str | None = None) -> dict[str, Any]:
    ruta = Path(ruta) if ruta else rutas.DIR_FASE1 / "config" / "fase1_spec.json"
    with open(ruta, encoding="utf8") as f:
        return json.load(f)


def cargar_fuentes_toml(ruta: Path | str | None = None) -> dict[str, Any]:
    ruta = Path(ruta) if ruta else rutas.DIR_CONFIG / "fuentes.toml"
    with open(ruta, "rb") as f:
        return tomllib.load(f)
