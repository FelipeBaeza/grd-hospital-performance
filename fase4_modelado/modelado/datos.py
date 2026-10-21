"""Carga de las matrices del Oro (Fase 3) con las garantías de la Fase 4: lista blanca de columnas y ningún dato de 2024."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from .config import RUTA_DICCIONARIO, cargar_reglas, sha256_archivo

CATEGORICAS = ["SEXO", "TIPO_INGRESO", "TIPO_PROCEDENCIA", "PROCEDENCIA_AGR", "CIE10_3C", "GRUPO_CLINICO"]
NUMERICAS = ["EDAD_ANIOS"]
BINARIAS = ["ES_FIN_SEMANA", "NIVEL_CUIDADO_INGRESO"] + [f"ELIX_{i:02d}" for i in range(1, 32)]
PREDICTORES = CATEGORICAS + NUMERICAS + BINARIAS
CONDICIONALES = ["TIPO_INGRESO", "TIPO_PROCEDENCIA", "PROCEDENCIA_AGR", "NIVEL_CUIDADO_INGRESO"]
CLAVES = ["ID_EPISODIO", "ANIO_EGRESO", "SPLIT", "COD_HOSPITAL"]
PROHIBIDAS = ["COD_HOSPITAL", "ANIO_EGRESO", "SPLIT", "ID_EPISODIO", "MORTALIDAD_BINARIA", "ESTANCIA_DIAS", "ESTANCIA_DIAS_TOPE",
              "N_COMORB_ELIX", "PUNTAJE_VAN_WALRAVEN", "N_DX_SECUNDARIOS", "EN_COHORTE_D", "EN_COHORTE_E"]
SPLITS_PERMITIDOS = {"entrenamiento", "calibracion"}


class DatoProhibido(PermissionError):
    """Se intentó usar una columna prohibida o un dato de 2024."""


def predictores_del_diccionario(ruta: Path | str = RUTA_DICCIONARIO) -> list[str]:
    d = pl.read_csv(ruta)
    ok = d.filter(pl.col("estado").str.starts_with("permitida") | (pl.col("estado") == "condicional"))
    return ok["columna"].to_list()


def verificar_lista_blanca(columnas: list[str]) -> None:
    """Falla si alguna columna pedida a un modelo no figura como permitida o condicional en el diccionario de la Fase 3."""
    permitidas = set(predictores_del_diccionario())
    fuera = [c for c in columnas if c not in permitidas]
    if fuera:
        raise DatoProhibido(f"columnas fuera de la lista blanca: {fuera}")


def verificar_sin_2024(df: pd.DataFrame, tope: int = 2023) -> None:
    if df["ANIO_EGRESO"].max() > tope or not set(df["SPLIT"].unique()) <= SPLITS_PERMITIDOS:
        raise DatoProhibido("la matriz contiene filas de 2024 o de un conjunto no permitido")


def huellas_entrada(capas: Path | str) -> dict[str, str]:
    capas = Path(capas)
    nombres = ["matriz_mortalidad", "matriz_estadia", "desenlace_mortalidad", "desenlace_estadia"]
    return {n: sha256_archivo(capas / "oro" / f"{n}.parquet") for n in nombres}


def cargar_matriz(capas: Path | str, tarea: str) -> pd.DataFrame:
    """Matriz de una tarea ('mortalidad' o 'estadia') con la columna objetivo 'y'. Las predictoras salen de la lista blanca."""
    capas = Path(capas)
    x = pl.read_parquet(capas / "oro" / f"matriz_{tarea}.parquet")
    muestra = os.environ.get("FASE4_MUESTRA")                       # solo para pruebas rápidas de código; la corrida oficial no lo define
    if muestra:
        x = x.sample(fraction=float(muestra), seed=1, shuffle=False).sort("ID_EPISODIO")
    d = pl.read_parquet(capas / "oro" / f"desenlace_{tarea}.parquet")
    obj = "MORTALIDAD_BINARIA" if tarea == "mortalidad" else "ESTANCIA_DIAS_TOPE"
    df = x.join(d.select("ID_EPISODIO", obj), on="ID_EPISODIO", how="inner")
    if df.height != x.height:
        raise ValueError("el desenlace no cubre todos los egresos de la matriz")
    df = df.rename({obj: "y"}).to_pandas()
    for c in CATEGORICAS:
        df[c] = df[c].astype("category")
    verificar_sin_2024(df)
    verificar_lista_blanca(PREDICTORES)
    return df


def X(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas que ve un modelo: solo los predictores de la lista blanca."""
    verificar_lista_blanca(PREDICTORES)
    return df[PREDICTORES]


def submuestra(n: int, fraccion: float, semilla: int) -> np.ndarray:
    """Máscara determinista de submuestreo (mismo resultado con la misma semilla y el mismo orden de filas)."""
    if fraccion >= 1:
        return np.ones(n, dtype=bool)
    return np.random.default_rng(semilla).random(n) < fraccion
