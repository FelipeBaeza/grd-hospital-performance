"""Esquema de validación hacia adelante: siempre se entrena con años anteriores al que se valida."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Reglas


def pliegues(df: pd.DataFrame, reglas: Reglas) -> list[tuple[np.ndarray, np.ndarray]]:
    """Dos pliegues (máscaras de entrenamiento y validación). Verifica que no haya fuga temporal."""
    v = reglas.validacion
    definicion = [(v["pliegue1_entrenamiento"], v["pliegue1_validacion"]), (v["pliegue2_entrenamiento"], v["pliegue2_validacion"])]
    anio = df["ANIO_EGRESO"].to_numpy()
    salida = []
    for ent, val in definicion:
        comprobar_pliegue(ent, val, v["anios_entrenamiento_final"])
        salida.append((np.isin(anio, ent), anio == val))
    return salida


def comprobar_pliegue(entrenamiento: list[int], validacion: int, anios_permitidos: list[int]) -> None:
    if max(entrenamiento) >= validacion:
        raise ValueError("fuga temporal: se entrena con un año igual o posterior al de validación")
    if validacion not in anios_permitidos or not set(entrenamiento) <= set(anios_permitidos):
        raise ValueError("un pliegue usa un año fuera de 2019–2022 (2023 se reserva para calibrar y 2024 está sellado)")


def mascaras_finales(df: pd.DataFrame, reglas: Reglas) -> tuple[np.ndarray, np.ndarray]:
    v = reglas.validacion
    anio = df["ANIO_EGRESO"].to_numpy()
    return np.isin(anio, v["anios_entrenamiento_final"]), anio == v["anio_calibracion"]
