"""Predicción con los modelos congelados. No entrena ni recalibra nada: solo carga y aplica."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import DIR_MODELOS_F4


def a_categorias(df: pd.DataFrame) -> pd.DataFrame:
    from modelado.datos import CATEGORICAS
    df = df.copy()
    for c in CATEGORICAS:
        df[c] = df[c].astype("category")
    return df


def predecir(df: pd.DataFrame, carpeta: Path) -> tuple[np.ndarray, np.ndarray]:
    """(predicción cruda, predicción recalibrada) de un modelo congelado ubicado en ``carpeta`` (modelo_completo.joblib y recalibrador.joblib)."""
    modelo = joblib.load(carpeta / "modelo_completo.joblib")
    rec = joblib.load(carpeta / "recalibrador.joblib")
    crudo = np.asarray(modelo.predecir(a_categorias(df)), float)
    return crudo, np.asarray(rec.transformar(crudo), float)


def carpeta_final(tarea: str) -> Path:
    return DIR_MODELOS_F4 / tarea
