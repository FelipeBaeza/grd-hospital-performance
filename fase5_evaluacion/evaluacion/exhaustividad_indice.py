"""Índice de exhaustividad de codificación por hospital: diagnósticos secundarios válidos registrados sobre los esperados por el modelo congelado."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import DIR_MODELOS_F4


def indice(df_anio: pd.DataFrame, carpeta_modelo: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """``df_anio``: salida de modelado.exhaustividad.cargar_anio (cohorte C del año, con y, CIE10_3C, banda_edad, SEXO, COD_HOSPITAL)."""
    m = joblib.load((carpeta_modelo or DIR_MODELOS_F4 / "exhaustividad") / "modelo_exhaustividad.joblib")
    esp = np.clip(m["modelo"].predict(df_anio[["CIE10_3C", "banda_edad", "SEXO"]]), 1e-6, None)
    d = pd.DataFrame({"h": df_anio["COD_HOSPITAL"].to_numpy(), "obs": df_anio["y"].to_numpy(float), "esp": esp})
    g = d.groupby("h").agg(n_egresos=("obs", "size"), observado=("obs", "sum"), esperado=("esp", "sum")).reset_index().rename(columns={"h": "cod_hospital"})
    glob = float(d["obs"].sum() / d["esp"].sum())
    g["indice_vs_2023"] = g["observado"] / (g["esperado"] * m["factor_global_2023"])      # 1,00 = nivel global de 2023
    g["indice_relativo_al_anio"] = (g["observado"] / g["esperado"]) / glob                    # 1,00 = promedio de ese año
    g = g.sort_values("cod_hospital").reset_index(drop=True)
    return g[["cod_hospital", "n_egresos", "indice_vs_2023", "indice_relativo_al_anio"]], {"indice_global_vs_2023": glob / m["factor_global_2023"], "factor_global_2023": m["factor_global_2023"]}
