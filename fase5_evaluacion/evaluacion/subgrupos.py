"""Calibración y error por subgrupos de pacientes; las celdas de menos de 10 casos se suprimen."""
from __future__ import annotations

import numpy as np
import pandas as pd

BANDAS = [18, 30, 40, 50, 60, 70, 80, 90, 200]


def _por(df: pd.DataFrame, col: str, tarea: str, minimo: int) -> pd.DataFrame:
    g = df.groupby(col, observed=True).agg(n=("y", "size"), observado=("y", "sum"), esperado=("p", "sum")).reset_index().rename(columns={col: "grupo"})
    g["variable"] = col
    g["oe"] = g["observado"] / g["esperado"]
    g = g[g["n"] >= minimo].copy()
    if tarea == "mortalidad":
        g["observado"] = g["observado"].astype(int)
        pequena = g["observado"] < minimo
        g.loc[pequena, ["observado", "oe"]] = np.nan                   # se suprime el conteo pequeño de eventos y el O/E, que lo revelaría
    return g[["variable", "grupo", "n", "observado", "esperado", "oe"]]


def tabla(df: pd.DataFrame, y, p, tarea: str, minimo: int = 10) -> pd.DataFrame:
    elix = [c for c in df.columns if c.startswith("ELIX_")]
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    d["edad"] = pd.cut(df["EDAD_ANIOS"].to_numpy(), BANDAS, right=False).astype(str)
    d["sexo"] = df["SEXO"].astype(str).to_numpy()
    d["comorbilidades"] = np.minimum(df[elix].fillna(0).sum(axis=1).to_numpy(), 4).astype(int).astype(str)
    d["grupo_clinico"] = df["GRUPO_CLINICO"].astype(str).to_numpy()
    vol = df.groupby("COD_HOSPITAL")["COD_HOSPITAL"].transform("size")
    d["tamano_hospital"] = pd.qcut(vol.rank(method="first"), 3, labels=["menor", "mediano", "mayor"]).astype(str).to_numpy()
    return pd.concat([_por(d, c, tarea, minimo) for c in ("edad", "sexo", "comorbilidades", "grupo_clinico", "tamano_hospital")], ignore_index=True)
