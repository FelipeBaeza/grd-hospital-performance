"""Cohortes D y E con las reglas de la Fase 3 (prep/cohortes.py): D = C con desenlace conocido; E = D, sobrevivientes con estadía mayor que cero, con tope de 53 días."""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLES_D = ("EVENTO", "NO_EVENTO")


def construir(pred_c: pd.DataFrame, desenlace: pd.DataFrame, tope: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``pred_c``: predictores de la cohorte C (ID_EPISODIO, COD_HOSPITAL, 40 predictores). ``desenlace``: ID_EPISODIO, ROL, ESTANCIA_DIAS.
    Devuelve (mortalidad, estadía), cada una con la columna objetivo ``y``."""
    d = pred_c.merge(desenlace[["ID_EPISODIO", "ROL", "ESTANCIA_DIAS"]], on="ID_EPISODIO", how="inner")
    if len(d) != len(pred_c):
        raise ValueError("el desenlace no cubre todos los egresos de la cohorte C")
    mort = d[d["ROL"].isin(ROLES_D)].copy()
    mort["y"] = (mort["ROL"] == "EVENTO").astype(np.int8)
    est = d[(d["ROL"] == "NO_EVENTO") & (d["ESTANCIA_DIAS"] > 0)].copy()
    est["y"] = np.minimum(est["ESTANCIA_DIAS"].to_numpy(), tope).astype(np.int32)
    return mort.drop(columns=["ROL", "ESTANCIA_DIAS"]).reset_index(drop=True), est.drop(columns=["ROL", "ESTANCIA_DIAS"]).reset_index(drop=True)
