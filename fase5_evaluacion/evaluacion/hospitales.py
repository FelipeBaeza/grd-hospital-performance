"""O/E por hospital, con volumen mínimo, supresión de celdas pequeñas y sin ranking (los hospitales salen ordenados por código)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import embudo


def _suprimir(x, minimo: int):
    return "<%d" % minimo if 0 < x < minimo else x


def tabla(hospital, y, esperado, tarea: str, prot) -> tuple[pd.DataFrame, dict]:
    """Una fila por hospital. ``y`` = 0/1 en mortalidad o días en estadía; ``esperado`` = probabilidad o días esperados (ya calibrados)."""
    h = prot.hospital
    d = pd.DataFrame({"h": np.asarray(hospital), "y": np.asarray(y, float), "e": np.asarray(esperado, float)})
    if tarea == "estadia":
        disp = float(np.mean((d["y"] - d["e"]) ** 2 / d["e"]))      # dispersión global (residuos de Pearson)
    else:
        disp = 1.0
    g = d.groupby("h").agg(n_egresos=("y", "size"), observado=("y", "sum"), esperado=("e", "sum")).reset_index()
    g["elegible"] = (g["n_egresos"] >= h["min_egresos"]) & (g["esperado"] >= h["min_esperado"])
    el = g[g["elegible"]].copy()
    t2 = embudo.tau2(el["observado"].to_numpy(), el["esperado"].to_numpy(), disp, h["winsor"])
    c = embudo.clasificar(el["observado"].to_numpy(), el["esperado"].to_numpy(), t2, h["z_alerta"], h["z_senal"], disp)
    el = pd.concat([el.reset_index(drop=True), c], axis=1)
    if tarea == "mortalidad":
        el["ic_inf"], el["ic_sup"] = embudo.ic_poisson(el["observado"].to_numpy(), el["esperado"].to_numpy())
    else:
        s = np.sqrt(disp / el["esperado"].to_numpy())
        el["ic_inf"], el["ic_sup"] = np.clip(el["oe"] - 1.96 * s, 0, None), el["oe"] + 1.96 * s
    el = el.sort_values("h").reset_index(drop=True)                 # orden por código: no es un ranking
    pub = el.rename(columns={"h": "cod_hospital"})
    if tarea == "mortalidad":
        pub["observado"] = pub["observado"].astype(int).map(lambda v: _suprimir(v, h["celda_minima"]))
        peq = pub["observado"].astype(str).str.startswith("<")
        pub["ic_inf"] = pub["ic_inf"].astype(float)
        pub["ic_sup"] = pub["ic_sup"].astype(float)
        pub.loc[peq, ["oe", "ic_inf", "ic_sup"]] = np.nan       # el O/E y su intervalo permitirían reconstruir el conteo (observado = O/E × esperado)
    resumen = {"n_hospitales_total": int(len(g)), "n_elegibles": int(len(el)), "n_bajo_volumen": int((~g["elegible"]).sum()), "tau2": float(t2), "dispersion": disp,
               "pct_fuera_95": float((el["zona"] != "dentro de lo esperado").mean()) if len(el) else float("nan"),
               "pct_senal_998": float(el["zona"].isin(["señal alta", "señal baja"]).mean()) if len(el) else float("nan"),
               "egresos_bajo_volumen": int(g.loc[~g["elegible"], "n_egresos"].sum())}
    pub = pub.drop(columns=["elegible"])
    return pub, resumen
