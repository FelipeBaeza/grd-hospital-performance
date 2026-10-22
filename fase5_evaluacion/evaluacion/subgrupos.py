"""Calibración y error por subgrupos de pacientes, sin celdas de menos de 10 casos.

Antes (hasta el 22-10-2026), los grupos con menos de 10 egresos se omitían y, en mortalidad, se ocultaba el observado
de los grupos con menos de 10 muertes. El escáner por propiedad mostró que eso no basta (incidente I-11): todas las
variables dividen la misma cohorte, así que lo que falta en una variable se obtiene restando del total los grupos
publicados.

Ahora, con ``agrupar_pequenos`` (por defecto), los grupos pequeños de cada variable se juntan en un grupo combinado
(«A o B»). Si el combinado sigue siendo pequeño, se le suma el grupo publicado que menos agrega (en mortalidad, el de
menos muertes; en estadía, el de menos egresos), hasta tener al menos 10 egresos y, en mortalidad, al menos 10 muertes. Así no queda ninguna celda oculta ni omitida, y ningún complemento
revela un conteo pequeño. ``agrupar_pequenos=False`` reproduce las tablas anteriores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BANDAS = [18, 30, 40, 50, 60, 70, 80, 90, 200]


def _agrupar(g: pd.DataFrame, tarea: str, minimo: int) -> pd.DataFrame:
    """Junta los grupos pequeños de una variable en un grupo combinado que no sea pequeño (incidente I-11)."""
    def pequeno(n, obs):
        return (n < minimo) | ((tarea == "mortalidad") & (obs < minimo))
    chico = pequeno(g["n"], g["observado"])
    if not chico.any():
        return g
    juntos = g[chico]
    # se suma primero el grupo publicado que menos agrega: en mortalidad, el de menos muertes; en estadía, el de menos egresos
    resto = g[~chico].sort_values(["observado", "n", "grupo"] if tarea == "mortalidad" else ["n", "grupo"])
    while bool(pequeno(juntos["n"].sum(), juntos["observado"].sum())) and len(resto):
        juntos, resto = pd.concat([juntos, resto.iloc[:1]]), resto.iloc[1:]
    combinado = pd.DataFrame([{"grupo": " o ".join(sorted(juntos["grupo"].astype(str))), "n": int(juntos["n"].sum()),
                               "observado": float(juntos["observado"].sum()), "esperado": float(juntos["esperado"].sum())}])
    return pd.concat([resto.sort_values("grupo"), combinado], ignore_index=True)


def _por(df: pd.DataFrame, col: str, tarea: str, minimo: int, agrupar_pequenos: bool = True) -> pd.DataFrame:
    g = df.groupby(col, observed=True).agg(n=("y", "size"), observado=("y", "sum"), esperado=("p", "sum")).reset_index().rename(columns={col: "grupo"})
    if agrupar_pequenos:
        g = _agrupar(g, tarea, minimo)
        g["variable"] = col
        g["oe"] = g["observado"] / g["esperado"]
        if tarea == "mortalidad":
            g["observado"] = g["observado"].astype(int)
            chico = g["observado"] < minimo                             # solo si toda la cohorte tiene menos de 10 muertes
            if chico.any():
                g["observado"] = g["observado"].astype(float)
                g.loc[chico, ["observado", "oe"]] = np.nan
        return g[["variable", "grupo", "n", "observado", "esperado", "oe"]]
    g["variable"] = col
    g["oe"] = g["observado"] / g["esperado"]
    g = g[g["n"] >= minimo].copy()
    if tarea == "mortalidad":
        g["observado"] = g["observado"].astype(int)
        pequena = g["observado"] < minimo
        g.loc[pequena, ["observado", "oe"]] = np.nan                   # se suprime el conteo pequeño de eventos y el O/E, que lo revelaría
    return g[["variable", "grupo", "n", "observado", "esperado", "oe"]]


def tabla(df: pd.DataFrame, y, p, tarea: str, minimo: int = 10, agrupar_pequenos: bool = True) -> pd.DataFrame:
    elix = [c for c in df.columns if c.startswith("ELIX_")]
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    d["edad"] = pd.cut(df["EDAD_ANIOS"].to_numpy(), BANDAS, right=False).astype(str)
    d["sexo"] = df["SEXO"].astype(str).to_numpy()
    d["comorbilidades"] = np.minimum(df[elix].fillna(0).sum(axis=1).to_numpy(), 4).astype(int).astype(str)
    d["grupo_clinico"] = df["GRUPO_CLINICO"].astype(str).to_numpy()
    vol = df.groupby("COD_HOSPITAL")["COD_HOSPITAL"].transform("size")
    d["tamano_hospital"] = pd.qcut(vol.rank(method="first"), 3, labels=["menor", "mediano", "mayor"]).astype(str).to_numpy()
    return pd.concat([_por(d, c, tarea, minimo, agrupar_pequenos) for c in ("edad", "sexo", "comorbilidades", "grupo_clinico", "tamano_hospital")], ignore_index=True)
