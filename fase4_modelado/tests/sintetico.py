"""Datos inventados con la misma forma que las matrices del Oro, para probar el código sin tocar datos reales."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from modelado.datos import BINARIAS, CATEGORICAS, PREDICTORES


def hacer_df(n: int = 4000, semilla: int = 0, anios=(2019, 2020, 2021, 2022, 2023), tarea: str = "mortalidad") -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    df = pd.DataFrame({
        "ID_EPISODIO": [f"{semilla}-{i}" for i in range(n)],
        "ANIO_EGRESO": rng.choice(anios, n),
        "COD_HOSPITAL": rng.choice(["101", "102", "103", "104"], n),
        "EDAD_ANIOS": rng.integers(18, 95, n).astype("int32"),
        "SEXO": rng.choice(["MUJER", "HOMBRE"], n),
        "TIPO_INGRESO": rng.choice(["URGENCIA", "PROGRAMADO"], n),
        "TIPO_PROCEDENCIA": rng.choice(["A", "B", "C"], n),
        "PROCEDENCIA_AGR": rng.choice(["URGENCIA", "PROGRAMADO"], n),
        "CIE10_3C": rng.choice(["J18", "I50", "K35", "RARO"], n),
        "GRUPO_CLINICO": rng.choice(["X", "IX", "XI"], n),
        "ES_FIN_SEMANA": rng.integers(0, 2, n).astype("int8"),
        "NIVEL_CUIDADO_INGRESO": rng.integers(0, 2, n).astype("int8"),
    })
    for c in BINARIAS:
        if c not in df:
            df[c] = (rng.random(n) < 0.1).astype("int8")
    riesgo = 1 / (1 + np.exp(-(-4 + 0.04 * (df["EDAD_ANIOS"] - 60) + 1.2 * df["ELIX_01"] + 0.8 * df["NIVEL_CUIDADO_INGRESO"])))
    if tarea == "mortalidad":
        df["y"] = (rng.random(n) < riesgo).astype(int)
    else:
        df["y"] = rng.poisson(3 + 6 * riesgo + 1.5 * df["ELIX_02"]).clip(1, 53)
    df["SPLIT"] = np.where(df["ANIO_EGRESO"] <= 2022, "entrenamiento", "calibracion")
    for c in CATEGORICAS:
        df[c] = df[c].astype("category")
    return df


def escribir_capas(carpeta, df: pd.DataFrame, tarea: str = "mortalidad") -> None:
    oro = carpeta / "oro"
    oro.mkdir(parents=True, exist_ok=True)
    x = pl.from_pandas(df.drop(columns=["y"]))
    x = x.with_columns([pl.col(c).cast(pl.String) for c in CATEGORICAS])
    x.write_parquet(oro / f"matriz_{tarea}.parquet")
    obj = "MORTALIDAD_BINARIA" if tarea == "mortalidad" else "ESTANCIA_DIAS_TOPE"
    pl.DataFrame({"ID_EPISODIO": df["ID_EPISODIO"].tolist(), obj: df["y"].tolist()}).write_parquet(oro / f"desenlace_{tarea}.parquet")
