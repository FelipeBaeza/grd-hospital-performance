"""Capas, carpetas de trabajo y candidatos sintéticos, con el mismo esquema que el proyecto, para probar la enmienda 01 de
punta a punta sin datos reales."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HOSPITALES = ["101100", "102100", "103100", "103101", "104100", "105100", "106100", "107100", "108100", "109100",
              "110100", "110110", "111100", "112100", "112103", "112104", "113100", "114100", "115100", "116100"]
CIE = [f"C{i:02d}" for i in range(12)]


def _predictores(rng, n: int, anio: int, inicio: int) -> pd.DataFrame:
    d = pd.DataFrame({
        "ID_EPISODIO": np.arange(inicio, inicio + n), "ANIO_EGRESO": anio, "COD_HOSPITAL": rng.choice(HOSPITALES, n),
        "SEXO": rng.choice(["HOMBRE", "MUJER"], n), "TIPO_INGRESO": rng.choice(["URGENCIA", "PROGRAMADO"], n),
        "TIPO_PROCEDENCIA": rng.choice(["DOMICILIO", "OTRO HOSPITAL", "OTRO"], n), "PROCEDENCIA_AGR": rng.choice(["DOMICILIO", "TRASLADO"], n),
        "CIE10_3C": rng.choice(CIE, n), "GRUPO_CLINICO": rng.choice(["I", "II", "III"], n), "EDAD_ANIOS": rng.uniform(18, 95, n),
        "ES_FIN_SEMANA": rng.integers(0, 2, n), "NIVEL_CUIDADO_INGRESO": rng.integers(0, 2, n),
    })
    for i in range(1, 32):
        d[f"ELIX_{i:02d}"] = (rng.random(n) < 0.08).astype(np.int8)
    return d


def crear(base: Path, por_hospital: int = 1200, semilla: int = 11) -> dict:
    rng = np.random.default_rng(semilla)
    capas, t4, t5 = base / "capas", base / "trabajo4", base / "trabajo5"
    for p in (capas / "oro", capas / "plata", t4, t5 / "evaluacion_2024"):
        p.mkdir(parents=True, exist_ok=True)
    filas, inicio = [], 1
    for anio in range(2019, 2025):
        n = por_hospital * len(HOSPITALES) if anio >= 2022 else por_hospital // 2 * len(HOSPITALES)
        d = _predictores(rng, n, anio, inicio)
        inicio += n
        riesgo = 1 / (1 + np.exp(-(-3.6 + 0.04 * (d["EDAD_ANIOS"] - 60) + 0.8 * d["ELIX_22"] + 0.5 * d["NIVEL_CUIDADO_INGRESO"])))
        d["MORTALIDAD_BINARIA"] = (rng.random(n) < riesgo).astype(np.int8)
        d["ESTANCIA_DIAS"] = rng.poisson(4 + 0.05 * (d["EDAD_ANIOS"] - 18) + 2 * d["ELIX_25"])
        d["ES_TRASLADO_ENTRANTE"] = d["TIPO_PROCEDENCIA"] == "OTRO HOSPITAL"
        filas.append(d)
    todo = pd.concat(filas, ignore_index=True)
    pred_cols = [c for c in todo.columns if c not in ("MORTALIDAD_BINARIA", "ESTANCIA_DIAS", "ES_TRASLADO_ENTRANTE")]
    hist = todo[todo["ANIO_EGRESO"] <= 2023].copy()
    hist["SPLIT"] = np.where(hist["ANIO_EGRESO"] == 2023, "calibracion", "entrenamiento")
    est = hist[(hist["MORTALIDAD_BINARIA"] == 0) & (hist["ESTANCIA_DIAS"] > 0)]
    hist[pred_cols + ["SPLIT"]].to_parquet(capas / "oro" / "matriz_mortalidad.parquet")
    hist[["ID_EPISODIO", "MORTALIDAD_BINARIA"]].to_parquet(capas / "oro" / "desenlace_mortalidad.parquet")
    est[pred_cols + ["SPLIT"]].to_parquet(capas / "oro" / "matriz_estadia.parquet")
    est.assign(ESTANCIA_DIAS_TOPE=np.minimum(est["ESTANCIA_DIAS"], 53))[["ID_EPISODIO", "ESTANCIA_DIAS_TOPE"]].to_parquet(capas / "oro" / "desenlace_estadia.parquet")
    (capas / "oro" / "parametros_oro.json").write_text(json.dumps({"tope_estancia_dias": 53}))
    for anio in (2023, 2024):
        d = todo[todo["ANIO_EGRESO"] == anio]
        d[pred_cols].drop(columns=["ANIO_EGRESO"]).assign(EN_COHORTE_D=True, EN_COHORTE_E=True, SPLIT="x").to_parquet(capas / "oro" / f"oro_predictores_{anio}.parquet")
        d[["ID_EPISODIO", "ES_TRASLADO_ENTRANTE"]].to_parquet(capas / "plata" / f"plata_{anio}.parquet")
        des = pd.DataFrame({"ID_EPISODIO": d["ID_EPISODIO"], "ROL": np.where(d["MORTALIDAD_BINARIA"] == 1, "EVENTO", "NO_EVENTO"),
                            "ESTANCIA_DIAS": d["ESTANCIA_DIAS"], "ES_TRASLADO_SALIENTE": False})
        des.to_parquet((t5 / "evaluacion_2024" / "desenlace_2024.parquet") if anio == 2024 else (capas / "plata" / "desenlace_2023.parquet"))
    lgbm = {"learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 20, "colsample": 0.9, "subsample": 0.9, "reg_alpha": 0.0, "reg_lambda": 0.0}
    (t4 / "candidatos_mortalidad.json").write_text(json.dumps({"lightgbm": {"tipo": "lightgbm", "params": lgbm, "n_arboles": 20},
                                                               "linea_base": {"tipo": "logistica", "params": {"C": 1.0}, "n_arboles": None}}))
    (t4 / "candidatos_estadia.json").write_text(json.dumps({"lightgbm": {"tipo": "lightgbm", "params": {**lgbm, "objetivo": "tweedie", "potencia": 1.5}, "n_arboles": 20},
                                                            "linea_base": {"tipo": "poisson", "params": {"alpha": 1.0}, "n_arboles": None}}))
    return {"capas": capas, "trabajo4": t4, "trabajo5": t5}


def prediccion_falsa(df: pd.DataFrame, carpeta: Path):
    """Reemplaza a los modelos congelados en las pruebas: predicción simple a partir de la edad."""
    if "mortalidad" in str(carpeta):
        p = 1 / (1 + np.exp(-(-4.2 + 0.04 * (df["EDAD_ANIOS"].to_numpy() - 60))))
    else:
        p = 4 + 0.05 * (df["EDAD_ANIOS"].to_numpy() - 18)
    return p, p
