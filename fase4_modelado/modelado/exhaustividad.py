"""Modelo simple de exhaustividad de codificación: cuántos diagnósticos secundarios se esperan según el diagnóstico principal, la edad y el sexo.
La razón observado/esperado por hospital se calcula en la Fase 5. No entra al modelo de riesgo."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import Reglas

DIAG = [f"DIAGNOSTICO{i}" for i in range(2, 36)]


def conteo_validos(df: pl.DataFrame, excluir_prefijos: list[str]) -> pl.Series:
    """Diagnósticos secundarios que cuentan: no vacíos y sin factores sociales ni embarazo normal (Fase 1, 11.3)."""
    cuentan = []
    for c in DIAG:
        v = pl.col(c).fill_null("")
        malo = pl.any_horizontal([v.str.starts_with(p) for p in excluir_prefijos])
        cuentan.append(((v != "") & ~malo).cast(pl.Int32))
    return df.select(pl.sum_horizontal(cuentan).alias("n_validos"))["n_validos"]


def cargar_anio(capas: Path, anio: int, reglas: Reglas, vocabulario: set[str]) -> pd.DataFrame:
    """Cohorte C (adultos hospitalizados) de un año, con las variables del modelo de exhaustividad. No usa ningún desenlace."""
    p = pl.read_parquet(capas / "plata" / f"plata_{anio}.parquet", columns=["ID_EPISODIO", "COD_HOSPITAL", "DIAGNOSTICO1", "EDAD_ANIOS", "SEXO"] + DIAG)
    c = pl.read_parquet(capas / "cohortes" / f"cohorte_{anio}.parquet", columns=["ID_EPISODIO", "C"])
    p = p.join(c, on="ID_EPISODIO").filter(pl.col("C"))
    n = conteo_validos(p, reglas.exhaustividad["excluir_prefijos"])
    cie = p["DIAGNOSTICO1"].fill_null("").str.slice(0, 3)
    base = p.select("ID_EPISODIO", "COD_HOSPITAL", "EDAD_ANIOS", "SEXO").with_columns(
        n.alias("y"), cie.alias("CIE10_3C"), pl.lit(anio).alias("ANIO"))
    base = base.with_columns(pl.when(pl.col("CIE10_3C").is_in(sorted(vocabulario))).then(pl.col("CIE10_3C")).otherwise(pl.lit("RARO")).alias("CIE10_3C"))
    edades = reglas.exhaustividad["bandas_edad"]
    df = base.to_pandas()
    df["banda_edad"] = pd.cut(df["EDAD_ANIOS"], edades, right=False).astype(str)
    return df


def vocabulario_cie(tabla_vocabularios: Path) -> set[str]:
    v = pl.read_csv(tabla_vocabularios, infer_schema_length=None)
    return set(v.filter((pl.col("variable") == "CIE10_3C") & pl.col("conservado"))["nivel"].to_list())


def modelo_poisson(alpha: float = 1e-6) -> Pipeline:
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32), ["CIE10_3C", "banda_edad", "SEXO"])], sparse_threshold=1.0)
    return Pipeline([("pre", pre), ("m", PoissonRegressor(alpha=alpha, max_iter=200))])


def entrenar_y_evaluar(entrenamiento: pd.DataFrame, evaluacion: pd.DataFrame, reglas: Reglas) -> dict:
    """Regresión Poisson (modelo que se congela) y LightGBM Poisson (control). Evaluación en 2023."""
    cols = ["CIE10_3C", "banda_edad", "SEXO"]
    glm = modelo_poisson().fit(entrenamiento[cols], entrenamiento["y"])
    pred_glm = np.clip(glm.predict(evaluacion[cols]), 1e-6, None)
    ctrl_tr, ctrl_ev = entrenamiento[cols].copy(), evaluacion[cols].copy()
    for c in cols:
        cats = pd.Categorical(ctrl_tr[c]).categories
        ctrl_tr[c], ctrl_ev[c] = pd.Categorical(ctrl_tr[c], categories=cats), pd.Categorical(ctrl_ev[c], categories=cats)
    ctrl_tr["EDAD"], ctrl_ev["EDAD"] = entrenamiento["EDAD_ANIOS"].to_numpy(), evaluacion["EDAD_ANIOS"].to_numpy()
    gbm = lgb.LGBMRegressor(objective="poisson", n_estimators=300, learning_rate=0.1, num_leaves=63, min_child_samples=200, n_jobs=reglas.ajuste["n_hilos"],
                            random_state=reglas.semilla["maestra"], deterministic=True, force_row_wise=True, verbose=-1).fit(ctrl_tr, entrenamiento["y"])
    pred_gbm = np.clip(gbm.predict(ctrl_ev), 1e-6, None)
    y = evaluacion["y"].to_numpy()

    def resumen(p):
        return {"devianza_poisson": float(mean_poisson_deviance(y, p)), "oe_global": float(y.sum() / p.sum()), "media_obs": float(y.mean()), "media_pred": float(p.mean())}
    return {"glm": glm, "metricas": {"poisson_glm": resumen(pred_glm), "lightgbm_control": resumen(pred_gbm), "solo_media": resumen(np.full(len(y), entrenamiento["y"].mean()))},
            "pred_glm": pred_glm}
