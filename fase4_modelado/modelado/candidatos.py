"""Los candidatos de cada tarea: líneas base (prevalencia, logística, lineal, Poisson) y los modelos de árboles (LightGBM, XGBoost)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from .datos import BINARIAS, CATEGORICAS, NUMERICAS, PREDICTORES, X

TIPOS_LINEALES = ("logistica", "lineal", "poisson")


def preproceso_lineal() -> ColumnTransformer:
    """Codificación para los modelos lineales. Todo lo que se aprende (categorías, medianas, nudos de la curva de edad) se aprende al ajustar, solo con el entrenamiento."""
    binarias_sin_faltante = [c for c in BINARIAS if c != "NIVEL_CUIDADO_INGRESO"]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32), CATEGORICAS),
        ("edad_lineal", StandardScaler(), NUMERICAS),
        ("edad_curva", SplineTransformer(n_knots=6, degree=3, include_bias=False, extrapolation="constant"), NUMERICAS),
        ("cuidado", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True), ["NIVEL_CUIDADO_INGRESO"]),
        ("bin", "passthrough", binarias_sin_faltante),
    ], sparse_threshold=1.0)


@dataclass
class Modelo:
    tarea: str
    tipo: str
    params: dict
    objeto: Any
    n_arboles: int | None = None
    extra: dict = field(default_factory=dict)

    def predecir(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilidad de morir (mortalidad) o estadía esperada en días (estadía)."""
        if self.tipo == "constante":
            return np.full(len(df), self.objeto)
        x = X(df)
        if self.tipo == "logistica":
            return self.objeto.predict_proba(x)[:, 1]
        if self.tipo in ("lineal", "poisson"):
            p = self.objeto.predict(x)
            return np.clip(p, 0.05, None) if self.tipo == "lineal" else p
        if self.tipo == "lightgbm":
            return self.objeto.predict_proba(x)[:, 1] if self.tarea == "mortalidad" else self.objeto.predict(x)
        if self.tipo == "xgboost":
            return self.objeto.predict_proba(x)[:, 1] if self.tarea == "mortalidad" else self.objeto.predict(x)
        raise ValueError(self.tipo)


def _estimador_gbm(tipo: str, tarea: str, p: dict, semilla: int, n_hilos: int, n_arboles: int, rondas: int | None):
    if tipo == "lightgbm":
        comunes = dict(n_estimators=n_arboles, learning_rate=p["learning_rate"], num_leaves=p["num_leaves"], min_child_samples=p["min_child_samples"],
                       colsample_bytree=p["colsample"], subsample=p["subsample"], subsample_freq=1, reg_alpha=p["reg_alpha"], reg_lambda=p["reg_lambda"],
                       n_jobs=n_hilos, random_state=semilla, deterministic=True, force_row_wise=True, verbose=-1)
        if tarea == "mortalidad":
            return lgb.LGBMClassifier(objective="binary", **comunes)
        extra = {"tweedie_variance_power": p["potencia"]} if p["objetivo"] == "tweedie" else {}
        return lgb.LGBMRegressor(objective=p["objetivo"], **extra, **comunes)
    comunes = dict(n_estimators=n_arboles, learning_rate=p["learning_rate"], max_depth=p["max_depth"], min_child_weight=p["min_child_weight"],
                   subsample=p["subsample"], colsample_bytree=p["colsample"], reg_alpha=p["reg_alpha"], reg_lambda=p["reg_lambda"], tree_method="hist",
                   enable_categorical=True, n_jobs=n_hilos, random_state=semilla, early_stopping_rounds=rondas)
    if tarea == "mortalidad":
        return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **comunes)
    if p["objetivo"] == "tweedie":
        return xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=p["potencia"], eval_metric=f"tweedie-nloglik@{p['potencia']}", **comunes)
    return xgb.XGBRegressor(objective="count:poisson", eval_metric="poisson-nloglik", **comunes)


def entrenar(tipo: str, tarea: str, df: pd.DataFrame, params: dict, semilla: int, n_hilos: int = 2, n_arboles: int | None = None,
             validacion: pd.DataFrame | None = None, rondas: int = 50, max_arboles: int = 1500) -> Modelo:
    """Entrena un candidato. Con `validacion` usa detención temprana; sin ella, usa `n_arboles` fijo."""
    y = df["y"].to_numpy()
    if tipo == "constante":
        return Modelo(tarea, tipo, {}, float(y.mean()))
    x = X(df)
    if tipo in TIPOS_LINEALES:
        if tipo == "logistica":
            est = LogisticRegression(C=params["C"], max_iter=300, solver="lbfgs")
        elif tipo == "lineal":
            est = Ridge(alpha=params["alpha"])
        else:
            est = PoissonRegressor(alpha=params["alpha"], max_iter=300)
        pipe = Pipeline([("pre", preproceso_lineal()), ("m", est)]).fit(x, y)
        return Modelo(tarea, tipo, params, pipe)
    con_valid = validacion is not None
    n = max_arboles if con_valid else int(n_arboles)
    est = _estimador_gbm(tipo, tarea, params, semilla, n_hilos, n, rondas if (con_valid and tipo == "xgboost") else None)
    if con_valid:
        xv, yv = X(validacion), validacion["y"].to_numpy()
        if tipo == "lightgbm":
            est.fit(x, y, eval_set=[(xv, yv)], callbacks=[lgb.early_stopping(rondas, verbose=False)])
            mejor = int(est.best_iteration_ or n)
        else:
            est.fit(x, y, eval_set=[(xv, yv)], verbose=False)
            mejor = int(est.best_iteration + 1)
        return Modelo(tarea, tipo, params, est, n_arboles=mejor)
    est.fit(x, y)
    return Modelo(tarea, tipo, params, est, n_arboles=n)
