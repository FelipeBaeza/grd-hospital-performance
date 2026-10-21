"""Controles de integridad: el hospital no se puede adivinar con las variables de ingreso, contribuciones por variable y coherencia clínica."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from .config import Reglas
from .datos import CONDICIONALES, PREDICTORES, X, submuestra


def prueba_identificacion_hospital(df: pd.DataFrame, reglas: Reglas, con_condicionales: bool) -> dict:
    """Modelo auxiliar que intenta adivinar el hospital con los predictores del Oro. Se entrena con una muestra de 2019–2022 y se evalúa en una muestra de 2023.
    Se compara la exactitud con y sin las variables condicionales (propuesta D5: la diferencia no debe pasar de 5 puntos porcentuales)."""
    i = reglas.integridad
    semilla = reglas.semilla["maestra"]
    cols = PREDICTORES if con_condicionales else [c for c in PREDICTORES if c not in CONDICIONALES]
    ent = df[df["ANIO_EGRESO"] <= 2022]
    cal = df[df["ANIO_EGRESO"] == 2023]
    ent = ent[submuestra(len(ent), min(1.0, i["muestra_entrenamiento"] / len(ent)), semilla)]
    cal = cal[submuestra(len(cal), min(1.0, i["muestra_evaluacion"] / len(cal)), semilla + 1)]
    clases = sorted(ent["COD_HOSPITAL"].unique())
    codigo = {c: k for k, c in enumerate(clases)}
    cal = cal[cal["COD_HOSPITAL"].isin(codigo)]                    # se descartan hospitales que no existían en el entrenamiento
    yt, yv = ent["COD_HOSPITAL"].map(codigo).to_numpy(), cal["COD_HOSPITAL"].map(codigo).to_numpy()
    m = lgb.LGBMClassifier(objective="multiclass", n_estimators=i["iteraciones"], learning_rate=0.1, num_leaves=31, min_child_samples=50, n_jobs=reglas.ajuste["n_hilos"],
                           random_state=semilla, deterministic=True, force_row_wise=True, verbose=-1).fit(ent[cols], yt)
    prob = m.predict_proba(cal[cols])
    exactitud = float((prob.argmax(axis=1) == yv).mean())
    mayoritaria = float(np.bincount(yt, minlength=len(clases)).max() / len(yt))
    return {"con_condicionales": bool(con_condicionales), "n_predictores": len(cols), "n_hospitales": len(clases), "n_evaluacion": int(len(yv)),
            "exactitud_top1": exactitud, "clase_mayoritaria": mayoritaria}


def contribuciones(modelo, df: pd.DataFrame, n: int, semilla: int) -> pd.DataFrame:
    """Contribución media absoluta de cada variable (valores SHAP nativos de los árboles; en la logística, coeficiente × desviación de cada columna)."""
    muestra = df[submuestra(len(df), min(1.0, n / len(df)), semilla)]
    x = X(muestra)
    if modelo.tipo == "lightgbm":
        c = modelo.objeto.predict(x, pred_contrib=True)[:, :-1]
        if c.ndim == 3:
            c = c[:, :, 0]
        valores = np.abs(c).mean(axis=0)
        nombres = list(x.columns)
    elif modelo.tipo == "xgboost":
        booster = modelo.objeto.get_booster()
        c = booster.predict(xgb.DMatrix(x, enable_categorical=True), pred_contribs=True)[:, :-1]
        valores, nombres = np.abs(c).mean(axis=0), list(x.columns)
    else:
        pipe = modelo.objeto
        z = pipe["pre"].transform(x)
        coef = pipe["m"].coef_.ravel()
        sd = np.sqrt(np.asarray(z.multiply(z).mean(axis=0)).ravel() - np.asarray(z.mean(axis=0)).ravel() ** 2)
        nombres_cols = pipe["pre"].get_feature_names_out()
        por_var: dict[str, float] = {}
        for nombre, c_, s_ in zip(nombres_cols, coef, sd):
            base = nombre.split("__", 1)[1]
            var = next((p for p in PREDICTORES if base == p or base.startswith(p + "_")), base)
            por_var[var] = por_var.get(var, 0.0) + abs(c_) * s_
        nombres, valores = list(por_var), np.array(list(por_var.values()))
    t = pd.DataFrame({"variable": nombres, "contribucion_media_abs": valores}).sort_values("contribucion_media_abs", ascending=False)
    t["proporcion"] = t["contribucion_media_abs"] / t["contribucion_media_abs"].sum()
    return t.reset_index(drop=True)


def coherencia(df: pd.DataFrame, pred: np.ndarray, tarea: str, minimo: int = 10) -> dict[str, pd.DataFrame]:
    """Observado y predicho por tramo de edad y por número de condiciones de Elixhauser (las celdas con menos de 10 egresos no se publican)."""
    d = pd.DataFrame({"y": df["y"].to_numpy(), "pred": pred, "edad": df["EDAD_ANIOS"].to_numpy()})
    d["n_elix"] = df[[c for c in df.columns if c.startswith("ELIX_")]].sum(axis=1).to_numpy()
    d["tramo_edad"] = pd.cut(d["edad"], [17, 29, 39, 49, 59, 69, 79, 89, 200], labels=["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"])
    d["n_condiciones"] = pd.cut(d["n_elix"], [-1, 0, 1, 2, 3, 100], labels=["0", "1", "2", "3", "4 o más"])
    salida = {}
    for nombre, col in (("edad", "tramo_edad"), ("comorbilidad", "n_condiciones")):
        t = d.groupby(col, observed=True).agg(n=("y", "size"), observado=("y", "mean"), predicho=("pred", "mean")).reset_index().rename(columns={col: "grupo"})
        t = t[t["n"] >= minimo]
        t["razon_oe"] = t["observado"] / t["predicho"]
        salida[nombre] = t
    return salida
