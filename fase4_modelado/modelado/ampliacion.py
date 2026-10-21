"""Ampliación exploratoria (después de ver los resultados de la Fase 4): más algoritmos de árboles bajo el MISMO protocolo.
Mismos pliegues hacia adelante, misma submuestra de ajuste, mismo método de búsqueda (TPE, semilla fija) y la misma métrica de selección.
NO usa 2023 para decidir ni toca 2024. No modifica el código ni los modelos congelados de la Fase 4."""
from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from .config import Reglas
from .datos import CATEGORICAS, PREDICTORES, X, submuestra
from .tuning import calcular_metricas, metrica_seleccion
from .validacion import pliegues

optuna.logging.set_verbosity(optuna.logging.WARNING)
NUEVOS = ("catboost", "lgbm_rf", "lgbm_extra")


def espacio(trial: optuna.Trial, tipo: str, tarea: str) -> dict:
    if tipo == "catboost":
        p = {"learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True), "depth": trial.suggest_int("depth", 4, 10),
             "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True), "rsm": trial.suggest_float("rsm", 0.5, 1.0),
             "subsample": trial.suggest_float("subsample", 0.5, 1.0), "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 1000, log=True)}
    elif tipo == "lgbm_rf":
        p = {"n_estimators": trial.suggest_int("n_estimators", 100, 400), "num_leaves": trial.suggest_int("num_leaves", 63, 1023, log=True),
             "min_child_samples": trial.suggest_int("min_child_samples", 20, 1000, log=True), "subsample": trial.suggest_float("subsample", 0.5, 0.9),
             "colsample": trial.suggest_float("colsample", 0.3, 0.9), "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)}
    else:   # lgbm_extra: boosting con umbrales de corte aleatorios (árboles extremadamente aleatorizados)
        p = {"learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True), "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
             "min_child_samples": trial.suggest_int("min_child_samples", 100, 2000, log=True), "subsample": trial.suggest_float("subsample", 0.5, 1.0),
             "colsample": trial.suggest_float("colsample", 0.5, 1.0), "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)}
    return p


def _cat_a_texto(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    for c in CATEGORICAS:
        x[c] = x[c].astype("object").where(x[c].notna(), "NA").astype(str)
    return x


class ModeloExtra:
    def __init__(self, tipo, tarea, objeto, n_arboles):
        self.tipo, self.tarea, self.objeto, self.n_arboles = tipo, tarea, objeto, n_arboles

    def predecir(self, df: pd.DataFrame) -> np.ndarray:
        x = X(df)
        if self.tipo == "catboost":
            x = _cat_a_texto(x)
            return self.objeto.predict_proba(x)[:, 1] if self.tarea == "mortalidad" else self.objeto.predict(x, prediction_type="Exponent")
        return self.objeto.predict_proba(x)[:, 1] if self.tarea == "mortalidad" else self.objeto.predict(x)


def entrenar_extra(tipo, tarea, train, params, semilla, n_hilos, valid=None, rondas=50, max_arboles=1500, n_arboles=None):
    x, y = X(train), train["y"].to_numpy()
    if tipo == "catboost":
        kw = dict(iterations=max_arboles if valid is not None else int(n_arboles), learning_rate=params["learning_rate"], depth=params["depth"],
                  l2_leaf_reg=params["l2_leaf_reg"], rsm=params["rsm"], subsample=params["subsample"], bootstrap_type="Bernoulli",
                  min_data_in_leaf=params["min_data_in_leaf"], grow_policy="Depthwise", thread_count=n_hilos, random_seed=semilla, verbose=False,
                  allow_writing_files=False, cat_features=CATEGORICAS)
        est = CatBoostClassifier(loss_function="Logloss", **kw) if tarea == "mortalidad" else CatBoostRegressor(loss_function="Poisson", **kw)
        xt = _cat_a_texto(x)
        if valid is not None:
            xv = _cat_a_texto(X(valid))
            est.fit(xt, y, eval_set=(xv, valid["y"].to_numpy()), early_stopping_rounds=rondas)
            n = int(est.get_best_iteration() + 1)
        else:
            est.fit(xt, y)
            n = int(n_arboles)
        return ModeloExtra(tipo, tarea, est, n)
    comunes = dict(num_leaves=params["num_leaves"], min_child_samples=params["min_child_samples"], colsample_bytree=params["colsample"],
                   subsample=params["subsample"], subsample_freq=1, reg_lambda=params["reg_lambda"], n_jobs=n_hilos, random_state=semilla,
                   deterministic=True, force_row_wise=True, verbose=-1)
    objetivo = "binary" if tarea == "mortalidad" else "poisson"
    Est = lgb.LGBMClassifier if tarea == "mortalidad" else lgb.LGBMRegressor
    if tipo == "lgbm_rf":
        est = Est(objective=objetivo, boosting_type="rf", n_estimators=params["n_estimators"] if valid is not None or n_arboles is None else int(n_arboles), **comunes)
        est.fit(x, y)
        return ModeloExtra(tipo, tarea, est, params["n_estimators"])
    est = Est(objective=objetivo, n_estimators=max_arboles if valid is not None else int(n_arboles), learning_rate=params["learning_rate"], extra_trees=True, **comunes)
    if valid is not None:
        est.fit(x, y, eval_set=[(X(valid), valid["y"].to_numpy())], callbacks=[lgb.early_stopping(rondas, verbose=False)])
        return ModeloExtra(tipo, tarea, est, int(est.best_iteration_ or max_arboles))
    est.fit(x, y)
    return ModeloExtra(tipo, tarea, est, int(n_arboles))


def _evaluar(tipo, tarea, df, tr, va, params, reglas, semilla, fraccion):
    a = reglas.ajuste
    idx = np.flatnonzero(tr)
    train = df.iloc[idx[submuestra(len(idx), fraccion, semilla)]]
    valid = df.iloc[np.flatnonzero(va)]
    m = entrenar_extra(tipo, tarea, train, params, semilla, a["n_hilos"], valid=valid, rondas=a["rondas_detencion_temprana"], max_arboles=a["max_arboles"])
    met = calcular_metricas(tarea, valid["y"].to_numpy(), m.predecir(valid), float(train["y"].mean()))
    return met, m.n_arboles


def ajustar(tarea, tipo, df, reglas: Reglas, almacen: Path, n_pruebas: int, avisar=print) -> dict:
    semilla = reglas.semilla["maestra"]
    folds = pliegues(df, reglas)
    estudio = optuna.create_study(study_name=f"ampl_{tarea}_{tipo}", storage=f"sqlite:///{almacen / ('ampl_' + tarea + '_' + tipo + '.db')}", load_if_exists=True,
                                  direction="maximize" if tarea == "mortalidad" else "minimize", sampler=optuna.samplers.TPESampler(seed=semilla),
                                  pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))

    def objetivo(trial):
        params = espacio(trial, tipo, tarea)
        pun = []
        for k, (tr, va) in enumerate(folds):
            met, _ = _evaluar(tipo, tarea, df, tr, va, params, reglas, semilla + k, reglas.ajuste["submuestra_ajuste"])
            pun.append(metrica_seleccion(tarea, met))
            trial.report(float(np.mean(pun)), k)
            if k == 0 and trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(pun))

    hechas = len([t for t in estudio.trials if t.state.is_finished()])
    t0 = time.time()
    for i in range(hechas, n_pruebas):
        estudio.optimize(objetivo, n_trials=1)
        avisar(f"[ampliacion {tarea}/{tipo}] prueba {i + 1}/{n_pruebas}; mejor {estudio.best_value:.4f}; {time.time() - t0:.0f} s")
    tabla = estudio.trials_dataframe(attrs=("number", "value", "state", "params")).rename(columns=lambda c: c.replace("params_", ""))
    tabla = tabla[tabla["state"].isin(["COMPLETE", "PRUNED"])].copy()
    tabla.loc[tabla["state"] == "PRUNED", "value"] = float("nan")
    return {"params": dict(estudio.best_trial.params), "mejor_valor_submuestra": float(estudio.best_value), "tabla": tabla}


def reajuste_completo(tipo, tarea, df, params, reglas: Reglas) -> dict:
    """Con los mejores hiperparámetros, cada pliegue se reentrena con todas sus filas (misma regla que los candidatos de la Fase 4)."""
    semilla = reglas.semilla["maestra"]
    mets, its, seg = [], [], []
    for k, (tr, va) in enumerate(pliegues(df, reglas)):
        t0 = time.time()
        met, it = _evaluar(tipo, tarea, df, tr, va, params, reglas, semilla + k, 1.0)
        mets.append(met); its.append(it); seg.append(round(time.time() - t0))
    return {"metricas_pliegues": mets, "n_arboles": its, "segundos": seg, "score": float(np.mean([metrica_seleccion(tarea, m) for m in mets]))}
