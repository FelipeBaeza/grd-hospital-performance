"""Ajuste de hiperparámetros con pliegues hacia adelante. Misma búsqueda (TPE) y mismo número de pruebas para LightGBM y XGBoost."""
from __future__ import annotations

import csv
import datetime as dt
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from .candidatos import entrenar
from .config import Reglas
from .datos import submuestra
from .metricas import metricas_estadia, metricas_mortalidad
from .validacion import pliegues

optuna.logging.set_verbosity(optuna.logging.WARNING)


def metrica_seleccion(tarea: str, m: dict) -> float:
    return m["pr_auc"] if tarea == "mortalidad" else m["devianza_poisson"]


def calcular_metricas(tarea: str, y, pred, prev_ref: float | None = None) -> dict:
    return metricas_mortalidad(y, pred, prev_ref) if tarea == "mortalidad" else metricas_estadia(y, pred)


def espacio(trial: optuna.Trial, tipo: str, tarea: str) -> dict:
    """Espacios de búsqueda acotados (plan, sección 7.1)."""
    p = {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True), "subsample": trial.suggest_float("subsample", 0.5, 1.0),
         "colsample": trial.suggest_float("colsample", 0.5, 1.0), "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
         "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)}
    if tipo == "lightgbm":
        p["num_leaves"] = trial.suggest_int("num_leaves", 31, 255, log=True)
        p["min_child_samples"] = trial.suggest_int("min_child_samples", 100, 2000, log=True)
    else:
        p["max_depth"] = trial.suggest_int("max_depth", 4, 10)
        p["min_child_weight"] = trial.suggest_float("min_child_weight", 1.0, 200.0, log=True)
    if tarea == "estadia":
        p["objetivo"] = trial.suggest_categorical("objetivo", ["poisson", "tweedie"])
        p["potencia"] = trial.suggest_float("potencia", 1.1, 1.9) if p["objetivo"] == "tweedie" else None
    return p


def _evaluar_pliegue(tipo, tarea, df, tr, va, params, reglas, semilla, fraccion):
    a = reglas.ajuste
    sel = tr & submuestra(len(df), 1.0, 0)
    idx = np.flatnonzero(sel)
    keep = submuestra(len(idx), fraccion, semilla)
    train = df.iloc[idx[keep]]
    valid = df.iloc[np.flatnonzero(va)]
    m = entrenar(tipo, tarea, train, params, semilla, a["n_hilos"], validacion=valid, rondas=a["rondas_detencion_temprana"], max_arboles=a["max_arboles"])
    pred = m.predecir(valid)
    met = calcular_metricas(tarea, valid["y"].to_numpy(), pred, float(train["y"].mean()))
    return met, m.n_arboles


REGISTRO_REANUDACIONES = "reanudaciones.csv"


def registrar_reanudacion(almacen: Path, estudio_nombre: str, terminadas: int, abiertas: list[int], n_pruebas: int, semilla_por_prueba: bool) -> Path:
    """Anota en ``almacen/reanudaciones.csv`` cada vez que un estudio se retoma (hallazgo H-19, incidente I-04).
    Una prueba que quedó abierta por un corte conserva sus parámetros y el muestreador nuevo la ve, así que una corrida
    retomada puede no dar las mismas pruebas que una de corrido. El registro deja constancia de cuándo pasó."""
    ruta = almacen / REGISTRO_REANUDACIONES
    nueva = not ruta.exists()
    with open(ruta, "a", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        if nueva:
            w.writerow(["momento_utc", "estudio", "pruebas_terminadas", "pruebas_abiertas", "pruebas_objetivo", "semilla_por_prueba"])
        w.writerow([dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(), estudio_nombre, terminadas,
                    " ".join(map(str, abiertas)), n_pruebas, semilla_por_prueba])
    return ruta


def ajustar_gbm(tarea: str, tipo: str, df: pd.DataFrame, reglas: Reglas, almacen: Path, n_pruebas: int | None = None, avisar=print,
                semilla_por_prueba: bool = False) -> dict:
    """Optuna (TPE, semilla fija), con almacenamiento en SQLite para poder retomar. Devuelve los mejores hiperparámetros y la tabla de pruebas.

    Cada reanudación queda en ``almacen/reanudaciones.csv``. Con ``semilla_por_prueba`` (apagado por defecto, para que
    la corrida congelada se reproduzca igual), las pruebas abiertas por un corte se marcan como fallidas y el
    muestreador se crea de nuevo antes de la prueba i-ésima con la semilla ``maestra + i``. Así, una corrida
    cortada y retomada da las mismas pruebas que una de corrido."""
    a = reglas.ajuste
    n_pruebas = n_pruebas or (a["pruebas_mortalidad"] if tarea == "mortalidad" else a["pruebas_estadia"])
    semilla = reglas.semilla["maestra"]
    folds = pliegues(df, reglas)
    direccion = "maximize" if tarea == "mortalidad" else "minimize"
    almacen.mkdir(parents=True, exist_ok=True)
    estudio = optuna.create_study(study_name=f"{tarea}_{tipo}", storage=f"sqlite:///{almacen / (tarea + '_' + tipo + '.db')}", load_if_exists=True, direction=direccion,
                                  sampler=optuna.samplers.TPESampler(seed=semilla), pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))

    def objetivo(trial: optuna.Trial) -> float:
        params = espacio(trial, tipo, tarea)
        puntajes = []
        for k, (tr, va) in enumerate(folds):
            met, _ = _evaluar_pliegue(tipo, tarea, df, tr, va, params, reglas, semilla + k, a["submuestra_ajuste"])
            puntajes.append(metrica_seleccion(tarea, met))
            trial.report(float(np.mean(puntajes)), k)
            if k == 0 and trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(puntajes))

    hechas = len([t for t in estudio.trials if t.state.is_finished()])
    abiertas = [t.number for t in estudio.trials if t.state == optuna.trial.TrialState.RUNNING]
    if hechas or abiertas:
        registrar_reanudacion(almacen, estudio.study_name, hechas, abiertas, n_pruebas, semilla_por_prueba)
        avisar(f"[ajuste {tarea}/{tipo}] se retoma con {hechas} pruebas terminadas y {len(abiertas)} abiertas; anotado en {REGISTRO_REANUDACIONES}")
    if semilla_por_prueba and abiertas:
        for t in estudio.trials:
            if t.state == optuna.trial.TrialState.RUNNING:
                estudio._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
    t0 = time.time()
    for i in range(hechas, n_pruebas):
        if semilla_por_prueba:
            estudio.sampler = optuna.samplers.TPESampler(seed=semilla + i)
        estudio.optimize(objetivo, n_trials=1)
        avisar(f"[ajuste {tarea}/{tipo}] prueba {i + 1}/{n_pruebas}; mejor {estudio.best_value:.4f}; {time.time() - t0:.0f} s")
    mejor = estudio.best_trial
    params = dict(mejor.params)
    if tarea == "estadia" and params.get("objetivo") == "poisson":
        params["potencia"] = None
    tabla = estudio.trials_dataframe(attrs=("number", "value", "state", "params")).rename(columns=lambda c: c.replace("params_", ""))
    tabla = tabla[tabla["state"].isin(["COMPLETE", "PRUNED"])].copy()                 # sin pruebas interrumpidas
    tabla.loc[tabla["state"] == "PRUNED", "value"] = float("nan")                      # una prueba podada solo se midió en un pliegue: su valor no es comparable
    return {"params": params, "mejor_valor_submuestra": float(estudio.best_value), "tabla": tabla}


def reajuste_completo(tarea: str, tipo: str, df: pd.DataFrame, params: dict, reglas: Reglas) -> dict:
    """Con los mejores hiperparámetros, reentrena cada pliegue con TODAS sus filas (detención temprana) y devuelve las métricas de validación."""
    a = reglas.ajuste
    semilla = reglas.semilla["maestra"]
    mets, iters = [], []
    for k, (tr, va) in enumerate(pliegues(df, reglas)):
        met, it = _evaluar_pliegue(tipo, tarea, df, tr, va, params, reglas, semilla + k, 1.0)
        mets.append(met)
        iters.append(it)
    n_arboles = int(max(50, round(np.mean(iters) * 1.1)))
    score = float(np.mean([metrica_seleccion(tarea, m) for m in mets]))
    return {"metricas_pliegues": mets, "n_arboles_pliegues": iters, "n_arboles_final": n_arboles, "score": score}


def ajustar_lineal(tarea: str, df: pd.DataFrame, reglas: Reglas, avisar=print) -> dict:
    """Malla pequeña de la regularización para la línea base (logística en mortalidad; Poisson en estadía). Se evalúa en los mismos pliegues."""
    a = reglas.ajuste
    semilla = reglas.semilla["maestra"]
    tipo = "logistica" if tarea == "mortalidad" else "poisson"
    malla = [{"C": c} for c in a["c_logistica"]] if tarea == "mortalidad" else [{"alpha": v} for v in a["alpha_poisson"]]
    folds = pliegues(df, reglas)
    filas = []
    for params in malla:
        pun = []
        for k, (tr, va) in enumerate(folds):
            idx = np.flatnonzero(tr)
            train = df.iloc[idx[submuestra(len(idx), a["submuestra_ajuste"], semilla + k)]]
            valid = df.iloc[np.flatnonzero(va)]
            m = entrenar(tipo, tarea, train, params, semilla)
            pun.append(metrica_seleccion(tarea, calcular_metricas(tarea, valid["y"].to_numpy(), m.predecir(valid), float(train["y"].mean()))))
        filas.append({**params, "valor": float(np.mean(pun))})
        avisar(f"[ajuste {tarea}/{tipo}] {params}: {filas[-1]['valor']:.4f}")
    t = pd.DataFrame(filas)
    mejor = t.loc[t["valor"].idxmax() if tarea == "mortalidad" else t["valor"].idxmin()]
    params = {k: float(mejor[k]) for k in malla[0]}
    return {"tipo": tipo, "params": params, "tabla": t}
