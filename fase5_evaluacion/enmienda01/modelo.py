"""Análisis 2.1 a 2.5 de la enmienda 01: variantes del modelo con la receta congelada, en los diseños anidado y original.

Cada variante usa el LightGBM de la tarea con los hiperparámetros y el número de árboles de la Fase 4, la semilla de la
Fase 4 y la misma recalibración (Platt en mortalidad, factor único en estadía). Se compara con la base del mismo diseño.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from . import comun

DISENOS = {"anidado": {"ent": [2019, 2020, 2021], "cal": 2022, "prueba": 2023},
           "original": {"ent": [2019, 2020, 2021, 2022], "cal": 2023, "prueba": 2024}}


def variantes(enm: dict) -> dict:
    v = {"base": {"quitar": [], "sin_anios": []}}
    for id_ in ("2.1", "2.2", "2.3", "2.4"):
        a = comun.analisis(enm, id_)
        v[id_] = {"quitar": comun.expandir_columnas(a.get("quitar", [])), "sin_anios": a.get("sin_anios_entrenamiento", [])}
    return v


def cohorte_2024(capas: Path, trabajo5: Path, tarea: str, categorias: dict) -> pd.DataFrame:
    """Cohorte de 2024 con el desenlace leído en la apertura (diseño original, rotulado «ya abierto»)."""
    from evaluacion import cohortes
    tope = json.loads((capas / "oro" / "parametros_oro.json").read_text())["tope_estancia_dias"]
    pred = pl.read_parquet(capas / "oro" / "oro_predictores_2024.parquet").drop(["EN_COHORTE_D", "EN_COHORTE_E", "SPLIT"]).to_pandas()
    des = pl.read_parquet(trabajo5 / "evaluacion_2024" / "desenlace_2024.parquet", columns=["ID_EPISODIO", "ROL", "ESTANCIA_DIAS"]).to_pandas()
    mort, est = cohortes.construir(pred, des, tope)
    d = (mort if tarea == "mortalidad" else est).copy()
    for c, cats in categorias.items():
        d[c] = pd.Categorical(d[c].astype(str), categories=cats)
    return d


def _recalibrador(tarea: str):
    from modelado.calibracion import FactorUnico, Platt
    return Platt() if tarea == "mortalidad" else FactorUnico()


def _metricas(tarea: str, y, p) -> dict:
    from modelado.metricas import metricas_estadia, metricas_mortalidad
    m = metricas_mortalidad(y, p) if tarea == "mortalidad" else metricas_estadia(y, p)
    claves = ["auc", "pr_auc", "bss", "pendiente"] if tarea == "mortalidad" else ["rmse", "devianza_poisson"]
    return {k: float(m[k]) for k in claves}


def correr(capas: Path, trabajo4: Path, trabajo5: Path, salida: Path, enm: dict, tareas=("mortalidad", "estadia"),
           disenos=("anidado", "original"), log=print) -> pd.DataFrame:
    """Entrena y evalúa las variantes. Guarda en ``salida`` (interna) las predicciones, las cohortes y las zonas por hospital."""
    from evaluacion import prediccion
    from evaluacion.config import DIR_PLANES_B
    from modelado.candidatos import _estimador_gbm, entrenar
    from modelado.datos import CATEGORICAS, PREDICTORES, cargar_matriz

    prot = comun.protocolo()
    salida.mkdir(parents=True, exist_ok=True)
    filas = []
    for tarea in tareas:
        cands = json.loads((trabajo4 / f"candidatos_{tarea}.json").read_text())
        lgbm, lb = cands["lightgbm"], cands["linea_base"]
        df = cargar_matriz(capas, tarea)
        categorias = {c: list(df[c].cat.categories) for c in CATEGORICAS}
        d24 = cohorte_2024(capas, trabajo5, tarea, categorias) if "original" in disenos else None
        for dis in disenos:
            cfg = DISENOS[dis]
            cal = df[df["ANIO_EGRESO"] == cfg["cal"]]
            prueba = d24 if cfg["prueba"] == 2024 else df[df["ANIO_EGRESO"] == cfg["prueba"]].reset_index(drop=True)
            h, y = prueba["COD_HOSPITAL"].astype(str).to_numpy(), prueba["y"].to_numpy(float)
            pd.DataFrame({"ID_EPISODIO": prueba["ID_EPISODIO"].to_numpy(), "cod_hospital": h, "y": y}).to_parquet(salida / f"cohorte_{tarea}_{dis}.parquet")
            tablas = {}
            for nombre, v in list(variantes(enm).items()) + [("2.5", None)]:
                t0 = time.time()
                if nombre == "2.5" and dis == "original":                    # planes B congelados de la Fase 5: no se reentrenan
                    carpeta = DIR_PLANES_B / ("mortalidad_logistica" if tarea == "mortalidad" else "estadia_poisson")
                    _, p = prediccion.predecir(prueba, carpeta)
                    n_ent = None
                elif nombre == "2.5":                                        # receta de la línea base, entrenada en el diseño anidado
                    ent = df[df["ANIO_EGRESO"].isin(cfg["ent"])]
                    mod = entrenar(lb["tipo"], tarea, ent, lb["params"], comun.SEMILLA_MODELO, 2,
                                   n_arboles=int(lb["n_arboles"]) if lb.get("n_arboles") else None)
                    rec = _recalibrador(tarea).fit(mod.predecir(cal), cal["y"].to_numpy(float))
                    p, n_ent = rec.transformar(mod.predecir(prueba)), len(ent)
                else:
                    cols = [c for c in PREDICTORES if c not in v["quitar"]]
                    ent = df[df["ANIO_EGRESO"].isin([a for a in cfg["ent"] if a not in v["sin_anios"]])]
                    est = _estimador_gbm("lightgbm", tarea, lgbm["params"], comun.SEMILLA_MODELO, 2, int(lgbm["n_arboles"]), None)
                    est.fit(ent[cols], ent["y"].to_numpy())
                    f = (lambda d: est.predict_proba(d[cols])[:, 1]) if tarea == "mortalidad" else (lambda d: est.predict(d[cols]))
                    rec = _recalibrador(tarea).fit(f(cal), cal["y"].to_numpy(float))
                    p, n_ent = rec.transformar(f(prueba)), len(ent)
                p = np.asarray(p, float)
                dif = None
                if nombre == "base" and dis == "original":                   # control: la receta reproduce el modelo congelado
                    _, p_cong = prediccion.predecir(prueba, prediccion.carpeta_final(tarea))
                    dif = float(np.max(np.abs(p_cong - p)))
                    log(f"[control {tarea}] máx |congelado − receta| = {dif:.2e}")
                    if dif > 1e-9:
                        log("  la receta no reproduce el modelo congelado: se usa el congelado como base")
                        p = np.asarray(p_cong, float)
                np.save(salida / f"pred_{tarea}_{dis}_{nombre}.npy", p)
                tab, res = comun.tabla_interna(h, y, p, tarea, prot)
                tab.to_csv(salida / f"zonas_{tarea}_{dis}_{nombre}.csv", index=False)
                tablas[nombre] = tab
                oe, lo, hi = comun.oe_con_ic(h, y, p)
                fila = {"tarea": tarea, "diseno": dis, "analisis": nombre, "n_entrenamiento": n_ent, "n_prueba": int(len(y)),
                        **_metricas(tarea, y, p), "oe_global": oe, "oe_ic_inf": lo, "oe_ic_sup": hi, **comun.resumen_res(res),
                        "control_dif_congelado": dif, "segundos": round(time.time() - t0)}
                if nombre != "base":
                    comp = comun.comparar(tablas["base"], tab)
                    fila.update(comp)
                    fila["cumple_regla"] = comun.cumple_regla(comp, enm)
                filas.append(fila)
                pd.DataFrame(filas).to_csv(salida / "modelo_resultados.csv", index=False)
                log(f"[{tarea} {dis} {nombre}] O/E {oe:.3f} [{lo:.3f}; {hi:.3f}] fuera95 {res['pct_fuera_95']:.3f} ({time.time() - t0:.0f} s)")
    return pd.DataFrame(filas)
