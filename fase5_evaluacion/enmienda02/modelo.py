"""Análisis E2.1 a E2.3 de la enmienda 02: la receta congelada reentrenada sobre las cohortes corregidas.

Cada análisis saca los mismos episodios de entrenamiento, calibración y prueba (filtros.py). Se compara con la base de
la enmienda 01 del mismo diseño y la misma tarea, que se lee de su carpeta interna (zonas y predicciones por
hospital). Las métricas del modelo se informan dos veces: en la cohorte corregida y, para la base, restringida a esos
mismos egresos, porque las poblaciones son distintas.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from enmienda01 import comun as c1
from enmienda01 import modelo as m1

from . import comun, filtros

ANALISIS = ("E2.1", "E2.2", "E2.3")


def base_enmienda01(interna01: Path, tarea: str, dis: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    coh = pd.read_parquet(interna01 / f"cohorte_{tarea}_{dis}.parquet")
    coh["cod_hospital"] = coh["cod_hospital"].astype(str)
    p = np.load(interna01 / f"pred_{tarea}_{dis}_base.npy")
    zonas = pd.read_csv(interna01 / f"zonas_{tarea}_{dis}_base.csv", dtype={"cod_hospital": str})
    if len(p) != len(coh):
        raise ValueError(f"la base de la enmienda 01 ({tarea}, {dis}) no cuadra con su cohorte")
    return coh, p, zonas


def correr(capas: Path, trabajo4: Path, trabajo5: Path, interna01: Path, salida: Path, enm: dict, eps: dict,
           tareas=("mortalidad", "estadia"), disenos=("anidado", "original"), log=print) -> pd.DataFrame:
    from modelado.candidatos import _estimador_gbm
    from modelado.datos import CATEGORICAS, PREDICTORES, cargar_matriz

    prot = c1.protocolo()
    salida.mkdir(parents=True, exist_ok=True)
    filas = []
    for tarea in tareas:
        lgbm = json.loads((trabajo4 / f"candidatos_{tarea}.json").read_text())["lightgbm"]
        df = cargar_matriz(capas, tarea)
        categorias = {c: list(df[c].cat.categories) for c in CATEGORICAS}
        d24 = m1.cohorte_2024(capas, trabajo5, tarea, categorias) if "original" in disenos else None
        for dis in disenos:
            cfg = m1.DISENOS[dis]
            coh_b, p_b, zonas_b = base_enmienda01(interna01, tarea, dis)
            for id_ in ANALISIS:
                if tarea not in comun.analisis(enm, id_)["tareas"]:
                    continue
                t0 = time.time()
                fuera = np.concatenate([filtros.quitar(eps[a], id_, tarea) for a in sorted(set(cfg["ent"]) | {cfg["cal"], cfg["prueba"]})])
                d = df[~df["ID_EPISODIO"].isin(fuera)]
                ent, cal = d[d["ANIO_EGRESO"].isin(cfg["ent"])], d[d["ANIO_EGRESO"] == cfg["cal"]]
                prueba = (d24 if cfg["prueba"] == 2024 else df[df["ANIO_EGRESO"] == cfg["prueba"]]).reset_index(drop=True)
                prueba = prueba[~prueba["ID_EPISODIO"].isin(fuera)].reset_index(drop=True)
                if len(fuera) == 0 or len(d) == len(df):
                    raise RuntimeError(f"{id_} ({tarea}, {dis}): el filtro no sacó ningún egreso; revisar los identificadores")
                est = _estimador_gbm("lightgbm", tarea, lgbm["params"], comun.SEMILLA_MODELO, 2, int(lgbm["n_arboles"]), None)
                est.fit(ent[PREDICTORES], ent["y"].to_numpy())
                f = (lambda x: est.predict_proba(x[PREDICTORES])[:, 1]) if tarea == "mortalidad" else (lambda x: est.predict(x[PREDICTORES]))
                rec = m1._recalibrador(tarea).fit(f(cal), cal["y"].to_numpy(float))
                p = np.asarray(rec.transformar(f(prueba)), float)
                h, y = prueba["COD_HOSPITAL"].astype(str).to_numpy(), prueba["y"].to_numpy(float)
                np.save(salida / f"pred_{tarea}_{dis}_{id_}.npy", p)
                pd.DataFrame({"ID_EPISODIO": prueba["ID_EPISODIO"].to_numpy(), "cod_hospital": h, "y": y}).to_parquet(salida / f"cohorte_{tarea}_{dis}_{id_}.parquet")
                tab, res = c1.tabla_interna(h, y, p, tarea, prot)
                tab.to_csv(salida / f"zonas_{tarea}_{dis}_{id_}.csv", index=False)
                comp = c1.comparar(zonas_b, tab)
                kb = coh_b["ID_EPISODIO"].isin(prueba["ID_EPISODIO"]).to_numpy()       # la base, en los mismos egresos
                oe, lo, hi = c1.oe_con_ic(h, y, p)
                fila = {"tarea": tarea, "diseno": dis, "analisis": id_, "n_entrenamiento": int(len(ent)), "n_prueba": int(len(y)),
                        "n_fuera_prueba": int((~kb).sum()),
                        **m1._metricas(tarea, y, p), **{f"{k}_base_mismos_egresos": v for k, v in m1._metricas(tarea, coh_b["y"].to_numpy(float)[kb], p_b[kb]).items()},
                        "oe_global": oe, "oe_ic_inf": lo, "oe_ic_sup": hi, **c1.resumen_res(res), **comp, "cumple_regla": c1.cumple_regla(comp, enm),
                        "segundos": round(time.time() - t0)}
                filas.append(fila)
                pd.DataFrame(filas).to_csv(salida / "modelo_resultados.csv", index=False)
                log(f"[{tarea} {dis} {id_}] O/E {oe:.3f} [{lo:.3f}; {hi:.3f}] misma zona {comp['pct_misma_zona']:.2f} ρ {comp['rho_oe']:.3f} ({time.time() - t0:.0f} s)")
    return pd.DataFrame(filas)
