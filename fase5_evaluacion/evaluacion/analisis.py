"""Análisis completo de un año con desenlace (ensayo con 2023 o evaluación real de 2024). Escribe tablas y figuras en ``destino``; no entrena ni recalibra nada."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import criterios, embudo, exhaustividad_indice, hospitales, prediccion, subgrupos
from .config import DIR_PLANES_B


def tabla_traslados(df_c: pd.DataFrame, traslados: pd.DataFrame, min_egresos: int) -> pd.DataFrame:
    """Tasa de traslado a otro hospital sobre la cohorte C completa (la cohorte D deja fuera a los trasladados por construcción)."""
    c = df_c[["ID_EPISODIO", "COD_HOSPITAL"]].merge(traslados, on="ID_EPISODIO", how="left")
    c["ES_TRASLADO_SALIENTE"] = c["ES_TRASLADO_SALIENTE"].fillna(False).astype(float)
    g = c.groupby("COD_HOSPITAL").agg(n=("ES_TRASLADO_SALIENTE", "size"), tasa_traslado_saliente=("ES_TRASLADO_SALIENTE", "mean")).reset_index().rename(columns={"COD_HOSPITAL": "cod_hospital"})
    return g[g["n"] >= min_egresos].drop(columns="n").sort_values("cod_hospital").reset_index(drop=True)


def evaluar(mort: pd.DataFrame, est: pd.DataFrame, df_exh: pd.DataFrame, traslados: pd.DataFrame | None, prot, destino: Path, B: int | None = None, planes_b: Path = DIR_PLANES_B, entrantes: pd.DataFrame | None = None) -> dict:
    """``mort`` y ``est``: cohortes D y E del año con columna ``y``. ``df_exh``: cohorte C del año para el índice de exhaustividad. ``traslados``: ID_EPISODIO y ES_TRASLADO_SALIENTE."""
    t = destino / "tablas"
    t.mkdir(parents=True, exist_ok=True)
    res = {}
    # ---- mortalidad
    ym, hm = mort["y"].to_numpy(float), mort["COD_HOSPITAL"].to_numpy()
    _, pm = prediccion.predecir(mort, prediccion.carpeta_final("mortalidad"))
    _, pl_ = prediccion.predecir(mort, planes_b / "mortalidad_logistica")
    crit_m = criterios.evaluar_mortalidad(ym, pm, pl_, hm, prot, B)
    crit_m.to_csv(t / "criterios_mortalidad.csv", index=False)
    tab_m, res_m = hospitales.tabla(hm, ym, pm, "mortalidad", prot)
    tab_m.to_csv(t / "oe_hospital_mortalidad.csv", index=False)
    subgrupos.tabla(mort, ym, pm, "mortalidad", prot.hospital["celda_minima"]).to_csv(t / "subgrupos_mortalidad.csv", index=False)
    # ---- estadía
    ye, he = est["y"].to_numpy(float), est["COD_HOSPITAL"].to_numpy()
    _, me = prediccion.predecir(est, prediccion.carpeta_final("estadia"))
    bases = {n: prediccion.predecir(est, planes_b / f"estadia_{n}")[1] for n in ("lineal", "poisson")}
    crit_e = criterios.evaluar_estadia(ye, me, bases, he, prot, B)
    crit_e.to_csv(t / "criterios_estadia.csv", index=False)
    tab_e, res_e = hospitales.tabla(he, ye, me, "estadia", prot)
    tab_e.to_csv(t / "oe_hospital_estadia.csv", index=False)
    subgrupos.tabla(est, ye, me, "estadia", prot.hospital["celda_minima"]).to_csv(t / "subgrupos_estadia.csv", index=False)
    # ---- exhaustividad y traslados
    ind, res_x = exhaustividad_indice.indice(df_exh)
    ind[ind["n_egresos"] >= prot.hospital["min_egresos"]].sort_values("cod_hospital").to_csv(t / "indice_exhaustividad_hospital.csv", index=False)
    if traslados is not None:
        tabla_traslados(df_exh, traslados, prot.hospital["min_egresos"]).to_csv(t / "tasa_traslado_hospital.csv", index=False)
    if entrantes is not None:  # decisión D9: proporción de traslados entrantes por hospital (contexto; no se excluyen)
        c = mort[["ID_EPISODIO", "COD_HOSPITAL"]].merge(entrantes[["ID_EPISODIO", "ES_TRASLADO_ENTRANTE"]], on="ID_EPISODIO", how="left")
        c["ES_TRASLADO_ENTRANTE"] = c["ES_TRASLADO_ENTRANTE"].fillna(False).astype(float)
        g = c.groupby("COD_HOSPITAL").agg(n=("ES_TRASLADO_ENTRANTE", "size"), proporcion_traslado_entrante=("ES_TRASLADO_ENTRANTE", "mean")).reset_index().rename(columns={"COD_HOSPITAL": "cod_hospital"})
        g[g["n"] >= prot.hospital["min_egresos"]].drop(columns="n").sort_values("cod_hospital").to_csv(t / "traslado_entrante_hospital.csv", index=False)
    # ---- calibración y deriva
    from modelado.metricas import tabla_calibracion
    cal = tabla_calibracion(ym, pm)
    pequena = (cal["observado"] * cal["n"]).round() < prot.hospital["celda_minima"]       # deciles con menos de 10 muertes: se oculta lo observado
    cal.loc[pequena, ["observado", "razon_oe"]] = np.nan
    cal.to_csv(t / "calibracion_deciles_mortalidad.csv", index=False)
    tabla_calibracion(ye, me).to_csv(t / "calibracion_deciles_estadia.csv", index=False)
    res = {"n_mortalidad": int(len(mort)), "n_estadia": int(len(est)), "hospitales_mortalidad": res_m, "hospitales_estadia": res_e, "exhaustividad": res_x,
           "oe_global_mortalidad": float(ym.sum() / pm.sum()), "oe_global_estadia": float(ye.sum() / me.sum())}
    return {"resumen": res, "pred": {"pm": pm, "pe": me}}
