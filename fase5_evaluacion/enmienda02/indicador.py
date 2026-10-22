"""Análisis E2.4 a E2.6 de la enmienda 02: cambios en el indicador, sin reentrenar.

La base de cada diseño es la de la enmienda 01 (su carpeta interna): en el original, los modelos congelados en 2024;
en el anidado, la base LightGBM del diseño anidado en 2023.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from enmienda01 import comun as c1
from enmienda01 import indicador as i1

from . import comun, filtros
from .modelo import base_enmienda01

ANIO = {"anidado": 2023, "original": 2024}


def _pendiente(tarea, dis, id_, motivo) -> dict:
    return {"tarea": tarea, "diseno": dis, "anio": ANIO[dis], "analisis": id_, "estado": f"pendiente: {motivo}"}


def correr(capas, trabajo5, interna01, salida, enm, eps: dict, raiz_tesis, usar_deis: bool,
           tareas=("mortalidad", "estadia"), disenos=("anidado", "original"), log=print) -> dict:
    prot = c1.protocolo()
    salida.mkdir(parents=True, exist_ok=True)
    primer = i1.primer_anio_grd(raiz_tesis)
    deis = pd.read_csv(comun.CATALOGO_DEIS, dtype={"cod_hospital": str}) if usar_deis else None
    filas, marcas, correl, tasas = [], [], [], []
    for dis in disenos:
        anio = ANIO[dis]
        ctx = filtros.tasas_contexto(eps[anio]).assign(diseno=dis, anio=anio)
        tasas.append(ctx)
        for tarea in tareas:
            coh, p, base = base_enmienda01(interna01, tarea, dis)
            h, y, ids = coh["cod_hospital"].to_numpy(), coh["y"].to_numpy(float), coh["ID_EPISODIO"].to_numpy()
            var = {}
            # E2.4a y E2.4b: catálogo DEIS
            if deis is None:
                filas += [_pendiente(tarea, dis, i, "falta fase3_preparacion/config/tipo_establecimiento.csv") for i in ("E2.4a", "E2.4b")]
            else:
                tipo = dict(zip(deis["cod_hospital"], deis["tipo"]))
                partes, t2 = [], []
                for g in sorted(set(tipo.get(c, "sin clasificar") for c in base["cod_hospital"])):
                    kk = np.array([tipo.get(c, "sin clasificar") == g for c in h])
                    if kk.sum() == 0:
                        continue
                    tb, rs = c1.tabla_interna(h[kk], y[kk], p[kk], tarea, prot)
                    partes.append(tb.assign(grupo=g))
                    t2.append(rs["tau2"])
                tp = pd.concat(partes, ignore_index=True)
                var["E2.4a"] = (tp, {"n_elegibles": len(tp), "tau2": float(np.mean(t2)), "pct_fuera_95": float(tp["zona"].isin(c1.FUERA).mean()),
                                     "pct_senal_998": float(tp["zona"].isin(["señal alta", "señal baja"]).mean())})
                mono = [c for c, t in tipo.items() if t == "instituto monográfico"]
                k = ~np.isin(h, mono)
                var["E2.4b"] = c1.tabla_interna(h[k], y[k], p[k], tarea, prot)
                for c in base[base["cod_hospital"].isin(mono)].itertuples():
                    marcas.append({"tarea": tarea, "diseno": dis, "anio": anio, "analisis": "E2.4b", "cod_hospital": c.cod_hospital, "zona_protocolo": c.zona})
            # E2.5: sin los hospitales con menos de 2 años en GRD
            nuevos = [c for c in base["cod_hospital"] if primer.get(c, anio) >= anio - 1]
            k = ~np.isin(h, nuevos)
            var["E2.5"] = c1.tabla_interna(h[k], y[k], p[k], tarea, prot)
            for c in base[base["cod_hospital"].isin(nuevos)].itertuples():
                marcas.append({"tarea": tarea, "diseno": dis, "anio": anio, "analisis": "E2.5", "cod_hospital": c.cod_hospital, "zona_protocolo": c.zona})
            # E2.6b: mortalidad sin Z51.5 en cualquier diagnóstico
            if tarea == "mortalidad":
                k = ~np.isin(ids, filtros.quitar(eps[anio], "E2.6b", tarea))
                var["E2.6b"] = c1.tabla_interna(h[k], y[k], p[k], tarea, prot)
            for id_, (tb, rs) in var.items():
                tb.to_csv(salida / f"indicador_{tarea}_{dis}_{id_}.csv", index=False)
                comp = c1.comparar(base, tb)
                filas.append({"tarea": tarea, "diseno": dis, "anio": anio, "analisis": id_, "estado": "corrido", **c1.resumen_res(rs), **comp,
                              "cumple_regla": c1.cumple_regla(comp, enm)})
            # E2.6a: correlación de las tasas de contexto con el O/E de la base
            m = base.merge(ctx, on="cod_hospital")
            for t in ("tasa_z515", "tasa_alta_domiciliaria", "tasa_voluntaria_fuga"):
                r = spearmanr(m["oe_interno"], m[t]).statistic if len(m) > 2 and m[t].nunique() > 1 else np.nan
                correl.append({"tarea": tarea, "diseno": dis, "anio": anio, "analisis": "E2.6a", "tasa": t, "comparados": int(len(m)), "rho_con_oe": float(r)})
            log(f"[e02 indicador {tarea} {dis}] " + " | ".join(f"{i} fuera95 {r[1]['pct_fuera_95']:.3f}" for i, r in var.items()))
    out = {"indicador": pd.DataFrame(filas), "marcas": pd.DataFrame(marcas), "correlaciones_contexto": pd.DataFrame(correl),
           "tasas_contexto": pd.concat(tasas, ignore_index=True)}
    for k, v in out.items():
        v.to_csv(salida / f"{k}_resultados.csv", index=False)
    return out
