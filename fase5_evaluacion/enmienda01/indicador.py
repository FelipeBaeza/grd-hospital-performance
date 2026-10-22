"""Análisis 2.6 y 3.1 a 3.6 de la enmienda 01: cambios en el indicador, sin reentrenar.

Base de cada diseño: en el original, los modelos congelados aplicados a 2024; en el anidado, la base LightGBM del diseño
anidado aplicada a 2023 (la calcula modelo.py). Se compara con la base del mismo diseño.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr

from . import comun

ANIO = {"anidado": 2023, "original": 2024}
CATALOGO_ANIOS = Path("fase3_preparacion/salidas/tablas/catalogo_hospitales_anio.csv")


def _entrantes(capas: Path, anio: int, ids: np.ndarray) -> np.ndarray:
    ent = pl.read_parquet(capas / "plata" / f"plata_{anio}.parquet", columns=["ID_EPISODIO", "ES_TRASLADO_ENTRANTE"]).to_pandas()
    m = pd.DataFrame({"ID_EPISODIO": ids}).merge(ent, on="ID_EPISODIO", how="left")
    return m["ES_TRASLADO_ENTRANTE"].fillna(False).astype(bool).to_numpy()


def primer_anio_grd(raiz_tesis: Path) -> pd.Series:
    cat = pd.read_csv(raiz_tesis / CATALOGO_ANIOS, dtype={"cod_hospital": str})
    return cat[cat["n_egresos"] > 0].groupby("cod_hospital")["anio"].min()


def _calibracion_estratos(tab: pd.DataFrame, h, y, p, estratos: dict, tarea: str, anio: int, dis: str) -> list[dict]:
    """O/E por estrato de hospitales elegibles, con intervalo bootstrap por hospital. Marca los estratos que contienen un
    hospital con conteo suprimido: su O/E permitiría reconstruirlo y no se publica."""
    filas = []
    d = pd.DataFrame({"h": h, "y": y, "p": p})
    for nombre, mapa in estratos.items():
        for grupo in sorted(set(mapa.values())):
            cods = [c for c, g in mapa.items() if g == grupo]
            k = d["h"].isin(cods).to_numpy()
            oe, lo, hi = comun.oe_con_ic(d["h"][k], d["y"][k], d["p"][k])
            sup = bool(tab.loc[tab["cod_hospital"].isin(cods), "suprimido"].any())
            filas.append({"diseno": dis, "anio": anio, "tarea": tarea, "estrato": nombre, "grupo": grupo, "n_hospitales": len(cods),
                          "oe": oe, "ic_inf": lo, "ic_sup": hi, "contiene_suprimido": sup})
    return filas


def correr(capas: Path, trabajo5: Path, interna: Path, enm: dict, raiz_tesis: Path, tareas=("mortalidad", "estadia"),
           disenos=("anidado", "original"), log=print) -> dict:
    from evaluacion import orquestador, prediccion
    prot, prot_sw = comun.protocolo(), comun.protocolo(winsor=0.0)
    mono = [str(c) for c in comun.analisis(enm, "3.1a")["excluir_hospitales"]]
    p5b = comun.analisis(enm, "3.4c")
    primer = primer_anio_grd(raiz_tesis)
    filas, marcas, estr, sim = [], [], [], []
    bases = {}
    for tarea in tareas:
        for dis in disenos:
            anio = ANIO[dis]
            coh = pd.read_parquet(interna / f"cohorte_{tarea}_{dis}.parquet")
            h, y, ids = coh["cod_hospital"].astype(str).to_numpy(), coh["y"].to_numpy(float), coh["ID_EPISODIO"].to_numpy()
            p = np.load(interna / f"pred_{tarea}_{dis}_base.npy")
            base, r0 = comun.tabla_interna(h, y, p, tarea, prot)
            bases[(tarea, dis)] = base
            filas.append({"tarea": tarea, "diseno": dis, "anio": anio, "analisis": "base", **comun.resumen_res(r0)})
            var = {}
            k = ~_entrantes(capas, anio, ids)
            var["2.6"] = comun.tabla_interna(h[k], y[k], p[k], tarea, prot)
            k = ~np.isin(h, mono)
            var["3.1a"] = comun.tabla_interna(h[k], y[k], p[k], tarea, prot)
            terc = pd.qcut(base["n"], 3, labels=["T1 menor", "T2", "T3 mayor"])
            grupos = dict(zip(base["cod_hospital"], terc.astype(str)))
            partes, t2 = [], []
            for g in ["T1 menor", "T2", "T3 mayor"]:
                kk = np.isin(h, [c for c, gg in grupos.items() if gg == g])
                tb, rs = comun.tabla_interna(h[kk], y[kk], p[kk], tarea, prot)
                partes.append(tb.assign(grupo=g))
                t2.append(rs["tau2"])
            tp = pd.concat(partes, ignore_index=True)
            var["3.1b"] = (tp, {"n_elegibles": len(tp), "tau2": float(np.mean(t2)), "pct_fuera_95": float(tp["zona"].isin(comun.FUERA).mean()),
                                "pct_senal_998": float(tp["zona"].isin(["señal alta", "señal baja"]).mean())})
            var["3.3"] = comun.tabla_interna(h, y, p * y.sum() / p.sum(), tarea, prot)
            var["3.4a"] = comun.tabla_interna(h, y, p, tarea, prot_sw)
            t2_tr = comun.tau2_truncado(base["obs"].to_numpy(), base["esp"].to_numpy(), r0["dispersion"], comun.analisis(enm, "3.4b")["truncamiento"])
            var["3.4b"] = comun.tabla_interna(h, y, p, tarea, prot, tau2_forzado=t2_tr)
            for id_, (tb, rs) in var.items():
                tb.to_csv(interna / f"indicador_{tarea}_{dis}_{id_}.csv", index=False)
                comp = comun.comparar(base, tb)
                filas.append({"tarea": tarea, "diseno": dis, "anio": anio, "analisis": id_, **comun.resumen_res(rs), **comp,
                              "cumple_regla": comun.cumple_regla(comp, enm)})
            # 3.1a: los institutos se informan aparte (zona en el embudo del protocolo)
            for c in base[base["cod_hospital"].isin(mono)].itertuples():
                marcas.append({"tarea": tarea, "diseno": dis, "anio": anio, "analisis": "3.1a", "cod_hospital": c.cod_hospital, "zona_protocolo": c.zona})
            # 3.2: hospitales con menos de 2 años en GRD, por zona
            nuevo = base["cod_hospital"].map(primer) >= anio - 1
            for z, n in base[nuevo].groupby("zona").size().items():
                marcas.append({"tarea": tarea, "diseno": dis, "anio": anio, "analisis": "3.2", "zona_protocolo": z, "n_hospitales_menos_2_anios": int(n)})
            # 3.4c: validación del embudo con la heterogeneidad real
            tau = float(np.sqrt(var["3.4a"][1]["tau2"]))
            sim.append({"tarea": tarea, "diseno": dis, "anio": anio, "tau_sin_winsor": tau,
                        "fuera95_esperado_protocolo": comun.simular_p5b(base["esp"].to_numpy(), r0["dispersion"], tau, prot.hospital["winsor"], p5b["replicas"], p5b["semilla"]),
                        "fuera95_esperado_sin_winsor": comun.simular_p5b(base["esp"].to_numpy(), r0["dispersion"], tau, 0.0, p5b["replicas"], p5b["semilla"]),
                        "fuera95_observado_protocolo": float(r0["pct_fuera_95"])})
            # 3.6: calibración por estrato
            est = {"tercil de egresos": grupos,
                   "tipo de hospital": {c: ("instituto monográfico" if c in mono else "general") for c in base["cod_hospital"]},
                   "años en GRD": {c: ("menos de 2" if primer.get(c, anio) >= anio - 1 else "2 o más") for c in base["cod_hospital"]}}
            kk = np.isin(h, base["cod_hospital"].to_numpy())
            estr += _calibracion_estratos(base, h[kk], y[kk], p[kk], est, tarea, anio, dis)
            log(f"[indicador {tarea} {dis}] fuera95 base {r0['pct_fuera_95']:.3f} | " +
                " | ".join(f"{i} {r[1]['pct_fuera_95']:.3f}" for i, r in var.items()))
    # 3.5: estabilidad 2023–2024 con los modelos congelados (2023 es el ensayo de la Fase 5)
    estab = []
    tope = json.loads((capas / "oro" / "parametros_oro.json").read_text())["tope_estancia_dias"]
    for tarea in tareas:
        if ("original" not in disenos):
            break
        mort, est23, _ = orquestador._cohortes_anio(capas, 2023, tope)
        d23 = mort if tarea == "mortalidad" else est23
        _, p23 = prediccion.predecir(d23, prediccion.carpeta_final(tarea))
        t23, _ = comun.tabla_interna(d23["COD_HOSPITAL"].astype(str).to_numpy(), d23["y"].to_numpy(float), p23, tarea, prot)
        t24 = bases[(tarea, "original")]
        m = t23.merge(t24, on="cod_hospital", suffixes=("_23", "_24"))
        f24 = m[m["zona_24"].isin(comun.FUERA)]
        misma_dir = (f24["zona_23"].str.contains("alta") & f24["zona_24"].str.contains("alta")) | (f24["zona_23"].str.contains("baja") & f24["zona_24"].str.contains("baja"))
        estab.append({"tarea": tarea, "n_comunes": len(m), "rho_oe_2023_2024": float(spearmanr(m["oe_interno_23"], m["oe_interno_24"]).statistic),
                      "n_fuera_2024": int(len(f24)), "fuera_2024_misma_direccion_2023": int(misma_dir.sum())})
        log(f"[3.5 {tarea}] ρ 2023–2024 {estab[-1]['rho_oe_2023_2024']:.3f}; alertas repetidas {estab[-1]['fuera_2024_misma_direccion_2023']}/{len(f24)}")
    out = {"indicador": pd.DataFrame(filas), "marcas": pd.DataFrame(marcas), "estratos": pd.DataFrame(estr),
           "validacion_p5b": pd.DataFrame(sim), "estabilidad": pd.DataFrame(estab)}
    for k, v in out.items():
        v.to_csv(interna / f"{k}_resultados.csv", index=False)
    return out
