"""Orquesta la Fase 4: contrato de entrada, ajuste, comparación, selección, calibración, integridad, exhaustividad, congelamiento y verificación."""
from __future__ import annotations

import argparse
import gc
import json
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import calibracion, congelar, exhaustividad, integridad, seleccion, tuning
from .candidatos import Modelo, entrenar
from .config import DIR_MODELOS, DIR_SALIDAS, RAIZ, cargar_reglas, sha256_archivo
from .datos import CATEGORICAS, PREDICTORES, cargar_matriz, huellas_entrada, X
from .metricas import metricas_estadia, metricas_mortalidad, tabla_calibracion
from .validacion import mascaras_finales, pliegues

ETAPAS = ("contrato", "ajuste", "comparacion", "seleccion", "calibracion", "integridad", "exhaustividad", "congelar", "verificacion")
TAREAS = ("mortalidad", "estadia")
FILAS_ESPERADAS = {"mortalidad": 2640061, "estadia": 2396356}
REGISTRO_FASE3 = RAIZ.parent / "fase3_preparacion" / "salidas" / "tablas" / "registro_accesos.csv"


def pico_ram_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def jw(ruta: Path, obj) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding="utf8")


def jr(ruta: Path):
    return json.loads(Path(ruta).read_text(encoding="utf8"))


def csv(df: pd.DataFrame, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False)


def met(tarea: str, y, p, ref=None) -> dict:
    return metricas_mortalidad(y, p, ref) if tarea == "mortalidad" else metricas_estadia(y, p)


def aplanar(m: dict, prefijo: str = "") -> dict:
    return {prefijo + k: v for k, v in m.items()}


# --------------------------------------------------------------------------------------------- referencias
def _ref_vw(tarea: str, df: pd.DataFrame, capas: Path, tr_mask: np.ndarray):
    """Referencia clínica sencilla: puntaje de van Walraven + edad + sexo. No participa en la selección."""
    lb = pl.concat([pl.read_parquet(capas / "oro" / f"linea_base_{a}.parquet", columns=["ID_EPISODIO", "PUNTAJE_VAN_WALRAVEN"]) for a in range(2019, 2024)]).to_pandas()
    d = df[["ID_EPISODIO", "EDAD_ANIOS", "SEXO", "y", "ANIO_EGRESO"]].merge(lb, on="ID_EPISODIO", how="left")
    d["EDAD2"] = d["EDAD_ANIOS"] ** 2
    pre = ColumnTransformer([("n", StandardScaler(), ["PUNTAJE_VAN_WALRAVEN", "EDAD_ANIOS", "EDAD2"]), ("c", OneHotEncoder(handle_unknown="ignore"), ["SEXO"])])
    est = LogisticRegression(C=1.0, max_iter=300) if tarea == "mortalidad" else PoissonRegressor(alpha=1e-6, max_iter=300)
    pipe = Pipeline([("pre", pre), ("m", est)]).fit(d[tr_mask], d.loc[tr_mask, "y"])
    return pipe, d


def _tuning_ridge(df: pd.DataFrame, reglas) -> dict:
    a = reglas.ajuste
    semilla = reglas.semilla["maestra"]
    from .datos import submuestra
    filas = []
    for alpha in a["alpha_lineal"]:
        r = []
        for k, (tr, va) in enumerate(pliegues(df, reglas)):
            idx = np.flatnonzero(tr)
            train = df.iloc[idx[submuestra(len(idx), a["submuestra_ajuste"], semilla + k)]]
            valid = df.iloc[np.flatnonzero(va)]
            m = entrenar("lineal", "estadia", train, {"alpha": alpha}, semilla)
            r.append(metricas_estadia(valid["y"].to_numpy(), m.predecir(valid))["rmse"])
        filas.append({"alpha": alpha, "rmse": float(np.mean(r))})
    t = pd.DataFrame(filas)
    return {"params": {"alpha": float(t.loc[t["rmse"].idxmin(), "alpha"])}, "tabla": t}


# --------------------------------------------------------------------------------------------- etapas
def etapa_contrato(capas: Path, trabajo: Path, salidas: Path, reglas) -> None:
    print("[contrato] comprobando entradas", flush=True)
    huellas = huellas_entrada(capas)
    filas = {}
    for t in TAREAS:
        df = pl.read_parquet(capas / "oro" / f"matriz_{t}.parquet", columns=["ANIO_EGRESO", "SPLIT"])
        filas[t] = {"filas": df.height, "anios": sorted(df["ANIO_EGRESO"].unique().to_list()), "splits": sorted(df["SPLIT"].unique().to_list())}
    registro = sha256_archivo(REGISTRO_FASE3) if REGISTRO_FASE3.exists() else "no existe"
    jw(trabajo / "contrato.json", {"huellas": huellas, "matrices": filas, "registro_accesos_antes": registro, "huella_reglas": reglas.huella()})
    csv(pd.DataFrame([{"archivo": k, "sha256": v} for k, v in huellas.items()]), salidas / "tablas" / "contrato_entrada.csv")


def etapa_ajuste(capas: Path, trabajo: Path, salidas: Path, reglas, tareas, avisar=print) -> None:
    for tarea in tareas:
        df = cargar_matriz(capas, tarea)
        avisar(f"[ajuste] {tarea}: {len(df):,} filas")
        res = tuning.ajustar_lineal(tarea, df, reglas, avisar)
        jw(trabajo / f"ajuste_{tarea}_linea_base.json", {"tipo": res["tipo"], "params": res["params"]})
        csv(res["tabla"], salidas / "tablas" / f"ajuste_linea_base_{tarea}.csv")
        if tarea == "estadia":
            rr = _tuning_ridge(df, reglas)
            jw(trabajo / "ajuste_estadia_lineal.json", {"tipo": "lineal", "params": rr["params"]})
            csv(rr["tabla"], salidas / "tablas" / "ajuste_lineal_estadia.csv")
        for tipo in ("lightgbm", "xgboost"):
            ruta = trabajo / f"ajuste_{tarea}_{tipo}.json"
            if ruta.exists():
                avisar(f"[ajuste] {tarea}/{tipo} ya estaba hecho")
                continue
            r = tuning.ajustar_gbm(tarea, tipo, df, reglas, trabajo / "optuna", avisar=avisar)
            csv(r["tabla"], salidas / "tablas" / f"ajuste_pruebas_{tarea}_{tipo}.csv")
            jw(ruta, {"tipo": tipo, "params": r["params"], "mejor_valor_submuestra": r["mejor_valor_submuestra"]})
        del df
        gc.collect()


def etapa_comparacion(capas: Path, trabajo: Path, salidas: Path, reglas, tareas, avisar=print) -> None:
    semilla = reglas.semilla["maestra"]
    a = reglas.ajuste
    for tarea in tareas:
        df = cargar_matriz(capas, tarea)
        tr_m, ca_m = mascaras_finales(df, reglas)
        entren, calib = df[tr_m], df[ca_m]
        folds = pliegues(df, reglas)
        filas, preds, scores = [], {}, {}
        bl_tipo = "logistica" if tarea == "mortalidad" else "poisson"
        candidatos = {"constante": ("constante", {}), "linea_base": (bl_tipo, jr(trabajo / f"ajuste_{tarea}_linea_base.json")["params"])}
        if tarea == "estadia":
            candidatos["lineal"] = ("lineal", jr(trabajo / "ajuste_estadia_lineal.json")["params"])
        for tipo in ("lightgbm", "xgboost"):
            candidatos[tipo] = (tipo, jr(trabajo / f"ajuste_{tarea}_{tipo}.json")["params"])
        for nombre, (tipo, params) in candidatos.items():
            avisar(f"[comparacion] {tarea}/{nombre}")
            t0 = time.time()
            if tipo in ("lightgbm", "xgboost"):
                rc = tuning.reajuste_completo(tarea, tipo, df, params, reglas)
                mets_f, n_arb = rc["metricas_pliegues"], rc["n_arboles_final"]
            else:
                mets_f, n_arb = [], None
                for tr, va in folds:
                    m = entrenar(tipo, tarea, df[tr], params, semilla, a["n_hilos"])
                    mets_f.append(met(tarea, df.loc[va, "y"].to_numpy(), m.predecir(df[va]), float(df.loc[tr, "y"].mean())))
            modelo = entrenar(tipo, tarea, entren, params, semilla, a["n_hilos"], n_arboles=n_arb)
            p = modelo.predecir(calib)
            m23 = met(tarea, calib["y"].to_numpy(), p, float(entren["y"].mean()))
            clave = tuning.metrica_seleccion(tarea, m23)
            sc = float(np.mean([tuning.metrica_seleccion(tarea, m) for m in mets_f]))
            scores[nombre] = {"score": sc, "metricas_2023": m23, "tipo": tipo, "params": params, "n_arboles": n_arb}
            preds[nombre] = p
            fila = {"candidato": nombre, "algoritmo": tipo, "n_arboles": n_arb, "segundos": round(time.time() - t0)}
            for k, m in enumerate(mets_f):
                fila[f"pliegue{k + 1}_{'pr_auc' if tarea == 'mortalidad' else 'devianza'}"] = tuning.metrica_seleccion(tarea, m)
            fila["media_pliegues"] = sc
            fila.update(aplanar(m23, "2023_"))
            filas.append(fila)
            joblib.dump(modelo, trabajo / f"modelo_{tarea}_{nombre}.joblib")
            gc.collect()
        # referencia clínica (no participa en la selección)
        vw, d = _ref_vw(tarea, df, capas, tr_m)
        pv = vw.predict_proba(d[ca_m])[:, 1] if tarea == "mortalidad" else vw.predict(d[ca_m])
        mv = met(tarea, calib["y"].to_numpy(), pv, float(entren["y"].mean()))
        filas.append({"candidato": "van_walraven_referencia", "algoritmo": "referencia", **aplanar(mv, "2023_")})
        t = pd.DataFrame(filas)
        csv(t, salidas / "tablas" / f"comparacion_{tarea}.csv")
        jw(trabajo / f"candidatos_{tarea}.json", scores)
        np.savez_compressed(trabajo / f"pred2023_{tarea}.npz", **preds, y=calib["y"].to_numpy(), ids=calib["ID_EPISODIO"].to_numpy().astype(str))
        del df, entren, calib
        gc.collect()


def etapa_seleccion(trabajo: Path, salidas: Path, reglas, tareas) -> None:
    for tarea in tareas:
        c = jr(trabajo / f"candidatos_{tarea}.json")
        cand = {n: c[n] for n in ("linea_base", "lightgbm", "xgboost")}
        r = seleccion.aplicar_regla(tarea, cand, reglas)
        jw(trabajo / f"seleccion_{tarea}.json", r)
        csv(pd.DataFrame(r["pasos"]).assign(ganador_final=r["ganador"]), salidas / "tablas" / f"seleccion_{tarea}.csv")
        print(f"[seleccion] {tarea}: gana {r['ganador']}", flush=True)


def _fechas_2023(capas: Path, ids: np.ndarray) -> np.ndarray:
    f = pl.read_parquet(capas / "plata" / "plata_2023.parquet", columns=["ID_EPISODIO", "FECHA_INGRESO"])
    orden = pl.DataFrame({"ID_EPISODIO": ids}).join(f, on="ID_EPISODIO", how="left")
    return orden["FECHA_INGRESO"].to_numpy()


def etapa_calibracion(capas: Path, trabajo: Path, salidas: Path, reglas, tareas) -> None:
    u = reglas.umbrales
    for tarea in tareas:
        sel = jr(trabajo / f"seleccion_{tarea}.json")["ganador"]
        z = np.load(trabajo / f"pred2023_{tarea}.npz", allow_pickle=False)
        pred, y, ids = z[sel], z["y"].astype(float), z["ids"]
        fechas = pd.Series(_fechas_2023(capas, ids))
        bloques = calibracion.bloques_temporales(fechas.fillna(fechas.min()), reglas.calibracion["bloques"])
        cmp = calibracion.comparar_metodos(tarea, pred, y, bloques, reglas.calibracion["empate_relativo"])
        elegido = cmp["elegido"]
        oof = cmp["oof"][elegido]
        m_oof = met(tarea, y, oof)
        ok_pend = (u["pendiente_min"] <= m_oof["pendiente"] <= u["pendiente_max"]) if tarea == "mortalidad" else True
        ok_oe = u["oe_min"] <= m_oof["oe_global"] <= u["oe_max"]
        anios = np.full(len(y), 2023)
        rec = calibracion.ajustar_final(tarea, elegido, pred, y, anios)
        calibrada = rec.transformar(pred)
        joblib.dump(rec, trabajo / f"recalibrador_{tarea}.joblib")
        filas = [{"metodo": k, **aplanar(v)} for k, v in cmp["metricas"].items()]
        csv(pd.DataFrame(filas).assign(elegido=lambda d: d["metodo"] == elegido), salidas / "tablas" / f"calibracion_metodos_{tarea}.csv")
        cur = pd.concat([tabla_calibracion(y, pred).assign(version="sin_calibrar"), tabla_calibracion(y, oof).assign(version="recalibrada_fuera_de_muestra")])
        csv(cur, salidas / "tablas" / f"calibracion_deciles_{tarea}.csv")
        jw(trabajo / f"calibracion_{tarea}.json", {"modelo": sel, "metodo": elegido, "metricas_fuera_de_muestra": m_oof, "cumple_pendiente": bool(ok_pend), "cumple_oe": bool(ok_oe)})
        np.savez_compressed(trabajo / f"pred2023_cal_{tarea}.npz", cruda=pred, calibrada=calibrada, y=y, ids=ids)
        print(f"[calibracion] {tarea}: {elegido}; OE fuera de muestra {m_oof['oe_global']:.3f}", flush=True)


def etapa_integridad(capas: Path, trabajo: Path, salidas: Path, reglas, tareas) -> None:
    df = cargar_matriz(capas, "mortalidad")
    r = [integridad.prueba_identificacion_hospital(df, reglas, True), integridad.prueba_identificacion_hospital(df, reglas, False)]
    dif = (r[0]["exactitud_top1"] - r[1]["exactitud_top1"]) * 100
    csv(pd.DataFrame(r), salidas / "tablas" / "integridad_identificacion_hospital.csv")
    jw(trabajo / "integridad_hospital.json", {"resultados": r, "diferencia_pp": dif, "umbral_pp": reglas.integridad["max_dif_exactitud_pp"], "cumple": bool(dif <= reglas.integridad["max_dif_exactitud_pp"])})
    del df
    gc.collect()
    for tarea in tareas:
        df = cargar_matriz(capas, tarea)
        calib = df[df["ANIO_EGRESO"] == 2023].reset_index(drop=True)
        sel = jr(trabajo / f"seleccion_{tarea}.json")["ganador"]
        modelo = joblib.load(trabajo / f"modelo_{tarea}_{sel}.joblib")
        t = integridad.contribuciones(modelo, calib, reglas.integridad["muestra_contribuciones"], reglas.semilla["maestra"])
        csv(t, salidas / "tablas" / f"contribuciones_{tarea}.csv")
        z = np.load(trabajo / f"pred2023_cal_{tarea}.npz", allow_pickle=False)
        for nombre, tabla in integridad.coherencia(calib, z["calibrada"], tarea).items():
            csv(tabla, salidas / "tablas" / f"coherencia_{tarea}_{nombre}.csv")
        prohibidas = set(t["variable"]) & {"COD_HOSPITAL", "ANIO_EGRESO", "SPLIT", "ID_EPISODIO"}
        jw(trabajo / f"integridad_{tarea}.json", {"prohibidas_en_contribuciones": sorted(prohibidas), "top5": t.head(5)["variable"].tolist()})
        del df, calib
        gc.collect()


def etapa_exhaustividad(capas: Path, trabajo: Path, salidas: Path, reglas) -> None:
    print("[exhaustividad] cargando 2019–2023", flush=True)
    voc = exhaustividad.vocabulario_cie(RAIZ.parent / "fase3_preparacion" / "salidas" / "tablas" / "oro_vocabularios.csv")
    partes = [exhaustividad.cargar_anio(capas, a, reglas, voc) for a in range(2019, 2024)]
    todo = pd.concat(partes, ignore_index=True)
    ent, ev = todo[todo["ANIO"] <= 2022], todo[todo["ANIO"] == 2023]
    r = exhaustividad.entrenar_y_evaluar(ent, ev, reglas)
    factor = float(ev["y"].sum() / r["pred_glm"].sum())              # nivel global de 2023 (la codificación crece cada año); el índice por hospital es relativo a este nivel
    joblib.dump({"modelo": r["glm"], "factor_global_2023": factor, "excluir_prefijos": reglas.exhaustividad["excluir_prefijos"], "bandas_edad": reglas.exhaustividad["bandas_edad"], "vocabulario_cie": sorted(voc)},
                trabajo / "exhaustividad.joblib")
    csv(pd.DataFrame([{"modelo": k, **v} for k, v in r["metricas"].items()]), salidas / "tablas" / "exhaustividad_modelos.csv")
    resumen = {"n_entrenamiento": int(len(ent)), "n_evaluacion": int(len(ev)), "media_observada_2023": float(ev["y"].mean()), "factor_global_2023": factor, "metricas": r["metricas"]}
    jw(trabajo / "exhaustividad.json", resumen)


def _entrenar_final(capas: Path, trabajo: Path, reglas, tarea: str):
    df = cargar_matriz(capas, tarea)
    tr_m, ca_m = mascaras_finales(df, reglas)
    sel = jr(trabajo / f"seleccion_{tarea}.json")["ganador"]
    c = jr(trabajo / f"candidatos_{tarea}.json")[sel]
    m = entrenar(c["tipo"], tarea, df[tr_m], c["params"], reglas.semilla["maestra"], reglas.ajuste["n_hilos"], n_arboles=c["n_arboles"])
    return m, df[ca_m].reset_index(drop=True), sel


def etapa_congelar(capas: Path, trabajo: Path, salidas: Path, reglas, tareas, modelos_dir: Path) -> None:
    """Reentrena dos veces el modelo elegido (control de reproducibilidad), guarda artefactos con huella y recarga para comprobar que predicen igual."""
    if modelos_dir.exists():
        shutil.rmtree(modelos_dir)
    modelos_dir.mkdir(parents=True)
    archivos, repro, extra = {}, {}, {}
    for tarea in tareas:
        print(f"[congelar] {tarea}: entrenamiento 1", flush=True)
        m1, calib, sel = _entrenar_final(capas, trabajo, reglas, tarea)
        p1 = m1.predecir(calib)
        print(f"[congelar] {tarea}: entrenamiento 2", flush=True)
        m2, _, _ = _entrenar_final(capas, trabajo, reglas, tarea)
        p2 = m2.predecir(calib)
        rec = joblib.load(trabajo / f"recalibrador_{tarea}.joblib")
        c = jr(trabajo / f"calibracion_{tarea}.json")
        carpeta = modelos_dir / tarea
        arch = congelar.guardar_modelo(m1, rec, carpeta)
        recargado = joblib.load(arch["modelo_completo"])
        p3 = recargado.predecir(calib)
        z = np.load(trabajo / f"pred2023_cal_{tarea}.npz", allow_pickle=False)
        dif_cal = float(np.max(np.abs(rec.transformar(p3) - z["calibrada"])))
        repro[tarea] = {"modelo": sel, "max_dif_entrenamientos": float(np.max(np.abs(p1 - p2))), "max_dif_recarga": float(np.max(np.abs(p1 - p3))), "max_dif_prediccion_calibrada": dif_cal,
                        "n_filas_comparadas": int(len(p1))}
        limites = ["Entrenado con 2019–2022; el nivel de mortalidad y de estadía cambió en 2023 y se corrigió con una recalibración global.",
                   "La codificación de diagnósticos crece cada año; los años no son totalmente comparables.",
                   "Solo adultos (18 años o más) hospitalizados; no sirve para niños ni recién nacidos.",
                   "El hospital no es un predictor; el modelo no distingue diferencias de recursos o de población entre hospitales.",
                   "No se evaluó con 2024: esa prueba se hace una sola vez en la Fase 5."]
        (carpeta / "ficha_modelo.md").write_text(congelar.ficha_modelo(tarea, m1, c["metodo"], c["metricas_fuera_de_muestra"], PREDICTORES, limites, reglas.huella()), encoding="utf8")
        arch["ficha"] = carpeta / "ficha_modelo.md"
        archivos[tarea] = arch
        extra[tarea] = {"modelo": sel, "recalibrador": c["metodo"], "n_arboles": m1.n_arboles}
        del calib
        gc.collect()
    carpeta = modelos_dir / "exhaustividad"
    carpeta.mkdir()
    shutil.copy(trabajo / "exhaustividad.joblib", carpeta / "modelo_exhaustividad.joblib")
    archivos["exhaustividad"] = {"modelo": carpeta / "modelo_exhaustividad.joblib"}
    congelar.escribir_manifiesto(modelos_dir, reglas, archivos, extra)
    jw(trabajo / "reproducibilidad.json", repro)
    jw(trabajo / "reproducibilidad_tuning.json", {"nota": "las pruebas de ajuste usan una semilla fija (TPE) y detección temprana determinista"})


def etapa_verificacion(trabajo: Path, salidas: Path, reglas, modelos_dir: Path, con_pytest: bool = True) -> pd.DataFrame:
    from .verificacion import verificar
    if con_pytest:
        r = subprocess.run([sys.executable, "-m", "pytest", "-o", "addopts=-p no:cacheprovider", "--tb=line"], cwd=RAIZ, capture_output=True, text=True)
        ult = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        import re
        pas = int(re.search(r"(\d+) passed", ult).group(1)) if re.search(r"(\d+) passed", ult) else 0
        fal = int(re.search(r"(\d+) failed", ult).group(1)) if re.search(r"(\d+) failed", ult) else (0 if r.returncode == 0 else 1)
        jw(trabajo / "pytest_resumen.json", {"pasaron": pas, "fallaron": fal, "linea_final": ult})
    v = verificar(trabajo, salidas, reglas, modelos_dir, REGISTRO_FASE3)
    csv(v, salidas / "tablas" / "tabla_verificacion_fase4.csv")
    return v


# --------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fase 4 (Modelado)")
    ap.add_argument("--capas", required=True, help="carpeta con las capas de la Fase 3 (oro/, plata/, cohortes/)")
    ap.add_argument("--trabajo", required=True, help="carpeta de trabajo (bases de Optuna, predicciones intermedias)")
    ap.add_argument("--etapas", nargs="*", default=list(ETAPAS))
    ap.add_argument("--tareas", nargs="*", default=list(TAREAS))
    ap.add_argument("--reglas", default=None, help="TOML alternativo de reglas (solo para pruebas rápidas)")
    ap.add_argument("--sin-pytest", action="store_true")
    ap.add_argument("--salidas", default=str(DIR_SALIDAS))
    ap.add_argument("--modelos", default=str(DIR_MODELOS))
    a = ap.parse_args(argv)
    capas, trabajo, salidas, modelos = Path(a.capas), Path(a.trabajo), Path(a.salidas), Path(a.modelos)
    reglas = cargar_reglas(a.reglas)
    trabajo.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tiempos = []
    for e in a.etapas:
        ti = time.time()
        if e == "contrato":
            etapa_contrato(capas, trabajo, salidas, reglas)
        elif e == "ajuste":
            etapa_ajuste(capas, trabajo, salidas, reglas, a.tareas)
        elif e == "comparacion":
            etapa_comparacion(capas, trabajo, salidas, reglas, a.tareas)
        elif e == "seleccion":
            etapa_seleccion(trabajo, salidas, reglas, a.tareas)
        elif e == "calibracion":
            etapa_calibracion(capas, trabajo, salidas, reglas, a.tareas)
        elif e == "integridad":
            etapa_integridad(capas, trabajo, salidas, reglas, a.tareas)
        elif e == "exhaustividad":
            etapa_exhaustividad(capas, trabajo, salidas, reglas)
        elif e == "congelar":
            etapa_congelar(capas, trabajo, salidas, reglas, a.tareas, modelos)
        elif e == "verificacion":
            print(etapa_verificacion(trabajo, salidas, reglas, modelos, not a.sin_pytest), flush=True)
        tiempos.append({"etapa": e, "segundos": round(time.time() - ti, 1), "pico_ram_gb": round(pico_ram_gb(), 2)})
        prev = []
        tf = salidas / "tablas" / "tiempos_etapas_fase4.csv"
        if tf.exists():
            prev = pd.read_csv(tf).to_dict("records")
        csv(pd.DataFrame([x for x in prev if x["etapa"] != e] + [tiempos[-1]]), tf)
    print(f"listo en {time.time() - t0:.0f} s; pico de RAM {pico_ram_gb():.2f} GB", flush=True)
    return 0
