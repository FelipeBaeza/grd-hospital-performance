"""Auditoría del modelo de la Fase 4 (exploratoria, DESPUÉS de ver los resultados; no modifica los modelos congelados).

Usa solo 2019–2022 para entrenar y 2023 para evaluar. 2024 no se toca (cargar_matriz lo impide).
Uso, desde fase4_modelado/:
    python run_auditoria.py --capas <capas> --trabajo <trabajo> [--tarea mortalidad|estadia] [--pruebas ablacion hospital permutacion]

REGLA DE DECISIÓN (escríbela y fírmala ANTES de correr; cámbiala aquí si quieres otra):
    - Ablación: si quitar CIE10_3C + las comorbilidades sospechosas baja el PR AUC de 2023 menos de 0.02 (mortalidad)
      o sube la devianza de Poisson menos de 1 % (estadía), el modelo NO depende de ellas: se puede usar la versión reducida.
    - Permutación: la MEDIA del AUC 2023 de 8 entrenamientos con etiquetas barajadas debe estar en 0.50 ± 0.02 (un solo AUC puede salir 0.44–0.55 por azar). Si la media se aleja, hay un error de tubería.
    - Por hospital: informar el rango del AUC; un AUC < 0.80 en un hospital con ≥ 20 muertes se anota como limitación.
Salidas: salidas/tablas/auditoria_*.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from modelado.candidatos import entrenar
from modelado.config import cargar_reglas
from modelado.datos import cargar_matriz
from modelado.metricas import metricas_estadia, metricas_mortalidad
from modelado.validacion import mascaras_finales

SOSPECHOSAS = ["ELIX_14", "ELIX_22", "ELIX_24", "ELIX_25", "ELIX_26"]   # insuficiencia renal, coagulopatía, pérdida de peso, hidroelectrolíticos, anemia aguda
DX = ["CIE10_3C"]
VARIANTES = {
    "base": [],
    "sin_diagnostico_principal": DX,
    "sin_comorbilidades_sospechosas": SOSPECHOSAS,
    "sin_ambos": DX + SOSPECHOSAS,
}


def neutralizar(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Anula una columna (misma en entrenamiento y evaluación) sin cambiar el esquema que espera el modelo."""
    d = df.copy()
    for c in columnas:
        if str(d[c].dtype) == "category":
            d[c] = pd.Categorical(["RARO"] * len(d), categories=["RARO"])
        else:
            d[c] = 0
    return d


def evaluar(tarea: str, y, p) -> dict:
    return metricas_mortalidad(y, p) if tarea == "mortalidad" else metricas_estadia(y, p)


def cargar_modelo_ganador(trabajo: Path, tarea: str):
    sel = json.loads((trabajo / f"seleccion_{tarea}.json").read_text(encoding="utf8"))["ganador"]
    c = json.loads((trabajo / f"candidatos_{tarea}.json").read_text(encoding="utf8"))[sel]
    return c["tipo"], c["params"], int(c["n_arboles"])


def prueba_ablacion(df, tr, ca, tipo, params, n_arb, tarea, reglas):
    filas, preds = [], {}
    d_tr, d_ca = df[tr], df[ca].reset_index(drop=True)
    for nombre, quitar in VARIANTES.items():
        t0 = time.time()
        m = entrenar(tipo, tarea, neutralizar(d_tr, quitar), params, reglas.semilla["maestra"], reglas.ajuste["n_hilos"], n_arboles=n_arb)
        p = m.predecir(neutralizar(d_ca, quitar))
        preds[nombre] = p
        met = evaluar(tarea, d_ca["y"].to_numpy(), p)
        filas.append({"variante": nombre, "quitadas": ",".join(quitar) or "-", **met, "segundos": round(time.time() - t0)})
        print(f"[ablacion] {nombre}: {({k: round(v, 4) for k, v in met.items() if k in ('pr_auc', 'auc', 'devianza_poisson', 'rmse')})}", flush=True)
    t = pd.DataFrame(filas)
    if tarea == "mortalidad":
        b = t.loc[t.variante == "base", "pr_auc"].iloc[0]
        t["delta_pr_auc"] = t["pr_auc"] - b
        t["depende"] = t["delta_pr_auc"] < -0.02
    else:
        b = t.loc[t.variante == "base", "devianza_poisson"].iloc[0]
        t["delta_devianza_rel"] = (t["devianza_poisson"] - b) / b
        t["depende"] = t["delta_devianza_rel"] > 0.01
    return t, preds["base"]


def prueba_hospital(d_ca: pd.DataFrame, p: np.ndarray, tarea: str) -> pd.DataFrame:
    d = d_ca[["COD_HOSPITAL", "y"]].copy()
    d["p"] = p
    glob = d["y"].sum() / d["p"].sum()
    filas = []
    for h, g in d.groupby("COD_HOSPITAL", observed=True):
        f = {"cod_hospital": h, "n": len(g), "observado": float(g["y"].sum()), "esperado": float(g["p"].sum())}
        f["oe_crudo"] = f["observado"] / f["esperado"]
        f["oe_relativo_al_global"] = f["oe_crudo"] / glob
        if tarea == "mortalidad" and 0 < g["y"].sum() < len(g):
            f["auc"] = float(roc_auc_score(g["y"], g["p"]))
            f["pr_auc"] = float(average_precision_score(g["y"], g["p"]))
        filas.append(f)
    t = pd.DataFrame(filas).sort_values("n", ascending=False)
    if "auc" in t:
        t["alerta_auc_bajo"] = (t["auc"] < 0.80) & (t["observado"] >= 20)
    return t


def prueba_permutacion(df, tr, ca, tipo, params, tarea, reglas, fraccion=0.2, n_arb=100, repeticiones=8):
    """Varias permutaciones: un solo AUC puede salir lejos de 0.5 por azar (los árboles ajustan ruido por categoría). Se juzga la media."""
    d_ca = df[ca].reset_index(drop=True)
    d_tr0 = df[tr]
    vals = []
    for r in range(repeticiones):
        rng = np.random.default_rng(reglas.semilla["maestra"] + r)
        d_tr = d_tr0.iloc[np.flatnonzero(rng.random(len(d_tr0)) < fraccion)].copy()
        d_tr["y"] = rng.permutation(d_tr["y"].to_numpy())                      # se rompe la relación entre predictores y desenlace
        p = entrenar(tipo, tarea, d_tr, params, reglas.semilla["maestra"] + r, reglas.ajuste["n_hilos"], n_arboles=n_arb).predecir(d_ca)
        vals.append(float(roc_auc_score(d_ca["y"], p)) if tarea == "mortalidad" else float(np.mean(np.abs(d_ca["y"] - p))))
    v = np.array(vals)
    if tarea == "mortalidad":
        res = {"repeticiones": repeticiones, "auc_media": float(v.mean()), "auc_sd": float(v.std(ddof=1)), "auc_min": float(v.min()), "auc_max": float(v.max()),
               "pasa": bool(abs(v.mean() - 0.5) <= max(0.02, 2 * v.std(ddof=1) / np.sqrt(len(v))))}
    else:
        base = float(np.mean(np.abs(d_ca["y"] - d_tr0["y"].mean())))
        res = {"repeticiones": repeticiones, "mae_media": float(v.mean()), "mae_constante": base, "pasa": bool(v.mean() >= base * 0.99)}
    print("[permutacion]", res, flush=True)
    return pd.DataFrame([res])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capas", required=True)
    ap.add_argument("--trabajo", required=True)
    ap.add_argument("--tarea", default="mortalidad", choices=["mortalidad", "estadia"])
    ap.add_argument("--pruebas", nargs="+", default=["ablacion", "hospital", "permutacion"], choices=["ablacion", "hospital", "permutacion"])
    a = ap.parse_args()
    reglas = cargar_reglas()
    salidas = Path(__file__).parent / "salidas" / "tablas"
    salidas.mkdir(parents=True, exist_ok=True)
    df = cargar_matriz(a.capas, a.tarea)
    tr, ca = mascaras_finales(df, reglas)
    tipo, params, n_arb = cargar_modelo_ganador(Path(a.trabajo), a.tarea)
    print(f"{a.tarea}: {tipo}, {n_arb} árboles; entrenamiento {int(tr.sum())} filas, 2023 {int(ca.sum())} filas", flush=True)
    p_base = None
    if "ablacion" in a.pruebas or "hospital" in a.pruebas:
        t, p_base = prueba_ablacion(df, tr, ca, tipo, params, n_arb, a.tarea, reglas)
        t.to_csv(salidas / f"auditoria_ablacion_{a.tarea}.csv", index=False)
    if "hospital" in a.pruebas:
        prueba_hospital(df[ca].reset_index(drop=True), p_base, a.tarea).to_csv(salidas / f"auditoria_hospital_{a.tarea}.csv", index=False)
    if "permutacion" in a.pruebas:
        prueba_permutacion(df, tr, ca, tipo, params, a.tarea, reglas).to_csv(salidas / f"auditoria_permutacion_{a.tarea}.csv", index=False)
    print("LISTO")


if __name__ == "__main__":
    main()
