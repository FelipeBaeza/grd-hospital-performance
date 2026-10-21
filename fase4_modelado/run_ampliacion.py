"""Ampliación exploratoria: más algoritmos de árboles con el mismo protocolo (pliegues hacia adelante; 2023 y 2024 no se usan).
Uso: python run_ampliacion.py --capas <capas> --trabajo <trabajo> [--tareas mortalidad estadia] [--tipos lgbm_extra lgbm_rf catboost]"""
import argparse, json, time
from pathlib import Path
import pandas as pd
from modelado import ampliacion as A
from modelado.config import cargar_reglas
from modelado.datos import cargar_matriz

p = argparse.ArgumentParser()
p.add_argument("--capas", required=True); p.add_argument("--trabajo", required=True)
p.add_argument("--tareas", nargs="+", default=["mortalidad", "estadia"]); p.add_argument("--tipos", nargs="+", default=list(A.NUEVOS))
a = p.parse_args()
reglas = cargar_reglas(); trabajo = Path(a.trabajo); salidas = Path(__file__).parent / "salidas" / "tablas"
alm = trabajo / "optuna_ampl"; alm.mkdir(parents=True, exist_ok=True)
for tarea in a.tareas:
    df = cargar_matriz(a.capas, tarea)
    n = reglas.ajuste["pruebas_mortalidad"] if tarea == "mortalidad" else reglas.ajuste["pruebas_estadia"]
    for tipo in a.tipos:
        f = trabajo / f"ampl_{tarea}_{tipo}.json"
        if f.exists():
            continue
        r = A.ajustar(tarea, tipo, df, reglas, alm, n)
        r["tabla"].to_csv(salidas / f"ampliacion_pruebas_{tarea}_{tipo}.csv", index=False)
        t0 = time.time(); rc = A.reajuste_completo(tipo, tarea, df, r["params"], reglas)
        clave = "pr_auc" if tarea == "mortalidad" else "devianza_poisson"
        out = {"tarea": tarea, "tipo": tipo, "params": r["params"], "pruebas": n, "score_submuestra": r["mejor_valor_submuestra"], "score_pliegues_completo": rc["score"],
               "por_pliegue": [m[clave] for m in rc["metricas_pliegues"]], "n_arboles": rc["n_arboles"], "segundos_pliegues": rc["segundos"]}
        f.write_text(json.dumps(out, indent=1))
        print("LISTO", tarea, tipo, out["score_pliegues_completo"], flush=True)
