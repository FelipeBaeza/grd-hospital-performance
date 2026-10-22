"""Resultados publicables de la enmienda 02. Mismas reglas que la enmienda 01, más dos:

- Una tasa de contexto por hospital se publica solo si su conteo es 0 o al menos 10. Con un conteo de 1 a 9, la tasa
  por el número de egresos lo reconstruiría.
- De las muertes obstétricas que salen de la cohorte D se publica solo el total 2019–2023 (o «<10»).
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from enmienda01.publicar import RENOMBRE as RENOMBRE01

RENOMBRE = {**RENOMBRE01, "n_fuera_prueba": "egresos_que_salen_de_la_prueba"}
CELDA_MINIMA = 10


def tasas_publicables(t: pd.DataFrame) -> pd.DataFrame:
    out = t[["diseno", "anio", "cod_hospital"]].copy()
    for k in ("z515", "alta_domiciliaria", "voluntaria_fuga"):
        n = t[k].to_numpy()
        ok = (n == 0) | (n >= CELDA_MINIMA)
        out[f"tasa_{k}_pct"] = np.where(ok, np.round(100 * t[f"tasa_{k}"].to_numpy(), 1), np.nan)
    out["nota"] = np.where(out.filter(like="tasa_").isna().any(axis=1), "tasa suprimida: menos de 10 casos", "")
    return out.sort_values(["diseno", "cod_hospital"]).reset_index(drop=True)


def escribir(res_modelo: pd.DataFrame, res_ind: dict, muertes_obst: int, salida: Path, meta: dict) -> dict:
    from evaluacion import privacidad
    salida.mkdir(parents=True, exist_ok=True)
    tablas = {
        "resumen_modelo.csv": res_modelo.drop(columns=["segundos"], errors="ignore"),
        "resumen_indicador.csv": res_ind["indicador"],
        "hospitales_informados_aparte.csv": res_ind["marcas"],
        "correlaciones_contexto.csv": res_ind["correlaciones_contexto"],
        "tasas_contexto_hospital.csv": tasas_publicables(res_ind["tasas_contexto"]),
        "muertes_obstetricas_excluidas.csv": pd.DataFrame([{"periodo": "2019-2023", "cohorte": "D",
                                                            "muertes": str(muertes_obst) if muertes_obst >= CELDA_MINIMA or muertes_obst == 0 else "<10"}]),
    }
    for nombre, t in tablas.items():
        t.rename(columns=RENOMBRE).to_csv(salida / nombre, index=False)
    import lightgbm
    import sklearn
    meta = {**meta, "python": platform.python_version(), "lightgbm": lightgbm.__version__, "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "archivos": sorted(tablas)}
    (salida / "metadatos.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    hallazgos = privacidad.escanear_carpeta(salida)
    if hallazgos:
        raise RuntimeError(f"el escáner de privacidad encontró problemas en las salidas publicables: {hallazgos}")
    return tablas
