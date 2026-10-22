"""Resultados publicables de la enmienda 01: resúmenes por análisis, sin conteos de personas por hospital.

Reglas (protocolo_fase5_v2.toml, [publicacion]):
- No se publican tablas por hospital con O/E ni observados. De cada análisis salen τ², porcentajes y la lista de códigos
  que cambian de zona.
- En mortalidad, el O/E de un estrato que contiene un hospital con conteo suprimido no se publica. Restando los
  observados de los demás hospitales del estrato se podría reconstruir ese conteo.
- Antes de escribir se pasa el escáner de privacidad de la Fase 5 y se falla si encuentra algo.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

RENOMBRE = {"n_entrenamiento": "egresos_entrenamiento", "n_prueba": "egresos_prueba", "n_elegibles": "elegibles",
            "n_comunes": "comparados", "n_cambian": "cambian_de_zona", "n_hospitales": "cantidad_hospitales",
            "n_hospitales_menos_2_anios": "con_menos_de_2_anios_en_grd", "n_fuera_2024": "fuera_en_2024",
            "fuera_2024_misma_direccion_2023": "repetidas_en_2023_misma_direccion"}

# exploración de la auditoría (incidente I-08) → identificador de la enmienda
EXPLORACION_MODELO = {"sin_sospechosas": "2.1", "sin_2020_2021": "2.2", "sin_condicionales": "2.3", "sin_comorbilidades": "2.4", "base_linea_base": "2.5"}
EXPLORACION_INDICADOR = {"sin_traslado_entrante": "2.6", "sin_monograficos": "3.1a", "pares_volumen": "3.1b", "rebase": "3.3", "sin_winsor": "3.4a",
                         "linea_base": "2.5"}
DISENO = {"A": "original", "B": "anidado"}


def suprimir_estratos(estr: pd.DataFrame) -> pd.DataFrame:
    e = estr.copy()
    k = (e["tarea"] == "mortalidad") & e["contiene_suprimido"]
    e.loc[k, ["oe", "ic_inf", "ic_sup"]] = np.nan
    e["nota"] = np.where(k, "O/E no publicado: el estrato contiene un hospital con menos de 10 muertes", "")
    return e.drop(columns=["contiene_suprimido"])


def comparar_exploracion(modelo: pd.DataFrame, indicador: pd.DataFrame, dir_exp: Path) -> pd.DataFrame:
    filas = []
    m = pd.read_csv(dir_exp / "resumen_sensibilidades.csv")
    for r in m.itertuples():
        id_ = EXPLORACION_MODELO.get(r.variante)
        if id_ is None:
            continue
        a = modelo[(modelo["tarea"] == r.tarea) & (modelo["diseno"] == DISENO[r.diseno]) & (modelo["analisis"] == id_)]
        if len(a):
            filas.append({"tarea": r.tarea, "diseno": DISENO[r.diseno], "analisis": id_, "pct_misma_zona_exploracion": r.pct_misma_zona,
                          "pct_misma_zona_enmienda": float(a["pct_misma_zona"].iloc[0]), "rho_exploracion": r.rho_oe, "rho_enmienda": float(a["rho_oe"].iloc[0])})
    ind = dir_exp / "indicador" / "indicador_resumen.csv"
    if ind.exists():
        i = pd.read_csv(ind)
        for r in i.itertuples():
            id_ = EXPLORACION_INDICADOR.get(r.variante)
            dis = "original" if r.anio == 2024 else "anidado"
            if id_ is None or (id_ == "2.5" and dis == "anidado"):
                continue
            fuente = modelo if id_ == "2.5" else indicador
            a = fuente[(fuente["tarea"] == r.tarea) & (fuente["diseno"] == dis) & (fuente["analisis"] == id_)]
            if len(a):
                filas.append({"tarea": r.tarea, "diseno": dis, "analisis": id_, "pct_misma_zona_exploracion": r.pct_misma_zona,
                              "pct_misma_zona_enmienda": float(a["pct_misma_zona"].iloc[0]), "rho_exploracion": r.rho_oe, "rho_enmienda": float(a["rho_oe"].iloc[0])})
    c = pd.DataFrame(filas).drop_duplicates(["tarea", "diseno", "analisis"])
    if len(c):
        c["igual"] = np.isclose(c["pct_misma_zona_exploracion"], c["pct_misma_zona_enmienda"]) & np.isclose(c["rho_exploracion"], c["rho_enmienda"], atol=1e-9)
    return c


def escribir(res_modelo: pd.DataFrame, res_ind: dict, salida: Path, meta: dict, dir_exp: Path | None = None) -> dict:
    from evaluacion import privacidad
    salida.mkdir(parents=True, exist_ok=True)
    tablas = {
        "resumen_modelo.csv": res_modelo.drop(columns=["segundos"], errors="ignore"),
        "resumen_indicador.csv": res_ind["indicador"],
        "institutos_y_hospitales_nuevos.csv": res_ind["marcas"],
        "calibracion_por_estrato.csv": suprimir_estratos(res_ind["estratos"]),
        "validacion_embudo_p5b.csv": res_ind["validacion_p5b"],
        "estabilidad_2023_2024.csv": res_ind["estabilidad"],
    }
    if dir_exp is not None and (dir_exp / "resumen_sensibilidades.csv").exists():
        tablas["comparacion_con_exploracion.csv"] = comparar_exploracion(res_modelo, res_ind["indicador"], dir_exp)
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
