"""Congelamiento de los artefactos: modelo, preprocesamiento, recalibrador y vocabularios, con huella SHA-256, versiones y ficha de modelo."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import optuna
import pandas as pd
import polars
import sklearn
import xgboost

from .candidatos import Modelo
from .config import Reglas, sha256_archivo


def versiones() -> dict:
    return {"python": platform.python_version(), "polars": polars.__version__, "pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__, "optuna": optuna.__version__}


def guardar_modelo(modelo: Modelo, recalibrador, carpeta: Path) -> dict[str, Path]:
    """Guarda el modelo en su formato nativo cuando existe (árboles) y el resto con joblib. Devuelve los archivos escritos."""
    carpeta.mkdir(parents=True, exist_ok=True)
    archivos = {}
    if modelo.tipo == "lightgbm":
        ruta = carpeta / "modelo_lightgbm.txt"
        modelo.objeto.booster_.save_model(str(ruta))
        archivos["modelo"] = ruta
    elif modelo.tipo == "xgboost":
        ruta = carpeta / "modelo_xgboost.json"
        modelo.objeto.get_booster().save_model(str(ruta))
        archivos["modelo"] = ruta
    # el objeto completo (con su preprocesamiento, si lo tiene) también se guarda para poder predecir sin reconstruirlo
    ruta = carpeta / "modelo_completo.joblib"
    joblib.dump(modelo, ruta)
    archivos["modelo_completo"] = ruta
    ruta = carpeta / "recalibrador.joblib"
    joblib.dump(recalibrador, ruta)
    archivos["recalibrador"] = ruta
    return archivos


def ficha_modelo(tarea: str, modelo: Modelo, recal_nombre: str, metricas: dict, columnas: list[str], limites: list[str], huella_reglas: str) -> str:
    objetivo = "Probabilidad de morir durante la hospitalización (adultos, cohorte D)" if tarea == "mortalidad" else "Días de hospitalización esperados de un sobreviviente, recortados a 53 días (cohorte E)"
    lineas = [f"# Ficha del modelo de {tarea}", "", f"**Qué predice.** {objetivo}.", "",
              f"**Algoritmo.** {modelo.tipo}; árboles: {modelo.n_arboles if modelo.n_arboles else 'no aplica'}.", f"**Hiperparámetros.** {json.dumps(modelo.params, ensure_ascii=False)}", "",
              "**Datos.** Entrenamiento con 2019–2022 y recalibración global con 2023 (método: " + recal_nombre + "). El año 2024 no se usó.", "",
              f"**Reglas.** Huella SHA-256 de reglas_fase4.toml: `{huella_reglas}`.", "", "**Variables usadas.** " + ", ".join(columnas), "",
              "**Métricas de validación (2023, fuera de muestra).**"]
    lineas += [f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in metricas.items()]
    lineas += ["", "**Límites conocidos.**"] + [f"- {l}" for l in limites]
    return "\n".join(lineas) + "\n"


def escribir_manifiesto(carpeta_modelos: Path, reglas: Reglas, archivos: dict[str, dict[str, Path]], extra: dict) -> Path:
    manifiesto = {"fase": 4, "version_reglas": reglas.datos["version"], "huella_reglas": reglas.huella(), "versiones": versiones(), "extra": extra, "archivos": {}}
    for grupo, d in archivos.items():
        manifiesto["archivos"][grupo] = {nombre: {"ruta": str(ruta.relative_to(carpeta_modelos)), "sha256": sha256_archivo(ruta)} for nombre, ruta in d.items()}
    ruta = carpeta_modelos / "manifiesto.json"
    ruta.write_text(json.dumps(manifiesto, indent=1, ensure_ascii=False), encoding="utf8")
    return ruta
