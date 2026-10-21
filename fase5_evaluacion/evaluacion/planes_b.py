"""Planes B congelados ANTES de abrir 2024: las líneas base ya entrenadas con 2019–2022, cada una con su propia recalibración global con 2023.
Si un criterio de la Fase 1 falla, el modelo final pasa a ser la línea base; para que ese cambio no obligue a tocar nada con 2024 abierto, se congelan ahora."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .config import sha256_archivo

CANDIDATOS = {"mortalidad": {"logistica": "linea_base"}, "estadia": {"poisson": "linea_base", "lineal": "lineal"}}


def congelar(trabajo_f4: Path, destino: Path) -> dict:
    """Recalibra con 2023 (solo con 2023) las líneas base y las guarda con huella. Reutiliza las predicciones de 2023 que dejó la Fase 4."""
    from modelado.calibracion import FactorUnico, Platt, ajustar_final  # noqa: F401
    destino.mkdir(parents=True, exist_ok=True)
    manifiesto = {"archivos": {}, "nota": "líneas base entrenadas con 2019–2022; recalibradas con 2023; 2024 no se usó"}
    for tarea, nombres in CANDIDATOS.items():
        z = np.load(trabajo_f4 / f"pred2023_{tarea}.npz", allow_pickle=False)
        y = z["y"]
        anios = np.full(len(y), 2023)
        for nombre, clave in nombres.items():
            modelo = joblib.load(trabajo_f4 / f"modelo_{tarea}_{clave}.joblib")
            rec = ajustar_final(tarea, "platt" if tarea == "mortalidad" else "factor", z[clave], y, anios)
            carpeta = destino / f"{tarea}_{nombre}"
            carpeta.mkdir(exist_ok=True)
            joblib.dump(modelo, carpeta / "modelo_completo.joblib")
            joblib.dump(rec, carpeta / "recalibrador.joblib")
            manifiesto["archivos"][f"{tarea}_{nombre}"] = {k: {"ruta": f"{tarea}_{nombre}/{k}.joblib", "sha256": sha256_archivo(carpeta / f"{k}.joblib")} for k in ("modelo_completo", "recalibrador")}
    (destino / "manifiesto.json").write_text(json.dumps(manifiesto, indent=1, ensure_ascii=False), encoding="utf8")
    return manifiesto
