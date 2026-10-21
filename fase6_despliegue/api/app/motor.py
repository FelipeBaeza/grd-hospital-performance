"""Motor de predicción de la demo. Carga los modelos congelados de la Fase 4 en su formato nativo de LightGBM, comprueba su huella SHA-256 contra el
manifiesto y calcula riesgo y contribuciones (TreeSHAP exacto del propio modelo). No entrena ni recalibra nada. No guarda ni registra el perfil."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

CATEGORICAS = ["SEXO", "TIPO_INGRESO", "TIPO_PROCEDENCIA", "PROCEDENCIA_AGR", "CIE10_3C", "GRUPO_CLINICO"]
ETIQUETAS = {"SEXO": "Sexo", "TIPO_INGRESO": "Tipo de ingreso", "TIPO_PROCEDENCIA": "Procedencia", "PROCEDENCIA_AGR": "Procedencia agrupada", "CIE10_3C": "Diagnóstico principal",
             "GRUPO_CLINICO": "Grupo clínico (capítulo CIE-10)", "EDAD_ANIOS": "Edad", "ES_FIN_SEMANA": "Ingreso en fin de semana", "NIVEL_CUIDADO_INGRESO": "Cuidado intensivo o intermedio al ingreso"}


class ModelosAlterados(RuntimeError):
    pass


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _sigmoide(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class MotorPrediccion:
    TAREAS = ("mortalidad", "estadia")

    def __init__(self, dir_modelos: Path, dir_datos: Path):
        self.manifiesto = json.loads((dir_modelos / "manifiesto.json").read_text(encoding="utf8"))
        self.recal = json.loads((dir_datos / "recalibracion.json").read_text(encoding="utf8"))
        self.catalogo = json.loads((dir_datos / "demo_catalogo.json").read_text(encoding="utf8"))
        self.huellas: dict[str, dict] = {}
        self.boosters: dict[str, lgb.Booster] = {}
        for t in self.TAREAS:
            ref = self.manifiesto["archivos"][t]["modelo"]
            ruta = dir_modelos / ref["ruta"]
            h = sha256_archivo(ruta)
            if h != ref["sha256"]:
                raise ModelosAlterados(f"el modelo de {t} no coincide con el manifiesto de la Fase 4")
            if self.recal[t]["fuente_sha256"] != self.manifiesto["archivos"][t]["recalibrador"]["sha256"]:
                raise ModelosAlterados(f"la recalibración de {t} no proviene del recalibrador congelado")
            self.huellas[t] = {"archivo": ref["ruta"], "sha256": h, "coincide": True}
            self.boosters[t] = lgb.Booster(model_file=str(ruta))
        b = self.boosters["mortalidad"]
        self.columnas = b.feature_name()
        self.niveles = dict(zip(CATEGORICAS, b.pandas_categorical))
        self.diag = {d["codigo"]: d for d in self.catalogo["diagnosticos"]}
        self.proc = {p["valor"]: p["procedencia_agr"] for p in self.catalogo["tipo_procedencia"]}
        self.elix = {c["id"]: c["nombre"] for c in self.catalogo["comorbilidades"]}

    # ---- entrada
    def marco(self, p: dict) -> pd.DataFrame:
        fila = {"SEXO": p["sexo"], "TIPO_INGRESO": p["tipo_ingreso"], "TIPO_PROCEDENCIA": p["tipo_procedencia"], "PROCEDENCIA_AGR": self.proc[p["tipo_procedencia"]],
                "CIE10_3C": p["diagnostico_principal"], "GRUPO_CLINICO": self.diag[p["diagnostico_principal"]]["grupo_clinico"], "EDAD_ANIOS": float(p["edad"]),
                "ES_FIN_SEMANA": int(bool(p["ingreso_fin_de_semana"])), "NIVEL_CUIDADO_INGRESO": int(bool(p["cuidado_intensivo_al_ingreso"]))}
        for i in range(1, 32):
            fila[f"ELIX_{i:02d}"] = int(f"ELIX_{i:02d}" in set(p.get("comorbilidades", [])))
        df = pd.DataFrame([fila])[self.columnas]
        for c in CATEGORICAS:
            df[c] = pd.Categorical(df[c], categories=self.niveles[c])
        return df

    # ---- salida
    def _valor_legible(self, col: str, p: dict) -> str:
        if col == "EDAD_ANIOS":
            return f"{int(p['edad'])} años"
        if col == "ES_FIN_SEMANA":
            return "sí" if p["ingreso_fin_de_semana"] else "no"
        if col == "NIVEL_CUIDADO_INGRESO":
            return "sí" if p["cuidado_intensivo_al_ingreso"] else "no"
        if col.startswith("ELIX_"):
            return "sí" if col in set(p.get("comorbilidades", [])) else "no"
        if col == "CIE10_3C":
            return f"{p['diagnostico_principal']} · {self.diag[p['diagnostico_principal']]['nombre']}"
        if col == "GRUPO_CLINICO":
            return f"Cap. {self.diag[p['diagnostico_principal']]['grupo_clinico']}"
        return {"SEXO": p["sexo"], "TIPO_INGRESO": p["tipo_ingreso"], "TIPO_PROCEDENCIA": p["tipo_procedencia"], "PROCEDENCIA_AGR": self.proc[p["tipo_procedencia"]]}[col]

    def _contribuciones(self, tarea: str, df: pd.DataFrame, p: dict) -> tuple[float, float, list[dict]]:
        """(valor bruto, base, contribuciones por variable) en la escala del modelo antes de recalibrar."""
        c = self.boosters[tarea].predict(df, pred_contrib=True)[0]
        base, contrib = float(c[-1]), c[:-1]
        filas = []
        for col, v in zip(self.columnas, contrib):
            etiqueta = ETIQUETAS.get(col) or self.elix.get(col, col)
            filas.append({"variable": col, "etiqueta": etiqueta, "valor": self._valor_legible(col, p), "efecto": float(v)})
        return float(contrib.sum() + base), base, filas

    def predecir(self, perfil: dict) -> dict:
        df = self.marco(perfil)
        out: dict = {}
        # mortalidad: Platt sobre el logit
        r = self.recal["mortalidad"]
        bruto, base, filas = self._contribuciones("mortalidad", df, perfil)
        a, b = r["a"], r["b"]
        for f in filas:
            f["efecto"] = b * f["efecto"]
        base_final = a + b * base
        logit = a + b * bruto
        out["mortalidad"] = {"probabilidad": _sigmoide(logit), "probabilidad_base": _sigmoide(base_final), "logit": logit, "base_logit": base_final,
                             "escala": "log-odds", "contribuciones": sorted(filas, key=lambda x: -abs(x["efecto"]))}
        # estadía: factor único sobre la media
        r = self.recal["estadia"]
        bruto, base, filas = self._contribuciones("estadia", df, perfil)
        lf = math.log(r["f"])
        out["estadia"] = {"dias_esperados": math.exp(bruto + lf), "dias_base": math.exp(base + lf), "log_dias": bruto + lf, "base_log_dias": base + lf,
                          "escala": "log-días", "contribuciones": sorted(filas, key=lambda x: -abs(x["efecto"]))}
        return out

    def estado(self) -> dict:
        return {t: self.huellas[t] for t in self.TAREAS}
