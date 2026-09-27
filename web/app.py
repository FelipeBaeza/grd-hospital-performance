"""Servidor Web y API de Rendimiento Hospitalario GRD (FastAPI).

Provee endpoints analíticos para:
- Resumen global y cascada CONSORT
- Benchmarks hospitalarios (HSMR, IEMC_ML, IEMC_FONASA)
- Comparación ML vs. Norma tradicional FONASA
- Métricas y calibración de modelos
- Calculadora / Simulador de riesgo clínico en tiempo real al ingreso

Ejecutar:
  python web/app.py
  (O con uvicorn: uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload)
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import polars as pl
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Asegurar raíz en PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

app = FastAPI(
    title="Sistema de Ajuste de Riesgo y Desempeño Hospitalario GRD",
    description="API analítica para evaluación de 5.8 millones de egresos hospitalarios de Chile (2019-2024)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelos y reportes al inicio
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
CONFIG_DIR = ROOT_DIR / "config"

model_mort = None
model_los = None

try:
    if (MODELS_DIR / "lgb_mortality.joblib").exists():
        model_mort = joblib.load(MODELS_DIR / "lgb_mortality.joblib")
    if (MODELS_DIR / "lgb_los.joblib").exists():
        model_los = joblib.load(MODELS_DIR / "lgb_los.joblib")
    print("Modelos LightGBM cargados en memoria exitosamente.")
except Exception as e:
    print(f"Advertencia al cargar modelos: {e}")

# Mapeador de grupos clínicos para la calculadora
try:
    df_gc = pl.read_csv(CONFIG_DIR / "agrupacion_grupo_clinico.csv", comment_prefix="#")
    ranges_gc = [
        (row["valor_origen"], row["valor_origen_hasta"], row["categoria_destino"])
        for row in df_gc.filter(pl.col("tipo_regla") == "RANGO").iter_rows(named=True)
    ]
except Exception:
    ranges_gc = []


class PatientAdmission(BaseModel):
    edad: int = Field(..., ge=0, le=110, example=65)
    sexo: str = Field(..., example="HOMBRE")
    prevision: str = Field(..., example="FONASA INSTITUCIONAL - (MAI) B")
    tipo_ingreso: str = Field(..., example="URGENCIA")
    procedencia_agr: str = Field(..., example="URGENCIA")
    diagnostico1_cie10: str = Field(..., example="I21.9")
    mes_ingreso: int = Field(6, ge=1, le=12)
    dia_semana_ingreso: int = Field(3, ge=1, le=7)
    es_fin_semana: int = Field(0, ge=0, le=1)
    n_egresos_12m: int = Field(0, ge=0)
    dias_desde_egreso_previo: float | None = Field(None, example=45)
    comorbilidades_elix: list[str] = Field(default_factory=list, example=["ELIX_01", "ELIX_06"])


@app.get("/api/summary")
def get_summary():
    """Devuelve métricas globales de la cohorte y del diagrama de exclusión CONSORT."""
    ruta_consort = REPORTS_DIR / "consort_summary.csv"
    if not ruta_consort.exists():
        raise HTTPException(status_code=404, detail="No existe consort_summary.csv")

    df_consort = pd.read_csv(ruta_consort)
    total_row = df_consort[df_consort["anio"].str.contains("TOTAL")].iloc[0].to_dict()

    return {
        "poblacion_inicial": int(total_row["n_inicial"]),
        "cohorte_dura_mortalidad": int(total_row["n_cohorte_dura"]),
        "pct_cohorte_dura": float(total_row["pct_cohorte_dura"]),
        "cohorte_estancia": int(total_row["n_cohorte_estancia"]),
        "pct_cohorte_estancia": float(total_row["pct_cohorte_estancia"]),
        "exclusiones": {
            "ex01_no_agrupable": int(total_row["ex01_no_agrupable"]),
            "ex02_recien_nacidos": int(total_row["ex02_recien_nacido"]),
            "ex03_obstetricia_sin_comp": int(total_row["ex03_obstetrico_sin_comp"]),
            "ex05_estancia_cero": int(total_row["ex05_estancia_cero"]),
            "ex06_fallecidos": int(total_row["ex06_fallecido"]),
            "ex07_cma": int(total_row["ex07_cma"]),
        },
        "por_anio": df_consort[~df_consort["anio"].str.contains("TOTAL")].to_dict(orient="records"),
    }


@app.get("/api/benchmarks")
def get_benchmarks(
    year: int = Query(2024, description="Año de evaluación (2019-2024)"),
    service: str | None = Query(None, description="Filtrar por Servicio de Salud"),
    category: str | None = Query(None, description="Filtrar por Cuadrante de Desempeño"),
):
    """Devuelve la lista de hospitales con sus índices HSMR e IEMC para un año específico."""
    ruta_b = REPORTS_DIR / f"hospital_benchmarks_{year}.csv"
    if not ruta_b.exists():
        ruta_b = REPORTS_DIR / "hospital_benchmarks_all_years.csv"
        if not ruta_b.exists():
            raise HTTPException(status_code=404, detail="No se encontraron datos de benchmarks.")
        df = pd.read_csv(ruta_b)
        df = df[df["anio"] == year]
    else:
        df = pd.read_csv(ruta_b)

    if service:
        df = df[df["SERVICIO_SALUD"].str.contains(service, case=False, na=False)]
    if category:
        df = df[df["clasificacion_desempeno"] == category]

    return df.to_dict(orient="records")


@app.get("/api/comparison")
def get_comparison():
    """Devuelve la comparativa metodológica entre IEMC ML y la norma estática FONASA."""
    ruta_comp = REPORTS_DIR / "comparison_ml_vs_fonasa.csv"
    if not ruta_comp.exists():
        raise HTTPException(status_code=404, detail="No existe comparison_ml_vs_fonasa.csv")

    df = pd.read_csv(ruta_comp)
    return df.to_dict(orient="records")


@app.get("/api/models")
def get_model_metrics():
    """Devuelve las métricas de validación y calibración de los modelos LightGBM."""
    m_mort_data = {}
    m_los_data = {}

    r_mort = REPORTS_DIR / "metrics_mortality.json"
    if r_mort.exists():
        with open(r_mort, "r", encoding="utf-8") as f:
            m_mort_data = json.load(f)

    r_los = REPORTS_DIR / "metrics_los.json"
    if r_los.exists():
        with open(r_los, "r", encoding="utf-8") as f:
            m_los_data = json.load(f)

    calib_data = []
    r_cal = REPORTS_DIR / "calibration_mortality_2024.csv"
    if r_cal.exists():
        calib_data = pd.read_csv(r_cal).to_dict(orient="records")

    imp_mort = []
    r_imp_m = REPORTS_DIR / "feature_importance_mortality.csv"
    if r_imp_m.exists():
        imp_mort = pd.read_csv(r_imp_m).head(15).to_dict(orient="records")

    imp_los = []
    r_imp_l = REPORTS_DIR / "feature_importance_los.csv"
    if r_imp_l.exists():
        imp_los = pd.read_csv(r_imp_l).head(15).to_dict(orient="records")

    return {
        "mortality": m_mort_data,
        "los": m_los_data,
        "calibration_deciles": calib_data,
        "feature_importance_mortality": imp_mort,
        "feature_importance_los": imp_los,
    }


@app.post("/api/predict")
def predict_patient_risk(admission: PatientAdmission):
    """Calcula el riesgo de mortalidad y la estancia esperada para un paciente al ingreso."""
    if model_mort is None or model_los is None:
        raise HTTPException(status_code=503, detail="Los modelos no están cargados en memoria.")

    # Derivar CIE10_3C y GRUPO_CLINICO
    c3 = admission.diagnostico1_cie10.strip().upper()[:3]
    gc = "NO_CLASIFICADO"
    if c3 == "DES":
        gc = "DESCONOCIDO"
    else:
        for d_from, d_to, cat_dest in ranges_gc:
            if d_from <= c3 <= d_to:
                gc = cat_dest
                break

    # Vector Elixhauser
    elix_dict = {f"ELIX_{i:02d}": 0 for i in range(1, 32)}
    for c in admission.comorbilidades_elix:
        if c in elix_dict:
            elix_dict[c] = 1

    n_comorb = sum(elix_dict.values())

    # Score Van Walraven con jerarquías
    vw_pesos = {
        "ELIX_01": 7, "ELIX_02": 5, "ELIX_03": -1, "ELIX_04": 4, "ELIX_05": 2,
        "ELIX_06": 0, "ELIX_07": 0, "ELIX_08": 7, "ELIX_09": 6, "ELIX_10": 3,
        "ELIX_11": 0, "ELIX_12": 0, "ELIX_13": 0, "ELIX_14": 5, "ELIX_15": 11,
        "ELIX_16": 0, "ELIX_17": 0, "ELIX_18": 9, "ELIX_19": 12, "ELIX_20": 4,
        "ELIX_21": 0, "ELIX_22": 3, "ELIX_23": -4, "ELIX_24": 6, "ELIX_25": 5,
        "ELIX_26": -2, "ELIX_27": -2, "ELIX_28": 0, "ELIX_29": -7, "ELIX_30": 0,
        "ELIX_31": -3,
    }
    score_vw = 0
    for cat, val in elix_dict.items():
        if val == 1:
            w = vw_pesos.get(cat, 0)
            if cat == "ELIX_20" and elix_dict.get("ELIX_19") == 1:
                w = 0
            score_vw += w

    row = {
        "SEXO": admission.sexo,
        "PREVISION": admission.prevision,
        "TIPO_INGRESO": admission.tipo_ingreso,
        "PROCEDENCIA_AGR": admission.procedencia_agr,
        "CIE10_3C": c3,
        "GRUPO_CLINICO": gc,
        "EDAD_ANIOS": admission.edad,
        "MES_INGRESO": admission.mes_ingreso,
        "DIA_SEMANA_INGRESO": admission.dia_semana_ingreso,
        "ES_FIN_SEMANA": admission.es_fin_semana,
        "N_EGRESOS_12M": admission.n_egresos_12m,
        "DIAS_DESDE_EGRESO_PREVIO": admission.dias_desde_egreso_previo if admission.dias_desde_egreso_previo is not None else np.nan,
        "HISTORIA_DISPONIBLE": 1,
        **elix_dict,
        "N_COMORB_ELIX": n_comorb,
        "SCORE_VANWALRAVEN": score_vw,
    }

    df_sample = pd.DataFrame([row])
    for c in ["SEXO", "PREVISION", "TIPO_INGRESO", "PROCEDENCIA_AGR", "CIE10_3C", "GRUPO_CLINICO"]:
        df_sample[c] = df_sample[c].astype("category")

    # Inferencia
    p_muerte = float(model_mort.predict_proba(df_sample)[:, 1][0])
    dias_esp = float(model_los.predict(df_sample)[0])
    dias_esp = max(1.0, dias_esp)

    # Nivel de riesgo
    if p_muerte < 0.01:
        nivel_riesgo = "BAJO"
    elif p_muerte < 0.05:
        nivel_riesgo = "MODERADO"
    elif p_muerte < 0.20:
        nivel_riesgo = "ALTO"
    else:
        nivel_riesgo = "CRÍTICO"

    return {
        "probabilidad_mortalidad": round(p_muerte, 4),
        "porcentaje_mortalidad": round(p_muerte * 100, 2),
        "nivel_riesgo": nivel_riesgo,
        "estancia_esperada_dias": round(dias_esp, 1),
        "n_comorbilidades": n_comorb,
        "score_van_walraven": score_vw,
        "grupo_clinico_asignado": gc,
    }


# Montar estáticos
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "API activa. El frontend se encuentra en /static/index.html"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
