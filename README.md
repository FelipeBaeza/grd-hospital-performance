# Evaluación del Desempeño Hospitalario Ajustado por Riesgo Clínico mediante Machine Learning sobre Egresos GRD (FONASA Chile, 2019–2024)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Polars](https://img.shields.io/badge/polars-1.43.2-blueviolet.svg)](https://pola.rs/)
[![LightGBM](https://img.shields.io/badge/lightgbm-4.7.0-green.svg)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.110.0-teal.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Proyecto de titulación para optar al título de Ingeniero Civil en Informática (USACH).  
**Autor:** Felipe Ignacio Baeza Muñoz  
**Profesor Guía:** Dr. Manuel Villalobos Cid  

---

## 📌 1. Resumen Ejecutivo y Resultados Principales

Este proyecto implementa un pipeline de machine learning para el ajuste de riesgo clínico y la evaluación del desempeño en la red hospitalaria pública chilena, procesando **5.808.536 episodios hospitalarios** de FONASA (período 2019–2024).

Frente a la metodología tradicional de FONASA basada en normas estáticas por Grupo Relacionado por el Diagnóstico (IR-GRD), este sistema introduce:
1. **Ajuste de riesgo individual al ingreso:** considerando edad, comorbilidades preexistentes de Elixhauser (Quan et al., 2005), historia previa retrospectiva causal (12 meses) y diagnóstico principal.
2. **Garantía estricta anti-fuga (Anti-Leakage):** exclusión sistemática de procedimientos intra-hospitalarios y desenlaces al alta.
3. **Métricas estandarizadas internacionales:** HSMR (*Hospital Standardized Mortality Ratio*) con intervalos de confianza de Byar al 95% e IEMC (*Índice de Estancia Media Ajustada por Casuística*).

### 🏆 Resultados en Test Set Fuera de Tiempo (2024 - 930.739 episodios)

| Modelo | Objetivo Clínico | Métrica Principal | Métrica Secundaria | Calibración Global (O/E) |
| :--- | :--- | :--- | :--- | :--- |
| **Mortalidad Intrahospitalaria** | Probabilidad de muerte al ingreso | **ROC-AUC: 0.9506** | **PR-AUC: 0.3676** (Base: 0.028) | **0.8752** (Brier: 0.021) |
| **Estancia Media (LOS)** | Días de estancia esperada | **MAE: 4.47 días** | **Mediana Error: 2.38 días** | **0.9442** (Spearman ρ: 0.603) |

---

## 🏗️ 2. Arquitectura de Datos y Cascada CONSORT

El procesamiento sigue una arquitectura Medallion optimizada con Apache Arrow y Polars:

```mermaid
flowchart TD
    RAW["Raw FONASA CSVs (2019-2024)<br/>N = 5,808,536"] --> B["01: Capa Bronce (Parquet)<br/>Compresión ZSTD (269 MB)"]
    B --> S["02-03: Capa Plata (Silver)<br/>Tipado de fechas, normalización e ID único"]
    S --> C{"04: Cascada CONSORT<br/>Exclusiones Clínicas Duras"}
    C -->|EX01: No agrupables (5,073)<br/>EX02: Recién nacidos (146,928)<br/>EX03: Partos normales (431,832)| EXCL["Casos Excluidos"]
    C --> CD["Cohorte Dura Mortalidad<br/>N = 5,222,340 (89.91%)"]
    CD --> CE["Cohorte Estancia Inpatient<br/>(Sin estancia 0, fallecidos ni CMA)<br/>N = 3,561,516 (61.32%)"]
    CD --> G["05: Capa Gold<br/>47 Features Basales conocidas al ingreso"]
    G --> M1["06: LightGBM Mortalidad<br/>Prevalencia Natural (3.8% Train)"]
    G --> M2["07: LightGBM Tweedie Estancia<br/>Tweedie power = 1.5"]
    M1 & M2 --> BM["08: Benchmarks Hospitalarios<br/>HSMR + IEMC (72 Hospitales)"]
    BM --> WEB["10: Dashboard Interactivo (FastAPI + JS)"]
```

---

## 📂 3. Estructura del Repositorio

```text
├── config/                      # Tablas maestras clínicas oficiales
│   ├── mapeo_grd_mdc.csv        # Catálogo IR_29301_COD_GRD -> (MDC, GRD_BASE)
│   ├── mapeo_elixhauser_quan2005.csv # Reglas de comorbilidad CIE-10 (Quan 2005)
│   ├── pesos_vanwalraven.csv    # Ponderaciones de mortalidad de Van Walraven (2009)
│   ├── jerarquia_elixhauser.csv # Reglas de dominancia comórbida
│   ├── agrupacion_grupo_clinico.csv # Capítulos CIE-10
│   └── agrupacion_procedencia.csv   # Categorías agregadas de procedencia
├── docs/                        # Evidencias metodológicas
│   ├── fase1_comprension_dominio/ # Objetivos, stakeholders, marcos de calidad
│   └── fase2_comprension_datos/   # Diccionarios de datos, calidad y selección
├── models/                      # Modelos ML entrenados (Joblib)
│   ├── lgb_mortality.joblib     # Clasificador de riesgo de mortalidad intrahospitalaria
│   └── lgb_los.joblib           # Regresor Tweedie de estancia esperada
├── reports/                     # Métricas y evidencias empíricas generadas
│   ├── consort_summary.csv      # Flujo de exclusión poblacional por año
│   ├── metrics_mortality.json   # Métricas de mortalidad (ROC-AUC, Brier, ECE)
│   ├── metrics_los.json         # Métricas de estancia (MAE, RMSE, Spearman)
│   ├── calibration_mortality_2024.csv # Calibración por deciles de riesgo
│   ├── hospital_benchmarks_2024.csv   # Indicadores HSMR e IEMC de 72 hospitales
│   └── comparison_ml_vs_fonasa.csv    # Comparativa metodológica ML vs. GRD FONASA
├── src/                         # Pipeline de código secuencial
│   ├── 00_schema_validator.py   # Validación de esquema crudo
│   ├── 01_ingest_bronze.py      # Ingesta masiva a Parquet
│   ├── 02_clean_silver.py       # Limpieza, tipado y censura
│   ├── 03_generate_id.py        # Generación de ID_EPISODIO único
│   ├── 04_filter_cohort.py      # Cascada CONSORT y comorbilidades Elixhauser
│   ├── 05_transform_gold.py     # Matriz de features con garantía anti-fuga
│   ├── 06_train_mortality.py    # Entrenamiento y calibración mortalidad
│   ├── 07_train_los.py          # Entrenamiento regresión estancia Tweedie
│   ├── 08_compute_benchmarks.py # Cálculo de indicadores y comparación vs FONASA
│   ├── 09_run_pipeline.py       # Orquestador maestro CLI
│   └── elixhauser.py            # Mapeador vectorial Polars Quan (2005)
├── web/                         # Aplicación Web y API interactiva
│   ├── app.py                   # Backend FastAPI
│   └── static/                  # Frontend HTML5/CSS/JS (Dark glassmorphism)
├── requirements.txt             # Dependencias del proyecto
└── README.md
```

---

## 🚀 4. Instalación y Uso

### 4.1. Configuración del Entorno
```bash
git clone git@github.com:FelipeBaeza/grd-hospital-performance.git
cd grd-hospital-performance
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4.2. Ejecución del Pipeline Completo
```bash
# Ejecutar los 9 pasos secuenciales
python src/09_run_pipeline.py --all

# O ejecutar un paso específico (ej. recalcular benchmarks)
python src/09_run_pipeline.py --step 08
```

### 4.3. Lanzar la Aplicación Web Interactiva
```bash
python web/app.py
```
Acceder en el navegador a: `http://localhost:8000`

---

## 📊 5. Conclusiones y Contribución de la Tesis

1. **Superación del Techo Predictivo Tradicional:** El modelo de machine learning alcanza un **ROC-AUC de 0.9506** y un **PR-AUC de 0.3676** en mortalidad, superando ampliamente los modelos lineales basados en tablas agregadas.
2. **Corrección de Distorsiones de Eficiencia:** Al evaluar el IEMC, la metodología tradicional de FONASA penaliza a hospitales de alta complejidad con pacientes crónicos o añosos al usar una media estática de estancia por GRD. El ajuste por ML descompone la severidad individual, revelando la verdadera eficiencia de camas hospitalarias.
3. **Reproducibilidad y Escalabilidad:** El pipeline completo procesa los **5.8 millones de filas en menos de 90 segundos** gracias a Polars y algoritmos vectoriales sin loops iterativos en Python.
