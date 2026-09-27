# Sistema de Evaluación del Desempeño Hospitalario Ajustado por Riesgo mediante Aprendizaje Automático

**Trabajo de Titulación**  
**Estudiante:** Felipe Ignacio Baeza Muñoz  
**Carrera:** Ingeniería de Ejecución en Computación e Informática  
**Universidad:** Universidad de Santiago de Chile (USACH)  
**Profesor Guía:** Dr. Manuel Villalobos Cid  

---

## 1. Descripción del Proyecto

Este proyecto desarrolla un sistema computacional basado en Machine Learning para calcular indicadores estandarizados de desempeño hospitalario (Razón Observado/Esperado, $O/E$) en la red pública de salud chilena, utilizando registros de Grupos Relacionados por Diagnóstico (GRD) entregados por FONASA (2019–2024, ~5,8 millones de egresos).

El sistema predice:
1. **Riesgo de Mortalidad Intrahospitalaria:** Clasificación binaria probabilística sobre prevalencia natural (~2,8%).
2. **Días de Estancia Esperados:** Regresión de conteo (Tweedie/Poisson) modelada exclusivamente en sobrevivientes.
3. **Indicador de Desempeño $O/E$ y Gráficos de Embudo (*Funnel Plots*):** Evaluación institucional con corrección de sobredispersión de Spiegelhalter (1995, 2005).
4. **Demostrador Ciudadano:** Inferencia en tiempo real (<2s) con explicabilidad SHAP y política estricta de cero persistencia de datos personales.

---

## 2. Metodología CRISP-DM

El desarrollo se organiza bajo las fases de CRISP-DM:
- **Fase 1: Comprensión del Dominio:** Evidencias y reglas en `docs/fase1_comprension_dominio/`.
- **Fase 2: Comprensión de los Datos:** Diccionario de variables, problemas de calidad y roles en `docs/fase2_comprension_datos/`.
- **Fase 3: Preparación de Datos:** Scripts `00_` a `06_` en `src/`.
- **Fase 4: Modelado:** Scripts `07_` y `08_` en `src/` (LightGBM).
- **Fase 5: Evaluación:** Script `09_` en `src/` (Razón $O/E$ e intervalos de control).
- **Fase 6: Despliegue:** API FastAPI y visualización en `web/`.

---

## 3. Estructura del Repositorio

```text
.
├── config/                 # Tablas y diccionarios clínicos de referencia (Elixhauser, van Walraven, etc.)
├── data/                   # Datos (ignorado por Git)
│   ├── raw/                # Archivos .txt de FONASA
│   ├── bronze/             # Parquet crudo 1:1
│   ├── silver/             # Parquet limpio, tipado y cohorte depurada
│   └── gold/               # Matrices de features y targets para ML
├── docs/                   # Evidencias metodológicas
│   ├── fase1_comprension_dominio/
│   └── fase2_comprension_datos/
├── src/                    # Pipeline secuencial de código
│   ├── 00_schema_validator.py
│   ├── 01_ingest_bronze.py
│   ├── 02_clean_silver.py
│   ├── 03_generate_id.py
│   ├── 04_filter_cohort.py
│   ├── 05_transform_gold.py
│   ├── 06_elixhauser.py
│   ├── 07_train_mortality.py
│   ├── 08_train_los.py
│   ├── 09_oe_indicator.py
│   └── pipeline.py
├── tests/                  # Pruebas clave de calidad y no-fuga
└── web/                    # Backend FastAPI y Frontend interactivo
```

---

## 4. Instalación y Uso

```bash
# Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el pipeline completo
python3 src/pipeline.py
```
