# Módulo de Análisis Exploratorio de Datos (EDA) y Validación Empírica — Fase 2

Este directorio contiene la suite completa de código, visualizaciones científicas y scripts de auditoría empírica desarrollados para la **Fase 2: Comprensión de los Datos (Data Understanding)** de la tesis:

> **"Sistema de evaluación del desempeño hospitalario ajustado por riesgo mediante aprendizaje automático"**  
> *Estudiante: Felipe Ignacio Baeza Muñoz | Carrera: Ingeniería de Ejecución en Computación e Informática — USACH*

---

## 📁 Estructura del Directorio

```text
Analisis exploratorio/
├── README.md                                    # Guía metodológica y catálogo de artefactos
├── 01_generar_graficos_eda.py                   # Generador de figuras de alta resolución (300 DPI)
├── 02_verificar_evidencias_fase2.py             # Auditoría cuantitativa de los 18 problemas de calidad
├── informe_verificacion_calidad_fase2.md        # Reporte formal de verificación empírica
└── figuras/                                     # Figuras listas para inserción en la tesis
    ├── fig01_distribucion_estancia_hist_box.png
    ├── fig02_piramide_poblacional_edad_sexo.png
    ├── fig03_top_capitulos_cie10.png
    ├── fig04_prevalencia_comorbilidades_elixhauser.png
    ├── fig05_evolucion_temporal_egresos_mortalidad.png
    └── fig06_distribucion_tipo_alta_censura.png
```

---

## 🚀 Ejecución Rápida

Para reproducir tanto los gráficos como la auditoría completa sobre los 5.8 millones de registros:

```bash
# 1. Generar las 6 figuras científicas (300 DPI)
python "Analisis exploratorio/01_generar_graficos_eda.py"

# 2. Ejecutar la auditoría y validar los 18 problemas de calidad
python "Analisis exploratorio/02_verificar_evidencias_fase2.py"
```

---

## 📊 Catálogo de Figuras Científicas Generadas

### 1. `fig01_distribucion_estancia_hist_box.png` — Distribución de Estancia Hospitalaria
- **Objetivo metodológico:** Justificar la formulación matemática del modelo de estancia (`ESTANCIA_DIAS`).
- **Hallazgo clave:** Marcada asimetría positiva (*right-skewed*) con una masa de probabilidad concentrada en estancia 0 (19.76% de casos ambulatorios) y una cola pesada de hospitalizaciones prolongadas (>30 días).
- **Decisión de modelado:** Descarta regresión lineal por mínimos cuadrados ordinarios (OLS) y fundamenta el uso de regresión **Tweedie / Poisson sobredispersa** con LightGBM.

### 2. `fig02_piramide_poblacional_edad_sexo.png` — Pirámide Demográfica Hospitalaria
- **Objetivo metodológico:** Caracterizar la estructura etaria y por sexo de los usuarios de la red pública FONASA.
- **Hallazgo clave:** Predominio femenino en edad fértil (20–39 años) por obstetricia/partos, y ensanchamiento bimodal en adultos mayores (>65 años) con mayor comorbilidad y riesgo de mortalidad intrahospitalaria.

### 3. `fig03_top_capitulos_cie10.png` — Principales Motivos de Ingreso por Capítulo CIE-10
- **Objetivo metodológico:** Resumir los más de 9.500 diagnósticos primarios en categorías clínicas estables.
- **Hallazgo clave:** Los capítulos líderes son Embarazo/Parto (Cap. 15), Enfermedades del Sistema Digestivo (Cap. 11), Traumatismos/Causas Externas (Cap. 19), Sistema Circulatorio (Cap. 9) y Neoplasias (Cap. 2).

### 4. `fig04_prevalencia_comorbilidades_elixhauser.png` — Prevalencia Comorbilidades de Elixhauser
- **Objetivo metodológico:** Evaluar la carga comórbida basal preexistente (algoritmo Quan et al., 2005) a partir de los diagnósticos secundarios (`DIAGNOSTICO2` a `DIAGNOSTICO35`).
- **Hallazgo clave:** Alta prevalencia de hipertensión no complicada, diabetes no complicada, trastornos hidroelectrolíticos, insuficiencia cardíaca congestiva (ICC) y enfermedad pulmonar crónica (EPOC).

### 5. `fig05_evolucion_temporal_egresos_mortalidad.png` — Evolución Temporal y Efecto Pandemia
- **Objetivo metodológico:** Analizar la estabilidad de la serie temporal (2019–2024) y cuantificar el impacto del COVID-19.
- **Hallazgo clave:** Drástica caída en el volumen de egresos durante 2020 (~781.000) por suspensión de cirugías electivas, acompañada de un salto en la tasa de mortalidad cruda (>3.7%) debido a la gravedad de los pacientes ingresados.

### 6. `fig06_distribucion_tipo_alta_censura.png` — Desenlaces de Egreso y Censura Estadística
- **Objetivo metodológico:** Fundamentar el tratamiento estadístico de la variable objetivo de mortalidad.
- **Hallazgo clave:** 90.9% egresos a domicilio con vida, 2.9% fallecidos intrahospitalarios confirmados, y un ~6.0% de egresos censurados (traslados a otros recintos, hospitalización domiciliaria o fugas) que requieren aislamiento para evitar etiquetado erróneo como sobrevivientes.

---

## 🔍 Resumen de la Auditoría Empírica (18 Problemas Verificados)

El script `02_verificar_evidencias_fase2.py` audita las capas Bronze (datos crudos) y Silver para corroborar que lo documentado en `docs/fase2_comprension_datos/` corresponde con la realidad matemática de los datos:

| ID | Problema Documentado | Evidencia Empírica Comprobada (5.8M Registros) |
|:--:|:---------------------|:-----------------------------------------------|
| **01** | Formato de fechas no estándar | 2023 usa `%d-%m-%Y` mientras nacimientos usa `%Y-%m-%d`. Resuelto con `coalesce`. |
| **02** | Coma decimal en `IR_29301_PESO` | 5.803.460 registros (99.91%) usan coma (ej. `0,7094`); 0 usan punto en crudo. |
| **03** | Ceros iniciales en procedimientos | 178.543 registros inician con `'0'` en `PROCEDIMIENTO1` (ej. `00.17`). |
| **04** | Puntos en diagnósticos CIE-10 | Más de 25 millones de códigos con puntos (ej. `J18.0`) estandarizados a `J180`. |
| **05** | Deriva de tipo entre años | `PROCEDIMIENTO2` infiere `float` y falla con `'DESCONOCIDO'`; forzado a String. |
| **06** | Alta anterior al ingreso | 12–18 episodios biológicamente imposibles (`FECHAALTA < INGRESO`) anulados. |
| **07** | Edad negativa o > 110 años | 27 registros anómalos (1 negativo, 26 > 110 años), inferior a los <500 estimados. |
| **08** | Estancias de 0 días | 1.147.555 episodios (19.76%) ambulatorios excluidos de estancia pero aptos para mortalidad. |
| **09** | Grafías duplicadas en `ETNIA` | `'OTRO'` (1.929.130) y `'OTRO '` (438.029) suman 2.367.159 registros unificados. |
| **10** | Desajuste en traslados | `FECHATRASLADO2` (333.365) vs `SERVICIOTRASLADO2` (333.372) con diferencia exacta de 7. |
| **11** | Columnas 100% vacías | `CONDICIONDEALTANEONATO3` y `4` tienen 0 valores no nulos en toda la base. |
| **12** | Desalineación estructural | `RN2ESTADO` tiene 1.444.072 valores frente a 1.576 de `CONDICIONDEALTANEONATO2`. |
| **13** | Columna sin semántica oficial | `RN1ESTADO` tiene exactamente 1.919.691 registros con códigos no tabulados (`'10'`, `'9'`). |
| **14** | RUTs en columna de fecha | Exactamente 9 registros en `FECHAPROCEDIMIENTO1` con RUT chileno en 2019, aislados. |
| **15** | Nulos en ID de paciente | Exactamente 2.044 nulos en `CIP_ENCRIPTADO` en toda la base de 5.8M. |
| **16** | GRD "DESCONOCIDO" | 75 casos entre 2019–2023 (90 casos en 6 años) sin código GRD convertible. |
| **17** | Renombre `ID_BENEFICIARIO` | Archivo 2019 contenía `ID_BENEFICIARIO`, armonizado por alias a `CIP_ENCRIPTADO`. |
| **18** | Completitud basal en `SEXO` | Exactamente 0 nulos en 5.808.536 episodios (41.2% Hombres, 58.8% Mujeres). |

---

## 🔒 Consideraciones Éticas y Cumplimiento
- **No Data Leakage:** Se verificó que ninguna variable post-ingreso (procedimientos, traslados, desenlaces intermedios) forma parte de la matriz predictiva basal.
- **Protección de Datos Sensibles (Ley 19.628 / 20.285):** Los 9 identificadores personales encontrados en columnas erróneas han sido aislados y enmascarados, impidiendo cualquier exposición en logs o figuras.
