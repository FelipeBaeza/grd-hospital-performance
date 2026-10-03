# Módulo de Análisis Exploratorio de Datos (EDA) y Auditoría Metodológica — Fase 2

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
├── 03_auditoria_profunda_fase2.py               # Métricas reales de cohorte, POA, upcoding y traslados
├── 04_protocolo_seleccion_empirica.py           # Protocolo LASSO, estabilidad y validación anidada
├── matriz_maestra_129_columnas.csv              # Matriz de 129 filas con rol único y conteos exactos
├── anexo_tipoalta_y_censura.csv                 # Mapeo exhaustivo de los 12 valores de TIPOALTA
├── informe_verificacion_calidad_fase2.md        # Reporte formal de verificación empírica
├── resolucion_observaciones_fase2.md            # Resolución punto por punto de la revisión de Fase 2
└── figuras/                                     # Figuras listas para inserción en la tesis
    ├── fig01_distribucion_estancia_hist_box.png
    ├── fig02_piramide_poblacional_edad_sexo.png
    ├── fig03_top_capitulos_cie10.png
    ├── fig04_prevalencia_comorbilidades_elixhauser.png
    ├── fig05_evolucion_temporal_egresos_mortalidad.png
    └── fig06_distribucion_tipo_alta_censura.png
```

---

## 🚀 Ejecución de Scripts

Para reproducir tanto los gráficos como las auditorías completas sobre los 5.8 millones de registros:

```bash
# 1. Generar las 6 figuras científicas (300 DPI) en < 3 segundos
python "Analisis exploratorio/01_generar_graficos_eda.py"

# 2. Ejecutar la auditoría y validar los 18 problemas de calidad en < 2 segundos
python "Analisis exploratorio/02_verificar_evidencias_fase2.py"

# 3. Ejecutar auditoría profunda de cohorte (mortalidad, estadía, upcoding, POA, traslados)
python "Analisis exploratorio/03_auditoria_profunda_fase2.py"

# 4. Ejecutar el protocolo de selección empírica con Stability Selection (Meinshausen & Bühlmann)
python "Analisis exploratorio/04_protocolo_seleccion_empirica.py"
```

---

## 📊 Catálogo de Figuras Científicas Generadas

### 1. `fig01_distribucion_estancia_hist_box.png` — Distribución de Estancia Hospitalaria
- **Objetivo metodológico:** Justificar la formulación matemática del modelo de estancia (`ESTANCIA_DIAS`).
- **Hallazgo clave:** Marcada asimetría positiva (*right-skewed*, asimetría = 14.89, curtosis = 629.20) con una masa de probabilidad concentrada en estancia 0 (19.76% de casos ambulatorios) y una cola pesada de hospitalizaciones prolongadas (>30 días).
- **Decisión de modelado:** Descarta regresión lineal por mínimos cuadrados ordinarios (OLS) y fundamenta el uso de regresión **Tweedie / Poisson sobredispersa** con LightGBM y *capping* a p99 (46 días).

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
- **Hallazgo clave:** Drástica caída en el volumen de egresos durante 2020 (~781.000) por suspensión de cirugías electivas, acompañada de un salto en la tasa de mortalidad cruda (>3.8% cruda, 4.07% no censurada) debido a la gravedad de los pacientes ingresados.

### 6. `fig06_distribucion_tipo_alta_censura.png` — Desenlaces de Egreso y Censura Estadística
- **Objetivo metodológico:** Fundamentar el tratamiento estadístico de la variable objetivo de mortalidad.
- **Hallazgo clave:** 89.62% egresos a domicilio con vida, 2.94% fallecidos intrahospitalarios confirmados, y un 6.11% de egresos censurados (traslados a otros recintos, hospitalización domiciliaria o fugas) aislados para evitar sesgo de sobrevida espuria.

---

## 🔒 Consideraciones Éticas y Cumplimiento
- **No Data Leakage:** Se verificó que ninguna variable post-ingreso (procedimientos, traslados, desenlaces intermedios) forma parte de la matriz predictiva basal.
- **Protección de Datos Sensibles (Ley 19.628 / 20.285):** Los 9 identificadores personales encontrados en columnas erróneas han sido aislados y enmascarados, impidiendo cualquier exposición en logs o figuras.
- **Equidad Algorítmica:** Las variables de etnia y vulnerabilidad socioeconómica no se incorporan como factores de riesgo permisibles en el modelo, preservándolas estrictamente para análisis post-hoc de justicia distributiva.
