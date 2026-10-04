# Módulo de Análisis Exploratorio de Datos (EDA) y Modelado Estadístico — Fase 2

Este directorio contiene la suite consolidada de scripts de cálculo, benchmarks canónicos, tablas de auditoría y documentos oficiales para la **Fase 2: Comprensión de los Datos y Modelado Exploratorio** de la tesis:

> **"Sistema de evaluación del desempeño hospitalario ajustado por riesgo mediante aprendizaje automático"**  
> *Estudiante: Felipe Ignacio Baeza Muñoz | Carrera: Ingeniería de Ejecución en Computación e Informática — USACH*

---

## 📁 Estructura Consolidada del Directorio

```text
Analisis exploratorio/
├── 01_modelado_benchmarks_mortalidad_m2cap.py   # Modelado M2+CAP, M2 Base, M3-Total, Bayes Empírico y benchmarks
├── 02_calibracion_preingreso_y_deriva_ndx.py    # Calibración preingreso y serie temporal N_dx de la red (2019–2024)
├── 03_sensibilidad_complicaciones_poa.py        # Sensibilidad a condiciones agudas y modelo estricto POA-26
├── 04_analisis_traslados_y_estres_censura.py    # Enlace determinístico a 30d, cotas lambda_salida y estrés Byar
├── 05_validacion_estadia_loho_y_bootstrap.py    # Validación LOHO out-of-sample y estabilidad bootstrap de estadía
├── 06_generar_graficos_eda.py                   # Generador de figuras científicas de alta resolución (300 DPI)
├── benchmarks_modelos_fase2_2023.csv            # Archivo canonical unificado de métricas O, E y P (K = 65)
├── tabla_canonica_elixhauser_31.csv             # Definición y clasificación POA de las 31 categorías Elixhauser
├── diccionario_base_datos_129.csv               # Diccionario exhaustivo de las 129 columnas con conteos de nulos
├── matriz_maestra_129_columnas.csv              # Matriz de metadatos de las 129 columnas con tipos y roles
├── anexo_tipoalta_y_censura.csv                 # Mapeo exhaustivo de los 12 valores observados de TIPOALTA
├── resolucion_observaciones_fase2.md            # Informe consolidado de resolución de observaciones (Fase 2)
├── resolucion_observaciones_fase2.docx          # Versión Word (DOCX) del informe de resolución
├── informe_verificacion_calidad_fase2.md        # Informe formal de calidad de datos y verificación CONSORT
├── informe_verificacion_calidad_fase2.docx      # Versión Word (DOCX) del informe de calidad
├── README.md                                    # Esta guía metodológica y catálogo de artefactos
├── README.docx                                  # Versión Word (DOCX) de esta guía metodológica
└── figuras/                                     # Figuras listas para inserción en la tesis
    ├── fig01_distribucion_estancia_hist_box.png
    ├── fig02_piramide_poblacional_edad_sexo.png
    ├── fig03_top_capitulos_cie10.png
    ├── fig04_prevalencia_comorbilidades_elixhauser.png
    ├── fig05_evolucion_temporal_egresos_mortalidad.png
    └── fig06_distribucion_tipo_alta_censura.png
```

---

## 🚀 Suite de Ejecución de Cálculos

Los cálculos de la Fase 2 se ejecutan de manera reproducible e independiente a través de los siguientes módulos:

```bash
# 1. Modelado primario M2+CAP, sensibilidades M2 y M3, y generación de benchmarks canonical
python "Analisis exploratorio/01_modelado_benchmarks_mortalidad_m2cap.py"

# 2. Evaluación de calibración multidimensional preingreso y deriva temporal de N_dx (2019-2024)
python "Analisis exploratorio/02_calibracion_preingreso_y_deriva_ndx.py"

# 3. Análisis de sensibilidad POA por supresión de agudos (I26, G93.4, R56) y POA-26
python "Analisis exploratorio/03_sensibilidad_complicaciones_poa.py"

# 4. Enlace determinístico de transferencias a 30 días, cotas de salida lambda_salida y estrés Byar
python "Analisis exploratorio/04_analisis_traslados_y_estres_censura.py"

# 5. Validación out-of-sample de estadía con Leave-One-Hospital-Out (LOHO) y bootstrap
python "Analisis exploratorio/05_validacion_estadia_loho_y_bootstrap.py"

# 6. Generación de las 6 figuras científicas EDA (300 DPI)
python "Analisis exploratorio/06_generar_graficos_eda.py"
```

---

## 📊 Documentos Oficiales en Formato Word (DOCX)

A solicitud de la comisión revisora, todos los reportes metodológicos cuentan con su versión correspondiente en Microsoft Word (`.docx`), formateada con estilos tipográficos profesionales, encabezados sombreados en azul marino, bordes limpios en tablas y cajas de llamado (*callouts*) para advertencias metodológicas:
1. [`resolucion_observaciones_fase2.docx`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/resolucion_observaciones_fase2.docx): Dictamen completo con especificación M2+CAP, cotas de salida, Byar stress reconciliado y deriva de codificación.
2. [`informe_verificacion_calidad_fase2.docx`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/informe_verificacion_calidad_fase2.docx): Auditoría cuantitativa de calidad de datos, nulos y consistencia CONSORT.
3. [`README.docx`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/README.docx): Catálogo de scripts y artefactos de la Fase 2.
