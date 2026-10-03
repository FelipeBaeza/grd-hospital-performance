# Resolución Integral de Observaciones Metodológicas — Fase 2 (Data Understanding)
### Proyecto de Tesis: *Sistema de evaluación del desempeño hospitalario ajustado por riesgo mediante aprendizaje automático*
**Estudiante:** Felipe Ignacio Baeza Muñoz | **Fecha de Elaboración:** Octubre 2026

---

## Introducción y Cambio de Paradigma: Ajuste de Riesgo para Indicadores O/E

El objetivo central de este trabajo **no es la predicción pronóstica individual a la cabecera del paciente**, sino la **construcción de un indicador de desempeño hospitalario ajustado por riesgo (O/E)** que permita comparar con justicia y validez causal a los 72 establecimientos de la red pública chilena (FONASA).

Este propósito impone tres restricciones metodológicas fundamentales:
1. **Prioridad Absoluta de la Calibración:** Para que el valor esperado $E = \sum_i \hat{p}_i$ sea válido a nivel de centro asistencial, el modelo de Machine Learning debe estar perfectamente calibrado en todas las regiones del espacio de riesgo. Una discriminación alta (ROC-AUC) con mala calibración distorsionaría el ratio O/E.
2. **Prohibición de Absorción del Efecto Hospital:** Ninguna variable predictiva basal puede actuar como *proxy* de la calidad, eficiencia o dotación interna del hospital. Si una variable absorbe la variabilidad asistencial del centro, el riesgo esperado absorberá la mala (o buena) práctica, neutralizando el O/E hacia 1.0 y ocultando las diferencias reales de desempeño.
3. **Control Estricto de Fuga de Datos (Data Leakage):** Basado en los principios de Kapoor & Narayanan (2023), toda variable posterior al ingreso o derivada del desenlace queda terminantemente excluida.

---

## 1. Unificación de Contradicciones Documentales

Se auditaron y unificaron de forma unánime las discrepancias identificadas entre el informe de texto, los diccionarios y las matrices de roles:

### 1.1 Postura Definitiva sobre `ETNIA`
- **Decisión Unificada:** Se clasifica como **`AUDITORIA_SENSIBILIDAD`**.
- **Justificación Bioética y Metodológica:** `ETNIA` **NUNCA entra al modelo predictivo de riesgo O/E** como predictor basal. Incluir el origen étnico en la función de riesgo implicaría normalizar como "esperable" una peor sobrevida o mayor estancia producto de barreras socioeconómicas o racismo estructural. Sin embargo, se deriva la variable binaria `PUEBLO_ORIGINARIO` para conservarla fuera del modelo con el propósito exclusivo de realizar **auditorías post-hoc de equidad algorítmica** (evaluar si el indicador O/E genera impacto dispar entre hospitales según la composición demográfica de su población atendida).

### 1.2 Definición Operativa de Desenlace Desconocido (Censura)
- **Decisión Unificada:**
  - **Alta a Domicilio (`DOMICILIO`):** Es un desenlace confirmado de **supervivencia** (`MORTALIDAD_BINARIA = 0`, `NO_CENSURADO`). Ingresa plenamente al modelo de mortalidad y de estancia.
  - **Hospitalización Domiciliaria (`HOSPITALIZACIÓN DOMICILIARIA`):** Representa un **desenlace no cerrado** en el recinto físico (el paciente continúa bajo cuidados agudos en su hogar). Se trata como **`CENSURADO`** en el modelo hospitalario base.
  - **Derivaciones a Otros Hospitales:** Son eventos censurados a nivel de centro emisor (`MORTALIDAD_BINARIA = NULL`), salvo que se aplique la regla de atribución de cadenas de traslado (Sección 4).

### 1.3 Reconciliación Exacta de las 129 Columnas
La tabla de bloques de la Fase 2 sumaba erróneamente 118 en el texto. La auditoría exhaustiva de los archivos crudos de FONASA y la capa Bronze confirma la siguiente partición canónica que suma **exactamente 129 columnas**:

| Bloque Temático | N° Columnas | Columnas Incluidas | Rol Principal |
|---|:---:|---|---|
| **1. Identificación y Red Asistencial** | **3** | `COD_HOSPITAL`, `CIP_ENCRIPTADO`, `SERVICIO_SALUD` | `LLAVE_AGREGACION` / `LONGITUDINAL` |
| **2. Demografía y Protección Social** | **7** | `SEXO`, `FECHA_NACIMIENTO`, `ETNIA`, `PROVINCIA`, `COMUNA`, `NACIONALIDAD`, `PREVISION` | `FEATURE_BASAL`, `FUENTE_DERIVADA`, `AUDITORIA` |
| **3. Contexto de Ingreso y Procedencia** | **7** | `TIPO_PROCEDENCIA`, `TIPO_INGRESO`, `ESPECIALIDAD_MEDICA`, `TIPO_ACTIVIDAD`, `FECHA_INGRESO`, `SERVICIOINGRESO`, `HOSPPROCEDENCIA` | `FEATURE_BASAL`, `FILTRO_COHORTE`, `CONDICIONAL` |
| **4. Traslados Internos Hospitalarios** | **18** | `FECHATRASLADO1`–`9` (9 cols) + `SERVICIOTRASLADO1`–`9` (9 cols) | `PROHIBIDA_FUGA` (Eventos intra-estadía) |
| **5. Diagnósticos Clínicos** | **35** | `DIAGNOSTICO1` (principal) + `DIAGNOSTICO2`–`DIAGNOSTICO35` (34 secundarios) | `FUENTE_DERIVADA` (deriva Elixhauser y CIE-10) |
| **6. Procedimientos, Quirófano y Pabellón** | **36** | `PROCEDIMIENTO1`–`30` (30), `FECHAPROCEDIMIENTO1` (1), `FECHAINTERV1` (1), `ESPECIALIDADINTERVENCION` (1), `USOSPABELLON` (1), `MEDICOINTERV1_ENCRIPTADO` (1), `MEDICOALTA_ENCRIPTADO` (1) | `PROHIBIDA_FUGA` (Tratamiento post-ingreso) |
| **7. Egreso y Familia Neonatal** | **19** | `FECHAALTA`, `SERVICIOALTA`, `TIPOALTA` (3) + 4 familias neonatales (`CONDICIONDEALTANEONATO1-4`, `PESORN1-4`, `SEXORN1-4`, `RN1-4ESTADO`) (16) | `TARGET_DERIVADA` (3) / `DESCARTAR` (16) |
| **8. Clasificación Agrupador IR-GRD** | **4** | `IR_29301_COD_GRD`, `IR_29301_PESO`, `IR_29301_SEVERIDAD`, `IR_29301_MORTALIDAD` | `FILTRO_COHORTE` (1) / `PROHIBIDA_FUGA` (3) |
| **TOTAL BASE FONASA** | **129** | **Suma matemáticamente verificada** | **129 columnas** |

### 1.4 Reconciliación de Obstetricia, Neonatos y `PESORN1`
- **El problema:** El borrador anterior mencionó informalmente que se excluían "~30.000 episodios", mientras que `PESORN1` registra **795.973 valores no nulos**.
- **La evidencia empírica:** 
  1. `PESORN1` (peso de nacimiento) se registra por duplicado en FONASA: en el egreso de la madre tras el parto (MDC 14) y en el egreso propio del recién nacido si ingresó a hospitalización/neonatología (MDC 15).
  2. La exclusión metodológica CONSORT se diseñó para retirar **partos y neonatos no complicados (sanos)**, ya que no presentan riesgo intrínseco de muerte comparable a la medicina y cirugía de adultos:
     - `ex02_recien_nacido` (GRD neonatal normal/sano): **146.928 episodios excluidos**.
     - `ex03_obstetrico_sin_comp` (Parto vaginal normal sin comorbilidad materna): **431.832 episodios excluidos**.
     - **Total cohorte sana excluida:** **578.760 episodios**.
  3. Los restantes **~217.213 episodios** con `PESORN1` corresponden a cesáreas complejas, patología materna severa (eclampsia, shock hemorrágico) o neonatos patológicos críticos en UCI, los cuales **sí se conservan** en la cohorte ajustada por riesgo.

### 1.5 Traslados Internos vs. Censura
- Las 18 columnas de traslados (`FECHATRASLADO1-9`, `SERVICIOTRASLADO1-9`) registran movimientos entre camas o unidades de un mismo hospital (ej. Urgencia a UCI). **No representan censura**, sino procesos intermedios de atención (`PROHIBIDA_FUGA`).
- La **censura** estadística está determinada única y exclusivamente por `TIPOALTA` al finalizar el episodio hospitalario.

### 1.6 Rol de `TIPO_ACTIVIDAD`
- Es estrictamente un **`FILTRO_COHORTE`**. Filtra pacientes ambulatorios o atenciones abreviadas de hospitalización de día, asegurando que la cohorte de evaluación corresponda a hospitalizaciones cerradas con pernoctación en cama básica o crítica.

### 1.7 Diferenciación entre Columnas Fuente Crudas y Variables Derivadas Finales
- En lugar de referirse a "~12 columnas que entran", se aclara:
  - **12 Columnas Fuente Crudas:** `SEXO`, `FECHA_NACIMIENTO`, `FECHA_INGRESO`, `TIPO_INGRESO`, `TIPO_PROCEDENCIA`, `DIAGNOSTICO1`, `DIAGNOSTICO2..35`, `COMUNA`, `PROVINCIA`, `PREVISION`, `ETNIA`.
  - **47+ Variables Derivadas / Features del Modelo:** Edad en años, sexo, procedencia agrupada, categoría diagnóstica CIE-10 (capítulos y 3 caracteres), 31 dummies de comorbilidad Elixhauser (Quan et al., 2005), Score ponderado de van Walraven, conteo de comorbilidades, indicadores de ingreso en fin de semana, variables de historia previa (`N_EGRESOS_12M`, `DIAS_DESDE_EGRESO_PREVIO`).

---

## 2. Exploración con Números Reales (5.808.536 Registros)

### 2.1 Tasas de Mortalidad Global y Evolución Temporal
Se auditó la totalidad de los egresos de la red pública chilena entre 2019 y 2024:

| Año | Egresos Totales ($N$) | Defunciones ($N$) | Tasa Mortalidad Cruda (%) | Tasa en No Censurados (%) | Egresos Censurados ($N$) | % Censura |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2019** | 1.151.475 | 29.587 | 2.57% | 2.70% | 55.696 | 4.84% |
| **2020** | 781.912 | 29.942 | **3.83%** | **4.07%** | 46.782 | 5.98% |
| **2021** | 816.909 | 31.965 | **3.91%** | **4.21%** | 57.022 | 6.98% |
| **2022** | 932.840 | 27.380 | 2.94% | 3.11% | 53.319 | 5.72% |
| **2023** | 1.039.587 | 25.136 | 2.42% | 2.58% | 64.705 | 6.22% |
| **2024** | 1.085.813 | 26.682 | 2.46% | 2.65% | 77.329 | 7.12% |
| **TOTAL** | **5.808.536** | **170.692** | **2.94%** | **3.13%** | **354.853** | **6.11%** |

*Hallazgo empírico:* La pandemia provocó un doble efecto: una caída de -32% en el volumen de egresos (2020) por suspensión de atenciones quirúrgicas electivas y un incremento del +50% en la tasa de mortalidad intrahospitalaria, reflejando una cohorte mucho más grave.

### 2.2 Distribución de la Estancia Hospitalaria (Sobrevivientes)
Calculada sobre la cohorte de pacientes sobrevivientes con estancia válida:
- **Media:** 5.36 días | **Desviación Estándar:** 11.00 días
- **Percentil 25 (p25):** 1.0 día
- **Mediana (p50):** 2.0 días
- **Percentil 75 (p75):** 6.0 días
- **Percentil 90 (p90):** 13.0 días
- **Percentil 95 (p95):** 20.0 días
- **Percentil 99 (p99):** 46.0 días
- **Máximo crudo:** > 365 días (casos sociales y hospitalizaciones crónicas prolongadas)
- **Asimetría (Skewness):** **14.89** (extrema asimetría a la derecha)
- **Curtosis:** **629.20** (distribución hiper-leptocúrtica con colas ultra pesadas)

*Decisión de Modelado:*
1. La asimetría de 14.89 descarta formalmente el uso de regresión OLS por mínimos cuadrados. Fundamenta el uso de la **distribución Tweedie compuesta Poisson-Gamma** ($1 < p < 2$, con $p=1.5$) mediante LightGBM.
2. **Tope Superior (Capping / Winsorizing):** Se establece un truncamiento a **p99 = 46 días** (o 60 días en análisis de sensibilidad). Esto evita que pacientes con abandono familiar o internaciones judiciales distorsionen el Índice de Estancia Media Ajustada (IEMC) del hospital.

### 2.3 Volumen por Hospital y Umbral Mínimo
- Total hospitales evaluados en la red pública: **72 establecimientos**.
- Distribución de volumen total acumulado (2019-2024):
  - Mínimo: 3.204 egresos | Mediana: 64.306 egresos | Media: 80.674 egresos | Máximo: 284.098 egresos.
- Volumen anual promedio por hospital:
  - Mediana: **10.718 egresos/año** | Media: **13.446 egresos/año**.
  - Hospitales con $\ge 200$ egresos/año: **72 de 72 (100.0%)**.
  - Hospitales con $\ge 500$ egresos/año: **72 de 72 (100.0%)**.
  - Hospitales con $\ge 1.000$ egresos/año: **69 de 72 (95.8%)**.
- **Definición de Umbral:** Se establece **$N \ge 500$ egresos/año**. Ningún hospital de mediana o alta complejidad queda excluido, garantizando suficiente masa crítica para la calibración del ratio O/E mediante *Funnel Plots* y contracción empírica de Bayes (*Empirical Bayes shrinkage*).

---

## 3. Fuga Temporal y Sesgo de Codificación en Diagnósticos (El Problema POA)

En Chile, el registro hospitalario DEIS/FONASA codifica los diagnósticos **únicamente al egreso**. No existe la marca `POA` (*Present on Admission*) de estándares como el CMS estadounidense. Esto genera tres riesgos críticos identificados por la revisión:

### 3.1 Comorbilidades de Elixhauser: Crónicas vs. Complicaciones Intrahospitalarias
Si una condición se produjo durante la estancia como complicación médica o quirúrgica, usarla para predecir el riesgo basal del ingreso constituye **fuga temporal inversa**: el modelo premia al hospital otorgándole un riesgo esperado $E$ mayor por haber complicado al paciente.

Se clasificaron las 31 condiciones de Elixhauser (Quan et al., 2005) en dos categorías operativas:
1. **Crónicas Puras (Invariablemente preexistentes al ingreso):**
   - Diabetes (`ELIX_11`, `ELIX_12`), Hipertensión (`ELIX_06`, `ELIX_07`), Cáncer sólido o metastásico (`ELIX_19`, `ELIX_20`), Enfermedad pulmonar crónica (`ELIX_10`), Parálisis (`ELIX_08`), Artritis reumatoide (`ELIX_23`), Hipotiroidismo (`ELIX_26`).
2. **Potenciales Complicaciones Intrahospitalarias (Requieren auditoría de sensibilidad):**
   - `ELIX_02` (Arritmias cardíacas): Fibrilación auricular perioperatoria o taquiarritmias nosocomiales.
   - `ELIX_25` (Trastornos hidroelectrolíticos): Frecuentemente secundarios al manejo hídrico o farmacológico.
   - `ELIX_22` (Coagulopatías): Consumo de factores post-quirúrgico o sepsis intrahospitalaria.
   - `ELIX_14` (Insuficiencia renal aguda): Falla renal nosocomial por nefrotoxicidad o sepsis.

*Protocolo de Sensibilidad Longitudinal:* Aprovechando que se dispone de `CIP_ENCRIPTADO`, para pacientes con hospitalizaciones previas se calcula una versión del índice de comorbilidad basada **únicamente en los diagnósticos documentados en ingresos anteriores** (garantía temporal estricta de preexistencia).

### 3.2 Intensidad de Codificación Diagnóstica (*Upcoding / Gaming*)
Un hospital docente con un departamento de codificación muy exhaustivo documenta en promedio más diagnósticos secundarios que un hospital regional con menor dotación administrativa.
- **Evidencia hallada en los datos:**
  - Hospital 106102 (alta complejidad metropolitana): **6.62 diagnósticos secundarios promedio por paciente**.
  - Hospital 110150 (hospital provincial): **2.43 diagnósticos secundarios promedio por paciente**.
  - **Brecha:** ¡**4.19 diagnósticos secundarios de diferencia por caso**!
- Si el modelo incluye el número total de diagnósticos (`N_DX_TOTAL`), el hospital que codifica más aumenta artificialmente su riesgo esperado $E$, reduciendo su razón $O/E$ y apareciendo falsamente como un centro de mejor desempeño.
- **Acción metodológica:** `N_DX_TOTAL` queda estrictamente **prohibido**. Las 31 comorbilidades de Elixhauser se tratan con penalización regularizada LASSO (L1) y binarización para limitar el impacto del exceso de codificación administrativa.

### 3.3 Fuga por Códigos Terminales en `DIAGNOSTICO1`
Se realizó una auditoría de códigos que anticipan directamente la muerte (citando a Kapoor & Narayanan, 2023):
- **Paro Cardíaco no Especificado (`I46.9`):** 898 episodios, con una **tasa de mortalidad intrahospitalaria del 95.3%**. El paciente ingresa en paro no resucitado o con maniobras agónicas. Tratar esto como un motivo de ingreso estándar sesga el modelo hacia la memorización del desenlace.
- **Muerte sin Asistencia (`R98`) / Otras Causas Desconocidas (`R99`):** 10 episodios con **100% de mortalidad**.
- **Cuidados Paliativos (`Z51.5`):** 140 episodios. Pacientes en adecuación de esfuerzo terapéutico.
- **Acción:** Se incorporan flags de auditoría y análisis de sensibilidad para evaluar el O/E hospitalario excluyendo pacientes en cuidados paliativos o ingresados en paro cardiorrespiratorio terminal.

---

## 4. Cadenas de Traslado y Vinculación Longitudinal

El análisis exploratorio reveló **354.853 egresos censurados** por traslados externos o derivaciones.

### 4.1 Riesgo de Contaminación de Variables Históricas
Cuando un paciente es trasladado del Hospital A al Hospital B el mismo día:
- El Hospital B ve un ingreso donde `N_EGRESOS_12M` cuenta el egreso de A como "hospitalización previa" a 0 días (`DIAS_DESDE_EGRESO_PREVIO = 0`), interpretándolo como un reingreso temprano fallido en lugar de un traslado continuo.

### 4.2 Regla de Enlace y Atribución (Protocolo CMS Transfer Attribution)
1. **Regla de Enlace Determinista:**
   - Mismo `CIP_ENCRIPTADO`.
   - Fecha de ingreso en receptor $\le$ Fecha de alta en emisor + 1 día.
   - `HOSPPROCEDENCIA` del receptor coincide con `COD_HOSPITAL` del emisor.
2. **Regla de Atribución del Desenlace (CMS Rule):**
   - Si el paciente es derivado y **fallece en el hospital receptor antes de 48 horas** de la admisión, el fallecimiento se atribuye al hospital emisor original (el paciente fue derivado en estado agónico o in extremis).
   - Si el paciente fallece tras una hospitalización prolongada en el receptor (>48 horas) o sobrevive, el desenlace se atribuye al hospital receptor.
3. **Manejo de los 2.044 Registros con CIP Nulo:**
   - No se pueden enlazar longitudinalmente. Se procesan como episodios huérfanos con variables de historial nulas (`HISTORIA_DISPONIBLE = 0`), sin imputación cruzada.

---

## 5. Anexo Exhaustivo de los 12 Valores de `TIPOALTA`

La extracción empírica sobre los 5.8 millones de registros identificó 12 valores en `TIPOALTA`:

| Valor en FONASA | Frecuencia | % | Rol en Mortalidad | Rol en Estancia | Justificación Metodológica |
|---|:---:|:---:|:---:|:---:|---|
| **DOMICILIO** | 5.205.550 | 89.62% | `NO_EVENTO (0)` | `INCLUIDO` | Alta médica estándar con vida. Conclusión clínica normal. |
| **FALLECIDO** | 170.692 | 2.94% | `EVENTO (1)` | `EXCLUIDO` | Muerte intrahospitalaria. Excluido del modelo de estancia para evitar premiar defunciones tempranas. |
| **HOSPITALIZACIÓN DOMICILIARIA** | 138.900 | 2.39% | `CENSURADO` | `EXCLUIDO` | Cuidados continuos en el hogar. Desenlace abierto a nivel hospitalario físico. |
| **DERIVACIÓN OTRO HOSPITAL DEL SERVICIO** | 123.285 | 2.12% | `CENSURADO` | `EXCLUIDO` | Traslado inter-hospitalario de la misma macrorred. Requiere enlace de cadena. |
| **ALTA VOLUNTARIA** | 58.281 | 1.00% | `NO_EVENTO (0)` | `SENSIBILIDAD` | Egreso por desistimiento del paciente. Con vida al egreso; estancia potencialmente truncada. |
| **DERIVACIÓN OTRO HOSPITAL RED NACIONAL** | 44.429 | 0.76% | `CENSURADO` | `EXCLUIDO` | Traslado extra-red a centro de alta complejidad. Desenlace no conocido en origen. |
| **DERIVACIÓN A OTROS CENTROS (HOGAR/CÁRCEL)** | 22.041 | 0.38% | `CENSURADO` | `EXCLUIDO` | Internación en instituciones residenciales o penitenciarias. |
| **FUGA DEL PACIENTE** | 19.160 | 0.33% | `NO_EVENTO (0)` | `SENSIBILIDAD` | Abandono no autorizado. Paciente con vida al egreso. |
| **DERIVACIÓN INST. PRIVADA (COMPRA CAMAS)** | 18.653 | 0.32% | `CENSURADO` | `EXCLUIDO` | Compra de servicios o Ley de Urgencia al sector privado. |
| **DERIVACIÓN INST. PRIVADA (VOLUNTARIO)** | 7.443 | 0.13% | `CENSURADO` | `EXCLUIDO` | Traslado electivo al sector privado financiado por seguros complementarios. |
| **NO IDENTIFICADA** | 80 | 0.00% | `CENSURADO` | `EXCLUIDO` | Error administrativo de interfaz en origen. |
| **DESCONOCIDO** | 22 | 0.00% | `CENSURADO` | `EXCLUIDO` | Registro sin valor codificado. |

---

## 6. Criterios Estadísticos Cuantitativos para Variables Condicionales

Las variables `NIVEL_CUIDADO_INGRESO`, `ESPECIALIDAD_MACRO` y `DERIVADO_OTRO_HOSPITAL` fueron auditadas en los datos de desarrollo (2019-2022) mediante tres métricas objetivas:

1. **V de Cramér contra `COD_HOSPITAL`:**
   - `TIPO_INGRESO`: $V = 0.1540$ (Asociación débil -> Variable basal segura).
   - `TIPO_PROCEDENCIA`: $V = 0.1874$ (Asociación moderada -> Permitida en 3 macro-grupos: urgencia, derivado, programado).
   - `ESPECIALIDAD_MEDICA` (158 categorías crudas): $V = 0.1682$ global, pero con concentración monopólica de ciertas especialidades en hospitales monográficos (ej. Traumatología en Instituto Traumatológico).
2. **Criterio Operativo de Decisión:**
   - Si la inclusión de la variable cambia drásticamente el ranking hospitalario O/E (correlación de Spearman $< 0.95$) o exhibe un coeficiente intraclase $ICC > 0.20$ atribuible al centro, **la variable se agrupa a nivel macro** para neutralizar la absorción del efecto establecimiento.

---

## 7. Protocolo de Selección Empírica de Características

Para superar la selección exclusivamente heurística a priori, se implementó un pipeline en 4 etapas:

1. **Diagnóstico de Colinealidad:**
   - Se demostró que `SCORE_VANWALRAVEN` y `N_COMORB_ELIX` son combinaciones lineales deterministas de las 31 comorbilidades.
   - En modelos lineales regularizados (LASSO/Elastic Net), se prohíbe incluir simultáneamente el score agregado y las 31 variables individuales.
   - En LightGBM, se permite su inclusión conjunta sujeta a la poda de importancia por permutación.
2. **Selección por Estabilidad (Stability Selection de Meinshausen & Bühlmann, 2010):**
   - Se ajustan $B = 50$ remuestreos estratificados sobre submuestras del 63.2% de los datos con penalización L1 regularizada (LASSO $C=0.05$).
   - Se retienen únicamente aquellas características que alcanzan una frecuencia de selección $\hat{\Pi}_k \ge 70\%$.
   - *Resultados confirmados en código:* Edad, Score de van Walraven, Falla cardíaca (`ELIX_01`), Enfermedad vascular periférica (`ELIX_04`), Hipertensión complicada (`ELIX_07`), Linfoma (`ELIX_17`), Cáncer metastásico (`ELIX_21`) y Trastornos hidroelectrolíticos (`ELIX_24`) alcanzaron **100% de estabilidad**.
3. **Validación Cruzada Anidada Agrupada por Paciente (*Nested GroupKFold*):**
   - Basado en Ambroise & McLachlan (2002): el escalado, selección de variables y ajuste de hiperparámetros se realizan estrictamente **dentro del pliegue de entrenamiento interno**, evitando cualquier fuga de optimismo.
   - La agrupación por `CIP_ENCRIPTADO` impide que dos hospitalizaciones del mismo paciente queden divididas entre entrenamiento y test.

---

## 8. Deriva Temporal y Robustez Multianual

1. **Códigos de Emergencia COVID-19 (`U07.1` y `U07.2`):**
   - Estos códigos CIE-10 no existían en 2019. Para evitar que el modelo sufra fallas por categorías fuera de vocabulario, se crearon macro-capítulos CIE-10 donde `U00-U49` se mapea a `CAP22_PROPOSITOS_ESPECIALES`.
2. **Período de Lavado Histórico (*Washout Period*):**
   - El año 2019 presenta obligatoriamente `HISTORIA_DISPONIBLE = 0` (no se dispone de datos de 2018). Se recomienda utilizar 2019 como ventana de acumulación histórica, evaluando los modelos de desempeño hospitalario formalmente sobre el período **2020–2024**.
3. **Estabilidad de Versiones IR-GRD:**
   - La base FONASA utiliza la versión IR-GRD v29.3 / v30.0 de forma estable en el período. Los filtros de cohorte basados en MDC (MDC 14 Obstetricia y MDC 15 Neonatología) permanecen semánticamente homogéneos entre 2019 y 2024.

---

## 9. Ajustes Específicos para Indicadores O/E y Comparación Hospitalaria

1. **Gráficos de Embudo (*Funnel Plots*):**
   - Se grafica el ratio O/E de mortalidad (HSMR) y de estancia (IEMC) frente al volumen de egresos de cada hospital, superponiendo límites de control a $\pm 2\sigma$ (alerta / 95% de confianza) y $\pm 3\sigma$ (alarma / 99.7% de confianza).
2. **Contracción Empírica de Bayes (*Empirical Bayes Shrinkage*):**
   - Para evitar que hospitales pequeños con pocos casos queden clasificados como "atípicos" por mera variabilidad estocástica, sus estimaciones O/E se contraen suavemente hacia el promedio de la red (1.0) en proporción inversa a su error estándar.
3. **Justificación Metodológica de No Ajustar por Nivel Socioeconómico:**
   - Ajustar el modelo de riesgo esperado por el nivel socioeconómico de la comuna o el tramo FONASA haría que un hospital vulnerable tuviera una "expectativa de mortalidad permitida más alta". Desde el punto de vista de la calidad asistencial sanitaria, un paciente de menores recursos merece la misma calidad técnica de atención que uno de mayores recursos. La inequidad territorial se documenta en auditorías estratificadas, pero no se normaliza en el denominador $E$.

---

## 10. Respaldo Bibliográfico y Guías de Reporte

- **Quan et al. (2005):** *Coding algorithms for defining comorbidities in ICD-9-CM and ICD-10 administrative data*. Medical Care, 43(11), 1130-1139. (Define las equivalencias de las 31 condiciones Elixhauser usadas en este código).
- **van Walraven et al. (2009):** *A modification of the Elixhauser comorbidity measure for predicting in-hospital mortality*. Medical Care, 47(6), 626-633. (Valida los pesos empíricos del Score van Walraven).
- **Kapoor & Narayanan (2023):** *Leakage and the reproducibility crisis in machine-learning-based science*. Patterns, 4(9), 100804. (Fundamenta la auditoría de fuga en códigos de ingreso terminales).
- **Riley et al. (2020):** *Calculating the sample size required for developing a clinical prediction model*. BMJ, 368, m441. (Garantiza suficiencia muestral en cohorte de 5.8M).
- **Meinshausen & Bühlmann (2010):** *Stability selection*. Journal of the Royal Statistical Society: Series B, 72(4), 417-473. (Algoritmo de selección de comorbilidades estables).
- **Ambroise & McLachlan (2002):** *Selection bias in gene extraction on the basis of microarray gene-expression data*. PNAS, 99(10), 6562-6566. (Valida la necesidad de validación cruzada anidada para evitar sobreoptimismo).
- **Guías Formales de Reporte:** El proyecto se estructura bajo las recomendaciones de **TRIPOD+AI (2024)** (*Transparent Reporting of a multivariable prediction model of an individual prognosis or diagnosis for Artificial Intelligence*) y se autoevalúa con la herramienta **PROBAST+AI** (*Prediction model Risk Of Bias Assessment Tool*).

---
*Documento metodológico formal integrado en la suite de Análisis Exploratorio de la Tesis.*
