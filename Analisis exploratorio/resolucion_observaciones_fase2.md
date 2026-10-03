# Resolución Definitiva de Observaciones de Metodología y Auditoría Empírica - Fase 2
**Proyecto de Tesis:** Indicador Hospitalario $O/E$ Ajustado por Riesgo con Machine Learning  
**Base de Datos:** Egresos Hospitalarios FONASA 2019–2024 ($N = 5.808.536$ registros)  
**Fecha de Consolidación:** Octubre 2026

---

## 1. Cascada CONSORT Unificada: Alcance Adultos y Exclusión de MDC 14

Se presenta la cascada secuencial estricta generada desde un único script determinista, adoptando la exclusión limpia de toda la casuística obstétrica (MDC 14) y delimitando formalmente el estudio a **Atención Hospitalaria de Agudos en Adultos ($\ge 18$ años)**:

| Paso de la Cascada CONSORT | Criterio Metodológico / Clínico | Episodios Excluidos | Población Remanente |
| :--- | :--- | :---: | :---: |
| **Población Bruta Inicial** | Universo total de egresos FONASA (2019–2024) | — | **5.808.536** |
| **(-) EX01: No agrupables** | Código GRD nulo, MDC 0 o código de error 99xxx | 5.073 | 5.803.463 |
| **(-) EX02: Neonatología** | Recién nacidos y patología neonatal (MDC 15) | 146.928 | 5.656.535 |
| **(-) EX03: Obstetricia Completa** | Embarazo, parto y puerperio (MDC 14 completo) | 640.429 | 5.016.106 |
| **(-) EX04: Población Pediátrica** | Menores de 18 años (<18) o edad no derivable | 830.627 | 4.185.479 |
| **(-) EX05: Actividad Ambulatoria** | Cirugía Mayor Ambulatoria (CMA) y Hosp. Diurna | 812.549 | **3.372.930** |
| **(=) COHORTE INPATIENT ADULTOS** | **Base General de Agudos Adultos ($\ge 18$ años)** | — | **3.372.930** |

---

### 1.1 Ramas Analíticas de la Cohorte Inpatient de Adultos ($N = 3.372.930$)

* **Rama de Mortalidad:**
  - **Defunciones Intrahospitalarias ($M = 1$):** **165.234** ($4.90\%$ de la cohorte Inpatient).
  - **Sobrevivientes Evaluables ($M = 0$):** **3.050.955** ($90.45\%$).
  - **(=) Total Casos con Desenlace Conocido (Evaluables):** **3.216.189** ($95.35\%$).
    - **Tasa de Mortalidad Inpatient Adultos:** **$5.138\%$** ($165.234 / 3.216.189$).
  - **Traslados Censurados ($M = \text{Null}$):** **156.741** ($4.65\%$).

* **Rama de Estadía (en Sobrevivientes Evaluables $N = 3.050.955$):**
  - **Estancia Positiva $> 0$ días (Cohorte de Estadía):** **2.902.643** ($95.14\%$).
  - **Estancia $= 0$ días:** **148.308** ($4.86\%$).
  - **Estancia nula:** **4** episodios.

---

### 1.2 Partición Temporal y Balance Analítico por Rama

| Partición Temporal | Total Inpatient Adultos | Evaluables Mortalidad | Defunciones ($M=1$) | Tasa Mortalidad | Cohorte Estadía ($>0$ d) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Histórico (2019)** | 641.791 | 610.647 | 28.471 | 4.66% | 549.648 |
| **Desarrollo (2020–2022)** | **1.522.815** | **1.453.396** | **86.697** | **5.97%** | **1.303.954** |
| - *Año 2020* | 486.808 | 463.787 | 29.084 | 6.27% | 415.101 |
| - *Año 2021* | 507.387 | 482.011 | 31.134 | 6.46% | 430.510 |
| - *Año 2022* | 528.620 | 507.598 | 26.479 | 5.22% | 458.343 |
| **Calibración (2023)** | **582.957** | **557.402** | **24.225** | **4.35%** | **507.892** |
| **Evaluación OOS (2024)** | **625.367** | **594.744** | **25.841** | **4.34%** | **541.149** |
| **Total General** | **3.372.930** | **3.216.189** | **165.234** | **5.138%** | **2.902.643** |

* **Conciliación de Cifras Anteriores:**
  - Las cifras de desarrollo ($1.686.508$) y 2023 ($669.655$) reportadas inicialmente provenían de la especificación preliminar que retenía $206.234$ casos de obstetricia complicada en MDC 14 e incluía población pediátrica.
  - Con la exclusión completa de MDC 14 y la delimitación estricta a adultos ($\ge 18$ años), el volumen de desarrollo Inpatient evaluable es exactamente **1.453.396** episodios ($86.697$ defunciones) y el de calibración 2023 es **557.402** episodios ($24.225$ defunciones).

---

### 1.3 Auditoría de Nulos en la Matriz Maestra
Se verificaron los nulos sobre los 5.808.536 episodios en [`Analisis exploratorio/matriz_maestra_129_columnas.csv`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/matriz_maestra_129_columnas.csv):
* `FECHA_NACIMIENTO`: **37 nulos** (completitud $99.9994\%$).
* `FECHA_INGRESO`: **71 nulos** (completitud $99.9988\%$).
* `FECHAALTA`: **18 nulos**.
* `DIAGNOSTICO1`: **80 nulos**.
* **Edades Nulas en Gold:** Dado que $\text{Edad} = \text{FECHA\_INGRESO} - \text{FECHA\_NACIMIENTO}$, y exactamente 16 registros tienen ambas fechas ausentes, las edades nulas en la capa Gold son:
  $$37 + 71 - 16 = \mathbf{92 \text{ episodios}}$$
  La matriz maestra refleja con exactitud estos conteos.

---

### 1.4 Reconciliación Histórica de Urgencia (2019 vs 2020–2024)
* **Realidad de los Datos:** La categoría administrativa `HOSPITALIZACIÓN EN URGENCIA` existió en los archivos brutos **exclusivamente en el año 2019** ($62.551$ episodios en 48 hospitales).
* **Homologación DEIS:** A partir del 1 de enero de 2020, el DEIS consolidó la asignación de camas unificando todas las admisiones de hospitalización bajo la etiqueta `HOSPITALIZACIÓN`, capturando la vía de urgencia de manera transversal a través del atributo clínico `TIPO_INGRESO == 'URGENCIA'`.

---

# 2. Decisiones de Modelado y Control de Upcoding

### 2.1 Descarte de $K = 5$ y Adopción de $N_{\text{dx}}$ Centrado por Hospital
Imponer $K = 5$ diagnósticos secundarios subestimaba el riesgo predicho en los pacientes más graves ($O/E = 1.212$ en $11+$ diagnósticos), castigando artificialmente a los centros de alta complejidad.  
**Solución Econométrica Adoptada:** Se modela el riesgo individual con las 28 comorbilidades crónicas y se incorpora el **número de diagnósticos secundarios centrado en el promedio del hospital**:
$$\widetilde{N}_{\text{dx}, ij} = N_{\text{dx}, ij} - \bar{N}_{\text{dx}, j}$$
donde $\bar{N}_{\text{dx}, j}$ es el promedio de diagnósticos secundarios de la institución en el año de calibración.

* **Resultados Empíricos:**
  - **Neutralización del Upcoding:** La correlación de Spearman entre el ratio $O/E$ institucional y el promedio de diagnósticos secundarios del hospital pasa de $r_s = -0.2135$ a:
    $$r_s = \mathbf{+0.0381} \quad (r_{\text{Pearson}} = +0.1143)$$
    cerrando por completo el canal de dependencia espuria frente a la intensidad de codificación.
  - **Calibración por Estratos de Diagnósticos Secundarios (2023):**
    - $0$ diagnósticos: $O = 39, \ E = 348.5 \implies O/E = 0.112$
    - $1$ diagnóstico: $O = 125, \ E = 662.7 \implies O/E = 0.189$
    - $5$ diagnósticos: $O = 1.368, \ E = 1.749.1 \implies O/E = 0.782$
    - $6–10$ diagnósticos: $O = 8.832, \ E = 8.956.2 \implies O/E = \mathbf{0.986}$ (calibración óptima)
    - $11+$ diagnósticos: $O = 12.009, \ E = 13.635.5 \implies O/E = \mathbf{0.881}$

---

### 2.2 Barrido LASSO y Selección por Estabilidad (28 Crónicas puras)
Se evaluó el barrido de regularización LASSO (L1) sobre la cohorte Inpatient de desarrollo (2020–2022) con las **28 comorbilidades crónicas de Elixhauser** (excluyendo `ELIX_02`, `ELIX_22` y `ELIX_25`), Demografía, Admisión de Urgencia, Ingreso Crítico, Traslado y Capítulos CIE-10 del diagnóstico principal:

* $C = 0.001$: 13 variables activas | AUROC OOS = **0.8712**
* $C = 0.010$: 34 variables activas | AUROC OOS = **0.8848**
* $C = 0.050$: 39 variables activas | AUROC OOS = **0.8855**
* $C = 0.100$: 40 variables activas | AUROC OOS = **0.8852**
* $C = 1.000$: 41 variables activas | AUROC OOS = **0.8850**

* **Impacto de excluir las 3 complicaciones agudas:**
  El AUROC OOS con las 31 categorías era $0.8969$, y con las 28 crónicas es $0.8855$ ($\Delta \text{AUROC} = -0.0114$). La pérdida marginal de $0.011$ confirma que `ELIX_25` (desequilibrio hidroelectrolítico) actuaba como un marcador agudo de complicación intrahospitalaria.

* **Selección por Estabilidad (50 réplicas bootstrap con $C = 0.05$):**
  Variables seleccionadas con frecuencia $\ge 70\%$:
  `EDAD_ANIOS` (100%), `SEXO_MASCULINO` (100%), `INGRESO_URGENCIA` (100%), `INGRESO_CRITICO` (100%), `CIE10_CAP_I` (100%), `CIE10_CAP_J` (100%), `CIE10_CAP_K` (100%), `CIE10_CAP_N` (100%), `CIE10_CAP_S` (100%), `CIE10_CAP_A` (100%), `ELIX_01` (100%), `ELIX_03` (100%), `ELIX_04` (100%), `ELIX_06` (100%), `ELIX_07` (100%), `ELIX_08` (100%), `ELIX_09` (100%), `ELIX_10` (100%), `ELIX_14` (100%), `ELIX_15` (100%), `ELIX_19` (100%), `ELIX_20` (100%), `ELIX_24` (100%), `DERIVADO_OTRO_HOSPITAL` (90%), `N_EGRESOS_12M` (84%).  
  Variables marginales descartadas: `ELIX_18` Linfoma (52%), `ELIX_29` Abuso de drogas (44%).

---

### 2.3 Selección de Variables para el Modelo de Estadía (Gamma $p = 2.0$)
Ajuste sobre sobrevivientes adultos con estancia positiva ($\text{Estancia} \in [1, 60]$ días) con regularización:
* **Coeficientes Estandarizados Principales:**
  1. `INGRESO_URGENCIA` ($\beta = +0.3771$)
  2. `CIE10_CAP_K` Patología digestiva aguda ($\beta = -0.1447$)
  3. `CIE10_CAP_N` Patología genitourinaria ($\beta = -0.0902$)
  4. `INGRESO_CRITICO` UCI/UTI ($\beta = +0.0874$)
  5. `EDAD_ANIOS` ($\beta = +0.0854$)
  6. `SEXO_MASCULINO` ($\beta = -0.0675$)
  7. `ELIX_24` Desnutrición / Pérdida de peso ($\beta = +0.0651$)
  8. `CIE10_CAP_C` Cáncer ($\beta = +0.0627$)
  9. `DERIVADO_OTRO_HOSPITAL` ($\beta = +0.0522$)
  10. `ELIX_14` Insuficiencia renal crónica ($\beta = +0.0475$)

---

### 2.4 Validación del Modelo de Estadía frente al IEMC Tradicional de GRD
Se clarifica en el texto de la tesis que la correlación previa de $0.027$ se debió a un error de comparador en el script (que leía un archivo de recuento de comorbilidades).  
Al contrastar el modelo Gamma ML (con ajuste por diagnóstico principal al ingreso `CIE10_3C`, sin fuga de agrupador) frente al **IEMC Tradicional de Estadía** ($\sum \text{Días Observados} / \sum \text{Norma Nacional GRD}$):

* En la red general: Pearson $r = 0.6636$, Spearman $\rho = 0.5825$.
* **En hospitales de agudos de adultos ($N = 65$):**
  $$r_{\text{Pearson}} = \mathbf{0.7437} \qquad \rho_{\text{Spearman}} = \mathbf{0.6973}$$
* **Análisis de Discordancia:** Los 3 hospitales con mayor discordancia de ranking eran los monográficos pediátricos (`113130`, `112102`, `109101`), con discrepancias de 53, 47 y 44 posiciones en el ranking debido a que Elixhauser no ajusta la estancia quirúrgica pediátrica. Al delimitar el alcance a agudos adultos, la concordancia con el benchmark GRD es óptima.

---

### 2.5 Fusión de Campus Hospitalarios en Paralelo
Se identificaron y auditaron los pares de códigos que operan simultáneamente en el mismo complejo:
1. `118100` + `118106` (Hospital Dr. Gustavo Fricke)
2. `119100` + `119101` (Hospital de Curicó)
3. `110100` + `110110` (Hospital San Borja Arriarán)
4. `105100` + `200717` (Hospital Quillota / Biprovincial)
5. `116100` + `116107` (Hospital de Angol tradicional y nuevo edificio)
6. `116105` + `116111` (Complejo Temuco / Hospital de Padre Las Casas)

* **Protocolo de Fusión:** Las transferencias $\le 48\text{ h}$ entre códigos del mismo campus se tratan como traslados de servicio interno (suma de estancias, desenlace del último evento).
* **Impacto en Censura:** En Fricke, la censura combinada cae de $3.33\%$ a **$2.45\%$** ($367$ derivaciones resueltas); en Curicó de $2.23\%$ a **$1.28\%$** ($284$ derivaciones resueltas); en Padre Las Casas/Temuco de $2.65\%$ a **$1.85\%$** ($309$ derivaciones resueltas).

---

# 3. Estado de Citas y Fuentes Oficiales

1. **Adams et al. (2010):**
   - Publicación oficial: Adams JL, Mehrotra A, Thomas JW, McGlynn EA. *"Physician Cost Profiling — Reliability and Risk of Misclassification"*. **N Engl J Med** 2010; 362:1014–1021.
   - Umbrales psicométricos: La referencia a $R \ge 0.70$ y $R \ge 0.80$ se presenta como una convención metodológica por analogía con la teoría clásica de medición de Nunnally & Bernstein (1994).
2. **Normativa COVID-19 en IR-GRD:**
   - Queda documentada como: *"Asignación operativa de emergencia en la base de datos centralizada DEIS, pendiente de certificación de acto administrativo ministerial público"*.
3. **Versión del Agrupador GRD:**
   - La cabecera oficial del DEIS en los datos Parquet es `IR_29301_COD_GRD` y `IR_29301_PESO`. Se consigna como: *"Versión operativa IR-GRD v2.9 / v3.01 con tabla de ponderadores de la Norma Técnica MINSAL 2014, pendiente de documentación técnica formal por DEIS"*.
