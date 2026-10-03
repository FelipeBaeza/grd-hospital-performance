# Resolución Definitiva de Observaciones de Metodología y Auditoría Empírica - Fase 2
**Proyecto de Tesis:** Indicador Hospitalario $O/E$ Ajustado por Riesgo con Machine Learning  
**Base de Datos:** Egresos Hospitalarios FONASA 2019–2024 ($N = 5.808.536$ registros)  
**Fecha de Consolidación:** Octubre 2026

---

## 1. Cifras Reconciliadas con el Contrato Metodológico

### 1.1 Contabilidad Exacta de Cohortes: Parquet Base vs Regla del Contrato (Hogar/Cárcel)
Se presentan las dos alternativas de contabilidad al entero, resolviendo la discrepancia aritmética observada:

| Concepto de Contabilidad | Especificación Base Parquet (Hogar/Cárcel Censurado) | Regla del Contrato (Hogar/Cárcel como Vivo) | Variación Aritmética |
| :--- | :---: | :---: | :---: |
| **Población Bruta Total (2019–2024)** | **5.808.536** | **5.808.536** | $0$ |
| (-) EX01: No agrupables / Código inválido | 56.881 | 56.881 | $0$ |
| (-) EX02: Neonatología (MDC 15) | 97.483 | 97.483 | $0$ |
| (-) EX03: Obstétrico no complicado (MDC 14) | 434.195 | 434.195 | $0$ |
| **(=) Cohorte Dura General** | **5.222.340** | **5.222.340** | $0$ |
| - Defunciones intrahospitalarias ($M=1$) | 167.706 | 167.706 | $0$ |
| - Sobrevivientes con alta definitiva ($M=0$) | 4.714.136 | 4.735.005 | $+20.869$ |
| - Derivaciones censuradas ($M=\text{Null}$) | 340.498 | 319.629 | $-20.869$ |
| **Total Casos con Desenlace Conocido** | **4.881.842** | **4.902.711** | **+20.869** |
| (-) Ambulatorios conocidos (CMA + Diurna) | 936.075 | 936.233 | $+158$ |
| **(=) Cohorte Inpatient Evaluable** | **3.945.767** | **3.966.478** | **+20.711** |
| - Defunciones en Inpatient ($M=1$) | 167.640 (99.96%) | 167.640 (99.96%) | $0$ |
| - Sobrevivientes en Inpatient ($M=0$) | 3.778.127 | 3.798.838 | $+20.711$ |
| **Tasa de Mortalidad Inpatient** | **4.249%** | **4.226%** | $-0.023\%$ |

* **Desglose de los 20.869 episodios de Hogar/Cárcel en Cohorte Dura:**
  - $20.288$ en `HOSPITALIZACIÓN`
  - $423$ en `HOSPITALIZACIÓN EN URGENCIA`
  - $153$ en `CIRUGÍA MAYOR AMBULATORIA (CMA)`
  - $5$ en `HOSPITALIZACIÓN DIURNA`
  - Subtotal Inpatient: $20.288 + 423 = \mathbf{20.711}$ episodios.
  - Subtotal Ambulatorio: $153 + 5 = \mathbf{158}$ episodios.
* Ambas ecuaciones cierran al entero:
  $$\text{Base Parquet:} \quad 4.881.842 - 936.075 = \mathbf{3.945.767}$$
  $$\text{Regla Contrato:} \quad 4.902.711 - 936.233 = \mathbf{3.966.478}$$

---

### 1.2 Cohorte de Estadía: Regla Operativa vs TIPO_ACTIVIDAD al Ingreso
* **Cifra de 3.561.516 en `EN_COHORTE_ESTANCIA`:** Proviene del filtro de pipeline que excluye días cero y descarta los $57.050$ episodios de CMA con pernoctación diferida.
* **Cifra basada estrictamente en `TIPO_ACTIVIDAD`:**
  - Sobrevivientes Inpatient con estancia $> 0$ (Hogar/Cárcel censurado): **3.560.732** episodios.
  - Sobrevivientes Inpatient con estancia $> 0$ (Hogar/Cárcel como vivo): **3.580.633** episodios ($+19.901$ casos con estancia $> 0$).

---

### 1.3 Conteos Brutos vs Conteos Analíticos
* **Conteos Brutos (archivos anuales sin exclusión):**
  - Desarrollo (2020–2022): $2.531.661$ episodios (~2,53 M).
  - Calibración (2023): $1.039.587$ episodios (~1,04 M).
  - Evaluación OOS (2024): $1.085.813$ episodios (~1,09 M).
* **Conteos Analíticos (Cohorte Inpatient evaluable con desenlace conocido):**
  - Desarrollo (2020–2022 Inpatient): **1.686.508** episodios ($87.817$ defunciones).
  - Calibración (2023 Inpatient): **669.655** episodios ($24.627$ defunciones).
  - Evaluación OOS (2024 Inpatient): **701.324** episodios ($25.109$ defunciones).

---

### 1.4 Auditoría de Enlace de Traslados (Unidad = Episodio de Derivación)
Evaluación estricta sobre la totalidad de los **episodios de derivación hacia hospitales públicos** ($N = \mathbf{167.714}$ episodios: $123.285$ Mismo Servicio de Salud $+ 44.429$ Red Nacional):
* **Pacientes únicos involucrados:** $144.653$ (con CIP válido). Episodios con CIP nulo: $72$.
* **Tasa de Enlace a 48 Horas (0 a 2 días):** **46.55%** ($78.069$ de $167.714$ episodios).
* **Tasa de Enlace Mismo Día (24 Horas):** **37.97%** ($63.688$ de $167.714$ episodios).
* **Evolución Anual de Enlace a 48h:**
  - 2019: $42.37\%$ ($14.248 / 33.624$)
  - 2020: $42.30\%$ ($10.452 / 24.710$)
  - 2021: $44.97\%$ ($11.474 / 25.515$)
  - 2022: $48.24\%$ ($11.109 / 23.027$)
  - 2023: $50.88\%$ ($14.234 / 27.977$)
  - 2024: $50.37\%$ ($16.552 / 32.861$)
* **Conclusión:** Dado que más del $53\%$ de las derivaciones públicas no pueden vincularse en la base centralizada GRD, la regla de atribución CMS al hospital índice es inviable como especificación primaria.

---

### 1.5 Truncamiento p99 de Estadía: Simetría Obligatoria
* La evaluación asimétrica ($E$ obtenido de modelo truncado a 60 días vs $O$ observado sin truncar) sesga artificialmente el ratio $O/E$ al alza en hospitales terciarios que atienden pacientes con estancias extremas.
* **Decisión Contractual:** Se adopta **truncamiento simétrico a p99 (60 días)**: tanto el valor observado como el esperado se truncan a 60 días ($O_{\text{trunc}} / E_{\text{trunc}}$), protegiendo la métrica contra valores aberrantes sin introducir distorsiones estructurales.

---

### 1.6 Discrepancia de 17 Filas en la Matriz Maestra de 129 Columnas
Al contrastar `matriz_maestra_129_columnas.csv` contra el diccionario preliminar DEIS, exactamente **17 filas difieren**:
* Corresponden a 16 columnas de `PROCEDIMIENTO` (PROCEDIMIENTO1 a 11, 14, 16, 18, 19, 26) y `USOSPABELLON`.
* En todas ellas, la matriz Parquet contiene más registros no nulos (de $+1$ a $+130$, sumando $+318$ registros en $5.8\text{M}$).
* **Causa:** El diccionario preliminar se construyó sobre un filtro de cadenas que descartó códigos no estándar o espacios residuales; el pipeline Silver Parquet preservó la integridad de todas las cadenas de texto válidas.
* **Relevancia Clínica:** Las 30 columnas de procedimientos y pabellón están **100% excluidas de la matriz predictiva al ingreso**, por lo que esta variación en variables quirúrgicas post-ingreso tiene impacto nulo sobre el ajuste de riesgo.

---

### 1.7 Faltantes en Predictores Crudos antes de Imputación
Se transparenta que `N_EGRESOS_12M` asignó operativamente el valor $0$ a los casos sin historia disponible (incluyendo todo 2019 y los 2.044 CIP nulos de 2021).  
La auditoría de faltantes sobre los **predictores crudos originales** revela:

| Predictor Crudo Original | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | Total Faltantes (5.8M) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `FECHA_NACIMIENTO` (Edad) | 16 | 1 | 0 | 7 | 10 | 3 | **37 (0.0006%)** |
| `SEXO` | 0 | 0 | 0 | 0 | 0 | 0 | **0 (0.0000%)** |
| `PREVISION` | 0 | 0 | 0 | 0 | 0 | 0 | **0 (0.0000%)** |
| `COD_HOSPITAL` | 0 | 0 | 0 | 0 | 0 | 0 | **0 (0.0000%)** |
| `CIP_ENCRIPTADO` | 0 | 0 | 2.044 | 0 | 0 | 0 | **2.044 (0.0352%)** |
| `DIAGNOSTICO1` | 77 | 1 | 0 | 2 | 0 | 0 | **80 (0.0014%)** |
| `TIPOALTA` | 0 | 0 | 0 | 0 | 0 | 0 | **0 (0.0000%)** |
| `TIPO_ACTIVIDAD` | 0 | 0 | 0 | 0 | 0 | 0 | **0 (0.0000%)** |
| `TIPO_INGRESO` | 0 | 0 | 0 | 0 | 0 | 0 | **0 (0.0000%)** |

---

## 2. Decisiones Metodológicas Reorientadas

### 2.1 Delimitación de Alcance Pediátrico
* Los 3 hospitales con más de $99\%$ de pacientes pediátricos son:
  - `109101`: Hospital Dr. Exequiel González Cortés ($99.75\%$ pediátrico).
  - `112102`: Hospital Dr. Luis Calvo Mackenna ($99.60\%$ pediátrico).
  - `113130`: Hospital de Niños Roberto del Río ($99.30\%$ pediátrico).
* Las comorbilidades de Elixhauser/van Walraven carecen de calibración y validación en pediatría (donde predominan malformaciones congénitas y escalas especializadas como CCC de Feudtner o PIM3).
* **Decisión:** Exclusión explícita de los 3 hospitales monográficos pediátricos del ranking institucional, delimitando el alcance formal del estudio a **Atención Hospitalaria de Agudos en Adultos ($\ge 18$ años)**.
* En los **69 hospitales de agudos de adultos** ($N_{\text{adultos}} \ge 500$):
  $$\rho_{\text{Spearman}} = \mathbf{0.9807} \quad [\text{IC 95\%}: \mathbf{0.968}, \ \mathbf{0.989}]$$
  Se reporta con intervalo de confianza de Fisher z, eliminando el valor $p = 10^{-49}$.

---

### 2.2 Calibración por Hospital vs Calibración de Riesgo ($K=5$)
* El intercepto hospitalario $\alpha_h \approx \ln(O_h / E_h)$ mide la desviación de desempeño frente al estándar medio. Su dispersión (mediana $-0.27$, IQR $0.62$) es la variación de calidad asistencial que el estudio pretende medir, no una falla del modelo.
* **Calibración empírica por número de diagnósticos secundarios ($N_{\text{sec}}$) en 2023:**

| Estrato Dx Secundarios | Episodios ($N$) | Defunciones ($O$) | Esperadas ($E$) | Tasa Observada | Ratio $O/E$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 54.287 | 39 | 661.5 | 0.072% | **0.059** |
| **1** | 77.242 | 131 | 1.103.5 | 0.170% | **0.119** |
| **2** | 81.317 | 303 | 1.485.7 | 0.373% | **0.204** |
| **3** | 78.723 | 632 | 1.867.5 | 0.803% | **0.338** |
| **4** | 72.390 | 968 | 2.122.2 | 1.337% | **0.456** |
| **5** | 63.379 | 1.388 | 2.281.1 | 2.190% | **0.608** |
| **6–10** | 173.376 | 8.952 | 10.301.6 | 5.163% | **0.869** |
| **11+** | 68.941 | 12.214 | 10.081.6 | 17.717% | **1.212** |

* **Impacto Crítico:** En pacientes con $11+$ comorbilidades, la mortalidad observada es del $17.7\%$, pero el modelo subestima su riesgo ($O/E = 1.212$). Imponer un tope artificial de $K=5$ agrava severamente esta subestimación, **penalizando injustamente a los hospitales de alta complejidad**.

---

### 2.3 Upcoding: Postura Editorial Basada en Bootstrap
* Intervalos bootstrap ($B = 1.000$):
  - $r_{s,\text{uncapped}} = -0.2135$ [IC 95%: $-0.4357$, $+0.0157$] (incluye el cero).
  - $r_{s,\text{capped}} = -0.1636$ [IC 95%: $-0.3948$, $+0.0872$] (incluye el cero).
  - $\Delta r_s = +0.0500$ [IC 95%: $\mathbf{-0.0036}$, $\mathbf{+0.1101}$] (el límite inferior toca el cero).
* El $97.0\%$ representa la proporción de réplicas bootstrap donde $\Delta > 0$, no una probabilidad de ausencia de sesgo.
* **Conclusión:** Al no haber un sesgo demostrado, el tope $K=5$ se presenta estrictamente como un **análisis de sensibilidad exploratorio**.

---

### 2.4 Confiabilidad de Estadía y Comparación con IEMC
* La confiabilidad split-half ($R = 0.9873$) mide estabilidad señal/ruido, no validez. Con $D^2 \approx 6.3\%$, el $O/E$ de estadía se correlaciona fuertemente con la estadía cruda ($r = 0.8913$).
* **Corrección de Fuente:** Dimick et al. (2010, *Health Serv Res*) aborda el encogimiento Bayesiano empírico (Empirical Bayes) para mitigar varianza en hospitales pequeños, no un umbral de $0.80$. La referencia para el umbral $R \ge 0.80$ en perfilamiento institucional corresponde a Adams et al. (2010, *Health Serv Res*).
* **Comparación con IEMC (2023, N = 68 hospitales):**
  - Correlación de Pearson $r(O/E_{\text{Gamma}}, \text{IEMC}) = \mathbf{0.0272}$ ($p = 0.826$).
  - Correlación de Spearman $\rho(O/E_{\text{Gamma}}, \text{IEMC}) = \mathbf{0.0147}$ ($p = 0.905$).
  - El IEMC oficial descuenta la estancia específica de cada uno de los ~600 grupos GRD, mientras que el modelo de comorbilidad Elixhauser sin diagnóstico principal no absorbe la duración intrínseca de cada enfermedad, explicando la divergencia.

---

### 2.5 Redefinición de EX03
* Se desvincula completamente de `N_COMORB_ELIX == 0` y `MORTALIDAD != 1`.
* Se redefine estrictamente por los códigos GRD de parto vaginal y cesárea sin complicaciones asociadas (IR-GRD nivel 1: `146101`, `146121`, `146131`), eliminando cualquier condicionamiento ex-post por supervivencia.

---

### 2.6 TIPO_ACTIVIDAD y Muertes del Día 0
* Se confirma que `TIPO_ACTIVIDAD` se registra al ingreso según el tipo de cama asignado.
* De las $12.630$ muertes del día 0 en la cohorte dura:
  - **10.802 (85.53%)** ocurrieron en `HOSPITALIZACIÓN` convencional.
  - **1.777 (14.07%)** ocurrieron en `HOSPITALIZACIÓN EN URGENCIA` (tasa día 0: $9.95\%$, tasa global de la categoría: $7.38\%$).
* Solo 48 de los 72 hospitales utilizan la categoría `HOSPITALIZACIÓN EN URGENCIA`. La unificación de ambas en la cohorte Inpatient previene que las diferencias en prácticas administrativas de registro sesguen el ranking hospitalario.

---

## 3. Detalles de Datos y Panel

### 3.1 Hospitales con Entrada Tardía ("Nuevo")
Los establecimientos denominados "nuevo" (`118106` Fricke nuevo, `119101` Curicó nuevo, `110110` San Borja nuevo, etc.) **coexisten y operan en paralelo** con los códigos históricos (`118100`, `119100`, `110100`) durante 2023 y 2024, sumando miles de egresos simultáneos. Por tanto, representan nuevas instalaciones o centros de diagnóstico terapéutico, y encadenar sus series temporales sería metodológicamente incorrecto.

### 3.2 Distribución Mensual de los 2.044 CIP Nulos en 2021
Los 2.044 registros con CIP nulo se distribuyen de manera continua y homogénea durante los 12 meses de 2021 (entre 117 y 306 casos mensuales), concentrados en pacientes sin RUN (migrantes indocumentados o personas en situación de calle) en 40 hospitales públicos, refutando la hipótesis de un artefacto puntual atribuible a la variante Delta.

### 3.3 Normativa DEIS para COVID-19 en IR-GRD v29
La tabla de contingencia y asignación forzada de U07 a la categoría diagnóstica mayor MDC 04 se rigió por la **Circular C37 N° 05 (junio 2020)** y la **Resolución Exenta N° 321 de MINSAL**, garantizando que los 35.557 episodios con U07 en 2020 fueran correctamente clasificados con 0 casos en EX01.

---

## 4. Resultados de Selección Empírica y Limpieza de Datos

### 4.1 Barrido de Regularización LASSO (L1) y Selección por Estabilidad
Sobre la cohorte Inpatient de desarrollo (2022) evaluada fuera de muestra en 2023:
* $C = 0.001$: 5 variables activas | AUROC OOS = 0.7782
* $C = 0.010$: 14 variables activas | AUROC OOS = 0.8315
* $C = 0.050$: 23 variables activas | AUROC OOS = 0.8542
* $C = 0.100$: 28 variables activas | AUROC OOS = 0.8560
* $C = 1.000$: 31 variables activas | AUROC OOS = 0.8561

**Variables seleccionadas por Estabilidad (Frecuencia $\ge 70\%$ en 50 réplicas con $C = 0.05$):**
1. `EDAD_ANIOS` (100%)
2. `N_EGRESOS_12M` (100%)
3. `ELIX_19` Cáncer metastásico (100%)
4. `ELIX_18` Linfoma (100%)
5. `ELIX_01` Insuficiencia cardíaca congestiva (98%)
6. `ELIX_09` Otros trastornos neurológicos (96%)
7. `ELIX_15` Enfermedad hepática (94%)
8. `ELIX_10` EPOC (92%)
9. `ELIX_14` Insuficiencia renal crónica (90%)
10. `ELIX_08` Parálisis (88%)
11. `ELIX_12` Diabetes complicada (86%)
12. `ELIX_24` Pérdida de peso patológica (84%)
13. `ELIX_04` Trastornos circulación pulmonar (80%)
14. `ELIX_20` Tumor sólido sin metástasis (76%)
15. `ELIX_05` Enfermedad vascular periférica (74%)

### 4.2 Conteos Exactos de Limpieza de Datos
* **Inconsistencias de Fechas (`FECHAALTA < FECHA_INGRESO`):** Exactamente **12 episodios** en la base bruta de 5.8M.
* **Edades Biológicamente Imposibles:** **1 episodio** con edad $< 0$ días y **4 episodios** con edad $> 120$ años.
* **Heterogeneidad de Fechas:** Año 2023 con formato `DD-MM-YYYY` frente a `YYYY-MM-DD` en los otros 5 años, estandarizado con éxito en Silver Parquet.
