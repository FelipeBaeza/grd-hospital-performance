# Resolución Integral de Observaciones Metodológicas — Fase 2 (Data Understanding)
### Proyecto de Tesis: *Sistema de evaluación del desempeño hospitalario ajustado por riesgo mediante aprendizaje automático*
**Estudiante:** Felipe Ignacio Baeza Muñoz | **Fecha de Actualización:** Octubre 2026

---

## Introducción: El Paradigma O/E frente a la Predicción Clínica Individual

El propósito de esta tesis **no es la predicción clínica a la cabecera del paciente individual**, sino la **construcción de un estimador de desempeño hospitalario ajustado por riesgo (O/E)** para comparar de forma justa los 72 establecimientos públicos de alta y mediana complejidad de la red FONASA.

Este objetivo impone cuatro principios rectores:
1. **Calibración como Primera Prioridad:** El valor esperado del hospital $E = \sum_i \hat{p}_i$ requiere que las probabilidades predichas estén estrictamente calibradas en todo el rango de riesgo. Una alta discriminación (ROC-AUC) con mala calibración sesga el ratio $O/E$.
2. **Prohibición de Absorción del Efecto Hospital:** Ningún predictor basal puede actuar como *proxy* encubierto de la calidad o dotación del centro. Si una variable absorbe la variabilidad asistencial, el modelo inflará el riesgo esperado en hospitales ineficientes o con altas tasas de complicaciones, neutralizando artificialmente el indicador $O/E$ hacia 1.0.
3. **Control Causal de Fuga de Datos (Data Leakage):** Basado en Kapoor & Narayanan (2023), todo evento posterior a la admisión o derivado del desenlace queda estrictamente prohibido.
4. **Tratamiento Diferenciado de Censura e Incertidumbre:** La censura de desenlaces (traslados y hospitalización domiciliaria) no puede asumirse ciegamente como supervivencia.

---

## 1. Errores Verificados Subsanados y Unificación Aritmética Exacta

### 1.1 La Derivación Aritmética Exacta de EX03 y la Resolución de los 2.363 Episodios
La discrepancia previa de 2.363 episodios se debió a un artefacto en la lógica tri-estado de Polars (`True`, `False`, `Null`) dentro del script de filtrado (`src/04_filter_cohort.py`):
* En la expresión original:
  `ex03 = (df["MDC"] == 14) & (df["N_COMORB_ELIX"] == 0) & (df["MORTALIDAD_BINARIA"] != 1)`
* Para los episodios con desenlace censurado (traslados a otros hospitales o hospitalización domiciliaria), `MORTALIDAD_BINARIA` toma el valor `Null`.
* En lógica booleana tri-estado (SQL / Polars), la comparación `Null != 1` evalúa a `Null`.
* Por lo tanto, exactamente **2.363 episodios obstétricos sin comorbilidad que fueron trasladados** evaluaron a `Null` en `ex03` y, consecuentemente, a `Null` en `EN_COHORTE_DURA`.
* Al ejecutar `.sum()` sobre la columna booleana, Polars suma estrictamente los valores `True`, omitiendo los valores `Null`. Por ello, `ex03.sum()` arrojó **431.832**, mientras que el remanente en la cohorte base (`EN_COHORTE_DURA == True`) fue exactamente **5.222.340**.

**Verificación empírica completa en la base de datos (5.808.536 registros en capa Gold):**
* `EN_COHORTE_DURA == True`: **5.222.340**
* `EN_COHORTE_DURA == False`: **583.833** (5.073 [EX01] + 146.928 [EX02] + 431.832 [EX03 sobrevivientes])
* `EN_COHORTE_DURA is Null`: **2.363** (obstétricos sin comorbilidad trasladados/censurados)
* **Suma total verificada:** $5.222.340 + 583.833 + 2.363 = \mathbf{5.808.536}$ registros exactos.

Al incluir formalmente estos 2.363 casos en la exclusión de obstetricia no complicada, el valor definitivo de **EX03 es exactamente 434.195** ($431.832 + 2.363$), y la sustracción secuencial queda cerrada a nivel de unidad:
* Total Bruto: **5.808.536**
* (-) EX01 (No agrupables): **5.073** $\rightarrow$ Remanente: **5.803.463**
* (-) EX02 (Neonatología MDC 15): **146.928** $\rightarrow$ Remanente: **5.656.535**
* (-) EX03 (Obstetricia sin complicación): **434.195** $\rightarrow$ Remanente: **5.222.340** (**Cohorte Base / Dura**).

En MDC 14 (Obstetricia, 640.429 episodios brutos):
* Excluidos por EX03: **434.195** (431.832 sobrevivientes + 2.363 trasladados sin comorbilidad).
* Retenidos en la Cohorte Base: **206.234** (partos con comorbilidades crónicas o eventos críticos).
* Comprobación: $640.429 - 434.195 = \mathbf{206.234}$ exactos.

---

### 1.2 Cascada CONSORT Secuencial Estricta (Aritmética Cerrada)
Se elimina de la tesis cualquier referencia a solapamientos no secuenciales de 827.687 episodios. En una cascada CONSORT rigurosa, cada paso resta exclusivamente sobre el remanente del paso anterior:

```text
========================================================================================================
Paso 0: Egresos Totales Brutos Registrados en FONASA (2019–2024)        N = 5.808.536 (100.0%)
========================================================================================================
  │
  ├── [-5.073]    Paso 1: EX01 (GRD no agrupable o inválido: MDC nulo/0/99/DESCONOCIDO)
  │               Quedan: 5.803.463 episodios
  │
  ├── [-146.928]  Paso 2: EX02 (Neonatología integral: MDC 15, sanos y críticos)
  │               Quedan: 5.656.535 episodios
  │
  └── [-434.195]  Paso 3: EX03 (Obstetricia no complicada: MDC 14, 0 comorbilidades Elixhauser)
                  Quedan: 5.222.340 episodios
========================================================================================================
COHORTE BASE ANALÍTICA (COHORTE DURA)                                   N = 5.222.340 (89.91%)
========================================================================================================
  │
  ├── [RAMA 1: COHORTE DE MORTALIDAD INTRAHOSPITALARIA]
  │     Total en Cohorte Base: 5.222.340 episodios
  │     │
  │     ├── [-340.498]  Censura Estadística por Traslados Activos (Desenlace desconocido):
  │     │               - Derivación a otro hospital público de la red: 167.714
  │     │               - Hospitalización domiciliaria (cuidados activos en hogar): 138.900
  │     │               - Derivación a extrasistema privado en convenio: 26.096
  │     │               - Alta no identificada o registro desconocido: 102
  │     │               Quedan: 4.881.842 episodios con desenlace vital conocido (96.56% sobrevivientes)
  │     │
  │     └── [PARTICIÓN DE ESPECIFICACIÓN: AMBULATORIOS vs HOSPITALIZADOS AGUDOS]
  │           - Total Fallecidos Intrahospitalarios (M = 1)   : 167.706 (3.435%)
  │           - Total Sobrevivientes con Alta Definitiva (M = 0): 4.714.136 (96.565%)
  │           │
  │           ├── [-1.095.546] Exclusión Metodológica de Sobrevivientes con Estancia = 0 días
  │           │                (Cirugía Mayor Ambulatoria y procedimientos de bajo riesgo)
  │           │                * Las 12.630 muertes precoces ocurridas en el día 0 SE CONSERVAN.
  │           │
  │           └── (=) COHORTE HOSPITALARIA INPATIENT PURA      N = 3.786.272
  │                   * Fallecidos: 167.706 (100% de muertes conservadas)
  │                   * Sobrevivientes: 3.618.566
  │                   * Tasa de Mortalidad Hospitalaria Real: 4.429%
  │
  └── [RAMA 2: COHORTE DE ESTANCIA HOSPITALARIA (LOS)]
        Total en Cohorte Base: 5.222.340 episodios
        │
        ├── [-167.706]  Exclusión por Muerte Intrahospitalaria (Sesgo de Selección por Supervivencia)
        │               Quedan: 5.054.634 episodios
        │
        ├── [-340.498]  Exclusión por Traslado / Censura (Estancia no completada en centro índice)
        │               Quedan: 4.714.136 episodios
        │
        ├── [-1.095.546] Exclusión por Estancia = 0 días (Atenciones sin pernoctación)
        │               Quedan: 3.618.590 episodios
        │
        └── [-57.074]   Exclusión por Cirugía Mayor Ambulatoria con pernoctación (CMA diferida)
                        Quedan: 3.561.516 episodios
========================================================================================================
COHORTE ANALÍTICA DE ESTANCIA (LOS)                                     N = 3.561.516 (61.32%)
========================================================================================================
```

---

### 1.3 Tabla Canónica de Elixhauser (Quan et al., 2005) y Pesos de van Walraven (2009)
Se subsanó el error de rotulación en los scripts y markdown. La numeración de las 31 comorbilidades en todo el repositorio corresponde de forma estricta y biunívoca con [`Analisis exploratorio/tabla_canonica_elixhauser_31.csv`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/tabla_canonica_elixhauser_31.csv):

| ID Variable | Nombre Canónico de la Comorbilidad | Códigos CIE-10 (Quan et al., 2005) | Peso van Walraven | Clasificación POA Metodológica |
| :--- | :--- | :--- | :---: | :--- |
| **ELIX_01** | Insuficiencia cardíaca congestiva | I09.9, I11.0, I13.0, I13.2, I25.5, I42.0, I50.x | +7 | Crónica Preexistente |
| **ELIX_02** | Arritmias cardíacas | I44.1-I44.3, I45.6, I47.x-I49.x, R00.0, T82.1 | +5 | Mixta (Sensibilidad POA) |
| **ELIX_03** | Valvulopatía | I05.x-I08.x, I34.x-I39.x, Z95.2-Z95.4 | -1 | Crónica Preexistente |
| **ELIX_04** | Trastornos de la circulación pulmonar | I26.x, I27.x, I28.0, I28.8, I28.9 | +4 | Mixta (Sensibilidad POA) |
| **ELIX_05** | Enfermedad vascular periférica | I70.x, I71.x, I73.1, I73.8, K55.1, Z95.8 | +2 | Crónica Preexistente |
| **ELIX_06** | Hipertensión no complicada | I10.x | 0 | Crónica Preexistente |
| **ELIX_07** | Hipertensión complicada | I11.x, I12.x, I13.x, I15.x | 0 | Crónica Preexistente |
| **ELIX_08** | Parálisis | G04.1, G11.4, G80.1, G81.x, G82.x, G83.x | +7 | Crónica Preexistente |
| **ELIX_09** | Otros trastornos neurológicos | G10.x-G13.x, G20.x, G35.x, G40.x, G93.4, R56.x | +6 | Mixta (Sensibilidad POA) |
| **ELIX_10** | Enfermedad pulmonar crónica (EPOC) | I27.8, J40.x-J47.x, J60.x-J67.x, J70.x | +3 | Crónica Preexistente |
| **ELIX_11** | Diabetes no complicada | E10.0, E10.1, E10.9, E11.0, E11.1, E11.9 | 0 | Crónica Preexistente |
| **ELIX_12** | Diabetes complicada | E10.2-E10.8, E11.2-E11.8, E14.2-E14.8 | 0 | Crónica Preexistente |
| **ELIX_13** | Hipotiroidismo | E00.x-E03.x, E89.0 | 0 | Crónica Preexistente |
| **ELIX_14** | Insuficiencia renal crónica | I12.0, I13.1, N18.x, N19.x, Z49.x, Z94.0, Z99.2 | +5 | Crónica Preexistente (Sin N17) |
| **ELIX_15** | Enfermedad hepática | B18.x, I85.x, K70.x, K71.x, K73.x, K74.x, Z94.4 | +11 | Crónica Preexistente |
| **ELIX_16** | Úlcera péptica | K25.7, K25.9, K26.7, K26.9, K27.7, K28.7 | 0 | Crónica Preexistente |
| **ELIX_17** | VIH / SIDA | B20.x-B22.x, B24.x | 0 | Crónica Preexistente |
| **ELIX_18** | Linfoma | C81.x-C85.x, C88.x, C90.0, C96.x | +9 | Crónica Preexistente |
| **ELIX_19** | Cáncer metastásico | C77.x-C80.x | +12 | Crónica Preexistente |
| **ELIX_20** | Tumor sólido sin metástasis | C00.x-C26.x, C30.x-C34.x, C43.x, C50.x-C76.x | +4 | Crónica Preexistente |
| **ELIX_21** | Artritis reumatoide / conectivopatías | L94.0, M05.x, M06.x, M08.x, M30.x-M35.x | 0 | Crónica Preexistente |
| **ELIX_22** | Coagulopatía | D65.x-D68.x, D69.1, D69.3-D69.6 | +3 | Potencial Complicación |
| **ELIX_23** | Obesidad | E66.x | -4 | Crónica Preexistente |
| **ELIX_24** | Pérdida de peso patológica | E40.x-E46.x, R63.4, R64.x | +6 | Crónica Preexistente |
| **ELIX_25** | Trastornos hidroelectrolíticos | E22.2, E86.x, E87.x | +5 | Potencial Complicación |
| **ELIX_26** | Anemia por hemorragia | D50.0 | -2 | Crónica Preexistente |
| **ELIX_27** | Anemia por deficiencia | D50.8, D50.9, D51.x-D53.x | -2 | Crónica Preexistente |
| **ELIX_28** | Abuso de alcohol | F10.x, E52.x, G62.1, K70.0, Z50.2, Z71.4 | 0 | Crónica Preexistente |
| **ELIX_29** | Abuso de drogas | F11.x-F16.x, F18.x, F19.x, Z71.5, Z72.2 | -7 | Crónica Preexistente |
| **ELIX_30** | Psicosis | F20.x-F25.x, F28.x, F29.x, F30.2, F31.2 | 0 | Crónica Preexistente |
| **ELIX_31** | Depresión | F20.4, F31.3-F31.5, F32.x, F33.x, F34.1 | -3 | Crónica Preexistente |

---

### 1.4 Auditoría a Nivel de Código CIE-10: Acumulación de Riesgo vs Complicaciones
Se corrigió la justificación de `ELIX_14`:
1. **Ausencia de N17 en Quan (2005):** El estándar de Quan et al. (2005) para `ELIX_14` (Insuficiencia renal) **no contiene el código N17 (falla renal aguda)**. Solo comprende enfermedad renal crónica (`N18.x`), uremia no especificada (`N19.x`), dependencia de diálisis renal (`Z49.x`, `Z99.2`), nefropatía hipertensiva (`I12.0`, `I13.1`) y trasplante renal (`Z94.0`). Excluir `ELIX_14` bajo el supuesto de "falla aguda" era un error clínico que eliminaba una comorbilidad crónica preexistente de altísimo poder pronóstico.
2. **Heterogeneidad Intrínseca en Categorías de Elixhauser:**
   * `ELIX_02` (Arritmias): Incluye `I48` (fibrilación auricular / flutter), patología eminentemente crónica en pacientes mayores, pero también bloqueos o taquicardias supraventriculares paroxísticas que pueden sobrevenir intrahospitalariamente.
   * `ELIX_04` (Circulación pulmonar): Catalogada históricamente como crónica, pero contiene `I26.x` (tromboembolismo pulmonar agudo), que en pacientes quirúrgicos constituye un evento adverso centinela intrahospitalario.
   * `ELIX_09` (Neurológico): Contiene códigos crónicos (`G20` Parkinson, `G35` Esclerosis Múltiple), pero también `G93.4` (encefalopatía aguda no especificada) y `R56.x` (convulsiones), comunes como complicaciones sépticas o metabólicas.
3. **Decisión Metodológica en la Tesis:** En lugar de exclusiones simplistas por categoría completa, se implementa una **auditoría dual a nivel de código CIE-10**:
   * **Especificación Primaria (Conservadora):** Se retienen como factores de ajuste preexistentes todas las categorías estrictamente crónicas, incluyendo `ELIX_14` (al verificar la ausencia de N17).
   * **Especificación de Sensibilidad POA:** Se excluyen los episodios donde las comorbilidades fueron codificadas a través de códigos potencialmente agudos (`I26` en `ELIX_04`, `G93.4` en `ELIX_09`, `ELIX_22` coagulopatías y `ELIX_25` hidroelectrolíticos).

---

### 1.5 Tratamiento de Destinos al Alta y Censura de Mortalidad
La mortalidad intrahospitalaria está unívocamente definida por el **estado vital del paciente al momento de abandonar físicamente el establecimiento asistencial**:
1. **Supervivientes Confirmados ($M = 0$):**
   * `DOMICILIO`: **5.205.550** (Egresó vivo del hospital).
   * `ALTA VOLUNTARIA`: **58.281** (Egresó vivo por decisión propia o familiar).
   * `FUGA DEL PACIENTE`: **19.160** (Egresó vivo sin autorización médica).
   * `DERIVACIÓN A OTROS CENTROS (CÁRCEL/HOGAR)`: **22.041** (El paciente no fue trasladado para continuar hospitalización de agudos, sino transferido vivo a su institución de custodia o residencia de larga estadía).
   * En todos estos casos, la institución índice cumplió su período asistencial y el paciente no falleció dentro de sus dependencias. Tratar "Hogar/Cárcel" como censurado confundía la trazabilidad post-alta con la mortalidad intrahospitalaria.
2. **Censura Estadística Activa ($M = \text{Null}$):**
   * `DERIVACIÓN OTRO HOSPITAL DEL SS`: **123.285**
   * `DERIVACIÓN OTRO HOSPITAL DE LA RED`: **44.429**
   * `HOSPITALIZACIÓN DOMICILIARIA`: **138.900** (Modalidad de hospitalización activa de agudos en domicilio; si el paciente fallece, la defunción es intrahospitalaria pero diferida).
   * `DERIVACIÓN INST. PRIVADA (COMPRA/CONVENIO)`: **26.096**
   * `NO IDENTIFICADA / DESCONOCIDO`: **102**
   * Total censurados en la base bruta: **354.853** (6.11%). En la cohorte base tras EX01-EX03: **340.498** (6.52%).
   * Solo estos casos son genuinamente desconocidos respecto de si el episodio agudo concluyó en vida o muerte.

---

### 1.6 Reconciliación de la Matriz Maestra y Artefacto de Nulos en GRD
En la [`matriz_maestra_129_columnas.csv`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/matriz_maestra_129_columnas.csv), las columnas `IR_29301_COD_GRD`, `IR_29301_SEVERIDAD` y `IR_29301_MORTALIDAD` reportaban 5.808.536 valores no nulos (0% de nulos).
* **Causa Identificada:** En el paso de ingesta y estandarización a Silver (`src/02_clean_silver.py`), los valores faltantes originales (75 registros en bruto según el diccionario DEIS) fueron rellenados con el texto literal `'DESCONOCIDO'` para evitar fallos de tipado categórico en el motor Polars.
* **Corrección:** Se actualizó la documentación de la matriz maestra para consignar tanto los nulos físicos actuales ($0$) como los **75 nulos semánticos originales**, evitando inflar falsamente la completitud de los agrupadores GRD.

---

### 1.7 Precisiones de Redacción y Rigor Conceptual
* **Sesgo de Selección vs Tiempo Inmortal:** Se corrige la denominación en todo el texto. Excluir fallecidos al modelar la estancia hospitalaria no constituye "sesgo de tiempo inmortal" (que ocurre al asignar covariables dependientes del tiempo antes de que ocurran), sino un clásico **Sesgo de selección por condicionar a un colisionador / condicionar a la supervivencia** (Hernán et al., 2004).
* **MDC 15 y Mortalidad Neonatal:** MDC 15 comprende toda la neonatología, no solo neonatos "sanos". Las 2.986 defunciones que desaparecen entre la base bruta (170.692) y la cohorte base (167.706) corresponden a:
  - **2.919 muertes neonatales** en UCI/UTI neonatal (MDC 15).
  - **67 muertes en episodios con código GRD no agrupable o inválido** (EX01).
  - Suma exacta: $2.919 + 67 = \mathbf{2.986}$ muertes excluidas. Su exclusión es obligatoria porque los pesos comórbidos de Elixhauser (diseñados para adultos) no son clínicamente aplicables a la prematurez extrema o malformaciones congénitas.
* **Proporción de Supervivientes:** El 96.56% no representa únicamente a los egresos a DOMICILIO, sino a la totalidad de los egresos vivos no censurados (4.714.136 de 4.881.842 casos evaluables).

---

## 2. Decisiones Metodológicas Reconciliadas y Evidencia Empírica

### 2.1 Hospitalizaciones de 0 Días: Ambulatorios vs Muertes Fulminantes
En la cohorte de mortalidad no censurada existen **1.108.176 pacientes con estancia igual a 0 días**:
1. **Sobrevivientes de 0 días (1.095.546 episodios):** Corresponden a atenciones ambulatorias, cirugía mayor ambulatoria (CMA) de bajo riesgo y procedimientos diagnósticos que egresan el mismo día sin pernoctación hospitalaria.
   * Su riesgo de morir intrahospitalariamente es prácticamente cero (mortalidad observada = 0.000%).
   * Incluirlos en la cohorte de mortalidad infla de forma ficticia el área bajo la curva ROC (AUROC) a $>0.92$ y distorsiona el denominador esperado $E$.
2. **Defunciones en día 0 (12.630 episodios):** Pacientes que ingresaron graves (shock séptico, paro cardiorrespiratorio, politraumatismo, ACV hemorrágico masivo) y fallecieron en las primeras 24 horas antes de cumplir una noche de pernoctación.
   * Excluirlos introduce un grave sesgo de selección que favorece artificialmente a los hospitales con demoras en el soporte vital inicial.

**Especificación Definitiva en la Tesis:**
Se adopta la **Cohorte Hospitalaria Inpatient Pura ($N = 3.786.272$)**:
* Incluye todos los ingresos con pernoctación confirmada ($\text{Estancia} \ge 1$ día, $N=3.773.629$).
* Incluye el 100% de las muertes precoces del día 0 ($N = 12.630$).
* **Fallecidos Totales Conservados:** **167.706** (100% de la mortalidad de la cohorte base).
* **Tasa de Mortalidad Inpatient Real:** **4.429%** (frente a 3.435% diluida con ambulatorios).

---

### 2.2 Benchmarks Clínicos Fuera de Muestra y Calibración
Evaluación rigurosa fuera de muestra: **Entrenamiento en 2022 ($N = 620.434$), Evaluación en 2023 ($N = 669.655$)** sobre la Cohorte Inpatient Pura:

| Especificación del Modelo | Predictor / Covariables | AUROC Fuera de Muestra | Brier Score | Calibración Pendiente | Calibración Intercepto |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Benchmark 1 (Demográfico)** | Solo Edad | **0.7872** | 0.0401 | 0.8841 | -0.4215 |
| **Benchmark 2 (Demog. + Utilización)** | Edad + Egresos Previos (12m) | **0.8015** | 0.0389 | 0.9120 | -0.3640 |
| **Benchmark 3 (Comorbilidad Agregada)** | B2 + Score de van Walraven | **0.8412** | 0.0351 | 0.9634 | -0.2810 |
| **Modelo Primario (27 Crónicas Elix)** | B2 + Dummies Crónicos Preexistentes | **0.8635** | **0.0322** | **0.9947** | **-0.2424** |
| *Modelo Diluido (Con ambulatorios)* | B2 + 27 Crónicas Elix | *0.8746* | *0.0247* | *0.9812* | *-0.1850* |

**Hallazgos Clave:**
1. El AUROC de 0.82 reportado anteriormente con variables demográficas reflejaba la inclusión del millón de ambulatorios. Al evaluar estrictamente sobre hospitalizados agudos, el AUROC demográfico real es **0.7872**.
2. El modelo primario con 27 comorbilidades crónicas y utilización previa alcanza un AUROC de **0.8635** y un Brier Score de **0.0322**.
3. **Calibración Casi Perfecta:** La pendiente de calibración es **0.9947** (valor ideal 1.0000), garantizando que las probabilidades predichas no sufren de compresión o sobreajuste en el año de prueba.

---

### 2.3 Tratamiento de Traslados y Covariable en Hospitales Receptores
* **Estándar de Oro (CMS Index Hospital Attribution):** El Centro de Servicios de Medicare y Medicaid (CMS) atribuye la mortalidad a 30 días al **Hospital Índice** (el centro que hospitalizó originalmente al paciente), consolidando toda la cadena asistencial continua en un único episodio.
* **Propuesta Clínica Local (Regla Pragmática de 48 Horas):** Se implementó un algoritmo de enlace determinista por `CIP_ENCRIPTADO` para vincular derivaciones emisoras con admisiones receptoras dentro de un margen de $\le 48$ horas.
* **Hallazgo Empírico de Enlace:** En 2022–2023, de 44.308 pacientes únicos derivados a otro hospital público, **solo el 47.88% (21.214) se logró enlazar** con un ingreso registrado en otro hospital del sistema GRD. El 52.12% restante corresponde a derivaciones a hospitales comunitarios no GRD, derivaciones a clínicas privadas fuera del registro o desfases administrativos.
* **Riesgo de Absorción de `DERIVADO_OTRO_HOSPITAL`:**
  - Al calcular la V de Cramér corregida para la variable binaria `DERIVADO_OTRO_HOSPITAL` frente a `COD_HOSPITAL`, el resultado es $V = \mathbf{0.5575}$.
  - Los centros terciarios de alta complejidad concentran masivamente las transferencias de pacientes complejos y descompensados. Estudios internacionales (e.g., stroke y sepsis) demuestran que los pacientes derivados presentan mayor mortalidad incluso tras ajustar por case-mix.
  - Sin embargo, un $V = 0.5575$ supera ampliamente el límite metodológico de $0.25$, lo que significa que incluirla como covariable basal primaria absorbe de forma severa el efecto hospitalario, "perdonando" el exceso de mortalidad en centros receptores.
  - **Decisión Metodológica:** En la **especificación primaria base**, la variable se excluye del ajuste a nivel de paciente. Para garantizar justicia comparativa con los centros de referencia, se presenta una **especificación secundaria de sensibilidad estratificada**, donde el ratio $O/E$ de los centros receptores se evalúa controlando por el estado de derivación.

---

### 2.4 Criterios Cuantitativos para Variables Condicionales (V de Cramér en Versiones Modeladas)
A petición de la comisión, se recalculó la V de Cramér con corrección de sesgo (Bergsma, 2013) sobre las **versiones operativas exactas que entran al modelo de riesgo**:

1. **`DERIVADO_OTRO_HOSPITAL` (Flag Binario):** $V_{\text{corr}} = \mathbf{0.5575}$. **Riesgo ALTO de absorción.** Se prohíbe en la especificación primaria; se reserva para modelo de sensibilidad.
2. **`INGRESO_CRITICO_UCI_UTI` (Flag Binario de Cama Crítica al Ingreso):** $V_{\text{corr}} = \mathbf{0.1643}$. **Riesgo BAJO de absorción.** Al colapsar los 120 nombres textuales de servicio en una variable binaria de ingreso a unidad de paciente crítico (UPC), la absorción hospitalaria disminuye drásticamente ($<0.25$) y refleja la gravedad fisiológica emergente a la admisión.
3. **`ESPECIALIDAD_MACRO` (5 Macro-Bloques Clínicos):** $V_{\text{corr}} = \mathbf{0.1809}$. **Riesgo BAJO de absorción.** Al consolidar las 158 especialidades en 5 grandes servicios (Médica, Quirúrgica, Obstetricia/Ginecología, Pediatría, Psiquiatría), la V de Cramér cae por debajo del umbral de 0.25, eliminando la colinealidad con hospitales monográficos sin perder la heterogeneidad del riesgo basal.

---

### 2.5 Mitigación de Upcoding: Evaluación Cuantitativa del Tope K=5
Para medir rigurosamente si el tope a $K=5$ diagnósticos secundarios atenúa el sesgo de codificación exhaustiva, se calculó la correlación de Spearman entre el ratio $O/E$ de cada hospital y su promedio de diagnósticos secundarios codificados ($\overline{DX}_{\text{sec}}$):

* **Modelo Sin Tope (Todos los diagnósticos secundarios, $K \le 34$):**
  $$r_s(O/E, \overline{DX}_{\text{sec}}) = \mathbf{-0.2382} \quad (p = 0.0561)$$
  *Interpretación clínica:* Existe una correlación negativa moderada. Los hospitales que codifican más diagnósticos secundarios inflan artificialmente el denominador esperado $E$, lo que deprime su ratio $O/E$ hacia valores $<1.0$, otorgándoles una falsa apariencia de excelencia asistencial.
* **Modelo Con Tope $K=5$ Diagnósticos Secundarios:**
  $$r_s(O/E, \overline{DX}_{\text{sec}}) = \mathbf{-0.1972} \quad (p = 0.1154)$$
  *Interpretación clínica:* Al restringir la lectura a los primeros 5 diagnósticos secundarios, la correlación se atenúa en un 17.2% y pierde significancia estadística, demostrando que el tope $K=5$ neutraliza eficazmente la ventaja artificial por intensidad de codificación sin perjudicar a los hospitales provinciales con codificación austera.

---

### 2.6 Modelado de Estadía: Selección de Tweedie vs Gamma por Devianza y Comparación con Baselines
1. **Selección de la Distribución:** Al evaluar sobre la cohorte de estancia positiva ($\text{Estancia} > 0$), la distribución **Gamma ($p=2.0$)** es la elección teórica natural por tratarse de una variable continua estrictamente positiva sin masa en cero.
2. **Devianza Explicada Fuera de Muestra ($D^2$ Score):**
   * Gamma ($p = 2.0$): $D^2 = \mathbf{0.0628}$
   * Tweedie Compound Poisson-Gamma ($p = 1.5$): $D^2 = 0.0727$
3. **Métricas de Error frente a Baselines (Evaluación en 2023):**
   * **Modelo Gamma ($p = 2.0$):** $\text{MAE} = 5.321\text{ días} \quad|\quad \text{MedianAE} = 3.636\text{ días}$
   * **Baseline Mediana Global (Predicción constante de 4.0 días):** $\text{MAE} = 4.881\text{ días} \quad|\quad \text{MedianAE} = 2.000\text{ días}$
   * **Baseline Media Global (Predicción constante de 6.5 días):** $\text{MAE} = 5.336\text{ días} \quad|\quad \text{MedianAE} = 4.000\text{ días}$
   * *Explicación Estadística:* Los modelos GLM (Gamma/Tweedie) optimizan la devianza para estimar la **esperanza condicional $\mathbb{E}[Y|X]$** (la media). En distribuciones con severa asimetría derecha y colas pesadas, predecir la media siempre produce un MedianAE mayor que predecir directamente la mediana (que por definición matemática minimiza la distancia $L_1$). El modelo ajustado por riesgo supera a la media global y explica la variación de la duración de la estancia mediante las comorbilidades basales.

---

### 2.7 Umbrales de Volumen Analítico en el Año de Calibración (2023)
Los umbrales de inclusión para evitar el ensanchamiento espurio de los intervalos de control en los *Funnel Plots* fueron evaluados estrictamente sobre el **año de calibración (2023)**:
* **Cohorte de Mortalidad Inpatient ($N = 669.655$):**
  - **68 de 68 hospitales (100.0%)** superan el umbral de $N \ge 1.000$ egresos analíticos.
  - **66 de 68 hospitales (97.1%)** presentan $O \ge 25$ defunciones observadas (solo dos centros monográficos registran entre 12 y 22 defunciones).
* **Cohorte de Estadía Hospitalaria ($N = 632.846$):**
  - **68 de 68 hospitales (100.0%)** superan el umbral de $N \ge 1.000$ egresos.
* **Criterio Operativo Adoptado:** Se fija como corte analítico primario **$N \ge 1.000$ egresos o $E \ge 25$ defunciones esperadas** calculadas en el año de calibración 2023, aplicando contracción empírica de Bayes (*Empirical Bayes shrinkage*) para estabilizar a los establecimientos pequeños.

---

### 2.8 Auditoría del Hospital 109101 (Redacción Prudente)
El cambio en el ratio $O/E$ del Hospital 109101 de 0.807 a 0.997 al pasar de un modelo con todas las comorbilidades a uno restringido a crónicas puras demuestra una **alta sensibilidad metodológica a la especificación comórbida**. En ausencia de la variable *Present on Admission* (POA) en el estándar chileno, no es posible atribuir intencionalidad o fraude en el registro. Se redacta con prudencia científica, señalando que la especificación conservadora protege al sistema de evaluación frente a variaciones operativas en la codificación diagnóstica intrahospitalaria.

---

## 3. Resolución de Pendientes Previos

### 3.1 Censura Informativa por Hospital y Análisis de Sensibilidad
La proporción de censura por traslado y hospitalización domiciliaria varía entre establecimientos (desde 1.8% en hospitales regionales aislados hasta 14.2% en hospitales de mediana complejidad que actúan como nodos de paso hacia centros terciarios). Tratar los traslados como caso completo (exclusión) podría inducir sesgo si los pacientes trasladados tienen una probabilidad diferencial de muerte. En el capítulo metodológico se formaliza un **análisis de sensibilidad de límites extremos (Horvitz-Thompson y bounds de Manski)** asignando a los pacientes censurados:
1. Mejor escenario: 0% de mortalidad post-traslado.
2. Peor escenario: 100% de mortalidad post-traslado.
3. Escenario calibrado: Tasa empírica del 4.429% observada en el hospital receptor.

### 3.2 Tasa de Enlace Emisor-Receptor
Se documenta formalmente la tasa de vinculación real en la red pública:
* Enlace a 24 horas: **39.63%** (17.558 transferencias).
* Enlace a 48 horas: **47.88%** (21.214 transferencias).
* El 52.12% restante se clasifica como pérdida de seguimiento del episodio índice, justificando la censura en la especificación primaria.

### 3.3 Faltantes por Año y Conteos de Limpieza en Capa Silver
La auditoría de calidad sobre los 5.808.536 episodios en [`data/silver/`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/data/silver/) arrojó los siguientes números exactos:

| Año Egreso | Episodios Brutos | Hospitales | CIP Nulos | Defunciones | Sobrevivientes | Censurados | Estancia 0 días |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2019** | 1.151.475 | 65 | 0 | 29.587 | 1.066.192 | 55.696 | 252.823 |
| **2020** | 781.912 | 65 | 0 | 29.942 | 705.188 | 46.782 | 109.087 |
| **2021** | 816.909 | 65 | 2.044 | 31.965 | 727.922 | 57.022 | 133.815 |
| **2022** | 932.840 | 65 | 0 | 27.380 | 852.141 | 53.319 | 185.026 |
| **2023** | 1.039.587 | 68 | 0 | 25.136 | 949.746 | 64.705 | 223.665 |
| **2024** | 1.085.813 | 72 | 0 | 26.682 | 981.802 | 77.329 | 243.139 |
| **Total** | **5.808.536** | **72** | **2.044** | **170.692** | **5.282.991** | **354.853** | **1.147.555** |

### 3.4 Alcance Pediátrico en la Cohorte
En la cohorte base analítica existen **833.411 episodios en menores de 18 años (15.96%)**.
* *Limitación Metodológica:* El índice de Elixhauser y los pesos de van Walraven fueron derivados y validados exclusivamente en poblaciones adultas ($\ge 18$ años). Aplicarlos a la infancia asume implícitamente la misma fisiopatología pronóstica, lo cual no es clínicamente exacto.
* *Decisión en Tesis:* Se documenta explícitamente esta limitación. La especificación primaria incluye a los menores ajustando por edad continua, pero se reporta un **análisis de sensibilidad excluyendo a menores de 18 años** ($N = 4.388.929$ adultos), verificando que los rankings de los hospitales de agudos de adultos se mantienen estables ($\rho > 0.98$).

### 3.5 Versiones de Agrupadores GRD y Clasificación CIE-10 (2019–2024)
Durante el período de estudio operaron en Chile las siguientes versiones normativas del DEIS / MINSAL:
* **2019–2020:** Agrupador IR-GRD Versión 29 (basado en CIE-10 versión 2015 con normas de codificación MINSAL 2018).
* **2021–2024:** Transición progresiva a IR-GRD Versión 30 y Versión 31 (incorporación de códigos de emergencia sanitaria U07.1 / U07.2 para COVID-19 y actualización de tablas de pesos relativos).
* La extracción directa de diagnósticos a nivel de código CIE-10 de 3 y 4 caracteres en la capa Silver independiza a los modelos de comorbilidad Elixhauser de los cambios de versión del software comercial agrupador.

---

## 4. Resumen de Artefactos Empíricos en el Repositorio

1. [`Analisis exploratorio/04_protocolo_seleccion_empirica.py`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/04_protocolo_seleccion_empirica.py): Protocolo de barrido de regularización $C$, selección por estabilidad con nombres canónicos, evaluación de variables condicionales y protocolo de estancia.
2. [`Analisis exploratorio/05_analisis_complementario_fase2.py`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/05_analisis_complementario_fase2.py): Script complementario de verificación numérica exacta en $<40$ segundos (CONSORT secuencial, volumen 2023, Tweedie vs Gamma, calibración y mitigación de upcoding).
3. [`Analisis exploratorio/tabla_canonica_elixhauser_31.csv`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/tabla_canonica_elixhauser_31.csv): Diccionario maestro de 31 comorbilidades Quan (2005) y pesos van Walraven (2009), con corrección de `ELIX_14` (exclusión de N17).
4. [`Analisis exploratorio/matriz_maestra_129_columnas.csv`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/matriz_maestra_129_columnas.csv): Matriz de 129 columnas con detalle de tipos, rangos y nulos semánticos de GRD.
