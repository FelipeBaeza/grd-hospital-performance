# Resolución Definitiva de Observaciones de Metodología y Auditoría Empírica - Fase 2
**Proyecto de Tesis:** Indicador Hospitalario $O/E$ Ajustado por Riesgo con Machine Learning  
**Base de Datos:** Egresos Hospitalarios FONASA 2019–2024 ($N = 5.808.536$ registros)  
**Fecha de Consolidación:** Octubre 2026

---

## 1. Cascada CONSORT Unificada: Alcance Adultos y Exclusión de MDC 14

Se presenta la cascada secuencial estricta generada desde un único script determinista, adoptando la exclusión limpia de toda la casuística obstétrica (MDC 14 completo: $640.429$ episodios) y delimitando formalmente el estudio a **Atención Hospitalaria de Agudos en Adultos ($\ge 18$ años)**:

| Paso de la Cascada CONSORT | Criterio Metodológico / Clínico | Episodios Excluidos | Población Remanente |
| :--- | :--- | :---: | :---: |
| **Población Bruta Inicial** | Universo total de egresos FONASA (2019–2024) | — | **5.808.536** |
| **(-) EX01: No agrupables** | Código GRD nulo, MDC 0 o código de error 99xxx | 5.073 | 5.803.463 |
| **(-) EX02: Neonatología** | Recién nacidos y patología perinatal (MDC 15) | 146.928 | 5.656.535 |
| **(-) EX03: Obstetricia Completa** | Embarazo, parto y puerperio (MDC 14 completo) | 640.429 | 5.016.106 |
| **(-) EX04: Población Pediátrica** | Menores de 18 años (<18) o edad no derivable | 830.627 | 4.185.479 |
| **(-) EX05: Actividad Ambulatoria** | Cirugía Mayor Ambulatoria (CMA) y Hosp. Diurna | 812.549 | **3.372.930** |
| **(=) COHORTE INPATIENT ADULTOS** | **Base General de Agudos Adultos ($\ge 18$ años)** | — | **3.372.930** |

---

### 1.1 Ramas Analíticas de la Cohorte Inpatient de Adultos ($N = 3.372.930$)

* **Rama de Mortalidad Inpatient Adultos:**
  - **Defunciones Intrahospitalarias ($M = 1$):** **165.234** ($4,90\%$ de la cohorte Inpatient).
  - **Sobrevivientes Evaluables ($M = 0$):** **3.050.955** ($90,45\%$).
  - **(=) Total Casos con Desenlace Conocido (Evaluables Mortalidad):** **3.216.189** ($95,35\%$).
    - **Tasa de Mortalidad Inpatient Adultos:** **$5,138\%$** ($165.234 / 3.216.189$).
  - **Traslados Censurados ($M = \text{Null}$):** **156.741** ($4,65\%$).

* **Rama de Estadía (en Sobrevivientes Evaluables $N = 3.050.955$):**
  - **Estancia Positiva $> 0$ días (Cohorte de Estadía Adultos):** **2.902.643** ($95,14\%$).
  - **Estancia $= 0$ días en sobrevivientes:** **148.308** ($4,86\%$).
  - **Estancia nula:** **4** episodios.

---

### 1.2 Partición Temporal y Balance Analítico por Rama

| Partición Temporal | Total Inpatient Adultos | Evaluables Mortalidad | Defunciones ($M=1$) | Tasa Mortalidad | Cohorte Estadía ($>0$ d) | Censurados (Traslado) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Histórico (2019)** | 641.791 | 610.647 | 28.471 | 4,66% | 549.648 | 31.144 |
| **Desarrollo (2020–2022)** | **1.522.815** | **1.453.396** | **86.697** | **5,97%** | **1.303.954** | **69.419** |
| - *Año 2020* | 486.808 | 463.787 | 29.084 | 6,27% | 415.101 | 23.021 |
| - *Año 2021* | 507.387 | 482.011 | 31.134 | 6,46% | 430.510 | 25.376 |
| - *Año 2022* | 528.620 | 507.598 | 26.479 | 5,22% | 458.343 | 21.022 |
| **Calibración (2023)** | **582.957** | **557.402** | **24.225** | **4,35%** | **507.892** | **25.555** |
| **Evaluación OOS (2024)** | **625.367** | **594.744** | **25.841** | **4,34%** | **541.149** | **30.623** |
| **Total General** | **3.372.930** | **3.216.189** | **165.234** | **5,138%** | **2.902.643** | **156.741** |

---

### 1.3 Auditoría de Nulos de FECHA_INGRESO y Edad

1. **FECHA_INGRESO:** Se encuentra corregida en la fila 15 de la matriz maestra, registrando con exactitud **71 nulos crudos** ($5.808.465$ registros válidos, $99,9988\%$).
2. **Identidad de los 92 nulos de edad:** Queda auditada y confirmada la relación booleana de la capa Gold:
   $$\text{Nulos Edad} = \text{Fecha Nac Nula } (37) + \text{Fecha Ing Nula } (71) - \text{Intersección } (16) = \mathbf{92}$$

---

### 1.4 Urgencia y Homologación DEIS

* La categoría de ingreso codificada como `HOSPITALIZACIÓN EN URGENCIA` existió en la base de datos nacional **únicamente durante el año 2019**, reportada por **48 de los 72 hospitales** (con una censura por traslado del $26,6\%$).
* Desde el año 2020 en adelante, el DEIS homologó administrativamente la captura consolidando todas las hospitalizaciones cerradas bajo la categoría `HOSPITALIZACIÓN`.

---

# 2. Decisiones de Modelado y Selección Empírica

### 2.1 Modelo de Mortalidad: Selección y Congelamiento de Hiperparámetro $C$

Dado el gran volumen muestral del conjunto de desarrollo ($N = 1.453.396$ episodios), la regularización con $C = 0,05$ retiene casi la totalidad de las covariables candidatas perdiendo capacidad discriminativa entre predictores genuinos y ruido marginal.

* **Comparación de Regularización:**
  - $C = 0,010$: 34 variables activas | AUROC OOS = **0,8848**
  - $C = 0,050$: 39 variables activas | AUROC OOS = **0,8855**
  - **Decisión de Congelamiento:** Se adopta formalmente **$C = 0,010$**. La diferencia de AUROC es de apenas $+0,0007$ (menos de una décima de punto porcentual), logrando un modelo más parsimonioso y robusto que elimina 5 variables marginales inestables.

* **Desglose de Comorbilidades de Elixhauser en Estabilidad:**
  De las 28 comorbilidades crónicas canónicas evaluadas (tras excluir formalmente `ELIX_02`, `ELIX_22` y `ELIX_25`), **exactamente 13 comorbilidades crónicas** alcanzan una frecuencia de selección del $100\%$ en las 50 réplicas bootstrap:
  - `ELIX_01` (Insuficiencia cardíaca congestiva)
  - `ELIX_03` (Valvulopatía)
  - `ELIX_04` (Trastornos de la circulación pulmonar)
  - `ELIX_06` (Hipertensión no complicada)
  - `ELIX_07` (Hipertensión complicada)
  - `ELIX_08` (Parálisis)
  - `ELIX_09` (Otros trastornos neurológicos)
  - `ELIX_10` (Enfermedad pulmonar crónica / EPOC)
  - `ELIX_14` (Insuficiencia renal crónica)
  - `ELIX_15` (Enfermedad hepática)
  - `ELIX_19` (Cáncer metastásico)
  - `ELIX_20` (Tumor sólido sin metástasis)
  - `ELIX_24` (Pérdida de peso patológica / Desnutrición)
  
  Otras 7 comorbilidades se retienen en el rango $70\%–98\%$ (`ELIX_05`, `ELIX_11`, `ELIX_12`, `ELIX_21`, `ELIX_23`, `ELIX_26`, `ELIX_28`).  
  Quedan formalmente descartadas por inestabilidad / coeficiente nulo: `ELIX_13` (Hipotiroidismo), `ELIX_16` (Úlcera péptica), `ELIX_17` (VIH/SIDA), `ELIX_18` (Linfoma, $52\%$), `ELIX_27` (Anemia por deficiencia), `ELIX_29` (Abuso de drogas, $44\%$), `ELIX_30` (Psicosis) y `ELIX_31` (Depresión).

* **Encuadre Metodológico de $\Delta\text{AUROC} = -0,0114$ y `ELIX_25`:**
  La reducción de AUROC en $0,0114$ al excluir `ELIX_02` (Arritmias), `ELIX_22` (Coagulopatía) y `ELIX_25` (Desequilibrio hidroelectrolítico) demuestra empíricamente que aportan señal estadística para la predicción de mortalidad. Sin embargo, debido a que la base nacional del DEIS carece del indicador *Present on Admission* (POA), resulta imposible deslindar estadísticamente si `ELIX_25` corresponde a una deshidratación basal o a una falla metabólica intrahospitalaria terminal. Por tanto, su exclusión se fundamenta como una **regla de gobernanza clínica y estructural**: evitar que un modelo de ajuste por riesgo compense o justifique a hospitales con altas tasas de complicaciones hidroelectrolíticas adquiridas durante la estancia.

---

### 2.2 Diagnósticos Secundarios Centrados, No-Linealidad y Complejidad Hospitalaria

* **Contexto de Calibración 2023:**
  La mortalidad hospitalaria nacional descendió desde $5,97\%$ en el trienio pandémico (2020–2022) a $4,35\%$ en 2023. En consecuencia, un modelo entrenado en 2020–2022 exhibe un **$O/E$ global previo a la recalibración de $0,728$** ($4,35 / 5,97$).
  Por ello, el estrato de $6–10$ diagnósticos con $O/E = 0,986$ crudo está calibrado respecto a la escala unitaria pero sobrepredicho respecto al promedio anual; tras la recalibración del intercepto para 2023 ($O/E_{\text{global}} = 1,000$), dicho estrato se sitúa en $O/E \approx 1,35$, mientras que los estratos de $0$ y $1$ diagnósticos ($O/E \approx 0,11–0,19$) reflejan una sobrepredicción persistente.
* **Escalamiento No-Lineal de Codificación:**
  La diferencia lineal ($\widetilde{N}_{\text{dx}, ij} = N_{\text{dx}, ij} - \bar{N}_{\text{dx}, j}$) asume una relación aditiva, mientras que la intensidad de codificación entre hospitales escala de manera multiplicativa. En el contrato de modelado definitivo se adopta la **razón de diagnósticos frente a la media hospitalaria** ($R_{\text{dx}, ij} = N_{\text{dx}, ij} / \bar{N}_{\text{dx}, j}$) o su percentil relativo, complementado con splines cúbicos restringidos para absorber la curvatura extrema en pacientes con $\ge 11$ diagnósticos.
* **Intervalo de Confianza del Coeficiente de Upcoding:**
  La correlación de Spearman entre el indicador $O/E$ hospitalario y el promedio de diagnósticos secundarios tras el centrado es:
  $$r_s = +0,0381 \quad [95\%\text{ IC}: -0,207 \text{ a } +0,283]$$
  El intervalo de confianza contiene al cero, confirmando la ausencia de correlación estadística significativa con el volumen de codificación.
* **Preservación del Gradiente de Complejidad Institucional:**
  Para verificar que el centrado no borra la señal legítima de gravedad entre centros de distinta complejidad, se auditó el $O/E$ antes y después del centrado:
  - **Hospitales de Alta Complejidad (Top 50% volumen):** $O/E_{\text{crudo}} = 1,0170 \longrightarrow O/E_{\text{centrado}} = 1,0186$ ($\Delta = +0,0016$).
  - **Hospitales de Mediana/Baja Complejidad:** $O/E_{\text{crudo}} = 0,9450 \longrightarrow O/E_{\text{centrado}} = 0,9404$ ($\Delta = -0,0045$).
  El gradiente de severidad estructural entre niveles de atención se mantiene completamente intacto.

---

### 2.3 Modelo de Estadía (Gamma $p = 2,0$): Regularización, Estabilidad y Escalas

* **Truncamiento $p99$ Auditado:**
  En la cohorte definitiva de adultos inpatient sobrevivientes, el percentil 99 de estancia real es de **$54,0$ días en la partición de desarrollo (2020–2022)** y de **$51,0$ días en 2023** (el valor previo de 60 días correspondía a la cohorte preliminar no filtrada).
* **Escala de los Coeficientes (Enlace Logarítmico):**
  La variable `EDAD_ANIOS` ingresa estandarizada ($z$-score, media $58,2$, desviación estándar $18,6$ años). Por consiguiente, el coeficiente $\beta = +0,0854$ equivale a un incremento multiplicativo de $\exp(0,0854) = 1,089$ ($+8,9\%$ de estancia) por cada desviación estándar de edad, lo que representa aproximadamente un $+0,47\%$ de mayor estancia por cada año cumplido. Las variables binarias ingresan sin escalar (efecto directo en log-días).
* **Barrido de Regularización L2 ($\alpha$) y Estabilidad en Estadía:**
  - $\alpha = 0,0001$: Deviance Explained $D^2 = 12,74\%$
  - $\alpha = 0,0010$: $D^2 = 12,72\%$
  - $\alpha = 0,0100$: $D^2 = 12,65\%$ (óptimo para estabilidad de coeficientes)
  - $\alpha = 0,1000$: $D^2 = 11,80\%$
* **Variables Retenidas al 100% en Estadía:**
  `INGRESO_URGENCIA` ($\beta = +0,377$), `CIE10_CAP_K` ($\beta = -0,145$), `CIE10_CAP_N` ($\beta = -0,090$), `INGRESO_CRITICO` ($\beta = +0,087$), `EDAD_ANIOS` ($\beta = +0,085$), `SEXO_MASCULINO` ($\beta = -0,068$), `ELIX_24` Desnutrición ($\beta = +0,065$), `CIE10_CAP_C` Cáncer ($\beta = +0,063$), `DERIVADO_OTRO_HOSPITAL` ($\beta = +0,052$), `ELIX_14` Renal crónica ($\beta = +0,048$).

---

### 2.4 Recálculo del IEMC sobre la Misma Cohorte y Análisis de Discordancia

Se recalculó tanto la norma GRD nacional como el modelo Gamma ML sobre la **misma cohorte estricta** (adultos $\ge 18$, sin obstetricia, sobrevivientes, estancia positiva $>0$, truncada a $p99 = 54$ días):
* **Correlación Nacional en los 65 Hospitales de Agudos de Adultos (2023):**
  $$r_{\text{Pearson}} = \mathbf{0,8395} \qquad \rho_{\text{Spearman}} = \mathbf{0,8330}$$
  Superando ampliamente el umbral contractual de validación ($r \ge 0,50$).

---

### 2.5 Recálculo de la Sensibilidad de Día 0 en la Cohorte Adulta

Se recalculó de manera simétrica la exclusión del día 0 (eliminando los episodios de estancia 0 tanto del numerador $O$ como del denominador ajustado $E$) sobre la cohorte adulta evaluable de 2023:
* **Correlación del Ranking Hospitalario $O/E$:**
  $$r_{\text{Pearson}} = \mathbf{0,9619} \qquad \rho_{\text{Spearman}} = \mathbf{0,9447}$$
  El valor de $\rho = 0,9447$ confirma la alta estabilidad del ranking de desempeño hospitalario, sustituyendo al $0,9975$ previo que correspondía a una corrida preliminar no simétrica.

---

# 3. Auditoría de Establecimientos, Nombres Oficiales y Fusión de Campus

### 3.1 Nombres Oficiales de Hospitales Pediátricos (Catálogo DEIS)

Verificados directamente contra el catálogo oficial de establecimientos del DEIS ([Hospitales.csv](file:///home/felipe/Documentos/Proyecto%20final/Seminario/Hospitales.csv)):
* **`109101`:** **Hospital Clínico de Niños Dr. Roberto del Río** (Servicio de Salud Metropolitano Norte, Independencia, Santiago).
* **`112102`:** **Hospital de Niños Dr. Luis Calvo Mackenna** (Servicio de Salud Metropolitano Oriente, Providencia, Santiago).
* **`113130`:** **Hospital Dr. Exequiel González Cortés** (Servicio de Salud Metropolitano Sur, San Miguel, Santiago).

---

### 3.2 Dinámica de la Red Hospitalaria y Unidad de Análisis

La composición del panel hospitalario evoluciona por incorporación progresiva de centros al sistema GRD centralizado:
* **2019–2022:** 65 hospitales reportantes en total (**62 hospitales de agudos de adultos**, al excluir los 3 pediátricos monográficos).
* **2023:** 68 hospitales reportantes (**65 hospitales de agudos de adultos**). Se incorporan 3 establecimientos:
  - `110110`: Instituto Traumatológico Dr. Teodoro Gebauer (Santiago)
  - `119101`: Hospital de Tomé (Talcahuano)
  - `118106`: Hospital de Lota (Concepción)
* **2024:** 72 hospitales reportantes (**68 hospitales de agudos de adultos** tras consolidar Quillota). Se incorporan 4 establecimientos:
  - `200717`: Hospital Biprovincial Quillota Petorca
  - `116111`: Hospital San Juan de Dios de Cauquenes
  - `116107`: Hospital de Constitución
  - `119102`: Hospital Penco - Lirquén

---

### 3.3 Auditoría de Pares de Campus: Corrección de Asignación Territorial

La revisión contra el catálogo oficial del DEIS reveló que varios pares propuestos previamente correspondían a **establecimientos distintos en comunas diferentes pertenecientes al mismo Servicio de Salud**, y no a un mismo campus físico:
1. **`116105` y `116111`:** `116105` es el Hospital Regional de Talca y `116111` es el Hospital de Cauquenes (separados por más de 100 km). No corresponden a Temuco ni a Padre Las Casas.
2. **`118100` y `118106`:** `118100` es el Hospital Guillermo Grant Benavente de Concepción y `118106` es el Hospital de Lota (distantes 40 km).
3. **`119100` y `119101`:** `119100` es el Hospital Las Higueras de Talcahuano y `119101` es el Hospital de Tomé.
4. **`110100` y `110110`:** `110100` es el Hospital San Juan de Dios y `110110` es el Instituto Traumatológico.
5. **`111100`:** Hospital Clínico San Borja Arriarán (pertenece al SS Metropolitano Central; no tiene relación institucional ni de campus con `110110`).

**Decisión Metodológica Fundamental:** Todos ellos son **establecimientos hospitalarios independientes con derivaciones interhospitalarias reales**. Tratar sus transferencias como movimientos internos habría vulnerado gravemente la regla de censura de emisores. Por tanto, se mantienen estrictamente como unidades independientes.

* **El único caso genuino de reemplazo y transición de campus:**
  - **Hospital San Martín de Quillota (`107101`) $\longrightarrow$ Hospital Biprovincial Quillota Petorca (`200717`):**
    En 2024 operaron en paralelo durante el traslado de pacientes al nuevo edificio.
    - Censura por traslado `107101`: **$6,68\%$** ($N = 14.245$)
    - Censura por traslado `200717`: **$3,88\%$** ($N = 10.354$)
    - Censura combinada de la entidad Quillota en 2024: **$5,50\%$** ($N = 24.599$).
* **Linares (`116108`):** El Hospital Presidente Carlos Ibáñez del Campo de Linares opera ininterrumpidamente bajo el código `116108`. El código `119102` corresponde en realidad al Hospital Penco-Lirquén (Talcahuano), el cual ingresó al registro GRD en 2024 como nuevo prestador sin constituir ningún quiebre de serie para Linares.

---

# 4. Estado de Citas y Normativa

1. **Adams et al. (2010):**
   - Cita oficial: Adams JL, Mehrotra A, Thomas JW, McGlynn EA. *"Physician Cost Profiling — Reliability and Risk of Misclassification"*. **N Engl J Med** 2010; 362:1014–1021.
   - Enmarcación: Los umbrales de confiabilidad ($0,70$, $0,80$, $0,90$) se definen como una **convención psicométrica por analogía** basada en Nunnally & Bernstein (*Psychometric Theory*, 3.ª ed., 1994).
2. **Normativa de Codificación COVID-19:**
   - Consignada como: *"Codificación operativa de emergencia adoptada centralizadamente por el DEIS en la base de datos nacional, clasificada como pendiente de validación mediante acto administrativo ministerial público"*.
3. **Versión del Agrupador GRD:**
   - Consignada textualmente como: *"Versión no documentada formalmente por el DEIS (la base de datos utiliza la cabecera `IR_29301`, sin ficha técnica pública oficial de especificación de release)"*.
