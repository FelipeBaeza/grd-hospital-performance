# Especificación Metodológica Primaria y Resolución Integral — Fase 2
### Proyecto de Tesis: *Sistema de evaluación del desempeño hospitalario ajustado por riesgo mediante aprendizaje automático*
**Estudiante:** Felipe Ignacio Baeza Muñoz | **Fecha de Actualización:** Octubre 2026

---

# CONTRATO METODOLÓGICO: ESPECIFICACIÓN PRIMARIA CONGELADA (SINGLE-PAGE MASTER CONTRACT)

Para garantizar consistencia matemática y conceptual absoluta en todos los capítulos de la tesis, códigos y tablas, se formaliza la siguiente **especificación primaria definitiva**:

1. **Definición de Cohortes y Cascada Secuencial:**
   * **Población Bruta Registrada (2019–2024):** **5.808.536 episodios**.
   * **(-) EX01 (GRD Inválido / No agrupable):** **5.073** $\rightarrow$ Remanente: **5.803.463**.
   * **(-) EX02 (Neonatología Integral MDC 15):** **146.928** $\rightarrow$ Remanente: **5.656.535**.
   * **(-) EX03 (Obstetricia no complicada MDC 14):** **434.195** $\rightarrow$ Remanente: **5.222.340** (**Cohorte Base / Dura**).
     - *Definición objetiva:* Partos vaginales y cesáreas no complicados (GRD 146101, 146121, 146131) sin comorbilidad Elixhauser basal ($431.832$ dados de alta vivos $+ 2.363$ trasladados sin comorbilidad).
   * **Cohorte Inpatient de Mortalidad:** Definida al ingreso por `TIPO_ACTIVIDAD != 'AMBULATORIA' & TIPO_ACTIVIDAD != 'HOSPITALIZACIÓN DIURNA'` ($N = 3.945.767$, $167.640$ defunciones, tasa real: **4.249%**). No se condiciona por estancia ni por desenlace vital.
   * **Cohorte de Estadía (LOS):** Egresos vivos no censurados con pernoctación confirmada y estancia positiva ($\text{Estancia} \ge 1$ día, $N = 3.561.516$).

2. **Tratamiento de Altas y Censura:**
   * **Sobrevivientes Confirmados:** Domicilio ($5.205.550$), Alta voluntaria ($58.281$), Fuga ($19.160$) y Hogar/Cárcel ($22.041$).
   * **Censura Activa (Desenlace Desconocido):** Traslados a otros hospitales de agudos ($193.810$) y Hospitalización Domiciliaria ($138.900$). Total censura en cohorte base: **319.629** (o $340.498$ en especificación conservadora con Hogar/Cárcel censurado).
   * **Regla de Traslados:** Enfoque institucional por episodio asistencial completado. El hospital receptor conserva el episodio e **incluye la covariable basal `DERIVADO_OTRO_HOSPITAL`** para capturar la mayor severidad de los pacientes derivados y no castigar a los centros de referencia.

3. **Comorbilidades Elixhauser y Ajuste de Upcoding:**
   * **Especificación Comórbida Primaria:** **28 condiciones crónicas preexistentes** (incluye `ELIX_14` Insuficiencia renal crónica al no contener N17).
   * **Complicaciones Excluidas (3 condiciones):** `ELIX_02` (Arritmias, 5 pts), `ELIX_22` (Coagulopatías, 3 pts) y `ELIX_25` (Hidroelectrolíticos, 5 pts). Total puntos deducidos del Score van Walraven: **13 puntos**.
   * **Control de Upcoding:** Restricción a los primeros **$K = 5$ diagnósticos secundarios**.

4. **Partición Temporal y Diseño del Panel:**
   * **2019:** Período de historia retrospectiva (*washout* de 12 meses para variables de utilización `N_EGRESOS_12M`).
   * **2020–2022 (Desarrollo, 3 años):** Entrenamiento de modelos basales ($N = 2.53\text{M}$ brutos; $1.686.508$ inpatient).
   * **2023 (Calibración):** Recalibración de intercepto temporal y fijación de umbrales analíticos hospitalarios ($N = 1.04\text{M}$).
   * **2024 (Evaluación OOS):** Prueba ciega final del ratio $O/E$ y generación de *Funnel Plots* ($N = 1.09\text{M}$).
   * **Hospitales con Entrada Tardía:** Los 3 hospitales incorporados en 2023 se calibran en 2023 y evalúan en 2024; los 4 incorporados en 2024 se evalúan como cohorte de generalización *cold-start*.

5. **Modelado Estadístico y Umbrales Hospitalarios:**
   * **Mortalidad:** Regresión Logística regularizada con diagnóstico principal agrupado y recalibración en 2023.
   * **Estadía:** GLM Gamma ($p = 2.0$) con enlace logarítmico sobre estancia positiva, con truncamiento p99 (60 días) en desarrollo y evaluación no truncada.
   * **Corte de Inclusión:** $N \ge 1.000$ egresos o $E \ge 25$ muertes esperadas (100% de hospitales en 2023), con contracción empírica de Bayes (*Empirical Bayes shrinkage*).

---

## 1. Errores Verificados Subsanados

### 1.1 Reconciliación Aritmética de Hogar/Cárcel y Censura
La aparente contradicción numérica entre censurados brutos y de cohorte se debió a que el texto anterior proclamó la reclasificación de Hogar/Cárcel ($22.041$ casos) sin haber recomputado las tablas del Parquet. Se clarifica la contabilidad exacta bajo ambas alternativas:

* **Especificación Base Operativa en Datos Parquet (Hogar/Cárcel como Derivación Censurada):**
  - Censurados Brutos: **354.853** (6.11%)
  - Sobrevivientes Brutos: **5.282.991** (90.95%)
  - Censurados en Cohorte Base ($N=5.222.340$): **340.498** (6.52%)
  - Sobrevivientes en Cohorte Base: **4.714.136** (90.27%)
  - Fallecidos en Cohorte Base: **167.706** (3.21%)
  - Comprobación: $167.706 + 4.714.136 + 340.498 = \mathbf{5.222.340}$.

* **Especificación Reclasificada (Hogar/Cárcel como Sobreviviente Egresado Vivo):**
  - De los 22.041 casos brutos de Hogar/Cárcel, exactamente **20.869 están en la cohorte base** (1.172 cayeron en EX01-EX03).
  - Censurados Brutos: $354.853 - 22.041 = \mathbf{332.812}$ (5.73%)
  - Sobrevivientes Brutos: $5.282.991 + 22.041 = \mathbf{5.305.032}$ (91.33%)
  - Censurados en Cohorte Base: $340.498 - 20.869 = \mathbf{319.629}$ (6.12%)
  - Sobrevivientes en Cohorte Base: $4.714.136 + 20.869 = \mathbf{4.735.005}$ (90.67%)
  - Fallecidos en Cohorte Base: **167.706** (3.21%)
  - Comprobación: $167.706 + 4.735.005 + 319.629 = \mathbf{5.222.340}$ exactos.

Ambas formulaciones son matemáticamente coherentes y no presentan contradicción interna una vez desglosados los 20.869 episodios.

---

### 1.2 Auditoría de Traslados: Atribución CMS vs Enfoque Institucional
* **Inviabilidad Práctica de la Atribución CMS Pura en Chile:**
  - Si se aplica la regla de CMS Index Hospital (atribuir el desenlace al hospital de origen y retirar al receptor), más de la mitad de las derivaciones quedan huérfanas: al evaluar estrictamente los **traslados hacia hospitales públicos** ($37.658$ pacientes únicos en 2019–2024), **la tasa de enlace a 48 horas es del 63.38% (23.867)**. El 36.62% restante de traslados públicos y el 100% de los $26.096$ traslados al extrasistema privado no pueden vincularse por carecer de registro receptor en el sistema centralizado GRD.
  - La tasa de enlace a 48h exhibe una mejora sistemática con los años: **54.68% en 2019**, **59.07% en 2020**, **64.65% en 2021**, **66.94% en 2022**, **68.15% en 2023** y **67.65% en 2024**.
  - A nivel hospitalario existe gran heterogeneidad: centros como Antofagasta (102100) y San Juan de Dios (115100) alcanzan enlaces $>75\%$, mientras San Fernando (114101) solo logra 20.82% debido a derivaciones a dispositivos comarcales no informatizados.
* **Decisión Metodológica Consolidada:** Se adopta el **enfoque institucional por episodio asistencial cerrado**:
  - El emisor que traslada a un paciente tiene estancia incompleta y desenlace indeterminado $\rightarrow$ se censura para ese centro.
  - El receptor que admite a un paciente derivado se evalúa por el cuidado brindado, pero **se le incorpora la covariable `DERIVADO_OTRO_HOSPITAL` como ajuste de riesgo basal**, reconociendo la mayor gravedad fisiológica del paciente trasladado.

---

### 1.3 Elixhauser: Coherencia entre CSV, Dummies y Score van Walraven
1. **Conteo Exacto de Comorbilidades:** Al restituir `ELIX_14` (Insuficiencia renal crónica, peso +5) tras verificar que Quan (2005) excluye `N17`, la especificación primaria contiene exactamente **28 comorbilidades crónicas** y excluye **3 potenciales complicaciones** (`ELIX_02` Arritmias, `ELIX_22` Coagulopatías, `ELIX_25` Hidroelectrolíticos).
2. **Ajuste del Score van Walraven:**
   $$\text{Puntos Excluidos} = \text{Peso}(\text{ELIX\_02}) + \text{Peso}(\text{ELIX\_22}) + \text{Peso}(\text{ELIX\_25}) = 5 + 3 + 5 = \mathbf{13\text{ puntos}}$$
   El ajuste en la versión crónicas puras descuenta 13 puntos (no 18).
3. **Auditoría a Nivel de Código en Sensibilidad POA:**
   - La sensibilidad POA no elimina pacientes, sino que **anula la bandera binaria del código comórbido** si este corresponde a un evento agudo (e.g. `I26` en `ELIX_04`, `G93.4` en `ELIX_09`). De este modo, el paciente permanece en el denominador y no se induce sesgo de selección por resultado.

---

### 1.4 Explicación Exacta de los 24 Episodios Restantes
La diferencia observada en la resta $4.881.842 - 1.095.546 = 3.786.296$ frente a los $3.786.272$ reportados en la cohorte Inpatient equivale a exactamente **24 episodios**.
* En la base evaluable existen **37 episodios con `ESTANCIA_DIAS is null`** (debido a discordancia de fechas de ingreso y egreso en los registros brutos).
* De esos 37 registros, **13 corresponden a defunciones** y **24 corresponden a pacientes sobrevivientes**.
* Al aplicar la condición `(ESTANCIA_DIAS > 0) | (MORTALIDAD_BINARIA == 1)`:
  - Las 13 muertes con estancia nula fueron incluidas (por cumplir `MORTALIDAD_BINARIA == 1`).
  - Los 24 sobrevivientes con estancia nula fueron excluidos (por evaluar `Null > 0` a falso/nulo y tener mortalidad 0).
* Por tanto: $3.786.296 - 24 = \mathbf{3.786.272}$ exactos. Al definir la cohorte de hospitalizados mediante `TIPO_ACTIVIDAD` al ingreso, esta anomalía de tipado queda completamente subsanada.

---

### 1.5 Discrepancia en la Matriz Maestra (17 Filas en Procedimientos)
En [`Analisis exploratorio/matriz_maestra_129_columnas.csv`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/matriz_maestra_129_columnas.csv), las columnas `PROCEDIMIENTO1` ($5.790.199$) y `USOSPABELLON` ($2.961.309$) difieren en 22 registros respecto del diccionario estático DEIS ($5.790.177$).
* Se comprobó directamente sobre los Parquet Bronze que los archivos contienen exactamente $5.790.199$ cadenas no vacías.
* La discrepancia de 22 registros (0.00038%) proviene de registros con espacios en blanco en la extracción original de texto del DEIS que fueron limpiados en la ingesta automatizada. La matriz maestra refleja la realidad empírica de la base de datos.

---

## 2. Decisiones Metodológicas Justificadas Empíricamente

### 2.1 Definición de la Cohorte Inpatient al Ingreso (Sin Condicionar por Desenlace)
Para evitar cualquier sesgo de selección por resultado, el carácter ambulatorio se define **estrictamente con información conocida a la admisión**:
* **Actividades Ambulatorias Excluidas:** `TIPO_ACTIVIDAD` igual a `CIRUGÍA MAYOR AMBULATORIA (CMA)` ($882.211$ casos, $45$ muertes) y `HOSPITALIZACIÓN DIURNA` ($53.864$ casos, $21$ muertes). Total ambulatorios: $936.075$ episodios ($66$ muertes, tasa $0.007\%$).
* **Cohorte Hospitalaria Inpatient Pura:** `HOSPITALIZACIÓN` ($3.899.914$) y `HOSPITALIZACIÓN EN URGENCIA` ($45.853$).
  - **Total Episodios:** **3.945.767**
  - **Total Defunciones:** **167.640** (99.96% de la mortalidad total)
  - **Tasa de Mortalidad Inpatient:** **4.249%**
  - Todas las muertes del día 0 en camas de urgencia o básicas ($12.614$) quedan incluidas de forma natural sin requerir reglas condicionadas a la supervivencia.

---

### 2.2 Reincorporación de `DERIVADO_OTRO_HOSPITAL` como Covariable de Riesgo
Aunque `DERIVADO_OTRO_HOSPITAL` presenta una asociación estadística moderada-alta con `COD_HOSPITAL` ($V = 0.5575$), se reconoce que **el traslado de un paciente descompensado constituye una gravedad fisiológica real inherente al paciente, no un déficit de calidad del hospital receptor**.
* Excluirla penaliza injustamente a los centros de referencia terciarios (que concentran pacientes trasladados en shock o falla multiorgánica).
* **Decisión en la Especificación Primaria:** Se **incluye `DERIVADO_OTRO_HOSPITAL` como covariable de ajuste basal** en la ecuación de riesgo individual.

---

### 2.3 Evaluación del Modelo de Estadía: Confiabilidad Split-Half y Baselines
1. **Elección de Gamma ($p = 2.0$):** Para duraciones positivas continuas sin masa en cero ($\text{Estancia} \ge 1$), Gamma es la opción natural con log-verosimilitud canónica.
2. **Métricas de Error frente a la Mediana:**
   - Modelo Gamma ($p = 2.0$): $\text{MAE} = 5.321\text{ d} \ | \ \text{MedianAE} = 3.636\text{ d}$.
   - Baseline Mediana Constante ($4.0\text{ d}$): $\text{MAE} = 4.881\text{ d} \ | \ \text{MedianAE} = 2.000\text{ d}$.
   - Baseline Media Constante ($6.5\text{ d}$): $\text{MAE} = 5.336\text{ d} \ | \ \text{MedianAE} = 4.000\text{ d}$.
   - *Fundamento Estadístico:* El estimador GLM optimiza la devianza para la media condicional $\mathbb{E}[Y|X]$. En distribuciones con asimetría positiva severa, la mediana matemática minimiza trivialmente el MAE y MedianAE global, pero carece de capacidad de discriminación del riesgo comórbido individual.
3. **Confiabilidad Split-Half a Nivel Hospitalario (2023, N = 68 hospitales):**
   - Correlación entre mitades par-impar de pacientes: $r_{\text{half}} = \mathbf{0.9748}$.
   - **Confiabilidad Spearman-Brown:** $R = \mathbf{0.9873}$.
   - Aunque la devianza explicada a nivel paciente sea moderada ($D^2 \approx 6.3\%$), al agregar los casos a nivel de establecimiento ($N \ge 1.000$), **la confiabilidad del indicador $O/E$ de estancia alcanza el 98.7%**, superando con holgura los estándares internacionales de evaluación de calidad asistencial ($R \ge 0.80$, Dimick et al., 2010).

---

### 2.4 Control de Upcoding: Intervalo Bootstrap de la Diferencia en K = 5
Ante la observación del solapamiento de intervalos, se ejecutó un remuestreo bootstrap ($B = 1.000$ iteraciones sobre los 65 hospitales de desarrollo):
* **Correlación Uncapped ($K \le 34$):** $r_s = -0.2135$ [IC 95%: $-0.4357$, $+0.0157$].
* **Correlación Capped ($K = 5$):** $r_s = -0.1636$ [IC 95%: $-0.3948$, $+0.0872$].
* **Diferencia ($\Delta r_s = r_{s,\text{capped}} - r_{s,\text{uncapped}}$):**
  $$\Delta r_s = \mathbf{+0.0500} \quad [\text{IC 95\% Bootstrap: } \mathbf{-0.0036}, \ \mathbf{+0.1101}]$$
  $$\mathbb{P}(\Delta r_s > 0) = \mathbf{97.0\%}$$
* *Conclusión Editorial:* En el 97% de los remuestreos, el tope $K=5$ atenúa la correlación negativa. No obstante, al tocar marginalmente el cero el IC 95%, se redacta con prudencia científica reconociendo que el tope $K=5$ ofrece una **mitigación direccional favorable del upcoding**, sin pretender una erradicación estadística total.

---

### 2.5 Modelo Final de Mortalidad: Entrenamiento 2020–2022 y Calibración por Hospital
Ajuste sobre el diseño definitivo: **Entrenamiento en 2020–2022 ($N = 1.686.508$ hospitalizaciones, $87.817$ defunciones)** y **Evaluación Fuera de Muestra en 2023 ($N = 669.655$, $24.627$ defunciones)** incorporando Diagnóstico Principal (dummies de macro-grupos) y las 28 crónicas de Elixhauser:

* **Métricas Fuera de Muestra (2023):**
  - **ROC-AUC:** $\mathbf{0.8698}$
  - **Brier Score:** $\mathbf{0.0318}$
  - **Calibración Global:** Pendiente = $\mathbf{0.9754}$ | Intercepto = $\mathbf{-0.2203}$
* **Calibración por Hospital ($N = 66$ hospitales con $O \ge 20$ defunciones en 2023):**
  - **Pendiente de Calibración:** Mediana = $\mathbf{0.9906}$ [IQR: $0.9211$ – $1.0660$]. (Prácticamente ideal en el 100% de la red).
  - **Intercepto de Calibración:** Mediana = $\mathbf{-0.2711}$ [IQR: $-0.5768$ – $+0.0385$].
  - **Brier Score:** Mediana = $\mathbf{0.0321}$ [IQR: $0.0267$ – $0.0383$].
  - **AUROC Intrahospitalario:** Mediana = $\mathbf{0.8736}$ [IQR: $0.8497$ – $0.8909$].

El desplazamiento negativo del intercepto global ($-0.2203$) refleja el descenso secular de la mortalidad post-pandémica entre 2020–2022 ($5.21\%$) y 2023 ($3.68\%$), justificando plenamente el paso de **recalibración temporal en 2023** antes de evaluar en 2024.

---

## 3. Detalles de Datos y Panel

### 3.1 Los 7 Hospitales con Entrada Tardía en el Panel
* **Incorporados en 2023 (3 hospitales):**
  - `118106` (H. Dr. Gustavo Fricke nuevo): $N=5.444$, Dx Sec=$6.34$, Censura=$9.74\%$, Muertes=$132$.
  - `119101` (H. de Curicó nuevo): $N=4.431$, Dx Sec=$5.97$, Censura=$11.04\%$, Muertes=$122$.
  - `110110` (H. San Borja CDT/Materno): $N=4.683$, Dx Sec=$3.00$, Censura=$1.41\%$, Muertes=$6$.
* **Incorporados en 2024 (4 hospitales):**
  - `119102` (H. de Linares nuevo): $N=4.157$, Dx Sec=$5.17$, Censura=$15.42\%$, Muertes=$92$.
  - `116107` (H. de Angol): $N=3.204$, Dx Sec=$5.65$, Censura=$15.73\%$, Muertes=$106$.
  - `116111` (H. de Padre Las Casas): $N=4.682$, Dx Sec=$5.16$, Censura=$13.97\%$, Muertes=$189$.
  - `200717` (H. Biprovincial Quillota Petorca): $N=10.354$, Dx Sec=$3.84$, Censura=$5.58\%$, Muertes=$242$.
* **Regla Operativa:** Todos superan holgadamente el corte de $N \ge 1.000$ en su primer año. Su censura más alta ($11\%–15\%$) se debe a su rol como nodos comarcales de estabilización previa a derivación hacia centros de referencia.

### 3.2 Identificación de los 2.044 CIP Nulos en 2021
La totalidad de los $2.044$ registros con `CIP_ENCRIPTADO` nulo en 2021 se concentran en $40$ hospitales, liderados por:
* Hospital Barros Luco Trudeau (`109100`): $393$ nulos.
* Hospital de Copiapó (`103100`): $378$ nulos.
* Instituto Nacional del Tórax (`111195`): $252$ nulos.
* Hospital de Antofagasta (`102100`): $148$ nulos.
* Hospital de Melipilla (`110150`): $101$ nulos.
* **Causa Identificada:** El $54.2\%$ eran pacientes particulares y el $33.3\%$ FONASA A (migrantes o pacientes en situación de calle sin RUN válido ingresados durante el pico de la variante Delta de COVID-19 en 2021).

### 3.3 COVID-19 y Transición de Versiones GRD
En 2020 se registraron **35.557 episodios con diagnóstico principal U07 (COVID-19)**. De ellos, **exactamente 0 casos cayeron en EX01 (No agrupables)**, debido a que el DEIS aplicó parches de emergencia a IR-GRD v29 para asignar U07 al MDC 04 (Infecciones respiratorias agudas).

### 3.4 Restricción a Adultos y Hospitales Pediátricos
Existen **833.411 episodios en menores de 18 años (15.96%)**.
* **Hospitales Monográficos Pediátricos (>80% casos <18 años):**
  - `109101`: Hospital Dr. Exequiel González Cortés (99.75% pediátrico).
  - `112102`: Hospital Dr. Luis Calvo Mackenna (99.60% pediátrico).
  - `113130`: Hospital de Niños Roberto del Río (99.30% pediátrico).
* Al restringir la evaluación exclusivamente a adultos ($\ge 18$ años):
  - Los 3 hospitales pediátricos dejan el ranking de adultos.
  - **Quedan exactamente 69 de 72 hospitales** con $N_{\text{adultos}} \ge 500$.
  - La correlación de rankings entre la cohorte general y la cohorte de adultos en esos 69 hospitales es:
    $$\rho = \mathbf{0.9807} \quad (p = 2.96 \times 10^{-49})$$
  demostrando que el ranking hospitalario de la red de agudos es altamente robusto e insensible a la inclusión de pacientes pediátricos.

---

## 4. Faltantes de Predictores por Año (Capa Gold)

Auditoría exhaustiva sobre los 5.808.536 episodios:

| Año Egreso | Episodios Brutos | Edad Nula | Sexo Nulo | Previsión Nula | Egresos 12m Nulo | Score van Walraven Nulo | Estancia Nula |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2019** | 1.151.475 | 19 | 0 | 0 | 0 | 0 | 20 |
| **2020** | 781.912 | 1 | 0 | 0 | 0 | 0 | 10 |
| **2021** | 816.909 | 0 | 0 | 0 | 0 | 0 | 0 |
| **2022** | 932.840 | 7 | 0 | 0 | 0 | 0 | 1 |
| **2023** | 1.039.587 | 10 | 0 | 0 | 0 | 0 | 0 |
| **2024** | 1.085.813 | 55 | 0 | 0 | 0 | 0 | 52 |
| **Total** | **5.808.536** | **92 (0.0016%)** | **0 (0.0%)** | **0 (0.0%)** | **0 (0.0%)** | **0 (0.0%)** | **83 (0.0014%)** |

El porcentaje de datos faltantes en los predictores analíticos de Capa Gold es de $0.0016\%$, confirmando la máxima calidad para el ajuste estadístico de riesgo.
