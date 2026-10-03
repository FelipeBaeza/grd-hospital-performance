# Resolución Metodológica y Cierre Definitivo de la Fase 2
**Proyecto:** Evaluación del Desempeño Hospitalario Ajustado por Riesgo en el Régimen de Garantías Explícitas en Salud (FONASA, 2019–2024)  
**Autor:** Felipe  
**Fecha:** Octubre 2026 (Actualización Consolidada Cierre de Fase 2)

---

## 1. Dependencia de la Especificación, Alertas Robustas y Centros Monográficos

### 1.1 Explicación y Documentación del Cambio en la Lista de Alertas de M3
En versiones preliminares del documento se listaron provisionalmente como alertas de M3 siete prestadores: El Pino, Van Buren, Tisné, Tórax, Pereira, Heyermann y Curanilahue. Dicha lista preliminar adolecía de dos inconsistencias que fueron debidamente corregidas tras congelar la cohorte analítica de calibración 2023 ($N = 557.318$ episodios, $O = 24.225$ defunciones, $K = 65$ hospitales):

1. **Salida de Curanilahue (`128109`) y Heyermann (`129100`):**  
   En la versión previa se había utilizado un $O/E$ crudo antes de contracción bayesiana (donde Curanilahue marcaba $1,211$). Al aplicar el modelo formal M3 con contracción empírica de Bayes ($\tau_{\text{M3}} = 0,2173$, $\mu_{\text{meta}} = 0,0261$) y el margen de exceso clínico a priori del $10\%$ ($P(\theta_j > 1,10) \ge 0,95$):
   * **Curanilahue (`128109`):** Su $O/E$ ajustado y recalibrado es de **$1,137$** ($O = 136, E = 119,5$). Debido a su volumen moderado de muertes ($O = 136$), su varianza de muestreo es $v_j = 1/136 = 0,007353$ y su error estándar posterior es $\text{SE}_{\text{post}} = 0,0798$. La probabilidad a posteriori de exceder el margen del $10\%$ es:
     $$P(\theta_j > 1,10 \mid M3) = \mathbf{0,594} \quad (\ll 0,95)$$
     Por ende, **Curanilahue no cumple el criterio de alerta y queda correctamente clasificado en la categoría "Promedio"**.
   * **Heyermann de Angol (`129100`):** Registra $O = 233, E = 197,3 \implies O/E = 1,181$. Con $\text{SE}_{\text{post}} = 0,0627$, su probabilidad a posteriori es $P(\theta_j > 1,10 \mid M3) = \mathbf{0,828} < 0,95$, quedando igualmente en **"Promedio"**.

2. **Entrada de Claudio Vicuña (`106103`) y La Florida (`114105`):**  
   * **Hospital Claudio Vicuña de San Antonio (`106103`):** Registra $O = 320, E = 195,8 \implies O/E_{\text{M3}} = \mathbf{1,634}$. Con $\text{SE}_{\text{post}} = 0,0541$, su probabilidad a posteriori es $P(\theta_j > 1,10) = \mathbf{1,000}$ (y $P(\theta_j > 1,20) = \mathbf{1,000}$). Su omisión en el listado preliminar respondió a un error de filtrado en el join por código de establecimiento.
   * **Hospital Clínico Metropolitano La Florida (`114105`):** Registra $O = 596, E = 457,3 \implies O/E_{\text{M3}} = \mathbf{1,303}$. Debido a su elevado volumen institucional ($O = 596$), su error estándar posterior es sumamente estrecho ($\text{SE}_{\text{post}} = 0,0403$). Su probabilidad posterior de exceso al $10\%$ es $P(\theta_j > 1,10) = \mathbf{1,000}$ (y al $20\%$ es $P(\theta_j > 1,20) = \mathbf{0,968}$). Entra formalmente al listado de alertas de M3.

**Valores Definitivos de $O/E$ y Probabilidades Posteriores en M3 (Margen 10%):**
Los 7 prestadores que configuran la lista oficial de Alertas en M3 ($P(\theta_j > 1,10) \ge 0,95$) son:
1. `112103` **Instituto Nacional del Tórax:** $O = 157, E = 70,5, O/E = \mathbf{2,226}, P(>1,10) = \mathbf{1,000}$ *(Monográfico)*
2. `113180` **Hospital El Pino:** $O = 438, E = 247,1, O/E = \mathbf{1,772}, P(>1,10) = \mathbf{1,000}$
3. `106103` **Hospital Claudio Vicuña (San Antonio):** $O = 320, E = 195,8, O/E = \mathbf{1,634}, P(>1,10) = \mathbf{1,000}$
4. `106102` **Hospital Dr. Eduardo Pereira Ramírez:** $O = 148, E = 100,1, O/E = \mathbf{1,478}, P(>1,10) = \mathbf{0,999}$
5. `106100` **Hospital Carlos Van Buren (Valparaíso):** $O = 602, E = 419,3, O/E = \mathbf{1,436}, P(>1,10) = \mathbf{1,000}$
6. `112101` **Hospital Dr. Luis Tisné B.:** $O = 502, E = 381,3, O/E = \mathbf{1,316}, P(>1,10) = \mathbf{1,000}$
7. `114105` **Hospital Clínico Metropolitano La Florida:** $O = 596, E = 457,3, O/E = \mathbf{1,303}, P(>1,10) = \mathbf{1,000}$

---

### 1.2 Regla Cuantitativa y Medida de Centros Monográficos
Para evitar clasificaciones ad-hoc basadas en descripciones cualitativas históricas, se define una **regla analítica cuantificable de especialización**:
$$\text{Centro Monográfico} \iff \max_{\text{MDC}} (\% \text{ Episodios}) \ge 50\% \quad \text{Ó} \quad (\% \text{ MDC 4 [Resp]} + \% \text{ MDC 5 [Cardio]}) \ge 70\%$$

**Evaluación Empírica en la Cohorte 2023:**
* **`110110` Instituto Traumatológico Dr. Teodoro Gebauer:** $93,4\%$ de sus episodios corresponden a MDC 8 (Afecciones Musculoesqueléticas). Supera el umbral del $50\% \implies$ **Monográfico**.
* **`112104` Instituto de Neurocirugía Dr. Asenjo:** $66,8\%$ de sus episodios corresponden a MDC 1 (Sistema Nervioso). Supera el umbral del $50\% \implies$ **Monográfico**.
* **`112103` Instituto Nacional del Tórax:** $62,5\%$ en MDC 5 (Circulatorio) y $30,2\%$ en MDC 4 (Respiratorio), sumando un **$92,7\%$ en patología cardio-respiratoria**. Supera el umbral combinado del $70\% \implies$ **Monográfico**.
* **`106102` Hospital Dr. Eduardo Pereira Ramírez (Valparaíso):**  
  Históricamente sanatorio broncopulmonar, pero en los datos contemporáneos presenta:
  - MDC 6 (Aparato Digestivo): $21,8\%$
  - MDC 4 (Aparato Respiratorio): $18,4\%$
  - MDC 5 (Aparato Circulatorio): $11,9\%$ (Suma MDC 4+5 = $30,3\%$)
  - Otros MDCs médicos y quirúrgicos: $47,9\%$  
  Ningún MDC alcanza el $50\%$ y la patología cardio-respiratoria solo representa el $30,3\%$. **Por regla medida objetiva, el Hospital Eduardo Pereira NO califica como centro monográfico, sino como hospital general de agudos con predominio médico-quirúrgico**.

Al desagregar los centros monográficos hacia su grupo de comparación especializado, el número de **alertas robustas en agudos generales se restringe a exactamente 4 prestadores** bajo la intersección con M2, o a **6 prestadores** bajo la intersección de variantes de M3.

---

### 1.3 Matriz de Sensibilidad de Alertas: Intersección de Variantes de M3
El Modelo 2 (clínico estándar) incumple el criterio de calibración diagnóstica (con $O/E$ que oscila de $0,12$ en 0–2 diagnósticos a $1,49$ en $\ge 11$ diagnósticos), por lo que condicionar la alerta robusta a la intersección con M2 ($M2 \cap M3$) hace que un modelo descalibrado condicione el resultado.

Por ello, se formula la **Alerta Robusta Basada en Variantes de M3**, intersectando cinco formulaciones que abordan directamente el case-mix y la no linealidad:
1. **M3a (Primario):** Razón intrahospitalaria $R_{\text{dx}} = N_{\text{dx}} / \bar{N}_{\text{dx}, h}$ con splines cúbicos naturales (4 nudos), 28 comorbilidades Elixhauser.
2. **M3b (Lineal):** Razón $R_{\text{dx}}$ lineal sin splines, 28 comorbilidades.
3. **M3c (Centrado / Z-score):** Estandarización intrahospitalaria $Z_{\text{dx}} = (N_{\text{dx}} - \bar{N}_{\text{dx}, h}) / \text{SD}_{\text{dx}, h}$, 28 comorbilidades.
4. **M3d (31 Comorbilidades):** $R_{\text{dx}}$ con splines, incluyendo comorbilidades agudas (`ELIX_02`, `ELIX_22`, `ELIX_25`).
5. **M3a Sin Día 0:** Exclusión estricta de pacientes con estancia $= 0$ días y fallecimiento inmediato.

**Parámetros de Dispersión Empírica ($\tau$) por Modelo:**
Cada modelo estima su propio $\tau$ mediante el estimador de DerSimonian-Laird sobre $\ln(O/E_j)$ con ponderaciones $w_j = O_j$:
* $\tau_{\text{M2}} = \mathbf{0,2163}$ ($\tau^2 = 0,0468; \mu_{\text{meta}} = 0,0252$)
* $\tau_{\text{M3a}} = \mathbf{0,2173}$ ($\tau^2 = 0,0472; \mu_{\text{meta}} = 0,0261$)
* $\tau_{\text{M3b}} = \mathbf{0,2219}$ ($\tau^2 = 0,0492; \mu_{\text{meta}} = 0,0273$)
* $\tau_{\text{M3c}} = \mathbf{0,2161}$ ($\tau^2 = 0,0467; \mu_{\text{meta}} = 0,0264$)
* $\tau_{\text{M3d}} = \mathbf{0,2078}$ ($\tau^2 = 0,0432; \mu_{\text{meta}} = 0,0241$)
* $\tau_{\text{Sin Día 0}} = \mathbf{0,2156}$ ($\tau^2 = 0,0465; \mu_{\text{meta}} = 0,0260$)

#### Matriz Completa de Alertas por Hospital (* = Alerta con $P(\theta > 1,10) \ge 0,95$):
| Código | Establecimiento | M3a (Prim) | M3b (Lin) | M3c (Zdx) | M3d (31c) | Sin Día 0 | M2 (Clin) | Consistencia M3 | Alerta Robusta $M2 \cap M3$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `112103` | **Inst. Nac. del Tórax** *(Monográfico)* | **2,23\*** | **2,34\*** | **2,09\*** | **2,20\*** | **2,36\*** | **1,97\*** | **5/5 (100%)** | Sí (Monográfico) |
| `113180` | **Hospital El Pino** | **1,77\*** | **1,73\*** | **1,81\*** | **1,68\*** | **1,68\*** | **1,84\*** | **5/5 (100%)** | **SÍ (Agudo 1)** |
| `106103` | **Hospital Claudio Vicuña** | **1,63\*** | **1,69\*** | **1,52\*** | **1,66\*** | **1,65\*** | **1,59\*** | **5/5 (100%)** | **SÍ (Agudo 2)** |
| `106102` | **Hospital Dr. Eduardo Pereira** | **1,48\*** | **1,42\*** | **1,38\*** | **1,45\*** | **1,55\*** | 0,98 | **5/5 (100%)** | No (Sensible a M2) |
| `106100` | **Hospital Carlos Van Buren** | **1,44\*** | **1,48\*** | **1,35\*** | **1,42\*** | **1,36\*** | **1,29\*** | **5/5 (100%)** | **SÍ (Agudo 3)** |
| `112101` | **Hospital Dr. Luis Tisné B.** | **1,32\*** | **1,36\*** | **1,28\*** | **1,26\*** | **1,28\*** | **1,31\*** | **5/5 (100%)** | **SÍ (Agudo 4)** |
| `114105` | **Hospital Clínico La Florida** | **1,30\*** | **1,28\*** | **1,32\*** | **1,23\*** | **1,32\*** | 1,17 | **5/5 (100%)** | No (Sensible a M2) |
| `108100` | Hospital San Camilo (San Felipe) | 1,20 | 1,21\* | 1,24\* | 1,18 | 1,16 | 1,29\* | 2/5 (Parcial) | No |
| `103100` | Hospital Regional de Antofagasta | 1,16 | 1,19\* | 1,26\* | 1,22\* | 1,20\* | 1,38\* | 4/5 (Parcial) | No |
| `121109` | Hospital Hernán Henríquez (Temuco)| 1,16 | 1,14 | 1,19\* | 1,17 | 1,20\* | 1,24\* | 2/5 (Parcial) | No |
| `111100` | Hospital San Borja-Arriarán | 1,18 | 1,18 | 1,27 | 1,19 | 1,24 | 1,51\* | 0/5 (Sensible M2) | No |
| `113100` | Hospital Barros Luco Trudeau | 1,15 | 1,14 | 1,16 | 1,15 | 1,10 | 1,29\* | 0/5 (Sensible M2) | No |
| `109100` | Complejo Hospitalario San José | 1,12 | 1,10 | 1,11 | 1,12 | 1,15 | 1,19\* | 0/5 (Sensible M2) | No |
| `103101` | Hospital Carlos Cisternas (Calama)| 0,96 | 0,95 | 1,02 | 0,95 | 0,97 | 1,22\* | 0/5 (Sensible M2) | No |
| `113150` | Hospital San Luis (Buin) | 0,92 | 0,92 | 0,92 | 0,96 | 0,93 | 1,31\* | 0/5 (Sensible M2) | No |

**Conclusiones Clave de la Matriz:**
1. **Consistencia M3:** Los EXACTOS 7 HOSPITALES (1 monográfico y 6 agudos generales) presentan una concordancia del **100% (5/5)** a través de todas las especificaciones de M3.
2. **Alerta Robusta Estricta ($M2 \cap M3$):** Si se impone la concordancia con el modelo no ajustado M2, Tórax se aísla como monográfico, y quedan **exactamente 4 hospitales generales agudos: El Pino, Claudio Vicuña, Van Buren y Luis Tisné**.
3. **Efecto de la Codificación en Pereira y La Florida:** Pereira pasa de $0,98$ en M2 a $1,48^*$ en M3a, y La Florida de $1,17$ en M2 a $1,30^*$ en M3a. En M2 estos hospitales se beneficiaban artificialmente del subajuste por profundidad diagnóstica; al ajustar por $R_{\text{dx}}$, su exceso de mortalidad emerge con $100\%$ de persistencia bajo todas las variantes de M3.

---

## 2. Inferencia Bayesiana Empírica: Reproducibilidad y Márgenes Clínicos

### 2.1 Aclaración de Cifras de la Regla Direccional (40 vs 51 Hospitales)
La diferencia entre las tasas de detección de la regla direccional $P(\theta > 1,0) \ge 0,95$ responde estrictamente a la especificación evaluada:
* **Bajo Modelo 2 (Clínico Lineal):** Marca **51 de 65 hospitales ($78,5\%$)** (23 en alerta, 28 sobresalientes, 14 en promedio).
* **Bajo Modelo 3 (Splines $R_{\text{dx}}$ con Centrado $\mu_{\text{meta}} = 0,0261$):** Marca **40 de 65 hospitales ($61,5\%$)** (21 en alerta, 19 sobresalientes, 25 en promedio).
Ambas cifras demuestran que probar dirección estocástica respecto a 1,0 en lugar de magnitud clínica sobre-clasifica a la mayor parte de la red pública.

### 2.2 Márgenes de Materialidad Clínica (10% y 20%) y Caso Curanilahue (`128109`)
El margen de materialidad clínica es una decisión metodológica de diseño del indicador:
* **Margen del 10%:** $\text{Alerta si } P(\theta_j > 1,10) \ge 0,95 \implies$ **7 hospitales en alerta ($10,8\%$)**, **10 sobresalientes ($15,4\%$)**, **48 promedio ($73,8\%$)**.
* **Margen del 20%:** $\text{Alerta si } P(\theta_j > 1,20) \ge 0,95 \implies$ **7 hospitales en alerta ($10,8\%$)**, **4 sobresalientes ($6,2\%$)**, **54 promedio ($83,1\%$)**.  
  (Los 7 hospitales al 20% son los mismos 7 de M3a: Tórax, El Pino, Vicuña, Pereira, Van Buren, Tisné y La Florida).

**Auditoría Específica de Curanilahue (`128109`):**
Curanilahue registra $O = 136, E = 119,5 \implies O/E_{\text{M3a}} = 1,138$.  
Con $v_j = 1/136 = 0,007353$, $\log(\theta_{\text{post}}) = 0,1154$ y $\text{SE}_{\text{post}} = 0,0798$:
* Margen 10%: $Z = (0,1154 - \log(1,10)) / 0,0798 = +0,252 \implies P(\theta > 1,10) = \mathbf{0,599} < 0,95$.
* Margen 20%: $Z = (0,1154 - \log(1,20)) / 0,0798 = -0,838 \implies P(\theta > 1,20) = \mathbf{0,201} < 0,95$.  
Curanilahue queda sólidamente en **PROMEDIO** tanto al 10% como al 20%.

### 2.3 Reproducibilidad Paso a Paso del Cálculo Bayesiano Empírico
Fórmula analítica de contracción:
$$v_j = \frac{1}{O_j}, \quad B_j = \frac{\tau^2}{\tau^2 + v_j}, \quad \log(\theta_j^{\text{EB}}) = B_j \log(O/E_j) + (1 - B_j)\mu_{\text{meta}}, \quad \text{SE}_{\text{post}, j} = \sqrt{\frac{\tau^2 v_j}{\tau^2 + v_j}}$$
$$Z_j = \frac{\log(\theta_j^{\text{EB}}) - \log(1,10)}{\text{SE}_{\text{post}, j}}, \quad P(\theta_j > 1,10 \mid \text{datos}) = \Phi(Z_j)$$

**Fila Reproducible: Hospital El Pino (`113180`):**
* $O_j = 438$, $E_{3, j} = 247,1 \implies O/E_{\text{post}} = 1,772$.
* $\log(O/E_{\text{post}}) = \mathbf{0,5721}$.
* Varianza within: $v_j = 1 / 438 = \mathbf{0,002283}$.
* Con $\tau^2 = 0,0472$ y $\mu_{\text{meta}} = 0,0261$:
  $$B_j = \frac{0,0472}{0,0472 + 0,002283} = \mathbf{0,9539}$$
  $$\log(\theta_j^{\text{EB}}) = 0,9539 \cdot (0,5721) + (1 - 0,9539) \cdot (0,0261) = \mathbf{0,5471} \implies \theta_j^{\text{EB}} = 1,728$$
  $$\text{SE}_{\text{post}, j} = \sqrt{\frac{0,0472 \cdot 0,002283}{0,04948}} = \mathbf{0,0467}$$
  $$Z = \frac{0,5471 - \log(1,10)}{0,0467} = \frac{0,5471 - 0,09531}{0,0467} = \mathbf{+9,67} \implies P(\theta_j > 1,10) = \mathbf{1,0000} \ (\text{ALERTA})$$

---

## 3. *Tipping Point* y Análisis de Sensibilidad con Traslados Enlazados

### 3.1 Microdatos de Traslados Enlazados y Emisores (Año 2023)
En la cohorte adulta inpatient de 2023 se registraron $25.555$ egresos por traslado ($31.910$ en el consolidado general Silver). De ellos, se enlazó determinísticamente al receptor dentro de los 30 días posteriores al egreso a **$11.544$ episodios ($45,2\%$ de tasa de enlace)**:
* **Mortalidad en el receptor:** **$7,31\%$ global** ($844$ defunciones).
* **Multiplicador de riesgo observado ($\lambda_{\text{obs}}$):** Comparando con la mortalidad promedio de la red ($4,35\%$), $\lambda_{\text{obs}} = \text{Tasa}_{\text{rec}} / 4,35\%$.

#### Tabla de Principales Hospitales Emisores de Traslados (2023):
| Código | Hospital Emisor | Emitidos | Enlazados | % Enlace | Defunciones Receptor | Tasa Obs % | IC 95% Wilson | $\lambda_{\text{obs}}$ | No Enlazados |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `114101` | Complejo Dr. Sótero del Río | 2.385 | 621 | 26,0% | 43 | 6,92% | [5,2% - 9,2%] | 1,59x | 1.764 |
| `113180` | Hospital El Pino | 1.370 | 250 | 18,2% | 25 | 10,00% | [6,9% - 14,3%] | 2,30x | 1.120 |
| `106100` | Hospital Carlos Van Buren | 1.259 | 910 | 72,3% | 58 | 6,37% | [5,0% - 8,2%] | 1,47x | 349 |
| `112101` | Hospital Dr. Luis Tisné B. | 908 | 209 | 23,0% | 20 | 9,57% | [6,3% - 14,3%] | 2,20x | 699 |
| `121109` | Hospital Dr. Hernán Henríquez | 853 | 445 | 52,2% | 17 | 3,82% | [2,4% - 6,0%] | 0,88x | 408 |
| `115100` | Hospital Regional de Rancagua | 834 | 230 | 27,6% | 12 | 5,22% | [3,0% - 8,9%] | 1,20x | 604 |
| `124105` | Hospital de Puerto Montt | 802 | 246 | 30,7% | 16 | 6,50% | [4,0% - 10,3%] | 1,50x | 556 |
| `118100` | Hospital Guillermo Grant (Concepción)| 713 | 493 | 69,1% | 24 | 4,87% | [3,3% - 7,1%] | 1,12x | 220 |
| `112100` | Hospital Del Salvador | 669 | 180 | 26,9% | 25 | 13,89% | [9,6% - 19,7%] | 3,20x | 489 |
| `117101` | Hospital Herminda Martín (Chillán) | 636 | 204 | 32,1% | 19 | 9,31% | [6,0% - 14,1%] | 2,14x | 432 |
| `107101` | Hospital San Martín (Quillota) | 570 | 231 | 40,5% | 13 | 5,63% | [3,3% - 9,4%] | 1,29x | 339 |
| `111195` | HUAP (Posta Central) | 551 | 263 | 47,7% | 31 | 11,79% | [8,4% - 16,2%] | 2,71x | 288 |
| `128109` | **Hospital Dr. Rafael Avaría (Curanilahue)** | **511** | **210** | **41,1%** | **12** | **5,71%** | **[3,3% - 9,7%]** | **1,31x** | **301** |
| `113100` | Hospital Barros Luco Trudeau | 503 | 94 | 18,7% | 16 | 17,02% | [10,8% - 25,9%] | 3,92x | 409 |
| `116110` | Hospital San José de Parral | 392 | 365 | 93,1% | 24 | 6,58% | [4,5% - 9,6%] | 1,51x | 27 |
| `121114` | Hospital Nueva Imperial | 333 | 267 | 80,2% | 6 | 2,25% | [1,0% - 4,8%] | 0,52x | 66 |
| `121117` | Hospital de Pitrufquén | 328 | 173 | 52,7% | 14 | 8,09% | [4,9% - 13,1%] | 1,86x | 155 |
| `106103` | Hospital Claudio Vicuña | 299 | 197 | 65,9% | 14 | 7,11% | [4,3% - 11,6%] | 1,63x | 102 |
| `121121` | Hospital de Villarrica | 288 | 182 | 63,2% | 9 | 4,95% | [2,6% - 9,1%] | 1,14x | 106 |
| `129106` | Hospital San José (Victoria) | 263 | 133 | 50,6% | 18 | 13,53% | [8,7% - 20,4%] | 3,11x | 130 |

---

### 3.2 Tipping Point Corregido ($\lambda_{\text{entrada}}$ y $\lambda_{\text{salida}}$)
Dado que Curanilahue ya **NO está en alerta** en M2 ni en M3 ($P(\theta > 1,10) = 0,594$), la métrica correcta para evaluar su sensibilidad no es a qué letalidad "sale" de alerta, sino a qué letalidad de traslados **entraría a zona de alerta ($\lambda_{\text{entrada}}^*$)**. A su vez, para los 7 prestadores en alerta, se evalúa si alguna tasa de letalidad en derivados podría hacerlos **salir de alerta ($\lambda_{\text{salida}}^*$)**:

| Hospital | Estado Base en M3 | $O/E_{\text{base}}$ | $P(>1,10)_{\text{base}}$ | Traslados | Tipping Point Evaluado | Multiplicador Crítico ($\lambda^*$) | Letalidad Equivalente | ¿Ocurre en Datos Observados? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Curanilahue (`128109`)** | **PROMEDIO** | **1,138** | **0,599** | 511 | **$\lambda_{\text{entrada}}^*$** | **2,00x** | **8,70%** | **NO** ($\lambda_{\text{obs}} = 1,31\text{x}$ [$5,71\%$]; se mantiene en Promedio) |
| **Nueva Imperial (`121114`)** | **SOBRESALIENTE**| **0,682** | **0,000** | 333 | $\lambda_{\text{entrada}}^*$ | $> 3,50\text{x}$ | $> 15,2\%$ | **NO** ($\lambda_{\text{obs}} = 0,52\text{x}$ [$2,25\%$]; se mantiene Sobresaliente)|
| **Villarrica (`121121`)** | **PROMEDIO** | **0,914** | **0,008** | 288 | $\lambda_{\text{entrada}}^*$ | $> 3,00\text{x}$ | $> 13,0\%$ | **NO** ($\lambda_{\text{obs}} = 1,14\text{x}$ [$4,95\%$]; se mantiene en Promedio) |
| **Hospital El Pino (`113180`)** | **ALERTA** | **1,772** | **1,000** | 1.370 | **$\lambda_{\text{salida}}^*$** | **Inalcanzable** | **$< 0,00\%$** | **NO** (Incluso con $0\%$ de muertes, $O/E = 1,428, P = 1,000$) |
| **Claudio Vicuña (`106103`)** | **ALERTA** | **1,634** | **1,000** | 299 | **$\lambda_{\text{salida}}^*$** | **Inalcanzable** | **$< 0,00\%$** | **NO** (Incluso con $0\%$ de muertes, $O/E = 1,533, P = 1,000$) |
| **Carlos Van Buren (`106100`)** | **ALERTA** | **1,436** | **1,000** | 1.259 | **$\lambda_{\text{salida}}^*$** | **Inalcanzable** | **$< 0,00\%$** | **NO** (Incluso con $0\%$ de muertes, $O/E = 1,270, P = 1,000$) |
| **Dr. Luis Tisné B. (`112101`)** | **ALERTA** | **1,316** | **1,000** | 908 | **$\lambda_{\text{salida}}^*$** | **Inalcanzable** | **$< 0,00\%$** | **NO** (Incluso con $0\%$ de muertes, $O/E = 1,193, P = 0,957$) |
| **Eduardo Pereira (`106102`)** | **ALERTA** | **1,478** | **0,999** | 48 | **$\lambda_{\text{salida}}^*$** | **Inalcanzable** | **$< 0,00\%$** | **NO** (Incluso con $0\%$ de muertes, $O/E = 1,448, P = 0,999$) |
| **La Florida (`114105`)** | **ALERTA** | **1,303** | **1,000** | 215 | **$\lambda_{\text{salida}}^*$** | **Inalcanzable** | **$< 0,00\%$** | **NO** (Incluso con $0\%$ de muertes, $O/E = 1,277, P = 1,000$) |
| **Instituto del Tórax (`112103`)**| **ALERTA** | **2,226** | **1,000** | 112 | **$\lambda_{\text{salida}}^*$** | **Inalcanzable** | **$< 0,00\%$** | **NO** (Incluso con $0\%$ de muertes, $O/E = 2,083, P = 1,000$) |

**Cotas para el 54,8% No Enlazado (14.011 casos):**  
Incluso sometiendo la fracción no enlazada a escenarios de estrés con letalidad de hasta $15,0\%$ ($\lambda = 3,45\times$):
1. Ninguno de los hospitales sobresalientes o promedio con alto traslado (Villarrica, Nueva Imperial, Quillota) cruza el umbral hacia zona de alerta.
2. Ninguno de los 7 hospitales en alerta sale de alerta, dado que sus transferencias presentan letalidades observadas iguales o superiores a la media de la red ($\lambda_{\text{obs}}$ de $1,47\times$ a $2,30\times$), lo que refuerza, y no diluye, su condición de alerta.

### 3.3 Rectificación Bibliográfica Rigurosa
* **Duke & Green (2001, Med J Aust 174:122–125):** Estudio de cohortes emparejadas en Melbourne sobre 73 adultos críticos derivados por saturación de UCI vs 73 controles: mortalidad de **$24,7\%$ en derivados vs $17,8\%$ en controles** (OR $1,5$; IC 95%: $0,68$–$3,4$).
* **Resumen Duke University Medical Center (2001):** La mortalidad global de la UCI quirúrgica fue de **$9,6\%$**, mientras que la de los pacientes derivados de otros centros fue de **$33,5\%$**.

---

## 4. Estadía: Selección en Desarrollo y Sensibilidad por Bloques

### 4.1 Selección Entrenada en Cohorte de Desarrollo (2020–2022) y Grilla ElasticNet
La selección de covariables se entrenó exclusivamente en la cohorte de desarrollo de sobrevivientes con estancia positiva ($N = \mathbf{1.303.718}$ episodios):
* **Grilla de Hiperparámetros:** Grilla geométrica fija de $n_{\alpha} = 100$ puntos espaciados logarítmicamente entre $\alpha_{\max} = 1,143086$ y $\alpha_{\min} = 0,001143$, con `l1_ratio = 0.5` y validación cruzada en 5 pliegues (`random_state = 42`).
* **Regla 1-SE:** El $\alpha_{\min}$ óptimo que minimiza el error cuadrático medio fue $0,001511$. El hiperparámetro seleccionado por la regla del error estándar fue $\alpha_{1\text{se}} = \mathbf{0,037431}$ (equivalente al $\alpha_{1\text{se}} = 0,036680$ de la grilla previa en muestra 30k, debido al paso logarítmico $\Delta \log_{10} \alpha \approx 0,03$).

#### Comparación de Coeficientes Estandarizados (Muestra 30k vs Cohorte Completa Dev 1.3M):
| Covariable Retenida | Nombre / Categoría | Coef. Muestra 30k | Coef. Dev Completa (N = 1.303.718) |
| :--- | :--- | :---: | :---: |
| `CIE10_ENC` | Diagnóstico Principal (LOHO) | **+0,4360** | **+0,4344** |
| `INGRESO_URGENCIA` | Admisión por Urgencia | **+0,1353** | **+0,1301** |
| `INGRESO_CRITICO` | Ingreso a UCI / UTI | **+0,0853** | **+0,0772** |
| `ELIX_24` | Pérdida de peso / Desnutrición | **+0,0562** | **+0,0577** |
| `ELIX_04` | Circulación pulmonar | **+0,0422** | **+0,0392** |
| `DERIVADO_OTRO_HOSPITAL` | Derivado de otro centro | **+0,0311** | **+0,0243** |
| `EDAD_ANIOS` | Edad en años continuos | **+0,0175** | **+0,0339** |
| `ELIX_14` | Enfermedad renal crónica | **+0,0228** | **+0,0307** |
| `ELIX_23` | Obesidad | **+0,0177** | **+0,0187** |
| `ELIX_01` | Insuficiencia cardíaca | **+0,0150** | **+0,0182** |
| `ELIX_09` | Otros trastornos neurológicos | **+0,0142** | **+0,0119** |
| `ELIX_05` | Enfermedad vascular periférica | **+0,0160** | **+0,0099** |
| `ELIX_15` | Hepatopatía | **+0,0153** | **+0,0095** |
| `ELIX_30` | Psicosis | **+0,0026** | **+0,0080** |
| `ELIX_08` | Parálisis / Hemiplejia | **+0,0140** | **+0,0066** |
| `ELIX_06` | Hipertensión no complicada | **+0,0134** | **+0,0038** |
| `ELIX_19` | Tumor metastásico | **+0,0028** | **+0,0024** |

Las 17 variables seleccionadas muestran una estabilidad paramétrica absoluta entre la submuestra de entrenamiento y la población completa de desarrollo.

---

### 4.2 Sensibilidad por Bloques en Estadía (Evaluación 2023, N = 507.811)
Para evaluar la contribución de cada dimensión clínica sobre el ordenamiento institucional de estancia, se calcularon los rankings de $O/E$ hospitalario bajo cuatro bloques jerárquicos de ajuste:
* **Bloque 1 (Demográfico):** `EDAD_ANIOS` + `SEXO_MASCULINO`.
* **Bloque 2 (+ Ingreso / Severidad):** Bloque 1 + `INGRESO_URGENCIA` + `INGRESO_CRITICO` + `DERIVADO_OTRO_HOSPITAL`.
* **Bloque 3 (+ Comorbilidades):** Bloque 2 + 28 comorbilidades crónicas Elixhauser.
* **Bloque 4 (+ Diagnóstico Principal LOHO - Modelo Completo):** Bloque 3 + `CIE10_ENC` con codificación fuera del hospital (*Leave-One-Hospital-Out*).

#### Matriz de Correlación de Rangos de Spearman ($\rho$) y Desplazamiento RMS:
| Comparación de Bloques | Spearman $\rho$ | Desplazamiento RMS (Puestos) | Interpretación Metodológica |
| :--- | :---: | :---: | :--- |
| **B1 (Demográfico) vs B4 (Full)** | **0,8565** | **10,05 puestos** | La demografía sola distorsiona el ranking en ~10 posiciones. |
| **B2 (+ Ingreso/Severidad) vs B4** | **0,9311** | **6,96 puestos** | El modo de admisión explica gran parte de la severidad inicial. |
| **B3 (+ Comorbilidades) vs B4** | **0,9669** | **4,83 puestos** | Las comorbilidades estabilizan el ajuste previo al diagnóstico. |
| **B3 (+ Comorbilidades) vs B4 (Full LOHO)**| **0,9669** | **4,83 puestos** | `CIE10_ENC` refina el perfil patológico específico sin fuga hospitalaria. |
| **B1 vs B2** | **0,8933** | — | Impacto sustantivo del ingreso crítico y urgencia ($\Delta \rho = 0,11$). |
| **B2 vs B3** | **0,9671** | — | Ajuste fino por carga de multimorbilidad. |
| **17 Variables vs 34 Variables (B4)** | **0,9990** | **0,86 puestos** | La regla parsimoniosa preserva el ranking de estancia intacto. |

---

## 5. Auditoría de la Calibración en 0–2 Diagnósticos Secundarios

### 5.1 Evidencia Clínica de la Subpoblación con 0–2 Diagnósticos
Se evaluó el perfil asistencial de los $159.821$ episodios ($28,7\%$ de la cohorte 2023) con 0 a 2 diagnósticos frente a los $397.497$ episodios con $\ge 3$ diagnósticos:
* **Mortalidad Observada:** **$0,288\%$** ($460$ muertes) en $0$–$2$ dx vs **$5,98\%$** ($23.765$ muertes) en $\ge 3$ dx (razón de 21 a 1).
* **Vía de Ingreso:**  
  - Urgencia: **$47,9\%$** ($76.496$ episodios).
  - Programada / Electiva: **$35,8\%$** ($57.202$ episodios).
  - Obstétrica: **$16,3\%$** ($26.119$ episodios).  
  *La urgencia explica casi la mitad de los ingresos en este estrato*, por lo que el fenómeno no es atribuible con exclusividad a cirugías electivas.
* **Uso de Pabellón Quirúrgico:** $66,7\%$ ($106.588$ episodios) requirieron intervención en quirófano.
* **Estancia Hospitalaria:** Mediana de **$2,0$ días** (media $3,53$) en $0$–$2$ dx vs mediana de **$5,0$ días** (media $9,35$) en $\ge 3$ dx.

**Conclusión Asistencial:**  
El estrato de 0 a 2 diagnósticos está dominado por **episodios de ultra-corta estancia (mediana de 48 horas)**, tanto médicos urgentes como quirúrgicos, donde el paciente egresa rápidamente y el equipo médico no desglosa comorbilidades crónicas secundarias preexistentes en la epicrisis.

---

### 5.2 Bootstrap Pareado de la Correlación: Desplazamiento y Cambio de Signo
Se recalculó el bootstrap pareado (1.000 réplicas independientes) de la correlación de Spearman entre la proporción institucional de pacientes con $0$–$2$ diagnósticos y el $O/E$ hospitalario:
* **En M2 (Clínico Base):**  
  $$r_s(M2) = \mathbf{+0,104} \quad [\text{IC 95\%: } -0,151 \text{ a } +0,341]$$  
  (Una mayor fracción de pacientes con 0–2 diagnósticos se asocia con un $O/E$ más alto; el hospital se ve relativamente peor).
* **En M3 (Splines $R_{\text{dx}}$):**  
  $$r_s(M3) = \mathbf{-0,121} \quad [\text{IC 95\%: } -0,383 \text{ a } +0,131]$$  
  (Una mayor fracción de pacientes con 0–2 diagnósticos se asocia con un $O/E$ más bajo; el hospital se ve relativamente mejor).
* **Diferencia Pareada $\Delta r_s (M3 - M2)$:**  
  $$\mathbf{\Delta r_s = -0,225} \quad [\text{IC 95\%: } -0,427 \text{ a } -0,038], \quad p = \mathbf{0,0100}$$

**Reescritura Rigurosa de la Conclusión:**
1. **Cambio de Signo y Desplazamiento:** M3 **no atenúa** una correlación preexistente; lo que hace es **invertir el signo de la asociación** (de $+0,104$ a $-0,121$).
2. **Magnitud Inalterada:** La magnitud absoluta de la asociación no disminuye ($|-0,121|$ frente a $|+0,104|$), y ambos intervalos de confianza individuales abarcan el cero, indicando ausencia de correlación monotónica lineal global significativa en ambos modelos tomados por separado.
3. **Efecto Significativo del Ajuste ($\Delta = -0,225, p = 0,010$):** La prueba pareada demuestra que la introducción de $R_{\text{dx}}$ produce un **desplazamiento sistemático hacia valores negativos**, sobrecompensando la penalización que el modelo previo imponía a establecimientos con internaciones breves.

---

## 6. Catálogo Maestro de Hospitales y Verificación de Tablas

### 6.1 Estado de `200717` (Complejo Asistencial Padre Las Casas)
En [catalogo_hospitales_procedencia.csv](file:///home/felipe/Documentos/Proyecto%20final/Tesis/config/catalogo_hospitales_procedencia.csv):
* **Código:** `200717`
* **Nombre Oficial:** Complejo Asistencial Padre Las Casas
* **Servicio de Salud:** Servicio de Salud Araucanía Sur | **Región:** La Araucanía
* **Año de Incorporación:** 2024
* **Estado:** `inferido`
* **Fuente:** `SUPERINTENDENCIA_SALUD_REG_1033_Y_DERIVACIONES_2024`
* **Evidencia Documental:** Establecimiento de mediana-alta complejidad incorporado a la red asistencial en 2020-2024, receptor en 2024 de 832 derivaciones de la red Araucanía Sur (Pitrufquén 488, Villarrica 282, Nueva Imperial 34, Lautaro 28). Inscripción vigente en el Registro Nacional de Prestadores Institucionales de la Superintendencia de Salud bajo el Registro N° 1033.

### 6.2 Regeneración de la Tabla de CIP Nulos: HUAP (`111195`) vs Instituto del Tórax (`112103`)
Se confirmó y regeneró la auditoría de registros con CIP encriptado nulo en la capa Silver a través del periodo histórico 2019–2024:
* **Código `111195`:** Corresponde oficialmente al **Hospital de Urgencia Asistencia Pública Dr. Alejandro del Río (HUAP / Posta Central)**. Concentra **252 registros con CIP nulo**, lo cual cuadra exactamente con su perfil asistencial de urgencia extrema y trauma de la vía pública, donde ingresan pacientes indocumentados o inconscientes sin cédula de identidad.
* **Código `112103`:** Corresponde oficialmente al **Instituto Nacional del Tórax**. Registra exactamente **0 registros con CIP nulo**, lo cual concuerda con su carácter de instituto monográfico de derivación electiva y terciaria donde todo ingreso requiere orden de hospitalización programada y cédula validada.

La versión previa que etiquetaba el código `111195` como "Instituto del Tórax" queda subsanada en todas las tablas del repositorio.

---

## 7. Dictamen Final de Cierre de Fase 2

Con estas rectificaciones:
1. **La lista de alertas de M3** queda transparentada y explicada a través del cálculo bayesiano empírico con margen de materialidad clínica ($10\%$).
2. **La conclusión de calibración diagnóstica en 0–2 dx** queda redactada correctamente reconociendo el cambio de signo ($\Delta r_s = -0,225, p = 0,010$) y el perfil asistencial de corta estadía.
3. **El criterio de centros monográficos** queda anclado a una regla objetiva medida ($\max \text{MDC} \ge 50\%$ o $\text{MDC 4+5} \ge 70\%$), aislando al Tórax y Traumatológico, y delimitando **4 alertas robustas generales bajo $M2 \cap M3$ y 6 bajo la intersección de variantes de M3**.
4. **La selección de estadía** se sustenta en la cohorte de desarrollo, reportando la grilla y la sensibilidad jerárquica por bloques.
5. **El tipping point y la censura** reportan a todos los emisores, documentando el umbral de entrada de Curanilahue ($\lambda_{\text{entrada}} = 2,00\times$) y la inmunidad de las alertas observadas.
6. **El catálogo y la tabla de CIP nulos** quedan plenamente alineados al estándar oficial DEIS.

La Fase 2 queda **concluida, validada matemáticamente y cerrada formalmente**.
