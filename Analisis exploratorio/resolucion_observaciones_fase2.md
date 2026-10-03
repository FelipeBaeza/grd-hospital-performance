# Resolución Definitiva de Observaciones de Metodología y Auditoría Empírica - Fase 2
**Proyecto de Tesis:** Indicador Hospitalario $O/E$ Ajustado por Riesgo con Machine Learning  
**Base de Datos:** Egresos Hospitalarios FONASA 2019–2024 ($N = 5.808.536$ registros)  
**Fecha de Consolidación:** Octubre 2026

---

## 1. Calibración Rigurosa sobre la Cohorte Inpatient Adulta

### 1.1 Corrección de Cohorte y Definición Canónica de Diagnósticos Secundarios ($N_{\text{dx}}$)
Se rectifica formalmente la cohorte analítica de calibración:
* **Población Inpatient Adulta Evaluable (2023):** $N = \mathbf{557.402}$ episodios y $O = \mathbf{24.225}$ defunciones intrahospitalarias observadas (tasa bruta $4,346\%$).
* **Definición de $N_{\text{dx}}$:** Corresponde estrictamente al recuento de diagnósticos secundarios codificados (`DIAGNOSTICO2` a `DIAGNOSTICO35`), con un rango observado de $0$ a $34$ diagnósticos por episodio.
* **Métricas Globales de Predicción (Desarrollo 2020–2022 $\longrightarrow$ Calibración 2023):**
  - Muertes Observadas ($O$): **$24.225$**
  - Muertes Esperadas del Modelo ($\sum E$): **$34.618,09$**
  - **$O/E$ Global Real Crudo:** **$0,6998$** (factor de recalibración $k = 0,6998$; multiplicador de intercepto $1/k = 1,4289$).

---

### 1.2 Tabla Canónica de Calibración por Estratos de $N_{\text{dx}}$ (Año 2023)

| Estrato $N_{\text{dx}}$ | N Episodios | Muertes Observadas ($O$) | Muertes Esperadas ($E$) | $O/E$ Crudo | $O/E$ Post-Recalibrado | Estado Contrato $[0,80, \ 1,25]$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 diagnósticos** | 40.495 | 39 | 794,9 | 0,0491 | **0,0701** | FUERA (Sobrepredicho) |
| **1 diagnóstico** | 56.530 | 125 | 1.410,6 | 0,0886 | **0,1266** | FUERA (Sobrepredicho) |
| **2 diagnósticos** | 62.840 | 296 | 1.965,9 | 0,1506 | **0,2152** | FUERA (Sobrepredicho) |
| **3 diagnósticos** | 63.670 | 615 | 2.500,2 | 0,2460 | **0,3515** | FUERA (Sobrepredicho) |
| **4 diagnósticos** | 58.992 | 954 | 2.799,1 | 0,3408 | **0,4870** | FUERA (Sobrepredicho) |
| **5 diagnósticos** | 51.883 | 1.367 | 2.925,1 | 0,4673 | **0,6678** | FUERA (Sobrepredicho) |
| **6–10 diagnósticos** | 150.094 | 8.831 | 12.132,6 | 0,7279 | **1,0401** | **DENTRO (Óptimo)** |
| **$\ge 11$ diagnósticos** | 72.898 | 11.998 | 10.089,7 | 1,1891 | **1,6993** | FUERA (Subpredicho) |
| **Total Red Inpatient** | **557.402** | **24.225** | **34.618,1** | **0,6998** | **1,0000** | **GLOBAL CALIBRADO** |

*Diagnóstico de Descalibración Estructural:*  
El estrato de $6–10$ diagnósticos secundarios ($N = 150.094$) queda perfectamente calibrado en **$1,0401$**. Sin embargo, la especificación puramente lineal de comorbilidades sobrepredice masivamente el riesgo de muerte en pacientes con $0$ a $4$ diagnósticos ($O/E \le 0,49$), mientras que subpredice el riesgo en pacientes hipercomplejos con $\ge 11$ diagnósticos ($O/E = 1,6993$), estrato que concentra casi la mitad de las muertes del país ($11.998$ de $24.225$).  
*Criterio Congelado para Fase 3:* Se adopta como requisito contractual el reemplazo de la escala lineal por la **razón intrahospitalaria de diagnósticos** ($R_{\text{dx}, ij} = N_{\text{dx}, ij} / \bar{N}_{\text{dx}, j}$) modelada mediante **splines cúbicos restringidos**, exigiendo que tras recalibrar, todo estrato con $E \ge 100$ se ubique dentro de $[0,80, \ 1,25]$.

---

## 2. Gráfico de Embudo y Sobredispersión (Spiegelhalter 2005)

### 2.1 Diagnóstico de Sobredispersión en la Red Hospitalaria
En modelos institucionales con gran volumen de egresos, la varianza entre hospitales supera ampliamente la varianza binomial/Poisson pura.
* **Factor de Sobredispersión Crudo ($\phi$):** $\phi_{\text{crudo}} = \mathbf{64,13}$
* **Factor de Sobredispersión Winsorizado al 10% (Spiegelhalter 2005):** $\phi_{\text{wins}} = \mathbf{58,66}$
* **Error Estándar Ajustado:** $\text{SE}_{\text{adj}, j} = \sqrt{\frac{\phi_{\text{wins}}}{E_j}} = \frac{7,66}{\sqrt{E_j}}$

### 2.2 Impacto en la Clasificación del Gráfico de Embudo ($\pm 3\sigma$)
* **Bajo Límites Poisson Naive (Sin Ajuste):** 16 hospitales clasificaban como *Sobresaliente*, 34 como *Promedio* y 15 como *Alerta de Mortalidad* (un $47,7\%$ de prestadores fuera de control, patrón típico de falso sobre-aislamiento).
* **Bajo Límites Ajustados por Sobredispersión (Spiegelhalter):** Los **65 hospitales de agudos de adultos clasifican dentro de la banda de variación institucional admisible (*Promedio*)**.
* **Sensibilidad del Día 0 bajo Límites Ajustados:**
  - Desplazamiento máximo de ranking: **9,0 puestos**.
  - Desplazamiento cuadrático medio (RMS): **2,51 puestos**.
  - **Spearman $\rho$ real (Base vs No-Día-0):** **$\mathbf{0,9911}$** (actualizado respecto al 0,945 previo que mezclaba escalas).
  - **Cambios de Categoría de Desempeño bajo Límites Ajustados:** **0 hospitales cambian de categoría** (los 65 permanecen en el rango esperado).

---

## 3. Upcoding: Evidencia Empírica de Atenuación

Conforme a la pauta de rigor metodológico, se consigna la formulación exacta:
> *"El centrado intrahospitalario desplaza la asociación con la intensidad de codificación en $\Delta r_s = +0,151$ [IC 95%: $+0,068$ a $+0,253$; $p < 0,001$ por bootstrap pareado con 1.000 réplicas]; la asociación basal era débil ($r_s = -0,089$ en modelo crudo y $r_s = +0,066$ en modelo centrado)."*

---

## 4. Censura, Ponderación IPW y Análisis de Punto de Inflexión (*Tipping Point*)

### 4.1 Análisis Sistemático de Punto de Inflexión en la Red
Para cada establecimiento $j$, se modeló la tasa crítica de mortalidad en sus transferencias derivadas ($p_{\text{cens}}^*$) requerida para que el hospital cruce el umbral de alerta institucional ($\text{HSMR} > 110$):
* **Hospitales con Alta Tasa de Derivación y su $p^*$ Crítico:**
  - `128109` (Hospital de Curanilahue, censura $17,79\%$): $p^* = \mathbf{0,99\%}$
  - `113180` (Hospital El Pino, censura $17,77\%$): Ya en alerta basal ($O/E = 1,058$)
  - `121110` (Hospital Dr. Abraham Godoy de Lautaro, censura $15,14\%$): $p^* = \mathbf{0,89\%}$
  - `121117` (Hospital de Pitrufquén, censura $14,39\%$): $p^* = \mathbf{9,09\%}$
  - `121121` (Hospital de Villarrica, censura $12,33\%$): $p^* = \mathbf{16,64\%}$
  - `116110` (Hospital San José de Parral, censura $10,83\%$): $p^* = \mathbf{16,93\%}$
  - `121114` (Hospital Intercultural de Nueva Imperial, censura $10,34\%$): $p^* = \mathbf{21,06\%}$
* **Comparación con Benchmark Clínico:**
  En la literatura internacional sobre traslados interhospitalarios de urgencia médica y quirúrgica compleja hacia centros terciarios (ej. Duke et al., *Crit Care Resusc* 2004; Rosenberg et al., *JAMA* 2014), la mortalidad intrahospitalaria de los pacientes derivados oscila entre un $8\%$ y un $15\%$.
  Bajo un escenario conservador de $10\%$ de mortalidad en derivados, **8 de los 65 hospitales de la red cruzarían el umbral a zona de alerta**, evidenciando que la censura no ajustada favorece artificialmente a los centros periféricos de derivación.
* **Modelo IPW:** Se congela para Fase 3 el ajuste por probabilidad inversa de censura ($w_i = 1 / P(\text{No Derivado} \mid X_i, \text{Hospital})$) incorporando efectos fijos por prestador.

---

## 5. Selección Empírica en Estadía y Auditoría de Casos Específicos

### 5.1 Selección ElasticNet y Estabilidad en el Modelo de Estadía
Se aplicó regularización ElasticNet con validación cruzada de 5 pliegues sobre la cohorte adulta de sobrevivientes con estancia positiva ($p99 = 54$ días):
* **Parámetros Óptimos:** $\alpha = \mathbf{0,000207}$, $L_1\text{-ratio} = \mathbf{1,00}$ (selección pura Lasso).
* **Variables Seleccionadas (25 de 29) y Coeficientes Estandarizados:**
  1. `EDAD_ANIOS` ($\beta_{\text{std}} = +0,1351 \implies \times 1,145$ por DE de edad)
  2. `ELIX_24` Desnutrición / Pérdida de peso ($\beta_{\text{std}} = +0,1092 \implies \times 1,115$)
  3. `ELIX_14` Insuficiencia renal crónica ($\beta_{\text{std}} = +0,0870 \implies \times 1,091$)
  4. `ELIX_29` Abuso de drogas ($\beta_{\text{std}} = +0,0832 \implies \times 1,087$)
  5. `ELIX_30` Psicosis ($\beta_{\text{std}} = +0,0561 \implies \times 1,058$)
  6. `ELIX_04` Trastornos de circulación pulmonar ($\beta_{\text{std}} = +0,0546 \implies \times 1,056$)
  7. `ELIX_06` Hipertensión no complicada ($\beta_{\text{std}} = +0,0540 \implies \times 1,056$)
  8. `ELIX_15` Enfermedad hepática ($\beta_{\text{std}} = +0,0537 \implies \times 1,055$)
  9. `ELIX_01` Insuficiencia cardíaca congestiva ($\beta_{\text{std}} = +0,0532 \implies \times 1,055$)
  10. `ELIX_09` Otros neurológicos ($\beta_{\text{std}} = +0,0492 \implies \times 1,050$)
* **Variables Descartadas (Coeficiente Nulo):**
  `ELIX_11` (Diabetes no complicada), `ELIX_12` (Diabetes complicada), `ELIX_16` (Úlcera péptica) y `ELIX_26` (Anemia hemorrágica).

---

### 5.2 Clarificación Geográfica y Administrativa: Quillota vs Padre Las Casas (`200717`)
La auditoría territorial de los datos crudos de 2024 desmintió la hipótesis preliminar de que `200717` fuera el reemplazo de Quillota:
* **Código `200717`:** Corresponde al **Hospital Complejo Asistencial Padre Las Casas**, ubicado en la Región de La Araucanía (*Servicio de Salud Araucanía Sur*, comuna de Padre Las Casas/Temuco). En 2024 registra $10.204$ episodios, y sus derivaciones provienen exclusivamente de la red de la Araucanía: Pitrufquén (488), Villarrica (282), Nueva Imperial (34) y Lautaro (28).
* **Código `107101`:** Es el **Hospital San Martín de Quillota** (*Servicio de Salud Viña del Mar - Quillota*, Región de Valparaíso).
* **Conclusión:** Existen cero transferencias y cero coincidencias de `CIP_ENCRIPTADO` entre `107101` y `200717` porque corresponden a establecimientos autónomos distantes por más de 700 kilómetros. Ambos se tratan de manera independiente.

---

### 5.3 Auditoría Sintáctica de Códigos CIE-10 (Capa Bronze 2019–2024)
Se auditó la totalidad de los registros de diagnóstico principal en la capa Bronze a lo largo de los seis años:
* **Heterogeneidad de Puntos Decimales:** Entre un **$5,13\%$ y un $6,27\%$** de los registros de cada año se reciben sin punto decimal (ej. $62.154$ registros en 2023, $5,98\%$).
* **Causa Estructural del Estándar CIE-10:** El **$100\%$ de los códigos sin punto corresponden a categorías de 3 caracteres de longitud** (`N47` Fimosis, `N10` Nefritis tubulointersticial, `N40` Hiperplasia prostática, `J46` Estado asmático, `C73` Cáncer tiroideo, etc.). En la clasificación CIE-10 internacional, estas categorías no poseen cuarto dígito decimal, por lo que su formato sin punto es normativamente correcto.
* **Validación de Capa Silver:** La función `REPLACE('.', '')` y la extracción canónica del código de 3 caracteres (`CIE10_3C`) unifican el $100\%$ de la base sin pérdida de información.

---

### 5.4 Robustez de Mortalidad y Gobernanza Clínica
* **Correlación de Rankings (28 Crónicas vs 31 con Agudas):**
  $$\rho_{\text{Spearman}} = \mathbf{0,9777}$$
* **Encuadre:** La correlación de $0,978$ constata que la inclusión de `ELIX_02` (Arritmias), `ELIX_22` (Coagulopatía) y `ELIX_25` (Desequilibrio hidroelectrolítico) introduce reordenamientos sensibles de prestadores. La decisión de mantenerlas fuera se sustenta en la **gobernanza clínica del indicador**, garantizando que el sistema no premie a hospitales que inducen o complican el manejo hidroelectrolítico y metabólico durante la internación.

---

### Verificación en el Repositorio
- [resolucion_observaciones_fase2.md](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/resolucion_observaciones_fase2.md)
- [07_calibracion_inpatient_fase2.py](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/07_calibracion_inpatient_fase2.py)
- Commit de consolidación: `5a37168`.
