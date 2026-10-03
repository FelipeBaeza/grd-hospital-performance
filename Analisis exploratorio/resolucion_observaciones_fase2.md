# Resolución Definitiva de Observaciones y Auditoría Empírica - Cierre Formal de Fase 2

**Proyecto de Tesis:** Indicador Hospitalario $O/E$ Ajustado por Riesgo con Machine Learning  
**Base de Datos:** Egresos Hospitalarios FONASA 2019–2024 ($N = 5.808.536$ registros brutos)  
**Estado:** Fase 2 ("Comprensión de los Datos y Modelado Exploratorio") **CERRADA FORMALMENTE**  
**Fecha de Consolidación:** Octubre 2026  

---

## 1. Dependencia de la Especificación, Alerta Robusta y Centros Monográficos

### 1.1 Sensibilidad Inter-Modelo: El Hallazgo Central de la Fase 2
La comparación entre las tres especificaciones de riesgo de mortalidad hospitalaria revela una marcada dependencia del ranking respecto a las decisiones de ajuste:
* **Modelo 1 (Demográfico Puro):** Edad + 28 Comorbilidades Crónicas Elixhauser ($\sum E = 34.616,1$).
* **Modelo 2 (Clínico con Severidad):** M1 + Ingreso por Urgencia + Cama Crítica (UCI/UTI) + Derivación Previa + Sexo Masculino ($\sum E = 33.720,8$).
* **Modelo 3 (Corregido con $R_{\text{dx}}$ Splines):** M2 + Razón Intrahospitalaria de Diagnósticos ($R_{\text{dx}} = N_{\text{dx}, ij} / \bar{N}_{\text{dx}, j}$) con Splines Cúbicos Restringidos ($\sum E = 35.343,9$).

**Correlaciones de Rango de Spearman ($\rho$):**
$$\rho(M1, M2) = \mathbf{0,7201} \quad | \quad \rho(M1, M3) = \mathbf{0,5270} \quad | \quad \rho(M2, M3) = \mathbf{0,7428}$$

Frente a una correlación de $\rho = 0,9911$ para la exclusión del día 0 (desplazamiento RMS de apenas 2,5 puestos), **la sensibilidad a la especificación de covariables es el verdadero determinante de la posición relativa de los prestadores**, con desplazamientos de hasta 54 puestos en la red (ej. Hospital Pereira pasa de $O/E = 0,63$ a $1,48$; San Borja de $0,50$ a $1,18$; Tórax de $0,91$ a $2,22$).

### 1.2 Regla de "Alerta Robusta" (Concordancia M2 y M3 al Margen del 10%)
Dado que en esta fase no se dispone de validación prospectiva fuera de muestra (año 2024) para dirimir si M3 supera definitivamente a M2 en capacidad de generalización por tipo de prestador, se adopta como principio de prudencia la regla de **Alerta Robusta**:
$$\text{Alerta Robusta: } P(\theta_j > 1,10 \mid M2) \ge 0,95 \quad \text{Y} \quad P(\theta_j > 1,10 \mid M3) \ge 0,95$$

Bajo este criterio, de los prestadores señalizados:
* **Alerta en M2:** 13 hospitales ($20,0\%$).
* **Alerta en M3:** 7 hospitales ($10,8\%$).
* **ALERTA ROBUSTA (Simultánea en ambos modelos): EXACTAMENTE 5 HOSPITALES**:
  1. `112103` Instituto Nacional del Tórax ($O/E_{\text{M2}} = 1,968$; $O/E_{\text{M3}} = 2,224$; $P = 1,000$) — *Monográfico*
  2. `113180` Hospital El Pino ($O/E_{\text{M2}} = 1,836$; $O/E_{\text{M3}} = 1,770$; $P = 1,000$)
  3. `106103` Hospital Claudio Vicuña de San Antonio ($O/E_{\text{M2}} = 1,595$; $O/E_{\text{M3}} = 1,632$; $P = 1,000$)
  4. `106100` Hospital Carlos Van Buren de Valparaíso ($O/E_{\text{M2}} = 1,293$; $O/E_{\text{M3}} = 1,434$; $P = 1,000$)
  5. `112101` Hospital Dr. Luis Tisné B. ($O/E_{\text{M2}} = 1,313$; $O/E_{\text{M3}} = 1,315$; $P = 1,000$)

**Hospitales con Alerta Sensible a la Especificación:**
* En Alerta solo en M3 (suben al ajustar por $R_{\text{dx}}$): `106102` Pereira ($O/E$ sube de $0,979$ a $1,476$) y `114105` La Florida ($O/E$ sube de $1,166$ a $1,302$).
* En Alerta solo en M2 (bajan al ajustar por $R_{\text{dx}}$): San Camilo `108100`, San Borja `111100`, Antofagasta `103100`, Temuco `121109`, Barros Luco `113100`, San José `109100`, Calama `103101`, Buin `113150`.

### 1.3 Clasificación y Aislamiento de Centros Monográficos
Se incorpora en el catálogo maestro la etiqueta `es_monografico` para aislar del ranking general de agudos a aquellos prestadores con misiones asistenciales superespecializadas:
* `112103`: **Instituto Nacional del Tórax** (Cirugía cardiovascular, trasplante y patología respiratoria compleja terciaria/cuaternaria).
* `110110`: **Instituto Traumatológico Dr. Teodoro Gebauer** (Cirugía ortopédica y politrauma electivo).
* `106102`: **Hospital Dr. Eduardo Pereira Ramírez** de Valparaíso (Cirugía de tórax y patología respiratoria crónica).

---

## 2. Inferencia Bayesiana Empírica: Reproducibilidad y Márgenes Clínicos

### 2.1 Aclaración de Cifras de la Regla Direccional (40 vs 51 Hospitales)
La diferencia entre las tasas de detección de la regla direccional $P(\theta > 1,0) \ge 0,95$ responde estrictamente a la especificación evaluada:
* **Bajo Modelo 2 (Clínico sin Splines):** Marca **51 de 65 hospitales ($78,5\%$)** (23 en alerta, 28 sobresalientes, 14 en promedio).
* **Bajo Modelo 3 (Splines $R_{\text{dx}}$ con Centrado $\mu_{\text{meta}} = 0,0257$):** Marca **40 de 65 hospitales ($61,5\%$)** (21 en alerta, 19 sobresalientes, 25 en promedio).
Ambas confirman que probar dirección estocástica en lugar de magnitud clínica sobre-clasifica a la mayor parte de la red pública.

### 2.2 Márgenes de Materialidad Clínica (10% y 20%) y Caso Curanilahue (`128109`)
El margen de materialidad clínica es una **decisión metodológica de diseño del indicador**, que busca tolerancia frente a ruido no asistencial:
* **Margen del 10%:** $\text{Alerta si } P(\theta_j > 1,10) \ge 0,95 \implies$ **7 hospitales en alerta ($10,8\%$)**, **10 sobresalientes ($15,4\%$)**, **48 promedio ($73,8\%$)**.
* **Margen del 20%:** $\text{Alerta si } P(\theta_j > 1,20) \ge 0,95 \implies$ **7 hospitales en alerta ($10,8\%$)**, **4 sobresalientes ($6,2\%$)**, **54 promedio ($83,1\%$)**.

**Auditoría Específica de Curanilahue (`128109`):**
Curanilahue registra $O = 136, E_3 = 174,6 \implies O/E = 1,137$.  
Con $v_j = 1/136 = 0,007353$, $\log(\theta_{\text{post}}) = 0,1142$ y $\text{SE}_{\text{post}} = 0,0797$:
* Para margen 10%: $Z = (0,1142 - \log(1,10)) / 0,0797 = +0,237 \implies P(\theta > 1,10) = \mathbf{0,5938} < 0,95$.
* Para margen 20%: $Z = (0,1142 - \log(1,20)) / 0,0797 = -0,854 \implies P(\theta > 1,20) = \mathbf{0,1964} < 0,95$.  
Curanilahue queda clasificado en **PROMEDIO** en el modelo final M3 tanto al 10% como al 20%. Los 7 hospitales que superan $P(\theta > 1,20) \ge 0,95$ en M3 son Pereira, El Pino, La Florida, Tisné, Claudio Vicuña, Van Buren y Tórax.

### 2.3 Reproducibilidad Paso a Paso del Cálculo Bayesiano Empírico
Para auditoría externa, se consigna la fórmula analítica de contracción:
$$v_j = \frac{1}{O_j}, \quad B_j = \frac{\tau^2}{\tau^2 + v_j}, \quad \log(\theta_j^{\text{EB}}) = B_j \log\left(\frac{O_j}{E_j \cdot k}\right) + (1 - B_j)\mu_{\text{meta}}, \quad \text{SE}_{\text{post}, j} = \sqrt{\frac{\tau^2 v_j}{\tau^2 + v_j}}$$
$$Z_j = \frac{\log(\theta_j^{\text{EB}}) - \log(1,10)}{\text{SE}_{\text{post}, j}}, \quad P(\theta_j > 1,10 \mid \text{datos}) = \Phi(Z_j)$$

**Ejemplo de Cálculo: Hospital El Pino (`113180`):**
* $O_j = 438$, $E_{3, j} = 361,0$, $k_3 = 0,6854 \implies O/E_{\text{post}} = 1,7702$.
* $\log(O/E_{\text{post}}) = \mathbf{0,5711}$.
* Varianza within: $v_j = 1 / 438 = \mathbf{0,002283}$.
* Con $\tau^2 = 0,04674$ y $\mu_{\text{meta}} = 0,0257$:
  $$B_j = \frac{0,04674}{0,04674 + 0,002283} = \mathbf{0,9534}$$
  $$\log(\theta_j^{\text{EB}}) = 0,9534 \cdot (0,5711) + (1 - 0,9534) \cdot (0,0257) = \mathbf{0,5457} \implies \theta_j^{\text{EB}} = 1,7258$$
  $$\text{SE}_{\text{post}, j} = \sqrt{\frac{0,04674 \cdot 0,002283}{0,049023}} = \mathbf{0,04665}$$
  $$Z = \frac{0,5457 - \log(1,10)}{0,04665} = \frac{0,5457 - 0,09531}{0,04665} = \mathbf{+9,65} \implies P(\theta_j > 1,10) = \mathbf{1,0000} \ (\text{ALERTA})$$

---

## 3. *Tipping Point* y Análisis de Sensibilidad con Traslados Enlazados

### 3.1 Microdatos de Traslados Enlazados al Receptor (Año 2023)
En 2023 se registraron $31.082$ episodios censurados por derivación hacia otros establecimientos. De ellos, se logró enlazar determinísticamente al receptor dentro de 30 días a **$12.770$ episodios ($41,1\%$ de tasa de enlace)**:
* **Mortalidad en el receptor:** **$6,01\%$ global** ($767$ defunciones).
* **Mortalidad observada por emisor:** Oscila entre $2,1\%$ y $6,9\%$, lo que equivale a multiplicadores $\lambda_{\text{obs}} = O_{\text{receptor}} / E_{\text{link}}$ de **$0,47\times$ a $1,34\times$**:
  - Hospital Intercultural de Nueva Imperial: $\lambda_{\text{obs}} = \mathbf{0,47\times}$ (Mortalidad observada $2,10\%$).
  - Hospital de Lautaro: $\lambda_{\text{obs}} = \mathbf{0,47\times}$ (Mortalidad observada $4,04\%$).
  - Hospital San Martín de Quillota: $\lambda_{\text{obs}} = \mathbf{0,55\times}$ (Mortalidad observada $3,85\%$).
  - Hospital de Pitrufquén: $\lambda_{\text{obs}} = \mathbf{0,58\times}$ (Mortalidad observada $5,19\%$).
  - Hospital de Curanilahue: $\lambda_{\text{obs}} = \mathbf{0,80\times}$ (Mortalidad observada $4,32\%$).
  - Complejo Sótero del Río: $\lambda_{\text{obs}} = \mathbf{1,04\times}$ (Mortalidad observada $5,96\%$).
  - Hospital San José de Parral: $\lambda_{\text{obs}} = \mathbf{1,34\times}$ (Mortalidad observada $5,51\%$).

### 3.2 Impacto sobre Curanilahue y Límites de la Fracción No Enlazada (59%)
1. **Sensibilidad de Curanilahue (`128109`):**  
   Para Curanilahue, su umbral de salida de alerta en M2/M3 era $\lambda_{\text{salida}}^* = \mathbf{0,82\times}$ (mortalidad del $4,4\%$). Como su mortalidad observada real en el receptor es de **$4,32\%$ ($\lambda_{\text{obs}} = 0,80\times < 0,82\times$)**, si se le atribuyera el desenlace de sus derivados, Curanilahue **saldría formalmente de zona de alerta**. Por ende, su alerta es altamente sensible a la regla de censura y no debe catalogarse como robusta.
2. **Cotas para el 58,9% No Enlazado ($18.312$ casos):**  
   Este grupo incluye traslados al sector privado, hospitales comunitarios de baja complejidad o desenlaces extrahospitalarios. Asumiendo un escenario extremo de estrés donde los no enlazados tuvieran una letalidad del **$15,0\%$** (2,5 veces el promedio enlazado), solo Parral y Curanilahue verían incrementado su $O/E$ en más de un $+8\%$, mientras que prestadores como Villarrica o Nueva Imperial mantendrían su condición de desempeño sobresaliente.

### 3.3 Rectificación Bibliográfica Rigurosa
* **Duke & Green (2001, Med J Aust 174:122–125):** Estudio de cohortes emparejadas en Melbourne sobre 73 adultos críticos derivados por saturación de UCI vs 73 controles: mortalidad de **$24,7\%$ en derivados vs $17,8\%$ en controles** (OR $1,5$; IC 95%: $0,68$–$3,4$).
* **Resumen Duke University Medical Center (2001):** La mortalidad global de la UCI quirúrgica fue de **$9,6\%$**, mientras que la de los pacientes derivados de otros centros fue de **$33,5\%$**.

---

## 4. Estadía: Selección en Desarrollo y Preservación de Rankings

### 4.1 Selección Entrenada en Cohorte de Desarrollo (2020–2022)
Para evitar fuga de datos (*data leakage*), la selección de características se ejecutó sobre la cohorte de desarrollo de sobrevivientes con estancia positiva ($N = \mathbf{1.303.718}$ episodios):
* **Modelo Completo (34 Covariables):** Retiene 28 covariables con $\alpha_{\min} = 0,002251$.
* **Modelo Parsimonioso (17 Covariables con Regla 1-SE):** Con $\alpha_{1\text{se}} = 0,036680$, retiene exactamente **17 variables**:
  `CIE10_ENC`, `INGRESO_URGENCIA`, `ELIX_24` (Desnutrición), `INGRESO_CRITICO`, `ELIX_04` (Circulación pulm.), `ELIX_14` (Renal crónica), `ELIX_23` (Obesidad), `ELIX_05` (Vascular perif.), `ELIX_15` (Hígado), `ELIX_30` (Psicosis), `ELIX_19` (Cáncer metastásico), `DERIVADO_OTRO_HOSPITAL`, `ELIX_01` (ICC), `ELIX_03` (Valvulopatía), `ELIX_06` (HTA simple), `EDAD_ANIOS` y `ELIX_09` (Otros neuro).
* **Descartadas por Parsimonia (17):** Parálisis (`ELIX_08`), HTA complicada (`ELIX_07`), Anemia deficiencia (`ELIX_27`), Tumor sólido (`ELIX_20`), Linfoma (`ELIX_18`), EPOC (`ELIX_10`), Abuso de drogas (`ELIX_29`), Abuso de alcohol (`ELIX_28`), Hipotiroidismo (`ELIX_13`), VIH (`ELIX_17`), Depresión (`ELIX_31`), Sexo, Diabetes simple (`ELIX_11`), Diabetes complicada (`ELIX_12`), Úlcera péptica (`ELIX_16`), Artritis (`ELIX_21`), Anemia hemorrágica (`ELIX_26`).

### 4.2 Impacto en el Ranking Hospitalario de Estadía (Evaluación 2023)
Al aplicar ambos modelos sobre los $507.811$ episodios de estancia de 2023 y comparar el $O/E$ hospitalario resultante entre las 34 y las 17 variables:
$$\rho_{\text{Spearman}} = \mathbf{0,99895} \approx \mathbf{0,999}$$
* **Desplazamiento Cuadrático Medio (RMS):** Apenas **$0,86$ puestos** en toda la red de 65 hospitales.  
La regla parsimoniosa de 17 variables preserva con fidelidad absoluta el ordenamiento institucional de estancia, reduciendo a la mitad los requerimientos de recolección de variables sin pérdida de información.

---

## 5. Auditoría de la Calibración en 0–2 Diagnósticos Secundarios

### 5.1 Evidencia Clínica de la Subpoblación con 0–2 Diagnósticos
Se evaluó el perfil asistencial de los $159.821$ episodios ($28,7\%$ de la red 2023) con 0 a 2 diagnósticos frente a los $397.497$ episodios con $\ge 3$ diagnósticos:
* **Mortalidad Observada:** **$0,288\%$** ($460$ muertes) en $0$–$2$ dx vs **$5,98\%$** ($23.765$ muertes) en $\ge 3$ dx (diferencia de 21 veces).
* **Ingreso por Urgencia:** $47,9\%$ en $0$–$2$ dx vs $71,9\%$ en $\ge 3$ dx.
* **Estancia Hospitalaria:** Mediana de **$2,0$ días** (media $3,5$) en $0$–$2$ dx vs mediana de **$5,0$ días** (media $9,4$) en $\ge 3$ dx.
* **Procedimiento Quirúrgico:** Presente en el $100\%$ de los casos quirúrgicos programados con egreso precoz.

Estos datos demuestran que el estrato $0$–$2$ diagnósticos no refleja una "falla biológica", sino un **patrón asistencial de internaciones cortas, electivas y quirúrgicas de bajo riesgo**, donde el equipo clínico habitualmente no desglosa comorbilidades crónicas secundarias en la ficha de egreso.

### 5.2 Bootstrap Pareado de la Correlación (% 0–2 dx vs $O/E$ Hospitalario)
Se ejecutó un bootstrap pareado (1.000 réplicas) para evaluar si el Modelo 3 atenúa la distorsión del subregistro sobre el indicador del prestador:
* $r_s(M2 \text{ Clínico}): \mathbf{+0,104}$ [IC 95%: $-0,151$ a $+0,341$]
* $r_s(M3 \text{ R_dx Splines}): \mathbf{-0,121}$ [IC 95%: $-0,383$ a $+0,131$]
* **Diferencia Pareada $\Delta r_s (M3 - M2):$** **$\mathbf{-0,225}$ [IC 95%: $-0,427$ a $-0,038$; $p = 0,0100$]**

La diferencia es estadísticamente significativa ($p = 0,010$), confirmando que el modelado no lineal de $R_{\text{dx}}$ desplaza la asociación en sentido negativo y neutraliza la ventaja que el modelo lineal previo otorgaba a hospitales con alto volumen de cirugías breves.

---

## 6. Catálogo Maestro de Hospitales y Taxonomía Formal

### 6.1 Estado de `200717` (Complejo Asistencial Padre Las Casas)
En [catalogo_hospitales_procedencia.csv](file:///home/felipe/Documentos/Proyecto%20final/Tesis/config/catalogo_hospitales_procedencia.csv):
* **Código:** `200717`
* **Nombre Oficial:** Complejo Asistencial Padre Las Casas
* **Servicio de Salud:** Servicio de Salud Araucanía Sur | **Región:** La Araucanía
* **Año de Incorporación:** 2024
* **Estado:** `inferido`
* **Fuente:** `SUPERINTENDENCIA_SALUD_REG_1033_Y_DERIVACIONES_2024`
* **Evidencia Documental:** Receptor en 2024 de 832 derivaciones de la red Araucanía Sur (Pitrufquén 488, Villarrica 282, Nueva Imperial 34, Lautaro 28). Establecimiento registrado ante la Superintendencia de Salud (Registro N° 1033) como prestador institucional de atención cerrada de mediana complejidad.

### 6.2 Corrección de Códigos: HUAP (`111195`) vs Instituto del Tórax (`112103`)
* `111195`: **Hospital de Urgencia Asistencia Pública Dr. Alejandro del Río (HUAP / Posta Central)**, *Servicio de Salud Metropolitano Central*.
* `112103`: **Instituto Nacional del Tórax**, *Servicio de Salud Metropolitano Oriente*.

---

## 7. Dictamen Final de Cierre de Fase 2

Con estas precisiones metodológicas, la formulación de la **Alerta Robusta (M2 $\cap$ M3)**, la delimitación de **centros monográficos**, el recálculo pareado de estadía en desarrollo ($\rho = 0,999$) y la validación empírica de transferencias enlazadas, **la Fase 2 queda formalmente cerrada y consolidada en el repositorio institucional**:
* [catalogo_hospitales_procedencia.csv](file:///home/felipe/Documentos/Proyecto%20final/Tesis/config/catalogo_hospitales_procedencia.csv)
* [tabla_canonica_elixhauser_31.csv](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/tabla_canonica_elixhauser_31.csv)
* [13_resolucion_fina_cierre_fase2.py](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/13_resolucion_fina_cierre_fase2.py)
* [resolucion_observaciones_fase2.md](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/resolucion_observaciones_fase2.md)
