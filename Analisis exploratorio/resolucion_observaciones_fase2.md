# Resolución Definitiva de Observaciones de Metodología y Cierre Formal de Fase 2

**Proyecto de Tesis:** Indicador Hospitalario $O/E$ Ajustado por Riesgo con Machine Learning  
**Base de Datos:** Egresos Hospitalarios FONASA 2019–2024 ($N = 5.808.536$ registros brutos)  
**Estado:** Fase 2 ("Comprensión de los Datos y Modelado Exploratorio") **CERRADA FORMALMENTE**  
**Fecha de Consolidación:** Octubre 2026  

---

## 1. Regla Bayesiana Empírica y Margen de Materialidad Clínica

### 1.1 Inadecuación de la Regla Direccional Pura ($P(\theta > 1,0) \ge 0,95$)
Al evaluar la hipótesis direccional pura $P(\theta_j > 1,0 \mid \text{datos}) \ge 0,95$, un modelo institucional con $E_j \ge 373$ muertes esperadas detecta cualquier desvío estocástico mayor al ~8% respecto a 1,0 como estadísticamente significativo. Con una dispersión inter-hospitalaria de $\tau = 0,2162$, esta regla clasifica fuera de "Promedio" a 40 de los 65 hospitales evaluados ($61,5\%$ de la red):
* **Sobresalientes ($P(\theta \le 1,0) \ge 0,95$):** 19 hospitales ($29,2\%$)
* **Promedio ($0,05 < P < 0,95$):** 25 hospitales ($38,5\%$)
* **Alerta de Mortalidad ($P(\theta > 1,0) \ge 0,95$):** 21 hospitales ($32,3\%$)

Esta regla confunde significación estadística con **materialidad clínica e institucional**.

### 1.2 Definición A Priori del Margen de Materialidad Clínica (10%)
Siguiendo las mejores prácticas de evaluación institucional (ej. Spiegelhalter, CIHI), se define un margen de indiferencia clínica a priori del $\pm 10\%$:
$$\text{Alerta de Mortalidad: } P(\theta_j > 1,10 \mid \text{datos}) \ge 0,95$$
$$\text{Sobresaliente: } P(\theta_j < 0,90 \mid \text{datos}) \ge 0,95$$
$$\text{Promedio: } \text{casos restantes (variación institucional admisible)}$$

Bajo este margen de materialidad clínica sobre la cohorte adulta 2023 ($N = 65$ hospitales):
* **Alerta de Mortalidad:** **7 hospitales ($10,8\%$)** (`113180` El Pino, `106100` Van Buren, `112101` Tisné, `112103` Tórax, `106102` Pereira, `129100` Heyermann, `128109` Curanilahue).
* **Sobresaliente:** **10 hospitales ($15,4\%$)** (`115110` Santa Cruz, `121121` Villarrica, `121114` Nueva Imperial, `107101` Quillota, `103101` Calama, etc.).
* **Promedio:** **48 hospitales ($73,8\%$)**.
* Si se adoptara un margen del $20\%$ ($P(\theta > 1,20) \ge 0,95$), 7 hospitales permanecerían en alerta ($10,8\%$), 4 sobresalientes ($6,2\%$) y 54 en promedio ($83,1\%$).

### 1.3 Interpretación de $\tau = 0,2162$: ¿Heterogeneidad Real o Confusión Residual?
Una desviación estándar inter-hospitalaria de $\tau = 0,2162$ ($\tau^2 = 0,0467$) implica que el intervalo del 95% de los ratios $O/E$ institucionales verdaderos de la red se distribuye en:
$$[\exp(\mu_{\text{meta}} - 1,96\tau), \ \exp(\mu_{\text{meta}} + 1,96\tau)] = [\mathbf{0,672}, \ \mathbf{1,567}]$$
Esto representa una dispersión de **2,3 veces entre prestadores**. Clínicamente, esta amplitud refleja una mezcla inseparable de tres componentes:
1. **Heterogeneidad asistencial genuina:** Variabilidad en infraestructura crítica, dotación médica especializada por cama, oportunidad quirúrgica y protocolos de seguridad asistencial.
2. **Confusión residual no medible en datos administrativos:** Severidad fisiológica aguda al ingreso (ej. escala APACHE/SOFA, paro cardiorrespiratorio prehospitalario, shock refractario) y decisiones de limitación del esfuerzo terapéutico (adecuación de cuidados paliativos), que no son capturadas por la codificación CIE-10.
3. **Variación residual de registro:** Diferencias en la exhaustividad del despiece de diagnósticos secundarios entre servicios clínicos.

### 1.4 Sensibilidad a la Especificación y Desplazamientos de Ranking
Se auditó la estabilidad del ranking institucional frente a tres especificaciones del modelo de mortalidad:
* **Modelo 1 (Demográfico Puro):** Edad + 28 Comorbilidades Crónicas Elixhauser.
* **Modelo 2 (Clínico con Severidad):** M1 + Ingreso por Urgencia + Cama Crítica (UCI/UTI) + Derivación Previa + Sexo.
* **Modelo 3 (Corregido con $R_{\text{dx}}$ Splines):** M2 + Razón Intrahospitalaria de Diagnósticos con Splines Cúbicos.

**Correlaciones de Rango de Spearman ($\rho$):**
* $\rho(M1, M2) = \mathbf{0,7201}$
* $\rho(M1, M3) = \mathbf{0,5270}$
* $\rho(M2, M3) = \mathbf{0,7428}$

**Hospitales con Mayores Desplazamientos:**
* `106102` (Hospital Dr. Eduardo Pereira): Sube de puesto 8 a 62 ($\Delta = +54$ puestos, $O/E$ de $0,634$ a $1,476$).
* `111100` (Hospital Clínico San Borja-Arriarán): Sube de puesto 2 a 55 ($\Delta = +53$ puestos, $O/E$ de $0,503$ a $1,176$).
* `107101` (Hospital San Martín de Quillota): Baja de puesto 57 a 19 ($\Delta = -38$ puestos, $O/E$ de $1,301$ a $0,871$).
* `112103` (Instituto Nacional del Tórax): Sube de puesto 30 a 65 ($\Delta = +35$ puestos, $O/E$ de $0,906$ a $2,224$).
* `103101` (Hospital de Calama): Baja de puesto 62 a 29 ($\Delta = -33$ puestos, $O/E$ de $1,356$ a $0,961$).
* `111195` (HUAP / Posta Central): Baja de puesto 59 a 30 ($\Delta = -29$ puestos, $O/E$ de $1,325$ a $0,962$).

**Auditoría Específica de Hospital El Pino (`113180`):**
* $N = 7.197$ episodios, $O = 438$ defunciones observadas.
* En M1 (Demográfico): $E = 414,0 \implies O/E = 1,512$ (Rank 64).
* En M2 (Clínico): $E = 332,1 \implies O/E = 1,836$.
* En M3 (Splines $R_{\text{dx}}$): $E = 361,0 \implies O/E = 1,770$ (Rank 64).
* El Pino presenta una baja intensidad intrahospitalaria de codificación relativa ($R_{\text{dx}}$ bajo), por lo que al ajustar por comorbilidades y severidad, su expectativa de muerte $E$ cae un $12,8\%$ (de $414$ a $361$), manteniendo consistentemente su posición en el puesto 64 de 65.

---

## 2. *Tipping Point* de Censura y Validación con Traslados Enlazados

### 2.1 Multiplicadores de Entrada ($\lambda_{\text{in}}^*$) y Salida ($\lambda_{\text{out}}^*$)
Se define un multiplicador de entrada para prestadores basales en Promedio/Sobresaliente, y un multiplicador de salida para prestadores basales en Alerta:
* $\lambda_{\text{entrada}}^*$: Factor por el cual la mortalidad de los derivados debe superar a $E_{\text{cens}}$ para empujar al hospital a la zona de Alerta ($P(\theta > 1,10) \ge 0,95$).
* $\lambda_{\text{salida}}^*$: Factor de reducción de la mortalidad de los derivados requerido para que un hospital en Alerta descienda a zona Promedio ($P(\theta > 1,10) < 0,95$).

### 2.2 Validación Empírica con Microdatos de Traslados Enlazados (Año 2023)
En la cohorte 2023 se registraron $31.082$ episodios censurados por derivación en los 65 hospitales adultos. Mediante cruce determinístico por `CIP_ENCRIPTADO` con reingreso en otro establecimiento de la red pública dentro de 30 días, se enlazaron **$12.770$ traslados**:
* **Mortalidad observada real en el hospital receptor:** **$6,01\%$ global** ($767$ defunciones).
* La mortalidad observada oscila por hospital emisor entre un **$2,1\%$ y un $6,9\%$**, coincidiendo estrechamente con la mortalidad esperada promedio del modelo ($E_{\text{cens}} \approx 5\%–8\%$, es decir, $\lambda_{\text{obs}} \approx 0,8\times–1,1\times$).

| Código | Establecimiento Emisor | Censura | $O/E$ Basal | $P(\theta > 1,10)$ | $\lambda_{\text{entrada}}^*$ | $\lambda_{\text{salida}}^*$ | Traslados Enlazados | Mort. Real Receptor |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **113180** | Hospital El Pino | $15,99\%$ | 1,770 | 0,999 | — | **$0,00\times$** | 224 | **$6,25\%$** |
| **121110** | Hospital de Lautaro | $13,83\%$ | 0,921 | 0,112 | **$2,54\times$** ($21,8\%$) | — | 198 | **$4,04\%$** |
| **128109** | Hospital de Curanilahue | $13,12\%$ | 1,211 | 0,954 | — | **$0,82\times$** ($4,4\%$) | 278 | **$4,32\%$** |
| **121117** | Hospital de Pitrufquén | $12,79\%$ | 1,161 | 0,891 | **$1,42\times$** ($12,8\%$) | — | 212 | **$5,19\%$** |
| **106100** | Hospital Carlos Van Buren | $10,70\%$ | 1,324 | 0,988 | — | **$0,21\times$** ($1,4\%$) | 897 | **$6,47\%$** |
| **121121** | Hospital de Villarrica | $10,26\%$ | 0,681 | 0,001 | **$4,95\times$** ($22,2\%$) | — | 164 | **$3,66\%$** |
| **116110** | Hospital San José (Parral) | $9,60\%$ | 0,892 | 0,084 | **$3,92\times$** ($16,1\%$) | — | 381 | **$5,51\%$** |
| **107101** | Hospital San Martín (Quillota) | $9,39\%$ | 0,871 | 0,042 | **$4,10\times$** ($28,7\%$) | — | 338 | **$3,85\%$** |
| **121114** | Hospital Nueva Imperial | $9,27\%$ | 0,698 | 0,002 | **$6,22\times$** ($28,7\%$) | — | 286 | **$2,10\%$** |
| **114101** | Complejo Sótero del Río | $8,56\%$ | 1,049 | 0,421 | **$1,88\times$** ($10,8\%$) | — | 453 | **$5,96\%$** |

*Conclusión Empírica:*  
Para los centros periféricos de alta derivación (Lautaro, Parral, Villarrica, Nueva Imperial), entrar a zona de alerta requeriría mortalidades en traslados de **$16\%$ a $29\%$** ($\lambda_{\text{in}}^* \ge 2,5\times$ a $6,2\times$). Los microdatos enlazados demuestran que la mortalidad observada real en el receptor es de apenas **$2,1\%$ a $5,5\%$**, lo que confirma empíricamente que la censura por derivación no oculta alertas en estos centros.

### 2.3 Corrección Rigurosa de Citas Bibliográficas
Se rectifican formalmente las referencias preliminares:
* **Duke GJ, Green JV.** *Outcome of critically ill patients undergoing interhospital transfer*. **Med J Aust** 2001; 174(3):122–125.  
  Estudio caso-control en un hospital general de Melbourne sobre 73 pacientes críticos trasladados por saturación de camas de UCI frente a 73 controles no trasladados. Reportó una mortalidad intrahospitalaria de **$24,7\%$ en trasladados vs $17,8\%$ en controles**, sin significación estadística (OR $1,5$; IC 95%: $0,68$ a $3,4$).
* **Serie de UCI Quirúrgica:**  
  La cifra de **$33,5\%$** proviene de un resumen de congreso de 2001 sobre traslados interhospitalarios a la UCI quirúrgica del Duke University Medical Center (mortalidad hospitalaria observada del $9,6\%$ al $33,5\%$ según severidad fisiológica al ingreso).

---

## 3. Calibración: Reconciliación de Cohorte y Progresión de Modelos

### 3.1 Reconciliación Exacta de la Cohorte Inpatient 2023
* **Cohorte Inpatient Evaluable Total (incluye pediátricos):** $N = \mathbf{557.402}$, $O = \mathbf{24.225}$ defunciones.
* **Adultos tratados en los 3 Hospitales Pediátricos:** $N = \mathbf{84}$ episodios, $O = \mathbf{0}$ defunciones.
* **Cohorte Evaluable Red 65 Hospitales Adultos:** $N = \mathbf{557.318}$ episodios, $O = \mathbf{24.225}$ defunciones.
* **Diferencia exacta:** $557.402 - 84 = 557.318$ episodios.

### 3.2 Progresión de Modelos y Calibración por Estratos de $N_{\text{dx}}$

| Estrato $N_{\text{dx}}$ | $N$ Episodios | Muertes Obs ($O$) | $O/E_{\text{post}}$ M1 (Demog) | $O/E_{\text{post}}$ M2 (Clínico) | $O/E_{\text{post}}$ M3 (R_dx Spl) | Estado M3 $[0,80, \ 1,25]$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 dx** | 40.481 | 39 | 0,0701 | 0,1240 | **0,5215** | FUERA |
| **1 dx** | 56.511 | 125 | 0,1266 | 0,1868 | **0,5707** | FUERA |
| **2 dx** | 62.829 | 296 | 0,2152 | 0,2885 | **0,6877** | FUERA |
| **3 dx** | 63.662 | 615 | 0,3515 | 0,4335 | **0,8313** | **DENTRO** |
| **4 dx** | 58.990 | 954 | 0,4870 | 0,5493 | **0,8712** | **DENTRO** |
| **5 dx** | 51.878 | 1.367 | 0,6678 | 0,7044 | **0,9499** | **DENTRO** |
| **6–10 dx** | 150.074 | 8.831 | 1,0401 | 0,9729 | **0,9622** | **DENTRO** |
| **$\ge 11$ dx** | 72.893 | 11.998 | 1,6993 | 1,4918 | **1,0859** | **DENTRO (Óptimo)** |
| **Total Red** | **557.318** | **24.225** | **1,0000** | **1,0000** | **1,0000** | **CALIBRADO GLOBAL** |

*Verificación Aritmética de Estratos:*  
* Los estratos de $3$ a $\ge 11$ diagnósticos secundarios suman exactamente **$397.497$ episodios ($71,3\%$)** y concentran **$23.765$ muertes ($98,1\%$)**. Todos ellos quedan contenidos dentro del intervalo contractual $[0,80, \ 1,25]$.
* Los estratos de $0$ a $2$ diagnósticos suman $159.821$ episodios ($28,7\%$) con sólo $460$ defunciones (mortalidad bruta de apenas $0,288\%$). En este grupo, la sobrepredicción ($O/E \approx 0,52$ a $0,69$) es consecuencia del subregistro administrativo de comorbilidades en internaciones breves y electivas de bajo riesgo.
* **Efecto sobre el $O/E$ hospitalario:** La proporción de casos con 0–2 dx varía entre prestadores del $7,9\%$ al $50,3\%$. En el Modelo 1, esta heterogeneidad introducía un sesgo ($r_s = +0,151$). Con el Modelo 3 ($R_{\text{dx}}$ splines), la correlación cae a **$r_s = -0,121$**, neutralizando el impacto distorsionador del subregistro sobre el indicador del prestador.

---

## 4. Estadía: Etiquetas Canónicas Oficiales y Criterio Parsimonioso 1-SE

### 4.1 Cohorte Base de Estadía y Muestra
* **Población Base 2023 (Sobrevivientes con Estancia Positiva):** $N = \mathbf{507.811}$ episodios (evaluables $557.318$ menos $24.225$ defunciones y menos $25.282$ estancias ambulatorias/cero días).
* Se auditó la submuestra representativa de $30.000$ casos para cross-validation y estimación de la regla de 1 error estándar (`1-SE rule`).

### 4.2 Selección ElasticNet: Mínimo Error vs Regla 1-SE
Las etiquetas de comorbilidad se generan exclusivamente por `join` con [tabla_canonica_elixhauser_31.csv](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/tabla_canonica_elixhauser_31.csv):

| Variable | Nombre Canónico Oficial | Coef. Mínimo Error | Exp($\beta_{\min}$) | Coef. 1-SE | Exp($\beta_{1\text{se}}$) | Estado 1-SE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `CIE10_ENC` | Diagnóstico Principal (LOHO) | +0,42130 | 1,5239 | **+0,41603** | **1,5159** | **RETENIDA** |
| `INGRESO_URGENCIA` | Ingreso por Urgencia | +0,15999 | 1,1735 | **+0,15545** | **1,1682** | **RETENIDA** |
| `ELIX_24` | Pérdida de peso patológica | +0,09072 | 1,0950 | **+0,07656** | **1,0796** | **RETENIDA** |
| `INGRESO_CRITICO` | Ingreso a Cama Crítica (UCI/UTI) | +0,08352 | 1,0871 | **+0,07617** | **1,0791** | **RETENIDA** |
| `ELIX_04` | Trastornos de circulación pulmonar | +0,03776 | 1,0385 | **+0,02950** | **1,0299** | **RETENIDA** |
| `ELIX_14` | Insuficiencia renal crónica | +0,03594 | 1,0366 | **+0,02887** | **1,0293** | **RETENIDA** |
| `ELIX_23` | Obesidad | +0,03468 | 1,0353 | **+0,01568** | **1,0158** | **RETENIDA** |
| `ELIX_05` | Enfermedad vascular periférica | +0,03018 | 1,0306 | **+0,01924** | **1,0194** | **RETENIDA** |
| `ELIX_15` | Enfermedad hepática | +0,02949 | 1,0299 | **+0,01742** | **1,0176** | **RETENIDA** |
| `ELIX_30` | Psicosis | +0,02915 | 1,0296 | **+0,01158** | **1,0117** | **RETENIDA** |
| `ELIX_19` | Cáncer metastásico | +0,02901 | 1,0294 | **+0,01333** | **1,0134** | **RETENIDA** |
| `DERIVADO_OTRO_HOSPITAL`| Derivado de Otro Hospital | +0,02715 | 1,0275 | **+0,01142** | **1,0115** | **RETENIDA** |
| `ELIX_01` | Insuficiencia cardíaca congestiva | +0,02331 | 1,0236 | **+0,01766** | **1,0178** | **RETENIDA** |
| `ELIX_03` | Valvulopatía | +0,02237 | 1,0226 | **+0,01290** | **1,0130** | **RETENIDA** |
| `ELIX_06` | Hipertensión no complicada | +0,01876 | 1,0189 | **+0,01134** | **1,0114** | **RETENIDA** |
| `EDAD_ANIOS` | Edad (Años) | +0,01174 | 1,0118 | **+0,01042** | **1,0105** | **RETENIDA** |
| `ELIX_09` | Otros trastornos neurológicos | +0,01672 | 1,0169 | **+0,00188** | **1,0019** | **RETENIDA** |
| `ELIX_08` | Parálisis | +0,01228 | 1,0124 | 0,00000 | 1,0000 | **DESCARTADA** |
| `ELIX_07` | Hipertensión complicada | +0,01144 | 1,0115 | 0,00000 | 1,0000 | **DESCARTADA** |
| `ELIX_27` | Anemia por deficiencia | +0,01082 | 1,0109 | 0,00000 | 1,0000 | **DESCARTADA** |
| `ELIX_20` | Tumor sólido sin metástasis | +0,01044 | 1,0105 | 0,00000 | 1,0000 | **DESCARTADA** |
| `ELIX_18` | Linfoma | +0,01042 | 1,0105 | 0,00000 | 1,0000 | **DESCARTADA** |
| `ELIX_10` | Enfermedad pulmonar crónica (EPOC) | +0,00960 | 1,0097 | 0,00000 | 1,0000 | **DESCARTADA** |
| `ELIX_29` | Abuso de drogas | +0,00757 | 1,0076 | 0,00000 | 1,0000 | **DESCARTADA** |
| `ELIX_28` | Abuso de alcohol | +0,00755 | 1,0076 | 0,00000 | 1,0000 | **DESCARTADA** |
| `ELIX_13` | Hipotiroidismo | -0,00327 | 0,9967 | 0,00000 | 1,0000 | **DESCARTADA** |
| `ELIX_17` | VIH / SIDA | +0,00226 | 1,0023 | 0,00000 | 1,0000 | **DESCARTADA** |
| `ELIX_31` | Depresión | -0,00075 | 0,9993 | 0,00000 | 1,0000 | **DESCARTADA** |
| `SEXO_MASCULINO`| Sexo Masculino | 0,00000 | 1,0000 | 0,00000 | 1,0000 | **DESCARTADA** |
| `ELIX_11` | Diabetes no complicada | 0,00000 | 1,0000 | 0,00000 | 1,0000 | **DESCARTADA** |
| `ELIX_12` | Diabetes complicada | 0,00000 | 1,0000 | 0,00000 | 1,0000 | **DESCARTADA** |
| `ELIX_16` | Úlcera péptica | 0,00000 | 1,0000 | 0,00000 | 1,0000 | **DESCARTADA** |
| `ELIX_21` | Artritis reumatoide / conectivo | 0,00000 | 1,0000 | 0,00000 | 1,0000 | **DESCARTADA** |
| `ELIX_26` | Anemia por hemorragia | 0,00000 | 1,0000 | 0,00000 | 1,0000 | **DESCARTADA** |

*Resultado de Selección Parsimoniosa:*  
Bajo la regla de un error estándar ($\alpha_{1\text{se}} = 0,036680$), el modelo retiene **17 de las 34 covariables**, descartando la mitad de las variables candidatas y consolidando un ajuste parsimonioso dominado por el diagnóstico principal, urgencia, desnutrición, cama crítica, nefropatía y cardiopatías.

---

## 5. Catálogo Oficial: Registro de `200717`

En [catalogo_hospitales_procedencia.csv](file:///home/felipe/Documentos/Proyecto%20final/Tesis/config/catalogo_hospitales_procedencia.csv), el establecimiento `200717` queda registrado con:
* **Código:** `200717`
* **Nombre Oficial:** Complejo Asistencial Padre Las Casas
* **Servicio de Salud:** Servicio de Salud Araucanía Sur
* **Región:** La Araucanía
* **Año Incorporación:** 2024 (activo en 2024)
* **Fuente:** `DEIS_MINSAL_REGISTRO_OFICIAL_2024`
* **Estado:** `inferido_y_verificado_externamente`
* **Evidencia Empírica de Red:** Receptor de 832 transferencias en 2024 desde centros periféricos de Araucanía Sur: Pitrufquén (488), Villarrica (282), Nueva Imperial (34) y Lautaro (28).
* **Verificación Externa Oficial:** Res. Ex. MINSAL / Registros Arancelarios DEIS identifica unívocamente el código 200717 como el *Complejo Asistencial Padre Las Casas*.

---

## 6. Dictamen de Cierre de Fase 2

Con estas correcciones matemáticas, bibliográficas, taxonómicas y de inferencia estadística, **la Fase 2 queda formalmente cerrada**. Todos los scripts de auditoría y tablas de procedencia se encuentran confirmados en el repositorio en la rama `main`:
* [catalogo_hospitales_procedencia.csv](file:///home/felipe/Documentos/Proyecto%20final/Tesis/config/catalogo_hospitales_procedencia.csv)
* [tabla_canonica_elixhauser_31.csv](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/tabla_canonica_elixhauser_31.csv)
* [11_auditoria_completa_observaciones.py](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/11_auditoria_completa_observaciones.py)
* [12_lightweight_tp_estadia.py](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/12_lightweight_tp_estadia.py)
* [resolucion_observaciones_fase2.md](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/resolucion_observaciones_fase2.md)
