import os
from pathlib import Path

content = """# Resolución Metodológica y Cierre Definitivo de la Fase 2: Especificación Primaria M2+MDC, Calibración Previa al Ingreso, Verificación LOHO y Sensibilidad a la Censura por Traslado

**Fecha de Cierre:** Octubre 2026  
**Repositorio:** `Tesis` (Rama `main`)  
**Cohorte Validada:** Egresos Inpatient FONASA 2019–2024 ($N = 5.808.536$ registros brutos consolidados en Plata), Cohorte de Desarrollo 2020–2022 ($N = 1.453.141$; $N_{\\text{stay}} = 1.303.718$; año 2019 con $N = 1.151.475$ empleado exclusivamente como ventana retrospectiva de búsqueda de comorbilidades a 1 año), Cohorte Evaluable de Validación 2023 ($N = 557.318$, $O = 24.225$ defunciones intrahospitalarias, $K = 65$ hospitales de la red pública, $K = 62$ agudos generales).

> **Nota de Conciliación de Cohorte y Denominadores:**
> 1. **Total Bruto Silver ($N = 5.808.536$):** La cifra de $5.820.730$ citada en borradores previos correspondía a una extracción preliminar de staging antes de la desduplicación unificada y depuración de registros sin bytes de egreso. El universo formal de egresos cerrados 2019–2024 es exactamente de $5.808.536$ episodios (2019: 1.151.475; 2020: 781.912; 2021: 816.909; 2022: 932.840; 2023: 1.039.587; 2024: 1.085.813).
> 2. **Cohorte de Desarrollo 2020–2022 ($N = 1.453.141$):** El período de modelado abarca estrictamente los egresos 2020 a 2022. La diferencia de 255 episodios respecto de los $1.453.396$ preliminares obedece a la exclusión explícita de pacientes adultos ($\ge 18$ años) hospitalizados en establecimientos pediátricos monográficos (Hospital Exequiel González Cortés, Hospital Dr. Roberto del Río y Hospital Dr. Luis Calvo Mackenna), consistente con la exclusión de 84 episodios en 2023 ($557.402 \\to 557.318$).

---

## 1. La Especificación Primaria Oficial: M2 + Diagnóstico Principal (M2+MDC)

### 1.1 Justificación Teórica y Calibración Preingreso Multidimensional
En estricta coherencia con el criterio de calibración por estratos previos al ingreso, el modelo clínico base sin diagnóstico principal (M2) presentaba una falla estructural en la dimensión de categoría diagnóstica mayor (MDC): un $O/E$ de **2,28** en patología respiratoria (MDC 4) y de **0,35** en patología musculoesquelética (MDC 8). Esta descalibración afectaba severamente la equidad de la evaluación asistencial: cualquier centro con mayor concentración de patología respiratoria aguda recibía una penalización espuria atribuible al perfil clínico de sus ingresos y no a su efectividad hospitalaria.

Al incorporar el diagnóstico principal agrupado (**Modelo M2+MDC**), los seis MDC principales se calibran dentro del margen de tolerancia $[0,93; 1,05]$, cumpliendo el estándar de calibración previa al ingreso.

> **Definición Rigurosa del Agrupamiento MDC:**  
> A fin de garantizar la **estricta ausencia de fuga de información (data leakage)**, el agrupamiento en Categorías Diagnósticas Mayores (MDC) del modelo M2+MDC se obtiene **directa y determinísticamente del código CIE-10 del diagnóstico principal consignado al ingreso (`DIAG1`)**, mapeado según los capítulos de la clasificación oficial de MINSAL. En ningún caso se emplean las salidas del agrupador IR-GRD (`IR_29301_COD_GRD` ni `MDC` del agrupador), las cuales quedan restringidas exclusivamente a los filtros de definición de cohorte (exclusión de neonatos EX01–EX03) y marcadas como prohibidas para el entrenamiento de predictores.

| Modelo | Definición de Covariables | $\\tau$ Agudos ($K=62$) | $\\mu$ Meta | Calibración MDC $[0,80; 1,25]$ | Rol Metodológico |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **M2+MDC** | Demografía + Vía Ingreso + 28 Elixhauser Crónicos + MDC (CIE-10 Ingreso) | **0,1948** | **-0,0359** | **CUMPLE** (MDC 0,93 a 1,05) | **ESPECIFICACIÓN PRIMARIA** |
| **M2 Base** | Demografía + Vía Ingreso + 28 Elixhauser Crónicos (sin MDC) | 0,2077 | -0,0267 | Falla (MDC 4: 2,28; MDC 8: 0,35) | **SENSIBILIDAD 1** (Sin Dx Principal) |
| **M3-Total** | M2 Base + Splines cúbicos de $R_{\\text{dx}}$ Total al Alta | 0,2041 | -0,0347 | Absorbe complicaciones de estadía | **SENSIBILIDAD 2** (Codificación Aguda) |

*Estimación de Hiperparámetros $(\\tau, \\mu)$:*  
Los hiperparámetros de efectos aleatorios se estimaron mediante el estimador clásico no iterativo de **DerSimonian-Laird** sobre los logaritmos de la razón estandarizada de mortalidad ($\ln(O/E)$), utilizando las varianzas de muestreo de Byar. El cambio en la media global $\\mu$ (que en versiones preliminares marcaba $+0,0224$ y pasa a $-0,0359$) responde a la **restricción estricta de la muestra de estimación a los $K=62$ hospitales generales de agudos de la red pública**, excluyendo los centros monográficos de alta mortalidad especializada (cuyo volumen sesgaba la media global por encima de cero).

### 1.2 Impacto sobre Centros Especializados: El Caso del Instituto Nacional del Tórax
El comportamiento del **Instituto Nacional del Tórax (`112103`)** ilustra el efecto del ajuste por diagnóstico principal:
* Sus egresos presentan un **92,7%** de concentración en patología respiratoria y cardiovascular (MDC 04 Respiratorio: 84,1%; MDC 05 Circulatorio: 8,6%).
* En M2 Base (sin MDC), el Tórax registraba $O/E = \\mathbf{1,968}$ ($P = 1,000$).
* En **M2+MDC (Primaria)**, al reconocer su perfil respiratorio, su exceso se reduce a $O/E = \\mathbf{1,429}$ ($P = 0,997$). El ajuste **reduce sustancialmente el sesgo** por mezcla de especialidad (más de medio punto de sobre-mortalidad aparente), manteniendo al establecimiento en alerta debido a la gravedad intrínseca no capturada.
* En M3-Total, debido a su baja codificación secundaria en relación a su estancia, su $O/E$ se distorsiona hasta $2,226$.

---

## 2. Taxonomía y Reconciliación de Alertas Asistenciales

### 2.1 Definición Rigurosa de Términos
Se establecen formalmente las tres categorías taxonómicas de alerta:
1. **Alerta Primaria:** Todo establecimiento de salud que en la **Especificación Primaria Oficial (M2+MDC)**, evaluado con los parámetros de la red de agudos generales ($\\tau = 0,1948, \\mu = -0,0359$), presenta una probabilidad posterior de exceso superior al margen de materialidad del 10%:
   $$P(\\theta_j > 1,10 \\mid \\text{M2+MDC}) \\ge 0,95$$
2. **Alerta Robusta:** Todo establecimiento que es **Alerta Primaria** y que, de forma concurrente, **persiste como alerta en el modelo de sensibilidad M3-Total** ($P \\ge 0,95$ en ambos modelos).
3. **Alerta Dependiente de Codificación Aguda:** Establecimiento que no califica como alerta en la especificación primaria preingreso, pero que alerta únicamente en M3-Total debido al ajuste por diagnósticos secundarios codificados al alta.

### 2.2 Lista Consolidada de Alertas en Hospitales Agudos Generales ($K=62$)
Bajo la especificación primaria M2+MDC, se identifican exactamente **10 Alertas Primarias** (y 52 hospitales en Desempeño Promedio):

| Código | Establecimiento | $O$ Defunciones | $O/E$ (M2+MDC) | $P_{\\text{Prim}}$ | $O/E$ (M2 Base) | $P_{\\text{M2}}$ | $O/E$ (M3-Tot) | $P_{\\text{M3}}$ | Categoría de Alerta |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `113180` | **Hospital El Pino** | 438 | **1,646** | **1,000** | 1,836 | 1,000 | 1,772 | 1,000 | **ALERTA ROBUSTA** |
| `111100` | **Hospital Clínico San Borja-Arriarán** | 145 | **1,432** | **0,996** | 1,515 | 1,000 | 1,177 | 0,697 | **Alerta Primaria** |
| `106103` | **Hospital Claudio Vicuña (San Antonio)** | 320 | **1,381** | **1,000** | 1,595 | 1,000 | 1,634 | 1,000 | **ALERTA ROBUSTA** |
| `113100` | **Hospital Barros Luco Trudeau** | 972 | **1,370** | **1,000** | 1,288 | 1,000 | 1,154 | 0,914 | **Alerta Primaria** |
| `107101` | **Hospital San Martín (Quillota)** | 285 | **1,365** | **1,000** | 1,223 | 0,935 | 0,872 | 0,000 | **Alerta Primaria** |
| `106100` | **Hospital Carlos Van Buren** | 602 | **1,334** | **1,000** | 1,293 | 1,000 | 1,436 | 1,000 | **ALERTA ROBUSTA** |
| `103100` | **Hospital Dr. Leonardo Guzmán (Antofagasta)**| 663 | **1,332** | **1,000** | 1,384 | 1,000 | 1,165 | 0,908 | **Alerta Primaria** |
| `121109` | **Hospital Dr. Hernán Henríquez (Temuco)** | 580 | **1,285** | **1,000** | 1,241 | 0,997 | 1,165 | 0,888 | **Alerta Primaria** |
| `112101` | **Hospital Dr. Luis Tisné B.** | 502 | **1,233** | **0,990** | 1,313 | 1,000 | 1,316 | 1,000 | **ALERTA ROBUSTA** |
| `109100` | **Complejo Hospitalario San José** | 815 | **1,192** | **0,984** | 1,188 | 0,981 | 1,125 | 0,700 | **Alerta Primaria** |
| `112103` | *Instituto Nacional del Tórax (Monográfico)*| 157 | **1,429** | **0,997** | 1,968 | 1,000 | 2,226 | 1,000 | *Alerta Robusta Monog.* |

### 2.3 Casos de Divergencia Extrema: Mecanismos Clínicos y Codificación
El análisis comparativo entre M2+MDC y M3-Total revela discrepancias extremas en dos centros asistenciales:

1. **Hospital San Martín de Quillota (`107101`):**  
   * **Comportamiento:** En M2+MDC registra $O/E = \\mathbf{1,365}$ ($P = 1,000$, Alerta Primaria), mientras que en M3-Total cae a $O/E = \\mathbf{0,872}$ ($P = 0,000$, Favorable), una divergencia de **$-0,49$**, la mayor de la red. En M2 Base registraba $O/E = 1,223$ ($P = 0,935$), activándose la alerta únicamente al agregar MDC.
   * **Mecanismo Operacional y Codificación:** Durante 2022–2023 el establecimiento completó el traslado físico y operacional hacia el nuevo **Hospital Biprovincial Quillota-Petorca**. Este proceso generó una disrupción severa en los registros clínicos, reflejada en una profunda infra-codificación secundaria ($R_{\\text{dx}} = 0,65$; promedio de solo 1,9 diagnósticos secundarios por egreso frente a una media de 3,0 en la red). En M3-Total, la penalización no lineal por splines interpreta erróneamente esta baja codificación inflando artificialmente las defunciones esperadas a $E_{\\text{M3}} = 326,8$ (frente a $O = 285$), enmascarando su exceso de mortalidad real. Por el contrario, el modelo primario M2+MDC, al basarse en comorbilidades crónicas y clasificar por MDC de ingreso, asigna $E_{\\text{M2+MDC}} = 208,7$, detectando un exceso real de $+76,3$ defunciones.

2. **Hospital Dr. Eduardo Pereira de Valparaíso (`106102`):**  
   * **Comportamiento:** En M2+MDC registra $O/E = \\mathbf{0,838}$ ($P = 0,000$, Desempeño Favorable), en M2 Base $O/E = 0,979$ ($P = 0,063$), pero en M3-Total salta a $O/E = \\mathbf{1,478^*}$ ($P = 0,999$, Alerta), una brecha de **$+0,64$**.
   * **Mecanismo Operacional y Codificación:** El Hospital Pereira es un centro quirúrgico torácico y broncopulmonar de referencia. Sus pacientes cursan estancias postquirúrgicas prolongadas donde se documentan exhaustivamente complicaciones intrahospitalarias agudas (atelectasias, derrames, neumonías nosocomiales e insuficiencias respiratorias agudas), alcanzando una tasa de codificación de $R_{\\text{dx}} = 1,42$. En M3-Total, el modelo sobre-ajusta por estas complicaciones colapsando las muertes esperadas a $E_{\\text{M3}} = 100,1$ (frente a $O = 148$), creando una alerta artificial. Al excluir la codificación aguda en M2+MDC, el riesgo esperado al ingreso es $E = 176,5$, confirmando que la mortalidad observada es inferior a la esperada por la condición basal de sus pacientes.

3. **Hospital de San Camilo (`108100`):**  
   * En M2 Base alertaba ($O/E = 1,292, P = 0,998$), pero en **M2+MDC (Primaria)** su razón desciende a $O/E = \\mathbf{1,174}$ con $P = \\mathbf{0,848} < 0,95$, quedando formalmente en **Desempeño Promedio**. En M3-Total registra $O/E = 1,197, P = 0,920 < 0,95$. El ajuste por diagnóstico principal explica adecuadamente su mezcla de casos.

---

## 3. Calibración, Descalce de $N_{\\text{dx}}$ y Verificaciones Técnicas

### 3.1 Criterio de Calibración Preingreso y Ausencia de Asociación con $N_{\\text{dx}}$
La especificación primaria M2+MDC satisface los márgenes pre-especificados de calibración previa al ingreso ($[0,80; 1,25]$ para todo estrato con $E \\ge 100$):
* **Edad:** $<65$ años ($1,061$), $65-74$ años ($0,971$), $\ge 75$ años ($0,977$).
* **Carga Crónica:** 0 comorb ($0,804$), 1 comorb ($0,989$), 2 comorb ($1,085$), $\ge 3$ comorb ($1,006$). Se destaca que el estrato de 0 comorbilidades se sitúa en $0,804$, en el límite inferior de tolerancia admisible.
* **Vía Ingreso:** Urgencia ($1,000$), Programada ($0,991$).
* **MDC Principal (CIE-10 de Ingreso):** MDC 6 Digestivo ($1,047$), MDC 8 Musculoesquelético ($0,932$), MDC 5 Circulatorio ($1,004$), MDC 4 Respiratorio ($1,008$), MDC 1 Nervioso ($0,979$), MDC 11 Riñón/Urinario ($0,982$).

*Asociación Empírica con $N_{\\text{dx}}$ por Hospital:*  
Dando respuesta directa al comité, se evaluó si el desempeño ajustado por M2+MDC está sesgado por la intensidad de codificación secundaria de los hospitales:
* Correlación de Spearman entre $O/E_{\\text{M2+MDC}}$ y $N_{\\text{dx}}$ medio del hospital: **$\\rho = -0,133$ ($p = 0,303$, no significativo)**.
* Correlación de Spearman entre $O/E_{\\text{M2+MDC}}$ y porcentaje de pacientes con $\ge 11$ diagnósticos: **$\\rho = -0,080$ ($p = 0,537$, no significativo)**.

**Conclusión:** En la red hospitalaria pública no se detecta correlación entre el desempeño primario y la profundidad de codificación secundaria. La verificación definitiva de calibración queda reservada para la validación prospectiva externa en la cohorte 2024.

### 3.2 Verificación Real de LOHO y Concentración Diagnóstica
Se ratifican las métricas out-of-sample del codificador Leave-One-Hospital-Out (LOHO) en la cohorte 2023 de estadía:
* **Concentración:** 144 categorías CIE-10 (9,9% de las 1.459 activas) tienen $>50\%$ de sus casos concentrados en un solo hospital.
* **Máxima Discrepancia:** $|\\Delta|_{\\text{máx}} = 0,5079$ log-días (CIE G47 en Puerto Montt; H36 con 96,0% en Los Ángeles, $|\\Delta|=0,4395$; T29 quemaduras con 55,5% en Posta Central, $|\\Delta|=0,4295$).
* **Impacto Institucional:** $\\rho = 0,99948$, $\\text{RMS} = 0,61$ puestos. El Instituto de Neurocirugía Dr. Asenjo (`112104`) presenta la mayor variación de la red ($-0,0294$ en $O/E$, de $0,8033$ a $0,7739$).

### 3.3 Estabilidad por Bootstrap de las Covariables de Estadía
Se congelan los resultados de 100 réplicas bootstrap en desarrollo ($N = 1.303.718$) en el archivo [`scratch/estadia_bootstrap_stability_17.json`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/scratch/estadia_bootstrap_stability_17.json):
* **16 Variables Núcleo al 100%:** Incluyen a `ELIX_03` Valvulopatía (100%) y `ELIX_08` Parálisis (100%).
* **Borde en Puesto 17:** `ELIX_06` HTA no complicada (69,0%), `ELIX_19` Cáncer metastásico (19,0%), `ELIX_10` EPOC (10,0%), `ELIX_07` HTA complicada (2,0%). Todos con coeficientes entre $+0,0014$ y $+0,0038$. El ordenamiento institucional es enteramente insensible a la alternancia en este borde ($\\rho \\ge 0,999$).

### 3.4 Desglose Paramétrico Empírico de Bayes (San Borja y San Camilo)
Se reproduce la inferencia empírica de Bayes empleando para cada modelo sus propios hiperparámetros $(\\tau, \\mu)$:

| Hospital | Especificación | $\\tau$ | $\\mu$ | $O$ | $E$ | $O/E$ | $v = 1/O$ | $B$ | $\\ln(\\theta^*)$ | $\\text{SE}_{\\text{post}}$ | $Z$ | $P(\\theta > 1,10)$ | Clasificación |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **San Borja (`111100`)** | **M2+MDC (Primaria)** | 0,1948 | -0,0359 | 145 | 101,3 | **1,432** | 0,006897 | 0,8463 | $+0,2985$ | 0,0764 | $+2,659$ | **0,9961** | **Alerta Primaria** |
| | M2 Base (Sensibilidad) | 0,2077 | -0,0267 | 145 | 95,7 | 1,515 | 0,006897 | 0,8621 | $+0,3543$ | 0,0771 | $+3,358$ | **0,9996** | Alerta M2 |
| | M3-Total (Sensibilidad)| 0,2041 | -0,0347 | 145 | 123,2 | 1,177 | 0,006897 | 0,8580 | $+0,1347$ | 0,0769 | $+0,512$ | **0,6967** | Promedio |
| **San Camilo (`108100`)**| **M2+MDC (Primaria)** | 0,1948 | -0,0359 | 365 | 310,9 | **1,174** | 0,002740 | 0,9326 | $+0,1473$ | 0,0506 | $+1,029$ | **0,8482** | **Promedio** |
| | M2 Base (Sensibilidad) | 0,2077 | -0,0267 | 365 | 282,6 | 1,292 | 0,002740 | 0,9402 | $+0,2389$ | 0,0508 | $+2,827$ | **0,9977** | Alerta M2 |
| | M3-Total (Sensibilidad)| 0,2041 | -0,0347 | 365 | 304,9 | 1,197 | 0,002740 | 0,9382 | $+0,1673$ | 0,0507 | $+1,419$ | **0,9204** | Promedio |

---

## 4. Enlace Unificado, Cotas de Salida y Sensibilidad a la Censura por Traslado

### 4.1 Criterio Unificado de Enlace Determinístico
Se fija de manera definitiva un único criterio de enlace para auditoría:
> **Criterio Unificado:** Se consideran enlazados todos los episodios de derivación con reingreso hospitalario subsiguiente en cualquier establecimiento de la red pública (silver 2023), identificado mediante `CIP_ENCRIPTADO`, con fecha de ingreso entre 0 y 30 días posteriores al egreso del emisor (`FECHA_INGRESO_REC >= FECHAALTA` y `FECHA_INGRESO_REC <= FECHAALTA + 30 días`).

Conteos oficiales consolidados:
* **Total egresos por traslado inpatient adultos 2023:** **25.555** episodios.
* **Total enlazados:** **11.544** episodios (tasa de enlace global: 45,2%).

### 4.2 Cotas de Salida ($\lambda_{\\text{salida}}$) para las Alertas Primarias
En respuesta a la observación del comité respecto de la cota inferior de mortalidad en pacientes derivados no enlazados que permitiría a un hospital salir de alerta ($P < 0,95$), se calculó el multiplicador umbral $\\lambda_{\\text{salida}}$ para las 10 Alertas Primarias (+ Tórax):

$$O_{\\text{eval}} = O_{\\text{base}} + O_{\\text{link}} + \\lambda_{\\text{salida}} \\cdot E_{\\text{nolink}}, \\quad E_{\\text{eval}} = E_{\\text{base}} + E_{\\text{link}} + E_{\\text{nolink}}$$

| Código | Establecimiento | $O_{\\text{base}}$ | $E_{\\text{base}}$ | $P_{\\text{base}}$ | No Enlaz. | $E_{\\text{nolink}}$ | Cota $\\lambda_{\\text{salida}}$ | ¿Factible Salir de Alerta? | Interpretación Clínica |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `112101` | **Hospital Dr. Luis Tisné B.** | 502 | 407,3 | 0,990 | 699 | 86,6 | **0,98x** | **SÍ (Factible, $\\lambda < 0,98$)** | **Alerta Marginal Sensible:** Si sus 699 derivados no enlazados experimentan una sobrevida estándar o ligeramente favorable ($\\lambda < 0,98\\times$), el establecimiento sale de alerta ($P < 0,95$). |
| `121109` | **Hospital Dr. Hernán Henríquez (Temuco)** | 580 | 451,4 | 1,000 | 408 | 36,4 | **0,47x** | Difícil (requiere $\\lambda < 0,47$) | Requiere que los no enlazados fallezcan a menos de la mitad de su riesgo esperado. |
| `107101` | **Hospital San Martín (Quillota)** | 285 | 208,7 | 1,000 | 339 | 26,4 | **0,19x** | Muy Remoto ($\\lambda < 0,19$) | Exige sobrevida casi total en pacientes trasladados graves. |
| `109100` | **Complejo Hospitalario San José** | 815 | 683,5 | 0,984 | 179 | 10,1 | **0,05x** | Casi Imposible ($\\lambda < 0,05$) | Requiere cero defunciones en pacientes derivados. |
| `113180` | **Hospital El Pino** | 438 | 266,0 | 1,000 | 1.120 | 83,2 | **No existe** | **NO (Alerta Inmune)** | Aun suponiendo $\\lambda = 0$ (cero muertes en 1.120 trasladados), $P_{\\text{min}} = 1,000$. |
| `111100` | **Hospital Clínico San Borja-Arriarán** | 145 | 101,3 | 0,996 | 34 | 1,3 | **No existe** | **NO (Alerta Inmune)** | Con $\\lambda = 0$, $O/E = 1,398$ y $P_{\\text{min}} = 0,995 \\ge 0,95$. Inmune a dilución. |
| `106103` | **Hospital Claudio Vicuña (San Antonio)** | 320 | 231,8 | 1,000 | 102 | 8,4 | **No existe** | **NO (Alerta Inmune)** | Con $\\lambda = 0$, $P_{\\text{min}} = 1,000$. |
| `113100` | **Hospital Barros Luco Trudeau** | 972 | 709,7 | 1,000 | 409 | 41,8 | **No existe** | **NO (Alerta Inmune)** | Con $\\lambda = 0$, $P_{\\text{min}} = 1,000$. |
| `106100` | **Hospital Carlos Van Buren** | 602 | 451,4 | 1,000 | 349 | 27,9 | **No existe** | **NO (Alerta Inmune)** | Con $\\lambda = 0$, $P_{\\text{min}} = 1,000$. |
| `103100` | **Hospital Dr. Leonardo Guzmán (Antofagasta)**| 663 | 497,8 | 1,000 | 54 | 3,2 | **No existe** | **NO (Alerta Inmune)** | Con $\\lambda = 0$, $P_{\\text{min}} = 1,000$. |
| `112103` | *Instituto Nacional del Tórax (Monográfico)*| 157 | 109,8 | 0,997 | 36 | 2,2 | **No existe** | **NO (Alerta Inmune)** | Con $\\lambda = 0$, $P_{\\text{min}} = 0,996 \\ge 0,95$. Inmune a dilución. |

### 4.3 Sensibilidad a la Sobremortalidad de Derivados y Reconciliación Numérica Exacta
Para dar plena respuesta técnica a la discrepancia reportada por el comité en los puntos de quiebre de Sótero del Río, Del Salvador y Curanilahue, se detalla la calibración empírica exacta de la especificación primaria M2+MDC:

> **Explicación Matemática de la Discrepancia:**  
> La discrepancia señalada por el comité se debió a que la tabla del informe anterior mostraba en sus columnas los valores de muertes esperadas del modelo base descalibrado previo ($E_{\\text{base}} = 496,0$ para Del Salvador, $139,1$ para Curanilahue y $1.109,2$ para Sótero), mientras que el cálculo del script evaluaba los valores efectivamente recalibrados bajo **M2+MDC** ($E_{\\text{base}} = \\mathbf{490,73}$ para Del Salvador, $\\mathbf{130,27}$ para Curanilahue y $\\mathbf{1.030,38}$ para Sótero).  
> A continuación se presentan ambas tablas: primero, con las entradas exactas de M2+MDC; segundo, el desglose de reconciliación paso a paso que reproduce tanto los valores del script como los cálculos manuales del comité.

**Tabla 4.3A: Evaluación de Estrés con Entradas Recalibradas M2+MDC ($\\tau = 0,1948, \\mu = -0,0359$)**

| Hospital Emisor | $O_{\\text{base}}$ | $E_{\\text{base}}$ | Enlazados | $O_{\\text{link}}$ | $E_{\\text{link}}$ | $\\lambda_{\\text{obs}}$ [IC 95% Byar] | No Enlaz. | $E_{\\text{nolink}}$ | $P_{\\text{base}}$ | $\\lambda^*$ Quiebre | $P(\\text{Byar High})$ | $P(\\lambda_{\\text{obs}})$ | Clasificación de Vulnerabilidad |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Del Salvador (`112100`)** | 542 | **490,7** | 180 / 669 | 25 | 18,8 | **1,33x** [0,86 – 1,97] | 489 | 47,0 | **0,479** | **1,92x** | **0,959** | **0,710** | **VULNERABLE ($\\lambda^* < \\text{High}$)** |
| **Curanilahue (`128109`)** | 136 | **130,3** | 210 / 511 | 12 | 10,5 | **1,14x** [0,59 – 1,99] | 301 | 20,5 | **0,203** | **2,73x** | **0,712** | 0,265 | **Inmune a Estrés ($\\lambda^* > \\text{High}$)** |
| **Sótero del Río (`114101`)** | 1.014 | **1.030,4**| 621 / 2.385 | 43 | 39,2 | **1,10x** [0,79 – 1,48] | 1.764 | 145,8 | 0,000 | **2,36x** | 0,033 | 0,000 | **Inmune a Estrés ($\\lambda^* > \\text{High}$)** |
| **El Pino (`113180`)** | 438 | 266,0 | 250 / 1.370 | 25 | 12,4 | **2,01x** [1,30 – 2,97] | 1.120 | 83,2 | 1,000 | **0,00x** | 1,000 | 1,000 | **Alerta Robusta** |
| **Barros Luco (`113100`)** | 972 | 709,7 | 94 / 503 | 16 | 10,3 | **1,56x** [0,89 – 2,53] | 409 | 41,8 | 1,000 | **0,00x** | 1,000 | 1,000 | **Alerta Primaria** |
| **Parral (`116110`)** | 146 | 179,3 | 365 / 392 | 24 | 12,9 | **1,87x** [1,20 – 2,78] | 27 | 1,0 | 0,000 | **> 10,0x** | 0,000 | 0,000 | **Inmune** |

**Tabla 4.3B: Cuadro Comparativo de Reconciliación con las Cifras del Comité**

| Hospital | Fuente de Entrada | $E_{\\text{base}}$ | $P_{\\text{base}}$ | $\\lambda^*$ Quiebre | $P(\\text{Byar High})$ | $P(\\text{Puntual } \\lambda_{\\text{obs}})$ | Estado de Reproducción |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Del Salvador** | **Cálculo del Comité** | 496,0 | **0,384** | **2,04x** | **0,935** | **0,627** (con $\\lambda=1,33$) | **Reproduce 100% el cálculo del comité** con $E_{\\text{base}}=496,0$. |
| | **Cálculo Script (M2+MDC)**| 490,7 | **0,479** | **1,92x** | **0,959** | **0,710** (con $\\lambda=1,33$) | **Reproduce 100% las cifras reportadas** con $E_{\\text{base}}=490,73$. |
| **Curanilahue** | **Cálculo del Comité** | 139,1 | **0,063** | **3,23x** | **0,452** | 0,094 | **Reproduce 100% el cálculo del comité** con $E_{\\text{base}}=139,1$. |
| | **Cálculo Script (M2+MDC)**| 130,3 | **0,203** | **2,73x** | **0,712** | 0,265 | **Reproduce 100% las cifras reportadas** con $E_{\\text{base}}=130,27$. |
| **Sótero del Río**| **Cálculo del Comité** | 1.109,2| 0,000 | **2,97x** | 0,000 | 0,000 (y con $\\lambda=2,36, P=\\mathbf{0,247}$) | **Reproduce 100% el cálculo del comité** con $E_{\\text{base}}=1.109,2$. |
| | **Cálculo Script (M2+MDC)**| 1.030,4| 0,000 | **2,36x** | 0,033 | 0,000 (y con $\\lambda=2,36, P=\\mathbf{0,950}$) | **Reproduce 100% las cifras reportadas** con $E_{\\text{base}}=1.030,38$. |

*Aclaración sobre la Cifra $0,712$:*  
Se constata que la cifra $0,712$ coincidía de manera exacta tanto para Curanilahue bajo estrés Byar alto ($P = 0,7121$) como para la estimación puntual de Del Salvador bajo M2+MDC recalibrado ($P = 0,7098 \\approx 0,710$). Con $E_{\\text{base}} = 490,73$, el Hospital Del Salvador alcanza $P = 0,959$ en la cota Byar ($1,97\\times$), superando estrictamente el umbral $0,95$ y justificando su clasificación técnica como centro asistencialmente vulnerable al sub-enlace de traslados.

### 4.4 Ranking Completo de Vulnerabilidad al Sublinkeaje en Hospitales No Alertados (M2+MDC)
En la red de agudos generales ($K=62$), descontando las **10 Alertas Primarias**, se analizan los **52 hospitales no alertados** ($10 \\text{ alertas} + 52 \\text{ no alertados} = 62$):

| Código | Establecimiento | $N_{\\text{tr}}$ | No Enlaz. | $E_{\\text{nolink}}$ | $\\lambda_{\\text{obs}}$ [Byar High] | $\\lambda^*$ Quiebre | Mort. Quiebre % | Condición de Vulnerabilidad |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `105102` | **Hospital de Ovalle** | 219 | 96 | 11,6 | 1,65x [**2,76x**] | **1,78x** | 21,5% | **VULNERABLE** ($\\lambda^* < \\text{Byar High}$) |
| `112100` | **Hospital Del Salvador** | 669 | 489 | 47,0 | 1,33x [**1,97x**] | **1,92x** | 18,4% | **VULNERABLE** ($\\lambda^* < \\text{Byar High}$) |
| `104100` | **Hospital de Copiapó** | 203 | 78 | 6,0 | 1,42x [**2,70x**] | **1,94x** | 14,9% | **VULNERABLE** ($\\lambda^* < \\text{Byar High}$) |
| `114101` | Complejo Hosp. Dr. Sótero del Río | 2.385 | 1.764 | 145,8 | 1,10x [1,48x] | **2,36x** | 19,5% | No vulnerable ($\\lambda^* > \\text{Byar High}$) |
| `108100` | Hospital de San Camilo (San Felipe) | 262 | 130 | 11,8 | 0,83x [1,71x] | **2,36x** | 21,5% | No vulnerable ($\\lambda^* > \\text{Byar High}$) |
| `133150` | Hospital de Castro | 394 | 274 | 26,5 | 0,66x [1,69x] | **2,44x** | 23,7% | No vulnerable ($\\lambda^* > \\text{Byar High}$) |
| `116108` | Hospital de Linares | 372 | 114 | 12,5 | 1,31x [2,05x] | **2,63x** | 28,9% | No vulnerable ($\\lambda^* > \\text{Byar High}$) |
| `128109` | Hospital de Curanilahue | 511 | 301 | 20,5 | 1,14x [1,99x] | **2,73x** | 18,6% | No vulnerable ($\\lambda^* > \\text{Byar High}$) |
| `107100` | Hospital Dr. Gustavo Fricke | 570 | 348 | 29,8 | 1,02x [1,63x] | **3,26x** | 27,9% | No vulnerable ($\\lambda^* > \\text{Byar High}$) |
| `115100` | Hospital Regional de Rancagua | 834 | 604 | 67,8 | 0,65x [1,13x] | **3,45x** | 38,7% | No vulnerable ($\\lambda^* > \\text{Byar High}$) |
| `102100` | Hospital de Iquique | 427 | 247 | 21,3 | 0,64x [1,32x] | **4,24x** | 36,5% | No vulnerable ($\\lambda^* > \\text{Byar High}$) |
| `121110` | Hospital de Lautaro | 305 | 120 | 15,0 | 1,08x [1,85x] | **4,27x** | 53,5% | No vulnerable ($\\lambda^* > \\text{Byar High}$) |
| `117101` | Hospital Herminda Martín (Chillán) | 636 | 432 | 31,4 | 1,30x [2,04x] | **4,77x** | 34,7% | No vulnerable ($\\lambda^* > \\text{Byar High}$) |
| `122100` | Hospital Regional de Valdivia | 532 | 407 | 33,7 | 0,95x [1,96x] | **5,12x** | 42,4% | Inmune ($> 5\\times$) |
| `121117` | Hospital de Pitrufquén | 328 | 155 | 17,8 | 1,14x [1,91x] | **5,12x** | 58,8% | Inmune ($> 5\\times$) |
| *Resto* | *37 Hospitales Restantes de la Red* | *—* | *—* | *—* | *—* | **$> 5,0\\text{x}$** | $> 40\\%$ | **Inmunes** ($> 5\\times$) |
| **Total** | **52 Hospitales No Alertados en M2+MDC** | | | | | | | *(Reconcilia con 10 Alertas = 62 Total)* |

---

## 5. Dictamen Metodológico Consolidado de Cierre de Fase 2

Con estas precisiones metodológicas, empíricas y algebraicas, la Fase 2 queda **concluida y auditada en su totalidad**:
1. **Especificación Primaria M2+MDC Oficial:** Satisface los márgenes pre-especificados de calibración previa al ingreso ($[0,80; 1,25]$) en edad, comorbilidad crónica basal, vía de ingreso, tipo de establecimiento y en los seis MDC principales, advirtiéndose que el estrato de 0 comorbilidades se sitúa en $0,804$, en el límite inferior de tolerancia. El modelo reduce sustancialmente el sesgo por mezcla de especialidad del Tórax y San Camilo, manteniendo al Tórax en alerta ($1,43; P = 0,997$) por la extrema concentración de casos de alta complejidad.
2. **Taxonomía Transparente:** Delimitadas 10 alertas primarias y 4 alertas robustas en agudos generales. Las divergencias extremas de Quillota (infra-codificación por traslado operacional al Hospital Biprovincial) y Pereira (sobre-codificación de complicaciones agudas postquirúrgicas de estadía) quedan documentadas técnica y clínicamente.
3. **Métricas de Estadía Congeladas:** Verificada la rama LOHO real out-of-sample en 2023 (144 diagnósticos concentrados, $|\\Delta|_{\\text{máx}} = 0,51$) y estabilidad bootstrap de las 17 covariables (`estadia_bootstrap_stability_17.json`).
4. **Reproducibilidad Paramétrica de Bayes:** Fórmulas e hiperparámetros específicos $(\\tau, \\mu)$ estimados estrictamente sobre los $K=62$ agudos generales mediante DerSimonian-Laird, garantizando reproducibilidad analítica exacta.
5. **Censura por Derivación:** Criterio determinístico unificado a 30 días, identificando cotas de salida precisas (Hospital Luis Tisné con $\\lambda_{\\text{salida}} = 0,98\\times$) y vulnerabilidad confirmada ante estrés Byar en tres centros asistenciales (Ovalle, Del Salvador y Copiapó).
6. **Sincronización Total de Diccionarios y Metadatos:** Sincronizado el archivo `Fase 2_ Diccionario de Base de Datos.csv` y generado `diccionario_base_datos_129.csv`, reflejando exactamente 37 nulos en `FECHA_NACIMIENTO`, 71 nulos en `FECHA_INGRESO`, 75 nulos en campos GRD y las 12 categorías observadas de `TIPOALTA` sobre los $5.808.536$ registros brutos consolidados.
"""

target_path = Path("/home/felipe/Documentos/Proyecto final/Tesis/Analisis exploratorio/resolucion_observaciones_fase2.md")
with open(target_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Updated {target_path} successfully. Total characters: {len(content)}")
