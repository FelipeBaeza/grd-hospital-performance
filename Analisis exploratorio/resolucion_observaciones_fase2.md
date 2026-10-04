# Resolución Metodológica y Cierre Definitivo de la Fase 2: Especificación Primaria M2+MDC, Calibración Previa al Ingreso, Verificación LOHO y Sensibilidad a la Censura por Traslado

**Fecha de Cierre:** Octubre 2026  
**Repositorio:** `Tesis` (Rama `main`)  
**Cohorte Validada:** Egresos Inpatient FONASA 2019–2024 ($N = 5.808.536$ registros brutos consolidados en Plata), Cohorte de Desarrollo 2020–2022 ($N = 1.453.141$; $N_{\text{stay}} = 1.303.718$; año 2019 con $N = 1.151.475$ empleado exclusivamente como ventana retrospectiva de búsqueda de contactos hospitalarios previos para `N_EGRESOS_12M`), Cohorte Evaluable de Validación 2023 ($N = 557.318$, $O = 24.225$ defunciones intrahospitalarias, $K = 65$ hospitales de la red pública, $K = 62$ agudos generales).

> **Conciliación de Denominadores y Muestra de Desarrollo:**
> 1. **Total Bruto Silver ($N = 5.808.536$):** La cifra de $5.820.730$ de borradores preliminares correspondía a una extracción de staging previa a la desduplicación unificada. El universo formal de egresos cerrados 2019–2024 en Plata es de exactamente $5.808.536$ episodios (2019: 1.151.475; 2020: 781.912; 2021: 816.909; 2022: 932.840; 2023: 1.039.587; 2024: 1.085.813).
> 2. **Desarrollo 2020–2022 ($N = 1.453.141$):** El entrenamiento se realiza estrictamente en 2020–2022. La diferencia de 255 episodios respecto de los $1.453.396$ preliminares obedece a la exclusión de adultos ($\ge 18$ años) egresados de hospitales pediátricos monográficos (Calvo Mackenna `112102`, Roberto del Río `109101` y Exequiel González Cortés `113130`), desglosados en:
>    * **Año 2020:** 85 episodios adultos.
>    * **Año 2021:** 86 episodios adultos.
>    * **Año 2022:** 96 episodios adultos.  
>    *(Total = 267 episodios brutos; 255 tras descartar censuras de traslados inter-hospitalarios).* En 2023 se excluyen de forma idéntica exactamente **84 episodios adultos** ($557.402 \to 557.318$).
> 3. **Origen de las Comorbilidades:** Las 28 comorbilidades crónicas Elixhauser de la especificación primaria provienen **exclusivamente de los diagnósticos secundarios del episodio actual** (`DIAGNOSTICO2` a `DIAGNOSTICO35`). El año 2019 se utiliza exclusivamente como ventana causal para computar el historial de contactos previos (`N_EGRESOS_12M` y `DIAS_DESDE_EGRESO_PREVIO`).

---

## 1. La Especificación Primaria Oficial: M2 + Diagnóstico Principal (M2+MDC)

### 1.1 Justificación Teórica y Calibración Preingreso Multidimensional
En estricta coherencia con el criterio de calibración por estratos previos al ingreso, el modelo clínico base sin diagnóstico principal (M2) presentaba una falla estructural en la dimensión de categoría diagnóstica mayor (MDC): un $O/E$ de **2,28** en patología respiratoria (MDC 4) y de **0,35** en patología musculoesquelética (MDC 8). Esta descalibración penalizaba espuriamente a cualquier centro con alta concentración de patología médica respiratoria.

Al incorporar el diagnóstico principal agrupado (**Modelo M2+MDC**), los seis MDC principales se calibran dentro del margen $[0,93; 1,05]$, satisfaciendo el estándar de equidad asistencial.

> **Definición Rigurosa del Diagnóstico Principal y Agrupamiento MDC:**  
> En la base de egresos hospitalarios de Chile no existe bandera de *Present on Admission* (POA); el diagnóstico principal (`DIAGNOSTICO1`) se consigna al egreso como la causa principal que motivó la internación. A fin de evitar cualquier fuga de información (*data leakage*) desde el agrupador IR-GRD (cuya asignación depende de la estancia y de procedimientos), el agrupamiento MDC se mapea **directa y determinísticamente desde los capítulos CIE-10 del diagnóstico principal**:

| Capítulo CIE-10 | Rango Códigos | Descripción Clínica del Capítulo | CDM / MDC Mapeada |
| :---: | :---: | :--- | :---: |
| **I** | A00 – B99 | Ciertas enfermedades infecciosas y parasitarias | **MDC 18** (Infecciosas y Parasitarias Sistémicas) |
| **II** | C00 – D48 | Neoplasias (tumores malignos e in situ) | **MDC 17** (Neoplasias y Mieloproliferativas) |
| **III** | D50 – D89 | Enfermedades de la sangre y hematopoyéticos | **MDC 16** (Trastornos de la Sangre e Inmunidad) |
| **IV** | E00 – E90 | Endocrinas, nutricionales y metabólicas | **MDC 10** (Endocrinas, Nutrición y Metabolismo) |
| **V** | F00 – F99 | Trastornos mentales y del comportamiento | **MDC 19** (Trastornos Mentales y Conductuales) |
| **VI** | G00 – G99 | Enfermedades del sistema nervioso | **MDC 01** (Sistema Nervioso) |
| **VII – VIII**| H00 – H95 | Enfermedades de ojo, oído y anexos | **MDC 02 / 03** (Ojo, Oído, Nariz y Garganta) |
| **IX** | I00 – I99 | Enfermedades del sistema circulatorio | **MDC 05** (Sistema Circulatorio) |
| **X** | J00 – J99 | Enfermedades del sistema respiratorio | **MDC 04** (Sistema Respiratorio) |
| **XI** | K00 – K93 | Enfermedades del sistema digestivo | **MDC 06** (Sistema Digestivo) |
| **XII** | L00 – L99 | Enfermedades de la piel y tejido subcutáneo | **MDC 09** (Piel y Tejido Subcutáneo) |
| **XIII** | M00 – M99 | Sistema osteomuscular y tejido conjuntivo | **MDC 08** (Sistema Musculoesquelético) |
| **XIV** | N00 – N99 | Enfermedades del sistema genitourinario | **MDC 11** (Riñón y Tracto Urinario) |
| **XVIII** | R00 – R99 | Síntomas, signos y hallazgos anormales | **MDC 23** (Factores que Influyen en Salud / Otros) |
| **XIX** | S00 – T98 | Traumatismos, envenenamientos y causas externas | **MDC 21 / 22** (Traumatismos y Lesiones) |
| **XXI – XXII**| U00 – Z99 | Códigos de emergencia (U07 COVID) y contacto | **MDC 04 / 23** (según afección sistémica) |

| Modelo | Definición de Covariables | $\tau$ Agudos ($K=62$) | $\mu$ Meta | Calibración MDC $[0,80; 1,25]$ | Rol Metodológico |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **M2+MDC** | Demografía + Vía Ingreso + 28 Elixhauser Crónicos + MDC (CIE-10) | **0,1948** | **-0,0359** | **CUMPLE** (MDC 0,93 a 1,05) | **ESPECIFICACIÓN PRIMARIA** |
| **M2 Base** | Demografía + Vía Ingreso + 28 Elixhauser Crónicos (sin MDC) | 0,2077 | -0,0267 | Falla (MDC 4: 2,28; MDC 8: 0,35) | **SENSIBILIDAD 1** (Sin Dx Principal) |
| **M3-Total** | M2 Base + Splines cúbicos de $R_{\text{dx}}$ Total al Alta | 0,2041 | -0,0347 | Absorbe complicaciones de estadía | **SENSIBILIDAD 2** (Codificación Aguda) |

*Causa Matemática de la Variación en $\mu$:*  
El paso de $\mu = +0,0224$ (o $+0,0204$) a $\mu = -0,0359$ responde estrictamente a la transición del **promedio ponderado de efectos fijos** (inverse-variance pooling: $\bar{y} = \sum O_i \ln(O_i/E_i) / \sum O_i = +0,0204$, donde los hospitales grandes con mayor mortalidad dominan el peso) al **estimador de efectos aleatorios de DerSimonian-Laird** ($\hat{\mu}_{\text{DL}} = \sum w_i^* y_i / \sum w_i^*$ con $w_i^* = 1/(1/O_i + \tau^2) = -0,0359$), el cual otorga un peso más equilibrado a los centros medianos de la red cuya razón observada es ligeramente inferior al promedio.

### 1.2 Impacto sobre Centros Especializados: El Caso del Instituto Nacional del Tórax
El **Instituto Nacional del Tórax (`112103`)** presenta una concentración del **92,8%** en patología cardiovascular y respiratoria:
* **MDC 05 Circulatorio:** **62,6%** (2.222 de 3.548 episodios evaluables; cumple la regla de MDC dominante $\ge 50\%$).
* **MDC 04 Respiratorio:** **30,1%** (1.069 de 3.548 episodios evaluables).
* En M2 Base (sin MDC), el Tórax registraba $O/E = \mathbf{1,968}$ ($P = 1,000$).
* En **M2+MDC (Primaria)**, al reconocer su perfil cardiovascular y respiratorio, su exceso se reduce a $O/E = \mathbf{1,429}$ ($P = 0,997$). El ajuste **reduce sustancialmente el sesgo** por mezcla de especialidad, manteniendo al establecimiento en alerta debido a la gravedad intrínseca de sus ingresos.
* En M3-Total, debido a que su codificación secundaria es inferior a su estancia, su $O/E$ se distorsiona hasta $2,226$.

---

## 2. Taxonomía y Reconciliación de Alertas Asistenciales

Todas las métricas de esta sección se derivan estrictamente del archivo canonical único de la tesis: [`benchmarks_modelos_fase2_2023.csv`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/benchmarks_modelos_fase2_2023.csv).

### 2.1 Definición Rigurosa de Términos
1. **Alerta Primaria:** Todo hospital que en la **Especificación Primaria Oficial (M2+MDC)**, con $\tau = 0,1948$ y $\mu = -0,0359$, presenta:
   $$P(\theta_j > 1,10 \mid \text{M2+MDC}) \ge 0,95$$
2. **Alerta Robusta:** Hospital que es **Alerta Primaria** y concurrentemente **persiste como alerta en el modelo de sensibilidad M3-Total** ($P \ge 0,95$ en ambos).
3. **Alerta Dependiente de Codificación Aguda:** Hospital que no califica como alerta en la especificación primaria, pero que alerta únicamente en M3-Total debido a diagnósticos secundarios codificados al alta.

### 2.2 Lista Consolidada de Alertas en Hospitales Agudos Generales ($K=62$)
Bajo la especificación primaria M2+MDC, se identifican exactamente **10 Alertas Primarias** (y 52 hospitales en Desempeño Promedio):

| Código | Establecimiento | $O$ Defunciones | $E_{\text{prim}}$ | $O/E$ (M2+MDC) | $P_{\text{Prim}}$ | $E_{\text{m2}}$ | $O/E$ (M2 Base) | $P_{\text{M2}}$ | $E_{\text{m3}}$ | $O/E$ (M3-Tot) | $P_{\text{M3}}$ | Categoría de Alerta |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `113180` | **Hospital El Pino** | 438 | 266,0 | **1,646** | **1,000** | 238,6 | 1,836 | 1,000 | 247,1 | 1,772 | 1,000 | **ALERTA ROBUSTA** |
| `111100` | **Hospital Clínico San Borja-Arriarán** | 145 | 101,3 | **1,432** | **0,996** | 95,7 | 1,515 | 1,000 | 123,2 | 1,177 | 0,697 | **Alerta Primaria** |
| `106103` | **Hospital Claudio Vicuña (San Antonio)** | 320 | 231,8 | **1,381** | **1,000** | 200,7 | 1,595 | 1,000 | 195,8 | 1,634 | 1,000 | **ALERTA ROBUSTA** |
| `113100` | **Hospital Barros Luco Trudeau** | 972 | 709,7 | **1,370** | **1,000** | 754,7 | 1,288 | 1,000 | 842,5 | 1,154 | 0,914 | **Alerta Primaria** |
| `107101` | **Hospital San Martín (Quillota)** | 285 | 208,7 | **1,365** | **1,000** | 235,3 | 1,211 | 0,919 | 326,7 | 0,872 | 0,000 | **Alerta Primaria** |
| `106100` | **Hospital Carlos Van Buren** | 602 | 451,4 | **1,334** | **1,000** | 465,7 | 1,293 | 1,000 | 419,3 | 1,436 | 1,000 | **ALERTA ROBUSTA** |
| `103100` | **Hospital Dr. Leonardo Guzmán (Antofagasta)**| 663 | 497,8 | **1,332** | **1,000** | 479,1 | 1,384 | 1,000 | 569,2 | 1,165 | 0,908 | **Alerta Primaria** |
| `121109` | **Hospital Dr. Hernán Henríquez (Temuco)** | 580 | 451,4 | **1,285** | **1,000** | 467,4 | 1,241 | 0,997 | 498,1 | 1,165 | 0,888 | **Alerta Primaria** |
| `112101` | **Hospital Dr. Luis Tisné B.** | 502 | 407,3 | **1,233** | **0,990** | 382,2 | 1,313 | 1,000 | 381,3 | 1,316 | 1,000 | **ALERTA ROBUSTA** |
| `109100` | **Complejo Hospitalario San José** | 815 | 683,5 | **1,192** | **0,984** | 685,8 | 1,188 | 0,981 | 724,5 | 1,125 | 0,700 | **Alerta Primaria** |
| `112103` | *Instituto Nacional del Tórax (Monográfico)*| 157 | 109,8 | **1,429** | **0,997** | 79,8 | 1,968 | 1,000 | 70,5 | 2,226 | 1,000 | *Alerta Robusta Monog.* |

> **Nota de Conciliación de Cifras:**
> * **Temuco (`121109`) y Carlos Van Buren (`106100`):** Ambos presentan $E_{\text{prim}} = 451,4$ en la tabla debido a una **coincidencia matemática de redondeo a 1 decimal**: el valor exacto de Van Buren es $E = \mathbf{451,4214}$, mientras que el de Temuco es $E = \mathbf{451,3872}$.
> * **Del Salvador (`112100`):** En el modelo canonical M2+MDC registra $O = 542, E_{\text{prim}} = \mathbf{490,73} \implies O/E = \mathbf{1,1045}$ con $P_{\text{prim}} = \mathbf{0,479}$. En M2 Base registra $E = 577,54 \implies O/E = 0,938$ ($P = 0,000$), y en M3-Total $E = 483,43 \implies O/E = 1,121$ ($P = 0,619$). La mención de $O/E = 1,093$ en versiones anteriores correspondía al cociente con $E = 496,0$ ($542/496 = 1,0927$). El valor canónico oficial es $1,1045$.

### 2.3 Casos de Divergencia Extrema: Mecanismos Clínicos y Codificación

1. **Hospital San Martín / Biprovincial Quillota (`107101`):**  
   * **Comportamiento:** En M2+MDC registra $O/E = \mathbf{1,365}$ ($P = 1,000$, Alerta Primaria), mientras que en M3-Total cae a $O/E = \mathbf{0,872}$ ($P = 0,000$), una divergencia de **$-0,49$**.
   * **Serie Histórica de Diagnósticos Secundarios (2019–2024):**  
     La serie anual de diagnósticos secundarios promedio en egresos hospitalizados de Quillota (`107101`) muestra una aceleración progresiva de codificación:
     * **2019:** 2,549 diagnósticos secundarios.
     * **2020:** 3,091 diagnósticos secundarios.
     * **2021:** 3,328 diagnósticos secundarios.
     * **2022:** 3,463 diagnósticos secundarios (Línea base cohorte de desarrollo = **3,181**).
     * **2023:** **4,763** diagnósticos secundarios.
     * **2024:** 5,434 diagnósticos secundarios.
   * **Mecanismo del Modelo M3:** En M3 out-of-sample, $R_{\text{dx}}$ se estandariza con respecto a la media histórica del propio hospital durante desarrollo: $R_{\text{dx}, j} = N_{\text{dx}, j} / \overline{N}_{\text{dx}, \text{Dev}(j)}$. Como Quillota aumentó su codificación secundaria en 2023 en un $+44\%$ respecto de su propia historia ($4,76 / 3,18 = \mathbf{1,44}$), el modelo M3 (cuyo efecto es monótonamente creciente con $R_{\text{dx}}$) predijo significativamente más defunciones ($E_{\text{M3}} = 326,7$ vs $E_{\text{M2+MDC}} = 208,7$), deprimiendo artificialmente su $O/E$ a $0,872$. En contraste, la especificación primaria M2+MDC, al basarse en comorbilidades crónicas y clasificar por MDC al egreso, evalúa $E = 208,7$, detectando un exceso real de $+76,3$ defunciones. *(Se aclara que el código `200717` citado previamente corresponde a Padre Las Casas y no tiene vinculación con Quillota).*

2. **Hospital Dr. Eduardo Pereira de Valparaíso (`106102`):**  
   * **Comportamiento:** En M2+MDC registra $O/E = \mathbf{0,838}$ ($P = 0,000$, Favorable), en M2 Base $O/E = 0,979$ ($P = 0,063$), pero en M3-Total salta a $O/E = \mathbf{1,478^*}$ ($P = 0,999$, Alerta), una brecha de **$+0,64$**.
   * **Mecanismo Operacional y Codificación:** Pereira es formalmente un **hospital de agudos generales** (`es_monografico == 'NO'`), con 30,3% de ingresos cardiovasculares y respiratorios y 21,8% digestivos. En la cohorte de desarrollo registraba una altísima codificación secundaria histórica ($7,59$ diagnósticos promedio). En 2023, su codificación secundaria media fue de $6,04$ ($R_{\text{dx}} = 6,04 / 7,59 = \mathbf{0,80}$). Al reducirse su codificación secundaria respecto de su línea base histórica, el modelo M3 predijo sustancialmente menos defunciones ($E_{\text{M3}} = 100,1$ vs $E_{\text{M2+MDC}} = 176,5$), haciendo saltar artificialmente su $O/E$ a $1,478^*$. La especificación primaria M2+MDC confirma que para la gravedad clínica de sus ingresos, sus defunciones observadas ($O = 148$) son inferiores a lo esperado ($E = 176,5$).

3. **Hospital de San Camilo (`108100`):**  
   * En M2 Base alertaba ($O/E = 1,292, P = 0,998$), pero en **M2+MDC (Primaria)** su razón desciende a $O/E = \mathbf{1,174}$ con $P = \mathbf{0,848} < 0,95$, quedando en **Desempeño Promedio**. En M3-Total registra $O/E = 1,197, P = 0,920 < 0,95$. El ajuste por diagnóstico principal explica adecuadamente su case-mix.

---

## 3. Calibración, Descalce de $N_{\text{dx}}$ y Verificaciones Técnicas

### 3.1 Criterio de Calibración Preingreso y Ausencia de Asociación con $N_{\text{dx}}$
La especificación primaria M2+MDC satisface los márgenes pre-especificados de calibración previa al ingreso ($[0,80; 1,25]$ para todo estrato con $E \ge 100$):
* **Edad:** $<65$ años ($1,061$), $65-74$ años ($0,971$), $\ge 75$ años ($0,977$).
* **Carga Crónica:** 0 comorb ($0,804$), 1 comorb ($0,989$), 2 comorb ($1,085$), $\ge 3$ comorb ($1,006$). Se destaca que el estrato de 0 comorbilidades se sitúa en $0,804$, en el límite inferior de tolerancia admisible.
* **Vía Ingreso:** Urgencia ($1,000$), Programada ($0,991$).
* **MDC Principal (CIE-10):** MDC 6 Digestivo ($1,047$), MDC 8 Musculoesquelético ($0,932$), MDC 5 Circulatorio ($1,004$), MDC 4 Respiratorio ($1,008$), MDC 1 Nervioso ($0,979$), MDC 11 Riñón/Urinario ($0,982$).

*Asociación Empírica con $N_{\text{dx}}$ por Hospital:*  
* Correlación de Spearman entre $O/E_{\text{M2+MDC}}$ y $N_{\text{dx}}$ medio del hospital: **$\rho = -0,133$ ($p = 0,303$, no significativo)**.
* Correlación de Spearman entre $O/E_{\text{M2+MDC}}$ y porcentaje de pacientes con $\ge 11$ diagnósticos: **$\rho = -0,080$ ($p = 0,537$, no significativo)**.

**Conclusión:** En la red hospitalaria pública no se detecta correlación entre el desempeño primario y la profundidad de codificación secundaria. La verificación definitiva de calibración queda reservada para la cohorte prospectiva externa de 2024.

### 3.2 Verificación Real de LOHO y Concentración Diagnóstica
Se ratifican las métricas out-of-sample del codificador Leave-One-Hospital-Out (LOHO) en la cohorte 2023 de estadía:
* **Concentración:** 144 categorías CIE-10 (9,9% de las 1.459 activas) tienen $>50\%$ de sus casos concentrados en un solo hospital.
* **Máxima Discrepancia:** $|\Delta|_{\text{máx}} = 0,5079$ log-días (CIE G47 en Puerto Montt; H36 con 96,0% en Los Ángeles, $|\Delta|=0,4395$; T29 quemaduras con 55,5% en Posta Central, $|\Delta|=0,4295$).
* **Impacto Institucional:** $\rho = 0,99948$, $\text{RMS} = 0,61$ puestos. El Instituto de Neurocirugía Dr. Asenjo (`112104`) presenta la mayor variación de la red ($-0,0294$ en $O/E$, de $0,8033$ a $0,7739$).

### 3.3 Estabilidad por Bootstrap de las Covariables de Estadía
Se congelan los resultados de 100 réplicas bootstrap en desarrollo ($N = 1.303.718$) en el archivo [`scratch/estadia_bootstrap_stability_17.json`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/scratch/estadia_bootstrap_stability_17.json):
* **16 Variables Núcleo al 100%:** Incluyen a `ELIX_03` Valvulopatía (100%) y `ELIX_08` Parálisis (100%).
* **Borde en Puesto 17:** `ELIX_06` HTA no complicada (69,0%), `ELIX_19` Cáncer metastásico (19,0%), `ELIX_10` EPOC (10,0%), `ELIX_07` HTA complicada (2,0%). Todos con coeficientes entre $+0,0014$ y $+0,0038$. El ordenamiento institucional es insensible a la alternancia en este borde ($\rho \ge 0,999$).

---

## 4. Enlace Unificado, Cotas de Salida y Sensibilidad a la Censura por Traslado

### 4.1 Criterio Unificado de Enlace Determinístico
* **Total egresos por traslado inpatient adultos 2023:** **25.555** episodios.
* **Total enlazados en $\le 30$ días:** **11.544** episodios (tasa de enlace global: 45,2%).

### 4.2 Cotas de Salida ($\lambda_{\text{salida}}$) para las Alertas Primarias
En respuesta a la solicitud del comité, se presentan los datos completos de enlace ($O_{\text{link}}$ y $E_{\text{link}}$) y el multiplicador umbral $\lambda_{\text{salida}}$ en derivados no enlazados para salir de alerta ($P < 0,95$):

$$O_{\text{eval}} = O_{\text{base}} + O_{\text{link}} + \lambda_{\text{salida}} \cdot E_{\text{nolink}}, \quad E_{\text{eval}} = E_{\text{base}} + E_{\text{link}} + E_{\text{nolink}}$$

| Código | Establecimiento | $O_{\text{base}}$ | $E_{\text{base}}$ | $P_{\text{base}}$ | $O_{\text{link}}$ | $E_{\text{link}}$ | No Enlaz. | $E_{\text{nolink}}$ | Cota $\lambda_{\text{salida}}$ | ¿Factible Salir? | Interpretación Clínica |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `112101` | **Hospital Dr. Luis Tisné B.** | 502 | 407,3 | 0,990 | 20 | 18,8 | 699 | 86,6 | **0,98x** | **SÍ (Factible, $\lambda < 0,98$)** | **Alerta Marginal Sensible:** Si sus 699 derivados no enlazados mueren a una tasa estándar o favorable ($\lambda < 0,98\times$), el hospital sale de alerta ($P < 0,95$). |
| `121109` | **Hospital Dr. Hernán Henríquez (Temuco)** | 580 | 451,4 | 1,000 | 17 | 31,1 | 408 | 36,4 | **0,47x** | Difícil (requiere $\lambda < 0,47$) | Requiere que los no enlazados fallezcan a menos de la mitad del riesgo esperado. |
| `107101` | **Hospital San Martín (Quillota)** | 285 | 208,7 | 1,000 | 13 | 11,6 | 339 | 26,4 | **0,19x** | Muy Remoto ($\lambda < 0,19$) | Exige sobrevida casi absoluta de trasladados. |
| `109100` | **Complejo Hospitalario San José** | 815 | 683,5 | 0,984 | 1 | 3,5 | 179 | 10,1 | **0,05x** | Casi Imposible ($\lambda < 0,05$) | Exige cero muertes en derivados. |
| `113180` | **Hospital El Pino** | 438 | 266,0 | 1,000 | 25 | 12,4 | 1.120 | 83,2 | **No existe** | **NO (Alerta Inmune)** | Con $\lambda = 0$ (cero muertes en 1.120 trasladados), $P_{\text{min}} = 1,000$. |
| `111100` | **Hospital Clínico San Borja-Arriarán** | 145 | 101,3 | 0,996 | 4 | 1,1 | 34 | 1,3 | **No existe** | **NO (Alerta Inmune)** | Con $\lambda = 0$, $O/E = 1,398$ y $P_{\text{min}} = 0,995 \ge 0,95$. Inmune a dilución. |
| `106103` | **Hospital Claudio Vicuña (San Antonio)** | 320 | 231,8 | 1,000 | 14 | 8,8 | 102 | 8,4 | **No existe** | **NO (Alerta Inmune)** | Con $\lambda = 0$, $P_{\text{min}} = 1,000$. |
| `113100` | **Hospital Barros Luco Trudeau** | 972 | 709,7 | 1,000 | 16 | 10,3 | 409 | 41,8 | **No existe** | **NO (Alerta Inmune)** | Con $\lambda = 0$, $P_{\text{min}} = 1,000$. |
| `106100` | **Hospital Carlos Van Buren** | 602 | 451,4 | 1,000 | 58 | 54,8 | 349 | 27,9 | **No existe** | **NO (Alerta Inmune)** | Con $\lambda = 0$, $P_{\text{min}} = 1,000$. |
| `103100` | **Hospital Dr. Leonardo Guzmán (Antofagasta)**| 663 | 497,8 | 1,000 | 6 | 4,2 | 54 | 3,2 | **No existe** | **NO (Alerta Inmune)** | Con $\lambda = 0$, $P_{\text{min}} = 1,000$. |
| `112103` | *Instituto Nacional del Tórax (Monográfico)*| 157 | 109,8 | 0,997 | 6 | 4,5 | 36 | 2,2 | **No existe** | **NO (Alerta Inmune)** | Con $\lambda = 0$, $P_{\text{min}} = 0,996 \ge 0,95$. Inmune a dilución. |

### 4.3 Sensibilidad a la Sobremortalidad de Derivados y Reconciliación con el Archivo Canónico
Se presentan los puntos de quiebre evaluados a partir de los datos exactos del archivo canonical [`benchmarks_modelos_fase2_2023.csv`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/benchmarks_modelos_fase2_2023.csv):

**Tabla 4.3A: Evaluación de Estrés con Entradas Canónicas M2+MDC ($\tau = 0,1948, \mu = -0,0359$)**

| Hospital Emisor | $O_{\text{base}}$ | $E_{\text{base}}$ | Enlazados | $O_{\text{link}}$ | $E_{\text{link}}$ | $\lambda_{\text{obs}}$ [IC 95% Byar] | No Enlaz. | $E_{\text{nolink}}$ | $P_{\text{base}}$ | $\lambda^*$ Quiebre | $P(\text{Byar High})$ | $P(\lambda_{\text{obs}})$ | Clasificación de Vulnerabilidad |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Del Salvador (`112100`)** | 542 | **490,7** | 180 / 669 | 25 | 18,8 | **1,33x** [0,86 – 1,97] | 489 | 47,0 | **0,479** | **1,92x** | **0,959** | **0,710** | **VULNERABLE ($\lambda^* < \text{High}$)** |
| **Curanilahue (`128109`)** | 136 | **130,3** | 210 / 511 | 12 | 10,5 | **1,14x** [0,59 – 1,99] | 301 | 20,5 | **0,203** | **2,73x** | **0,712** | 0,265 | **Inmune a Estrés ($\lambda^* > \text{High}$)** |
| **Sótero del Río (`114101`)** | 1.014 | **1.030,4**| 621 / 2.385 | 43 | 39,2 | **1,10x** [0,79 – 1,48] | 1.764 | 145,8 | 0,000 | **2,36x** | 0,033 | 0,000 | **Inmune a Estrés ($\lambda^* > \text{High}$)** |
| **El Pino (`113180`)** | 438 | 266,0 | 250 / 1.370 | 25 | 12,4 | **2,01x** [1,30 – 2,97] | 1.120 | 83,2 | 1,000 | **0,00x** | 1,000 | 1,000 | **Alerta Robusta** |
| **Barros Luco (`113100`)** | 972 | 709,7 | 94 / 503 | 16 | 10,3 | **1,56x** [0,89 – 2,53] | 409 | 41,8 | 1,000 | **0,00x** | 1,000 | 1,000 | **Alerta Primaria** |
| **Parral (`116110`)** | 146 | 179,3 | 365 / 392 | 24 | 12,9 | **1,87x** [1,20 – 2,78] | 27 | 1,0 | 0,000 | **> 10,0x** | 0,000 | 0,000 | **Inmune** |

**Tabla 4.3B: Cuadro Comparativo de Reconciliación con las Cifras del Comité**

| Hospital | Fuente de Entrada | $E_{\text{base}}$ | $P_{\text{base}}$ | $\lambda^*$ Quiebre | $P(\text{Byar High})$ | $P(\text{Puntual } \lambda_{\text{obs}})$ | Estado de Reproducción |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Del Salvador** | **Cálculo del Comité** | 496,0 | **0,384** | **2,04x** | **0,935** | **0,627** (con $\lambda=1,33$) | **Reproduce 100% el cálculo del comité** con $E_{\text{base}}=496,0$. |
| | **Cálculo Canónico M2+MDC**| **490,73** | **0,479** | **1,92x** | **0,959** | **0,710** (con $\lambda=1,33$) | **Reproduce 100% las cifras reportadas** con $E_{\text{base}}=490,73$. |
| **Curanilahue** | **Cálculo del Comité** | 139,1 | **0,063** | **3,23x** | **0,452** | 0,094 | **Reproduce 100% el cálculo del comité** con $E_{\text{base}}=139,1$. |
| | **Cálculo Canónico M2+MDC**| **130,27** | **0,203** | **2,73x** | **0,712** | 0,265 | **Reproduce 100% las cifras reportadas** con $E_{\text{base}}=130,27$. |
| **Sótero del Río**| **Cálculo del Comité** | 1.109,2| 0,000 | **2,97x** | 0,000 | 0,000 (y con $\lambda=2,36, P=\mathbf{0,247}$) | **Reproduce 100% el cálculo del comité** con $E_{\text{base}}=1.109,2$. |
| | **Cálculo Canónico M2+MDC**| **1.030,38**| 0,000 | **2,36x** | 0,033 | 0,000 (y con $\lambda=2,36, P=\mathbf{0,950}$) | **Reproduce 100% las cifras reportadas** con $E_{\text{base}}=1.030,38$. |

*Aclaración sobre la Cifra $0,712$:*  
Se constata que la cifra $0,712$ coincidía de manera exacta tanto para Curanilahue bajo estrés Byar alto ($P = 0,7121$) como para la estimación puntual de Del Salvador bajo M2+MDC recalibrado ($P = 0,7098 \approx 0,710$). Con el valor canonical $E_{\text{base}} = 490,73$, el Hospital Del Salvador alcanza $P = 0,959$ en la cota Byar ($1,97\times$), superando estrictamente el umbral $0,95$ y justificando su clasificación técnica como centro asistencialmente vulnerable al sub-enlace de traslados.

### 4.4 Ranking Completo de Vulnerabilidad al Sublinkeaje en Hospitales No Alertados (M2+MDC)
En la red de agudos generales ($K=62$), descontando las **10 Alertas Primarias**, se analizan los **52 hospitales no alertados** ($10 \text{ alertas} + 52 \text{ no alertados} = 62$):

| Código | Establecimiento | $N_{\text{tr}}$ | No Enlaz. | $E_{\text{nolink}}$ | $\lambda_{\text{obs}}$ [Byar High] | $\lambda^*$ Quiebre | Mort. Quiebre % | Condición de Vulnerabilidad |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `105102` | **Hospital de Ovalle** | 219 | 96 | 11,6 | 1,65x [**2,76x**] | **1,78x** | 21,5% | **VULNERABLE** ($\lambda^* < \text{Byar High}$) |
| `112100` | **Hospital Del Salvador** | 669 | 489 | 47,0 | 1,33x [**1,97x**] | **1,92x** | 18,4% | **VULNERABLE** ($\lambda^* < \text{Byar High}$) |
| `104100` | **Hospital de Copiapó** | 203 | 78 | 6,0 | 1,42x [**2,70x**] | **1,94x** | 14,9% | **VULNERABLE** ($\lambda^* < \text{Byar High}$) |
| `114101` | Complejo Hosp. Dr. Sótero del Río | 2.385 | 1.764 | 145,8 | 1,10x [1,48x] | **2,36x** | 19,5% | No vulnerable ($\lambda^* > \text{Byar High}$) |
| `108100` | Hospital de San Camilo (San Felipe) | 262 | 130 | 11,8 | 0,83x [1,71x] | **2,36x** | 21,5% | No vulnerable ($\lambda^* > \text{Byar High}$) |
| `133150` | Hospital de Castro | 394 | 274 | 26,5 | 0,66x [1,69x] | **2,44x** | 23,7% | No vulnerable ($\lambda^* > \text{Byar High}$) |
| `116108` | Hospital de Linares | 372 | 114 | 12,5 | 1,31x [2,05x] | **2,63x** | 28,9% | No vulnerable ($\lambda^* > \text{Byar High}$) |
| `128109` | Hospital de Curanilahue | 511 | 301 | 20,5 | 1,14x [1,99x] | **2,73x** | 18,6% | No vulnerable ($\lambda^* > \text{Byar High}$) |
| `107100` | Hospital Dr. Gustavo Fricke | 570 | 348 | 29,8 | 1,02x [1,63x] | **3,26x** | 27,9% | No vulnerable ($\lambda^* > \text{Byar High}$) |
| `115100` | Hospital Regional de Rancagua | 834 | 604 | 67,8 | 0,65x [1,13x] | **3,45x** | 38,7% | No vulnerable ($\lambda^* > \text{Byar High}$) |
| `102100` | Hospital de Iquique | 427 | 247 | 21,3 | 0,64x [1,32x] | **4,24x** | 36,5% | No vulnerable ($\lambda^* > \text{Byar High}$) |
| `121110` | Hospital de Lautaro | 305 | 120 | 15,0 | 1,08x [1,85x] | **4,27x** | 53,5% | No vulnerable ($\lambda^* > \text{Byar High}$) |
| `117101` | Hospital Herminda Martín (Chillán) | 636 | 432 | 31,4 | 1,30x [2,04x] | **4,77x** | 34,7% | No vulnerable ($\lambda^* > \text{Byar High}$) |
| `122100` | Hospital Regional de Valdivia | 532 | 407 | 33,7 | 0,95x [1,96x] | **5,12x** | 42,4% | Inmune ($> 5\times$) |
| `121117` | Hospital de Pitrufquén | 328 | 155 | 17,8 | 1,14x [1,91x] | **5,12x** | 58,8% | Inmune ($> 5\times$) |
| *Resto* | *37 Hospitales Restantes de la Red* | *—* | *—* | *—* | *—* | **$> 5,0\text{x}$** | $> 40\%$ | **Inmunes** ($> 5\times$) |
| **Total** | **52 Hospitales No Alertados en M2+MDC** | | | | | | | *(Reconcilia con 10 Alertas = 62 Total)* |

---

## 5. Dictamen Metodológico Consolidado de Cierre de Fase 2

Con estas precisiones metodológicas, empíricas y algebraicas, la Fase 2 queda **concluida y auditada en su totalidad**:
1. **Especificación Primaria M2+MDC Oficial:** Satisface los márgenes pre-especificados de calibración previa al ingreso ($[0,80; 1,25]$) en edad, comorbilidad crónica basal, vía de ingreso, tipo de establecimiento y en los seis MDC principales, advirtiéndose que el estrato de 0 comorbilidades se sitúa en $0,804$, en el límite inferior de tolerancia. El modelo reduce sustancialmente el sesgo por mezcla de especialidad del Tórax y San Camilo, manteniendo al Tórax en alerta ($1,43; P = 0,997$) por la extrema concentración de casos de alta complejidad.
2. **Taxonomía Transparente:** Delimitadas 10 alertas primarias y 4 alertas robustas en agudos generales. Las divergencias extremas de Quillota (aumento en la captura de diagnósticos secundarios en 2023) y Pereira (disminución en la intensidad de codificación respecto de su línea base) quedan documentadas técnica y clínicamente a partir de la fórmula $R_{\text{dx}, j} = N_{\text{dx}, j} / \overline{N}_{\text{dx}, \text{Dev}(j)}$.
3. **Métricas de Estadía Congeladas:** Verificada la rama LOHO real out-of-sample en 2023 (144 diagnósticos concentrados, $|\Delta|_{\text{máx}} = 0,51$) y estabilidad bootstrap de las 17 covariables (`estadia_bootstrap_stability_17.json`).
4. **Reproducibilidad Paramétrica de Bayes y Archivo Canonical:** Fórmulas e hiperparámetros específicos $(\tau, \mu)$ estimados estrictamente sobre los $K=62$ agudos generales mediante DerSimonian-Laird, garantizando reproducibilidad analítica exacta a partir de [`benchmarks_modelos_fase2_2023.csv`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/benchmarks_modelos_fase2_2023.csv).
5. **Censura por Derivación:** Criterio determinístico unificado a 30 días, identificando cotas de salida precisas (Hospital Luis Tisné con $\lambda_{\text{salida}} = 0,98\times$) y vulnerabilidad confirmada ante estrés Byar en tres centros asistenciales (Ovalle, Del Salvador y Copiapó).
6. **Sincronización Total de Diccionarios y Metadatos:** Sincronizado el archivo `Fase 2_ Diccionario de Base de Datos.csv` y generado `diccionario_base_datos_129.csv`, reflejando exactamente 37 nulos en `FECHA_NACIMIENTO`, 71 nulos en `FECHA_INGRESO`, 75 nulos en campos GRD y las 12 categorías observadas de `TIPOALTA` sobre los $5.808.536$ registros brutos consolidados.
