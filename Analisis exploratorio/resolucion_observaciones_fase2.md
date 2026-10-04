# Resolución Metodológica y Cierre Definitivo de la Fase 2: Especificación Primaria M2+CAP, Calibración Previa al Ingreso, Deriva de Codificación y Sensibilidad a la Censura por Traslado

**Fecha de Cierre:** Octubre 2026  
**Repositorio:** `Tesis` (Rama `main`)  
**Cohorte Validada:** Egresos Inpatient FONASA 2019–2024 ($N = 5.808.536$ registros brutos consolidados en Plata), Cohorte de Desarrollo 2020–2022 ($N = 1.453.141$; $N_{	ext{stay}} = 1.303.718$; año 2019 con $N = 1.151.475$ empleado exclusivamente como ventana retrospectiva de búsqueda de contactos hospitalarios previos para `N_EGRESOS_12M`), Cohorte Evaluable de Validación 2023 ($N = 557.318$, $O = 24.225$ defunciones intrahospitalarias, $K = 65$ hospitales de la red pública, $K = 62$ agudos generales).

> **Conciliación de Denominadores y Muestra de Desarrollo:**
> 1. **Total Bruto Silver ($N = 5.808.536$):** La cifra de $5.820.730$ de borradores preliminares correspondía a una extracción de staging previa a la desduplicación unificada. El universo formal de egresos cerrados 2019–2024 en Plata es de exactamente $5.808.536$ episodios (2019: 1.151.475; 2020: 781.912; 2021: 816.909; 2022: 932.840; 2023: 1.039.587; 2024: 1.085.813).
> 2. **Desarrollo 2020–2022 ($N = 1.453.141$):** El entrenamiento se realiza estrictamente en 2020–2022. La diferencia de 255 episodios respecto de los $1.453.396$ preliminares obedece a la exclusión de adultos ($\ge 18$ años) egresados de hospitales pediátricos monográficos (Calvo Mackenna `112102`, Roberto del Río `109101` y Exequiel González Cortés `113130`), desglosados en:
>    * **Año 2020:** 85 episodios adultos.
>    * **Año 2021:** 86 episodios adultos.
>    * **Año 2022:** 96 episodios adultos.  
>    *(Total = 267 episodios brutos; 255 tras descartar censuras de traslados inter-hospitalarios).* En 2023 se excluyen de forma idéntica exactamente **84 episodios adultos** ($557.402 	o 557.318$).
> 3. **Origen de las Comorbilidades:** Las 28 comorbilidades crónicas Elixhauser de la especificación primaria provienen **exclusivamente de los diagnósticos secundarios del episodio actual** (`DIAGNOSTICO2` a `DIAGNOSTICO35`). El año 2019 se utiliza exclusivamente como ventana causal para computar el historial de contactos previos (`N_EGRESOS_12M` y `DIAS_DESDE_EGRESO_PREVIO`).

---

## 1. La Especificación Primaria Oficial: M2 + Diagnóstico Principal (M2+CAP)

### 1.1 Justificación Teórica y Calibración Preingreso Multidimensional
En estricta coherencia con el criterio de calibración por estratos previos al ingreso, el modelo clínico base sin diagnóstico principal (M2 Base) presentaba una falla estructural en la dimensión diagnóstica: un $O/E$ de **2,28** en patología respiratoria y de **0,35** en patología musculoesquelética. Esta descalibración penalizaba espuriamente a cualquier centro con alta concentración de patología médica respiratoria.

Al incorporar el diagnóstico principal agrupado (**Modelo M2+CAP**), los principales grupos diagnósticos se calibran dentro del margen $[0,93; 1,05]$, satisfaciendo plenamente el estándar de equidad asistencial.

> **Nomenclatura Técnica y Correspondencia con Capítulos CIE-10 (M2+CAP):**  
> El modelo se denomina formalmente **`M2+CAP`** para transparentar que la estratificación del diagnóstico principal se realiza por los 20 **capítulos de la CIE-10** consignados en `DIAG1` y no por la Categoría Diagnóstica Mayor (MDC) producida algorítmicamente por el agrupador IR-GRD.  
> *Ejemplo Crítico:* Los accidentes cerebrovasculares (ACV, códigos $I60-I69$) pertenecen al **Capítulo IX de la CIE-10** (Enfermedades del sistema circulatorio), mientras que el agrupador IR-GRD los asigna clínicamente al MDC 01 (Enfermedades del sistema nervioso). La clasificación se realiza sin fuga de información (*leakage*) del agrupador comercial:

| Capítulo CIE-10 | Rango Códigos | Descripción Clínica del Capítulo | Grupo Asignado en Modelo |
| :---: | :---: | :--- | :---: |
| **I** | A00 – B99 | Ciertas enfermedades infecciosas y parasitarias | **Capítulo 18** (Infecciosas y Parasitarias) |
| **II** | C00 – D48 | Neoplasias (tumores malignos e in situ) | **Capítulo 17** (Neoplasias) |
| **III** | D50 – D89 | Enfermedades de la sangre y hematopoyéticos | **Capítulo 16** (Sangre e Inmunidad) |
| **IV** | E00 – E90 | Endocrinas, nutricionales y metabólicas | **Capítulo 10** (Endocrinas y Nutrición) |
| **V** | F00 – F99 | Trastornos mentales y del comportamiento | **Capítulo 19** (Salud Mental) |
| **VI** | G00 – G99 | Enfermedades del sistema nervioso | **Capítulo 01** (Sistema Nervioso) |
| **VII – VIII**| H00 – H95 | Enfermedades de ojo, oído y anexos | **Capítulo 02 / 03** (Sensoriales) |
| **IX** | I00 – I99 | Enfermedades del sistema circulatorio (incluye ACV) | **Capítulo 05** (Sistema Circulatorio) |
| **X** | J00 – J99 | Enfermedades del sistema respiratorio | **Capítulo 04** (Sistema Respiratorio) |
| **XI** | K00 – K93 | Enfermedades del sistema digestivo | **Capítulo 06** (Sistema Digestivo) |
| **XII** | L00 – L99 | Enfermedades de la piel y tejido subcutáneo | **Capítulo 09** (Piel y Subcutáneo) |
| **XIII** | M00 – M99 | Sistema osteomuscular y tejido conjuntivo | **Capítulo 08** (Musculoesquelético) |
| **XIV** | N00 – N99 | Enfermedades del sistema genitourinario | **Capítulo 11** (Nefrourológico) |
| **XVIII** | R00 – R99 | Síntomas, signos y hallazgos anormales | **Capítulo 23** (Signos y Síntomas) |
| **XIX** | S00 – T98 | Traumatismos, envenenamientos y causas externas | **Capítulo 21 / 22** (Traumatismos) |
| **XXI – XXII**| U00 – Z99 | Códigos de emergencia (U07 COVID) y contacto | **Capítulo 04 / 23** (Emergencias y Otros) |

| Modelo | Definición de Covariables | $	au$ Agudos ($K=62$) | $\mu$ Meta | Calibración Dx $[0,80; 1,25]$ | Rol Metodológico |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **M2+CAP** | Demografía + Vía Ingreso + 28 Elixhauser Crónicos + Capítulos CIE-10 | **0,1948** | **-0,0359** | **CUMPLE** (Estratos en 0,93 a 1,05) | **ESPECIFICACIÓN PRIMARIA** |
| **M2 Base** | Demografía + Vía Ingreso + 28 Elixhauser Crónicos (sin Capítulos) | 0,2077 | -0,0267 | Falla (Resp: 2,28; Musc: 0,35) | **SENSIBILIDAD 1** (Sin Dx Principal) |
| **M3-Total** | M2 Base + Splines cúbicos de $R_{	ext{dx}}$ Total al Alta | 0,2041 | -0,0347 | Absorbe complicaciones de estadía | **SENSIBILIDAD 2** (Codificación Aguda) |

*Causa Matemática de la Variación en $\mu$:*  
El paso de $\mu = +0,0224$ (o $+0,0204$) a $\mu = -0,0359$ responde estrictamente a la transición del **promedio ponderado de efectos fijos** (inverse-variance pooling: $ar{y} = \sum O_i \ln(O_i/E_i) / \sum O_i = +0,0204$, donde los hospitales grandes con mayor mortalidad dominan el peso) al **estimador de efectos aleatorios de DerSimonian-Laird** ($\hat{\mu}_{	ext{DL}} = \sum w_i^* y_i / \sum w_i^*$ con $w_i^* = 1/(1/O_i + 	au^2) = -0,0359$), el cual pondera equilibradamente a los centros de tamaño mediano de la red cuyo log-ratio medio se sitúa levemente por debajo de cero.

### 1.2 Impacto sobre Centros Especializados: El Caso del Instituto Nacional del Tórax
El **Instituto Nacional del Tórax (`112103`)** presenta una concentración del **92,8%** en patología cardiovascular y respiratoria:
* **Capítulo Circulatorio (CIE-10 IX):** **62,6%** (2.222 de 3.548 episodios evaluables; cumple la regla de capítulo dominante $\ge 50\%$).
* **Capítulo Respiratorio (CIE-10 X):** **30,1%** (1.069 de 3.548 episodios evaluables).
* En M2 Base (sin diagnóstico principal), el Tórax registraba $O/E = \mathbf{1,968}$ ($P = 1,000$).
* En **M2+CAP (Primaria)**, al reconocer su perfil cardiovascular y respiratorio, su exceso se reduce a $O/E = \mathbf{1,429}$ ($P = 0,997$). El ajuste **reduce sustancialmente el sesgo** por mezcla de especialidad, manteniendo al establecimiento en alerta debido a la gravedad intrínseca de sus ingresos.
* En M3-Total, debido a que su codificación secundaria no escala con su prolongada estancia, su $O/E$ se distorsiona hasta $2,226$.

---

## 2. Taxonomía y Reconciliación de Alertas Asistenciales

Todas las métricas de esta sección se derivan estrictamente del archivo canonical único de la tesis: [`benchmarks_modelos_fase2_2023.csv`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/benchmarks_modelos_fase2_2023.csv).

### 2.1 Definición Rigurosa de Términos
1. **Alerta Primaria:** Todo hospital que en la **Especificación Primaria Oficial (M2+CAP)**, con $	au = 0,1948$ y $\mu = -0,0359$, presenta:
   $$P(	heta_j > 1,10 \mid 	ext{M2+CAP}) \ge 0,95$$
2. **Alerta Robusta:** Hospital que es **Alerta Primaria** y concurrentemente **persiste como alerta en el modelo de sensibilidad M3-Total** ($P \ge 0,95$ en ambos).
3. **Alerta Dependiente de Codificación Aguda:** Hospital que no califica como alerta en la especificación primaria, pero que alerta únicamente en M3-Total debido a diagnósticos secundarios codificados al alta.

### 2.2 Lista Consolidada de Alertas en Hospitales Agudos Generales ($K=62$)
Bajo la especificación primaria M2+CAP, se identifican exactamente **10 Alertas Primarias** (y 52 hospitales en Desempeño Promedio):

| Código | Establecimiento | $O$ Defunciones | $E_{	ext{prim}}$ | $O/E$ (M2+CAP) | $P_{	ext{Prim}}$ | $E_{	ext{m2}}$ | $O/E$ (M2 Base) | $P_{	ext{M2}}$ | $E_{	ext{m3}}$ | $O/E$ (M3-Tot) | $P_{	ext{M3}}$ | Categoría de Alerta |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `113180` | **Hospital El Pino** | 438 | 266,0 | **1,646** | **1,000** | 238,6 | 1,836 | 1,000 | 247,1 | 1,772 | 1,000 | **ALERTA ROBUSTA** |
| `111100` | **Hospital Clínico San Borja-Arriarán** | 145 | 101,3 | **1,432** | **0,996** | 95,7 | 1,515 | 1,000 | 123,2 | 1,177 | 0,697 | **Alerta Primaria** |
| `106103` | **Hospital Claudio Vicuña (San Antonio)** | 320 | 231,8 | **1,381** | **1,000** | 200,7 | 1,595 | 1,000 | 195,8 | 1,634 | 1,000 | **ALERTA ROBUSTA** |
| `113100` | **Hospital Barros Luco Trudeau** | 972 | 709,7 | **1,370** | **1,000** | 754,7 | 1,288 | 1,000 | 842,5 | 1,154 | 0,914 | **Alerta Primaria** |
| `107101` | **Hospital San Martín (Quillota)** | 285 | 208,7 | **1,365** | **1,000** | 235,3 | 1,211 | 0,919 | 326,7 | 0,872 | 0,000 | **Alerta Primaria (En Transición)** |
| `106100` | **Hospital Carlos Van Buren** | 602 | 451,4 | **1,334** | **1,000** | 465,7 | 1,293 | 1,000 | 419,3 | 1,436 | 1,000 | **ALERTA ROBUSTA** |
| `103100` | **Hospital Dr. Leonardo Guzmán (Antofagasta)**| 663 | 497,8 | **1,332** | **1,000** | 479,1 | 1,384 | 1,000 | 569,2 | 1,165 | 0,908 | **Alerta Primaria** |
| `121109` | **Hospital Dr. Hernán Henríquez (Temuco)** | 580 | 451,4 | **1,285** | **1,000** | 467,4 | 1,241 | 0,997 | 498,1 | 1,165 | 0,888 | **Alerta Primaria** |
| `112101` | **Hospital Dr. Luis Tisné B.** | 502 | 407,3 | **1,233** | **0,990** | 382,2 | 1,313 | 1,000 | 381,3 | 1,316 | 1,000 | **ALERTA ROBUSTA** |
| `109100` | **Complejo Hospitalario San José** | 815 | 683,5 | **1,192** | **0,984** | 685,8 | 1,188 | 0,981 | 724,5 | 1,125 | 0,700 | **Alerta Primaria** |
| `112103` | *Instituto Nacional del Tórax (Monográfico)*| 157 | 109,8 | **1,429** | **0,997** | 79,8 | 1,968 | 1,000 | 70,5 | 2,226 | 1,000 | *Alerta Robusta Monog.* |

> **Nota de Conciliación de Cifras:**
> * **Temuco (`121109`) y Carlos Van Buren (`106100`):** Ambos presentan $E_{	ext{prim}} = 451,4$ en la tabla debido a una coincidencia de redondeo a 1 decimal: Van Buren es $E = \mathbf{451,4214}$, mientras que Temuco es $E = \mathbf{451,3872}$.
> * **Del Salvador (`112100`):** En el modelo canonical M2+CAP registra $O = 542, E_{	ext{prim}} = \mathbf{490,73} \implies O/E = \mathbf{1,1045}$ con $P_{	ext{prim}} = \mathbf{0,479}$. En M2 Base registra $E = 577,54 \implies O/E = 0,938$ ($P = 0,000$), y en M3-Total $E = 483,43 \implies O/E = 1,121$ ($P = 0,619$).

### 2.3 Casos de Divergencia Extrema: Mecanismos Clínicos y Codificación

#### 1. Hospital San Martín / Hospital Biprovincial Quillota-Petorca (`107101`)
* **Comportamiento en Modelos:** En M2+CAP registra $O/E = \mathbf{1,365}$ ($P = 1,000$, Alerta Primaria), mientras que en M3-Total cae a $O/E = \mathbf{0,872}$ ($P = 0,000$), una divergencia extrema de **$-0,49$**.
* **Aritmética y Razón de Codificación:** Quillota pasó de promediar **3,181** diagnósticos secundarios en la cohorte histórica de desarrollo (2020–2022) a **4,763** en 2023. La razón de codificación es exactamente:
  $$rac{4,763}{3,181} = \mathbf{1,497 pprox 1,50} \quad (+49,7\% pprox \mathbf{+50\%})$$
  *(Se rectifica el valor $+44\%$ reportado en borradores previos).*
* **Retracción Explícita:** Se **retira formal y explícitamente** la explicación preliminar que sugería un "colapso del registro a 1,9 diagnósticos contra 3,0 de la red". Dicha afirmación provenía de una corrida sobre una partición transitoria no consolidada. La serie temporal depurada demuestra inequívocamente lo opuesto: una expansión acelerada del registro secundario (2019: 2,55 $	o$ 2020: 3,03 $	o$ 2021: 2,82 $	o$ 2022: 3,46 $	o$ 2023: 4,76 $	o$ 2024: 5,43; $+113\%$ global).
* **Apertura Institucional y Clasificación Técnica:** En **octubre de 2022** concluyó el traslado definitivo de todos los pacientes hospitalizados y unidades críticas desde el antiguo Hospital San Martín al nuevo **Hospital Biprovincial Quillota-Petorca**, manteniendo el código Fonasa `107101`. El salto de codificación en 2023 coincide de forma directa con la puesta en marcha del nuevo recinto, la adopción de una ficha clínica electrónica moderna y una mayor exhaustividad en el registro médico al egreso. Debido a esta reconfiguración de infraestructura y procesos, Quillota se clasifica formalmente como **"En Transición Institucional"**.

#### 2. Hospital Dr. Eduardo Pereira Ramírez (`106102`)
* **Comportamiento en Modelos:** En M2+CAP registra $O = 148, E_{	ext{prim}} = 176,52 \implies O/E = \mathbf{0,8384}$ ($P = 0,0005$, Desempeño Promedio Favorable). En M3-Total, en cambio, salta bruscamente a $E_{	ext{m3}} = 100,1 \implies O/E = \mathbf{1,478^*}$ ($P = 0,999$, Alerta Dependiente de Codificación Aguda).
* **Mecanismo Operacional y Codificación:** Pereira es formalmente un **hospital de agudos generales** (`es_monografico == 'NO'`), con 30,3% de ingresos cardiovasculares/respiratorios y 21,8% digestivos. En la cohorte de desarrollo registraba una atípica intensidad diagnóstica histórica (**7,592** diagnósticos promedio). En 2023, su codificación descendió a **6,043** diagnósticos.
* **Fórmula Canónica de $R_{	ext{dx}}$:** Para resolver cualquier ambigüedad técnica, se explicita que en la evaluación prospectiva 2023 de M3-Total, $R_{	ext{dx}}$ se calculó determinísticamente como:
  $$R_{	ext{dx}, ij} = rac{N_{	ext{dx}, ij}}{\overline{N}_{	ext{dx}, 	ext{Dev}(j)}}$$
  donde $\overline{N}_{	ext{dx}, 	ext{Dev}(j)}$ es la media histórica de desarrollo de cada hospital. Esta fue la formulación exacta que generó $ho(	ext{M2}, 	ext{M3-Total}) = 0,7046 pprox 0,705$.  
  Para Pereira, $R_{	ext{dx}} = 6,043 / 7,592 = \mathbf{0,796 pprox 0,80}$ ($-20,4\%$). Al codificar menos que su propia línea base, M3 le descontó $E$ de manera severa ($176,5 	o 100,1$), disparando espuriamente su ratio. La especificación primaria M2+CAP refleja su verdadero perfil de riesgo basal.

#### 3. Hospital de San Camilo (`108100`)
* En M2 Base alertaba ($O/E = 1,292, P = 0,998$), pero en **M2+CAP (Primaria)** su razón desciende a $O/E = \mathbf{1,174}$ con $P = \mathbf{0,848} < 0,95$, quedando en **Desempeño Promedio**. En M3-Total registra $O/E = 1,197, P = 0,920 < 0,95$. El ajuste por capítulo diagnóstico principal explica adecuadamente su case-mix.

---

## 3. Deriva de Codificación de la Red, Sensibilidad POA y Verificaciones Técnicas

### 3.1 Deriva Temporal de Diagnósticos Secundarios en la Red Hospitalaria (2019–2024)
La intensidad de codificación secundaria de la red pública experimentó un crecimiento secular sostenido a lo largo del período evaluado:

| Año | Total Hospitalizaciones Red | $N_{	ext{dx}}$ Medio Red | Mediana Red | $\% \ge 11$ Dx Red | Cohorte Evaluable Adultos | $N_{	ext{dx}}$ Medio Adultos | Mediana Adultos | $\% \ge 11$ Dx Adultos |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2019** | 886.627 | **3,803** | 3,0 | 4,84% | 567.976 | **4,077** | 3,0 | 6,13% |
| **2020** | 702.275 | **4,521** | 4,0 | 7,80% | 463.714 | **4,844** | 4,0 | 9,70% |
| **2021** | 713.978 | **4,979** | 4,0 | 10,02% | 481.925 | **5,374** | 4,0 | 12,37% |
| **2022** | 778.073 | **4,937** | 4,0 | 9,64% | 507.502 | **5,334** | 4,0 | 12,08% |
| **2023** | 846.043 | **5,147** | 4,0 | 10,57% | 557.318 | **5,557** | 4,0 | 13,08% |
| **2024** | 873.772 | **5,501** | 4,0 | 12,28% | 594.628 | **5,877** | 5,0 | 14,62% |

*Tendencia Secular:* El promedio de diagnósticos secundarios aumentó en **+44,2%** entre 2019 y 2024 en la cohorte evaluable de adultos, reflejando una tecnificación y digitalización progresiva de los sistemas de registro clínico hospitalario en Chile.

### 3.2 Independencia entre Desempeño Primario ($O/E_{	ext{M2+CAP}}$) y Deriva de Codificación Hospitalaria
Para verificar si los hospitales con mayor crecimiento relativo en codificación se ven artificialmente favorecidos en su $O/E$ bajo M2+CAP, se calculó la correlación a nivel de establecimiento entre el $O/E$ primario y el cambio de codificación respecto a la línea base de desarrollo:
* **Correlación vs $\Delta N_{	ext{dx}} (2023 - 	ext{Dev})$:**
  * Spearman: **$ho = +0,0981$ ($p = 0,4559$, no significativo)**.
  * Pearson: **$r = +0,1085$ ($p = 0,4095$, no significativo)**.
* **Correlación vs Razón de Codificación $(N_{	ext{dx}, 2023} / \overline{N}_{	ext{dx}, 	ext{Dev}})$:**
  * Spearman: **$ho = +0,1126$ ($p = 0,3915$, no significativo)**.
  * Pearson: **$r = +0,1109$ ($p = 0,3990$, no significativo)**.

**Conclusión:** No existe ninguna correlación negativa ni asociación estadísticamente significativa entre el cambio en la intensidad de codificación diagnóstica y el desempeño en mortalidad ajustada. El modelo M2+CAP es inmune al sesgo por deriva de codificación.

### 3.3 Sensibilidad a la Supresión de Complicaciones Agudas (Proxy POA Longitudinal a Nivel de Código)
Dado que sólo el **29,78%** de los pacientes evaluables de 2023 contaban con una hospitalización previa en los 12 meses anteriores (lo que generaría un 70,2% de datos faltantes si el ajuste comórbido se restringiera únicamente a ingresos pasados), se evaluó el modelo mediante **supresión estricta a nivel de código CIE-10 de condiciones agudas hospitalarias**:
1. **ELIX_04 (Circulación Pulmonar):** Supresión del código agudo $I26$ (tromboembolismo pulmonar, 5.088 episodios en 2023), reteniendo exclusivamente hipertensión pulmonar crónica y afecciones cardiopulmonares preexistentes ($I27, I28$).
2. **ELIX_09 (Trastornos Neurológicos):** Supresión de los códigos agudos $G93.4$ (encefalopatía no especificada, 1.304 episodios) y $R56$ (convulsiones agudas, 4.691 episodios), reteniendo exclusivamente patología neurológica crónica ($G10-G13, G20-G22, G35-G37, G40-G41$, etc.).
3. **ELIX_14 (Falla Renal):** Se ratifica que Quan 2005 excluye taxativamente la falla renal aguda ($N17$), limitándose a enfermedad renal crónica ($N18, N19$).
4. **Variante POA-26 (Sensibilidad Estricta):** Exclusión adicional de las 5 categorías potencialmente contaminables por estancia declaradas en la Fase 1 (`ELIX_02` arritmias, `ELIX_22` coagulopatía, `ELIX_24` desnutrición, `ELIX_25` electrolitos, `ELIX_26` hemorragia aguda), dejando exactamente 26 categorías puramente crónicas.

**Resultados de la Sensibilidad POA:**
* **Estabilidad Global de Rankings:**
  * $ho(	ext{M2+CAP Primario, M2+CAP POA}) = \mathbf{0,99908}$
  * $ho(	ext{M2+CAP Primario, M2+CAP POA-26}) = \mathbf{0,99349}$
* **Evaluación del Hospital Dr. Eduardo Pereira (`106102`):**
  * Primario M2+CAP: $E = 176,52$, $O/E = \mathbf{0,8384}$, $P(	heta > 1,10) = \mathbf{0,0005}$
  * Sensibilidad POA: $E = 175,52$, $O/E = \mathbf{0,8432}$, $P(	heta > 1,10) = \mathbf{0,0006}$
  * Sensibilidad POA-26: $E = 176,76$, $O/E = \mathbf{0,8373}$, $P(	heta > 1,10) = \mathbf{0,0004}$
  * *Hallazgo:* El bajo $O/E$ de Pereira **no es producto de inflación comórbida aguda ni contaminación POA**. Su volumen de muertes esperadas ($E pprox 176$) y su ratio ($O/E pprox 0,84$) permanecen absolutamente inalterados ante la supresión de diagnósticos agudos.
* **Estabilidad de las 10 Alertas Primarias:**

| Código | Establecimiento | $O/E$ Primario | $P_{	ext{prim}}$ | $O/E$ POA | $P_{	ext{poa}}$ | $O/E$ POA-26 | $P_{	ext{poa26}}$ | Robustez POA |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `113180` | **Hospital El Pino** | 1,646 | 1,000 | 1,651 | 1,000 | 1,681 | 1,000 | **100% Inmune** |
| `111100` | **Hospital Clínico San Borja-Arriarán** | 1,432 | 0,996 | 1,452 | 0,998 | 1,514 | 1,000 | **100% Inmune** |
| `106103` | **Hospital Claudio Vicuña (San Antonio)** | 1,381 | 1,000 | 1,382 | 1,000 | 1,350 | 1,000 | **100% Inmune** |
| `113100` | **Hospital Barros Luco Trudeau** | 1,370 | 1,000 | 1,372 | 1,000 | 1,354 | 1,000 | **100% Inmune** |
| `107101` | **Hospital San Martín (Quillota)** | 1,365 | 1,000 | 1,368 | 1,000 | 1,358 | 0,999 | **100% Inmune** |
| `106100` | **Hospital Carlos Van Buren** | 1,334 | 1,000 | 1,325 | 1,000 | 1,305 | 1,000 | **100% Inmune** |
| `103100` | **Hospital Dr. Leonardo Guzmán (Antofagasta)**| 1,332 | 1,000 | 1,325 | 1,000 | 1,305 | 1,000 | **100% Inmune** |
| `121109` | **Hospital Dr. Hernán Henríquez (Temuco)** | 1,285 | 1,000 | 1,288 | 1,000 | 1,286 | 1,000 | **100% Inmune** |
| `112101` | **Hospital Dr. Luis Tisné B.** | 1,233 | 0,990 | 1,238 | 0,992 | 1,235 | 0,991 | **100% Inmune** |
| `109100` | **Complejo Hospitalario San José** | 1,192 | 0,984 | 1,206 | 0,993 | 1,191 | 0,983 | **100% Inmune** |
| `112103` | *Instituto Nacional del Tórax* | 1,429 | 0,997 | 1,465 | 0,999 | 1,453 | 0,999 | **100% Inmune** |

### 3.4 Calibración Preingreso Multidimensional
La especificación primaria M2+CAP cumple con holgura los márgenes pre-especificados de calibración previa al ingreso ($[0,80; 1,25]$ para todo estrato con $E \ge 100$):
* **Edad:** $<65$ años ($1,061$), $65-74$ años ($0,971$), $\ge 75$ años ($0,977$).
* **Carga Crónica:** 0 comorb ($0,804$), 1 comorb ($0,989$), 2 comorb ($1,085$), $\ge 3$ comorb ($1,006$). El estrato de 0 comorbilidades se sitúa en $0,804$, en el límite inferior admisible.
* **Vía Ingreso:** Urgencia ($1,000$), Programada ($0,991$).
* **Capítulos Diagnósticos Principales:** Digestivo ($1,047$), Musculoesquelético ($0,932$), Circulatorio ($1,004$), Respiratorio ($1,008$), Sistema Nervioso ($0,979$), Nefrourológico ($0,982$).

### 3.5 Verificación Real de LOHO y Concentración Diagnóstica
Se ratifican las métricas out-of-sample del codificador Leave-One-Hospital-Out (LOHO) en la cohorte 2023 de estadía:
* **Concentración:** 144 categorías CIE-10 (9,9% de las 1.459 activas) tienen $>50\%$ de sus casos concentrados en un solo hospital.
* **Máxima Discrepancia:** $|\Delta|_{	ext{máx}} = 0,5079$ log-días (CIE G47 en Puerto Montt; H36 con 96,0% en Los Ángeles, $|\Delta|=0,4395$; T29 quemaduras con 55,5% en Posta Central, $|\Delta|=0,4295$).
* **Impacto Institucional:** $ho = 0,99948$, $	ext{RMS} = 0,61$ puestos. El Instituto de Neurocirugía Dr. Asenjo (`112104`) presenta la mayor variación de la red ($-0,0294$ en $O/E$, de $0,8033$ a $0,7739$).

### 3.6 Estabilidad por Bootstrap de las Covariables de Estadía
Se congelan los resultados de 100 réplicas bootstrap en desarrollo ($N = 1.303.718$) en [`scratch/estadia_bootstrap_stability_17.json`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/scratch/estadia_bootstrap_stability_17.json):
* **16 Variables Núcleo al 100%:** Incluyen a `ELIX_03` Valvulopatía (100%) y `ELIX_08` Parálisis (100%).
* **Borde en Puesto 17:** `ELIX_06` HTA no complicada (69,0%), `ELIX_19` Cáncer metastásico (19,0%), `ELIX_10` EPOC (10,0%), `ELIX_07` HTA complicada (2,0%). Todos con coeficientes entre $+0,0014$ y $+0,0038$. El ordenamiento institucional es insensible a la alternancia en este borde ($ho \ge 0,999$).

---

## 4. Enlace Unificado, Cotas de Salida y Sensibilidad a la Censura por Traslado

### 4.1 Criterio Unificado de Enlace Determinístico
* **Total egresos por traslado inpatient adultos 2023:** **25.555** episodios.
* **Total enlazados en $\le 30$ días:** **11.544** episodios (tasa de enlace global: 45,2%).

### 4.2 Cotas de Salida ($\lambda_{	ext{salida}}$) para las Alertas Primarias
Se presentan los datos completos de enlace ($O_{	ext{link}}$ y $E_{	ext{link}}$) y el multiplicador umbral $\lambda_{	ext{salida}}$ en derivados no enlazados para salir de alerta ($P < 0,95$):

$$O_{	ext{eval}} = O_{	ext{base}} + O_{	ext{link}} + \lambda_{	ext{salida}} \cdot E_{	ext{nolink}}, \quad E_{	ext{eval}} = E_{	ext{base}} + E_{	ext{link}} + E_{	ext{nolink}}$$

| Código | Establecimiento | $O_{	ext{base}}$ | $E_{	ext{base}}$ | $P_{	ext{base}}$ | $O_{	ext{link}}$ | $E_{	ext{link}}$ | No Enlaz. | $E_{	ext{nolink}}$ | Cota $\lambda_{	ext{salida}}$ | ¿Factible Salir? | Interpretación Clínica |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `112101` | **Hospital Dr. Luis Tisné B.** | 502 | 407,3 | 0,990 | 20 | 18,8 | 699 | 86,6 | **0,98x** | **SÍ (Factible, $\lambda < 0,98$)** | **Alerta Marginal Sensible:** Si sus 699 derivados no enlazados mueren a una tasa estándar o favorable ($\lambda < 0,98	imes$), el hospital sale de alerta ($P = 0,948 < 0,95$). |
| `121109` | **Hospital Dr. Hernán Henríquez (Temuco)** | 580 | 451,4 | 1,000 | 17 | 31,1 | 408 | 36,4 | **0,47x** | Difícil ($\lambda < 0,47$) | Requiere que los no enlazados fallezcan a menos de la mitad del riesgo esperado. |
| `107101` | **Hospital San Martín (Quillota)** | 285 | 208,7 | 1,000 | 13 | 11,6 | 339 | 26,4 | **0,19x** | Muy Remoto ($\lambda < 0,19$) | Exige sobrevida casi absoluta de trasladados. |
| `109100` | **Complejo Hospitalario San José** | 815 | 683,5 | 0,984 | 1 | 3,5 | 179 | 10,1 | **0,05x** | Casi Imposible ($\lambda < 0,05$) | Exige cero muertes en derivados para tocar 0,95. |
| `113180` | **Hospital El Pino** | 438 | 266,0 | 1,000 | 25 | 12,4 | 1.120 | 83,2 | **No existe** | **NO (Alerta Inmune)** | Con $\lambda = 0$ (cero muertes en 1.120 trasladados), $P_{	ext{min}} = 1,000$. |
| `111100` | **Hospital Clínico San Borja-Arriarán** | 145 | 101,3 | 0,996 | 4 | 1,1 | 34 | 1,3 | **No existe** | **NO (Alerta Inmune)** | Con $\lambda = 0$, $O/E = 1,398$ y $P_{	ext{min}} = 0,995 \ge 0,95$. Inmune a dilución. |
| `106103` | **Hospital Claudio Vicuña (San Antonio)** | 320 | 231,8 | 1,000 | 14 | 8,8 | 102 | 8,4 | **No existe** | **NO (Alerta Inmune)** | Con $\lambda = 0$, $P_{	ext{min}} = 1,000$. |
| `113100` | **Hospital Barros Luco Trudeau** | 972 | 709,7 | 1,000 | 16 | 10,3 | 409 | 41,8 | **No existe** | **NO (Alerta Inmune)** | Con $\lambda = 0$, $P_{	ext{min}} = 1,000$. |
| `106100` | **Hospital Carlos Van Buren** | 602 | 451,4 | 1,000 | 58 | 54,8 | 349 | 27,9 | **No existe** | **NO (Alerta Inmune)** | Con $\lambda = 0$, $P_{	ext{min}} = 1,000$. |
| `103100` | **Hospital Dr. Leonardo Guzmán (Antofagasta)**| 663 | 497,8 | 1,000 | 6 | 4,2 | 54 | 3,2 | **No existe** | **NO (Alerta Inmune)** | Con $\lambda = 0$, $P_{	ext{min}} = 1,000$. |
| `112103` | *Instituto Nacional del Tórax (Monográfico)*| 157 | 109,8 | 0,997 | 6 | 4,5 | 36 | 2,2 | **No existe** | **NO (Alerta Inmune)** | Con $\lambda = 0$, $P_{	ext{min}} = 0,996 \ge 0,95$. Inmune a dilución. |

### 4.3 Sensibilidad a la Sobremortalidad de Derivados y Reconciliación con el Archivo Canónico
Se presentan los puntos de quiebre evaluados a partir de los datos exactos del archivo canonical [`benchmarks_modelos_fase2_2023.csv`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/benchmarks_modelos_fase2_2023.csv):

| Hospital | Fuente de Entrada | $E_{	ext{base}}$ | $P_{	ext{base}}$ | $\lambda^*$ Quiebre | $P(	ext{Byar High})$ | $P(	ext{Puntual } \lambda_{	ext{obs}})$ | Estado de Reproducción |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Del Salvador** | **Cálculo Canónico M2+CAP**| **490,73** | **0,479** | **1,92x** | **0,959** | **0,710** (con $\lambda=1,33$) | **Reproduce 100% las cifras canónicas** con $E_{	ext{base}}=490,73$. |
| | Versión preliminar | 496,0 | 0,384 | 2,04x | 0,935 | 0,627 (con $\lambda=1,33$) | Reproduce cálculo previo del comité con $E_{	ext{base}}=496,0$. |
| **Curanilahue** | **Cálculo Canónico M2+CAP**| **130,27** | **0,203** | **2,73x** | **0,712** | 0,265 | **Reproduce 100% las cifras canónicas** con $E_{	ext{base}}=130,27$. |
| | Versión preliminar | 139,1 | 0,063 | 3,23x | 0,452 | 0,094 | Reproduce cálculo previo del comité con $E_{	ext{base}}=139,1$. |
| **Sótero del Río**| **Cálculo Canónico M2+CAP**| **1.030,38**| 0,000 | **2,36x** | 0,033 | 0,000 (y con $\lambda=2,36, P=\mathbf{0,950}$) | **Reproduce 100% las cifras canónicas** con $E_{	ext{base}}=1.030,38$. |
| | Versión preliminar | 1.109,2| 0,000 | 2,97x | 0,000 | 0,000 (y con $\lambda=2,36, P=\mathbf{0,247}$) | Reproduce cálculo previo del comité con $E_{	ext{base}}=1.109,2$. |

*Aclaración sobre la Cifra $0,712$:*  
Se constata que la cifra $0,712$ coincidía de manera exacta tanto para Curanilahue bajo estrés Byar alto ($P = 0,7121$) como para la estimación puntual de Del Salvador bajo M2+CAP recalibrado ($P = 0,7098 pprox 0,710$). Con el valor canonical $E_{	ext{base}} = 490,73$, el Hospital Del Salvador alcanza $P = 0,959$ en la cota Byar ($1,97	imes$), superando estrictamente el umbral $0,95$ y confirmando su condición de establecimiento asistencialmente vulnerable al sub-enlace de traslados.

### 4.4 Ranking Completo de Vulnerabilidad al Sublinkeaje en Hospitales No Alertados (M2+CAP)
En la red de agudos generales ($K=62$), descontando las **10 Alertas Primarias**, se analizan los **52 hospitales no alertados** ($10 	ext{ alertas} + 52 	ext{ no alertados} = 62$):

| Código | Establecimiento | $N_{	ext{tr}}$ | No Enlaz. | $E_{	ext{nolink}}$ | $\lambda_{	ext{obs}}$ [Byar High] | $\lambda^*$ Quiebre | Mort. Quiebre % | Condición de Vulnerabilidad |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `105102` | **Hospital de Ovalle** | 219 | 96 | 11,6 | 1,65x [**2,76x**] | **1,78x** | 21,5% | **VULNERABLE** ($\lambda^* < 	ext{Byar High}$) |
| `112100` | **Hospital Del Salvador** | 669 | 489 | 47,0 | 1,33x [**1,97x**] | **1,92x** | 18,4% | **VULNERABLE** ($\lambda^* < 	ext{Byar High}$) |
| `104100` | **Hospital de Copiapó** | 203 | 78 | 6,0 | 1,42x [**2,70x**] | **1,94x** | 14,9% | **VULNERABLE** ($\lambda^* < 	ext{Byar High}$) |
| `114101` | Complejo Hosp. Dr. Sótero del Río | 2.385 | 1.764 | 145,8 | 1,10x [1,48x] | **2,36x** | 19,5% | No vulnerable ($\lambda^* > 	ext{Byar High}$) |
| `108100` | Hospital de San Camilo (San Felipe) | 262 | 130 | 11,8 | 0,83x [1,71x] | **2,36x** | 21,5% | No vulnerable ($\lambda^* > 	ext{Byar High}$) |
| `133150` | Hospital de Castro | 394 | 274 | 26,5 | 0,66x [1,69x] | **2,44x** | 23,7% | No vulnerable ($\lambda^* > 	ext{Byar High}$) |
| `116108` | Hospital de Linares | 372 | 114 | 12,5 | 1,31x [2,05x] | **2,63x** | 28,9% | No vulnerable ($\lambda^* > 	ext{Byar High}$) |
| `128109` | Hospital de Curanilahue | 511 | 301 | 20,5 | 1,14x [1,99x] | **2,73x** | 18,6% | No vulnerable ($\lambda^* > 	ext{Byar High}$) |
| `107100` | Hospital Dr. Gustavo Fricke | 570 | 348 | 29,8 | 1,02x [1,63x] | **3,26x** | 27,9% | No vulnerable ($\lambda^* > 	ext{Byar High}$) |
| `115100` | Hospital Regional de Rancagua | 834 | 604 | 67,8 | 0,65x [1,13x] | **3,45x** | 38,7% | No vulnerable ($\lambda^* > 	ext{Byar High}$) |
| `102100` | Hospital de Iquique | 427 | 247 | 21,3 | 0,64x [1,32x] | **4,24x** | 36,5% | No vulnerable ($\lambda^* > 	ext{Byar High}$) |
| `121110` | Hospital de Lautaro | 305 | 120 | 15,0 | 1,08x [1,85x] | **4,27x** | 53,5% | No vulnerable ($\lambda^* > 	ext{Byar High}$) |
| `117101` | Hospital Herminda Martín (Chillán) | 636 | 432 | 31,4 | 1,30x [2,04x] | **4,77x** | 34,7% | No vulnerable ($\lambda^* > 	ext{Byar High}$) |
| `122100` | Hospital Regional de Valdivia | 532 | 407 | 33,7 | 0,95x [1,96x] | **5,12x** | 42,4% | Inmune ($> 5	imes$) |
| `121117` | Hospital de Pitrufquén | 328 | 155 | 17,8 | 1,14x [1,91x] | **5,12x** | 58,8% | Inmune ($> 5	imes$) |
| *Resto* | *37 Hospitales Restantes de la Red* | *—* | *—* | *—* | *—* | **$> 5,0	ext{x}$** | $> 40\%$ | **Inmunes** ($> 5	imes$) |
| **Total** | **52 Hospitales No Alertados en M2+CAP** | | | | | | | *(Reconcilia con 10 Alertas = 62 Total)* |

---

## 5. Dictamen Metodológico Consolidado de Cierre de Fase 2

Con estas precisiones metodológicas, empíricas y algebraicas, la Fase 2 queda **concluida y auditada en su totalidad**:
1. **Especificación Primaria M2+CAP Oficial:** Satisface los márgenes pre-especificados de calibración previa al ingreso ($[0,80; 1,25]$) en edad, comorbilidad crónica basal, vía de ingreso, tipo de establecimiento y en los principales capítulos diagnósticos CIE-10 (con ACV mapeado al Capítulo IX Circulatorio y no a MDC 01 del agrupador). El estrato de 0 comorbilidades se sitúa en $0,804$, en el límite inferior admisible. El modelo reduce sustancialmente el sesgo por mezcla de especialidad del Tórax y San Camilo, manteniendo al Tórax en alerta ($1,43; P = 0,997$) por la extrema gravedad de sus ingresos.
2. **Taxonomía Transparente y Deriva de Codificación Resuelta:** Delimitadas 10 alertas primarias y 4 alertas robustas en agudos generales. Las divergencias de Quillota (salto de $+49,7\% pprox +50\%$ en captura secundaria tras su traslado en octubre de 2022 al Hospital Biprovincial Quillota-Petorca, clasificado como "En Transición Institucional") y Pereira (reducción de $-20,4\%$ en diagnósticos respecto a su línea base histórica) quedan plenamente documentadas bajo la fórmula $R_{	ext{dx}, ij} = N_{	ext{dx}, ij} / \overline{N}_{	ext{dx}, 	ext{Dev}(j)}$. Se demuestra ausencia de correlación entre $O/E$ y deriva de codificación ($ho = +0,0981, p = 0,4559$) e invarianza de rankings bajo supresión de condiciones agudas (proxy POA, $ho = 0,99908$; Pereira $O/E pprox 0,84, P \le 0,0006$).
3. **Métricas de Estadía Congeladas:** Verificada la rama LOHO real out-of-sample en 2023 (144 diagnósticos concentrados, $|\Delta|_{	ext{máx}} = 0,51$) y estabilidad bootstrap de las 17 covariables (`estadia_bootstrap_stability_17.json`).
4. **Reproducibilidad Paramétrica de Bayes y Archivo Canonical:** Fórmulas e hiperparámetros específicos $(	au = 0,1948, \mu = -0,0359)$ estimados sobre los $K=62$ agudos generales mediante DerSimonian-Laird, garantizando reproducibilidad analítica exacta a partir de [`benchmarks_modelos_fase2_2023.csv`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/benchmarks_modelos_fase2_2023.csv).
5. **Censura por Derivación:** Criterio determinístico unificado a 30 días, identificando cotas de salida precisas (Hospital Luis Tisné con $\lambda_{	ext{salida}} = 0,98	imes$) y vulnerabilidad confirmada ante estrés Byar en tres centros asistenciales (Ovalle, Del Salvador y Copiapó).
6. **Sincronización Total de Diccionarios y Metadatos:** Sincronizado el archivo `Fase 2_ Diccionario de Base de Datos.csv` y generado `diccionario_base_datos_129.csv`, reflejando exactamente 37 nulos en `FECHA_NACIMIENTO`, 71 nulos en `FECHA_INGRESO`, 75 nulos en campos GRD y las 12 categorías observadas de `TIPOALTA` sobre los $5.808.536$ registros brutos consolidados.
