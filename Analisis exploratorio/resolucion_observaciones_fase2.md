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

## 1. Errores Subsanados y Unificación de Cifras Exactas

### 1.1 Cascada CONSORT Secuencial Estricta (Sin Solapamientos)
La aparente inconsistencia previa se originó porque las exclusiones de estancia cero (`EX05`) y cirugía mayor ambulatoria (`EX07`) se habían reportado como conteos univariados marginales, solapando a 827.687 pacientes que compartían ambas condiciones. 

La siguiente es la **cascada CONSORT estrictamente secuencial**, donde cada paso descuenta exactamente del remanente del paso inmediatamente anterior sobre los **5.808.536 episodios brutos**:

```text
========================================================================================================
Paso 0: Egresos Totales Brutos Registrados en FONASA (2019–2024)        N = 5.808.536 (100.0%)
========================================================================================================
  │
  ├── [-5.073]    Paso 1: EX01 (GRD no agrupable o inválido: MDC nulo/0/99/DESCONOCIDO)
  │               Quedan: 5.803.463 episodios
  │
  ├── [-146.928]  Paso 2: EX02 (Recién nacidos y período perinatal sanos: MDC 15)
  │               Quedan: 5.656.535 episodios
  │
  └── [-431.832]  Paso 3: EX03 (Obstetricia sin complicación: MDC 14, 0 comorb. Elixhauser, no fallecida)
                  Quedan: 5.224.703 episodios
========================================================================================================
COHORTE BASE ANALÍTICA (COHORTE DURA)                                   N = 5.224.703 (89.95%)
(Verificación aritmética: 5.808.536 - 5.073 - 146.928 - 431.832 = 5.224.703 exactos)
========================================================================================================
  │
  ├── [RAMA 1: EVALUACIÓN DE MORTALIDAD INTRAHOSPITALARIA]
  │     Total en Cohorte Base: 5.224.703 episodios
  │     │
  │     └── [-342.861]  Exclusión por Censura Estadística Informativa:
  │                     - Derivaciones a otros hospitales de la red/privados: 205.998
  │                     - Hospitalización domiciliaria (cuidados activos en hogar): 136.783
  │                     - Alta no identificada / desconocida: 80
  │                     ============================================================================
  │                     COHORTE EVALUACIÓN MORTALIDAD NO CENSURADA: N = 4.881.842 episodios
  │                     - Sobrevivientes confirmados a domicilio/alta voluntaria: 4.714.136 (96.56%)
  │                     - Fallecidos intrahospitalarios confirmados:              167.706 (3.44%)
  │
  └── [RAMA 2: EVALUACIÓN DE ESTANCIA HOSPITALARIA (SECUENCIAL ESTRICTA)]
        Total en Cohorte Base: 5.224.703 episodios
        │
        ├── [-167.706]   Paso E1: EX06 (Fallecidos intrahospitalarios; evita premiar muerte precoz)
        │                Quedan: 5.056.997 episodios
        │
        ├── [-342.861]   Paso E2: Censurados / Traslados no concluidos (estancia física no cerrada)
        │                Quedan: 4.714.136 episodios
        │
        ├── [-1.095.546] Paso E3: EX05 (Estancia 0 días ambulatoria en sobrevivientes)
        │                Quedan: 3.618.590 episodios
        │
        └── [-57.055]    Paso E4: EX07 (Cirugía Mayor Ambulatoria restante con estancia > 0 días)
                         Quedan: 3.561.535 episodios
        ============================================================================================
        COHORTE FINAL EVALUACIÓN ESTANCIA HOSPITALARIA: N = 3.561.535 episodios (61.32%)
        (Verificación aritmética: 5.224.703 - 167.706 - 342.861 - 1.095.546 - 57.055 = 3.561.535 exactos)
========================================================================================================
```

*Nota Metodológica sobre `TIPO_ACTIVIDAD` y Día 0:*
En la cohorte base existen **12.630 pacientes que fallecieron en el día 0** de hospitalización ($<24\text{h}$ tras el ingreso). Excluir el día 0 del modelo de mortalidad habría introducido un severo **sesgo de tiempo inmortal**, eliminando el 7.5% de todas las muertes intrahospitalarias. Por ello, el día 0 se preserva en la rama de mortalidad y se descuenta únicamente en la rama de estancia para evaluar el consumo de camas de pacientes internados.

### 1.2 Conteo Directo de Maternidad y Neonatología (MDC 14 y MDC 15)
Se descartó la estimación indirecta anterior. El conteo directo por código en la base revela:
* **MDC 15 (Recién Nacidos y Período Perinatal):**
  * Volumen bruto: **146.928 episodios**.
  * En Cohorte Dura final: **0 episodios (100.0% excluidos)**. Ningún neonato entra al modelo, debido a que el Índice de Elixhauser fue desarrollado y validado exclusivamente para adultos.
* **MDC 14 (Embarazo, Parto y Puerperio):**
  * Volumen bruto: **640.429 episodios**.
  * En Cohorte Dura final: **206.234 episodios conservados**. Corresponden a partos con complicaciones severas, cesáreas complejas, eclampsia o pacientes con comorbilidades basales crónicas.
  * Excluidos: **434.195 episodios de obstetricia no complicada** (431.832 en CONSORT secuencial).
* **Campo `PESORN1` (795.973 registros no nulos en bruto):** Se registra tanto en el egreso materno como en el del neonato hospitalizado. En la cohorte dura final, solo subsiste como variable descriptiva descartada para el modelo predictivo.

### 1.3 Matriz Maestra de 129 Columnas con Conteos Exactos por Código
El archivo [`Analisis exploratorio/matriz_maestra_129_columnas.csv`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/matriz_maestra_129_columnas.csv) fue recalculado fila por fila directamente desde los datos Parquet en Bronze. Sus conteos coinciden al dígito con el diccionario oficial:
* `DIAGNOSTICO1`: 5.808.456 no nulos (80 nulos).
* `DIAGNOSTICO2`: **5.013.342 no nulos** (795.194 nulos).
* `FECHATRASLADO1`: **1.001.017 no nulos**.
* `FECHATRASLADO9`: **973 no nulos**.
* `HOSPPROCEDENCIA`: **703.854 no nulos**.
* Total columnas: **exactamente 129**.

### 1.4 Aclaración sobre las Categorías de `TIPOALTA`
* El manual normativo DEIS de FONASA lista **18 categorías nominales teóricas**.
* En la base empírica de 5.808.536 registros (2019–2024), **únicamente 12 categorías presentan registros ($N > 0$)**. Las 6 categorías restantes (ej. "ALTA POR TRASLADO EXTRAORDINARIO", "ALTA POR DISPOSICIÓN JUDICIAL") tienen **0 registros** en todo el sexenio.
* **Porcentajes Brutos vs. Condicionales:**
  * El valor `DOMICILIO` representa el **89.62% del total bruto** (5.205.550 / 5.808.536), y el **96.56% de la cohorte de mortalidad no censurada** (4.714.136 / 4.881.842).
* **Total de Derivaciones Externas:** Suman **215.851 episodios** (123.285 a hospital del mismo servicio, 44.429 a hospital de la red nacional, 22.041 a otros centros residenciales/cárcel, y 26.096 a prestadores privados). La cifra histórica de 89.438 correspondía a una anualidad previa.

---

## 2. Reglas de Atribución de Traslados y Cadenas Inter-Hospitalarias

### 2.1 La Regla de Hospital Índice de CMS frente a la Regla de 48 Horas
Siguiendo las observaciones del revisor:
1. **Regla de Hospital Índice (Estándar CMS / NIH PMC3319769):**
   * Metodología de CMS para medidas de mortalidad a 30 días en ACV, IAM e Insuficiencia Cardíaca: todos los traslados continuos se consolidan en un único episodio y el desenlace final se atribuye **100% al hospital que ingresó inicialmente al paciente (Hospital Índice)**.
2. **Propuesta Clínica Pragmática (Regla de 48 Horas):**
   * En el sistema público chileno, los hospitales provinciales de baja complejidad derivan pacientes inestables a hospitales metropolitanos. Si un paciente llega en shock refractario y fallece en $<48\text{h}$, la responsabilidad clínica recae en el emisor. Sin embargo, si el paciente sobrevive a la fase aguda y fallece tras 15 días en la UCI del receptor por una neumonía nosocomial, atribuir la defunción al emisor desincentiva la derivación oportuna y distorsiona el O/E del hospital receptor.
3. **Modelo Fraccional (Estudio Noruego / DOAJ):**
   * Distribución ponderada del desenlace según los días cama consumidos en cada establecimiento ($w_A = \text{LOS}_A / \text{LOS}_{\text{total}}$, $w_B = \text{LOS}_B / \text{LOS}_{\text{total}}$).
4. **Descontaminación del Historial:**
   * Las variables `N_EGRESOS_12M` y `DIAS_DESDE_EGRESO_PREVIO` colapsan la cadena: el episodio del emisor **NO se cuenta como reingreso previo** para el hospital receptor.
   * De las 215.851 derivaciones, el 68.4% logra vincularse determinísticamente por `CIP_ENCRIPTADO` con una admisión receptora en $\le 1$ día. Los 2.044 registros con `CIP` nulo se excluyen de la vinculación y se procesan con `HISTORIA_DISPONIBLE = 0`.

---

## 3. Comorbilidades de Elixhauser: Especificación Primaria vs. Sensibilidad

Para resolver la contradicción de que las complicaciones aparezcan como predictoras basales y en el Score de van Walraven, se establece la siguiente estructura formal:

### 3.1 Tabla Canónica de las 31 Condiciones (Quan et al., 2005)
Guardada en [`Analisis exploratorio/tabla_canonica_elixhauser_31.csv`](file:///home/felipe/Documentos/Proyecto%20final/Tesis/Analisis%20exploratorio/tabla_canonica_elixhauser_31.csv):

| Código | Condición Clínica | Códigos CIE-10 (Quan et al., 2005) | Peso vW | Clasificación Operativa |
|:---:|---|---|:---:|:---:|
| **ELIX_01** | Insuficiencia cardíaca congestiva | I09.9, I11.0, I13.0, I13.2, I25.5, I42.0, I50.x | +7 | Crónica Preexistente |
| **ELIX_02** | Arritmias cardíacas | I44.1-I44.3, I45.6, I47.x-I49.x, R00.0, T82.1 | +5 | **Potencial Complicación** |
| **ELIX_03** | Valvulopatía | I05.x-I08.x, I34.x-I39.x, Z95.2-Z95.4 | -1 | Crónica Preexistente |
| **ELIX_04** | Trastornos circulación pulmonar | I26.x, I27.x, I28.0, I28.8, I28.9 | +4 | Crónica Preexistente |
| **ELIX_05** | Enfermedad vascular periférica | I70.x, I71.x, I73.1, I73.8, K55.1, Z95.8 | +2 | Crónica Preexistente |
| **ELIX_06** | Hipertensión no complicada | I10.x | 0 | Crónica Preexistente |
| **ELIX_07** | Hipertensión complicada | I11.x, I12.x, I13.x, I15.x | 0 | Crónica Preexistente |
| **ELIX_08** | Parálisis | G04.1, G11.4, G80.1, G81.x, G82.x, G83.x | +7 | Crónica Preexistente |
| **ELIX_09** | Otros trastornos neurológicos | G10.x-G13.x, G20.x-G22.x, G35.x-G37.x, G40.x | +6 | Crónica Preexistente |
| **ELIX_10** | Enfermedad pulmonar crónica | I27.8, J40.x-J47.x, J60.x-J67.x, J70.1 | +3 | Crónica Preexistente |
| **ELIX_11** | Diabetes no complicada | E10.0, E10.1, E11.0, E11.1, E13.0, E14.0 | 0 | Crónica Preexistente |
| **ELIX_12** | Diabetes complicada | E10.2-E10.8, E11.2-E11.8, E13.2-E13.8 | 0 | Crónica Preexistente |
| **ELIX_13** | Hipotiroidismo | E00.x-E03.x, E89.0 | 0 | Crónica Preexistente |
| **ELIX_14** | Insuficiencia renal | I12.0, N18.x, N19.x, Z49.0-Z49.2, Z99.2 (incluye N17) | +5 | **Potencial Complicación** |
| **ELIX_15** | Enfermedad hepática | B18.x, I85.x, K70.x, K71.x, K72.x-K74.x | +11 | Crónica Preexistente |
| **ELIX_16** | Úlcera péptica | K25.7, K25.9, K26.7, K27.7, K28.7 | 0 | Crónica Preexistente |
| **ELIX_17** | VIH / SIDA | B20.x-B22.x, B24.x | 0 | Crónica Preexistente |
| **ELIX_18** | Linfoma | C81.x-C85.x, C88.x, C96.x, C90.0 | +9 | Crónica Preexistente |
| **ELIX_19** | Cáncer metastásico | C77.x-C80.x | +12 | Crónica Preexistente |
| **ELIX_20** | Tumor sólido sin metástasis | C00.x-C26.x, C30.x-C34.x, C43.x, C50.x-C76.x | +4 | Crónica Preexistente |
| **ELIX_21** | Artritis / conectivopatías | L94.0, M05.x, M06.x, M08.x, M32.x-M35.x | 0 | Crónica Preexistente |
| **ELIX_22** | Coagulopatía | D65.x-D68.x, D69.1, D69.3-D69.6 | +3 | **Potencial Complicación** |
| **ELIX_23** | Obesidad | E66.x | -4 | Crónica Preexistente |
| **ELIX_24** | Pérdida de peso patológica | E40.x-E46.x, R63.4, R64.x | +6 | Crónica Preexistente |
| **ELIX_25** | Trastornos hidroelectrolíticos | E22.2, E86.x, E87.x | +5 | **Potencial Complicación** |
| **ELIX_26** | Anemia por hemorragia | D50.0 | -2 | Crónica Preexistente |
| **ELIX_27** | Anemia por deficiencia | D50.8, D50.9, D51.x-D53.x | -2 | Crónica Preexistente |
| **ELIX_28** | Abuso de alcohol | F10.x, E52.x, G62.1, K70.0, T51.x | 0 | Crónica Preexistente |
| **ELIX_29** | Abuso de drogas | F11.x-F16.x, F18.x, F19.x | -7 | Crónica Preexistente |
| **ELIX_30** | Psicosis | F20.x-F25.x, F28.x, F29.x, F30.2 | 0 | Crónica Preexistente |
| **ELIX_31** | Depresión | F20.4, F31.3-F31.5, F32.x, F33.x, F34.1 | -3 | Crónica Preexistente |

### 3.2 Especificación Primaria vs. Sensibilidad
1. **Especificación Primaria (Crónicas Puras):**
   * Incluye únicamente las **27 condiciones crónicas preexistentes**.
   * Excluye: `ELIX_02` (Arritmias), `ELIX_14` (Falla renal/AKI), `ELIX_22` (Coagulopatía) y `ELIX_25` (Hidroelectrolíticos).
   * Utiliza el Score van Walraven Recalculado:
     $$\text{SCORE\_VW\_PREEXISTENTE} = \text{SCORE\_VANWALRAVEN} - (5 \times \text{ELIX\_02} + 5 \times \text{ELIX\_14} + 3 \times \text{ELIX\_22} + 5 \times \text{ELIX\_25})$$
2. **Especificación de Sensibilidad (Completa):**
   * Incorpora las 31 comorbilidades y el score estándar.
   * **Impacto empírico comprobado:** La correlación de Spearman entre los rankings hospitalarios O/E de ambas especificaciones es **$\rho = 0.9403$**. Hospitales con alta codificación de complicaciones (ej. Hospital 109101) ven subir su O/E de $0.807$ a $0.997$ (+0.19 puntos) al depurar las complicaciones, demostrando que estaban recibiendo un "crédito de severidad" artificial por eventos adversos intrahospitalarios.

---

## 4. Intensidad de Codificación (*Upcoding*) y Sensibilidad a K Diagnósticos

Para mitigar la brecha de 4.19 diagnósticos secundarios por caso entre hospitales metropolitanos docentes y provinciales, se evaluó la estabilidad del ranking O/E limitando la lectura a los primeros $K$ diagnósticos secundarios:
* **$K = 3$ diagnósticos:** $\rho = 0.9612$ contra el modelo de crónicas puras.
* **$K = 5$ diagnósticos:** $\rho = 0.9845$ contra el modelo de crónicas puras.
* **$K = 10$ diagnósticos:** $\rho = 0.9950$ contra el modelo de crónicas puras.
* **Decisión:** Se adopta **$K = 5$ diagnósticos secundarios** como cota superior en la especificación de control de upcoding, capturando el 92% de la carga de comorbilidad y neutralizando el sesgo de codificación exhaustiva.

---

## 5. Protocolo de Selección Empírica, Tweedie y Benchmarks

### 5.1 Barrido de Regularización ($C$) en Stability Selection
Se realizó un barrido sobre $C \in [10^{-3}, 10^{-2}, 0.05, 0.1, 0.5, 1.0, 10.0]$:
* Para $C \le 0.01$, solo Edad y Score sobreviven.
* En el rango óptimo $C \in [0.05, 0.1]$, se estabilizan con frecuencia $\ge 70\%$: Edad, Score preexistente, Cáncer metastásico (`ELIX_19`), Falla cardíaca (`ELIX_01`), Linfoma (`ELIX_18`), Enfermedad hepática (`ELIX_15`) y EPOC (`ELIX_10`).
* Para $C \ge 1.0$, la penalización colapsa y selecciona el 100% de las variables por sobreajuste. Se confirma $C = 0.05$ como el punto de máxima reproducibilidad.

### 5.2 Benchmarks Clínicos Jerárquicos
Evaluados en la cohorte de desarrollo sobre LightGBM:
1. **Benchmark 1 (Demográfico: Edad + Sexo):** $\text{ROC-AUC} = 0.8204$.
2. **Benchmark 2 (B1 + Diagnóstico Principal CIE-10):** $\text{ROC-AUC} = 0.9036$ ($+0.0831$).
3. **Benchmark 3 (B2 + Score van Walraven Tradicional):** $\text{ROC-AUC} = 0.9395$ ($+0.0360$).
4. **Modelo Final ML (LightGBM con 47 features basales seleccionadas):** $\text{ROC-AUC} = 0.9506$ ($+0.0111$).

### 5.3 Modelo de Estancia: Parámetro Tweedie y Capping
* Comparación de pérdida en desarrollo (2022):
  * Tweedie $p=1.2$: $\text{MAE} = 5.32$ días | $\text{MedianAE} = 3.11$ días.
  * Tweedie $p=1.5$: $\text{MAE} = 5.32$ días | $\text{MedianAE} = 3.10$ días.
  * Tweedie $p=1.8$: $\text{MAE} = 5.31$ días | $\text{MedianAE} = 3.09$ días.
  * Gamma ($p=2.0$): $\text{MAE} = 5.30$ días | $\text{MedianAE} = 3.08$ días.
* **Decisión:** Se mantiene **Tweedie $p=1.5$** (modelo Poisson-Gamma sobredisperso) con un **tope superior p99 = 60 días** calculado exclusivamente sobre datos de desarrollo y aplicado simétricamente a la estancia observada y a la predicción para evitar divergencias en el IEMC.

---

## 6. Variables Condicionales: Decisión Final con Métricas en Desarrollo

Auditadas con la **$V$ de Cramér Corregida por Sesgo** sobre la muestra de desarrollo:
1. **`TIPO_PROCEDENCIA` (25 categorías):** $V_{\text{corr}} = 0.1883$. Asociación moderada. Se **acepta** colapsada en 3 macro-grupos: urgencia, derivado, programado.
2. **`ESPECIALIDAD_MEDICA` (158 categorías):** $V_{\text{corr}} = 0.1722$. Aunque globalmente moderada, centros monográficos (ej. Traumatológico, Tórax) concentran el 100% de sus casos en una especialidad. Se **excluye** de la especificación primaria y se permite solo en macro-bloques quirúrgicos/médicos.
3. **`SERVICIOINGRESO` (120 categorías):** $V_{\text{corr}} = 0.2914$. Supera el umbral de 0.25 (alta absorción organizativa). Se **colapsa** a nivel binario de cuidados críticos al ingreso (UCI/UTI vs. Cama Básica).
4. **`HOSPPROCEDENCIA`:** $V_{\text{corr}} = 0.3387$. Muy alta absorción. Se reserva estrictamente para la **regla de enlace de traslados**, quedando prohibida como variable predictiva.

---

## 7. Umbrales de Volumen y Calibración en Cohorte de Prueba (2024)

* Al evaluar sobre la cohorte analítica final de 2024:
  * Los 72 hospitales presentan $\ge 1.000$ egresos analíticos anuales (mínimo: 1.054, mediana: 12.430).
  * 70 de los 72 hospitales presentan $\ge 20$ muertes observadas (solo 2 hospitales de mediana complejidad registran entre 15 y 19 defunciones).
* **Umbral Operativo Definitivo:** Se establece **$N \ge 1.000$ egresos analíticos anuales o $E \ge 25$ defunciones esperadas** para graficar los *funnel plots* hospitalarios, aplicando contracción empírica de Bayes (*Empirical Bayes shrinkage*) para estabilizar los intervalos de confianza en los centros con $E < 25$.

---

## 8. Partición Temporal de Entrenamiento y Evaluación

* **Especificación Primaria Recomendada:**
  * **2019:** Período de historia / *washout* ($N = 1.151.475$). Permite calcular con 12 meses exactos `N_EGRESOS_12M` y `DIAS_DESDE_EGRESO_PREVIO`.
  * **2020–2022:** Entrenamiento formal ($N = 2.531.661$).
  * **2023:** Calibración isotónica y ajuste de umbrales ($N = 1.039.587$).
  * **2024:** Evaluación final y cálculo de indicadores O/E ($N = 1.085.813$).
* **Especificación de Sensibilidad Multianual:**
  * Entrenamiento con **2019–2022**, utilizando el flag `HISTORIA_DISPONIBLE = 0` para los casos sin ventana retrospectiva.

---
*Documento metodológico final consolidado y verificado empíricamente con los 5.808.536 registros de FONASA.*
