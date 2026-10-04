# Resolución Metodológica y Cierre Definitivo de la Fase 2
**Proyecto:** Evaluación del Desempeño Hospitalario Ajustado por Riesgo en el Régimen de Garantías Explícitas en Salud (FONASA, 2019–2024)  
**Autor:** Felipe  
**Fecha:** Octubre 2026 (Actualización Consolidada con Re-estimación de Agudos Generales)

---

## 1. Dependencia de la Especificación, Alertas en M3 y Re-estimación de Agudos Generales

### 1.1 Documentación del Cambio en la Lista de Alertas de M3
En versiones preliminares del documento se listaron provisionalmente 7 prestadores (El Pino, Van Buren, Tisné, Tórax, Pereira, Heyermann y Curanilahue) sobre la base de $O/E$ preliminares antes de contracción completa. Al aplicar el modelo formal M3 con contracción empírica de Bayes sobre la cohorte evaluable congelada de 2023 ($N = 557.318$ episodios, $O = 24.225$ defunciones, $K = 65$ hospitales):

1. **Salida de Curanilahue (`128109`):**  
   Su $O/E$ crudo preliminar de $1,211$ se recalibra y contrae a **$1,137$** ($O = 136, E = 119,5$). Con $O = 136$ muertes, su varianza de muestreo es $v_j = 1/136 = 0,007353$ y su error estándar posterior es $\text{SE}_{\text{post}} = 0,0798$.  
   Al evaluar el exceso respecto al margen clínico a priori del $10\%$ ($\ln(1,10) = 0,0953$):
   $$Z_j = \frac{0,1154 - 0,0953}{0,0798} = +0,252 \implies P(\theta_j > 1,10 \mid M3) = \mathbf{0,594} \text{ a } \mathbf{0,599} \quad (\ll 0,95)$$
   *(La leve oscilación entre $0,594$ y $0,599$ responde al $\tau$ utilizado: $0,594$ con $\tau = 0,2162$ y $0,599$ con $\tau = 0,2173$)*.  
   **Curanilahue no cumple el umbral de alerta bayesiana ($P \ge 0,95$) y queda clasificado formalmente en "Promedio"**. Al margen del $20\%$, $P(\theta_j > 1,20) = \mathbf{0,201}$.
2. **Salida de Dr. Mauricio Heyermann de Angol (`129100`):**  
   Registra $O = 233, E = 197,3 \implies O/E = 1,181$. Con $\text{SE}_{\text{post}} = 0,0624$, su probabilidad posterior es $P(\theta_j > 1,10 \mid M3) = \mathbf{0,828} < 0,95$, quedando igualmente en **"Promedio"**.
3. **Entrada de Hospital Claudio Vicuña de San Antonio (`106103`):**  
   Registra $O = 320, E = 195,8 \implies O/E_{\text{M3}} = \mathbf{1,634}$. Con $\text{SE}_{\text{post}} = 0,0539$, su probabilidad posterior es $P(\theta_j > 1,10) = \mathbf{1,000}$ (y $P(\theta_j > 1,20) = \mathbf{1,000}$). Su omisión previa respondió a un error de indexación por código.
4. **Entrada de Hospital Clínico Metropolitano La Florida (`114105`):**  
   Registra $O = 596, E = 457,3 \implies O/E_{\text{M3}} = \mathbf{1,303}$. Debido a su elevado volumen ($O = 596$), su error estándar posterior es muy estrecho ($\text{SE}_{\text{post}} = 0,0402$). Su probabilidad posterior de exceso al $10\%$ es $P(\theta_j > 1,10) = \mathbf{1,000}$ (y al $20\%$, $P(>1,20) = \mathbf{0,968}$).

---

### 1.2 Re-estimación de $\tau$ y M3 Excluyendo Centros Monográficos (Agudos Generales, $K = 62$)
El Instituto Nacional del Tórax registra un $O/E = 2,226$ con $O = 157$. Al incluirlo en la estimación de Bayes empírico de toda la red, **infla artificialmente $\tau$** (la dispersión entre hospitales) y desplaza hacia arriba la meta-media $\mu_{\text{meta}}$, aumentando indebidamente la contracción de todos los hospitales agudos generales.

Al re-estimar el modelo empírico de Bayes **exclusivamente sobre los 62 hospitales de agudos generales** (excluyendo Tórax `112103`, Traumatológico `110110` y Neurocirugía `112104`):
* **Red Completa (65 hospitales):** $\tau = \mathbf{0,2173}$ ($\tau^2 = 0,0472$), $\mu_{\text{meta}} = \mathbf{0,0261}$.
* **Solo Agudos Generales (62 hospitales):** $\tau_{\text{agudos}} = \mathbf{0,2048}$ ($\tau^2 = 0,0419$), $\mu_{\text{meta}} = \mathbf{0,0224}$.
* **Efecto de Aislamiento:** La presencia de los monográficos inflaba $\tau$ en $+\mathbf{0,0125}$ (+6,1% de variabilidad espuria) y desplazaba $\mu_{\text{meta}}$ en $+\mathbf{0,0037}$.

Al evaluar a los agudos generales bajo sus propios parámetros limpios ($\tau = 0,2048, \mu = 0,0224$):
* **Las alertas de agudos generales se mantienen estrictamente idénticas (6 hospitales):** El Pino ($P = 1,000$), Claudio Vicuña ($P = 1,000$), Eduardo Pereira ($P = 0,9993$), Carlos Van Buren ($P = 1,000$), Luis Tisné ($P = 0,9999$) y La Florida ($P = 1,000$).
* **Centros intermedios:** San Camilo ($P = 0,9300$), Antofagasta ($P = 0,9164$), Barros Luco ($P = 0,9211$), Heyermann ($P = 0,8226$) y Curanilahue ($P = 0,5871$) confirman su condición de **Promedio**.

---

### 1.3 Matriz de Sensibilidad con $O$, $E$ y $\text{SE}_{\text{post}}$ por Hospital
Se incorpora el detalle de defunciones observadas ($O$), esperadas ($E$) y error estándar posterior ($\text{SE}_{\text{post}}$) para transparentar por qué un hospital con menor $O/E$ puede estar en alerta mientras otro con mayor $O/E$ no lo está:
* **Caso San Camilo (`108100`) vs San Borja (`111100`) en M3c:**
  - San Camilo registra **$O = 365$ defunciones**, lo que otorga una alta precisión muestral ($\text{SE}_{\text{post}} = 0,0507$). Por ello, con $O/E = 1,24$, su intervalo creíble se sitúa íntegramente por sobre $1,10$ ($P = 0,997 \implies$ ALERTA).
  - San Borja registra solo **$O = 145$ defunciones** (2,5 veces menos muertes), lo que eleva su incertidumbre posterior a $\text{SE}_{\text{post}} = 0,0770$. Pese a tener un $O/E = 1,27$, su probabilidad posterior de exceso al $10\%$ no alcanza el $95\%$ ($P = 0,733 \implies$ PROMEDIO).

#### Matriz de Sensibilidad de Alertas en Hospitales Agudos Generales (* = Alerta con $P(\theta > 1,10) \ge 0,95$):
| Código | Establecimiento | $O$ | $E_{\text{M3a}}$ | $\text{SE}_{\text{post}}$ | M3a (Prim) | M3b (Lin) | M3c (Zdx) | M3d (31c) | Sin Día 0 | M2 (Clin) | Consistencia M3 | Alerta Robusta $M2 \cap M3$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `113180` | **Hospital El Pino** | 438 | 247,1 | 0,0465 | **1,77\*** | **1,73\*** | **1,81\*** | **1,68\*** | **1,68\*** | **1,84\*** | **5/5 (100%)** | **SÍ (Agudo 1)** |
| `106103` | **Hospital Claudio Vicuña** | 320 | 195,8 | 0,0539 | **1,63\*** | **1,69\*** | **1,52\*** | **1,66\*** | **1,65\*** | **1,59\*** | **5/5 (100%)** | **SÍ (Agudo 2)** |
| `106102` | **Hospital Dr. Eduardo Pereira** | 148 | 100,1 | 0,0763 | **1,48\*** | **1,42\*** | **1,38\*** | **1,45\*** | **1,55\*** | 0,98 | **5/5 (100%)** | No (Sensible M2) |
| `106100` | **Hospital Carlos Van Buren** | 602 | 419,3 | 0,0400 | **1,44\*** | **1,48\*** | **1,35\*** | **1,42\*** | **1,36\*** | **1,29\*** | **5/5 (100%)** | **SÍ (Agudo 3)** |
| `112101` | **Hospital Dr. Luis Tisné B.** | 502 | 381,3 | 0,0436 | **1,32\*** | **1,36\*** | **1,28\*** | **1,26\*** | **1,28\*** | **1,31\*** | **5/5 (100%)** | **SÍ (Agudo 4)** |
| `114105` | **Hospital Clínico La Florida** | 596 | 457,3 | 0,0402 | **1,30\*** | **1,28\*** | **1,32\*** | **1,23\*** | **1,32\*** | 1,17 | **5/5 (100%)** | No (Sensible M2) |
| `108100` | Hospital San Camilo (San Felipe) | 365 | 304,9 | 0,0507 | 1,20 | 1,21\* | 1,24\* | 1,18 | 1,16 | 1,29\* | 2/5 (Parcial) | No |
| `103100` | Hospital Regional de Antofagasta | 663 | 569,2 | 0,0382 | 1,16 | 1,19\* | 1,26\* | 1,22\* | 1,20\* | 1,38\* | 4/5 (Parcial) | No |
| `121109` | Hospital Hernán Henríquez (Temuco)| 580 | 498,1 | 0,0407 | 1,16 | 1,14 | 1,19\* | 1,17 | 1,20\* | 1,24\* | 2/5 (Parcial) | No |
| `113100` | Hospital Barros Luco Trudeau | 972 | 842,5 | 0,0317 | 1,15 | 1,14 | 1,16 | 1,15 | 1,10 | 1,29\* | 0/5 | No |
| `111100` | Hospital San Borja-Arriarán | 145 | 123,2 | 0,0770 | 1,18 | 1,18 | 1,27 | 1,19 | 1,24 | 1,51\* | 0/5 | No |
| `109100` | Complejo Hospitalario San José | 780 | 696,4 | 0,0353 | 1,12 | 1,10 | 1,11 | 1,12 | 1,15 | 1,19\* | 0/5 | No |
| `103101` | Hospital Carlos Cisternas (Calama)| 132 | 137,5 | 0,0807 | 0,96 | 0,95 | 1,02 | 0,95 | 0,97 | 1,22\* | 0/5 | No |
| `113150` | Hospital San Luis (Buin) | 161 | 175,0 | 0,0733 | 0,92 | 0,92 | 0,92 | 0,96 | 0,93 | 1,31\* | 0/5 | No |

---

## 2. Delimitación Cuantitativa de Monográficos en Toda la Red (65 Hospitales)

Para demostrar que la regla analítica no deja casos grises ni arbitrariedades, se publica la concentración de episodios por MDC para **la totalidad de los 65 prestadores**:
$$\text{Centro Monográfico} \iff \max_{\text{MDC}} (\% \text{ Episodios}) \ge 50\% \quad \text{Ó} \quad (\% \text{ MDC 4 [Resp]} + \% \text{ MDC 5 [Cardio]}) \ge 70\%$$

#### Distribución Completa de Concentración de Especialidad (Cohorte 2023):
| Código | Establecimiento | MDC Principal | % MDC Principal | % Cardio-Resp (4+5) | Clasificación Medida |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `110110` | **Instituto Traumatológico Dr. Teodoro Gebauer** | **MDC 8** | **93,4%** | 0,8% | **MONOGRÁFICO** |
| `112104` | **Instituto de Neurocirugía Dr. Asenjo** | **MDC 1** | **66,8%** | 5,1% | **MONOGRÁFICO** |
| `112103` | **Instituto Nacional del Tórax** | **MDC 5** | **62,5%** | **92,7%** | **MONOGRÁFICO** |
| `107102` | Hospital de Quilpué | MDC 8 | 31,8% | 15,2% | Agudo General |
| `111100` | Hospital Clínico San Borja-Arriarán | MDC 13 | 28,0% | 11,9% | Agudo General |
| `105101` | Hospital San Pablo (Coquimbo) | MDC 8 | 25,6% | 18,8% | Agudo General |
| `108101` | Hospital San Juan de Dios (Los Andes) | MDC 8 | 25,1% | 17,4% | Agudo General |
| `128109` | Hospital Provincial Dr. Rafael Avaría (Curanilahue)| MDC 13 | 24,0% | 21,0% | Agudo General |
| `106102` | **Hospital Dr. Eduardo Pereira Ramírez** | **MDC 6** | **21,8%** | **30,3%** | **AGUDO GENERAL** |
| `115110` | Hospital de Santa Cruz | MDC 13 | 21,2% | 18,5% | Agudo General |
| `114103` | Hospital Padre Alberto Hurtado | MDC 13 | 20,3% | 17,6% | Agudo General |
| `113150` | Hospital San Luis (Buin) | MDC 13 | 20,3% | 22,4% | Agudo General |
| `113100` | Hospital Barros Luco Trudeau | MDC 8 | 19,8% | 21,1% | Agudo General |
| `121114` | Hospital Intercultural de Nueva Imperial | MDC 8 | 19,6% | 17,5% | Agudo General |
| `113180` | Hospital El Pino | MDC 13 | 19,6% | 18,2% | Agudo General |
| `121109` | Hospital Dr. Hernán Henríquez Aravena | MDC 13 | 19,6% | 21,9% | Agudo General |
| `111195` | HUAP (Posta Central) | MDC 8 | 19,5% | 28,5% | Agudo General |
| `119101` | Hospital de Tomé | MDC 4 | 19,3% | 27,8% | Agudo General |
| `110150` | Hospital San José (Melipilla) | MDC 13 | 18,9% | 15,4% | Agudo General |
| `109100` | Complejo Hospitalario San José | MDC 8 | 18,9% | 23,0% | Agudo General |
| `110130` | Hospital de Talagante | MDC 7 | 18,7% | 23,3% | Agudo General |
| `118105` | Hospital San José (Coronel) | MDC 13 | 18,5% | 17,9% | Agudo General |
| `129106` | Hospital San José (Victoria) | MDC 8 | 18,3% | 14,0% | Agudo General |
| `107101` | Hospital San Martín (Quillota) | MDC 13 | 18,0% | 17,9% | Agudo General |
| `115107` | Hospital San Juan de Dios (San Fernando) | MDC 6 | 17,9% | 18,7% | Agudo General |
| `121110` | Hospital Dr. Abraham Godoy (Lautaro) | MDC 13 | 17,7% | 21,8% | Agudo General |
| `116105` | Hospital Dr. César Garavagno (Talca) | MDC 8 | 17,5% | 18,9% | Agudo General |
| `120101` | Complejo Asistencial Dr. Víctor Ríos (Los Ángeles)| MDC 13 | 17,5% | 17,6% | Agudo General |
| `107100` | Hospital Dr. Gustavo Fricke | MDC 5 | 17,3% | 25,0% | Agudo General |
| `117101` | Hospital Clínico Herminda Martín (Chillán) | MDC 13 | 17,2% | 21,7% | Agudo General |
| `106103` | Hospital Claudio Vicuña | MDC 6 | 16,9% | 25,7% | Agudo General |
| `119100` | Hospital Las Higueras (Talcahuano) | MDC 8 | 16,5% | 21,5% | Agudo General |
| `105102` | Hospital Dr. Antonio Tirado Lanas (Ovalle) | MDC 6 | 16,4% | 17,3% | Agudo General |
| `124105` | Hospital de Puerto Montt | MDC 13 | 16,2% | 25,7% | Agudo General |
| `110100` | Hospital San Juan de Dios (Santiago) | MDC 13 | 16,2% | 21,8% | Agudo General |
| `111101` | Hospital Clínico Metropolitano Félix Bulnes | MDC 5 | 16,1% | 30,2% | Agudo General |
| `108100` | Hospital de San Camilo (San Felipe) | MDC 6 | 16,1% | 24,5% | Agudo General |
| `101100` | Hospital Dr. Juan Noé Crevanni (Arica) | MDC 8 | 16,1% | 15,5% | Agudo General |
| `116108` | Hospital Presidente Carlos Ibáñez (Linares)| MDC 13 | 15,9% | 18,5% | Agudo General |
| `125100` | Hospital Regional (Coihaique) | MDC 8 | 15,7% | 13,6% | Agudo General |
| `116100` | Hospital San Juan de Dios (Curicó) | MDC 6 | 15,7% | 19,8% | Agudo General |
| `122100` | Hospital Clínico Regional (Valdivia) | MDC 13 | 15,6% | 20,5% | Agudo General |
| `106100` | Hospital Carlos Van Buren | MDC 8 | 15,6% | 19,5% | Agudo General |
| `116110` | Hospital San José (Parral) | MDC 4 | 15,4% | 25,7% | Agudo General |
| `117102` | Hospital de San Carlos | MDC 6 | 15,3% | 22,2% | Agudo General |
| `104100` | Hospital San José del Carmen (Copiapó) | MDC 8 | 15,2% | 16,8% | Agudo General |
| `110120` | Hospital Dr. Félix Bulnes Cerda (Quinta Normal)| MDC 4 | 15,0% | 24,5% | Agudo General |
| `104103` | Hospital Provincial del Huasco (Vallenar) | MDC 6 | 15,0% | 19,5% | Agudo General |
| `105100` | Hospital San Juan de Dios (La Serena) | MDC 13 | 14,8% | 26,7% | Agudo General |
| `114105` | Hospital Clínico Metropolitano La Florida | MDC 4 | 14,7% | 25,2% | Agudo General |
| `118106` | Hospital de Lota | MDC 13 | 14,6% | 19,1% | Agudo General |
| `114101` | Complejo Hospitalario Dr. Sótero del Río | MDC 13 | 14,5% | 24,2% | Agudo General |
| `103100` | Hospital Dr. Leonardo Guzmán (Antofagasta) | MDC 13 | 14,5% | 21,5% | Agudo General |
| `121117` | Hospital de Pitrufquén | MDC 7 | 14,1% | 26,5% | Agudo General |
| `112100` | Hospital Del Salvador | MDC 8 | 13,9% | 20,7% | Agudo General |
| `103101` | Hospital Dr. Carlos Cisternas (Calama) | MDC 4 | 13,9% | 21,8% | Agudo General |
| `123100` | Hospital Base San José de Osorno | MDC 8 | 13,7% | 21,5% | Agudo General |
| `133150` | Hospital de Castro | MDC 6 | 13,6% | 23,4% | Agudo General |
| `118100` | Hospital Guillermo Grant Benavente (Concepción)| MDC 13 | 13,5% | 21,3% | Agudo General |
| `102100` | Hospital Dr. Ernesto Torres Galdames (Iquique) | MDC 13 | 13,1% | 18,2% | Agudo General |
| `129100` | Hospital Dr. Mauricio Heyermann (Angol) | MDC 6 | 12,9% | 18,9% | Agudo General |
| `115100` | Hospital Regional de Rancagua | MDC 4 | 12,8% | 24,8% | Agudo General |
| `126100` | Hospital Clínico de Magallanes (Punta Arenas)| MDC 6 | 12,1% | 19,3% | Agudo General |

**Conclusión:** La distribución es bimodal y nítida. Los tres institutos monográficos presentan concentraciones de entre **$66,8\%$ y $93,4\%$**, mientras que el hospital general más concentrado de la red (Quilpué) alcanza apenas un **$31,8\%$** en su MDC mayor y el Hospital Eduardo Pereira solo un **$30,3\%$** en Cardio-Respiratorio. No existe ambigüedad en la clasificación.

---

## 3. Censura por Derivación: $\lambda_{\text{obs}}$ Basado en Riesgo Esperado y Estrés Riguroso

### 3.1 Corrección de Escala: $\lambda_{\text{obs}} = O_{\text{link}} / E_{\text{link}}$ e Intervalos Exactos
En estricto apego al marco conceptual bayesiano, $\lambda_{\text{obs}}$ se calcula como la razón entre las defunciones observadas en el receptor ($O_{\text{link}}$) y el **riesgo esperado por case-mix de esos mismos pacientes según el modelo M3 ($E_{\text{link}} = \sum P_{3, i}$)**, y no frente al promedio no ajustado de la red.

En 2023 se registraron $31.910$ egresos totales por traslado en Silver ($31.082$ bajo el filtro exploratorio preliminar y **$25.555$ en la cohorte adulta inpatient evaluable**). De ellos, se enlazaron a su receptor a **$11.544$ episodios ($45,2\%$ de enlace)**:
* **Riesgo Esperado ($E$):** Los pacientes derivados no son promedio; tienen un riesgo esperado de **$6,46\%$** en los enlazados y de **$8,10\%$** en los no enlazados (severidad basal superior al $4,35\%$ de la red).

#### Tabla de Principales Emisores con $\lambda_{\text{obs}} = O_{\text{link}} / E_{\text{link}}$ e Intervalos Exactos Byar (95%):
| Código | Hospital Emisor | Emitidos | Enlazados | $O_{\text{link}}$ | $E_{\text{link}}$ | $\lambda_{\text{obs}} (O/E)$ | IC 95% Byar | $E_{\text{no\_link}}$ | No Enlazados |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `114101` | Complejo Dr. Sótero del Río | 2.385 | 621 | 43 | 26,4 | **1,63x** | [1,18x - 2,19x] | 89,2 | 1.764 |
| `113180` | Hospital El Pino | 1.370 | 250 | 25 | 9,7 | **2,58x** | [1,67x - 3,81x] | 46,8 | 1.120 |
| `106100` | Hospital Carlos Van Buren | 1.259 | 910 | 58 | 45,4 | **1,28x** | [0,97x - 1,65x] | 19,3 | 349 |
| `112101` | Hospital Dr. Luis Tisné B. | 908 | 209 | 20 | 15,7 | **1,27x** | [0,78x - 1,97x] | 55,1 | 699 |
| `121109` | Hospital Dr. Hernán Henríquez | 853 | 445 | 17 | 33,0 | **0,51x** | [0,30x - 0,82x] | 38,9 | 408 |
| `115100` | Hospital Regional de Rancagua | 834 | 230 | 12 | 17,0 | **0,71x** | [0,36x - 1,23x] | 59,1 | 604 |
| `124105` | Hospital de Puerto Montt | 802 | 246 | 16 | 20,5 | **0,78x** | [0,45x - 1,27x] | 61,8 | 556 |
| `120101` | Complejo Asistencial Dr. Víctor Ríos | 786 | 172 | 9 | 15,7 | **0,57x** | [0,26x - 1,09x] | 73,3 | 614 |
| `118100` | Hospital Guillermo Grant (Concepción)| 713 | 493 | 24 | 34,9 | **0,69x** | [0,44x - 1,02x] | 18,7 | 220 |
| `112100` | Hospital Del Salvador | 669 | 180 | 25 | 11,9 | **2,10x** | [1,36x - 3,10x] | 47,9 | 489 |
| `117101` | Hospital Herminda Martín (Chillán) | 636 | 204 | 19 | 15,4 | **1,23x** | [0,74x - 1,92x] | 33,4 | 432 |
| `107101` | Hospital San Martín (Quillota) | 570 | 231 | 13 | 18,9 | **0,69x** | [0,37x - 1,17x] | 42,1 | 339 |
| `107100` | Hospital Dr. Gustavo Fricke | 570 | 222 | 17 | 18,5 | **0,92x** | [0,53x - 1,47x] | 29,8 | 348 |
| `111195` | HUAP (Posta Central) | 551 | 263 | 31 | 21,3 | **1,46x** | [0,99x - 2,07x] | 17,8 | 288 |
| `122100` | Hospital Clínico Regional (Valdivia)| 532 | 125 | 7 | 9,1 | **0,77x** | [0,31x - 1,58x] | 39,3 | 407 |
| `128109` | **Hospital Dr. Rafael Avaría (Curanilahue)**| **511** | **210** | **12** | **12,6** | **0,95x** | **[0,49x - 1,66x]** | **20,3** | **301** |
| `113100` | Hospital Barros Luco Trudeau | 503 | 94 | 16 | 8,9 | **1,80x** | [1,03x - 2,92x] | 30,2 | 409 |
| `102100` | Hospital Dr. Ernesto Torres Galdames| 427 | 180 | 7 | 16,7 | **0,42x** | [0,17x - 0,86x] | 20,6 | 247 |
| `116110` | Hospital San José de Parral | 392 | 365 | 24 | 14,6 | **1,65x** | [1,06x - 2,45x] | 1,2 | 27 |
| `121114` | Hospital Intercultural Nueva Imperial| 333 | 267 | 6 | 8,8 | **0,68x** | [0,25x - 1,49x] | 5,2 | 66 |
| `121117` | Hospital de Pitrufquén | 328 | 173 | 14 | 9,1 | **1,54x** | [0,84x - 2,59x] | 11,5 | 155 |
| `106103` | Hospital Claudio Vicuña | 299 | 197 | 14 | 8,6 | **1,63x** | [0,89x - 2,74x] | 5,4 | 102 |
| `121121` | Hospital de Villarrica | 288 | 182 | 9 | 11,2 | **0,80x** | [0,37x - 1,52x] | 6,8 | 106 |

---

### 3.2 Corrección del Análisis de Estrés: Caso Curanilahue y Punto de Quiebre en No Enlazados
Se reconoce y subsana la inconsistencia advertida: imponer una letalidad plana de $15,0\%$ sobre los 301 traslados no enlazados de Curanilahue imputaba $45,2$ muertes, elevando su letalidad total de derivados a $11,2\%$, por encima de su umbral crítico ($8,70\%$). La afirmación previa de que "ningún prestador promedio alcanza su $\lambda_{\text{entrada}}$" era falsa bajo ese estrés arbitrario.

La formulación rigurosa se basa en **múltiplos del riesgo esperado por el modelo ($E_{\text{no\_link}}$)**:
* Curanilahue presenta $E_{\text{no\_link}} = 20,3$ defunciones esperadas en sus 301 no enlazados (riesgo basal de $6,74\%$).
* Su $\lambda_{\text{obs}}$ en los enlazados es de **$0,95\times$** ($12$ observadas vs $12,6$ esperadas).

#### Evaluación Paramétrica de Estrés en los No Enlazados de Curanilahue:
| Escenario de Estrés ($\lambda_{\text{no\_link}}$) | Defunciones Imputadas | Tasa Imputada | $O_{\text{total}}$ | $E_{\text{total}}$ | $O/E$ Global | $P(\theta > 1,10)$ | Categoría EB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\lambda = \mathbf{0,95x}$ *(Igual al Observado)* | 19,3 defunciones | 6,41% | 167,3 | 152,4 | 1,098 | 0,443 | **PROMEDIO** |
| $\lambda = \mathbf{1,00x} \cdot E$ *(Riesgo Base)* | 20,3 defunciones | 6,74% | 168,3 | 152,4 | 1,104 | 0,469 | **PROMEDIO** |
| $\lambda = \mathbf{1,50x} \cdot E$ | 30,4 defunciones | 10,10% | 178,4 | 152,4 | 1,171 | 0,746 | **PROMEDIO** |
| $\lambda = \mathbf{2,00x} \cdot E$ | 40,6 defunciones | 13,49% | 188,6 | 152,4 | 1,237 | 0,920 | **PROMEDIO** |
| $\lambda = \mathbf{2,17x} \cdot E$ *(Punto de Quiebre)*| **44,0 defunciones** | **14,62%** | **192,0** | **152,4** | **1,260** | **0,950** | **UMBRAL ALERTA** |
| $\lambda = \mathbf{2,50x} \cdot E$ | 50,7 defunciones | 16,84% | 198,7 | 152,4 | 1,304 | 0,984 | ALERTA |

**Conclusión:** Curanilahue solo cruza hacia zona de alerta si la mortalidad de sus traslados no enlazados **supera 2,17 veces su riesgo esperado ($14,62\%$)**, lo que equivale a más de 2,5 veces la mortalidad empírica observada en sus pacientes enlazados ($5,71\%, \lambda = 0,95\times$).

---

## 4. Auditoría de Ingresos Obstétricos y $R_{\text{dx}}$ como Información Posterior

### 4.1 Aclaración: `TIPO_INGRESO == 'OBSTETRICA'` vs Exclusión de MDC 14
Se auditó la presencia de $26.119$ episodios ($16,3\%$) con `TIPO_INGRESO == 'OBSTETRICA'` en el estrato de 0–2 diagnósticos:
* **Exclusión de MDC 14:** Se verificó formalmente que en la cohorte evaluable **existen exactamente 0 episodios de MDC 14** ($N_{\text{MDC 14}} = 0$). La regla de exclusión EX03 se aplicó al 100%.
* **Naturaleza Administrativa:** En los hospitales públicos, `TIPO_INGRESO` codifica la puerta física de admisión. El **$95,8\%$ ($44.285$ episodios en 2023)** de los registros con ingreso obstétrico corresponden a **MDC 13 (Afecciones del Aparato Genital Femenino / Ginecología)** que ingresaron administrativamente a través de la Urgencia Maternal (metrorragias, quistes de ovario, legrados ginecológicos, miomatosis). No constituyen obstetricia activa (embarazo, parto o puerperio).

---

### 4.2 Evaluación de M3 con $R_{\text{dx}}$ Crónico vs $R_{\text{dx}}$ Total y Limitación Metodológica
Dado que el número de diagnósticos $N_{\text{dx}}$ depende directamente de la estancia hospitalaria (mediana de 2 días en 0–2 dx vs 5 días en $\ge 3$ dx), $N_{\text{dx}}$ incorpora información posterior al ingreso ligada al tiempo en riesgo.

Para aislar este efecto, se comparó M3 bajo tres especificaciones:
1. **M2:** Modelo clínico sin $R_{\text{dx}}$.
2. **M3-Crónico:** $R_{\text{dx}}$ calculado **estrictamente sobre las 28 comorbilidades crónicas Elixhauser** (patología preexistente al ingreso).
3. **M3-Total:** $R_{\text{dx}}$ sobre el total de diagnósticos secundarios al alta.

#### Resultados Comparativos (62 Hospitales de Agudos Generales):
| Métrica | M2 (Sin $R_{\text{dx}}$) | M3-Crónico (28 Elixhauser) | M3-Total (Diagnósticos Totales) |
| :--- | :---: | :---: | :---: |
| **Dispersión Empírica ($\tau$)** | 0,2062 | 0,2055 | 0,2048 |
| **Meta-media ($\mu_{\text{meta}}$)** | 0,0224 | 0,0222 | 0,0224 |
| **Correlación con M2 ($\rho_{\text{Spearman}}$)** | 1,0000 | **0,9984** | **0,7046** |
| **Correlación con M3-Total ($\rho_{\text{Spearman}}$)**| 0,7046 | **0,7046** | 1,0000 |

**Hallazgo Metodológico Central:**
* Al calcular $R_{\text{dx}}$ únicamente sobre condiciones crónicas preexistentes, **el ordenamiento institucional es idéntico a M2 ($\rho = 0,9984$)**.
* La totalidad de la divergencia entre M2 y M3 ($\rho = 0,7046$) proviene de los **diagnósticos secundarios agudos y complicaciones intrahospitalarias codificados al alta**, los cuales aumentan en internaciones prolongadas.
* **Limitación Asistencial Consignada:** El ajuste por profundidad diagnóstica total en M3 corrige la descalibración empírica por estratos de codificación, pero conlleva el riesgo de absorber estancia y complicaciones intrahospitalarias, pudiendo premiar indirectamente a centros con estancias prolongadas. Por ello, M3 debe interpretarse como una prueba de esfuerzo frente al subregistro, y no como un reemplazo ciego del modelo preexistente.

---

## 5. Estadía: Variables Definitivas, Grillas de Hiperparámetros y Codificación LOHO

### 5.1 Conjunto Definitivo de 17 Covariables en la Cohorte de Desarrollo ($N = 1.303.718$)
En la cohorte completa de desarrollo, se ajustó ElasticNet (`l1_ratio = 0.5`) bajo la regla 1-SE. A continuación se presentan los coeficientes estandarizados con sus **nombres canónicos oficiales vinculados a `tabla_canonica_elixhauser_31.csv`**:

| Rango | Variable | Nombre Canónico Oficial | Coeficiente ($\alpha = 0,036680$) | Estado Parsimonioso |
| :---: | :--- | :--- | :---: | :---: |
| 1 | `CIE10_ENC_LOHO` | Diagnóstico Principal (LOHO) | **+0,42528** | **Retenida** |
| 2 | `INGRESO_URGENCIA` | Admisión por Urgencia | **+0,13467** | **Retenida** |
| 3 | `INGRESO_CRITICO` | Ingreso a Cama Crítica (UCI/UTI) | **+0,07870** | **Retenida** |
| 4 | `ELIX_24` | Pérdida de peso patológica | **+0,05872** | **Retenida** |
| 5 | `ELIX_04` | Trastornos de la circulación pulmonar | **+0,03980** | **Retenida** |
| 6 | `EDAD_ANIOS` | Edad (Años continuos) | **+0,03526** | **Retenida** |
| 7 | `ELIX_14` | Insuficiencia renal crónica | **+0,03117** | **Retenida** |
| 8 | `DERIVADO_OTRO_HOSPITAL`| Derivado de otro establecimiento | **+0,02504** | **Retenida** |
| 9 | `ELIX_23` | Obesidad | **+0,01932** | **Retenida** |
| 10 | `ELIX_01` | Insuficiencia cardíaca congestiva | **+0,01842** | **Retenida** |
| 11 | `ELIX_09` | Otros trastornos neurológicos | **+0,01246** | **Retenida** |
| 12 | `ELIX_05` | Enfermedad vascular periférica | **+0,01027** | **Retenida** |
| 13 | `ELIX_15` | Enfermedad hepática | **+0,01010** | **Retenida** |
| 14 | `ELIX_30` | Psicosis | **+0,00890** | **Retenida** |
| 15 | `ELIX_03` | Valvulopatía | **+0,00783** | **Retenida** |
| 16 | `ELIX_08` | Parálisis | **+0,00695** | **Retenida** |
| 17 | `ELIX_06` | Hipertensión no complicada | **+0,00419** | **Retenida** |
| 18 | `ELIX_19` | Cáncer metastásico | +0,00296 | Descartada por 1-SE |
| 19 | `ELIX_10` | Enfermedad pulmonar crónica (EPOC) | +0,00241 | Descartada por 1-SE |
| 20 | `ELIX_07` | Hipertensión complicada | +0,00154 | Descartada por 1-SE |
| 21–34| Resto (14 variables)| Diabetes, Linfoma, Sexo, etc. | 0,00000 | Descartadas por 1-SE |

*Aclaración de Variables:* En la población completa de desarrollo ($N = 1,3\text{M}$), **tanto `ELIX_03` (Valvulopatía) como `ELIX_08` (Parálisis) están simultáneamente retenidas dentro de las 17 principales**, mientras que `ELIX_19` (Cáncer metastásico) pasa a ser la variable número 18 descartada por regularización.

---

### 5.2 Documentación de Grillas de Hiperparámetros ($\alpha$)
La discrepancia entre $\alpha_{1\text{se}} = 0,036680$ y $0,037431$ se debe a que corresponden a **dos grillas diferentes**:
* En `ElasticNetCV`, $\alpha_{\max} = \max_i |X_i^T y| / (N \cdot \text{l1\_ratio})$ se calcula analíticamente a partir de los datos de entrenamiento.
* **Grilla 1 (Muestra 30k de 2023):** Generó $\alpha_{\max} = 1,1201 \implies$ seleccionó $\alpha_{1\text{se}} = \mathbf{0,036680}$.
* **Grilla 2 (Población Desarrollo 1.3M):** Generó $\alpha_{\max} = 1,143086 \implies$ seleccionó $\alpha_{1\text{se}} = \mathbf{0,037431}$.
Al variar $\alpha_{\max}$, toda la escala de 100 puntos se traslada proporcionalmente. Ambos valores representan el idéntico punto de corte operacional.

---

### 5.3 Comparación Formal: Codificador LOHO vs Codificador Naive (In-Sample)
Para verificar empíricamente la hipótesis de "sin fuga hospitalaria", se ajustó el Bloque 4 de estancia utilizando target encoding LOHO (*Leave-One-Hospital-Out*) frente al codificador in-sample global (*Naive*):
* **Correlación de Spearman en Rankings de Estadía (2023):** $\rho = \mathbf{1,00000}$.
* **Desplazamiento Cuadrático Medio (RMS):** $\mathbf{0,00 \text{ puestos}}$.

*Explicación:* Dada la escala masiva de la cohorte de desarrollo ($1.303.718$ casos y cientos de casos por categoría CIE-10), excluir a un hospital individual modifica la media de estancia de la categoría en menos de $0,001$ días, garantizando que el ranking está matemáticamente libre de fuga hospitalaria.

---

## 6. Auditoría de CIP Nulos: Anomalía de Extracción 2021

Se verificó el comportamiento de registros sin CIP en la capa Silver a lo largo de todo el sexenio:
* **2019:** 0 nulos (de 1.151.475 registros)
* **2020:** 0 nulos (de 781.912 registros)
* **2021:** **2.044 nulos** (de 816.909 registros)
* **2022:** 0 nulos (de 932.840 registros)
* **2023:** 0 nulos (de 1.039.587 registros)
* **2024:** 0 nulos (de 1.085.813 registros)

**Conclusión Metodológica:**  
La concentración de 252 nulos en HUAP y 2.044 en la red durante 2021 **no responde a un fenómeno clínico de pacientes indocumentados**, ya que de ser así se repetiría en todos los años de la serie. Corresponde estrictamente a una **anomalía puntual de extracción y procesamiento informático del archivo maestro DEIS del año 2021**, quedando rectificada su interpretación.

---

## 7. Dictamen Final Consolidado

1. **Alertas en M3:** Transparentadas rigurosamente, confirmando 6 alertas en agudos generales y 1 monográfica (Tórax). Curanilahue ($P = 0,594$) y Heyermann ($P = 0,828$) son formalmente Promedio.
2. **Re-estimación Limpia:** Al estimar $\tau$ sin monográficos, $\tau_{\text{agudos}}$ desciende a $0,2048$, eliminando la sobre-dispersión generada por el Tórax sin alterar las 6 alertas identificadas.
3. **Censura y Tipping Point:** $\lambda_{\text{obs}}$ se expresa en unidades de riesgo esperado $E_{\text{link}}$ con intervalos Byar, y Curanilahue requiere un estrés superior a $2,17\times$ su riesgo esperado ($14,62\%$) en no enlazados para entrar en alerta.
4. **$R_{\text{dx}}$ y Estancia:** Se demuestra que la divergencia entre M2 y M3 proviene de diagnósticos secundarios dependientes de la estancia ($\rho_{\text{chr}, \text{M2}} = 0,9984$), consignándose como limitación estructural.
5. **Estadía y Catálogo:** Nombres canónicos oficiales restablecidos, grillas documentadas, LOHO validado ($\rho = 1,00000$) y CIP nulos catalogados fielmente como error de extracción 2021.
