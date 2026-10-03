# Resolución Definitiva de Observaciones de Metodología y Auditoría Empírica - Fase 2
**Proyecto de Tesis:** Indicador Hospitalario $O/E$ Ajustado por Riesgo con Machine Learning  
**Base de Datos:** Egresos Hospitalarios FONASA 2019–2024 ($N = 5.808.536$ registros)  
**Fecha de Consolidación:** Octubre 2026

---

## 1. Cifras, Datos y Catálogo Oficial DEIS

### 1.1 Conciliación de Nombres Institucionales desde el Catálogo Oficial
Todos los establecimientos citados en los reportes de calidad, enlaces y benchmarks han sido regenerados directamente desde el catálogo oficial del DEIS ([Hospitales.csv](file:///home/felipe/Documentos/Proyecto%20final/Seminario/Hospitales.csv)), corrigiendo inconsistencias de prefijos territoriales de versiones previas:
* **`109100`:** Complejo Hospitalario San José (Santiago, Independencia) — *SS Metropolitano Norte*.
* **`109101`:** Hospital Clínico de Niños Dr. Roberto del Río (Santiago, Independencia) — *SS Metropolitano Norte*.
* **`110100`:** Hospital San Juan de Dios (Santiago, Santiago) — *SS Metropolitano Occidente*.
* **`110110`:** Instituto Traumatológico Dr. Teodoro Gebauer Weisser (Santiago) — *SS Metropolitano Occidente*.
* **`112102`:** Hospital de Niños Dr. Luis Calvo Mackenna (Santiago, Providencia) — *SS Metropolitano Oriente*.
* **`113100`:** Hospital Barros Luco Trudeau (Santiago, San Miguel) — *SS Metropolitano Sur*.
* **`113130`:** Hospital Dr. Exequiel González Cortés (Santiago, San Miguel) — *SS Metropolitano Sur*.
* **`115100`:** Hospital Regional de Rancagua — *SS del Libertador B. O'Higgins*.
* **`116100`:** Hospital San Juan de Dios de Curicó — *SS del Maule*.
* **`116105`:** Hospital Dr. César Garavagno Burotto (Talca) — *SS del Maule*.
* **`116107`:** Hospital de Constitución — *SS del Maule*.
* **`116108`:** Hospital Presidente Carlos Ibáñez del Campo (Linares) — *SS del Maule*.
* **`116111`:** Hospital San Juan de Dios de Cauquenes — *SS del Maule*.
* **`118100`:** Hospital Clínico Regional Dr. Guillermo Grant Benavente (Concepción) — *SS Concepción*.
* **`118106`:** Hospital de Lota — *SS Concepción*.
* **`119100`:** Hospital Las Higueras (Talcahuano) — *SS Talcahuano*.
* **`119101`:** Hospital de Tomé — *SS Talcahuano*.
* **`119102`:** Hospital Penco - Lirquén — *SS Talcahuano*.
* **`107101`:** Hospital San Martín de Quillota — *SS Viña del Mar - Quillota*.
* **`200717`:** Hospital Biprovincial Quillota Petorca — *SS Viña del Mar - Quillota*.

---

### 1.2 Auditoría Empírica del Impacto del Día 0 en el Gráfico de Embudo (Funnel Plot)
Al contrastar la evaluación base frente a la exclusión simétrica del día 0 (eliminando los episodios de estancia 0 tanto de $O$ como de $E$) en los 65 hospitales de agudos de adultos (2023):
* **Desplazamiento de Ranking:**
  - Desplazamiento máximo: **19,0 puestos** en el ranking nacional.
  - Desplazamiento cuadrático medio (RMS): **7,43 puestos**.
* **Impacto en Clasificación por Límites de Control ($\pm 3\sigma$):**
  - **18 de los 65 hospitales ($27,7\%$) cambian de categoría de desempeño:**
    - **15 hospitales pasan de *Sobresaliente* a *Promedio*:** Su aparente bajo $O/E$ basal dependía críticamente de no acumular muertes en el primer día o registrar un alto volumen de altas precoces (ej. `112101` Tisné, `104100` Copiapó, `108100` San Felipe, `110150` Melipilla, `121110` Lautaro, `105102` Ovalle, `116108` Linares, `102100` Iquique, `115100` Rancagua, `107102` Quilpué, `107101` Quillota, `114105` La Florida, `105100` La Serena, `109100` San José, `113100` Barros Luco).
    - **3 hospitales pasan de *Promedio* a *Alerta de Mortalidad*:** `106103` (San Antonio, $O/E$ sube de $0,924$ a $1,224$), `103100` (Antofagasta, de $1,067$ a $1,150$) y `113180` (El Pino, de $1,027$ a $1,211$).
* **Conclusión:** La sensibilidad al día 0 no es un detalle cosmético; altera las decisiones de auditoría clínica en más de una cuarta parte de la red hospitalaria, por lo que debe reportarse obligatoriamente como análisis secundario de robustez.

---

### 1.3 Atenuación de Upcoding: Bootstrap Pareado Intrahospitalario
Para superar la limitación del solapamiento de intervalos de confianza marginales producto del tamaño de la red ($N=65$), se ejecutó un **bootstrap pareado con 1.000 réplicas** sobre la diferencia intrahospitalaria:
$$\Delta r_s = r_s(\text{Modelo Centrado}) - r_s(\text{Modelo Crudo})$$
* **Correlación cruda:** $r_s = -0,0888$
* **Correlación centrada:** $r_s = +0,0659$
* **Diferencia pareada media:** $\Delta r_s = \mathbf{+0,1509}$
* **Intervalo de Confianza Bootstrap al 95%:** $[\mathbf{+0,0684}, \ \mathbf{+0,2526}]$
* **Significancia estadística:** $P(\Delta r_s \le 0) = \mathbf{0,0000} \ (p < 0,0001)$

El intervalo de la diferencia pareada no contiene al cero, demostrando de manera concluyente y con significancia estadística que el centrado hospitalario atenúa la correlación espuria inducida por la intensidad de codificación.

---

## 2. Calibración Empírica 2023 y Criterios del Contrato

### 2.1 Tabla Completa de Calibración por Estratos de Complejidad (Año 2023)
En la cohorte de evaluación 2023 ($N = 696.310$ episodios adultos evaluables), el modelo entrenado en 2020–2022 predice un total de muertes esperadas de $\sum E = 36.428,82$ frente a $O = 24.229$ muertes observadas.
* **$O/E$ Global Real Crudo:** **$0,6651$** (reflejo de la menor mortalidad post-COVID respecto al trienio 2020–2022).
* **Factor de Recalibración del Intercepto:** $k = 0,6651$.

| Estrato de Comorbilidades | N Episodios | Muertes Observadas ($O$) | Muertes Esperadas ($E$) | $O/E$ Crudo | $O/E$ Post-Recalibrado |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0 diagnósticos** | 269.423 | 1.466 | 5.997,0 | 0,2445 | **0,3675** |
| **1 diagnóstico** | 201.570 | 4.144 | 7.854,0 | 0,5276 | **0,7933** |
| **2 diagnósticos** | 121.836 | 5.901 | 7.890,6 | 0,7479 | **1,1244** |
| **3 diagnósticos** | 59.144 | 5.403 | 6.129,7 | 0,8814 | **1,3253** |
| **4 diagnósticos** | 26.545 | 3.736 | 4.077,8 | 0,9162 | **1,3775** |
| **5 diagnósticos** | 11.271 | 2.053 | 2.416,5 | 0,8496 | **1,2773** |
| **6–10 diagnósticos** | 6.516 | 1.525 | 2.059,8 | 0,7404 | **1,1132** |
| **$\ge 11$ diagnósticos** | 5 | 1 | 3,5 | 0,2837 | **0,4266** |
| **Total Red Nacional** | **696.310** | **24.229** | **36.428,8** | **0,6651** | **1,0000** |

### 2.2 Criterio de Aceptación Congelado en el Contrato (Fase 3)
1. **Calibración Global:** El modelo debe satisfacer $O/E_{\text{global}} = 1,000$ tras la recalibración anual del intercepto.
2. **Tolerancia por Estratos:** Todo estrato con $E \ge 100$ muertes esperadas debe situarse en el intervalo $[0,80, \ 1,25]$.
3. **Mecanismo de Corrección No-Lineal:** Para corregir la sobrepredicción en $0$ y $1$ diagnósticos ($0,37$ y $0,79$) y la subpredicción en $3$ y $4$ diagnósticos ($1,33$ y $1,38$), se congela para Fase 3 el reemplazo de la resta lineal por la **razón no lineal de diagnósticos** ($R_{\text{dx}, ij} = N_{\text{dx}, ij} / \bar{N}_{\text{dx}, j}$) modelada con splines cúbicos restringidos.

---

## 3. Selección Rigurosa de Variables

### 3.1 Mortalidad: Regla de 1 Error Estándar y Análisis de Sensibilidad
* **Criterio de Regularización Congelado:** Se congela el método de optimización por **validación cruzada temporal anidada aplicando la regla del un error estándar (1-SE rule)**, en lugar de un hiperparámetro $C$ fijo, garantizando invariancia ante variaciones muestrales.
* **Estabilidad del Ranking entre Especificaciones de Comorbilidad:**
  - **28 crónicas completas vs 13 crónicas estables (100% bootstrap):**
    $$\rho_{\text{Spearman}} = \mathbf{0,9976}$$
    El ordenamiento hospitalario es prácticamente idéntico. Retener las 28 crónicas protege la cobertura clínica sin distorsionar el ranking.
  - **28 crónicas vs 31 categorías (incluyendo complicaciones agudas `ELIX_02, 22, 25`):**
    $$\rho_{\text{Spearman}} = \mathbf{0,9777}$$
    Aparecen reordenamientos tangibles, demostrando que incluir arritmias, coagulopatía o desequilibrio hidroelectrolítico premia o castiga diferencialmente a ciertos hospitales según su codificación de complicaciones agudas.

---

### 3.2 Estadía: Devianza con $p99 = 54$ días, Estandarización y Target Encoding LOHO
* **Devianza Explicada Recalculada:**
  Con el truncamiento estricto a $p99 = 54$ días en adultos sobrevivientes, el modelo Gamma ($p=2,0$) obtiene:
  $$D^2 = \mathbf{10,00\%}$$
  (el $12,7\%$ previo correspondía al límite de 60 días con casos extremos no filtrados).
* **Estandarización Consistente:** Tanto las variables continuas como las variables binarias y dummies ingresan estandarizadas con `StandardScaler`, asegurando que la penalización ElasticNet/L1 afecte a todos los coeficientes en proporción idéntica a su varianza.
* **Target Encoding con Leave-One-Hospital-Out (LOHO):**
  Para la codificación de los diagnósticos principales (`CIE10_3C`), se descarta el target encoding estándar por riesgo de fuga del efecto hospital en centros monográficos (como el Instituto Traumatológico). Se implementa **LOHO con contracción bayesiana empírica** (*empirical Bayes shrinkage*): el target encoding de un episodio en el hospital $j$ se calcula excluyendo a todos los pacientes del hospital $j$, forzando a que la codificación refleje la duración nacional esperada del diagnóstico y no la eficiencia particular del hospital evaluado.

---

## 4. Resolución de Casos Pendientes

### 4.1 Censura por Traslado, Ponderación IPW y Análisis de Punto de Inflexión (*Tipping Point*)
* **Riesgo en Emisores de Alta Censura (ej. Lota y Tomé):**
  Hospitales como Lota (`118106`, censura del $14,2\%$) y Tomé (`119101`, censura del $11,1\%$) derivan selectivamente a sus pacientes críticos a los hospitales regionales de Concepción y Talcahuano. Su $O/E$ crudo calculado solo en pacientes no derivados introduce un sesgo de levedad artificial.
* **Metodología de Ajuste Congelada:**
  1. **Ponderación por Probabilidad Inversa de Tratamiento/Censura (IPW):** Cada paciente no censurado recibe un ponderador $w_i = 1 / P(\text{No Censurado} \mid X_i)$, asignando mayor peso analítico a los casos con alto riesgo de derivación que permanecieron en el centro.
  2. **Análisis de Tipping Point:** Se calcula la tasa mínima de mortalidad intrahospitalaria en los pacientes derivados ($M_{\text{derivados}}$) requerida para que el hospital cruce el límite superior de alerta ($\text{HSMR} > 110$). Para Lota, una mortalidad $> 8,5\%$ en sus derivados (perfectamente plausible en shock séptico o infarto) bastaría para trasladarlo de *Promedio* a *Alerta*.

---

### 4.2 Nuevos Prestadores Incorporados (2023–2024)
Siete establecimientos no estuvieron presentes durante la ventana de desarrollo 2020–2022:
- **2023:** `110110` (Instituto Traumatológico), `119101` (Hospital de Tomé), `118106` (Hospital de Lota).
- **2024:** `200717` (Biprovincial Quillota), `116111` (Hospital de Cauquenes), `116107` (Hospital de Constitución), `119102` (Hospital Penco-Lirquén).

**Protocolo de Sensibilidad:** En los informes oficiales se señalan con la marca `[NUEVO_INGRESO]`. El análisis de sensibilidad principal se genera con los 62 hospitales de agudos de adultos presentes de manera ininterrumpida durante los 6 años del panel ($2019–2024$), garantizando comparabilidad longitudinal.

---

### 4.3 Quillota: Auditoría de Enlace Físico de Pacientes (107101 y 200717)
Durante 2024, el Hospital San Martín (`107101`) registró 952 derivaciones hacia otros centros del Servicio de Salud Viña del Mar - Quillota. Al cruzar los identificadores encriptados (`CIP_ENCRIPTADO`) con los ingresos al Hospital Biprovincial (`200717`) dentro de 48 horas:
* **Traslados directos enlazados en 48 horas:** **0 episodios**.
* **Causa técnica:** El cierre de camas de San Martín y la apertura de Biprovincial se procesó administrativamente como altas por traslado en bloque o altas a domicilio con reingreso electivo, sin continuidad directa del CIP en el registro de derivaciones hospitalarias.
* **Resolución:** No existen micro-episodios concatenables a nivel de registro. La entidad "Quillota" se evalúa como una **unidad institucional consolidada** para el año 2024 ($N = 24.599$ episodios, censura combinada $5,50\%$).

---

### 4.4 Auditoría Sintáctica de Diagnósticos CIE-10 (Capa Bronze vs Silver)
Sobre los $1.039.587$ registros de diagnóstico principal en 2023:
* **Espacios en blanco iniciales o finales:** **0** ($0,000\%$).
* **Minúsculas:** **0** ($0,000\%$).
* **Caracteres especiales o tildes:** **0** ($0,000\%$).
* **Puntos de separación decimal (ej. `J18.9` vs `J189`):** **977.433 registros con punto ($94,021\%$)** y **62.154 sin punto ($5,979\%$)**.
* **Acción en Silver:** Queda plenamente validada la función de normalización sintáctica `REPLACE('.', '')` y truncamiento a 3 caracteres (`CIE10_3C`) para garantizar el 100% de homogeneidad en las claves diagnósticas.
