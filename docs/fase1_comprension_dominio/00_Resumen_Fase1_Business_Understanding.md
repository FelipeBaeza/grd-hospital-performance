**Fase 1 — Comprensión del Dominio**  
**Proyecto de Tesis: ***Sistema de evaluación del desempeño hospitalario ajustado por riesgo mediante aprendizaje automático*  
**Estudiante:** Felipe Ignacio Baeza Muñoz |  **Carrera:** Ingeniería de Ejecución en Computación e Informática — USACH  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSPBCj7fFwtCmJHAjAU2QtIq6DIzW7UHAMBfnGt1V8fHEQAA3rsexOkF3va0dq8AAAAASUVORK5CYII=)  
**1. ¿De qué trata el proyecto y por qué es importante?**  
En Chile, cada vez que un paciente sale de un hospital —ya sea de alta, trasladado o fallecido— se registra un **"egreso hospitalario"** con toda su información clínica: diagnósticos, días de hospitalización, resultado, médicos tratantes, entre otros. El sistema que clasifica y organiza esta información se llama  **GRD (Grupos Relacionados por Diagnóstico)**, y actualmente genera más de un millón de registros al año en la red hospitalaria pública.  
El problema es que **nadie usa esos datos para evaluar si los hospitales están funcionando bien o mal**. Las plataformas oficiales solo publican promedios muy simples (como la tasa de mortalidad general) sin ningún ajuste por la gravedad de los pacientes que atiende cada centro.  
Esto crea una comparación injusta: un hospital que atiende casos muy complejos (pacientes mayores, con múltiples enfermedades previas, diagnósticos graves) va a tener naturalmente más muertes y estadías más largas que uno que atiende casos simples. Comparar sus tasas directamente no dice nada sobre la calidad de la atención.  
**Mi proyecto busca resolver eso.** Construyo un sistema computacional que:  
1. **Procesa** los 5.808.536 registros históricos de egresos entre 2019 y 2024, obtenidos formalmente desde FONASA mediante la Ley de Transparencia.  
2. **Predice**, usando Machine Learning, cuántas muertes y cuántos días de hospitalización *debería* tener un hospital, dado el perfil clínico real de sus pacientes.  
3. **Compara** lo predicho (Esperado) con lo que realmente ocurrió (Observado), generando un indicador llamado  **Razón O/E** (Observado sobre Esperado).  
4. **Muestra** esa comparación en una plataforma web para que gestores y ciudadanos puedan interpretar el desempeño hospitalario de forma justa.  
El sistema no pretende rankear hospitales ni señalar culpables. Su propósito es **generar señales de alerta objetivas** que ayuden a los propios hospitales a identificar en qué procesos pueden mejorar.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNhwgJmkPYLLpnRgQU2QtIq6DIze3UGAMBf3Gu1VcfHEQAA3rseaHkEMn1wK7sAAAAASUVORK5CYII=)  
**2. Las reglas fundamentales del modelo**  
Para que el sistema sea válido científicamente, tiene reglas muy estrictas que no se pueden violar.  
**Regla 1 — Solo información disponible al momento del ingreso**  
El modelo de predicción solo puede "saber" lo que sabría un médico cuando el paciente llega al hospital: su edad, su sexo, su diagnóstico de ingreso y sus enfermedades previas. **No puede usar información que ocurrió después**, como cuántos días estuvo en UCI, si lo operaron, o cuántos traslados tuvo.  
Si violamos esta regla, el modelo "hace trampa": ya sabe el resultado y no está realmente prediciendo nada. En ciencias de datos, esto se llama **fuga de datos (data leakage)** y es el error más grave en modelos clínicos.  
**Regla 2 — El hospital no puede ser variable predictora**  
No puedo decirle al modelo "este paciente está en el Hospital X". Si lo hiciera, el modelo aprendería el historial de resultados de ese hospital y la predicción sería circular: estaría evaluando el desempeño *usando* el desempeño histórico como insumo. El código del hospital solo se usa al final, para  *agregar* los resultados por institución.  
**Regla 3 — Las comorbilidades son el corazón del ajuste**  
Una comorbilidad es una enfermedad que el paciente ya tenía antes de ingresar: diabetes, insuficiencia cardíaca, cáncer, entre otras. Un paciente con más enfermedades crónicas previas tiene naturalmente más riesgo, independiente del hospital donde esté. El estándar internacional para medir esto a partir de diagnósticos clínicos se llama **Índice de Elixhauser**, que clasifica las comorbilidades en 31 categorías distintas. Mi modelo lo implementa completo.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OQQmAABRAsSdYxKY/jMFMIZ7ECt5E2BJsmZmt2gMA4C+Otbqr8+sJAACvXQ85QgYXd/O+eQAAAABJRU5ErkJggg==)  
**3. Los datos con los que trabajo**  
| | |  
|-|-|  
| **Característica** | **Detalle** |   
| **Fuente** | FONASA — obtenida formalmente mediante Ley 20.285 de Transparencia |   
| **Período** | Egresos hospitalarios 2019 – 2024 |   
| **Total de registros** | 5.808.536 egresos |   
| **Columnas por registro** | 129 variables clínicas y administrativas |   
| **Formato original** | Archivos de texto plano separados por \|, uno por año |   
| **Hardware de procesamiento** | Intel Core i5 13ª gen, 16 GB RAM, Linux (Ubuntu) |   
   
Dado el volumen (5.8 millones de registros en 16 GB de RAM), uso una librería especializada llamada **Polars** que procesa los datos en bloques sin cargarlos todos en memoria al mismo tiempo.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNhYMMAKlD4OzrxgQU2QtIq6DIzR3UFAMBf3Gu1VefXEwAAXtsfSqADWz4G/HUAAAAASUVORK5CYII=)  
**4. Cómo se organiza el procesamiento — Arquitectura Medallón**  
El pipeline de datos sigue una arquitectura en capas, donde cada capa tiene un propósito específico:  
Archivos originales (.txt)  
          ↓  
     CAPA BRONCE  
     (Ingesta cruda: datos tal como vienen, solo validación de estructura)  
          ↓  
     CAPA PLATA  
     (Limpieza: tipado de fechas, normalización de texto, filtros de cohorte)  
          ↓  
     CAPA ORO  
     (Features finales: variables transformadas, listas para el modelo ML)  
          ↓  
     MODELO ML  
     (Predicción de mortalidad + días de estadía)  
          ↓  
     INDICADOR O/E  
     (Comparación observado vs. esperado por hospital)  
          ↓  
     PLATAFORMA WEB  
     (Visualización interactiva para gestores y ciudadanos)  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSPBCUZfEnoYmFDBhAU2QtIq6DIzW7UHAMBfnGt1V8fXEwAAXrse/wcF74lXkIsAAAAASUVORK5CYII=)  
**5. Qué hace cada etapa del pipeline**  
**Etapa 1 — Validación del esquema**  
Antes de procesar nada, el sistema verifica que los archivos tengan exactamente las 129 columnas esperadas. Si falta alguna o hay una desconocida, el pipeline se detiene y muestra exactamente cuál es el problema. Esto existe porque en 2019 una columna llamada ID_BENEFICIARIO fue renombrada a CIP_ENCRIPTADO en los años siguientes, y el sistema maneja ese cambio automáticamente.  
**Etapa 2 — Ingesta (Capa Bronce)**  
Lee los 6 archivos de texto (uno por año) y los guarda en un formato eficiente llamado Parquet. Si una fila está tan corrupta que no se puede leer, se guarda aparte en una tabla de "rechazos" con el motivo del error. Al final, el sistema verifica matemáticamente que el número de filas leídas más las rechazadas sume exactamente el número de líneas del archivo original.  
**También en esta etapa se genera el ** **ID_EPISODIO** **:** un identificador único para cada registro que no existía en los datos originales. Lo calculo usando una fórmula matemática a partir del archivo de origen, el número de línea y el contenido de la fila. Sin este ID, sería imposible cruzar tablas o rastrear a un paciente entre hospitalizaciones.  
**Etapa 3 — Limpieza (Capa Plata)**  
- Convierte las fechas al formato correcto (había dos formatos distintos en los datos).  
- Corrige los números decimales escritos con coma en vez de punto.  
- Estandariza el texto de diagnósticos: todo a mayúsculas, sin tildes, sin espacios extra.  
- Aplica los **filtros de cohorte**: excluye los registros que no corresponde analizar (recién nacidos sanos, episodios obstétricos sin complicación, GRD no agrupables).  
- Si un valor es imposible (ej: edad negativa, fecha de alta anterior al ingreso), ese valor específico se marca como inválido, pero **el registro completo del paciente no se elimina**.  
- Clasifica el tipo de alta en tres categorías: **muerte confirmada**,  **sobrevida confirmada** o  **desenlace desconocido (censurado)**.  
**Etapa 4 — Features para el modelo (Capa Oro)**  
Crea las variables que el modelo de ML va a usar como insumo:  
| | | |  
|-|-|-|  
| **Variable** | **Qué mide** | **Fuente** |   
| EDAD_ANIOS | Edad exacta en años al ingreso | FECHA_INGRESO - FECHA_NACIMIENTO |   
| MES_INGRESO | Mes del año en que ingresó | FECHA_INGRESO |   
| ES_FIN_SEMANA | Si ingresó sábado o domingo (1/0) | FECHA_INGRESO |   
| CIE10_3C | Diagnóstico principal (primeros 3 caracteres del código CIE-10) | DIAGNOSTICO1 |   
| GRUPO_CLINICO | Categoría amplia del diagnóstico (capítulo CIE-10) | DIAGNOSTICO1 |   
| TIPO_INGRESO | Si el ingreso fue urgencia o programado | TIPO_INGRESO |   
| N_EGRESOS_12M | Cuántas veces fue hospitalizado en el último año | CIP_ENCRIPTADO + FECHA_INGRESO |   
| ELIX_01…ELIX_31 | 31 indicadores de comorbilidades crónicas (Elixhauser) | DIAGNOSTICO2–DIAGNOSTICO35 |   
| SCORE_VANWALRAVEN | Suma ponderada de las 31 comorbilidades | Calculado de ELIX_01…ELIX_31 |   
   
Y dos variables **objetivo** que el modelo aprende a predecir:  
| | | |  
|-|-|-|  
| **Variable** | **Tipo de problema ML** | **Fuente** |   
| MORTALIDAD_BINARIA | Clasificación binaria (1=falleció, 0=sobrevivió) | TIPOALTA |   
| ESTANCIA | Regresión (número de días) | FECHAALTA - FECHA_INGRESO |   
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNhYMMAKlD4OzrxgQU2QtIq6DIzR3UFAMBf3Gu1VefXEwAAXtsfSqADWz4G/HUAAAAASUVORK5CYII=)  
**6. El Índice de Elixhauser — el ajuste clave del modelo**  
Este es el componente más importante del ajuste por riesgo. Funciona así:  
Cada paciente tiene registrados hasta 35 diagnósticos: uno principal (el motivo del ingreso) y hasta 34 diagnósticos secundarios (sus enfermedades previas). El índice de Elixhauser **ignora el diagnóstico principal** (para no "contaminar" la predicción con el motivo del ingreso) y escanea los 34 diagnósticos secundarios buscando códigos CIE-10 que correspondan a cada una de las 31 categorías de comorbilidades crónicas.  
**¿Por qué 31 categorías en vez de un solo número?**  
   
 Porque las enfermedades interactúan entre sí de formas complejas. Un paciente con diabetes *sin* complicaciones tiene un riesgo diferente al que tiene diabetes  *con* complicaciones renales. Al mantener los 31 indicadores separados, el modelo puede detectar esas interacciones automáticamente. Si los colapsara en un solo número, perdería esa información.  
**Las 31 categorías incluyen:** insuficiencia cardíaca, arritmias, enfermedad valvular, hipertensión, diabetes (con y sin complicación), enfermedad pulmonar crónica, cáncer, insuficiencia renal, enfermedad hepática, VIH/SIDA, obesidad, depresión, entre otras.  
**Un detalle importante sobre los datos chilenos:** Como la base de datos no incluye un marcador que indique si un diagnóstico ya existía al ingreso o apareció *durante* la hospitalización (se llama "Present On Admission" o POA, y en Chile no está registrado), existe un riesgo de que algunas comorbilidades registradas sean en realidad complicaciones que ocurrieron dentro del hospital. Para mitigar esto, uso la "lista blanca" de Elixhauser, que excluye por diseño los diagnósticos que estadísticamente tienen mayor probabilidad de ser complicaciones intrahospitalarias.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AUBBAsUeCE4yeIiT9CRVMWGAjJK2CbjNzVGcAAPzF2qu7Wl9PAAB47XoA/vcF8exqpY4AAAAASUVORK5CYII=)  
**7. Qué hace el sistema con los datos sensibles (Privacidad)**  
Los archivos de FONASA contienen identificadores encriptados de pacientes y médicos. Si el sistema detectara accidentalmente un RUT real, un correo o un teléfono en cualquier columna, tiene la instrucción de:  
1. **Contar** cuántos encontró y en qué columna.  
2. **Aislar** el registro en cuarentena silenciosa.  
3. **Jamás escribir** el valor real en ningún archivo de log, reporte de error o tabla.  
Adicionalmente, el sistema es **stateless** en la plataforma web: cuando un ciudadano ingresa un perfil clínico para consultar el riesgo estimado, esos datos no se guardan en ningún servidor.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AUBBAsUfyVTCg9UygEBVsWGAjJK2CbjNzVGcAAPzFtapV7V9PAAB47X4AEWIEM8iQs0EAAAAASUVORK5CYII=)  
**8. Cómo se valida el modelo (Criterios de éxito)**  
Construyo **dos modelos de Machine Learning independientes**: uno para predecir mortalidad y otro para predecir días de hospitalización. Ambos se entrenan con datos de los años  **2019 a 2023** y se prueban con datos del año  **2024** que el modelo nunca vio durante el entrenamiento (validación temporal).  
El modelo se considera válido si supera estos umbrales:  
| | | |  
|-|-|-|  
| **Criterio** | **Métrica** | **Umbral** |   
| Poder de discriminación (mortalidad) | PR AUC (Precision-Recall) | Mayor que la línea base logística |   
| Error de predicción (estadía) | RMSE | Menor que la línea base logística |   
| Calibración estadística | Brier Skill Score | Mayor que 0 |   
| Calibración de probabilidades | Pendiente de calibración | Entre 0.8 y 1.2 |   
| Razón O/E global | Observado / Esperado total | Entre 0.95 y 1.05 |   
   
***¿Por qué Precision-Recall y no el AUC estándar?*** * Porque la * *mortalidad intrahospitalaria es un evento raro (~2.8% de los egresos). En datos tan desbalanceados, el AUC estándar puede verse artificialmente bien aunque el modelo falle en detectar los casos positivos. Precision-Recall AUC es más exigente y más honesto.*  
Una vez validados los modelos, los resultados se agregan por hospital para calcular la **Razón O/E** y se visualizan en  **gráficos de embudo (funnel plots)** con límites estadísticos al 95% y 99.8%, siguiendo el método de Spiegelhalter (2005). Esto permite identificar qué hospitales tienen un desempeño genuinamente inusual, corrigiendo el efecto de que los hospitales pequeños tengan naturalmente más variabilidad en sus estadísticas.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OMQ2AABAAsSNBCkJfFEIwwIgHRiywEZJWQZeZ2ao9AAD+4lyruzq+ngAA8Nr1AOHsBegrsOrIAAAAAElFTkSuQmCC)  
**9. Glosario de términos clave**  
| | |  
|-|-|  
| **Término** | **Explicación simple** |   
| **GRD** | Sistema que clasifica los egresos hospitalarios por tipo de diagnóstico y recursos consumidos |   
| **Egreso hospitalario** | Cada vez que un paciente sale del hospital (alta, traslado o fallecimiento) |   
| **O/E (Observado/Esperado)** | Indicador que compara lo que realmente pasó vs. lo que el modelo esperaba |   
| **Elixhauser** | Estándar internacional de 31 categorías para medir enfermedades crónicas previas |   
| **Data leakage** | Error grave en ML: usar información del futuro para predecir el futuro |   
| **Calibración** | Que las probabilidades del modelo coincidan con la realidad (si dice 30%, pase en ~30% de los casos) |   
| **Funnel plot** | Gráfico estadístico que compara hospitales ajustando por su tamaño y evitando falsas alarmas |   
| **Upcoding** | Registrar diagnósticos adicionales de mayor gravedad para inflar el perfil del paciente |   
| **Censura estadística** | Pacientes cuyo desenlace es desconocido (trasladados) — se excluyen del target de mortalidad |   
| **POA** | "Present On Admission" — marcador que indica si un diagnóstico existía antes del ingreso |   
| **IEMC** | Índice de Exhaustividad y Mérito de Codificación — mide si un hospital registra diagnósticos acorde a su casuística |   
| **Pipeline** | Secuencia de pasos automatizados que transforma los datos crudos en resultados finales |   
| **Parquet** | Formato de archivo columnar eficiente para grandes volúmenes de datos analíticos |   
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OQQmAABRAsSd4NIGJjPWxpgGsYQVvImwJtszMXp0BAPAX91pt1fH1BACA164HhZwEOFrXVOsAAAAASUVORK5CYII=)  
**10. Alcance y limitaciones reconocidas**  
El sistema **no** pretende:  
- Ser una medición definitiva de la calidad asistencial de un hospital.  
- Señalar culpables ni generar rankings punitivos para el cuerpo médico.  
- Corregir el sesgo de alta anticipada (pacientes trasladados antes de fallecer para que la muerte no quede registrada en el hospital emisor) — este sesgo existe en todos los sistemas similares del mundo y se declara como limitación explícita.  
El sistema **sí** pretende:  
- Generar señales estandarizadas y objetivas que sirvan como punto de partida para que los gestores de salud identifiquen qué procesos internos requieren revisión.  
- Ser completamente reproducible: cualquier persona con acceso a los datos originales puede ejecutar el mismo pipeline y obtener exactamente los mismos resultados.  
- Ser transparente: cada decisión metodológica está documentada y justificada.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSPBCUZfEnoYmFDBhAU2QtIq6DIzW7UHAMBfnGt1V8fXEwAAXrse/wcF74lXkIsAAAAASUVORK5CYII=)  
*Documento de Fase 1 — Comprensión del Dominio*  
   
 *Universidad de Santiago de Chile | Departamento de Ingeniería Informática*  
   
 *Fecha: Septiembre 2026*  
