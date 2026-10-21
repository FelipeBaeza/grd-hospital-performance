# Informe de recolección inicial de los datos (CRISP-DM 2.1)

**Fuente de los datos: extracto.** Los extractos son una fracción menor de los datos y no son una muestra aleatoria simple: sirven para desarrollar y probar, no para conteos absolutos.

## 1. Qué se recolectó

Se recolectaron 6 archivos anuales de egresos hospitalarios de FONASA (2019 a 2024). Cada archivo se leyó completo como texto, por posición de columna, con 129 columnas por archivo. En total se leyeron 299.994 filas y se rechazaron 0; en cada archivo las filas leídas más las rechazadas suman las líneas de datos.

| Año | Archivo | Bytes | Líneas de datos | BOM | Fin de línea | Codificación | Filas leídas | Rechazadas |
|---|---|---|---|---|---|---|---|---|
| 2019 | GRD_PUBLICO_2019.txt | 24.503.085 | 49.999 | no | CRLF | UTF-8 válido | 49.999 | 0 |
| 2020 | GRD_PUBLICO_2020.txt | 25.154.950 | 49.999 | no | CRLF | UTF-8 válido | 49.999 | 0 |
| 2021 | GRD_PUBLICO_2021.txt | 26.217.228 | 49.999 | si | CRLF | UTF-8 válido | 49.999 | 0 |
| 2022 | GRD_PUBLICO_EXTERNO_2022.txt | 25.709.179 | 49.999 | si | CRLF | UTF-8 válido | 49.999 | 0 |
| 2023 | GRD_PUBLICO_2023.txt | 25.861.939 | 49.999 | si | CRLF | UTF-8 válido | 49.999 | 0 |
| 2024 | GRD_PUBLICO_2024.txt | 26.024.993 | 49.999 | no | CRLF | UTF-8 válido | 49.999 | 0 |

*Archivos anuales y cuadratura de lectura.* *Tabla completa en `tablas/t21_archivos.csv`*

![Barras con el número de filas leídas de cada archivo anual de egresos, de 2019 a 2024.](../figuras/F02_01.png)

*Figura F02_01. Filas leídas por archivo y año. Datos de origen: `tablas/t21_archivos.csv`.*

## 2. Deriva de esquema

Se comparó el nombre de cada una de las 129 columnas de cada año con el esquema canónico. Número total de diferencias de nombre: 1. La segunda columna (identificador del paciente) se llama ID_BENEFICIARIO en 2024 y CIP_ENCRIPTADO en el resto; el lector la trata por posición.

| Posición | Nombre canónico | Año | Nombre en el encabezado |
|---|---|---|---|
| 2 | CIP_ENCRIPTADO | 2024 | ID_BENEFICIARIO |

*Diferencias de nombre de columna respecto del esquema canónico.* *Tabla completa en `tablas/t21_esquema_diferencias.csv`*

## 3. Deriva de formato

Se midió, para columnas clave, la proporción de valores informados en cada formato. Las fechas de ingreso, alta e intervención vienen en AAAA-MM-DD salvo en 2023, donde vienen en DD-MM-AAAA; la fecha de nacimiento mantiene el formato ISO todos los años.

![Mapa de calor con el porcentaje de valores de cada formato (fechas, decimales, puntos en códigos, identificadores) por año.](../figuras/F03_01.png)

*Figura F03_01. Deriva de formato por año. Datos de origen: `tablas/t21_deriva_formato.csv`.*

## 4. Bitácora de problemas de carga

La bitácora lista los problemas de carga por año con su severidad y la acción aplicada. Los conteos de líneas del archivo no se consideran casos de pacientes; los conteos de valores sí se suprimen si tienen menos de diez casos.

| Año | Problema | Severidad | N | Detalle | Acción |
|---|---|---|---|---|---|
| 2019 | Texto (centinela) en columna de formato definido | media | 310 | CIP_ENCRIPTADO | Contar como faltante disfrazado |
| 2020 | Peso GRD con coma y sin cero inicial | baja | 35.558 | IR_29301_PESO | Convertir coma a punto y aceptar ,xxxx |
| 2020 | Texto (centinela) en columna de formato definido | media | 143 | CIP_ENCRIPTADO | Contar como faltante disfrazado |
| 2021 | BOM UTF-8 al inicio del archivo | baja | 1 | El primer nombre de columna llega con BOM si no se decodifica como utf-8-sig | El lector lo retira |
| 2021 | Peso GRD con coma y sin cero inicial | baja | 33.632 | IR_29301_PESO | Convertir coma a punto y aceptar ,xxxx |
| 2022 | BOM UTF-8 al inicio del archivo | baja | 1 | El primer nombre de columna llega con BOM si no se decodifica como utf-8-sig | El lector lo retira |
| 2022 | Texto (centinela) en columna de formato definido | media | 150 | CIP_ENCRIPTADO | Contar como faltante disfrazado |
| 2023 | BOM UTF-8 al inicio del archivo | baja | 1 | El primer nombre de columna llega con BOM si no se decodifica como utf-8-sig | El lector lo retira |
| 2023 | Fechas en formato DD-MM-AAAA | media | 49.999 | Columna FECHAALTA | El lector de fechas acepta ISO y DD-MM-AAAA |
| 2023 | Fechas en formato DD-MM-AAAA | media | 24.559 | Columna FECHAINTERV1 | El lector de fechas acepta ISO y DD-MM-AAAA |
| 2023 | Fechas en formato DD-MM-AAAA | media | 7.844 | Columna FECHATRASLADO1 | El lector de fechas acepta ISO y DD-MM-AAAA |
| 2023 | Fechas en formato DD-MM-AAAA | media | 49.999 | Columna FECHA_INGRESO | El lector de fechas acepta ISO y DD-MM-AAAA |
| 2023 | Texto (centinela) en columna de formato definido | media | 78 | CIP_ENCRIPTADO | Contar como faltante disfrazado |
| 2024 | Nombre alternativo de la segunda columna (identificador) | media | 1 | El encabezado dice ID_BENEFICIARIO en lugar de CIP_ENCRIPTADO | Alias por posición |
| 2024 | Texto (centinela) en columna de formato definido | media | 52 | CIP_ENCRIPTADO | Contar como faltante disfrazado |
| 2024 | Texto (centinela) en columna de formato definido | media | 907 | FECHAINTERV1 | Contar como faltante disfrazado |

*Problemas de carga detectados.* *Tabla completa en `tablas/t21_problemas_carga.csv`*

## 5. Verificaciones contra lo documentado

Cada fila contrasta un hecho de la carga con lo que decían los documentos de la Fase 2 o con lo observado en los extractos. Una discrepancia es un hallazgo, no un error de la suite.

| Id | Descripción | Esperado | Observado | Resultado |
|---|---|---|---|---|
| V01 | Todos los archivos tienen el mismo número de columnas | 129 | 129 | COINCIDE |
| V02 | Las filas leídas más las rechazadas suman las líneas de datos | si | si | COINCIDE |
| V03 | Filas por año en el extracto | 49999 | 49999 | COINCIDE |
| V04 | Años en que la segunda columna se llama ID_BENEFICIARIO | 2019 | 2024 | DISCREPA |
| V05 | Años con FECHA_INGRESO en DD-MM-AAAA | 2023 | 2023 | COINCIDE |
| V06 | FECHA_NACIMIENTO viene en formato ISO en todos los años | si | si | COINCIDE |
| V07 | Años con BOM | 2021;2022;2023 | 2021;2022;2023 | COINCIDE |
| V08 | Cobertura de años | 2019;2020;2021;2022;2023;2024 | 2019;2020;2021;2022;2023;2024 | COINCIDE |

*Verificaciones de la carga.* *Tabla completa en `tablas/t21_verificaciones.csv`*

## 6. Procedencia de los datos y de los documentos

| Elemento | Estado | Evidencia | Acción |
|---|---|---|---|
| Oficio FONASA 23364/2026 (respuesta a la solicitud AO004T0010901) | PRESENTE | archivo con SHA-256 4456ae03697fc7a4; entrega el catálogo de GRD (código y glosa) y explica la estructura del código IR-GRD | Citar como fuente del catálogo GRD; no es respuesta por los microdatos |
| Respuesta de FONASA por los microdatos de egresos 2019 a 2024 | AUSENTE | no hay en la carpeta un documento de respuesta que acompañe los archivos de egresos | Incorporar la respuesta (con su número de oficio y fecha) al expediente |
| Condiciones de uso y de publicación de resultados por hospital | NO DOCUMENTADAS | el oficio 23364 es un catálogo técnico, sin registros de pacientes, y no se pronuncia sobre publicación por hospital | Consultar por escrito a FONASA (Fase 1, sección 14.4); decisión 10 sigue abierta |
| Diccionario oficial de todas las versiones del agrupador | NO EXISTE SEGÚN EL OFICIO | el oficio advierte que identificar el CDM y el grupo base por la estructura del código no equivale a un diccionario oficial de las versiones históricas | Usar el catálogo GRDs_T10901.xlsx solo como referencia de códigos vigentes |
| Archivo de egresos 2019 | PRESENTE | GRD_PUBLICO_2019.txt; SHA-256 56d6b5fde2c447d1; 49999 líneas de datos | Conservar sin modificar (regla R4) |
| Archivo de egresos 2020 | PRESENTE | GRD_PUBLICO_2020.txt; SHA-256 d58b595cd0d3457e; 49999 líneas de datos | Conservar sin modificar (regla R4) |
| Archivo de egresos 2021 | PRESENTE | GRD_PUBLICO_2021.txt; SHA-256 a7737925ac670ce8; 49999 líneas de datos | Conservar sin modificar (regla R4) |
| Archivo de egresos 2022 | PRESENTE | GRD_PUBLICO_EXTERNO_2022.txt; SHA-256 cb9818fec7e1f03e; 49999 líneas de datos | Conservar sin modificar (regla R4) |
| Archivo de egresos 2023 | PRESENTE | GRD_PUBLICO_2023.txt; SHA-256 f35ad4589856552e; 49999 líneas de datos | Conservar sin modificar (regla R4) |
| Archivo de egresos 2024 | PRESENTE | GRD_PUBLICO_2024.txt; SHA-256 c25831d80986a452; 49999 líneas de datos | Conservar sin modificar (regla R4) |

*Hoja de procedencia.* *Tabla completa en `tablas/t21_procedencia.csv`*

## 7. Límites

- Los hashes SHA-256 completos están en `tablas/t21_archivos.csv` y en el manifiesto de la corrida.
- Falta la respuesta de FONASA por los microdatos y la confirmación de las condiciones de publicación por hospital.
- Los datos de 2024 se usan solo para estructura y calidad; las columnas de resultado de 2024 llegan enmascaradas.
