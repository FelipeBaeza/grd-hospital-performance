# Fase 2 — Comprensión de los Datos
### Proyecto de Tesis: *Sistema de evaluación del desempeño hospitalario ajustado por riesgo mediante aprendizaje automático*
**Estudiante:** Felipe Ignacio Baeza Muñoz | **Carrera:** Ingeniería de Ejecución en Computación e Informática — USACH

---

> Esta fase documenta todo lo que aprendí al explorar los datos en crudo antes de programar cualquier transformación. Su objetivo es registrar qué hay en los datos, qué problemas presentan, y qué decisiones tomé al respecto. Está escrita para que cualquier persona con conocimiento general de computación pueda entenderla sin haber visto el proyecto antes.

---

## 1. ¿Cómo obtuve los datos y qué encontré al abrirlos?

Los datos provienen de una solicitud formal a FONASA mediante la **Ley 20.285 de Transparencia**. Son registros de egresos hospitalarios de la red pública chilena para los años 2019 a 2024, entregados como seis archivos de texto separados por el carácter `|`, uno por año.

Al abrir los archivos, lo primero que descubrí es que **no existe un identificador único por registro**: FONASA no asigna ningún ID a cada egreso. Esto es un problema importante porque sin un identificador propio, es imposible rastrear a un paciente entre distintas hospitalizaciones o cruzar de forma segura la tabla principal con tablas auxiliares. El primer trabajo fue diseñar una solución para esto (ver Sección 4).

Lo segundo que encontré es que los archivos tienen **129 columnas** y aproximadamente 970.000 filas por año. A simple vista, muchas columnas parecen vacías o tienen datos en formatos inconsistentes. El trabajo de esta fase fue sistematizar qué hay exactamente en cada una de esas 129 columnas.

### Estructura general de las 129 columnas

Los datos se pueden clasificar en seis grandes bloques:

| Bloque | Columnas | Qué contiene |
|--------|---------|--------------|
| **Identificación** | 3 | Código del hospital, identificador encriptado del paciente, servicio de salud |
| **Datos demográficos** | 5 | Sexo, fecha de nacimiento, etnia, provincia, comuna, nacionalidad, previsión |
| **Contexto del ingreso** | 6 | Fecha de ingreso, tipo de ingreso, tipo de procedencia, especialidad médica, servicio de ingreso, hospital de procedencia |
| **Diagnósticos** | 35 | Un diagnóstico principal (DIAGNOSTICO1) y hasta 34 diagnósticos secundarios (DIAGNOSTICO2–DIAGNOSTICO35) |
| **Procedimientos y evolución** | 39 | Hasta 30 códigos de procedimientos, 3 pares de traslados internos, médico interventor, fechas de intervención, uso de pabellón |
| **Egreso y resultado** | 8 | Fecha de alta, tipo de alta, servicio de alta, datos de recién nacidos (4 familias de columnas) |
| **Clasificación GRD** | 4 | Código GRD asignado, peso del GRD, severidad, riesgo de mortalidad del agrupador |
| **Traslados externos** | 18 | 9 pares de fecha+servicio de traslados internos durante la hospitalización |

---

## 2. El problema más importante: ¿qué información no puedo usar en el modelo?

Antes de analizar los datos, necesitaba entender un concepto fundamental: **fuga de datos (data leakage)**. Si el modelo aprende con información que solo existe porque el paciente ya tuvo cierto resultado, entonces el modelo hace trampa — no está prediciendo, está describiendo lo que ya pasó.

Esto me llevó a clasificar cada columna según si puede o no entrar al modelo de Machine Learning:

### Las 7 categorías de rol de las columnas

| Categoría | Significado | ¿Entra al modelo? | Ejemplo |
|-----------|-------------|:-----------------:|---------|
| `FEATURE_BASAL` | Información conocida al ingreso | ✅ **SÍ** | Edad, sexo, diagnóstico de ingreso |
| `TARGET` | Lo que el modelo debe predecir | ❌ No como predictor | Mortalidad, días de estadía |
| `PROHIBIDA_FUGA` | Información que ocurre después del ingreso | ❌ **NUNCA** | Procedimientos quirúrgicos, fechas de traslado interno |
| `FUENTE_DERIVADA` | Variable cruda que sirve para calcular otra | ❌ Solo como insumo | Fecha de nacimiento → calcula edad |
| `LLAVE_AGREGACION` | Identifica hospital, año o región | ❌ Solo para agrupar resultados | Código del hospital |
| `FILTRO_COHORTE` | Indica si el paciente aplica o no al estudio | ❌ Solo para filtrar | GRD no agrupable, recién nacido sano |
| `AUDITORIA_SENSIBILIDAD` | Variables sensibles reservadas para análisis de equidad | ❌ Fuera del modelo principal | Etnia, comuna, nivel socioeconómico |
| `DESCARTAR` | Sin propósito para el proyecto | ❌ Ignorar | Columnas vacías en todos los años |

### Los bloques de columnas más importantes para el modelo

**Las que SÍ entran al modelo (FEATURE_BASAL):**
- `SEXO` (0 nulos en 5.8M registros)
- `TIPO_INGRESO` — si el ingreso fue urgencia o programado (predictor documentado de mortalidad)
- `TIPO_PROCEDENCIA` → deriva en `PROCEDENCIA_AGR` (urgencia vs. derivado vs. programado)
- `DIAGNOSTICO1` → deriva en `GRUPO_CLINICO` y `CIE10_3C` (el diagnóstico de ingreso en dos niveles de detalle)
- `DIAGNOSTICO2` a `DIAGNOSTICO35` → sirven para calcular los 31 indicadores Elixhauser

**Las que NUNCA pueden entrar (PROHIBIDA_FUGA):**
- `PROCEDIMIENTO1` a `PROCEDIMIENTO30` — las cirugías y procedimientos que el paciente tuvo (ocurren después del ingreso)
- `FECHATRASLADO1` a `FECHATRASLADO9` — cuántas veces fue trasladado dentro del hospital
- `FECHAINTERV1` — cuándo lo intervinieron quirúrgicamente
- `USOSPABELLON` — si usó pabellón (quirófano)
- `IR_29301_MORTALIDAD` — el riesgo de mortalidad que ya calculó el agrupador GRD (copiar esto sería hacer trampa completa)

> **¿Por qué los procedimientos son tan problemáticos?** Porque si el modelo sabe que un paciente tuvo 15 procedimientos distintos, ya sabe que estuvo hospitalizado mucho tiempo y que era un caso grave. No está prediciendo el riesgo; está describiendo el desenlace. Esto explica por qué hay 30 columnas de procedimientos que simplemente no puedo tocar.

---

## 3. Los problemas de calidad que encontré en los datos

Los datos administrativos hospitalarios nunca son perfectos. Al explorar las 129 columnas en detalle, identifiqué los siguientes problemas:

### 3.1 Problemas de formato y codificación

| Problema | Dónde ocurre | Qué hago |
|---------|-------------|----------|
| **Dos formatos de fecha distintos** | `FECHA_INGRESO` usa `dd-mm-yyyy` pero `FECHA_NACIMIENTO` usa `yyyy-mm-dd` | Convertir ambas al mismo formato durante la limpieza |
| **Decimales con coma** | `IR_29301_PESO` tiene valores como `"0,7094"` en vez de `"0.7094"` | Reemplazar coma por punto antes de convertir |
| **Procedimientos leídos como decimales** | `PROCEDIMIENTO1` tiene `"0.17"` cuando debería ser `"00.17"` (código CIE-9-MC) | Leerlos siempre como texto, nunca como número |
| **Deriva de tipo entre años** | `PROCEDIMIENTO2` se infiere como texto en algunos años y como entero en otros | Forzar lectura como texto en todos los años |

### 3.2 Valores imposibles o biológicamente incoherentes

| Problema | Ejemplo | Qué hago |
|---------|---------|----------|
| **Edad calculada negativa** | `FECHA_NACIMIENTO` posterior a `FECHA_INGRESO` — error de digitación | Anular solo ese campo (el resto del registro se conserva) |
| **Edad > 110 años** | Error de tipeo en la fecha de nacimiento | Anular solo ese campo |
| **Fecha de alta anterior al ingreso** | `FECHAALTA < FECHA_INGRESO` — 18 casos en 5.8M | Marcar para revisión, excluir del cálculo de estadía |
| **Estadía de 0 días** | Ingreso y alta el mismo día | Excluir del modelo de estadía, conservar para mortalidad |

> **Regla clave de calidad:** Cuando encuentro un valor imposible, **no elimino al paciente completo**. Solo anulo ese valor específico y conservo el resto de la información clínica. Una fecha mal digitada no invalida los 34 diagnósticos del paciente.

### 3.3 Inconsistencias en categorías de texto

**El caso de ETNIA:** La categoría `"OTRO"` aparece con dos grafías diferentes en los datos originales, generando dos conteos distintos: **1.929.129** y **409.537** registros que deberían ser la misma categoría. Si no corrijo esto antes de derivar la variable `PUEBLO_ORIGINARIO`, el modelo aprendería una distinción que no existe clínicamente.

**Solución:** Antes de cualquier análisis, aplicar normalización de texto: `strip()` (eliminar espacios), `upper()` (todo mayúsculas), y eliminación de tildes. Con esto, las dos grafías colapsan en una sola.

**El caso de los diagnósticos CIE-10:** Los códigos de diagnóstico deben estar en mayúsculas y sin espacios. Encontré casos como `"j18.0"` (minúscula), `"I48 "` (espacio al final) y `"Ó30"` (tilde). La normalización los unifica en `"J180"`, `"I48"` y `"O30"` respectivamente.

### 3.4 Columnas vacías o sin propósito

Varias columnas de la familia de recién nacidos están completamente vacías en los 6 años de datos:
- `CONDICIONDEALTANEONATO3`: 0 valores no nulos (columna completamente vacía)
- `CONDICIONDEALTANEONATO4`: 0 valores no nulos (columna completamente vacía)

Otras tienen valores que no corresponden a sus familias (ej: `RN2ESTADO` tiene 1.444.072 valores no nulos cuando `CONDICIONDEALTANEONATO2` solo tiene 1.576), lo que indica una desalineación de columnas entre años. Todas estas columnas se marcan como `DESCARTAR`.

### 3.5 El problema de privacidad en `FECHAPROCEDIMIENTO1`

Al explorar los datos, descubrí que la columna `FECHAPROCEDIMIENTO1` tiene solo **9 valores no nulos** en toda la base, y esos 9 valores tienen la estructura de un RUT chileno, no de una fecha. Esto es probablemente un error de carga del sistema hospitalario.

La política es: detectar el problema, aislar esos 9 registros en cuarentena, **pero jamás escribir el valor sospechoso en ningún log o reporte**. Solo se registra que en esa columna hay 9 valores con estructura de identificador personal.

---

## 4. El identificador único que tuve que crear

Como los datos de FONASA no traen un ID propio por registro, diseñé una función matemática que genera un identificador único llamado `ID_EPISODIO`. Lo construí combinando:

```
ID_EPISODIO = [DOMINIO] + [AÑO_ARCHIVO] + [NUMERO_LINEA] + [HASH_CONTENIDO_LINEA]
```

Donde:
- `DOMINIO`: un prefijo fijo para evitar colisiones si el sistema se integra con otros proyectos.
- `AÑO_ARCHIVO`: el año del archivo de origen (2019, 2020... 2024).
- `NUMERO_LINEA`: la posición exacta de la fila en el archivo de texto original.
- `HASH_CONTENIDO_LINEA`: una huella digital del contenido de la fila, calculada con el algoritmo `blake2b`.

**¿Por qué esta combinación?** Porque garantiza que el ID sea:
- **Único**: dos filas idénticas en archivos distintos tendrán diferente `AÑO_ARCHIVO` y `NUMERO_LINEA`.
- **Determinista**: si proceso el mismo archivo dos veces, cada fila obtiene exactamente el mismo ID.
- **Resistente al orden**: no depende de funciones internas de Python que varíen entre ejecuciones.

**Restricción importante:** Este `ID_EPISODIO` sirve para cruzar tablas internamente, pero **el modelo de ML nunca lo ve**. Si el modelo aprendiera a partir del ID, simplemente memorizaría los registros en lugar de aprender patrones clínicos.

---

## 5. El problema de los traslados y la censura estadística

Este es uno de los hallazgos más importantes de la exploración de datos.

Cuando un paciente es **trasladado de un hospital a otro**, aparece en la base de datos como **dos registros separados**: uno en el hospital que lo envía (con `TIPOALTA = "TRASLADO EXTERNO"`) y otro en el hospital que lo recibe (con `TIPO_PROCEDENCIA = "DERIVADO_OTRO_HOSPITAL"`).

Esto crea un problema doble:

**Problema 1 — Para el modelo de mortalidad:** El paciente trasladado no tiene un desenlace registrado en el hospital emisor. No sabemos si después falleció en el hospital receptor o se recuperó. Si lo clasificamos como "sobreviviente" porque no figura como fallecido en nuestra base, estaríamos cometiendo un error. La solución es marcarlo como **"censurado"** — un término estadístico que significa "desenlace desconocido en este registro".

En los datos hay **89.438 episodios censurados** (traslados + hospitalizaciones domiciliarias), lo que representa aproximadamente el 1.5% de los registros.

**Problema 2 — Para el modelo de estadía:** Si el hospital emisor registra 2 días de estadía antes del traslado, y el hospital receptor registra 8 días más, los **días reales del episodio son 10**, no 2 ni 8 por separado. Para calcular correctamente la estadía, necesito encadenar los episodios del mismo paciente usando el identificador encriptado `CIP_ENCRIPTADO`.

**El sesgo de alta anticipada:** Existe un fenómeno documentado en sistemas de salud donde algunos hospitales trasladan pacientes muy graves a otros centros justo antes de que fallezcan, para que la muerte no quede registrada en su institución. Esta práctica distorsiona cualquier indicador de mortalidad. Mi sistema lo reconoce como una **limitación declarada**, no como algo que pueda corregirse computacionalmente con los datos disponibles.

---

## 6. Los diagnósticos en dos niveles de detalle

El diagnóstico principal (`DIAGNOSTICO1`) tiene **9.575 códigos CIE-10 distintos** en los datos. Usar ese código tal cual en el modelo crearía un problema de sobreajuste severo: muchos códigos tienen tan pocos pacientes que el modelo no puede aprender nada confiable de ellos.

La solución fue representar el diagnóstico en **dos niveles de abstracción**:

| Variable | Cómo se construye | Cardinalidad | Para qué sirve |
|---------|-------------------|:------------:|----------------|
| `GRUPO_CLINICO` | Primera letra del código CIE-10 | ~22 categorías | Eje de ajuste estable, comparable con estándares internacionales |
| `CIE10_3C` | Primeros 3 caracteres del código CIE-10 | ~1.400 categorías | Captura señal clínica específica sin sobreajustar |

**Ejemplo:** Un paciente con diagnóstico `"J189"` (neumonía no especificada):
- `GRUPO_CLINICO` → `"J"` (enfermedades del sistema respiratorio)
- `CIE10_3C` → `"J18"` (neumonía)

El modelo puede aprender tanto que "los pacientes con enfermedades respiratorias tienen mayor riesgo" (`GRUPO_CLINICO`) como que "los pacientes con neumonía específicamente tienen un riesgo diferente al de otras enfermedades respiratorias" (`CIE10_3C`).

---

## 7. Las variables de historial del paciente

Una de las features más valiosas que puedo construir es el **historial previo de hospitalizaciones** del paciente. Un paciente que fue hospitalizado 5 veces en el último año claramente tiene una carga de enfermedad crónica mayor que uno que no tiene ingresos previos.

Para construir esto, uso el identificador encriptado `CIP_ENCRIPTADO` para conectar todos los episodios de un mismo paciente y calcular:

- **`N_EGRESOS_12M`**: cuántas veces fue hospitalizado en los 12 meses anteriores al ingreso actual.
- **`DIAS_DESDE_EGRESO_PREVIO`**: cuántos días pasaron desde su último egreso (un reingreso en menos de 30 días es señal de inestabilidad clínica).
- **`HISTORIA_DISPONIBLE`**: un indicador binario que marca si tengo 12 meses completos de historia o no.

**¿Por qué ese tercer indicador es importante?** Porque la base de datos comienza en 2019. Un paciente que ingresa en febrero de 2019 aparecerá con "0 egresos previos" simplemente porque no tengo datos de 2018, no porque sea un paciente sin historial. Sin este indicador, el modelo confundiría "sin datos registrados" con "paciente sano".

---

## 8. Las variables que quedan fuera del modelo principal (Auditoría de equidad)

Hay un conjunto de variables que son potencialmente valiosas para entender el contexto social del paciente, pero que **no pueden entrar al modelo predictivo principal** por una razón metodológica específica: en Chile, el territorio (comuna) y el hospital están tan relacionados geográficamente que si le doy la comuna al modelo, este termina "adivinando" en qué hospital está el paciente. Eso volvería el ajuste circular.

Estas variables se reservan para un **análisis de equidad posterior**, donde verificaré si el sistema genera resultados distintos según el nivel socioeconómico o el origen étnico de los pacientes:

| Variable derivada | Fuente | Uso |
|------------------|--------|-----|
| `PUEBLO_ORIGINARIO` | `ETNIA` (normalizada) | ¿Existen diferencias en el O/E según pertenencia a pueblo originario? |
| `ES_EXTRANJERO` | `NACIONALIDAD` | ¿Existen diferencias para pacientes extranjeros? |
| `TRAMO_FONASA` | `PREVISION` (18 categorías agrupadas en A/B/C/D) | ¿Existen diferencias según nivel de ingresos? |
| `INDICE_PRIVACION` | `COMUNA` cruzada con tabla oficial de vulnerabilidad | ¿Los hospitales en zonas más vulnerables tienen peor O/E? |
| `FLAG_PANDEMIA` | `ANIO_EGRESO` (1 si es 2020 o 2021, 0 si no) | ¿Cómo afectó el COVID-19 a los indicadores? |

---

## 9. Variables condicionales — lo que requiere verificación empírica

Hay tres variables que podrían ser features útiles del modelo, pero que requieren verificación antes de incluirlas. Las marco como **"condicionales"** porque su validez depende de si miden el riesgo del paciente al ingreso o el contexto del hospital:

**`SERVICIOINGRESO` → `NIVEL_CUIDADO_INGRESO`:** Indica si el paciente ingresó directamente a UCI (Unidad de Cuidados Intensivos). El problema es que ingresa a UCI puede reflejar tanto la gravedad del paciente como el simple hecho de que ese hospital tiene más camas de UCI disponibles. Si lo segundo domina, la variable mide capacidad instalada, no riesgo clínico, y no puede entrar al modelo.

**`ESPECIALIDAD_MEDICA` → `ESPECIALIDAD_MACRO`:** Con 158 especialidades distintas y asignaciones que dependen de cómo cada hospital organiza sus departamentos internamente, existe un riesgo alto de que el modelo aprenda a identificar hospitales por su estructura administrativa en vez de aprender riesgo clínico del paciente.

**`DERIVADO_OTRO_HOSPITAL`:** Un paciente que llega derivado de otro hospital suele ser más complejo que uno que llega directamente. Pero como la derivación también depende de acuerdos entre establecimientos y no solo de la gravedad, requiere verificación estadística.

---

## 10. Resumen: el mapa completo de las 129 columnas

| Rol | Cantidad | Qué pasa con ellas |
|-----|:--------:|-------------------|
| `FEATURE_BASAL` (entra al modelo) | ~12 | Se transforman y pasan a la capa oro |
| `FUENTE_DERIVADA` (genera otras variables) | ~39 | Se procesan para crear derivadas, luego se descartan |
| `PROHIBIDA_FUGA` (información post-ingreso) | ~54 | Bloqueadas del modelo. Algunos slots de traslado se usan solo para auditoría de censura |
| `TARGET` (lo que predigo) | 2 | Mortalidad binaria y días de estadía |
| `LLAVE_AGREGACION` (identificadores) | 3 | Solo para agrupar resultados finales |
| `FILTRO_COHORTE` (criterios de exclusión) | 5 | Aplican los filtros, luego se descartan |
| `AUDITORIA_SENSIBILIDAD` (equidad) | 7 | Reservadas para análisis de equidad |
| `DESCARTAR` (sin propósito) | ~7 | Ignoradas desde el inicio |
| **Total** | **~129** | |

---
*Documento de Fase 2 — Comprensión de los Datos*
*Universidad de Santiago de Chile | Departamento de Ingeniería Informática*
*Fecha: Septiembre 2026*
