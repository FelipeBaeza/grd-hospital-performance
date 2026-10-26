# Informe de cierre de la Fase 2 (CRISP-DM 2.4)

**Fuente de los datos: completo.** Todas las cifras provienen de los archivos completos.

Este informe cierra la Fase 2: resume el resultado de los criterios de terminado, los cambios que se hicieron al protocolo después de ver resultados, las limitaciones y lo que falta con los datos completos.

## 1. Resultado de la verificación

La definición de terminado de la Fase 2 tiene 12 criterios. En esta corrida 12 pasan, 0 fallan y 0 quedan pendientes. Un criterio pendiente no es un fallo: depende de una opción de la corrida (las pruebas automáticas y la corrida repetida que comprueba el determinismo) o de que se corra con los datos completos.

Las Tareas 3 a 12 produjeron 97 tablas y 33 figuras, reunidas en 7 informes. Cada cifra de los informes sale de una tabla y cada figura lleva su leyenda y su tabla de origen.

| Criterio | Descripción | Estado | Detalle |
|---|---|---|---|
| DT01 | Las 17 decisiones pendientes están cubiertas por una hipótesis o marcadas como no dependientes de datos | PASS | 17 de 17 decisiones cubiertas; el protocolo trae 26 hipótesis y 15 supuestos |
| DT02 | Cada cifra de los informes proviene de un CSV | PASS | 508 cifras de 6 informes con su CSV de origen; 444 cifras escalares se reproducen desde su columna y 6 tablas derivadas tienen columnas calculadas desde su CSV; sin llaves sobrantes |
| DT03 | Cero lecturas de resultado de 2024 | PASS | 180 accesos registrados, 30 del año 2024: ninguna lectura de resultado y cero violaciones; bloqueos 1, enmascaradas 4 y reglas de calidad autorizadas 4 |
| DT04 | Cero identificadores, fechas completas o celdas con 1 a 9 casos en las salidas | PASS | 108 archivos escaneados (tablas e informes) sin identificadores, fechas completas ni celdas de 1 a 9 casos |
| DT05 | La cascada de exclusiones cuadra con el total en cada año | PASS | 36 identidades secuenciales, 6 totales anuales contra las filas leídas, 23 sumas de períodos agregados y 6 identidades de la cascada del pipeline cuadran |
| DT06 | Las 129 filas del diccionario están conciliadas | PASS | 129 de 129 variables del diccionario conciliadas: 0 coinciden, 125 discrepan y 4 no se evalúan |
| DT07 | El ledger de cifras está completo | PASS | 129 de 129 cifras documentadas con estado: 46 confirman, 33 discrepan y 50 no se pueden verificar, cada una con su motivo o su cifra medida |
| DT08 | Todas las figuras están indexadas con su CSV de respaldo | PASS | 33 de 33 figuras del plan indexadas con su PNG y su CSV de respaldo (huellas iguales) e incrustadas en un informe con su leyenda y su CSV de origen |
| DT09 | La matriz de decisión cubre todas las variables candidatas | PASS | 199 variables en la matriz, igual al universo documentado y a la tabla de disponibilidad; las 59 variables del ranking están en la matriz |
| DT10 | Las pruebas automáticas pasan | PASS | 1.834 pruebas pasaron |
| DT11 | La corrida es determinista | PASS | dos corridas completas dieron los mismos 145 archivos con los mismos SHA-256 |
| DT12 | Corrida con datos completos | PASS | 97 tablas y 1.128 métricas con la etiqueta completo; se leyeron los 6 años |

*Resultado de los criterios de terminado.* *Tabla completa en `tablas/verificacion_dt.csv`*

## 2. Cambios al protocolo

Después de ver resultados se hicieron 6 cambios al protocolo congelado en la Tarea 1, que quedó en la versión v1.6. La tabla dice qué cambió y por qué.

| Cambio | Descripción |
|---|---|
| 1.1 | se agrega conciliacion_dif_relevante_pp tras la primera corrida sobre extractos. No cambia el estado COINCIDE o DISCREPANCIA (regla por z); solo agrega la columna de magnitud. |
| 1.2 | se agrega id_largo_minimo (regla Q11) antes de ejecutar la Tarea 5. El valor 5 se apoyó en una inspección previa de los patrones de largo de los identificadores, por lo que no es independiente de los datos. |
| 1.3 | se agrega edad_neonato_max (regla Q26) antes de ejecutar la Tarea 6, sin haber visto resultados de esa regla. El valor 1 año es un criterio clínico de tolerancia sobre el límite de 28 días del neonato. |
| 1.4 | la comparación mensual de H17b se define contra el mismo mes del año anterior, que controla la estacionalidad y coincide con el cambio anual que describe cambio_relativo_relevante. La primera versión comparaba cada mes con los tres meses previos y se descartó después de ver que marcaba cerca de un tercio de los meses por el ciclo invernal. No cambia ningún valor de umbral. |
| 1.5 | el criterio de S09 se amplía con el contraste de los egresos con traslado interno, porque la coincidencia entre SERVICIOINGRESO y SERVICIOALTA sin traslado interno es esperable por construcción y no discrimina entre el ingreso y la estadía. Se advirtió después de ver, en los extractos, 99,2 % de coincidencia sin traslado y 15,2 % con traslado. La regla pasa a ser: cada grupo con al menos celda_minima egresos; se acepta si coincide en la mayoría sin traslado y en la minoría con traslado, se rechaza si coincide en la mayoría también con traslado, y no es evaluable si algún grupo es menor al mínimo o si no coincide en la mayoría sin traslado. Aceptarlo no levanta las sospechas de proxy ni de fuga de la Tarea 11 sobre SERVICIOINGRESO. No cambia ningún valor de umbral. |
| 1.6 | cambio posterior, de octubre de 2026, hecho después de abrir 2024 y de la revisión técnica del profesor guía sobre la Fase 1. Se alinean cuatro hipótesis con la Fase 1 v6. H11 ya no propone informar a FONASA: los valores con forma de RUT solo se cuentan y su columna no se usa (la regla Q17 pasa de informar_fonasa a excluir, y las referencias de Q07 y Q17 dejan de citar conteos de menos de diez casos). H14 confirma que la severidad del GRD queda prohibida, también en sensibilidad (R6), en vez de proponer una sensibilidad aparte. H13 matiza si el identificador tiene un quiebre (H01a), como ya pedía su criterio («un espacio de IDs consistente»). H17a deja de contar el quiebre del identificador cuando el historial se descarta (H01c contradice), porque entonces no afecta a ninguna variable. Se agregan los análisis posteriores R1 a R5 (eda/revision_profesor.py), que piden las observaciones O2, O3, O7 y O8 y la decisión 6; son descriptivos y no cambian ninguna decisión ejecutada. Además se agrega la supresión complementaria (eda/proteccion.py y eda/proteccion_tablas.py): la revisión encontró conteos de 1 a 9 que se despejaban restando desde lo publicado (por ejemplo, el total de valores con forma de RUT menos el de una de sus dos columnas), y ahora ninguna suma conocida queda con una sola celda oculta: las sumas de cada tabla, la resta de los períodos 2019-2023 y 2019-2022, la resta de dos variantes vecinas de la cohorte y el total de fallecidos de los estratos cuando, restado de la cohorte D, deja la actividad ambulatoria con menos de diez; las tablas que publican las mismas celdas ocultan las mismas, y las métricas del ledger se leen de las tablas ya protegidas. El diccionario de la Fase 2 deja de publicar sus conteos de menos de diez casos y esas cuatro columnas quedan NO EVALUABLE en la conciliación. El registro de cifras documentadas (config/cifras_documentadas.csv) tampoco copia los cinco conteos de menos de diez casos de los documentos: dice <10 y esas cifras siguen NO VERIFICABLE. En los estratos de la decisión 15, la estadía sin dato se une a «1 día o más»: como categoría propia, con menos de diez egresos, obligaba a ocultar el estrato de 0 días que evalúa H15b. La tabla por edad y sexo deja de publicar los sobrevivientes con estadía válida (junto con los egresos y los fallecidos daban cuántos sobrevivientes no tenían estadía) y solo declara el total de cada período, que es lo que el lector conoce. Una auditoría posterior agregó tres identidades y una regla: la suma de cada variante de la cohorte (n = fallecidos + no evento + censurados + sin dato), la cadena de la cascada (TOTAL - EX01 - EX02 - EX03 = A, etc., que en los extractos dejaba despejar EX01), el resto implícito de los enlaces de traslado (traslados ni inválidos ni elegibles) y, en toda identidad, que dos o más celdas pequeñas ocultas no sumen de 1 a 9, porque esa suma también se deduce. No cambia ningún valor de umbral. |

*Cambios al protocolo después de ver resultados.* *Tabla completa en `tablas/cambios_protocolo.csv`*

## 3. Limitaciones

- El desenlace de 2024 nunca se leyó. Las cifras que lo requieren se verifican solo con 2019 a 2023 o con las cotas de la Tarea 7; el resto queda no verificable.
- Los intervalos de confianza ignoran el agrupamiento por paciente y por hospital.
- El ledger compara cada cifra con una sola medición: si el documento y la suite usan definiciones distintas, la cifra queda no verificable y no se declara una discrepancia.
- El mapeo de tres caracteres del pipeline es una reconstrucción aproximada: el código original no está en el repositorio revisado.
- El AUC del modelo L1 no es un desempeño esperable: incluye variables cercanas a proxy o fuga y la Tarea 11 no midió un L1 sin ellas.
- El conocimiento clínico que no proviene de las cuatro referencias verificadas en fuente queda pendiente de revisión clínica (decisión 12).
- La evidencia recomienda y no decide: toda decisión sigue pendiente de decisión humana.

## 4. Qué falta con los datos completos

3 de las 26 hipótesis quedan no evaluables, 50 de las 129 cifras documentadas quedan no verificables y 2 de los 15 supuestos no se pueden evaluar. Entre las hipótesis, unas se repiten con los datos completos antes de decidir (la corrida usa `run_fase2_eda.py` con la fuente completa y la carpeta de datos) y otras no dependen de datos: son decisiones metodológicas o institucionales que ninguna corrida resuelve.

| Hipótesis | Decisión | Clase | Motivo |
|---|---|---|---|
| H04 | 4 | informativa | no depende de datos |
| H10 | 10 | informativa | no depende de datos |
| H12 | 12 | informativa | no depende de datos |

*Hipótesis que no dependen de datos.* *Tabla completa en `tablas/t25_hipotesis_evaluadas.csv`*
