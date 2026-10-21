# Tarea 12: ledger de cifras, evidencia por decisión y cierre (segunda pasada)

La Tarea 12 compara las 129 cifras documentadas de las Fases 1 y 2 con las métricas de la corrida, evalúa 26 hipótesis y 15 supuestos, deriva el semáforo de las 17 decisiones y sugiere 23 cambios a los documentos, sin leer datos. Esta pasada revisa las correcciones a los dos MEDIUM de la primera: S09 contrasta ahora los egresos sin y con traslado interno (cambio de protocolo 1.5) y la evidencia de la decisión 5 cita una sola base. La suite sigue verde (1.587 pruebas), la corrida nueva de T3 a T12 sobre los extractos dio privacidad 0 y cortafuegos sin violaciones, y un oráculo escrito aparte reprodujo los 129 estados del ledger, el semáforo y los supuestos que recalculé. No quedan HIGH ni MEDIUM; los 7 NIT son un orden de filas inestable en una tabla de la Tarea 7, una limitación del semáforo, formato del informe, un borde sin prueba, una redacción y dos arrastres declarados.

Watch for: (1) la regla nueva de S09 se fijó después de ver 99,2 % y 15,2 %; está declarada como cambio 1.5 y ningún umbral cambió, pero la autorización del orquestador solo consta en el registro del plan (confirmed). (2) `t23_resultado_estratos.csv` sale con otro orden de filas en cada proceso y haría fallar DT11 en la Tarea 13; es de la Tarea 7 y queda fuera de este cambio (confirmed).

**Verdict**: APPROVED

## High-level view

El ledger resuelve las 129 cifras con las reglas del encargo: 33 CONFIRMA, 16 DISCREPA (solo C076 es de magnitud relevante) y 80 NO VERIFICABLE, todas con motivo. El oráculo coincide en estado, magnitud y z en las 129 filas, y los cuatro hallazgos del encargo se reproducen desde las tablas de la corrida.

El semáforo de las 17 decisiones (3 verdes, 7 amarillas, 0 rojas, 7 grises) y los 15 supuestos (6 aceptados, 6 rechazados, 3 no evaluables) se reproducen desde las tablas fuente. S09 pasa de rechazado a aceptado al leer también el 15,2 % de coincidencia con traslado interno, y CF1-008 deja de pedir el descarte de SERVICIOINGRESO aclarando que la sospecha de proxy de la Tarea 11 sigue en pie. La decisión 5 cita 3,63 %, 4,31 % y 0,67 puntos sobre la misma base.

El cambio de protocolo 1.5 amplía el criterio de S09 después de ver resultados. Está declarado en `umbrales.toml`, `hipotesis.csv`, `cambios_protocolo.csv` y el informe de cierre, y ningún valor de umbral cambió.

T12 no lee datos, y sus nueve tablas y dos figuras salen byte a byte iguales en dos procesos distintos. Lo que no es estable entre procesos es una tabla de la Tarea 7.

Las 45 pruebas nuevas fijan el comportamiento corregido: de 21 mutantes sobre el código nuevo mueren 20, y el que sobrevive es un borde de la nota de redondeo. F12_01 y F12_02 no tienen defectos de lectura; la tabla de discrepancias del informe pierde decimales en tres cifras.

<details>
<summary>Issues (7)</summary>

1. **Orden de filas inestable en `t23_resultado_estratos.csv` (fuera de T12)**: `cohorte.py` línea 670 ordena por variable, período y n sin desempate y tres pares empatados cambian de posición entre procesos; agregar `categoria` como cuarta clave junto con B0 para que DT11 pase (confirmed).
2. **Decisión 1 gris sin la consecuencia de S01**: S01 está RECHAZADO (1,82 %) y su criterio manda descartar las variables de historial, pero la evidencia de la decisión 1 solo cita la cifra y el veredicto no la menciona; agregar una frase fija con la consecuencia (likely).
3. **Tabla de discrepancias del informe de evidencia**: "Medido" usa los decimales de lo documentado (C076 40 % en vez de 40,4; C128 y C129 4 y 68 en vez de 3,97 y 67,64), C018 muestra 0,2 % de diferencia relativa y un motivo de igualdad en una fila que discrepa; mostrar al menos dos decimales y no publicar la diferencia relativa de un año (confirmed).
4. **S11 sin años**: el 40,4 % es de 2019 a 2024 y el criterio dice 2019 a 2023 (33,6 %); nombrar los años en la cifra y declarar la elección (confirmed).
5. **Borde de la nota de redondeo sin prueba**: cambiar `z_tol > z_umbral` por `>=` en `_nota_redondeo` sobrevive a las 115 pruebas del ledger; agregar el caso `_nota_redondeo("DISCREPA", 6.0, 3.0, 2.0, 0, 3.0)` (confirmed).
6. **Salvedad de H11 solo genérica**: el cero del extracto deja la decisión 11 amarilla por decisión del orquestador, pero `que_falta_con_datos_completos` no dice que un cero en un extracto no descarta valores en los datos completos; agregar esa frase (confirmed).
7. **Prioridad alta de CF2-004**: S03 con 99,9993 % de códigos válidos sale en prioridad alta por la regla de los supuestos rechazados que afectan una decisión; declarar en el texto propuesto la diferencia de 0,0007 puntos o bajar la prioridad (confirmed).

</details>

<details>
<summary>Details</summary>

### S09 con traslado interno y base única en la decisión 5

Recalculé S09 desde `t23_fuga.csv`: 91.706 de 92.445 sin traslado (99,20 %, Wilson de 99,14 a 99,26) y 3.454 de 22.696 con traslado (15,22 %, de 14,76 a 15,69). Hay mayoría solo en el primer grupo, así que la regla da ACEPTADO. CF1-008 baja de prioridad alta a media y es coherente con la matriz de la Tarea 11, que mantiene SERVICIOINGRESO como CONDICIONAL por PROXY_MOD (asociación 0,256). El semáforo de la decisión 2 no cambia (amarillo) porque los supuestos aportan solo texto.

La decisión 5 lee `prevalencia_conocida`: A 3,63 % (7.615 de 209.496 con desenlace conocido) y B 4,31 % (7.497 de 173.954), con una diferencia de 0,6749 puntos que el texto publica como 0,67. La resta de los porcentajes ya redondeados da 0,68, y el texto lo declara ("calculada con los valores sin redondear"). CF1-009 conserva la base de todos los egresos (3,41 % y 4,02 %) y la rotula, de modo que los dos textos no se contradicen.

De los nueve NIT de la primera pasada, seis se ven resueltos en las salidas: la ubicación 9.1 de C078 (los 68 hospitales están en la sección 9.1 de la Fase 1), el par C030 y C100 como aclaración baja (CF2-013), las bases de S01 (8.975 IDs), S04 (0 de 10) y S15 (86 visibles y 13 suprimidas), la ventana de H14 (la confirmación de 2023), el `dif_abs` de C114 (2.404), la nota de tolerancia en C032, C057, C062, C070 y C121 (z de 2,41; 2,17; 2,97; 2,90 y 2,41) y la sección 3 del cierre separada. Las reglas de S03, S12 y S13 quedaron declaradas en el motivo con la prioridad sin cambio, el formato quedó parcial y H11 se mantuvo por decisión del orquestador; los restos están en la lista de issues.

### Ledger y hallazgos del encargo recalculados

El oráculo (sin importar `eda/`) coincide con `ledger_cifras.csv` en estado, magnitud y z, con diferencia máxima menor que 0,01, en las 129 filas. Cuentas a mano: C055 sin intervalo, n = 299.994, 47.881 casos contra 47.999 esperados, z = -0,588; C076 con error estándar 0,9134 tomado del intervalo de la métrica, z = -3,896 y diferencia de 3,56 puntos; C070 con z = 3,153 que baja a 2,90 con media unidad de tolerancia; C114 con intervalo de 19.899 a 22.569 (conteo de 2019 a 2023 más la cota de 2024, 2.670) que contiene al 22.303 documentado, con `dif_abs` 2.404.

Las razones de la Fase 2 sobre la Fase 1 son 67,64 (EX01), 18,93 (EX02), 19,36 (EX03) y 3,97 (censurados), y el inverso de la fracción de muestreo es 5.808.536 sobre 299.994, o sea 19,36. Solo EX02 y EX03 caen entre 10 y 20; la tabla 5.2 de la Fase 2 (leída con pdftotext) lista esas cuatro filas, de modo que C128 y C129 discrepan con razón. C018 es real: leí solo la primera línea de cada archivo y la columna 2 se llama CIP_ENCRIPTADO de 2019 a 2023 e ID_BENEFICIARIO en 2024, al revés de lo que dice la sección 1.3.

Los 129 estados traen motivo. Los 49 CONFIRMA y DISCREPA traen valor medido y tabla, y de los 80 NO VERIFICABLE, 65 conservan el valor medido de apoyo; los otros 15 son 8 de menos de diez casos y 7 sin métrica. El único valor de 1 a 9 en las columnas protegidas es la diferencia de hospitales de C077 y C078 (3 y 4), que no es un conteo de casos.

Los hallazgos del encargo, desde las tablas de la corrida. Cohorte de estadía de 2019 a 2023: 169.595 de 249.995 egresos son 67,84 % con la lectura literal y 157.109 son 62,84 % sin censurados; el 61,31 % documentado abarca 2019 a 2024, por eso C051 queda no verificable y cita las dos lecturas. Diagnósticos secundarios en adultos hospitalizados: 4,14; 4,73; 5,35; 5,24; 5,53 y 5,82 contra 4,08 a 5,88, un aumento de 40,44 % (38,66 a 42,24) contra 44 % documentado. Diabetes: 30.904 de 208.227 (14,84 %) en ELIX_11 y 8.715 (4,19 %) en ELIX_12 con el mapeo completo, y 0 con tres caracteres. ELIX_06: 71.053 de 208.227 (34,12 %) en adultos hospitalizados; el 27,5 % de todos los egresos con tres caracteres es la base contra la que se compara el 27,8 % de C062.

### Semáforo y supuestos recalculados desde las tablas fuente

La decisión 2 es amarilla porque SERVICIOINGRESO (0,256) y ESPECIALIDAD_MEDICA (0,167) son proxies moderados, ninguna supera el umbral de fuga de 0,85 (0,797 la primera) y HOSPPROCEDENCIA coincide con el hospital del egreso en 30,6 % contra un máximo de 5 %. La 3 es verde porque PROVINCIA (0,725) y COMUNA (0,653) son proxies altos y ES_EXTRANJERO (0,265) moderado, aunque PREVISION (0,094) y NACIONALIDAD (0,073) son bajos: el criterio de H03 solo respalda y nunca contradice. La 7 es verde: las reglas p99 de la base E (51 días) y de D sobrevivientes (50 días) cumplen en la definición y en 2023. La 14 es verde con margen estrecho: el AUC de IR_29301_SEVERIDAD en la confirmación (0,858) supera el umbral y las dos referencias, el de ranking (0,842) no. La 17 es amarilla: 4 de 47 variables con PSI alto, pero hay quiebre del identificador.

Supuestos: S02, 11 de 31 categorías con diferencia relativa mayor que 20 %; S01, 163 de 8.975 IDs (1,82 %) con nacimiento inconsistente y 65 (0,72 %) con sexo, y el grupo de IDs de menos de 5 dígitos tiene 23, de modo que excluirlo no cambia el 1,8 %; S04, 0 de 10 etiquetas con 10 casos o más sin rol; S15, 86 en celdas visibles y 13 suprimidas.

### Cambio de protocolo 1.5 y su autorización

La regla de S09 sigue un argumento estructural y no el valor observado: sin traslado interno el servicio de ingreso coincide con el de alta por construcción, así que solo el grupo con traslado discrimina, y 15,2 % queda muy por debajo del 50 % que habría si la variable registrara la estadía. El cambio sube `umbrales.toml` a la versión 1.5 con cinco cambios declarados, enmienda el `criterio_decision` de S09 en `hipotesis.csv` y dice que se advirtió al ver 99,2 % y 15,2 %; los 24 umbrales del encargo de las Tareas 10 y 11 conservan su valor. Que el orquestador lo autorizó solo consta en el registro del plan (sección 13, vuelta 2, punto 1); no tuve acceso al mensaje original. Con extractos y una regla definida después de ver los datos, el resultado es indicativo, como ya pide CF1-008 al recomendar confirmarlo con los datos completos.

### 2024, privacidad y estabilidad entre procesos

El registro de accesos tiene 78 filas (T03 12, T04 6, T05 12, T06 6, T07 25, T08 12 y T10 5), las mismas antes y después de T12, y los 13 accesos de 2024 son de T3 a T8, con el resultado enmascarado, bajo una regla de calidad autorizada o bloqueados. Los tres módulos de T12 no llaman al lector, a la red ni a procesos; su única lectura directa es `tablas/metricas_eda.csv`. El escaneo dio 0 violaciones y el cortafuegos 0 violaciones con 1 bloqueo (la autoprueba de 2024 de T7 sobre TIPOALTA). Las nueve tablas no traen nombres de hospitales, fechas ni patrones de RUT.

Mi corrida y la del implementador (mismo código, procesos distintos, `/tmp/real_t12b`) producen 137 archivos cada una y 4 difieren; las nueve tablas de T12 y los dos PNG son idénticos. Correr T12 solo desde los CSV (la ruta `--solo`) reproduce seis tablas y los dos PNG; las otras tres cambian en el quinto decimal de z, por el redondeo de las métricas en disco, y en el texto de S10 (35 identidades en vez de 36, porque la cascada sin suprimir solo la recibe una corrida completa), sin cambiar ningún estado.

### Orden de filas en `t23_resultado_estratos.csv` (fuera de T12)

De los 4 archivos distintos, `F08_02.png` y los dos índices (`figuras_indice.csv` y `tablas_indice.csv`) corresponden al defecto de `comorbilidad.py` que la etapa B ya tiene en B0. El cuarto, `t23_resultado_estratos.csv`, no está en el plan: tres pares de filas empatadas en n intercambian su posición (2019 capítulos XX y XVI, ambos con n suprimido; 2022 capítulos XXII y VI, con 555; 2023 CMA y otras instituciones, con 119). `cohorte.py` línea 670 ordena por variable, período y n sin desempate por categoría, y el orden de un empate no es estable. El contenido es el mismo y T12 lee la tabla por clave, pero cada proceso escribe otros bytes, de modo que DT11 fallará. Basta agregar `categoria` como cuarta clave del orden; la sección 0 del plan permite corregir un defecto de una tarea anterior con una prueba y anotarlo en el registro.

### F12_01, F12_02 y la tabla de discrepancias del informe

Los conteos escritos en las figuras coinciden con las tablas: F12_01 tiene 12 decisiones cuantitativas (2 verdes, 6 amarillas, 4 grises), 2 parciales (1 y 1), 2 institucionales y 1 sin datos, estas tres últimas grises; F12_02 tiene 2, 2 y 1 en la Fase 1 y 31, 14 y 79 en la Fase 2. El pie repite "Celdas con 1 a 9 casos suprimidas" aunque las barras cuentan decisiones y cifras; es texto de plantilla de la infraestructura.

Compuse los informes de T12 sobre las tablas de mi corrida: se guardan sin cifras sueltas ni llaves sobrantes y con 196 cifras registradas con CSV y columna. La tabla de discrepancias da a cada cifra el formato de su unidad, pero "Medido" toma los decimales de lo documentado. C076 muestra 44 % contra 40 % (40,44 en el CSV), de modo que la diferencia relativa de 8,1 % no se reproduce con lo que se ve (9,1 %), y C128 y C129 muestran 4 y 68 en vez de 3,97 y 67,64. C018 aparece con una diferencia relativa de 0,2 % (2019 contra 2024), que para un año no informa, y su motivo dice "igualdad con lo que midió la suite" en una fila que discrepa.

### Mutación sobre el código nuevo

Apliqué 21 mutantes a una copia en /tmp, cada uno contra su archivo de pruebas: 10 en `decisiones.py` (los dos bordes de mayoría y el mínimo de egresos de S09, su veredicto y la cita de la segunda proporción, base de H05a, normalización del texto, grupo de S01, borde de S04, ventana de H14), 5 en `ledger.py` y 6 en `cierre.py`. Mueren 20. Sobrevive el cambio de `z_tol > z_umbral` por `>=` en `_nota_redondeo`: las pruebas cubren z con tolerancia en 2,81 y 2,79 contra un umbral de 2,80, pero no la igualdad exacta. La llamada `_nota_redondeo("DISCREPA", 6.0, 3.0, 2.0, 0, 3.0)` da z = 3,00 y distingue el mutante.

### S11 sin años, decisión 1 gris y dos arrastres declarados

S11 cita 40,4 % "entre el primer y el último año", que es 2019 a 2024 (métrica `codificacion.aumento_pct_2019_2024`), mientras el criterio de `hipotesis.csv` dice 2019 a 2023, que en `t23_deriva_codificacion.csv` es 33,6 % (31,9 a 35,3). El estado no cambia, pero la cifra no nombra los años ni dice por qué se aparta del criterio.

La decisión 1 sale gris porque H01b y H01c requieren datos completos, aunque S01 está RECHAZADO y su criterio dice que en ese caso se descartan las variables de historial. La evidencia de la decisión 1 cita la cifra de S01, pero ni ella ni el veredicto dicen esa consecuencia. Es una limitación del diseño del semáforo (los supuestos aportan solo texto) y no un error, pero deja a quien mira solo el color sin la señal más fuerte de esa decisión (likely).

H11 trata el cero del extracto como "matiza" y deja la decisión 11 amarilla; el orquestador decidió mantenerlo. El registro del plan dice que la salvedad (un cero en un extracto no descarta valores en los datos completos, donde la Fase 2 documenta menos de diez) queda en `que_falta_con_datos_completos`, pero esa columna trae solo la frase genérica del extracto. Y CF2-004 (S03, 99,9993 % de códigos válidos) sale en prioridad alta por la regla de los supuestos rechazados que afectan una decisión; la diferencia con el 100 % es de 0,0007 puntos.

### Lo que no pude verificar

El modo completo con datos reales (no hay archivos completos en esta máquina), la autorización del cambio de protocolo 1.5 fuera del registro del plan, las citas `texto_actual` del `.docx` de la Fase 1 (contrasté las de los cambios con el texto de los PDF y los CSV de `docs/`, no todas con la misma profundidad) y la reproducibilidad de la tubería completa en DT11, que corresponde a la Tarea 13.

</details>

<details>
<summary>File map</summary>

- `eda/ledger.py`: `dif_abs` de la cota, nota de tolerancia de redondeo y formato por unidad.
- `eda/decisiones.py`: regla nueva de S09, base de H05a, ventana de H14, textos de S01, S03, S04, S12, S13 y S15, normalización del texto de las decisiones.
- `eda/cierre.py`: cambio de SERVICIOINGRESO, base de la prevalencia, par reconciliado, ubicación de C078, texto de reemplazo de supuestos y sección 3 del cierre.
- `config/umbrales.toml` y `config/hipotesis.csv`: cambio de protocolo 1.5 (versión, estado y entrada 1.5; criterio de S09).
- `tests/ayudas_t12.py`, `tests/test_t12_ledger.py`, `tests/test_t12_decisiones.py` y `tests/test_t12_cierre.py`: 45 pruebas nuevas y las que fijaban el comportamiento anterior, actualizadas.

Diff completo no disponible (el proyecto no usa git); los archivos modificados después de la primera revisión son exactamente estos.

</details>
