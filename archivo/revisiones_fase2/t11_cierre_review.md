# Cierre de la Tarea 11: selección estadística de variables (revisión independiente, vuelta 1)

La etapa C deja cuatro módulos nuevos (`modelos`, `sospechas`, `sensibilidad_elix`, `seleccion`), la parte 2 de `decision`, dos ayudas nuevas en `graficos`, 11 tablas y las figuras F11_01 a F11_06 sobre un panel de 143.350 egresos (115.310 de 2019-2022 y 28.040 de 2023, con 7.326 fallecidos). La suite completa termina sin fallos (1.080 pruebas), dos regeneraciones de la Tarea 11 con el código actual cumplen privacidad y cortafuegos con 0 violaciones, y la primera deja las 11 tablas y los seis PNG de la Tarea 11 idénticos byte a byte a los del implementador. Reproduje de forma independiente con scipy y scikit-learn 42 AUC, 62 odds ratio de Elixhauser, 24 asociaciones con el hospital, 20 PSI, la sobredispersión de los ingresos a cuidados especiales, la sensibilidad de estadía, el árbol de decisión de las 199 variables y el modelo L1 (AUC de 2023 de 0,9251), todo hasta el redondeo de seis decimales. No hay hallazgos HIGH ni MEDIUM; quedan siete NIT.

Watch for: la frase del resumen del implementador que atribuye el AUC de 0,925 del L1 a ESPECIALIDAD_MEDICA y SERVICIOINGRESO solo se sostiene en parte, porque sin ellas el modelo sigue en 0,9096 y sin las cuatro condicionales en 0,9033 (NIT, confirmed); F08_02 de la Tarea 8 cambia de bytes entre corridas con el mismo código, de modo que `figuras_indice.csv` no sirve para comprobar determinismo en toda la carpeta (NIT, confirmed, preexistente); el reinicio del equipo borró `/tmp/real_t10` y `/tmp/real_t11`, que `verificacion-global` espera encontrar (NIT, confirmed).

**Verdict**: APPROVED

## Vista general

Señal univariada, asociación con el hospital, PSI, sobredispersión, odds ratio de Elixhauser, sensibilidad de estadía y modelo L1 se recalcularon con implementaciones propias y coinciden con las tablas hasta el redondeo. El top 5 de complicación probable (ELIX_24, 04, 25, 30 y 27) y sus coincidencias de 3 y 2 con las listas de la Fase 1 y del repositorio salen también de mi cálculo.

La matriz de 199 variables aplica el árbol sin desviaciones: reaplicado desde cero da el mismo resultado en todas las filas, y sus 785 cifras comparadas vienen de la fila correcta de cada tabla fuente. El año 2024 no entra: hay cinco lecturas de la Tarea 11, todas de 2019 a 2023, y `faltante_2024` solo se muestra.

La lectura de que dos condicionales inflan el AUC de 0,925 del L1 exagera: quitarlas lo baja 0,016.

La corrida real con el código actual es idéntica a la del implementador salvo F08_02, una figura de la Tarea 8 que cambia de bytes entre corridas, y el reinicio del equipo borró las salidas que la verificación global esperaba conservar.

Las seis figuras son legibles; los defectos son de rotulado, en F11_02 y en el pie de F11_05. Los p y q muy pequeños se publican como 0,0 por el redondeo a seis decimales.

<details>
<summary>Issues (7)</summary>

1. **Afirmación de AUC inflado del L1**: NIT, confirmed. El resumen del implementador dice que el AUC de 0,925 está inflado por ESPECIALIDAD_MEDICA y SERVICIOINGRESO, pero sin ellas es 0,9096 y sin las cuatro condicionales 0,9033. Matizar la frase o agregar una fila de sensibilidad (L1 sin condicionales) a `t23_importancia.csv`.
2. **Rótulos de F11_02 cruzados por líneas y sin guía**: NIT, confirmed. Las líneas punteadas atraviesan ocho rótulos y los de ELIX_14, 01, 22 y 15 no tienen guía hacia su punto (el "5" de ELIX_15 toca su marcador). Dar fondo blanco semitransparente a los rótulos o agregar líneas guía.
3. **Pie de F11_05 con 0,93**: NIT, confirmed. `eda/seleccion.py:652` formatea el AUC del L1 con dos decimales y el pie dice 0,93 cuando la tabla y el informe dicen 0,925. Pasar tres decimales a `_coma`.
4. **P y q publicados como 0,0**: NIT, confirmed. El redondeo a seis decimales deja `p_crudo` en 0,0 en 44 de 60 filas, `p_ajustado` en 28 de 60 y `p_larga` en 27 de 31. Declarar en la descripción de las tablas que 0,0 significa menor que 0,000001, o publicar el logaritmo del p.
5. **F08_02 no reproducible (Tarea 8, preexistente)**: NIT, confirmed. `eda/comorbilidad.py:424` recorre `group_by` sin `maintain_order=True` y el PNG cambia entre corridas, y con él `figuras_indice.csv`. Agregar el argumento o limitar la comparación de determinismo a las filas de las Tareas 10 y 11.
6. **Salidas reales borradas por el reinicio**: NIT, confirmed. `/tmp/real_t10`, `/tmp/real_t11` y `/tmp/real_t11_run.py` ya no existen. `verificacion-global` debe regenerar la Tarea 11 (3 a 4 minutos) y reescribir el script antes de revisar los PNG.
7. **Edición por shell de `test_t11_seleccion.py`**: NIT, confirmed. El implementador la declara como incumplimiento de la regla del plan; el archivo está íntegro (1.551 líneas, 114 pruebas, sin restos de depuración). No requiere acción salvo no repetirlo.

</details>

<details>
<summary>Details</summary>

### Señal, asociación con el hospital y deriva recalculadas con scipy y scikit-learn

Con el panel de `panel.construir_panel` en un contexto propio y sin importar `modelos`, `sospechas` ni `seleccion`, recalculé 42 AUC con `roc_auc_score`: numéricas y binarias en el ranking y en 2023, categóricas con la tasa suavizada por nivel (m igual a 20) en muestra, por validación cruzada dejando un año fuera y en 2023, y cinco variables de fuga. La diferencia máxima contra las tablas es 4,7e-7, la dirección coincide en todas y los límites de Hanley y McNeil coinciden a cuatro decimales. N_DX_SEC da 0,809856 en 2019-2022 y 0,824285 en 2023; ESPECIALIDAD_MEDICA da 0,819030 en muestra, 0,809850 por validación cruzada (el `auc_senal`) y 0,819785 en 2023; IR_29301_MORTALIDAD da 0,860286 y 0,870696, e IR_29301_SEVERIDAD 0,842146 y 0,857847, de modo que la primera supera 0,85 en las dos ventanas y la segunda solo en la confirmación.

Para la asociación con el hospital, `chi2_contingency` sin corrección de Yates más la corrección de sesgo de Bergsma (y la raíz de eta al cuadrado de `f_oneway` en las numéricas) reproduce 24 valores (12 variables en dos ventanas) con diferencia máxima de 4,9e-7: SERVICIO_SALUD 0,909540 y 0,944698, PROVINCIA 0,724523, COMUNA 0,653392, DERIVADO_OTRO_HOSPITAL 0,571993, SERVICIOINGRESO 0,256076, ESPECIALIDAD_MEDICA 0,166590, NIVEL_CUIDADO_INGRESO 0,155405 (n de 115.167 por los nulos), N_DX_SEC 0,227716 y EDAD_ANIOS 0,118254. El PSI con deciles de la ventana de ranking, el nulo como bin propio y piso de 1e-4 en las proporciones coincide en 20 valores (10 variables, dos comparaciones) con diferencia máxima de 4,6e-7: ETNIA da 6,352185 contra 2023, y ESPECIALIDAD_MEDICA 0,743038 entre 2019 y 2020-2022, el mayor de los cuatro con PSI alto de H17a.

La sobredispersión de Spiegelhalter de los ingresos a cuidados especiales, con los 62 hospitales de al menos 100 filas, proporción global de 0,090044 y winsorización del 10 %, da 29,335646 (45,74 sin winsorizar); los cuantiles p10 de 0,037851 y p90 de 0,146527 dan la razón de 3,871166. La concordancia de servicios da 91.706 de 92.445 egresos sin traslado interno (0,992006, con Wilson de `binomtest` de 0,991411 a 0,992560) y 3.454 de 22.696 con traslado (0,152185). Los conteos del informe que dependen de estas tablas también salen de mi cálculo: 21 OR ajustados relevantes, clases de señal de 9, 6, 5 y 36 sobre 56 candidatas, 11 variables con caída de AUC de al menos 0,001 y 4 de 47 variables clave con PSI alto en H17a.

### Odds ratio de Elixhauser, sensibilidad de estadía e índice de complicación probable

Para las 31 categorías en los dos períodos (62 filas), el OR crudo con el intervalo de Woolf coincide con diferencia máxima de 5,0e-7. El OR ajustado por edad, edad al cuadrado y sexo (`LogisticRegression` sin penalización y error de Wald desde la inversa de la información de Fisher) coincide con diferencia relativa máxima de 4,9e-6 en el OR y sus límites (1,7e-5 en valor absoluto). Los q de Benjamini y Hochberg de `scipy.stats.false_discovery_control` coinciden con diferencia máxima de 4,7e-7 en el crudo y 1,9e-5 en el ajustado. ELIX_24 da 3,6358 (3,2253 a 4,0986) frente a 5,8166 crudo, ELIX_25 4,2274, ELIX_04 3,5128, ELIX_22 5,6465 y ELIX_17 3,7581 contra 1,5747 crudo. Los dos conteos suprimidos de 2023 (ELIX_26 con 3 eventos entre expuestos y ELIX_30 con 6) están entre 1 y 9 en mi recuento.

En la sensibilidad de estadía, los cortes de la cohorte E (104.612 sobrevivientes de la ventana de ranking) son 2, 4 y 8 días, con tramos de 34.567, 21.941, 18.435 y 29.669. La razón de prevalencias entre el último y el primer tramo con el intervalo de Katz coincide con diferencia de 4,9e-7, el OR de estadía larga con 7,4e-6 y el q con 4,6e-7. El índice de complicación probable, recalculado con `rankdata` desde mis propios límites inferiores, da los mismos 31 rangos (0 discrepancias) y el mismo top 5: ELIX_24, ELIX_04, ELIX_25, ELIX_30 y ELIX_27, con 3 coincidencias con las cinco de la Fase 1 (ELIX_04, 24 y 25) y 2 con la lista del repositorio (ELIX_24 y 25).

### Árbol de decisión de las 199 variables y cruce con las tablas fuente

Reapliqué a la matriz publicada las nueve reglas de `arbol_decision.csv` con los umbrales de `umbrales.toml` (fuga 0,85, proxy 0,30 y 0,15, faltante 0,50, PSI 0,25, señal baja 0,55, incremento 0,005, y representante de cluster por mayor `auc_senal`, luego menor faltante y nombre). Las 199 filas reproducen `regla_decisiva`, `reglas_activadas`, `recomendacion` y `contraste_fase1`, también las cinco con cifras anuladas (SEXO, PROVINCIA, PREVISION, SERVICIO_SALUD y ESPECIALIDAD_MEDICA), donde la recomendación no depende de la cifra que se anuló. Los conteos son 110 EXCLUIR por la Fase 1, 39 NO EVALUABLE más 3 NO EVALUABLE en extracto, 20 MANTENER, 11 MANTENER por criterio clínico, 8 BAJA SEÑAL, 6 CONDICIONAL por proxy, 1 ELIMINAR por redundancia y 1 EXCLUIR por proxy; los contrastes son 92, 42, 18, 13, 11, 9, 7 y 7. Ninguna variable con medición incompleta sale MANTENER ni con etiqueta de estadística que no objeta, e IR_29301_MORTALIDAD queda excluida por la Fase 1 con FUGA también activa.

Además comparé 785 celdas de la matriz con su tabla fuente: AUC del ranking y de 2023 desde el ranking o desde la fuga, `auc_senal`, incremento, asociación con el hospital, los dos PSI, cambio de signo, faltante desde el panel o desde t22, y cluster y tamaño de cluster contra la Tarea 10. No hay una sola discrepancia.

### Cortafuegos de 2024, privacidad y restricciones del flujo

El registro de accesos de mi corrida tiene 23 entradas, 22 permitidas y una permitida enmascarada. Las de 2024 son de la Tarea 4 (una, enmascarada, con propósito de estructura) y de la Tarea 8 (dos, sin columnas de resultado); las cinco de la Tarea 11 son de 2019 a 2023, con propósito `seleccion` y 98 columnas. Ningún módulo de la etapa llama a `ctx.lector`: la búsqueda de `ctx.lector` y `.leer(` solo encuentra docstrings, y la única lectura es `panel.construir_panel`. Las 377 métricas de la Tarea 11 (133 de 2019-2022, 233 de 2023 y 11 de 2019-2023) llevan alcance explícito y ninguna cubre 2024; las 97 con 2024 en el alcance son de la Tarea 8. `disponibilidad_variables.csv` muestra `faltante_2024` sin usarlo: `faltante_max_t22` es igual al máximo de 2019 a 2023 en las 199 filas, aunque en 11 de ellas el valor de 2024 lo supera y habría cambiado la cifra.

El escáner de la suite da 0 violaciones de privacidad en las dos regeneraciones. Mi barrido independiente de las tablas de la Tarea 11 no encuentra ningún valor entre 1 y 9 en las columnas de conteo declaradas en `tablas_indice.csv`; los que aparecen en columnas `n_*` no declaradas son conteos de variables, de años o de columnas del diseño (`n_proxy_alto`, `n_anios_efecto_suficiente`, `n_columnas_diseno`), no de casos. Las 33 sumas de verificación del índice coinciden con los archivos. Fuera de `fase2_eda` solo cambiaron documentos de `.agents`; no hay `.md` nuevos fuera de `.agents`, `salidas/` y los catálogos de la raíz están intactos, `config/umbrales.toml`, `hipotesis.csv`, `conocimiento_clinico.csv` y `requirements.txt` no cambiaron después del encargo, y el código no importa `statsmodels` ni `pandas`.

### Modelo L1 reproducido y la afirmación de AUC inflado

Con un diseño propio (las 47 variables que la Fase 1 no excluye, numéricas estandarizadas con el entrenamiento, nulos rellenados con la mediana más indicador y una columna por nivel de cada categórica) y `LogisticRegression` con penalización L1 y `liblinear` al C publicado de 0,464159, salen 294 columnas, AUC de 0,9251 en 2023 y 0,9248 en el ranking, 176 coeficientes no nulos y 42 variables que sobreviven, todo igual a `t23_importancia.csv`. Lo que no se sostiene es la lectura. El resumen del implementador y el registro del plan dicen que esa cifra "está inflada por ESPECIALIDAD_MEDICA y SERVICIOINGRESO" y que "no es un desempeño esperable", y ninguna salida de la suite lo mide. Al reajustar al mismo C sin esas dos variables el AUC de 2023 baja a 0,9096 (45 variables, 170 columnas), y sin las cuatro condicionales, con NIVEL_CUIDADO_INGRESO y DERIVADO_OTRO_HOSPITAL, a 0,9033 (43 variables, 167 columnas). Aportan 0,016 y 0,022 puntos y el modelo sigue entre 0,90 y 0,91; el reajuste usa el C publicado y no repite la validación cruzada de cada variante. El informe provisional solo advierte que los coeficientes "pueden codificar el hospital" y no necesita cambio. La cifra comunicada fuera del informe conviene matizarla o respaldarla con una fila de sensibilidad en `t23_importancia.csv` (NIT, confirmed).

### Corrida real con el código actual y la figura F08_02 de la Tarea 8

`eda/sensibilidad_elix.py` y `tests/test_t11_seleccion.py` cambiaron a las 13:49 del 10 de octubre, después de la corrida real del implementador (13:12 a 13:15), por la retirada de un import sin uso y dos retoques en una prueba (un nombre con tilde y un espacio sobrante), según el registro del plan. Por eso regeneré las Tareas 4, 8 y 11 sobre los extractos con el código actual. En la primera regeneración, 50 de los 52 archivos de `tablas`, `figuras` e `informes` son idénticos byte a byte a `/tmp/real_t11` del implementador, entre ellos las 11 tablas de la Tarea 11, el informe provisional y los seis PNG, así que lo que se revisó visualmente es lo que produce el código actual. Los dos distintos son `figuras/F08_02.png` y `tablas/figuras_indice.csv`, este último por el hash de ese PNG.

F08_02 es la deriva de codificación de la Tarea 8: `eda/comorbilidad.py:424` recorre `hosp.group_by("cod_hospital")` sin `maintain_order=True` y dibuja una línea gris por hospital en orden variable. Cuatro corridas con el mismo código dieron cuatro hashes distintos de F08_02 (1854107f, 97dc874d, 553137a5 y 69f0e8a6), mientras F08_01, F08_03 y `t23_deriva_codificacion.csv` salieron iguales en las tres que comparé; contra el implementador cambian 2.668 píxeles del panel derecho, con diferencia máxima de 6 sobre la suma de canales. El archivo no cambió desde el 8 de octubre, así que el defecto es anterior a esta etapa y queda fuera de su diff, pero hace que `figuras_indice.csv` no sea reproducible entre corridas y obliga a limitar cualquier comparación global a las filas de las Tareas 10 y 11 (NIT, confirmed).

La Tarea 11 tardó 201 s en la primera regeneración y 250 s en la segunda (con mis propias verificaciones en paralelo), contra 165 s del registro, y el pico de memoria fue de 768 y 737 MB contra 819 MB; sigue bajo los cinco minutos que fija la sección de riesgos del plan.

### Legibilidad de F11_01 a F11_06

Abrí los seis PNG con `read_file` y recorté y amplié F11_02. Ningún texto toca el borde de la imagen, las leyendas quedan fuera de los datos en las seis, el eje de F11_01 declara que las barras parten de 0,5 y la escala logarítmica de F11_04 lleva marcas en 0,5, 1, 2, 4 y 8. F11_03 ya no muestra "-0,00": el único negativo es el "-0,01" real de ELIX_13 en 2022 y 2023. Los defectos son dos.

En F11_02 las líneas punteadas en 0,15 y 0,30 atraviesan ocho rótulos (GRUPO_CLINICO, PUNTAJE_VAN_WALRAVEN, N_COMORB_ELIX, PUNTAJE_VAN_WALRAVEN_SIN_5, ELIX_25, SERVICIOINGRESO, ES_EXTRANJERO y ENTRANTE_TIPO_PROCEDENCIA); todos siguen legibles, pero `etiquetas_sin_solape` solo esquiva rótulos y puntos, no las líneas. En el rincón inferior izquierdo, ELIX_14, ELIX_01, ELIX_22 y ELIX_15 quedan a la izquierda de una pila de cuatro triángulos sin guía hacia su punto, y el "5" de ELIX_15 toca su marcador (NIT, confirmed). El pie de F11_05 dice "AUC del modelo L1 en 2023: 0,93" porque `eda/seleccion.py:652` llama a `_coma(auc_modelo)` con dos decimales, mientras la tabla y el informe dan 0,925 (NIT, confirmed).

### P y q publicados como 0,0

`Contexto.escribir_tabla` redondea los flotantes a seis decimales (`_limpiar_numericos` en `eda/contexto.py`), de modo que un p de 5,4e-99, como el de ELIX_24, se publica como 0,0. En `t23_or_elix.csv` salen 0,0 el `p_crudo` en 44 de 60 filas con valor, el `p_ajustado` en 28 de 60 y el `q_ajustado` en 27; en `t23_sensibilidad_elix.csv`, `p_larga` y `q_larga` en 27 de 31. Los q se calculan antes de redondear y no se contaminan, pero quien lea la tabla o recalcule Benjamini y Hochberg desde los p publicados encontrará ceros exactos, que no existen (NIT, confirmed).

### Pruebas, el mutante M53 y la edición por shell de test_t11_seleccion.py

`pytest.ini` ya trae `-q` y un segundo `-q` en la línea de comandos oculta la línea final de resumen, por lo que conté los símbolos del avance: 1.080 puntos y ninguna otra línea, 408 más que la línea base de 672 del plan. El registro del implementador decía que M53 (rango mínimo sin desempate) lo mató primero la prueba del gradiente y no la dedicada a los tres empates. En una copia en `/tmp` apliqué `rankdata(..., method="min")` en lugar del orden total y corrí solo esa prueba: pasa sin mutar y falla con la mutación (`[1, 2, 3, 4, 4, 4, ...]` contra `[1, 2, 3, 4, 5, 6, ...]`), así que también lo detecta. No repetí los otros 23 mutantes.

`tests/test_t11_seleccion.py`, que el implementador declara haber editado dos veces por shell contra la regla del plan, está íntegro: 1.551 líneas, UTF-8 sin retornos de carro, termina en su marca `# FIN-test_t11_seleccion`, recolecta 114 pruebas y no tiene restos de depuración. Con `ast` no hay imports sin uso ni identificadores no ASCII en 14 archivos entre módulos y pruebas de las etapas B y C, ningún nombre de prueba se repite dentro de un archivo en toda la carpeta `tests` (839 funciones) y ninguna prueba de las Tareas 10 y 11 carece de aserciones. La violación de la regla de edición no tuvo consecuencias observables (NIT, confirmed).

### Salidas reales borradas por el reinicio del equipo

Entre mi primera pasada y esta, el equipo se reinició (en `/tmp` solo quedan entradas del arranque de las 15:26): desaparecieron `/tmp/real_t10`, `/tmp/real_t11`, `/tmp/real_t11_run.py` y los registros de esa corrida, y con ellos mi primera regeneración. La sección S14 del plan pedía conservarlos hasta `verificacion-global` para abrir los PNG desde ahí; ese paso tendrá que regenerarlos y reescribir el script. Repetí la regeneración y los cruces que dependían de ella; la identidad byte a byte con el implementador y la revisión visual corresponden a la primera regeneración (NIT, confirmed).

### Lo que no se pudo verificar

El modo completo con datos reales (hasta 800.000 filas): memoria y tiempo no son extrapolables desde los extractos, y el historial es no evaluable en extractos por construcción. La cardinalidad de ESPECIALIDAD_MEDICA (75 niveles y 12 en un solo hospital, contra los 76 y 13 de la medición de la revisión 3) depende de la lectura cruda por hospital que ya resolvió la etapa A, y no la recalculé. Los mapeos y fichas pendientes de revisión clínica (decisión 12) quedan fuera del alcance estadístico. Del registro de mutantes de la etapa no pude comprobar la variante 28, que el registro cita sin individualizar (enumera 27).

</details>

<details>
<summary>Mapa de archivos</summary>

- `eda/modelos.py` (nuevo, 681 líneas): diseños, ajustes logísticos, referencias, incremento de AUC, L1, permutación y AUC univariado compartido.
- `eda/sospechas.py` (nuevo, 487 líneas): proxy del hospital y fuga, con F11_02 y F11_06.
- `eda/sensibilidad_elix.py` (nuevo, 395 líneas): OR de las 31 categorías, sensibilidad de estadía, índice de complicación probable y F11_04.
- `eda/seleccion.py` (nuevo, 1.215 líneas): ranking, estabilidad, importancia, parámetros, resumen, orquestación, métricas, informe de nueve secciones y F11_01, F11_03 y F11_05.
- `eda/decision.py` (ampliado, 1.020 líneas): árbol, matriz, publicación, escritura, resumen y la sección 8 del informe.
- `eda/graficos.py`: `etiquetas_sin_solape`, gancho `antes_de_guardar` y anotación sin "-0,00" en `mapa_calor`.
- `tests/test_t11_modelos.py`, `test_t11_evidencias.py`, `test_t11_seleccion.py`, `test_t11_decision.py`, `test_graficos_t10t11.py` y `ayudas_figuras.py`: pruebas de la etapa.

Sin git en este flujo: el cambio completo es el contenido de estos archivos. La salida revisada se regeneró en una carpeta temporal de `/tmp` (tablas, figuras e informe provisional), borrada al terminar junto con los scripts de apoyo.

</details>
