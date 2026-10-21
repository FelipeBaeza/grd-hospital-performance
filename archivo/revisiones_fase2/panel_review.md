# Panel compartido de las Tareas 10 y 11: estadística bivariada, supresión por bloques y cardinalidad de especialidad (Etapa A)

La etapa A entrega las funciones estadísticas nuevas de `eda/estadistica.py`, `suprimir_bloques`, la parte 1 de `eda/decision.py` (universo, roles y disponibilidad), `eda/panel.py` con su lectura única de 2019 a 2023, el panel artificial de pruebas y 285 pruebas nuevas sobre las 286 originales. El panel fija los 60 niveles con 2019-2022, no imputa nada y calcula la cardinalidad de ESPECIALIDAD_MEDICA sobre el texto sin tope, como exige el MEDIUM 1. Reproduje 33 comparaciones de estadísticos con scipy y scikit-learn usando datos propios, recalculé la cohorte D y ocho variables derivadas sobre los extractos reales (143.350 filas, 0 diferencias) y reproduje la discrepancia de la cardinalidad (75 y 12 frente a 76 y 13) con su causa exacta. No hay hallazgos HIGH ni MEDIUM.

Watch for: (a) la cifra de H02b que verá el usuario es 75 niveles y 12 exclusivos de un hospital, sobre los hospitales con al menos 100 filas; no es un error, pero C3 y C7 deben declarar esa base o alguien la "corregirá" a los 76 y 13 de la revisión 3 (confirmed). (b) Ocho llamadas `group_by` o `unique` de `panel.py` no llevan `maintain_order=True`, lo que incumple la letra de DT11 sin efecto medible (confirmed).

**Verdict**: APPROVED

## High-level view

Estadística. Las funciones nuevas de `estadistica.py` coinciden con scipy y scikit-learn en todas las comparaciones que armé con semillas y tamaños propios. `redondear_publicado` usa `np.round` y reproduce el redondeo de polars, que es quien escribe el CSV, incluso en los empates binarios exactos.

Supresión por bloques. `suprimir_bloques` coincide con un oráculo ingenuo y no depende del orden de los bloques; `escribir_por_bloques` declara la desviación de la regla del encargo y suma las celdas anuladas al índice de tablas (NIT 6).

Lectura y partición temporal. `construir_panel` es el único punto de lectura de las dos tareas: una llamada por año de 2019 a 2023 y una guarda que corta antes de leer si 2024 entra a la partición. Sobre los extractos reales deja 0 violaciones, 0 bloqueos y ningún acceso de 2024.

Fidelidad de las variables. La cohorte D, ocho variables derivadas, el tope de 60 niveles, la disponibilidad, el historial y el registro de variables coinciden con recálculos propios hechos con código y tablas tipeados por mí, no con las pruebas del repositorio.

Cardinalidad de ESPECIALIDAD_MEDICA. El MEDIUM 1 está implementado como pide la revisión y M52 muere con la prueba de la columna recortada. La diferencia entre 75/12 y 76/13 es un único nivel de menos de diez filas que el código excluye por dos razones independientes y que la revisión 3 pudo contar por cualquiera de las dos; el código no se ajustó a la cifra. El MEDIUM 2 pertenece a la etapa C y no existe aún en el código.

Límites. De los 41 mutantes del registro solo reproduje M52. El modo completo con datos reales, el tiempo con la fuente completa y los mapeos clínicos de la decisión 12 no son verificables con extractos.

<details>
<summary>Issues (2)</summary>

1. **Base de la cardinalidad sin declarar en las salidas de C** — la cifra de H02b será 75 niveles y 12 exclusivos (hospitales con al menos 100 filas) frente a los 76 y 13 de la revisión 3; en C3 y C7, anotar la base en la fila `cardinalidad_especialidad` de `t23_proxy_hospital.csv` y en la sección 4 del informe, sin dígitos (NIT, confirmed).
2. **DT11 literal en `panel.py`** — ocho `group_by` o `unique` sin `maintain_order=True` (líneas 258, 272, 406, 459, 577, 581, 582 y 742), sin efecto medible; agregar el argumento antes de que B y C copien el patrón (NIT, confirmed).

</details>

<details>
<summary>Details</summary>

### Qué se corrió

La suite se corrió una vez: `cd fase2_eda && python -B -m pytest tests` dio `571 passed in 23.10s`, y `pytest --collect-only` confirma 286 pruebas originales más 129, 42, 67, 30 y 17 en los cinco archivos nuevos. Desde las 10:24 del 9 de octubre solo cambiaron `estadistica.py`, `supresion.py`, `decision.py`, `panel.py`, los archivos de prueba nuevos y el plan (`find -newermt`); `config/`, `docs/`, `hospitales.py`, `cohorte.py`, `contexto.py`, `elixhauser.py` y `requirements.txt` no cambiaron, y no hay `.md` nuevos fuera del plan. Mis scripts, escritos en `/tmp` y borrados al terminar junto con la copia de mutación y los directorios de pytest de mis corridas, importan las funciones del repositorio, comparan con scipy, scikit-learn o Python puro e imprimen solo agregados.

### Estadísticos reproducidos con scipy y scikit-learn

Las semillas y los tamaños son distintos de los de las pruebas del repositorio. Spearman sobre 5.000 pares con empates y 5 % de NaN da 0,215351370891 en el repositorio y en `scipy.stats.spearmanr`. El AUC de 4.000 puntajes con empates da 0,757902908558 igual que `roc_auc_score`. La V de Cramér corregida coincide a 1e-12 con la fórmula de Bergsma armada sobre `chi2_contingency` en seis tablas de 2x2 a 4x6, y al multiplicar una tabla por 1000 converge a 0,417822 frente a 0,417855 de `association` sin corregir. El OR de Woolf y su intervalo coinciden con `odds_ratio(kind="sample")` en tres tablas (8,142857 con 3,784040 a 17,522575 en la primera). La regresión logística difiere de `LogisticRegression(C=np.inf)` en 1,2e-8 en los coeficientes y de `C = 1/ridge` en 1,1e-8 y 7,8e-9 con ridge 0,5 y 3,0, y sus errores estándar coinciden con la inversa de X'WX en 4,4e-16 relativo. Benjamini y Hochberg, la información mutua normalizada, eta cuadrado, phi con faltantes por columna, el VIF, el enlace promedio y los componentes también coinciden con `false_discovery_control`, `normalized_mutual_info_score`, `f_oneway`, `matthews_corrcoef`, `diag(inv(R))`, `linkage` y una búsqueda en anchura propia, con diferencias de 1e-12 o menos. Los límites de los intervalos de Hanley y McNeil y de Woolf se compararon con tolerancia 1e-5 y 1e-6 porque dependen del valor de z.

El NIT 3 se probó contra polars con 400.005 valores (la mitad de la forma k/1e6 + 0,5e-6), 4.504.500 fracciones n/total con total hasta 3.000 y 256 empates binarios exactos (j/128 con j impar): 0 discrepancias entre `redondear_publicado` y `pl.Series.round(6)`, mientras que `round(0.7999995, 6)` de Python da 0.799999. `componentes_redundantes` compara el valor redondeado (`estadistica.py:743`), de modo que un par en 0,7999999999999999 es arista y queda en el mismo grupo del dendrograma, como fija su prueba.

### Cortafuegos, partición temporal y privacidad sobre los extractos reales

`Contexto.crear("extracto", ...)`, `panel.construir_panel`, `panel.escribir_tablas` y `ctx.escribir_registros` dejan cinco entradas en el registro de accesos (`T10`, 2019 a 2023, `seleccion`, `PERMITIDO`), `violaciones() == 0`, `n_bloqueos() == 0` y ninguna entrada de 2024; `privacidad.escanear_salidas` suma 0. Una búsqueda estática confirma que `panel.py:729` es la única llamada a `ctx.lector.leer` entre los módulos nuevos y que la guarda de `panel.py:711` revisa tanto los años a leer como `anios_resultado`; las demás menciones de 2024 son docstrings. `COLUMNAS_PANEL` excluye `FECHAPROCEDIMIENTO1`, `MEDICOINTERV1_ENCRIPTADO`, `MEDICOALTA_ENCRIPTADO`, `CIP_ENCRIPTADO` y el bloque de recién nacidos, y `datos` (71 columnas en extracto) no trae identificadores, fechas ni `TIPOALTA`. Dos corridas en carpetas distintas dan el mismo hash de `datos` y los mismos `sha256` en las dos tablas del panel, `tablas_indice.csv`, `metricas_eda.csv` y `registro_accesos.csv`. `t23_panel_muestreo.csv` escribe 5 filas con 0 celdas suprimidas y `t23_panel_niveles.csv` 1.150 filas con 114; las cuatro métricas `panel.*` tienen alcance explícito (2019-2022 o 2023) y valen 115.310, 6.182, 28.040 y 1.144.

### Cardinalidad de ESPECIALIDAD_MEDICA: 75 y 12 frente a 76 y 13

El MEDIUM 1 está en `panel.py:754` y `809-810`: el texto de las filas D muestreadas de la ventana de ranking (solo con los espacios de los bordes quitados) se acumula antes de toda recodificación y `cardinalidad_especialidad` (`panel.py:555`) devuelve solo seis agregados. La prueba de 70 niveles con tres exclusivos fuera de los 60 más frecuentes da `n_niveles_totales = 70` y `n_niveles_en_un_hospital = 3`; sobre la columna ya recortada a 60 niveles daría 61 y 0. Apliqué M52 en una copia en `/tmp` (cardinalidad sobre la columna ya recortada) y `pytest tests/test_panel.py -k cardinalidad` pasa de 5 verdes a 4 verdes y 1 roja: `test_la_cardinalidad_se_calcula_antes_del_tope_de_niveles_aunque_la_columna_quede_recortada` falla con `n_niveles_totales` 6 frente a 18 y `n_niveles_en_un_hospital` 5 frente a 18.

Recalculé la cifra en Python puro, sin llamar a la función del repositorio, con la cohorte D de 2019 a 2022 (115.310 filas). Hay 65 hospitales, 62 con al menos 100 filas y 3 con menos (23 filas en total). La lectura vigente da 62 hospitales, cardinalidad de 3 a 39 con mediana 16, 75 niveles y 12 exclusivos, que es lo que devuelven `panel.cardinalidad_especialidad` y `construir_panel`. Contar los niveles sobre los 65 hospitales da 76 y 13, igual que contar el centinela "DESCONOCIDO" como nivel; hacer ambas cosas da 77 y 14; recortar espacios, pasar a mayúsculas o quitar tildes no cambia nada (192 filas con espacios al borde, 0 nulos, 0 vacíos); y el tope de 60, que es el error del MEDIUM 1, da 60 niveles, 5 exclusivos y cardinalidad de 3 a 38, como midió la revisión 3. La causa exacta es un único nivel que el código no cuenta y que la revisión 3 sí contó: el único nivel que usan solo hospitales chicos (2 filas, un hospital) o el centinela "DESCONOCIDO" (3 filas, un hospital que es de los 62). Hay siete niveles compartidos entre hospitales grandes y chicos y ninguno queda como exclusivo por culpa del filtro, así que 76 = 75 + 1 y 13 = 12 + 1. La revisión 3 no dice cuál de los dos caminos tomó, y por eso su cifra no permite decidir cuál lectura corregir.

Acepto la lectura literal. Es la que escriben DIS 10.4 ("por hospital con al menos `MIN_FILAS_HOSPITAL` filas"), la corrección del hallazgo 1 de la revisión 3 (`# solo agregados, con MIN_FILAS_HOSPITAL`) y la nota S18 del plan, y ajustar el código a 76 y 13 obligaría a elegir entre dos exclusiones sin evidencia de cuál midió la revisión. La sección "Resolución de la decisión abierta" del registro (matriz de escenarios, las dos causas y la alternativa de dos líneas) coincide con mi recálculo en cada fila que reproduje, y es un buen ejemplo de la regla de no ajustar el código a la cifra. Decidir más tarde cuesta poco porque B y C todavía no consumen estas cifras.

### Supresión por bloques y NIT 6

`suprimir_bloques` frente a una implementación ingenua en 400 tablas aleatorias (nulos, ceros, 1 a 9, 10 y más, `minimo` 5 y 10, uno a tres bloques con dependientes que a veces son el conteo de otro bloque) da 0 diferencias en la tabla y en el número de celdas, y el mismo resultado con los bloques invertidos. `escribir_por_bloques` (`panel.py:847`) lleva la desviación declarada en el docstring, falla con una columna ausente antes de escribir y suma `n_celdas` a `celdas_suprimidas` del índice (`panel.py:871`). Las pruebas fijan la fila (40, 4), el índice con `columnas_conteo = "n;n_2023"` y `celdas_suprimidas == 2`, el escáner que detecta una celda de 2023 introducida a mano y el umbral 5 frente a 10.

### Variables, niveles, disponibilidad e historial

Recalculé con el mapeo de 7.4 tipeado por mí DIA_SEMANA, MES_INGRESO, ES_FIN_SEMANA, NIVEL_CUIDADO_INGRESO, DERIVADO_OTRO_HOSPITAL, ES_EXTRANJERO, PUEBLO_ORIGINARIO y PROCEDENCIA_AGR sobre las 143.350 filas D de 2019 a 2023 y las comparé fila a fila con el panel: 0 diferencias y 0 fechas sin parsear. En PROCEDENCIA_AGR, "(sin mapeo)" son 2.728 de 115.310 en 2019-2022 (2,37 %) y 606 de 28.040 en 2023. Las identidades de Elixhauser son exactas (`N_COMORB_ELIX` es la suma de las 31 categorías y los dos puntajes son la combinación con `e.pesos`). Para ESPECIALIDAD_MEDICA, SERVICIOINGRESO, COMUNA, NACIONALIDAD y PROVINCIA la lista de niveles es idéntica a los 60 más frecuentes de 2019-2022 y las filas en OTROS coinciden en ambas ventanas; con 2019-2023 la lista cambiaría, de modo que la prueba no es vacía.

`decision.construir_disponibilidad` con el `t22_perfil_columnas.csv` real da 199 filas (129 crudas, 66 derivadas, 4 de reglas), 134 no evaluables con motivo y 1.194 celdas de faltante por año iguales a mi recálculo de `pct_faltante / 100`; `faltante_2024` supera a `faltante_max_t22` en 11 variables, así que 2024 no entra al máximo. El registro coincide en ambos modos con la tabla de 7.3 que tipié aparte (75 variables; 58 candidatas evaluables en extracto, con 5 numéricas, 37 binarias y 16 categóricas, y 61 en modo completo). `historial_por_episodio` coincide con una definición de fuerza bruta de 7.7 en 1.500 episodios aleatorios con nulos, fechas invertidas, reingresos del mismo día, identificadores de 15 dígitos y dos espacios de identificadores (828 viables, 787 con previo, 0 diferencias). El panel artificial cumple la verdad de diseño de 14.2 medida con scipy y scikit-learn (AUC de EDAD_ANIOS 0,704, AUC de ELIX_03 0,508, V de 0,997 o más en los pares anidados, V con el hospital de 0,995 y 0,226 en ESPECIALIDAD_MEDICA y COMUNA, y la cadena da un componente con los grupos {A, B} y {C}).

### Lo que no se pudo verificar

De los 41 mutantes del registro solo se reprodujo M52; los demás se apoyan en las pruebas que el registro les asigna. El modo completo con datos reales, el historial sobre datos reales y el tiempo y la memoria con la fuente completa no son verificables con extractos. `MAPA_PROCEDENCIA_AGR`, `PATRONES_CUIDADO_ESPECIAL` y la codificación de PUEBLO_ORIGINARIO se verificaron contra el diseño, no contra criterio clínico, que sigue pendiente de la decisión 12.

### NIT: `group_by` y `unique` sin `maintain_order=True` (confirmed)

La regla DT11 del plan (sección 0) pide `maintain_order=True` o un orden posterior con desempate único en todo `group_by` o `unique` de polars del código nuevo. Cumplen `calcular_niveles` (ordena por `["n", "v"]`) y `calcular_niveles_referencia` (usa `sorted`). No cumplen `panel.py:258` (`_por_distintos`), `272` (`limpiar_categorica`), `406` (`calcular_niveles`), `459` (`tabla_de_niveles`), `577`, `581` y `582` (`cardinalidad_especialidad`) y `742` (`ids_validos[anio]`). En todos el resultado alimenta un conjunto, un diccionario o un agregado simétrico (mínimo, máximo, mediana) y dos corridas reales dan hashes idénticos, así que no hay efecto medible. El riesgo es de copia: B y C seguirán el patrón de este módulo. La corrección es agregar el argumento, sin rehacer mutaciones.

### NIT: declarar la base de la cardinalidad en las salidas de la etapa C (confirmed)

Cuando C3 copie los agregados a la fila `cardinalidad_especialidad` de `t23_proxy_hospital.csv` y C7 los lleve al resumen y a la sección 4 del informe, la cifra visible será 75 y 12, mientras la revisión 3 cita 76 y 13 y la tabla de la Tarea 4 cita 89 niveles de la variable. Una nota en la fila o una frase sin dígitos en el informe (hospitales con al menos el mínimo de filas, niveles reales, ventana de ranking) evita que un lector o un revisor posterior lo tome por un error.

</details>

<details>
<summary>Mapa de archivos</summary>

- `eda/estadistica.py`: constantes técnicas, validaciones `_exigir_*`, medidas bivariadas, regresión logística propia, VIF, agrupamiento, puntaje categórico, deciles y PSI de códigos.
- `eda/supresion.py`: `suprimir_bloques`.
- `eda/decision.py` (nuevo): universo de 199 variables, `rol_fase1`, motivos de no evaluables, `construir_disponibilidad`.
- `eda/panel.py` (nuevo): `Variable` y `Panel`, registro de 75 variables, definiciones operativas, niveles, muestreo, historial, cardinalidad de especialidad, `construir_panel`, `escribir_tablas`, `escribir_por_bloques`.
- `tests/panel_artificial.py` (nuevo): panel con verdad de diseño y `umbrales_temporales`.
- `tests/test_estadistica_asociacion.py`, `test_supresion_bloques.py`, `test_t11_decision.py`, `test_panel.py`, `test_panel_artificial.py` (nuevos): 129, 17, 42, 67 y 30 pruebas.
- `.agents/tasks/t10t11_plan_impl.md`: casillas A1 a A8 y registro de evidencia de la etapa A (sección 10).

No hay diff de git (la consigna excluye git); la revisión se hizo leyendo los archivos y ejecutando los scripts descritos arriba.

</details>
