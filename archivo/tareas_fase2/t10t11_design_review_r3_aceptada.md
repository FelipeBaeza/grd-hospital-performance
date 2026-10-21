# Revisión del diseño de las Tareas 10 y 11 (revisión 3)

Veredicto: CHANGES_REQUESTED (0 HIGH, 2 MEDIUM, 10 NIT).

Regla mecánica aplicada: cualquier hallazgo HIGH o MEDIUM bloquea. Hay dos MEDIUM, así que el diseño vuelve a su autor antes de implementar. Los 15 hallazgos de la revisión 2 están atendidos en el texto (los verifiqué uno por uno contra las secciones 8.1, 10.1 a 10.11, 11 y 14); los dos MEDIUM de esta revisión son nuevos y se arreglan con párrafos cortos.

Qué se leyó y cómo se verificó. El encargo (`fase2_brief_t10_t11.txt`, 290 líneas) y el diseño (`t10t11_diseno.md`, 1145 líneas) se leyeron completos. Se contrastaron con el código real de `fase2_eda/eda` (`contexto`, `supresion`, `privacidad`, `cortafuegos`, `lector`, `config`, `rutas`, `cohorte`, `elixhauser`, `estadistica`, `graficos`, `informes`, `hospitales`, `comorbilidad`, `esquema`, `transformaciones`, `protocolo`, `conocimiento`, `ejecucion`), con `config/` (umbrales, árbol, plan de figuras, hipótesis, definición de terminado, conocimiento clínico, categorías de Elixhauser), con `docs/` (Reglas del Modelo R1 a R8, diccionario, variables derivadas, `reglas_variables.csv`, decisiones, `fase1_spec.json`) y con las salidas ya existentes de las Tareas 1 a 6. Se corrió la suite (286 pruebas verdes), se recalcularon con numpy, scipy y scikit-learn las cuentas a mano del diseño y se ejecutaron tres verificaciones: el generador sintético (conteos de la cohorte D, categorías constantes, AUC de las variables de fuga), la cohorte D de los extractos 2019 a 2023 y la cardinalidad de ESPECIALIDAD_MEDICA. Las lecturas reales pasaron por `ctx.lector.leer`, solo imprimieron conteos y el cortafuegos terminó con 0 violaciones, 0 bloqueos y ningún acceso a 2024. No se evaluó modelo ni esfuerzo. Todo lo temporal estaba en `/tmp` y se borró al terminar.

## Criterios de rechazo del encargo

| Criterio | Resultado |
|---|---|
| Lectura de 2024 | Sin hallazgo. Única lectura en `panel.construir_panel`, solo años de `anios_resultado`, `RuntimeError` si aparece el año expuesto, ninguna métrica con alcance que cubra 2024. `faltante_2024` sale de `t22_perfil_columnas.csv` (tabla de la Tarea 4, estructura y calidad). Verifiqué en `cortafuegos.autorizar` que los años no expuestos admiten el propósito `seleccion` y solo se registran. |
| Imputación de faltantes | Sin hallazgo. El panel conserva nulos y "(sin dato)". El relleno con la mediana del entrenamiento más indicador de nulos vive solo en `modelos`; R5 dice textualmente que "las líneas base los completan usando solo datos de entrenamiento" y el diseño lo declara en el informe y en `t23_parametros_seleccion.csv`. |
| Umbrales de `umbrales.toml` | Sin hallazgo. No se cambia ni se agrega ninguno (versión 1,4). Las constantes técnicas de módulo tienen precedente real (`SIMILITUD_MINIMA` y `MIN_EPISODIOS_EMBUDO` en `hospitales.py`) y se publican. |
| Conocimiento clínico fuera de las cuatro fuentes | Sin hallazgo. Mapeo de procedencia, patrones de cuidado especial, lista del repositorio y PUEBLO_ORIGINARIO quedan "pendiente de revisión clínica (decisión 12)". La escala de Cohen que usa `ETIQUETA_V_RESULTADO` la cita la justificación de `proxy_hospital_alto` en el protocolo. |
| COD_HOSPITAL como predictor | Sin hallazgo. Es `referencia`, solo entra a la medida de proxy. El L1 admite condicionales cercanas a proxy (ESPECIALIDAD_MEDICA, SERVICIOINGRESO) y lo declara con `clase_proxy`, en F11_05 y en el informe. |
| IR_29301_MORTALIDAD como candidata | Sin hallazgo. Es `solo_fuga`: fuera de ranking, estabilidad, proxy, importancia y modelos; aparece en `t23_fuga.csv` con `prohibida_siempre` y en la matriz con R_FASE1. |
| Medida de asociación, categóricas, regla de decisión, partición temporal | Definidas (9.1, 7.5, 10.9, sección 6). Los hallazgos 1 y 2 son de evidencia o de orden incompletos dentro de esas piezas, no de falta de definición de las piezas. |
| Tope de muestreo reproducible y ruta de modo completo | Presentes (7.6, 7.7, 14.5), con el oráculo `default_rng(semilla + anio)` e índices ordenados. |
| Tablas y figuras exigidas | Cubiertas: las 15 tablas con nombre en el encargo más 4 de apoyo, y F10_01 a F11_06 con sus CSV y los valores de `decision` del plan. |
| Estándar de pruebas | No es vacuo: cuentas a mano (14 recalculadas, todas correctas), oráculos de scipy y sklearn, bordes de umbral con igualdad, integración sintética, ruta de modo completo y 51 mutantes. |

## Hallazgos

### 1. MEDIUM. La cardinalidad de ESPECIALIDAD_MEDICA por hospital (H02b) se calcula sobre la variable ya recortada a 60 niveles

Dónde: 10.4 (fila `cardinalidad_especialidad`), 7.5 (tope de 60 niveles para toda categórica que no sea la referencia), 10.10 (`espec_n_niveles_totales`, `espec_n_niveles_en_un_hospital`), 10.12 (sección 4 del informe) y la guía de H02b de la sección 20.

Problema: el encargo (6.4) pide "la cardinalidad de ESPECIALIDAD_MEDICA por hospital (H02b)". El diseño la cuenta como "número de niveles reales de ESPECIALIDAD_MEDICA (sin OTROS ni '(sin dato)')" de la columna del panel, que por 7.5 ya conserva solo los 60 más frecuentes. Entonces `n_niveles_totales` nunca pasa de 60 y los niveles raros, que son los que con más frecuencia pertenecen a un solo hospital, desaparecen en OTROS. Medido en los extractos (cohorte D de 2019 a 2022, solo conteos, 62 hospitales con al menos 100 filas):

- sin tope: 76 niveles, 13 usados por un solo hospital, cardinalidad por hospital de 3 a 39 (mediana 16);
- con el tope: 60 niveles, 5 usados por un solo hospital (62 % menos), cardinalidad de 3 a 38.

La cifra que el informe presentaría como "niveles que identifican a un solo hospital" quedaría en 5 en vez de 13, y `n_niveles_totales = 60` contradice los 89 niveles que la sección 18 del propio diseño cita para esta variable.

Corrección:
```text
# panel.construir_panel, antes de aplicar 7.5, sobre las mismas filas D muestreadas de la ventana de ranking
esp = crudo["ESPECIALIDAD_MEDICA"].str.strip_chars()        # texto sin tope y sin OTROS
Panel.cardinalidad_especialidad = {n_hospitales, card_minima, card_mediana, card_maxima,
                                   n_niveles_totales, n_niveles_en_un_hospital}   # solo agregados, con MIN_FILAS_HOSPITAL
```
`sospechas.tabla_proxy` copia esos agregados a la fila `cardinalidad_especialidad`; las filas de variable siguen usando la columna recortada. Prueba: un panel artificial con 70 niveles de especialidad, de los cuales 3 fuera de los 60 más frecuentes pertenecen a un solo hospital, da `n_niveles_totales = 70` y `n_niveles_en_un_hospital` los cuenta. Mutante nuevo M52: calcular la cardinalidad sobre la columna recortada.

### 2. MEDIUM. El índice de complicación probable no define desempates, así que `rango_indice`, `en_top5` y `elix_top5` quedan indeterminados, y el reescalado divide por cero con una sola categoría

Dónde: 10.5 (viñetas "Índice de complicación probable" y "Comparaciones"; columnas `indice_complicacion`, `rango_indice`, `en_top5`), 10.10 (`elix_top5`, `elix_n_coinciden_fase1`, `elix_n_coinciden_repo`), 14.3 (prueba del índice con cuatro categorías) y 14.8 (DT11).

Problema: el índice es el promedio de dos rangos (con rango promedio en empates), de modo que los empates son la regla. Simulé 31 categorías con dos vectores de rangos (20.000 repeticiones por cada correlación de 0; 0,5; 0,8 y 0,95): hubo al menos un empate en el 100 % de las corridas y un empate justo entre la quinta y la sexta categoría, o sea en el corte del top 5, entre el 21 % y el 24 % de ellas. El texto solo dice que `rango_indice` vale 1 para la más probable y que `en_top5` marca "las cinco de menor `rango_indice`". Sin regla de desempate, el conjunto que se compara con la lista de la Fase 1 (el entregable de H08 y de las cifras que pide el mensaje final del encargo) depende de la implementación: un rango "mínimo" marca 4, 6 o más categorías, y `DataFrame.sort` de polars no promete estabilidad sin `maintain_order=True`, lo que choca con DT11. Además `1 - (promedio_de_rangos - 1) / (m - 1)` divide por cero si solo una categoría tiene índice, caso posible con datos chicos (el generador sintético tiene 19 categorías no constantes y pocos expuestos en los tramos extremos).

Corrección:
```text
rango_indice = posición 1..m al ordenar por (indice_complicacion desc, rango de or_larga_bajo asc, rango de razon_bajo asc, elix_id asc)   # orden total
en_top5      = "si" si rango_indice <= 5 (exactamente min(5, m) categorías); "no" en las demás con índice; nulo sin índice
m < 2        -> indice_complicacion, rango_indice y en_top5 nulos; elix_top5 vacío
elix_top5    = elix_id de las en_top5 en orden de rango_indice, separados por "; "
```
Pruebas: tres categorías empatadas en las posiciones 4 a 6 dan exactamente cinco `en_top5` y la misma selección en dos corridas; m = 1 y m = 0 no lanzan excepción. Mutante nuevo M53: rango "mínimo" sin desempate.

### 3. NIT. `redondear_publicado` no fija la función de redondeo y el corte del dendrograma usa distancias sin redondear

Dónde: 8.1 (`redondear_publicado` y la regla transversal), 9.3 (`fcluster(Z, t=1 - redundancia_abs)`), 13 (invariante "los umbrales se comparan con el valor publicado"), 14.3.

Problema: el diseño escribe `round(x, 6)` "igual que `_limpiar_numericos`". `round` de Python y `np.round` o polars difieren en los empates decimales exactos: medí 97.756 diferencias en 200.000 valores de la forma k + 0,5e-6, y `round(0.7999995, 6)` da 0.799999 mientras `np.round` y `pl.Series.round` dan 0.8 (el CSV lo escribe polars). Con una implementación en Python puro tabla y decisión discreparían justo en el caso que la regla quiere evitar. Además las distancias del dendrograma usan `1 - asociacion_abs` sin redondear: un par en 0.7999999999999999 es arista del componente (publica 0,8) pero `fcluster` con `t = 0.19999999999999996` no lo une (verificado con scipy 1,18,1).

Corrección: `redondear_publicado(x) = np.round(x, 6)` (acepta escalares y arreglos); casos nuevos en 14.3: 0.7999995 pasa a 0.8 y 0.8499995 a 0.85; distancia del dendrograma `1 - redondear_publicado(A)` y corte en `1 - u["redundancia_abs"] + 1e-12`; prueba con un par en 0.7999999999999999 que es arista del componente y queda en el mismo grupo.

### 4. NIT. El AUC citado para IR_29301_SEVERIDAD en el generador es el ordinal; con el puntaje categórico que fija el diseño da otro valor

Dónde: 10.7 (último punto: "llega a 0,79"), 14.2 punto 1 ("0,7937") y 18.

Problema: 0,7937 es el AUC del valor numérico 0 a 3. El diseño trata la variable como categórica (tasa suavizada por nivel) y en el generador los niveles no son monótonos (tasas por nivel de 0, 0, 0,295 y 0,233 en 2019 a 2023). Con el método del diseño medí 0,8312 en 2019-2022 (en muestra) y 0,7837 en 2023. La conclusión "no supera 0,85" se mantiene, pero con un margen de 0,019 y no de 0,056, y una prueba fijada en 0,79 fallaría. El AUC de IR_29301_MORTALIDAD (0,9222) sí coincide porque sus niveles son monótonos.

Corrección: reescribir 10.7, 14.2 y 18 con "0,83 en el ranking (puntaje categórico en muestra) y 0,78 en 2023; FUGA usa el máximo, 0,83, bajo 0,85" y probar `0.80 < auc_ranking < 0.85` en lugar de un valor puntual.

### 5. NIT. `t23_parametros_seleccion.csv` no lista `EPS_COLINEALIDAD` ni `DECIMALES_PUBLICADOS`, aunque 8.1 dice que todas las constantes se publican

Dónde: 8.1 (párrafo de constantes: "Todas se publican en `t23_parametros_seleccion.csv`"), sección 4, 10.10 (lista de filas) y 14.3 (prueba de `tabla_parametros`).

Corrección: agregar las filas `EPS_COLINEALIDAD` (estadistica; criterio de colinealidad exacta del encargo, R2 mayor que 1 menos 1e-9) y `DECIMALES_PUBLICADOS` (estadistica; decimales con que se comparan los umbrales), o quitar la frase de 8.1. La prueba de `tabla_parametros()` debe exigir una fila por cada constante enumerada en la sección 4.

### 6. NIT. La escritura por bloques se presenta como una forma de cumplir la regla de privacidad del encargo, que manda separar en varias tablas, y no se declara como desviación

Dónde: 11 ("Hay dos formas de cumplirlo"), 9.3, 10.4, 10.6, 10.7 y el encargo, sección 1 (Privacidad).

Problema: el encargo dice "Si una tabla mezcla conteos de bases distintas, sepárala en varias tablas", y a la vez pide la asociación de 2023 "en columna aparte" dentro de `pares_redundantes.csv`: las dos instrucciones chocan y el diseño las concilia con `suprimir_bloques`. Es razonable (el efecto buscado, no anular cifras ajenas, se cumple y todos los conteos quedan en `cols_n` para el escáner), pero la sección 11 no dice que reemplaza la regla literal, y las celdas anuladas en el paso previo no entran en `celdas_suprimidas` del índice de tablas (11 lo admite), de modo que la auditoría subinforma.

Corrección: agregar a la sección 11 una frase de desviación declarada (qué regla del encargo se reemplaza, por qué y qué la compensa) y hacer que `escribir_por_bloques` sume las celdas anuladas en el paso previo a `celdas_suprimidas` de la entrada de la tabla en el índice.

### 7. NIT. Un conteo de faltantes de 1 a 9 se deduce de dos columnas publicadas aunque `prop_faltante` se anule

Dónde: 10.2 (`n_filas_ventana`, `n`, `prop_faltante`) y 11 (caso especial 1).

Problema: para numéricas y binarias, `n_filas_ventana - n` es el conteo de faltantes. El diseño anula `prop_faltante` cuando ese conteo está entre 1 y 9, pero deja publicadas las dos columnas que lo revelan. En los extractos ninguna variable numérica o binaria tiene tan pocos faltantes (por eso es NIT), pero el propósito declarado no se logra.

Corrección: agregar `n_faltante` a la tabla y un tercer bloque de `escribir_por_bloques` con `cols_n = ["n_faltante"]` y `dependientes = ["n", "prop_faltante", "clase_faltante", "faltante_informativo", "or_faltante", "or_faltante_bajo", "or_faltante_alto"]`; prueba con una variable de 5 nulos en 100.000 filas.

### 8. NIT. DT11 no tiene prueba automática, el hilo único de BLAS solo se fija en `modelos` y los pliegues del L1 dependen del orden de las filas

Dónde: 10.1 (`threadpool_limits` y "`cv=3` con `StratifiedKFold` sin barajar es determinista"), 10.5 (`regresion_logistica` en Elixhauser), 14.4 y 14.8.

Corrección: (a) en la integración de 14.4 correr una segunda vez en otra carpeta y comparar `sha256` de `tablas_indice.csv` y `sha256_png` y `sha256_csv` de `figuras_indice.csv`. (b) Poner `threadpool_limits(limits=1)` dentro de `regresion_logistica` y del producto matricial del VIF, para cubrir también `sensibilidad_elix`. (c) Pasar `cv=StratifiedKFold(PLIEGUES_CV, shuffle=True, random_state=semilla)` a `LogisticRegressionCV`: las filas del panel van por año y por posición en el archivo, así que un `StratifiedKFold` sin barajar arma bloques contiguos (años o grupos de hospitales, según el orden del archivo) y el C elegido cambiaría con el orden de los datos completos. Sigue siendo el "equivalente determinista" que admite el encargo.

### 9. NIT. El L1 no registra si liblinear convergió ni qué se hace con `ConvergenceWarning`

Dónde: 10.1 (`ResultadoL1.estado` solo "ok" o "no evaluable: pocos eventos"; `catch_warnings` solo para `FutureWarning`) y 12.2.

Corrección: con unas 450 columnas, varias colineales (los agregados de Elixhauser son combinaciones exactas de las 31 categorías) y `max_iter=200`, algún C puede agotar iteraciones. Agregar `convergio: bool` a `ResultadoL1` (falso si algún elemento de `modelo.n_iter_` es igual a `max_iter`), la columna `l1_convergio` en la fila `MODELO_L1` de `t23_importancia.csv`, capturar `ConvergenceWarning` junto con `FutureWarning` dentro de `ajustar_l1` y probarlo con `max_iter=1`.

### 10. NIT. 14.6 dice a la vez que se aplican las 51 mutaciones y que el mínimo es de 15 a 20 priorizando 31

Dónde: 14.6, párrafo previo a la tabla.

Corrección: elegir una regla, por ejemplo "se aplican M01 a M51; si el tiempo no alcanza, como mínimo las 31 priorizadas, que superan el mínimo de 15 a 20 del encargo", y documentar los mutantes equivalentes que sobrevivan.

### 11. NIT. 10.9 no dice de dónde sale `faltante_max` para ETNIA y PUEBLO_ORIGINARIO, y `medicion_completa` no exige proxy ni PSI

Dónde: 10.9 (entradas por variable y "Medición completa").

Corrección: (a) ETNIA y PUEBLO_ORIGINARIO son del panel y "parcial", pero no tienen fila en `t23_ranking_variables.csv` (no analizan el resultado): declarar que su `faltante_max` es `faltante_max_t22` con `fuente_faltante = "t22"`. (b) Para toda candidata del panel agregar a `medicion_completa` que existan los valores reales de `asociacion_hospital` y `psi_vs_2023`, para que MANTENER no salga de una variable cuyo proxy o deriva no se pudo medir (hoy casi siempre coincide con `estado_medicion` distinto de "ok", pero no está garantizado).

### 12. NIT. Dos detalles del texto se contradicen: nivel base de SEXO y puntaje de un nivel nuevo en 2023

Dónde: 10.1 (modelo base "contra MUJER" frente a `omitir_nivel_base` "el más frecuente del entrenamiento") y 7.5 (nivel nuevo en la confirmación).

Corrección: (a) fijar MUJER como nivel base de SEXO (excepción explícita a `omitir_nivel_base`), para que el modelo base, el OR de Elixhauser y sus oráculos coincidan aunque HOMBRE fuera el nivel más frecuente. (b) 7.5 dice que un nivel nuevo en 2023 "recibe la tasa global", pero lo recodifica a OTROS, que tiene tasa propia cuando la variable pasa de 60 niveles: escribir "recibe la tasa de OTROS si OTROS existe en la ventana de ranking y la tasa global si no".

## Supuestos verificados

1. Umbrales: `umbrales.toml` versión 1,4 contiene todos los que el diseño lee (incluidos `ventana_historial_dias`, `z_95`, `winsor`, `id_quiebre_ratio`, `nivel_confianza`) con los valores de la prueba `u` de 14.1; `[particion]` trae `anios_resultado`, `anios_ranking`, `anio_confirmacion` y `anio_expuesto`; `semilla` está en `[meta]` y `ctx.semilla` la lee.
2. Cortafuegos: `propositos_permitidos = [estructura, calidad]` solo rige para 2024; los años no expuestos admiten `seleccion` y se registran; `violaciones()` solo mira 2024. `Fuente.anios()`, `Contexto.crear(fuente=, umbrales=)`, `escribir_tabla(tarea: str | int)`, `registrar_metrica(valor=None, conteo, alcance, nivel)` y `leer_tabla` existen como los usa el diseño.
3. `supresion.suprimir` anula la celda pequeña y las dependientes de la fila; `escribir_tabla` redondea a seis decimales con polars; `privacidad.escanear_salidas` revisa solo las columnas registradas en `cols_n`; el escáner de números largos no marca textos como "2019-2022".
4. `cohorte`: `COLS_BASE` no trae el identificador; `tipar`, `agregar_comorbilidad` (solo MDC 14) y `marcar` (D y E) tienen las firmas y salidas que usa el diseño; `ing`, `id_num` y `alta` solo salen con CIP_ENCRIPTADO, por eso el panel calcula `ing` desde `crudo`. Cohorte D real: 115.310 filas y 6.182 eventos en 2019-2022, 28.040 y 1.144 en 2023 (coincide con el diseño).
5. `elixhauser.flags_desde_mascara` devuelve `SCORE_VANWALRAVEN` (no `PUNTAJE_VAN_WALRAVEN`, como dice el encargo); los agregados se calculan después de la jerarquía (identidad con diferencia 0 sobre el generador); las cinco sospechosas son ELIX_04, 22, 24, 25 y 26.
6. `hospitales`: `regimenes_de(ids_por_anio, u)`, `historial_regimen` (previo estrictamente anterior y 365 días inclusive), `normalizar_nombre`, `TEXTO_ENTRANTE`, y las tablas `t23_procedencia_autorref.csv` y `t23_historial_factibilidad.csv` que cita la columna `evidencia`.
7. `graficos`: `mapa_calor` devuelve `None` hoy y sus 8 llamadas (en `calidad`, `calidad_catalogos`, `recoleccion`, `descripcion` y `comorbilidad`) ignoran el retorno; `guardar_figura` se llama con argumentos nombrados y `tarea` entera; `nueva_figura_multiple` pasa `width_ratios`; `mapa_calor` anota solo hasta 400 celdas; `validar_plan_figuras` solo exige columnas y valores no vacíos, así que cambiar la tabla de F10_02 es compatible.
8. `informes`: formatos `entero`, `dec1` a `dec4`, `pct1`, `pct2`, `prop1`, `prop2`, `texto`; la leyenda de figura se escanea como prosa y el texto alternativo no; `descripcion.Cifra_umbral` existe. Los títulos de figura del diseño solo llevan años y códigos.
9. Config: `arbol_decision.csv` trae los nueve códigos en el orden que el diseño exige; `plan_figuras.csv` trae F10_01 a F11_06 con las decisiones citadas; `conocimiento_clinico.csv` tiene 57 filas (24 verificadas, solo Elixhauser con peso, el puntaje y Spiegelhalter) y cita `t23_fuga.csv`, `t23_proxy_hospital.csv` y `t23_sensibilidad_elix.csv`; las hipótesis H01c, H02a, H02b, H03, H08, H14, H17a y S09 apuntan a tablas que el diseño produce.
10. Documentos: el diccionario tiene 129 variables, iguales a `esquema.COLUMNAS`, todas con rol; solo ESPECIALIDAD_MEDICA, SERVICIOINGRESO y HOSPPROCEDENCIA difieren de `reglas_variables.csv` (CONDICIONAL prevalece); las derivadas son 34 filas (62 nombres) y los prefijos de "¿Entra al modelo?" del diseño cubren los 34 valores; R2 y R6 como las cita el diseño; `fase1_spec.json` fija 2023 como calibración y 2024 como prueba.
11. Cuentas a mano recalculadas: V de Bergsma 0,490290, 0,381613, 0,459457, 0, 1 y NaN en las tablas casi vacías; Yates de scipy 0,475; Hanley y McNeil 0,044502 y 0,060087; Woolf 4,75 (2,128388 a 10,600746) y 0,081908 (0,004406 a 1,522707 con Haldane); eta cuadrado 54/58; Spearman 0,8; AUC 0,75 y 0,5; VIF 5,263158; alturas de enlace 0,8, 0,7 y 0,9; cadena 0,90, 0,85 y 0,50 con alturas 0,10 y 0,325; tramos (8, 0, 0, 12) y (5, 5, 0, 10) con `percentil_de_conteos` real; la V corregida nunca superó a la sin corregir en 29.944 tablas aleatorias.
12. scikit-learn 1,9,1: `LogisticRegressionCV(l1_ratios=(1.0,), solver="liblinear")` con matriz dispersa da el mismo C y los mismos coeficientes que `penalty="l1"` (1 `FutureWarning` frente a 3); `LogisticRegression(C=np.inf)` coincide con `penalty=None`; `threadpoolctl` 3,7,0 está fijado en `requirements.txt`; polars 2,0,0 cumple `dt.weekday()` de 1 a 7, `dt.epoch("d")` y `replace_strict(default=)`.
13. Generador sintético (n = 700, semilla 29): cohorte D de 421, 442, 429, 440 y 415 filas con 49, 58, 47, 66 y 46 fallecidos; 12 categorías de Elixhauser constantes en 2019-2022 y en 2023 (la misma lista del diseño); 280 expuestos como máximo; AUC de IR_29301_MORTALIDAD 0,9222 y de IR_29301_PESO 0,7453; USOSPABELLON vale "1" en el 31 % de la cohorte D.
14. Extractos (salidas de la Tarea 4): TIPO_PROCEDENCIA 19 valores, ESPECIALIDAD_MEDICA 89, SERVICIOINGRESO 86, NACIONALIDAD 95, ETNIA 13 y COD_HOSPITAL 68; los valores truncados de "OTRAS INSTITUCIONES" y los demás que el diseño deja en "(sin mapeo)" aparecen en `t22_top_valores.csv`.
15. La suite existente tiene 286 pruebas y todas pasan; ninguna prueba fija la lista de módulos de `eda/`, así que los siete módulos nuevos no la rompen (`ejecucion.archivos_codigo` los recoge solos).

## Supuestos no verificados o incorrectos

Incorrectos:
1. 10.4: que los "niveles reales" de la columna recortada sirvan como cardinalidad de ESPECIALIDAD_MEDICA por hospital. Con los extractos da 60 y 5 en vez de 76 y 13 (hallazgo 1).
2. 10.5: que el promedio de dos rangos entregue un orden total para formar el top 5. Hay empates en el corte en 21 % a 24 % de los casos simulados (hallazgo 2).
3. 8.1: que `round(x, 6)` sea "igual que `_limpiar_numericos`" en cualquier implementación (hallazgo 3).
4. 10.7, 14.2 y 18: que el AUC de IR_29301_SEVERIDAD del generador sea 0,7937 con el puntaje categórico del diseño; es 0,8312 en el ranking y 0,7837 en 2023 (hallazgo 4).
5. 8.1: que todas las constantes técnicas queden en `t23_parametros_seleccion.csv`; faltan dos (hallazgo 5).

No verificables hasta que exista el código (el diseño ya los reconoce en 14.4, 14.7 y 17):
- Tiempo de la Tarea 11 con el L1 de validación cruzada y los incrementos, y memoria del L1 y de la lectura en modo completo.
- Legibilidad real de los PNG (F10_01 con 59 etiquetas de 6,5 pt) y el comportamiento de `etiquetas_sin_solape` con el gancho `antes_de_guardar` bajo `layout="constrained"`.
- El modo completo con datos reales y el historial en datos reales (en extractos es no evaluable por construcción).
- Que la bandera de separación por coeficiente se active en el panel artificial con una cuarta parte de las filas en una columna sin eventos (el diseño midió 15 % en una simulación).
