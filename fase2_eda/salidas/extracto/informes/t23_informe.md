# Informe de exploración de los datos (CRISP-DM 2.3)

**Fuente de los datos: extracto.** Los extractos son una fracción menor de los datos y no son una muestra aleatoria simple: sirven para desarrollar y probar, no para conteos absolutos.

Este informe reúne la exploración de los datos: la cohorte, el resultado y la estadía (Tarea 7), la comorbilidad (Tarea 8) y los hospitales (Tarea 9).

## 1. Cohorte y cascada de exclusiones

La cascada resta, en orden, los egresos que quedan fuera de cada variante de cohorte. Cada paso excluye solo lo que queda del anterior, así que la cohorte final y las exclusiones suman el total en cada año. Los pasos EX01 a EX03 no miran el desenlace: la cohorte dura A no depende de quién falleció. Entre 2019 y 2023 se midieron 249.995 egresos.

![Barras horizontales en cascada: del total de egresos se restan los GRD no agrupables, los recién nacidos y los obstétricos sin comorbilidad para llegar a la cohorte dura A; luego los menores, la actividad no hospitalaria, los censurados, los fallecidos y las estadías de 0 días para llegar a las variantes B, C, D y E.](../figuras/F07_01.png)

*Figura F07_01. Cascada de exclusiones y variantes de cohorte (2019-2023). Datos de origen: `tablas/t23_cascada.csv`.*

| Paso | Descripción | Egresos | % del total | Usa el desenlace |
|---|---|---|---|---|
| TOTAL | Egresos leídos | 249.995 | 100,00 % | no |
| EX01 | GRD no agrupable: sin código, centinela, formato inválido o MDC 99 | 157 | 0,06 % | no |
| EX02 | Recién nacidos (MDC 15) | 6.687 | 2,67 % | no |
| EX03 | Obstétricos sin comorbilidades (MDC 14 sin categorías de Elixhauser) | 19.777 | 7,91 % | no |
| A | Variante A: cohorte dura | 223.374 | 89,35 % | no |
| EX04 | Menores de 18 años o con edad desconocida | 36.862 | 14,75 % | no |
| B | Variante B: adultos | 186.512 | 74,61 % | no |
| EX05 | Actividad distinta de hospitalización | 31.253 | 12,50 % | no |
| C | Variante C: adultos hospitalizados | 155.259 | 62,10 % | no |
| EX06 | Censurados por tipo de alta o sin dato de desenlace | 11.909 | 4,76 % | si |
| D | Variante D: cohorte del modelo de mortalidad | 143.350 | 57,34 % | si |
| EX07 | Fallecidos (salen del modelo de estadía) | 7.326 | 2,93 % | si |
| EX08 | Sobrevivientes con estadía de 0 días o desconocida | 5.677 | 2,27 % | si |
| E | Variante E: cohorte del modelo de estadía | 130.347 | 52,14 % | si |

*Cascada de 2019 a 2023.* *Tabla completa en `tablas/t23_cascada.csv`*

Los GRD no agrupables (EX01) suman 157 egresos (0,06 %); los recién nacidos (EX02), 6.687 (2,7 %); y los obstétricos sin comorbilidades (EX03), 19.777 (7,9 %). La cohorte dura A queda con 223.374 egresos (89,4 %). Los menores de 18 años o con edad desconocida suman 36.862, la actividad distinta de hospitalización 31.253 y los censurados por tipo de alta 11.909.

Las exclusiones que dependen del desenlace (EX06 a EX08) solo existen para 2019 a 2023, y el año expuesto 2024 aporta únicamente los pasos que no lo usan. El intento de leer el desenlace de 2024 fue rechazado por el cortafuegos y quedó en el registro de accesos.

## 2. Prevalencia de la mortalidad por variante de cohorte

La decisión 5 limita la población a adultos y la decisión 15 a la actividad de hospitalización. La prevalencia se informa de dos formas: sobre todos los egresos de la variante y sobre los egresos con desenlace conocido (fallecidos más sobrevivientes, sin censurados). Al pasar de A a B la prevalencia con desenlace conocido cambia de 3,63 % a 4,31 %, y llega a 5,11 % en la variante C. En 2023 la variante C tiene 3,74 % sobre todos sus egresos y 4,08 % con desenlace conocido.

| Período | Variante | Egresos | Fallecidos | Censurados | Prevalencia (todos) | Prevalencia (conocido) | IC bajo | IC alto | Cambio (pp) |
|---|---|---|---|---|---|---|---|---|---|
| 2019-2023 | A | 223.374 | 7.615 | 13.878 | 3,41 % | 3,63 % | 3,56 % | 3,72 % | n/d |
| 2019-2023 | B | 186.512 | 7.497 | 12.558 | 4,02 % | 4,31 % | 4,22 % | 4,41 % | 0,67 |
| 2019-2023 | C | 155.259 | 7.326 | 11.909 | 4,72 % | 5,11 % | 5,00 % | 5,23 % | 0,80 |
| 2019-2023 | D | 143.350 | 7.326 | 0 | 5,11 % | 5,11 % | 5,00 % | 5,23 % | 0,00 |
| 2019-2023 | E | 130.347 | 0 | 0 | n/d | n/d | n/d | n/d | n/d |
| 2023 | A | 45.555 | 1.165 | 2.870 | 2,56 % | 2,73 % | 2,58 % | 2,89 % | n/d |
| 2023 | B | 37.810 | 1.144 | 2.548 | 3,03 % | 3,24 % | 3,06 % | 3,43 % | 0,51 |
| 2023 | C | 30.571 | 1.144 | 2.531 | 3,74 % | 4,08 % | 3,85 % | 4,32 % | 0,84 |
| 2023 | D | 28.040 | 1.144 | 0 | 4,08 % | 4,08 % | 3,85 % | 4,32 % | 0,00 |
| 2023 | E | 25.735 | 0 | 0 | n/d | n/d | n/d | n/d | n/d |

*Variantes de cohorte; intervalos de Wilson indicativos.* *Tabla completa en `tablas/t23_cascada_variantes.csv`*

La definición del pipeline se reproduce de forma aproximada (mapeo de comorbilidades de tres caracteres y exclusión de obstétricos solo si sobreviven), porque el código original no está en el repositorio revisado. Su cascada se compara con las cifras documentadas en el cierre.

| Paso | Descripción | Egresos | % del total |
|---|---|---|---|
| TOTAL | Egresos leídos | 249.995 | 100,00 % |
| EX01 | GRD no agrupable | 157 | 0,06 % |
| EX02 | Recién nacidos (MDC 15) | 6.687 | 2,67 % |
| EX03_SIN_DESENLACE | MDC 14 sin comorbilidades con mapeo de tres caracteres, sin mirar el desenlace (cota superior de EX03) | 20.008 | 8,00 % |
| AMBULATORIA | Actividad distinta de hospitalización (se solapa con las demás) | 36.593 | 14,64 % |
| EX03_PIPELINE | Obstétricos sin comorbilidades, solo sobrevivientes (mapeo de tres caracteres) | 19.899 | 7,96 % |
| COHORTE_DURA_PIPELINE | Cohorte dura con la definición del pipeline | 223.252 | 89,30 % |
| ESTANCIA_CERO | Estadía de 0 días (se solapa con las demás) | 43.354 | 17,34 % |
| FALLECIDOS | Fallecidos (se solapan con las demás) | 7.748 | 3,10 % |
| CENSURADOS | Censurados por tipo de alta | 14.450 | 5,78 % |
| COHORTE_ESTANCIA_PIPELINE | Cohorte dura del pipeline sin estadía de 0 días, actividad ambulatoria ni fallecidos | 169.595 | 67,84 % |
| COHORTE_ESTANCIA_PIPELINE_SIN_CENSURADOS | Lectura alternativa: la cohorte anterior sin censurados | 157.109 | 62,84 % |

*Cascada con la definición aproximada del pipeline.* *Tabla completa en `tablas/t23_cascada_pipeline.csv`*

## 3. Desenlace: tipo de alta y mortalidad

El rol de cada tipo de alta viene de la decisión de la Fase 1 (archivo de roles): los fallecidos son el evento, el alta a domicilio, la voluntaria y la fuga son no evento, y los traslados y la hospitalización domiciliaria son censurados. Las etiquetas cortadas se asignan por prefijo.

| Tipo de alta | Rol | Egresos | % del total |
|---|---|---|---|
| DOMICILIO | NO_EVENTO | 224.254 | 89,70 % |
| FALLECIDO | EVENTO | 7.748 | 3,10 % |
| HOSPITALIZACIÓN DOMICILIARIA | CENSURADO | 5.520 | 2,21 % |
| DERIVACIÓN OTRO HOSPITAL DEL SERVICIO | CENSURADO | 4.910 | 1,96 % |
| ALTA VOLUNTARIA | NO_EVENTO | 2.681 | 1,07 % |
| DERIVACIÓN OTRO HOSPITAL DE LA RED NACIONAL | CENSURADO | 1.888 | 0,76 % |
| DERIVACIÓN A OTROS CENTROS (CÁRCEL, HOGAR) | CENSURADO | 1.018 | 0,41 % |
| FUGA DEL PACIENTE | NO_EVENTO | 862 | 0,34 % |
| DERIVACIÓN INST. PRIVADA (COMPRA DE SERVICIOS) | CENSURADO | 779 | 0,31 % |
| DERIVACIÓN INST. PRIVADA (VOLUNTARIO) | CENSURADO | 330 | 0,13 % |
| NO IDENTIFICADA | CENSURADO | <10 | n/d |

*Tipo de alta de 2019 a 2023.* *Tabla completa en `tablas/t23_tipoalta.csv`*

La mortalidad cruda sobre todos los egresos fue 2,54 % en 2019, 3,65 % en 2020, 4,01 % en 2021, 2,93 % en 2022, 2,37 % en 2023. La mortalidad agrupada de 2020-2022 es 1,39 veces la de 2019 (IC 1,31 a 1,48), evidencia para la decisión 17. Un año se marca como alza o baja cuando el intervalo de la razón contra el año previo excluye la igualdad y el cambio relativo alcanza 20,0 %.

![Dos series de puntos con barras de error por año: mortalidad sobre todos los egresos y sobre la cohorte del modelo de mortalidad.](../figuras/F07_02.png)

*Figura F07_02. Mortalidad cruda por año con intervalo de confianza. Datos de origen: `tablas/t23_resultado_anio.csv`.*

| Año | Base | Egresos | Fallecidos | Mortalidad | IC bajo | IC alto | Razón vs año previo | Cambio anual |
|---|---|---|---|---|---|---|---|---|
| 2019 | todos | 49.999 | 1.270 | 2,54 % | 2,41 % | 2,68 % | n/d | n/d |
| 2019 | D | 26.688 | 1.059 | 3,97 % | 3,74 % | 4,21 % | n/d | n/d |
| 2020 | todos | 49.999 | 1.824 | 3,65 % | 3,49 % | 3,82 % | 1,44 | alza |
| 2020 | D | 29.880 | 1.767 | 5,91 % | 5,65 % | 6,19 % | 1,49 | alza |
| 2021 | todos | 49.999 | 2.003 | 4,01 % | 3,84 % | 4,18 % | 1,10 | no |
| 2021 | D | 29.972 | 1.939 | 6,47 % | 6,20 % | 6,75 % | 1,09 | no |
| 2022 | todos | 49.999 | 1.466 | 2,93 % | 2,79 % | 3,08 % | 0,73 | baja |
| 2022 | D | 28.770 | 1.417 | 4,93 % | 4,68 % | 5,18 % | 0,76 | baja |
| 2023 | todos | 49.999 | 1.185 | 2,37 % | 2,24 % | 2,51 % | 0,81 | no |
| 2023 | D | 28.040 | 1.144 | 4,08 % | 3,85 % | 4,32 % | 0,83 | no |

*Mortalidad cruda por año; base D es la cohorte del modelo de mortalidad.* *Tabla completa en `tablas/t23_resultado_anio.csv`*

La serie mensual compara cada mes con el mismo mes del año anterior, lo que controla la estacionalidad. De los 48 meses con referencia, 11 muestran un alza relevante y 11 una baja relevante. Los intervalos no se ajustan por comparaciones múltiples y son indicativos.

![Dos paneles con el mismo eje de meses: arriba los egresos por mes y abajo la mortalidad mensual con su intervalo de confianza; los triángulos marcan los meses cuya mortalidad cambia de forma relevante frente al mismo mes del año anterior.](../figuras/F07_04.png)

*Figura F07_04. Egresos y mortalidad por mes con cambios relevantes. Datos de origen: `tablas/t23_serie_mensual.csv`.*

La mortalidad y la estadía también cambian con la edad y el sexo, y la cohorte dura incluye a los menores hasta que la decisión 5 los saque.

![Dos paneles por grupo de edad, separados por sexo y período: a la izquierda la mortalidad con escala logarítmica y a la derecha la estadía media de los sobrevivientes.](../figuras/F07_05.png)

*Figura F07_05. Mortalidad y estadía por grupo de edad y sexo. Datos de origen: `tablas/t23_resultado_edad_sexo.csv`.*

## 4. Estadía de los sobrevivientes

En 2019 a 2022 la estadía de los sobrevivientes de la variante E tiene mediana 4 días, media 7,4 y varianza 166,6. El índice de dispersión (varianza sobre media) es 22,5; un valor mayor que uno indica sobredispersión frente a la distribución de Poisson, evidencia para elegir entre Poisson y Tweedie en la decisión 9. En 2023 el índice es 14,8.

![Barras por tramo de días de estadía, con escala logarítmica en el eje vertical, para los sobrevivientes de la cohorte del modelo de mortalidad en 2019 a 2022 y en 2023.](../figuras/F07_03.png)

*Figura F07_03. Histograma logarítmico de la estadía de los sobrevivientes. Datos de origen: `tablas/t23_estancia_hist.csv`.*

| Período | Sobrevivientes | Media | Varianza | Índice de dispersión | Mediana | p90 | p99 | Máximo |
|---|---|---|---|---|---|---|---|---|
| 2019 | 24.655 | 7,2 | 303,4 | 42,3 | 4 | 16 | 49 | 2.211 |
| 2020 | 26.884 | 7,4 | 128,5 | 17,4 | 4 | 16 | 50 | 450 |
| 2021 | 26.918 | 7,7 | 123,7 | 16,0 | 4 | 17 | 53 | 371 |
| 2022 | 26.155 | 7,3 | 120,9 | 16,6 | 4 | 16 | 51 | 288 |
| 2023 | 25.735 | 6,9 | 102,7 | 14,8 | 4 | 15 | 48 | 278 |
| 2019-2022 | 104.612 | 7,4 | 166,6 | 22,5 | 4 | 16 | 51 | 2.211 |
| 2019-2023 | 130.347 | 7,3 | 154,0 | 21,1 | 4 | 16 | 50 | 2.211 |

*Estadía en días de los sobrevivientes de la variante E.* *Tabla completa en `tablas/t23_estancia_dispersion.csv`*

La decisión 7 se evalúa con 2019 a 2022 y se confirma con 2023. El corte más bajo que afecta a no más de 1,0 % de los sobrevivientes es la regla p99, con 51 días: afectaría a 0,97 % de los sobrevivientes de 2019 a 2022 y a 0,78 % de los de 2023. El máximo observado en 2019 a 2022 fue de 2.211 días.

| Regla | Corte (días) | Afectados 2019-2022 | % 2019-2022 | Cumple | Afectados 2023 | % 2023 | Cumple 2023 | Propuesta |
|---|---|---|---|---|---|---|---|---|
| p99 | 51 | 1.010 | 0,97 % | si | 201 | 0,78 % | si | si |
| p99,5 | 66 | 515 | 0,49 % | si | 90 | 0,35 % | si | no |
| p99,9 | 112 | 102 | 0,10 % | si | 25 | 0,10 % | si | no |
| 365 días | 365 | <10 | n/d | si | 0 | n/d | si | no |

*Reglas de corte de estadías largas, base E.* *Tabla completa en `tablas/t23_cola_estancia.csv`*

## 5. Estadía cero y actividad ambulatoria

De los 43.354 egresos con estadía cero en 2019 a 2023, 75,7 % son de actividad ambulatoria, 1,34 % fallecieron el mismo día y 2,73 % son censurados. Entre los 36.593 egresos de actividad ambulatoria, 89,7 % tienen estadía cero. Entre los egresos con estadía cero y desenlace conocido, la mortalidad es 1,38 %.

| Actividad | Desenlace | Egresos | Con estadía de 0 días |
|---|---|---|---|
| CIRUGÍA MAYOR AMBULATORIA (CMA) | CENSURADO | 105 | 91 |
| CIRUGÍA MAYOR AMBULATORIA (CMA) | EVENTO | <10 | <10 |
| CIRUGÍA MAYOR AMBULATORIA (CMA) | NO_EVENTO | 31.717 | 29.699 |
| HOSPITALIZACIÓN | CENSURADO | 13.754 | 898 |
| HOSPITALIZACIÓN | EVENTO | 7.575 | 515 |
| HOSPITALIZACIÓN | NO_EVENTO | 192.073 | 9.118 |
| HOSPITALIZACIÓN DIURNA | CENSURADO | 23 | 23 |
| HOSPITALIZACIÓN DIURNA | NO_EVENTO | 2.376 | 2.359 |
| HOSPITALIZACIÓN EN URGENCIA | CENSURADO | 568 | 173 |
| HOSPITALIZACIÓN EN URGENCIA | EVENTO | 169 | 64 |
| HOSPITALIZACIÓN EN URGENCIA | NO_EVENTO | 1.631 | 411 |

*Estadías cero por actividad y desenlace, 2019 a 2023.* *Tabla completa en `tablas/t23_estancia_cero.csv`*

Para la decisión 15 se compara la mortalidad de cada estrato con la del resto en adultos de la cohorte dura con desenlace conocido. Si la razón es menor que 0,25, el estrato diluye el cociente observado sobre esperado y se propone sacarlo del modelo de mortalidad. La decisión final es humana.

| Período | Variable | Estrato | Egresos | Fallecidos | Mortalidad | Razón vs resto | IC bajo | IC alto | Bajo el umbral |
|---|---|---|---|---|---|---|---|---|---|
| 2019-2022 | actividad_grupo | ambulatoria | 23.382 | 171 | 0,73 % | 0,14 | 0,12 | 0,16 | si |
| 2023 | actividad_grupo | ambulatoria | 7.222 | 0 | 0,00 % | 0,00 | 0,00 | 0,01 | si |
| 2019-2022 | estancia_cero | 0 días | 25.775 | 457 | 1,77 % | 0,34 | 0,31 | 0,37 | no |
| 2023 | estancia_cero | 0 días | 7.899 | 78 | 0,99 % | 0,25 | 0,20 | 0,32 | no |

*Estratos de la decisión 15.* *Tabla completa en `tablas/t23_resultado_estratos.csv`*

## 6. Límites

- Los extractos no son una muestra aleatoria simple: las prevalencias y las razones son indicativas y los conteos absolutos no se comparan con los datos completos.
- Los intervalos de confianza ignoran el agrupamiento por paciente y por hospital.
- La cascada del pipeline es una reproducción aproximada de su definición.
- El desenlace y la fecha de alta de 2024 no se leen: el cortafuegos lo impide y deja registro del intento.

## 7. Prevalencia de las categorías de Elixhauser

Las 31 categorías se calculan con el mapeo de Quan et al., contrastado con una segunda fuente pública, sobre los diagnósticos secundarios (regla R3) y con las jerarquías habituales: la hipertensión complicada anula a la no complicada, la diabetes complicada a la no complicada y el cáncer metastásico al tumor sólido. La base son los adultos hospitalizados. Considerando todos los años, la categoría más frecuente es Hipertensión no complicada (34,1 %), seguida de Diabetes no complicada (14,8 %) y Obesidad (13,4 %).

![Mapa de calor con el porcentaje de adultos hospitalizados que tiene cada una de las 31 categorías de Elixhauser en cada año, ordenadas de mayor a menor prevalencia.](../figuras/F08_01.png)

*Figura F08_01. Prevalencia de las 31 categorías de Elixhauser por año. Datos de origen: `tablas/t23_elix_prevalencia.csv`.*

| Rango | Categoría | Nombre | Egresos | Prevalencia | IC bajo | IC alto |
|---|---|---|---|---|---|---|
| 1 | ELIX_06 | Hipertensión no complicada | 71.053 | 34,1 % | 33,9 % | 34,3 % |
| 2 | ELIX_11 | Diabetes no complicada | 30.904 | 14,8 % | 14,7 % | 15,0 % |
| 3 | ELIX_23 | Obesidad | 27.803 | 13,4 % | 13,2 % | 13,5 % |
| 4 | ELIX_13 | Hipotiroidismo | 20.335 | 9,8 % | 9,6 % | 9,9 % |
| 5 | ELIX_14 | Insuficiencia renal | 15.786 | 7,6 % | 7,5 % | 7,7 % |
| 6 | ELIX_25 | Trastornos hidroelectrolíticos | 15.276 | 7,3 % | 7,2 % | 7,4 % |
| 7 | ELIX_10 | Enfermedad pulmonar crónica | 13.985 | 6,7 % | 6,6 % | 6,8 % |
| 8 | ELIX_02 | Arritmias cardíacas | 13.343 | 6,4 % | 6,3 % | 6,5 % |
| 9 | ELIX_01 | Insuficiencia cardíaca congestiva | 11.154 | 5,4 % | 5,3 % | 5,5 % |
| 10 | ELIX_28 | Abuso de alcohol | 9.632 | 4,6 % | 4,5 % | 4,7 % |
| 11 | ELIX_15 | Enfermedad hepática | 8.975 | 4,3 % | 4,2 % | 4,4 % |
| 12 | ELIX_12 | Diabetes complicada | 8.715 | 4,2 % | 4,1 % | 4,3 % |

*Las doce categorías más frecuentes, todos los años.* *Tabla completa en `tablas/t23_elix_prevalencia.csv`*

## 8. Mapeo completo frente al de tres caracteres

El mapeo del repositorio compara solo tres caracteres del código, así que no reconoce las categorías que Quan define con cuatro. La diabetes no complicada pasa de 14,8 % con el mapeo completo a 0,0 % con tres caracteres, y la complicada de 4,2 % a 0,0 %. De las 31 categorías, 11 difieren en más de 20,0 % de su prevalencia relativa; el supuesto S02 se rechaza: el mapeo de tres caracteres no equivale al completo.

| Categoría | Nombre | Largos de prefijo | Prevalencia completa | Prevalencia con 3 caracteres | Diferencia relativa |
|---|---|---|---|---|---|
| ELIX_11 | Diabetes no complicada | 4 | 14,84 % | 0,00 % | -100,0 % |
| ELIX_12 | Diabetes complicada | 4 | 4,19 % | 0,00 % | -100,0 % |
| ELIX_16 | Úlcera péptica sin sangrado | 4 | 0,42 % | 0,00 % | -100,0 % |
| ELIX_26 | Anemia por pérdida de sangre | 4 | 0,31 % | 0,00 % | -100,0 % |
| ELIX_27 | Anemia por deficiencia | 3 y 4 | 1,20 % | 0,27 % | -77,5 % |
| ELIX_22 | Coagulopatía | 3 y 4 | 1,79 % | 0,69 % | -61,4 % |
| ELIX_05 | Enfermedad vascular periférica | 3 y 4 | 3,48 % | 1,48 % | -57,3 % |
| ELIX_15 | Enfermedad hepática | 3 y 4 | 4,31 % | 2,16 % | -49,9 % |
| ELIX_28 | Abuso de alcohol | 3 y 4 | 4,63 % | 3,36 % | -27,4 % |
| ELIX_18 | Linfoma | 3 y 4 | 0,63 % | 0,48 % | -23,5 % |
| ELIX_02 | Arritmias cardíacas | 3 y 4 | 6,41 % | 4,96 % | -22,6 % |

*Categorías que superan el umbral de diferencia relativa.* *Tabla completa en `tablas/t23_elix_vs_3c.csv`*

## 9. Diagnóstico principal y regla R3

La regla R3 calcula las categorías solo con diagnósticos secundarios, porque el diagnóstico principal es el motivo del ingreso y no una comorbilidad previa. Si se incluyera, la categoría con mayor aumento sería Tumor sólido sin metástasis, que pasaría de 3,1 % a 6,4 % de los adultos hospitalizados (110,5 % más).

| Categoría | Nombre | Con la categoría (secundarios) | Con la categoría (principal) | Prevalencia (secundarios) | Prevalencia (con principal) | Aumento relativo |
|---|---|---|---|---|---|---|
| ELIX_20 | Tumor sólido sin metástasis | 6.377 | 9.482 | 3,06 % | 6,45 % | 110,5 % |
| ELIX_30 | Psicosis | 967 | 995 | 0,46 % | 0,88 % | 88,7 % |
| ELIX_18 | Linfoma | 1.318 | 621 | 0,63 % | 0,92 % | 45,3 % |
| ELIX_17 | VIH/SIDA | 889 | 383 | 0,43 % | 0,55 % | 29,5 % |
| ELIX_12 | Diabetes complicada | 8.715 | 2.869 | 4,19 % | 5,38 % | 28,7 % |
| ELIX_04 | Trastornos de la circulación pulmonar | 3.128 | 989 | 1,50 % | 1,93 % | 28,2 % |
| ELIX_01 | Insuficiencia cardíaca congestiva | 11.154 | 3.349 | 5,36 % | 6,68 % | 24,7 % |
| ELIX_09 | Otros trastornos neurológicos | 6.636 | 1.322 | 3,19 % | 3,73 % | 17,2 % |

*Las categorías que más cambiarían al incluir el diagnóstico principal.* *Tabla completa en `tablas/t23_elix_principal.csv`*

## 10. Asociación entre categorías

El mayor coeficiente phi entre dos categorías distintas en 2019-2022 es 0,40, entre ELIX_06 y ELIX_11. 0 de 434 pares con casos suficientes alcanzan el umbral de redundancia de 0,80; las categorías de Elixhauser se mantienen como indicadores separados y la agrupación formal de variables redundantes se hace en la selección de variables.

![Mapa de calor cuadrado con el coeficiente phi entre cada par de las 31 categorías de Elixhauser en adultos hospitalizados; la diagonal vale uno.](../figuras/F08_03.png)

*Figura F08_03. Matriz phi entre las 31 categorías de Elixhauser (2019-2022). Datos de origen: `tablas/t23_elix_phi.csv`.*

| Categoría A | Nombre A | Categoría B | Nombre B | Egresos con ambas | Phi |
|---|---|---|---|---|---|
| ELIX_06 | Hipertensión no complicada | ELIX_11 | Diabetes no complicada | 16.014 | 0,40 |
| ELIX_01 | Insuficiencia cardíaca congestiva | ELIX_02 | Arritmias cardíacas | 2.413 | 0,27 |
| ELIX_15 | Enfermedad hepática | ELIX_28 | Abuso de alcohol | 1.573 | 0,24 |
| ELIX_06 | Hipertensión no complicada | ELIX_14 | Insuficiencia renal | 7.310 | 0,23 |
| ELIX_12 | Diabetes complicada | ELIX_14 | Insuficiencia renal | 1.867 | 0,21 |
| ELIX_01 | Insuficiencia cardíaca congestiva | ELIX_14 | Insuficiencia renal | 2.147 | 0,20 |
| ELIX_14 | Insuficiencia renal | ELIX_25 | Trastornos hidroelectrolíticos | 2.528 | 0,20 |
| ELIX_01 | Insuficiencia cardíaca congestiva | ELIX_03 | Valvulopatía | 968 | 0,20 |
| ELIX_07 | Hipertensión complicada | ELIX_14 | Insuficiencia renal | 877 | 0,20 |
| ELIX_01 | Insuficiencia cardíaca congestiva | ELIX_07 | Hipertensión complicada | 706 | 0,19 |

*Los diez pares con mayor asociación.* *Tabla completa en `tablas/t23_elix_phi.csv`*

## 11. Deriva de la codificación

El promedio de diagnósticos secundarios por egreso de adultos hospitalizados pasa de 4,14 en 2019 a 5,82 en 2024, un cambio relativo de 40,4 % (IC 38,7 % a 42,2 %). Si la codificación crece sin que cambien los pacientes, el puntaje de comorbilidad sube y un modelo entrenado en los años antiguos subestima el riesgo de los nuevos; es evidencia para la decisión 17.

![Dos paneles por año: a la izquierda el promedio nacional de diagnósticos secundarios por egreso de adultos hospitalizados con su intervalo de confianza; a la derecha una línea gris por hospital y el total nacional destacado.](../figuras/F08_02.png)

*Figura F08_02. Deriva de la codificación: diagnósticos secundarios por egreso. Datos de origen: `tablas/t23_deriva_codificacion.csv`.*

Entre los 53 hospitales con al menos 100 egresos en el primer y el último año, el cambio relativo del promedio va de 8,4 % (percentil diez) a 78,8 % (percentil noventa), y 50 aumentan. La codificación no cambia igual en todos los hospitales, por eso el cambio pesa sobre la comparación entre ellos. Esta figura es de uso interno: muestra códigos de hospital y no nombres.

| Año | Adultos hospitalizados | Promedio | IC bajo | IC alto | Cambio relativo | IC bajo del cambio | IC alto del cambio |
|---|---|---|---|---|---|---|---|
| 2019 | 31.755 | 4,14 | 4,10 | 4,18 | n/d | n/d | n/d |
| 2020 | 37.465 | 4,73 | 4,69 | 4,77 | 14,2 % | 12,8 % | 15,7 % |
| 2021 | 36.781 | 5,35 | 5,31 | 5,40 | 29,3 % | 27,6 % | 30,9 % |
| 2022 | 34.833 | 5,24 | 5,20 | 5,29 | 26,5 % | 24,9 % | 28,1 % |
| 2023 | 33.717 | 5,53 | 5,49 | 5,58 | 33,6 % | 31,9 % | 35,3 % |
| 2024 | 33.676 | 5,82 | 5,77 | 5,87 | 40,4 % | 38,7 % | 42,2 % |

*Promedio de diagnósticos secundarios, total nacional.* *Tabla completa en `tablas/t23_deriva_codificacion.csv`*

- Las categorías no distinguen comorbilidades presentes al ingreso de complicaciones de la hospitalización: los diagnósticos están codificados al egreso y no hay marcador de presente al ingreso.
- Los intervalos ignoran el agrupamiento por hospital y son indicativos.
- En el extracto los conteos absolutos no se comparan con los datos completos.

## 12. Hospitales: perfil y volumen

En el último año con desenlace (2023) hay 68 hospitales con egresos, 3 de ellos pediátricos o monográficos (más de 50,0 % de sus egresos son de menores) y 0 fuera del catálogo. Con el volumen mínimo propuesto (500 egresos elegibles y 10 muertes esperadas) cumplen 23. Las muertes esperadas salen de estandarizar por edad y sexo con las tasas nacionales del mismo año. En 2023, 3,0 % de los hospitales tienen menos de 10 egresos elegibles y quedan con la celda suprimida.

| Año | Hospitales | Fuera del catálogo | Pediátricos o monográficos | Con cohorte | Cumplen el volumen | Celdas bajo el mínimo |
|---|---|---|---|---|---|---|
| 2019 | 65 | 0 | 3 | 63 | 18 | 1,6 % |
| 2020 | 65 | 0 | 3 | 65 | 25 | 4,6 % |
| 2021 | 65 | 0 | 3 | 65 | 27 | 4,6 % |
| 2022 | 65 | 0 | 3 | 64 | 22 | 3,1 % |
| 2023 | 68 | 0 | 3 | 67 | 23 | 3,0 % |
| 2024 | 72 | 1 | 3 | n/d | n/d | n/d |

*Hospitales por año.* *Tabla completa en `tablas/t23_hospitales_resumen.csv`*

| Egresos mínimos | Muertes esperadas mínimas | Hospitales | Cumplen egresos | Cumplen muertes | Cumplen ambos | Ambos, sin pediátricos |
|---|---|---|---|---|---|---|
| 200 | 5 | 67 | 52 | 63 | 52 | 52 |
| 200 | 10 | 67 | 52 | 43 | 43 | 43 |
| 200 | 20 | 67 | 52 | 24 | 24 | 24 |
| 500 | 5 | 67 | 23 | 63 | 23 | 23 |
| 500 | 10 | 67 | 23 | 43 | 23 | 23 |
| 500 | 20 | 67 | 23 | 24 | 22 | 22 |
| 1.000 | 5 | 67 | 2 | 63 | 2 | 2 |
| 1.000 | 10 | 67 | 2 | 43 | 2 | 2 |
| 1.000 | 20 | 67 | 2 | 24 | 2 | 2 |

*Cuadrícula de volumen mínimo en el último año con desenlace.* *Tabla completa en `tablas/t23_volumen_umbrales.csv`*

![Puntos por año, uno por hospital, con el número de egresos de la cohorte del modelo de mortalidad en escala logarítmica; las líneas horizontales marcan los volúmenes mínimos candidatos.](../figuras/F09_02.png)

*Figura F09_02. Distribución del volumen de egresos por hospital. Datos de origen: `tablas/t23_hospitales_volumen.csv`.*

## 13. Embudo crudo exploratorio

El embudo del período 2019-2023 compara la mortalidad cruda de cada hospital con el promedio nacional de la cohorte del modelo (5,11 %) y dibuja los límites de control al 95 % y al 99,8 % que corresponden a su volumen. Se estima un factor de sobredispersión de 6,34 con los puntajes z winsorizados (método de Spiegelhalter). De 66 hospitales evaluables, 9 quedan fuera del límite de 95 % y 3 fuera del de 99,8 %. Es una vista exploratoria sin ajuste por riesgo: no es el indicador. La figura incluye también el último año solo, con su propio factor.

![Gráficos de embudo: cada punto es un hospital con su mortalidad cruda y su número de egresos de la cohorte del modelo; las líneas son los límites de control al 95 % y al 99,8 % alrededor del promedio nacional, ajustados por sobredispersión.](../figuras/F09_01.png)

*Figura F09_01. Embudo crudo exploratorio de la mortalidad por hospital. Datos de origen: `tablas/t23_embudo.csv`.*

## 14. Identificador, historial y traslados

El identificador del paciente forma 2 espacios sin pacientes en común (2019-2020; 2021-2023). El historial de doce meses solo es posible dentro de un espacio, y el primer año de cada uno no tiene ventana completa por construcción. Un egreso es viable si su identificador es válido y su ventana queda dentro del espacio.

| Año | Espacio de identificadores | Egresos | Identificador válido | Ventana completa | Viable | Con egreso previo | Cumple el mínimo |
|---|---|---|---|---|---|---|---|
| 2019 | 2019-2020 | 49.999 | 98,7 % | 0,0 % | 0,0 % | n/d | no |
| 2020 | 2019-2020 | 49.999 | 99,0 % | 98,1 % | 97,1 % | 3,9 % | si |
| 2021 | 2021-2023 | 49.999 | 99,7 % | 0,0 % | 0,0 % | n/d | no |
| 2022 | 2021-2023 | 49.999 | 99,7 % | 98,7 % | 98,4 % | 2,7 % | si |
| 2023 | 2021-2023 | 49.999 | 99,8 % | 100,0 % | 99,8 % | 2,5 % | si |

*Factibilidad del historial.* *Tabla completa en `tablas/t23_historial_factibilidad.csv`*

![Barras por año con el porcentaje de egresos con identificador válido, con ventana de doce meses completa dentro del espacio de identificadores y con ambas condiciones; un rombo marca el porcentaje con egreso previo y una línea el mínimo de factibilidad.](../figuras/F09_03.png)

*Figura F09_03. Factibilidad del historial de doce meses por año. Datos de origen: `tablas/t23_historial_factibilidad.csv`.*

En 2023 la factibilidad del historial es 99,8 %, igual o superior al mínimo de 80,0 % de la decisión 1. El porcentaje de egresos con un egreso previo dentro de la ventana es 2,5 % entre los viables.

| Período | Dirección | Ventana (días) | Traslados | Sin identificador válido | Elegibles | Enlazados | Tasa de enlace | IC bajo | IC alto |
|---|---|---|---|---|---|---|---|---|---|
| 2019-2023 | entrante | 1 | 15.785 | 80 | 15.502 | 103 | 0,7 % | 0,5 % | 0,8 % |
| 2023 | entrante | 1 | 3.142 | 0 | 3.142 | 28 | 0,9 % | 0,6 % | 1,3 % |
| 2019-2023 | entrante | 2 | 15.785 | 80 | 15.481 | 105 | 0,7 % | 0,6 % | 0,8 % |
| 2023 | entrante | 2 | 3.142 | 0 | 3.142 | 28 | 0,9 % | 0,6 % | 1,3 % |
| 2019-2023 | entrante | 7 | 15.785 | 80 | 15.383 | 105 | 0,7 % | 0,6 % | 0,8 % |
| 2023 | entrante | 7 | 3.142 | 0 | 3.142 | 28 | 0,9 % | 0,6 % | 1,3 % |
| 2019-2023 | entrante | 30 | 15.785 | 80 | 14.939 | 106 | 0,7 % | 0,6 % | 0,9 % |
| 2023 | entrante | 30 | 3.142 | 0 | 3.142 | 29 | 0,9 % | 0,6 % | 1,3 % |
| 2019-2023 | saliente | 1 | 5.246 | 28 | 5.214 | 118 | 2,3 % | 1,9 % | 2,7 % |
| 2023 | saliente | 1 | 995 | 0 | 993 | 27 | 2,7 % | 1,9 % | 3,9 % |
| 2019-2023 | saliente | 2 | 5.246 | 28 | 5.205 | 122 | 2,3 % | 2,0 % | 2,8 % |
| 2023 | saliente | 2 | 995 | 0 | 990 | 27 | 2,7 % | 1,9 % | 3,9 % |
| 2019-2023 | saliente | 7 | 5.246 | 28 | 5.189 | 124 | 2,4 % | 2,0 % | 2,8 % |
| 2023 | saliente | 7 | 995 | 0 | 983 | 27 | 2,7 % | 1,9 % | 4,0 % |
| 2019-2023 | saliente | 30 | 5.246 | 28 | 5.017 | 130 | 2,6 % | 2,2 % | 3,1 % |
| 2023 | saliente | 30 | 995 | 0 | 918 | 26 | 2,8 % | 1,9 % | 4,1 % |

*Enlace de traslados salientes y entrantes.* *Tabla completa en `tablas/t23_enlaces_traslado.csv`*

En 2023, 2,8 % de los 918 traslados salientes elegibles se enlaza con un ingreso en otro hospital dentro de 30 días. Esa tasa no alcanza el mínimo de 30,0 % que justifica el rastreo (decisión 13). Un traslado cuyo plazo de enlace se sale del último día observado no se cuenta como no enlazado: se excluye del denominador. Los ingresos de 2024 no se buscan.

En un extracto, el ingreso siguiente de un paciente casi nunca está en el mismo extracto, de modo que la factibilidad del historial, el egreso previo y el enlace de traslados se subestiman por construcción. Estas cifras solo prueban la mecánica y no sirven para decidir: quedan no evaluables hasta correr con los datos completos.

## 15. Hospital de procedencia

HOSPPROCEDENCIA trae el nombre del hospital de procedencia. En 30,7 % de los egresos que lo informan, el nombre coincide con el del hospital del egreso (30,6 % con nombres idénticos tras normalizar). El criterio de la decisión 2 acepta hasta 5,0 %: se supera, así que HOSPPROCEDENCIA no sirve para definir el traslado entrante y se propone usar TIPO_PROCEDENCIA.

| Tipo de procedencia | Egresos | Con hospital de procedencia | Del mismo hospital | % del mismo hospital |
|---|---|---|---|---|
| (todas) | 49.999 | 6.823 | 2.155 | 31,6 % |
| OTROS HOSPITALES DE LA RED | 3.582 | 3.564 | 38 | 1,1 % |
| SERVICIO EMERGENCIA (DOMICILIO) | 23.344 | 1.179 | 1.098 | 93,1 % |
| CENTRO ESPECIALIDADES (CDT, CRS, CONSULTORIO ADOS. ESP) | 15.460 | 821 | 800 | 97,4 % |
| OTROS HOSPITALES RED NACIONAL | 526 | 521 | 0 | 0,0 % |
| OTRAS INSTITUCIONES SALUD (CLÍNICAS PRIVADAS, DE REHABILITAC | 549 | 503 | <10 | n/d |
| CARDIOCIRUGÍA PAGO GRD | 143 | 64 | 64 | 100,0 % |
| PLAN DE RESOLUCIÓN LE | 814 | 58 | 53 | 91,4 % |

*Hospital de procedencia por tipo de procedencia, último año con desenlace.* *Tabla completa en `tablas/t23_procedencia_autorref.csv`*

## 16. Mortalidad de los traslados entrantes

Se llama entrante al ingreso cuyo TIPO_PROCEDENCIA es otro hospital de la red. En adultos de la cohorte dura con desenlace conocido, la mortalidad de los entrantes es 7,53 % y la de los demás ingresos 4,06 %; la razón es 1,86 (IC 1,74 a 1,98). Es evidencia para la decisión 16, que decide si se excluyen los entrantes o solo los enlazados con un traslado saliente.

| Período | Entrante | Egresos | Fallecidos | Mortalidad | IC bajo | IC alto | Razón vs no entrantes |
|---|---|---|---|---|---|---|---|
| 2019 | si | 2.042 | 122 | 5,97 % | 5,03 % | 7,09 % | 1,78 |
| 2019 | no | 33.023 | 1.106 | 3,35 % | 3,16 % | 3,55 % | n/d |
| 2020 | si | 2.794 | 226 | 8,09 % | 7,13 % | 9,16 % | 1,60 |
| 2020 | no | 30.494 | 1.541 | 5,05 % | 4,81 % | 5,30 % | n/d |
| 2021 | si | 2.873 | 279 | 9,71 % | 8,68 % | 10,85 % | 1,89 |
| 2021 | no | 32.399 | 1.662 | 5,13 % | 4,89 % | 5,38 % | n/d |
| 2022 | si | 2.419 | 168 | 6,94 % | 6,00 % | 8,03 % | 1,82 |
| 2022 | no | 32.648 | 1.249 | 3,83 % | 3,62 % | 4,04 % | n/d |
| 2023 | si | 2.472 | 154 | 6,23 % | 5,34 % | 7,25 % | 2,06 |
| 2023 | no | 32.790 | 990 | 3,02 % | 2,84 % | 3,21 % | n/d |
| 2019-2023 | si | 12.600 | 949 | 7,53 % | 7,08 % | 8,01 % | 1,86 |
| 2019-2023 | no | 161.354 | 6.548 | 4,06 % | 3,96 % | 4,16 % | n/d |

*Mortalidad cruda de entrantes y no entrantes.* *Tabla completa en `tablas/t23_resultado_entrantes.csv`*

## 17. Límites

- Los intervalos y el embudo ignoran el agrupamiento por paciente y por hospital.
- El embudo es crudo: sin ajuste por riesgo, no es el indicador de desempeño.
- El factor de sobredispersión multiplicativo crece con el volumen de cada hospital, por lo que no es comparable entre períodos de distinta duración: el período agrupado da un factor mayor que un año solo.
- La comparación de nombres de hospital es aproximada: un nombre escrito distinto puede quedar como no coincidente.
- Los enlaces y el historial dependen de que el identificador sea consistente; las inconsistencias de nacimiento y sexo por identificador están en el informe de calidad.
- Las figuras por hospital son de uso interno y usan códigos, no nombres, mientras la decisión 10 siga abierta.
