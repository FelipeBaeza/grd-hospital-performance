# Tarea 10: asociación mixta, clusters de redundancia y VIF (etapa B, vuelta 1)

La etapa B agrega `eda/asociacion.py` (1.156 líneas), cuatro ayudas gráficas y un cambio de retorno en `eda/graficos.py`, una línea en `config/plan_figuras.csv` y 101 pruebas. Sobre el panel de la etapa A, sin leer datos, mide la asociación de las 58 variables candidatas más la referencia COD_HOSPITAL con la medida propia de cada tipo de par, agrupa por redundancia (0,80 con igualdad) con componentes conexas y un dendrograma de enlace promedio, y calcula el VIF con y sin los agregados de comorbilidad; escribe seis tablas y las figuras F10_01 a F10_04. La suite completa pasa (672 pruebas), mis recálculos con scipy y scikit-learn coinciden con las tablas hasta el redondeo de seis decimales y una segunda corrida en otra carpeta reproduce byte a byte tablas y PNG. El registro de accesos de la Tarea 10 no tiene lecturas de 2024 ni bloqueos. No hay HIGH ni MEDIUM; quedan seis NIT de comunicación y rotulado.

Watch for: F10_02 deja casi invisibles cinco de los siete grupos redundantes (los de asociación casi perfecta) y repite colores entre grupos (NIT, confirmed); F10_01 a F10_03 rotulan las categorías de Elixhauser como ELIX_nn sin nombre (NIT, confirmed); el modo completo con datos reales no se pudo verificar, límite que el implementador ya declara (possible).

**Verdict**: APPROVED

## Vista general

Cada tipo de par usa su propia medida (Spearman, phi, eta, V de Cramér con corrección de Bergsma) con eliminación de faltantes por pares. Los pares de binarias salen de una llamada vectorizada y una prueba obliga a que coincida con la ruta par a par.

La redundancia se define dos veces a propósito. El cluster es una componente conexa de las aristas con asociación de al menos 0,80 y es lo que usará la selección de variables; el grupo del dendrograma es lo que queda junto al cortar el árbol. En el extracto coinciden (50 y 50) y la cadena A~B~C de las pruebas cubre el caso en que no.

El VIF parte de una sola matriz de correlación de casos completos y detecta por rango la colinealidad exacta de los agregados de Elixhauser (42 columnas, rango 39, tres relaciones) en lugar de invertir una matriz singular.

La lectura de datos sigue concentrada en `panel.construir_panel`. Las tablas se escriben con supresión por celda pequeña, por bloques en los pares redundantes para que la reestimación de 2023 viva en columnas aparte, y el resumen y las métricas toman la versión publicable.

Las figuras se arman solo desde su CSV de respaldo. Los defectos son de comunicación, no de legibilidad.

<details>
<summary>Issues (6)</summary>

1. **Colores repetidos en F10_02**: NIT, confirmed. El color de cada subárbol sale de `paleta[(grupo - 1) % 6]` (asociacion.py:802), así que los grupos 22, 34 y 46 quedan los tres en bermellón y los grupos 6 y 18 en púrpura, mientras el pie (línea 821) dice "Cada color es un grupo". Corregir el pie ("los colores se reutilizan entre grupos") o asignar el color por orden de aparición.
2. **Grupos casi perfectos invisibles en F10_02**: NIT, confirmed. Cinco de los siete grupos se unen a una altura de 0,000013 a 0,001081, es decir de 0,01 a 1,2 píxeles a 150 dpi: marcas pegadas al eje, justo los pares que motivan la figura. Marcarlos con un punto o un rótulo (uniones por debajo de 0,01) o añadir un recuadro ampliado.
3. **Elixhauser sin nombre en F10_01 a F10_03**: NIT, confirmed. Las 31 categorías se rotulan ELIX_nn y solo F10_04 las nombra, lo que obliga al lector clínico a cruzar figuras. Rotular con "nn nombre corto" o incluir una tabla de equivalencias en el informe.
4. **Rotulado menor de F10_01 y F10_04**: NIT, confirmed. Los ejes de F10_01 dicen "Variable (ordenadas por similitud)" (asociacion.py:767-768) y el pie de F10_04 no dice que las celdas grises son las suprimidas (el de F10_01 sí). Poner "Variables" y agregar la frase al pie.
5. **Leyenda "quince pares" con nueve filas**: NIT, confirmed. `agregar_secciones` fija "Los quince pares más asociados" (asociacion.py:1085) pero `pares_redundantes.csv` solo lista los redundantes, nueve en el extracto. Escribir "hasta quince" o armar la leyenda con el número de filas.
6. **`git status` ejecutado pese a la restricción**: NIT, confirmed. El registro de la etapa lo declara; `.git/index` conserva su `mtime` (2026-10-07 20:43) y no hay `index.lock`, solo cambió el `mtime` del directorio `.git`. No requiere acción, salvo no repetirlo.

</details>

<details>
<summary>Details</summary>

### Recálculo independiente de la matriz, el VIF y los clusters con scipy y scikit-learn

Se reconstruyó el panel con `panel.construir_panel` en un contexto propio (143.350 filas: 115.310 de 2019-2022 y 28.040 de 2023) y se recalcularon, sin importar `eda.asociacion`, 16 pares que cubren los seis tipos de combinación, varios con nulos (ES_EXTRANJERO con PUEBLO_ORIGINARIO n 114.780; NIVEL_CUIDADO_INGRESO con ES_EXTRANJERO n 114.637; EDAD_ANIOS con NIVEL_CUIDADO_INGRESO n 115.167). `spearmanr` (EDAD_ANIOS con N_DX_SEC 0,362677), `matthews_corrcoef` (ELIX_01 con ELIX_02 0,265588), `f_oneway` (EDAD_ANIOS con GRUPO_CLINICO 0,548391) y la V de Bergsma a mano sobre `chi2_contingency` (TIPO_INGRESO con GRUPO_CLINICO 0,637774; COD_HOSPITAL con SERVICIO_SALUD 0,909540) coinciden con `matriz_asociacion.csv` con diferencia máxima de 4,9e-07, o sea el redondeo, y con el mismo `n_pares` y el mismo signo.

Para el VIF, `np.linalg.inv(np.corrcoef(...))` sobre los 114.637 casos completos da el conjunto base de 38 columnas y rango 38, con máximo 2,474475 (ENTRANTE_TIPO_PROCEDENCIA) y diferencia máxima de 4,5e-07 contra `t23_vif.csv`; `LinearRegression` reproduce 1,751828 (EDAD_ANIOS) y 2,463186 (DERIVADO_OTRO_HOSPITAL). Con los agregados hay 42 columnas y rango 39 (`matrix_rank`, tol 1e-9): el R² de ELIX_01, ELIX_17, N_COMORB_ELIX y los dos puntajes contra las demás vale 1,000000000000, los ocho VIF de las variables no exactas coinciden con scikit-learn hasta 4,2e-07 y la identidad de N_COMORB_ELIX con la suma de las 31 categorías da diferencia 0.

Para el dendrograma, `linkage(method="average")` sobre `1 - asociacion_abs` de la matriz publicada reproduce las alturas de `t23_dendrograma.csv` hasta 5,0e-07, con los mismos nodos hijos, el mismo `n_hojas` y el mismo orden de hojas (`leaves_list` contra `posicion_dendrograma`). `fcluster` a 0,20 más 1e-12 y `connected_components` con aristas de al menos 0,80 dan las mismas particiones de 50 grupos y 50 clusters que las tablas, y el conjunto de 9 pares redundantes es el mismo. También coinciden la numeración de grupos y clusters, `altura_union`, `max_asociacion_otra` y `n_pares_redundantes` de las 59 filas de `clusters_variables.csv`. Los nueve pares reproducen `asociacion_2023` y `n_pares_2023`; PROVINCIA con SERVICIO_SALUD pasa de 0,847834 a 0,869254 y es el único par que no es casi nominal. El par más cercano al umbral sin cruzarlo es DERIVADO_OTRO_HOSPITAL con TIPO_PROCEDENCIA (0,790855): la regla de 0,80 lo deja fuera del cluster de procedencia, dato a tener presente en la Tarea 11. Las seis tablas traen las columnas y el orden de DIS 9.2 a 9.6, y la matriz publicada es simétrica, con diagonal 1 y valores en [0, 1].

### Cortafuegos de 2024, privacidad, determinismo y restricciones del flujo

`asociacion.py` no llama a `ctx.lector` (la búsqueda de `lector`, `.leer(` y `2024` solo encuentra docstrings); la única lectura es `panel.construir_panel`, que filtra por `anios_resultado` y lanza `RuntimeError` antes de leer si el año expuesto aparece entre ellos (panel.py:710-712; la lectura está en la línea 729). En mi corrida el registro tiene cinco entradas T10 (2019 a 2023, propósito `seleccion`), 0 violaciones, 0 bloqueos, 0 entradas de 2024 y 0 violaciones de privacidad. En la corrida del implementador (`/tmp/real_t10`) las únicas entradas de 2024 son de T04 y T08, con propósito `estructura`. Las 15 métricas de la Tarea 10 (11 de `asociacion.` y `vif.` más 4 del panel) llevan alcance explícito 2019-2022 o 2023.

Una segunda corrida en otra carpeta y otro proceso da los mismos `sha256` en las ocho tablas y los mismos `sha256_png` y `sha256_csv` en las cuatro figuras que el registro del implementador, así que los PNG revisados son los que produce el código actual; `figuras_indice.csv` coincide con `plan_figuras.csv` en `csv` y en `decision_relacionada`.

Fuera de `fase2_eda/` solo cambiaron los documentos de `.agents/tasks` del orquestador y, aparte del `git status` del punto 6, no hay `.md` nuevos fuera de `.agents`, `salidas/` está intacta y no quedan restos de mutación en `/tmp`.

### Legibilidad de F10_01 a F10_04

Se abrieron los cuatro PNG completos con `read_file` y recortes a resolución nativa de las etiquetas, la leyenda y los rótulos. Ningún píxel no blanco toca el borde de las imágenes (sin texto cortado), no hay solapes entre las 59 etiquetas de 6,5 pt de F10_01 ni entre las 31 de F10_04, la leyenda de F10_03 queda fuera de los datos y el rótulo del corte de F10_02 queda sobre el borde superior del área, sin tapar hojas. La coma decimal, las escalas (la logarítmica de F10_03 de 1 a 20 con marcas en 1, 2, 5, 10 y 20) y los pies con fuente, años y nota de supresión cumplen la lista de cotejo de DIS 14.7; la diagonal saturada de F10_04 queda declarada en su pie.

Los defectos son los de los puntos 1 a 4 de la lista, todos de comunicación. En F10_02 el recuento de píxeles por color Okabe-Ito da 3 grupos en bermellón (6 bandas de hojas), 2 en púrpura (4 bandas), uno en naranja y uno en celeste. El eje va de 0 a 1 en 1.064 píxeles, de modo que la unión a 0,001081 mide 1,2 píxeles y las de 0,000013 a 0,000257 menos de uno. La información de esos grupos sí está en F10_01, donde los recuadros nombran a cada miembro, por eso no bloquea.

### Informe provisional y trazabilidad de cifras

Las 41 cifras y tablas del informe de la corrida real figuran en `cifras_informe.csv` con su CSV y columna de origen, y coinciden con `t23_resumen_asociacion.csv` y con mis recálculos (58 variables, 1.653 pares, 9 redundantes, 7 clusters de 15 variables, 114.637 casos completos, rango 39 de 42 columnas). El único desajuste es el del punto 5: la leyenda dice quince pares y la tabla trae nueve.

### Pruebas y mutantes asignados a la etapa B

Las 101 pruebas nuevas (8 gráficas y 93 de asociación) coinciden con el registro del implementador (672 menos 571). Los cinco mutantes asignados (M17, M18, M19, M20, M42) tienen su prueba por nombre y cada una fija un caso que distingue la mutación: phi exacto 0,80 con la tabla (432, 48, 48, 432) y comprobación en `clusters_variables.csv`, rho de -0,9 con distancia por valor absoluto, eliminación por pares contra relleno con cero (diferencia mayor que 0,05), referencia fuera de componentes, pares, dendrograma y VIF, y 19 % contra 20 % exacto con variables genéricas. Como muestreo apliqué M42 (exclusión por faltante solo para DIAS_DESDE_EGRESO_PREVIO) sobre una copia en `/tmp`: fallan exactamente las dos pruebas que el registro indica, la del 19 por ciento (800 filas en lugar de 810) y la integración de modo completo (1 exclusión en lugar de 2). No se reejecutaron los otros 31 mutantes del registro.

### Lo que no se pudo verificar

El modo completo con datos reales (hasta 800.000 filas): la memoria de `phi_pares`, que el implementador estima en unos 0,7 GB, el tiempo y la legibilidad de F10_01 con 62 etiquetas (possible; sin los datos completos no hay forma de comprobarlo). El historial en extractos es no evaluable por construcción.

</details>

<details>
<summary>Mapa de archivos</summary>

- `eda/asociacion.py` (nuevo, 1.156 líneas): medidas por par, matriz, pares, clusters, dendrograma, VIF, resumen, métricas, cuatro figuras e informe.
- `eda/graficos.py`: `mapa_calor` devuelve la `Colorbar`; nuevas `formato_barra_decimal`, `recuadros_de_grupo`, `linea_gruesa` y `altura_filas`.
- `config/plan_figuras.csv`: F10_02 apunta a `t23_dendrograma.csv`.
- `tests/test_t10_asociacion.py` (93 pruebas) y `tests/test_graficos_t10t11.py` (8 pruebas): nuevos.

Sin git en este flujo: el cambio completo es el contenido de estos archivos. La salida revisada está en `/tmp/real_t10/extracto/` (tablas, figuras e informe provisional).

</details>
