# Reescritura del historial de git (8 de noviembre de 2026)

Por decisión del autor, el 08-11-2026 se reescribió el historial público de git para quitar de las 4 ramas (y de la etiqueta `v1-congelado` local) las versiones
que rompían el criterio DT04 (nada de conteos de 1 a 9 casos, ni reconstruibles). Cierra la decisión pendiente de **I-09** y las medidas de historial de
**I-11** e **I-12**.

## Qué se hizo

Se reemplazaron **17 versiones de archivos** (contenido de los *blobs*). La estructura, los mensajes y las fechas de cada commit no cambian: el historial
tiene los mismos 77 commits en el mismo orden y con las mismas fechas de autoría y de confirmación. **El contenido final de `main`, de `fase2-v7` y de
`fase2-revision-profesor-2` queda idéntico** (mismo hash de árbol antes y después). Solo `orden-v5`, la rama antigua del pull request #1, difiere, y solo en los
5 archivos de I-11.

| Incidente | Archivos | Versión que había | Versión que queda |
|---|---|---|---|
| I-12 (Fase 3) | `catalogo_hospitales_anio.csv` y `catalogo_hospitales.sql` en el commit de la Fase 3 v7 | conteos exactos de C y D por hospital y año | los redondeados a la decena, que ya publicaba el repositorio |
| I-12 (Fase 5) | 7 tablas por hospital y subgrupo de `ensayo_2023_v4` en el commit de la verificación previa | observado, esperado y límites del embudo exactos | la versión publicable (sin esos valores, egresos a la decena) |
| I-11 | 5 tablas por subgrupo del protocolo v1.0 y de la Fase 6 en el congelado del 21-10 | 13 conteos de 1 a 9 reconstruibles por complemento | la versión corregida del 22-10 |
| I-09b | `reports/hospital_benchmarks_2024.csv`, `hospital_benchmarks_all_years.csv` y `calibration_mortality_2024.csv` del prototipo | 6 celdas de 1 a 9 muertes, con su HSMR o razón O/E | celdas en «<10» y los derivados en blanco |

Las demás exposiciones de I-01 (las métricas de 2024 del prototipo y los mensajes de commit que las citan) **no se tocaron**: son el registro de lo que pasó.

## Cómo se verificó

- Un escáner recorrió el árbol completo de **cada uno de los 77 commits** (16 commits de `bbf4252` a `b3dc35a` tenían 13 hallazgos de I-11, y 10 commits de `9bbdf32` a `4143048` tenían 264 o 353 de I-12):
  antes, 26 de los 77 árboles tenían hallazgos; ahora los **77 dan 0 hallazgos** (privacidad por propiedad de la Fase 5, con la comparación entre tablas de I-12).
- `git fsck` sin errores. El número de commits no cambió (77).
- Los árboles de las puntas de `main`, `fase2-v7` y `fase2-revision-profesor-2` son idénticos a los de antes de la reescritura.

## Lo que cambia y hay que saber

1. **Todos los identificadores de commit cambiaron**, incluido el primero. Los que citan los informes y los documentos (por ejemplo `9bbdf32`, `a225dd9`,
   `b718454`, `ca3c40b`) son los **anteriores a la reescritura**; la tabla de equivalencias está al final de este archivo. Los anexos de las enmiendas 01, 02 y 03
   y el registro de la enmienda 02 siguen citando los identificadores antiguos, y no se modificaron para no alterar documentos ya firmados o sellados.
2. **El manifiesto `MANIFIESTO_v1-congelado.sha256` ya no coincide con la etiqueta en 5 líneas** (656, 657, 670, 671 y 710): son las 5 tablas de I-11. El manifiesto
   conserva las huellas del congelado original; la etiqueta ahora trae la versión corregida. Las huellas SHA-256 del contenido anterior y del nuevo están abajo.
3. **Las fechas se conservan**, pero los eventos de *push* de GitHub del 27-09 que cita I-01 son los del repositorio anterior. Las horas de I-01 siguen siendo las de esos eventos; las fechas de los commits son las mismas.
4. **El respaldo del estado anterior** (un archivo `.bundle` con todas las ramas y la etiqueta) está **fuera del repositorio** y no debe publicarse: contiene lo que se quitó.
   Huella SHA-256 del respaldo: `f7e7dc65f3f91c1580dec6d9e18cc84c74974da24428e766e4913a01cbd54d90`.

## Lo que la reescritura no resuelve (requiere a GitHub o al autor)

- **La etiqueta `v1-congelado` de GitHub sigue apuntando al commit anterior** (`0953162`), que conserva todo lo que se quitó. Desde el entorno de trabajo no se pueden modificar etiquetas
  (el servidor intermedio responde HTTP 403). El autor debe reemplazarla: borrarla en GitHub (página de etiquetas) y crearla de nuevo sobre el commit `fb8c3f0` (el `0953162` de
  antes), o desde un clon: `git tag -fa v1-congelado fb8c3f0 -m "Congelado v1.0"` y `git push -f origin v1-congelado`. Mientras tanto, el historial anterior sigue alcanzable por esa etiqueta.
- El commit `7248d78` de I-09a (la prueba de privacidad con valores reales) **ya no era alcanzable desde ninguna rama**: solo existe en los servidores de GitHub y se ve si se conoce su identificador.
- La referencia `refs/pull/1/head` del pull request #1 apunta al `orden-v5` anterior, y GitHub no deja cambiarla.
- Las vistas en caché, y cualquier fork o clon que alguien ya haya hecho.

Para esos tres puntos hay que escribir al soporte de GitHub (opción «Remove cached views and references to sensitive data in pull requests», en la página de soporte de contacto de GitHub).

## Huellas de los archivos reemplazados (SHA-256, primeros 16 caracteres)

| Archivo | Incidente | Antes | Después |
|---|---|---|---|
| `fase3_preparacion/salidas/tablas/catalogo_hospitales_anio.csv` | I-12 Fase 3 | `60db65cf405b5104` | `fb54859a10911ad3` |
| `fase3_preparacion/salidas/tablas/catalogo_hospitales.sql` | I-12 Fase 3 | `a606328fd5c6bd78` | `3acac168babceccc` |
| `fase5_evaluacion/salidas/ensayo_2023_v4/tablas/oe_hospital_mortalidad.csv` | I-12 Fase 5 | `9e1bea40dbf7d8f6` | `72197a4988e90583` |
| `fase5_evaluacion/salidas/ensayo_2023_v4/tablas/oe_hospital_estadia.csv` | I-12 Fase 5 | `976d1252334e204a` | `1f6ba08e5dcfd564` |
| `fase5_evaluacion/salidas/ensayo_2023_v4/tablas/subgrupos_mortalidad.csv` | I-12 Fase 5 | `677cbfb4d2100c5e` | `8546c75e0acc3bce` |
| `fase5_evaluacion/salidas/ensayo_2023_v4/tablas/subgrupos_estadia.csv` | I-12 Fase 5 | `6be7042ad0ba46a2` | `53582f6f4667acd0` |
| `fase5_evaluacion/salidas/ensayo_2023_v4/tablas/indice_exhaustividad_hospital.csv` | I-12 Fase 5 | `7095d2e2e23acf83` | `6e8eb5a92148f377` |
| `fase5_evaluacion/salidas/ensayo_2023_v4/tablas/tasa_traslado_hospital.csv` | I-12 Fase 5 | `c446f4f72dbf1786` | `8502ffd1e7ebcb9b` |
| `fase5_evaluacion/salidas/ensayo_2023_v4/tablas/traslado_entrante_hospital.csv` | I-12 Fase 5 | `b3aefb20f3f55288` | `60297c9ce690b4de` |
| `fase5_evaluacion/salidas/ensayo_2023/tablas/subgrupos_mortalidad.csv` | I-11 | `2cc6d533b4dec6d5` | `a69384a9ceda22e0` |
| `fase5_evaluacion/salidas/ensayo_2023/tablas/subgrupos_estadia.csv` | I-11 | `c1a53f6d8404fc86` | `78d9b1de0c89dcdb` |
| `fase5_evaluacion/salidas/evaluacion_2024/tablas/subgrupos_mortalidad.csv` | I-11 | `23af84c78231a268` | `4e26237a9a7c776f` |
| `fase5_evaluacion/salidas/evaluacion_2024/tablas/subgrupos_estadia.csv` | I-11 | `4b8bbec22cc1a521` | `ee95795aef880cba` |
| `fase6_despliegue/datos/publicables/subgrupo.csv` | I-11 | `00f3de9642a0b96d` | `b75dfc05f7dce23a` |
| `reports/hospital_benchmarks_2024.csv` | I-09b prototipo | `3c293575390ade2f` | `07728b5a46314cfe` |
| `reports/hospital_benchmarks_all_years.csv` | I-09b prototipo | `211bebc6b41c1d05` | `1a759ba6c66f43e3` |
| `reports/calibration_mortality_2024.csv` | I-09b prototipo | `6c68d3d304099a12` | `aa194e535ce865c5` |

## Equivalencia de identificadores de commit (antes → después)

| Antes | Después | Fecha | Mensaje |
|---|---|---|---|
| `2210ce6` | `b874231` | 27-09-2026 17:30 | Initial commit |
| `58d23a4` | `048fd04` | 27-09-2026 17:19 | feat: inicialización del repositorio de tesis con evidencias de Fases 1 y 2 y andamiaje ba |
| `9d502d6` | `47c0969` | 27-09-2026 17:20 | feat(ingesta): implementar validación de esquema y conversión a capa bronce en Parquet |
| `58aa3ef` | `94ef8b8` | 27-09-2026 17:40 | feat(silver): implementar limpieza, tipado a Capa Silver y generación de ID_EPISODIO |
| `3ee570b` | `8edb639` | 27-09-2026 17:53 | feat(pipeline): implementar paso 04 filtrado de cohorte CONSORT y comorbilidades Elixhause |
| `fc43cdf` | `d909c5f` | 27-09-2026 17:55 | feat(pipeline): implementar paso 05 transformacion capa Gold con garantia estricta anti-fu |
| `652827f` | `20c9fef` | 27-09-2026 17:58 | feat(ml): entrenar y evaluar modelo de riesgo de mortalidad intrahospitalaria con LightGBM |
| `311bd03` | `327adad` | 27-09-2026 17:59 | feat(ml): entrenar y evaluar modelo de regresion Tweedie de estancia media hospitalaria (L |
| `edbede3` | `abdde15` | 27-09-2026 18:01 | feat(benchmarks): calcular HSMR e IEMC ajustados por riesgo y comparar ML vs norma GRD FON |
| `f69f73c` | `5a1cb8d` | 27-09-2026 18:03 | feat(web): implementar CLI orquestador maestro y aplicacion web interactiva de desempeno h |
| `ca631e1` | `ed48705` | 27-09-2026 18:04 | docs: actualizar README con arquitectura completa, diagramas y resultados experimentales |
| `d109827` | `dd579bb` | 27-09-2026 20:17 | feat(eda): agregar suite de visualizaciones científicas y auditoría de calidad de Fase 2 |
| `128701a` | `0345fab` | 02-10-2026 22:51 | docs(fase2): resolver observaciones metodologicas, agregar matriz maestra 129 cols y selec |
| `452e745` | `7e2ce12` | 02-10-2026 23:18 | docs(fase2): resolver inconsistencias CONSORT secuencial, elixhauser puro y traslados CMS |
| `55f18c3` | `15ce105` | 02-10-2026 23:45 | fix(fase2): reconciliar CONSORT secuencial 2363 episodios, calibracion inpatient, K=5 upco |
| `0775e23` | `ecfc816` | 03-10-2026 07:35 | docs(fase2): congelar especificacion primaria, bootstrap K=5, enlace publico y calibracion |
| `37184c7` | `454cdcc` | 03-10-2026 07:51 | docs: resolucion matematica y metodologica completa de observaciones Fase 2 |
| `db21523` | `ea46fd8` | 03-10-2026 08:06 | fix: corregir nulos de FECHA_NACIMIENTO en matriz maestra (37 nulos) |
| `31dd9ab` | `50d3df7` | 03-10-2026 13:05 | docs(fase2): consolidar resolucion de observaciones metodologicas y propagacion completa |
| `daf1685` | `3ecf14e` | 03-10-2026 13:24 | docs(fase2): afinar rigor empirico, catalogo oficial DEIS de hospitales y recalculos preci |
| `5a37168` | `2fb859a` | 03-10-2026 13:33 | docs(fase2): auditoria fina, bootstrap pareado upcoding, funnel plot dia 0 y calibracion c |
| `8fba478` | `c5cdeaa` | 03-10-2026 13:49 | docs(fase2): calibracion canonica inpatient con N_dx, sobredispersion Spiegelhalter y reso |
| `4ffac13` | `3f29a2c` | 03-10-2026 14:03 | docs(fase2): cierre definitivo de fase 2 con catalogo oficial, regla unificada EB, retract |
| `0d1c374` | `73db1ab` | 03-10-2026 18:37 | docs(fase2): cierre definitivo formal con margen de materialidad clinica 10%, validacion e |
| `f41eab1` | `37cdf10` | 03-10-2026 18:49 | docs(fase2): resolucion fina definitiva con alerta robusta M2-M3, centros monograficos, es |
| `49cdafe` | `fe571ef` | 03-10-2026 19:37 | docs: resolucion integral de observaciones finales de fase 2 (alertas m3, sensibilidad blo |
| `9076614` | `dcd3389` | 03-10-2026 23:29 | docs: incorporacion de correcciones metodologicas finas (lambda_obs por E_link, tau sin mo |
| `cef34b6` | `c4add93` | 03-10-2026 23:43 | fix(fase2): re-establecer M2 primario, calibracion preingreso, LOHO out-of-sample real, es |
| `2245600` | `0d2e69d` | 03-10-2026 23:52 | feat(fase2): formalizar M2+MDC como primaria, taxonomía de alertas, tau específico por mod |
| `7af9302` | `d8a71a7` | 04-10-2026 10:42 | docs(fase2): reconciliacion exacta de puntos de quiebre, cotas de salida de alertas primar |
| `b81cfc7` | `4cd01d8` | 04-10-2026 12:15 | docs(fase2): unificacion canonica de E por hospital en CSV, serie historica Quillota, diag |
| `62b63a0` | `59ed17e` | 04-10-2026 12:43 | docs y modelado: cierre final fase 2 con M2+CAP, sensibilidad POA y deriva de codificación |
| `62c46a7` | `c267e19` | 04-10-2026 12:53 | refactor: suite consolidada de scripts de cálculo (01 a 06), eliminación de obsoletos y co |
| `0953162` | `fb8c3f0` | 21-10-2026 23:47 | Congela el estado del proyecto antes de ordenarlo (fases 2 a 6, informes v4) |
| `bbf4252` | `08c5f68` | 21-10-2026 23:47 | Ordena el proyecto: informes v5, guía, historial de versiones y archivo de proceso |
| `12f21c9` | `f13a724` | 21-10-2026 23:36 | Documenta qué queda fuera del repositorio público y por qué |
| `b8a9834` | `208d7d8` | 21-10-2026 20:56 | Fusiona el proyecto ordenado (fases 2 a 6, informes v5) y retira la versión antigua (#1) |
| `eeea204` | `3cbc941` | 22-10-2026 01:32 | Etapa 0 del plan de corrección: gobernanza |
| `77333d4` | `f74d7e5` | 22-10-2026 02:07 | Declara que el profesor guía autorizó después de conocer los resultados (I-10) |
| `89b9b56` | `47e6a45` | 22-10-2026 02:15 | Registra la enmienda 01 con la aprobación verbal del profesor guía |
| `b718454` | `0bc7981` | 22-10-2026 02:24 | Fase 5: código de los 15 análisis de la enmienda 01, registrado antes de correr |
| `e5bd614` | `4857254` | 22-10-2026 02:27 | Entorno fijado: requirements-lock.txt y Dockerfile (H-17) |
| `4b3f2bd` | `0da8234` | 22-10-2026 04:22 | Fase 5: resultados de la enmienda 01 (corrida única del 22-10-2026) |
| `a5535db` | `e7bfaa3` | 22-10-2026 04:23 | Enmienda 02 (borrador): datos y cohortes; no se corre hasta registrarla |
| `e4475e0` | `1589bdf` | 22-10-2026 04:23 | Fase 4: registro de reanudaciones de Optuna y semilla por prueba (plan 4.8, H-19) |
| `64f9759` | `cec657b` | 22-10-2026 04:23 | Defensa: citas del informe de seminario verificadas (plan 5.1) |
| `ba759d1` | `95624d4` | 22-10-2026 04:34 | Privacidad por propiedad (plan 4.2) e incidente I-11 |
| `dec2c3d` | `cce7198` | 22-10-2026 04:34 | Ingeniería (plan 4.3, 4.5, 4.7, 4.8): sin ajustes al evaluar, sello cripto y pruebas falta |
| `b3dc35a` | `05882f5` | 22-10-2026 04:36 | Defensa: limitaciones y respuestas al comité con la evidencia de la enmienda 01 (plan 5.2  |
| `0c3416d` | `a860f55` | 22-10-2026 15:02 | I-11 corregido: tablas por subgrupo regeneradas sin conteos pequeños reconstruibles |
| `b1cfe7d` | `ca04efb` | 22-10-2026 15:05 | Catálogo DEIS de tipo de establecimiento (plan 1.3 y 1.4) y texto final de la enmienda 02 |
| `ca3c40b` | `599deb2` | 22-10-2026 15:05 | Registra la enmienda 02 y su código antes de correr |
| `0fe32b3` | `0552996` | 22-10-2026 15:53 | Fase 5: resultados de la enmienda 02 (corrida única del 22-10-2026) |
| `7ea2839` | `9e45e1a` | 22-10-2026 15:56 | Enmienda 02: resumen, anexo y lectura en la documentación |
| `550256f` | `effd875` | 25-10-2026 19:52 | Fase 1 v6: restaura la versión 2 del informe y la corrige según lo ejecutado |
| `6321de3` | `b7bb2b8` | 25-10-2026 20:16 | Revisión del profesor guía: Fase 1 v6 corregida y Fase 2 v6 |
| `ae53810` | `2e35464` | 25-10-2026 22:21 | Fase 1 v6: sin anexos, sin el primer párrafo de «Sobre esta versión» y sin el aviso a FONA |
| `6763488` | `0e91bca` | 26-10-2026 03:50 | Fase 2 v7 (en curso): cambio 1.6, análisis del profesor y supresión complementaria |
| `61c1f98` | `fe81405` | 26-10-2026 04:39 | Fase 2 v7: ajustes de privacidad tras la corrida de prueba |
| `0cfd138` | `e7c9634` | 26-10-2026 05:23 | Fase 2 v7: identidades que faltaban en la supresión complementaria |
| `53792eb` | `3d2b6ac` | 26-10-2026 12:17 | Fase 2 v7: corrida oficial completa, análisis del profesor e informe Word |
| `367b0d0` | `f8e78be` | 02-11-2026 14:58 | Fase 2: análisis de la segunda revisión del profesor (S1 a S9) |
| `568206b` | `eef83da` | 02-11-2026 22:08 | Fase 2 v8: informe con las correcciones y los análisis de la segunda revisión del profesor |
| `a2178b9` | `191b1e1` | 02-11-2026 23:34 | Fase 3 v6: opciones de cohorte de la segunda revisión del profesor |
| `9bbdf32` | `e86309c` | 03-11-2026 14:52 | Fase 3 v7: fase rehecha con las cuatro reglas de la revisión en la cohorte oficial |
| `83fe557` | `4ff96a3` | 03-11-2026 22:01 | Fase 4 y 5: preparación para la cohorte v7 y la enmienda 03 |
| `fe53ca0` | `71589e9` | 03-11-2026 22:02 | Fase 4 (en curso): tablas parciales de la corrida con la cohorte v7 |
| `14c2224` | `64739d2` | 03-11-2026 23:27 | WIP Fase 4 v7: tablas parciales del ajuste (mortalidad completa, estadía en curso) |
| `670530d` | `83a4a7d` | 04-11-2026 00:27 | WIP Fase 4 v7: ajuste y comparación completos |
| `dedc0ba` | `cf06c62` | 04-11-2026 01:21 | WIP Fase 4 v7: selección, calibración e integridad |
| `8871633` | `32297e3` | 04-11-2026 01:33 | Fase 4 v7: modelos reentrenados con la cohorte de la Fase 3 v7 (G1–G8 PASS) |
| `ada010f` | `1d01f45` | 04-11-2026 01:34 | Fase 5 enmienda 03: registro previo a la segunda apertura de 2024 y planes B con la cohort |
| `a225dd9` | `3b051c9` | 04-11-2026 01:38 | Fase 5 enmienda 03: verificación previa (P1–P8 PASS) y segunda apertura autorizada de 2024 |
| `4143048` | `cd89b2e` | 04-11-2026 01:38 | Registro de accesos: segunda lectura concedida del desenlace de 2024 (enmienda 03) |
| `c1b609d` | `5b2eb19` | 04-11-2026 02:03 | Fase 5 enmienda 03: evaluación de 2024 (G1–G8 PASS) e incidente I-12 |
| `dcb1387` | `53a9516` | 04-11-2026 02:11 | Informes: Fase 4 v6 y anexo de resultados de la enmienda 03 |
| `97b7e5b` | `46cf7bb` | 08-11-2026 20:48 | Fase 6: datos y web con la cohorte de la enmienda 03 |
