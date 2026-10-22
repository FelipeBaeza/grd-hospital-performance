# Registro de incidentes

Un incidente es cualquier hecho que se aparta del protocolo, o que expuso información de 2024 o de la evaluación
fuera del procedimiento previsto. Cada entrada indica qué pasó, con qué evidencia, qué impacto tiene y qué se hace.
Están en orden cronológico. Las decisiones de diseño que cambiaron entre versiones de la Fase 1 no están aquí: se
registran en `docs/fase1_comprension_dominio/VERSIONES_Y_DESVIACIONES.md`.

Hora de referencia (UTC), 19-10-2026:

- 12:16:38: pre-registro del protocolo v1.0 (huella `5624bdae…`).
- 13:04:55: única lectura concedida del desenlace de 2024 (`registro_accesos.csv` de la Fase 3).

| ID | Fecha | Fase | Hecho | ¿Antes o después de abrir 2024? | ¿Cambia algún resultado? |
|---|---|---|---|---|---|
| I-01 | 27-09-2026 | Prototipo | Un prototipo anterior se evaluó con el desenlace de 2024 y se publicó en GitHub | Antes del sello; usa el desenlace de 2024 | No cambia cifras; 2024 deja de ser ciego para el autor |
| I-02 | antes del 16-10-2026 | F2 | Una versión del informe de la Fase 2 mostró la mortalidad de 2024 | Antes | No |
| I-03 | octubre de 2026 | F2 | La Fase 2 usó 2024 para describir predictores (Elixhauser, deriva de codificación) | Antes | No |
| I-04 | 18-10-2026 | F4 | El ajuste de Optuna se cortó dos veces y las reanudaciones no se registraron | Antes | No (mismo modelo elegido) |
| I-05 | 19-10-2026 | F5 | Lectura directa del archivo sellado (solo nombres de columnas) | Antes | No |
| I-06 | 19-10-2026, 15:06 | F5 | Cambios en el código de evaluación después de la apertura | Después | No (solo el reporte) |
| I-07 | 20-10-2026 | Auditoría | Relectura de 2024 en una reproducción independiente aislada | Después | No (resultados idénticos) |
| I-08 | 20-10-2026 | Auditoría | Sensibilidades exploratorias corridas antes de pre-registrar la enmienda 01 | Después | No; sus resultados ya se conocen |
| I-09 | 21-10-2026 | Repositorio | Publicación de valores reales en GitHub: la prueba de privacidad y tablas por hospital con celdas menores que 10 | Después | No; decisión de publicación pendiente |

---

## I-01. Un prototipo anterior se evaluó en 2024 y se publicó (27-09-2026)

**Qué pasó.** Antes del pipeline actual, el repositorio tenía un prototipo (`src/`, `reports/`, `models/`). El
27-09-2026 ese prototipo entrenó un LightGBM de mortalidad con 2019–2022, lo validó con 2023 y lo **probó en
2024**. También calculó el HSMR y la estadía ajustada **por hospital** para 2019–2024. Los resultados se subieron a
GitHub, que es público, ese mismo día:

| Commit | Hora (UTC) | Qué publicó |
|---|---|---|
| `652827f` | 20:58 | `reports/metrics_mortality.json` y `calibration_mortality_2024.csv`: AUC, PR AUC, Brier y O/E global de 2024 |
| `edbede3` | 21:01 | `hospital_benchmarks_2024.csv` y `hospital_benchmarks_all_years.csv`: muertes observadas y esperadas, y HSMR por hospital y año |

El mensaje del commit `652827f` cita el AUC de 2024. Las horas son las de los eventos de push de GitHub.

**Evidencia.** El historial de git (commits `652827f` y `edbede3`, y el `62c46a7` que los conserva) y los eventos
de push del repositorio.

**Impacto.**
- *Lo que se mantiene.*
  - Los criterios de aceptación vienen del informe de seminario (03-08-2026), anterior al prototipo.
  - El pipeline definitivo (Fases 2 a 5) se construyó después, del 05 al 19-10, con un cortafuegos que bloquea el desenlace de 2024.
  - Ese pipeline leyó 2024 una sola vez, a las 13:04:55 UTC del 19-10, después del pre-registro del protocolo.
- *Lo que cambia.* El autor conocía cómo se había comportado un modelo parecido en 2024 mientras diseñaba el pipeline definitivo. No se puede descartar que eso influyera en alguna decisión tomada entre el 27-09 y el 19-10, por ejemplo recalibrar siempre con 2023. Por eso **2024 no fue un año ciego para el autor**: fue un año reservado para el pipeline definitivo.

**Medida.**
1. Se declara. En los informes, «2024 nunca se vio» se reemplaza por «los modelos definitivos no usaron 2024 antes de la apertura; un prototipo anterior sí se evaluó en 2024 (I-01)».
2. La enmienda 01 usa como validación principal un diseño anidado que no toca 2024: entrenar con 2019–2021, calibrar con 2022 y probar en 2023.
3. Se recomienda pedir los egresos de 2025 y sellarlos como prueba limpia.
4. La privacidad de las tablas publicadas se trata en I-09.

---

## I-02. La mortalidad de 2024 apareció en una versión del informe de la Fase 2 (antes del 16-10-2026)

**Qué pasó.** Una versión del informe de la Fase 2 traía una tabla anual con la mortalidad cruda de 2024 (2,46 %).
Esa tabla no salió de la suite de la Fase 2 y contradecía el cortafuegos (DT03) y la regla R7. Lo más probable es
que viniera de los cálculos del prototipo de I-01.

**Evidencia.**
- `docs/historial/fase2/Fase2_Comprension_Datos_Analisis_Revisado.pdf` (16-10-2026), sección 8.2 y tabla de problemas, detectó la tabla.
- La versión del 18-10-2026 (`Fase2_Comprension_Datos_Base_Completa.docx`) la retiró y lo anota en su tabla de correcciones.
- La versión que contenía la cifra no se conserva en el proyecto.

**Impacto.** Es una cifra agregada de todos los egresos, no de las cohortes D y E. Ningún parámetro se ajustó con
ella: la calibración se fijó con 2023 y el protocolo prohíbe recalibrar con 2024.

**Medida.** Se declara.

---

## I-03. La Fase 2 usó 2024 para describir predictores (octubre de 2026)

**Qué pasó.** La suite de la Fase 2 usó los seis años, 2024 incluido, para la estructura y la calidad de los
datos, la prevalencia de las categorías de Elixhauser y la deriva de codificación. Por ejemplo, los diagnósticos
secundarios por egreso subieron de 4,16 en 2019 a 5,89 en 2024 (+41,8 %). El cortafuegos bloqueó el desenlace:
2024 no aportó muertes ni estadías.

**Evidencia.** Informe de la Fase 2, secciones 4.2 y 13.2; `fase2_eda/salidas/completo/tablas/t23_elix_prevalencia.csv`.

**Impacto.** Se conoció cómo se distribuían los predictores en el año de prueba, no su relación con el desenlace.
La deriva medida motivó el índice de exhaustividad, que solo se informa y no entra al modelo. No cambia ningún
resultado.

**Medida.** Se declara.

---

## I-04. Reanudaciones de Optuna sin registrar (Fase 4, 18-10-2026)

**Qué pasó.** El ajuste de XGBoost se cortó dos veces: en la prueba 28 de mortalidad y en la 18 de estadía. Se
reanudó sin anotarlo. Al reanudar, Optuna crea un muestreador nuevo y toma en cuenta la prueba que quedó abierta,
así que las pruebas siguientes no son las que daría una corrida de una sola vez (hallazgo H-19 de la auditoría).

**Evidencia.** `fase4_modelado/salidas/tablas/ajuste_pruebas_*_xgboost.csv` (faltan esas dos pruebas). La
reproducción independiente obtuvo las 140 pruebas exactas solo después de emular los dos cortes.

**Impacto.** Las tablas de XGBoost dependen de los cortes. El modelo elegido (LightGBM en las dos tareas) no
cambia.

**Medida.** Se declara. Cada reanudación se anotará en un registro, o se guardará el estado del muestreador (plan,
etapa 4.8).

---

## I-05. Lectura directa del archivo sellado (preparación de la Fase 5, 19-10-2026)

*Texto original del registro:*

> Durante la preparación, para conocer el esquema del archivo `desenlace_2024.parquet`, el asistente lo cargó
> completo con una lectura directa (fuera del cortafuegos de la Fase 3 y sin pasar por el registro de accesos).
> Solo se imprimieron los nombres de las columnas: no se mostró, calculó ni guardó ningún valor y el archivo no se
> volvió a abrir. El registro de accesos no registra este evento porque no pasó por el cortafuegos.
> Medida: se declara al profesor guía junto con la solicitud de autorización; el esquema se consulta en adelante
> solo por metadatos del archivo.

**Impacto.** Ninguno en las cifras. Muestra que el sello es de procedimiento, no técnico: una lectura directa no
deja rastro (H-16).

**Estado.** Informado al profesor guía en `autorizacion_apertura_2024.txt`. También figura en el protocolo v1.0.

---

## I-06. Cambios en el código de evaluación después de la apertura (19-10-2026)

**Qué pasó.** La huella pre-registrada cubre solo `config/protocolo_fase5.toml`, no el código. Después de la
apertura (13:04:55 UTC) se modificaron, con fecha de guardado 15:06 UTC:

- en `evaluacion/`: `analisis.py`, `figuras.py`, `hospitales.py`, `orquestador.py`, `privacidad.py`, `subgrupos.py` y `verificacion.py`;
- la prueba `tests/test_privacidad_derivada.py`.

Los cambios conocidos son tres:

1. La tabla de traslados salientes se recalculó sobre la cohorte C (con la cohorte D salía en cero).
2. Se agregó la supresión de conteos reconstruibles (observado = O/E × esperado).
3. Subieron las pruebas automáticas del cierre de 15 a 18. El informe de cierre dice 15; la tabla final dice 18.

**Evidencia.** Fechas de guardado de los archivos y `salidas/tablas/tabla_verificacion_cierre_fase5.csv`. No
existe una copia del código anterior a la apertura, así que no se puede publicar la diferencia exacta línea por
línea.

**Impacto.** Solo en el reporte. No cambió ningún criterio, modelo ni umbral del protocolo.

**Medida.** Se declara. Desde la enmienda 01, el registro incluye la huella de cada archivo de `evaluacion/`
(`registrar_enmienda.py`, hallazgo H-13).

---

## I-07. Relectura de 2024 en una reproducción independiente (auditoría, 20-10-2026)

**Qué pasó.** Para verificar que los resultados se reproducen, un auditor externo corrió el proyecto completo desde
los archivos originales en una copia aislada. En esa copia se leyó una vez el desenlace de 2024. La autorización
iba rotulada «reproducción independiente de auditoría» y se anotó en el registro de accesos de la copia, no en el
del proyecto.

**Evidencia.** Plan de corrección y reproducción independiente (`docs/auditoria/`). El registro de accesos del
proyecto no cambió (huella `e46fba7d…`).

**Impacto.** Las 12 tablas de la evaluación 2024 y las puertas G1–G8 salieron idénticas byte a byte. La relectura no
es un segundo examen: no se eligió ni se ajustó nada con ella.

**Medida.** Se declara al profesor guía junto con la enmienda 01.

---

## I-08. Sensibilidades exploratorias antes de la enmienda (auditoría, 20-10-2026)

**Qué pasó.** Sobre la reproducción, la auditoría corrió por adelantado:

- 20 variantes del modelo y 2 líneas base, en 2023 con el diseño anidado y en 2024 con el diseño original;
- 24 variantes del indicador.

Sus resultados ya se conocen (plan de corrección, §4) antes de que exista la enmienda que los pre-registra.

**Impacto.** La enmienda 01 no puede presentarse como ciega frente a esos resultados. Su valor es otro: fija **antes
de volver a correrlos** qué análisis se informan y con qué regla se leen, para que no se elija la variante que «sale
mejor».

**Medida.** La enmienda 01 lo declara en su sección 6. Se informan todos los análisis de su lista, cumplan o no la
regla. Si al repetirlos un resultado difiere del exploratorio, se informa la diferencia y su causa.

---

## I-09. Valores reales publicados en GitHub (21-10-2026)

El repositorio es público. El criterio DT04 del proyecto (no publicar conteos de 1 a 9 casos, ni conteos que se
puedan reconstruir) se rompió de dos maneras.

**a) La prueba de privacidad con valores reales.** `tests/test_privacidad_derivada.py` usaba el esperado y el O/E
de un hospital con muertes suprimidas, y con esos dos valores se reconstruye el conteo oculto (H-09).

- La corrección quedó en el commit de congelado `0953162`.
- Antes de eso, un commit de la primera subida (`7248d78`, rama `orden-v5`) estuvo público unos 15 minutos con la prueba original.
- La rama se reemplazó, pero GitHub sigue mostrando ese commit si se conoce su identificador.

**b) Tablas por hospital del prototipo (I-01).** El historial público conserva, desde el 27-09-2026, tablas con las
muertes observadas por hospital y año, de 2019 a 2024. Hay una celda menor que 10 en 2023 y otra en 2024. Estas
tablas ya no están en `main`, pero siguen en el historial.

**Impacto.** No hay datos de pacientes. Los microdatos de origen son públicos (GRD público de FONASA), así que esas
cifras también se pueden calcular desde la fuente. Aun así, el repositorio contradice la regla de privacidad que el
propio proyecto se fijó.

**Medida pendiente, decisión del autor.** Hay dos opciones:

1. **Reescribir el historial** para quitar esos archivos y la prueba original, y pedir al soporte de GitHub que borre las vistas en caché y la referencia del pull request #1. Cambian los identificadores de todos los commits posteriores al 27-09, también los que citan los informes.
2. **Mantener el historial y declararlo aquí**, como queda hoy.

Mientras no se decida, se mantiene la opción 2.
