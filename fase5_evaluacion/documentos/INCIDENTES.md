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
| I-10 | Después del 19-10-2026 | F5 | El profesor guía autorizó por escrito después de conocer los resultados de 2024 | Después | No cambia cifras; la autorización previa no operó |
| I-11 | 22-10-2026 | F5 y F6 | El escáner por propiedad encontró conteos nacionales de 1 a 9 reconstruibles por complemento en las tablas por subgrupo | Después | No; tablas regeneradas el mismo día |

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

**Resultado (22-10-2026).** La corrida registrada de la enmienda 01 repitió los análisis con el código registrado
antes de correr (commit `b718454`). Las 40 comparaciones posibles con la exploración (% en la misma zona y ρ, por
análisis, tarea y diseño) son idénticas (`salidas/enmienda_01/comparacion_con_exploracion.csv`).

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

---

## I-10. El profesor guía autorizó después de conocer los resultados de 2024

**Qué pasó.** La apertura del 19-10-2026, a las 13:04:55 UTC, se activó con `autorizacion_apertura_2024.txt`. Ese
archivo lo firmó el estudiante e «informa la aprobación del profesor guía». A su vez, `DECISIONES_FASE5.md` dice que
el estudiante tomó las decisiones del profesor como aprobatorias.

El profesor guía dio su autorización **después de conocer los resultados de 2024**, según lo informó el autor el
21-10-2026. Su constancia escrita es, por lo tanto, una **ratificación posterior** y no una autorización previa.

**Evidencia.**
- `autorizacion_apertura_2024.txt` y `DECISIONES_FASE5.md` (19-10-2026).
- La declaración del autor (21-10-2026).
- La Parte A de la constancia del profesor, con la fecha de su primera aprobación (pendiente de firma).

**Impacto.** No cambia ninguna cifra. Los criterios y la regla de veredicto se pre-registraron antes de abrir
(12:16:38 UTC) y los modelos ya estaban congelados. Lo que no operó fue la salvaguarda de gobernanza: que un tercero
autorizara la apertura antes de que ocurriera. Por eso la evaluación de 2024 se describe así: «realizada con la
aprobación informada por el estudiante y ratificada por el profesor guía después de conocer los resultados».

**Medida.**
1. La Parte A de la constancia se firma como ratificación posterior e indica cuándo dio el profesor su primera aprobación.
2. Para un año de prueba nuevo (2025), la autorización del profesor se registra antes de abrir. Idealmente se usa el sello criptográfico del plan (etapa 4.5), para que sin su clave el año no se pueda leer.
3. Por decisión del autor y del profesor (21-10-2026), los análisis de la enmienda 01 se informan tal cual resulten. No se modifican ni se descartan porque no mejoren el modelo.

---

## I-11. Conteos pequeños reconstruibles por complemento en las tablas por subgrupo (22-10-2026)

**Qué pasó.** El escáner de privacidad por propiedad (`evaluacion/privacidad_propiedad.py`, plan 4.2) revisó todo lo
publicado por las Fases 3 a 6. Encontró 13 casos del mismo tipo, todos en las tablas de calibración por subgrupo de
pacientes y en su copia de la Fase 6:

- `salidas/ensayo_2023/tablas/subgrupos_*.csv`;
- `salidas/evaluacion_2024/tablas/subgrupos_*.csv`;
- `fase6_despliegue/datos/publicables/subgrupo.csv`.

Cada tabla divide la misma cohorte según varias variables (edad, sexo, comorbilidades, grupo clínico y tamaño del
hospital). Una variable con todos sus grupos publicados da el total. En otra variable, lo que falta para llegar a ese
total es un conteo de 1 a 9, en dos situaciones:

- **Celda oculta:** el grupo «sexo sin dato» o un grupo clínico poco frecuente tienen el observado suprimido.
- **Fila omitida:** el grupo no se publicó por tener menos de 10 egresos.

La lista exacta estaba en `config/privacidad_hallazgos_declarados.csv`, sin los conteos (commit `ba759d1`).

**Impacto.** Ningún resultado cambia.

- Los conteos son **nacionales**, de pacientes con sexo sin dato o de grupos clínicos raros; no son por hospital.
  El riesgo de identificar a alguien es bajo.
- Aun así, el criterio DT04 del proyecto prohíbe publicar conteos de 1 a 9, también los reconstruibles. Igual que en
  I-09, las tablas están en el repositorio público.
- El escáner original no lo detectaba porque revisa patrones dentro de un archivo, no lo que se puede deducir
  combinando celdas.

**Medida (decisión del autor, 22-10-2026: regenerar las tablas).**

1. `evaluacion/subgrupos.py` ya no oculta ni omite grupos pequeños: los **combina**. Los grupos con menos de 10
   egresos o, en mortalidad, con menos de 10 muertes se juntan en un grupo «A o B». Si el combinado sigue siendo
   pequeño, se le suma el grupo publicado que menos agrega (en mortalidad, el de menos muertes). Así no queda ninguna
   celda oculta, y ningún complemento revela un conteo pequeño. El método anterior sigue disponible
   (`agrupar_pequenos=False`) y una prueba comprueba las dos cosas.
2. `regenerar_subgrupos.py` rehízo las 4 tablas de la Fase 5 (ensayo 2023 y evaluación 2024) con los modelos
   congelados y la guardia sin ajustes.
   - **Control:** antes de escribir comprobó que el método anterior reproduce byte a byte las tablas publicadas.
   - **Registro:** las huellas anteriores y nuevas están en `salidas/tablas/regeneracion_subgrupos_I-11.csv`.
   - **Lectura de 2024:** para 2024 volvió a leer el desenlace ya abierto (`trabajo5`), con la autorización del autor.
3. `fase6_despliegue/datos/publicables/subgrupo.csv` se rehízo desde las tablas nuevas, igual que en
   `exportar_publicables.py`, y se actualizó su huella en `huellas.json`.
4. **Qué cambió en las tablas:**
   - Sexo: el grupo «sin dato», cuando es pequeño, se une a otro sexo.
   - Grupos clínicos: los capítulos raros se unen entre sí y, en mortalidad 2024, con el capítulo XV.
   - Ningún otro valor cambia. El capítulo XXII de mortalidad 2024 sigue igual (O/E 0,58). En estadía 2024 pasa a
     ser «XXII o sin capítulo», con menos de diez egresos más.
5. El escáner por propiedad ya no encuentra ningún conteo reconstruible en lo publicado, y la lista de hallazgos
   declarados queda vacía. Las puertas P7 y G7 fallan ante cualquier hallazgo nuevo.
6. Las versiones anteriores de esas tablas siguen en el **historial público** de git. Reescribirlo o no es la misma
   decisión pendiente de I-09.

