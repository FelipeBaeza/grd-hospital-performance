# Enmienda 02 al protocolo de la Fase 5: datos y cohortes

**Estado: borrador.** Archivo de la enmienda: `config/protocolo_fase5_v3.toml`. Protocolo base: la enmienda 01
(`config/protocolo_fase5_v2.toml`, huella `0f199fb8…`), que a su vez se apoya en el protocolo v1.0
(`5624bdae…`). Los dos siguen vigentes.

> **Validez.** Este texto y `protocolo_fase5_v3.toml` valen solo cuando existe
> `documentos/registro_enmienda_02.json`, que registra sus huellas, la firma del autor y la aprobación del
> profesor guía. Hasta entonces son un borrador: se pueden corregir, pero ningún análisis de la lista se corre
> como resultado.

## 1. Por qué existe

La enmienda 01 dejó fuera, a propósito, las correcciones que cambian **quién entra a las cohortes** o **cómo se
agrupan los hospitales** (su §9). Son las de la etapa 1 del plan de corrección:

| Plan | Problema | Hallazgo |
|---|---|---|
| 1.1 | La configuración dice que el alta voluntaria y la fuga van solo en una sensibilidad de estadía (`rol_estancia = INCLUIDO_SENSIBILIDAD`), pero ningún módulo lee esa columna. Ambas entran a la cohorte E principal: son alrededor del 2 % de E | H-08 |
| 1.2 | EX03 excluye solo los obstétricos **sin** comorbilidades de Elixhauser. El resto del MDC 14 (entre el 5 y el 7 % de la cohorte D, según el año) queda en el modelo, mezclado con adultos que no son obstétricos. Nadie con formación clínica validó esa regla | H-07 |
| 1.3 | El informe habla de hospitales «de alta y mediana complejidad», pero la población incluye institutos monográficos y hospitales comunitarios. La lista de institutos de la enmienda 01 es provisional: se armó con los nombres de `Hospitales.csv`, que no trae el nivel de complejidad | H-01, H-18 |
| 1.4 | Varios hospitales entraron a GRD en 2023 o 2024. En su primer año la codificación todavía se está ajustando | H-06 |
| 1.5 | Prácticas que mueven el O/E sin cambiar la calidad: codificar cuidados paliativos (Z51.5), dar el alta a hospitalización domiciliaria y el alta voluntaria | H-15 |

## 2. Qué no cambia

Lo mismo que en la enmienda 01: los modelos congelados de la Fase 4, el protocolo v1.0 y las tablas de
`salidas/evaluacion_2024/` son el **resultado primario** de la tesis. No se modifican ni se reemplazan.

El código de las Fases 3 y 5 que cambia con esta enmienda lo hace **detrás de una opción que viene apagada**. Sin
la opción, las cascadas, las capas Oro y las cohortes evaluadas quedan idénticas byte a byte. Las pruebas lo
verifican.

## 3. Decisiones ya tomadas

- **EX03 (1.2), opción conservadora.** Se excluye el **MDC 14 completo**, no solo los obstétricos sin
  comorbilidades. La alternativa, definir el parto sin complicación con los GRD del catálogo IR-GRD, necesita una
  contraparte clínica que el proyecto no tiene. Excluir el MDC 14 completo es más fácil de defender y se declara
  como limitación: el indicador deja de cubrir la atención obstétrica. Decisión del autor del 21-10-2026.
- **Alta voluntaria y fuga (1.1).** Salen de la cohorte E, como ya decía la configuración. En la cohorte D
  (mortalidad) siguen como «no evento», sin cambios.

## 4. Diseños y base de comparación

Los mismos dos diseños de la enmienda 01, con los mismos roles:

| Diseño | Entrenar | Calibrar | Probar | Rol |
|---|---|---|---|---|
| **Anidado** | 2019–2021 | 2022 | 2023 | Resultado de la enmienda: no usa 2024 |
| Original | 2019–2022 | 2023 | 2024 | Secundario, rotulado «ya abierto» |

La receta es la misma: el LightGBM congelado de cada tarea, con sus hiperparámetros, su número de árboles y su
recalibración. Cada análisis se compara con la **base del mismo diseño y la misma tarea**, que es la base de la
enmienda 01.

Cuando un análisis cambia una cohorte, el cambio se aplica a la vez en entrenamiento, calibración y prueba. Así
se compara el indicador que se obtendría con esa regla. La comparación por hospital se hace sobre los hospitales
elegibles en las dos versiones.

## 5. Lista cerrada de análisis

Ninguno de estos análisis se corrió antes, ni como exploración: a diferencia de la enmienda 01, en esta lista no
hay nada «ya conocido».

| ID | Análisis | ¿Reentrena? | Regla de lectura | Plan |
|---|---|---|---|---|
| E2.1 | Sin alta voluntaria ni fuga en la cohorte E | Sí, solo estadía | Sí | 1.1 |
| E2.2 | Sin el MDC 14 completo en las cohortes D y E | Sí, las dos tareas | Sí | 1.2 |
| E2.3 | E2.1 y E2.2 juntas: la cohorte que tendría una versión 2 del indicador | Sí, las dos tareas | Sí | 1.1 y 1.2 |
| E2.4a | Grupos de pares por tipo de establecimiento DEIS: un embudo por grupo | No | Sí | 1.3 |
| E2.4b | Análisis 3.1a de la enmienda 01, repetido con la lista de institutos del catálogo DEIS | No | Sí | 1.3 |
| E2.5 | Embudo sin los hospitales que llevan menos de 2 años en GRD; se informan aparte | No | Sí | 1.4 |
| E2.6a | Tasas de contexto por hospital (Z51.5 en cualquier diagnóstico, alta a hospitalización domiciliaria, alta voluntaria y fuga) y su correlación de rangos con el O/E | No | No (se informan las correlaciones) | 1.5 |
| E2.6b | Mortalidad sin los episodios con Z51.5 en cualquier diagnóstico | No | Sí | 1.5 |

**Detalle de cada análisis:**

- **E2.1.** La regla sale de `fase3_preparacion/config/tipoalta_roles.csv`: se excluyen los tipos de alta con
  `rol_estancia = INCLUIDO_SENSIBILIDAD`, hoy `ALTA VOLUNTARIA` y `FUGA DEL PACIENTE`.
- **E2.2.**
  - Se informa cuántas muertes obstétricas salen de la cohorte D, sumadas en todos los años (2019–2023: 46),
    sin desglose por hospital ni por año.
  - Un modelo sin MDC 14 conserva las categorías de las variables, pero no ve episodios obstétricos.
- **E2.4a y E2.4b** necesitan `fase3_preparacion/config/tipo_establecimiento.csv`. Ese archivo clasifica los
  72 códigos de hospital desde el registro de establecimientos del DEIS, con la fuente y la fecha de descarga.
  - Su huella se registra antes de correr los dos análisis.
  - Si el catálogo no existe al registrar la enmienda, E2.4a y E2.4b se corren después, sin cambiar la lista,
    apenas se agregue y se registre su huella.
- **E2.5.** El primer año en GRD sale de `catalogo_hospitales_anio.csv`: el primer año con egresos. La columna
  `anios_en_grd` se agrega al catálogo de la Fase 3.
- **E2.6a.**
  - Las tasas son por hospital y por año evaluado, sobre la cohorte C (adultos hospitalizados). No se usa la
    cohorte D porque el alta a hospitalización domiciliaria está censurada y no entra a ella.
  - Por hospital se publica la tasa solo si su conteo es 0 o al menos 10. Si el conteo va de 1 a 9, la tasa se
    suprime: la tasa por el número de egresos lo reconstruiría.
- **E2.6b.** El modelo no se reentrena: se quitan esos episodios de la cohorte evaluada, como en el análisis 2.6
  de la enmienda 01. En 2019–2023, esos episodios son alrededor del 3 % de las muertes de la cohorte D.

## 6. Métricas, regla de lectura y publicación

Son las de la enmienda 01 (§5, §7 y §8), sin cambios:

- **Regla de lectura:** al menos el 80 % de los hospitales conserva su zona y la ρ de Spearman del O/E es de al
  menos 0,90.
- **Intervalos:** 1.000 réplicas bootstrap de hospitales completos.
- **Publicación:**
  - Los resúmenes por análisis, sin tablas por hospital con O/E ni observados.
  - En mortalidad, el O/E de un estrato no se publica si el estrato contiene un hospital con conteo suprimido.

Estos análisis miden si el indicador resiste las correcciones de la auditoría; no buscan mejorar el modelo.
**Ningún análisis de la lista se modifica, se repite con otra configuración ni se descarta porque no mejore el
modelo.** Si E2.3 resultara «mejor», no reemplaza al resultado primario: es la candidata a versión 2 del
indicador y debe validarse en un año nuevo (2025) antes de usarse.

## 7. Cómo se registra

Igual que la enmienda 01:

1. El autor y el profesor guía revisan este texto y `protocolo_fase5_v3.toml`, y lo corrigen.
2. `python registrar_enmienda.py --enmienda 2 huellas` muestra las dos huellas.
3. `python registrar_enmienda.py --enmienda 2 firmar …` crea `documentos/registro_enmienda_02.json`.
4. Antes de la primera corrida, `python registrar_enmienda.py --enmienda 2 antes-de-correr <scripts>` registra la
   huella del código. El código se sube a git **antes** de correr, para que la hora de GitHub sirva de prueba.
5. Se corren los análisis y se informan en un anexo de la Fase 5, rotulados «enmienda 02».

## 8. Fuera de esta enmienda

- **El catálogo DEIS en sí.** Es un dato de entrada, no un análisis. Se descarga del DEIS y se guarda con su
  fuente y fecha.
- **El año 2025.** Cuando FONASA lo entregue, la versión 2 del indicador (E2.3, si se adopta) se valida en ese
  año con un protocolo nuevo, pre-registrado y sellado antes de abrirlo.
