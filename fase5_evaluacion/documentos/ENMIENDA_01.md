# Enmienda 01 al protocolo de la Fase 5

Archivo de la enmienda: `config/protocolo_fase5_v2.toml`. Protocolo base: `config/protocolo_fase5.toml` v1.0
(huella `5624bdae…`), que sigue vigente.

> **Validez.** Este texto y `protocolo_fase5_v2.toml` valen solo cuando existe
> `documentos/registro_enmienda_01.json`, que registra sus huellas, la firma del autor y la constancia del profesor
> guía. Hasta entonces son un borrador: se pueden corregir, pero ningún análisis de la lista se corre como
> resultado.

## 1. Por qué existe

La auditoría metodológica de las Fases 1 a 5 y la reproducción independiente (20 y 21-10-2026) confirmaron que los
resultados se reproducen byte a byte. También encontraron decisiones que conviene poner a prueba: los institutos
monográficos en el embudo, la winsorización, las comorbilidades que pueden ser complicaciones, las variables
condicionales y las sensibilidades prometidas que no se hicieron.

El desenlace de 2024 ya se abrió (19-10-2026). Además, 2024 no era un año ciego para el autor: un prototipo
anterior se había evaluado en ese año el 27-09-2026 (incidente I-01). Por eso ninguna corrección puede presentarse
como probada a ciegas en 2024. Esta enmienda fija **antes de correrlos** la lista cerrada de análisis, sus métricas
y la regla para leerlos.

## 2. Qué no cambia

Los modelos congelados de la Fase 4, el protocolo v1.0 y las tablas de `salidas/evaluacion_2024/` son el
**resultado primario** de la tesis. No se modifican, no se re-etiquetan y no se reemplazan, aunque un análisis de
esta enmienda «salga mejor».

## 3. Diseños

| Diseño | Entrenar | Calibrar | Probar | Rol |
|---|---|---|---|---|
| **Anidado** | 2019–2021 | 2022 | 2023 | Resultado de la enmienda: no usa 2024 |
| Original | 2019–2022 | 2023 | 2024 | Secundario, rotulado «ya abierto» |

Cada variante del modelo usa la receta congelada: el LightGBM de cada tarea con sus hiperparámetros y su número de
árboles, y la misma recalibración (Platt en mortalidad, factor único en estadía). Cada análisis se compara con la
**variante base del mismo diseño y la misma tarea**. Las cohortes son las de la Fase 3 y del protocolo v1.0, y el
indicador es el embudo del protocolo v1.0, salvo en los análisis que cambian justamente eso.

## 4. Lista cerrada de análisis

«Ya conocido» significa que la auditoría lo corrió como exploración y su resultado se sabe (sección 6).

| ID | Análisis | ¿Reentrena? | Regla de lectura | Ya conocido | Hallazgo |
|---|---|---|---|---|---|
| 2.1 | Sin las 5 categorías de Elixhauser sospechosas de la Fase 1 (ELIX_04, 22, 24, 25 y 26) | Sí | Sí | Sí | H-02 |
| 2.2 | Sin 2020 y 2021 en el entrenamiento | Sí | Sí | Sí | H-04 |
| 2.3 | Sin las 4 variables condicionales (`TIPO_INGRESO`, `TIPO_PROCEDENCIA`, `PROCEDENCIA_AGR`, `NIVEL_CUIDADO_INGRESO`) | Sí | Sí | Sí | H-10 |
| 2.4 | Sin ninguna comorbilidad secundaria (O/E «resistente a la codificación») | Sí | Sí | Sí | H-02, H-12 |
| 2.5 | Línea base en vez de LightGBM, a nivel de hospital. En el diseño original se usan los planes B congelados; en el anidado, la receta de la línea base | Solo en el anidado | Sí | Sí | Pregunta 8 del comité |
| 2.6 | Sin traslados entrantes en la evaluación | No | Sí | Sí | H-03 |
| 3.1a | Sin los institutos monográficos (110110, 112103 y 112104) en el embudo general; se informan aparte | No | Sí | Sí | H-01 |
| 3.1b | Grupos de pares: un embudo por tercil de egresos | No | Sí | Sí | H-01, H-06 |
| 3.2 | Marca de los hospitales con menos de 2 años en GRD | No | No (se informa cuántos hay en cada zona) | No | H-06 |
| 3.3 | Re-basado anual del esperado (suma de observados igual a suma de esperados) | No | Sí | Sí | H-11 |
| 3.4a | Embudo sin winsorizar | No | Sí | Sí | H-05 |
| 3.4b | Embudo con truncamiento al 10 % por cola, al estilo SHMI | No | Sí | No | H-05 |
| 3.4c | Validación del embudo con la heterogeneidad real (puerta P5b): 400 réplicas, semilla 7 | No | No (se informa la proporción esperada fuera del 95 %) | Sí | H-05 |
| 3.5 | Estabilidad del O/E por hospital entre 2023 y 2024 | No | No (se informan la correlación y las alertas repetidas) | Sí | Auditoría §3.4 |
| 3.6 | Calibración por estrato: tercil de egresos, general o monográfico, años en GRD | No | No (O/E por estrato con intervalo) | En parte | Auditoría §3.1 |

La lista de los institutos monográficos es provisional: se tomó de los nombres de `Hospitales.csv`. Se reemplaza
por el catálogo DEIS cuando exista (plan, etapa 1.3), y ese cambio se declara.

## 5. Métricas

- **Modelo de mortalidad:** AUC, PR AUC, Brier Skill Score, pendiente de calibración y O/E global con intervalo del 95 %.
- **Modelo de estadía:** RMSE, devianza de Poisson y O/E global con intervalo del 95 %.
- **Indicador:**
  - porcentaje de hospitales fuera del 95 % y del 99,8 %, y τ²;
  - porcentaje de hospitales en la misma zona que la base y correlación de rangos (Spearman) del O/E por hospital;
  - cuántos hospitales cambian de zona, y en qué dirección (de fuera a dentro o de dentro a fuera).

Los intervalos usan 1.000 réplicas bootstrap de hospitales completos, como el protocolo v1.0.

## 6. Lo que ya se conoce

La auditoría corrió por adelantado la mayoría de estos análisis (incidente I-08; plan de corrección, §4). En
resumen, la conclusión resistió casi todo. Solo dos análisis no cumplieron la regla, y los dos en mortalidad:

- 2.3, sin las variables condicionales: ρ 0,80 en 2024 y 0,89 en el diseño anidado;
- 2.4, sin ninguna comorbilidad: ρ 0,88 en 2024.

Por eso esta enmienda **no es ciega**. Lo que garantiza es otra cosa:

1. Se informan todos los análisis de la lista, cumplan o no la regla.
2. Nada que no esté en la lista se presenta como resultado; a lo más, como exploración rotulada.
3. Si al repetirlos un resultado difiere del exploratorio, se informa la diferencia y su causa.

## 7. Regla de lectura

La conclusión de la Fase 5 se mantiene frente a un análisis si se cumplen las dos condiciones:

- al menos el **80 %** de los hospitales conserva su zona (dentro de lo esperado, alerta alta o baja, señal alta o baja);
- la correlación de rangos del O/E por hospital con la base es de al menos **0,90**.

Si no se cumple, se informa que el indicador no es robusto a esa decisión. El resultado primario no se reemplaza.
Una variante que resulte mejor solo puede adoptarse como **versión 2** para el despliegue, y debe validarse en un año
nuevo (2025). Nunca se elige entre variantes mirando 2024.

## 8. Qué se publica

Las tablas por hospital siguen las reglas de supresión del protocolo v1.0: celda mínima de 10, también para los
conteos que se puedan reconstruir. Se publican el resumen de cada análisis y los códigos de los hospitales que
cambian de zona, sin conteos.

## 9. Fuera de esta enmienda

Las correcciones de datos y cohortes quedan para una **enmienda 02**, que se registra antes de correrlas:

- alta voluntaria y fuga en la cohorte E (H-08);
- la exclusión obstétrica EX03, que necesita la contraparte clínica (H-07);
- el catálogo de tipo de establecimiento DEIS (H-01, H-18);
- las tasas de contexto por hospital: paliativos Z51.5, hospitalización domiciliaria y alta voluntaria (H-15).

## 10. Cómo se registra y en qué orden

1. El autor y el profesor guía revisan este texto y `protocolo_fase5_v2.toml`, y corrigen lo que haga falta.
2. Con el texto final, `python registrar_enmienda.py huellas` muestra las dos huellas que van en la constancia del profesor (Parte C).
3. El profesor firma la constancia. El autor la guarda escaneada en `documentos/constancia_profesor_guia_firmada.pdf`.
4. `python registrar_enmienda.py firmar --autor "…" --profesor "…" --fecha-firma AAAA-MM-DD` crea `documentos/registro_enmienda_01.json`. Ese registro guarda:
   - las huellas de los dos archivos de la enmienda y de la constancia;
   - la huella de cada archivo de `evaluacion/`;
   - la hora.
5. El registro se sube a git con la etiqueta `enmienda-01`.
6. Antes de la primera corrida, `python registrar_enmienda.py antes-de-correr <scripts>` registra la huella del código que corre los análisis. Si la enmienda cambió desde la firma, se detiene.
7. Recién entonces se corren los análisis. Sus resultados se informan en un anexo de la Fase 5, rotulados «enmienda 01».
