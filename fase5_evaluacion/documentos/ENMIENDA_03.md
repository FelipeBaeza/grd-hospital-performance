# Enmienda 03 al protocolo de la Fase 5: nueva evaluación en 2024 con la cohorte de la Fase 3 v7

Archivo de la enmienda: `config/protocolo_fase5_v4.toml`. Protocolo base: v1.0 (`config/protocolo_fase5.toml`, huella
`5624bdae…`). Las enmiendas 01 y 02 siguen vigentes para sus propios análisis.

> **Validez.** Este texto y `protocolo_fase5_v4.toml` valen cuando existe `documentos/registro_preapertura_v4.json`,
> con la huella del protocolo, de los modelos y del registro de accesos, escrito **antes** de la segunda lectura del
> desenlace de 2024. Después no se cambian: un cambio exige una enmienda nueva.
>
> **Aprobación del profesor guía.** El autor informó el 03-11-2026 que el profesor guía dio su visto bueno a las cuatro
> reglas de la cohorte y a volver a evaluar 2024 (`documentos/autorizacion_reapertura_2024.txt`). Falta adjuntar la
> constancia escrita del propio profesor, como en la apertura de octubre.

## 1. Por qué existe

La segunda revisión técnica del profesor guía (02-11-2026, sobre la Fase 2 v7) observó cuatro reglas de la cohorte
que pesan en la comparación entre hospitales:

| Observación | Regla de la cohorte v7 | Paso en la Fase 3 |
|---|---|---|
| 1. Censura | La hospitalización domiciliaria deja de ser censura: egresó vivo. No entra al modelo de estadía | EX06 y EX11 |
| 7. Filas repetidas | Las filas repetidas por clave natural (Q15) salen de la cohorte de mortalidad | EX10 |
| Decisión 16 | Los traslados entrantes enlazados con un traslado saliente previo (30 días, otro hospital, mismo régimen de identificadores) salen de la cohorte de mortalidad | EX12 |
| 4. Regla obstétrica | Sale todo el MDC 14 | EX03 |

La Fase 3 se rehízo con esas reglas (v7, umbrales v2.0) y la Fase 4 volvió a entrenar y calibrar los modelos con la
nueva cohorte, con las mismas reglas de la Fase 4 (`reglas_fase4.toml` sin cambios). Esta enmienda evalúa esos
modelos en 2024.

## 2. Qué cambia y qué no

- **No cambia:** los siete criterios, sus umbrales, la regla del veredicto, la semilla, las 1.000 réplicas, el volumen
  mínimo (500 egresos y 10 muertes esperadas, con el modelo final) y el embudo.
- **Cambia:** la cohorte (las cuatro reglas), los modelos y los planes B (Fase 4 rehecha) y el número de lecturas del
  desenlace de 2024 que el protocolo permite: **dos en total**. La primera fue la apertura del protocolo v1.0
  (19-10-2026, 13:04 UTC).
- **Se conserva:** el protocolo v1.0 y sus resultados en `salidas/evaluacion_2024/`. Los de esta enmienda van a
  `salidas/evaluacion_2024_v4/` y sus tablas de puertas llevan el sufijo `_v4`.

## 3. Cómo se aplican las reglas en 2024

La Fase 3 no puede aplicar en 2024 las reglas que usan el desenlace, porque está sellado. La Fase 5 las aplica con el
desenlace que lee en la apertura autorizada, con el mismo código de la Fase 3 (`prep/cohortes.py`):

- la hospitalización domiciliaria se reconoce por su tipo de alta;
- las filas repetidas usan la fecha de alta de 2024;
- el enlace de traslados busca salientes de 2023 (Plata) y de 2024 (el desenlace abierto), dentro del régimen 2021–2024.

El ensayo con 2023 comprueba que este camino da exactamente las cohortes D y E de la Fase 3 (mismos identificadores).

## 4. Lo que hay que declarar

- **2024 no es ciego.** El autor y el asistente conocen los resultados del protocolo v1.0. Esta es una segunda mirada
  al mismo año de prueba: no reemplaza al resultado primario ni se presenta como una validación independiente.
- **Las reglas se eligieron después de ver datos de 2019 a 2024.** Las propuso la revisión del profesor a partir de la
  Fase 2, que ya había leído la estructura de 2024 y algunos conteos de calidad con su fecha de alta (Fase 2 v8,
  sección 6.3). Ninguna regla se eligió mirando el O/E de 2024.
- **Las enmiendas 01 y 02 no se repiten con la cohorte v7.** Sus análisis de sensibilidad siguen hechos con la cohorte
  anterior.
