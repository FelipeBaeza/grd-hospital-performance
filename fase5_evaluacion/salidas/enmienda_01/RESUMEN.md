# Enmienda 01: resumen de resultados

Los 15 análisis de la enmienda 01 (`documentos/ENMIENDA_01.md`) se corrieron una sola vez, el 22 de octubre de 2026
entre las 02:25 y las 04:10 UTC. El código se registró y se subió a GitHub antes (commit `b718454`, 02:24 UTC). El
informe completo está en `docs/fase5_evaluacion/Anexo_Enmienda01_resultados.docx`.

El resultado primario (modelos congelados, protocolo v1.0 y `salidas/evaluacion_2024/`) no cambia.

## En una frase

El indicador cumple la regla de lectura en **41 de 44** combinaciones de análisis, tarea y diseño. La regla pide que
al menos el 80 % de los hospitales conserve su zona y que la ρ del O/E sea de al menos 0,90. Las 3 combinaciones que
no cumplen son de mortalidad:

| Análisis | Diseño anidado (2023) | Diseño original (2024, ya abierto) |
|---|---|---|
| 2.3 Sin las 4 variables condicionales | 92,3 % · ρ 0,891 | 91,3 % · ρ 0,802 |
| 2.4 Sin ninguna comorbilidad | cumple (96,9 % · ρ 0,909) | 88,4 % · ρ 0,879 |

## Lo demás

- **Algoritmo (2.5).** La logística deja al 94–100 % de los hospitales en la misma zona que LightGBM, con ρ 0,98 en
  las dos tareas. El indicador no depende del algoritmo.
- **Comorbilidades sospechosas (2.1), pandemia (2.2), traslados entrantes (2.6), institutos (3.1a), re-basado (3.3)
  y método del embudo (3.4a y 3.4b):** todos cumplen.
- **Grupos de pares por volumen (3.1b).**
  - En estadía, los hospitales fuera del 95 % bajan de 12,3 a 1,5 % en 2023 y de 15,9 a 5,8 % en 2024.
  - Todos los que pasan de fuera a dentro son del tercil de menos egresos.
- **Calibración por estrato (3.6).** El O/E de estadía es 0,864 en 2023 y 0,850 en 2024 en el tercil de menos
  egresos, frente a 0,957–0,979 en los otros. En los hospitales con menos de 2 años en GRD es 0,794 y 0,829. El
  O/E de estadía no es comparable entre hospitales de distinto tamaño.
- **Validación del embudo (3.4c).** Con la variación real entre hospitales, el embudo del protocolo deja fuera del
  95 % entre el 11,6 y el 13,2 % aunque ningún hospital sea anómalo; sin winsorizar, cerca del 5 %.
- **Estabilidad (3.5).** ρ del O/E entre 2023 y 2024: 0,821 en mortalidad y 0,882 en estadía. De las alertas de
  2024, 8 de 11 (mortalidad) y 6 de 9 (estadía) ya estaban en 2023, en la misma dirección.
- **Exploración de la auditoría (incidente I-08).** Las 40 comparaciones posibles son idénticas a las
  exploratorias.

## Archivos

| Archivo | Contenido |
|---|---|
| `resumen_modelo.csv` | 2.1 a 2.5 |
| `resumen_indicador.csv` | 2.6 y 3.1 a 3.4 |
| `calibracion_por_estrato.csv` | 3.6 (en mortalidad, sin el O/E de los estratos con un hospital suprimido) |
| `validacion_embudo_p5b.csv` | 3.4c |
| `estabilidad_2023_2024.csv` | 3.5 |
| `institutos_y_hospitales_nuevos.csv` | 3.1a y 3.2 |
| `comparacion_con_exploracion.csv` | Comparación con la exploración |
| `metadatos.json` | Huellas, horas, semillas y versiones |
