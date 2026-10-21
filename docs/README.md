# Documentos del proyecto

## Orden de lectura

1. `00_guia_del_proyecto/Guia_del_proyecto.docx` — la guía de entrada.
2. `fase1_comprension_dominio/Informe_Fase1_v5.docx` — diseño, reglas y decisiones (con los cambios y sus fechas).
3. `fase5_evaluacion/Informe_Fase5_v5.docx` — resultados en 2024 y cómo leerlos.
4. `fase2_comprension_datos/Informe_Fase2_v5.docx`, `fase3_preparacion_datos/Informe_Fase3_v5.docx` y
   `fase4_modelado/Informe_Fase4_v5.docx`, según lo que se necesite revisar.
5. `auditoria/` — auditoría metodológica externa y resumen del plan de corrección.

Los informes **v5** describen solo lo que se ejecutó, con cifras comprobadas en una reproducción independiente
desde cero (octubre de 2026).

## Qué hay en cada carpeta

| Carpeta | Contenido |
| --- | --- |
| `00_guia_del_proyecto/` | Guía del proyecto |
| `00_propuesta_seminario/` | Informe de seminario (agosto de 2026): el punto de partida |
| `fase1_comprension_dominio/` | Informe v5 y tablas de la Fase 1 |
| `fase2_comprension_datos/` | Informe v5 y tablas de respaldo de la Fase 2 |
| `fase3_preparacion_datos/` | Informe v5 y plan original de la Fase 3 |
| `fase4_modelado/` | Informe v5 y plan original de la Fase 4 |
| `fase5_evaluacion/` | Informe v5 (preparación y evaluación) y plan original de la Fase 5 |
| `fase6_despliegue/` | Plan de la Fase 6 (en desarrollo) |
| `auditoria/` | Auditoría externa y resumen del plan de corrección |
| `historial/` | Versiones anteriores de los informes, sin cambios |
| `comunicaciones_fonasa/` | Borradores de cartas a FONASA (no enviadas; solo en la copia local, fuera de git) |

Los **planes** de cada fase se conservan como se escribieron: registran lo que se planificó, no lo que se hizo.

## Archivos que el código lee (no mover ni renombrar)

La Fase 2 (`fase2_eda/eda/ejecucion.py`) lee estos archivos como insumo:

- `fase1_comprension_dominio/`: `decisiones_pendientes.csv`, `reglas_variables.csv`, `tipoalta_roles.csv`,
  `matriz_trazabilidad.csv`, `config/fase1_spec.json` y los cuatro `Fase_1__Comprensión_del_dominio_-_*.csv`.
- `fase2_comprension_datos/`: `Fase_2__Diccionario_de_Base_de_Datos.csv`, `Fase_2__Problemas_de_Calidad_de_Datos.csv`
  y `Fase_2__Variables_Derivadas.csv`.
