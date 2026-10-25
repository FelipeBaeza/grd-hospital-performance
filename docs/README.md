# Documentos del proyecto

## Orden de lectura

1. `00_guia_del_proyecto/Guia_del_proyecto.docx` — la guía de entrada.
2. `fase1_comprension_dominio/Informe_Fase1_v6.docx` — diseño, reglas y decisiones. Restaura la versión 2 del
   informe, la corrige según lo ejecutado e incorpora la revisión técnica del profesor guía.
3. `fase5_evaluacion/Informe_Fase5_v5.docx` — resultados en 2024 y cómo leerlos; y
   `fase5_evaluacion/Anexo_Enmienda01_resultados.docx` y `Anexo_Enmienda02_resultados.docx` — los análisis de
   robustez de las enmiendas 01 y 02.
4. `fase2_comprension_datos/Informe_Fase2_v6.docx` (su sección 9 responde lo que el profesor pidió y la Fase 2 ya
   resolvió), `fase3_preparacion_datos/Informe_Fase3_v5.docx` y
   `fase4_modelado/Informe_Fase4_v5.docx`, según lo que se necesite revisar.
5. `auditoria/` — auditoría metodológica externa y resumen del plan de corrección.
6. `gobernanza/` — etapa 0 del plan de corrección: constancia para el profesor guía, enmienda 01, versiones de la
   Fase 1 e incidentes.
7. `defensa/` — material para la defensa (etapa 5): limitaciones y respuestas al comité con la evidencia de la enmienda 01, y citas del informe de seminario verificadas.

Los informes **v5** (y la **v6** de las Fases 1 y 2) describen solo lo que se ejecutó, con cifras comprobadas en una
reproducción independiente desde cero (octubre de 2026).

## Qué hay en cada carpeta

| Carpeta | Contenido |
| --- | --- |
| `00_guia_del_proyecto/` | Guía del proyecto |
| `00_propuesta_seminario/` | Informe de seminario (agosto de 2026): el punto de partida |
| `fase1_comprension_dominio/` | Informe v6, tablas de la Fase 1 y `VERSIONES_Y_DESVIACIONES.md` (qué versión es el pre-registro y qué cambió) |
| `fase2_comprension_datos/` | Informe v6 y tablas de respaldo de la Fase 2 |
| `fase3_preparacion_datos/` | Informe v5 y plan original de la Fase 3 |
| `fase4_modelado/` | Informe v5 y plan original de la Fase 4 |
| `fase5_evaluacion/` | Informe v5 (preparación y evaluación), anexos de las enmiendas 01 y 02 y plan original de la Fase 5 |
| `fase6_despliegue/` | Plan de la Fase 6 (en desarrollo) |
| `auditoria/` | Auditoría externa y resumen del plan de corrección |
| `gobernanza/` | Documento para revisión y firma del profesor guía (etapa 0) |
| `defensa/` | Material para la defensa (etapa 5): limitaciones, respuestas al comité y citas verificadas |
| `historial/` | Versiones anteriores de los informes, sin cambios (en la Fase 1: el PDF de la versión 2, la v3, la v4 y la v5; en la Fase 2, la v5) |
| `comunicaciones_fonasa/` | Borradores de cartas a FONASA (no enviadas; solo en la copia local, fuera de git) |

Los **planes** de cada fase se conservan como se escribieron: registran lo que se planificó, no lo que se hizo.

## Archivos que el código lee (no mover ni renombrar)

La Fase 2 (`fase2_eda/eda/ejecucion.py`) lee estos archivos como insumo:

- `fase1_comprension_dominio/`: `decisiones_pendientes.csv`, `reglas_variables.csv`, `tipoalta_roles.csv`,
  `matriz_trazabilidad.csv`, `config/fase1_spec.json` y los cuatro `Fase_1__Comprensión_del_dominio_-_*.csv`.
- `fase2_comprension_datos/`: `Fase_2__Diccionario_de_Base_de_Datos.csv`, `Fase_2__Problemas_de_Calidad_de_Datos.csv`
  y `Fase_2__Variables_Derivadas.csv`.
