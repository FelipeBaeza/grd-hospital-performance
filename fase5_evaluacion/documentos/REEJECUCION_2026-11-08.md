# Re-ejecución completa de la Fase 5, enmienda 03 (8 de noviembre de 2026)

**Pregunta.** Después de rehacer la Fase 6 y de reescribir el historial de git (I-09), ¿la Fase 5 sigue dando lo mismo?

**Respuesta.** Sí. Todos los resultados son **idénticos byte a byte** a los publicados el 04-11-2026, y las 16 compuertas
dan el mismo estado que en la corrida oficial, salvo P3. P3 está en FAIL porque se diseñó para correr antes de abrir
2024, y esta re-ejecución se hizo después.

## Cómo se corrió

- **Código:** el del commit `f3ddb6e` (historial ya reescrito), en una copia aislada del repositorio (`git worktree`) y
  de la carpeta de trabajo, para no sobrescribir nada de lo publicado.
- **Datos:** capas de la Fase 3 v7 y la carpeta de trabajo de la Fase 4 v7. Protocolo `config/protocolo_fase5_v4.toml`.
- **Etapas:** huellas, planes B, pre-registro, validación del embudo, ensayo con 2023 (100 réplicas), verificación
  previa, evaluación de 2024 (1000 réplicas, semilla 20261019), publicación y cierre.

**No se abrió 2024 de nuevo.** Se omitió la etapa `apertura`. La evaluación de 2024 se recalculó desde el desenlace que
guardó la segunda apertura autorizada (huella SHA-256 `afc164a36df29d04…`, la misma registrada el 04-11-2026), que es
como está diseñada esa etapa. El archivo sellado no se leyó, el registro de accesos de la Fase 3 no cambió y siguen
constando 2 lecturas concedidas. Nada se reajustó: la etapa corre con los ajustes prohibidos (G4).

**Tiempos:** ensayo con 2023, 173 s; evaluación de 2024, 797 s.

## Qué se comparó

| Salida | Archivos | Resultado |
|---|---|---|
| Tablas, figuras y resúmenes publicados de `ensayo_2023_v4` y `evaluacion_2024_v4` | 28 | Idénticos byte a byte |
| Tablas completas (carpeta de trabajo, fuera del repositorio) | 16 | Idénticas byte a byte |
| Criterios de 2024 (7) y su veredicto | — | Iguales: 7 cumplen, los dos O/E globales (0,962 y 0,956) «cumple con incertidumbre» |
| Indicador por hospital (69 + 69) | — | Igual: mismas zonas, mismos O/E e intervalos |

## Compuertas

| Puerta | Oficial (04-11) | Re-ejecución (08-11) | Por qué |
|---|---|---|---|
| G1 a G8 (cierre) | 8 PASS | 8 PASS | Iguales |
| P1, P2, P4 a P8 (preparación) | PASS | PASS | Iguales |
| P3 «2024 sin abrir con este protocolo» | PASS | **FAIL** | Esperado: la puerta verifica que la segunda lectura todavía no ocurrió; el 08-11 ya había ocurrido |

**Dos detalles del procedimiento**, para quien repita esta prueba:

- **Pre-registro.** Si se vuelve a correr la etapa `preregistro`, se reescribe `registro_preapertura_v4.json` con la fecha
  del día, y G3 falla, porque el registro queda posterior a la apertura. En esta re-ejecución se repuso el registro original
  y G3 dio PASS. El registro del repositorio no se tocó.
- **Orden de las etapas.** Si la verificación previa corre antes de `publicacion`, P7 y P8 fallan: el ensayo deja las
  tablas completas en `salidas/` y el escáner entre tablas las compara con las del protocolo v1.0 (89 hallazgos). Después
  de `publicacion`, las dos dan PASS. El orden correcto está en `GUIA_FASE5.md`.

## Conclusión

La reescritura del historial y los cambios de la Fase 6 no alteraron la Fase 5: con el mismo código, los mismos datos y
el desenlace guardado, la evaluación de 2024 de la enmienda 03 se reproduce exactamente.
