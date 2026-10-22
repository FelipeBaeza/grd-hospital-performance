# Gobernanza (etapa 0 del plan de corrección)

La etapa 0 ordena el registro del proyecto antes de tocar código. No reentrena nada ni cambia ningún resultado ya
calculado.

| Paso | Dónde está | Estado |
| --- | --- | --- |
| 0.1 Congelar el estado | Etiqueta `v1-congelado` y `docs/historial/MANIFIESTO_v1-congelado.sha256` | Hecho |
| 0.2 Versiones de la Fase 1 y desviaciones | `docs/fase1_comprension_dominio/VERSIONES_Y_DESVIACIONES.md` | Hecho |
| 0.3 Registro de incidentes | `fase5_evaluacion/documentos/INCIDENTES.md` (I-01 a I-09) | Hecho; falta decidir I-09 |
| 0.4 Constancia del profesor guía | `Etapa0_Gobernanza_para_firma.docx` (sección 3) y `fase5_evaluacion/documentos/PLANTILLA_constancia_profesor_guia.md` | **Falta la firma** |
| 0.5 Enmienda 01 | `fase5_evaluacion/config/protocolo_fase5_v2.toml` y `fase5_evaluacion/documentos/ENMIENDA_01.md` | Borrador; **falta aprobarla y registrarla** |

## Cómo se cierra

1. Imprimir `Etapa0_Gobernanza_para_firma.docx` y revisarlo con el profesor guía. Si la enmienda cambia, corregir
   los dos archivos de la enmienda antes de seguir.
2. Desde `fase5_evaluacion/`, ejecutar `python registrar_enmienda.py huellas` y anotar las dos huellas en la Parte C.
3. El profesor firma las Partes A, B y C.
4. Escanear la constancia firmada como `fase5_evaluacion/documentos/constancia_profesor_guia_firmada.pdf`.
5. Registrar:

   ```
   python registrar_enmienda.py firmar --autor "Felipe Ignacio Baeza Muñoz" --profesor "Manuel Villalobos Cid" --fecha-firma AAAA-MM-DD
   git add documentos/ && git commit -m "Registra la enmienda 01" && git tag -a enmienda-01 -m "Enmienda 01 firmada" && git push --follow-tags
   ```

6. Decidir qué hacer con el incidente I-09: reescribir o no el historial público.
