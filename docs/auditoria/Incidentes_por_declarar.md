# Incidentes por declarar en `fase5_evaluacion/documentos/INCIDENTES.md`

Texto propuesto por la auditoría (plan de corrección, etapa 0.3). Copiarlo al final de `INCIDENTES.md` cuando la
autora y el profesor guía lo revisen. No se agregó automáticamente porque es un registro de la autora.

## Mortalidad de 2024 publicada en una versión anterior del informe de la Fase 2
Una versión anterior del informe de la Fase 2 informó la mortalidad de 2024 (2,46 %), un dato de desenlace que el
cortafuegos debía impedir conocer antes de la apertura. La versión 4 del informe lo retiró. La Fase 2 también usó
2024 para la prevalencia de Elixhauser y la deriva de la codificación (predictores, no desenlace).
Medida: se declara. No hay evidencia de que haya cambiado la selección ni la calibración de los modelos.

## Cambios de código de la evaluación después de la apertura
La huella pre-registrada cubre `config/protocolo_fase5.toml`, no el código de `evaluacion/`. Después de la apertura
se recalculó la tabla de traslados salientes sobre la cohorte C (salía en cero con la cohorte D) y se agregaron
pruebas automáticas (15 en la preparación, 18 en el cierre). No cambió ningún criterio, modelo ni gráfico.
Medida: se declara; en adelante, la huella del código de evaluación se incluye en el pre-registro.

## Relectura de 2024 en una reproducción independiente (auditoría, 20-10-2026)
Para verificar que los resultados se reproducen, un auditor externo corrió el proyecto completo desde los archivos
originales en una copia aislada. En esa copia se leyó una vez el desenlace de 2024, con una autorización rotulada
«reproducción independiente de auditoría» y anotada en el registro de accesos de la copia, no en el original.
Resultado: las 12 tablas de la evaluación 2024 y las puertas G1–G8 salieron idénticas byte a byte. La relectura no
es un segundo examen: no se eligió ni se ajustó nada con ella.
Medida: se declara al profesor guía junto con la enmienda del protocolo.
