# Incidentes de la Fase 5

## Lectura directa del archivo sellado (preparación)
Durante la preparación, para conocer el esquema del archivo `desenlace_2024.parquet`, el asistente lo cargó completo con una lectura directa (fuera del cortafuegos de la Fase 3 y sin pasar por el registro de accesos). Solo se imprimieron los nombres de las columnas: no se mostró, calculó ni guardó ningún valor y el archivo no se volvió a abrir. El registro de accesos no registra este evento porque no pasó por el cortafuegos.
Medida: se declara al profesor guía junto con la solicitud de autorización; el esquema se consulta en adelante solo por metadatos del archivo.
