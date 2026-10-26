# Correr la Fase 2 con la base completa (5,8 millones de egresos) en un equipo de 16 GB

## La idea en simple

Los archivos de FONASA pesan cerca de 3 GB en texto. El problema no era «cargarlos»: la suite **releía el archivo de
texto completo cada vez que una tarea pedía una columna** y, en dos tareas, tenía las 129 columnas de un año en
memoria al mismo tiempo. Eso hace que la RAM suba a saltos y que todo sea lento.

Ahora la suite trabaja así:

1. **Conversión única a Parquet (capa Bronce).** La primera vez que lee un año, convierte el `.txt` a un archivo
   Parquet *en streaming* (va leyendo y escribiendo por trozos, sin cargar el archivo en RAM). Todo queda como texto,
   con las mismas filas y el mismo orden. El `.txt` original no se toca (regla R4).
2. **Lectura por columnas.** Parquet guarda los datos por columna: cuando una tarea pide 10 columnas, se leen solo
   esas 10. Los 3 GB de texto quedan en ~270 MB de Parquet comprimido.
3. **Bloques de columnas.** Las dos tareas que recorren las 129 columnas (T4 descripción y el escaneo de RUT de T5)
   ahora lo hacen de 16 en 16 columnas.
4. **Sin muestreo.** Las Tareas 10 y 11 usan **todos** los egresos de la cohorte D (2,07 millones en 2019–2022 y 568.737 en
   2023). El umbral `max_filas_asociacion_por_anio` vale 0 (sin muestreo) y `modelos.TOPE_FILAS_MODELOS` es `None`.
   Única excepción: el ajuste del modelo L1 usa una muestra fija de 400.000 egresos (`modelos.TOPE_FILAS_L1`), porque con
   toda la cohorte tardaba más de 2 horas; su AUC y su importancia se evalúan con todos los egresos de 2023. Para volver a
   muestrear (equipos con poca memoria) pon un entero positivo en `max_filas_asociacion_por_anio` (en `config/umbrales.toml`).

Los resultados son **idénticos** a los del camino anterior: las pruebas `tests/test_memoria_completo.py` comparan
fila por fila y tabla por tabla los dos caminos.

## Lo que se midió (equipo de prueba con 7 GB de RAM y 2 núcleos)

| | Suite anterior | Suite con baja memoria |
|---|---|---|
| Pico de RAM | 4,1 GB ya en T3–T4; T5 llevaba más de 11 min sin terminar cuando se detuvo | **5,7 GB** en la corrida oficial completa (todos los egresos, también en variables y modelos) |
| T5 (calidad) | más de 11 min sin terminar (el cálculo de episodios solapados crecía casi al cuadrado con el número de pacientes) | **32 s** |
| T3 a T9 y T12 | no terminó | **≈ 15 min** |
| T10 y T11 (todos los egresos) | — | **≈ 27 min** (caché ya creada) |
| Corrida oficial (con pytest y determinismo) | — | **1 h 54 min**, resultado **12 PASS, 0 PENDIENTE, 0 FAIL** (1.759 pruebas) |

Con 16 GB y 12 núcleos debería ser más rápido y con holgura de memoria.

## Cómo correrlo

Desde una terminal en tu equipo:

```bash
cd ~/Documentos/"Proyecto final"/Tesis
source venv/bin/activate
cd fase2_eda
python run_fase2_eda.py --fuente completo --data-root ~/Documentos/"Proyecto final"/Seminario/"data seminario_full"
```

- La primera corrida crea la caché en `Seminario/data seminario_full/_cache_parquet_fase2/` (unos 270 MB). Las
  siguientes la reutilizan; si un `.txt` cambia, su Parquet se regenera solo (el nombre lleva la huella SHA-256).
- Los resultados quedan en `fase2_eda/salidas/completo/` (no se toca `salidas/extracto/`, que desde el cambio 1.6 queda solo en la copia local: con el 5 % de los egresos, combinar filas de la tabla de variantes dejaba deducir un conteo de menos de diez).
- Para una prueba más rápida sin la suite de pruebas ni la corrida repetida de determinismo, agrega
  `--sin-pytest --sin-determinismo`.
- Para correr solo algunas tareas: `--solo T7,T8`.

### Opciones de la caché

| Variable de entorno | Efecto |
|---|---|
| (sin definir) | Con `--fuente completo` la caché va en `<data-root>/_cache_parquet_fase2`. Con el extracto no se usa. |
| `FASE2_CACHE_PARQUET=/otra/carpeta` | Guarda la caché en otra carpeta (por ejemplo un disco con más espacio). |
| `FASE2_CACHE_PARQUET=0` | Desactiva la caché (vuelve al modo antiguo; no recomendado con la base completa). |

**Privacidad:** la caché contiene los mismos datos que los `.txt` (incluidas fechas exactas). Vive junto a los datos
crudos, fuera del repositorio de la tesis; no la subas a GitHub.

## Si aun así la memoria se pone justa

- Cierra el navegador y otras aplicaciones pesadas durante la corrida.
- Limita los hilos de Polars (menos hilos = menos memoria simultánea, algo más lento):
  `POLARS_MAX_THREADS=4 python run_fase2_eda.py ...`
- Corre por partes con `--solo` (por ejemplo `--solo T3,T4,T5,T6` y luego `--solo T7,T8,T9`).

## Análisis posteriores pedidos por el profesor guía (R1 a R5)

Después de la corrida principal (para reutilizar la caché):

```bash
python run_revision_profesor.py --fuente completo --data-root ~/Documentos/"Proyecto final"/Seminario/"data seminario_full"
```

Escribe en `fase2_eda/salidas/revision_profesor/completo/` las tablas `r1` a `r5` y las figuras `R01` a `R03`: mortalidad y estadía por
año y por diagnóstico frecuente (O8), faltantes de las variables del modelo (O2), códigos raros y nuevos del diagnóstico principal (O3),
traslados por tipo de hospital del catálogo DEIS 2025 (O7) y volumen de las cohortes por hospital (decisión 6). Son análisis
posteriores a la apertura de 2024, declarados en el cambio 1.6 del protocolo: solo leen el desenlace de 2019 a 2023 y no cambian
ninguna decisión ya ejecutada.

## Privacidad: supresión primaria y complementaria

Toda tabla oculta los conteos de 1 a 9 casos. Desde el cambio 1.6, las tablas declaradas en `eda/proteccion_tablas.py` aplican además
la supresión complementaria (`eda/proteccion.py`): si una celda oculta es la única oculta de una suma que el lector conoce (el total del
año, el total de la fila, un período que suma sus años), se oculta también la de menor valor de esa suma. Las tablas que publican las
mismas celdas (por ejemplo la distribución de TIPOALTA de las Tareas 4 y 7) heredan las celdas que ocultó la primera.

Informe explicado: `docs/fase2_comprension_datos/Informe_Fase2_v7.docx`.
