# Análisis complementario (decisiones 5 y 7)

Respalda las secciones 7.1 y 8.5 del informe de la Fase 2 y 9.6 y 10.4 del informe de la Fase 1.

- `analisis_decisiones.py`: lee solo 2019–2023 de la caché Parquet (el cortafuegos de 2024 se respeta).
  Uso: `python analisis_decisiones.py <carpeta _cache_parquet_fase2> <carpeta de salida>`
- `graficos_decisiones.py`: genera las figuras g17 y g18 a partir de los CSV.
- Resultados: `a_*.csv` (recién nacidos y menores de 18) y `b_*.csv/json` (percentil 99 de la estadía).

Los modelos son de referencia (edad, sexo, número de diagnósticos secundarios, ingreso, procedencia y diagnóstico principal; sin Elixhauser). Sirven para comparar alternativas; no son el modelo final.
