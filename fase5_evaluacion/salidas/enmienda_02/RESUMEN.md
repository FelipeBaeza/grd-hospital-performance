# Enmienda 02: resumen de resultados

Los análisis de la enmienda 02 (`documentos/ENMIENDA_02.md`) se corrieron una sola vez, el 22 de octubre de 2026
entre las 15:05 y las 15:53 UTC. El código y el catálogo DEIS se registraron y se subieron a GitHub antes (commit
`ca3c40b`). El profesor guía revisa y firma la enmienda después de los resultados, por decisión del autor. Informe
completo: `docs/fase5_evaluacion/Anexo_Enmienda02_resultados.docx`.

El resultado primario de la tesis no cambia.

## En una frase

**Los 24 análisis con regla la cumplen. Las correcciones de cohorte no cambian el indicador:**

- **MDC 14 completo fuera, y alta voluntaria y fuga fuera de estadía** (E2.1 a E2.3): en 2023 ningún hospital
  cambia de zona y la ρ del O/E es de al menos 0,996. En 2024 cambian 1 o 2 hospitales.
- **Muertes obstétricas que salen de la cohorte D** (2019–2023): 46.

## Lo demás

- **Grupos de pares DEIS (E2.4a).** El catálogo tiene 52 hospitales generales de alta complejidad, 14 de mediana,
  3 institutos y 3 pediátricos. Con estos grupos, los hospitales fuera del 95 % bajan de 15,9 a 10,1 % en
  mortalidad 2024 y de 15,9 a 11,6 % en estadía 2024. En estadía, los grupos por volumen de la enmienda 01 los bajan
  más (5,8 %): el tamaño pesa más que la complejidad.
- **Institutos (E2.4b).** La lista del DEIS es la misma que la provisional, así que el resultado es idéntico al
  análisis 3.1a.
- **Hospitales nuevos en GRD (E2.5).** Sin los 7 hospitales con menos de 2 años en GRD, los hospitales fuera en
  estadía 2024 bajan de 15,9 a 12,9 %.
- **Paliativos (E2.6).** Sin los episodios con Z51.5, el indicador resiste: 96,9 y 95,7 % de los hospitales en la
  misma zona. La tasa de Z51.5 por hospital no se relaciona con el O/E (|ρ| de 0,02 o menos).
- **Altas a domicilio y voluntarias (E2.6a).** La hospitalización domiciliaria y el alta voluntaria tienen una
  relación débil con el O/E (|ρ| de 0,14 a 0,26). Los hospitales con más altas a domicilio tienen un O/E de estadía
  algo más bajo.

## Archivos

| Archivo | Contenido |
|---|---|
| `resumen_modelo.csv` | E2.1 a E2.3 |
| `resumen_indicador.csv` | E2.4a, E2.4b, E2.5 y E2.6b |
| `hospitales_informados_aparte.csv` | Institutos y hospitales nuevos |
| `correlaciones_contexto.csv`, `tasas_contexto_hospital.csv` | E2.6a |
| `muertes_obstetricas_excluidas.csv` | Total 2019–2023 |
| `metadatos.json` | Huellas, horas, semillas y versiones |
