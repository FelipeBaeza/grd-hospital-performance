# Guía de la Fase 5 (Evaluación)

Estado: **completa**. 2024 se abrió una vez (ver `documentos/autorizacion_apertura_2024.txt` y `documentos/DECISIONES_FASE5.md`).
Incidentes: `documentos/INCIDENTES.md` (I-01 a I-11). El profesor guía ratificó la apertura después de conocer los resultados (I-10).

**Enmienda 01** (`config/protocolo_fase5_v2.toml` y `documentos/ENMIENDA_01.md`): registrada el 21-10-2026 en
`documentos/registro_enmienda_01.json`, con la aprobación verbal del profesor guía anotada. Falta adjuntar su
constancia escrita (`docs/gobernanza/`).

- El código de los 15 análisis (`run_enmienda01.py` y `enmienda01/`) se registró antes de correr
  (`documentos/registro_corrida_enmienda_01.json`, commit `b718454`).
- Se corrió una vez, el 22-10-2026.
- Resultados publicables: `salidas/enmienda_01/` (empezar por `RESUMEN.md`). Anexo:
  `docs/fase5_evaluacion/Anexo_Enmienda01_resultados.docx`.
- En resumen: 41 de 44 combinaciones cumplen la regla. Las 3 que no cumplen son de mortalidad: sin las variables
  condicionales y sin comorbilidades.

```
# solo si la enmienda y el código están registrados; las tablas por hospital quedan en --interna (fuera del repositorio)
python run_enmienda01.py --capas <capas> --trabajo4 <trabajo4> --trabajo5 <trabajo5> --interna <carpeta interna> [--exploracion <salida de la auditoría>]
```

**Enmienda 02** (datos y cohortes): `config/protocolo_fase5_v3.toml` y `documentos/ENMIENDA_02.md`, con su código
en `run_enmienda02.py` y `enmienda02/`. Se registra con `python registrar_enmienda.py --enmienda 2 ...` antes de
correr; el profesor guía la firma después de los resultados (decisión del autor, 22-10-2026). Usa el catálogo DEIS
de `fase3_preparacion/config/tipo_establecimiento.csv`.

- Registrada el 22-10-2026 y corrida una vez, con el código y el catálogo registrados antes (`ca3c40b`).
- Resultados en `salidas/enmienda_02/` (empezar por `RESUMEN.md`) y en
  `docs/fase5_evaluacion/Anexo_Enmienda02_resultados.docx`.
- En resumen: los 24 análisis cumplen la regla, y las correcciones de cohorte no cambian el indicador.

```
python run_enmienda02.py --capas <capas> --trabajo4 <trabajo4> --trabajo5 <trabajo5> --interna01 <carpeta interna de la enmienda 01> --interna <carpeta interna>
```

`evaluacion/enmienda.py` y `registrar_enmienda.py` se generalizaron **después** de la corrida de la enmienda 01, para
registrar también la 02 (`--enmienda 2`). Para la 01 se comportan igual, y las pruebas lo verifican. Por eso su
huella ya no coincide con la que guarda `registro_corrida_enmienda_01.json`, que es la del código con que se corrió.

```
pip install -r requirements.txt
# preparación
python run_fase5.py --capas <capas> --trabajo <trabajo5> --trabajo-fase4 <trabajo4> --etapas huellas planes_b preregistro validar_embudo ensayo verificacion
# apertura única y evaluación de 2024 (no repetir la apertura)
python run_fase5.py ... --etapas apertura
python run_fase5.py ... --etapas evaluacion_2024 --bootstrap 1000
python run_fase5.py ... --etapas cierre
pytest
```

- Resultados reales: `salidas/evaluacion_2024/`. Puertas de cierre: `salidas/tablas/tabla_verificacion_cierre_fase5.csv`.
- `salidas/ensayo_2023/` es solo un ensayo de código (2023 se usó para calibrar).
- La etapa `apertura` falla si 2024 ya fue abierto. El protocolo (`config/protocolo_fase5.toml`) no debe cambiar tras el pre-registro.
- No ejecutar la etapa `verificacion` (puertas de preparación) después de la apertura: su puerta P3 ya no aplica.
- Puertas reforzadas por el plan de corrección (octubre de 2026):
  - **G4** ya no busca texto: corre la evaluación completa con cualquier entrenamiento o recalibración interceptado
    (`evaluacion/sin_ajuste.py`). La evaluación real de 2024 también corre dentro de esa guardia.
  - **P7 y G7** suman el escáner por propiedad (`evaluacion/privacidad_propiedad.py`), que intenta reconstruir
    conteos de 1 a 9 combinando tablas (tasa × tamaño, O/E × esperado y complementos) y busca valores reales en
    pruebas y documentos. Los 13 hallazgos del incidente I-11 se corrigieron regenerando las tablas por subgrupo
    (`regenerar_subgrupos.py`). La lista de hallazgos declarados (`config/privacidad_hallazgos_declarados.csv`) queda
    vacía, así que cualquier hallazgo hace fallar la puerta.
- `run_fase5.py` crea la carpeta de trabajo si no existe.

Informe explicado: `docs/fase5_evaluacion/Informe_Fase5_v5.docx`.
