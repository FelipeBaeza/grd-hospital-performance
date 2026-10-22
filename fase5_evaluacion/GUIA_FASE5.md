# Guía de la Fase 5 (Evaluación)

Estado: **completa**. 2024 se abrió una vez (ver `documentos/autorizacion_apertura_2024.txt` y `documentos/DECISIONES_FASE5.md`).
Incidentes: `documentos/INCIDENTES.md` (I-01 a I-10). El profesor guía ratificó la apertura después de conocer los resultados (I-10).

**Enmienda 01** (`config/protocolo_fase5_v2.toml` y `documentos/ENMIENDA_01.md`): registrada el 21-10-2026 en
`documentos/registro_enmienda_01.json`, con la aprobación verbal del profesor guía anotada. Falta adjuntar su
constancia escrita (`docs/gobernanza/`).

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

Informe explicado: `docs/fase5_evaluacion/Informe_Fase5_v5.docx`.
