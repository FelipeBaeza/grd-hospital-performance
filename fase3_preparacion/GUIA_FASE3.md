# Fase 3 – Preparación de los datos (guía de la carpeta)

Informe explicado paso a paso: `docs/fase3_preparacion_datos/Informe_Fase3_v5.docx`.

## Cómo ejecutar
```
pip install -r requirements.txt            # polars, pyarrow, numpy, matplotlib, pytest
python run_fase3.py --data-root <carpeta con GRD_PUBLICO_<año>.txt> --capas <carpeta de trabajo>
python generar_figuras.py --capas <carpeta de trabajo>
```
Opciones: `--etapas bronce plata cohortes oro catalogo`, `--anios 2019 2020`, `--sin-pytest`, `--sin-determinismo`.
Las capas (Bronce, Plata, Oro, sellado; unos 800 MB) se escriben en `--capas`, no en esta carpeta.

## Contenido
- `prep/` código por capa: bronce, plata, cohortes, oro, sellado, catalogo, privacidad, verificacion, orquestador.
- `config/` umbrales congelados y tablas de referencia (Elixhauser, roles de TIPOALTA, cifras de la Fase 2).
- `tests/` 61 pruebas con datos sintéticos (`python -m pytest`).
- `salidas/tablas/` resultados en CSV; `tabla_verificacion.csv` tiene las 8 puertas (P1 a P8, todas PASS en la corrida final).
- `salidas/figuras/` gráficos del informe.

## Reglas que el código hace cumplir
1. El desenlace de 2024 va a `sellado/`, no se agrega y solo se abre con la variable de entorno `FASE3_AUTORIZA_DESENLACE_2024` (cada intento queda en `registro_accesos.csv`).
2. Vocabularios y tope de estadía (53 días, percentil 99) se calculan solo con 2019–2022.
3. Las filas de cada capa cuadran con la anterior; ningún egreso se elimina sin motivo (5.808.535 en total).
4. Valores imposibles se anulan y se cuentan; nunca se imputan.
5. Salidas sin RUT, fechas completas ni celdas de 1 a 9 casos.
6. Severidad, peso y riesgo de mortalidad del GRD no pasan a Plata ni a Oro.

## Agregado por el plan de corrección (octubre de 2026)
- `prep/cohortes.py` acepta las opciones de la enmienda 02 (MDC 14 completo fuera; alta voluntaria y fuga fuera de E,
  paso EX09). Vienen **apagadas**: sin ellas las cohortes quedan idénticas byte a byte.
- `prep/sello_cripto.py`: sello criptográfico para un año de prueba nuevo (plan 4.5). El profesor guía cifra el
  desenlace con su clave (`python -m prep.sello_cripto sellar <archivo>`) y solo él lo puede abrir.

## Pendiente
Lectura enmascarada de TIPOALTA 2024 (requiere aprobación expresa del profesor guía), cartas a FONASA, revisión clínica y carga del `catalogo_hospitales.sql` en PostgreSQL.
