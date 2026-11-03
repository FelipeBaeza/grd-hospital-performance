# Fase 3 – Preparación de los datos (guía de la carpeta)

Informe explicado paso a paso: `docs/fase3_preparacion_datos/Informe_Fase3_v7.docx`.

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
- `tests/` 74 pruebas con datos sintéticos (`python -m pytest`).
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
- `catalogo_deis.py`: arma `config/tipo_establecimiento.csv` desde la Base de Establecimientos 2025 del DEIS
  (`config/fuentes/`). Los 72 hospitales GRD quedan en: 52 generales de alta complejidad, 14 de mediana, 3 institutos
  monográficos y 3 pediátricos. Incluye el primer año en GRD (plan 1.3 y 1.4).
- `prep/sello_cripto.py`: sello criptográfico para un año de prueba nuevo (plan 4.5). El profesor guía cifra el
  desenlace con su clave (`python -m prep.sello_cripto sellar <archivo>`) y solo él lo puede abrir.

## Agregado por la segunda revisión del profesor (2 de noviembre de 2026)
- `prep/cohortes.py` suma tres opciones, también **apagadas** (sin ellas las cohortes son idénticas a las oficiales):
  `tipoalta_vivos` (hospitalización domiciliaria como egreso vivo en D; sale de E en EX11), `deduplicar_q15` (filas repetidas
  fuera de D, EX10) y `excluir_entrantes_enlazados` (entrantes enlazados con un traslado saliente a 30 días, fuera de D, EX12).
  `opciones_revision()` las junta con `mdc14_completo`. Solo afectan a 2019–2023: en 2024 se aplican en la Fase 5.
- `run_cohortes_revision.py --capas <carpeta>` arma las seis variantes en `<capas>/cohortes_revision/` y escribe
  `salidas/tablas/cascada_revision.csv` (solo el período 2019–2023, por privacidad).
- `config/umbrales_fase3.toml` v1.1: `ventana_enlace_traslado_dias` (30) y `regimenes_id`.

## Versión 7: la fase rehecha con las reglas de la revisión (3 de noviembre de 2026)
- Por decisión del autor, `config/umbrales_fase3.toml` v2.0 activa las cuatro reglas en la sección `[cohorte]`: la cohorte oficial
  pasa a D = 2.537.454 y E = 2.205.838 (2019–2023). Cada regla se apaga ahí sin tocar el código.
- La cohorte anterior se reconstruye como control en cada corrida; la puerta P3 exige que siga reproduciendo las 79 cifras de la Fase 2
  (`comparacion_cascada_fase2.csv`) y `cascada_control_vs_oficial.csv` compara las dos en 2019–2023.
- `cascada_fase3.csv` publica por año EX10 sumado a EX06 (las filas repetidas son de 1 a 9 en algunos años).
- P5 cuenta solo las lecturas de 2024 posteriores al sellado de la corrida (el registro guarda también la apertura de la Fase 5).
- Las Fases 4 y 5 deben volver a correrse con estas capas; volver a evaluar en 2024 requiere el visto bueno del profesor guía.

## Pendiente
Lectura enmascarada de TIPOALTA 2024 (requiere aprobación expresa del profesor guía), cartas a FONASA, revisión clínica y carga del `catalogo_hospitales.sql` en PostgreSQL.
