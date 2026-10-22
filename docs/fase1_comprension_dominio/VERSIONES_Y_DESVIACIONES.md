# Fase 1: versiones del informe y desviaciones

Este documento cumple la etapa 0.2 del plan de corrección (hallazgo H-03). Dice qué versión de la Fase 1 sirve
como pre-registro y cuál es una enmienda posterior. Después lista cada decisión que cambió, con su fecha y con si
el cambio se decidió antes o después de abrir 2024. Las horas son UTC.

## 1. Qué versión es qué

| Documento | Fecha verificable | Qué es | Por qué |
|---|---|---|---|
| Informe de seminario (`docs/00_propuesta_seminario/informe.pdf`) | 03-08-2026 | **Pre-registro de objetivos y criterios** | Fija los criterios de aceptación (PR AUC mayor que la logística, BSS > 0, pendiente 0,8–1,2, O/E 0,95–1,05), la partición temporal y las sensibilidades prometidas. Es anterior a todo uso de los datos de 2024 |
| Fase 1 del 27-09-2026 (`docs/historial/fase1/2026-09-27_github/`) | PDF del 26-09 a las 23:31; publicado en GitHub el 27-09 a las 20:33 | **Pre-registro verificable de la Fase 1** | GitHub fija la hora de publicación (commit `58d23a4`). Salió 25 minutos antes de que el prototipo se evaluara en 2024 (incidente I-01) |
| Fase 1 v3 (`docs/historial/fase1/Informe_Fase1_v3_actualizado.docx`) | Se guardó por última vez el 19-10 a las 21:36, después de la apertura | **Versión de trabajo previa a la consolidación** | Su contenido refleja el diseño anterior a la apertura (por ejemplo, los traslados entrantes excluidos). Pero se escribió entre el 05-10 y el 19-10, después de I-01, y su fecha de guardado es posterior a la apertura. Sirve como referencia del diseño, no como prueba de fecha |
| Protocolo de la Fase 5 v1.0 (`fase5_evaluacion/config/protocolo_fase5.toml`) | 19-10-2026 12:16:38 (huella `5624bdae…`) | **Pre-registro de la evaluación** | Fija la regla de veredicto, el volumen mínimo, el embudo y la apertura única, 48 minutos antes de abrir 2024 |
| Fase 1 v4 (`docs/historial/fase1/Informe_Fase1_v4_consenso.docx`) | Archivo del 19-10 a las 22:53. Su texto cita resultados de la Fase 5 de 2024 | **Enmienda posterior a la apertura** | Adopta como decisiones del proyecto lo que ya se había ejecutado. En los traslados entrantes da como motivo que excluirlos «obligaría a… repetir el examen de 2024» |
| Fase 1 v5 (`docs/fase1_comprension_dominio/Informe_Fase1_v5.docx`) | 21-10-2026 | Versión explicativa | Explica lo que se ejecutó, sin tomar decisiones nuevas |

**Por qué el pre-registro es la versión del 27-09 y no la v3.** El plan de corrección proponía restituir la v3. Al
revisar las fechas se ve que la v3 no puede probar que es anterior a la apertura. Además, se escribió después de
que un prototipo se evaluara en 2024 (I-01). La versión del 27-09 tiene una fecha que un tercero (GitHub) confirma
y es anterior a todo contacto con el desenlace de 2024.

## 2. Desviaciones: qué cambió, cuándo y con qué efecto

Tipo: **aclaración** (documenta algo ya decidido o ejecutado, sin cambiar el diseño) o **desviación** (se aparta de
lo pre-registrado).

| # | Tema | Pre-registro | v3 (versión de trabajo) | v4 y lo ejecutado | Cuándo se decidió el cambio | Tipo | Efecto en los resultados | Qué hace la enmienda 01 |
|---|---|---|---|---|---|---|---|---|
| 1 | **Traslados entrantes** (decisión 16) | Informe de seminario: excluidos | Se marcan pero «no entran a los modelos ni al O/E» | Incluidos en las cohortes D y E y en el O/E, ajustados por `TIPO_PROCEDENCIA` | Las Fases 3 y 4 los incluyeron el 18-10, antes de abrir, sin que una puerta lo detectara. La Fase 5 lo declaró (D9). La v4 lo adoptó después de abrir, con un motivo que cita el examen de 2024 | Desviación de ejecución, formalizada después | Ninguno en las cifras: el pipeline estaba congelado. En la exploración, entre el 94 y el 100 % de los hospitales conserva su zona sin ellos | Análisis 2.6: el indicador sin traslados entrantes |
| 2 | **Sensibilidad sin 2020–2021** (decisión 17) | Seminario y Fase 1 del 27-09: «se realizará» | «El plan no incluye análisis de sensibilidad» | La Fase 5 (D4) la dejó fuera. La v4 la rebaja a «planificada, sin resultados» | Se retiró en la v3, antes de abrir. La v4 la reformuló después | Desviación: análisis prometido que no se hizo | No hay resultado del proyecto. En 2024, el capítulo XXII (COVID-19) tiene O/E 0,58. En la exploración, entre el 95 y el 100 % de los hospitales conserva su zona | Análisis 2.2 |
| 3 | **Sensibilidad sin las 5 categorías sospechosas de Elixhauser** (decisión 8) | La lista de decisiones pendientes (06-10) la proponía | «Rechazo» | «Retirada; se declara como limitación» | En la v3 (sin fecha verificable); la v4 lo confirma | Desviación: salvaguarda retirada | Hallazgo H-02. En la exploración el PR AUC baja 0,028 y entre el 91 y el 100 % de los hospitales conserva su zona | Análisis 2.1 |
| 4 | **Rastreo de traslados salientes** (decisión 13) | Seminario: censurados; alta anticipada declarada | «Verde: viable; rastrear dentro de 2021–2024» | Retirado: quedan censurados y se muestra la tasa de traslado por hospital | La Fase 3 no lo implementó (antes de abrir). La v4 lo formalizó después | Desviación justificada: el identificador cambia entre 2020 y 2021 | El sesgo de alta anticipada no se corrige (H-06) | No se reanaliza; queda como limitación. La tasa de traslado se muestra como condición de lectura (3.1) |
| 5 | **Variables condicionales** (decisión 2) | Seminario: solo CIE-10 en bruto | Servicio, especialidad y derivado: «Rechazado». El nivel de cuidado al ingreso queda permitido | Excluye esas tres; incluye el nivel de cuidado. Agrega `PROCEDENCIA_AGR` (dos niveles de procedencia), como en la Fase 3 | Fase 3, antes de abrir | Aclaración: alinea el texto con lo ejecutado | Hallazgo H-10. Sin las 4 condicionales, el ranking de mortalidad cambia (ρ 0,80 y 0,89): es la única variante del modelo que no cumple la regla | Análisis 2.3; decisión clínica pendiente |
| 6 | **Validación clínica** (decisión 12) | — | «Generar un informe para un médico pero seguir con las fases» | «Conseguir una contraparte clínica antes de la Fase 3»; no se consiguió | — | Desviación pendiente | Nadie validó clínicamente EX03, la procedencia ni el nivel de cuidado | Fuera de la enmienda 01: las correcciones de cohortes van en la enmienda 02 |
| 7 | **Aprobación del profesor guía y cartas a FONASA** | La Fase 1 del 27-09 deja cada decisión para que el profesor la apruebe o la rechace | «Falta la confirmación del profesor guía en cada una»; cartas a FONASA pendientes | «El profesor otorgó plena libertad»; «se descartó enviar cartas a FONASA y adjuntar una constancia del profesor». En cambio, `DECISIONES_FASE5.md` deja la constancia pendiente | Después de abrir (v4) | Desviación de gobernanza: aprobación sin registro escrito | Ninguno en las cifras; afecta la credibilidad del pre-registro | Etapa 0.4: constancia del profesor (Partes A a C) |
| 8 | **Tope de estadía en el percentil 99** (53 días) | Lista de pendientes (06-10): definir la regla solo con 2019–2022 | Sin sección propia | Sección 10.4 y decisión 7 | Fase 3 (18-10) y protocolo v1.0 (`tope_dias = 53`), antes de abrir | Aclaración | Ninguno | — |
| 9 | **Volumen mínimo** (decisión 6) | Seminario: «número suficiente» | 500 egresos y 10 muertes esperadas, «para confirmar» | Adoptado, «con muertes esperadas en todas las fases». La Fase 3 había usado muertes observadas | Protocolo v1.0 (`min_esperado = 10`), antes de abrir | Aclaración | Ninguno: la Fase 5 usó muertes esperadas | — |
| 10 | **Nombres de hospitales** (decisión 10) | — | «Pendiente definir» / «Aceptado» | Solo códigos, sin nombres | Fase 5 (D6), el 19-10 | Aclaración | Ninguno. Los códigos DEIS no anonimizan (H-18) | — |
| 11 | **Qué no es el indicador** (§11.4) | Seminario §4.3 y §5.2 | Sin sección | Sección nueva | v4 | Aclaración | Ninguno | — |

## 3. Cómo leer esta tabla

- Lo único que cambió el análisis respecto de lo pre-registrado son las filas 1 a 4. Las cuatro se decidieron
  **en la ejecución** (Fases 3 a 5) y la v4 las formalizó después de abrir 2024. Ninguna cambió una cifra ya
  calculada, pero no deben presentarse como decisiones previas.
- La enmienda 01 (`fase5_evaluacion/documentos/ENMIENDA_01.md`) convierte las filas 1, 2, 3 y 5 en análisis
  declarados, con una regla de lectura fijada antes de correrlos.
- Las filas 6 y 7 dependen de terceros: el profesor guía (constancia) y una contraparte clínica.
