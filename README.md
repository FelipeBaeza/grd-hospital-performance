# Indicador O/E hospitalario ajustado por riesgo (GRD 2019–2024)

Proyecto de tesis — Ingeniería de Ejecución en Computación e Informática, Universidad de Santiago de Chile.
Estudiante: Felipe Ignacio Baeza Muñoz · Profesor guía: Manuel Villalobos Cid.

Con los 5.808.535 egresos hospitalarios 2019–2024 que FONASA entregó por la Ley de Transparencia, el proyecto
entrena dos modelos (mortalidad intrahospitalaria y días de estadía) que estiman lo **esperado** para cada paciente
según su perfil al ingresar, y compara por hospital lo **observado** con lo esperado (razón O/E) en gráficos de embudo.
No es un ranking ni una medida directa de calidad: es una señal para orientar revisiones.

## Empieza aquí

1. `docs/00_guia_del_proyecto/Guia_del_proyecto.docx` — qué se hizo, dónde está y cómo se reproduce.
2. `docs/README.md` — índice de todos los informes y orden de lectura.

## Estado

| Fase (CRISP-DM) | Carpeta | Estado |
| --- | --- | --- |
| 1. Comprensión del dominio | `docs/fase1_comprension_dominio/` | Cerrada |
| 2. Comprensión de los datos | `fase2_eda/` | Cerrada |
| 3. Preparación de los datos | `fase3_preparacion/` | Cerrada |
| 4. Modelado | `fase4_modelado/` | Cerrada; modelos congelados con huella SHA-256 |
| 5. Evaluación | `fase5_evaluacion/` | Cerrada; 2024 abierto una sola vez (19-10-2026) |
| 6. Despliegue | `fase6_despliegue/` | En desarrollo |

Resultado en 2024: los 7 criterios de aceptación se cumplen (5 con holgura y 2 en el límite: O/E global de
mortalidad 0,979 y de estadía 0,957). Una reproducción independiente desde cero (octubre de 2026) obtuvo
resultados idénticos byte a byte.

2024 quedó reservado para el pipeline definitivo, que lo leyó una sola vez después de pre-registrar el protocolo.
No fue un año ciego para el autor: un prototipo anterior se evaluó en 2024 el 27-09-2026 (incidente I-01 en
`fase5_evaluacion/documentos/INCIDENTES.md`).

**Gobernanza (etapa 0 del plan de corrección):** los incidentes, las versiones de la Fase 1 y la enmienda 01 al
protocolo están listos. Falta la firma del profesor guía (`docs/gobernanza/`).

## Estructura

```
.
├── README.md                  este archivo
├── docs/                      informes v5 por fase, guía, auditoría e historial de versiones
├── fase2_eda/ … fase6_despliegue/   código de cada fase (cada una con su GUIA_*.md, config/, tests/ y salidas/)
├── archivo/                   artefactos del proceso de desarrollo que ya no se usan para correr nada
├── CIE-10.csv, CIE-9 .csv, IR-GRD.csv, Hospitales.csv, GRDs_T10901.xlsx, …   catálogos (la Fase 2 los lee aquí)
└── data seminario/            extracto de los datos (no se sube a git)
```

## Cómo correrlo

Python 3.13 con las versiones del manifiesto de la Fase 4 (`fase4_modelado/modelos/manifiesto.json`); bastan
2 núcleos y 7 GB de RAM. Los datos completos de FONASA y las capas intermedias viven **fuera** del repositorio.

| Fase | Comando (desde la carpeta de la fase) | Tiempo aprox. |
| --- | --- | --- |
| 2 | `python run_fase2_eda.py --fuente completo --data-root <data seminario_full>` | 2 h |
| 3 | `python run_fase3.py --data-root <data seminario_full> --capas <capas>` | 7 min |
| 4 | `python run_fase4.py --capas <capas> --trabajo <trabajo4>` | 5 h |
| 5 | `python run_fase5.py --capas <capas> --trabajo <trabajo5> --trabajo-fase4 <trabajo4>` | 5 min (preparación) |

## Reglas que el código hace cumplir

- 2024 es el examen final: su desenlace se sella y se abre una sola vez con autorización registrada. Cualquier
  corrección que cambie un modelo debe validarse sin 2024 (diseño 2019–2021 / 2022 / 2023) o con un año nuevo.
- Nada que revele el desenlace ni identifique al hospital entra al modelo.
- Ninguna salida publicada tiene RUT, fechas completas ni cifras de 1 a 9 casos; los hospitales van por código y sin ranking.

## Datos y privacidad

Los archivos de FONASA traen fechas exactas de nacimiento, ingreso y alta: no se suben a git ni a ningún
repositorio público.

Este repositorio es público. Quedan solo en la copia local del autor, fuera de git (ver `.gitignore`):

- `Oficio_Respuesta_N°23364_T10901.pdf` y `docs/comunicaciones_fonasa/`: correspondencia con FONASA. Sin el
  oficio, la Fase 2 registra su procedencia como «AUSENTE»; no cambia ningún cálculo.
- `fase4_modelado/salidas/tablas/auditoria_extracto_hospital_*.csv`: tablas por hospital con muertes observadas
  de 1 a 9 sin suprimir. Se regeneran con `python run_auditoria.py --capas <capas> --trabajo <trabajo>
  --tarea mortalidad|estadia --pruebas hospital` en `fase4_modelado/`.

La prueba de privacidad de la Fase 5 usa valores inventados (hallazgo H-09 de la auditoría, `docs/auditoria/`).
