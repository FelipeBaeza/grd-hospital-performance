# Fase 6: decisiones del autor sobre el plan (8 de noviembre de 2026)

El plan de la Fase 6 (`Plan_Fase6_Despliegue.docx`, sección 11) dejaba ocho decisiones con una propuesta cada una. El 08-11-2026 el autor
las resolvió así. Las etapas F0 a F4 ya están hechas con la cohorte de la enmienda 03; estas decisiones cambian lo que muestra la plataforma
y dónde vive.

| # | Tema | Propuesta del plan | Decisión del autor | Qué se hizo |
|---|---|---|---|---|
| E1 | Dónde queda el prototipo | Entorno académico restringido, sin publicar en internet | **En el servidor del profesor guía** | `docker-compose.yml` y `.env.example` permiten elegir dirección y puertos (`ENLACE`, `PUERTO_WEB`, `CORS_ORIGENES`); la API queda solo local; guía de instalación en `GUIA_FASE6.md`. Se recomienda acceso restringido y HTTPS |
| E2 | Nombres o códigos | Solo códigos | **Se muestra el nombre del hospital** | Nombres desde `Hospitales.csv` en la base, la API y la web. **Desviación** de la decisión 10 de la Fase 1 (fila 13 de `VERSIONES_Y_DESVIACIONES.md`) |
| E3 | Prueba con usuarios | 5 participantes, 6 tareas, 4 de 5 aciertos por tarea | **5 usuarios, más adelante** | Pendiente. La meta (tareas y aciertos) se fija antes de las sesiones |
| E4 | Quién consigue participantes | El estudiante, con apoyo del profesor | **El autor** | — |
| E5 | Orden de las tablas | Solo por código | **Por código** | Ya era así; se mantiene con los nombres |
| E6 | Hospitales con menos de 500 egresos | No se muestran | **No se muestran** | Ya era así (3 hospitales en 2024) |
| E7 | Catálogos de nombres | CIE-10 público y, si se consigue, DEIS o FONASA | **El catálogo que estaba en la carpeta** (`Hospitales.csv`; los diagnósticos ya usaban `CIE-10.csv`) | Cubre 68 de los 69 hospitales con volumen suficiente |
| E8 | Pendientes de la Fase 5 | Constancia del profesor y sensibilidad 2020–2021 | **Para después** | — |

## Hospital sin nombre

El hospital **200717** cumple el volumen mínimo, pero no figura en `Hospitales.csv`. Por decisión del autor **no se muestra hasta conseguir
su nombre**; si no se consigue, queda fuera de la plataforma. Su O/E sigue en las tablas de la Fase 5. El exportador lo deja fuera de forma
automática y lo anota en `metadatos.json` (`hospitales_excluidos_sin_nombre`). Para agregarlo: sumar su fila a `Hospitales.csv` y volver a
correr `herramientas/exportar_publicables.py`.

## Lo que hay que tener presente con los nombres

- La Fase 1 decidió mostrar solo códigos mientras FONASA no responda sobre las condiciones de publicación por hospital. Esa respuesta
  sigue pendiente.
- Con nombres, un O/E alto se lee más fácil como juicio sobre un hospital. Por eso se mantienen el orden por código, la ausencia de
  ranking y las advertencias visibles en cada vista.
- Los nombres no agregan riesgo de reidentificar pacientes: no cambian ningún conteo publicado.
