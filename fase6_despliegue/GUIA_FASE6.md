# Fase 6 · Despliegue del prototipo (API + PostgreSQL + web)

Prototipo académico. Solo escucha en `127.0.0.1`. No contiene datos de pacientes: solo el catálogo de códigos de hospital y resultados agregados de la Fase 5 (2024).

## Estructura
- `api/` FastAPI (`app/`), cargador de base (`herramientas/cargar_bd.py`), exportador de datos publicables (`herramientas/exportar_publicables.py`).
- `datos/publicables/` salida de la puerta de privacidad (CSV/JSON agregados + `huellas.json`).
- `web/` Vue 3 + Vite (5 vistas). `docker-compose.yml`, `.env.example`.
- `tests/` pruebas automáticas. `docs/openapi.json` contrato de la API.

## Levantar con Docker (recomendado)
```
cd fase6_despliegue
docker compose up --build
```
Web: http://127.0.0.1:8080 · API: http://127.0.0.1:8000/docs. La carga a la base corre sola (servicio `carga`) y verifica huellas, privacidad y coincidencia base-CSV antes de arrancar la API. Los modelos se leen de `../fase4_modelado/modelos` (solo lectura) y la API se niega a arrancar si su huella SHA-256 no coincide con el manifiesto.

## Regenerar los datos publicables (solo si cambia la Fase 5)
```
python -m herramientas.exportar_publicables --fase5 ../fase5_evaluacion/salidas/evaluacion_2024 --modelos ../fase4_modelado/modelos \
  --capas <ruta capas> --cie10 <CIE-10.csv> --elix ../fase3_preparacion/config/elixhauser_categorias.csv --salida datos/publicables
```
(ejecutar con `PYTHONPATH=api`). Requiere `polars`, `joblib` además de `api/requirements.txt`.

## Pruebas
Con la API y la base en marcha (`LIMITE_DEMO_POR_MINUTO=100000` al arrancar la API para la prueba de carga):
```
pip install -r api/requirements-dev.txt
pytest -q          # ~1 min; LATENCIA_SEGUNDOS=60 por defecto 30
```
Cubren: modelos intactos y rechazo de alteraciones, demo igual a los modelos congelados de la Fase 4 y SHAP que suma el resultado, coincidencia API/BD/tablas de la Fase 5 (69 hospitales × 2 tareas), ausencia de ranking y de nombres, validación sin eco de entradas, marcador de privacidad en registros del servidor, escaneo de datos publicables, latencia p95 < 2 s con 20 usuarios.

## Reglas que el prototipo hace cumplir
Sin ranking ni orden por O/E (solo por código) · solo códigos de hospital · sin datos de pacientes · celdas < 10 suprimidas (también las derivadas: observado = O/E × esperado) · advertencias de la Fase 5 siempre visibles · el perfil de la demo no se guarda ni se registra.

## Pendiente (no se puede automatizar)
Prueba con usuarios (F6), análisis (F7) e informe de cierre con puertas H1–H8 (F8). Docker no se pudo ejecutar en el entorno de desarrollo: `docker-compose.yml` se validó sintácticamente y los mismos pasos se probaron localmente (pip, uvicorn, npm build, PostgreSQL 16).
