# Fase 6 · Despliegue del prototipo (API + PostgreSQL + web)

Prototipo académico que se instala en el servidor del profesor guía (decisión E1). No contiene datos de pacientes: solo el catálogo de hospitales (código y nombre) y resultados agregados de la Fase 5 (2024), los de la **enmienda 03** (cohorte de la Fase 3 v7, modelos de la Fase 4 v6; carpeta `evaluacion_2024_v4`).

## Estructura
- `api/` FastAPI (`app/`), cargador de base (`herramientas/cargar_bd.py`), exportador de datos publicables (`herramientas/exportar_publicables.py`).
- `datos/publicables/` salida de la puerta de privacidad (CSV/JSON agregados + `huellas.json`).
- `web/` Vue 3 + Vite (5 vistas). `docker-compose.yml`, `.env.example`.
- `tests/` pruebas automáticas. `docs/openapi.json` contrato de la API.

## Decisiones del 08-11-2026
Ver `docs/fase6_despliegue/DECISIONES_FASE6.md`. En resumen: servidor del profesor (E1), **nombre del hospital visible** (E2, desviación de la decisión 10 de la Fase 1), orden por código (E5), sin hospitales con menos de 500 egresos (E6), nombres desde `Hospitales.csv` (E7). Un hospital sin nombre en el catálogo no se muestra: hoy, el **200717** (`metadatos.json`, `hospitales_excluidos_sin_nombre`). Si se consigue su nombre, se agrega a `Hospitales.csv` y se regeneran los datos publicables.

## Instalar en el servidor del profesor
1. Instalar Docker con Compose y copiar el repositorio (basta con `fase4_modelado/modelos/` y `fase6_despliegue/`, en esas mismas rutas relativas).
2. `cp .env.example .env` y cambiar la clave de la base, `ENLACE`, `PUERTO_WEB` y `CORS_ORIGENES` (la dirección con que se abrirá la web).
3. Recomendado: dejar `ENLACE=127.0.0.1` y publicar la web detrás del proxy con HTTPS del servidor, con acceso restringido (usuario y clave o red de la universidad). La web ya envía `X-Robots-Tag: noindex`.
4. `docker compose up -d --build`. La carga verifica huellas y privacidad antes de abrir la API; si algo no coincide, no arranca.
5. Comprobar: `curl http://127.0.0.1:${PUERTO_API}/salud` en el servidor y abrir la web.

## Levantar con Docker (recomendado)
```
cd fase6_despliegue
docker compose up --build
```
Web: http://127.0.0.1:8080 · API: http://127.0.0.1:8000/docs. La carga a la base corre sola (servicio `carga`) y verifica huellas, privacidad y coincidencia base-CSV antes de arrancar la API. Los modelos se leen de `../fase4_modelado/modelos` (solo lectura) y la API se niega a arrancar si su huella SHA-256 no coincide con el manifiesto.

## Regenerar los datos publicables (solo si cambia la Fase 5)
```
python -m herramientas.exportar_publicables --fase5 ../fase5_evaluacion/salidas/evaluacion_2024_v4 --modelos ../fase4_modelado/modelos \
  --capas <ruta capas> --cie10 <CIE-10.csv> --elix ../fase3_preparacion/config/elixhauser_categorias.csv --salida datos/publicables
```
(ejecutar con `PYTHONPATH=api`). Requiere `polars`, `joblib` además de `api/requirements.txt`.

## Pruebas
Con la API y la base en marcha (`LIMITE_DEMO_POR_MINUTO=100000` al arrancar la API para la prueba de carga):
```
pip install -r api/requirements-dev.txt
pytest -q          # ~1 min; LATENCIA_SEGUNDOS=60 por defecto 30
```
Cubren: modelos intactos y rechazo de alteraciones, demo igual a los modelos congelados de la Fase 4 y SHAP que suma el resultado, coincidencia API/BD/tablas de la Fase 5 (69 hospitales × 2 tareas), ausencia de ranking, nombres tomados solo del catálogo, validación sin eco de entradas, marcador de privacidad en registros del servidor, escaneo de datos publicables, latencia p95 < 2 s con 20 usuarios.

## Qué cambió con la enmienda 03
- La web ya no dibuja el gráfico de embudo, porque necesitaba el esperado y los límites de cada hospital. Muestra el O/E de cada hospital contra su volumen de egresos (redondeado a la decena), con su intervalo de confianza; el color es la zona calculada en la Fase 5.
- La base y la API ya no tienen las columnas de observado, esperado ni límites; el exportador copia solo columnas con nombre y redondea los conteos.
- Los textos de advertencia se actualizaron con las cifras de la enmienda 03 (O/E globales 0,962 y 0,956; 14,5 % y 15,9 % de hospitales fuera del límite del 95 %).

## Reglas que el prototipo hace cumplir
Sin ranking ni orden por O/E (solo por código) · nombre del hospital desde el catálogo (sin nombre, no se muestra) · sin datos de pacientes · celdas < 10 suprimidas; egresos a la decena; **no se publican observado, esperado ni límites del embudo** por hospital ni por subgrupo (enmienda 03, incidente I-12: restando dos versiones de la cohorte se reconstruirían conteos de 1 a 9) · advertencias de la Fase 5 siempre visibles · el perfil de la demo no se guarda ni se registra.

## Pendiente (no se puede automatizar)
Prueba con usuarios (F6), análisis (F7) e informe de cierre con puertas H1–H8 (F8). Docker no se pudo ejecutar en el entorno de desarrollo (el 08-11-2026 el daemon arrancó, pero Docker Hub está bloqueado): `docker-compose.yml` se validó sintácticamente y los mismos pasos se probaron localmente (pip, uvicorn, npm build, PostgreSQL 16).
