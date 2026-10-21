"""API de solo lectura sobre resultados agregados + demo pedagógica. No expone orden por O/E, nombres de hospitales ni datos de pacientes."""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from . import config, db, textos
from .esquemas import PerfilDemo
from .motor import MotorPrediccion

TAREAS = ("mortalidad", "estadia")
VERSION = "1.0.0"
LIMITE_DEMO_POR_MINUTO = int(os.environ.get("LIMITE_DEMO_POR_MINUTO", "120"))


@asynccontextmanager
async def vida(app: FastAPI):
    app.state.motor = MotorPrediccion(config.DIR_MODELOS, config.DIR_DATOS)      # falla si los modelos no coinciden con el manifiesto
    app.state.visitas = defaultdict(deque)
    yield


app = FastAPI(title="Indicadores hospitalarios GRD", version=VERSION, lifespan=vida,
              description="Indicador O/E de mortalidad y estadía por hospital (códigos, sin ranking) y demo pedagógica con perfil sintético.")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGENES, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.exception_handler(RequestValidationError)
async def validacion(_: Request, exc: RequestValidationError):
    """Responde qué campo falló sin devolver el valor ingresado."""
    return JSONResponse(status_code=422, content={"detalle": [{"campo": ".".join(str(x) for x in e["loc"][1:]), "mensaje": e["msg"]} for e in exc.errors()]})


@app.middleware("http")
async def cabeceras(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    if request.url.path.startswith("/demo"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


def _consultar(sql: str, **params) -> list[dict]:
    with db.motor().connect() as cx:
        return [dict(r._mapping) for r in cx.execute(text(sql), params)]


def _tarea(tarea: str) -> str:
    if tarea not in TAREAS:
        raise HTTPException(404, "tarea desconocida: use 'mortalidad' o 'estadia'")
    return tarea


def _meta() -> dict:
    return _consultar("SELECT valor FROM metadatos WHERE clave='paquete'")[0]["valor"]


def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


# ---------------------------------------------------------------- estado y metodología
@app.get("/salud")
def salud():
    with db.motor().connect() as cx:
        cx.execute(text("SELECT 1"))
    return {"estado": "ok", "version_api": VERSION, "modelos": app.state.motor.estado(), "huella_reglas_fase4": _meta()["huella_reglas_fase4"]}


@app.get("/metodologia")
def metodologia():
    meta = _meta()
    return {"que_es": textos.QUE_ES, "no_es": textos.NO_ES, "advertencias": textos.ADVERTENCIAS, "limites_datos": textos.LIMITES_DATOS, "demo_avisos": textos.DEMO_AVISOS,
            "volumen": {"min_egresos": 500, "min_esperado": 10, "celda_minima": 10}, "anio_evaluado": meta["anio_evaluado"],
            "cohortes": {k: meta["resumen_fase5"][k] for k in ("n_cohorte_c", "n_mortalidad", "n_estadia")}}


@app.get("/criterios")
def criterios():
    filas = _consultar("SELECT tarea, criterio, valor, ic_inf, ic_sup, umbral, veredicto, es_criterio FROM criterio_global ORDER BY tarea DESC, orden")
    return {"criterios": [f for f in filas if f["es_criterio"]], "informativas": [f for f in filas if not f["es_criterio"]]}


# ---------------------------------------------------------------- hospitales
@app.get("/hospitales")
def hospitales():
    """Códigos disponibles, siempre en orden de código (no hay orden por desempeño)."""
    filas = _consultar("SELECT cod_hospital FROM hospital ORDER BY cod_hospital")
    return {"hospitales": [f["cod_hospital"] for f in filas], "n": len(filas)}


@app.get("/indicadores/{tarea}")
def indicadores(tarea: str):
    t = _tarea(tarea)
    filas = _consultar("""SELECT cod_hospital, n_egresos, observado_pub, esperado, oe, ic_inf, ic_sup, lim_inf_95, lim_sup_95, lim_inf_998, lim_sup_998, zona
                          FROM indicador_hospital WHERE tarea=:t ORDER BY cod_hospital""", t=t)
    resumen = _meta()["resumen_fase5"][f"hospitales_{t}"]
    esp = np.array([f["esperado"] for f in filas], float)
    x = np.exp(np.linspace(np.log(esp.min() * 0.8), np.log(esp.max() * 1.2), 120))
    curva = {"esperado": [round(float(v), 4) for v in x]}
    for z, nombre in ((1.96, "95"), (3.09, "998")):
        s = np.sqrt(resumen["dispersion"] / x + resumen["tau2"])
        curva[f"inf_{nombre}"] = [round(float(v), 5) for v in 1 - z * s]
        curva[f"sup_{nombre}"] = [round(float(v), 5) for v in 1 + z * s]
    puntos = []
    for f in filas:
        puntos.append({**{k: (_num(v) if isinstance(v, float) else v) for k, v in f.items()}, "oculto_por_privacidad": f["oe"] is None})
    return {"tarea": t, "anio": _meta()["anio_evaluado"], "n_hospitales": len(filas), "puntos": puntos, "curva": curva,
            "resumen": {k: _num(resumen[k]) if isinstance(resumen[k], float) else resumen[k] for k in ("n_elegibles", "n_bajo_volumen", "tau2", "dispersion", "pct_fuera_95", "pct_senal_998")},
            "advertencias": textos.ADVERTENCIAS, "nota": "Orden por código de hospital. No es un ranking."}


@app.get("/hospitales/{codigo}")
def ficha(codigo: str):
    if not _consultar("SELECT 1 FROM hospital WHERE cod_hospital=:c", c=codigo):
        raise HTTPException(404, "código de hospital no disponible (sin datos suficientes o inexistente)")
    ind = {f["tarea"]: f for f in _consultar("""SELECT tarea, n_egresos, observado_pub, esperado, oe, ic_inf, ic_sup, lim_inf_95, lim_sup_95, lim_inf_998, lim_sup_998, zona
                                               FROM indicador_hospital WHERE cod_hospital=:c""", c=codigo)}
    ctx = _consultar("SELECT indice_vs_2023, indice_relativo_al_anio, tasa_traslado_saliente, proporcion_traslado_entrante FROM contexto_hospital WHERE cod_hospital=:c", c=codigo)[0]
    return {"cod_hospital": codigo, "anio": _meta()["anio_evaluado"], "indicadores": ind, "contexto": ctx, "advertencias": textos.ADVERTENCIAS,
            "lectura": "Una zona fuera de los límites es un motivo para revisar, no un juicio sobre el hospital."}


@app.get("/subgrupos/{tarea}")
def subgrupos(tarea: str):
    t = _tarea(tarea)
    return {"tarea": t, "subgrupos": _consultar("SELECT variable, grupo, n, observado_pub, esperado, oe FROM subgrupo WHERE tarea=:t ORDER BY variable, grupo", t=t)}


@app.get("/calibracion/{tarea}")
def calibracion(tarea: str):
    t = _tarea(tarea)
    return {"tarea": t, "deciles": _consultar("SELECT decil, n, observado, predicho, razon_oe FROM calibracion_decil WHERE tarea=:t ORDER BY decil", t=t)}


# ---------------------------------------------------------------- demo pedagógica
def _limitar(request: Request) -> None:
    ahora = time.monotonic()
    cola = request.app.state.visitas[request.client.host if request.client else "?"]
    while cola and ahora - cola[0] > 60:
        cola.popleft()
    if len(cola) >= LIMITE_DEMO_POR_MINUTO:
        raise HTTPException(429, "demasiadas solicitudes a la demo; espere un minuto")
    cola.append(ahora)


@app.get("/demo/catalogo")
def demo_catalogo():
    c = app.state.motor.catalogo
    return {**c, "avisos": textos.DEMO_AVISOS}


@app.post("/demo/riesgo")
def demo_riesgo(perfil: PerfilDemo, request: Request):
    """Calcula riesgo y días esperados de un perfil sintético. El perfil vive solo en memoria durante la llamada: no se guarda ni se registra."""
    _limitar(request)
    m = app.state.motor
    p = perfil.model_dump()
    errores = []
    if p["sexo"] not in m.catalogo["sexo"]:
        errores.append({"campo": "sexo", "mensaje": "valor fuera del catálogo"})
    if p["tipo_ingreso"] not in m.catalogo["tipo_ingreso"]:
        errores.append({"campo": "tipo_ingreso", "mensaje": "valor fuera del catálogo"})
    if p["tipo_procedencia"] not in m.proc:
        errores.append({"campo": "tipo_procedencia", "mensaje": "valor fuera del catálogo"})
    if p["diagnostico_principal"] not in m.diag:
        errores.append({"campo": "diagnostico_principal", "mensaje": "código CIE-10 fuera del catálogo de la demo"})
    if any(c not in m.elix for c in p["comorbilidades"]):
        errores.append({"campo": "comorbilidades", "mensaje": "comorbilidad desconocida"})
    if errores:
        return JSONResponse(status_code=422, content={"detalle": errores})
    res = m.predecir(p)
    res["avisos"] = textos.DEMO_AVISOS
    res["lectura"] = ("Probabilidad de morir durante la hospitalización y días esperados de estadía si el paciente sobrevive, según los modelos congelados. "
                      "Cada contribución indica cuánto empuja esa variable el resultado respecto del perfil promedio (en la escala del modelo).")
    return res
