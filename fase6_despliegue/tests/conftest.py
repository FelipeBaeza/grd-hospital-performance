"""Fixtures. Las pruebas de integración usan una API real en marcha (variable API_URL, por defecto http://127.0.0.1:8000)
y la base de datos de DATABASE_URL; si no están disponibles se omiten con un mensaje claro."""
import json
import os
import sys
from pathlib import Path

import httpx
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "api"))
DIR_MODELOS = Path(os.environ.get("DIR_MODELOS", RAIZ.parent / "fase4_modelado" / "modelos"))
DIR_DATOS = Path(os.environ.get("DIR_DATOS", RAIZ / "datos" / "publicables"))
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

PERFIL = {"sexo": "HOMBRE", "edad": 71, "tipo_ingreso": "URGENCIA", "tipo_procedencia": "APS CONSULTORIO (CESFAM)", "diagnostico_principal": "J18",
          "ingreso_fin_de_semana": True, "cuidado_intensivo_al_ingreso": False, "comorbilidades": ["ELIX_01", "ELIX_05"]}


@pytest.fixture(scope="session")
def cliente():
    try:
        c = httpx.Client(base_url=API_URL, timeout=30)
        c.get("/salud").raise_for_status()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"la API no responde en {API_URL}: {e}")
    return c


@pytest.fixture(scope="session")
def catalogo():
    return json.loads((DIR_DATOS / "demo_catalogo.json").read_text(encoding="utf8"))


@pytest.fixture(scope="session")
def perfil(catalogo):
    p = dict(PERFIL)
    p["sexo"] = catalogo["sexo"][0] if p["sexo"] not in catalogo["sexo"] else p["sexo"]
    if p["tipo_ingreso"] not in catalogo["tipo_ingreso"]:
        p["tipo_ingreso"] = catalogo["tipo_ingreso"][0]
    p["tipo_procedencia"] = catalogo["tipo_procedencia"][0]["valor"]
    codigos = {d["codigo"] for d in catalogo["diagnosticos"]}
    p["diagnostico_principal"] = "J18" if "J18" in codigos else sorted(codigos)[0]
    p["comorbilidades"] = [c["id"] for c in catalogo["comorbilidades"][:2]]
    return p
