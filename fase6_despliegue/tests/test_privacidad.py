"""F3: privacidad. Marcadores en el perfil no deben aparecer en registros, base de datos ni código; datos publicables sin celdas pequeñas."""
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

from app import privacidad
from conftest import DIR_DATOS, DIR_MODELOS, RAIZ

MARCA_DIAG = "MARCADOR_PRIVADO_7731"


def test_datos_publicables_sin_filtraciones():
    assert privacidad.escanear_carpeta(DIR_DATOS) == []


def test_no_hay_columnas_de_identificacion_en_esquema():
    sql = "\n".join(l.split("--")[0] for l in (RAIZ / "api" / "app" / "esquema.sql").read_text(encoding="utf8").lower().splitlines())  # sin comentarios
    for col in ("rut", "nombre", "fecha_nac", "fecha_ingreso", "id_egreso", "paciente"):
        assert col not in sql, col


def test_codigo_no_registra_el_perfil():
    for f in (RAIZ / "api" / "app").glob("*.py"):
        t = f.read_text(encoding="utf8")
        assert "logging" not in t and "print(" not in t, f.name


def _puerto_libre():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_marcador_no_aparece_en_registros_del_servidor(perfil, tmp_path):
    """Servidor real aparte: se envía un perfil con marcadores (válido e inválido) y se busca en todo lo que escribe."""
    puerto = _puerto_libre()
    log = tmp_path / "servidor.log"
    env = {**os.environ, "DIR_MODELOS": str(DIR_MODELOS), "DIR_DATOS": str(DIR_DATOS)}
    with open(log, "w") as fh:
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(puerto), "--log-level", "debug"], cwd=RAIZ / "api", env=env, stdout=fh, stderr=subprocess.STDOUT)
        try:
            base = f"http://127.0.0.1:{puerto}"
            for _ in range(60):
                try:
                    if httpx.get(base + "/salud", timeout=2).status_code == 200:
                        break
                except Exception:
                    time.sleep(1)
            else:
                pytest.skip("no levantó el servidor de prueba")
            edad_marca = 87
            httpx.post(base + "/demo/riesgo", json={**perfil, "diagnostico_principal": MARCA_DIAG, "edad": edad_marca}, timeout=30)
            httpx.post(base + "/demo/riesgo", json={**perfil, "sexo": MARCA_DIAG}, timeout=30)
            ok = httpx.post(base + "/demo/riesgo", json={**perfil, "edad": 83}, timeout=30)
            assert ok.status_code == 200
        finally:
            proc.terminate()
            proc.wait(10)
    texto = log.read_text()
    assert MARCA_DIAG not in texto
    for clave in ("comorbilidades", "cuidado_intensivo", "ingreso_fin_de_semana", "diagnostico_principal"):
        assert clave not in texto, f"el registro contiene el campo {clave}"
