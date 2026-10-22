"""Piezas comunes de la enmienda 02: guardia de registro y lectura del protocolo v3. Las tablas por hospital, la
comparación con la base, la regla de lectura y el bootstrap son los de la enmienda 01 (enmienda01/comun.py), cuya
huella también se registra antes de correr."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

from enmienda01 import comun as c1
from evaluacion.config import FASE3, RAIZ, sha256_archivo

FUERA = c1.FUERA
SEMILLA_MODELO, SEMILLA_BOOTSTRAP = c1.SEMILLA_MODELO, c1.SEMILLA_BOOTSTRAP
REGISTRO = Path("documentos/registro_enmienda_02.json")
REGISTRO_CORRIDA = Path("documentos/registro_corrida_enmienda_02.json")
PROTOCOLO_V3 = Path("config/protocolo_fase5_v3.toml")
ARCHIVOS_CODIGO = ["run_enmienda02.py", "enmienda02/__init__.py", "enmienda02/comun.py", "enmienda02/filtros.py",
                   "enmienda02/modelo.py", "enmienda02/indicador.py", "enmienda02/publicar.py",
                   "enmienda01/comun.py", "enmienda01/modelo.py", "../fase3_preparacion/prep/cohortes.py"]
CATALOGO_DEIS = FASE3 / "config" / "tipo_establecimiento.csv"
RUTA_CATALOGO_DEIS = "../fase3_preparacion/config/tipo_establecimiento.csv"   # como se registra en antes-de-correr

EnmiendaNoAutorizada = c1.EnmiendaNoAutorizada


def verificar_autorizacion(raiz: Path = RAIZ, archivos: list[str] = ARCHIVOS_CODIGO) -> dict:
    """Exige la enmienda 02 registrada y sin cambios, y el código igual al registrado antes de la primera corrida."""
    reg, cor = raiz / REGISTRO, raiz / REGISTRO_CORRIDA
    if not reg.exists():
        raise EnmiendaNoAutorizada("la enmienda 02 no está registrada (falta documentos/registro_enmienda_02.json)")
    if not cor.exists():
        raise EnmiendaNoAutorizada("falta registrar el código antes de correr: python registrar_enmienda.py --enmienda 2 antes-de-correr ...")
    firmado, corrida = json.loads(reg.read_text(encoding="utf8")), json.loads(cor.read_text(encoding="utf8"))
    for ruta, h in firmado["huellas"].items():
        if sha256_archivo(raiz / ruta) != h:
            raise EnmiendaNoAutorizada(f"{ruta} cambió después de firmarse")
    registrados = corrida["scripts"]
    faltan = [a for a in archivos if a not in registrados]
    if faltan:
        raise EnmiendaNoAutorizada(f"código sin registrar: {faltan}")
    distintos = [a for a in archivos if sha256_archivo(raiz / a) != registrados[a]]
    if distintos:
        raise EnmiendaNoAutorizada(f"el código cambió después de registrarse: {distintos}")
    return {"registro_enmienda": sha256_archivo(reg), "registro_corrida": sha256_archivo(cor), "momento_corrida_registrada": corrida["momento_utc"],
            "scripts_registrados": registrados}


def catalogo_deis_autorizado(guardia: dict | None, ruta: Path = CATALOGO_DEIS) -> bool:
    """El catálogo DEIS solo se usa si existe y su huella se registró antes de correr. Sin él, E2.4a y E2.4b quedan pendientes."""
    if not ruta.exists():
        return False
    if guardia is None:                      # pruebas sobre datos sintéticos
        return True
    h = guardia["scripts_registrados"].get(RUTA_CATALOGO_DEIS)
    if h is None:
        raise EnmiendaNoAutorizada("el catálogo DEIS existe pero su huella no se registró antes de correr")
    if sha256_archivo(ruta) != h:
        raise EnmiendaNoAutorizada("el catálogo DEIS cambió después de registrarse")
    return True


def cargar_enmienda(raiz: Path = RAIZ) -> dict:
    with open(raiz / PROTOCOLO_V3, "rb") as f:
        return tomllib.load(f)


def analisis(enm: dict, id_: str) -> dict:
    return next(a for a in enm["analisis"] if a["id"] == id_)
