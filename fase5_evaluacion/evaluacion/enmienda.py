"""Registro de las enmiendas al protocolo de la Fase 5.

- Enmienda 01: protocolo_fase5_v2.toml + documentos/ENMIENDA_01.md (base: protocolo v1.0).
- Enmienda 02: protocolo_fase5_v3.toml + documentos/ENMIENDA_02.md (base: la enmienda 01, que debe estar registrada).

Una enmienda vale solo cuando existe documentos/registro_enmienda_0N.json. Ese registro guarda las huellas SHA-256 de
los dos archivos de la enmienda, de la constancia del profesor guía y del código de evaluacion/, con la hora. Antes de
la primera corrida, registro_corrida_enmienda_0N.json agrega la huella de los scripts que corren los análisis y
comprueba que la enmienda no cambió desde la firma. Las funciones toman ``n`` (1 por defecto, como antes).
"""
from __future__ import annotations

import datetime as dt
import json
import tomllib
from pathlib import Path

from .config import RAIZ, sha256_archivo

PROTOCOLO_V2 = Path("config/protocolo_fase5_v2.toml")
PROTOCOLO_V1 = Path("config/protocolo_fase5.toml")
TEXTO = Path("documentos/ENMIENDA_01.md")
CONSTANCIA = Path("documentos/constancia_profesor_guia_firmada.pdf")
REGISTRO = Path("documentos/registro_enmienda_01.json")
REGISTRO_CORRIDA = Path("documentos/registro_corrida_enmienda_01.json")
ENMIENDAS = {
    1: {"protocolo": PROTOCOLO_V2, "texto": TEXTO, "base": PROTOCOLO_V1, "constancia": CONSTANCIA, "registro": REGISTRO, "corrida": REGISTRO_CORRIDA},
    2: {"protocolo": Path("config/protocolo_fase5_v3.toml"), "texto": Path("documentos/ENMIENDA_02.md"), "base": PROTOCOLO_V2,
        "constancia": Path("documentos/constancia_profesor_guia_enmienda_02.pdf"), "registro": Path("documentos/registro_enmienda_02.json"),
        "corrida": Path("documentos/registro_corrida_enmienda_02.json"), "requiere": REGISTRO},
}


def _e(n: int) -> dict:
    if n not in ENMIENDAS:
        raise ErrorEnmienda(f"no existe la enmienda {n}")
    return ENMIENDAS[n]


class ErrorEnmienda(RuntimeError):
    pass


def _ahora(ahora: dt.datetime | None) -> dt.datetime:
    return (ahora or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc).replace(microsecond=0)


def huellas(raiz: Path = RAIZ, n: int = 1) -> dict[str, str]:
    """Huellas de los dos archivos de la enmienda (las que van en la constancia del profesor)."""
    e = _e(n)
    faltan = [str(p) for p in (e["protocolo"], e["texto"]) if not (raiz / p).exists()]
    if faltan:
        raise ErrorEnmienda(f"faltan archivos de la enmienda: {faltan}")
    return {str(e["protocolo"]): sha256_archivo(raiz / e["protocolo"]), str(e["texto"]): sha256_archivo(raiz / e["texto"])}


def huellas_codigo(raiz: Path = RAIZ) -> dict[str, str]:
    return {str(p.relative_to(raiz)): sha256_archivo(p) for p in sorted((raiz / "evaluacion").glob("*.py"))}


def _verificar_base(raiz: Path, n: int = 1) -> dict:
    e = _e(n)
    if e.get("requiere") and not (raiz / e["requiere"]).exists():
        raise ErrorEnmienda(f"la enmienda {n:02d} se apoya en una enmienda anterior que no está registrada ({e['requiere']})")
    with open(raiz / e["protocolo"], "rb") as f:
        datos = tomllib.load(f)
    if datos.get("enmienda") != n:
        raise ErrorEnmienda(f"{e['protocolo']} declara la enmienda {datos.get('enmienda')}, no la {n}")
    actual = sha256_archivo(raiz / e["base"])
    if datos.get("huella_base") != actual:
        raise ErrorEnmienda(f"el protocolo base ({e['base']}) no coincide con la huella base declarada en la enmienda ({actual[:8]}…)")
    ids = [a["id"] for a in datos.get("analisis", [])]
    if not ids or len(ids) != len(set(ids)):
        raise ErrorEnmienda("la lista de análisis está vacía o tiene identificadores repetidos")
    return datos


def firmar(autor: str, profesor: str, fecha_firma: str, raiz: Path = RAIZ, sin_constancia: bool = False,
           ahora: dt.datetime | None = None, nota: str | None = None, n: int = 1) -> Path:
    """Crea documentos/registro_enmienda_0N.json. No sobrescribe un registro existente."""
    e = _e(n)
    if (raiz / e["registro"]).exists():
        raise ErrorEnmienda(f"la enmienda {n:02d} ya está registrada; un cambio exige una enmienda nueva")
    if not autor.strip() or not profesor.strip():
        raise ErrorEnmienda("faltan los nombres del autor y del profesor guía")
    dt.date.fromisoformat(fecha_firma)
    datos = _verificar_base(raiz, n)
    constancia = raiz / e["constancia"]
    if not constancia.exists() and not sin_constancia:
        raise ErrorEnmienda(f"falta la constancia firmada del profesor guía ({e['constancia']})")
    t = _ahora(ahora)
    registro = {
        "enmienda": datos["enmienda"], "version": datos["version"],
        "huellas": huellas(raiz, n), ("huella_protocolo_v1" if n == 1 else "huella_protocolo_base"): datos["huella_base"],
        "constancia": {"ruta": str(e["constancia"]), "sha256": sha256_archivo(constancia)} if constancia.exists()
                      else {"ruta": None, "sha256": None, "nota": "registrada sin constancia escrita del profesor guía"},
        "codigo_evaluacion": huellas_codigo(raiz),
        "analisis": [a["id"] for a in datos["analisis"]],
        "firmas": {"autor": autor.strip(), "profesor_guia": profesor.strip(), "fecha_firma": fecha_firma},
        "nota": (nota or "").strip() or None,
        "momento_utc": t.isoformat(), "momento_epoch": int(t.timestamp()),
    }
    (raiz / e["registro"]).write_text(json.dumps(registro, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    return raiz / e["registro"]


def antes_de_correr(scripts: list[str], raiz: Path = RAIZ, ahora: dt.datetime | None = None, n: int = 1) -> Path:
    """Registra la huella de los scripts de la enmienda antes de la primera corrida. Falla si la enmienda cambió."""
    e = _e(n)
    if not (raiz / e["registro"]).exists():
        raise ErrorEnmienda(f"la enmienda {n:02d} no está firmada: falta {e['registro']}")
    if (raiz / e["corrida"]).exists():
        raise ErrorEnmienda(f"la corrida de la enmienda {n:02d} ya está registrada")
    if not scripts:
        raise ErrorEnmienda("indica los scripts que corren los análisis de la enmienda")
    firmado = json.loads((raiz / e["registro"]).read_text(encoding="utf8"))
    if huellas(raiz, n) != firmado["huellas"]:
        raise ErrorEnmienda("la enmienda cambió después de firmarse; un cambio exige una enmienda nueva")
    rutas = [Path(s) if Path(s).is_absolute() else raiz / s for s in scripts]
    faltan = [str(p) for p in rutas if not p.is_file()]
    if faltan:
        raise ErrorEnmienda(f"no existen: {faltan}")
    codigo = huellas_codigo(raiz)
    t = _ahora(ahora)
    registro = {
        "enmienda": firmado["enmienda"], "huellas_enmienda": firmado["huellas"],
        "scripts": {str(p.relative_to(raiz) if p.is_relative_to(raiz) else p): sha256_archivo(p) for p in rutas},
        "codigo_evaluacion": codigo,
        "codigo_cambiado_desde_firma": sorted(k for k in set(codigo) | set(firmado["codigo_evaluacion"])
                                              if codigo.get(k) != firmado["codigo_evaluacion"].get(k)),
        "momento_utc": t.isoformat(), "momento_epoch": int(t.timestamp()),
    }
    (raiz / e["corrida"]).write_text(json.dumps(registro, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    return raiz / e["corrida"]
