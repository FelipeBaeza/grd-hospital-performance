"""Registro de la enmienda 01 (protocolo_fase5_v2.toml + documentos/ENMIENDA_01.md).

La enmienda vale solo cuando existe documentos/registro_enmienda_01.json. Ese registro guarda las huellas SHA-256 de
los dos archivos de la enmienda, de la constancia del profesor guía y del código de evaluacion/, con la hora. Antes de
la primera corrida, registro_corrida_enmienda_01.json agrega la huella de los scripts que corren los análisis y
comprueba que la enmienda no cambió desde la firma.
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


class ErrorEnmienda(RuntimeError):
    pass


def _ahora(ahora: dt.datetime | None) -> dt.datetime:
    return (ahora or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc).replace(microsecond=0)


def huellas(raiz: Path = RAIZ) -> dict[str, str]:
    """Huellas de los dos archivos de la enmienda (las que van en la constancia del profesor)."""
    faltan = [str(p) for p in (PROTOCOLO_V2, TEXTO) if not (raiz / p).exists()]
    if faltan:
        raise ErrorEnmienda(f"faltan archivos de la enmienda: {faltan}")
    return {str(PROTOCOLO_V2): sha256_archivo(raiz / PROTOCOLO_V2), str(TEXTO): sha256_archivo(raiz / TEXTO)}


def huellas_codigo(raiz: Path = RAIZ) -> dict[str, str]:
    return {str(p.relative_to(raiz)): sha256_archivo(p) for p in sorted((raiz / "evaluacion").glob("*.py"))}


def _verificar_base(raiz: Path) -> dict:
    with open(raiz / PROTOCOLO_V2, "rb") as f:
        datos = tomllib.load(f)
    actual = sha256_archivo(raiz / PROTOCOLO_V1)
    if datos.get("huella_base") != actual:
        raise ErrorEnmienda(f"el protocolo v1.0 no coincide con la huella base declarada en la enmienda ({actual[:8]}…)")
    ids = [a["id"] for a in datos.get("analisis", [])]
    if not ids or len(ids) != len(set(ids)):
        raise ErrorEnmienda("la lista de análisis está vacía o tiene identificadores repetidos")
    return datos


def firmar(autor: str, profesor: str, fecha_firma: str, raiz: Path = RAIZ, sin_constancia: bool = False,
           ahora: dt.datetime | None = None, nota: str | None = None) -> Path:
    """Crea documentos/registro_enmienda_01.json. No sobrescribe un registro existente."""
    if (raiz / REGISTRO).exists():
        raise ErrorEnmienda("la enmienda 01 ya está registrada; un cambio exige una enmienda nueva")
    if not autor.strip() or not profesor.strip():
        raise ErrorEnmienda("faltan los nombres del autor y del profesor guía")
    dt.date.fromisoformat(fecha_firma)
    datos = _verificar_base(raiz)
    constancia = raiz / CONSTANCIA
    if not constancia.exists() and not sin_constancia:
        raise ErrorEnmienda(f"falta la constancia firmada del profesor guía ({CONSTANCIA})")
    t = _ahora(ahora)
    registro = {
        "enmienda": datos["enmienda"], "version": datos["version"],
        "huellas": huellas(raiz), "huella_protocolo_v1": datos["huella_base"],
        "constancia": {"ruta": str(CONSTANCIA), "sha256": sha256_archivo(constancia)} if constancia.exists()
                      else {"ruta": None, "sha256": None, "nota": "registrada sin constancia escrita del profesor guía"},
        "codigo_evaluacion": huellas_codigo(raiz),
        "analisis": [a["id"] for a in datos["analisis"]],
        "firmas": {"autor": autor.strip(), "profesor_guia": profesor.strip(), "fecha_firma": fecha_firma},
        "nota": (nota or "").strip() or None,
        "momento_utc": t.isoformat(), "momento_epoch": int(t.timestamp()),
    }
    (raiz / REGISTRO).write_text(json.dumps(registro, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    return raiz / REGISTRO


def antes_de_correr(scripts: list[str], raiz: Path = RAIZ, ahora: dt.datetime | None = None) -> Path:
    """Registra la huella de los scripts de la enmienda antes de la primera corrida. Falla si la enmienda cambió."""
    if not (raiz / REGISTRO).exists():
        raise ErrorEnmienda("la enmienda 01 no está firmada: falta documentos/registro_enmienda_01.json")
    if (raiz / REGISTRO_CORRIDA).exists():
        raise ErrorEnmienda("la corrida de la enmienda 01 ya está registrada")
    if not scripts:
        raise ErrorEnmienda("indica los scripts que corren los análisis de la enmienda")
    firmado = json.loads((raiz / REGISTRO).read_text(encoding="utf8"))
    if huellas(raiz) != firmado["huellas"]:
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
    (raiz / REGISTRO_CORRIDA).write_text(json.dumps(registro, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    return raiz / REGISTRO_CORRIDA
