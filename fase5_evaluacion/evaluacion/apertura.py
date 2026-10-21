"""Apertura única del desenlace de 2024. Se exige, en este orden: autorización escrita, protocolo con la misma huella que se registró antes, modelos intactos y ninguna lectura previa concedida.
La lectura pasa por el cortafuegos de la Fase 3 (prep.sellado), que la anota en el registro de accesos."""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

from .config import sha256_archivo


class AperturaNoAutorizada(PermissionError):
    pass


def lecturas_concedidas(registro: Path) -> int:
    if not Path(registro).exists():
        return 0
    with open(registro, newline="", encoding="utf8") as f:
        return sum(1 for r in csv.DictReader(f) if r["evento"] == "lectura_desenlace" and r["anio"] == "2024" and r["concedido"] == "si")


def comprobar_autorizacion(ruta: Path, frase: str) -> dict:
    """El archivo debe existir y traer la frase, quién autoriza y cuándo (líneas ``firmante:`` y ``fecha:`` con contenido)."""
    if not Path(ruta).exists():
        raise AperturaNoAutorizada("falta la autorización escrita del profesor guía")
    texto = Path(ruta).read_text(encoding="utf8")
    campos = {}
    for linea in texto.splitlines():
        if ":" in linea:
            k, v = linea.split(":", 1)
            campos[k.strip().lower()] = v.strip()
    if frase not in texto or not campos.get("firmante") or not campos.get("fecha"):
        raise AperturaNoAutorizada("la autorización debe incluir la frase, el firmante y la fecha")
    return campos


def registrar_preapertura(destino: Path, protocolo, huellas: dict) -> Path:
    """Guarda la huella del protocolo y de lo congelado ANTES de abrir; la apertura exige que sigan iguales."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({"huella_protocolo": protocolo.huella(), "huellas_congeladas": huellas, "momento_epoch": int(time.time())}, indent=1), encoding="utf8")
    return destino


def abrir_2024(dir_sellado: Path, protocolo, preapertura: Path, autorizacion: Path, registro: Path, verificar_modelos, destino: Path):
    """Devuelve el desenlace de 2024 (polars) y lo guarda en ``destino``. Falla si algo del orden anterior no se cumple."""
    from prep import sellado
    comprobar_autorizacion(autorizacion, protocolo.apertura["frase"])
    if not Path(preapertura).exists():
        raise AperturaNoAutorizada("falta el registro previo a la apertura (huella del protocolo)")
    pre = json.loads(Path(preapertura).read_text(encoding="utf8"))
    if pre["huella_protocolo"] != protocolo.huella():
        raise AperturaNoAutorizada("el protocolo cambió después de registrarse")
    verificar_modelos()
    if lecturas_concedidas(registro) >= protocolo.apertura["lecturas_concedidas_permitidas"]:
        raise AperturaNoAutorizada("2024 ya fue abierto: solo se permite una lectura")
    os.environ[sellado.VARIABLE_AUTORIZACION] = sellado.FRASE
    try:
        d = sellado.leer_desenlace(dir_sellado, 2024, proposito="Fase 5 evaluación: apertura unica")
    finally:
        os.environ.pop(sellado.VARIABLE_AUTORIZACION, None)
    if lecturas_concedidas(registro) != 1:
        raise AperturaNoAutorizada("el registro no muestra exactamente una lectura concedida")
    destino.parent.mkdir(parents=True, exist_ok=True)
    d.write_parquet(destino)
    (destino.parent / "huella_desenlace_2024.json").write_text(json.dumps({"sha256": sha256_archivo(destino)}), encoding="utf8")
    return d
