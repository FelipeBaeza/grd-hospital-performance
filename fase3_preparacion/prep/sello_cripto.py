"""Sello criptográfico para un año de prueba nuevo (plan de corrección 4.5; hallazgo H-16).

El sello de 2024 era procedimental: una frase escrita en el código y un registro de accesos que una lectura directa
del archivo no toca (incidente I-05). Para un año nuevo (2025) se propone esto:

1. Apenas llegan los datos, el **profesor guía** cifra el archivo del desenlace con una clave que solo él conoce:
   ``python -m prep.sello_cripto sellar <desenlace_2025.parquet>``. Se escribe ``<archivo>.sellado`` (AES-256-CBC
   con PBKDF2, de OpenSSL) y ``<archivo>.sellado.json``, con las huellas SHA-256 del original y del cifrado y la
   hora.
2. El original se borra. El .json se sube a git: la hora de GitHub prueba que el sello existía antes de evaluar.
3. Después de pre-registrar el protocolo, el profesor abre el archivo con su clave:
   ``python -m prep.sello_cripto abrir <archivo>.sellado <destino>``. La apertura comprueba que el resultado tenga
   la huella registrada.

Sin la clave no hay lectura posible, ni siquiera del esquema. La clave se pide por teclado; solo para pruebas
automáticas se acepta la variable de entorno SELLO_CLAVE_PRUEBA.
"""
from __future__ import annotations

import argparse
import datetime as dt
import getpass
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

CIFRADO = ["-aes-256-cbc", "-pbkdf2", "-iter", "600000", "-salt"]


class SelloError(RuntimeError):
    pass


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _clave(confirmar: bool) -> str:
    if os.environ.get("SELLO_CLAVE_PRUEBA"):
        return os.environ["SELLO_CLAVE_PRUEBA"]
    c = getpass.getpass("Clave del sello (la guarda el profesor guía): ")
    if confirmar and c != getpass.getpass("Repítela: "):
        raise SelloError("las claves no coinciden")
    if len(c) < 12:
        raise SelloError("la clave debe tener al menos 12 caracteres")
    return c


def _openssl(args: list[str], clave: str) -> None:
    r = subprocess.run(["openssl", "enc", *CIFRADO, *args, "-pass", "fd:0"], input=clave.encode(), capture_output=True)
    if r.returncode != 0:
        raise SelloError("OpenSSL no pudo completar la operación (¿clave equivocada o archivo dañado?)")


def sellar(original: Path | str, clave: str | None = None) -> Path:
    original = Path(original)
    destino = original.with_name(original.name + ".sellado")
    if destino.exists():
        raise SelloError(f"ya existe {destino.name}: un sello no se sobrescribe")
    clave = clave or _clave(confirmar=True)
    _openssl(["-e", "-in", str(original), "-out", str(destino)], clave)
    t = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    reg = {"archivo": original.name, "sha256_original": _sha256(original), "sha256_sellado": _sha256(destino),
           "cifrado": "openssl enc " + " ".join(CIFRADO), "momento_utc": t.isoformat()}
    destino.with_name(destino.name + ".json").write_text(json.dumps(reg, indent=1, ensure_ascii=False) + "\n", encoding="utf8")
    return destino


def abrir(sellado: Path | str, destino: Path | str, clave: str | None = None) -> Path:
    sellado, destino = Path(sellado), Path(destino)
    reg = json.loads(sellado.with_name(sellado.name + ".json").read_text(encoding="utf8"))
    if _sha256(sellado) != reg["sha256_sellado"]:
        raise SelloError("el archivo sellado no es el registrado")
    _openssl(["-d", "-in", str(sellado), "-out", str(destino)], clave or _clave(confirmar=False))
    if _sha256(destino) != reg["sha256_original"]:
        destino.unlink(missing_ok=True)
        raise SelloError("lo abierto no coincide con la huella del original")
    return destino


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sello criptográfico del desenlace de un año de prueba")
    sub = ap.add_subparsers(dest="accion", required=True)
    s = sub.add_parser("sellar")
    s.add_argument("original")
    a_ = sub.add_parser("abrir")
    a_.add_argument("sellado")
    a_.add_argument("destino")
    a = ap.parse_args(argv)
    try:
        print("escrito:", sellar(a.original) if a.accion == "sellar" else abrir(a.sellado, a.destino))
    except SelloError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
