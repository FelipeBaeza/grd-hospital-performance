"""Archivos sintéticos con el formato de FONASA (129 columnas separadas por |) para probar la Fase 3 sin usar datos reales."""
from __future__ import annotations

from pathlib import Path

from prep import esquema

DEFECTO = {
    "COD_HOSPITAL": "101100", "CIP_ENCRIPTADO": "123456", "SEXO": "HOMBRE", "FECHA_NACIMIENTO": "1960-01-01", "TIPO_PROCEDENCIA": "SERVICIO EMERGENCIA (DOMICILIO)",
    "TIPO_INGRESO": "URGENCIA", "TIPO_ACTIVIDAD": "HOSPITALIZACIÓN", "FECHA_INGRESO": "2020-03-02", "SERVICIOINGRESO": "MEDICINA ADULTO",
    "FECHAALTA": "2020-03-07", "TIPOALTA": "DOMICILIO", "DIAGNOSTICO1": "J18.9", "IR_29301_COD_GRD": "041101", "IR_29301_SEVERIDAD": "2", "IR_29301_MORTALIDAD": "1",
}


def fila(**kw) -> str:
    valores = {c: "" for c in esquema.COLUMNAS}
    valores.update(DEFECTO)
    valores.update(kw)
    return "|".join(valores[c] for c in esquema.COLUMNAS)


def escribir(ruta: Path, filas: list[str], bom: bool = False, crlf: bool = False, alias_id: bool = False, extra_lineas: list[str] | None = None) -> Path:
    enc = list(esquema.COLUMNAS)
    if alias_id:
        enc[esquema.POS_ID] = "ID_BENEFICIARIO"
    nl = "\r\n" if crlf else "\n"
    lineas = ["|".join(enc), *filas, *(extra_lineas or [])]
    datos = nl.join(lineas) + nl
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes((b"\xef\xbb\xbf" if bom else b"") + datos.encode("utf8"))
    return ruta
