"""Catálogo de tipo de establecimiento de los hospitales GRD, desde el registro del DEIS (plan de corrección 1.3;
hallazgos H-01 y H-18). Desde fase3_preparacion/:

    python catalogo_deis.py

Lee `config/fuentes/DEIS_Base_de_Establecimientos_2025.xlsx` (DEIS, Base de Establecimientos 2025; archivo entregado
por el autor el 22-10-2026) y escribe `config/tipo_establecimiento.csv`, con una fila por código de hospital GRD.

La clave es `EstablecimientoCodigo` del DEIS, que coincide con `COD_HOSPITAL` en los 72 hospitales GRD.

Regla de `tipo`, en este orden:

1. **instituto monográfico:** el nombre DEIS empieza con «Instituto».
2. **pediátrico:** el nombre DEIS contiene «Niños», o el hospital está en `PEDIATRICOS_POR_NOMBRE_PROPIO` (su nombre
   no lo dice). Ninguno llega al volumen mínimo de adultos, así que no entran a ningún embudo.
3. **general de alta / mediana / baja complejidad:** según `NivelComplejidadEstabGlosa` del DEIS.

El DEIS no distingue los institutos monográficos ni los hospitales pediátricos: todos son «Hospital» de alta o
mediana complejidad. Por eso las reglas 1 y 2 usan el nombre.

También agrega `primer_anio_grd`, el primer año con egresos en `salidas/tablas/catalogo_hospitales_anio.csv`
(plan 1.4).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
FUENTE = RAIZ / "config" / "fuentes" / "DEIS_Base_de_Establecimientos_2025.xlsx"
CATALOGO_ANIOS = RAIZ / "salidas" / "tablas" / "catalogo_hospitales_anio.csv"
DESTINO = RAIZ / "config" / "tipo_establecimiento.csv"
PEDIATRICOS_POR_NOMBRE_PROPIO = {"113130": "Hospital Dr. Exequiel González Cortés: hospital pediátrico del Servicio de Salud Metropolitano Sur"}
COMPLEJIDAD = {"Alta Complejidad": "general de alta complejidad", "Mediana Complejidad": "general de mediana complejidad",
               "Baja Complejidad": "general de baja complejidad"}


def tipo(cod: str, nombre: str, complejidad: str) -> tuple[str, str]:
    if nombre.strip().lower().startswith("instituto"):
        return "instituto monográfico", "nombre DEIS empieza con «Instituto»"
    if "niños" in nombre.lower():
        return "pediátrico", "nombre DEIS contiene «Niños»"
    if cod in PEDIATRICOS_POR_NOMBRE_PROPIO:
        return "pediátrico", PEDIATRICOS_POR_NOMBRE_PROPIO[cod]
    if complejidad in COMPLEJIDAD:
        return COMPLEJIDAD[complejidad], "NivelComplejidadEstabGlosa del DEIS"
    raise ValueError(f"{cod}: nivel de complejidad DEIS desconocido ({complejidad!r})")


def construir(fuente: Path = FUENTE, catalogo_anios: Path = CATALOGO_ANIOS) -> pd.DataFrame:
    deis = pd.read_excel(fuente, dtype=str)
    deis["EstablecimientoCodigo"] = deis["EstablecimientoCodigo"].str.strip()
    anios = pd.read_csv(catalogo_anios, dtype={"cod_hospital": str})
    primer = anios[anios["n_egresos"] > 0].groupby("cod_hospital")["anio"].min()
    cods = sorted(anios["cod_hospital"].unique())
    m = deis.set_index("EstablecimientoCodigo")
    faltan = [c for c in cods if c not in m.index]
    if faltan:
        raise ValueError(f"códigos GRD sin fila en el DEIS: {faltan}")
    filas = []
    for c in cods:
        r = m.loc[c]
        t, regla = tipo(c, r["EstablecimientoGlosa"], r["NivelComplejidadEstabGlosa"])
        filas.append({"cod_hospital": c, "nombre_deis": r["EstablecimientoGlosa"].strip(), "nivel_complejidad_deis": r["NivelComplejidadEstabGlosa"],
                      "nivel_atencion_deis": r["NivelAtencionEstabglosa"], "tipo": t, "regla": regla, "primer_anio_grd": int(primer[c])})
    return pd.DataFrame(filas)


def main() -> int:
    t = construir()
    t.to_csv(DESTINO, index=False)
    h = hashlib.sha256(FUENTE.read_bytes()).hexdigest()
    print(f"escrito {DESTINO.relative_to(RAIZ)}: {len(t)} hospitales; fuente SHA-256 {h}")
    print(t["tipo"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
