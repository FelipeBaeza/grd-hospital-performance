"""Regenera las tablas por subgrupo publicadas, sin conteos pequeños reconstruibles (incidente I-11). Desde
fase5_evaluacion/:

    python regenerar_subgrupos.py --capas <capas> --trabajo5 <trabajo5>

Para cada año publicado (ensayo 2023 y evaluación 2024):

1. Arma las mismas cohortes que la Fase 5 y aplica los modelos congelados. No entrena nada: corre dentro de
   ``sin_ajuste.prohibir_ajustes``.
2. Control: con el método anterior (``agrupar_pequenos=False``), las tablas deben ser idénticas byte a byte a las
   publicadas. Si no, se detiene sin escribir nada.
3. Escribe las tablas con los grupos pequeños combinados.

En 2024 lee el desenlace ya abierto (``trabajo5/evaluacion_2024/desenlace_2024.parquet``), no el archivo sellado.
Es una lectura más de un archivo ya abierto, autorizada por el autor el 22-10-2026, y queda anotada en I-11.
"""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd
import polars as pl

from evaluacion import orquestador, prediccion, sin_ajuste, subgrupos
from evaluacion.config import DIR_SALIDAS, RUTA_PROTOCOLO, cargar_protocolo

CARPETAS = {2023: DIR_SALIDAS / "ensayo_2023" / "tablas", 2024: DIR_SALIDAS / "evaluacion_2024" / "tablas"}


def _csv(df: pd.DataFrame) -> bytes:
    b = io.StringIO()
    df.to_csv(b, index=False)
    return b.getvalue().encode("utf8")


def tablas_del_anio(capas: Path, trabajo5: Path, anio: int, minimo: int) -> dict[str, dict[bool, bytes]]:
    tope = json.loads((capas / "oro" / "parametros_oro.json").read_text())["tope_estancia_dias"]
    des = None
    if anio == 2024:
        des = pl.read_parquet(trabajo5 / "evaluacion_2024" / "desenlace_2024.parquet", columns=["ID_EPISODIO", "ROL", "ESTANCIA_DIAS", "ES_TRASLADO_SALIENTE"]).to_pandas()
    mort, est, _ = orquestador._cohortes_anio(capas, anio, tope, des)
    out = {}
    with sin_ajuste.prohibir_ajustes():
        for tarea, d in (("mortalidad", mort), ("estadia", est)):
            _, p = prediccion.predecir(d, prediccion.carpeta_final(tarea))
            out[tarea] = {ag: _csv(subgrupos.tabla(d, d["y"].to_numpy(float), p, tarea, minimo, agrupar_pequenos=ag)) for ag in (False, True)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenera las tablas por subgrupo sin conteos pequeños reconstruibles (I-11)")
    ap.add_argument("--capas", required=True)
    ap.add_argument("--trabajo5", required=True)
    a = ap.parse_args()
    prot = cargar_protocolo(RUTA_PROTOCOLO)
    nuevas, informe = {}, []
    for anio, carpeta in CARPETAS.items():
        t = tablas_del_anio(Path(a.capas), Path(a.trabajo5), anio, prot.hospital["celda_minima"])
        for tarea, v in t.items():
            ruta = carpeta / f"subgrupos_{tarea}.csv"
            if v[False] != ruta.read_bytes():
                print(f"ERROR: el método anterior no reproduce {ruta.relative_to(DIR_SALIDAS)}; no se escribe nada", file=sys.stderr)
                return 1
            nuevas[ruta] = v[True]
            informe.append({"archivo": str(ruta.relative_to(DIR_SALIDAS)), "sha256_anterior": hashlib.sha256(v[False]).hexdigest(),
                            "sha256_nuevo": hashlib.sha256(v[True]).hexdigest()})
            print(f"{ruta.relative_to(DIR_SALIDAS)}: el método anterior reproduce la tabla publicada; se reemplaza")
    for ruta, b in nuevas.items():
        ruta.write_bytes(b)
    pd.DataFrame(informe).to_csv(DIR_SALIDAS / "tablas" / "regeneracion_subgrupos_I-11.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
