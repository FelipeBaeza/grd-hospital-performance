"""Cohortes de la segunda revisión del profesor guía (02-11-2026): las opciones nuevas de ``prep/cohortes.py``, una a una y juntas.

Uso (después de ``run_fase3.py``, con las capas ya construidas):

    python run_cohortes_revision.py --capas <carpeta de trabajo>

Escribe las cohortes de cada variante en ``<capas>/cohortes_revision/<variante>/`` (nunca en ``<capas>/cohortes``, que alimenta a las Fases 4
y 5) y la tabla ``salidas/tablas/cascada_revision.csv``. Solo trabaja con 2019–2023: las opciones cambian D y E, que en 2024 no existen
hasta la Fase 5. Comprueba que la variante con las reglas de la configuración (umbrales v2.0: las cuatro de la revisión) es idéntica,
fila por fila, a la cohorte oficial de ``<capas>/cohortes``; la variante «base» es la cohorte de la Fase 2, sin reglas.

Privacidad: la tabla publica solo el período 2019–2023 de cada variante. Por año, la diferencia con la cascada oficial (que se publica por
año) dejaría despejar pasos pequeños, como las filas repetidas de 2019.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from prep import cohortes, elixhauser
from prep.config import ANIOS_RESULTADO, DIR_SALIDAS, cargar_umbrales

VARIANTES = {
    "base": cohortes.Opciones(),
    "hosp_domiciliaria_viva": cohortes.opciones_revision(True, False, False, False),
    "sin_filas_repetidas": cohortes.opciones_revision(False, True, False, False),
    "sin_entrantes_enlazados": cohortes.opciones_revision(False, False, True, False),
    "mdc14_completo": cohortes.opciones_revision(False, False, False, True),
    "revision_completa": cohortes.opciones_revision(),
}
DESCRIPCION = {
    "base": "Cohorte de la Fase 2 (sin reglas nuevas)",
    "hosp_domiciliaria_viva": "Hospitalización domiciliaria como egreso vivo",
    "sin_filas_repetidas": "Sin filas repetidas (Q15)",
    "sin_entrantes_enlazados": "Sin traslados entrantes enlazados",
    "mdc14_completo": "MDC 14 completo fuera",
    "revision_completa": "Las cuatro reglas de la revisión",
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--capas", type=Path, required=True)
    a = p.parse_args(argv)
    u, e = cargar_umbrales(), elixhauser.cargar()
    op_oficial = cohortes.opciones_config(u)
    d_p, oficial, salida = a.capas / "plata", a.capas / "cohortes", a.capas / "cohortes_revision"
    filas = []
    for nombre, op in VARIANTES.items():
        casc = []
        for anio in ANIOS_RESULTADO:
            print(f"[{nombre}] {anio}", flush=True)
            casc.append(cohortes.construir_cohorte(anio, d_p, salida / nombre, u, e, op))
            if op == op_oficial:
                nuevo = pl.read_parquet(salida / nombre / f"cohorte_{anio}.parquet")
                if not nuevo.equals(pl.read_parquet(oficial / f"cohorte_{anio}.parquet")):
                    raise SystemExit(f"la variante {nombre} de {anio} no es idéntica a la cohorte oficial")
        c = pl.concat(casc)
        if not all(cohortes.cuadra(c.filter(pl.col("periodo") == str(x))) for x in ANIOS_RESULTADO):
            raise SystemExit(f"la cascada de {nombre} no cuadra")
        tot = c.group_by("orden", "paso", "tipo", "descripcion").agg(pl.col("n").sum()).sort("orden")
        n0 = int(tot.filter(pl.col("paso") == "TOTAL")["n"][0])
        for r in tot.iter_rows(named=True):
            filas.append({"variante": nombre, "descripcion_variante": DESCRIPCION[nombre], "periodo": "2019-2023", "orden": r["orden"],
                          "paso": r["paso"], "descripcion": r["descripcion"], "n": r["n"], "prop_del_total": r["n"] / n0})
    t = pl.DataFrame(filas)
    minimo = int(u["celda_minima"])
    if t.filter((pl.col("n") >= 1) & (pl.col("n") < minimo)).height:
        raise SystemExit("hay un paso de 1 a 9 egresos en el período: revisar antes de publicar")
    (DIR_SALIDAS / "tablas").mkdir(parents=True, exist_ok=True)
    t.write_csv(DIR_SALIDAS / "tablas" / "cascada_revision.csv")
    resumen = t.filter(pl.col("paso").is_in(["C", "D", "E", "EX06", "EX07", "EX10", "EX11", "EX12", "EX03"])).pivot(
        on="paso", index="variante", values="n")
    print(resumen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
