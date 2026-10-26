"""Análisis posteriores de la Fase 2 que pidió el profesor guía (cambio 1.6 del protocolo): R1 a R5.

Uso (después de la corrida principal, para reutilizar la caché Parquet):

    python run_revision_profesor.py --fuente completo --data-root <carpeta data seminario_full>

Salidas en ``salidas/revision_profesor/<fuente>/`` (tablas, figuras y registros, con su propio registro de accesos). No toca las salidas de la corrida
principal ni el protocolo pre-registrado: son análisis descriptivos agregados después de abrir 2024 y se declaran así.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from eda import revision_profesor, rutas
from eda.contexto import Contexto
from eda import privacidad


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Análisis posteriores pedidos por el profesor guía (R1 a R5)")
    p.add_argument("--fuente", choices=rutas.FUENTES_VALIDAS, default="completo")
    p.add_argument("--data-root", type=Path, default=None, help="carpeta con los archivos de egresos (obligatoria con --fuente completo)")
    p.add_argument("--salida-base", type=Path, default=rutas.DIR_SALIDAS / "revision_profesor")
    a = p.parse_args(argv)
    if a.fuente == "completo" and a.data_root is None:
        p.error("--data-root es obligatorio con --fuente completo")
    t0 = time.time()
    ctx = Contexto.crear(a.fuente, data_root=a.data_root, dir_salida_base=a.salida_base)
    revision_profesor.ejecutar(ctx)
    ctx.escribir_registros()
    esc = privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)
    n = privacidad.total_violaciones(esc)
    print(f"Revisión del profesor, fuente {a.fuente}: {len(ctx.indice_tablas())} tablas y {len(ctx.figuras.ids())} figuras en {ctx.dir_salida}")
    print(f"Privacidad: {n} violaciones. Tiempo: {time.time() - t0:,.0f} s".replace(",", "."))
    return 0 if n == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
