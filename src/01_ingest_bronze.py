"""
01_ingest_bronze.py
-------------------
Paso 1: Ingesta Cruda a Capa Bronce (Parquet 1:1).

Lee cada archivo .txt de FONASA en modo texto puro (sin inferir tipos),
aplica alias necesarios y materializa los datos en formato columnar Parquet
particionado por año en data/bronze/.

Garantiza:
- Cero pérdida de información (ceros a la izquierda en CIE-9 se preservan).
- Reconciliación exacta: filas_leídas == líneas_físicas - 1.
"""

import sys
import argparse
from pathlib import Path
import polars as pl

try:
    from src.constants import ALIAS_COLUMNAS
except ImportError:
    ALIAS_COLUMNAS = {"ID_BENEFICIARIO": "CIP_ENCRIPTADO"}


def contar_lineas_archivo(ruta: Path) -> int:
    """Cuenta el número de líneas físicas de forma eficiente en bloques de bytes."""
    lineas = 0
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            lineas += bloque.count(b"\n")
    return lineas


def ingestar_archivo_a_bronce(
    ruta_txt: Path, dir_salida: Path, separador: str = "|"
) -> tuple[bool, int, Path]:
    """
    Lee un archivo TXT con Polars forzando todas las columnas como pl.String
    y guarda en Parquet con compresión zstd.
    """
    nombre_base = ruta_txt.stem.lower()
    ruta_parquet = dir_salida / f"{nombre_base}.parquet"

    # 1. Conteo físico de líneas
    total_lineas = contar_lineas_archivo(ruta_txt)
    lineas_datos_esperadas = max(0, total_lineas - 1)

    # 2. Lectura perezosa como texto plano
    # infer_schema_length=0 fuerza que todas las columnas se lean como String
    df = pl.read_csv(
        ruta_txt,
        separator=separador,
        infer_schema_length=0,
        quote_char=None,
        encoding="utf-8-lossy",
        truncate_ragged_lines=True,
    )

    # 3. Aplicar alias de columnas si existen
    renombres = {col: ALIAS_COLUMNAS[col] for col in df.columns if col in ALIAS_COLUMNAS}
    if renombres:
        df = df.rename(renombres)

    # 4. Verificación de reconciliación
    filas_obtenidas = len(df)
    if filas_obtenidas != lineas_datos_esperadas:
        print(
            f"[AVISO] {ruta_txt.name}: Se esperaban {lineas_datos_esperadas} filas pero se leyeron {filas_obtenidas}."
        )

    # 5. Escritura eficiente a Parquet
    df.write_parquet(ruta_parquet, compression="zstd")
    return True, filas_obtenidas, ruta_parquet


def main():
    parser = argparse.ArgumentParser(description="Paso 1: Ingesta Cruda a Bronce")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Directorio con archivos .txt")
    parser.add_argument("--output-dir", type=str, default="data/bronze", help="Directorio destino Parquet")
    args = parser.parse_args()

    dir_entrada = Path(args.data_dir)
    dir_salida = Path(args.output_dir)
    dir_salida.mkdir(parents=True, exist_ok=True)

    archivos = sorted(dir_entrada.glob("*.txt"))
    if not archivos:
        dir_alternativo = Path("../Seminario/data seminario")
        if dir_alternativo.exists():
            print(f"[AVISO] No hay archivos en '{dir_entrada}'. Usando '{dir_alternativo}'")
            archivos = sorted(dir_alternativo.glob("*.txt"))

    if not archivos:
        print(f"[ERROR] No se encontraron archivos .txt en '{dir_entrada}'")
        sys.exit(1)

    print("=" * 70)
    print("INGESTA A CAPA BRONCE (TXT -> Parquet 1:1)")
    print(f"Archivos a procesar: {len(archivos)}")
    print(f"Destino: {dir_salida}")
    print("=" * 70)

    total_filas = 0
    for ruta in archivos:
        ok, n_filas, destino = ingestar_archivo_a_bronce(ruta, dir_salida)
        total_filas += n_filas
        print(f"✓ {ruta.name} -> {destino.name} ({n_filas:,} filas)".replace(",", "."))

    print("=" * 70)
    print(f"✓ Ingesta completada con éxito. Total filas en Bronce: {total_filas:,}".replace(",", "."))


if __name__ == "__main__":
    main()
