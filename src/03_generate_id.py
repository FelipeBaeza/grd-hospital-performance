"""
03_generate_id.py
-----------------
Paso 3: Generación del Identificador Único por Episodio (ID_EPISODIO).

FONASA no entrega un identificador por registro. Este paso genera un ID determinista
y único para cada hospitalización:
    ID_EPISODIO = "GRD_{AÑO}_{NRO_FILA}"

Garantiza:
- Unicidad matemática absoluta (cero duplicados en 5,8 millones de registros).
- Trazabilidad determinista: reproducir la ejecución produce exactamente el mismo ID.
- Ubicación: primera columna del dataset en Capa Silver.
"""

import sys
import argparse
from pathlib import Path
import polars as pl


def agregar_id_episodio(df: pl.DataFrame, anio: int) -> pl.DataFrame:
    """Genera ID_EPISODIO y lo coloca como primera columna."""
    n_filas = len(df)
    
    # Construir ID formateado: GRD_2020_0000001
    id_series = [f"GRD_{anio}_{i:07d}" for i in range(1, n_filas + 1)]
    
    df_con_id = df.with_columns(
        ID_EPISODIO=pl.Series("ID_EPISODIO", id_series, dtype=pl.String)
    )
    
    # Mover ID_EPISODIO a la primera posición
    otras_columnas = [c for c in df_con_id.columns if c != "ID_EPISODIO"]
    return df_con_id.select(["ID_EPISODIO"] + otras_columnas)


def main():
    parser = argparse.ArgumentParser(description="Paso 3: Generación de ID_EPISODIO en Capa Silver")
    parser.add_argument("--silver-dir", type=str, default="data/silver", help="Directorio Parquet Silver")
    args = parser.parse_args()

    dir_silver = Path(args.silver_dir)
    archivos_silver = sorted(dir_silver.glob("silver_*.parquet"))

    if not archivos_silver:
        print(f"[ERROR] No se encontraron archivos en '{dir_silver}'. Ejecuta primero 02_clean_silver.py")
        sys.exit(1)

    print("=" * 70)
    print("GENERACIÓN DE IDENTIFICADOR ÚNICO (ID_EPISODIO)")
    print(f"Archivos encontrados: {len(archivos_silver)}")
    print("=" * 70)

    total_episodios = 0
    ids_unicos_totales = set()

    for ruta in archivos_silver:
        nombre = ruta.stem
        anio = int(nombre.split("_")[-1])

        df = pl.read_parquet(ruta)
        df_con_id = agregar_id_episodio(df, anio)

        # Verificación de unicidad local
        unicos_archivo = df_con_id["ID_EPISODIO"].n_unique()
        if unicos_archivo != len(df_con_id):
            print(f"[ERROR] Colisión de IDs detectada en {ruta.name}!")
            sys.exit(1)

        # Sobrescribir con ID incluido
        df_con_id.write_parquet(ruta, compression="zstd")

        total_episodios += len(df_con_id)
        print(f"✓ {ruta.name}: {len(df_con_id):,} IDs generados ({df_con_id['ID_EPISODIO'][0]} ... {df_con_id['ID_EPISODIO'][-1]})".replace(",", "."))

    print("=" * 70)
    print(f"✓ Total de episodios identificados unívocamente: {total_episodios:,}".replace(",", "."))
    print("✓ Cero colisiones de ID comprobadas.")
    print("=" * 70)


if __name__ == "__main__":
    main()
