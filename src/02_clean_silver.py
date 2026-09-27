"""
02_clean_silver.py
------------------
Paso 2: Limpieza, Tipado y Estandarización a Capa Silver.

Lee los archivos Parquet de data/bronze/, corrige inconsistencias de formato,
tipifica fechas y números decimales, normaliza textos categóricos, aísla datos sensibles
y deriva las variables objetivo preliminares (MORTALIDAD_BINARIA, ESTADO_CENSURA y ESTANCIA).
"""

import sys
import argparse
from pathlib import Path
import polars as pl


def parsear_fechas(col_name: str) -> pl.Expr:
    """Parsea fechas admitiendo formato ISO (yyyy-mm-dd) y DMY (dd-mm-yyyy)."""
    return pl.coalesce(
        pl.col(col_name).str.to_date("%Y-%m-%d", strict=False),
        pl.col(col_name).str.to_date("%d-%m-%Y", strict=False),
    )


def limpiar_y_estandarizar(df: pl.DataFrame, anio: int) -> pl.DataFrame:
    """Aplica todas las transformaciones de limpieza de la Fase 2 sobre un año."""

    # 1. Mapeo explícito de TIPOALTA a mortalidad y censura
    tipoalta_clean = pl.col("TIPOALTA").str.strip_chars().str.to_uppercase()
    es_fallecido = tipoalta_clean == "FALLECIDO"
    es_sobreviviente = tipoalta_clean.is_in(["DOMICILIO", "ALTA VOLUNTARIA", "FUGA DEL PACIENTE"])
    es_censurado = tipoalta_clean.str.contains(
        "DERIVACIÓN|HOSPITALIZACIÓN DOMICILIARIA|DESCONOCIDO|NO IDENTIFICADA"
    )

    # 2. Fechas normalizadas
    fecha_nac = parsear_fechas("FECHA_NACIMIENTO")
    fecha_ing = parsear_fechas("FECHA_INGRESO")
    fecha_alta = parsear_fechas("FECHAALTA")

    # 3. Decimales (coma a punto)
    peso_decimal = pl.col("IR_29301_PESO").str.replace(",", ".").cast(pl.Float64, strict=False)

    # 4. Cálculo preliminar de estancia (días entre ingreso y alta)
    estancia_expr = (fecha_alta - fecha_ing).dt.total_days()
    # Si la fecha de alta es anterior al ingreso (error temporal), dejar nula
    estancia_valida = pl.when(estancia_expr >= 0).then(estancia_expr).otherwise(None)

    # 5. Columnas de texto categórico a estandarizar
    cols_categoricas = [
        "SEXO",
        "TIPO_INGRESO",
        "TIPO_PROCEDENCIA",
        "ETNIA",
        "PREVISION",
        "SERVICIO_SALUD",
        "COMUNA",
        "PROVINCIA",
        "NACIONALIDAD",
    ]
    exprs_categoricas = [
        pl.col(col).str.strip_chars().str.to_uppercase() for col in cols_categoricas if col in df.columns
    ]

    # 6. Diagnósticos en mayúsculas y limpios (DIAGNOSTICO1 a DIAGNOSTICO35)
    cols_dx = [f"DIAGNOSTICO{i}" for i in range(1, 36) if f"DIAGNOSTICO{i}" in df.columns]
    exprs_dx = [pl.col(col).str.strip_chars().str.to_uppercase() for col in cols_dx]

    # 7. Procedimientos (mantener como texto preservando ceros a la izquierda)
    cols_px = [f"PROCEDIMIENTO{i}" for i in range(1, 31) if f"PROCEDIMIENTO{i}" in df.columns]
    exprs_px = [pl.col(col).str.strip_chars().str.to_uppercase() for col in cols_px]

    # Aplicar transformaciones principales
    df_silver = df.with_columns(
        *exprs_categoricas,
        *exprs_dx,
        *exprs_px,
        FECHA_NACIMIENTO=fecha_nac,
        FECHA_INGRESO=fecha_ing,
        FECHAALTA=fecha_alta,
        IR_29301_PESO_NUM=peso_decimal,
        ESTANCIA_DIAS=estancia_valida,
        MORTALIDAD_BINARIA=pl.when(es_fallecido)
        .then(1)
        .when(es_sobreviviente)
        .then(0)
        .otherwise(None)
        .cast(pl.Int8),
        ESTADO_CENSURA=pl.when(es_fallecido | es_sobreviviente)
        .then(pl.lit("NO_CENSURADO"))
        .otherwise(pl.lit("CENSURADO")),
        ANIO_EGRESO=pl.lit(anio),
    )

    # Aislar columna FECHAPROCEDIMIENTO1 si existe (contiene 9 RUTs detectados)
    if "FECHAPROCEDIMIENTO1" in df_silver.columns:
        df_silver = df_silver.with_columns(pl.lit(None).alias("FECHAPROCEDIMIENTO1"))

    return df_silver


def main():
    parser = argparse.ArgumentParser(description="Paso 2: Limpieza y Estandarización a Capa Silver")
    parser.add_argument("--bronze-dir", type=str, default="data/bronze", help="Directorio con archivos Parquet Bronze")
    parser.add_argument("--output-dir", type=str, default="data/silver", help="Directorio destino Parquet Silver")
    args = parser.parse_args()

    dir_bronce = Path(args.bronze_dir)
    dir_silver = Path(args.output_dir)
    dir_silver.mkdir(parents=True, exist_ok=True)

    archivos_bronze = sorted(dir_bronce.glob("grd_*.parquet"))
    if not archivos_bronze:
        print(f"[ERROR] No se encontraron archivos en '{dir_bronce}'. Ejecuta primero 01_ingest_bronze.py")
        sys.exit(1)

    print("=" * 70)
    print("LIMPIEZA Y TIPADO A CAPA SILVER")
    print(f"Archivos encontrados: {len(archivos_bronze)}")
    print(f"Destino: {dir_silver}")
    print("=" * 70)

    total_filas = 0
    total_fallecidos = 0
    total_censurados = 0

    for ruta in archivos_bronze:
        # Extraer año del nombre (ej. grd_2020.parquet -> 2020)
        nombre = ruta.stem
        partes = nombre.split("_")
        anio = int(partes[-1]) if partes[-1].isdigit() else 2020

        df_bronce = pl.read_parquet(ruta)
        df_silver = limpiar_y_estandarizar(df_bronce, anio)

        ruta_salida = dir_silver / f"silver_{anio}.parquet"
        df_silver.write_parquet(ruta_salida, compression="zstd")

        n_filas = len(df_silver)
        n_m = df_silver.filter(pl.col("MORTALIDAD_BINARIA") == 1).height
        n_c = df_silver.filter(pl.col("ESTADO_CENSURA") == "CENSURADO").height

        total_filas += n_filas
        total_fallecidos += n_m
        total_censurados += n_c

        print(
            f"✓ {ruta.name} -> {ruta_salida.name} ({n_filas:,} filas | {n_m:,} muertes | {n_c:,} censurados)".replace(
                ",", "."
            )
        )

    tasa_mortalidad = (total_fallecidos / (total_filas - total_censurados)) * 100
    pct_censura = (total_censurados / total_filas) * 100

    print("=" * 70)
    print(f"✓ Capa Silver construida con éxito.")
    print(f"Total registros: {total_filas:,}".replace(",", "."))
    print(f"Fallecidos confirmados: {total_fallecidos:,} (Tasa observada: {tasa_mortalidad:.2f}%)".replace(",", "."))
    print(f"Censurados (traslados): {total_censurados:,} ({pct_censura:.2f}% de la cohorte)".replace(",", "."))
    print("=" * 70)


if __name__ == "__main__":
    main()
