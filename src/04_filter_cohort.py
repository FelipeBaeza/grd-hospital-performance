"""Paso 4: Filtrado de Cohorte según Diagrama CONSORT y Criterios Clínicos.

Aplica los criterios de inclusión/exclusión oficiales:
1. Exclusiones Duras (Cohorte de Mortalidad y Riesgo Basal):
   - EX01: GRD no agrupable o código de error (MDC nulo, 0, 99xxxx, DESCONOCIDO).
   - EX02: Recién nacidos y período perinatal (MDC 15).
   - EX03: Obstetricia no complicada (MDC 14, 0 comorbilidades Elixhauser, no fallecida).
2. Exclusiones Blandas (Cohorte de Estancia / Prolongada):
   - EX05: Estancia cero días (ambulatoria/hospital de día).
   - EX06: Paciente fallecido durante la estancia.
   - EX07: Cirugía Mayor Ambulatoria (CMA).
   - EX08: Incoherencia de fechas (estancia negativa).

Guarda:
- data/silver/silver_filtered_{anio}.parquet
- data/silver/consort_summary.csv
"""

import sys
from pathlib import Path
import time
import polars as pl

# Asegurar raíz en PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.elixhauser import calcular_elixhauser


def cargar_mapeo_mdc(config_path: Path | str = "config/mapeo_grd_mdc.csv") -> pl.DataFrame:
    """Carga la correspondencia oficial IR_29301_COD_GRD -> (MDC, GRD_BASE)."""
    mapeo = pl.read_csv(
        config_path,
        comment_prefix="#",
        schema_overrides={
            "codigo_grd": pl.String,
            "mdc": pl.Int32,
            "grd_base": pl.Int32,
        },
    )
    return (
        mapeo.select([
            pl.col("codigo_grd"),
            pl.col("mdc").alias("MDC"),
            pl.col("grd_base").alias("GRD_BASE"),
        ])
        .unique(subset=["codigo_grd"])
    )


def procesar_cohorte_anio(
    anio: int,
    mapeo_mdc: pl.DataFrame,
    silver_dir: Path = Path("data/silver"),
) -> dict:
    """Procesa un año individual, aplicando cascada CONSORT y guardando silver_filtered_{anio}.parquet."""
    ruta_silver = silver_dir / f"silver_{anio}.parquet"
    if not ruta_silver.exists():
        raise FileNotFoundError(f"No existe {ruta_silver}")

    t0 = time.time()
    df = pl.read_parquet(ruta_silver)
    n_inicial = len(df)

    # 1. Comorbilidades Elixhauser y Score Van Walraven
    elix = calcular_elixhauser(df)
    df = df.join(elix, on="ID_EPISODIO", how="left")

    # 2. Mapeo a MDC y GRD_BASE
    df = df.join(mapeo_mdc, left_on="IR_29301_COD_GRD", right_on="codigo_grd", how="left")

    # 3. Predicados de Exclusiones Duras (Cohorte de Mortalidad)
    ex01 = (
        df["IR_29301_COD_GRD"].is_null()
        | df["MDC"].is_null()
        | (df["MDC"] == 0)
        | df["IR_29301_COD_GRD"].str.starts_with("99")
        | (df["IR_29301_COD_GRD"] == "DESCONOCIDO")
    )
    ex02 = df["MDC"] == 15
    ex03 = (df["MDC"] == 14) & (df["N_COMORB_ELIX"] == 0) & (df["MORTALIDAD_BINARIA"] != 1)

    # Atribución jerárquica de motivo de exclusión
    motivo_dura = (
        pl.when(ex01)
        .then(pl.lit("EX01_NO_AGRUPABLE"))
        .when(ex02)
        .then(pl.lit("EX02_RECIEN_NACIDO"))
        .when(ex03)
        .then(pl.lit("EX03_OBSTETRICO_SIN_COMPLICACION"))
        .otherwise(None)
    )

    cohorte_dura = (~ex01) & (~ex02) & (~ex03)

    # 4. Predicados de Exclusiones Blandas (Cohorte de Estancia)
    ex05 = df["ESTANCIA_DIAS"] == 0
    ex06 = df["MORTALIDAD_BINARIA"] == 1
    ex07 = df["TIPO_ACTIVIDAD"].str.contains("(?i)CMA|AMBULATORIA")
    ex08 = df["ESTANCIA_DIAS"] < 0

    cohorte_estancia = cohorte_dura & (~ex05) & (~ex06) & (~ex07) & (~ex08)

    # 5. Añadir marcas y columnas derivadas
    df_filtrado = df.with_columns([
        ex01.alias("EX01_NO_AGRUPABLE"),
        ex02.alias("EX02_RECIEN_NACIDO"),
        ex03.alias("EX03_OBSTETRICO_SIN_COMPLICACION"),
        cohorte_dura.alias("EN_COHORTE_DURA"),
        motivo_dura.alias("MOTIVO_EXCLUSION_DURA"),
        ex05.alias("EX05_ESTANCIA_CERO"),
        ex06.alias("EX06_FALLECIDO"),
        ex07.alias("EX07_CMA"),
        ex08.alias("EX08_ESTANCIA_NEGATIVA"),
        cohorte_estancia.alias("EN_COHORTE_ESTANCIA"),
    ])

    # Guardar silver_filtered_{anio}.parquet
    ruta_salida = silver_dir / f"silver_filtered_{anio}.parquet"
    df_filtrado.write_parquet(ruta_salida, compression="zstd")

    tiempo_total = time.time() - t0

    metricas = {
        "anio": anio,
        "n_inicial": n_inicial,
        "ex01_no_agrupable": int(ex01.sum()),
        "ex02_recien_nacido": int(ex02.sum()),
        "ex03_obstetrico_sin_comp": int(ex03.sum()),
        "n_cohorte_dura": int(cohorte_dura.sum()),
        "pct_cohorte_dura": round(float(cohorte_dura.sum()) / n_inicial * 100, 2),
        "ex05_estancia_cero": int(ex05.sum()),
        "ex06_fallecido": int(ex06.sum()),
        "ex07_cma": int(ex07.sum()),
        "ex08_estancia_negativa": int(ex08.sum()),
        "n_cohorte_estancia": int(cohorte_estancia.sum()),
        "pct_cohorte_estancia": round(float(cohorte_estancia.sum()) / n_inicial * 100, 2),
        "tiempo_seg": round(tiempo_total, 2),
    }

    print(
        f"[{anio}] Procesados {n_inicial:,} episodios -> "
        f"Cohorte Dura: {metricas['n_cohorte_dura']:,} ({metricas['pct_cohorte_dura']}%) | "
        f"Cohorte Estancia: {metricas['n_cohorte_estancia']:,} ({metricas['pct_cohorte_estancia']}%) "
        f"en {tiempo_total:.2f}s"
    )

    return metricas


def main():
    print("=" * 80)
    print("EJECUTANDO PASO 4: FILTRADO DE COHORTE CONSORT (2019-2024)")
    print("=" * 80)

    silver_dir = Path("data/silver")
    mapeo_mdc = cargar_mapeo_mdc()

    anios = [2019, 2020, 2021, 2022, 2023, 2024]
    resumen = []

    for anio in anios:
        m = procesar_cohorte_anio(anio, mapeo_mdc, silver_dir)
        resumen.append(m)

    df_resumen = pl.DataFrame(resumen)

    # Fila de totales agregados
    total_inicial = df_resumen["n_inicial"].sum()
    total_dura = df_resumen["n_cohorte_dura"].sum()
    total_estancia = df_resumen["n_cohorte_estancia"].sum()

    fila_total = {
        "anio": "TOTAL (2019-2024)",
        "n_inicial": total_inicial,
        "ex01_no_agrupable": df_resumen["ex01_no_agrupable"].sum(),
        "ex02_recien_nacido": df_resumen["ex02_recien_nacido"].sum(),
        "ex03_obstetrico_sin_comp": df_resumen["ex03_obstetrico_sin_comp"].sum(),
        "n_cohorte_dura": total_dura,
        "pct_cohorte_dura": round(total_dura / total_inicial * 100, 2),
        "ex05_estancia_cero": df_resumen["ex05_estancia_cero"].sum(),
        "ex06_fallecido": df_resumen["ex06_fallecido"].sum(),
        "ex07_cma": df_resumen["ex07_cma"].sum(),
        "ex08_estancia_negativa": df_resumen["ex08_estancia_negativa"].sum(),
        "n_cohorte_estancia": total_estancia,
        "pct_cohorte_estancia": round(total_estancia / total_inicial * 100, 2),
        "tiempo_seg": round(df_resumen["tiempo_seg"].sum(), 2),
    }

    # Guardar resumen CONSORT en CSV
    df_resumen_completo = pl.concat([df_resumen.cast({"anio": pl.String}), pl.DataFrame([fila_total])])
    ruta_csv = silver_dir / "consort_summary.csv"
    df_resumen_completo.write_csv(ruta_csv)

    print("\n" + "=" * 80)
    print("RESUMEN GENERAL DEL FLUJO CONSORT (2019-2024)")
    print("=" * 80)
    print(df_resumen_completo.select([
        "anio", "n_inicial", "ex01_no_agrupable", "ex02_recien_nacido", 
        "ex03_obstetrico_sin_comp", "n_cohorte_dura", "pct_cohorte_dura",
        "n_cohorte_estancia", "pct_cohorte_estancia"
    ]))
    print(f"\nResumen persistido con éxito en: {ruta_csv}")


if __name__ == "__main__":
    main()
