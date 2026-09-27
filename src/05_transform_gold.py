"""Paso 5: Construcción de la Capa Gold (Matriz de Features Basales y Anti-Leakage).

Construye la matriz analítica de modelamiento clínico con estricta garantía de NO FUGA:
- Solo features conocidas al momento del ingreso hospitalario.
- Exclusión total de outcomes clínicos y variables post-ingreso (procedimientos, alta, etc.).
- Historial del paciente causal y retrospectivo (N_EGRESOS_12M, DIAS_DESDE_EGRESO_PREVIO).
- Agrupación clínica de admisión (CIE10_3C, GRUPO_CLINICO por capítulo CIE-10).
- Agrupación de procedencia (PROCEDENCIA_AGR).
- 31 comorbilidades basales Elixhauser y Score Van Walraven.

Guarda:
- data/gold/gold_{anio}.parquet
- reports/gold_features_summary.csv
- data/gold/gold_metadata.json
"""

import sys
from pathlib import Path
import json
import time
import polars as pl

# Asegurar raíz en PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Columnas estrictamente prohibidas como features por fuga de datos o circularidad
COLUMNAS_PROHIBIDAS_FUGA = {
    "FECHAALTA", "TIPOALTA", "SERVICIOALTA",
    "DIAS_ESTADA", "DIAS_ESTADA_CRUDA", "ESTANCIA_DIAS",
    "MORTALIDAD_BINARIA", "ESTADO_CENSURA",
    "FECHAPROCEDIMIENTO1", "PROCEDIMIENTO1",
    "RN1ESTADO", "RN2ESTADO", "RN3ESTADO", "RN4ESTADO",
    "IR_29301_PESO", "IR_29301_COD_GRD", "MDC", "GRD_BASE",
    "EN_COHORTE_DURA", "EN_COHORTE_ESTANCIA", "MOTIVO_EXCLUSION_DURA",
}

# 31 Categorías canónicas de Elixhauser
CATEGORIAS_ELIX = [f"ELIX_{i:02d}" for i in range(1, 32)]

# Lista oficial de columnas predictoras basales para modelos ML
FEATURES_BASALES = [
    # Demográficas
    "EDAD_ANIOS",
    "SEXO",
    "PREVISION",
    # Admisión y Temporales
    "ANIO_INGRESO",
    "MES_INGRESO",
    "DIA_SEMANA_INGRESO",
    "ES_FIN_SEMANA",
    "TIPO_INGRESO",
    "PROCEDENCIA_AGR",
    # Clínicas de Admisión
    "CIE10_3C",
    "GRUPO_CLINICO",
    # Historial Previo del Paciente (Causal)
    "N_EGRESOS_12M",
    "DIAS_DESDE_EGRESO_PREVIO",
    "HISTORIA_DISPONIBLE",
    # Comorbilidades Basales (Quan 2005)
    *CATEGORIAS_ELIX,
    "N_COMORB_ELIX",
    "SCORE_VANWALRAVEN",
]


def precalcular_historial_pacientes(silver_dir: Path = Path("data/silver")) -> pl.DataFrame:
    """Calcula las variables de historial retrospectivo en ventana de 365 días a través de todos los años."""
    print("Precalculando variables de historial causal de pacientes (2019-2024)...")
    t0 = time.time()
    
    files = sorted(silver_dir.glob("silver_filtered_*.parquet"))
    dfs = [
        pl.read_parquet(f, columns=["ID_EPISODIO", "CIP_ENCRIPTADO", "FECHA_INGRESO", "FECHAALTA"])
        for f in files
    ]
    all_df = pl.concat(dfs)

    # Tabla de eventos de egreso por paciente y fecha
    eventos = (
        all_df.filter(pl.col("CIP_ENCRIPTADO").is_not_null() & pl.col("FECHAALTA").is_not_null())
        .group_by(["CIP_ENCRIPTADO", "FECHAALTA"])
        .agg(pl.len().alias("ALTAS_EN_FECHA"))
        .sort(["CIP_ENCRIPTADO", "FECHAALTA"])
        .with_columns(
            pl.col("ALTAS_EN_FECHA").cum_sum().over("CIP_ENCRIPTADO").cast(pl.Int32).alias("C_ALTA")
        )
    )

    # Tabla de consultas (fechas de ingreso de cada episodio)
    consultas = (
        all_df.filter(pl.col("CIP_ENCRIPTADO").is_not_null() & pl.col("FECHA_INGRESO").is_not_null())
        .select([
            "ID_EPISODIO",
            "CIP_ENCRIPTADO",
            pl.col("FECHA_INGRESO").alias("T_INGRESO"),
            (pl.col("FECHA_INGRESO") - pl.duration(days=364)).alias("CORTE_365D"),
        ])
        .sort(["CIP_ENCRIPTADO", "T_INGRESO"])
    )

    # Join asof para C(t) y última alta
    asof_t = consultas.join_asof(
        eventos.select(["CIP_ENCRIPTADO", pl.col("FECHAALTA").alias("ULTIMA_ALTA"), pl.col("C_ALTA").alias("C_T")]),
        left_on="T_INGRESO",
        right_on="ULTIMA_ALTA",
        by="CIP_ENCRIPTADO",
        strategy="backward",
        allow_exact_matches=False,
    )

    # Join asof para C(t - 365d)
    asof_365 = asof_t.join_asof(
        eventos.select(["CIP_ENCRIPTADO", pl.col("FECHAALTA").alias("FECHA_CORTE"), pl.col("C_ALTA").alias("C_365")]),
        left_on="CORTE_365D",
        right_on="FECHA_CORTE",
        by="CIP_ENCRIPTADO",
        strategy="backward",
        allow_exact_matches=False,
    )

    # Variables finales de historial
    history_df = asof_365.select([
        "ID_EPISODIO",
        (pl.col("C_T").fill_null(0) - pl.col("C_365").fill_null(0)).cast(pl.Int32).alias("N_EGRESOS_12M"),
        (pl.col("T_INGRESO") - pl.col("ULTIMA_ALTA")).dt.total_days().cast(pl.Int32).alias("DIAS_DESDE_EGRESO_PREVIO"),
        (pl.col("T_INGRESO") >= pl.date(2020, 1, 1)).alias("HISTORIA_DISPONIBLE"),
    ])

    print(f"Historial causal calculado para {len(history_df):,} episodios en {time.time()-t0:.2f}s.")
    return history_df


def cargar_mapeadores_clinicos():
    """Carga mapeadores de procedencia y grupos clínicos desde config/."""
    # 1. Procedencia
    df_proc = pl.read_csv("config/agrupacion_procedencia.csv", comment_prefix="#")
    map_proc = df_proc.filter(pl.col("tipo_regla") == "VALOR").select([
        pl.col("valor_origen").str.strip_chars().str.to_uppercase().alias("TIPO_PROCEDENCIA_KEY"),
        pl.col("categoria_destino").alias("PROCEDENCIA_AGR"),
    ])

    # 2. Grupo Clínico (Capítulos CIE-10)
    df_gc = pl.read_csv("config/agrupacion_grupo_clinico.csv", comment_prefix="#")
    ranges_gc = [
        (row["valor_origen"], row["valor_origen_hasta"], row["categoria_destino"])
        for row in df_gc.filter(pl.col("tipo_regla") == "RANGO").iter_rows(named=True)
    ]
    return map_proc, ranges_gc


def procesar_gold_anio(
    anio: int,
    history_df: pl.DataFrame,
    map_proc: pl.DataFrame,
    ranges_gc: list,
    silver_dir: Path = Path("data/silver"),
    gold_dir: Path = Path("data/gold"),
) -> dict:
    """Transforma un archivo silver_filtered en el dataset analítico Gold."""
    ruta_silver = silver_dir / f"silver_filtered_{anio}.parquet"
    if not ruta_silver.exists():
        raise FileNotFoundError(f"No existe {ruta_silver}")

    t0 = time.time()
    df = pl.read_parquet(ruta_silver)

    # 1. Unir historial previo de paciente
    df = df.join(history_df, on="ID_EPISODIO", how="left")
    # Para los episodios sin CIP (2.044 casos en todo el período), N_EGRESOS_12M queda 0 e HISTORIA_DISPONIBLE False
    df = df.with_columns([
        pl.col("N_EGRESOS_12M").fill_null(0),
        pl.col("HISTORIA_DISPONIBLE").fill_null(False),
    ])

    # 2. Derivar Edad
    edad_expr = (
        (pl.col("FECHA_INGRESO") - pl.col("FECHA_NACIMIENTO")).dt.total_days() / 365.25
    ).floor().clip(0, 110).cast(pl.Int32)

    # 3. Derivar Temporales
    mes_expr = pl.col("FECHA_INGRESO").dt.month().cast(pl.Int32)
    dia_sem_expr = pl.col("FECHA_INGRESO").dt.weekday().cast(pl.Int32)
    es_fin_sem_expr = (pl.col("FECHA_INGRESO").dt.weekday() >= 6).cast(pl.Int32)
    anio_expr = pl.col("FECHA_INGRESO").dt.year().cast(pl.Int32)

    # 4. Derivar Procedencia Agrupada
    df = df.with_columns(
        pl.col("TIPO_PROCEDENCIA").str.strip_chars().str.to_uppercase().alias("TIPO_PROCEDENCIA_KEY")
    ).join(map_proc, on="TIPO_PROCEDENCIA_KEY", how="left")
    df = df.with_columns(pl.col("PROCEDENCIA_AGR").fill_null("NO_CLASIFICADO")).drop("TIPO_PROCEDENCIA_KEY")

    # 5. Derivar CIE10_3C y GRUPO_CLINICO
    c3 = pl.col("DIAGNOSTICO1").str.strip_chars().str.to_uppercase().str.slice(0, 3)
    gc_expr = pl.when(c3 == "DES").then(pl.lit("DESCONOCIDO"))
    for d_from, d_to, cat_dest in ranges_gc:
        gc_expr = gc_expr.when((c3 >= d_from) & (c3 <= d_to)).then(pl.lit(cat_dest))
    gc_expr = gc_expr.otherwise(pl.lit("NO_CLASIFICADO"))

    # Aplicar transformaciones
    df_gold = df.with_columns([
        edad_expr.alias("EDAD_ANIOS"),
        mes_expr.alias("MES_INGRESO"),
        dia_sem_expr.alias("DIA_SEMANA_INGRESO"),
        es_fin_sem_expr.alias("ES_FIN_SEMANA"),
        anio_expr.alias("ANIO_INGRESO"),
        c3.alias("CIE10_3C"),
        gc_expr.alias("GRUPO_CLINICO"),
    ])

    # Convertir binarias Elixhauser a Int8 (para optimizar memoria y compatibilidad con LightGBM)
    elix_cast = [pl.col(cat).cast(pl.Int8) for cat in CATEGORIAS_ELIX]
    df_gold = df_gold.with_columns(elix_cast)

    # 6. Selección de Columnas Gold (Metadatos + Features + Targets + Agrupadores)
    columnas_metadatos = [
        "ID_EPISODIO",
        "CIP_ENCRIPTADO",
        "COD_HOSPITAL",
        "SERVICIO_SALUD",
        "IR_29301_COD_GRD",
        "MDC",
        "GRD_BASE",
        "EN_COHORTE_DURA",
        "EN_COHORTE_ESTANCIA",
        "MOTIVO_EXCLUSION_DURA",
    ]
    columnas_targets = [
        "MORTALIDAD_BINARIA",
        "ESTANCIA_DIAS",
        "ESTADO_CENSURA",
    ]

    columnas_finales = columnas_metadatos + columnas_targets + FEATURES_BASALES

    # Comprobación de Anti-Leakage
    for feat in FEATURES_BASALES:
        if feat in COLUMNAS_PROHIBIDAS_FUGA:
            raise ValueError(f"VIOLACIÓN DE ANTI-LEAKAGE: la columna '{feat}' está en COLUMNAS_PROHIBIDAS_FUGA.")

    df_gold_final = df_gold.select(columnas_finales)

    # Guardar en data/gold/gold_{anio}.parquet
    gold_dir.mkdir(parents=True, exist_ok=True)
    ruta_salida = gold_dir / f"gold_{anio}.parquet"
    df_gold_final.write_parquet(ruta_salida, compression="zstd")

    tiempo_total = time.time() - t0
    print(
        f"[{anio}] Gold generado exitosamente: {len(df_gold_final):,} filas, "
        f"{len(df_gold_final.columns)} columnas en {tiempo_total:.2f}s -> {ruta_salida}"
    )

    return {
        "anio": anio,
        "n_filas": len(df_gold_final),
        "n_features": len(FEATURES_BASALES),
        "tiempo_seg": round(tiempo_total, 2),
    }


def main():
    print("=" * 80)
    print("EJECUTANDO PASO 5: CONSTRUCCIÓN DE LA CAPA GOLD (2019-2024)")
    print("=" * 80)

    silver_dir = Path("data/silver")
    gold_dir = Path("data/gold")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Precalcular historial temporal (cross-year causal)
    history_df = precalcular_historial_pacientes(silver_dir)

    # 2. Cargar mapeadores clínicos
    map_proc, ranges_gc = cargar_mapeadores_clinicos()

    # 3. Procesar año por año
    anios = [2019, 2020, 2021, 2022, 2023, 2024]
    metricas = []

    for anio in anios:
        m = procesar_gold_anio(anio, history_df, map_proc, ranges_gc, silver_dir, gold_dir)
        metricas.append(m)

    df_metricas = pl.DataFrame(metricas)
    df_metricas.write_csv(reports_dir / "gold_features_summary.csv")

    # 4. Generar Diccionario / Metadatos de la Capa Gold
    # Leer muestra de 2022 para registrar tipos
    sample = pl.read_parquet(gold_dir / "gold_2022.parquet", n_rows=100)
    metadata = {
        "descripcion": "Capa Gold analítica para entrenamiento de modelos de desempeño hospitalario",
        "n_anios": len(anios),
        "total_registros": int(df_metricas["n_filas"].sum()),
        "features_basales_count": len(FEATURES_BASALES),
        "features_basales": FEATURES_BASALES,
        "anti_leakage_garantia": "Estricta. Solo features conocidas al ingreso.",
        "esquema_tipos": {col: str(sample.schema[col]) for col in sample.columns},
    }

    with open(gold_dir / "gold_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with open(reports_dir / "gold_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("RESUMEN DE LA CAPA GOLD GENERADA (2019-2024)")
    print("=" * 80)
    print(df_metricas)
    print(f"Total registros Gold: {df_metricas['n_filas'].sum():,}")
    print(f"Total Features Basales: {len(FEATURES_BASALES)}")
    print(f"Metadatos guardados en: {gold_dir / 'gold_metadata.json'}")


if __name__ == "__main__":
    main()
