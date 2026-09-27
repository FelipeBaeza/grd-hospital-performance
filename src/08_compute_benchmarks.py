"""Paso 8: Cálculo de Indicadores de Desempeño Hospitalario (HSMR e IEMC).

Calcula y compara los índices oficiales de evaluación hospitalaria:
1. HSMR (Hospital Standardized Mortality Ratio):
   - Muertes observadas vs. Muertes esperadas ajustadas por riesgo basal (LightGBM).
   - Intervalos de confianza del 95% mediante aproximación de Byar.
2. IEMC (Índice de Estancia Media Ajustada por Casuística):
   - IEMC_ML: Estancia observada vs. Estancia esperada ajustada por ML (LightGBM Tweedie).
   - IEMC_FONASA: Estancia observada vs. Norma estándar de estancia por código GRD.
   - Demuestra el valor agregado del ajuste granular individual frente al promedio estático.
3. Cuadrante de Desempeño Clínico:
   - Clasificación bidimensional de cada hospital según Supervivencia y Eficiencia.

Guarda:
- reports/hospital_benchmarks_2024.csv
- reports/hospital_benchmarks_all_years.csv
- reports/comparison_ml_vs_fonasa.csv
"""

import sys
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import polars as pl

# Asegurar raíz en PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CATEGORICAL_FEATURES = [
    "SEXO",
    "PREVISION",
    "TIPO_INGRESO",
    "PROCEDENCIA_AGR",
    "CIE10_3C",
    "GRUPO_CLINICO",
]

NUMERICAL_FEATURES = [
    "EDAD_ANIOS",
    "MES_INGRESO",
    "DIA_SEMANA_INGRESO",
    "ES_FIN_SEMANA",
    "N_EGRESOS_12M",
    "DIAS_DESDE_EGRESO_PREVIO",
    "HISTORIA_DISPONIBLE",
]

ELIX_FEATURES = [f"ELIX_{i:02d}" for i in range(1, 32)] + ["N_COMORB_ELIX", "SCORE_VANWALRAVEN"]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + ELIX_FEATURES


def calcular_norma_grd_nacional(gold_dir: Path = Path("data/gold")) -> pl.DataFrame:
    """Calcula la estancia media de referencia por código GRD sobre el período de entrenamiento (2019-2022)."""
    print("Calculando norma estándar nacional de estancia por código GRD (2019-2022)...")
    dfs = [
        pl.read_parquet(
            gold_dir / f"gold_{y}.parquet",
            columns=["IR_29301_COD_GRD", "EN_COHORTE_ESTANCIA", "ESTANCIA_DIAS"],
        ).filter(pl.col("EN_COHORTE_ESTANCIA") & (pl.col("ESTANCIA_DIAS") >= 1))
        for y in [2019, 2020, 2021, 2022]
    ]
    train_los = pl.concat(dfs)

    norma = (
        train_los.group_by("IR_29301_COD_GRD")
        .agg([
            pl.col("ESTANCIA_DIAS").mean().alias("NORMA_ESTANCIA_GRD"),
            pl.len().alias("N_CASOS_NORMA"),
        ])
    )
    # Media global de respaldo para GRD con pocos casos
    media_global = float(train_los["ESTANCIA_DIAS"].mean())
    norma = norma.with_columns(
        pl.when(pl.col("N_CASOS_NORMA") < 10)
        .then(media_global)
        .otherwise(pl.col("NORMA_ESTANCIA_GRD"))
        .alias("NORMA_ESTANCIA_GRD")
    )
    print(f"Norma estándar calculada para {len(norma):,} códigos GRD.")
    return norma


def calcular_benchmarks_anio(
    anio: int,
    model_mort,
    model_los,
    norma_grd: pl.DataFrame,
    gold_dir: Path = Path("data/gold"),
) -> pd.DataFrame:
    """Calcula indicadores HSMR, IEMC_ML e IEMC_FONASA para todos los hospitales en un año."""
    t0 = time.time()
    ruta_gold = gold_dir / f"gold_{anio}.parquet"
    cols_load = [
        "ID_EPISODIO",
        "COD_HOSPITAL",
        "SERVICIO_SALUD",
        "IR_29301_COD_GRD",
        "EN_COHORTE_DURA",
        "EN_COHORTE_ESTANCIA",
        "MORTALIDAD_BINARIA",
        "ESTANCIA_DIAS",
    ] + ALL_FEATURES

    df_pl = pl.read_parquet(ruta_gold, columns=cols_load)

    # Unir norma estándar de GRD
    df_pl = df_pl.join(norma_grd.select(["IR_29301_COD_GRD", "NORMA_ESTANCIA_GRD"]), on="IR_29301_COD_GRD", how="left")
    df_pl = df_pl.with_columns(pl.col("NORMA_ESTANCIA_GRD").fill_null(6.8))

    # Conversión a Pandas para inferencia LightGBM
    pdf = df_pl.to_pandas()
    for c in CATEGORICAL_FEATURES:
        pdf[c] = pdf[c].astype("category")
    pdf["HISTORIA_DISPONIBLE"] = pdf["HISTORIA_DISPONIBLE"].astype(int)

    # 1. Predicción de Mortalidad Esperada
    mask_mort = pdf["EN_COHORTE_DURA"] & pdf["MORTALIDAD_BINARIA"].notna()
    pdf["P_MORTALIDAD_ESP"] = np.nan
    if mask_mort.sum() > 0:
        pdf.loc[mask_mort, "P_MORTALIDAD_ESP"] = model_mort.predict_proba(pdf.loc[mask_mort, ALL_FEATURES])[:, 1]

    # 2. Predicción de Estancia Esperada (ML)
    mask_los = pdf["EN_COHORTE_ESTANCIA"] & (pdf["ESTANCIA_DIAS"] >= 1)
    pdf["ESTANCIA_ESP_ML"] = np.nan
    if mask_los.sum() > 0:
        pred_los = model_los.predict(pdf.loc[mask_los, ALL_FEATURES])
        pdf.loc[mask_los, "ESTANCIA_ESP_ML"] = np.clip(pred_los, a_min=1.0, a_max=None)

    # 3. Agregación a nivel de Hospital
    registros = []
    for hosp, g in pdf.groupby("COD_HOSPITAL"):
        serv = g["SERVICIO_SALUD"].iloc[0]
        n_tot = len(g)

        # Mortalidad
        gm = g[g["EN_COHORTE_DURA"] & g["MORTALIDAD_BINARIA"].notna()]
        n_mort = len(gm)
        o_mort = float(gm["MORTALIDAD_BINARIA"].sum())
        e_mort = float(gm["P_MORTALIDAD_ESP"].sum())

        hsmr = (o_mort / e_mort * 100.0) if e_mort > 0 else np.nan

        # Intervalo de confianza Byar (Poisson)
        if o_mort > 0 and e_mort > 0:
            term = 1.96 / (3.0 * np.sqrt(o_mort))
            ci_low = 100.0 * (o_mort / e_mort) * ((1.0 - 1.0 / (9.0 * o_mort) - term) ** 3)
            ci_high = 100.0 * (o_mort / e_mort) * ((1.0 - 1.0 / (9.0 * o_mort) + term) ** 3)
        else:
            ci_low, ci_high = np.nan, np.nan

        # Estancia
        gl = g[g["EN_COHORTE_ESTANCIA"] & (g["ESTANCIA_DIAS"] >= 1)]
        n_los = len(gl)
        o_los_sum = float(gl["ESTANCIA_DIAS"].sum())
        e_los_ml_sum = float(gl["ESTANCIA_ESP_ML"].sum())
        e_los_fonasa_sum = float(gl["NORMA_ESTANCIA_GRD"].sum())

        iemc_ml = (o_los_sum / e_los_ml_sum) if e_los_ml_sum > 0 else np.nan
        iemc_fonasa = (o_los_sum / e_los_fonasa_sum) if e_los_fonasa_sum > 0 else np.nan

        los_obs_mean = gl["ESTANCIA_DIAS"].mean() if n_los > 0 else np.nan
        los_esp_ml_mean = gl["ESTANCIA_ESP_ML"].mean() if n_los > 0 else np.nan
        los_esp_fonasa_mean = gl["NORMA_ESTANCIA_GRD"].mean() if n_los > 0 else np.nan

        # Clasificación de desempeño
        if pd.isna(hsmr) or pd.isna(iemc_ml):
            categoria = "DATOS_INSUFICIENTES"
        elif hsmr < 90.0 and iemc_ml < 0.95:
            categoria = "SOBRESALIENTE"
        elif iemc_ml < 0.95 and 90.0 <= hsmr <= 110.0:
            categoria = "ALTA_EFICIENCIA"
        elif hsmr < 90.0 and 0.95 <= iemc_ml <= 1.05:
            categoria = "ALTA_SUPERVIVENCIA"
        elif 90.0 <= hsmr <= 110.0 and 0.95 <= iemc_ml <= 1.05:
            categoria = "PROMEDIO_ESPERADO"
        elif hsmr > 110.0 and iemc_ml > 1.05:
            categoria = "ALERTA_CRITICA"
        elif hsmr > 110.0:
            categoria = "ALERTA_MORTALIDAD"
        elif iemc_ml > 1.05:
            categoria = "ALERTA_ESTANCIA"
        else:
            categoria = "COMPLEJO_MIXTO"

        registros.append({
            "anio": anio,
            "COD_HOSPITAL": hosp,
            "SERVICIO_SALUD": serv,
            "n_episodios_totales": n_tot,
            "n_episodios_mortalidad": n_mort,
            "muertes_observadas": int(o_mort),
            "muertes_esperadas": round(e_mort, 1),
            "HSMR": round(hsmr, 1),
            "HSMR_IC95_INF": round(ci_low, 1),
            "HSMR_IC95_SUP": round(ci_high, 1),
            "n_episodios_estancia": n_los,
            "estancia_dias_observada_media": round(los_obs_mean, 2) if pd.notna(los_obs_mean) else np.nan,
            "estancia_dias_esperada_ml_media": round(los_esp_ml_mean, 2) if pd.notna(los_esp_ml_mean) else np.nan,
            "estancia_dias_esperada_fonasa_media": round(los_esp_fonasa_mean, 2) if pd.notna(los_esp_fonasa_mean) else np.nan,
            "IEMC_ML": round(iemc_ml, 3),
            "IEMC_FONASA": round(iemc_fonasa, 3),
            "diferencia_IEMC_ML_FONASA": round(iemc_ml - iemc_fonasa, 3) if pd.notna(iemc_ml) and pd.notna(iemc_fonasa) else np.nan,
            "clasificacion_desempeno": categoria,
        })

    bdf = pd.DataFrame(registros).sort_values("n_episodios_totales", ascending=False)
    print(f"[{anio}] Calculados benchmarks para {len(bdf)} hospitales en {time.time()-t0:.2f}s.")
    return bdf


def main():
    print("=" * 80)
    print("EJECUTANDO PASO 8: CÁLCULO DE BENCHMARKS HOSPITALARIOS (HSMR E IEMC)")
    print("=" * 80)

    models_dir = Path("models")
    reports_dir = Path("reports")
    gold_dir = Path("data/gold")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar modelos entrenados
    print("Cargando modelos ML entrenados...")
    m_mort = joblib.load(models_dir / "lgb_mortality.joblib")
    m_los = joblib.load(models_dir / "lgb_los.joblib")

    # 2. Calcular norma estándar FONASA de GRD
    norma_grd = calcular_norma_grd_nacional(gold_dir)

    # 3. Calcular para todos los años
    anios = [2019, 2020, 2021, 2022, 2023, 2024]
    dfs_anios = []

    for anio in anios:
        b = calcular_benchmarks_anio(anio, m_mort, m_los, norma_grd, gold_dir)
        dfs_anios.append(b)

    df_todos = pd.concat(dfs_anios, ignore_index=True)
    df_2024 = dfs_anios[-1]  # 2024 test period

    # Guardar benchmarks en CSV
    ruta_all = reports_dir / "hospital_benchmarks_all_years.csv"
    ruta_2024 = reports_dir / "hospital_benchmarks_2024.csv"
    df_todos.to_csv(ruta_all, index=False)
    df_2024.to_csv(ruta_2024, index=False)

    # 4. Tabla de comparación ML vs FONASA para el período 2024
    comp = df_2024[df_2024["n_episodios_totales"] >= 1000][[
        "COD_HOSPITAL",
        "SERVICIO_SALUD",
        "n_episodios_totales",
        "HSMR",
        "IEMC_ML",
        "IEMC_FONASA",
        "diferencia_IEMC_ML_FONASA",
        "clasificacion_desempeno",
    ]].sort_values("n_episodios_totales", ascending=False)

    ruta_comp = reports_dir / "comparison_ml_vs_fonasa.csv"
    comp.to_csv(ruta_comp, index=False)

    print("\n" + "=" * 80)
    print("RESUMEN DE BENCHMARKS HOSPITALARIOS 2024 (TOP 10 HOSPITALES POR CASUÍSTICA)")
    print("=" * 80)
    print(comp.head(10).to_string(index=False))

    print("\nDistribución de Categorías de Desempeño 2024:")
    print(df_2024["clasificacion_desempeno"].value_counts())

    print(f"\nResultados persistidos exitosamente en:")
    print(f" - {ruta_2024}")
    print(f" - {ruta_all}")
    print(f" - {ruta_comp}")


if __name__ == "__main__":
    main()
