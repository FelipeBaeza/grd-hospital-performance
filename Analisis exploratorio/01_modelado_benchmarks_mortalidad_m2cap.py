"""
01_modelado_benchmarks_mortalidad_m2cap.py
=========================================
Propósito:
  Entrenamiento, recalibración y benchmarking empírico de modelos de mortalidad intrahospitalaria:
  1. M2+CAP (Especificación Primaria Oficial): Demografía + Vía Ingreso + 28 Elixhauser Crónicos + Capítulos CIE-10.
  2. M2 Base (Sensibilidad 1): Demografía + Vía Ingreso + 28 Elixhauser Crónicos (sin diagnóstico principal).
  3. M3-Total (Sensibilidad 2): M2 Base + Splines cúbicos de R_dx total al alta.

Genera el archivo canonical unificado de la tesis:
  Analisis exploratorio/benchmarks_modelos_fase2_2023.csv
"""

from pathlib import Path
import numpy as np
import polars as pl
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

DIR_BASE = Path("/home/felipe/Documentos/Proyecto final/Tesis")
CACHE_DEV = DIR_BASE / "scratch/dev_eval_cache.parquet"
CACHE_23 = DIR_BASE / "scratch/eval_23_cache.parquet"
CAT_PATH = DIR_BASE / "config/catalogo_hospitales_procedencia.csv"
OUT_CSV = DIR_BASE / "Analisis exploratorio/benchmarks_modelos_fase2_2023.csv"
OUT_SCRATCH = DIR_BASE / "scratch/benchmarks_modelos_fase2_2023.csv"


def ejecutar_benchmarks():
    print("1. Cargando datos...")
    df_dev = pl.read_parquet(CACHE_DEV)
    df_23 = pl.read_parquet(CACHE_23)
    cat = pl.read_csv(CAT_PATH).with_columns(pl.col("cod_hospital").cast(pl.String))

    elix_28 = [
        f"ELIX_{i:02d}"
        for i in range(1, 32)
        if f"ELIX_{i:02d}" not in ["ELIX_02", "ELIX_22", "ELIX_25"]
    ]
    f_base = [
        "EDAD_ANIOS",
        "INGRESO_URGENCIA",
        "INGRESO_CRITICO",
        "DERIVADO_OTRO_HOSPITAL",
        "SEXO_MASCULINO",
    ]

    # OHE de Capítulos CIE-10 del diagnóstico principal
    mdcs_dev = df_dev["MDC"].to_pandas().values.reshape(-1, 1)
    mdcs_23 = df_23["MDC"].to_pandas().values.reshape(-1, 1)
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    X_mdc_dev = ohe.fit_transform(mdcs_dev)
    X_mdc_23 = ohe.transform(mdcs_23)

    # Scaler base
    sc = StandardScaler()
    X_dev_base = sc.fit_transform(
        df_dev.select(f_base + elix_28).fill_null(0).to_pandas().values
    )
    X_23_base = sc.transform(
        df_23.select(f_base + elix_28).fill_null(0).to_pandas().values
    )

    X_dev_m2cap = np.hstack([X_dev_base, X_mdc_dev])
    X_23_m2cap = np.hstack([X_23_base, X_mdc_23])

    # R_dx normalizado contra línea base histórica de desarrollo
    h_dev_stats = df_dev.group_by("COD_HOSPITAL").agg(
        pl.col("N_DX").mean().alias("mean_ndx_tot")
    )
    g_mean_tot = df_dev["N_DX"].mean()
    df_dev_m3 = df_dev.join(h_dev_stats, on="COD_HOSPITAL", how="left").with_columns(
        (pl.col("N_DX") / pl.col("mean_ndx_tot").fill_null(g_mean_tot)).alias(
            "R_DX_TOT"
        )
    )
    df_23_m3 = df_23.join(h_dev_stats, on="COD_HOSPITAL", how="left").with_columns(
        (pl.col("N_DX") / pl.col("mean_ndx_tot").fill_null(g_mean_tot)).alias(
            "R_DX_TOT"
        )
    )

    sp_tot = SplineTransformer(n_knots=4, degree=3, include_bias=False)
    sp_dev_tot = sp_tot.fit_transform(df_dev_m3.select(["R_DX_TOT"]).to_pandas().values)
    sp_23_tot = sp_tot.transform(df_23_m3.select(["R_DX_TOT"]).to_pandas().values)

    X_dev_m3 = np.hstack([X_dev_base, sp_dev_tot])
    X_23_m3 = np.hstack([X_23_base, sp_23_tot])

    y_dev = df_dev["MORTALIDAD_BINARIA"].to_numpy()
    o_tot = df_23["MORTALIDAD_BINARIA"].sum()

    print("2. Ajustando modelos M2+CAP, M2 Base y M3-Total...")
    m2_cap = LogisticRegression(max_iter=200, random_state=42).fit(
        X_dev_m2cap, y_dev
    )
    m2_base = LogisticRegression(max_iter=200, random_state=42).fit(
        X_dev_base, y_dev
    )
    m3_tot = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_m3, y_dev)

    p_m2cap = m2_cap.predict_proba(X_23_m2cap)[:, 1]
    p_m2base = m2_base.predict_proba(X_23_base)[:, 1]
    p_m3tot = m3_tot.predict_proba(X_23_m3)[:, 1]

    df_23_scored = df_23.with_columns(
        [
            pl.Series("E_M2CAP", p_m2cap * (o_tot / p_m2cap.sum())),
            pl.Series("E_M2", p_m2base * (o_tot / p_m2base.sum())),
            pl.Series("E_M3", p_m3tot * (o_tot / p_m3tot.sum())),
        ]
    )

    # Agregación por hospital
    h_res = (
        df_23_scored.group_by("COD_HOSPITAL")
        .agg(
            [
                pl.len().alias("N"),
                pl.col("MORTALIDAD_BINARIA").sum().alias("O"),
                pl.col("E_M2CAP").sum().alias("E_m2cap"),
                pl.col("E_M2").sum().alias("E_m2"),
                pl.col("E_M3").sum().alias("E_m3"),
                pl.col("N_DX").mean().alias("mean_ndx"),
                (pl.col("N_DX") >= 11).mean().alias("pct_ndx_ge11"),
            ]
        )
        .with_columns(
            [
                (pl.col("O") / pl.col("E_m2cap")).alias("OE_m2cap"),
                (pl.col("O") / pl.col("E_m2")).alias("OE_m2"),
                (pl.col("O") / pl.col("E_m3")).alias("OE_m3"),
            ]
        )
        .join(
            cat.select(["cod_hospital", "nombre_oficial", "es_monografico"]),
            left_on="COD_HOSPITAL",
            right_on="cod_hospital",
            how="left",
        )
    )

    # Inferencia Empírica Bayesiana con DerSimonian-Laird
    c = np.log(1.10)

    def calc_p(o_series, e_series, tau, mu):
        p_vals = []
        for o, e in zip(o_series, e_series):
            if o == 0 or e == 0:
                p_vals.append(0.0)
                continue
            y = np.log(o / e)
            v = 1.0 / o
            B = (tau**2) / (tau**2 + v)
            theta = B * y + (1 - B) * mu
            se = np.sqrt(B * v)
            z = (theta - c) / se
            p_vals.append(float(stats.norm.cdf(z)))
        return p_vals

    h_res = h_res.with_columns(
        [
            pl.Series(
                "P_m2cap", calc_p(h_res["O"], h_res["E_m2cap"], 0.1948, -0.0359)
            ),
            pl.Series("P_m2", calc_p(h_res["O"], h_res["E_m2"], 0.2077, -0.0267)),
            pl.Series("P_m3", calc_p(h_res["O"], h_res["E_m3"], 0.2041, -0.0347)),
            # Columnas de compatibilidad hacia atrás
            pl.col("E_m2cap").alias("E_m2mdc"),
            pl.col("OE_m2cap").alias("OE_m2mdc"),
            pl.col("P_m2cap").alias("P_m2mdc"),
        ]
    ).sort("COD_HOSPITAL")

    h_res.write_csv(OUT_CSV)
    h_res.write_csv(OUT_SCRATCH)
    print(f"3. Archivo canónico guardado en {OUT_CSV} (K = {len(h_res)})")


if __name__ == "__main__":
    ejecutar_benchmarks()
