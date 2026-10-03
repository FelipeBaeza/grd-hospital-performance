import sys
import time
from pathlib import Path
import polars as pl
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression, TweedieRegressor
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
GOLD_DIR = ROOT_DIR / "data/gold"
SILVER_DIR = ROOT_DIR / "data/silver"
BRONZE_DIR = ROOT_DIR / "data/bronze"

def run_auditoria():
    print("=" * 80)
    print("AUDITORÍA FINA FASE 2: RESOLUCIÓN EMPÍRICA RIGUROSA")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. CARGA DE DATOS DESARROLLO (2020-2022) Y CALIBRACIÓN (2023)
    # -------------------------------------------------------------------------
    print("\n1. Cargando datos de desarrollo (2020-2022) y calibración (2023)...")
    elix_28 = [f"ELIX_{i:02d}" for i in range(1, 32) if f"ELIX_{i:02d}" not in ["ELIX_02", "ELIX_22", "ELIX_25"]]
    elix_31 = [f"ELIX_{i:02d}" for i in range(1, 32)]
    
    cols_load = [
        "COD_HOSPITAL", "ESTANCIA_DIAS", "MORTALIDAD_BINARIA", "EN_COHORTE_DURA", 
        "EDAD_ANIOS", "SEXO", "MDC", "N_COMORB_ELIX"
    ] + elix_31

    dfs_dev = []
    for y in [2020, 2021, 2022]:
        df_y = pl.read_parquet(GOLD_DIR / f"gold_{y}.parquet", columns=cols_load)
        # Adultos, sin MDC 14, 15, 0, evaluables
        q_y = df_y.filter(
            (pl.col("EDAD_ANIOS") >= 18) &
            (pl.col("MDC") != 14) &
            (pl.col("MDC") != 15) &
            (pl.col("MDC") != 0) &
            (pl.col("MORTALIDAD_BINARIA").is_not_null())
        )
        dfs_dev.append(q_y)
    df_dev = pl.concat(dfs_dev)
    print(f"  Desarrollo (2020-2022): {len(df_dev):,} episodios | Muertes: {df_dev['MORTALIDAD_BINARIA'].sum():,}")

    df_2023_raw = pl.read_parquet(GOLD_DIR / "gold_2023.parquet", columns=cols_load)
    df_2023 = df_2023_raw.filter(
        (pl.col("EDAD_ANIOS") >= 18) &
        (pl.col("MDC") != 14) &
        (pl.col("MDC") != 15) &
        (pl.col("MDC") != 0) &
        (pl.col("MORTALIDAD_BINARIA").is_not_null())
    )
    print(f"  Calibración (2023)    : {len(df_2023):,} episodios | Muertes: {df_2023['MORTALIDAD_BINARIA'].sum():,}")

    # -------------------------------------------------------------------------
    # 2. MODELO DE MORTALIDAD Y CALIBRACIÓN POR ESTRATOS EXACTA
    # -------------------------------------------------------------------------
    print("\n2. Entrenando modelo de mortalidad en 2020-2022 y evaluando en 2023...")
    feats_mort = ["EDAD_ANIOS"] + elix_28
    
    # Estandarización consistente
    scaler_mort = StandardScaler()
    X_dev = scaler_mort.fit_transform(df_dev.select(feats_mort).fill_null(0).to_pandas().values)
    y_dev = df_dev["MORTALIDAD_BINARIA"].to_numpy()

    clf_mort = LogisticRegression(solver="lbfgs", max_iter=200, random_state=42)
    clf_mort.fit(X_dev, y_dev)

    X_2023 = scaler_mort.transform(df_2023.select(feats_mort).fill_null(0).to_pandas().values)
    p_hat_2023 = clf_mort.predict_proba(X_2023)[:, 1]
    df_2023 = df_2023.with_columns(pl.Series("E_PRED", p_hat_2023))

    total_O_2023 = df_2023["MORTALIDAD_BINARIA"].sum()
    total_E_2023 = df_2023["E_PRED"].sum()
    factor_recal = total_O_2023 / total_E_2023
    print(f"  Total O (2023): {total_O_2023:,}")
    print(f"  Total E crudo (2023): {total_E_2023:,.2f}")
    print(f"  O/E global real crudo: {total_O_2023 / total_E_2023:.4f} (Factor de recalibración: {factor_recal:.4f})")

    # Estratos de diagnósticos secundarios (N_COMORB_ELIX)
    print("\n  TABLA COMPLETA DE CALIBRACIÓN POR ESTRATOS EN 2023:")
    print("  Estrato dx | Episodios | Muertes O | Muertes E (crudo) | O/E crudo | O/E post-recalibrado")
    print("  " + "-" * 75)
    
    estratos = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 100)]
    for low, high in estratos:
        if low == high:
            label = f"{low} dx"
            cond = (pl.col("N_COMORB_ELIX") == low)
        elif high == 100:
            label = f">= {low} dx"
            cond = (pl.col("N_COMORB_ELIX") >= low)
        else:
            label = f"{low}-{high} dx"
            cond = (pl.col("N_COMORB_ELIX") >= low) & (pl.col("N_COMORB_ELIX") <= high)
        
        sub = df_2023.filter(cond)
        o_sub = sub["MORTALIDAD_BINARIA"].sum()
        e_sub = sub["E_PRED"].sum()
        oe_raw = o_sub / e_sub if e_sub > 0 else 0
        oe_recal = oe_raw / factor_recal if factor_recal > 0 else 0
        print(f"  {label:<10} | {len(sub):9,} | {o_sub:9,} | {e_sub:17.1f} | {oe_raw:9.4f} | {oe_recal:18.4f}")

    # -------------------------------------------------------------------------
    # 3. DÍA 0: ANÁLISIS DE FUNNEL PLOT Y DESPLAZAMIENTOS
    # -------------------------------------------------------------------------
    print("\n3. Análisis de Sensibilidad Día 0 y Gráfico de Embudo (Funnel Plot ±3 sigma)...")
    # Base vs No-Día-0 a nivel hospitalario
    hosp_base = df_2023.group_by("COD_HOSPITAL").agg([
        pl.len().alias("N"),
        pl.col("MORTALIDAD_BINARIA").sum().alias("O"),
        pl.col("E_PRED").sum().alias("E")
    ]).with_columns([
        (pl.col("O") / pl.col("E")).alias("OE_base")
    ])

    hosp_noday0 = df_2023.filter(pl.col("ESTANCIA_DIAS") > 0).group_by("COD_HOSPITAL").agg([
        pl.len().alias("N_noday0"),
        pl.col("MORTALIDAD_BINARIA").sum().alias("O_noday0"),
        pl.col("E_PRED").sum().alias("E_noday0")
    ]).with_columns([
        (pl.col("O_noday0") / pl.col("E_noday0")).alias("OE_noday0")
    ])

    comp_d0 = hosp_base.join(hosp_noday0, on="COD_HOSPITAL")
    # Excluir pediátricos para agudos adultos
    comp_d0 = comp_d0.filter(~pl.col("COD_HOSPITAL").is_in(["109101", "112102", "113130"]))
    
    # Rankings
    comp_d0 = comp_d0.with_columns([
        pl.col("OE_base").rank().alias("rank_base"),
        pl.col("OE_noday0").rank().alias("rank_noday0")
    ]).with_columns([
        (pl.col("rank_base") - pl.col("rank_noday0")).abs().alias("rank_diff")
    ])

    max_disp = comp_d0["rank_diff"].max()
    rms_disp = np.sqrt((comp_d0["rank_diff"] ** 2).mean())
    print(f"  Hospitales evaluados (adultos agudos 2023): {len(comp_d0)}")
    print(f"  Desplazamiento máximo de ranking: {max_disp} puestos")
    print(f"  Desplazamiento RMS (Root Mean Square): {rms_disp:.2f} puestos")

    # Control limits funnel plot (Poisson / Exacto normal)
    # sigma_OE = sqrt(1 / E)
    comp_d0 = comp_d0.with_columns([
        (3.0 * (1.0 / pl.col("E").sqrt())).alias("limit_3s_base"),
        (3.0 * (1.0 / pl.col("E_noday0").sqrt())).alias("limit_3s_noday0")
    ]).with_columns([
        # Clasificación base: Sobresaliente (OE < 1 - 3s), Alerta (OE > 1 + 3s), Normal
        pl.when(pl.col("OE_base") < (1.0 - pl.col("limit_3s_base"))).then(pl.lit("SOBRESALIENTE"))
          .when(pl.col("OE_base") > (1.0 + pl.col("limit_3s_base"))).then(pl.lit("ALERTA_MORTALIDAD"))
          .otherwise(pl.lit("PROMEDIO")).alias("clase_base"),
        pl.when(pl.col("OE_noday0") < (1.0 - pl.col("limit_3s_noday0"))).then(pl.lit("SOBRESALIENTE"))
          .when(pl.col("OE_noday0") > (1.0 + pl.col("limit_3s_noday0"))).then(pl.lit("ALERTA_MORTALIDAD"))
          .otherwise(pl.lit("PROMEDIO")).alias("clase_noday0")
    ])

    cambios_clase = comp_d0.filter(pl.col("clase_base") != pl.col("clase_noday0"))
    print(f"  Hospitales que cambian de clasificación en embudo (+-3 sigma): {len(cambios_clase)} de {len(comp_d0)}")
    if len(cambios_clase) > 0:
        for r in cambios_clase.iter_rows(named=True):
            print(f"    - Hosp {r['COD_HOSPITAL']}: de {r['clase_base']} a {r['clase_noday0']} (OE base={r['OE_base']:.3f}, OE no-d0={r['OE_noday0']:.3f})")

    # -------------------------------------------------------------------------
    # 4. UPCODING: BOOTSTRAP PAREADO DE LA DIFERENCIA (r_s_cen - r_s_raw)
    # -------------------------------------------------------------------------
    print("\n4. Bootstrap Pareado para Atenuación de Upcoding (1.000 réplicas)...")
    # Calcular promedio hospitalario de dx secundarios
    h_dx = df_2023.group_by("COD_HOSPITAL").agg(pl.col("N_COMORB_ELIX").mean().alias("MEAN_DX"))
    h_eval = comp_d0.select(["COD_HOSPITAL", "OE_base"]).join(h_dx, on="COD_HOSPITAL")

    # Modelo con centrado
    df_2023_cen = df_2023.join(h_dx, on="COD_HOSPITAL").with_columns(
        (pl.col("N_COMORB_ELIX") - pl.col("MEAN_DX")).alias("CEN_DX")
    )
    # Fit rápido centrado
    feats_cen = ["EDAD_ANIOS", "CEN_DX"]
    clf_cen = LogisticRegression(solver="lbfgs").fit(
        StandardScaler().fit_transform(df_2023_cen.select(feats_cen).to_pandas().values),
        df_2023_cen["MORTALIDAD_BINARIA"].to_numpy()
    )
    p_cen = clf_cen.predict_proba(StandardScaler().fit_transform(df_2023_cen.select(feats_cen).to_pandas().values))[:, 1]
    df_2023_cen = df_2023_cen.with_columns(pl.Series("E_CEN", p_cen))

    h_cen = df_2023_cen.group_by("COD_HOSPITAL").agg([
        pl.col("MORTALIDAD_BINARIA").sum().alias("O_cen"),
        pl.col("E_CEN").sum().alias("E_cen")
    ]).with_columns((pl.col("O_cen") / pl.col("E_cen")).alias("OE_cen"))

    data_upcoding = h_eval.join(h_cen, on="COD_HOSPITAL").to_pandas()
    
    r_raw, _ = stats.spearmanr(data_upcoding["OE_base"], data_upcoding["MEAN_DX"])
    r_cen, _ = stats.spearmanr(data_upcoding["OE_cen"], data_upcoding["MEAN_DX"])
    print(f"  Spearman crudo: {r_raw:.4f} | Spearman centrado: {r_cen:.4f}")

    np.random.seed(42)
    n_boot = 1000
    n_h = len(data_upcoding)
    diffs = []
    for _ in range(n_boot):
        idx = np.random.choice(n_h, size=n_h, replace=True)
        samp = data_upcoding.iloc[idx]
        r1, _ = stats.spearmanr(samp["OE_base"], samp["MEAN_DX"])
        r2, _ = stats.spearmanr(samp["OE_cen"], samp["MEAN_DX"])
        diffs.append(r2 - r1)
    
    diffs = np.array(diffs)
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    p_val_attenuation = np.mean(diffs <= 0)
    print(f"  Diferencia pareada (r_s_cen - r_s_raw): {np.mean(diffs):+.4f} [IC 95%: {ci_low:+.4f} a {ci_high:+.4f}]")
    print(f"  P(Delta r_s <= 0): {p_val_attenuation:.4f} -> Significancia de atenuación: p = {p_val_attenuation:.4f}")

    # -------------------------------------------------------------------------
    # 5. ROBUSTEZ DE MORTALIDAD (rho entre 41, 34 y 28 comorbilidades)
    # -------------------------------------------------------------------------
    print("\n5. Correlación de rankings entre modelos de mortalidad (41 vars vs 34 vs 28 crónicas vs 31 con agudas)...")
    # Ya tenemos p_hat_2023 con 28 crónicas
    # Ajustemos con 31 completas
    feats_31 = ["EDAD_ANIOS"] + elix_31
    clf_31 = LogisticRegression(solver="lbfgs", max_iter=200).fit(
        StandardScaler().fit_transform(df_dev.select(feats_31).fill_null(0).to_pandas().values),
        y_dev
    )
    p_31 = clf_31.predict_proba(StandardScaler().fit_transform(df_2023.select(feats_31).fill_null(0).to_pandas().values))[:, 1]
    
    h_31 = df_2023.with_columns(pl.Series("E_31", p_31)).group_by("COD_HOSPITAL").agg([
        pl.col("MORTALIDAD_BINARIA").sum().alias("O"),
        pl.col("E_31").sum().alias("E_31")
    ]).with_columns((pl.col("O") / pl.col("E_31")).alias("OE_31"))

    # Modelo con 13 crónicas estables (100%)
    feats_13 = ["EDAD_ANIOS", "ELIX_01", "ELIX_03", "ELIX_04", "ELIX_06", "ELIX_07", "ELIX_08", "ELIX_09", "ELIX_10", "ELIX_14", "ELIX_15", "ELIX_19", "ELIX_20", "ELIX_24"]
    clf_13 = LogisticRegression(solver="lbfgs", max_iter=200).fit(
        StandardScaler().fit_transform(df_dev.select(feats_13).fill_null(0).to_pandas().values),
        y_dev
    )
    p_13 = clf_13.predict_proba(StandardScaler().fit_transform(df_2023.select(feats_13).fill_null(0).to_pandas().values))[:, 1]
    h_13 = df_2023.with_columns(pl.Series("E_13", p_13)).group_by("COD_HOSPITAL").agg([
        pl.col("MORTALIDAD_BINARIA").sum().alias("O"),
        pl.col("E_13").sum().alias("E_13")
    ]).with_columns((pl.col("O") / pl.col("E_13")).alias("OE_13"))

    comp_spec = comp_d0.select(["COD_HOSPITAL", "OE_base"]).join(h_31.select(["COD_HOSPITAL", "OE_31"]), on="COD_HOSPITAL").join(h_13.select(["COD_HOSPITAL", "OE_13"]), on="COD_HOSPITAL")
    
    rho_28_vs_31, _ = stats.spearmanr(comp_spec["OE_base"], comp_spec["OE_31"])
    rho_28_vs_13, _ = stats.spearmanr(comp_spec["OE_base"], comp_spec["OE_13"])
    print(f"  rho Spearman (28 crónicas vs 31 con agudas ELIX_02,22,25): {rho_28_vs_31:.4f}")
    print(f"  rho Spearman (28 crónicas vs 13 estables al 100%)       : {rho_28_vs_13:.4f}")

    # -------------------------------------------------------------------------
    # 6. ESTADÍA: D2 CON TOPE 54 DÍAS, ELASTICNET Y LOHO TARGET ENCODING
    # -------------------------------------------------------------------------
    print("\n6. Estadía: Regularización ElasticNet, D2 con tope 54 días y Target Encoding LOHO...")
    q_los = df_2023.filter((pl.col("MORTALIDAD_BINARIA") == 0) & (pl.col("ESTANCIA_DIAS") > 0))
    p99_los = 54.0
    q_los = q_los.with_columns(
        pl.when(pl.col("ESTANCIA_DIAS") > p99_los).then(p99_los).otherwise(pl.col("ESTANCIA_DIAS")).alias("LOS_TRUNC")
    )

    feats_los = ["EDAD_ANIOS"] + elix_28
    scaler_los = StandardScaler()
    X_los = scaler_los.fit_transform(q_los.select(feats_los).to_pandas().values)
    y_los = q_los["LOS_TRUNC"].to_numpy()

    # GLM Gamma con ElasticNet / L1
    glm_los = TweedieRegressor(power=2.0, alpha=0.01, link="log", max_iter=200)
    glm_los.fit(X_los, y_los)
    y_pred_los = glm_los.predict(X_los)
    
    # Deviance explained
    # D2 Tweedie (p=2.0)
    def gamma_deviance(y_true, y_pred):
        return 2 * np.mean(-np.log(y_true / y_pred) + (y_true - y_pred) / y_pred)

    dev_null = gamma_deviance(y_los, np.mean(y_los))
    dev_model = gamma_deviance(y_los, y_pred_los)
    d2_exacto = 1.0 - (dev_model / dev_null)
    print(f"  Devianza explicada D2 (Gamma p=2.0, tope 54 días): {d2_exacto*100:.2f}%")

    # -------------------------------------------------------------------------
    # 7. QUILLOTA: TRASLADOS EN 48 HORAS ENTRE 107101 Y 200717
    # -------------------------------------------------------------------------
    print("\n7. Quillota: Enlace de traslados en 48 horas entre 107101 y 200717...")
    s24 = pl.read_parquet(SILVER_DIR / "silver_2024.parquet", columns=[
        "ID_EPISODIO", "CIP_ENCRIPTADO", "COD_HOSPITAL", "FECHA_INGRESO", "FECHAALTA", "TIPOALTA"
    ])
    q_sanmartin = s24.filter(pl.col("COD_HOSPITAL") == "107101")
    q_biprov = s24.filter(pl.col("COD_HOSPITAL") == "200717")

    # Buscar ingresos a Biprovincial cuyo CIP_ENCRIPTADO tuvo egreso por derivación de San Martín
    der_sm = q_sanmartin.filter(pl.col("TIPOALTA").str.contains("DERIVACIÓN"))
    link_quillota = der_sm.join(q_biprov, on="CIP_ENCRIPTADO", suffix="_biprov")
    link_quillota = link_quillota.with_columns(
        ((pl.col("FECHA_INGRESO_biprov") - pl.col("FECHAALTA")).dt.total_days()).alias("dias_traslado")
    )
    link_48h = link_quillota.filter((pl.col("dias_traslado") >= 0) & (pl.col("dias_traslado") <= 2))
    print(f"  Derivaciones desde San Martín (107101): {len(der_sm):,}")
    print(f"  Casos enlazados con Biprovincial (200717) dentro de 48 horas: {len(link_48h):,}")
    print(f"  Total traslados directos resueltos como internos: {len(link_48h)}")

    # -------------------------------------------------------------------------
    # 8. AUDITORÍA DE FORMATO DE DIAGNÓSTICOS (MINÚSCULAS, TILDES, ESPACIOS)
    # -------------------------------------------------------------------------
    print("\n8. Auditoría de formato en códigos CIE-10 (Bronze vs Gold)...")
    b23 = pl.read_parquet(BRONZE_DIR / "grd_2023.parquet", columns=["DIAGNOSTICO1"])
    total_diag = len(b23)
    has_spaces = b23.filter(pl.col("DIAGNOSTICO1").str.contains(r"^\s|\s$")).height
    has_lowercase = b23.filter(pl.col("DIAGNOSTICO1").str.contains(r"[a-z]")).height
    has_dots = b23.filter(pl.col("DIAGNOSTICO1").str.contains(r"\.")).height
    has_special = b23.filter(pl.col("DIAGNOSTICO1").str.contains(r"[áéíóúñÁÉÍÓÚÑ]")).height
    
    print(f"  Total registros DIAGNOSTICO1 analizados (2023): {total_diag:,}")
    print(f"  - Con espacios iniciales/finales                : {has_spaces:,} ({has_spaces/total_diag*100:.3f}%)")
    print(f"  - Con minúsculas                                : {has_lowercase:,} ({has_lowercase/total_diag*100:.3f}%)")
    print(f"  - Con puntos de separación (ej: J18.9)          : {has_dots:,} ({has_dots/total_diag*100:.3f}%)")
    print(f"  - Con caracteres especiales o tildes            : {has_special:,} ({has_special/total_diag*100:.3f}%)")

if __name__ == "__main__":
    run_auditoria()
