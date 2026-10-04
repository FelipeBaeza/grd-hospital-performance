"""
04_analisis_traslados_y_estres_censura.py
=========================================
Propósito:
  1. Enlace determinístico unificado a 30 días de egresos por traslado inpatient adultos 2023.
  2. Cálculo de cotas de salida (lambda_salida) para las 10 Alertas Primarias bajo M2+CAP.
  3. Matriz de estrés de sobremortalidad en derivados (Byar CI) y puntos de quiebre (lambda*)
     para los 52 hospitales no alertados de agudos generales.
"""

from pathlib import Path
import numpy as np
import polars as pl
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DIR_BASE = Path("/home/felipe/Documentos/Proyecto final/Tesis")
SILVER_DIR = DIR_BASE / "data/silver"
GOLD_DIR = DIR_BASE / "data/gold"
CAT_PATH = DIR_BASE / "config/catalogo_hospitales_procedencia.csv"
CACHE_DEV = DIR_BASE / "scratch/dev_eval_cache.parquet"
CACHE_23 = DIR_BASE / "scratch/eval_23_cache.parquet"


def ejecutar_analisis_traslados():
    print("1. Cargando datos de hospitalizaciones y traslados 2023...")
    s23 = pl.read_parquet(SILVER_DIR / "silver_2023.parquet")
    g23 = pl.read_parquet(GOLD_DIR / "gold_2023.parquet")
    cat = pl.read_csv(CAT_PATH).with_columns(
        pl.col("cod_hospital").cast(pl.String)
    )

    c_ex01 = (
        (g23["IR_29301_COD_GRD"].is_null())
        | (g23["MDC"] == 0)
        | (g23["IR_29301_COD_GRD"].str.starts_with("99"))
    )
    c_ex02 = (~c_ex01) & (g23["MDC"] == 15)
    c_ex03 = (~c_ex01) & (~c_ex02) & (g23["MDC"] == 14)
    c_ex04 = (
        (~c_ex01)
        & (~c_ex02)
        & (~c_ex03)
        & ((g23["EDAD_ANIOS"] < 18) | (g23["EDAD_ANIOS"].is_null()))
    )
    c_ex05 = (
        (~c_ex01)
        & (~c_ex02)
        & (~c_ex03)
        & (~c_ex04)
        & (s23["TIPO_ACTIVIDAD"] != "HOSPITALIZACIÓN")
    )
    inp_mask = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (~c_ex05)

    is_transfer = s23["TIPOALTA"].str.contains(
        "DERIVACIÓN OTRO HOSPITAL|DERIVACIÓN INST. PRIVADA"
    )
    transf_df = s23.filter(inp_mask & is_transfer).select(
        ["ID_EPISODIO", "COD_HOSPITAL", "CIP_ENCRIPTADO", "FECHAALTA"]
    )

    print(
        f"  Total traslados elegibles 2023: {len(transf_df):,} episodios en {transf_df['COD_HOSPITAL'].n_unique()} hospitales"
    )

    df_dev = pl.read_parquet(CACHE_DEV)
    df_23 = pl.read_parquet(CACHE_23)

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

    mdcs_dev = df_dev["MDC"].to_pandas().values.reshape(-1, 1)
    mdcs_23 = df_23["MDC"].to_pandas().values.reshape(-1, 1)
    ohe = OneHotEncoder(
        drop="first", sparse_output=False, handle_unknown="ignore"
    )
    X_mdc_dev = ohe.fit_transform(mdcs_dev)
    X_mdc_23 = ohe.transform(mdcs_23)

    sc = StandardScaler()
    X_dev_base = sc.fit_transform(
        df_dev.select(f_base + elix_28).fill_null(0).to_pandas().values
    )
    X_23_base = sc.transform(
        df_23.select(f_base + elix_28).fill_null(0).to_pandas().values
    )

    X_dev_m2cap = np.hstack([X_dev_base, X_mdc_dev])
    X_23_m2cap = np.hstack([X_23_base, X_mdc_23])

    y_dev = df_dev["MORTALIDAD_BINARIA"].to_numpy()
    m2_cap = LogisticRegression(max_iter=200, random_state=42).fit(
        X_dev_m2cap, y_dev
    )

    o_tot = df_23["MORTALIDAD_BINARIA"].sum()
    p_23 = m2_cap.predict_proba(X_23_m2cap)[:, 1]
    factor = o_tot / p_23.sum()
    df_23 = df_23.with_columns(pl.Series("E_M2CAP", p_23 * factor))

    # Predecir riesgo esperado en los traslados
    s_tr = s23.filter(inp_mask & is_transfer).with_columns(
        [
            s23.filter(inp_mask & is_transfer)["TIPO_INGRESO"]
            .str.to_uppercase()
            .str.contains("URGEN")
            .cast(pl.Int32)
            .alias("INGRESO_URGENCIA"),
            s23.filter(inp_mask & is_transfer)["SERVICIOINGRESO"]
            .str.to_uppercase()
            .str.contains("UCI|UTI|CRITIC|INTENS|INTERMED")
            .cast(pl.Int32)
            .alias("INGRESO_CRITICO"),
            (
                s23.filter(inp_mask & is_transfer)["HOSPPROCEDENCIA"].is_not_null()
                | s23.filter(inp_mask & is_transfer)["TIPO_PROCEDENCIA"]
                .str.to_uppercase()
                .str.contains("DERIV|OTRO|HOSP")
            )
            .cast(pl.Int32)
            .alias("DERIVADO_OTRO_HOSPITAL"),
            (g23.filter(inp_mask & is_transfer)["SEXO"] == 1)
            .cast(pl.Int32)
            .alias("SEXO_MASCULINO"),
            g23.filter(inp_mask & is_transfer)["EDAD_ANIOS"].alias(
                "EDAD_ANIOS"
            ),
            g23.filter(inp_mask & is_transfer)["MDC"].alias("MDC"),
        ]
    )
    for col in elix_28:
        s_tr = s_tr.with_columns(g23.filter(inp_mask & is_transfer)[col].alias(col))

    X_tr_base = sc.transform(
        s_tr.select(f_base + elix_28).fill_null(0).to_pandas().values
    )
    mdcs_tr = s_tr["MDC"].to_pandas().values.reshape(-1, 1)
    X_mdc_tr = ohe.transform(mdcs_tr)
    X_tr_m2cap = np.hstack([X_tr_base, X_mdc_tr])
    p_tr = m2_cap.predict_proba(X_tr_m2cap)[:, 1] * factor
    transf_df = transf_df.with_columns(pl.Series("P_expected", p_tr))

    # 2. Enlace determinístico en ventana <= 30 días
    all_episodes = s23.select(
        [
            "ID_EPISODIO",
            "COD_HOSPITAL",
            "CIP_ENCRIPTADO",
            "FECHA_INGRESO",
            "FECHAALTA",
            "TIPOALTA",
        ]
    ).with_columns(
        (pl.col("TIPOALTA") == "FALLECIDO")
        .cast(pl.Int32)
        .alias("FALLECIDO_RECEPTOR")
    )

    linked = (
        transf_df.filter(
            pl.col("CIP_ENCRIPTADO").is_not_null()
            & (pl.col("CIP_ENCRIPTADO") != "")
        )
        .join(
            all_episodes.filter(
                pl.col("CIP_ENCRIPTADO").is_not_null()
                & (pl.col("CIP_ENCRIPTADO") != "")
            ),
            on="CIP_ENCRIPTADO",
            suffix="_rec",
        )
        .filter(
            (pl.col("ID_EPISODIO") != pl.col("ID_EPISODIO_rec"))
            & (pl.col("FECHA_INGRESO") >= pl.col("FECHAALTA"))
            & (
                pl.col("FECHA_INGRESO")
                <= pl.col("FECHAALTA") + pl.duration(days=30)
            )
        )
        .sort(["ID_EPISODIO", "FECHA_INGRESO"])
        .group_by("ID_EPISODIO")
        .first()
    )

    transf_df = transf_df.join(
        linked.select(["ID_EPISODIO", "FALLECIDO_RECEPTOR"]),
        on="ID_EPISODIO",
        how="left",
    ).with_columns(
        pl.col("FALLECIDO_RECEPTOR").is_not_null().alias("es_enlazado")
    )

    h_base = df_23.group_by("COD_HOSPITAL").agg(
        [
            pl.len().alias("N_base"),
            pl.col("MORTALIDAD_BINARIA").sum().alias("O_base"),
            pl.col("E_M2CAP").sum().alias("E_base"),
        ]
    )

    h_tr = transf_df.group_by("COD_HOSPITAL").agg(
        [
            pl.len().alias("N_tr"),
            pl.col("es_enlazado").cast(pl.Int32).sum().alias("N_link"),
            pl.col("FALLECIDO_RECEPTOR").fill_null(0).sum().alias("O_link"),
            pl.when(pl.col("es_enlazado"))
            .then(pl.col("P_expected"))
            .otherwise(0)
            .sum()
            .alias("E_link"),
            pl.when(~pl.col("es_enlazado")).then(1).otherwise(0).sum().alias("N_nolink"),
            pl.when(~pl.col("es_enlazado"))
            .then(pl.col("P_expected"))
            .otherwise(0)
            .sum()
            .alias("E_nolink"),
        ]
    )

    h_all = (
        h_base.join(h_tr, on="COD_HOSPITAL", how="left")
        .join(
            cat.select(["cod_hospital", "nombre_oficial", "es_monografico"]),
            left_on="COD_HOSPITAL",
            right_on="cod_hospital",
            how="left",
        )
        .sort("COD_HOSPITAL")
    )

    tau = 0.1948
    mu = -0.0359
    c = np.log(1.10)
    z_thresh = 1.64485

    def eb(O, E):
        oe = O / E
        y = np.log(oe)
        v = 1.0 / O
        B = (tau**2) / (tau**2 + v)
        theta = B * y + (1 - B) * mu
        se = np.sqrt(B * v)
        z = (theta - c) / se
        p = stats.norm.cdf(z)
        return oe, z, p

    print("\n--- 2. COTAS DE SALIDA (LAMBDA_SALIDA) PARA LAS 10 ALERTAS PRIMARIAS ---")
    alerts_ids = [
        "113180",
        "111100",
        "106103",
        "113100",
        "107101",
        "106100",
        "103100",
        "121109",
        "112101",
        "109100",
        "112103",
    ]
    for hid in alerts_ids:
        r = h_all.filter(pl.col("COD_HOSPITAL") == hid).to_dicts()[0]
        O_min = r["O_base"] + r["O_link"]
        E_tot = r["E_base"] + r["E_link"] + r["E_nolink"]
        _, z_min, p_min = eb(O_min, E_tot)

        if z_min >= z_thresh:
            lam_str = "No existe (P_min >= 0.95)"
            factible = "NO (Robusta/Inmune)"
        else:
            low, high = 0.0, 5.0
            for _ in range(50):
                mid = (low + high) / 2.0
                O_new = r["O_base"] + r["O_link"] + mid * r["E_nolink"]
                _, z_mid, _ = eb(O_new, E_tot)
                if z_mid < z_thresh:
                    low = mid
                else:
                    high = mid
            lam_str = f"{low:.2f}x"
            factible = "SÍ (Marginal Sensible)" if low > 0.5 else "Remoto"
        print(
            f"  [{hid}] {r['nombre_oficial'][:25]}: O_base={r['O_base']}, E_base={r['E_base']:.1f} | N_nolink={r['N_nolink']}, E_nolink={r['E_nolink']:.1f} -> Cota lambda_salida = {lam_str} ({factible})"
        )

    print("\n--- 3. PUNTOS DE QUIEBRE (LAMBDA*) EN HOSPITALES DE INTERÉS ---")
    for hid in ["112100", "128109", "114101", "105102", "104100"]:
        r = h_all.filter(pl.col("COD_HOSPITAL") == hid).to_dicts()[0]
        E_tot = r["E_base"] + r["E_link"] + r["E_nolink"]

        # Resolver lambda tal que Z >= z_thresh
        low, high = 0.0, 10.0
        for _ in range(50):
            mid = (low + high) / 2.0
            O_new = r["O_base"] + r["O_link"] + mid * r["E_nolink"]
            _, z_mid, _ = eb(O_new, E_tot)
            if z_mid < z_thresh:
                low = mid
            else:
                high = mid
        print(
            f"  [{hid}] {r['nombre_oficial'][:25]}: E_base={r['E_base']:.2f}, E_nolink={r['E_nolink']:.2f} -> lambda* quiebre = {low:.2f}x"
        )


if __name__ == "__main__":
    ejecutar_analisis_traslados()
