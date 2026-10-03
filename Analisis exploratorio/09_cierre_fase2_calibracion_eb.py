import sys
from pathlib import Path
import polars as pl
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, SplineTransformer

GOLD_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/gold")
SILVER_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/silver")
CAT_PATH = Path("/home/felipe/Documentos/Proyecto final/Tesis/config/catalogo_hospitales_procedencia.csv")

def get_cohort(year):
    g = pl.read_parquet(GOLD_DIR / f"gold_{year}.parquet")
    s = pl.read_parquet(SILVER_DIR / f"silver_{year}.parquet")
    
    c_ex01 = (g['IR_29301_COD_GRD'].is_null()) | (g['MDC'] == 0) | (g['IR_29301_COD_GRD'].str.starts_with('99'))
    c_ex02 = (~c_ex01) & (g['MDC'] == 15)
    c_ex03 = (~c_ex01) & (~c_ex02) & (g['MDC'] == 14)
    c_ex04 = (~c_ex01) & (~c_ex02) & (~c_ex03) & ((g['EDAD_ANIOS'] < 18) | (g['EDAD_ANIOS'].is_null()))
    c_ex05 = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (s['TIPO_ACTIVIDAD'] != 'HOSPITALIZACIÓN')
    inp_mask = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (~c_ex05)
    
    cens_deriv = s['TIPOALTA'].str.contains('DERIVACIÓN OTRO HOSPITAL|DERIVACIÓN INST. PRIVADA')
    eval_mask = inp_mask & (~cens_deriv)
    
    # Calculate N_DX (DIAGNOSTICO2..35)
    diag_cols = [f'DIAGNOSTICO{i}' for i in range(2, 36)]
    n_dx_expr = pl.sum_horizontal([
        pl.when(pl.col(c).is_not_null() & (pl.col(c) != '') & (pl.col(c) != '0')).then(1).otherwise(0)
        for c in diag_cols
    ])
    
    urg = s['TIPO_INGRESO'].str.to_uppercase().str.contains('URGEN').cast(pl.Int32)
    crit = s['SERVICIOINGRESO'].str.to_uppercase().str.contains('UCI|UTI|CRITIC|INTENS|INTERMED').cast(pl.Int32)
    der = (
        s['HOSPPROCEDENCIA'].is_not_null() & (s['HOSPPROCEDENCIA'] != '0') & (s['HOSPPROCEDENCIA'] != '')
        | s['TIPO_PROCEDENCIA'].str.to_uppercase().str.contains('DERIV|OTRO|HOSP')
    ).cast(pl.Int32)
    sex_m = (g['SEXO'] == 1).cast(pl.Int32)
    
    elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]
    g_sub = g.filter(eval_mask).select(['ID_EPISODIO', 'COD_HOSPITAL', 'EDAD_ANIOS', 'ESTANCIA_DIAS'] + elix_28)
    s_sub = s.filter(eval_mask).select(['ID_EPISODIO', 'TIPOALTA'] + diag_cols).with_columns([
        n_dx_expr.alias('N_DX'),
        urg.filter(eval_mask).alias('INGRESO_URGENCIA'),
        crit.filter(eval_mask).alias('INGRESO_CRITICO'),
        der.filter(eval_mask).alias('DERIVADO_OTRO_HOSPITAL'),
        sex_m.filter(eval_mask).alias('SEXO_MASCULINO')
    ]).select(['ID_EPISODIO', 'TIPOALTA', 'N_DX', 'INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL', 'SEXO_MASCULINO'])
    
    m = g_sub.join(s_sub, on='ID_EPISODIO')
    m = m.with_columns(
        pl.when(pl.col('TIPOALTA') == 'FALLECIDO').then(1).otherwise(0).alias('MORTALIDAD_BINARIA')
    )
    return m

def main():
    print("=" * 80)
    print("RECALCULO DE CALIBRACION, SOBREDISPERSION Y GRUPOS DE HOSPITALES")
    print("=" * 80)

    print("Cargando cohortes...")
    df_dev = pl.concat([get_cohort(y) for y in [2020, 2021, 2022]])
    df_2023 = get_cohort(2023)

    # Excluir pediatricos
    ped_codes = ['109101', '112102', '113130']
    df_dev = df_dev.filter(~pl.col('COD_HOSPITAL').is_in(ped_codes))
    df_2023 = df_2023.filter(~pl.col('COD_HOSPITAL').is_in(ped_codes))

    print(f"Dev (2020-2022): N={len(df_dev):,}, O={df_dev['MORTALIDAD_BINARIA'].sum():,}")
    print(f"Calibración (2023): N={len(df_2023):,}, O={df_2023['MORTALIDAD_BINARIA'].sum():,}")

    # Calcular media de N_DX por hospital en desarrollo para centrado/razon
    h_ndx_dev = df_dev.group_by('COD_HOSPITAL').agg(pl.col('N_DX').mean().alias('mean_ndx_hosp'))
    df_dev = df_dev.join(h_ndx_dev, on='COD_HOSPITAL', how='left').with_columns(
        (pl.col('N_DX') / pl.col('mean_ndx_hosp').fill_null(pl.col('N_DX').mean())).alias('R_DX')
    )
    df_2023 = df_2023.join(h_ndx_dev, on='COD_HOSPITAL', how='left').with_columns(
        (pl.col('N_DX') / pl.col('mean_ndx_hosp').fill_null(pl.col('N_DX').mean())).alias('R_DX')
    )

    elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]
    base_feats = ['EDAD_ANIOS', 'INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL', 'SEXO_MASCULINO'] + elix_28

    # 1. MODELO BASE (Sin R_DX)
    print("\n--- 1. Modelo Base (Sin R_DX) ---")
    scaler_base = StandardScaler()
    X_dev_base = scaler_base.fit_transform(df_dev.select(base_feats).fill_null(0).to_pandas().values)
    y_dev = df_dev['MORTALIDAD_BINARIA'].to_numpy()

    clf_base = LogisticRegression(solver='lbfgs', max_iter=200, random_state=42)
    clf_base.fit(X_dev_base, y_dev)

    X_23_base = scaler_base.transform(df_2023.select(base_feats).fill_null(0).to_pandas().values)
    p_hat_base = clf_base.predict_proba(X_23_base)[:, 1]
    df_2023 = df_2023.with_columns(pl.Series('P_HAT_BASE', p_hat_base))

    # 2. MODELO CORREGIDO (Con R_DX splines)
    print("\n--- 2. Modelo Corregido (Con R_DX splines) ---")
    spline = SplineTransformer(n_knots=4, degree=3, include_bias=False)
    spline_dev = spline.fit_transform(df_dev.select(['R_DX']).to_pandas().values)
    spline_23 = spline.transform(df_2023.select(['R_DX']).to_pandas().values)

    X_dev_corr = np.hstack([X_dev_base, spline_dev])
    X_23_corr = np.hstack([X_23_base, spline_23])

    clf_corr = LogisticRegression(solver='lbfgs', max_iter=200, random_state=42)
    clf_corr.fit(X_dev_corr, y_dev)
    p_hat_corr = clf_corr.predict_proba(X_23_corr)[:, 1]
    df_2023 = df_2023.with_columns(pl.Series('P_HAT_CORR', p_hat_corr))

    # Definir estratos de N_DX
    df_2023 = df_2023.with_columns(
        pl.when(pl.col('N_DX') == 0).then(pl.lit('0 dx'))
          .when(pl.col('N_DX') == 1).then(pl.lit('1 dx'))
          .when(pl.col('N_DX') == 2).then(pl.lit('2 dx'))
          .when(pl.col('N_DX') == 3).then(pl.lit('3 dx'))
          .when(pl.col('N_DX') == 4).then(pl.lit('4 dx'))
          .when(pl.col('N_DX') == 5).then(pl.lit('5 dx'))
          .when(pl.col('N_DX') <= 10).then(pl.lit('6-10 dx'))
          .otherwise(pl.lit('>=11 dx')).alias('ESTRATO_NDX')
    )

    # Definir Complejidad Hospitalaria (Complejos vs Provinciales)
    # Hospitales Complejos: >= 8,000 admisiones o > 15% camas críticas
    h_vols = df_2023.group_by('COD_HOSPITAL').agg([
        pl.len().alias('n_hosp'),
        (pl.col('INGRESO_CRITICO') == 1).cast(pl.Float64).mean().alias('pct_crit')
    ]).with_columns(
        pl.when((pl.col('n_hosp') >= 6000) | (pl.col('pct_crit') >= 0.12)).then(pl.lit('COMPLEJO_ALTA'))
          .otherwise(pl.lit('PROVINCIAL_MEDIANA')).alias('TIPO_HOSP')
    )
    df_2023 = df_2023.join(h_vols.select(['COD_HOSPITAL', 'TIPO_HOSP']), on='COD_HOSPITAL', how='left')

    # Factores globales de recalibracion
    O_tot = df_2023['MORTALIDAD_BINARIA'].sum()
    E_tot_base = df_2023['P_HAT_BASE'].sum()
    E_tot_corr = df_2023['P_HAT_CORR'].sum()
    k_base = O_tot / E_tot_base
    k_corr = O_tot / E_tot_corr
    print(f"Global Base: O={O_tot:,}, E={E_tot_base:,.1f}, k={k_base:.4f}")
    print(f"Global Corr: O={O_tot:,}, E={E_tot_corr:,.1f}, k={k_corr:.4f}")

    # Tabla Calibracion por Estrato NDX
    estratos_orden = ['0 dx', '1 dx', '2 dx', '3 dx', '4 dx', '5 dx', '6-10 dx', '>=11 dx']
    print("\n" + "=" * 95)
    print("CALIBRACIÓN POR ESTRATOS DE N_DX: BASE VS CORREGIDO (CON R_DX SPLINES)")
    print("=" * 95)
    print(f"{'Estrato N_dx':<12} | {'N':>8} | {'O':>6} | {'E_base':>9} | {'OE_post_base':>12} | {'E_corr':>9} | {'OE_post_corr':>12} | {'Estado Corr':>15}")
    print("-" * 95)
    for est in estratos_orden:
        sub = df_2023.filter(pl.col('ESTRATO_NDX') == est)
        n_s = len(sub)
        o_s = sub['MORTALIDAD_BINARIA'].sum()
        e_b = sub['P_HAT_BASE'].sum()
        e_c = sub['P_HAT_CORR'].sum()
        oe_p_b = (o_s / e_b) * (1.0 / k_base)
        oe_p_c = (o_s / e_c) * (1.0 / k_corr)
        est_str = "DENTRO" if (0.80 <= oe_p_c <= 1.25) else "FUERA"
        print(f"{est:<12} | {n_s:8,} | {o_s:6,} | {e_b:9.1f} | {oe_p_b:12.4f} | {e_c:9.1f} | {oe_p_c:12.4f} | {est_str:>15}")

    # Tabla Calibracion por Grupo de Hospital
    print("\n" + "=" * 95)
    print("CALIBRACIÓN POR GRUPO DE HOSPITAL (COMPLEJOS VS PROVINCIALES)")
    print("=" * 95)
    print(f"{'Grupo Hospital':<20} | {'K':>3} | {'N':>8} | {'O':>6} | {'E_base':>9} | {'OE_post_base':>12} | {'E_corr':>9} | {'OE_post_corr':>12}")
    print("-" * 95)
    for grp in ['COMPLEJO_ALTA', 'PROVINCIAL_MEDIANA']:
        sub = df_2023.filter(pl.col('TIPO_HOSP') == grp)
        k_grp = sub['COD_HOSPITAL'].n_unique()
        n_s = len(sub)
        o_s = sub['MORTALIDAD_BINARIA'].sum()
        e_b = sub['P_HAT_BASE'].sum()
        e_c = sub['P_HAT_CORR'].sum()
        oe_p_b = (o_s / e_b) * (1.0 / k_base)
        oe_p_c = (o_s / e_c) * (1.0 / k_corr)
        print(f"{grp:<20} | {k_grp:3d} | {n_s:8,} | {o_s:6,} | {e_b:9.1f} | {oe_p_b:12.4f} | {e_c:9.1f} | {oe_p_c:12.4f}")

    # Calcular Sobredispersion de Spiegelhalter para ambos modelos
    for mod_name, e_col in [('Modelo Base', 'P_HAT_BASE'), ('Modelo Corregido (Splines R_dx)', 'P_HAT_CORR')]:
        h_agg = df_2023.group_by('COD_HOSPITAL').agg([
            pl.col('MORTALIDAD_BINARIA').sum().alias('O'),
            pl.col(e_col).sum().alias('E')
        ])
        k_mod = h_agg['O'].sum() / h_agg['E'].sum()
        h_agg = h_agg.with_columns((pl.col('E') * k_mod).alias('E_calib'))
        h_agg = h_agg.with_columns([
            ((pl.col('O') - pl.col('E_calib')) / pl.col('E_calib').sqrt()).alias('Z')
        ])
        z_vals = h_agg['Z'].to_numpy()
        K = len(z_vals)
        phi_crudo = np.sum(z_vals ** 2) / (K - 1)
        z_wins = stats.mstats.winsorize(z_vals, limits=[0.10, 0.10])
        phi_wins = np.sum(z_wins ** 2) / (K - 1)
        
        # Meta-analysis tau
        oe_vals = (h_agg['O'] / h_agg['E_calib']).to_numpy()
        log_oe = np.log(oe_vals)
        v_w = 1.0 / h_agg['O'].to_numpy()
        w_f = 1.0 / v_w
        mu_m = np.sum(w_f * log_oe) / np.sum(w_f)
        Q = np.sum(w_f * (log_oe - mu_m)**2)
        tau2 = max(0.0, (Q - (K - 1)) / (np.sum(w_f) - np.sum(w_f**2) / np.sum(w_f)))
        tau = np.sqrt(tau2)
        print(f"\n{mod_name}:")
        print(f"  phi_crudo = {phi_crudo:.2f}, phi_wins = {phi_wins:.2f} (sqrt(phi) = {np.sqrt(phi_wins):.2f})")
        print(f"  tau = {tau:.4f} (tau^2 = {tau2:.4f})")

if __name__ == "__main__":
    main()
