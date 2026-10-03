import gc
from pathlib import Path
import polars as pl
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.preprocessing import StandardScaler, SplineTransformer

GOLD_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/gold")
SILVER_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/silver")
CAT_PATH = Path("/home/felipe/Documentos/Proyecto final/Tesis/config/catalogo_hospitales_procedencia.csv")
ELIX_PATH = Path("/home/felipe/Documentos/Proyecto final/Tesis/Analisis exploratorio/tabla_canonica_elixhauser_31.csv")

ped_codes = ['109101', '112102', '113130']
mono_codes = ['110110', '112103', '106102'] # Traumatologico, Torax, Pereira

def get_eval_cohort(year):
    g = pl.read_parquet(GOLD_DIR / f"gold_{year}.parquet")
    s = pl.read_parquet(SILVER_DIR / f"silver_{year}.parquet")
    
    c_ex01 = (g['IR_29301_COD_GRD'].is_null()) | (g['MDC'] == 0) | (g['IR_29301_COD_GRD'].str.starts_with('99'))
    c_ex02 = (~c_ex01) & (g['MDC'] == 15)
    c_ex03 = (~c_ex01) & (~c_ex02) & (g['MDC'] == 14)
    c_ex04 = (~c_ex01) & (~c_ex02) & (~c_ex03) & ((g['EDAD_ANIOS'] < 18) | (g['EDAD_ANIOS'].is_null()))
    c_ex05 = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (s['TIPO_ACTIVIDAD'] != 'HOSPITALIZACIÓN')
    inp_mask = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (~c_ex05)
    
    cens_deriv = s['TIPOALTA'].str.contains('DERIVACIÓN OTRO HOSPITAL|DERIVACIÓN INST. PRIVADA')
    eval_mask = inp_mask & (~cens_deriv) & (~g['COD_HOSPITAL'].is_in(ped_codes))
    
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
    cie3 = s['DIAGNOSTICO1'].str.replace(r'\.', '').str.slice(0, 3)
    proc1_sn = (s['PROCEDIMIENTO1'].is_not_null() & (s['PROCEDIMIENTO1'] != '') & (s['PROCEDIMIENTO1'] != '0')).cast(pl.Int32)
    
    elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]
    
    g_sub = g.filter(eval_mask).select(['ID_EPISODIO', 'COD_HOSPITAL', 'EDAD_ANIOS', 'ESTANCIA_DIAS'] + elix_28)
    s_sub = s.filter(eval_mask).select(['ID_EPISODIO', 'TIPOALTA', 'CIP_ENCRIPTADO', 'TIPO_INGRESO'] + diag_cols).with_columns([
        n_dx_expr.alias('N_DX'),
        urg.filter(eval_mask).alias('INGRESO_URGENCIA'),
        crit.filter(eval_mask).alias('INGRESO_CRITICO'),
        der.filter(eval_mask).alias('DERIVADO_OTRO_HOSPITAL'),
        sex_m.filter(eval_mask).alias('SEXO_MASCULINO'),
        cie3.filter(eval_mask).alias('CIE10_3C'),
        proc1_sn.filter(eval_mask).alias('TIENE_PROCEDIMIENTO'),
        (pl.col('TIPOALTA') == 'FALLECIDO').cast(pl.Int32).alias('MORTALIDAD_BINARIA')
    ])
    
    return g_sub.join(s_sub, on='ID_EPISODIO')

def main():
    print("=" * 80)
    print("RESOLUCIÓN DEFINITIVA DE OBSERVACIONES Y MODELADO CONJUNTO")
    print("=" * 80)
    
    # 1. Cargar Dev (2020-2022) y Calibracion 2023
    print("Cargando datos...")
    df_dev = pl.concat([get_eval_cohort(y) for y in [2020, 2021, 2022]])
    df_23 = get_eval_cohort(2023)
    
    # R_DX
    h_ndx_dev = df_dev.group_by('COD_HOSPITAL').agg(pl.col('N_DX').mean().alias('mean_ndx'))
    df_dev = df_dev.join(h_ndx_dev, on='COD_HOSPITAL', how='left').with_columns(
        (pl.col('N_DX') / pl.col('mean_ndx').fill_null(pl.col('N_DX').mean())).alias('R_DX')
    )
    df_23 = df_23.join(h_ndx_dev, on='COD_HOSPITAL', how='left').with_columns(
        (pl.col('N_DX') / pl.col('mean_ndx').fill_null(pl.col('N_DX').mean())).alias('R_DX')
    )
    
    elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]
    f_clin = ['EDAD_ANIOS', 'INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL', 'SEXO_MASCULINO'] + elix_28
    
    # Ajustar Modelo 2 (Clínico) y Modelo 3 (Splines R_DX)
    y_dev = df_dev['MORTALIDAD_BINARIA'].to_numpy()
    sc2 = StandardScaler()
    X_dev_2 = sc2.fit_transform(df_dev.select(f_clin).fill_null(0).to_pandas().values)
    X_23_2 = sc2.transform(df_23.select(f_clin).fill_null(0).to_pandas().values)
    m2 = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_2, y_dev)
    p2 = m2.predict_proba(X_23_2)[:, 1]
    
    sp = SplineTransformer(n_knots=4, degree=3, include_bias=False)
    sp_dev = sp.fit_transform(df_dev.select(['R_DX']).to_pandas().values)
    sp_23 = sp.transform(df_23.select(['R_DX']).to_pandas().values)
    X_dev_3 = np.hstack([X_dev_2, sp_dev])
    X_23_3 = np.hstack([X_23_2, sp_23])
    m3 = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_3, y_dev)
    p3 = m3.predict_proba(X_23_3)[:, 1]
    
    df_23 = df_23.with_columns([
        pl.Series('P2', p2),
        pl.Series('P3', p3)
    ])
    
    O_tot = df_23['MORTALIDAD_BINARIA'].sum()
    k2 = O_tot / df_23['P2'].sum()
    k3 = O_tot / df_23['P3'].sum()
    
    # Hospital aggregates
    h_agg = df_23.group_by('COD_HOSPITAL').agg([
        pl.len().alias('N'),
        pl.col('MORTALIDAD_BINARIA').sum().alias('O'),
        pl.col('P2').sum().alias('E2'),
        pl.col('P3').sum().alias('E3'),
        (pl.col('N_DX') <= 2).cast(pl.Int32).sum().alias('N_low_dx')
    ]).with_columns([
        (pl.col('O') / pl.col('E2') * (1.0/k2)).alias('OE_M2'),
        (pl.col('O') / pl.col('E3') * (1.0/k3)).alias('OE_M3'),
        (1.0 / pl.col('O')).alias('v_j'),
        (pl.col('N_low_dx') / pl.col('N') * 100).alias('pct_low_dx')
    ])
    
    cat = pl.read_csv(CAT_PATH)
    h_agg = h_agg.join(cat.select(['cod_hospital', 'nombre_oficial', 'es_pediatrico']), left_on='COD_HOSPITAL', right_on=pl.col('cod_hospital').cast(pl.String), how='left')
    h_agg = h_agg.with_columns(
        pl.col('COD_HOSPITAL').is_in(mono_codes).alias('es_monografico')
    )
    
    # EB en M2 y M3
    def run_eb(oe_series, o_series):
        K = len(oe_series)
        log_oe = np.log(oe_series.to_numpy())
        v_w = 1.0 / o_series.to_numpy()
        w_f = 1.0 / v_w
        mu_m = np.sum(w_f * log_oe) / np.sum(w_f)
        Q = np.sum(w_f * (log_oe - mu_m)**2)
        tau2 = max(0.0, (Q - (K - 1)) / (np.sum(w_f) - np.sum(w_f**2) / np.sum(w_f)))
        tau = np.sqrt(tau2)
        B_j = tau2 / (tau2 + v_w)
        log_th_post = B_j * log_oe + (1.0 - B_j) * mu_m
        se_post = np.sqrt(tau2 * v_w / (tau2 + v_w))
        p_100 = 1.0 - stats.norm.cdf((0.0 - log_th_post) / se_post)
        p_110 = 1.0 - stats.norm.cdf((np.log(1.10) - log_th_post) / se_post)
        p_120 = 1.0 - stats.norm.cdf((np.log(1.20) - log_th_post) / se_post)
        p_090 = stats.norm.cdf((np.log(0.90) - log_th_post) / se_post)
        return tau, mu_m, log_th_post, se_post, p_100, p_110, p_120, p_090

    tau_m2, mu_m2, th_m2, se_m2, p100_m2, p110_m2, p120_m2, p090_m2 = run_eb(h_agg['OE_M2'], h_agg['O'])
    tau_m3, mu_m3, th_m3, se_m3, p100_m3, p110_m3, p120_m3, p090_m3 = run_eb(h_agg['OE_M3'], h_agg['O'])
    
    h_agg = h_agg.with_columns([
        pl.Series('log_th_M2', th_m2),
        pl.Series('se_post_M2', se_m2),
        pl.Series('p_gt_110_M2', p110_m2),
        pl.Series('p_gt_120_M2', p120_m2),
        pl.Series('p_gt_100_M2', p100_m2),
        pl.Series('log_th_M3', th_m3),
        pl.Series('se_post_M3', se_m3),
        pl.Series('p_gt_110_M3', p110_m3),
        pl.Series('p_gt_120_M3', p120_m3),
        pl.Series('p_gt_100_M3', p100_m3)
    ]).with_columns([
        (pl.col('p_gt_110_M2') >= 0.95).alias('alerta_M2'),
        (pl.col('p_gt_110_M3') >= 0.95).alias('alerta_M3')
    ]).with_columns([
        (pl.col('alerta_M2') & pl.col('alerta_M3')).alias('alerta_robusta')
    ])
    
    print("\n--- 1. Alerta Robusta (M2 y M3 Simultáneamente con Margen 10%) ---")
    print(f"Hospitales en Alerta M2 (P > 1.10 >= 0.95): {h_agg['alerta_M2'].sum()}")
    print(f"Hospitales en Alerta M3 (P > 1.10 >= 0.95): {h_agg['alerta_M3'].sum()}")
    print(f"Hospitales en ALERTA ROBUSTA (M2 y M3): {h_agg['alerta_robusta'].sum()} de 65")
    
    print("\nDetalle de Hospitales en Alerta en M2 o M3:")
    print(f"{'Cod':<7} | {'Nombre':<35} | {'Mono':<5} | {'OE_M2':>7} | {'P_M2':>6} | {'OE_M3':>7} | {'P_M3':>6} | {'P_120_M3':>8} | {'Alerta Robusta':<14}")
    print("-" * 115)
    alert_union = h_agg.filter(pl.col('alerta_M2') | pl.col('alerta_M3')).sort('OE_M3', descending=True)
    for r in alert_union.iter_rows(named=True):
        mono_str = "SI" if r['es_monografico'] else "NO"
        rob_str = "ROBUSTA" if r['alerta_robusta'] else "SENSIBLE"
        print(f"{r['COD_HOSPITAL']:<7} | {r['nombre_oficial'][:35]:<35} | {mono_str:<5} | {r['OE_M2']:7.3f} | {r['p_gt_110_M2']:6.3f} | {r['OE_M3']:7.3f} | {r['p_gt_110_M3']:6.3f} | {r['p_gt_120_M3']:8.3f} | {rob_str:<14}")
    
    print("\n--- 2. Detalle de Curanilahue (128109) en Margen 10% vs 20% ---")
    curan = h_agg.filter(pl.col('COD_HOSPITAL') == '128109').row(0, named=True)
    print(f"  O = {curan['O']}, E3 = {curan['E3']:.1f}, OE_M3 = {curan['OE_M3']:.3f}, v_j = {curan['v_j']:.6f}")
    print(f"  log(theta_post) = {curan['log_th_M3']:.4f}, se_post = {curan['se_post_M3']:.4f}")
    print(f"  P(theta > 1.10) = {curan['p_gt_110_M3']:.4f} (>= 0.95: ALERTA AL 10%)")
    print(f"  P(theta > 1.20) = {curan['p_gt_120_M3']:.4f} (< 0.95: SALE DE ALERTA AL 20%)")
    
    # Hospitales en alerta al 20% en M3
    n_alert_20 = (h_agg['p_gt_120_M3'] >= 0.95).sum()
    print(f"\nHospitales en Alerta al 20% en M3 (P(theta > 1.20) >= 0.95): {n_alert_20} de 65 (Curanilahue cae fuera)")
    for r in h_agg.filter(pl.col('p_gt_120_M3') >= 0.95).iter_rows(named=True):
        print(f"  - Hosp {r['COD_HOSPITAL']} ({r['nombre_oficial'][:30]}): OE_M3={r['OE_M3']:.3f}, P(>1.20)={r['p_gt_120_M3']:.3f}")

    # --- 3. Auditoría de la Hipótesis del Subregistro en 0-2 Diagnósticos ---
    print("\n" + "=" * 80)
    print("--- 3. Auditoría de la Hipótesis de 0-2 Diagnósticos ---")
    print("=" * 80)
    
    sub_02 = df_23.filter(pl.col('N_DX') <= 2)
    sub_ge3 = df_23.filter(pl.col('N_DX') >= 3)
    
    pct_urg_02 = sub_02['INGRESO_URGENCIA'].mean() * 100
    pct_urg_ge3 = sub_ge3['INGRESO_URGENCIA'].mean() * 100
    
    pct_proc_02 = sub_02['TIENE_PROCEDIMIENTO'].mean() * 100
    pct_proc_ge3 = sub_ge3['TIENE_PROCEDIMIENTO'].mean() * 100
    
    med_los_02 = sub_02['ESTANCIA_DIAS'].median()
    med_los_ge3 = sub_ge3['ESTANCIA_DIAS'].median()
    
    mean_los_02 = sub_02['ESTANCIA_DIAS'].mean()
    mean_los_ge3 = sub_ge3['ESTANCIA_DIAS'].mean()
    
    mort_02 = sub_02['MORTALIDAD_BINARIA'].mean() * 100
    mort_ge3 = sub_ge3['MORTALIDAD_BINARIA'].mean() * 100
    
    print(f"Comparación Clínica de Pacientes con 0-2 dx vs >=3 dx:")
    print(f"  Episodios:              0-2 dx = {len(sub_02):,} ({len(sub_02)/len(df_23)*100:.1f}%) | >=3 dx = {len(sub_ge3):,} ({len(sub_ge3)/len(df_23)*100:.1f}%)")
    print(f"  Mortalidad observada:   0-2 dx = {mort_02:.3f}% ({sub_02['MORTALIDAD_BINARIA'].sum():,} muertes) | >=3 dx = {mort_ge3:.2f}% ({sub_ge3['MORTALIDAD_BINARIA'].sum():,} muertes)")
    print(f"  Ingreso por Urgencia:   0-2 dx = {pct_urg_02:.1f}% | >=3 dx = {pct_urg_ge3:.1f}%")
    print(f"  Procedimiento Quirúrg:  0-2 dx = {pct_proc_02:.1f}% | >=3 dx = {pct_proc_ge3:.1f}%")
    print(f"  Estancia Mediana:       0-2 dx = {med_los_02:.1f} días (Media: {mean_los_02:.1f}) | >=3 dx = {med_los_ge3:.1f} días (Media: {mean_los_ge3:.1f})")

    # Bootstrap pareado de la diferencia de correlación r_s(M1) vs r_s(M3)
    np.random.seed(42)
    B = 1000
    K = len(h_agg)
    diffs = []
    r1_list = []
    r3_list = []
    p_low = h_agg['pct_low_dx'].to_numpy()
    oe_1 = h_agg['OE_M2'].to_numpy() # O M1
    oe_3 = h_agg['OE_M3'].to_numpy()
    
    for _ in range(B):
        idx = np.random.choice(K, size=K, replace=True)
        r1, _ = stats.spearmanr(p_low[idx], oe_1[idx])
        r3, _ = stats.spearmanr(p_low[idx], oe_3[idx])
        diffs.append(r3 - r1)
        r1_list.append(r1)
        r3_list.append(r3)
        
    diffs = np.array(diffs)
    r1_arr = np.array(r1_list)
    r3_arr = np.array(r3_list)
    print(f"\nBootstrap Pareado (1,000 réplicas) de Correlación (% 0-2 dx vs OE):")
    print(f"  r_s(M2 Clínico):     {np.mean(r1_arr):+.3f} [IC 95%: {np.percentile(r1_arr, 2.5):+.3f} a {np.percentile(r1_arr, 97.5):+.3f}]")
    print(f"  r_s(M3 R_dx Spl):   {np.mean(r3_arr):+.3f} [IC 95%: {np.percentile(r3_arr, 2.5):+.3f} a {np.percentile(r3_arr, 97.5):+.3f}]")
    print(f"  Delta r_s (M3 - M2): {np.mean(diffs):+.3f} [IC 95%: {np.percentile(diffs, 2.5):+.3f} a {np.percentile(diffs, 97.5):+.3f}], p-valor = {np.mean(diffs >= 0):.4f}")

    # --- 4. Selección de Estadía en Desarrollo (2020-2022) y Comparación de Rankings ---
    print("\n" + "=" * 80)
    print("--- 4. Estadía: Selección en Desarrollo (2020-2022) y Comparación 34 vs 17 ---")
    print("=" * 80)
    
    # Submuestra de desarrollo para estadía
    stay_dev = df_dev.filter((pl.col('MORTALIDAD_BINARIA') == 0) & (pl.col('ESTANCIA_DIAS') > 0))
    print(f"Población de Estadía en Desarrollo (2020-2022): N = {len(stay_dev):,}")
    
    stay_dev = stay_dev.with_columns(
        pl.when(pl.col('ESTANCIA_DIAS') > 54.0).then(54.0).otherwise(pl.col('ESTANCIA_DIAS')).alias('LOS_TRUNC')
    ).with_columns(pl.col('LOS_TRUNC').log().alias('LOG_LOS'))
    
    g_mean_dev = stay_dev['LOG_LOS'].mean()
    cie_stats_dev = stay_dev.group_by('CIE10_3C').agg([
        pl.len().alias('n_cie'),
        pl.col('LOG_LOS').mean().alias('mean_cie')
    ]).with_columns([
        ((pl.col('n_cie') * pl.col('mean_cie') + 10.0 * g_mean_dev) / (pl.col('n_cie') + 10.0)).alias('CIE10_ENC')
    ])
    stay_dev = stay_dev.join(cie_stats_dev.select(['CIE10_3C', 'CIE10_ENC']), on='CIE10_3C', how='left').with_columns(
        pl.col('CIE10_ENC').fill_null(g_mean_dev)
    )
    
    all_stay_feats = [
        'CIE10_ENC', 'INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL', 
        'SEXO_MASCULINO', 'EDAD_ANIOS'
    ] + elix_28
    
    feats_17 = [
        'CIE10_ENC', 'INGRESO_URGENCIA', 'ELIX_24', 'INGRESO_CRITICO', 'ELIX_04',
        'ELIX_14', 'ELIX_23', 'ELIX_05', 'ELIX_15', 'ELIX_30', 'ELIX_19',
        'DERIVADO_OTRO_HOSPITAL', 'ELIX_01', 'ELIX_03', 'ELIX_06', 'EDAD_ANIOS', 'ELIX_09'
    ]
    
    scaler_34 = StandardScaler()
    X_dev_34 = scaler_34.fit_transform(stay_dev.select(all_stay_feats).fill_null(0).to_pandas().values)
    y_dev_s = stay_dev['LOG_LOS'].to_numpy()
    
    # Ajustar modelo 34 variables y modelo 17 variables en desarrollo
    m_stay_34 = ElasticNet(alpha=0.002, l1_ratio=0.5, random_state=42, max_iter=1000).fit(X_dev_34, y_dev_s)
    
    scaler_17 = StandardScaler()
    X_dev_17 = scaler_17.fit_transform(stay_dev.select(feats_17).fill_null(0).to_pandas().values)
    m_stay_17 = ElasticNet(alpha=0.002, l1_ratio=0.5, random_state=42, max_iter=1000).fit(X_dev_17, y_dev_s)
    
    # Evaluar en 2023
    stay_23 = df_23.filter((pl.col('MORTALIDAD_BINARIA') == 0) & (pl.col('ESTANCIA_DIAS') > 0))
    stay_23 = stay_23.with_columns(
        pl.when(pl.col('ESTANCIA_DIAS') > 54.0).then(54.0).otherwise(pl.col('ESTANCIA_DIAS')).alias('LOS_TRUNC')
    ).with_columns(pl.col('LOS_TRUNC').log().alias('LOG_LOS'))
    stay_23 = stay_23.join(cie_stats_dev.select(['CIE10_3C', 'CIE10_ENC']), on='CIE10_3C', how='left').with_columns(
        pl.col('CIE10_ENC').fill_null(g_mean_dev)
    )
    
    X_23_34 = scaler_34.transform(stay_23.select(all_stay_feats).fill_null(0).to_pandas().values)
    X_23_17 = scaler_17.transform(stay_23.select(feats_17).fill_null(0).to_pandas().values)
    
    pred_34 = np.exp(m_stay_34.predict(X_23_34))
    pred_17 = np.exp(m_stay_17.predict(X_23_17))
    
    stay_23 = stay_23.with_columns([
        pl.Series('pred_34', pred_34),
        pl.Series('pred_17', pred_17)
    ])
    
    h_stay = stay_23.group_by('COD_HOSPITAL').agg([
        pl.col('ESTANCIA_DIAS').sum().alias('obs_dias'),
        pl.col('pred_34').sum().alias('exp_34'),
        pl.col('pred_17').sum().alias('exp_17')
    ]).with_columns([
        (pl.col('obs_dias') / pl.col('exp_34')).alias('OE_stay_34'),
        (pl.col('obs_dias') / pl.col('exp_17')).alias('OE_stay_17')
    ])
    
    rho_stay, _ = stats.spearmanr(h_stay['OE_stay_34'], h_stay['OE_stay_17'])
    print(f"Correlación de Rangos de Spearman (rho) en Estadía (34 variables vs 17 variables): rho = {rho_stay:.5f}")
    print(f"Desplazamiento RMS de ranking entre 34 y 17 variables: {np.sqrt(np.mean((h_stay['OE_stay_34'].rank().to_numpy() - h_stay['OE_stay_17'].rank().to_numpy())**2)):.2f} puestos")

if __name__ == "__main__":
    main()
