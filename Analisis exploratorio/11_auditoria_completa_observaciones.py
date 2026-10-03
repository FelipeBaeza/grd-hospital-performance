import sys
from pathlib import Path
import polars as pl
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression, ElasticNetCV, ElasticNet
from sklearn.preprocessing import StandardScaler, SplineTransformer

GOLD_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/gold")
SILVER_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/silver")
CAT_PATH = Path("/home/felipe/Documentos/Proyecto final/Tesis/config/catalogo_hospitales_procedencia.csv")
ELIX_PATH = Path("/home/felipe/Documentos/Proyecto final/Tesis/Analisis exploratorio/tabla_canonica_elixhauser_31.csv")

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
    
    elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]
    
    g_sub = g.filter(inp_mask).select(['ID_EPISODIO', 'COD_HOSPITAL', 'EDAD_ANIOS', 'ESTANCIA_DIAS'] + elix_28)
    s_sub = s.filter(inp_mask).select(['ID_EPISODIO', 'TIPOALTA', 'CIP_ENCRIPTADO'] + diag_cols).with_columns([
        n_dx_expr.alias('N_DX'),
        urg.filter(inp_mask).alias('INGRESO_URGENCIA'),
        crit.filter(inp_mask).alias('INGRESO_CRITICO'),
        der.filter(inp_mask).alias('DERIVADO_OTRO_HOSPITAL'),
        sex_m.filter(inp_mask).alias('SEXO_MASCULINO'),
        cie3.filter(inp_mask).alias('CIE10_3C'),
        cens_deriv.filter(inp_mask).alias('ES_CENSURADO')
    ])
    
    m = g_sub.join(s_sub, on='ID_EPISODIO')
    m = m.with_columns(
        pl.when(pl.col('TIPOALTA') == 'FALLECIDO').then(1).otherwise(0).alias('MORTALIDAD_BINARIA')
    )
    return m

def main():
    print("=" * 80)
    print("AUDITORIA INTEGRAL DE OBSERVACIONES Y MODELADO DEFINITIVO")
    print("=" * 80)
    
    ped_codes = ['109101', '112102', '113130']
    
    print("\n--- 1. Reconciliación de Cohorte 2023 ---")
    df_23_raw = get_cohort(2023)
    n_tot_eval = len(df_23_raw.filter(pl.col('ES_CENSURADO') == False))
    o_tot_eval = df_23_raw.filter(pl.col('ES_CENSURADO') == False)['MORTALIDAD_BINARIA'].sum()
    
    df_23_ped = df_23_raw.filter((pl.col('ES_CENSURADO') == False) & (pl.col('COD_HOSPITAL').is_in(ped_codes)))
    n_ped = len(df_23_ped)
    o_ped = df_23_ped['MORTALIDAD_BINARIA'].sum()
    
    df_23 = df_23_raw.filter(~pl.col('COD_HOSPITAL').is_in(ped_codes))
    eval_23 = df_23.filter(pl.col('ES_CENSURADO') == False)
    cens_23 = df_23.filter(pl.col('ES_CENSURADO') == True)
    
    print(f"Total Evaluable Inpatient 2023 (con pediátricos): N = {n_tot_eval:,}, O = {o_tot_eval:,}")
    print(f"Adultos en 3 hospitales pediátricos: N = {n_ped:,}, O = {o_ped:,}")
    print(f"Cohorte Evaluable Red 65 Hospitales Adultos: N = {len(eval_23):,}, O = {eval_23['MORTALIDAD_BINARIA'].sum():,}")
    print(f"Diferencia exacta: {n_tot_eval} - {n_ped} = {len(eval_23)}")

    # 2. Cargar Desarrollo
    print("\nCargando cohortes de desarrollo (2020-2022)...")
    df_dev_raw = pl.concat([get_cohort(y) for y in [2020, 2021, 2022]])
    df_dev = df_dev_raw.filter(~pl.col('COD_HOSPITAL').is_in(ped_codes))
    eval_dev = df_dev.filter(pl.col('ES_CENSURADO') == False)
    
    # Calcular R_DX
    h_ndx_dev = eval_dev.group_by('COD_HOSPITAL').agg(pl.col('N_DX').mean().alias('mean_ndx'))
    eval_dev = eval_dev.join(h_ndx_dev, on='COD_HOSPITAL', how='left').with_columns(
        (pl.col('N_DX') / pl.col('mean_ndx').fill_null(pl.col('N_DX').mean())).alias('R_DX')
    )
    eval_23 = eval_23.join(h_ndx_dev, on='COD_HOSPITAL', how='left').with_columns(
        (pl.col('N_DX') / pl.col('mean_ndx').fill_null(pl.col('N_DX').mean())).alias('R_DX')
    )
    cens_23 = cens_23.join(h_ndx_dev, on='COD_HOSPITAL', how='left').with_columns(
        (pl.col('N_DX') / pl.col('mean_ndx').fill_null(pl.col('N_DX').mean())).alias('R_DX')
    )

    elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]
    f_m1 = ['EDAD_ANIOS'] + elix_28
    f_m2 = ['EDAD_ANIOS', 'INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL', 'SEXO_MASCULINO'] + elix_28
    y_dev = eval_dev['MORTALIDAD_BINARIA'].to_numpy()

    # M1: Demográfico
    sc1 = StandardScaler()
    X_dev_1 = sc1.fit_transform(eval_dev.select(f_m1).fill_null(0).to_pandas().values)
    X_23_1 = sc1.transform(eval_23.select(f_m1).fill_null(0).to_pandas().values)
    m1 = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_1, y_dev)
    p1 = m1.predict_proba(X_23_1)[:, 1]

    # M2: Clínico
    sc2 = StandardScaler()
    X_dev_2 = sc2.fit_transform(eval_dev.select(f_m2).fill_null(0).to_pandas().values)
    X_23_2 = sc2.transform(eval_23.select(f_m2).fill_null(0).to_pandas().values)
    m2 = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_2, y_dev)
    p2 = m2.predict_proba(X_23_2)[:, 1]

    # M3: Clínico + R_DX Splines
    sp = SplineTransformer(n_knots=4, degree=3, include_bias=False)
    sp_dev = sp.fit_transform(eval_dev.select(['R_DX']).to_pandas().values)
    sp_23 = sp.transform(eval_23.select(['R_DX']).to_pandas().values)
    X_dev_3 = np.hstack([X_dev_2, sp_dev])
    X_23_3 = np.hstack([X_23_2, sp_23])
    m3 = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_3, y_dev)
    p3 = m3.predict_proba(X_23_3)[:, 1]

    eval_23 = eval_23.with_columns([
        pl.Series('P1', p1),
        pl.Series('P2', p2),
        pl.Series('P3', p3)
    ])

    O_tot = eval_23['MORTALIDAD_BINARIA'].sum()
    E1_tot = eval_23['P1'].sum()
    E2_tot = eval_23['P2'].sum()
    E3_tot = eval_23['P3'].sum()
    k1 = O_tot / E1_tot
    k2 = O_tot / E2_tot
    k3 = O_tot / E3_tot

    print("\n--- 2. Comparación de los Tres Modelos de Mortalidad en 2023 ---")
    print(f"Modelo 1 (Demográfico puro):       O = {O_tot:,}, E = {E1_tot:,.1f}, k = {k1:.4f}")
    print(f"Modelo 2 (Con Severidad Ingreso):  O = {O_tot:,}, E = {E2_tot:,.1f}, k = {k2:.4f}")
    print(f"Modelo 3 (Corregido R_dx Splines): O = {O_tot:,}, E = {E3_tot:,.1f}, k = {k3:.4f}")

    # Tabla Calibración por Estratos de N_DX para los 3 Modelos
    eval_23 = eval_23.with_columns(
        pl.when(pl.col('N_DX') == 0).then(pl.lit('0 dx'))
          .when(pl.col('N_DX') == 1).then(pl.lit('1 dx'))
          .when(pl.col('N_DX') == 2).then(pl.lit('2 dx'))
          .when(pl.col('N_DX') == 3).then(pl.lit('3 dx'))
          .when(pl.col('N_DX') == 4).then(pl.lit('4 dx'))
          .when(pl.col('N_DX') == 5).then(pl.lit('5 dx'))
          .when(pl.col('N_DX') <= 10).then(pl.lit('6-10 dx'))
          .otherwise(pl.lit('>=11 dx')).alias('ESTRATO_NDX')
    )

    estratos_orden = ['0 dx', '1 dx', '2 dx', '3 dx', '4 dx', '5 dx', '6-10 dx', '>=11 dx']
    print("\n" + "=" * 110)
    print("CALIBRACIÓN EN 2023 A TRAVÉS DE LOS TRES MODELOS POR ESTRATOS DE N_DX")
    print("=" * 110)
    print(f"{'Estrato':<10} | {'N':>8} | {'O':>6} | {'OE_post M1':>12} | {'OE_post M2':>12} | {'OE_post M3':>12} | {'Estado M3':>15}")
    print("-" * 110)
    for est in estratos_orden:
        sub = eval_23.filter(pl.col('ESTRATO_NDX') == est)
        n_s = len(sub)
        o_s = sub['MORTALIDAD_BINARIA'].sum()
        oe1 = (o_s / sub['P1'].sum()) * (1.0 / k1)
        oe2 = (o_s / sub['P2'].sum()) * (1.0 / k2)
        oe3 = (o_s / sub['P3'].sum()) * (1.0 / k3)
        st3 = "DENTRO" if 0.80 <= oe3 <= 1.25 else "FUERA"
        print(f"{est:<10} | {n_s:8,} | {o_s:6,} | {oe1:12.4f} | {oe2:12.4f} | {oe3:12.4f} | {st3:>15}")

    # Suma de estratos 3 a >=11 dx
    sub_3_11 = eval_23.filter(pl.col('N_DX') >= 3)
    n_3_11 = len(sub_3_11)
    o_3_11 = sub_3_11['MORTALIDAD_BINARIA'].sum()
    print(f"\nSuma de Estratos 3 a >=11 dx: N = {n_3_11:,} ({n_3_11/len(eval_23)*100:.1f}%), O = {o_3_11:,} ({o_3_11/O_tot*100:.1f}%)")

    # Efecto del estrato 0-2 dx en hospitales
    h_low_dx = eval_23.group_by('COD_HOSPITAL').agg([
        pl.len().alias('N_hosp'),
        (pl.col('N_DX') <= 2).cast(pl.Int32).sum().alias('N_low_dx'),
        pl.col('MORTALIDAD_BINARIA').sum().alias('O_hosp'),
        pl.col('P1').sum().alias('E1'),
        pl.col('P2').sum().alias('E2'),
        pl.col('P3').sum().alias('E3')
    ]).with_columns([
        (pl.col('N_low_dx') / pl.col('N_hosp') * 100).alias('pct_low_dx'),
        (pl.col('O_hosp') / pl.col('E1') * (1.0/k1)).alias('OE_M1'),
        (pl.col('O_hosp') / pl.col('E2') * (1.0/k2)).alias('OE_M2'),
        (pl.col('O_hosp') / pl.col('E3') * (1.0/k3)).alias('OE_M3')
    ])
    r_low_oe1, _ = stats.spearmanr(h_low_dx['pct_low_dx'], h_low_dx['OE_M1'])
    r_low_oe2, _ = stats.spearmanr(h_low_dx['pct_low_dx'], h_low_dx['OE_M2'])
    r_low_oe3, _ = stats.spearmanr(h_low_dx['pct_low_dx'], h_low_dx['OE_M3'])
    print(f"\nCorrelación Spearman entre % de episodios con 0-2 dx y OE hospitalario:")
    print(f"  Modelo 1: r_s = {r_low_oe1:+.3f}")
    print(f"  Modelo 2: r_s = {r_low_oe2:+.3f}")
    print(f"  Modelo 3: r_s = {r_low_oe3:+.3f}")

    # Correlación de rankings entre M1, M2 y M3
    r_m1_m2, _ = stats.spearmanr(h_low_dx['OE_M1'], h_low_dx['OE_M2'])
    r_m1_m3, _ = stats.spearmanr(h_low_dx['OE_M1'], h_low_dx['OE_M3'])
    r_m2_m3, _ = stats.spearmanr(h_low_dx['OE_M2'], h_low_dx['OE_M3'])
    print(f"\nCorrelación de rankings (Spearman rho) entre especificaciones de mortalidad:")
    print(f"  M1 (Demog) vs M2 (Clinico):    rho = {r_m1_m2:.4f}")
    print(f"  M1 (Demog) vs M3 (R_dx Spl):   rho = {r_m1_m3:.4f}")
    print(f"  M2 (Clinico) vs M3 (R_dx Spl): rho = {r_m2_m3:.4f}")

    # Hospitales que más se mueven entre M1 y M3
    h_ranks = h_low_dx.with_columns([
        pl.col('OE_M1').rank().alias('rank_M1'),
        pl.col('OE_M3').rank().alias('rank_M3')
    ]).with_columns([
        (pl.col('rank_M3') - pl.col('rank_M1')).alias('delta_rank'),
        (pl.col('rank_M3') - pl.col('rank_M1')).abs().alias('abs_delta')
    ])
    cat = pl.read_csv(CAT_PATH)
    h_ranks = h_ranks.join(cat.select(['cod_hospital', 'nombre_oficial']), left_on='COD_HOSPITAL', right_on=pl.col('cod_hospital').cast(pl.String), how='left')
    
    print("\nHospitales con mayores desplazamientos de ranking (M1 Demog -> M3 R_dx Splines):")
    top_movers = h_ranks.sort('abs_delta', descending=True).head(8)
    for r in top_movers.iter_rows(named=True):
        print(f"  Hosp {r['COD_HOSPITAL']} ({r['nombre_oficial'][:30]}): Rank M1={r['rank_M1']:.0f} -> Rank M3={r['rank_M3']:.0f} (Delta={r['delta_rank']:+2.0f} puestos, OE1={r['OE_M1']:.3f}, OE3={r['OE_M3']:.3f})")
    
    # Caso El Pino
    elpino = h_ranks.filter(pl.col('COD_HOSPITAL') == '113180').row(0, named=True)
    print(f"\nDetalle de Hospital El Pino (113180):")
    print(f"  N = {elpino['N_hosp']:,}, O = {elpino['O_hosp']:,}")
    print(f"  M1 Demog:   E = {elpino['E1']:.1f}, OE = {elpino['OE_M1']:.3f}, Rank = {elpino['rank_M1']:.0f}")
    print(f"  M2 Clinico: E = {elpino['E2']:.1f}, OE = {elpino['OE_M2']:.3f}, Rank = {elpino['rank_M2'] if 'rank_M2' in elpino else '-'}")
    print(f"  M3 Splines: E = {elpino['E3']:.1f}, OE = {elpino['OE_M3']:.3f}, Rank = {elpino['rank_M3']:.0f}")
    print(f"  Cambio de E: de {elpino['E1']:.1f} a {elpino['E3']:.1f} (caída de {(elpino['E1']-elpino['E3'])/elpino['E1']*100:.1f}%)")

    # --- 3. Bayes Empírico y Margen de Materialidad Clínica ---
    print("\n" + "=" * 80)
    print("3. BAYES EMPÍRICO Y MARGEN DE MATERIALIDAD CLÍNICA")
    print("=" * 80)
    # Modelo M2/M3 EB
    # Usando M2/M3:
    K = len(h_low_dx)
    log_oe = np.log(h_low_dx['OE_M3'].to_numpy())
    v_w = 1.0 / h_low_dx['O_hosp'].to_numpy()
    w_f = 1.0 / v_w
    mu_meta = np.sum(w_f * log_oe) / np.sum(w_f)
    Q = np.sum(w_f * (log_oe - mu_meta)**2)
    tau2 = max(0.0, (Q - (K - 1)) / (np.sum(w_f) - np.sum(w_f**2) / np.sum(w_f)))
    tau = np.sqrt(tau2)

    B_j = tau2 / (tau2 + v_w)
    log_th_post = B_j * log_oe + (1.0 - B_j) * mu_meta
    se_post = np.sqrt(tau2 * v_w / (tau2 + v_w))
    theta_eb = np.exp(log_th_post)

    # Probabilidades bajo diferentes márgenes de materialidad:
    # 0% margen: P(theta > 1.00)
    # 10% margen: P(theta > 1.10) y P(theta < 0.90)
    # 20% margen: P(theta > 1.20) y P(theta < 0.80)
    p_gt_100 = 1.0 - stats.norm.cdf((0.0 - log_th_post) / se_post)
    p_lt_100 = stats.norm.cdf((0.0 - log_th_post) / se_post)
    
    p_gt_110 = 1.0 - stats.norm.cdf((np.log(1.10) - log_th_post) / se_post)
    p_lt_090 = stats.norm.cdf((np.log(0.90) - log_th_post) / se_post)

    p_gt_120 = 1.0 - stats.norm.cdf((np.log(1.20) - log_th_post) / se_post)
    p_lt_080 = stats.norm.cdf((np.log(0.80) - log_th_post) / se_post)

    print(f"Parámetros EB: tau = {tau:.4f}, tau^2 = {tau2:.4f}, mu_meta = {mu_meta:.4f}")
    print(f"Rango 95% inter-hospitalario de O/E verdadero: [{np.exp(mu_meta - 1.96*tau):.3f}, {np.exp(mu_meta + 1.96*tau):.3f}]")

    print("\nDistribución de Clasificación según Margen de Materialidad Clínica (N = 65 hospitales):")
    for nombre_margen, p_hi, p_lo in [
        ("0% Margen (P(theta > 1.0) >= 0.95)", p_gt_100, p_lt_100),
        ("10% Margen (P(theta > 1.10) >= 0.95 / P(theta < 0.90) >= 0.95)", p_gt_110, p_lt_090),
        ("20% Margen (P(theta > 1.20) >= 0.95 / P(theta < 0.80) >= 0.95)", p_gt_120, p_lt_080),
    ]:
        n_alerta = np.sum(p_hi >= 0.95)
        n_sob = np.sum(p_lo >= 0.95)
        n_prom = K - n_alerta - n_sob
        print(f"  {nombre_margen:<65}: Alerta = {n_alerta:2d} ({n_alerta/K*100:4.1f}%), Promedio = {n_prom:2d} ({n_prom/K*100:4.1f}%), Sobresaliente = {n_sob:2d} ({n_sob/K*100:4.1f}%) | Marcados = {n_alerta+n_sob:2d} ({(n_alerta+n_sob)/K*100:4.1f}%)")

    # --- 4. Tipping Point: Entrada y Salida, y Comparación con Derivaciones Enlazadas ---
    print("\n" + "=" * 80)
    print("4. TIPPING POINT (ENTRADA Y SALIDA) Y DATOS EMPÍRICOS DE TRASLADOS ENLAZADOS")
    print("=" * 80)
    # Censurados 2023
    X_cens_2 = sc2.transform(cens_23.select(f_m2).fill_null(0).to_pandas().values)
    sp_cens = sp.transform(cens_23.select(['R_DX']).to_pandas().values)
    X_cens_3 = np.hstack([X_cens_2, sp_cens])
    p3_cens = m3.predict_proba(X_cens_3)[:, 1]
    cens_23 = cens_23.with_columns(pl.Series('P3_CENS', p3_cens))

    cens_h = cens_23.group_by('COD_HOSPITAL').agg([
        pl.len().alias('N_cens'),
        pl.col('P3_CENS').sum().alias('E_cens'),
        pl.col('P3_CENS').mean().alias('p_esperado_cens')
    ])

    h_tp = h_low_dx.join(cens_h, on='COD_HOSPITAL', how='left').with_columns([
        pl.col('N_cens').fill_null(0),
        pl.col('E_cens').fill_null(0.0),
        pl.col('p_esperado_cens').fill_null(0.0)
    ]).with_columns([
        (pl.col('N_cens') / (pl.col('N_hosp') + pl.col('N_cens')) * 100).alias('tasa_censura'),
        pl.Series('theta_eb', theta_eb),
        pl.Series('se_post', se_post),
        pl.Series('p_gt_110', p_gt_110)
    ]).join(cat.select(['cod_hospital', 'nombre_oficial']), left_on='COD_HOSPITAL', right_on=pl.col('cod_hospital').cast(pl.String), how='left')

    # Linkage empírico: buscar desenlace de derivados en el mismo año en otros hospitales
    # Unir cens_23 con silver_2023 por CIP_ENCRIPTADO
    s23_all = pl.read_parquet(SILVER_DIR / "silver_2023.parquet")
    receptores = s23_all.filter(~s23_all['COD_HOSPITAL'].is_in(ped_codes)).select(['CIP_ENCRIPTADO', 'COD_HOSPITAL', 'FECHAINGRESO', 'FECHAALTA', 'TIPOALTA'])
    
    cens_link = cens_23.select(['CIP_ENCRIPTADO', 'COD_HOSPITAL', 'P3_CENS']).join(
        receptores, on='CIP_ENCRIPTADO', how='inner'
    ).filter(pl.col('COD_HOSPITAL') != pl.col('COD_HOSPITAL_right'))
    
    link_h = cens_link.group_by('COD_HOSPITAL').agg([
        pl.len().alias('n_enlazados'),
        (pl.col('TIPOALTA') == 'FALLECIDO').cast(pl.Int32).sum().alias('o_receptor'),
        pl.col('P3_CENS').sum().alias('e_link')
    ]).with_columns([
        (pl.col('o_receptor') / pl.col('e_link')).alias('lambda_obs'),
        (pl.col('o_receptor') / pl.col('n_enlazados') * 100).alias('mort_obs_receptor')
    ])

    h_tp = h_tp.join(link_h, on='COD_HOSPITAL', how='left')

    # Función para calcular lambda_entrada* y lambda_salida* con margen 10%
    # Alerta si P(theta > 1.10) >= 0.95
    # (log_th_star - log(1.10)) / se_star >= 1.64485
    def calc_lambdas(o_eval, e_eval, e_cens, k_factor, p_base):
        if e_cens <= 0:
            return None, None
        
        def z_val(lam):
            o_tot = o_eval + lam * e_cens
            e_tot = e_eval + e_cens
            oe = (o_tot / e_tot) * (1.0 / k_factor)
            if oe <= 0:
                return -10.0
            log_oe_s = np.log(oe)
            v_s = 1.0 / max(1.0, o_tot)
            b_s = tau2 / (tau2 + v_s)
            log_th_s = b_s * log_oe_s + (1.0 - b_s) * mu_meta
            se_s = np.sqrt(tau2 * v_s / (tau2 + v_s))
            return (log_th_s - np.log(1.10)) / se_s

        z_zero = z_val(0.0)
        # Si ya está en alerta (z_zero >= 1.645 o p_base >= 0.95):
        # Buscamos lambda_salida*: valor de lambda por debajo del cual sale de alerta (z < 1.645)
        # En general, si z(lam) baja cuando lam disminuye
        lam_entrada = None
        lam_salida = None

        if p_base >= 0.95:
            # Ya en alerta. lambda_entrada no aplica. Buscamos lambda_salida (reducción de mortalidad)
            # z_val con lambda bajo: ¿puede bajar de 1.64485?
            if z_val(0.0) < 1.64485:
                # Encuentra punto de cruce en [0, 5]
                low, high = 0.0, 5.0
                for _ in range(30):
                    mid = (low + high) / 2.0
                    if z_val(mid) < 1.64485:
                        low = mid
                    else:
                        high = mid
                lam_salida = (low + high) / 2.0
            else:
                lam_salida = 0.0 # Ni con 0 muertes sale de alerta
        else:
            # No está en alerta. Buscamos lambda_entrada
            if z_val(50.0) >= 1.64485:
                low, high = 0.0, 50.0
                for _ in range(30):
                    mid = (low + high) / 2.0
                    if z_val(mid) < 1.64485:
                        low = mid
                    else:
                        high = mid
                lam_entrada = (low + high) / 2.0

        return lam_entrada, lam_salida

    l_in_list = []
    l_out_list = []
    for r in h_tp.iter_rows(named=True):
        li, lo = calc_lambdas(r['O_hosp'], r['E3'], r['E_cens'], k3, r['p_gt_110'])
        l_in_list.append(li)
        l_out_list.append(lo)

    h_tp = h_tp.with_columns([
        pl.Series('lambda_entrada', l_in_list),
        pl.Series('lambda_salida', l_out_list)
    ])

    print("\nTabla de Tipping Point y Validación Empírica con Derivaciones Enlazadas:")
    print(f"{'Cod':<7} | {'Nombre':<30} | {'Cens%':>6} | {'OE_base':>7} | {'P(>1.10)':>8} | {'Lam_In*':>8} | {'Lam_Out*':>8} | {'Lam_Obs':>8} | {'Mort_Obs_Recep':>14}")
    print("-" * 115)
    for r in h_tp.sort('tasa_censura', descending=True).head(12).iter_rows(named=True):
        lin_str = f"{r['lambda_entrada']:6.2f}x" if r['lambda_entrada'] is not None else "-"
        lout_str = f"{r['lambda_salida']:6.2f}x" if r['lambda_salida'] is not None else "-"
        lobs_str = f"{r['lambda_obs']:6.2f}x" if r['lambda_obs'] is not None else "N/D"
        mobs_str = f"{r['mort_obs_receptor']:5.1f}%" if r['mort_obs_receptor'] is not None else "N/D"
        print(f"{r['COD_HOSPITAL']:<7} | {r['nombre_oficial'][:30]:<30} | {r['tasa_censura']:5.1f}% | {r['OE_M3']:7.3f} | {r['p_gt_110']:7.3f}  | {lin_str:>8} | {lout_str:>8} | {lobs_str:>8} | {mobs_str:>14}")

    # --- 5. Estadía: Etiquetas Canónicas y Criterio 1-SE ---
    print("\n" + "=" * 80)
    print("5. MODELO DE ESTADÍA: ETIQUETAS CANÓNICAS Y CRITERIO 1-SE")
    print("=" * 80)
    elix_canon = pl.read_csv(ELIX_PATH)
    
    # Evaluar cohorte de estadía 2023 de sobrevivientes con estancia > 0
    # Base: eval_23 (que tiene 557,318)
    q_stay = eval_23.filter((pl.col('MORTALIDAD_BINARIA') == 0) & (pl.col('ESTANCIA_DIAS') > 0))
    print(f"Base de Estadía 2023 (Sobrevivientes con estancia > 0): N = {len(q_stay):,}")
    
    # Truncamiento p99
    p99 = 54.0
    q_stay = q_stay.with_columns(
        pl.when(pl.col('ESTANCIA_DIAS') > p99).then(p99).otherwise(pl.col('ESTANCIA_DIAS')).alias('LOS_TRUNC')
    ).with_columns(pl.col('LOS_TRUNC').log().alias('LOG_LOS'))

    # Target encoding CIE10
    g_mean = q_stay['LOG_LOS'].mean()
    cie_stats = q_stay.group_by('CIE10_3C').agg([
        pl.len().alias('n_cie'),
        pl.col('LOG_LOS').mean().alias('mean_cie')
    ]).with_columns([
        ((pl.col('n_cie') * pl.col('mean_cie') + 10.0 * g_mean) / (pl.col('n_cie') + 10.0)).alias('CIE10_ENC')
    ])
    q_stay = q_stay.join(cie_stats.select(['CIE10_3C', 'CIE10_ENC']), on='CIE10_3C', how='left').with_columns(
        pl.col('CIE10_ENC').fill_null(g_mean)
    )

    all_stay_feats = [
        'CIE10_ENC', 'INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL', 
        'SEXO_MASCULINO', 'EDAD_ANIOS'
    ] + elix_28

    scaler_s = StandardScaler()
    X_s = scaler_s.fit_transform(q_stay.select(all_stay_feats).fill_null(0).to_pandas().values)
    y_s = q_stay['LOG_LOS'].to_numpy()

    # Submuestra para CV y 1-SE
    np.random.seed(42)
    idx_sub = np.random.choice(len(X_s), size=min(100000, len(X_s)), replace=False)
    encv = ElasticNetCV(l1_ratio=[0.5, 0.7, 0.9, 1.0], cv=5, random_state=42)
    encv.fit(X_s[idx_sub], y_s[idx_sub])

    # 1-SE rule
    # Mean MSE across folds for the optimal l1_ratio
    l1_idx = np.where(encv.l1_ratio == encv.l1_ratio_)[0][0]
    mse_path = encv.mse_path_[l1_idx] # shape: (n_alphas, n_folds)
    mean_mse = mse_path.mean(axis=-1)
    se_mse = mse_path.std(axis=-1) / np.sqrt(mse_path.shape[-1])
    min_idx = np.argmin(mean_mse)
    min_mse = mean_mse[min_idx]
    target_mse = min_mse + se_mse[min_idx]
    
    # 1-SE alpha is largest alpha whose mean MSE <= target_mse
    eligible = np.where(mean_mse <= target_mse)[0]
    alpha_1se = encv.alphas_[eligible[0]]
    print(f"ElasticNet CV: Alpha Min = {encv.alpha_:.6f} (L1={encv.l1_ratio_:.2f}) | Alpha 1-SE = {alpha_1se:.6f}")

    # Modelo con Alpha 1-SE
    m_1se = ElasticNet(alpha=alpha_1se, l1_ratio=encv.l1_ratio_, random_state=42)
    m_1se.fit(X_s[idx_sub], y_s[idx_sub])

    # Crear tabla con nombres canónicos por join
    df_feats = pl.DataFrame({
        'variable': all_stay_feats,
        'coef_min': encv.coef_,
        'coef_1se': m_1se.coef_
    })

    # Join con catalogo canonico
    df_feats = df_feats.join(elix_canon.select(['categoria_id', 'nombre_comorbilidad']), left_on='variable', right_on='categoria_id', how='left')
    df_feats = df_feats.with_columns(
        pl.when(pl.col('variable') == 'CIE10_ENC').then(pl.lit('Diagnóstico Principal (LOHO)'))
          .when(pl.col('variable') == 'INGRESO_URGENCIA').then(pl.lit('Ingreso por Urgencia'))
          .when(pl.col('variable') == 'INGRESO_CRITICO').then(pl.lit('Ingreso a Cama Crítica (UCI/UTI)'))
          .when(pl.col('variable') == 'DERIVADO_OTRO_HOSPITAL').then(pl.lit('Derivado de Otro Hospital'))
          .when(pl.col('variable') == 'SEXO_MASCULINO').then(pl.lit('Sexo Masculino'))
          .when(pl.col('variable') == 'EDAD_ANIOS').then(pl.lit('Edad (Años)'))
          .otherwise(pl.col('nombre_comorbilidad')).alias('nombre_estandar')
    )

    print("\nResultados de Selección con Criterio de Mínimo Error vs Criterio 1-SE:")
    print(f"{'Variable':<12} | {'Nombre Canónico':<35} | {'Coef Min':>10} | {'Exp(B_min)':>10} | {'Coef 1-SE':>10} | {'Exp(B_1se)':>10} | {'Estado 1-SE':>12}")
    print("-" * 115)
    for r in df_feats.sort(pl.col('coef_min').abs(), descending=True).iter_rows(named=True):
        st_1se = "RETENIDA" if abs(r['coef_1se']) > 1e-4 else "DESCARTADA"
        print(f"{r['variable']:<12} | {r['nombre_estandar'][:35]:<35} | {r['coef_min']:+10.5f} | {np.exp(r['coef_min']):10.4f} | {r['coef_1se']:+10.5f} | {np.exp(r['coef_1se']):10.4f} | {st_1se:>12}")

    n_ret_min = np.sum(np.abs(encv.coef_) > 1e-4)
    n_ret_1se = np.sum(np.abs(m_1se.coef_) > 1e-4)
    print(f"\nVariables retenidas bajo criterio de mínimo error: {n_ret_min} de {len(all_stay_feats)}")
    print(f"Variables retenidas bajo criterio más parsimonioso 1-SE: {n_ret_1se} de {len(all_stay_feats)}")

if __name__ == "__main__":
    main()
