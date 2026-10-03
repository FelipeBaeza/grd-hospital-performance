import sys
from pathlib import Path
import polars as pl
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import ElasticNetCV, ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler

GOLD_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/gold")
SILVER_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/silver")

def main():
    print("=" * 80)
    print("CIERRE DEFINITIVO FASE 2: ESPECIFICACIÓN COMPLETA Y MODELO BAYESIANO EMPÍRICO")
    print("=" * 80)

    # 1. Carga de cohorte
    g23 = pl.read_parquet(GOLD_DIR / "gold_2023.parquet")
    s23 = pl.read_parquet(SILVER_DIR / "silver_2023.parquet")

    c_ex01 = (g23['IR_29301_COD_GRD'].is_null()) | (g23['MDC'] == 0) | (g23['IR_29301_COD_GRD'].str.starts_with('99'))
    c_ex02 = (~c_ex01) & (g23['MDC'] == 15)
    c_ex03 = (~c_ex01) & (~c_ex02) & (g23['MDC'] == 14)
    c_ex04 = (~c_ex01) & (~c_ex02) & (~c_ex03) & ((g23['EDAD_ANIOS'] < 18) | (g23['EDAD_ANIOS'].is_null()))
    c_ex05 = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (s23['TIPO_ACTIVIDAD'] != 'HOSPITALIZACIÓN')
    inp_mask = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (~c_ex05)

    cens_deriv = s23['TIPOALTA'].str.contains('DERIVACIÓN OTRO HOSPITAL|DERIVACIÓN INST. PRIVADA')
    eval_mask = inp_mask & (~cens_deriv)

    urg = s23['TIPO_INGRESO'].str.to_uppercase().str.contains('URGEN').cast(pl.Int32)
    crit = s23['SERVICIOINGRESO'].str.to_uppercase().str.contains('UCI|UTI|CRITIC|INTENS|INTERMED').cast(pl.Int32)
    der = (
        s23['HOSPPROCEDENCIA'].is_not_null() & (s23['HOSPPROCEDENCIA'] != '0') & (s23['HOSPPROCEDENCIA'] != '')
        | s23['TIPO_PROCEDENCIA'].str.to_uppercase().str.contains('DERIV|OTRO|HOSP')
    ).cast(pl.Int32)
    sex_m = (g23['SEXO'] == 1).cast(pl.Int32)
    cie3 = s23['DIAGNOSTICO1'].str.replace(r'\.', '').str.slice(0, 3)

    elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]

    df_eval = g23.filter(eval_mask).select(['ID_EPISODIO', 'COD_HOSPITAL', 'EDAD_ANIOS', 'ESTANCIA_DIAS', 'MORTALIDAD_BINARIA'] + elix_28)
    s_feat = s23.filter(eval_mask).select(['ID_EPISODIO']).with_columns([
        urg.filter(eval_mask).alias('INGRESO_URGENCIA'),
        crit.filter(eval_mask).alias('INGRESO_CRITICO'),
        der.filter(eval_mask).alias('DERIVADO_OTRO_HOSPITAL'),
        sex_m.filter(eval_mask).alias('SEXO_MASCULINO'),
        cie3.filter(eval_mask).alias('CIE10_3C')
    ])
    df_full = df_eval.join(s_feat, on='ID_EPISODIO')
    print(f"  Cohorte Inpatient evaluable 2023: {len(df_full):,} episodios")

    # 2. Estadía: Selección ElasticNet y Bootstrap de Estabilidad
    print("\n2. Modelo de Estadía: Selección ElasticNet y Bootstrap de Estabilidad (50 réplicas)...")
    q_stay = df_full.filter((pl.col('MORTALIDAD_BINARIA') == 0) & (pl.col('ESTANCIA_DIAS') > 0))
    p99 = 54.0
    q_stay = q_stay.with_columns(
        pl.when(pl.col('ESTANCIA_DIAS') > p99).then(p99).otherwise(pl.col('ESTANCIA_DIAS')).alias('LOS_TRUNC')
    )
    q_stay = q_stay.with_columns(pl.col('LOS_TRUNC').log().alias('LOG_LOS'))
    global_mean_los = q_stay['LOG_LOS'].mean()
    
    cie_stats = q_stay.group_by('CIE10_3C').agg([
        pl.len().alias('n_cie'),
        pl.col('LOG_LOS').mean().alias('mean_cie')
    ]).with_columns([
        ((pl.col('n_cie') * pl.col('mean_cie') + 10.0 * global_mean_los) / (pl.col('n_cie') + 10.0)).alias('CIE10_ENC')
    ])
    q_stay = q_stay.join(cie_stats.select(['CIE10_3C', 'CIE10_ENC']), on='CIE10_3C', how='left').with_columns(
        pl.col('CIE10_ENC').fill_null(global_mean_los)
    )

    all_stay_feats = [
        'EDAD_ANIOS', 'INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL', 
        'SEXO_MASCULINO', 'CIE10_ENC'
    ] + elix_28

    scaler_stay = StandardScaler()
    X_stay = scaler_stay.fit_transform(q_stay.select(all_stay_feats).fill_null(0).to_pandas().values)
    y_stay = q_stay['LOG_LOS'].to_numpy()

    np.random.seed(42)
    sample_idx = np.random.choice(len(X_stay), size=min(50000, len(X_stay)), replace=False)
    X_sub = X_stay[sample_idx]
    y_sub = y_stay[sample_idx]

    encv = ElasticNetCV(l1_ratio=[0.5, 0.7, 0.9, 1.0], cv=5, random_state=42, max_iter=2000)
    encv.fit(X_sub, y_sub)
    print(f"  Hiperparámetros óptimos: alpha = {encv.alpha_:.6f}, L1-ratio = {encv.l1_ratio_:.2f}")

    B = 50
    counts = np.zeros(len(all_stay_feats))
    coeffs_sum = np.zeros(len(all_stay_feats))
    n_b = int(0.632 * len(X_sub))

    for b in range(B):
        b_idx = np.random.choice(len(X_sub), size=n_b, replace=False)
        m_b = ElasticNet(alpha=encv.alpha_, l1_ratio=encv.l1_ratio_, max_iter=2000, random_state=b)
        m_b.fit(X_sub[b_idx], y_sub[b_idx])
        active = (np.abs(m_b.coef_) > 1e-4).astype(int)
        counts += active
        coeffs_sum += m_b.coef_

    stability_freq = counts / B
    mean_coeffs = coeffs_sum / B

    print(f"\n  RESULTADOS DE SELECCIÓN POR ESTABILIDAD EN ESTADÍA (Umbral >= 70%):")
    print(f"  {'Variable':<25} | {'Frecuencia':>10} | {'Beta Estandarizado':>18} | {'Exp(Beta)':>10} | {'Estado':>12}")
    print("  " + "-" * 85)
    for feat, freq, b_val in sorted(zip(all_stay_feats, stability_freq, mean_coeffs), key=lambda x: -abs(x[2])):
        estado = "RETENIDA" if freq >= 0.70 else "DESCARTADA"
        print(f"  {feat:<25} | {freq*100:9.1f}% | {b_val:+18.5f} | {np.exp(b_val):10.4f} | {estado:>12}")

    # 3. Bayes Empírico en Mortalidad
    print("\n3. Regla Única de Clasificación: Modelo de Efectos Aleatorios / Bayes Empírico...")
    feats_m = ['EDAD_ANIOS', 'INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL', 'SEXO_MASCULINO'] + elix_28
    df_m_eval = df_full.filter(pl.col('MORTALIDAD_BINARIA').is_not_null())
    scaler_m = StandardScaler()
    X_m = scaler_m.fit_transform(df_m_eval.select(feats_m).fill_null(0).to_pandas().values)
    y_m = df_m_eval['MORTALIDAD_BINARIA'].to_numpy()

    clf_m = LogisticRegression(solver='lbfgs', max_iter=200, random_state=42).fit(X_m, y_m)
    p_hat_m = clf_m.predict_proba(X_m)[:, 1]
    df_m_eval = df_m_eval.with_columns(pl.Series('P_HAT', p_hat_m))

    h_m = df_m_eval.group_by('COD_HOSPITAL').agg([
        pl.len().alias('N'),
        pl.col('MORTALIDAD_BINARIA').sum().alias('O'),
        pl.col('P_HAT').sum().alias('E')
    ]).filter(~pl.col('COD_HOSPITAL').is_in(['109101', '112102', '113130']))
    K = len(h_m)

    h_m = h_m.with_columns([
        (pl.col('O') / pl.col('E')).alias('OE_crudo'),
        (pl.col('O') / pl.col('E')).log().alias('log_OE'),
        (1.0 / pl.col('O')).alias('var_within')
    ])

    log_oe = h_m['log_OE'].to_numpy()
    v_w = h_m['var_within'].to_numpy()
    w_fixed = 1.0 / v_w
    mu_meta = np.sum(w_fixed * log_oe) / np.sum(w_fixed)
    Q = np.sum(w_fixed * (log_oe - mu_meta) ** 2)
    tau2 = max(0.0, (Q - (K - 1)) / (np.sum(w_fixed) - np.sum(w_fixed ** 2) / np.sum(w_fixed)))
    tau = np.sqrt(tau2)
    print(f"  Desviación estándar entre hospitales (tau): {tau:.4f} (Varianza tau^2 = {tau2:.4f})")

    B_j = tau2 / (tau2 + v_w)
    log_theta_post = B_j * log_oe + (1.0 - B_j) * mu_meta
    se_post = np.sqrt(tau2 * v_w / (tau2 + v_w))
    theta_eb = np.exp(log_theta_post)

    z_post = (log_theta_post - 0.0) / se_post
    p_exceso = stats.norm.cdf(z_post)

    h_m = h_m.with_columns([
        pl.Series('THETA_EB', theta_eb),
        pl.Series('SE_POST', se_post),
        pl.Series('P_EXCESO', p_exceso)
    ]).with_columns([
        pl.when(pl.col('P_EXCESO') >= 0.95).then(pl.lit('ALERTA_MORTALIDAD'))
          .when(pl.col('P_EXCESO') <= 0.05).then(pl.lit('SOBRESALIENTE'))
          .otherwise(pl.lit('PROMEDIO')).alias('CLASE_EB')
    ])

    print("\n  REGLA ÚNICA DE CLASIFICACIÓN (BAYES EMPÍRICO AL 95% POSTERIOR):")
    print(h_m['CLASE_EB'].value_counts())

    # 4. Tipping Point
    print("\n4. Tipping Point Unificado en Función del Case-Mix Derivado...")
    cens_mask = inp_mask & cens_deriv
    df_cens = g23.filter(cens_mask).select(['ID_EPISODIO', 'COD_HOSPITAL', 'EDAD_ANIOS'] + elix_28)
    s_cens = s23.filter(cens_mask).select(['ID_EPISODIO']).with_columns([
        urg.filter(cens_mask).alias('INGRESO_URGENCIA'),
        crit.filter(cens_mask).alias('INGRESO_CRITICO'),
        der.filter(cens_mask).alias('DERIVADO_OTRO_HOSPITAL'),
        sex_m.filter(cens_mask).alias('SEXO_MASCULINO')
    ])
    cens_full = df_cens.join(s_cens, on='ID_EPISODIO')
    
    X_cens = scaler_m.transform(cens_full.select(feats_m).fill_null(0).to_pandas().values)
    p_cens = clf_m.predict_proba(X_cens)[:, 1]
    cens_full = cens_full.with_columns(pl.Series('P_HAT_CENS', p_cens))

    cens_agg = cens_full.group_by('COD_HOSPITAL').agg([
        pl.len().alias('N_cens'),
        pl.col('P_HAT_CENS').sum().alias('E_cens'),
        pl.col('P_HAT_CENS').mean().alias('p_esperado_cens')
    ])

    comp_tp = h_m.join(cens_agg, on='COD_HOSPITAL', how='left').with_columns([
        pl.col('N_cens').fill_null(0),
        pl.col('E_cens').fill_null(0.0),
        pl.col('p_esperado_cens').fill_null(0.0)
    ]).with_columns([
        (pl.col('N_cens') / (pl.col('N') + pl.col('N_cens')) * 100).alias('tasa_censura')
    ])

    # Multiplicador lambda* tal que (O + lambda * E_cens) / (E + E_cens) alcance alerta posterior
    # Alerta en EB equivale aproximadamente a log(O_tot / E_tot) >= 1.645 * se_post
    print("  Hospitales con mayor censura y riesgo esperado de sus derivados:")
    top_c = comp_tp.sort('tasa_censura', descending=True).head(8)
    for r in top_c.iter_rows(named=True):
        print(f"    - Hosp {r['COD_HOSPITAL']}: N_eval={r['N']:5,}, N_cens={r['N_cens']:4,}, Censura={r['tasa_censura']:5.2f}%, Riesgo esperado derivados={r['p_esperado_cens']*100:4.1f}%, OE actual={r['OE_crudo']:.3f}, Clase EB={r['CLASE_EB']}")

if __name__ == "__main__":
    main()
