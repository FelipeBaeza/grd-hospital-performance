import sys
from pathlib import Path
import polars as pl
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
    
    urg = s['TIPO_INGRESO'].str.to_uppercase().str.contains('URGEN').cast(pl.Int32)
    crit = s['SERVICIOINGRESO'].str.to_uppercase().str.contains('UCI|UTI|CRITIC|INTENS|INTERMED').cast(pl.Int32)
    der = (
        s['HOSPPROCEDENCIA'].is_not_null() & (s['HOSPPROCEDENCIA'] != '0') & (s['HOSPPROCEDENCIA'] != '')
        | s['TIPO_PROCEDENCIA'].str.to_uppercase().str.contains('DERIV|OTRO|HOSP')
    ).cast(pl.Int32)
    sex_m = (g['SEXO'] == 1).cast(pl.Int32)
    
    elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]
    
    g_sub = g.filter(inp_mask).select(['ID_EPISODIO', 'COD_HOSPITAL', 'EDAD_ANIOS', 'ESTANCIA_DIAS'] + elix_28)
    s_sub = s.filter(inp_mask).select(['ID_EPISODIO', 'TIPOALTA']).with_columns([
        urg.filter(inp_mask).alias('INGRESO_URGENCIA'),
        crit.filter(inp_mask).alias('INGRESO_CRITICO'),
        der.filter(inp_mask).alias('DERIVADO_OTRO_HOSPITAL'),
        sex_m.filter(inp_mask).alias('SEXO_MASCULINO'),
        cens_deriv.filter(inp_mask).alias('ES_CENSURADO')
    ])
    
    m = g_sub.join(s_sub, on='ID_EPISODIO')
    m = m.with_columns(
        pl.when(pl.col('TIPOALTA') == 'FALLECIDO').then(1).otherwise(0).alias('MORTALIDAD_BINARIA')
    )
    return m

def main():
    print("Cargando cohortes...")
    df_2023 = get_cohort(2023)
    ped_codes = ['109101', '112102', '113130']
    df_2023 = df_2023.filter(~pl.col('COD_HOSPITAL').is_in(ped_codes))

    eval_23 = df_2023.filter(pl.col('ES_CENSURADO') == False)
    cens_23 = df_2023.filter(pl.col('ES_CENSURADO') == True)

    elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]
    feats = ['EDAD_ANIOS', 'INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL', 'SEXO_MASCULINO'] + elix_28

    scaler = StandardScaler()
    X_eval = scaler.fit_transform(eval_23.select(feats).fill_null(0).to_pandas().values)
    y_eval = eval_23['MORTALIDAD_BINARIA'].to_numpy()

    clf = LogisticRegression(solver='lbfgs', max_iter=200, random_state=42).fit(X_eval, y_eval)
    p_hat_eval = clf.predict_proba(X_eval)[:, 1]
    eval_23 = eval_23.with_columns(pl.Series('P_HAT', p_hat_eval))

    X_cens = scaler.transform(cens_23.select(feats).fill_null(0).to_pandas().values)
    p_hat_cens = clf.predict_proba(X_cens)[:, 1]
    cens_23 = cens_23.with_columns(pl.Series('P_HAT_CENS', p_hat_cens))

    # Agregados de evaluables por hospital
    h_eval = eval_23.group_by('COD_HOSPITAL').agg([
        pl.len().alias('N_eval'),
        pl.col('MORTALIDAD_BINARIA').sum().alias('O_eval'),
        pl.col('P_HAT').sum().alias('E_eval')
    ])

    # Agregados de censurados por hospital
    h_cens = cens_23.group_by('COD_HOSPITAL').agg([
        pl.len().alias('N_cens'),
        pl.col('P_HAT_CENS').sum().alias('E_cens'),
        pl.col('P_HAT_CENS').mean().alias('p_esperado_cens')
    ])

    h_tot = h_eval.join(h_cens, on='COD_HOSPITAL', how='left').with_columns([
        pl.col('N_cens').fill_null(0),
        pl.col('E_cens').fill_null(0.0),
        pl.col('p_esperado_cens').fill_null(0.0)
    ]).with_columns([
        (pl.col('N_cens') / (pl.col('N_eval') + pl.col('N_cens')) * 100).alias('tasa_censura'),
        (pl.col('O_eval') / pl.col('E_eval')).alias('OE_crudo'),
        (1.0 / pl.col('O_eval')).alias('v_w'),
        (pl.col('O_eval') / pl.col('E_eval')).log().alias('log_oe')
    ])

    cat = pl.read_csv(CAT_PATH)
    h_tot = h_tot.join(cat.select(['cod_hospital', 'nombre_oficial']), left_on='COD_HOSPITAL', right_on=pl.col('cod_hospital').cast(pl.String), how='left')

    # Empirical Bayes parameters
    K = len(h_tot)
    log_oe = h_tot['log_oe'].to_numpy()
    v_w = h_tot['v_w'].to_numpy()
    w_fixed = 1.0 / v_w
    mu_meta = np.sum(w_fixed * log_oe) / np.sum(w_fixed)
    Q = np.sum(w_fixed * (log_oe - mu_meta) ** 2)
    tau2 = max(0.0, (Q - (K - 1)) / (np.sum(w_fixed) - np.sum(w_fixed ** 2) / np.sum(w_fixed)))
    tau = np.sqrt(tau2)

    # Baseline posterior
    B_j = tau2 / (tau2 + v_w)
    log_theta_post = B_j * log_oe + (1.0 - B_j) * mu_meta
    se_post = np.sqrt(tau2 * v_w / (tau2 + v_w))
    z_post = (log_theta_post - 0.0) / se_post
    p_exceso = stats.norm.cdf(z_post)

    h_tot = h_tot.with_columns([
        pl.Series('tau', np.full(K, tau)),
        pl.Series('log_theta_post', log_theta_post),
        pl.Series('se_post', se_post),
        pl.Series('p_exceso_base', p_exceso)
    ]).with_columns(
        pl.when(pl.col('p_exceso_base') >= 0.95).then(pl.lit('ALERTA_MORTALIDAD'))
          .when(pl.col('p_exceso_base') <= 0.05).then(pl.lit('SOBRESALIENTE'))
          .otherwise(pl.lit('PROMEDIO')).alias('clase_base')
    )

    # Function to find lambda* such that P(theta* > 1.0) >= 0.95
    # For a hypothetical mortality O_cens = lambda * E_cens:
    # O_tot = O_eval + lambda * E_cens
    # E_tot = E_eval + E_cens
    # OE_star = O_tot / E_tot
    # log_oe_star = log(OE_star)
    # v_w_star = 1.0 / O_tot
    # B_star = tau2 / (tau2 + v_w_star)
    # log_theta_star = B_star * log_oe_star + (1 - B_star) * mu_meta
    # se_star = sqrt(tau2 * v_w_star / (tau2 + v_w_star))
    # z_star = log_theta_star / se_star
    # We want z_star = 1.64485 (which corresponds to p_exceso = 0.95)

    def calc_lambda_star(o_eval, e_eval, e_cens):
        if e_cens <= 0:
            return None
        # Binary search or root finding for lambda in [0, 50]
        def z_diff(lam):
            o_tot = o_eval + lam * e_cens
            e_tot = e_eval + e_cens
            if o_tot <= 0:
                return -10.0
            oe = o_tot / e_tot
            log_oe_s = np.log(oe)
            v_s = 1.0 / o_tot
            b_s = tau2 / (tau2 + v_s)
            log_th_s = b_s * log_oe_s + (1.0 - b_s) * mu_meta
            se_s = np.sqrt(tau2 * v_s / (tau2 + v_s))
            return (log_th_s / se_s) - 1.644853

        if z_diff(0.0) >= 0:
            return 0.0 # Already in alert!
        if z_diff(50.0) < 0:
            return None # Impossible even at 50x risk
        
        low, high = 0.0, 50.0
        for _ in range(50):
            mid = (low + high) / 2.0
            if z_diff(mid) < 0:
                low = mid
            else:
                high = mid
        return (low + high) / 2.0

    lambda_stars = []
    for r in h_tot.iter_rows(named=True):
        lam = calc_lambda_star(r['O_eval'], r['E_eval'], r['E_cens'])
        lambda_stars.append(lam)

    h_tot = h_tot.with_columns(pl.Series('lambda_star', lambda_stars))

    print("\n" + "=" * 115)
    print("TIPPING POINT UNIFICADO: MULTIPLICADOR DE RIESGO ESPERADO (LAMBDA*) PARA ALCANZAR ALERTA POSTERIOR (P >= 0.95)")
    print("=" * 115)
    print(f"{'Cod':<7} | {'Nombre':<35} | {'Censura':>7} | {'E_cens':>7} | {'E_eval':>7} | {'OE_base':>7} | {'Clase Base':<17} | {'Lambda*':>10} | {'Mortalidad Cens Eq':>18}")
    print("-" * 115)
    top_cens = h_tot.sort('tasa_censura', descending=True).head(15)
    for r in top_cens.iter_rows(named=True):
        lam = r['lambda_star']
        if lam == 0.0:
            lam_str = "Ya en alerta"
            mort_eq = "Basal >= 0.95"
        elif lam is not None:
            lam_str = f"{lam:7.2f}x"
            mort_eq = f"{lam * r['p_esperado_cens'] * 100:6.1f}%"
        else:
            lam_str = "> 50x"
            mort_eq = "Inalcanzable"
        print(f"{r['COD_HOSPITAL']:<7} | {r['nombre_oficial'][:35]:<35} | {r['tasa_censura']:6.2f}% | {r['E_cens']:7.1f} | {r['E_eval']:7.1f} | {r['OE_crudo']:7.3f} | {r['clase_base']:<17} | {lam_str:>10} | {mort_eq:>18}")

if __name__ == "__main__":
    main()
