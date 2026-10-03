from pathlib import Path
import sys
import polars as pl
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

GOLD_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/gold")
SILVER_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/silver")

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
    
    # Calculate N_DX
    diag_cols = [f'DIAGNOSTICO{i}' for i in range(2, 36)]
    n_dx_expr = pl.sum_horizontal([
        pl.when(pl.col(c).is_not_null() & (pl.col(c) != '') & (pl.col(c) != '0')).then(1).otherwise(0)
        for c in diag_cols
    ])
    
    elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]
    g_sub = g.filter(eval_mask).select(['ID_EPISODIO', 'COD_HOSPITAL', 'EDAD_ANIOS', 'ESTANCIA_DIAS'] + elix_28)
    s_sub = s.filter(eval_mask).select(['ID_EPISODIO', 'TIPOALTA'] + diag_cols).with_columns(n_dx_expr.alias('N_DX')).select(['ID_EPISODIO', 'TIPOALTA', 'N_DX'])
    
    m = g_sub.join(s_sub, on='ID_EPISODIO')
    m = m.with_columns(
        pl.when(pl.col('TIPOALTA') == 'FALLECIDO').then(1).otherwise(0).alias('MORTALIDAD_BINARIA')
    )
    return m

def main():
    print("=" * 80)
    print("RECALCULO RIGUROSO DE CALIBRACIÓN EN COHORTE INPATIENT ADULTOS")
    print("=" * 80)
    
    print("Cargando desarrollo (2020-2022)...")
    df_dev = pl.concat([get_cohort(y) for y in [2020, 2021, 2022]])
    print(f"Dev: N={len(df_dev):,}, Defunciones={df_dev['MORTALIDAD_BINARIA'].sum():,}")

    print("Cargando calibración (2023)...")
    df_2023 = get_cohort(2023)
    print(f"2023: N={len(df_2023):,}, Defunciones={df_2023['MORTALIDAD_BINARIA'].sum():,}")

    elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]
    feats = ['EDAD_ANIOS'] + elix_28

    scaler = StandardScaler()
    X_dev = scaler.fit_transform(df_dev.select(feats).fill_null(0).to_pandas().values)
    y_dev = df_dev['MORTALIDAD_BINARIA'].to_numpy()

    clf = LogisticRegression(solver='lbfgs', max_iter=200, random_state=42)
    clf.fit(X_dev, y_dev)

    X_2023 = scaler.transform(df_2023.select(feats).fill_null(0).to_pandas().values)
    p_2023 = clf.predict_proba(X_2023)[:, 1]
    df_2023 = df_2023.with_columns(pl.Series('E_PRED', p_2023))

    total_O = df_2023['MORTALIDAD_BINARIA'].sum()
    total_E = df_2023['E_PRED'].sum()
    global_OE = total_O / total_E
    print(f"\nTotal O 2023 = {total_O:,}")
    print(f"Total E 2023 = {total_E:,.2f}")
    print(f"O/E Global Real Crudo = {global_OE:.4f} (Factor recalibración = {global_OE:.4f})")

    print("\n" + "=" * 90)
    print("TABLA CANÓNICA DE CALIBRACIÓN POR ESTRATOS DE N_DX (ADULTOS INPATIENT 2023)")
    print("=" * 90)
    print(f"{'Estrato N_dx':<14} | {'N Episodios':>11} | {'Muertes O':>9} | {'Muertes E':>11} | {'O/E Crudo':>10} | {'O/E Post-Recal':>14} | {'Estado':>12}")
    print("-" * 90)

    estratos = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 35)]
    for low, high in estratos:
        if low == high:
            label = f"{low} dx"
            sub = df_2023.filter(pl.col('N_DX') == low)
        elif high >= 35:
            label = f">= {low} dx"
            sub = df_2023.filter(pl.col('N_DX') >= low)
        else:
            label = f"{low}-{high} dx"
            sub = df_2023.filter((pl.col('N_DX') >= low) & (pl.col('N_DX') <= high))
        
        n_c = len(sub)
        o_c = sub['MORTALIDAD_BINARIA'].sum()
        e_c = sub['E_PRED'].sum()
        oe_r = o_c / e_c
        oe_p = oe_r / global_OE
        
        # Criterio contrato [0.80, 1.25]
        estado = "DENTRO" if (0.80 <= oe_p <= 1.25) else "FUERA"
        print(f"{label:<14} | {n_c:11,} | {o_c:9,} | {e_c:11.1f} | {oe_r:10.4f} | {oe_p:14.4f} | {estado:>12}")

    print("-" * 90)

    # -------------------------------------------------------------------------
    # SOBREDISPERSIÓN Y FUNNEL PLOT (SPIEGELHALTER 2005)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("AJUSTE DE SOBREDISPERSIÓN EN FUNNEL PLOT (SPIEGELHALTER 2005)")
    print("=" * 90)

    # Base vs No-Día-0
    h_b = df_2023.group_by('COD_HOSPITAL').agg([
        pl.len().alias('N'),
        pl.col('MORTALIDAD_BINARIA').sum().alias('O'),
        pl.col('E_PRED').sum().alias('E')
    ]).with_columns((pl.col('O') / pl.col('E')).alias('OE'))

    h_nd0 = df_2023.filter(pl.col('ESTANCIA_DIAS') > 0).group_by('COD_HOSPITAL').agg([
        pl.len().alias('N_nd0'),
        pl.col('MORTALIDAD_BINARIA').sum().alias('O_nd0'),
        pl.col('E_PRED').sum().alias('E_nd0')
    ]).with_columns((pl.col('O_nd0') / pl.col('E_nd0')).alias('OE_nd0'))

    comp_h = h_b.join(h_nd0, on='COD_HOSPITAL').filter(~pl.col('COD_HOSPITAL').is_in(['109101', '112102', '113130']))
    K = len(comp_h)
    print(f"Total hospitales de agudos evaluados: {K}")

    # Spearman rho entre rankings base y no-dia-0
    comp_h = comp_h.with_columns([
        pl.col('OE').rank().alias('rank_b'),
        pl.col('OE_nd0').rank().alias('rank_nd0')
    ]).with_columns((pl.col('rank_b') - pl.col('rank_nd0')).abs().alias('rank_diff'))
    
    rho_d0, _ = stats.spearmanr(comp_h['OE'], comp_h['OE_nd0'])
    max_d = comp_h['rank_diff'].max()
    rms_d = np.sqrt((comp_h['rank_diff'] ** 2).mean())
    print(f"Spearman rho real (Base vs No-Día-0): {rho_d0:.4f}")
    print(f"Desplazamiento RMS: {rms_d:.2f} puestos | Desplazamiento máximo: {max_d} puestos")

    # Factor de sobredispersión phi de Spiegelhalter (winsorizado o Poisson residual deviance)
    # phi = (1 / (K - 1)) * sum((O - E)^2 / E)
    # Para O/E: Var(Y_j) = 1/E_j + tau^2 (aditiva) o phi/E_j (multiplicativa)
    z_scores = ((comp_h['O'] - comp_h['E']) / np.sqrt(comp_h['E'])).to_numpy()
    phi_crudo = np.sum(z_scores ** 2) / (K - 1)
    
    # Winsorización al 10% de Spiegelhalter para evitar que outliers inflen phi
    z_wins = np.clip(z_scores, np.percentile(z_scores, 10), np.percentile(z_scores, 90))
    phi_wins = np.sum(z_wins ** 2) / (K - 1)
    
    # Entre-hospitales variance tau^2 (DerSimonian-Laird / Spiegelhalter)
    s_inv_e = np.sum(1.0 / comp_h['E'].to_numpy())
    tau2 = max(0, (np.sum(z_scores ** 2) - (K - 1)) / (np.sum(comp_h['E'].to_numpy()) - np.sum(comp_h['E'].to_numpy() ** 2) / np.sum(comp_h['E'].to_numpy())))
    
    print(f"Factor de sobredispersión de Poisson (phi crudo): {phi_crudo:.2f}")
    print(f"Factor de sobredispersión Winsorizado (Spiegelhalter): {phi_wins:.2f}")

    # Límites de control ajustados por sobredispersión:
    # SE_adj = sqrt(phi_wins / E_j)
    comp_h = comp_h.with_columns([
        # Poisson no ajustado
        (3.0 / pl.col('E').sqrt()).alias('lim_3s_pois'),
        # Ajustado por sobredispersión
        (3.0 * np.sqrt(phi_wins) / pl.col('E').sqrt()).alias('lim_3s_adj'),
        (3.0 * np.sqrt(phi_wins) / pl.col('E_nd0').sqrt()).alias('lim_3s_adj_nd0')
    ]).with_columns([
        pl.when(pl.col('OE') < (global_OE - pl.col('lim_3s_adj'))).then(pl.lit('SOBRESALIENTE'))
          .when(pl.col('OE') > (global_OE + pl.col('lim_3s_adj'))).then(pl.lit('ALERTA_MORTALIDAD'))
          .otherwise(pl.lit('PROMEDIO')).alias('clase_adj_base'),
        pl.when(pl.col('OE_nd0') < (global_OE - pl.col('lim_3s_adj_nd0'))).then(pl.lit('SOBRESALIENTE'))
          .when(pl.col('OE_nd0') > (global_OE + pl.col('lim_3s_adj_nd0'))).then(pl.lit('ALERTA_MORTALIDAD'))
          .otherwise(pl.lit('PROMEDIO')).alias('clase_adj_nd0'),
        # Crudo sin ajuste
        pl.when(pl.col('OE') < (global_OE - pl.col('lim_3s_pois'))).then(pl.lit('SOBRESALIENTE'))
          .when(pl.col('OE') > (global_OE + pl.col('lim_3s_pois'))).then(pl.lit('ALERTA_MORTALIDAD'))
          .otherwise(pl.lit('PROMEDIO')).alias('clase_pois_base')
    ])

    print("\nClasificación en embudo basal SIN ajuste de sobredispersión:")
    print(comp_h['clase_pois_base'].value_counts())

    print("\nClasificación en embudo basal CON ajuste de sobredispersión (Spiegelhalter):")
    print(comp_h['clase_adj_base'].value_counts())

    cambios_adj = comp_h.filter(pl.col('clase_adj_base') != pl.col('clase_adj_nd0'))
    print(f"\nHospitales que cambian de categoría con y sin día 0 BAJO LÍMITES AJUSTADOS: {len(cambios_adj)} de {K}")
    for r in cambios_adj.iter_rows(named=True):
        print(f"  - Hosp {r['COD_HOSPITAL']}: de {r['clase_adj_base']} a {r['clase_adj_nd0']} (OE base={r['OE']:.3f}, OE no-d0={r['OE_nd0']:.3f})")

    # -------------------------------------------------------------------------
    # TIPPING POINT ANALYSIS EN CENSURA
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("ANÁLISIS DE PUNTO DE INFLEXIÓN (TIPPING POINT) DE CENSURA")
    print("=" * 90)
    # Censurados en 2023 por hospital
    s23_all = pl.read_parquet(SILVER_DIR / "silver_2023.parquet")
    cens_h = s23_all.filter(
        pl.col('TIPOALTA').str.contains('DERIVACIÓN OTRO HOSPITAL|DERIVACIÓN INST. PRIVADA')
    ).group_by('COD_HOSPITAL').agg(pl.len().alias('N_cens'))

    comp_tipping = comp_h.join(cens_h, on='COD_HOSPITAL', how='left').with_columns(
        pl.col('N_cens').fill_null(0)
    ).with_columns([
        (pl.col('N_cens') / (pl.col('N') + pl.col('N_cens')) * 100).alias('tasa_censura')
    ])

    # Umbral de alerta: HSMR > 110 (o OE > global_OE * 1.10) o cruce de límite 3-sigma
    # Si derivados tienen mortalidad p_c: O_total = O + p_c * N_cens
    # E_total = E + E_cens. Para ser conservadores, E_cens estimado a la tasa media del hospital o de la red
    # p_c* tal que (O + p_c * N_cens) / E = global_OE * 1.10
    comp_tipping = comp_tipping.with_columns(
        pl.when(pl.col('N_cens') > 0).then(
            ((global_OE * 1.10 * pl.col('E')) - pl.col('O')) / pl.col('N_cens')
        ).otherwise(pl.lit(None)).alias('p_tipping_hsmr110')
    )

    print("Hospitales con mayor censura y su punto de inflexión p*:")
    top_cens = comp_tipping.sort('tasa_censura', descending=True).head(10)
    for r in top_cens.iter_rows(named=True):
        pt = r['p_tipping_hsmr110']
        pt_str = f"{pt*100:5.2f}%" if (pt is not None and pt >= 0) else "Ya en alerta / N/A"
        print(f"  Hosp {r['COD_HOSPITAL']}: N_eval={r['N']:5,}, N_cens={r['N_cens']:4,}, Censura={r['tasa_censura']:5.2f}%, OE actual={r['OE']:.3f} -> p* inflexión={pt_str}")

    # Cuántos hospitales cruzarían a alerta si sus derivados tuvieran p_c = 10% (mortalidad observada en transferencias críticas)
    tipping_10 = comp_tipping.filter((pl.col('clase_adj_base') != 'ALERTA_MORTALIDAD') & (pl.col('p_tipping_hsmr110') <= 0.10) & (pl.col('p_tipping_hsmr110') > 0))
    print(f"\nHospitales que pasarían a ALERTA si la mortalidad de derivados fuera del 10%: {len(tipping_10)} de {K}")

if __name__ == "__main__":
    main()
