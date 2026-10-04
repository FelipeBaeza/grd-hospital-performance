import polars as pl
import numpy as np
import scipy.stats as stats
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SplineTransformer

df_dev = pl.read_parquet('/home/felipe/Documentos/Proyecto final/Tesis/scratch/dev_eval_cache.parquet')
df_23 = pl.read_parquet('/home/felipe/Documentos/Proyecto final/Tesis/scratch/eval_23_cache.parquet')
cat = pl.read_csv('/home/felipe/Documentos/Proyecto final/Tesis/config/catalogo_hospitales_procedencia.csv').with_columns(pl.col('cod_hospital').cast(pl.String))

elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]
f_base = ['EDAD_ANIOS', 'INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL', 'SEXO_MASCULINO']

mdcs_dev = df_dev['MDC'].to_pandas().values.reshape(-1, 1)
mdcs_23 = df_23['MDC'].to_pandas().values.reshape(-1, 1)
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
X_mdc_dev = ohe.fit_transform(mdcs_dev)
X_mdc_23 = ohe.transform(mdcs_23)

sc = StandardScaler()
X_dev_base = sc.fit_transform(df_dev.select(f_base + elix_28).fill_null(0).to_pandas().values)
X_23_base = sc.transform(df_23.select(f_base + elix_28).fill_null(0).to_pandas().values)

X_dev_m2mdc = np.hstack([X_dev_base, X_mdc_dev])
X_23_m2mdc = np.hstack([X_23_base, X_mdc_23])

# R_DX total for M3
h_dev_stats = df_dev.group_by('COD_HOSPITAL').agg(pl.col('N_DX').mean().alias('mean_ndx_tot'))
g_mean_tot = df_dev['N_DX'].mean()
df_dev = df_dev.join(h_dev_stats, on='COD_HOSPITAL', how='left').with_columns(
    (pl.col('N_DX') / pl.col('mean_ndx_tot').fill_null(g_mean_tot)).alias('R_DX_TOT')
)
df_23 = df_23.join(h_dev_stats, on='COD_HOSPITAL', how='left').with_columns(
    (pl.col('N_DX') / pl.col('mean_ndx_tot').fill_null(g_mean_tot)).alias('R_DX_TOT')
)

sp_tot = SplineTransformer(n_knots=4, degree=3, include_bias=False)
sp_dev_tot = sp_tot.fit_transform(df_dev.select(['R_DX_TOT']).to_pandas().values)
sp_23_tot = sp_tot.transform(df_23.select(['R_DX_TOT']).to_pandas().values)

X_dev_m3 = np.hstack([X_dev_base, sp_dev_tot])
X_23_m3 = np.hstack([X_23_base, sp_23_tot])

y_dev = df_dev['MORTALIDAD_BINARIA'].to_numpy()
o_tot = df_23['MORTALIDAD_BINARIA'].sum()

m2_mdc = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_m2mdc, y_dev)
m2_base = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_base, y_dev)
m3_tot = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_m3, y_dev)

p_m2mdc = m2_mdc.predict_proba(X_23_m2mdc)[:, 1]
p_m2base = m2_base.predict_proba(X_23_base)[:, 1]
p_m3tot = m3_tot.predict_proba(X_23_m3)[:, 1]

df_23 = df_23.with_columns([
    pl.Series('E_M2MDC', p_m2mdc * (o_tot / p_m2mdc.sum())),
    pl.Series('E_M2', p_m2base * (o_tot / p_m2base.sum())),
    pl.Series('E_M3', p_m3tot * (o_tot / p_m3tot.sum()))
])

# Hospital metrics
h_res = df_23.group_by('COD_HOSPITAL').agg([
    pl.len().alias('N'),
    pl.col('MORTALIDAD_BINARIA').sum().alias('O'),
    pl.col('E_M2MDC').sum().alias('E_m2mdc'),
    pl.col('E_M2').sum().alias('E_m2'),
    pl.col('E_M3').sum().alias('E_m3'),
    pl.col('N_DX').mean().alias('mean_ndx'),
    (pl.col('N_DX') >= 11).mean().alias('pct_ndx_ge11')
]).with_columns([
    (pl.col('O') / pl.col('E_m2mdc')).alias('OE_m2mdc'),
    (pl.col('O') / pl.col('E_m2')).alias('OE_m2'),
    (pl.col('O') / pl.col('E_m3')).alias('OE_m3')
]).join(cat.select(['cod_hospital', 'nombre_oficial', 'es_monografico']), left_on='COD_HOSPITAL', right_on='cod_hospital', how='left')

# Tau y mu específicos
tau_m2mdc, mu_m2mdc = 0.1948, -0.0359
tau_m2, mu_m2 = 0.2077, -0.0267
tau_m3, mu_m3 = 0.2041, -0.0347

c = np.log(1.10)

def calc_p(oe_arr, o_arr, tau_val, mu_val):
    y = np.log(oe_arr)
    v = 1.0 / o_arr
    B = (tau_val**2) / (tau_val**2 + v)
    theta = B * y + (1 - B) * mu_val
    se = np.sqrt(B * v)
    z = (theta - c) / se
    return stats.norm.cdf(z)

h_res = h_res.with_columns([
    pl.Series('P_m2mdc', calc_p(h_res['OE_m2mdc'].to_numpy(), h_res['O'].to_numpy(), tau_m2mdc, mu_m2mdc)),
    pl.Series('P_m2', calc_p(h_res['OE_m2'].to_numpy(), h_res['O'].to_numpy(), tau_m2, mu_m2)),
    pl.Series('P_m3', calc_p(h_res['OE_m3'].to_numpy(), h_res['O'].to_numpy(), tau_m3, mu_m3))
])

# General acute hospitals only
h_ag = h_res.filter(pl.col('es_monografico') == 'NO')

print(f"=== ALERTAS EN MODELO PRIMARIO M2+MDC (P >= 0.95 en Agudos Generales) ===")
alerts_prim = h_ag.filter(pl.col('P_m2mdc') >= 0.95).sort('OE_m2mdc', descending=True)
print(f"Total Alertas Primarias en Agudos Generales: {alerts_prim.height}")
for r in alerts_prim.iter_rows(named=True):
    is_rob = "SÍ (ROBUSTA)" if r['P_m3'] >= 0.95 else "NO (Solo Primaria)"
    print(f"  {r['COD_HOSPITAL']} | {r['nombre_oficial'][:32]:<32} | O={r['O']:4d} | OE_prim={r['OE_m2mdc']:.3f} (P={r['P_m2mdc']:.3f}) | OE_m3={r['OE_m3']:.3f} (P={r['P_m3']:.3f}) | {is_rob}")

print(f"\n=== ALERTAS EN M2 BASE (P >= 0.95 en Agudos Generales) ===")
alerts_m2 = h_ag.filter(pl.col('P_m2') >= 0.95).sort('OE_m2', descending=True)
print(f"Total Alertas M2 Base: {alerts_m2.height}")
for r in alerts_m2.iter_rows(named=True):
    print(f"  {r['COD_HOSPITAL']} | {r['nombre_oficial'][:32]:<32} | O={r['O']:4d} | OE_m2={r['OE_m2']:.3f} (P={r['P_m2']:.3f})")

print(f"\n=== ALERTAS EN M3-TOTAL (P >= 0.95 en Agudos Generales) ===")
alerts_m3 = h_ag.filter(pl.col('P_m3') >= 0.95).sort('OE_m3', descending=True)
print(f"Total Alertas M3-Total: {alerts_m3.height}")
for r in alerts_m3.iter_rows(named=True):
    print(f"  {r['COD_HOSPITAL']} | {r['nombre_oficial'][:32]:<32} | O={r['O']:4d} | OE_m3={r['OE_m3']:.3f} (P={r['P_m3']:.3f})")

print(f"\n=== COMPORTAMIENTO DE CENTROS CRÍTICOS Y MONOGRÁFICOS ===")
special_ids = ['112103', '106102', '114105', '111100', '108100', '113180', '106103', '106100', '112101']
for r in h_res.filter(pl.col('COD_HOSPITAL').is_in(special_ids)).sort('COD_HOSPITAL').iter_rows(named=True):
    print(f"{r['COD_HOSPITAL']} | {r['nombre_oficial'][:30]:<30} | O={r['O']:4d} | M2+MDC: {r['OE_m2mdc']:.3f} (P={r['P_m2mdc']:.3f}) | M2: {r['OE_m2']:.3f} (P={r['P_m2']:.3f}) | M3: {r['OE_m3']:.3f} (P={r['P_m3']:.3f})")

# Asociación entre O/E primario y N_dx
r_mean, p_mean = stats.spearmanr(h_ag['OE_m2mdc'], h_ag['mean_ndx'])
r_pct, p_pct = stats.spearmanr(h_ag['OE_m2mdc'], h_ag['pct_ndx_ge11'])
print(f"\n=== ASOCIACIÓN ENTRE O/E PRIMARIO (M2+MDC) Y N_DX POR HOSPITAL ===")
print(f"Correlación con N_dx medio: rho = {r_mean:+.3f} (p = {p_mean:.3f})")
print(f"Correlación con % pacientes >= 11 dx: rho = {r_pct:+.3f} (p = {p_pct:.3f})")

