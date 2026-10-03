import sys
from pathlib import Path
import polars as pl
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression, ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler, SplineTransformer

DATA_DEV_CACHE = Path("/home/felipe/Documentos/Proyecto final/Tesis/scratch/dev_eval_cache.parquet")
DATA_23_CACHE = Path("/home/felipe/Documentos/Proyecto final/Tesis/scratch/eval_23_cache.parquet")
SILVER_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/silver")
CAT_PATH = Path("/home/felipe/Documentos/Proyecto final/Tesis/config/catalogo_hospitales_procedencia.csv")

print("Cargando datos cacheados...")
df_dev = pl.read_parquet(DATA_DEV_CACHE)
df_23 = pl.read_parquet(DATA_23_CACHE)
cat = pl.read_csv(CAT_PATH)

# ==============================================================================
# 1. CRITERIO CUANTITATIVO DE MONOGRÁFICOS
# ==============================================================================
print("\n" + "=" * 90)
print("1. ANÁLISIS CUANTITATIVO DE CONCENTRACIÓN DE MDC (CRITERIO MONOGRÁFICO)")
print("=" * 90)

# Agrupar por hospital y MDC en 2023
hosp_mdc = df_23.group_by(['COD_HOSPITAL', 'MDC']).agg(pl.len().alias('n_mdc')).join(
    df_23.group_by('COD_HOSPITAL').agg(pl.len().alias('n_total')), on='COD_HOSPITAL'
).with_columns((pl.col('n_mdc') / pl.col('n_total') * 100).alias('pct_mdc'))

# Hospitales con max(pct_mdc) >= 40% o concentración en respiratorio+cardio (MDC 4+5)
top_mdc = hosp_mdc.sort('pct_mdc', descending=True).group_by('COD_HOSPITAL').first()
resp_cardio = hosp_mdc.filter(pl.col('MDC').is_in([4, 5])).group_by('COD_HOSPITAL').agg(
    pl.col('pct_mdc').sum().alias('pct_resp_cardio')
)

hosp_specialty = df_23.group_by('COD_HOSPITAL').agg(pl.len().alias('N_total')).join(
    top_mdc.select(['COD_HOSPITAL', 'MDC', 'pct_mdc']), on='COD_HOSPITAL'
).join(
    resp_cardio, on='COD_HOSPITAL', how='left'
).with_columns(pl.col('pct_resp_cardio').fill_null(0.0)).join(
    cat.select(['cod_hospital', 'nombre_oficial']), left_on='COD_HOSPITAL', right_on=pl.col('cod_hospital').cast(pl.String), how='left'
)

print(f"{'Cod':<7} | {'Nombre':<32} | {'Max MDC':<8} | {'% Max MDC':>10} | {'% MDC 4+5 (Cardio-Resp)':>24} | {'Clasificación Criterio':<22}")
print("-" * 115)
for r in hosp_specialty.sort('pct_mdc', descending=True).iter_rows(named=True):
    es_mono = (r['pct_mdc'] >= 50.0) or (r['pct_resp_cardio'] >= 70.0)
    clase = "MONOGRÁFICO" if es_mono else "AGUDO GENERAL"
    if es_mono or r['COD_HOSPITAL'] in ['106102', '110110', '112103', '114105', '113180']:
        print(f"{r['COD_HOSPITAL']:<7} | {r['nombre_oficial'][:32]:<32} | MDC {r['MDC']:<4} | {r['pct_mdc']:9.1f}% | {r['pct_resp_cardio']:23.1f}% | {clase:<22}")

# ==============================================================================
# 2. MODELOS M2 Y VARIANTES DE M3
# ==============================================================================
print("\n" + "=" * 90)
print("2. ENTRENAMIENTO DE M2 Y VARIANTES DE M3 (CALIBRACIÓN Y TAU)")
print("=" * 90)

h_ndx_dev = df_dev.group_by('COD_HOSPITAL').agg([
    pl.col('N_DX').mean().alias('mean_ndx'),
    pl.col('N_DX').std().alias('std_ndx')
])
g_mean_ndx = df_dev['N_DX'].mean()
g_std_ndx = df_dev['N_DX'].std()

df_dev = df_dev.join(h_ndx_dev, on='COD_HOSPITAL', how='left').with_columns([
    (pl.col('N_DX') / pl.col('mean_ndx').fill_null(g_mean_ndx)).alias('R_DX'),
    ((pl.col('N_DX') - pl.col('mean_ndx').fill_null(g_mean_ndx)) / pl.col('std_ndx').fill_null(g_std_ndx)).alias('Z_NDX')
])
df_23 = df_23.join(h_ndx_dev, on='COD_HOSPITAL', how='left').with_columns([
    (pl.col('N_DX') / pl.col('mean_ndx').fill_null(g_mean_ndx)).alias('R_DX'),
    ((pl.col('N_DX') - pl.col('mean_ndx').fill_null(g_mean_ndx)) / pl.col('std_ndx').fill_null(g_std_ndx)).alias('Z_NDX')
])

elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]
elix_31 = [f'ELIX_{i:02d}' for i in range(1, 32)]
f_base = ['EDAD_ANIOS', 'INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL', 'SEXO_MASCULINO']

sc28 = StandardScaler()
X_dev_28 = sc28.fit_transform(df_dev.select(f_base + elix_28).fill_null(0).to_pandas().values)
X_23_28 = sc28.transform(df_23.select(f_base + elix_28).fill_null(0).to_pandas().values)

sc31 = StandardScaler()
X_dev_31 = sc31.fit_transform(df_dev.select(f_base + elix_31).fill_null(0).to_pandas().values)
X_23_31 = sc31.transform(df_23.select(f_base + elix_31).fill_null(0).to_pandas().values)

sp = SplineTransformer(n_knots=4, degree=3, include_bias=False)
sp_dev = sp.fit_transform(df_dev.select(['R_DX']).to_pandas().values)
sp_23 = sp.transform(df_23.select(['R_DX']).to_pandas().values)

y_dev = df_dev['MORTALIDAD_BINARIA'].to_numpy()

# Ajustes
print("Ajustando modelos...")
m2 = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_28, y_dev)
p_m2 = m2.predict_proba(X_23_28)[:, 1]

X_dev_m3a = np.hstack([X_dev_28, sp_dev])
X_23_m3a = np.hstack([X_23_28, sp_23])
m3a = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_m3a, y_dev)
p_m3a = m3a.predict_proba(X_23_m3a)[:, 1]

X_dev_m3b = np.hstack([X_dev_28, df_dev.select(['R_DX']).to_pandas().values])
X_23_m3b = np.hstack([X_23_28, df_23.select(['R_DX']).to_pandas().values])
m3b = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_m3b, y_dev)
p_m3b = m3b.predict_proba(X_23_m3b)[:, 1]

X_dev_m3c = np.hstack([X_dev_28, df_dev.select(['Z_NDX']).to_pandas().values])
X_23_m3c = np.hstack([X_23_28, df_23.select(['Z_NDX']).to_pandas().values])
m3c = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_m3c, y_dev)
p_m3c = m3c.predict_proba(X_23_m3c)[:, 1]

X_dev_m3d = np.hstack([X_dev_31, sp_dev])
X_23_m3d = np.hstack([X_23_31, sp_23])
m3d = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_m3d, y_dev)
p_m3d = m3d.predict_proba(X_23_m3d)[:, 1]

df_23 = df_23.with_columns([
    pl.Series('p_m2', p_m2),
    pl.Series('p_m3a', p_m3a),
    pl.Series('p_m3b', p_m3b),
    pl.Series('p_m3c', p_m3c),
    pl.Series('p_m3d', p_m3d)
])

def run_eb_eval(df_in, p_col, margin=1.10):
    o_tot = df_in['MORTALIDAD_BINARIA'].sum()
    e_tot = df_in[p_col].sum()
    k_recal = o_tot / e_tot
    
    h = df_in.group_by('COD_HOSPITAL').agg([
        pl.len().alias('N'),
        pl.col('MORTALIDAD_BINARIA').sum().alias('O'),
        pl.col(p_col).sum().alias('E_raw')
    ]).with_columns([
        (pl.col('E_raw') * k_recal).alias('E'),
        (pl.col('O') / (pl.col('E_raw') * k_recal)).alias('OE')
    ])
    
    K = len(h)
    log_oe = np.log(h['OE'].to_numpy())
    v_w = 1.0 / h['O'].to_numpy()
    w_f = 1.0 / v_w
    mu_meta = np.sum(w_f * log_oe) / np.sum(w_f)
    Q = np.sum(w_f * (log_oe - mu_meta)**2)
    tau2 = max(0.0, (Q - (K - 1)) / (np.sum(w_f) - np.sum(w_f**2) / np.sum(w_f)))
    tau = np.sqrt(tau2)
    B_j = tau2 / (tau2 + v_w)
    log_th_post = B_j * log_oe + (1.0 - B_j) * mu_meta
    se_post = np.sqrt(tau2 * v_w / (tau2 + v_w))
    
    z_100 = (log_th_post - np.log(1.00)) / se_post
    p_100 = 1.0 - stats.norm.cdf(-z_100)
    
    z_110 = (log_th_post - np.log(margin)) / se_post
    p_110 = 1.0 - stats.norm.cdf(-z_110)
    
    z_120 = (log_th_post - np.log(1.20)) / se_post
    p_120 = 1.0 - stats.norm.cdf(-z_120)
    
    h = h.with_columns([
        pl.Series('log_th_post', log_th_post),
        pl.Series('se_post', se_post),
        pl.Series('P_100', p_100),
        pl.Series('P_110', p_110),
        pl.Series('P_120', p_120),
        (pl.Series('P_110', p_110) >= 0.95).alias('alerta_110'),
        (pl.Series('P_120', p_120) >= 0.95).alias('alerta_120')
    ])
    return tau, mu_meta, h

tau_m2, mu_m2, h_m2 = run_eb_eval(df_23, 'p_m2')
tau_m3a, mu_m3a, h_m3a = run_eb_eval(df_23, 'p_m3a')
tau_m3b, mu_m3b, h_m3b = run_eb_eval(df_23, 'p_m3b')
tau_m3c, mu_m3c, h_m3c = run_eb_eval(df_23, 'p_m3c')
tau_m3d, mu_m3d, h_m3d = run_eb_eval(df_23, 'p_m3d')

# Sin día 0
df_23_nd0 = df_23.filter(~((pl.col('MORTALIDAD_BINARIA') == 1) & (pl.col('ESTANCIA_DIAS') == 0)))
tau_nd0, mu_nd0, h_nd0 = run_eb_eval(df_23_nd0, 'p_m3a')

print("\nPARÁMETROS DE DISPERSIÓN EMPÍRICA (TAU) POR MODELO:")
print(f"  M2 (Clínico Base):           tau = {tau_m2:.4f} (tau^2 = {tau_m2**2:.4f}, mu = {mu_m2:.4f})")
print(f"  M3a (Primario R_dx Splines):  tau = {tau_m3a:.4f} (tau^2 = {tau_m3a**2:.4f}, mu = {mu_m3a:.4f})")
print(f"  M3b (Lineal R_dx):            tau = {tau_m3b:.4f} (tau^2 = {tau_m3b**2:.4f}, mu = {mu_m3b:.4f})")
print(f"  M3c (Centrado Z_dx):          tau = {tau_m3c:.4f} (tau^2 = {tau_m3c**2:.4f}, mu = {mu_m3c:.4f})")
print(f"  M3d (Splines 31 comorb):      tau = {tau_m3d:.4f} (tau^2 = {tau_m3d**2:.4f}, mu = {mu_m3d:.4f})")
print(f"  M3a Sin Día 0:                tau = {tau_nd0:.4f} (tau^2 = {tau_nd0**2:.4f}, mu = {mu_nd0:.4f})")

# Unir matriz de sensibilidad completa
m_sens = h_m3a.select(['COD_HOSPITAL', 'O', 'E', 'OE', 'P_110', 'P_120', 'alerta_110']).rename({
    'E': 'E_M3a', 'OE': 'OE_M3a', 'P_110': 'P110_M3a', 'P_120': 'P120_M3a', 'alerta_110': 'Alt_M3a'
}).join(
    h_m3b.select(['COD_HOSPITAL', 'OE', 'P_110', 'alerta_110']).rename({'OE': 'OE_M3b', 'P_110': 'P110_M3b', 'alerta_110': 'Alt_M3b'}), on='COD_HOSPITAL'
).join(
    h_m3c.select(['COD_HOSPITAL', 'OE', 'P_110', 'alerta_110']).rename({'OE': 'OE_M3c', 'P_110': 'P110_M3c', 'alerta_110': 'Alt_M3c'}), on='COD_HOSPITAL'
).join(
    h_m3d.select(['COD_HOSPITAL', 'OE', 'P_110', 'alerta_110']).rename({'OE': 'OE_M3d', 'P_110': 'P110_M3d', 'alerta_110': 'Alt_M3d'}), on='COD_HOSPITAL'
).join(
    h_nd0.select(['COD_HOSPITAL', 'OE', 'P_110', 'alerta_110']).rename({'OE': 'OE_nd0', 'P_110': 'P110_nd0', 'alerta_110': 'Alt_nd0'}), on='COD_HOSPITAL'
).join(
    h_m2.select(['COD_HOSPITAL', 'OE', 'P_110', 'alerta_110']).rename({'OE': 'OE_M2', 'P_110': 'P110_M2', 'alerta_110': 'Alt_M2'}), on='COD_HOSPITAL'
).join(
    cat.select(['cod_hospital', 'nombre_oficial']), left_on='COD_HOSPITAL', right_on=pl.col('cod_hospital').cast(pl.String), how='left'
)

# Filtro de hospitales con alerta en al menos un modelo
m_sens_alert = m_sens.filter(
    pl.col('Alt_M3a') | pl.col('Alt_M3b') | pl.col('Alt_M3c') | pl.col('Alt_M3d') | pl.col('Alt_nd0') | pl.col('Alt_M2')
).sort('OE_M3a', descending=True)

print("\n" + "=" * 120)
print("MATRIZ COMPLETA DE SENSIBILIDAD DE ALERTAS (M2 vs M3a, M3b, M3c, M3d, Sin Día 0)")
print("=" * 120)
print(f"{'Cod':<7} | {'Nombre':<30} | {'M3a Prim':>9} | {'M3b Lin':>8} | {'M3c Zdx':>8} | {'M3d 31c':>8} | {'Sin Día0':>9} | {'M2 Clin':>8} | {'Consist M3':<11} | {'Robusta Total':<13}")
print("-" * 120)

for r in m_sens_alert.iter_rows(named=True):
    n_m3 = sum([r['Alt_M3a'], r['Alt_M3b'], r['Alt_M3c'], r['Alt_M3d'], r['Alt_nd0']])
    cons_m3 = f"5/5 (100%)" if n_m3 == 5 else f"{n_m3}/5"
    rob_tot = "SÍ (M2+M3)" if (r['Alt_M3a'] and r['Alt_M2']) else "NO"
    
    s_m3a = f"{r['OE_M3a']:.2f}*" if r['Alt_M3a'] else f"{r['OE_M3a']:.2f}"
    s_m3b = f"{r['OE_M3b']:.2f}*" if r['Alt_M3b'] else f"{r['OE_M3b']:.2f}"
    s_m3c = f"{r['OE_M3c']:.2f}*" if r['Alt_M3c'] else f"{r['OE_M3c']:.2f}"
    s_m3d = f"{r['OE_M3d']:.2f}*" if r['Alt_M3d'] else f"{r['OE_M3d']:.2f}"
    s_nd0 = f"{r['OE_nd0']:.2f}*" if r['Alt_nd0'] else f"{r['OE_nd0']:.2f}"
    s_m2 = f"{r['OE_M2']:.2f}*" if r['Alt_M2'] else f"{r['OE_M2']:.2f}"
    
    print(f"{r['COD_HOSPITAL']:<7} | {r['nombre_oficial'][:30]:<30} | {s_m3a:>9} | {s_m3b:>8} | {s_m3c:>8} | {s_m3d:>8} | {s_nd0:>9} | {s_m2:>8} | {cons_m3:<11} | {rob_tot:<13}")

