import polars as pl
import numpy as np
import scipy.stats as stats
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, SplineTransformer

DATA_DEV_CACHE = Path("/home/felipe/Documentos/Proyecto final/Tesis/scratch/dev_eval_cache.parquet")
DATA_23_CACHE = Path("/home/felipe/Documentos/Proyecto final/Tesis/scratch/eval_23_cache.parquet")
CAT_PATH = Path("/home/felipe/Documentos/Proyecto final/Tesis/config/catalogo_hospitales_procedencia.csv")

df_dev = pl.read_parquet(DATA_DEV_CACHE)
df_23 = pl.read_parquet(DATA_23_CACHE)
cat = pl.read_csv(CAT_PATH)

# 1. Proporciones de MDC para TODOS los hospitales en 2023
print("=== 1. DISTRIBUCIÓN DE CONCENTRACIÓN DE MDC (TODOS LOS 65 HOSPITALES) ===")
hosp_mdc = df_23.group_by(['COD_HOSPITAL', 'MDC']).agg(pl.len().alias('n_mdc')).join(
    df_23.group_by('COD_HOSPITAL').agg(pl.len().alias('n_total')), on='COD_HOSPITAL'
).with_columns((pl.col('n_mdc') / pl.col('n_total') * 100).alias('pct_mdc'))

top_mdc = hosp_mdc.sort('pct_mdc', descending=True).group_by('COD_HOSPITAL').first()
resp_cardio = hosp_mdc.filter(pl.col('MDC').is_in([4, 5])).group_by('COD_HOSPITAL').agg(
    pl.col('pct_mdc').sum().alias('pct_resp_cardio')
)

all_hosp_spec = df_23.group_by('COD_HOSPITAL').agg(pl.len().alias('N_total')).join(
    top_mdc.select(['COD_HOSPITAL', 'MDC', 'pct_mdc']), on='COD_HOSPITAL'
).join(
    resp_cardio, on='COD_HOSPITAL', how='left'
).with_columns(pl.col('pct_resp_cardio').fill_null(0.0)).join(
    cat.select(['cod_hospital', 'nombre_oficial']), left_on='COD_HOSPITAL', right_on=pl.col('cod_hospital').cast(pl.String), how='left'
).sort('pct_mdc', descending=True)

print(f"{'Cod':<7} | {'Nombre':<32} | {'Max MDC':<8} | {'% Max':>6} | {'% Resp+Card (4+5)':>17} | {'Mono?':<5}")
print("-" * 85)
for r in all_hosp_spec.iter_rows(named=True):
    is_mono = (r['pct_mdc'] >= 50.0) or (r['pct_resp_cardio'] >= 70.0)
    mono_str = "SÍ" if is_mono else "no"
    print(f"{r['COD_HOSPITAL']:<7} | {r['nombre_oficial'][:32]:<32} | MDC {r['MDC']:<4} | {r['pct_mdc']:5.1f}% | {r['pct_resp_cardio']:16.1f}% | {mono_str:<5}")

# Monográficos identificados por la regla
mono_hospitals = all_hosp_spec.filter((pl.col('pct_mdc') >= 50.0) | (pl.col('pct_resp_cardio') >= 70.0))['COD_HOSPITAL'].to_list()
print(f"\nHospitales monográficos identificados (N = {len(mono_hospitals)}): {mono_hospitals}")

# 2. Ajuste de M3a
h_ndx_dev = df_dev.group_by('COD_HOSPITAL').agg(pl.col('N_DX').mean().alias('mean_ndx'))
g_mean_ndx = df_dev['N_DX'].mean()
df_dev = df_dev.join(h_ndx_dev, on='COD_HOSPITAL', how='left').with_columns(
    (pl.col('N_DX') / pl.col('mean_ndx').fill_null(g_mean_ndx)).alias('R_DX')
)
df_23 = df_23.join(h_ndx_dev, on='COD_HOSPITAL', how='left').with_columns(
    (pl.col('N_DX') / pl.col('mean_ndx').fill_null(g_mean_ndx)).alias('R_DX')
)

elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]
f_base = ['EDAD_ANIOS', 'INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL', 'SEXO_MASCULINO']

sc28 = StandardScaler()
X_dev_28 = sc28.fit_transform(df_dev.select(f_base + elix_28).fill_null(0).to_pandas().values)
X_23_28 = sc28.transform(df_23.select(f_base + elix_28).fill_null(0).to_pandas().values)

sp = SplineTransformer(n_knots=4, degree=3, include_bias=False)
sp_dev = sp.fit_transform(df_dev.select(['R_DX']).to_pandas().values)
sp_23 = sp.transform(df_23.select(['R_DX']).to_pandas().values)

y_dev = df_dev['MORTALIDAD_BINARIA'].to_numpy()

X_dev_m3a = np.hstack([X_dev_28, sp_dev])
X_23_m3a = np.hstack([X_23_28, sp_23])
m3a = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_m3a, y_dev)
p_m3a = m3a.predict_proba(X_23_m3a)[:, 1]

df_23 = df_23.with_columns(pl.Series('P3', p_m3a))

# Recalibración
k_tot = df_23['MORTALIDAD_BINARIA'].sum() / df_23['P3'].sum()
df_23 = df_23.with_columns((pl.col('P3') * k_tot).alias('E3'))

h_all = df_23.group_by('COD_HOSPITAL').agg([
    pl.len().alias('N'),
    pl.col('MORTALIDAD_BINARIA').sum().alias('O'),
    pl.col('E3').sum().alias('E')
]).with_columns((pl.col('O') / pl.col('E')).alias('OE'))

# Función EB
def run_eb(h_sub):
    K = len(h_sub)
    log_oe = np.log(h_sub['OE'].to_numpy())
    v_w = 1.0 / h_sub['O'].to_numpy()
    w_f = 1.0 / v_w
    mu_m = np.sum(w_f * log_oe) / np.sum(w_f)
    Q = np.sum(w_f * (log_oe - mu_m)**2)
    tau2 = max(0.0, (Q - (K - 1)) / (np.sum(w_f) - np.sum(w_f**2) / np.sum(w_f)))
    tau = np.sqrt(tau2)
    B_j = tau2 / (tau2 + v_w)
    log_th_post = B_j * log_oe + (1.0 - B_j) * mu_m
    se_post = np.sqrt(tau2 * v_w / (tau2 + v_w))
    z_110 = (log_th_post - np.log(1.10)) / se_post
    p_110 = 1.0 - stats.norm.cdf(-z_110)
    z_120 = (log_th_post - np.log(1.20)) / se_post
    p_120 = 1.0 - stats.norm.cdf(-z_120)
    
    return tau, mu_m, h_sub.with_columns([
        pl.Series('log_th_post', log_th_post),
        pl.Series('se_post', se_post),
        pl.Series('P_110', p_110),
        pl.Series('P_120', p_120),
        (pl.Series('P_110', p_110) >= 0.95).alias('alerta_110')
    ])

tau_all, mu_all, h_eb_all = run_eb(h_all)

# Re-estimar EB SOLO CON AGUDOS GENERALES (excluyendo mono_hospitals)
h_agudos = h_all.filter(~pl.col('COD_HOSPITAL').is_in(mono_hospitals))
tau_agudos, mu_agudos, h_eb_agudos = run_eb(h_agudos)

print("\n" + "=" * 90)
print("=== 2. IMPACTO DE EXCLUIR CENTROS MONOGRÁFICOS EN PARÁMETROS EMPÍRICOS BAYESIANOS ===")
print("=" * 90)
print(f"Red Completa (65 hospitales):         tau = {tau_all:.4f} (tau^2 = {tau_all**2:.4f}), mu_meta = {mu_all:.4f}")
print(f"Solo Agudos Generales (62 hospitales): tau = {tau_agudos:.4f} (tau^2 = {tau_agudos**2:.4f}), mu_meta = {mu_agudos:.4f}")
print(f"Diferencia producida por monográficos: Delta tau = {tau_all - tau_agudos:+.4f}, Delta mu = {mu_all - mu_agudos:+.4f}")

# Comparar alertas en Agudos Generales
comp = h_eb_agudos.select(['COD_HOSPITAL', 'O', 'E', 'OE', 'se_post', 'P_110', 'alerta_110']).join(
    h_eb_all.select(['COD_HOSPITAL', 'P_110', 'alerta_110']).rename({'P_110': 'P_110_con_mono', 'alerta_110': 'alt_con_mono'}),
    on='COD_HOSPITAL'
).join(
    cat.select(['cod_hospital', 'nombre_oficial']), left_on='COD_HOSPITAL', right_on=pl.col('cod_hospital').cast(pl.String), how='left'
).sort('OE', descending=True)

print("\nComparación de Alertas en Hospitales Agudos Generales (Con vs Sin Monográficos en EB):")
print(f"{'Cod':<7} | {'Nombre':<32} | {'O':>5} | {'E':>6} | {'OE':>6} | {'SE_post':>7} | {'P_110 (Sin)':>11} | {'P_110 (Con)':>11} | {'Alerta Sin':<10}")
print("-" * 115)
for r in comp.filter(pl.col('alerta_110') | pl.col('alt_con_mono') | (pl.col('OE') >= 1.15)).iter_rows(named=True):
    alt_s = "ALERTA" if r['alerta_110'] else "prom"
    print(f"{r['COD_HOSPITAL']:<7} | {r['nombre_oficial'][:32]:<32} | {r['O']:5d} | {r['E']:6.1f} | {r['OE']:6.3f} | {r['se_post']:7.4f} | {r['P_110']:11.4f} | {r['P_110_con_mono']:11.4f} | {alt_s:<10}")

