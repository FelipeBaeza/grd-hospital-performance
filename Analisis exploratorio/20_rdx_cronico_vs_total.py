import polars as pl
import numpy as np
import scipy.stats as stats
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, SplineTransformer

df_dev = pl.read_parquet("/home/felipe/Documentos/Proyecto final/Tesis/scratch/dev_eval_cache.parquet")
df_23 = pl.read_parquet("/home/felipe/Documentos/Proyecto final/Tesis/scratch/eval_23_cache.parquet")
CAT_PATH = Path("/home/felipe/Documentos/Proyecto final/Tesis/config/catalogo_hospitales_procedencia.csv")
cat = pl.read_csv(CAT_PATH)

elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]
f_base = ['EDAD_ANIOS', 'INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL', 'SEXO_MASCULINO']

# 1. N_crónico = suma de comorbilidades crónicas (Elixhauser 28)
n_chron_expr = pl.sum_horizontal(elix_28)
df_dev = df_dev.with_columns(n_chron_expr.alias('N_CHRONIC'))
df_23 = df_23.with_columns(n_chron_expr.alias('N_CHRONIC'))

# R_DX total vs R_DX crónico
h_dev_stats = df_dev.group_by('COD_HOSPITAL').agg([
    pl.col('N_DX').mean().alias('mean_ndx_tot'),
    pl.col('N_CHRONIC').mean().alias('mean_ndx_chron')
])
g_mean_tot = df_dev['N_DX'].mean()
g_mean_chron = df_dev['N_CHRONIC'].mean()

df_dev = df_dev.join(h_dev_stats, on='COD_HOSPITAL', how='left').with_columns([
    (pl.col('N_DX') / pl.col('mean_ndx_tot').fill_null(g_mean_tot)).alias('R_DX_TOT'),
    (pl.col('N_CHRONIC') / pl.col('mean_ndx_chron').fill_null(g_mean_chron)).alias('R_DX_CHRON')
])

df_23 = df_23.join(h_dev_stats, on='COD_HOSPITAL', how='left').with_columns([
    (pl.col('N_DX') / pl.col('mean_ndx_tot').fill_null(g_mean_tot)).alias('R_DX_TOT'),
    (pl.col('N_CHRONIC') / pl.col('mean_ndx_chron').fill_null(g_mean_chron)).alias('R_DX_CHRON')
])

# Preparar matrices
sc = StandardScaler()
X_dev_base = sc.fit_transform(df_dev.select(f_base + elix_28).fill_null(0).to_pandas().values)
X_23_base = sc.transform(df_23.select(f_base + elix_28).fill_null(0).to_pandas().values)

# Splines para total y crónico
sp_tot = SplineTransformer(n_knots=4, degree=3, include_bias=False)
sp_dev_tot = sp_tot.fit_transform(df_dev.select(['R_DX_TOT']).to_pandas().values)
sp_23_tot = sp_tot.transform(df_23.select(['R_DX_TOT']).to_pandas().values)

sp_chr = SplineTransformer(n_knots=4, degree=3, include_bias=False)
sp_dev_chr = sp_chr.fit_transform(df_dev.select(['R_DX_CHRON']).to_pandas().values)
sp_23_chr = sp_chr.transform(df_23.select(['R_DX_CHRON']).to_pandas().values)

y_dev = df_dev['MORTALIDAD_BINARIA'].to_numpy()

# Ajustar los 3 modelos
print("Ajustando modelos...")
# 1. M2 (Sin R_dx)
m2 = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_base, y_dev)
p_m2 = m2.predict_proba(X_23_base)[:, 1]

# 2. M3-Total (R_dx sobre diagnósticos totales)
m3_tot = LogisticRegression(max_iter=200, random_state=42).fit(np.hstack([X_dev_base, sp_dev_tot]), y_dev)
p_m3_tot = m3_tot.predict_proba(np.hstack([X_23_base, sp_23_tot]))[:, 1]

# 3. M3-Crónico (R_dx sobre comorbilidades crónicas)
m3_chr = LogisticRegression(max_iter=200, random_state=42).fit(np.hstack([X_dev_base, sp_dev_chr]), y_dev)
p_m3_chr = m3_chr.predict_proba(np.hstack([X_23_base, sp_23_chr]))[:, 1]

df_23 = df_23.with_columns([
    pl.Series('p_m2', p_m2),
    pl.Series('p_m3_tot', p_m3_tot),
    pl.Series('p_m3_chr', p_m3_chr)
])

# Recalibración
k_m2 = df_23['MORTALIDAD_BINARIA'].sum() / df_23['p_m2'].sum()
k_tot = df_23['MORTALIDAD_BINARIA'].sum() / df_23['p_m3_tot'].sum()
k_chr = df_23['MORTALIDAD_BINARIA'].sum() / df_23['p_m3_chr'].sum()

# Agregados por hospital (solo agudos generales, excluyendo monográficos 110110, 112104, 112103)
mono_codes = ['110110', '112104', '112103']
h = df_23.filter(~pl.col('COD_HOSPITAL').is_in(mono_codes)).group_by('COD_HOSPITAL').agg([
    pl.len().alias('N'),
    pl.col('MORTALIDAD_BINARIA').sum().alias('O'),
    (pl.col('p_m2') * k_m2).sum().alias('E_m2'),
    (pl.col('p_m3_tot') * k_tot).sum().alias('E_tot'),
    (pl.col('p_m3_chr') * k_chr).sum().alias('E_chr')
]).with_columns([
    (pl.col('O') / pl.col('E_m2')).alias('OE_m2'),
    (pl.col('O') / pl.col('E_tot')).alias('OE_tot'),
    (pl.col('O') / pl.col('E_chr')).alias('OE_chr')
]).join(
    cat.select(['cod_hospital', 'nombre_oficial']), left_on='COD_HOSPITAL', right_on=pl.col('cod_hospital').cast(pl.String), how='left'
)

# EB para cada uno
def calc_eb(oe_s, o_s):
    K = len(oe_s)
    log_oe = np.log(oe_s.to_numpy())
    v = 1.0 / o_s.to_numpy()
    w = 1.0 / v
    mu = np.sum(w * log_oe) / np.sum(w)
    Q = np.sum(w * (log_oe - mu)**2)
    tau2 = max(0.0, (Q - (K - 1)) / (np.sum(w) - np.sum(w**2) / np.sum(w)))
    tau = np.sqrt(tau2)
    B = tau2 / (tau2 + v)
    lth = B * log_oe + (1.0 - B) * mu
    se = np.sqrt(tau2 * v / (tau2 + v))
    z = (lth - np.log(1.10)) / se
    p = 1.0 - stats.norm.cdf(-z)
    return tau, mu, p, (p >= 0.95)

t_m2, mu_m2, p_m2, alt_m2 = calc_eb(h['OE_m2'], h['O'])
t_tot, mu_tot, p_tot, alt_tot = calc_eb(h['OE_tot'], h['O'])
t_chr, mu_chr, p_chr, alt_chr = calc_eb(h['OE_chr'], h['O'])

h = h.with_columns([
    pl.Series('P_m2', p_m2), pl.Series('alt_m2', alt_m2),
    pl.Series('P_tot', p_tot), pl.Series('alt_tot', alt_tot),
    pl.Series('P_chr', p_chr), pl.Series('alt_chr', alt_chr)
])

print("\n" + "=" * 95)
print("=== COMPARACIÓN DE M3: R_DX TOTAL vs R_DX CRÓNICO vs M2 (SOLO AGUDOS GENERALES) ===")
print("=" * 95)
print(f"Parámetros de Dispersión Empírica (tau) y Meta-media (mu):")
print(f"  M2 (Sin R_dx):                    tau = {t_m2:.4f}, mu = {mu_m2:.4f}")
print(f"  M3-Crónico (R_dx sobre 28 Elix):  tau = {t_chr:.4f}, mu = {mu_chr:.4f}")
print(f"  M3-Total (R_dx sobre diag total):  tau = {t_tot:.4f}, mu = {mu_tot:.4f}")

r_tot_chr, _ = stats.spearmanr(h['OE_tot'], h['OE_chr'])
r_tot_m2, _ = stats.spearmanr(h['OE_tot'], h['OE_m2'])
r_chr_m2, _ = stats.spearmanr(h['OE_chr'], h['OE_m2'])
print(f"\nCorrelaciones de Spearman entre Rankings de O/E:")
print(f"  M3-Total vs M3-Crónico: rho = {r_tot_chr:.4f}")
print(f"  M3-Total vs M2:         rho = {r_tot_m2:.4f}")
print(f"  M3-Crónico vs M2:       rho = {r_chr_m2:.4f}")

print("\nHospitales con Alerta en alguna variante (Margen 10%, P >= 0.95):")
print(f"{'Cod':<7} | {'Nombre':<32} | {'OE_m2':>6} | {'P_m2':>6} | {'OE_chr':>6} | {'P_chr':>6} | {'OE_tot':>6} | {'P_tot':>6} | {'Alerta Chr':<10} | {'Alerta Tot':<10}")
print("-" * 115)
for r in h.filter(pl.col('alt_m2') | pl.col('alt_tot') | pl.col('alt_chr')).sort('OE_tot', descending=True).iter_rows(named=True):
    a_chr = "ALERTA" if r['alt_chr'] else "prom"
    a_tot = "ALERTA" if r['alt_tot'] else "prom"
    print(f"{r['COD_HOSPITAL']:<7} | {r['nombre_oficial'][:32]:<32} | {r['OE_m2']:6.2f} | {r['P_m2']:6.3f} | {r['OE_chr']:6.2f} | {r['P_chr']:6.3f} | {r['OE_tot']:6.2f} | {r['P_tot']:6.3f} | {a_chr:<10} | {a_tot:<10}")

