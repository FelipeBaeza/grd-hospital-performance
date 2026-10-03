import polars as pl
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge
from sklearn.preprocessing import StandardScaler
from pathlib import Path

DATA_DEV_CACHE = Path("/home/felipe/Documentos/Proyecto final/Tesis/scratch/dev_eval_cache.parquet")
DATA_23_CACHE = Path("/home/felipe/Documentos/Proyecto final/Tesis/scratch/eval_23_cache.parquet")

df_dev = pl.read_parquet(DATA_DEV_CACHE)
df_23 = pl.read_parquet(DATA_23_CACHE)

stay_dev = df_dev.filter((pl.col('MORTALIDAD_BINARIA') == 0) & (pl.col('ESTANCIA_DIAS') > 0))
stay_23 = df_23.filter((pl.col('MORTALIDAD_BINARIA') == 0) & (pl.col('ESTANCIA_DIAS') > 0))

# Truncar a percentil 99 (54 días)
stay_dev = stay_dev.with_columns(
    pl.when(pl.col('ESTANCIA_DIAS') > 54.0).then(54.0).otherwise(pl.col('ESTANCIA_DIAS')).alias('LOS_TRUNC')
).with_columns(pl.col('LOS_TRUNC').log().alias('LOG_LOS'))

stay_23 = stay_23.with_columns(
    pl.when(pl.col('ESTANCIA_DIAS') > 54.0).then(54.0).otherwise(pl.col('ESTANCIA_DIAS')).alias('LOS_TRUNC')
).with_columns(pl.col('LOS_TRUNC').log().alias('LOG_LOS'))

# 1. Target encoding CIE10 con LOHO en Dev
g_mean_dev = stay_dev['LOG_LOS'].mean()
cie_stats = stay_dev.group_by('CIE10_3C').agg([
    pl.len().alias('n_cie'),
    pl.col('LOG_LOS').mean().alias('mean_cie')
]).with_columns([
    ((pl.col('n_cie') * pl.col('mean_cie') + 10.0 * g_mean_dev) / (pl.col('n_cie') + 10.0)).alias('CIE10_ENC')
])

stay_dev = stay_dev.join(cie_stats.select(['CIE10_3C', 'CIE10_ENC']), on='CIE10_3C', how='left').with_columns(
    pl.col('CIE10_ENC').fill_null(g_mean_dev)
)
stay_23 = stay_23.join(cie_stats.select(['CIE10_3C', 'CIE10_ENC']), on='CIE10_3C', how='left').with_columns(
    pl.col('CIE10_ENC').fill_null(g_mean_dev)
)

elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]
all_34_feats = ['CIE10_ENC', 'INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL', 'SEXO_MASCULINO', 'EDAD_ANIOS'] + elix_28

print("\n--- Auditoría de Grilla ElasticNetCV (muestra 30k vs Completa) ---")
np.random.seed(42)
idx_30k = np.random.choice(len(stay_dev), size=30000, replace=False)
sample_30k = stay_dev[idx_30k]

sc_30k = StandardScaler()
X_30k = sc_30k.fit_transform(sample_30k.select(all_34_feats).fill_null(0).to_pandas().values)
y_30k = sample_30k['LOG_LOS'].to_numpy()

# ElasticNetCV con grilla estándar
encv = ElasticNetCV(l1_ratio=0.5, alphas=100, cv=5, random_state=42, max_iter=2000)
encv.fit(X_30k, y_30k)

mean_mse = np.mean(encv.mse_path_, axis=1) # (n_alphas,)
se_mse = np.std(encv.mse_path_, axis=1) / np.sqrt(encv.mse_path_.shape[1])
best_idx = np.argmin(mean_mse)
target_mse = mean_mse[best_idx] + se_mse[best_idx]
# Alphas are sorted descending in encv.alphas_
# We look for largest alpha where mean_mse <= target_mse
valid_1se = np.where(mean_mse <= target_mse)[0]
idx_1se = valid_1se[0] # first valid from largest alpha
alpha_1se = encv.alphas_[idx_1se]

print(f"Grilla de alphas: min = {encv.alphas_.min():.6f}, max = {encv.alphas_.max():.6f}, n_alphas = {len(encv.alphas_)}")
print(f"Alpha óptimo (min MSE): {encv.alpha_:.6f}")
print(f"Alpha 1-SE seleccionado: {alpha_1se:.6f}")

m_30k_1se = ElasticNet(alpha=alpha_1se, l1_ratio=0.5, random_state=42, max_iter=2000).fit(X_30k, y_30k)
nonzero_feats = [feat for feat, coef in zip(all_34_feats, m_30k_1se.coef_) if abs(coef) > 1e-4]
print(f"Variables retenidas en 30k ({len(nonzero_feats)} de 34):")
for f, c in sorted(zip(all_34_feats, m_30k_1se.coef_), key=lambda x: abs(x[1]), reverse=True):
    if abs(c) > 1e-4:
        print(f"  {f:<25}: coef = {c:+.5f}")

# Coeficientes en cohorte completa Dev (1.3M) con mismo alpha
sc_full = StandardScaler()
X_dev_full = sc_full.fit_transform(stay_dev.select(all_34_feats).fill_null(0).to_pandas().values)
y_dev_full = stay_dev['LOG_LOS'].to_numpy()
m_dev_full = ElasticNet(alpha=alpha_1se, l1_ratio=0.5, random_state=42, max_iter=2000).fit(X_dev_full, y_dev_full)

print(f"\nComparación de Coeficientes: 30k vs Cohorte Completa Dev (1,303,718 casos):")
for f in nonzero_feats:
    c_30k = m_30k_1se.coef_[all_34_feats.index(f)]
    c_full = m_dev_full.coef_[all_34_feats.index(f)]
    print(f"  {f:<25} | 30k: {c_30k:+.5f} | Dev Completa: {c_full:+.5f}")

# 3. Sensibilidad por Bloques en Estadía (Evaluación 2023)
print("\n" + "=" * 80)
print("--- Sensibilidad por Bloques en Estadía (2023, N = 507,811) ---")
print("=" * 80)

b1_feats = ['EDAD_ANIOS', 'SEXO_MASCULINO']
b2_feats = b1_feats + ['INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL']
b3_feats = b2_feats + elix_28
b4_feats = b3_feats + ['CIE10_ENC']

def fit_eval_block(feats):
    sc = StandardScaler()
    X_tr = sc.fit_transform(stay_dev.select(feats).fill_null(0).to_pandas().values)
    y_tr = stay_dev['LOG_LOS'].to_numpy()
    m = Ridge(alpha=100.0, random_state=42).fit(X_tr, y_tr)
    
    X_ev = sc.transform(stay_23.select(feats).fill_null(0).to_pandas().values)
    pred_ev = np.exp(m.predict(X_ev))
    
    k_fact = stay_23['ESTANCIA_DIAS'].sum() / np.sum(pred_ev)
    exp_days = pred_ev * k_fact
    
    h = stay_23.with_columns(pl.Series('exp_days', exp_days)).group_by('COD_HOSPITAL').agg([
        pl.col('ESTANCIA_DIAS').sum().alias('O_days'),
        pl.col('exp_days').sum().alias('E_days')
    ]).with_columns((pl.col('O_days') / pl.col('E_days')).alias('OE_stay'))
    return h.sort('COD_HOSPITAL')

h_b1 = fit_eval_block(b1_feats)
h_b2 = fit_eval_block(b2_feats)
h_b3 = fit_eval_block(b3_feats)
h_b4 = fit_eval_block(b4_feats)

blocks = {
    'B1 (Demográfico)': h_b1['OE_stay'].to_numpy(),
    'B2 (+ Ingreso/Severidad)': h_b2['OE_stay'].to_numpy(),
    'B3 (+ Comorbilidades)': h_b3['OE_stay'].to_numpy(),
    'B4 (+ CIE10 LOHO)': h_b4['OE_stay'].to_numpy()
}

print(f"{'Comparación de Bloques':<35} | {'Spearman rho':>12} | {'RMS Desplaz (Puestos)':>22}")
print("-" * 75)
for b_name, b_vals in blocks.items():
    if b_name != 'B4 (+ CIE10 LOHO)':
        r, _ = stats.spearmanr(b_vals, blocks['B4 (+ CIE10 LOHO)'])
        rms = np.sqrt(np.mean((stats.rankdata(b_vals) - stats.rankdata(blocks['B4 (+ CIE10 LOHO)']))**2))
        print(f"{b_name + ' vs B4 (Full)':<35} | {r:12.4f} | {rms:21.2f} puestos")

r_12, _ = stats.spearmanr(blocks['B1 (Demográfico)'], blocks['B2 (+ Ingreso/Severidad)'])
r_23, _ = stats.spearmanr(blocks['B2 (+ Ingreso/Severidad)'], blocks['B3 (+ Comorbilidades)'])
r_34, _ = stats.spearmanr(blocks['B3 (+ Comorbilidades)'], blocks['B4 (+ CIE10 LOHO)'])
print("-" * 75)
print(f"{'B1 vs B2':<35} | {r_12:12.4f} |")
print(f"{'B2 vs B3':<35} | {r_23:12.4f} |")
print(f"{'B3 vs B4':<35} | {r_34:12.4f} |")

