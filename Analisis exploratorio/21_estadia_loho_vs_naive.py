import polars as pl
import numpy as np
import scipy.stats as stats
from pathlib import Path
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

DATA_DEV_CACHE = Path("/home/felipe/Documentos/Proyecto final/Tesis/scratch/dev_eval_cache.parquet")
DATA_23_CACHE = Path("/home/felipe/Documentos/Proyecto final/Tesis/scratch/eval_23_cache.parquet")
CANON_PATH = Path("/home/felipe/Documentos/Proyecto final/Tesis/Analisis exploratorio/tabla_canonica_elixhauser_31.csv")

df_dev = pl.read_parquet(DATA_DEV_CACHE)
df_23 = pl.read_parquet(DATA_23_CACHE)
canon = pl.read_csv(CANON_PATH)

stay_dev = df_dev.filter((pl.col('MORTALIDAD_BINARIA') == 0) & (pl.col('ESTANCIA_DIAS') > 0))
stay_23 = df_23.filter((pl.col('MORTALIDAD_BINARIA') == 0) & (pl.col('ESTANCIA_DIAS') > 0))

# Truncar a percentil 99 (54 días)
stay_dev = stay_dev.with_columns(
    pl.when(pl.col('ESTANCIA_DIAS') > 54.0).then(54.0).otherwise(pl.col('ESTANCIA_DIAS')).alias('LOS_TRUNC')
).with_columns(pl.col('LOS_TRUNC').log().alias('LOG_LOS'))

stay_23 = stay_23.with_columns(
    pl.when(pl.col('ESTANCIA_DIAS') > 54.0).then(54.0).otherwise(pl.col('ESTANCIA_DIAS')).alias('LOS_TRUNC')
).with_columns(pl.col('LOS_TRUNC').log().alias('LOG_LOS'))

# 1. Target encoding NAIVE (in-sample global)
g_mean_dev = stay_dev['LOG_LOS'].mean()
cie_stats_naive = stay_dev.group_by('CIE10_3C').agg([
    pl.len().alias('n_cie'),
    pl.col('LOG_LOS').mean().alias('mean_cie')
]).with_columns([
    ((pl.col('n_cie') * pl.col('mean_cie') + 10.0 * g_mean_dev) / (pl.col('n_cie') + 10.0)).alias('CIE10_ENC_NAIVE')
])

# 2. Target encoding LOHO (Leave-One-Hospital-Out)
# Para cada hospital h, calculamos el target encoding excluyendo al hospital h
hosp_cie_stats = stay_dev.group_by(['COD_HOSPITAL', 'CIE10_3C']).agg([
    pl.len().alias('n_hc'),
    pl.col('LOG_LOS').sum().alias('sum_hc')
])
cie_tot_stats = stay_dev.group_by('CIE10_3C').agg([
    pl.len().alias('n_tot'),
    pl.col('LOG_LOS').sum().alias('sum_tot')
])

loho_stats = hosp_cie_stats.join(cie_tot_stats, on='CIE10_3C').with_columns([
    (pl.col('n_tot') - pl.col('n_hc')).alias('n_loho'),
    (pl.col('sum_tot') - pl.col('sum_hc')).alias('sum_loho')
]).with_columns([
    pl.when(pl.col('n_loho') > 0).then(pl.col('sum_loho') / pl.col('n_loho')).otherwise(g_mean_dev).alias('mean_loho')
]).with_columns([
    ((pl.col('n_loho') * pl.col('mean_loho') + 10.0 * g_mean_dev) / (pl.col('n_loho') + 10.0)).alias('CIE10_ENC_LOHO')
])

stay_dev = stay_dev.join(loho_stats.select(['COD_HOSPITAL', 'CIE10_3C', 'CIE10_ENC_LOHO']), on=['COD_HOSPITAL', 'CIE10_3C'], how='left').with_columns(
    pl.col('CIE10_ENC_LOHO').fill_null(g_mean_dev)
).join(cie_stats_naive.select(['CIE10_3C', 'CIE10_ENC_NAIVE']), on='CIE10_3C', how='left').with_columns(
    pl.col('CIE10_ENC_NAIVE').fill_null(g_mean_dev)
)

stay_23 = stay_23.join(cie_stats_naive.select(['CIE10_3C', 'CIE10_ENC_NAIVE']), on='CIE10_3C', how='left').with_columns(
    pl.col('CIE10_ENC_NAIVE').fill_null(g_mean_dev)
).with_columns(
    pl.col('CIE10_ENC_NAIVE').alias('CIE10_ENC_LOHO') # En 2023, el modelo entrenado en Dev se evalúa out-of-sample
)

elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]
f_base = ['EDAD_ANIOS', 'INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL', 'SEXO_MASCULINO']

# Entrenar Ridge B4 con LOHO vs B4 con Naive en Dev y evaluar en 2023
feats_loho = f_base + elix_28 + ['CIE10_ENC_LOHO']
feats_naive = f_base + elix_28 + ['CIE10_ENC_NAIVE']

sc_l = StandardScaler()
X_tr_l = sc_l.fit_transform(stay_dev.select(feats_loho).fill_null(0).to_pandas().values)
m_l = Ridge(alpha=100.0, random_state=42).fit(X_tr_l, stay_dev['LOG_LOS'].to_numpy())
X_ev_l = sc_l.transform(stay_23.select(feats_loho).fill_null(0).to_pandas().values)
pred_l = np.exp(m_l.predict(X_ev_l))
k_l = stay_23['ESTANCIA_DIAS'].sum() / np.sum(pred_l)
exp_l = pred_l * k_l

sc_n = StandardScaler()
X_tr_n = sc_n.fit_transform(stay_dev.select(feats_naive).fill_null(0).to_pandas().values)
m_n = Ridge(alpha=100.0, random_state=42).fit(X_tr_n, stay_dev['LOG_LOS'].to_numpy())
X_ev_n = sc_n.transform(stay_23.select(feats_naive).fill_null(0).to_pandas().values)
pred_n = np.exp(m_n.predict(X_ev_n))
k_n = stay_23['ESTANCIA_DIAS'].sum() / np.sum(pred_n)
exp_n = pred_n * k_n

h_stay = stay_23.with_columns([
    pl.Series('exp_loho', exp_l),
    pl.Series('exp_naive', exp_n)
]).group_by('COD_HOSPITAL').agg([
    pl.col('ESTANCIA_DIAS').sum().alias('obs_days'),
    pl.col('exp_loho').sum().alias('exp_loho'),
    pl.col('exp_naive').sum().alias('exp_naive')
]).with_columns([
    (pl.col('obs_days') / pl.col('exp_loho')).alias('OE_loho'),
    (pl.col('obs_days') / pl.col('exp_naive')).alias('OE_naive')
])

rho_loho_naive, _ = stats.spearmanr(h_stay['OE_loho'], h_stay['OE_naive'])
rms_loho_naive = np.sqrt(np.mean((stats.rankdata(h_stay['OE_loho']) - stats.rankdata(h_stay['OE_naive']))**2))

print("=== COMPARACIÓN FORMAL: LOHO VS TARGET ENCODING NAIVE EN ESTADÍA ===")
print(f"Correlación de Spearman (rho): {rho_loho_naive:.5f}")
print(f"Desplazamiento Cuadrático Medio (RMS): {rms_loho_naive:.2f} puestos")

# Auditoría exacta de las 17 variables en Dev Completo (N = 1,303,718)
print("\n=== AJUSTE ELASTICNET EN DEV COMPLETO (N = 1,303,718) ===")
all_34 = f_base + elix_28 + ['CIE10_ENC_LOHO']
sc_all = StandardScaler()
X_all = sc_all.fit_transform(stay_dev.select(all_34).fill_null(0).to_pandas().values)
y_all = stay_dev['LOG_LOS'].to_numpy()

# Ajustar ElasticNet con alpha = 0.036680 y alpha = 0.037431
m_36 = ElasticNet(alpha=0.036680, l1_ratio=0.5, random_state=42, max_iter=2000).fit(X_all, y_all)
m_37 = ElasticNet(alpha=0.037431, l1_ratio=0.5, random_state=42, max_iter=2000).fit(X_all, y_all)

# Unir con nombres canónicos de tabla_canonica_elixhauser_31.csv
labels_dict = {}
for r in canon.iter_rows(named=True):
    labels_dict[r["categoria_id"]] = r["nombre_comorbilidad"]
labels_dict['CIE10_ENC_LOHO'] = 'Diagnóstico Principal (LOHO)'
labels_dict['INGRESO_URGENCIA'] = 'Admisión por Urgencia'
labels_dict['INGRESO_CRITICO'] = 'Ingreso a Cama Crítica (UCI/UTI)'
labels_dict['DERIVADO_OTRO_HOSPITAL'] = 'Derivado de otro establecimiento'
labels_dict['SEXO_MASCULINO'] = 'Sexo Masculino'
labels_dict['EDAD_ANIOS'] = 'Edad (Años continuos)'

coefs_df = pl.DataFrame({
    'variable': all_34,
    'coef_36': m_36.coef_,
    'coef_37': m_37.coef_
}).with_columns([
    pl.col('variable').replace(labels_dict).alias('nombre_canonico'),
    pl.col('coef_36').abs().alias('abs_coef')
]).sort('abs_coef', descending=True)

print("\nCoeficientes en Cohorte de Desarrollo Completa (N = 1.303.718):")
print(f"{'Variable':<18} | {'Nombre Canónico':<35} | {'Coef (0.036680)':>15} | {'Coef (0.037431)':>15}")
print("-" * 90)
for r in coefs_df.iter_rows(named=True):
    print(f"{r['variable']:<18} | {r['nombre_canonico'][:35]:<35} | {r['coef_36']:+15.5f} | {r['coef_37']:+15.5f}")

