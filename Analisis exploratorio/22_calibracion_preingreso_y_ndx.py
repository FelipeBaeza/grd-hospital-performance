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

# Suma de crónicos
df_dev = df_dev.with_columns(pl.sum_horizontal(elix_28).alias('N_CHRONIC'))
df_23 = df_23.with_columns(pl.sum_horizontal(elix_28).alias('N_CHRONIC'))

# R_DX
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

sc = StandardScaler()
X_dev_base = sc.fit_transform(df_dev.select(f_base + elix_28).fill_null(0).to_pandas().values)
X_23_base = sc.transform(df_23.select(f_base + elix_28).fill_null(0).to_pandas().values)

sp_tot = SplineTransformer(n_knots=4, degree=3, include_bias=False)
sp_dev_tot = sp_tot.fit_transform(df_dev.select(['R_DX_TOT']).to_pandas().values)
sp_23_tot = sp_tot.transform(df_23.select(['R_DX_TOT']).to_pandas().values)

sp_chr = SplineTransformer(n_knots=4, degree=3, include_bias=False)
sp_dev_chr = sp_chr.fit_transform(df_dev.select(['R_DX_CHRON']).to_pandas().values)
sp_23_chr = sp_chr.transform(df_23.select(['R_DX_CHRON']).to_pandas().values)

y_dev = df_dev['MORTALIDAD_BINARIA'].to_numpy()

print("Ajustando modelos M2, M3-Crónico y M3-Total...")
m2 = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_base, y_dev)
m3_chr = LogisticRegression(max_iter=200, random_state=42).fit(np.hstack([X_dev_base, sp_dev_chr]), y_dev)
m3_tot = LogisticRegression(max_iter=200, random_state=42).fit(np.hstack([X_dev_base, sp_dev_tot]), y_dev)

p_m2 = m2.predict_proba(X_23_base)[:, 1]
p_chr = m3_chr.predict_proba(np.hstack([X_23_base, sp_23_chr]))[:, 1]
p_tot = m3_tot.predict_proba(np.hstack([X_23_base, sp_23_tot]))[:, 1]

# Recalibración 2023
k_m2 = df_23['MORTALIDAD_BINARIA'].sum() / np.sum(p_m2)
k_chr = df_23['MORTALIDAD_BINARIA'].sum() / np.sum(p_chr)
k_tot = df_23['MORTALIDAD_BINARIA'].sum() / np.sum(p_tot)

df_23 = df_23.with_columns([
    (pl.Series('p_m2', p_m2) * k_m2).alias('E_m2'),
    (pl.Series('p_chr', p_chr) * k_chr).alias('E_chr'),
    (pl.Series('p_tot', p_tot) * k_tot).alias('E_tot')
])

# 1. Calibración por estratos de N_DX
print("\n=== 1. CALIBRACIÓN POR ESTRATOS DE N_DX (DIAGNÓSTICOS SECUNDARIOS TOTALES) ===")
df_23 = df_23.with_columns(
    pl.when(pl.col('N_DX') <= 2).then(pl.lit('0-2 dx'))
    .when(pl.col('N_DX') <= 5).then(pl.lit('3-5 dx'))
    .when(pl.col('N_DX') <= 10).then(pl.lit('6-10 dx'))
    .otherwise(pl.lit('>=11 dx')).alias('estrato_ndx')
)

cal_ndx = df_23.group_by('estrato_ndx').agg([
    pl.len().alias('N'),
    pl.col('MORTALIDAD_BINARIA').sum().alias('O'),
    pl.col('E_m2').sum().alias('E_m2'),
    pl.col('E_chr').sum().alias('E_chr'),
    pl.col('E_tot').sum().alias('E_tot')
]).with_columns([
    (pl.col('O') / pl.col('E_m2')).alias('OE_m2'),
    (pl.col('O') / pl.col('E_chr')).alias('OE_chr'),
    (pl.col('O') / pl.col('E_tot')).alias('OE_tot'),
    (pl.col('O') / pl.col('N') * 100).alias('tasa_mort_pct')
]).sort('N', descending=True)

print(f"{'Estrato N_dx':<12} | {'N':>8} | {'O':>6} | {'Tasa %':>7} | {'OE (M2 Primario)':>17} | {'OE (M3-Crónico)':>16} | {'OE (M3-Total Sens)':>19}")
print("-" * 95)
for r in cal_ndx.iter_rows(named=True):
    print(f"{r['estrato_ndx']:<12} | {r['N']:8d} | {r['O']:6d} | {r['tasa_mort_pct']:6.2f}% | {r['OE_m2']:17.3f} | {r['OE_chr']:16.3f} | {r['OE_tot']:19.3f}")

# 2. Calibración del Modelo Primario (M2) por estratos PREVIOS AL INGRESO
print("\n=== 2. CALIBRACIÓN DE M2 POR ESTRATOS PREVIOS AL INGRESO ===")

# Estratos de Edad
df_23 = df_23.with_columns(
    pl.when(pl.col('EDAD_ANIOS') < 65).then(pl.lit('<65 años'))
    .when(pl.col('EDAD_ANIOS') < 75).then(pl.lit('65-74 años'))
    .otherwise(pl.lit('>=75 años')).alias('estrato_edad'),
    
    pl.when(pl.col('N_CHRONIC') == 0).then(pl.lit('0 comorb'))
    .when(pl.col('N_CHRONIC') == 1).then(pl.lit('1 comorb'))
    .when(pl.col('N_CHRONIC') == 2).then(pl.lit('2 comorb'))
    .otherwise(pl.lit('>=3 comorb')).alias('estrato_cronico'),
    
    pl.when(pl.col('INGRESO_URGENCIA') == 1).then(pl.lit('Urgencia'))
    .otherwise(pl.lit('Programada')).alias('estrato_ingreso')
)

def print_calib(dim_col, dim_name):
    print(f"\n--- Estratificación por {dim_name} ---")
    df_top = df_23 if dim_col != "mdc_str" else df_top_mdc
    cal = df_top.group_by(dim_col).agg([
        pl.len().alias('N'),
        pl.col('MORTALIDAD_BINARIA').sum().alias('O'),
        pl.col('E_m2').sum().alias('E')
    ]).with_columns([
        (pl.col('O') / pl.col('E')).alias('OE'),
        (pl.col('O') / pl.col('N') * 100).alias('tasa_pct')
    ]).sort('N', descending=True)
    
    print(f"{dim_name:<25} | {'N':>8} | {'O':>6} | {'E':>7} | {'Tasa %':>7} | {'O/E M2':>8} | {'¿En [0.80, 1.25]?':<18}")
    print("-" * 90)
    for r in cal.iter_rows(named=True):
        ok = "SÍ [Calibrado]" if 0.80 <= r['OE'] <= 1.25 else "FUERA"
        print(f"{str(r[dim_col]):<25} | {r['N']:8d} | {r['O']:6d} | {r['E']:7.1f} | {r['tasa_pct']:6.2f}% | {r['OE']:8.3f} | {ok:<18}")

print_calib('estrato_edad', 'Edad Previa')
print_calib('estrato_cronico', 'Carga Comorbilidad Crónica')
print_calib('estrato_ingreso', 'Vía de Ingreso')

# Principales MDCs
top_mdcs = [5, 4, 6, 8, 1, 11]
df_top_mdc = df_23.filter(pl.col('MDC').is_in(top_mdcs)).with_columns(
    pl.col('MDC').cast(pl.String).alias('mdc_str')
)
print_calib('mdc_str', 'MDC Principal (Top)')

