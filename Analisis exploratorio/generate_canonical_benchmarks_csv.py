import polars as pl
import numpy as np
import scipy.stats as stats
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SplineTransformer

print("1. Cargando caches de datos...")
df_dev = pl.read_parquet('/home/felipe/Documentos/Proyecto final/Tesis/scratch/dev_eval_cache.parquet')
df_23 = pl.read_parquet('/home/felipe/Documentos/Proyecto final/Tesis/scratch/eval_23_cache.parquet')
cat = pl.read_csv('/home/felipe/Documentos/Proyecto final/Tesis/config/catalogo_hospitales_procedencia.csv').with_columns(pl.col('cod_hospital').cast(pl.String))

elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]
f_base = ['EDAD_ANIOS', 'INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL', 'SEXO_MASCULINO']

# OHE de MDC
mdcs_dev = df_dev['MDC'].to_pandas().values.reshape(-1, 1)
mdcs_23 = df_23['MDC'].to_pandas().values.reshape(-1, 1)
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
X_mdc_dev = ohe.fit_transform(mdcs_dev)
X_mdc_23 = ohe.transform(mdcs_23)

# Scaler base
sc = StandardScaler()
X_dev_base = sc.fit_transform(df_dev.select(f_base + elix_28).fill_null(0).to_pandas().values)
X_23_base = sc.transform(df_23.select(f_base + elix_28).fill_null(0).to_pandas().values)

X_dev_m2mdc = np.hstack([X_dev_base, X_mdc_dev])
X_23_m2mdc = np.hstack([X_23_base, X_mdc_23])

# R_DX total para M3 (estandarizado con media de hospital en dev)
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

print("2. Entrenando modelos M2+MDC, M2 Base y M3-Total...")
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

# Agregacion por hospital
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

# Inferencia Empirica de Bayes
c = np.log(1.10)

def calc_p(o_series, e_series, tau, mu):
    p_vals = []
    for o, e in zip(o_series, e_series):
        if o == 0 or e == 0:
            p_vals.append(0.0)
            continue
        y = np.log(o / e)
        v = 1.0 / o
        B = (tau**2) / (tau**2 + v)
        theta = B * y + (1 - B) * mu
        se = np.sqrt(B * v)
        z = (theta - c) / se
        p_vals.append(float(stats.norm.cdf(z)))
    return p_vals

h_res = h_res.with_columns([
    pl.Series('P_m2mdc', calc_p(h_res['O'], h_res['E_m2mdc'], 0.1948, -0.0359)),
    pl.Series('P_m2', calc_p(h_res['O'], h_res['E_m2'], 0.2077, -0.0267)),
    pl.Series('P_m3', calc_p(h_res['O'], h_res['E_m3'], 0.2041, -0.0347))
])

# Ordenar por COD_HOSPITAL
h_res = h_res.sort('COD_HOSPITAL')

out_csv1 = Path('/home/felipe/Documentos/Proyecto final/Tesis/Analisis exploratorio/benchmarks_modelos_fase2_2023.csv')
out_csv2 = Path('/home/felipe/Documentos/Proyecto final/Tesis/scratch/benchmarks_modelos_fase2_2023.csv')
h_res.write_csv(out_csv1)
h_res.write_csv(out_csv2)
print(f"3. Archivo unico canonical de E y O guardado en {out_csv1} (K = {h_res.height})")

# Validacion de Del Salvador, Sotero, Curanilahue, Van Buren, Temuco
for hid in ['112100', '114101', '128109', '106100', '121109']:
    r = h_res.filter(pl.col('COD_HOSPITAL') == hid).to_dicts()[0]
    print(f"[{hid}] {r['nombre_oficial'][:25]}: O={r['O']}, E_m2mdc={r['E_m2mdc']:.4f}, OE_m2mdc={r['OE_m2mdc']:.4f}, P_m2mdc={r['P_m2mdc']:.4f}")
