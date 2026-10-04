import polars as pl
import numpy as np
import scipy.stats as stats
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df_dev = pl.read_parquet('/home/felipe/Documentos/Proyecto final/Tesis/scratch/dev_poa_cache.parquet')
df_23 = pl.read_parquet('/home/felipe/Documentos/Proyecto final/Tesis/scratch/eval_23_poa_cache.parquet')
cat = pl.read_csv('/home/felipe/Documentos/Proyecto final/Tesis/config/catalogo_hospitales_procedencia.csv').with_columns(pl.col('cod_hospital').cast(pl.String))

# Variables
f_base = ['EDAD_ANIOS', 'INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL', 'SEXO_MASCULINO']
elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]

# Lista de features para Modelo POA (reemplazando ELIX_04 y ELIX_09 por sus versiones depuradas de I26, G93.4, R56)
elix_28_poa = [c if c not in ['ELIX_04', 'ELIX_09'] else f'{c}_POA' for c in elix_28]

# Lista estricta de 26 crónicas (excluyendo también ELIX_24 y ELIX_26)
elix_26_poa = [c for c in elix_28_poa if c not in ['ELIX_24', 'ELIX_26']]

# OHE MDC
mdcs_dev = df_dev['MDC'].to_pandas().values.reshape(-1, 1)
mdcs_23 = df_23['MDC'].to_pandas().values.reshape(-1, 1)
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
X_mdc_dev = ohe.fit_transform(mdcs_dev)
X_mdc_23 = ohe.transform(mdcs_23)

# Scaler
sc1 = StandardScaler()
X_dev_m2cap = np.hstack([sc1.fit_transform(df_dev.select(f_base + elix_28).fill_null(0).to_pandas().values), X_mdc_dev])
X_23_m2cap = np.hstack([sc1.transform(df_23.select(f_base + elix_28).fill_null(0).to_pandas().values), X_mdc_23])

sc2 = StandardScaler()
X_dev_poa = np.hstack([sc2.fit_transform(df_dev.select(f_base + elix_28_poa).fill_null(0).to_pandas().values), X_mdc_dev])
X_23_poa = np.hstack([sc2.transform(df_23.select(f_base + elix_28_poa).fill_null(0).to_pandas().values), X_mdc_23])

sc3 = StandardScaler()
X_dev_poa26 = np.hstack([sc3.fit_transform(df_dev.select(f_base + elix_26_poa).fill_null(0).to_pandas().values), X_mdc_dev])
X_23_poa26 = np.hstack([sc3.transform(df_23.select(f_base + elix_26_poa).fill_null(0).to_pandas().values), X_mdc_23])

y_dev = df_dev['MORTALIDAD_BINARIA'].to_numpy()
o_tot = df_23['MORTALIDAD_BINARIA'].sum()

print("Ajustando modelos...")
m_prim = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_m2cap, y_dev)
m_poa = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_poa, y_dev)
m_poa26 = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_poa26, y_dev)

p_prim = m_prim.predict_proba(X_23_m2cap)[:, 1]
p_poa = m_poa.predict_proba(X_23_poa)[:, 1]
p_poa26 = m_poa26.predict_proba(X_23_poa26)[:, 1]

df_23 = df_23.with_columns([
    pl.Series('E_prim', p_prim * (o_tot / p_prim.sum())),
    pl.Series('E_poa', p_poa * (o_tot / p_poa.sum())),
    pl.Series('E_poa26', p_poa26 * (o_tot / p_poa26.sum()))
])

# Agregacion por hospital
h = df_23.group_by('COD_HOSPITAL').agg([
    pl.len().alias('N'),
    pl.col('MORTALIDAD_BINARIA').sum().alias('O'),
    pl.col('E_prim').sum().alias('E_prim'),
    pl.col('E_poa').sum().alias('E_poa'),
    pl.col('E_poa26').sum().alias('E_poa26')
]).with_columns([
    (pl.col('O') / pl.col('E_prim')).alias('OE_prim'),
    (pl.col('O') / pl.col('E_poa')).alias('OE_poa'),
    (pl.col('O') / pl.col('E_poa26')).alias('OE_poa26')
]).join(cat.select(['cod_hospital', 'nombre_oficial', 'es_monografico']), left_on='COD_HOSPITAL', right_on='cod_hospital', how='left')

# Inferencia Bayesiana
c = np.log(1.10)
def calc_p(o_series, e_series, tau=0.1948, mu=-0.0359):
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

h = h.with_columns([
    pl.Series('P_prim', calc_p(h['O'], h['E_prim'])),
    pl.Series('P_poa', calc_p(h['O'], h['E_poa'])),
    pl.Series('P_poa26', calc_p(h['O'], h['E_poa26']))
])

# Correlaciones
r_poa, _ = stats.spearmanr(h['OE_prim'], h['OE_poa'])
r_poa26, _ = stats.spearmanr(h['OE_prim'], h['OE_poa26'])
print(f"\nCorrelación de Spearman entre rankings O/E:")
print(f"  M2+CAP Primario vs M2+CAP POA (depurado I26, G93.4, R56):  rho = {r_poa:.5f}")
print(f"  M2+CAP Primario vs M2+CAP POA-26 (estricto 26 crónicas):     rho = {r_poa26:.5f}")

# Evaluación de Pereira
per = h.filter(pl.col('COD_HOSPITAL') == '106102').to_dicts()[0]
print("\n" + "=" * 90)
print(f"EVALUACIÓN DE HOSPITAL DR. EDUARDO PEREIRA (106102):")
print(f"  M2+CAP Primario:  E = {per['E_prim']:.2f}, O/E = {per['OE_prim']:.4f}, P(theta > 1.10) = {per['P_prim']:.4f}")
print(f"  M2+CAP POA:       E = {per['E_poa']:.2f}, O/E = {per['OE_poa']:.4f}, P(theta > 1.10) = {per['P_poa']:.4f}")
print(f"  M2+CAP POA-26:    E = {per['E_poa26']:.2f}, O/E = {per['OE_poa26']:.4f}, P(theta > 1.10) = {per['P_poa26']:.4f}")
print("=" * 90)

# Alertas Primarias
alert_codes = ['113180', '106103', '112101', '107101', '112103', '111100', '108100', '103100', '121109', '113100']
print("\nEVALUACIÓN DE LAS 10 ALERTAS PRIMARIAS BAJO SENSIBILIDAD POA:")
print(f"{'Cod':<7} | {'Nombre':<30} | {'OE_prim':>7} | {'P_prim':>7} | {'OE_poa':>7} | {'P_poa':>7} | {'OE_poa26':>8} | {'P_poa26':>8}")
print("-" * 95)
for hid in alert_codes:
    r = h.filter(pl.col('COD_HOSPITAL') == hid).to_dicts()[0]
    print(f"{r['COD_HOSPITAL']:<7} | {r['nombre_oficial'][:30]:<30} | {r['OE_prim']:7.3f} | {r['P_prim']:7.4f} | {r['OE_poa']:7.3f} | {r['P_poa']:7.4f} | {r['OE_poa26']:8.3f} | {r['P_poa26']:8.4f}")

# Otros hospitales de interés
print("\nHOSPITALES DE INTERÉS DE LA COMISIÓN:")
for hid in ['128109', '114101', '112100', '109100']:
    r = h.filter(pl.col('COD_HOSPITAL') == hid).to_dicts()[0]
    print(f"{r['COD_HOSPITAL']:<7} | {r['nombre_oficial'][:30]:<30} | {r['OE_prim']:7.3f} | {r['P_prim']:7.4f} | {r['OE_poa']:7.3f} | {r['P_poa']:7.4f} | {r['OE_poa26']:8.3f} | {r['P_poa26']:8.4f}")
