import polars as pl
import numpy as np
import scipy.stats as stats
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder

SILVER_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/silver")
GOLD_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/gold")
CAT_PATH = Path("/home/felipe/Documentos/Proyecto final/Tesis/config/catalogo_hospitales_procedencia.csv")

s23 = pl.read_parquet(SILVER_DIR / "silver_2023.parquet")
g23 = pl.read_parquet(GOLD_DIR / "gold_2023.parquet")
cat = pl.read_csv(CAT_PATH).with_columns(pl.col('cod_hospital').cast(pl.String))

# Inpatient evaluable adults filters
c_ex01 = (g23['IR_29301_COD_GRD'].is_null()) | (g23['MDC'] == 0) | (g23['IR_29301_COD_GRD'].str.starts_with('99'))
c_ex02 = (~c_ex01) & (g23['MDC'] == 15)
c_ex03 = (~c_ex01) & (~c_ex02) & (g23['MDC'] == 14)
c_ex04 = (~c_ex01) & (~c_ex02) & (~c_ex03) & ((g23['EDAD_ANIOS'] < 18) | (g23['EDAD_ANIOS'].is_null()))
c_ex05 = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (s23['TIPO_ACTIVIDAD'] != 'HOSPITALIZACIÓN')
inp_mask = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (~c_ex05)

is_transfer = s23['TIPOALTA'].str.contains('DERIVACIÓN OTRO HOSPITAL|DERIVACIÓN INST. PRIVADA')
transf_df = s23.filter(inp_mask & is_transfer).select([
    'ID_EPISODIO', 'COD_HOSPITAL', 'CIP_ENCRIPTADO', 'FECHAALTA'
])
print(f"Total traslados inpatient adultos 2023: {len(transf_df):,}")

# Model M2+MDC trained on Dev
df_dev = pl.read_parquet("/home/felipe/Documentos/Proyecto final/Tesis/scratch/dev_eval_cache.parquet")
df_23 = pl.read_parquet("/home/felipe/Documentos/Proyecto final/Tesis/scratch/eval_23_cache.parquet")

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

y_dev = df_dev['MORTALIDAD_BINARIA'].to_numpy()
m2_mdc = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_m2mdc, y_dev)

o_tot = df_23['MORTALIDAD_BINARIA'].sum()
p_23 = m2_mdc.predict_proba(X_23_m2mdc)[:, 1]
factor = o_tot / p_23.sum()
df_23 = df_23.with_columns(pl.Series('E_M2MDC', p_23 * factor))

# Predict on transfers
s_tr = s23.filter(inp_mask & is_transfer).with_columns([
    s23.filter(inp_mask & is_transfer)['TIPO_INGRESO'].str.to_uppercase().str.contains('URGEN').cast(pl.Int32).alias('INGRESO_URGENCIA'),
    s23.filter(inp_mask & is_transfer)['SERVICIOINGRESO'].str.to_uppercase().str.contains('UCI|UTI|CRITIC|INTENS|INTERMED').cast(pl.Int32).alias('INGRESO_CRITICO'),
    (s23.filter(inp_mask & is_transfer)['HOSPPROCEDENCIA'].is_not_null() | s23.filter(inp_mask & is_transfer)['TIPO_PROCEDENCIA'].str.to_uppercase().str.contains('DERIV|OTRO|HOSP')).cast(pl.Int32).alias('DERIVADO_OTRO_HOSPITAL'),
    (g23.filter(inp_mask & is_transfer)['SEXO'] == 1).cast(pl.Int32).alias('SEXO_MASCULINO'),
    g23.filter(inp_mask & is_transfer)['EDAD_ANIOS'].alias('EDAD_ANIOS'),
    g23.filter(inp_mask & is_transfer)['MDC'].alias('MDC')
])

for col in elix_28:
    s_tr = s_tr.with_columns(g23.filter(inp_mask & is_transfer)[col].alias(col))

X_tr_base = sc.transform(s_tr.select(f_base + elix_28).fill_null(0).to_pandas().values)
mdcs_tr = s_tr['MDC'].to_pandas().values.reshape(-1, 1)
X_mdc_tr = ohe.transform(mdcs_tr)
X_tr_m2mdc = np.hstack([X_tr_base, X_mdc_tr])

p_tr = m2_mdc.predict_proba(X_tr_m2mdc)[:, 1] * factor
transf_df = transf_df.with_columns(pl.Series('P_expected', p_tr))

# Deterministic Linkage (Script 19 standard definition)
all_episodes = s23.select(['ID_EPISODIO', 'COD_HOSPITAL', 'CIP_ENCRIPTADO', 'FECHA_INGRESO', 'FECHAALTA', 'TIPOALTA']).with_columns(
    (pl.col('TIPOALTA') == 'FALLECIDO').cast(pl.Int32).alias('FALLECIDO_RECEPTOR')
)

linked = transf_df.filter(pl.col('CIP_ENCRIPTADO').is_not_null() & (pl.col('CIP_ENCRIPTADO') != '')).join(
    all_episodes.filter(pl.col('CIP_ENCRIPTADO').is_not_null() & (pl.col('CIP_ENCRIPTADO') != '')),
    on='CIP_ENCRIPTADO',
    suffix='_rec'
).filter(
    (pl.col('ID_EPISODIO') != pl.col('ID_EPISODIO_rec')) &
    (pl.col('FECHA_INGRESO') >= pl.col('FECHAALTA')) &
    (pl.col('FECHA_INGRESO') <= pl.col('FECHAALTA') + pl.duration(days=30))
).sort(['ID_EPISODIO', 'FECHA_INGRESO']).group_by('ID_EPISODIO').first()

transf_df = transf_df.join(
    linked.select(['ID_EPISODIO', 'FALLECIDO_RECEPTOR']),
    on='ID_EPISODIO',
    how='left'
).with_columns(
    pl.col('FALLECIDO_RECEPTOR').is_not_null().alias('es_enlazado')
)

print(f"Total enlazados con definicion estandar: {transf_df['es_enlazado'].sum():,}")

# Hospital base in 2023
h_base = df_23.group_by('COD_HOSPITAL').agg([
    pl.len().alias('N_base'),
    pl.col('MORTALIDAD_BINARIA').sum().alias('O_base'),
    pl.col('E_M2MDC').sum().alias('E_base')
])

# Transfer aggregates
h_tr = transf_df.group_by('COD_HOSPITAL').agg([
    pl.len().alias('N_tr'),
    pl.col('es_enlazado').cast(pl.Int32).sum().alias('N_link'),
    pl.col('FALLECIDO_RECEPTOR').fill_null(0).sum().alias('O_link'),
    pl.when(pl.col('es_enlazado')).then(pl.col('P_expected')).otherwise(0).sum().alias('E_link'),
    pl.when(~pl.col('es_enlazado')).then(1).otherwise(0).sum().alias('N_nolink'),
    pl.when(~pl.col('es_enlazado')).then(pl.col('P_expected')).otherwise(0).sum().alias('E_nolink')
])

h_all = h_base.join(h_tr, on='COD_HOSPITAL', how='left').with_columns([
    pl.col('N_tr').fill_null(0),
    pl.col('N_link').fill_null(0),
    pl.col('O_link').fill_null(0),
    pl.col('E_link').fill_null(0.0),
    pl.col('N_nolink').fill_null(0),
    pl.col('E_nolink').fill_null(0.0)
]).join(cat.select(['cod_hospital', 'nombre_oficial', 'es_monografico']), left_on='COD_HOSPITAL', right_on='cod_hospital', how='left')

h_agudos = h_all.filter(pl.col('es_monografico') == 'NO')

tau = 0.1948
mu = -0.0359
c = np.log(1.10)
z_thresh = 1.64485 # P >= 0.95

def get_eb(O, E):
    oe = O / E
    y = np.log(oe)
    v = 1.0 / O
    B = (tau**2) / (tau**2 + v)
    theta = B * y + (1 - B) * mu
    se = np.sqrt(B * v)
    z = (theta - c) / se
    p = stats.norm.cdf(z)
    return oe, z, p

base_eb = [get_eb(r['O_base'], r['E_base']) for r in h_agudos.iter_rows(named=True)]
h_agudos = h_agudos.with_columns([
    pl.Series('OE_base', [x[0] for x in base_eb]),
    pl.Series('Z_base', [x[1] for x in base_eb]),
    pl.Series('P_base', [x[2] for x in base_eb])
])

def calc_byar(O, E):
    if E == 0 or O == 0:
        return 0.0, 0.0, 0.0
    lam = O / E
    low = (O / E) * (1 - 1/(9*O) - 1.96 / (3 * np.sqrt(O)))**3
    high = ((O + 1) / E) * (1 - 1/(9*(O+1)) + 1.96 / (3 * np.sqrt(O+1)))**3
    return lam, max(0.0, low), high

byar_res = [calc_byar(r['O_link'], r['E_link']) for r in h_agudos.iter_rows(named=True)]
h_agudos = h_agudos.with_columns([
    pl.Series('lambda_obs', [x[0] for x in byar_res]),
    pl.Series('byar_low', [x[1] for x in byar_res]),
    pl.Series('byar_high', [x[2] for x in byar_res])
])

# Tipping point calculation
def get_tp(r):
    if r['P_base'] >= 0.95:
        return 0.0, 0.0
    if r['E_nolink'] <= 0.01:
        return 999.0, 999.0
    low, high = 0.0, 50.0
    for _ in range(50):
        mid = (low + high) / 2.0
        O_new = r['O_base'] + r['O_link'] + mid * r['E_nolink']
        E_new = r['E_base'] + r['E_link'] + r['E_nolink']
        _, z_new, _ = get_eb(O_new, E_new)
        if z_new < z_thresh:
            low = mid
        else:
            high = mid
    mort = (high * r['E_nolink'] / r['N_nolink']) * 100.0 if r['N_nolink'] > 0 else 0.0
    return high, mort

tp_res = [get_tp(r) for r in h_agudos.iter_rows(named=True)]
h_agudos = h_agudos.with_columns([
    pl.Series('lambda_star', [x[0] for x in tp_res]),
    pl.Series('mort_star_pct', [x[1] for x in tp_res])
])

print("\n=== 1. LOS 5 HOSPITALES REVISADOS CON ENLACE ESTÁNDAR BAJO M2+MDC ===")
for hid in ['114101', '113180', '112100', '113100', '116110', '128109']:
    r = h_agudos.filter(pl.col('COD_HOSPITAL') == hid).to_dicts()[0]
    # Byar stress
    O_stress = r['O_base'] + r['O_link'] + r['byar_high'] * r['E_nolink']
    E_stress = r['E_base'] + r['E_link'] + r['E_nolink']
    oe_str, _, p_str = get_eb(O_stress, E_stress)
    
    # Point estimate stress
    O_pt = r['O_base'] + r['O_link'] + r['lambda_obs'] * r['E_nolink']
    oe_pt, _, p_pt = get_eb(O_pt, E_stress)
    
    print(f"{r['COD_HOSPITAL']} | {r['nombre_oficial'][:30]:<30} | Enlaz={r['N_link']:3d}/{r['N_tr']:4d} | O_link={r['O_link']:2d}, E_link={r['E_link']:4.1f} | lam={r['lambda_obs']:.2f}x [{r['byar_low']:.2f}-{r['byar_high']:.2f}] | P_base={r['P_base']:.3f} | lam*={r['lambda_star']:.2f}x | P_pt={p_pt:.3f} | P_byarHigh={p_str:.3f}")

print("\n=== 2. HOSPITALES NO ALERTADOS EN M2+MDC (K=52) ORDENADOS POR VULNERABILIDAD ===")
no_alert = h_agudos.filter(pl.col('P_base') < 0.95).sort('lambda_star')
print(f"Total No Alertados: {no_alert.height}")
for r in no_alert.head(15).iter_rows(named=True):
    is_vuln = "VULNERABLE (lam* < Byar_High)" if r['lambda_star'] < r['byar_high'] else "No vulnerable"
    print(f"{r['COD_HOSPITAL']} | {r['nombre_oficial'][:30]:<30} | N_tr={r['N_tr']:4d} | NoLink={r['N_nolink']:4d} (E={r['E_nolink']:4.1f}) | lam_obs={r['lambda_obs']:.2f}x [High={r['byar_high']:.2f}x] | lam*={r['lambda_star']:.2f}x (Mort={r['mort_star_pct']:4.1f}%) | {is_vuln}")

