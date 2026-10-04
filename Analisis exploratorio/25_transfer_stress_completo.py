import polars as pl
import numpy as np
import scipy.stats as stats
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SILVER_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/silver")
GOLD_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/gold")
CAT_PATH = Path("/home/felipe/Documentos/Proyecto final/Tesis/config/catalogo_hospitales_procedencia.csv")

s23 = pl.read_parquet(SILVER_DIR / "silver_2023.parquet")
g23 = pl.read_parquet(GOLD_DIR / "gold_2023.parquet")
cat = pl.read_csv(CAT_PATH).with_columns(pl.col('cod_hospital').cast(pl.String))

# 1. Filtros de cohorte evaluable inpatient adultos
c_ex01 = (g23['IR_29301_COD_GRD'].is_null()) | (g23['MDC'] == 0) | (g23['IR_29301_COD_GRD'].str.starts_with('99'))
c_ex02 = (~c_ex01) & (g23['MDC'] == 15)
c_ex03 = (~c_ex01) & (~c_ex02) & (g23['MDC'] == 14)
c_ex04 = (~c_ex01) & (~c_ex02) & (~c_ex03) & ((g23['EDAD_ANIOS'] < 18) | (g23['EDAD_ANIOS'].is_null()))
c_ex05 = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (s23['TIPO_ACTIVIDAD'] != 'HOSPITALIZACIÓN')
inp_mask = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (~c_ex05)

is_transfer = s23['TIPOALTA'].str.contains('DERIVACIÓN OTRO HOSPITAL|DERIVACIÓN INST. PRIVADA')

# 2. Entrenar M2 (Modelo Primario) en Dev
df_dev = pl.read_parquet("/home/felipe/Documentos/Proyecto final/Tesis/scratch/dev_eval_cache.parquet")
df_23 = pl.read_parquet("/home/felipe/Documentos/Proyecto final/Tesis/scratch/eval_23_cache.parquet")

elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]
f_base = ['EDAD_ANIOS', 'INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL', 'SEXO_MASCULINO']

sc28 = StandardScaler()
X_dev = sc28.fit_transform(df_dev.select(f_base + elix_28).fill_null(0).to_pandas().values)
y_dev = df_dev['MORTALIDAD_BINARIA'].to_numpy()

m2 = LogisticRegression(max_iter=200, random_state=42).fit(X_dev, y_dev)

# Predecir en cohorte evaluable 2023
X_23 = sc28.transform(df_23.select(f_base + elix_28).fill_null(0).to_pandas().values)
p_23 = m2.predict_proba(X_23)[:, 1]
factor = df_23['MORTALIDAD_BINARIA'].sum() / p_23.sum()
df_23 = df_23.with_columns(pl.Series('E_M2', p_23 * factor))

# Predecir en los traslados
diag_cols = [f'DIAGNOSTICO{i}' for i in range(2, 36)]
n_dx_expr = pl.sum_horizontal([
    pl.when(pl.col(c).is_not_null() & (pl.col(c) != '') & (pl.col(c) != '0')).then(1).otherwise(0)
    for c in diag_cols
])

s_tr = s23.filter(inp_mask & is_transfer).with_columns([
    n_dx_expr.alias('N_DX'),
    s23.filter(inp_mask & is_transfer)['TIPO_INGRESO'].str.to_uppercase().str.contains('URGEN').cast(pl.Int32).alias('INGRESO_URGENCIA'),
    s23.filter(inp_mask & is_transfer)['SERVICIOINGRESO'].str.to_uppercase().str.contains('UCI|UTI|CRITIC|INTENS|INTERMED').cast(pl.Int32).alias('INGRESO_CRITICO'),
    (s23.filter(inp_mask & is_transfer)['HOSPPROCEDENCIA'].is_not_null() | s23.filter(inp_mask & is_transfer)['TIPO_PROCEDENCIA'].str.to_uppercase().str.contains('DERIV|OTRO|HOSP')).cast(pl.Int32).alias('DERIVADO_OTRO_HOSPITAL'),
    (g23.filter(inp_mask & is_transfer)['SEXO'] == 1).cast(pl.Int32).alias('SEXO_MASCULINO'),
    g23.filter(inp_mask & is_transfer)['EDAD_ANIOS'].alias('EDAD_ANIOS')
])

for col in elix_28:
    s_tr = s_tr.with_columns(g23.filter(inp_mask & is_transfer)[col].alias(col))

X_tr = sc28.transform(s_tr.select(f_base + elix_28).fill_null(0).to_pandas().values)
p_tr = m2.predict_proba(X_tr)[:, 1] * factor
s_tr = s_tr.with_columns(pl.Series('P_M2', p_tr))

# Enlace de traslados a receptor en <= 30 días
admissions = s23.filter(
    (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (~c_ex05)
).select([
    'ID_EPISODIO', 'COD_HOSPITAL', 'CIP_ENCRIPTADO', 'FECHA_INGRESO', 'FECHAALTA', 'TIPOALTA'
]).rename({
    'ID_EPISODIO': 'ID_EPISODIO_REC',
    'COD_HOSPITAL': 'COD_HOSPITAL_REC',
    'FECHA_INGRESO': 'FECHA_INGRESO_REC',
    'FECHAALTA': 'FECHAALTA_REC',
    'TIPOALTA': 'TIPOALTA_REC'
})

tr_with_cip = s_tr.filter(pl.col('CIP_ENCRIPTADO').is_not_null() & (pl.col('CIP_ENCRIPTADO') != ''))
matches = tr_with_cip.join(admissions, on='CIP_ENCRIPTADO', how='inner')
matches = matches.filter(
    (pl.col('COD_HOSPITAL') != pl.col('COD_HOSPITAL_REC')) &
    (pl.col('FECHA_INGRESO_REC') >= pl.col('FECHAALTA')) &
    (pl.col('FECHA_INGRESO_REC') <= pl.col('FECHAALTA') + pl.duration(days=30))
).sort(['ID_EPISODIO', 'FECHA_INGRESO_REC']).unique(subset=['ID_EPISODIO'], keep='first')

matches = matches.with_columns(
    pl.col('TIPOALTA_REC').str.to_uppercase().str.contains('FALLECIDO').cast(pl.Int32).alias('DIED_AT_REC')
)

# Unir con s_tr
s_tr = s_tr.join(matches.select(['ID_EPISODIO', 'DIED_AT_REC', 'ID_EPISODIO_REC']), on='ID_EPISODIO', how='left').with_columns(
    pl.col('ID_EPISODIO_REC').is_not_null().alias('LINKED')
)

# Estadísticas base por hospital evaluable
h_base = df_23.group_by('COD_HOSPITAL').agg([
    pl.len().alias('N_base'),
    pl.col('MORTALIDAD_BINARIA').sum().alias('O_base'),
    pl.col('E_M2').sum().alias('E_base')
])

# Estadísticas de traslados por hospital emisor
tr_stats = s_tr.group_by('COD_HOSPITAL').agg([
    pl.len().alias('N_tr'),
    pl.col('LINKED').sum().alias('N_link'),
    pl.col('DIED_AT_REC').fill_null(0).sum().alias('O_link'),
    pl.when(pl.col('LINKED')).then(pl.col('P_M2')).otherwise(0).sum().alias('E_link'),
    (pl.col('LINKED') == False).sum().alias('N_nolink'),
    pl.when(pl.col('LINKED') == False).then(pl.col('P_M2')).otherwise(0).sum().alias('E_nolink')
])

h_all = h_base.join(tr_stats, on='COD_HOSPITAL', how='left').with_columns([
    pl.col('N_tr').fill_null(0),
    pl.col('N_link').fill_null(0),
    pl.col('O_link').fill_null(0),
    pl.col('E_link').fill_null(0.0),
    pl.col('N_nolink').fill_null(0),
    pl.col('E_nolink').fill_null(0.0)
]).join(cat.select(['cod_hospital', 'nombre_oficial', 'es_monografico']), left_on='COD_HOSPITAL', right_on='cod_hospital', how='left')

# Filtro agudos generales (K=62)
h_agudos = h_all.filter(pl.col('es_monografico') == 'NO')

tau = 0.2048
mu = 0.0224
c = np.log(1.10)
z_thresh = 1.64485 # P >= 0.95

# Calcular estado basal de M2
def get_eb_stats(O, E):
    oe = O / E
    y = np.log(oe)
    v = 1.0 / O
    B = (tau**2) / (tau**2 + v)
    theta = B * y + (1 - B) * mu
    se = np.sqrt(B * v)
    z = (theta - c) / se
    p = stats.norm.cdf(z)
    return oe, z, p

base_eb = [get_eb_stats(r['O_base'], r['E_base']) for r in h_agudos.iter_rows(named=True)]
h_agudos = h_agudos.with_columns([
    pl.Series('OE_base', [x[0] for x in base_eb]),
    pl.Series('Z_base', [x[1] for x in base_eb]),
    pl.Series('P_base', [x[2] for x in base_eb])
])

# Byar 95% CI para lambda_obs
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

# Calcular lambda_quiebre para no alertados
def find_lambda_quiebre(r):
    if r['P_base'] >= 0.95:
        return 0.0, 0.0 # Ya es alerta
    if r['E_nolink'] <= 0.01:
        return 999.0, 999.0
    
    low_lam, high_lam = 0.0, 50.0
    for _ in range(50):
        mid = (low_lam + high_lam) / 2.0
        O_new = r['O_base'] + r['O_link'] + mid * r['E_nolink']
        E_new = r['E_base'] + r['E_link'] + r['E_nolink']
        _, z_new, _ = get_eb_stats(O_new, E_new)
        if z_new < z_thresh:
            low_lam = mid
        else:
            high_lam = mid
    lam_star = high_lam
    mort_rate = (lam_star * r['E_nolink'] / r['N_nolink']) * 100.0 if r['N_nolink'] > 0 else 0.0
    return lam_star, mort_rate

quiebre_res = [find_lambda_quiebre(r) for r in h_agudos.iter_rows(named=True)]
h_agudos = h_agudos.with_columns([
    pl.Series('lambda_quiebre', [x[0] for x in quiebre_res]),
    pl.Series('mort_pct_quiebre', [x[1] for x in quiebre_res])
])

print("=== 1. HOSPITALES CON LAMBDA_OBS SIGNIFICATIVAMENTE > 1.0 (BYAR LOW > 1.0) ===")
sig_hosp = h_agudos.filter(pl.col('byar_low') > 1.0).sort('lambda_obs', descending=True)
print(f"{'Cod':<7} | {'Nombre':<30} | {'N_link':>6} | {'O_link':>6} | {'E_link':>6} | {'lambda_obs':>10} | {'95% Byar CI':>16} | {'P_base':>7}")
print("-" * 100)
for r in sig_hosp.iter_rows(named=True):
    print(f"{r['COD_HOSPITAL']:<7} | {r['nombre_oficial'][:30]:<30} | {r['N_link']:6d} | {r['O_link']:6d} | {r['E_link']:6.1f} | {r['lambda_obs']:9.2f}x | [{r['byar_low']:.2f}, {r['byar_high']:.2f}] | {r['P_base']:7.3f}")

print("\n=== 2. ESCENARIO DE ESTRÉS: APLICAR BYAR HIGH A LOS NO ENLAZADOS DE ESTOS 5 HOSPITALES ===")
print(f"{'Cod':<7} | {'Nombre':<30} | {'N_nolink':>8} | {'E_nolink':>8} | {'Byar High':>9} | {'O_estres':>8} | {'OE_estres':>9} | {'P_estres':>8} | {'¿Alerta?':<8}")
print("-" * 105)
for r in sig_hosp.iter_rows(named=True):
    O_new = r['O_base'] + r['O_link'] + r['byar_high'] * r['E_nolink']
    E_new = r['E_base'] + r['E_link'] + r['E_nolink']
    oe_new, z_new, p_new = get_eb_stats(O_new, E_new)
    alerta = "SÍ" if p_new >= 0.95 else "NO"
    print(f"{r['COD_HOSPITAL']:<7} | {r['nombre_oficial'][:30]:<30} | {r['N_nolink']:8d} | {r['E_nolink']:8.1f} | {r['byar_high']:8.2f}x | {O_new:8.1f} | {oe_new:9.3f} | {p_new:8.3f} | {alerta:<8}")

print("\n=== 3. TIPPING POINT (LAMBDA_QUIEBRE) PARA TODOS LOS HOSPITALES NO ALERTADOS (TOP 20 MÁS VULNERABLES) ===")
no_alert = h_agudos.filter(pl.col('P_base') < 0.95).sort('lambda_quiebre')
print(f"{'Cod':<7} | {'Nombre':<30} | {'N_tr':>5} | {'N_nolink':>8} | {'E_nolink':>8} | {'lambda_obs':>10} | {'lambda*':>8} | {'Mort Quiebre %':>15}")
print("-" * 105)
for r in no_alert.head(20).iter_rows(named=True):
    print(f"{r['COD_HOSPITAL']:<7} | {r['nombre_oficial'][:30]:<30} | {r['N_tr']:5d} | {r['N_nolink']:8d} | {r['E_nolink']:8.1f} | {r['lambda_obs']:9.2f}x | {r['lambda_quiebre']:7.2f}x | {r['mort_pct_quiebre']:14.1f}%")

# Save table to scratch
h_agudos.write_csv("/home/felipe/Documentos/Proyecto final/Tesis/scratch/tabla_completa_estres_traslados.csv")
print("\nTabla completa guardada en /home/felipe/Documentos/Proyecto final/Tesis/scratch/tabla_completa_estres_traslados.csv")
