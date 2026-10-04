import polars as pl
import numpy as np
import scipy.stats as stats
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, SplineTransformer

SILVER_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/silver")
GOLD_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/gold")
CAT_PATH = Path("/home/felipe/Documentos/Proyecto final/Tesis/config/catalogo_hospitales_procedencia.csv")

s23 = pl.read_parquet(SILVER_DIR / "silver_2023.parquet")
g23 = pl.read_parquet(GOLD_DIR / "gold_2023.parquet")
cat = pl.read_csv(CAT_PATH).with_columns(pl.col('cod_hospital').cast(pl.String))

# 1. Filtros cohorte evaluable inpatient adultos
c_ex01 = (g23['IR_29301_COD_GRD'].is_null()) | (g23['MDC'] == 0) | (g23['IR_29301_COD_GRD'].str.starts_with('99'))
c_ex02 = (~c_ex01) & (g23['MDC'] == 15)
c_ex03 = (~c_ex01) & (~c_ex02) & (g23['MDC'] == 14)
c_ex04 = (~c_ex01) & (~c_ex02) & (~c_ex03) & ((g23['EDAD_ANIOS'] < 18) | (g23['EDAD_ANIOS'].is_null()))
c_ex05 = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (s23['TIPO_ACTIVIDAD'] != 'HOSPITALIZACIÓN')
inp_mask = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (~c_ex05)

is_transfer = s23['TIPOALTA'].str.contains('DERIVACIÓN OTRO HOSPITAL|DERIVACIÓN INST. PRIVADA')

# 2. Modelos M2 (Primario) y M3-Total (Sensibilidad)
df_dev = pl.read_parquet("/home/felipe/Documentos/Proyecto final/Tesis/scratch/dev_eval_cache.parquet")
df_23 = pl.read_parquet("/home/felipe/Documentos/Proyecto final/Tesis/scratch/eval_23_cache.parquet")

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
X_dev_base = sc28.fit_transform(df_dev.select(f_base + elix_28).fill_null(0).to_pandas().values)
sp = SplineTransformer(n_knots=4, degree=3, include_bias=False)
sp_dev = sp.fit_transform(df_dev.select(['R_DX']).to_pandas().values)
X_dev_m3 = np.hstack([X_dev_base, sp_dev])

y_dev = df_dev['MORTALIDAD_BINARIA'].to_numpy()

m2 = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_base, y_dev)
m3 = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_m3, y_dev)

# Predecir en cohorte evaluable 2023
X_23_base = sc28.transform(df_23.select(f_base + elix_28).fill_null(0).to_pandas().values)
sp_23 = sp.transform(df_23.select(['R_DX']).to_pandas().values)
X_23_m3 = np.hstack([X_23_base, sp_23])

p23_m2 = m2.predict_proba(X_23_base)[:, 1]
p23_m3 = m3.predict_proba(X_23_m3)[:, 1]

o_tot = df_23['MORTALIDAD_BINARIA'].sum()
df_23 = df_23.with_columns([
    pl.Series('E_M2', p23_m2 * (o_tot / p23_m2.sum())),
    pl.Series('E_M3', p23_m3 * (o_tot / p23_m3.sum()))
])

# Traslados
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

s_tr = s_tr.join(h_ndx_dev, on='COD_HOSPITAL', how='left').with_columns(
    (pl.col('N_DX') / pl.col('mean_ndx').fill_null(g_mean_ndx)).alias('R_DX')
)

X_tr_base = sc28.transform(s_tr.select(f_base + elix_28).fill_null(0).to_pandas().values)
sp_tr = sp.transform(s_tr.select(['R_DX']).to_pandas().values)
X_tr_m3 = np.hstack([X_tr_base, sp_tr])

p_tr_m2 = m2.predict_proba(X_tr_base)[:, 1] * (o_tot / p23_m2.sum())
p_tr_m3 = m3.predict_proba(X_tr_m3)[:, 1] * (o_tot / p23_m3.sum())

s_tr = s_tr.with_columns([
    pl.Series('P_M2', p_tr_m2),
    pl.Series('P_M3', p_tr_m3)
])

# Enlace de traslados a receptor en <= 30 días
admissions = s23.filter(
    (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (~c_ex05)
).select([
    'ID_EPISODIO', 'COD_HOSPITAL', 'CIP_ENCRIPTADO', 'FECHA_INGRESO', 'FECHAALTA', 'TIPOALTA'
]).rename({
    'ID_EPISODIO': 'ID_EPISODIO_REC',
    'COD_HOSPITAL': 'COD_HOSPITAL_REC',
    'FECHA_INGRESO': 'FECHAINGRESO_REC',
    'FECHAALTA': 'FECHAALTA_REC',
    'TIPOALTA': 'TIPOALTA_REC'
})

tr_with_cip = s_tr.filter(pl.col('CIP_ENCRIPTADO').is_not_null() & (pl.col('CIP_ENCRIPTADO') != ''))
matches = tr_with_cip.join(admissions, on='CIP_ENCRIPTADO', how='inner')
matches = matches.filter(
    (pl.col('COD_HOSPITAL') != pl.col('COD_HOSPITAL_REC')) &
    (pl.col('FECHAINGRESO_REC') >= pl.col('FECHAALTA')) &
    (pl.col('FECHAINGRESO_REC') <= pl.col('FECHAALTA') + pl.duration(days=30))
).sort(['ID_EPISODIO', 'FECHAINGRESO_REC']).unique(subset=['ID_EPISODIO'], keep='first')

matches = matches.with_columns(
    pl.col('TIPOALTA_REC').str.to_uppercase().str.contains('FALLECIDO').cast(pl.Int32).alias('DIED_AT_REC')
)

s_tr = s_tr.join(matches.select(['ID_EPISODIO', 'DIED_AT_REC', 'ID_EPISODIO_REC']), on='ID_EPISODIO', how='left').with_columns(
    pl.col('ID_EPISODIO_REC').is_not_null().alias('LINKED')
)

# EB parameters (agudos generales)
tau = 0.2048
mu = 0.0224
c = np.log(1.10)
z_thresh = 1.64485

def calc_eb(o, e):
    oe = o / e
    y = np.log(oe)
    v = 1.0 / o
    B = (tau**2) / (tau**2 + v)
    theta = B * y + (1 - B) * mu
    se = np.sqrt(B * v)
    z = (theta - c) / se
    p = stats.norm.cdf(z)
    return oe, z, p

def calc_byar(O, E):
    if E == 0 or O == 0:
        return 0.0, 0.0, 0.0
    lam = O / E
    low = (O / E) * (1 - 1/(9*O) - 1.96 / (3 * np.sqrt(O)))**3
    high = ((O + 1) / E) * (1 - 1/(9*(O+1)) + 1.96 / (3 * np.sqrt(O+1)))**3
    return lam, max(0.0, low), high

# 5 Hospitales señalados con lambda_obs significativo > 1.0
target_5 = [
    ('114101', 'Complejo Hospitalario Dr. Sótero del Río'),
    ('113180', 'Hospital El Pino (San Bernardo)'),
    ('112100', 'Hospital del Salvador (Providencia)'),
    ('113100', 'Hospital Barros Luco Trudeau (San Miguel)'),
    ('116110', 'Hospital San José de Parral')
]

print("========================================================================================================================")
print("TABLA 1: ANÁLISIS DE ESTRÉS CON LÍMITE SUPERIOR DE BYAR (95% CI) EN LOS 5 HOSPITALES CON LAMBDA_OBS > 1.0 (M2 Y M3)")
print("========================================================================================================================")

for model_name, col_e, col_ptr in [('M2 (PRIMARIO)', 'E_M2', 'P_M2'), ('M3-TOTAL (SENSIBILIDAD)', 'E_M3', 'P_M3')]:
    print(f"\n--- MODELO {model_name} ---")
    print(f"{'Hospital':<36} | {'O_base':>6} | {'E_base':>7} | {'O_link':>6} | {'E_link':>6} | {'lambda_obs [95% Byar]':>23} | {'N_nolink':>8} | {'E_nolink':>8} | {'OE_base':>7} | {'P_base':>6} | {'OE_estres':>9} | {'P_estres':>8} | {'Alerta?':<7}")
    print("-" * 155)
    for hid, name in target_5:
        # Base
        sub_base = df_23.filter(pl.col('COD_HOSPITAL') == hid)
        O_base = sub_base['MORTALIDAD_BINARIA'].sum()
        E_base = sub_base[col_e].sum()
        
        # Transfers
        sub_tr = s_tr.filter(pl.col('COD_HOSPITAL') == hid)
        linked = sub_tr.filter(pl.col('LINKED'))
        unlinked = sub_tr.filter(pl.col('LINKED') == False)
        
        O_link = linked['DIED_AT_REC'].fill_null(0).sum()
        E_link = linked[col_ptr].sum()
        N_nolink = unlinked.height
        E_nolink = unlinked[col_ptr].sum()
        
        lam, low, high = calc_byar(O_link, E_link)
        oe_base, z_base, p_base = calc_eb(O_base, E_base)
        
        # Stress scenario: unlinked die at Byar Upper Limit
        O_stress = O_base + O_link + high * E_nolink
        E_stress = E_base + E_link + E_nolink
        oe_stress, z_stress, p_stress = calc_eb(O_stress, E_stress)
        alerta = "SÍ" if p_stress >= 0.95 else "NO"
        
        lam_str = f"{lam:4.2f}x [{low:4.2f}-{high:4.2f}]"
        print(f"{name[:36]:<36} | {O_base:6d} | {E_base:7.1f} | {O_link:6d} | {E_link:6.1f} | {lam_str:>23} | {N_nolink:8d} | {E_nolink:8.1f} | {oe_base:7.3f} | {p_base:6.3f} | {oe_stress:9.3f} | {p_stress:8.3f} | {alerta:<7}")

# Tipping point across all non-alert hospitals in M2
h_base_all = df_23.group_by('COD_HOSPITAL').agg([
    pl.len().alias('N_base'),
    pl.col('MORTALIDAD_BINARIA').sum().alias('O_base'),
    pl.col('E_M2').sum().alias('E_base_m2'),
    pl.col('E_M3').sum().alias('E_base_m3')
])

tr_stats_all = s_tr.group_by('COD_HOSPITAL').agg([
    pl.len().alias('N_tr'),
    pl.col('LINKED').sum().alias('N_link'),
    pl.col('DIED_AT_REC').fill_null(0).sum().alias('O_link'),
    pl.when(pl.col('LINKED')).then(pl.col('P_M2')).otherwise(0).sum().alias('E_link_m2'),
    pl.when(pl.col('LINKED')).then(pl.col('P_M3')).otherwise(0).sum().alias('E_link_m3'),
    (pl.col('LINKED') == False).sum().alias('N_nolink'),
    pl.when(pl.col('LINKED') == False).then(pl.col('P_M2')).otherwise(0).sum().alias('E_nolink_m2'),
    pl.when(pl.col('LINKED') == False).then(pl.col('P_M3')).otherwise(0).sum().alias('E_nolink_m3')
])

full_tab = h_base_all.join(tr_stats_all, on='COD_HOSPITAL', how='left').with_columns([
    pl.col('N_tr').fill_null(0),
    pl.col('N_link').fill_null(0),
    pl.col('O_link').fill_null(0),
    pl.col('E_link_m2').fill_null(0.0),
    pl.col('E_nolink_m2').fill_null(0.0)
]).join(cat.select(['cod_hospital', 'nombre_oficial', 'es_monografico']), left_on='COD_HOSPITAL', right_on='cod_hospital', how='left')

full_agudos = full_tab.filter(pl.col('es_monografico') == 'NO')

def get_tp(r):
    oe, z, p = calc_eb(r['O_base'], r['E_base_m2'])
    if p >= 0.95:
        return 0.0, 0.0, p
    if r['E_nolink_m2'] <= 0.01:
        return 999.0, 999.0, p
    low, high = 0.0, 50.0
    for _ in range(50):
        mid = (low + high) / 2.0
        O_new = r['O_base'] + r['O_link'] + mid * r['E_nolink_m2']
        E_new = r['E_base_m2'] + r['E_link_m2'] + r['E_nolink_m2']
        _, z_new, _ = calc_eb(O_new, E_new)
        if z_new < z_thresh:
            low = mid
        else:
            high = mid
    mort = (high * r['E_nolink_m2'] / r['N_nolink']) * 100.0 if r['N_nolink'] > 0 else 0.0
    return high, mort, p

tp_results = [get_tp(r) for r in full_agudos.iter_rows(named=True)]
full_agudos = full_agudos.with_columns([
    pl.Series('lambda_star', [x[0] for x in tp_results]),
    pl.Series('mort_star_pct', [x[1] for x in tp_results]),
    pl.Series('P_base_m2', [x[2] for x in tp_results])
])

print("\n========================================================================================================================")
print("TABLA 2: RANKING COMPLETO DE VULNERABILIDAD AL SUBLINKEAJE (PUNTOS DE QUIEBRE LAMBDA* EN MODELO PRIMARIO M2)")
print("========================================================================================================================")
print(f"{'Cod':<7} | {'Establecimiento':<36} | {'N_tr':>5} | {'N_nolink':>8} | {'E_nolink':>8} | {'P_base':>6} | {'lambda*':>8} | {'Mortalidad Quiebre %':>20} | {'Vulnerabilidad':<15}")
print("-" * 125)

non_alert_sorted = full_agudos.filter(pl.col('P_base_m2') < 0.95).sort('lambda_star')
for r in non_alert_sorted.iter_rows(named=True):
    vuln = "Crítica (<2x)" if r['lambda_star'] < 2.0 else ("Alta (2x-3x)" if r['lambda_star'] < 3.0 else ("Moderada (3x-5x)" if r['lambda_star'] < 5.0 else "Inmune (>5x)"))
    print(f"{r['COD_HOSPITAL']:<7} | {r['nombre_oficial'][:36]:<36} | {r['N_tr']:5d} | {r['N_nolink']:8d} | {r['E_nolink_m2']:8.1f} | {r['P_base_m2']:6.3f} | {r['lambda_star']:7.2f}x | {r['mort_star_pct']:19.1f}% | {vuln:<15}")

