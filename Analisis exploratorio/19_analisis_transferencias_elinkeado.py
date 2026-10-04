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
cat = pl.read_csv(CAT_PATH)

# Identificar traslados de adultos inpatient
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

# Cargar M3 para predecir P en los pacientes trasladados
df_dev = pl.read_parquet("/home/felipe/Documentos/Proyecto final/Tesis/scratch/dev_eval_cache.parquet")
h_ndx_dev = df_dev.group_by('COD_HOSPITAL').agg(pl.col('N_DX').mean().alias('mean_ndx'))
g_mean_ndx = df_dev['N_DX'].mean()

df_dev = df_dev.join(h_ndx_dev, on='COD_HOSPITAL', how='left').with_columns(
    (pl.col('N_DX') / pl.col('mean_ndx').fill_null(g_mean_ndx)).alias('R_DX')
)

elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]
f_base = ['EDAD_ANIOS', 'INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL', 'SEXO_MASCULINO']

sc28 = StandardScaler()
X_dev = sc28.fit_transform(df_dev.select(f_base + elix_28).fill_null(0).to_pandas().values)
sp = SplineTransformer(n_knots=4, degree=3, include_bias=False)
sp_dev = sp.fit_transform(df_dev.select(['R_DX']).to_pandas().values)
X_dev_m3 = np.hstack([X_dev, sp_dev])
y_dev = df_dev['MORTALIDAD_BINARIA'].to_numpy()

m3 = LogisticRegression(max_iter=200, random_state=42).fit(X_dev_m3, y_dev)

# Predecir sobre todos los traslados
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

X_tr = sc28.transform(s_tr.select(f_base + elix_28).fill_null(0).to_pandas().values)
sp_tr = sp.transform(s_tr.select(['R_DX']).to_pandas().values)
X_tr_m3 = np.hstack([X_tr, sp_tr])

p_tr = m3.predict_proba(X_tr_m3)[:, 1]

# Recalibración
df_23 = pl.read_parquet("/home/felipe/Documentos/Proyecto final/Tesis/scratch/eval_23_cache.parquet")
df_23 = df_23.join(h_ndx_dev, on='COD_HOSPITAL', how='left').with_columns(
    (pl.col('N_DX') / pl.col('mean_ndx').fill_null(g_mean_ndx)).alias('R_DX')
)
X_23 = sc28.transform(df_23.select(f_base + elix_28).fill_null(0).to_pandas().values)
sp_23 = sp.transform(df_23.select(['R_DX']).to_pandas().values)
p_23 = m3.predict_proba(np.hstack([X_23, sp_23]))[:, 1]
k_fact = df_23['MORTALIDAD_BINARIA'].sum() / np.sum(p_23)

p_tr_cal = p_tr * k_fact
transf_df = transf_df.with_columns(pl.Series('P_expected', p_tr_cal))

# Enlace determinístico
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

# Marcar enlace en transf_df
transf_df = transf_df.join(
    linked.select(['ID_EPISODIO', 'FALLECIDO_RECEPTOR']),
    on='ID_EPISODIO',
    how='left'
).with_columns(
    pl.col('FALLECIDO_RECEPTOR').is_not_null().alias('es_enlazado')
)

print(f"Total enlazados: {transf_df['es_enlazado'].sum():,}")
print(f"Riesgo esperado promedio en trasladados: {transf_df['P_expected'].mean()*100:.2f}%")
print(f"Riesgo esperado promedio en enlazados: {transf_df.filter(pl.col('es_enlazado'))['P_expected'].mean()*100:.2f}%")
print(f"Riesgo esperado promedio en no enlazados: {transf_df.filter(~pl.col('es_enlazado'))['P_expected'].mean()*100:.2f}%")

# Agrupar por hospital emisor
h_transf = transf_df.group_by('COD_HOSPITAL').agg([
    pl.len().alias('n_emitidos'),
    pl.col('es_enlazado').cast(pl.Int32).sum().alias('n_enlazados'),
    pl.col('FALLECIDO_RECEPTOR').sum().alias('muertes_enlazadas'),
    pl.when(pl.col('es_enlazado')).then(pl.col('P_expected')).otherwise(None).sum().alias('E_enlazado'),
    pl.when(~pl.col('es_enlazado')).then(pl.col('P_expected')).otherwise(None).sum().alias('E_no_enlazado'),
    pl.col('P_expected').sum().alias('E_total_transf')
]).with_columns([
    (pl.col('n_emitidos') - pl.col('n_enlazados')).alias('n_no_enlazados'),
    (pl.col('muertes_enlazadas').fill_null(0) / pl.col('E_enlazado')).alias('lambda_obs_E')
]).join(
    cat.select(['cod_hospital', 'nombre_oficial']), left_on='COD_HOSPITAL', right_on=pl.col('cod_hospital').cast(pl.String), how='left'
)

# Intervalo exacto de Poisson para O/E (Byar / exact chi2)
def byar_ci(o, e, conf=0.95):
    if o == 0 or e == 0 or o is None or e is None:
        return 0.0, 0.0
    alpha = 1 - conf
    lo = stats.chi2.ppf(alpha / 2, 2 * o) / (2 * e)
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * (o + 1)) / (2 * e)
    return lo, hi

ci_lo, ci_hi = [], []
for r in h_transf.iter_rows(named=True):
    o = r['muertes_enlazadas']
    e = r['E_enlazado']
    l, h = byar_ci(o, e)
    ci_lo.append(l)
    ci_hi.append(h)

h_transf = h_transf.with_columns([
    pl.Series('lambda_lo', ci_lo),
    pl.Series('lambda_hi', ci_hi)
])

print("\n" + "=" * 135)
print("TABLA DE TRASLADOS CON LAMBDA BASADO EN E (RIESGO ESPERADO MODELO M3) E INTERVALO 95% EXACTO")
print("=" * 135)
print(f"{'Cod':<7} | {'Nombre':<30} | {'Emit':>5} | {'Enlaz':>5} | {'O_link':>6} | {'E_link':>6} | {'lambda_obs (O/E)':>16} | {'IC 95% Byar':>17} | {'E_no_link':>9} | {'No_link':>7}")
print("-" * 135)
for r in h_transf.sort('n_emitidos', descending=True).head(30).iter_rows(named=True):
    ci_s = f"[{r['lambda_lo']:.2f}x - {r['lambda_hi']:.2f}x]"
    lam_s = f"{r['lambda_obs_E']:.2f}x" if r['lambda_obs_E'] is not None else "N/A"
    print(f"{r['COD_HOSPITAL']:<7} | {r['nombre_oficial'][:30]:<30} | {r['n_emitidos']:5d} | {r['n_enlazados']:5d} | {r['muertes_enlazadas']:6.0f} | {r['E_enlazado']:6.1f} | {lam_s:>16} | {ci_s:>17} | {r['E_no_enlazado']:9.1f} | {r['n_no_enlazados']:7d}")

# 3. Punto de quiebre específico de Curanilahue y estrés
print("\n" + "=" * 90)
print("=== 3. ANÁLISIS DE ESTRÉS RIGUROSO PARA CURANILAHUE (128109) ===")
print("=" * 90)
cur = h_transf.filter(pl.col('COD_HOSPITAL') == '128109').row(0, named=True)
print(f"Curanilahue: Emitidos={cur['n_emitidos']}, Enlazados={cur['n_enlazados']}, O_link={cur['muertes_enlazadas']}, E_link={cur['E_enlazado']:.1f}")
print(f"  lambda_obs = {cur['lambda_obs_E']:.2f}x [IC 95%: {cur['lambda_lo']:.2f}x - {cur['lambda_hi']:.2f}x]")
print(f"  No enlazados = {cur['n_no_enlazados']}, E_no_link = {cur['E_no_enlazado']:.1f}")

# Cargar O y E base de Curanilahue en M3 (agudos generales)
# O_base = 136, E_base = 119.5
tau2_ag = 0.2048**2
mu_ag = 0.0224

def eval_curanilahue_unlinked(lambda_no_link):
    o_link = cur['muertes_enlazadas']
    e_link = cur['E_enlazado']
    o_nolink = lambda_no_link * cur['E_no_enlazado']
    e_nolink = cur['E_no_enlazado']
    
    o_tot = 136 + o_link + o_nolink
    e_tot = 119.5 + e_link + e_nolink
    oe_tot = o_tot / e_tot
    
    v = 1.0 / o_tot
    b = tau2_ag / (tau2_ag + v)
    lth = b * np.log(oe_tot) + (1.0 - b) * mu_ag
    se = np.sqrt(tau2_ag * v / (tau2_ag + v))
    z = (lth - np.log(1.10)) / se
    p = 1.0 - stats.norm.cdf(-z)
    return o_tot, e_tot, oe_tot, p

print("\nEvaluación de escenarios de estrés basados en múltiplos de E sobre los no enlazados de Curanilahue:")
for mult in [0.5, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
    o_t, e_t, oe_t, p = eval_curanilahue_unlinked(mult)
    status = "ALERTA" if p >= 0.95 else "PROMEDIO"
    print(f"  lambda_no_link = {mult:.2f}x E (Muertes no enlazadas: {mult*cur['E_no_enlazado']:4.1f} de {cur['n_no_enlazados']}): O_tot={o_t:5.1f}, E_tot={e_t:5.1f}, OE = {oe_t:.3f}, P(>1.10) = {p:.4f} --> {status}")

# Punto de quiebre exacto
found = False
for mult in np.arange(1.0, 5.0, 0.01):
    o_t, e_t, oe_t, p = eval_curanilahue_unlinked(mult)
    if p >= 0.95:
        print(f"\n>>> PUNTO DE QUIEBRE EXACTO PARA CURANILAHUE EN NO ENLAZADOS:")
        print(f"    lambda_no_link* = {mult:.2f}x del riesgo esperado E_no_link")
        print(f"    Muertes requeridas en los 301 no enlazados: {mult*cur['E_no_enlazado']:.1f} defunciones (Tasa = {mult*cur['E_no_enlazado']/cur['n_no_enlazados']*100:.2f}%)")
        print(f"    Mortalidad observada en enlazados fue solo {cur['muertes_enlazadas']/cur['n_enlazados']*100:.2f}% (lambda_obs = {cur['lambda_obs_E']:.2f}x)")
        found = True
        break

