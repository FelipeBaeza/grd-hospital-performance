import gc
from pathlib import Path
import polars as pl
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression, ElasticNetCV, ElasticNet
from sklearn.preprocessing import StandardScaler

GOLD_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/gold")
SILVER_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/silver")
CAT_PATH = Path("/home/felipe/Documentos/Proyecto final/Tesis/config/catalogo_hospitales_procedencia.csv")
ELIX_PATH = Path("/home/felipe/Documentos/Proyecto final/Tesis/Analisis exploratorio/tabla_canonica_elixhauser_31.csv")

ped_codes = ['109101', '112102', '113130']

# --- PARTE A: ESTADÍA Y REGLA 1-SE ---
print("--- 1. Estadía: Selección con ElasticNet, Regla 1-SE y Etiquetas Canónicas ---")
elix_canon = pl.read_csv(ELIX_PATH)

# Leer solo 2023 para estadia
g23 = pl.read_parquet(GOLD_DIR / "gold_2023.parquet")
s23 = pl.read_parquet(SILVER_DIR / "silver_2023.parquet")

c_ex01 = (g23['IR_29301_COD_GRD'].is_null()) | (g23['MDC'] == 0) | (g23['IR_29301_COD_GRD'].str.starts_with('99'))
c_ex02 = (~c_ex01) & (g23['MDC'] == 15)
c_ex03 = (~c_ex01) & (~c_ex02) & (g23['MDC'] == 14)
c_ex04 = (~c_ex01) & (~c_ex02) & (~c_ex03) & ((g23['EDAD_ANIOS'] < 18) | (g23['EDAD_ANIOS'].is_null()))
c_ex05 = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (s23['TIPO_ACTIVIDAD'] != 'HOSPITALIZACIÓN')
inp_mask = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (~c_ex05)
cens_deriv = s23['TIPOALTA'].str.contains('DERIVACIÓN OTRO HOSPITAL|DERIVACIÓN INST. PRIVADA')
eval_mask = inp_mask & (~cens_deriv) & (~g23['COD_HOSPITAL'].is_in(ped_codes))

elix_28 = [f'ELIX_{i:02d}' for i in range(1, 32) if f'ELIX_{i:02d}' not in ['ELIX_02', 'ELIX_22', 'ELIX_25']]

# Sobrevivientes con estancia > 0
stay_mask = eval_mask & (s23['TIPOALTA'] != 'FALLECIDO') & (g23['ESTANCIA_DIAS'] > 0)
n_stay = stay_mask.sum()
print(f"Población base de estadía 2023 (sobrevivientes con estancia > 0): N = {n_stay:,}")

urg = s23['TIPO_INGRESO'].str.to_uppercase().str.contains('URGEN').cast(pl.Int32)
crit = s23['SERVICIOINGRESO'].str.to_uppercase().str.contains('UCI|UTI|CRITIC|INTENS|INTERMED').cast(pl.Int32)
der = (
    s23['HOSPPROCEDENCIA'].is_not_null() & (s23['HOSPPROCEDENCIA'] != '0') & (s23['HOSPPROCEDENCIA'] != '')
    | s23['TIPO_PROCEDENCIA'].str.to_uppercase().str.contains('DERIV|OTRO|HOSP')
).cast(pl.Int32)
sex_m = (g23['SEXO'] == 1).cast(pl.Int32)
cie3 = s23['DIAGNOSTICO1'].str.replace(r'\.', '').str.slice(0, 3)

df_stay = g23.filter(stay_mask).select(['ID_EPISODIO', 'EDAD_ANIOS', 'ESTANCIA_DIAS'] + elix_28).with_columns([
    urg.filter(stay_mask).alias('INGRESO_URGENCIA'),
    crit.filter(stay_mask).alias('INGRESO_CRITICO'),
    der.filter(stay_mask).alias('DERIVADO_OTRO_HOSPITAL'),
    sex_m.filter(stay_mask).alias('SEXO_MASCULINO'),
    cie3.filter(stay_mask).alias('CIE10_3C')
])

p99 = 54.0
df_stay = df_stay.with_columns(
    pl.when(pl.col('ESTANCIA_DIAS') > p99).then(p99).otherwise(pl.col('ESTANCIA_DIAS')).alias('LOS_TRUNC')
).with_columns(pl.col('LOS_TRUNC').log().alias('LOG_LOS'))

g_mean = df_stay['LOG_LOS'].mean()
cie_stats = df_stay.group_by('CIE10_3C').agg([
    pl.len().alias('n_cie'),
    pl.col('LOG_LOS').mean().alias('mean_cie')
]).with_columns([
    ((pl.col('n_cie') * pl.col('mean_cie') + 10.0 * g_mean) / (pl.col('n_cie') + 10.0)).alias('CIE10_ENC')
])
df_stay = df_stay.join(cie_stats.select(['CIE10_3C', 'CIE10_ENC']), on='CIE10_3C', how='left').with_columns(
    pl.col('CIE10_ENC').fill_null(g_mean)
)

all_stay_feats = [
    'CIE10_ENC', 'INGRESO_URGENCIA', 'INGRESO_CRITICO', 'DERIVADO_OTRO_HOSPITAL', 
    'SEXO_MASCULINO', 'EDAD_ANIOS'
] + elix_28

scaler_s = StandardScaler()
X_s = scaler_s.fit_transform(df_stay.select(all_stay_feats).fill_null(0).to_pandas().values)
y_s = df_stay['LOG_LOS'].to_numpy()

# Liberar memoria de dataframes pesados
del g23, s23, df_stay, cie_stats
gc.collect()

# Submuestra representativa para CV (30,000 casos)
np.random.seed(42)
idx_sub = np.random.choice(len(X_s), size=30000, replace=False)
X_sub = X_s[idx_sub]
y_sub = y_s[idx_sub]

encv = ElasticNetCV(l1_ratio=[0.5, 0.7, 0.9, 1.0], cv=5, random_state=42, max_iter=2000)
encv.fit(X_sub, y_sub)

l1_idx = np.where(encv.l1_ratio == encv.l1_ratio_)[0][0]
mse_path = encv.mse_path_[l1_idx]
mean_mse = mse_path.mean(axis=-1)
se_mse = mse_path.std(axis=-1) / np.sqrt(mse_path.shape[-1])
min_idx = np.argmin(mean_mse)
min_mse = mean_mse[min_idx]
target_mse = min_mse + se_mse[min_idx]

eligible = np.where(mean_mse <= target_mse)[0]
alpha_1se = encv.alphas_[l1_idx][eligible[0]]
print(f"Hiperparámetros CV: Alpha Min = {encv.alpha_:.6f}, L1 = {encv.l1_ratio_:.2f} | Alpha 1-SE = {alpha_1se:.6f}")

m_1se = ElasticNet(alpha=alpha_1se, l1_ratio=encv.l1_ratio_, max_iter=2000, random_state=42)
m_1se.fit(X_sub, y_sub)

df_feats = pl.DataFrame({
    'variable': all_stay_feats,
    'coef_min': encv.coef_,
    'coef_1se': m_1se.coef_
})

df_feats = df_feats.join(elix_canon.select(['categoria_id', 'nombre_comorbilidad']), left_on='variable', right_on='categoria_id', how='left')
df_feats = df_feats.with_columns(
    pl.when(pl.col('variable') == 'CIE10_ENC').then(pl.lit('Diagnóstico Principal (LOHO)'))
      .when(pl.col('variable') == 'INGRESO_URGENCIA').then(pl.lit('Ingreso por Urgencia'))
      .when(pl.col('variable') == 'INGRESO_CRITICO').then(pl.lit('Ingreso a Cama Crítica (UCI/UTI)'))
      .when(pl.col('variable') == 'DERIVADO_OTRO_HOSPITAL').then(pl.lit('Derivado de Otro Hospital'))
      .when(pl.col('variable') == 'SEXO_MASCULINO').then(pl.lit('Sexo Masculino'))
      .when(pl.col('variable') == 'EDAD_ANIOS').then(pl.lit('Edad (Años)'))
      .otherwise(pl.col('nombre_comorbilidad')).alias('nombre_estandar')
)

print(f"\n{'Variable':<12} | {'Nombre Canónico Oficial':<38} | {'Coef Min':>10} | {'Exp(B_min)':>10} | {'Coef 1-SE':>10} | {'Exp(B_1se)':>10} | {'Estado 1-SE':>12}")
print("-" * 125)
for r in df_feats.sort(pl.col('coef_min').abs(), descending=True).iter_rows(named=True):
    st_1se = "RETENIDA" if abs(r['coef_1se']) > 1e-4 else "DESCARTADA"
    print(f"{r['variable']:<12} | {r['nombre_estandar'][:38]:<38} | {r['coef_min']:+10.5f} | {np.exp(r['coef_min']):10.4f} | {r['coef_1se']:+10.5f} | {np.exp(r['coef_1se']):10.4f} | {st_1se:>12}")

n_ret_min = np.sum(np.abs(encv.coef_) > 1e-4)
n_ret_1se = np.sum(np.abs(m_1se.coef_) > 1e-4)
print(f"\nVariables retenidas bajo mínimo error: {n_ret_min} de {len(all_stay_feats)}")
print(f"Variables retenidas bajo regla parsimoniosa 1-SE: {n_ret_1se} de {len(all_stay_feats)}")

del X_s, y_s, X_sub, y_sub
gc.collect()

# --- PARTE B: TIPPING POINT Y TRASLADOS ENLAZADOS ---
print("\n" + "=" * 80)
print("--- 2. Tipping Point (Entrada y Salida) y Desenlaces Empíricos de Traslados Enlazados ---")
print("=" * 80)

# Cargar silver 2023 censurados y receptores de forma ligera
s23_light = pl.read_parquet(
    SILVER_DIR / "silver_2023.parquet",
    columns=['ID_EPISODIO', 'CIP_ENCRIPTADO', 'COD_HOSPITAL', 'FECHA_INGRESO', 'FECHAALTA', 'TIPOALTA', 'TIPO_ACTIVIDAD']
).filter((pl.col('TIPO_ACTIVIDAD') == 'HOSPITALIZACIÓN') & (~pl.col('COD_HOSPITAL').is_in(ped_codes)))

cens_23 = s23_light.filter(pl.col('TIPOALTA').str.contains('DERIVACIÓN OTRO HOSPITAL|DERIVACIÓN INST. PRIVADA'))
eval_23 = s23_light.filter(~pl.col('TIPOALTA').str.contains('DERIVACIÓN OTRO HOSPITAL|DERIVACIÓN INST. PRIVADA'))

print(f"Episodios censurados por derivación en 2023: {len(cens_23):,}")
print(f"Episodios evaluables en 2023: {len(eval_23):,}")

# Linkage con receptores: buscar el CIP_ENCRIPTADO en otro hospital donde FECHA_INGRESO_receptor >= FECHAALTA_emisor
link = cens_23.select(['CIP_ENCRIPTADO', 'COD_HOSPITAL', 'FECHAALTA']).join(
    s23_light.select(['CIP_ENCRIPTADO', 'COD_HOSPITAL', 'FECHA_INGRESO', 'TIPOALTA']),
    on='CIP_ENCRIPTADO',
    how='inner'
).filter(
    (pl.col('COD_HOSPITAL') != pl.col('COD_HOSPITAL_right')) &
    (pl.col('FECHA_INGRESO') >= pl.col('FECHAALTA'))
)

# Quedarse con el primer reingreso dentro de 30 días
link = link.with_columns(
    ((pl.col('FECHA_INGRESO') - pl.col('FECHAALTA')).dt.total_days()).alias('dias_traslado')
).filter(pl.col('dias_traslado') <= 30).sort(['CIP_ENCRIPTADO', 'dias_traslado']).unique(subset=['CIP_ENCRIPTADO', 'COD_HOSPITAL'], keep='first')

print(f"Total traslados enlazados a receptor dentro de 30 días en 2023: {len(link):,}")
mort_link_global = (link['TIPOALTA'] == 'FALLECIDO').mean() * 100
print(f"Mortalidad observada global en receptores en 2023: {mort_link_global:.2f}% ({(link['TIPOALTA'] == 'FALLECIDO').sum():,} fallecidos)")

link_by_hosp = link.group_by('COD_HOSPITAL').agg([
    pl.len().alias('n_enlazados'),
    (pl.col('TIPOALTA') == 'FALLECIDO').cast(pl.Int32).sum().alias('o_receptor')
]).with_columns([
    (pl.col('o_receptor') / pl.col('n_enlazados') * 100).alias('tasa_mort_receptor')
])

cat = pl.read_csv(CAT_PATH)
link_by_hosp = link_by_hosp.join(cat.select(['cod_hospital', 'nombre_oficial']), left_on='COD_HOSPITAL', right_on=pl.col('cod_hospital').cast(pl.String), how='left')

print("\nHospitales con mayor volumen de derivados enlazados y su mortalidad observada en el receptor:")
print(f"{'Cod':<7} | {'Establecimiento Emisor':<35} | {'Enlazados':>9} | {'Muertes Recep':>13} | {'Mort Observada':>15}")
print("-" * 90)
for r in link_by_hosp.sort('n_enlazados', descending=True).head(12).iter_rows(named=True):
    print(f"{r['COD_HOSPITAL']:<7} | {r['nombre_oficial'][:35]:<35} | {r['n_enlazados']:9,d} | {r['o_receptor']:13,d} | {r['tasa_mort_receptor']:14.2f}%")

