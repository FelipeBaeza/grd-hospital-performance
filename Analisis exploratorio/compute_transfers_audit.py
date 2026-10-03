import polars as pl
import numpy as np
import scipy.stats as stats
from pathlib import Path

SILVER_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/silver")
GOLD_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/gold")
CAT_PATH = Path("/home/felipe/Documentos/Proyecto final/Tesis/config/catalogo_hospitales_procedencia.csv")

s23 = pl.read_parquet(SILVER_DIR / "silver_2023.parquet")
g23 = pl.read_parquet(GOLD_DIR / "gold_2023.parquet")
cat = pl.read_csv(CAT_PATH)

# Revisar total bruto de traslados en Silver 2023
is_transfer_raw = s23['TIPOALTA'].str.contains('DERIVACIÓN OTRO HOSPITAL|DERIVACIÓN INST. PRIVADA')
print(f"Total bruto traslados Silver 2023: {is_transfer_raw.sum():,}")

# Identificar traslados emitidos en 2023 (inpatient adultos)
c_ex01 = (g23['IR_29301_COD_GRD'].is_null()) | (g23['MDC'] == 0) | (g23['IR_29301_COD_GRD'].str.starts_with('99'))
c_ex02 = (~c_ex01) & (g23['MDC'] == 15)
c_ex03 = (~c_ex01) & (~c_ex02) & (g23['MDC'] == 14)
c_ex04 = (~c_ex01) & (~c_ex02) & (~c_ex03) & ((g23['EDAD_ANIOS'] < 18) | (g23['EDAD_ANIOS'].is_null()))
c_ex05 = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (s23['TIPO_ACTIVIDAD'] != 'HOSPITALIZACIÓN')
inp_mask = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (~c_ex05)

# Traslados a analizar (inpatient adulto)
transf_episodes = s23.filter(inp_mask & is_transfer_raw).select([
    'ID_EPISODIO', 'COD_HOSPITAL', 'CIP_ENCRIPTADO', 'FECHAALTA'
])
print(f"Total traslados inpatient adultos 2023: {len(transf_episodes):,}")

# Todos los episodios 2023 para enlace
all_episodes = s23.select(['ID_EPISODIO', 'COD_HOSPITAL', 'CIP_ENCRIPTADO', 'FECHA_INGRESO', 'FECHAALTA', 'TIPOALTA']).with_columns(
    (pl.col('TIPOALTA') == 'FALLECIDO').cast(pl.Int32).alias('FALLECIDO_RECEPTOR')
)

# Join por CIP_ENCRIPTADO
linked = transf_episodes.filter(pl.col('CIP_ENCRIPTADO').is_not_null() & (pl.col('CIP_ENCRIPTADO') != '')).join(
    all_episodes.filter(pl.col('CIP_ENCRIPTADO').is_not_null() & (pl.col('CIP_ENCRIPTADO') != '')),
    on='CIP_ENCRIPTADO',
    suffix='_rec'
).filter(
    (pl.col('ID_EPISODIO') != pl.col('ID_EPISODIO_rec')) &
    (pl.col('FECHA_INGRESO') >= pl.col('FECHAALTA')) &
    (pl.col('FECHA_INGRESO') <= pl.col('FECHAALTA') + pl.duration(days=30))
).sort(['ID_EPISODIO', 'FECHA_INGRESO']).group_by('ID_EPISODIO').first()

print(f"Traslados exitosamente enlazados (<=30 días): {len(linked):,} ({len(linked)/len(transf_episodes)*100:.1f}%)")
print(f"No enlazados: {len(transf_episodes) - len(linked):,} ({(len(transf_episodes) - len(linked))/len(transf_episodes)*100:.1f}%)")
print(f"Muertes observadas en receptor: {linked['FALLECIDO_RECEPTOR'].sum():,} ({linked['FALLECIDO_RECEPTOR'].mean()*100:.2f}%)")

# Agrupar por hospital emisor
h_transf = transf_episodes.group_by('COD_HOSPITAL').agg(pl.len().alias('n_emitidos')).join(
    linked.group_by('COD_HOSPITAL').agg([
        pl.len().alias('n_enlazados'),
        pl.col('FALLECIDO_RECEPTOR').sum().alias('muertes_enlazadas')
    ]), on='COD_HOSPITAL', how='left'
).with_columns([
    pl.col('n_enlazados').fill_null(0),
    pl.col('muertes_enlazadas').fill_null(0),
    (pl.col('n_emitidos') - pl.col('n_enlazados').fill_null(0)).alias('n_no_enlazados')
]).with_columns([
    (pl.col('n_enlazados') / pl.col('n_emitidos') * 100).alias('pct_enlace'),
    pl.when(pl.col('n_enlazados') > 0).then(pl.col('muertes_enlazadas') / pl.col('n_enlazados') * 100).otherwise(0.0).alias('tasa_obs_pct'),
    pl.when(pl.col('n_enlazados') > 0).then((pl.col('muertes_enlazadas') / pl.col('n_enlazados')) / 0.043467).otherwise(0.0).alias('lambda_obs')
]).join(
    cat.select(['cod_hospital', 'nombre_oficial']), left_on='COD_HOSPITAL', right_on=pl.col('cod_hospital').cast(pl.String), how='left'
)

# Wilson score interval para tasa
def wilson_ci(k, n, conf=0.95):
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - conf)/2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - spread) * 100, min(1.0, center + spread) * 100

ci_low, ci_high = [], []
for r in h_transf.iter_rows(named=True):
    lo, hi = wilson_ci(r['muertes_enlazadas'], r['n_enlazados'])
    ci_low.append(lo)
    ci_high.append(hi)

h_transf = h_transf.with_columns([
    pl.Series('ci_low', ci_low),
    pl.Series('ci_high', ci_high)
])

print("\n" + "=" * 135)
print("TABLA GENERAL DE TRASLADOS EMITIDOS, ENLACE Y MORTALIDAD EN RECEPTOR (2023)")
print("=" * 135)
print(f"{'Cod':<7} | {'Nombre':<30} | {'Emitidos':>8} | {'Enlazados':>9} | {'% Enlace':>8} | {'O_rec':>5} | {'Tasa Obs %':>10} | {'IC 95% Wilson':>17} | {'lambda_obs':>10} | {'No Enlaz':>8}")
print("-" * 135)
for r in h_transf.sort('n_emitidos', descending=True).head(35).iter_rows(named=True):
    ci_str = f"[{r['ci_low']:.1f}% - {r['ci_high']:.1f}%]"
    print(f"{r['COD_HOSPITAL']:<7} | {r['nombre_oficial'][:30]:<30} | {r['n_emitidos']:8d} | {r['n_enlazados']:9d} | {r['pct_enlace']:7.1f}% | {r['muertes_enlazadas']:5d} | {r['tasa_obs_pct']:9.2f}% | {ci_str:>17} | {r['lambda_obs']:9.2f}x | {r['n_no_enlazados']:8d}")

# Detalle Curanilahue
print("\n" + "=" * 90)
print("DETALLE ESPECÍFICO CURANILAHUE (128109):")
cur_t = h_transf.filter(pl.col('COD_HOSPITAL') == '128109').row(0, named=True)
print(f"  Emitidos: {cur_t['n_emitidos']}, Enlazados: {cur_t['n_enlazados']} ({cur_t['pct_enlace']:.1f}%), No enlazados: {cur_t['n_no_enlazados']}")
print(f"  Muertes enlazadas: {cur_t['muertes_enlazadas']} ({cur_t['tasa_obs_pct']:.2f}%, IC 95%: [{cur_t['ci_low']:.1f}% - {cur_t['ci_high']:.1f}%])")
print(f"  lambda_obs = {cur_t['lambda_obs']:.2f}x (comparado con tasa red 4.35%)")

