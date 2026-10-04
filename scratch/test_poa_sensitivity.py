import polars as pl
import numpy as np
import scipy.stats as stats
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder

GOLD_DIR = Path('/home/felipe/Documentos/Proyecto final/Tesis/data/gold')
SILVER_DIR = Path('/home/felipe/Documentos/Proyecto final/Tesis/data/silver')
CAT_PATH = Path('/home/felipe/Documentos/Proyecto final/Tesis/config/catalogo_hospitales_procedencia.csv')
cat = pl.read_csv(CAT_PATH).with_columns(pl.col('cod_hospital').cast(pl.String))

ped_codes = ['109101', '112102', '113130']

def load_with_poa(year):
    print(f"Cargando año {year}...")
    g = pl.read_parquet(GOLD_DIR / f'gold_{year}.parquet')
    s = pl.read_parquet(SILVER_DIR / f'silver_{year}.parquet')
    
    c_ex01 = (g['IR_29301_COD_GRD'].is_null()) | (g['MDC'] == 0) | (g['IR_29301_COD_GRD'].str.starts_with('99'))
    c_ex02 = (~c_ex01) & (g['MDC'] == 15)
    c_ex03 = (~c_ex01) & (~c_ex02) & (g['MDC'] == 14)
    c_ex04 = (~c_ex01) & (~c_ex02) & (~c_ex03) & ((g['EDAD_ANIOS'] < 18) | (g['EDAD_ANIOS'].is_null()))
    c_ex05 = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (s['TIPO_ACTIVIDAD'] != 'HOSPITALIZACIÓN')
    inp_mask = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (~c_ex05)
    
    cens_deriv = s['TIPOALTA'].str.contains('DERIVACIÓN OTRO HOSPITAL|DERIVACIÓN INST. PRIVADA')
    eval_mask = inp_mask & (~cens_deriv) & (~g['COD_HOSPITAL'].is_in(ped_codes))
    
    diag_cols = [f'DIAGNOSTICO{i}' for i in range(2, 36)]
    
    # Flags agudos a suprimir
    has_i26 = s.select(
        pl.sum_horizontal([
            pl.col(c).str.replace(r'\.', '').str.to_uppercase().str.starts_with('I26').fill_null(False)
            for c in diag_cols
        ]) > 0
    ).to_series()
    
    has_g934 = s.select(
        pl.sum_horizontal([
            pl.col(c).str.replace(r'\.', '').str.to_uppercase().str.starts_with('G934').fill_null(False)
            for c in diag_cols
        ]) > 0
    ).to_series()
    
    has_r56 = s.select(
        pl.sum_horizontal([
            pl.col(c).str.replace(r'\.', '').str.to_uppercase().str.starts_with('R56').fill_null(False)
            for c in diag_cols
        ]) > 0
    ).to_series()
    
    has_elix04_other = s.select(
        pl.sum_horizontal([
            pl.col(c).str.replace(r'\.', '').str.to_uppercase().str.starts_with('I27').fill_null(False) |
            pl.col(c).str.replace(r'\.', '').str.to_uppercase().str.starts_with('I28').fill_null(False)
            for c in diag_cols
        ]) > 0
    ).to_series()
    
    has_elix09_other = s.select(
        pl.sum_horizontal([
            pl.col(c).str.replace(r'\.', '').str.to_uppercase().str.slice(0, 3).is_in(['G10', 'G11', 'G12', 'G13', 'G20', 'G21', 'G22', 'G32', 'G35', 'G36', 'G37', 'G40', 'G41']) |
            pl.col(c).str.replace(r'\.', '').str.to_uppercase().str.starts_with('G254').fill_null(False) |
            pl.col(c).str.replace(r'\.', '').str.to_uppercase().str.starts_with('G255').fill_null(False) |
            pl.col(c).str.replace(r'\.', '').str.to_uppercase().str.starts_with('G312').fill_null(False) |
            pl.col(c).str.replace(r'\.', '').str.to_uppercase().str.starts_with('G318').fill_null(False) |
            pl.col(c).str.replace(r'\.', '').str.to_uppercase().str.starts_with('G319').fill_null(False) |
            pl.col(c).str.replace(r'\.', '').str.to_uppercase().str.starts_with('G931').fill_null(False) |
            pl.col(c).str.replace(r'\.', '').str.to_uppercase().str.starts_with('R470').fill_null(False)
            for c in diag_cols
        ]) > 0
    ).to_series()
    
    urg = s['TIPO_INGRESO'].str.to_uppercase().str.contains('URGEN').cast(pl.Int32)
    crit = s['SERVICIOINGRESO'].str.to_uppercase().str.contains('UCI|UTI|CRITIC|INTENS|INTERMED').cast(pl.Int32)
    der = (
        s['HOSPPROCEDENCIA'].is_not_null() & (s['HOSPPROCEDENCIA'] != '0') & (s['HOSPPROCEDENCIA'] != '')
        | s['TIPO_PROCEDENCIA'].str.to_uppercase().str.contains('DERIV|OTRO|HOSP')
    ).cast(pl.Int32)
    sex_m = (g['SEXO'] == 1).cast(pl.Int32)
    
    elix_31 = [f'ELIX_{i:02d}' for i in range(1, 32)]
    
    g_sub = g.filter(eval_mask).select(['ID_EPISODIO', 'COD_HOSPITAL', 'MDC', 'EDAD_ANIOS', 'ESTANCIA_DIAS'] + elix_31)
    s_sub = s.filter(eval_mask).select(['ID_EPISODIO', 'TIPOALTA']).with_columns([
        urg.filter(eval_mask).alias('INGRESO_URGENCIA'),
        crit.filter(eval_mask).alias('INGRESO_CRITICO'),
        der.filter(eval_mask).alias('DERIVADO_OTRO_HOSPITAL'),
        sex_m.filter(eval_mask).alias('SEXO_MASCULINO'),
        has_i26.filter(eval_mask).alias('HAS_I26'),
        has_g934.filter(eval_mask).alias('HAS_G934'),
        has_r56.filter(eval_mask).alias('HAS_R56'),
        has_elix04_other.filter(eval_mask).alias('HAS_ELIX04_OTHER'),
        has_elix09_other.filter(eval_mask).alias('HAS_ELIX09_OTHER'),
        (pl.col('TIPOALTA') == 'FALLECIDO').cast(pl.Int32).alias('MORTALIDAD_BINARIA')
    ])
    
    df = g_sub.join(s_sub, on='ID_EPISODIO')
    
    df = df.with_columns([
        (pl.col('ELIX_04').cast(pl.Boolean) & pl.col('HAS_ELIX04_OTHER')).cast(pl.Int32).alias('ELIX_04_POA'),
        (pl.col('ELIX_09').cast(pl.Boolean) & pl.col('HAS_ELIX09_OTHER')).cast(pl.Int32).alias('ELIX_09_POA')
    ])
    return df

print('1. Cargando 2020-2022 y 2023 con supresion de agudos...')
df_dev = pl.concat([load_with_poa(y) for y in [2020, 2021, 2022]])
df_23 = load_with_poa(2023)

df_dev.write_parquet('/home/felipe/Documentos/Proyecto final/Tesis/scratch/dev_poa_cache.parquet')
df_23.write_parquet('/home/felipe/Documentos/Proyecto final/Tesis/scratch/eval_23_poa_cache.parquet')
print('Caches guardadas con éxito.')
