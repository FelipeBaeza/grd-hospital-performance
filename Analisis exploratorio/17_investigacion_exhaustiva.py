import polars as pl
import numpy as np
import scipy.stats as stats
from pathlib import Path

SILVER_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/silver")
GOLD_DIR = Path("/home/felipe/Documentos/Proyecto final/Tesis/data/gold")
CAT_PATH = Path("/home/felipe/Documentos/Proyecto final/Tesis/config/catalogo_hospitales_procedencia.csv")

# 1. CIP nulos por año en Silver
print("=== 1. AUDITORÍA DE CIP NULOS POR AÑO (2019-2024) ===")
for y in range(2019, 2025):
    s = pl.read_parquet(SILVER_DIR / f"silver_{y}.parquet")
    n_null = s.filter(pl.col('CIP_ENCRIPTADO').is_null() | (pl.col('CIP_ENCRIPTADO') == '') | (pl.col('CIP_ENCRIPTADO') == '0')).height
    print(f"  Año {y}: Total registros = {len(s):,}, CIP nulos = {n_null:,}")

# 2. Análisis de 'ingreso obstétrico' en cohorte evaluable 2023
print("\n=== 2. INGRESO OBSTÉTRICO VS MDC EN COHORTE EVALUABLE 2023 ===")
s23 = pl.read_parquet(SILVER_DIR / "silver_2023.parquet")
g23 = pl.read_parquet(GOLD_DIR / "gold_2023.parquet")

c_ex01 = (g23['IR_29301_COD_GRD'].is_null()) | (g23['MDC'] == 0) | (g23['IR_29301_COD_GRD'].str.starts_with('99'))
c_ex02 = (~c_ex01) & (g23['MDC'] == 15)
c_ex03 = (~c_ex01) & (~c_ex02) & (g23['MDC'] == 14)
c_ex04 = (~c_ex01) & (~c_ex02) & (~c_ex03) & ((g23['EDAD_ANIOS'] < 18) | (g23['EDAD_ANIOS'].is_null()))
c_ex05 = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (s23['TIPO_ACTIVIDAD'] != 'HOSPITALIZACIÓN')
inp_mask = (~c_ex01) & (~c_ex02) & (~c_ex03) & (~c_ex04) & (~c_ex05)
cens_deriv = s23['TIPOALTA'].str.contains('DERIVACIÓN OTRO HOSPITAL|DERIVACIÓN INST. PRIVADA')
ped_codes = ['109101', '112102', '113130']
eval_mask = inp_mask & (~cens_deriv) & (~g23['COD_HOSPITAL'].is_in(ped_codes))

print(f"Total evaluable 2023: {eval_mask.sum():,}")
obs_ingreso = s23.filter(eval_mask).filter(pl.col('TIPO_INGRESO') == 'OBSTETRICA')
print(f"Episodios con TIPO_INGRESO == 'OBSTETRICA': {len(obs_ingreso):,}")

# Distribución de MDC en esos episodios de ingreso obstétrico
g_obs = g23.filter(eval_mask).filter(s23.filter(eval_mask)['TIPO_INGRESO'] == 'OBSTETRICA')
print("Distribución de MDC en episodios con TIPO_INGRESO == 'OBSTETRICA':")
print(g_obs['MDC'].value_counts().sort('count', descending=True))

# Verificar si hay algún MDC 14 en la cohorte evaluable
n_mdc14 = g23.filter(eval_mask).filter(pl.col('MDC') == 14).height
print(f"MDC == 14 en cohorte evaluable: {n_mdc14} (debe ser 0 por EX03)")

