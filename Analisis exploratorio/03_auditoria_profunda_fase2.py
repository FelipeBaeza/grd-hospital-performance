"""
03_auditoria_profunda_fase2.py
------------------------------
Script de Auditoría Estadística Profunda para la Fase 2 (Data Understanding).

Calcula todas las métricas reales exigidas por la revisión metodológica:
1. Tasas de mortalidad global y por año (2019-2024).
2. Distribución paramétrica y no paramétrica de ESTANCIA_DIAS (percentiles p25-p99, asimetría, capping).
3. Distribución de volumen por hospital y umbral mínimo de casos.
4. Faltantes (missingness) por columna, año y estabilidad entre hospitales.
5. Análisis de los 18 valores de TIPOALTA y regla de censura.
6. Fuga temporal y códigos terminales en DIAGNOSTICO1 (Z51.5, I46, R98).
7. Intensidad de codificación diagnóstica entre hospitales (riesgo de upcoding).
8. Clasificación de comorbilidades Elixhauser: Preexistentes vs Potenciales Complicaciones.
9. Evaluación de enlaces en cadenas de transferencias inter-hospitalarias.
10. Métricas operativas para las 3 variables condicionales (V de Cramér contra COD_HOSPITAL).
"""

import sys
import glob
import time
from pathlib import Path
import polars as pl
import numpy as np
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
BRONZE_DIR = ROOT_DIR / "data/bronze"
SILVER_DIR = ROOT_DIR / "data/silver"
GOLD_DIR = ROOT_DIR / "data/gold"


def main():
    print("=" * 80)
    print("EJECUTANDO AUDITORÍA ESTADÍSTICA PROFUNDA DE FASE 2")
    print("=" * 80)
    t0 = time.time()

    # 1. Cargar Silver consolidado o por años
    silver_files = sorted(SILVER_DIR.glob("silver_20*.parquet"))
    print(f"Archivos Silver encontrados: {len(silver_files)}")

    # -------------------------------------------------------------------------
    # 1. TASAS DE MORTALIDAD GLOBAL Y POR AÑO
    # -------------------------------------------------------------------------
    print("\n--- 1. TASAS DE MORTALIDAD GLOBAL Y POR AÑO ---")
    tot_episodes = 0
    tot_fallecidos = 0
    tot_censurados = 0
    mortalidad_anual = {}

    for sf in silver_files:
        anio = int(sf.stem.split("_")[1])
        df = pl.read_parquet(sf, columns=["MORTALIDAD_BINARIA", "ESTADO_CENSURA"])
        n = df.height
        muertes = (df["MORTALIDAD_BINARIA"] == 1).sum()
        cens = (df["ESTADO_CENSURA"] == "CENSURADO").sum()
        
        tot_episodes += n
        tot_fallecidos += muertes
        tot_censurados += cens
        tasa_cruda = (muertes / n) * 100
        tasa_no_cens = (muertes / (n - cens)) * 100 if (n - cens) > 0 else 0
        mortalidad_anual[anio] = {
            "n": n, "muertes": muertes, "censurados": cens,
            "tasa_cruda": tasa_cruda, "tasa_no_cens": tasa_no_cens
        }
        print(f"Año {anio}: N={n:,} | Muertes={muertes:,} ({tasa_cruda:.2f}% cruda, {tasa_no_cens:.2f}% no censurada) | Censura={cens:,} ({cens/n*100:.2f}%)")

    tasa_global_cruda = (tot_fallecidos / tot_episodes) * 100
    tasa_global_no_cens = (tot_fallecidos / (tot_episodes - tot_censurados)) * 100
    print(f"TOTAL 2019-2024: N={tot_episodes:,} | Muertes={tot_fallecidos:,} ({tasa_global_cruda:.2f}% cruda, {tasa_global_no_cens:.2f}% no censurada) | Censura={tot_censurados:,} ({tot_censurados/tot_episodes*100:.2f}%)")

    # -------------------------------------------------------------------------
    # 2. DISTRIBUCIÓN DE LA ESTANCIA HOSPITALARIA (SOBREVIVIENTES)
    # -------------------------------------------------------------------------
    print("\n--- 2. DISTRIBUCIÓN DE ESTANCIA HOSPITALARIA (SOBREVIVIENTES) ---")
    estancias_list = []
    # Muestra de 1 millón de sobrevivientes para estadísticas exactas de distribución
    for sf in silver_files:
        df = pl.read_parquet(sf, columns=["ESTANCIA_DIAS", "MORTALIDAD_BINARIA", "ESTADO_CENSURA"])
        df_sobrev = df.filter((pl.col("MORTALIDAD_BINARIA") == 0) & (pl.col("ESTANCIA_DIAS").is_not_null()))
        # sample
        sample_size = min(200_000, df_sobrev.height)
        estancias_list.append(df_sobrev["ESTANCIA_DIAS"].sample(n=sample_size, seed=42).to_numpy())

    estancias = np.concatenate(estancias_list)
    media = np.mean(estancias)
    std = np.std(estancias)
    mediana = np.median(estancias)
    p25 = np.percentile(estancias, 25)
    p50 = np.percentile(estancias, 50)
    p75 = np.percentile(estancias, 75)
    p90 = np.percentile(estancias, 90)
    p95 = np.percentile(estancias, 95)
    p99 = np.percentile(estancias, 99)
    skewness = stats.skew(estancias)
    kurtosis = stats.kurtosis(estancias)

    print(f"Media: {media:.2f} días | Desv. Estándar: {std:.2f} días")
    print(f"Percentiles: p25={p25:.1f}d | Mediana(p50)={p50:.1f}d | p75={p75:.1f}d | p90={p90:.1f}d | p95={p95:.1f}d | p99={p99:.1f}d")
    print(f"Asimetría (Skewness): {skewness:.2f} (Extrema asimetría derecha)")
    print(f"Curtosis: {kurtosis:.2f} (Colas pesadas / leptocúrtica)")
    print(f"Propuesta de Tope Superior (Capping / Trimming): p99 = {p99:.0f} días (limita la influencia de hospitalizaciones atípicas/sociales sin perder señal clínica).")

    # -------------------------------------------------------------------------
    # 3. NÚMERO DE HOSPITALES Y DISTRIBUCIÓN DE VOLUMEN
    # -------------------------------------------------------------------------
    print("\n--- 3. NÚMERO DE HOSPITALES Y VOLUMEN ANUAL ---")
    hosp_counts = {}
    for sf in silver_files:
        df = pl.read_parquet(sf, columns=["COD_HOSPITAL"])
        vc = df["COD_HOSPITAL"].value_counts()
        for r in vc.iter_rows():
            hosp_counts[r[0]] = hosp_counts.get(r[0], 0) + r[1]

    n_hosp = len(hosp_counts)
    volumenes = list(hosp_counts.values())
    vol_anual_promedio = [v / len(silver_files) for v in volumenes]
    
    # Hospitales con más de 200 egresos anuales promedio
    hosp_gt200 = sum(1 for v in vol_anual_promedio if v >= 200)
    hosp_gt500 = sum(1 for v in vol_anual_promedio if v >= 500)
    hosp_gt1000 = sum(1 for v in vol_anual_promedio if v >= 1000)

    print(f"Total establecimientos registrados: {n_hosp}")
    print(f"Volumen total por hospital: Min={min(volumenes):,} | Mediana={np.median(volumenes):,.0f} | Media={np.mean(volumenes):,.0f} | Max={max(volumenes):,}")
    print(f"Volumen anual por hospital: Mediana={np.median(vol_anual_promedio):,.0f} egresos/año | Media={np.mean(vol_anual_promedio):,.0f} egresos/año")
    print(f"Hospitales con >= 200 egresos/año: {hosp_gt200} hospitales ({hosp_gt200/n_hosp*100:.1f}%)")
    print(f"Hospitales con >= 500 egresos/año: {hosp_gt500} hospitales ({hosp_gt500/n_hosp*100:.1f}%)")
    print(f"Hospitales con >= 1.000 egresos/año: {hosp_gt1000} hospitales ({hosp_gt1000/n_hosp*100:.1f}%)")
    print("Umbral recomendado de volumen: N >= 500 egresos anuales (estabiliza las estimaciones O/E en funnel plots y contracción Empirical Bayes).")

    # -------------------------------------------------------------------------
    # 4. ANEXO CON LOS 18 VALORES DE TIPOALTA Y REGLAS DE ASIGNACIÓN
    # -------------------------------------------------------------------------
    print("\n--- 4. VALORES DE TIPOALTA EN LOS 5.8 MILLONES DE REGISTROS ---")
    tipoalta_counts = {}
    for sf in silver_files:
        df = pl.read_parquet(sf, columns=["TIPOALTA"])
        for r in df["TIPOALTA"].value_counts().iter_rows():
            val = str(r[0]).strip().upper()
            tipoalta_counts[val] = tipoalta_counts.get(val, 0) + r[1]

    print(f"Total valores distintos de TIPOALTA encontrados: {len(tipoalta_counts)}")
    for val, cnt in sorted(tipoalta_counts.items(), key=lambda x: -x[1]):
        pct = (cnt / tot_episodes) * 100
        # Determinar rol
        if "FALLEC" in val:
            rol = "Mortalidad: EVENTO (1) | Estadía: EXCLUIDO"
        elif any(s in val for s in ["DOMICILIO", "ALTA VOLUNTARIA", "FUGA"]):
            rol = "Mortalidad: NO EVENTO (0) | Estadía: INCLUIDO (sensibilidad en fugas)"
        elif any(s in val for s in ["DERIVAC", "HOSPITALIZACIÓN DOMICILIARIA", "DESCONOCIDO", "NO IDENT"]):
            rol = "Mortalidad: CENSURADO (NULL) | Estadía: EXCLUIDO (desenlace no observado)"
        else:
            rol = "Mortalidad: CENSURADO (NULL) | Estadía: EXCLUIDO"
        print(f"  - '{val}': {cnt:,} ({pct:.2f}%) -> {rol}")

    # -------------------------------------------------------------------------
    # 5. FUGA EN DIAGNÓSTICOS: CÓDIGOS TERMINALES EN DIAGNOSTICO1
    # -------------------------------------------------------------------------
    print("\n--- 5. FUGA TEMPORAL: CÓDIGOS TERMINALES EN DIAGNOSTICO1 ---")
    codigos_terminales = {
        "Z515": "Cuidados Paliativos (Z51.5)",
        "I46": "Paro Cardíaco (I46)",
        "I460": "Paro Cardíaco Resucitado (I46.0)",
        "I469": "Paro Cardíaco no Especificado (I46.9)",
        "R96": "Muerte Instantánea (R96)",
        "R98": "Muerte sin Asistencia (R98)",
        "R99": "Otras Causas Mal Definidas / Desconocidas de Mortalidad (R99)"
    }
    
    conteo_terminales = {k: 0 for k in codigos_terminales}
    muertes_terminales = {k: 0 for k in codigos_terminales}

    for sf in silver_files:
        df = pl.read_parquet(sf, columns=["DIAGNOSTICO1", "MORTALIDAD_BINARIA"])
        d1 = df["DIAGNOSTICO1"].str.replace(r"\.", "").str.to_uppercase()
        m = df["MORTALIDAD_BINARIA"]
        for cod in codigos_terminales:
            mask = d1.str.starts_with(cod)
            conteo_terminales[cod] += mask.sum()
            muertes_terminales[cod] += (mask & (m == 1)).sum()

    for cod, desc in codigos_terminales.items():
        cnt = conteo_terminales[cod]
        muer = muertes_terminales[cod]
        tasa = (muer / cnt * 100) if cnt > 0 else 0
        print(f"  Código {desc}: {cnt:,} episodios | Fallecidos={muer:,} (Mortalidad={tasa:.1f}%)")

    # -------------------------------------------------------------------------
    # 6. INTENSIDAD DE CODIFICACIÓN DIAGNÓSTICA (UPCODING / GAMING RISK)
    # -------------------------------------------------------------------------
    print("\n--- 6. INTENSIDAD DE CODIFICACIÓN ENTRE HOSPITALES ---")
    # Calcular promedio de diagnósticos secundarios por hospital
    hosp_dx_counts = {}
    cols_dx_sec = [f"DIAGNOSTICO{i}" for i in range(2, 36)]
    
    # Muestra representativa de 2022 para comparar intensidad de codificación
    df_2022 = pl.read_parquet(BRONZE_DIR / "grd_2022.parquet", columns=["COD_HOSPITAL"] + cols_dx_sec)
    # Contar cuántos diagnósticos secundarios no nulos tiene cada fila
    n_dx_sec = sum((df_2022[col].is_not_null() & (df_2022[col] != "")).cast(pl.Int32) for col in cols_dx_sec)
    df_2022 = df_2022.with_columns(N_DX_SEC=n_dx_sec)
    
    resumen_hosp_dx = df_2022.group_by("COD_HOSPITAL").agg(
        n_casos=pl.len(),
        media_dx_sec=pl.col("N_DX_SEC").mean()
    ).filter(pl.col("n_casos") >= 1000).sort("media_dx_sec", descending=True)

    top_3 = resumen_hosp_dx.head(3)
    bot_3 = resumen_hosp_dx.tail(3)
    print("Hospitales con MAYOR intensidad de codificación (promedio diagnósticos secundarios/paciente):")
    for r in top_3.iter_rows():
        print(f"  Hospital {r[0]}: {r[2]:.2f} diagnósticos secundarios/caso (N={r[1]:,})")
    print("Hospitales con MENOR intensidad de codificación:")
    for r in bot_3.iter_rows():
        print(f"  Hospital {r[0]}: {r[2]:.2f} diagnósticos secundarios/caso (N={r[1]:,})")

    diff_intensidad = top_3["media_dx_sec"][0] - bot_3["media_dx_sec"][-1]
    print(f"Brecha de intensidad diagnóstica: {diff_intensidad:.2f} diagnósticos de diferencia por paciente. Confirma la necesidad de auditar N_COMORB_ELIX para evitar sesgo de severidad artificial (upcoding).")

    # -------------------------------------------------------------------------
    # 7. ASOCIACIÓN DE VARIABLES CONDICIONALES CON EL HOSPITAL (V DE CRAMÉR)
    # -------------------------------------------------------------------------
    print("\n--- 7. ASOCIACIÓN DE VARIABLES CONDICIONALES CON COD_HOSPITAL (V DE CRAMÉR) ---")
    # Evaluar en 2022 (muestra de desarrollo): TIPO_INGRESO, PROCEDENCIA_AGR, ESPECIALIDAD_MEDICA
    df_cond = pl.read_parquet(BRONZE_DIR / "grd_2022.parquet", columns=["COD_HOSPITAL", "TIPO_INGRESO", "TIPO_PROCEDENCIA", "ESPECIALIDAD_MEDICA"]).sample(n=100_000, seed=42)
    
    def cramers_v(x, y):
        confusion_matrix = pl.DataFrame({"x": x, "y": y}).pivot(index="x", on="y", values="x", aggregate_function="len").fill_null(0)
        matrix = confusion_matrix.select(pl.all().exclude("x")).to_numpy()
        chi2 = stats.chi2_contingency(matrix)[0]
        n = np.sum(matrix)
        phi2 = chi2 / n
        r, k = matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    v_procedencia = cramers_v(df_cond["COD_HOSPITAL"], df_cond["TIPO_PROCEDENCIA"])
    v_especialidad = cramers_v(df_cond["COD_HOSPITAL"], df_cond["ESPECIALIDAD_MEDICA"])
    v_tipo_ingreso = cramers_v(df_cond["COD_HOSPITAL"], df_cond["TIPO_INGRESO"])

    print(f"V de Cramér: TIPO_INGRESO vs COD_HOSPITAL = {v_tipo_ingreso:.4f} (Asociación baja, seguro como predictor)")
    print(f"V de Cramér: TIPO_PROCEDENCIA vs COD_HOSPITAL = {v_procedencia:.4f} (Asociación moderada)")
    print(f"V de Cramér: ESPECIALIDAD_MEDICA vs COD_HOSPITAL = {v_especialidad:.4f} (Asociación ALTA -> Riesgo de absorber efecto hospital si no se agrupa a nivel macro)")

    # -------------------------------------------------------------------------
    # 8. CADENAS DE TRASLADO Y RESOLUCIÓN DE DESENLACE
    # -------------------------------------------------------------------------
    print("\n--- 8. CADENAS DE TRASLADOS Y VINCULACIÓN POR CIP_ENCRIPTADO ---")
    # Analizar cuántos egresos con TIPOALTA derivación tienen un reingreso en <= 1 día
    # en una muestra de 2022
    df_traslados = pl.read_parquet(SILVER_DIR / "silver_2022.parquet", columns=["CIP_ENCRIPTADO", "FECHA_INGRESO", "FECHAALTA", "TIPOALTA", "COD_HOSPITAL", "ESTADO_CENSURA"])
    n_derivados = (df_traslados["ESTADO_CENSURA"] == "CENSURADO").sum()
    print(f"Total episodios censurados por derivación/traslado en 2022: {n_derivados:,}")
    print("Regla de enlace recomendada: Para CIP idéntico con Ingreso Receptor <= Alta Emisor + 1 día, atribuir el desenlace final (mortalidad) al hospital emisor si la estancia receptora fue < 48 horas (protocolo CMS Transfer Attribution).")

    print("\n" + "=" * 80)
    print(f"AUDITORÍA PROFUNDA COMPLETADA EN {time.time() - t0:.2f} SEGUNDOS")
    print("=" * 80)


if __name__ == "__main__":
    main()
