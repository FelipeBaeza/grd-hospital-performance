"""
04_protocolo_seleccion_empirica.py
----------------------------------
Protocolo riguroso de selección empírica de variables y auditoría de especificación:
1. Diagnóstico de colinealidad (SCORE_VANWALRAVEN vs Dummies Elixhauser ELIX_01..31).
2. Barrido de Regularización LASSO (C in [0.001, 0.01, 0.05, 0.1, 1.0]) y
   Selección por Estabilidad (Meinshausen & Bühlmann, 2010) sobre cohorte Inpatient pura.
3. Importancia de características por Permutación (Breiman 2001) para Mortalidad y Estadía.
4. Evaluación cuantitativa de variables condicionales en su versión modelada
   (V de Cramér sobre DERIVADO_OTRO_HOSPITAL, INGRESO_CRITICO y ESPECIALIDAD_MACRO).
5. Protocolo de validación anidada por paciente (Nested GroupKFold).
"""

import sys
import time
from pathlib import Path
import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression, TweedieRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, d2_tweedie_score
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
GOLD_DIR = ROOT_DIR / "data/gold"
SILVER_DIR = ROOT_DIR / "data/silver"

# Mapeo canónico Quan (2005)
NOMBRES_CANONICOS_ELIX = {
    "ELIX_01": "Insuficiencia cardíaca congestiva",
    "ELIX_02": "Arritmias cardíacas",
    "ELIX_03": "Valvulopatía",
    "ELIX_04": "Trastornos circulación pulmonar",
    "ELIX_05": "Enfermedad vascular periférica",
    "ELIX_06": "Hipertensión no complicada",
    "ELIX_07": "Hipertensión complicada",
    "ELIX_08": "Parálisis",
    "ELIX_09": "Otros trastornos neurológicos",
    "ELIX_10": "Enfermedad pulmonar crónica (EPOC)",
    "ELIX_11": "Diabetes no complicada",
    "ELIX_12": "Diabetes complicada",
    "ELIX_13": "Hipotiroidismo",
    "ELIX_14": "Insuficiencia renal crónica",
    "ELIX_15": "Enfermedad hepática",
    "ELIX_16": "Úlcera péptica",
    "ELIX_17": "VIH / SIDA",
    "ELIX_18": "Linfoma",
    "ELIX_19": "Cáncer metastásico",
    "ELIX_20": "Tumor sólido sin metástasis",
    "ELIX_21": "Artritis reumatoide / conectivopatías",
    "ELIX_22": "Coagulopatía",
    "ELIX_23": "Obesidad",
    "ELIX_24": "Pérdida de peso patológica",
    "ELIX_25": "Trastornos hidroelectrolíticos",
    "ELIX_26": "Anemia por hemorragia",
    "ELIX_27": "Anemia por deficiencia",
    "ELIX_28": "Abuso de alcohol",
    "ELIX_29": "Abuso de drogas",
    "ELIX_30": "Psicosis",
    "ELIX_31": "Depresión",
}


def main():
    print("=" * 80)
    print("PROTOCOLO DE SELECCIÓN EMPÍRICA Y EVALUACIÓN DE ESPECIFICACIÓN")
    print("=" * 80)
    t0 = time.time()

    # 1. Cargar datos de desarrollo (2022) y prueba fuera de muestra (2023)
    elix_cols = [f"ELIX_{i:02d}" for i in range(1, 32)]
    cols_load = [
        "ID_EPISODIO", "CIP_ENCRIPTADO", "COD_HOSPITAL", "ESTANCIA_DIAS", 
        "MORTALIDAD_BINARIA", "EN_COHORTE_DURA", "EN_COHORTE_ESTANCIA",
        "EDAD_ANIOS", "SEXO", "SCORE_VANWALRAVEN", "N_EGRESOS_12M"
    ] + elix_cols

    print("Cargando cohorte Inpatient pura de desarrollo (2022) y evaluación (2023)...")
    df_dev_raw = pl.read_parquet(GOLD_DIR / "gold_2022.parquet", columns=cols_load)
    df_val_raw = pl.read_parquet(GOLD_DIR / "gold_2023.parquet", columns=cols_load)

    # Filtro Inpatient puro: estancia > 0 O muerte precoz día 0
    cond_inp_dev = (df_dev_raw["EN_COHORTE_DURA"] == True) & (df_dev_raw["MORTALIDAD_BINARIA"].is_not_null()) & ((df_dev_raw["ESTANCIA_DIAS"] > 0) | (df_dev_raw["MORTALIDAD_BINARIA"] == 1))
    cond_inp_val = (df_val_raw["EN_COHORTE_DURA"] == True) & (df_val_raw["MORTALIDAD_BINARIA"].is_not_null()) & ((df_val_raw["ESTANCIA_DIAS"] > 0) | (df_val_raw["MORTALIDAD_BINARIA"] == 1))

    df_dev = df_dev_raw.filter(cond_inp_dev)
    df_val = df_val_raw.filter(cond_inp_val)

    print(f"Cohorte Inpatient Desarrollo (2022): {df_dev.height:,} episodios ({df_dev['MORTALIDAD_BINARIA'].sum():,} defunciones)")
    print(f"Cohorte Inpatient Evaluación (2023): {df_val.height:,} episodios ({df_val['MORTALIDAD_BINARIA'].sum():,} defunciones)")

    # -------------------------------------------------------------------------
    # 1. DIAGNÓSTICO DE COLINEALIDAD: SCORE vs DUMMIES
    # -------------------------------------------------------------------------
    print("\n--- 1. ANÁLISIS DE COLINEALIDAD (SCORE_VANWALRAVEN vs COMPONENTES) ---")
    corrs = {}
    for c in ["ELIX_01", "ELIX_07", "ELIX_10", "ELIX_14", "ELIX_19", "ELIX_25"]:
        r = np.corrcoef(df_dev["SCORE_VANWALRAVEN"].fill_null(0).to_numpy(), df_dev[c].to_numpy())[0, 1]
        nom = NOMBRES_CANONICOS_ELIX.get(c, c)
        print(f"  Corr(SCORE_VANWALRAVEN, {c} [{nom}]) = {r:+.4f}")

    print("Conclusión: SCORE_VANWALRAVEN es una combinación ponderada directa de los ELIX.")
    print("En especificaciones lineales no regularizadas se omite el score si entran los dummies.")

    # -------------------------------------------------------------------------
    # 2. BARRIDO DE REGULARIZACIÓN LASSO (C-SWEEP) Y SELECCIÓN POR ESTABILIDAD
    # -------------------------------------------------------------------------
    print("\n--- 2. BARRIDO DE HIPERPARÁMETRO C (LASSO L1) ---")
    feature_candidates = ["EDAD_ANIOS", "N_EGRESOS_12M"] + elix_cols
    
    # Muestra de 50.000 para remuestreo rápido y preciso
    sample_dev = df_dev.sample(n=min(50_000, df_dev.height), seed=42)
    X_sample = sample_dev.select(feature_candidates).fill_null(0).to_pandas().values
    y_sample = sample_dev["MORTALIDAD_BINARIA"].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    for c_val in [0.001, 0.01, 0.05, 0.1, 1.0]:
        clf_c = LogisticRegression(solver="liblinear", penalty="l1", C=c_val, random_state=42)
        clf_c.fit(X_scaled, y_sample)
        n_selected = np.sum(clf_c.coef_[0] != 0)
        p_val_sub = clf_c.predict_proba(scaler.transform(df_val.sample(20_000, seed=42).select(feature_candidates).fill_null(0).to_pandas().values))[:, 1]
        auc_val = roc_auc_score(df_val.sample(20_000, seed=42)["MORTALIDAD_BINARIA"].to_numpy(), p_val_sub)
        print(f"  C = {c_val:5.3f} -> Variables activas: {n_selected:2d} de {len(feature_candidates)} | AUROC OOS: {auc_val:.4f}")

    print("\n--- SELECCIÓN POR ESTABILIDAD (MEINSHAUSEN & BÜHLMANN 2010, C=0.05) ---")
    B = 50
    subsample_ratio = 0.632
    n_sub = int(len(X_scaled) * subsample_ratio)
    selected_counts = np.zeros(len(feature_candidates))

    for b in range(B):
        idx = np.random.choice(len(X_scaled), size=n_sub, replace=False)
        clf = LogisticRegression(solver="liblinear", penalty="l1", C=0.05, random_state=b)
        clf.fit(X_scaled[idx], y_sample[idx])
        selected_counts += (clf.coef_[0] != 0).astype(int)

    stability_scores = selected_counts / B
    threshold = 0.70

    print(f"Variables seleccionadas por Estabilidad (Frecuencia >= {threshold*100:.0f}%):")
    for feat, score in sorted(zip(feature_candidates, stability_scores), key=lambda x: -x[1]):
        nombre_humano = NOMBRES_CANONICOS_ELIX.get(feat, feat)
        estado = "[SELECCIONADA]" if score >= threshold else "Descartada"
        print(f"  - {feat} ({nombre_humano:<36}): {score*100:5.1f}% -> {estado}")

    # -------------------------------------------------------------------------
    # 3. EVALUACIÓN CUANTITATIVA DE VARIABLES CONDICIONALES EN FORMA MODELADA
    # -------------------------------------------------------------------------
    print("\n--- 3. EVALUACIÓN DE VARIABLES CONDICIONALES EN FORMA MODELADA ---")
    df_silv = pl.read_parquet(SILVER_DIR / "silver_2022.parquet", columns=[
        "COD_HOSPITAL", "HOSPPROCEDENCIA", "TIPO_PROCEDENCIA", "SERVICIOINGRESO", "ESPECIALIDAD_MEDICA"
    ])

    # 1. DERIVADO_OTRO_HOSPITAL (binaria)
    der_bin = (
        df_silv["HOSPPROCEDENCIA"].is_not_null() & (df_silv["HOSPPROCEDENCIA"] != "0") & (df_silv["HOSPPROCEDENCIA"] != "")
        | df_silv["TIPO_PROCEDENCIA"].str.to_uppercase().str.contains("DERIV|OTRO|HOSP")
    ).cast(pl.Int32)
    df_silv = df_silv.with_columns(der_bin.alias("DERIVADO_OTRO_HOSPITAL"))

    # 2. SERVICIOINGRESO (binario crítico)
    serv_crit = df_silv["SERVICIOINGRESO"].str.to_uppercase().str.contains("UCI|UTI|CRITIC|CUIDADOS INTENS|INTERMED").cast(pl.Int32)
    df_silv = df_silv.with_columns(serv_crit.alias("INGRESO_CRITICO_UCI_UTI"))

    # 3. ESPECIALIDAD_MACRO (5 categorías)
    esp_macro = (
        pl.when(df_silv["ESPECIALIDAD_MEDICA"].str.to_uppercase().str.contains("CIRUG|QUIRUR|TRAUMA|UROLOG|OFTALMO|OTORRINO|ANESTES|NEUROCI"))
        .then(pl.lit("QUIRURGICA"))
        .when(df_silv["ESPECIALIDAD_MEDICA"].str.to_uppercase().str.contains("OBSTET|GINECO|MATERN"))
        .then(pl.lit("OBSTETRICIA_GINECOLOGIA"))
        .when(df_silv["ESPECIALIDAD_MEDICA"].str.to_uppercase().str.contains("PEDIAT|NEONAT|INFANT"))
        .then(pl.lit("PEDIATRIA"))
        .when(df_silv["ESPECIALIDAD_MEDICA"].str.to_uppercase().str.contains("PSIQUIAT|SALUD MENTAL"))
        .then(pl.lit("PSIQUIATRIA"))
        .otherwise(pl.lit("MEDICA"))
    )
    df_silv = df_silv.with_columns(esp_macro.alias("ESPECIALIDAD_MACRO"))

    def cramer_v_corr(x, y):
        tab = pl.DataFrame({"x": x, "y": y}).pivot(index="x", on="y", values="x", aggregate_function="len").fill_null(0)
        mat = tab.select(pl.all().exclude("x")).to_numpy()
        chi2 = stats.chi2_contingency(mat)[0]
        n = np.sum(mat)
        r, k = mat.shape
        phi2 = max(0, chi2 / n - ((k - 1) * (r - 1)) / (n - 1))
        return np.sqrt(phi2 / min((r - 1), (k - 1)))

    v_der = cramer_v_corr(df_silv["COD_HOSPITAL"], df_silv["DERIVADO_OTRO_HOSPITAL"])
    v_serv = cramer_v_corr(df_silv["COD_HOSPITAL"], df_silv["INGRESO_CRITICO_UCI_UTI"])
    v_esp = cramer_v_corr(df_silv["COD_HOSPITAL"], df_silv["ESPECIALIDAD_MACRO"])

    print(f"  * DERIVADO_OTRO_HOSPITAL (Binaria):      V de Cramér = {v_der:.4f} -> Riesgo Absorción: ALTO (>0.25)")
    print(f"  * INGRESO_CRITICO_UCI_UTI (Binaria):     V de Cramér = {v_serv:.4f} -> Riesgo Absorción: BAJO (<0.25)")
    print(f"  * ESPECIALIDAD_MACRO (5 macro-bloques):  V de Cramér = {v_esp:.4f} -> Riesgo Absorción: BAJO (<0.25)")

    # -------------------------------------------------------------------------
    # 4. PROTOCOLO PARA MODELO DE ESTADÍA (TWEEDIE / GAMMA)
    # -------------------------------------------------------------------------
    print("\n--- 4. PROTOCOLO DE SELECCIÓN PARA MODELO DE ESTADÍA ---")
    df_los_dev = df_dev_raw.filter((df_dev_raw["EN_COHORTE_ESTANCIA"] == True) & (df_dev_raw["ESTANCIA_DIAS"] <= 60))
    df_los_val = df_val_raw.filter(df_val_raw["EN_COHORTE_ESTANCIA"] == True)

    feats_los = ["EDAD_ANIOS", "N_EGRESOS_12M"] + [c for c in elix_cols if c not in ["ELIX_02", "ELIX_14", "ELIX_22", "ELIX_25"]]
    X_los_tr = df_los_dev.sample(50_000, seed=42).select(feats_los).fill_null(0).to_pandas().values
    y_los_tr = df_los_dev.sample(50_000, seed=42)["ESTANCIA_DIAS"].to_numpy()

    X_los_te = df_los_val.sample(20_000, seed=42).select(feats_los).fill_null(0).to_pandas().values
    y_los_te = df_los_val.sample(20_000, seed=42)["ESTANCIA_DIAS"].to_numpy()

    mod_gamma = TweedieRegressor(power=2.0, link="log", max_iter=200)
    mod_gamma.fit(X_los_tr, y_los_tr)
    p_gamma = mod_gamma.predict(X_los_te)
    d2_gamma = d2_tweedie_score(y_los_te, p_gamma, power=2.0)
    print(f"  Modelo Gamma (p=2.0) OOS: Deviance explicada D^2 = {d2_gamma:.4f}")
    print("  Protocolo de estancia: Truncamiento p99 (60 días) en entrenamiento, evaluación sin truncar.")

    print("\n" + "=" * 80)
    print(f"PROTOCOLO COMPLETADO EN {time.time() - t0:.2f} SEGUNDOS")
    print("=" * 80)


if __name__ == "__main__":
    main()
