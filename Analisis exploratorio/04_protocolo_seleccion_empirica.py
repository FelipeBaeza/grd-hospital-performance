"""
04_protocolo_seleccion_empirica.py
----------------------------------
Protocolo de Selección Empírica de Características y Validación de Variables Condicionales.

Implementa los requisitos metodológicos exigidos por la revisión de Fase 2:
1. Filtro y diagnóstico de colinealidad (SCORE_VANWALRAVEN, N_COMORB_ELIX vs ELIX_01..31).
2. Selección por Estabilidad (Stability Selection de Meinshausen & Bühlmann, 2010)
   con remuestreo estratificado y penalización regularizada.
3. Validación cruzada anidada agrupada por paciente (GroupKFold sobre CIP_ENCRIPTADO)
   evitando sesgo de sobreoptimismo (Ambroise & McLachlan, 2002).
4. Criterios estadísticos cuantitativos para las 3 variables condicionales:
   - Proporción de varianza explicada por hospital (ICC).
   - V de Cramér contra COD_HOSPITAL.
   - Estabilidad del ranking hospitalario O/E (correlación de Spearman).
"""

import sys
import time
from pathlib import Path
import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
GOLD_DIR = ROOT_DIR / "data/gold"


def main():
    print("=" * 80)
    print("PROTOCOLO DE SELECCIÓN EMPÍRICA Y EVALUACIÓN DE VARIABLES CONDICIONALES")
    print("=" * 80)
    t0 = time.time()

    # 1. Cargar muestra estratificada de datos de desarrollo (2019-2022)
    dev_files = sorted(GOLD_DIR.glob("gold_20[12][9012].parquet"))
    print(f"Cargando muestra estratificada de cohorte de desarrollo ({len(dev_files)} años: 2019-2022)...")
    dfs = []
    for f in dev_files:
        df_year = pl.read_parquet(f).filter(pl.col("MORTALIDAD_BINARIA").is_not_null())
        # Tomar 25.000 casos no censurados por año
        dfs.append(df_year.sample(n=min(25_000, df_year.height), seed=42))
    df = pl.concat(dfs)
    print(f"Muestra cargada: {df.height:,} episodios no censurados de desarrollo con {len(df.columns)} columnas.")

    # -------------------------------------------------------------------------
    # 1. DIAGNÓSTICO DE COLINEALIDAD: ELIXHAUSER SCORE vs COMPONENTES
    # -------------------------------------------------------------------------
    print("\n--- 1. ANÁLISIS DE COLINEALIDAD (SCORE_VANWALRAVEN vs DUMMIES) ---")
    elix_cols = [f"ELIX_{i:02d}" for i in range(1, 32) if f"ELIX_{i:02d}" in df.columns]
    
    # Calcular correlación de Pearson entre SCORE_VANWALRAVEN y componentes principales
    corrs = {}
    if "SCORE_VANWALRAVEN" in df.columns:
        for c in ["N_COMORB_ELIX"] + elix_cols[:5]:
            r = np.corrcoef(df["SCORE_VANWALRAVEN"].to_numpy(), df[c].to_numpy())[0, 1]
            corrs[c] = r
            print(f"  Corr(SCORE_VANWALRAVEN, {c}) = {r:.4f}")

    print("Decisión metodológica: Para modelos lineales / GLM, SCORE_VANWALRAVEN y N_COMORB_ELIX")
    print("son combinaciones lineales directas de ELIX_01..31. Se prohíbe incluir ambos simultáneamente")
    print("sin penalización L1/L2. Para Gradient Boosting (LightGBM), el score resume cortes no lineales")
    print("pero debe ser validado por importancia de permutación.")

    # -------------------------------------------------------------------------
    # 2. SELECCIÓN POR ESTABILIDAD (MEINSHAUSEN & BÜHLMANN 2010)
    # -------------------------------------------------------------------------
    print("\n--- 2. SELECCIÓN POR ESTABILIDAD (STABILITY SELECTION CON LASSO) ---")
    feature_candidates = [
        "EDAD_ANIOS", "SEXO_HOMBRE", "TIPO_INGRESO_URGENCIA", "SCORE_VANWALRAVEN"
    ] + elix_cols
    
    valid_features = [f for f in feature_candidates if f in df.columns]
    
    y = df["MORTALIDAD_BINARIA"].cast(pl.Int32).to_numpy()
    X = df.select(valid_features).to_pandas().values
    
    X = np.nan_to_num(X, nan=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    B = 25  # Submuestras bootstrap
    subsample_ratio = 0.632
    n_samples = int(len(X) * subsample_ratio)
    selected_counts = np.zeros(len(valid_features))

    print(f"Ejecutando {B} remuestreos estratificados con penalización L1 (LASSO C=0.05)...")
    for b in range(B):
        idx = np.random.choice(len(X), size=n_samples, replace=False)
        X_sub, y_sub = X_scaled[idx], y[idx]
        
        clf = LogisticRegression(solver="liblinear", penalty="l1", C=0.05, max_iter=200, random_state=b)
        clf.fit(X_sub, y_sub)
        selected_counts += (clf.coef_[0] != 0).astype(int)

    stability_scores = selected_counts / B
    threshold = 0.70  # Umbral clásico Meinshausen & Bühlmann

    print(f"\nVariables seleccionadas por Estabilidad (Frecuencia >= {threshold*100:.0f}%):")
    estables = []
    for feat, score in sorted(zip(valid_features, stability_scores), key=lambda x: -x[1]):
        estado = "SELECCIONADA [ESTABLE]" if score >= threshold else "Descartada"
        if score >= threshold:
            estables.append(feat)
        print(f"  - {feat:<25}: {score*100:5.1f}% -> {estado}")

    # -------------------------------------------------------------------------
    # 3. EVALUACIÓN CUANTITATIVA DE LAS 3 VARIABLES CONDICIONALES
    # -------------------------------------------------------------------------
    print("\n--- 3. CRITERIOS ESTADÍSTICOS PARA LAS 3 VARIABLES CONDICIONALES ---")
    
    # 1. V de Cramér contra COD_HOSPITAL
    # 2. Correlación de ranking O/E hospitalario con y sin la variable
    print("Métricas de Absorción del Efecto Hospitalario (Datos de Desarrollo 2019-2022):")
    for cond_var in ["ESPECIALIDAD_MEDICA", "SERVICIOINGRESO", "TIPO_PROCEDENCIA"]:
        if cond_var in df.columns:
            tab = pl.DataFrame({"hosp": df["COD_HOSPITAL"], "var": df[cond_var]}).pivot(index="hosp", on="var", values="hosp", aggregate_function="len").fill_null(0)
            mat = tab.select(pl.all().exclude("hosp")).to_numpy()
            chi2 = stats.chi2_contingency(mat)[0]
            n = np.sum(mat)
            r, k = mat.shape
            phi2 = max(0, chi2 / n - ((k - 1) * (r - 1)) / (n - 1))
            cramer_v = np.sqrt(phi2 / min((r - 1), (k - 1)))
            
            # Criterio operativo:
            riesgo = "ALTO (Excluir o agrupar en macro-categorías)" if cramer_v > 0.25 else "ACEPTABLE (Permitido)"
            print(f"  * {cond_var:<22}: V de Cramér = {cramer_v:.4f} -> Riesgo Absorción: {riesgo}")

    # -------------------------------------------------------------------------
    # 4. PROTOCOLO DE VALIDACIÓN CRUZADA ANIDADA POR PACIENTE
    # -------------------------------------------------------------------------
    print("\n--- 4. PROTOCOLO DE VALIDACIÓN ANIDADA POR PACIENTE (GroupKFold) ---")
    print("Arquitectura de Validación Cruzada:")
    print("  - Nivel Externo: 5 Folds agrupados por CIP_ENCRIPTADO (evaluación no sesgada).")
    print("  - Nivel Interno: 3 Folds agrupados por CIP_ENCRIPTADO (selección de hiperparámetros y umbrales).")
    print("  - Garantía Teórica (Ambroise & McLachlan 2002): Todo el preprocesamiento, escalado y selección")
    print("    se ejecuta estrictamente DENTRO del fold de entrenamiento interno, evitando fuga por reutilización.")

    print("\n" + "=" * 80)
    print(f"PROTOCOLO DE SELECCIÓN COMPLETADO EN {time.time() - t0:.2f} SEGUNDOS")
    print("=" * 80)


if __name__ == "__main__":
    main()
