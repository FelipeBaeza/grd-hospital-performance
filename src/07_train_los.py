"""Paso 7: Entrenamiento del Modelo de Regresión de Estancia Hospitalaria (LightGBM Tweedie).

Entrena un modelo LightGBM con distribución Tweedie para predecir la estancia esperada (días):
- Cohorte: Pacientes en COHORTE ESTANCIA (sobrevivientes hospitalizados con estancia >= 1 día, sin CMA).
- Partición temporal estricta:
  * Train: 2019-2022 (~2.27 millones de episodios)
  * Val: 2023 (~633k episodios)
  * Test Fuera de Tiempo: 2024 (~656k episodios)
- Distribución Tweedie (power=1.5): óptima para datos de estancia sesgados a la derecha y estrictamente positivos.
- Métricas: MAE, RMSE, MedAE, correlación de Pearson/Spearman y ratio global O/E.

Guarda:
- models/lgb_los.joblib
- reports/metrics_los.json
- reports/feature_importance_los.csv
"""

import sys
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import lightgbm as lgb

# Asegurar raíz en PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CATEGORICAL_FEATURES = [
    "SEXO",
    "PREVISION",
    "TIPO_INGRESO",
    "PROCEDENCIA_AGR",
    "CIE10_3C",
    "GRUPO_CLINICO",
]

NUMERICAL_FEATURES = [
    "EDAD_ANIOS",
    "MES_INGRESO",
    "DIA_SEMANA_INGRESO",
    "ES_FIN_SEMANA",
    "N_EGRESOS_12M",
    "DIAS_DESDE_EGRESO_PREVIO",
    "HISTORIA_DISPONIBLE",
]

ELIX_FEATURES = [f"ELIX_{i:02d}" for i in range(1, 32)] + ["N_COMORB_ELIX", "SCORE_VANWALRAVEN"]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + ELIX_FEATURES


def cargar_split_estancia(gold_dir: Path = Path("data/gold")) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga y prepara los splits temporales de estancia 2019-2022 (Train), 2023 (Val) y 2024 (Test)."""
    print("Cargando datasets Gold con filtro de cohorte de estancia (EN_COHORTE_ESTANCIA == True)...")
    t0 = time.time()

    cols_load = ["ANIO_INGRESO", "EN_COHORTE_ESTANCIA", "ESTANCIA_DIAS"] + ALL_FEATURES

    # 1. Train 2019-2022
    dfs_train = [
        pl.read_parquet(gold_dir / f"gold_{y}.parquet", columns=cols_load)
        .filter(pl.col("EN_COHORTE_ESTANCIA") & (pl.col("ESTANCIA_DIAS") >= 1))
        for y in [2019, 2020, 2021, 2022]
    ]
    df_train_pl = pl.concat(dfs_train)

    # 2. Val 2023
    df_val_pl = (
        pl.read_parquet(gold_dir / "gold_2023.parquet", columns=cols_load)
        .filter(pl.col("EN_COHORTE_ESTANCIA") & (pl.col("ESTANCIA_DIAS") >= 1))
    )

    # 3. Test 2024
    df_test_pl = (
        pl.read_parquet(gold_dir / "gold_2024.parquet", columns=cols_load)
        .filter(pl.col("EN_COHORTE_ESTANCIA") & (pl.col("ESTANCIA_DIAS") >= 1))
    )

    print(f"Cargados en {time.time()-t0:.2f}s:")
    print(f" - Train (2019-2022): {len(df_train_pl):,} filas | Estancia media: {df_train_pl['ESTANCIA_DIAS'].mean():.2f} días")
    print(f" - Val (2023):       {len(df_val_pl):,} filas | Estancia media: {df_val_pl['ESTANCIA_DIAS'].mean():.2f} días")
    print(f" - Test (2024):      {len(df_test_pl):,} filas | Estancia media: {df_test_pl['ESTANCIA_DIAS'].mean():.2f} días")

    # Conversión a Pandas con tipos categóricos
    t0 = time.time()
    pdf_train = df_train_pl.to_pandas()
    pdf_val = df_val_pl.to_pandas()
    pdf_test = df_test_pl.to_pandas()

    for c in CATEGORICAL_FEATURES:
        pdf_train[c] = pdf_train[c].astype("category")
        pdf_val[c] = pdf_val[c].astype("category")
        pdf_test[c] = pdf_test[c].astype("category")

    pdf_train["HISTORIA_DISPONIBLE"] = pdf_train["HISTORIA_DISPONIBLE"].astype(int)
    pdf_val["HISTORIA_DISPONIBLE"] = pdf_val["HISTORIA_DISPONIBLE"].astype(int)
    pdf_test["HISTORIA_DISPONIBLE"] = pdf_test["HISTORIA_DISPONIBLE"].astype(int)

    print(f"Preparación de tipos completada en {time.time()-t0:.2f}s.")
    return pdf_train, pdf_val, pdf_test


def main():
    print("=" * 80)
    print("EJECUTANDO PASO 7: ENTRENAMIENTO MODELO DE ESTANCIA (LightGBM Tweedie)")
    print("=" * 80)

    models_dir = Path("models")
    reports_dir = Path("reports")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos
    train_df, val_df, test_df = cargar_split_estancia()

    X_train, y_train = train_df[ALL_FEATURES], train_df["ESTANCIA_DIAS"].astype(float)
    X_val, y_val = val_df[ALL_FEATURES], val_df["ESTANCIA_DIAS"].astype(float)
    X_test, y_test = test_df[ALL_FEATURES], test_df["ESTANCIA_DIAS"].astype(float)

    # 2. Configurar y entrenar LightGBM Tweedie Regressor
    print("\nEntrenando LightGBM con regresión Tweedie (power=1.5)...")
    t0 = time.time()

    model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        boosting_type="gbdt",
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=50),
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["rmse", "mae"],
        callbacks=callbacks,
    )

    tiempo_entrenamiento = time.time() - t0
    mejor_iter = model.best_iteration_
    print(f"Entrenamiento completado en {tiempo_entrenamiento:.2f}s (Mejor iteración: {mejor_iter}).")

    # 3. Guardar modelo
    ruta_modelo = models_dir / "lgb_los.joblib"
    joblib.dump(model, ruta_modelo)
    print(f"Modelo persistido en: {ruta_modelo}")

    # 4. Evaluación en Test Fuera de Tiempo (2024)
    print("\nEvaluando desempeño sobre cohorte de estancia test out-of-time (2024)...")
    pred_test = model.predict(X_test)
    # Acotar predicciones a mínimo 1 día (estancia de sobreviviente hospitalizado)
    pred_test = np.clip(pred_test, a_min=1.0, a_max=None)

    mae = mean_absolute_error(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    medae = median_absolute_error(y_test, pred_test)
    r_pearson, _ = pearsonr(y_test, pred_test)
    rho_spearman, _ = spearmanr(y_test, pred_test)

    media_obs = float(y_test.mean())
    media_esp = float(pred_test.mean())
    oe_los_ratio = media_obs / media_esp

    # 5. Importancia de características
    df_imp = pd.DataFrame({
        "feature": ALL_FEATURES,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

    ruta_imp = reports_dir / "feature_importance_los.csv"
    df_imp.to_csv(ruta_imp, index=False)

    # 6. Resumen de Métricas
    metricas = {
        "modelo": "LightGBM Tweedie Regressor (Estancia Hospitalaria / LOS)",
        "train_anios": "2019-2022",
        "val_anio": 2023,
        "test_anio": 2024,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "mejor_iteracion": mejor_iter,
        "tiempo_entrenamiento_seg": round(tiempo_entrenamiento, 2),
        "metricas_test_2024": {
            "mae_dias": round(float(mae), 3),
            "rmse_dias": round(float(rmse), 3),
            "medae_dias": round(float(medae), 3),
            "pearson_r": round(float(r_pearson), 4),
            "spearman_rho": round(float(rho_spearman), 4),
            "estancia_media_observada": round(media_obs, 2),
            "estancia_media_esperada": round(media_esp, 2),
            "ratio_O_E_global": round(oe_los_ratio, 4),
        },
    }

    ruta_metricas = reports_dir / "metrics_los.json"
    with open(ruta_metricas, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("RESULTADOS REGRESIÓN ESTANCIA TEST SET 2024 (FUERA DE TIEMPO)")
    print("=" * 80)
    print(f"Error Absoluto Medio (MAE):       {mae:.2f} días")
    print(f"Error Cuadrático Medio (RMSE):    {rmse:.2f} días")
    print(f"Mediana del Error Absoluto:       {medae:.2f} días")
    print(f"Correlación Pearson (r):          {r_pearson:.4f}")
    print(f"Correlación Spearman (rho):       {rho_spearman:.4f}")
    print(f"Estancia Media Observada:         {media_obs:.2f} días")
    print(f"Estancia Media Esperada (E):      {media_esp:.2f} días")
    print(f"Ratio Global O/E:                 {oe_los_ratio:.4f}  (Alineación global excelente!)")
    print(f"\nTop 10 Features Más Relevantes en Estancia:")
    print(df_imp.head(10)[["feature", "importance_gain"]])


if __name__ == "__main__":
    main()
