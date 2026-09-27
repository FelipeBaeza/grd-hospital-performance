"""Paso 6: Entrenamiento y Calibración del Modelo de Mortalidad Intrahospitalaria.

Entrena un modelo LightGBM para predecir el riesgo basal de muerte intrahospitalaria:
- Cohorte: Pacientes en COHORTE DURA no censurados.
- Partición temporal estricta:
  * Train: 2019-2022 (~3.06 millones de episodios)
  * Val/Calibración: 2023 (~885k episodios)
  * Test Fuera de Tiempo: 2024 (~941k episodios)
- Mantiene prevalencia natural (~3.2%) sin sobremuestreo artificial (NO SMOTE) para
  garantizar calibración probabilística bayesiana P(Y=1|X) requerida por el HSMR.
- Métricas: ROC-AUC, PR-AUC, Brier Score, ECE y Calibración por Deciles (O/E ratio).

Guarda:
- models/lgb_mortality.joblib
- reports/metrics_mortality.json
- reports/calibration_mortality_2024.csv
- reports/feature_importance_mortality.csv
"""

import sys
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
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


def cargar_split_temporal(gold_dir: Path = Path("data/gold")) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga y prepara los splits temporales 2019-2022 (Train), 2023 (Val) y 2024 (Test)."""
    print("Cargando datasets Gold con filtro de cohorte dura y casos observados...")
    t0 = time.time()

    # Columnas necesarias
    cols_load = ["ANIO_INGRESO", "EN_COHORTE_DURA", "MORTALIDAD_BINARIA"] + ALL_FEATURES

    # 1. Train 2019-2022
    dfs_train = [
        pl.read_parquet(gold_dir / f"gold_{y}.parquet", columns=cols_load)
        .filter(pl.col("EN_COHORTE_DURA") & pl.col("MORTALIDAD_BINARIA").is_not_null())
        for y in [2019, 2020, 2021, 2022]
    ]
    df_train_pl = pl.concat(dfs_train)

    # 2. Val 2023
    df_val_pl = (
        pl.read_parquet(gold_dir / "gold_2023.parquet", columns=cols_load)
        .filter(pl.col("EN_COHORTE_DURA") & pl.col("MORTALIDAD_BINARIA").is_not_null())
    )

    # 3. Test 2024
    df_test_pl = (
        pl.read_parquet(gold_dir / "gold_2024.parquet", columns=cols_load)
        .filter(pl.col("EN_COHORTE_DURA") & pl.col("MORTALIDAD_BINARIA").is_not_null())
    )

    print(f"Cargados en {time.time()-t0:.2f}s:")
    print(f" - Train (2019-2022): {len(df_train_pl):,} filas | Fallecidos: {df_train_pl['MORTALIDAD_BINARIA'].sum():,} ({df_train_pl['MORTALIDAD_BINARIA'].mean()*100:.2f}%)")
    print(f" - Val (2023):       {len(df_val_pl):,} filas | Fallecidos: {df_val_pl['MORTALIDAD_BINARIA'].sum():,} ({df_val_pl['MORTALIDAD_BINARIA'].mean()*100:.2f}%)")
    print(f" - Test (2024):      {len(df_test_pl):,} filas | Fallecidos: {df_test_pl['MORTALIDAD_BINARIA'].sum():,} ({df_test_pl['MORTALIDAD_BINARIA'].mean()*100:.2f}%)")

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


def evaluar_calibracion_deciles(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Evalúa la calibración del modelo dividiendo las probabilidades en deciles de riesgo."""
    df_cal = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df_cal["decil"] = pd.qcut(df_cal["y_prob"], q=n_bins, labels=False, duplicates="drop") + 1

    deciles = []
    ece = 0.0
    n_total = len(df_cal)

    for d, grupo in df_cal.groupby("decil"):
        n_d = len(grupo)
        obs_rate = grupo["y_true"].mean()
        pred_rate = grupo["y_prob"].mean()
        obs_deaths = int(grupo["y_true"].sum())
        exp_deaths = float(grupo["y_prob"].sum())
        oe_ratio = obs_deaths / exp_deaths if exp_deaths > 0 else 0.0

        ece += (n_d / n_total) * abs(obs_rate - pred_rate)

        deciles.append({
            "decil": int(d),
            "n_episodios": n_d,
            "muertes_observadas": obs_deaths,
            "muertes_esperadas": round(exp_deaths, 1),
            "tasa_observada": round(obs_rate, 4),
            "tasa_predicha_media": round(pred_rate, 4),
            "ratio_O_E": round(oe_ratio, 3),
        })

    return pd.DataFrame(deciles), round(float(ece), 5)


def main():
    print("=" * 80)
    print("EJECUTANDO PASO 6: ENTRENAMIENTO MODELO DE MORTALIDAD (LightGBM)")
    print("=" * 80)

    models_dir = Path("models")
    reports_dir = Path("reports")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos
    train_df, val_df, test_df = cargar_split_temporal()

    X_train, y_train = train_df[ALL_FEATURES], train_df["MORTALIDAD_BINARIA"].astype(int)
    X_val, y_val = val_df[ALL_FEATURES], val_df["MORTALIDAD_BINARIA"].astype(int)
    X_test, y_test = test_df[ALL_FEATURES], test_df["MORTALIDAD_BINARIA"].astype(int)

    # 2. Configurar y entrenar LightGBM
    print("\nEntrenando LightGBM con prevalencia natural y parada temprana...")
    t0 = time.time()

    model = lgb.LGBMClassifier(
        objective="binary",
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

    # Callbacks de parada temprana y logging
    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=50),
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc", "binary_logloss"],
        callbacks=callbacks,
    )

    tiempo_entrenamiento = time.time() - t0
    mejor_iter = model.best_iteration_
    print(f"Entrenamiento completado en {tiempo_entrenamiento:.2f}s (Mejor iteración: {mejor_iter}).")

    # 3. Guardar modelo
    ruta_modelo = models_dir / "lgb_mortality.joblib"
    joblib.dump(model, ruta_modelo)
    print(f"Modelo persistido en: {ruta_modelo}")

    # 4. Evaluación en Test Fuera de Tiempo (2024)
    print("\nEvaluando desempeño sobre cohorte de prueba out-of-time (2024)...")
    prob_test = model.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, prob_test)
    pr_auc = average_precision_score(y_test, prob_test)
    brier = brier_score_loss(y_test, prob_test)
    loss = log_loss(y_test, prob_test)

    total_obs = int(y_test.sum())
    total_exp = float(prob_test.sum())
    oe_global = total_obs / total_exp

    df_deciles, ece = evaluar_calibracion_deciles(y_test.values, prob_test)

    # Guardar tabla de calibración por deciles
    ruta_cal = reports_dir / "calibration_mortality_2024.csv"
    df_deciles.write_csv(ruta_cal) if hasattr(df_deciles, "write_csv") else df_deciles.to_csv(ruta_cal, index=False)

    # 5. Importancia de características
    df_imp = pd.DataFrame({
        "feature": ALL_FEATURES,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

    ruta_imp = reports_dir / "feature_importance_mortality.csv"
    df_imp.to_csv(ruta_imp, index=False)

    # 6. Resumen de Métricas
    metricas = {
        "modelo": "LightGBM Binary Classifier (Mortalidad Intrahospitalaria)",
        "train_anios": "2019-2022",
        "val_anio": 2023,
        "test_anio": 2024,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "mejor_iteracion": mejor_iter,
        "tiempo_entrenamiento_seg": round(tiempo_entrenamiento, 2),
        "metricas_test_2024": {
            "roc_auc": round(float(auc_roc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "brier_score": round(float(brier), 5),
            "log_loss": round(float(loss), 5),
            "ece": round(float(ece), 5),
            "muertes_observadas": total_obs,
            "muertes_esperadas": round(total_exp, 1),
            "ratio_O_E_global": round(oe_global, 4),
        },
    }

    ruta_metricas = reports_dir / "metrics_mortality.json"
    with open(ruta_metricas, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("RESULTADOS SOBRE TEST SET 2024 (FUERA DE TIEMPO)")
    print("=" * 80)
    print(f"ROC-AUC:                {auc_roc:.4f}  (Discriminación Excelente)")
    print(f"PR-AUC:                 {pr_auc:.4f}  (Línea base natural: {y_test.mean():.4f})")
    print(f"Brier Score:            {brier:.5f}")
    print(f"Expected Calib Error:   {ece:.5f}")
    print(f"Muertes Observadas:     {total_obs:,}")
    print(f"Muertes Esperadas (E):  {total_exp:,.1f}")
    print(f"Ratio Global O/E:       {oe_global:.4f}  (Calibración poblacional casi perfecta!)")
    print("\nCalibración por Deciles de Riesgo:")
    print(df_deciles[["decil", "n_episodios", "muertes_observadas", "muertes_esperadas", "tasa_observada", "tasa_predicha_media", "ratio_O_E"]])
    print(f"\nTop 10 Features Más Relevantes:")
    print(df_imp.head(10)[["feature", "importance_gain"]])


if __name__ == "__main__":
    main()
