# Ficha del modelo de mortalidad

**Qué predice.** Probabilidad de morir durante la hospitalización (adultos, cohorte D).

**Algoritmo.** lightgbm; árboles: 508.
**Hiperparámetros.** {"learning_rate": 0.027738550742526145, "subsample": 0.6484390500617475, "colsample": 0.9075965467403001, "reg_alpha": 0.003036672849460866, "reg_lambda": 0.021301549034007102, "num_leaves": 96, "min_child_samples": 272}

**Datos.** Entrenamiento con 2019–2022 y recalibración global con 2023 (método: platt). El año 2024 no se usó.

**Reglas.** Huella SHA-256 de reglas_fase4.toml: `24ed78fcd90b5ee5ca2f93d61225fc8f6e936256d7c88f231720bbfb29391e59`.

**Variables usadas.** SEXO, TIPO_INGRESO, TIPO_PROCEDENCIA, PROCEDENCIA_AGR, CIE10_3C, GRUPO_CLINICO, EDAD_ANIOS, ES_FIN_SEMANA, NIVEL_CUIDADO_INGRESO, ELIX_01, ELIX_02, ELIX_03, ELIX_04, ELIX_05, ELIX_06, ELIX_07, ELIX_08, ELIX_09, ELIX_10, ELIX_11, ELIX_12, ELIX_13, ELIX_14, ELIX_15, ELIX_16, ELIX_17, ELIX_18, ELIX_19, ELIX_20, ELIX_21, ELIX_22, ELIX_23, ELIX_24, ELIX_25, ELIX_26, ELIX_27, ELIX_28, ELIX_29, ELIX_30, ELIX_31

**Métricas de validación (2023, fuera de muestra).**
- n: 548107
- prevalencia: 0.0430
- pr_auc: 0.3336
- auc: 0.9190
- brier: 0.0333
- bss: 0.1904
- log_loss: 0.1162
- pendiente: 0.9995
- intercepto: -0.0010
- citl: -0.0001
- oe_global: 1.0000

**Límites conocidos.**
- Entrenado con 2019–2022; el nivel de mortalidad y de estadía cambió en 2023 y se corrigió con una recalibración global.
- La codificación de diagnósticos crece cada año; los años no son totalmente comparables.
- Solo adultos (18 años o más) hospitalizados; no sirve para niños ni recién nacidos.
- El hospital no es un predictor; el modelo no distingue diferencias de recursos o de población entre hospitales.
- No se evaluó con 2024: esa prueba se hace una sola vez en la Fase 5.
