# Ficha del modelo de estadia

**Qué predice.** Días de hospitalización esperados de un sobreviviente, recortados a 53 días (cohorte E).

**Algoritmo.** lightgbm; árboles: 872.
**Hiperparámetros.** {"learning_rate": 0.02163344146706896, "subsample": 0.8295381286173298, "colsample": 0.5546242185298835, "reg_alpha": 5.122619410737736e-07, "reg_lambda": 0.12081713737965877, "num_leaves": 230, "min_child_samples": 109, "objetivo": "tweedie", "potencia": 1.7959876071348773}

**Datos.** Entrenamiento con 2019–2022 y recalibración global con 2023 (método: factor). El año 2024 no se usó.

**Reglas.** Huella SHA-256 de reglas_fase4.toml: `24ed78fcd90b5ee5ca2f93d61225fc8f6e936256d7c88f231720bbfb29391e59`.

**Variables usadas.** SEXO, TIPO_INGRESO, TIPO_PROCEDENCIA, PROCEDENCIA_AGR, CIE10_3C, GRUPO_CLINICO, EDAD_ANIOS, ES_FIN_SEMANA, NIVEL_CUIDADO_INGRESO, ELIX_01, ELIX_02, ELIX_03, ELIX_04, ELIX_05, ELIX_06, ELIX_07, ELIX_08, ELIX_09, ELIX_10, ELIX_11, ELIX_12, ELIX_13, ELIX_14, ELIX_15, ELIX_16, ELIX_17, ELIX_18, ELIX_19, ELIX_20, ELIX_21, ELIX_22, ELIX_23, ELIX_24, ELIX_25, ELIX_26, ELIX_27, ELIX_28, ELIX_29, ELIX_30, ELIX_31

**Métricas de validación (2023, fuera de muestra).**
- n: 519833
- media_obs: 6.7981
- media_pred: 6.7937
- devianza_poisson: 4.1423
- rmse: 7.0684
- mae: 4.1129
- oe_global: 1.0007

**Límites conocidos.**
- Entrenado con 2019–2022; el nivel de mortalidad y de estadía cambió en 2023 y se corrigió con una recalibración global.
- La codificación de diagnósticos crece cada año; los años no son totalmente comparables.
- Solo adultos (18 años o más) hospitalizados; no sirve para niños ni recién nacidos.
- El hospital no es un predictor; el modelo no distingue diferencias de recursos o de población entre hospitales.
- No se evaluó con 2024: esa prueba se hace una sola vez en la Fase 5.
