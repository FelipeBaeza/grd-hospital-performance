import numpy as np
import pandas as pd
import polars as pl
import pytest

from modelado import candidatos, congelar, exhaustividad, integridad
from modelado.candidatos import entrenar
from modelado.datos import CONDICIONALES, PREDICTORES
from tests.sintetico import hacer_df

PARAMS_LGBM = {"learning_rate": 0.1, "subsample": 0.8, "colsample": 0.8, "reg_alpha": 0.0, "reg_lambda": 0.0, "num_leaves": 15, "min_child_samples": 20}
PARAMS_XGB = {"learning_rate": 0.1, "subsample": 0.8, "colsample": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0, "max_depth": 4, "min_child_weight": 1.0}


def test_mortalidad_todos_los_candidatos_dan_probabilidades():
    df = hacer_df(6000)
    tr, ca = df[df["ANIO_EGRESO"] <= 2022], df[df["ANIO_EGRESO"] == 2023]
    for tipo, params in (("constante", {}), ("logistica", {"C": 1.0}), ("lightgbm", PARAMS_LGBM), ("xgboost", PARAMS_XGB)):
        m = entrenar(tipo, "mortalidad", tr, params, 1, 2, n_arboles=30)
        p = m.predecir(ca)
        assert p.shape == (len(ca),) and (p >= 0).all() and (p <= 1).all(), tipo


def test_estadia_todos_los_candidatos_dan_promedios_positivos():
    df = hacer_df(6000, tarea="estadia")
    tr, ca = df[df["ANIO_EGRESO"] <= 2022], df[df["ANIO_EGRESO"] == 2023]
    casos = [("poisson", {"alpha": 1e-4}), ("lineal", {"alpha": 1.0}), ("lightgbm", {**PARAMS_LGBM, "objetivo": "tweedie", "potencia": 1.5}),
             ("lightgbm", {**PARAMS_LGBM, "objetivo": "poisson", "potencia": None}), ("xgboost", {**PARAMS_XGB, "objetivo": "tweedie", "potencia": 1.5}),
             ("xgboost", {**PARAMS_XGB, "objetivo": "poisson", "potencia": None})]
    for tipo, params in casos:
        p = entrenar(tipo, "estadia", tr, params, 1, 2, n_arboles=30).predecir(ca)
        assert p.shape == (len(ca),) and (p > 0).all() and np.isfinite(p).all(), (tipo, params)


def test_un_modelo_bien_entrenado_supera_a_la_prevalencia():
    from modelado.metricas import metricas_mortalidad
    df = hacer_df(20000)
    tr, ca = df[df["ANIO_EGRESO"] <= 2022], df[df["ANIO_EGRESO"] == 2023]
    m = entrenar("lightgbm", "mortalidad", tr, PARAMS_LGBM, 1, 2, n_arboles=60)
    assert metricas_mortalidad(ca["y"], m.predecir(ca))["auc"] > 0.65


def test_la_deteccion_temprana_devuelve_el_numero_de_arboles():
    df = hacer_df(8000)
    tr, va = df[df["ANIO_EGRESO"] <= 2021], df[df["ANIO_EGRESO"] == 2022]
    for tipo, params in (("lightgbm", PARAMS_LGBM), ("xgboost", PARAMS_XGB)):
        m = entrenar(tipo, "mortalidad", tr, params, 1, 2, validacion=va, rondas=10, max_arboles=200)
        assert 1 <= m.n_arboles <= 200


def test_el_entrenamiento_es_determinista():
    df = hacer_df(6000)
    tr, ca = df[df["ANIO_EGRESO"] <= 2022], df[df["ANIO_EGRESO"] == 2023]
    for tipo, params in (("lightgbm", PARAMS_LGBM), ("xgboost", PARAMS_XGB), ("logistica", {"C": 1.0})):
        a = entrenar(tipo, "mortalidad", tr, params, 7, 2, n_arboles=40).predecir(ca)
        b = entrenar(tipo, "mortalidad", tr, params, 7, 2, n_arboles=40).predecir(ca)
        assert np.max(np.abs(a - b)) == 0, tipo


def test_el_preprocesamiento_lineal_aprende_solo_del_entrenamiento():
    df = hacer_df(3000)
    tr = df[df["ANIO_EGRESO"] <= 2022].copy()
    nuevo = df[df["ANIO_EGRESO"] == 2023].copy()
    nuevo["CIE10_3C"] = pd.Categorical(["Z99"] * len(nuevo))                   # categoría que no existía al entrenar
    pre = candidatos.preproceso_lineal().fit(tr[PREDICTORES])
    a = pre.transform(tr[PREDICTORES])
    b = pre.transform(nuevo[PREDICTORES])
    assert a.shape[1] == b.shape[1]
    nombres = pre.get_feature_names_out()
    cols_cie = [i for i, n in enumerate(nombres) if "CIE10_3C" in n]
    assert b[:, cols_cie].sum() == 0 and not any("Z99" in n for n in nombres)


def test_un_modelo_no_ve_hospital_ni_anio():
    df = hacer_df(3000)
    m = entrenar("logistica", "mortalidad", df, {"C": 1.0}, 1)
    usadas = set(m.objeto["pre"].feature_names_in_)
    assert usadas == set(PREDICTORES) and "COD_HOSPITAL" not in usadas and "ANIO_EGRESO" not in usadas


def test_identificacion_del_hospital_se_detecta_si_una_variable_lo_delata():
    from modelado.config import cargar_reglas
    r = cargar_reglas()
    r.integridad = {**r.integridad, "muestra_entrenamiento": 3000, "muestra_evaluacion": 1500, "iteraciones": 15}
    df = hacer_df(6000)
    df["TIPO_PROCEDENCIA"] = pd.Categorical(df["COD_HOSPITAL"].map({"101": "A", "102": "B", "103": "C", "104": "D"}))       # variable condicional que delata al hospital
    con = integridad.prueba_identificacion_hospital(df, r, True)
    sin = integridad.prueba_identificacion_hospital(df, r, False)
    assert con["exactitud_top1"] > 0.95 and sin["exactitud_top1"] < 0.5 and con["n_predictores"] == 40 and sin["n_predictores"] == 40 - len(CONDICIONALES)


def test_coherencia_no_publica_celdas_pequenas():
    df = hacer_df(2000)
    df.loc[df.index[:5], "EDAD_ANIOS"] = 95
    df["EDAD_ANIOS"] = df["EDAD_ANIOS"].clip(upper=89)
    df.loc[df.index[:5], "EDAD_ANIOS"] = 95
    t = integridad.coherencia(df, np.full(len(df), 0.1), "mortalidad")
    assert "90+" not in set(t["edad"]["grupo"].astype(str)) and (t["edad"]["n"] >= 10).all()


def test_contribuciones_de_una_logistica_y_de_un_arbol_suman_uno():
    df = hacer_df(6000)
    for tipo, params in (("logistica", {"C": 1.0}), ("lightgbm", PARAMS_LGBM), ("xgboost", PARAMS_XGB)):
        m = entrenar(tipo, "mortalidad", df, params, 1, 2, n_arboles=30)
        t = integridad.contribuciones(m, df, 2000, 1)
        assert abs(t["proporcion"].sum() - 1) < 1e-6 and set(t["variable"]) <= set(PREDICTORES), tipo
        assert t.iloc[0]["variable"] in ("EDAD_ANIOS", "ELIX_01", "NIVEL_CUIDADO_INGRESO"), tipo


def test_conteo_de_diagnosticos_excluye_factores_sociales_y_embarazo_normal():
    cols = {f"DIAGNOSTICO{i}": [None, None, None] for i in range(2, 36)}
    cols["DIAGNOSTICO2"] = ["E119", "Z550", "O800"]
    cols["DIAGNOSTICO3"] = ["I10", "I10", "Z340"]
    cols["DIAGNOSTICO4"] = ["Z59", None, "J189"]
    n = exhaustividad.conteo_validos(pl.DataFrame(cols), ["Z55", "Z56", "Z57", "Z58", "Z59", "Z34", "O80"])
    assert n.to_list() == [2, 1, 1]


def test_el_modelo_de_exhaustividad_espera_mas_diagnosticos_en_los_mayores():
    from modelado.config import cargar_reglas
    rng = np.random.default_rng(0)
    n = 20000
    edad = rng.integers(18, 95, n)
    df = pd.DataFrame({"CIE10_3C": rng.choice(["J18", "I50", "RARO"], n), "EDAD_ANIOS": edad, "SEXO": rng.choice(["MUJER", "HOMBRE"], n), "y": rng.poisson(1 + edad / 20)})
    df["banda_edad"] = pd.cut(df["EDAD_ANIOS"], [18, 30, 40, 50, 60, 70, 80, 90, 200], right=False).astype(str)
    r = exhaustividad.entrenar_y_evaluar(df.iloc[:15000], df.iloc[15000:], cargar_reglas())
    p = r["glm"].predict(df.iloc[15000:][["CIE10_3C", "banda_edad", "SEXO"]])
    jov, vie = p[df.iloc[15000:]["EDAD_ANIOS"].to_numpy() < 30].mean(), p[df.iloc[15000:]["EDAD_ANIOS"].to_numpy() > 80].mean()
    assert vie > jov and abs(r["metricas"]["poisson_glm"]["oe_global"] - 1) < 0.05


def test_el_manifiesto_guarda_huellas_que_se_pueden_verificar(tmp_path):
    from modelado.config import cargar_reglas, sha256_archivo
    df = hacer_df(3000)
    m = entrenar("lightgbm", "mortalidad", df, PARAMS_LGBM, 1, 2, n_arboles=20)
    from modelado.calibracion import Platt
    rec = Platt().fit(m.predecir(df), df["y"].to_numpy())
    arch = congelar.guardar_modelo(m, rec, tmp_path / "mortalidad")
    man = congelar.escribir_manifiesto(tmp_path, cargar_reglas(), {"mortalidad": arch}, {})
    import json
    d = json.loads(man.read_text())
    for x in d["archivos"]["mortalidad"].values():
        assert sha256_archivo(tmp_path / x["ruta"]) == x["sha256"]
    import joblib
    assert np.max(np.abs(joblib.load(arch["modelo_completo"]).predecir(df) - m.predecir(df))) == 0


def test_el_codigo_de_la_fase_4_no_llama_al_cortafuegos_de_2024():
    from pathlib import Path
    import re
    for f in (Path(__file__).resolve().parents[1] / "modelado").glob("*.py"):
        if f.name == "verificacion.py":
            continue                                                    # la verificación contiene los nombres a modo de búsqueda
        assert not re.search(r"leer_desenlace|marca_cohorte_d_2024|FASE3_AUTORIZA|desenlace_2024", f.read_text(encoding="utf8")), f.name


def test_la_privacidad_de_las_salidas_se_escanea(tmp_path):
    from modelado import privacidad
    (tmp_path / "a.csv").write_text("n,x\n5,1\n500,2\n", encoding="utf8")
    (tmp_path / "b.md").write_text("12.345.678-9 y 2023-04-05", encoding="utf8")
    h = {x["archivo"]: x for x in privacidad.escanear_carpeta(tmp_path)}
    assert h["a.csv"]["celdas_pequenas"] == 1 and h["b.md"]["rut"] == 1 and h["b.md"]["fechas"] == 1
