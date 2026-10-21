import math

import numpy as np
import pytest

from modelado import metricas, seleccion
from modelado.config import cargar_reglas


def simular(n=200000, shift=0.0, escala=1.0, semilla=0):
    rng = np.random.default_rng(semilla)
    z = rng.normal(-3, 1.2, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    p = 1 / (1 + np.exp(-(escala * z + shift)))
    return y, p


def test_un_modelo_bien_calibrado_da_pendiente_uno_y_oe_uno():
    y, p = simular()
    m = metricas.metricas_mortalidad(y, p)
    assert abs(m["pendiente"] - 1) < 0.03 and abs(m["oe_global"] - 1) < 0.03 and abs(m["citl"]) < 0.03


def test_un_modelo_que_sobreestima_tiene_oe_menor_que_uno():
    y, p = simular(shift=0.5)
    m = metricas.metricas_mortalidad(y, p)
    assert m["oe_global"] < 0.8 and m["citl"] < -0.3


def test_un_modelo_demasiado_extremo_tiene_pendiente_menor_que_uno():
    y, p = simular(escala=1.6)
    assert metricas.metricas_mortalidad(y, p)["pendiente"] < 0.8


def test_prediccion_constante_no_tiene_pendiente_pero_el_brier_skill_es_cero():
    y, _ = simular(10000)
    p = np.full(len(y), y.mean())
    m = metricas.metricas_mortalidad(y, p)
    assert math.isnan(m["pendiente"]) and abs(m["bss"]) < 1e-6


def test_un_buen_modelo_supera_a_la_prevalencia_en_brier():
    y, p = simular()
    assert metricas.metricas_mortalidad(y, p)["bss"] > 0.05


def test_pr_auc_de_un_azar_es_la_prevalencia():
    rng = np.random.default_rng(1)
    y = (rng.random(100000) < 0.05).astype(float)
    assert abs(metricas.metricas_mortalidad(y, rng.random(100000))["pr_auc"] - 0.05) < 0.01


def test_metricas_de_estadia():
    rng = np.random.default_rng(2)
    mu = rng.uniform(2, 12, 50000)
    y = rng.poisson(mu).astype(float)
    m = metricas.metricas_estadia(y, mu)
    assert abs(m["oe_global"] - 1) < 0.02 and m["rmse"] > 0 and m["devianza_poisson"] > 0
    assert metricas.metricas_estadia(y, mu * 1.5)["oe_global"] < 0.7


def test_tabla_de_calibracion_tiene_diez_grupos():
    y, p = simular(50000)
    t = metricas.tabla_calibracion(y, p)
    assert len(t) == 10 and t["n"].sum() == 50000 and (t["razon_oe"].between(0.5, 1.5)).all()


def cand(score, oe, pend=1.0):
    return {"score": score, "metricas_2023": {"oe_global": oe, "pendiente": pend}}


def test_si_los_arboles_no_mejoran_lo_suficiente_gana_la_logistica():
    r = cargar_reglas()
    s = seleccion.aplicar_regla("mortalidad", {"linea_base": cand(0.30, 0.9), "lightgbm": cand(0.305, 0.9), "xgboost": cand(0.306, 0.9)}, r)
    assert s["ganador"] == "linea_base" and not any(p["reemplaza"] for p in s["pasos"])


def test_un_retador_que_mejora_el_margen_y_calibra_igual_reemplaza():
    r = cargar_reglas()
    s = seleccion.aplicar_regla("mortalidad", {"linea_base": cand(0.30, 0.9), "lightgbm": cand(0.32, 0.9), "xgboost": cand(0.321, 0.9)}, r)
    assert s["ganador"] == "lightgbm"            # xgboost solo mejora 0,001 sobre lightgbm


def test_un_retador_con_peor_calibracion_no_reemplaza():
    r = cargar_reglas()
    s = seleccion.aplicar_regla("mortalidad", {"linea_base": cand(0.30, 0.95), "lightgbm": cand(0.35, 0.70), "xgboost": cand(0.30, 0.95)}, r)
    assert s["ganador"] == "linea_base" and s["pasos"][0]["mejora_suficiente"] and not s["pasos"][0]["calibracion_no_peor"]


def test_regla_de_estadia_usa_mejora_relativa_de_la_devianza():
    r = cargar_reglas()
    nada = seleccion.aplicar_regla("estadia", {"linea_base": cand(2.50, 1.0), "lightgbm": cand(2.49, 1.0), "xgboost": cand(2.48, 1.0)}, r)
    algo = seleccion.aplicar_regla("estadia", {"linea_base": cand(2.50, 1.0), "lightgbm": cand(2.40, 1.0), "xgboost": cand(2.40, 1.0)}, r)
    assert nada["ganador"] == "linea_base" and algo["ganador"] == "lightgbm"


def test_un_empate_entre_complejos_se_queda_con_el_mas_simple():
    r = cargar_reglas()
    s = seleccion.aplicar_regla("mortalidad", {"linea_base": cand(0.30, 1.0), "lightgbm": cand(0.34, 1.0), "xgboost": cand(0.345, 1.0)}, r)
    assert s["ganador"] == "lightgbm"
