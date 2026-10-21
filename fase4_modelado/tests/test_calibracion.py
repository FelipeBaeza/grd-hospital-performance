import numpy as np
import pandas as pd
import pytest

from modelado import calibracion, metricas


def datos(n=60000, shift=0.6, semilla=3):
    rng = np.random.default_rng(semilla)
    z = rng.normal(-3, 1.2, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    return y, 1 / (1 + np.exp(-(z + shift)))            # el modelo sobreestima


def test_platt_corrige_el_nivel():
    y, p = datos()
    antes = metricas.metricas_mortalidad(y, p)
    r = calibracion.Platt().fit(p, y)
    despues = metricas.metricas_mortalidad(y, r.transformar(p))
    assert antes["oe_global"] < 0.85 and abs(despues["oe_global"] - 1) < 0.02 and abs(despues["pendiente"] - 1) < 0.03


def test_factor_unico_deja_oe_en_uno():
    rng = np.random.default_rng(1)
    mu = rng.uniform(2, 10, 20000)
    y = rng.poisson(mu * 0.8).astype(float)
    r = calibracion.FactorUnico().fit(mu, y)
    assert abs(y.sum() / r.transformar(mu).sum() - 1) < 1e-9


def test_el_recalibrador_exige_filas_de_2023():
    y, p = datos(1000)
    with pytest.raises(calibracion.FueraDeAnio):
        calibracion.ajustar_final("mortalidad", "platt", p, y, np.array([2022] * 500 + [2023] * 500))
    r = calibracion.ajustar_final("mortalidad", "platt", p, y, np.full(1000, 2023))
    assert isinstance(r, calibracion.Platt)


def test_los_bloques_son_contiguos_en_el_tiempo_y_del_mismo_tamano():
    fechas = pd.Series(pd.date_range("2023-01-01", periods=1000, freq="D").to_numpy()[np.random.default_rng(0).permutation(1000)])
    b = calibracion.bloques_temporales(fechas, 5)
    assert sorted(np.bincount(b)) == [200] * 5
    for k in range(4):
        assert fechas[b == k].max() < fechas[b == k + 1].min()


def test_la_validacion_cruzada_nunca_ajusta_con_el_bloque_que_predice():
    y, p = datos(20000)
    bloques = calibracion.bloques_temporales(pd.Series(np.arange(len(y))), 5)

    class Espia(calibracion.Platt):
        vistos = []

        def fit(self, p_, y_):
            Espia.vistos.append(len(p_))
            return super().fit(p_, y_)
    oof = calibracion.validacion_cruzada_bloques("mortalidad", p, y, bloques, Espia)
    assert Espia.vistos == [16000] * 5 and len(oof) == 20000 and np.isfinite(oof).all()


def test_se_prefiere_el_metodo_simple_si_hay_empate():
    y, p = datos(30000)
    bloques = calibracion.bloques_temporales(pd.Series(np.arange(len(y))), 5)
    r = calibracion.comparar_metodos("mortalidad", p, y, bloques, empate=0.5)       # empate enorme: siempre el simple
    assert r["elegido"] == "platt" and set(r["metricas"]) == {"sin_calibrar", "platt", "isotonica"}
    assert r["metricas"]["platt"]["brier"] < r["metricas"]["sin_calibrar"]["brier"]


def test_la_isotonica_gana_si_la_forma_no_es_logistica():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.001, 0.4, 80000)
    y = (rng.random(80000) < np.where(p < 0.2, 0.01, 0.5)).astype(float)
    bloques = calibracion.bloques_temporales(pd.Series(np.arange(len(y))), 5)
    assert calibracion.comparar_metodos("mortalidad", p, y, bloques, empate=0.005)["elegido"] == "isotonica"
