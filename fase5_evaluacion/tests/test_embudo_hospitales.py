import numpy as np
import pandas as pd
from evaluacion import embudo, hospitales
from evaluacion.config import cargar_protocolo


def test_embudo_sin_diferencias_reales_marca_poco():
    esp = np.exp(np.random.default_rng(0).uniform(np.log(15), np.log(900), 65))
    r = embudo.simular_nulo(esp, 300, 2)
    assert 0.02 <= r["fuera_95"] <= 0.09 and r["fuera_998"] <= 0.01


def test_sobredispersion_ensancha_limites():
    esp = np.array([50.0, 200.0, 800.0])
    lo0, hi0 = embudo.limites(esp, 0.0, 1.96)
    lo1, hi1 = embudo.limites(esp, 0.02, 1.96)
    assert (hi1 > hi0).all() and (lo1 < lo0).all()


def test_tau2_cero_sin_variacion_real_y_positiva_con_ella():
    rng = np.random.default_rng(1)
    esp = np.full(60, 300.0)
    assert embudo.tau2(rng.poisson(esp), esp) < 0.01
    real = np.exp(rng.normal(0, 0.2, 60))
    assert embudo.tau2(rng.poisson(esp * real), esp) > 0.01


def test_ic_poisson_contiene_el_observado():
    obs, esp = np.array([20, 100]), np.array([22.0, 90.0])
    lo, hi = embudo.ic_poisson(obs, esp)
    assert (lo < obs / esp).all() and (hi > obs / esp).all()


def test_tabla_hospitales_volumen_minimo_y_sin_ranking():
    rng = np.random.default_rng(2)
    h = np.concatenate([np.full(2000, 1), np.full(1500, 2), np.full(100, 3)])
    e = np.full(len(h), 0.05)
    y = rng.binomial(1, e)
    t, r = hospitales.tabla(h, y, e, "mortalidad", cargar_protocolo())
    assert list(t["cod_hospital"]) == [1, 2]                      # el hospital de 100 egresos queda fuera
    assert r["n_bajo_volumen"] == 1 and r["egresos_bajo_volumen"] == 100
    assert t["cod_hospital"].is_monotonic_increasing             # orden por código, no por O/E


def test_oe_de_mortalidad_es_observado_sobre_esperado():
    h = np.full(1000, 7)
    e = np.full(1000, 0.1)
    y = np.zeros(1000); y[:80] = 1
    t, _ = hospitales.tabla(h, y, e, "mortalidad", cargar_protocolo())
    assert abs(t["oe"].iloc[0] - 0.8) < 1e-9
