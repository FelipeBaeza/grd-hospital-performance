"""Pruebas de valor conocido de las funciones de eda/estadistica.py (se amplía en cada tarea)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eda import estadistica as est


# ------------------------------------------------------------------------- Chao1 (T4)
def test_chao1_valor_conocido_corregido_por_sesgo():
    # S_obs=6, f1=3, f2=2 -> 6 + 3*2/(2*(2+1)) = 7
    assert est.chao1([1, 1, 1, 2, 2, 5]) == pytest.approx(7.0)
    # sin singletons no hay corrección
    assert est.chao1([3, 4, 5]) == 3.0
    # todo singleton: S + f1(f1-1)/2
    assert est.chao1([1, 1, 1, 1]) == pytest.approx(4 + 4 * 3 / 2)
    assert est.chao1([]) == 0.0
    assert est.frecuencia_de_frecuencias([1, 1, 2, 7]) == (4, 2, 1)


# --------------------------------------------------------------------------- PSI (T4)
def test_psi_cero_en_distribuciones_iguales():
    assert est.psi({"a": 10, "b": 30}, {"a": 100, "b": 300}) == pytest.approx(0.0, abs=1e-12)
    assert est.psi([5, 5], [5, 5]) == pytest.approx(0.0, abs=1e-12)


def test_psi_valor_conocido():
    # p=(0.5,0.5), q=(0.6,0.4): 0.1 ln(1.2) + (-0.1) ln(0.8)
    esperado = 0.1 * math.log(1.2) - 0.1 * math.log(0.8)
    assert est.psi([50, 50], [60, 40]) == pytest.approx(esperado, rel=1e-9)
    assert est.psi({"x": 50, "y": 50}, {"y": 40, "x": 60}) == pytest.approx(esperado, rel=1e-9)


def test_psi_es_simetrico_y_detecta_categorias_nuevas():
    a, b = {"a": 70, "b": 30}, {"a": 40, "b": 60}
    assert est.psi(a, b) == pytest.approx(est.psi(b, a))
    nueva = est.psi({"a": 100}, {"a": 50, "b": 50})
    assert nueva > 0.25  # una categoría que aparece es un cambio grande
    assert math.isnan(est.psi({"a": 0}, {"a": 1}))


def test_clasificar_psi():
    assert est.clasificar_psi(0.05, 0.10, 0.25) == "estable"
    assert est.clasificar_psi(0.10, 0.10, 0.25) == "moderado"
    assert est.clasificar_psi(0.30, 0.10, 0.25) == "alto"
    assert est.clasificar_psi(float("nan"), 0.10, 0.25) == "no evaluable"


# ------------------------------------------------------------------ proporciones (T7)
@pytest.mark.parametrize(
    "k, n, bajo, alto",
    [  # Newcombe (1998), tabla 1: método de Wilson sin corrección
        (81, 263, 0.2553, 0.3662),
        (15, 148, 0.0624, 0.1605),
        (0, 20, 0.0, 0.1611),
        (0, 29, 0.0, 0.1170),
        (5, 10, 0.2366, 0.7634),
    ],
)
def test_wilson_reproduce_los_valores_publicados(k, n, bajo, alto):
    lo, hi = est.wilson(k, n)
    assert float(lo) == pytest.approx(bajo, abs=1e-4)
    assert float(hi) == pytest.approx(alto, abs=1e-4)


def test_wilson_acepta_arreglos_y_devuelve_nan_sin_observaciones():
    lo, hi = est.wilson(np.array([0, 5, 3]), np.array([0, 10, 10]))
    assert np.isnan(lo[0]) and np.isnan(hi[0])
    assert lo[1] == pytest.approx(0.2366, abs=1e-4)
    assert (lo <= hi)[1:].all() and (lo[1:] >= 0).all() and (hi[1:] <= 1).all()


def test_razon_de_proporciones_katz_valor_conocido():
    # 20/100 contra 10/100: razón 2; EE del log = sqrt(1/20 - 1/100 + 1/10 - 1/100) = sqrt(0,13)
    rr, lo, hi = est.razon_de_proporciones(20, 100, 10, 100)
    se = math.sqrt(0.13)
    assert rr == pytest.approx(2.0)
    assert lo == pytest.approx(2.0 * math.exp(-1.959964 * se), rel=1e-6)
    assert hi == pytest.approx(2.0 * math.exp(1.959964 * se), rel=1e-6)
    assert lo < 1.0 < hi  # con estos conteos el IC incluye 1


def test_razon_de_proporciones_sin_eventos_en_el_comparado_o_la_referencia():
    # sin eventos en el grupo comparado: razón 0 y límite superior desde Wilson
    rr, lo, hi = est.razon_de_proporciones(0, 1000, 50, 1000)
    assert rr == 0.0 and lo == 0.0
    _, alto = est.wilson(0, 1000)
    assert hi == pytest.approx(float(alto) / 0.05)
    # sin eventos en la referencia o sin denominador: no se puede estimar
    for args in ((5, 100, 0, 100), (5, 0, 5, 100), (5, 100, 5, 0)):
        assert all(math.isnan(x) for x in est.razon_de_proporciones(*args))


def test_percentil_de_conteos_es_el_primer_valor_que_alcanza_la_fraccion():
    valores, conteos = [0, 1, 2, 3, 10], [10, 30, 30, 20, 10]  # total 100
    assert est.percentil_de_conteos(valores, conteos, 0.5) == 2  # acumulado 10, 40, 70
    assert est.percentil_de_conteos(valores, conteos, 0.4) == 1
    assert est.percentil_de_conteos(valores, conteos, 0.9) == 3
    assert est.percentil_de_conteos(valores, conteos, 0.99) == 10
    assert est.percentil_de_conteos([5, 1], [1, 1], 0.5) == 1  # no exige orden de entrada
    assert math.isnan(est.percentil_de_conteos([], [], 0.5))


def test_momentos_de_conteos_coincide_con_numpy():
    valores, conteos = [0, 1, 4, 9], [3, 5, 2, 1]
    datos = np.repeat(valores, conteos)
    n, media, var = est.momentos_de_conteos(valores, conteos)
    assert n == len(datos)
    assert media == pytest.approx(datos.mean())
    assert var == pytest.approx(datos.var(ddof=1))
    assert est.momentos_de_conteos([], [])[0] == 0


# ------------------------------------------------------------- embudo (T9)
def test_winsorizar_por_rango_reemplaza_los_extremos_por_el_vecino():
    z = [-3.0, -1.0, 0.0, 1.0, 3.0]
    assert est.winsorizar(z, 0.2).tolist() == [-1.0, -1.0, 0.0, 1.0, 1.0]       # floor(0,2 * 5) = 1 valor por cola
    assert est.winsorizar(z, 0.1).tolist() == sorted(z)                         # floor(0,1 * 5) = 0: no cambia
    assert est.winsorizar([5.0, None, float("nan"), -5.0, 0.0], 0.0).tolist() == [-5.0, 0.0, 5.0]   # descarta faltantes
    assert est.winsorizar([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0], 0.1).tolist()[-2:] == [9.0, 9.0]


def test_phi_de_spiegelhalter_es_el_promedio_de_z_cuadrado_winsorizado_y_nunca_baja_de_uno():
    z = [-3.0, -1.0, 0.0, 1.0, 3.0]
    assert est.phi_spiegelhalter(z, 0.2) == 1.0                                 # el promedio es 0,8: no se ajusta
    z_ancho = [-4.0, -2.0, -2.0, 0.0, 2.0, 2.0, 4.0, 40.0, -40.0, 3.0]
    esperado = float(np.mean(est.winsorizar(z_ancho, 0.1) ** 2))
    assert est.phi_spiegelhalter(z_ancho, 0.1) == pytest.approx(esperado) and esperado > 1.0
    assert est.phi_spiegelhalter([10.0, -10.0], 0.1) == 1.0                     # con menos de tres instituciones no se estima
    # un valor extremo no domina: sin winsorizar el promedio sería mucho mayor
    assert est.phi_spiegelhalter(z_ancho, 0.1) < float(np.mean(np.square(z_ancho)))


def test_limites_del_embudo_y_puntaje_z():
    lo, hi = est.limites_proporcion(0.05, [100, 10000], 1.959964)
    se = math.sqrt(0.05 * 0.95 / 100)
    assert float(hi[0]) == pytest.approx(0.05 + 1.959964 * se) and float(lo[0]) == pytest.approx(0.05 - 1.959964 * se)
    assert (hi - lo)[1] < (hi - lo)[0]                                          # más volumen, límites más estrechos
    lo1, hi1 = est.limites_proporcion(0.2, [400], 1.959964)
    lo2, hi2 = est.limites_proporcion(0.2, [400], 1.959964, phi=4.0)
    assert float(hi2[0] - lo2[0]) == pytest.approx(2 * float(hi1[0] - lo1[0]))   # phi = 4 duplica el ancho (raíz de 4)
    lo3, _ = est.limites_proporcion(0.01, [50], 3.090232)
    assert float(lo3[0]) == 0.0                                                 # el límite inferior no baja de cero
    z = est.puntaje_z_proporcion([10, 5], [100, 100], 0.05)
    assert z[0] == pytest.approx((0.10 - 0.05) / math.sqrt(0.05 * 0.95 / 100)) and z[1] == pytest.approx(0.0)
    assert np.isnan(est.puntaje_z_proporcion([1], [0], 0.05)[0])
