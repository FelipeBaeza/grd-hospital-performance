"""Funciones estadísticas de las Tareas 10 y 11 (asociación y selección de variables): cuentas a mano, bordes y oráculos de scipy y scikit-learn."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import sparse, stats
from scipy.cluster import hierarchy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, normalized_mutual_info_score, roc_auc_score

from eda import estadistica as E


def _sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------------------- publicación
def test_redondear_publicado_usa_el_redondeo_de_polars_y_no_el_de_python():
    # empates decimales exactos: np.round y polars dan 0.8 donde round de Python da 0.799999
    assert E.redondear_publicado(0.7999995) == 0.8
    assert E.redondear_publicado(0.8499995) == 0.85
    assert E.redondear_publicado(0.7999996) == 0.8
    assert E.redondear_publicado(0.7999999999999999) == 0.8
    assert E.redondear_publicado(0.799999) == 0.799999
    assert math.isnan(E.redondear_publicado(float("nan")))
    assert E.DECIMALES_PUBLICADOS == 6
    # arreglos: mismo resultado elemento a elemento y NaN intacto
    r = E.redondear_publicado(np.array([0.7999995, np.nan, 0.5]))
    assert r[0] == 0.8 and np.isnan(r[1]) and r[2] == 0.5


# ------------------------------------------------------------------------------------------ Spearman
def test_spearman_a_mano():
    rho, n = E.spearman([1, 2, 3, 4], [1, 3, 2, 4])
    assert rho == pytest.approx(0.8, abs=1e-12) and n == 4
    assert E.spearman([1, 2, 3, 4], [4, 3, 2, 1])[0] == pytest.approx(-1.0)
    assert E.spearman([1, 2, 3, 4], [2, 4, 6, 8])[0] == pytest.approx(1.0)


def test_spearman_con_empates_coincide_con_scipy_en_200_pares_aleatorios():
    rng = np.random.default_rng(11)
    for _ in range(200):
        n = int(rng.integers(5, 60))
        x = rng.integers(0, 6, size=n).astype(float)           # valores repetidos a propósito
        y = (x + rng.integers(0, 4, size=n)).astype(float)
        if x.min() == x.max() or y.min() == y.max():
            continue
        rho, m = E.spearman(x, y)
        assert m == n and rho == pytest.approx(stats.spearmanr(x, y).statistic, abs=1e-12)


def test_spearman_descarta_pares_con_nan_y_cuenta_los_usados():
    x = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]
    y = [2.0, 1.0, 9.0, 4.0, np.nan, 7.0]
    rho, n = E.spearman(x, y)
    ok = [0, 1, 3, 5]
    assert n == 4 and rho == pytest.approx(stats.spearmanr(np.array(x)[ok], np.array(y)[ok]).statistic)


def test_spearman_degenerado_devuelve_nan_sin_excepcion():
    assert math.isnan(E.spearman([1, 1, 1, 1], [1, 2, 3, 4])[0])          # constante
    assert math.isnan(E.spearman([1, 2, 3, 4], [5, 5, 5, 5])[0])
    rho, n = E.spearman([1, 2], [2, 1])                                    # menos de 3 pares
    assert math.isnan(rho) and n == 2
    rho, n = E.spearman([1, 2, np.nan, 4], [1, np.nan, 3, np.nan])       # un solo par completo
    assert math.isnan(rho) and n == 1
    assert E.spearman([], [])[1] == 0


# ---------------------------------------------------------------------------------------------- phi
def test_phi_a_mano_con_dos_tablas():
    def phi_de(a, b, c, d):
        x = np.array([1] * a + [1] * b + [0] * c + [0] * d, dtype=float)
        y = np.array([1] * a + [0] * b + [1] * c + [0] * d, dtype=float)
        phi, npares = E.phi_pares(np.c_[x, y], np.ones((x.size, 2), dtype=bool))
        assert npares[0, 1] == x.size
        return phi[0, 1]

    assert phi_de(40, 10, 10, 40) == pytest.approx(0.6, abs=1e-12)        # (40*40 - 10*10) / (50*50)
    assert phi_de(45, 5, 5, 45) == pytest.approx(0.8, abs=1e-12)
    assert phi_de(10, 40, 40, 10) == pytest.approx(-0.6, abs=1e-12)


def test_phi_coincide_con_matthews_y_con_el_calculo_par_a_par_con_nulos():
    rng = np.random.default_rng(3)
    n, p = 400, 5
    X = (rng.random((n, p)) < np.array([0.5, 0.3, 0.7, 0.2, 0.45])).astype(float)
    X[:, 1] = np.where(rng.random(n) < 0.4, X[:, 0], X[:, 1])             # asociación real
    phi, npares = E.phi_pares(X, np.ones((n, p), dtype=bool))
    for i in range(p):
        for j in range(p):
            if i != j:
                assert phi[i, j] == pytest.approx(matthews_corrcoef(X[:, i], X[:, j]), abs=1e-12)
    assert np.allclose(np.diag(phi), 1.0) and np.all(npares == n)
    # con observaciones inválidas: solo cuentan las filas válidas en ambas variables de cada par
    V = rng.random((n, p)) > 0.15
    Xn = np.where(V, X, 7.0)                                               # el valor de una fila inválida no se mira
    phi2, np2 = E.phi_pares(Xn, V)
    for i in range(p):
        for j in range(p):
            ok = V[:, i] & V[:, j]
            assert np2[i, j] == ok.sum()
            if i != j:
                esperado = matthews_corrcoef(X[ok, i], X[ok, j])
                assert phi2[i, j] == pytest.approx(esperado, abs=1e-12), (i, j)
    assert np.allclose(phi2, phi2.T, equal_nan=True)


def test_phi_diagonal_es_nan_si_la_variable_no_varia_y_el_llamador_impone_el_uno():
    X = np.c_[np.zeros(50), np.r_[np.ones(20), np.zeros(30)], np.r_[np.ones(25), np.zeros(25)]]
    phi, _ = E.phi_pares(X, np.ones(X.shape, dtype=bool))
    assert math.isnan(phi[0, 0]) and np.isnan(phi[0, 1]) and np.isnan(phi[1, 0])
    assert phi[1, 1] == pytest.approx(1.0) and phi[2, 2] == pytest.approx(1.0)
    assert not math.isnan(phi[1, 2])


# ------------------------------------------------------------------------------------ contingencia
def test_tabla_de_contingencia_cuenta_y_descarta_codigos_negativos():
    a = np.array([0, 0, 1, 2, 2, 2, -1, 1])
    b = np.array([0, 1, 1, 0, 0, 2, 0, -1])
    t = E.tabla_contingencia(a, b, 3, 3)
    esperado = np.zeros((3, 3), dtype=int)
    for u, v in zip(a, b):
        if u >= 0 and v >= 0:
            esperado[u, v] += 1
    assert t.shape == (3, 3) and np.array_equal(t, esperado) and t.sum() == 6
    # un nivel declarado que no aparece da una fila de ceros, no se pierde
    assert E.tabla_contingencia([0, 0], [0, 1], 3, 2).shape == (3, 2)


# --------------------------------------------------------------------------------------- V de Cramér
@pytest.mark.parametrize(
    "tabla, esperado",
    [
        ([[30, 10], [10, 30]], 0.490290),                      # la de scipy con correction=True da 0,475 (Yates): no es esta
        ([[20, 10, 5], [5, 15, 10]], 0.381613),
        ([[8, 2, 1], [3, 9, 4], [2, 2, 10]], 0.459457),
        ([[10, 10], [10, 10]], 0.0),
        ([[50, 0], [0, 50]], 1.0),
        ([[2, 0], [0, 2]], 1.0),                               # n = 4
    ],
)
def test_v_corregida_a_mano(tabla, esperado):
    assert E.v_cramer_sesgo(tabla) == pytest.approx(esperado, abs=1e-6)


def test_v_corregida_no_es_la_correccion_de_yates_de_scipy():
    t = np.array([[30, 10], [10, 30]])
    assert stats.contingency.association(t, method="cramer", correction=True) == pytest.approx(0.475, abs=1e-3)
    assert E.v_cramer_sesgo(t) == pytest.approx(0.490290, abs=1e-6)


@pytest.mark.parametrize("tabla", [[[1, 0], [0, 1]], np.eye(3), [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[5]], [[0, 0], [0, 0]], [[7, 3]]])
def test_v_corregida_de_tablas_casi_vacias_es_nan_sin_excepcion(tabla):
    assert math.isnan(E.v_cramer_sesgo(tabla))


def test_v_corregida_nunca_supera_a_la_sin_corregir_ni_sale_de_cero_a_uno_y_es_simetrica():
    rng = np.random.default_rng(5)
    for _ in range(300):
        r, c = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        t = rng.integers(0, 30, size=(r, c))
        v = E.v_cramer_sesgo(t)
        if math.isnan(v):
            continue
        assert 0.0 <= v <= 1.0
        assert v <= stats.contingency.association(t[t.sum(1) > 0][:, t.sum(0) > 0], method="cramer") + 1e-12
        assert v == pytest.approx(E.v_cramer_sesgo(t.T), abs=1e-12)


def test_v_corregida_converge_a_la_v_de_scipy_al_multiplicar_la_tabla_y_descarta_filas_vacias():
    t = np.array([[20, 10, 5], [5, 15, 10]])
    grande = E.v_cramer_sesgo(t * 1000)
    assert grande == pytest.approx(stats.contingency.association(t * 1000, method="cramer"), abs=1e-3)
    con_vacias = np.array([[20, 10, 5, 0], [0, 0, 0, 0], [5, 15, 10, 0]])
    assert E.v_cramer_sesgo(con_vacias) == pytest.approx(E.v_cramer_sesgo(t), abs=1e-12)
    assert math.isnan(E.v_cramer_sesgo([[10, 0], [20, 0]]))              # un solo nivel no vacío


# ----------------------------------------------------------------------------------------------- eta
def test_eta_cuadrado_a_mano_y_contra_f_oneway():
    eta2, n = E.eta_cuadrado([1, 2, 3, 7, 8, 9], [0, 0, 0, 1, 1, 1])
    assert eta2 == pytest.approx(54 / 58, abs=1e-12) and n == 6
    rng = np.random.default_rng(2)
    g = rng.integers(0, 5, size=300)
    v = rng.normal(size=300) + 0.4 * g
    eta2, n = E.eta_cuadrado(v, g, 5)
    f = stats.f_oneway(*[v[g == k] for k in range(5)]).statistic
    k = 5
    assert eta2 == pytest.approx(f * (k - 1) / (f * (k - 1) + n - k), abs=1e-12) and n == 300


def test_eta_cuadrado_descarta_nan_y_codigos_negativos_y_marca_lo_degenerado():
    v = [1.0, 2.0, 3.0, np.nan, 7.0, 8.0, 9.0, 100.0]
    g = [0, 0, 0, 0, 1, 1, 1, -1]
    eta2, n = E.eta_cuadrado(v, g, 2)
    assert n == 6 and eta2 == pytest.approx(54 / 58)
    assert math.isnan(E.eta_cuadrado([1, 2, 3], [0, 0, 0])[0])            # un solo grupo no vacío
    assert math.isnan(E.eta_cuadrado([4, 4, 4, 4], [0, 0, 1, 1])[0])       # SST = 0
    assert E.eta_cuadrado([4.0, 5.0], [0, 1])[0] == pytest.approx(1.0)
    assert E.eta_cuadrado([], [])[1] == 0


# ----------------------------------------------------------------------------------------------- AUC
def test_auc_a_mano_y_contra_scikit_learn_con_empates():
    assert E.auc_por_rangos([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == (0.75, 2, 2)
    assert E.auc_por_rangos([1, 1, 2, 2], [0, 1, 0, 1]) == (0.5, 2, 2)
    rng = np.random.default_rng(8)
    for _ in range(50):
        n = int(rng.integers(20, 200))
        puntaje = rng.integers(0, 8, size=n).astype(float)
        y = (rng.random(n) < _sigmoide(puntaje - 4)).astype(float)
        if y.min() == y.max():
            continue
        auc, n1, n0 = E.auc_por_rangos(puntaje, y)
        assert auc == pytest.approx(roc_auc_score(y, puntaje), abs=1e-12) and (n1, n0) == (int(y.sum()), int(n - y.sum()))


def test_auc_descarta_nan_y_una_sola_clase_da_nan():
    auc, n1, n0 = E.auc_por_rangos([0.1, np.nan, 0.35, 0.8, 0.9], [0, 1, 1, 1, np.nan])
    assert (n1, n0) == (2, 1) and auc == pytest.approx(1.0)
    auc, n1, n0 = E.auc_por_rangos([1, 2, 3], [1, 1, 1])
    assert math.isnan(auc) and (n1, n0) == (3, 0)


def test_hanley_y_mcneil_a_mano_y_con_n1_y_n0_asimetricos():
    bajo, alto = E.ic_hanley_mcneil(0.8, 50, 50)
    assert (alto - bajo) / 2 == pytest.approx(1.959964 * 0.044502, abs=1e-6)            # error estándar 0,044502
    bajo, alto = E.ic_hanley_mcneil(0.7, 30, 70)
    assert (alto - bajo) / 2 == pytest.approx(1.959964 * 0.060087, abs=1e-6)            # n1 y n0 intercambiados darían otro valor
    b2, a2 = E.ic_hanley_mcneil(0.7, 70, 30)
    assert (a2 - b2) / 2 != pytest.approx((alto - bajo) / 2, abs=1e-6)
    assert E.ic_hanley_mcneil(0.7, 30, 70, z=1.0)[1] - E.ic_hanley_mcneil(0.7, 30, 70, z=1.0)[0] == pytest.approx(2 * 0.060087, abs=1e-6)


def test_hanley_y_mcneil_recorta_a_cero_y_uno_y_devuelve_nan_si_falta_una_clase():
    bajo, alto = E.ic_hanley_mcneil(0.99, 5, 5)
    assert alto == 1.0 and 0.0 <= bajo < 0.99
    bajo, alto = E.ic_hanley_mcneil(0.01, 5, 5)
    assert bajo == 0.0
    for args in ((0.8, 0, 50), (0.8, 50, 0), (float("nan"), 10, 10)):
        b, a = E.ic_hanley_mcneil(*args)
        assert math.isnan(b) and math.isnan(a)


# ---------------------------------------------------------------------------------------------- Woolf
def test_woolf_a_mano_tres_casos():
    razon, bajo, alto, corregido = E.or_woolf(20, 80, 10, 190)
    assert razon == pytest.approx(4.75) and bajo == pytest.approx(2.128388, abs=1e-5) and alto == pytest.approx(10.600746, abs=1e-5) and corregido is False
    razon, bajo, alto, corregido = E.or_woolf(0, 50, 5, 45)                                   # celda en cero: Haldane-Anscombe
    assert razon == pytest.approx(0.081908, abs=1e-6) and bajo == pytest.approx(0.004406, abs=1e-6) and alto == pytest.approx(1.522707, abs=1e-5) and corregido is True
    razon, bajo, alto, corregido = E.or_woolf(30, 30, 30, 30)
    assert razon == pytest.approx(1.0) and bajo < 1.0 < alto and corregido is False


def test_woolf_coincide_con_scipy_y_marca_la_correccion_solo_con_ceros():
    for tabla in ([[20, 80], [10, 190]], [[5, 15], [40, 60]], [[100, 300], [50, 700]]):
        (a, b), (c, d) = tabla
        razon, bajo, alto, corregido = E.or_woolf(a, b, c, d)
        ic = stats.contingency.odds_ratio(tabla, kind="sample").confidence_interval(0.95)
        assert razon == pytest.approx(stats.contingency.odds_ratio(tabla, kind="sample").statistic, rel=1e-9)
        assert bajo == pytest.approx(ic.low, rel=1e-5) and alto == pytest.approx(ic.high, rel=1e-5) and corregido is False


@pytest.mark.parametrize("celdas", [(0, 0, 5, 5), (5, 5, 0, 0), (0, 5, 0, 5), (5, 0, 5, 0), (0, 0, 0, 0)])
def test_woolf_con_una_fila_o_columna_completa_en_cero_es_nan(celdas):
    razon, bajo, alto, corregido = E.or_woolf(*celdas)
    assert math.isnan(razon) and math.isnan(bajo) and math.isnan(alto) and corregido is False


# ------------------------------------------------------------------ Benjamini-Hochberg y Wald
def test_benjamini_hochberg_contra_scipy_incluido_un_caso_no_monotono():
    p = np.array([0.001, 0.2, 0.012, 0.03, 0.04, 0.5, 0.9, 0.0001])
    assert np.allclose(E.benjamini_hochberg(p), stats.false_discovery_control(p, method="bh"), atol=1e-15)
    no_monotono = np.array([0.01, 0.04, 0.03, 0.5, 0.0301])                         # p(2)·m/2 supera a p(3)·m/3: exige el mínimo acumulado
    q = E.benjamini_hochberg(no_monotono)
    assert np.allclose(q, stats.false_discovery_control(no_monotono, method="bh"), atol=1e-15)
    orden = np.argsort(no_monotono, kind="stable")
    assert np.all(np.diff(q[orden]) >= -1e-15)                                        # monótono en el orden de los p
    assert np.all(q >= no_monotono) and np.all(q <= 1.0)


def test_benjamini_hochberg_conserva_los_nan_y_no_los_cuenta_en_m():
    p = np.array([0.01, np.nan, 0.04, 0.03, np.nan])
    q = E.benjamini_hochberg(p)
    assert np.isnan(q[1]) and np.isnan(q[4])
    assert np.allclose(q[[0, 2, 3]], stats.false_discovery_control([0.01, 0.04, 0.03], method="bh"))
    assert np.isnan(E.benjamini_hochberg([np.nan, np.nan])).all() and E.benjamini_hochberg([]).size == 0


def test_p_wald_coincide_con_la_normal_y_marca_los_errores_estandar_invalidos():
    beta = np.array([0.0, 0.5, -1.96, 3.0, 1.0, 1.0, 1.0])
    se = np.array([1.0, 0.25, 1.0, 1.0, 0.0, -1.0, np.nan])
    p = E.p_wald(beta, se)
    esperado = 2 * stats.norm.sf(np.abs(beta[:4] / se[:4]))
    assert np.allclose(p[:4], esperado, atol=1e-15) and np.isnan(p[4:]).all()
    assert E.p_wald(1.96, 1.0) == pytest.approx(0.0499958, abs=1e-6) and isinstance(E.p_wald(1.0, 1.0), float)


# ------------------------------------------------------------------------- información mutua
def test_informacion_mutua_normalizada_contra_scikit_learn_y_a_mano():
    rng = np.random.default_rng(4)
    for _ in range(30):
        n = int(rng.integers(50, 300))
        a = rng.integers(0, int(rng.integers(2, 6)), size=n)
        b = np.where(rng.random(n) < 0.5, a, rng.integers(0, int(rng.integers(2, 6)), size=n))
        assert E.informacion_mutua_normalizada(a, b) == pytest.approx(normalized_mutual_info_score(a, b), abs=1e-12)
    assert E.informacion_mutua_normalizada([0, 0, 1, 1, 2, 2], [5, 5, 7, 7, 9, 9]) == pytest.approx(1.0)       # idénticas salvo el nombre de los códigos
    assert E.informacion_mutua_normalizada([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0, abs=1e-12)        # independientes exactas


def test_informacion_mutua_con_constantes_y_faltantes():
    assert E.informacion_mutua_normalizada([0, 0, 0, 0], [0, 1, 0, 1]) == 0.0
    assert E.informacion_mutua_normalizada([0, 1, 0, 1], [3, 3, 3, 3]) == 0.0
    assert math.isnan(E.informacion_mutua_normalizada([2, 2, 2], [4, 4, 4]))
    assert E.informacion_mutua_normalizada([0, 0, 1, 1, -1], [0, 0, 1, 1, 0]) == pytest.approx(1.0)       # el código negativo (faltante) se descarta
    assert math.isnan(E.informacion_mutua_normalizada([-1, -1], [0, 1]))


# ---------------------------------------------------------------------------- entradas inválidas
CASOS_INVALIDOS = [
    ("spearman", lambda: E.spearman([1, 2, 3], [1, 2]), "'x' e 'y'"),
    ("phi_pares", lambda: E.phi_pares(np.zeros((4, 2)), np.ones((4, 3), dtype=bool)), "'X' y 'V'"),
    ("phi_pares", lambda: E.phi_pares(np.array([[0.0, 2.0], [1.0, 1.0]]), np.ones((2, 2), dtype=bool)), "'X'"),
    ("tabla_contingencia", lambda: E.tabla_contingencia([0, 1, 3], [0, 1, 0], 3, 2), "'a'"),               # el caso que antes pasaba en silencio
    ("tabla_contingencia", lambda: E.tabla_contingencia([0, 1], [0, 2], 3, 2), "'b'"),
    ("tabla_contingencia", lambda: E.tabla_contingencia([0, 1], [0], 2, 2), "'a' e 'b'"),
    ("tabla_contingencia", lambda: E.tabla_contingencia([0, 1], [0, 1], 0, 2), "'na'"),
    ("v_cramer_sesgo", lambda: E.v_cramer_sesgo([[1, -2], [3, 4]]), "'tabla'"),
    ("v_cramer_sesgo", lambda: E.v_cramer_sesgo([[1, np.nan], [3, 4]]), "'tabla'"),
    ("eta_cuadrado", lambda: E.eta_cuadrado([1.0, 2.0, 3.0], [0, 1]), "'valores' e 'grupos'"),
    ("eta_cuadrado", lambda: E.eta_cuadrado([1.0, 2.0, 3.0], [0, 1, 2], 2), "'grupos'"),
    ("eta_cuadrado", lambda: E.eta_cuadrado([1.0, 2.0], [0, 1], 0), "'n_grupos'"),
    ("auc_por_rangos", lambda: E.auc_por_rangos(np.arange(10.0), np.arange(9) % 2), "'puntaje' e 'y'"),
    ("auc_por_rangos", lambda: E.auc_por_rangos([1.0, 2.0, 3.0], [0, 1, 2]), "'y'"),
    ("ic_hanley_mcneil", lambda: E.ic_hanley_mcneil(1.2, 10, 10), "'auc'"),
    ("ic_hanley_mcneil", lambda: E.ic_hanley_mcneil(0.7, -1, 10), "'n1'"),
    ("ic_hanley_mcneil", lambda: E.ic_hanley_mcneil(0.7, 10, float("inf")), "'n0'"),
    ("or_woolf", lambda: E.or_woolf(-1, 5, 5, 5), "'a'"),
    ("or_woolf", lambda: E.or_woolf(1, 5, float("nan"), 5), "'c'"),
    ("p_wald", lambda: E.p_wald([1.0, 2.0], [1.0]), "'beta' y 'se'"),
    ("benjamini_hochberg", lambda: E.benjamini_hochberg([0.1, 1.5]), "'p'"),
    ("benjamini_hochberg", lambda: E.benjamini_hochberg([-0.1, 0.5]), "'p'"),
    ("informacion_mutua_normalizada", lambda: E.informacion_mutua_normalizada([0, 1, 2], [0, 1]), "'a' e 'b'"),
    ("informacion_mutua_normalizada", lambda: E.informacion_mutua_normalizada([0.5, 1.5], [0, 1]), "'a'"),
    ("regresion_logistica", lambda: E.regresion_logistica(np.ones((5, 2)), [0, 1, 0, 1]), "'X' y 'y'"),
    ("regresion_logistica", lambda: E.regresion_logistica(np.ones((4, 2)), [0, 1, 2, 1]), "'y'"),
    ("regresion_logistica", lambda: E.regresion_logistica(np.ones((4, 2)), [0, 1, np.nan, 1]), "'y'"),
    ("regresion_logistica", lambda: E.regresion_logistica(np.array([[1.0, np.inf], [1.0, 2.0]]), [0, 1]), "'X'"),
    ("regresion_logistica", lambda: E.regresion_logistica(np.ones((4, 2)), [0, 1, 0, 1], ridge=-1.0), "'ridge'"),
    ("regresion_logistica", lambda: E.regresion_logistica(np.ones((4, 2)), [0, 1, 0, 1], bloque=0), "'bloque'"),
    ("regresion_logistica", lambda: E.regresion_logistica(np.ones((4, 2)), [0, 1, 0, 1], max_iter=0), "'max_iter'"),
    ("regresion_logistica", lambda: E.regresion_logistica(np.ones((0, 2)), []), "'X'"),                     # sin filas no hay modelo que ajustar
    ("regresion_logistica", lambda: E.regresion_logistica(np.ones((4, 0)), [0, 1, 0, 1], intercepto=False), "'X'"),
    ("vif_desde_correlacion", lambda: E.vif_desde_correlacion(np.ones((2, 3))), "'R'"),
    ("vif_desde_correlacion", lambda: E.vif_desde_correlacion(np.array([[1.0, np.nan], [np.nan, 1.0]])), "'R'"),
    ("orden_por_clusters", lambda: E.orden_por_clusters(np.ones((2, 3))), "'distancia'"),
    ("orden_por_clusters", lambda: E.orden_por_clusters(np.zeros((3, 3)), "ward"), "'metodo'"),
    ("componentes_redundantes", lambda: E.componentes_redundantes(np.ones((2, 3)), 0.8), "'A'"),
    ("componentes_redundantes", lambda: E.componentes_redundantes(np.eye(3), float("nan")), "'umbral'"),
    ("puntaje_categorico", lambda: E.puntaje_categorico([0, 1, 2], [0, 1, 0], 2), "'codigos'"),
    ("puntaje_categorico", lambda: E.puntaje_categorico([0, 1], [0, 1, 0], 2), "'codigos' e 'y'"),
    ("puntaje_categorico", lambda: E.puntaje_categorico([0, 1], [0, 2], 2), "'y'"),
    ("puntaje_categorico", lambda: E.puntaje_categorico([0, 1], [0, 1], 2, m=-1.0), "'m'"),
    ("bordes_deciles", lambda: E.bordes_deciles([1.0, 2.0, 3.0], n_bins=1), "'n_bins'"),
    ("asignar_bins", lambda: E.asignar_bins([1.0, 2.0], [5.0, 3.0]), "'bordes'"),
    ("psi_de_codigos", lambda: E.psi_de_codigos([0, 1, 2], [0, 1, 1], 2), "'base'"),
    ("psi_de_codigos", lambda: E.psi_de_codigos([0, 1], [0, 3], 3), "'nuevo'"),
    ("psi_de_codigos", lambda: E.psi_de_codigos([0, 1], [0, 1], 0), "'n_codigos'"),
]


@pytest.mark.parametrize("funcion, llamada, argumento", CASOS_INVALIDOS, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASOS_INVALIDOS)])
def test_una_entrada_invalida_lanza_value_error_que_nombra_la_funcion_y_el_argumento(funcion, llamada, argumento):
    with pytest.raises(ValueError) as excepcion:
        llamada()
    mensaje = str(excepcion.value)
    assert mensaje.startswith(funcion + ":") and argumento in mensaje, mensaje


def test_el_mensaje_de_largos_distintos_es_el_del_diseno():
    with pytest.raises(ValueError, match=r"auc_por_rangos: 'puntaje' e 'y' tienen largos distintos \(10 y 9\)"):
        E.auc_por_rangos(np.arange(10.0), np.arange(9) % 2)


def test_lo_degenerado_es_valido_y_no_lanza_excepcion():
    assert math.isnan(E.spearman([1, 1, 1], [1, 2, 3])[0])
    assert math.isnan(E.v_cramer_sesgo([[1, 0], [0, 1]]))
    assert math.isnan(E.eta_cuadrado([1.0, 2.0], [0, 0])[0])
    assert math.isnan(E.auc_por_rangos([1.0, 2.0], [1, 1])[0])
    assert math.isnan(E.or_woolf(0, 0, 3, 4)[0])
    assert E.benjamini_hochberg([np.nan]).shape == (1,)
    assert math.isnan(E.informacion_mutua_normalizada([0, 0], [1, 1]))
    assert np.isnan(E.puntaje_categorico([-1, -1], [0, 1], 2)).all()


# ============================================================================== regresión logística
@pytest.fixture(scope="module")
def datos_logit():
    rng = np.random.default_rng(21)
    n = 6000
    X = rng.normal(size=(n, 4))
    X[:, 3] = (X[:, 3] > 0.3).astype(float)
    eta = -1.2 + X @ np.array([0.9, -0.5, 0.0, 0.7])
    y = (rng.random(n) < _sigmoide(eta)).astype(float)
    return X, y


def _coef_sklearn(X, y, **kw):
    m = LogisticRegression(tol=1e-10, max_iter=1000, **kw).fit(X, y)
    return np.r_[m.intercept_, m.coef_[0]]


def test_logistica_coincide_con_scikit_learn_sin_penalizacion(datos_logit):
    X, y = datos_logit
    r = E.regresion_logistica(X, y)
    assert r.convergio and not r.separacion and r.n == len(y) and r.coef.shape == (5,)
    assert np.allclose(r.coef, _coef_sklearn(X, y, C=np.inf), rtol=1e-4, atol=1e-6)


def test_logistica_da_lo_mismo_con_matriz_densa_dispersa_y_con_bloques_chicos(datos_logit):
    X, y = datos_logit
    base = E.regresion_logistica(X, y)
    for variante in (E.regresion_logistica(sparse.csr_matrix(X), y), E.regresion_logistica(sparse.coo_matrix(X), y, bloque=1000),
                     E.regresion_logistica(X, y, bloque=7), E.regresion_logistica(sparse.csc_matrix(X), y, bloque=7)):
        assert np.allclose(variante.coef, base.coef, rtol=0, atol=1e-10) and np.allclose(variante.se, base.se, rtol=0, atol=1e-10)
        assert variante.devianza == pytest.approx(base.devianza, abs=1e-8)


def test_logistica_errores_estandar_de_wald_contra_la_inversa_de_x_w_x(datos_logit):
    X, y = datos_logit
    r = E.regresion_logistica(X, y)
    A = np.c_[np.ones(len(y)), X]
    mu = _sigmoide(A @ r.coef)
    inversa = np.linalg.inv(A.T @ ((mu * (1 - mu))[:, None] * A))
    assert np.allclose(r.se, np.sqrt(np.diag(inversa)), rtol=1e-8) and np.allclose(r.cov, inversa, rtol=1e-6, atol=1e-10)
    assert r.devianza == pytest.approx(-2 * np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)), rel=1e-9)


@pytest.mark.parametrize("ridge", [1e-6, 1.0, 5.0])
def test_logistica_con_ridge_sigue_la_convencion_de_scikit_learn(datos_logit, ridge):
    X, y = datos_logit
    r = E.regresion_logistica(X, y, ridge=ridge)
    assert r.convergio and np.allclose(r.coef, _coef_sklearn(X, y, C=1.0 / ridge), rtol=1e-4, atol=1e-6)
    sin = E.regresion_logistica(X, y)
    assert np.abs(r.coef[1:]).sum() <= np.abs(sin.coef[1:]).sum() + 1e-9                                  # el ridge encoge


def test_logistica_sin_intercepcion_y_solo_con_intercepcion(datos_logit):
    X, y = datos_logit
    r = E.regresion_logistica(X, y, intercepto=False)
    assert r.coef.shape == (4,) and np.allclose(r.coef, LogisticRegression(C=np.inf, fit_intercept=False, tol=1e-10, max_iter=1000).fit(X, y).coef_[0], atol=1e-6)
    solo = E.regresion_logistica(X[:, :0], y)
    assert solo.coef.shape == (1,) and solo.coef[0] == pytest.approx(math.log(y.mean() / (1 - y.mean())), abs=1e-6)
    assert solo.se[0] == pytest.approx(1 / math.sqrt(len(y) * y.mean() * (1 - y.mean())), rel=1e-6)


def test_separacion_por_coef_es_exactamente_la_regla_coeficiente_a_coeficiente(datos_logit):
    X, y = datos_logit
    for r in (E.regresion_logistica(X, y), E.regresion_logistica(np.c_[X, np.where(y == 0, 1.0, 0.0)], y, max_iter=60)):
        regla = (np.abs(r.coef) > E.SEPARACION_COEF_MAX) | (r.se > E.SEPARACION_SE_MAX)
        assert np.array_equal(r.separacion_por_coef, regla) and r.separacion == bool(regla.any())
    assert E.SEPARACION_COEF_MAX == 12.0 and E.SEPARACION_SE_MAX == 1e3


def test_una_columna_de_molestia_sin_eventos_se_marca_sola_y_no_contamina_a_las_demas():
    rng = np.random.default_rng(31)
    n = 20_000
    X = rng.normal(size=(n, 3))
    y = (rng.random(n) < _sigmoide(-1.5 + X @ np.array([0.8, -0.4, 0.3]))).astype(float)
    ceros = np.flatnonzero(y == 0)[:3000]
    molestia = np.zeros(n)
    molestia[ceros] = 1.0
    r = E.regresion_logistica(np.c_[X, molestia], y)
    assert r.separacion and r.convergio and r.separacion_por_coef[-1] and not r.separacion_por_coef[:-1].any()
    assert r.coef[-1] < -E.SEPARACION_COEF_MAX                                                            # medido: unos -17
    mantener = np.ones(n, dtype=bool)
    mantener[ceros] = False
    esperado = _coef_sklearn(X[mantener], y[mantener], C=np.inf)
    assert np.allclose(r.coef[:4], esperado, atol=1e-6)


def test_cuatro_filas_sin_eventos_en_una_base_grande_no_llegan_a_la_bandera():
    rng = np.random.default_rng(32)
    n = 115_310
    X = rng.normal(size=(n, 2))
    y = (rng.random(n) < _sigmoide(-2.5 + X @ np.array([0.7, 0.2]))).astype(float)
    ceros = np.flatnonzero(y == 0)[:4]
    molestia = np.zeros(n)
    molestia[ceros] = 1.0
    r = E.regresion_logistica(np.c_[X, molestia], y)
    assert not r.separacion_por_coef[-1] and not r.separacion and r.convergio
    assert -E.SEPARACION_COEF_MAX < r.coef[-1] < -5.0
    mantener = np.ones(n, dtype=bool)
    mantener[ceros] = False
    assert np.allclose(r.coef[:3], _coef_sklearn(X[mantener], y[mantener], C=np.inf), atol=1e-3)


def test_con_una_sola_iteracion_no_converge_aunque_no_haya_separacion(datos_logit):
    X, y = datos_logit
    r = E.regresion_logistica(X, y, max_iter=1)
    assert r.n_iter == 1 and not r.convergio and not r.separacion


def test_matriz_singular_se_resuelve_con_minimos_cuadrados_y_se_marca(datos_logit):
    X, y = datos_logit
    duplicada = np.c_[X, X[:, 0]]
    r = E.regresion_logistica(duplicada, y)
    assert r.convergio and r.separacion
    assert r.separacion_por_coef[1] and r.separacion_por_coef[5] and not r.separacion_por_coef[[0, 2, 3, 4]].any()
    assert np.isinf(r.se[1]) and np.isinf(r.se[5])
    sin = E.regresion_logistica(X, y)
    assert r.coef[1] + r.coef[5] == pytest.approx(sin.coef[1], abs=1e-6)                                  # el efecto total se reparte entre las copias
    assert r.devianza == pytest.approx(sin.devianza, abs=1e-6)
    assert np.allclose(r.coef[[0, 2, 3, 4]], sin.coef[[0, 2, 3, 4]], atol=1e-6)


@pytest.mark.parametrize("valor, signo", [(0.0, -1), (1.0, 1)])
def test_un_desenlace_constante_no_lanza_y_marca_separacion_en_la_intercepcion(valor, signo):
    rng = np.random.default_rng(5)
    r = E.regresion_logistica(rng.normal(size=(300, 2)), np.full(300, valor))
    assert r.convergio and r.separacion and r.separacion_por_coef.all()                                    # medido: intercepción de unos ±28
    assert signo * r.coef[0] > E.SEPARACION_COEF_MAX and np.isfinite(r.se).all()
    assert np.allclose(r.coef[1:], 0.0, atol=1e-6)


def test_una_columna_constante_junto_a_la_intercepcion_queda_con_error_infinito_y_marcada():
    rng = np.random.default_rng(5)
    X = np.c_[rng.normal(size=(300, 2)), np.full(300, 2.0)]
    y = (rng.random(300) < 0.3).astype(float)
    r = E.regresion_logistica(X, y)
    assert r.convergio and r.separacion
    assert np.isinf(r.se[[0, 3]]).all() and np.isfinite(r.se[[1, 2]]).all()                                # intercepción y columna son colineales
    assert r.separacion_por_coef[[0, 3]].all() and not r.separacion_por_coef[[1, 2]].any()
    assert np.isnan(r.cov[np.ix_([0, 3], [0, 1, 2, 3])]).all() and np.isfinite(r.cov[np.ix_([1, 2], [1, 2])]).all()
    sin = E.regresion_logistica(X[:, :2], y)
    assert r.devianza == pytest.approx(sin.devianza, abs=1e-6) and np.allclose(r.coef[1:3], sin.coef[1:], atol=1e-6)


def test_una_sola_fila_no_lanza_y_deja_todos_los_errores_estandar_infinitos():
    r = E.regresion_logistica(np.array([[0.3, -1.2]]), [1.0])
    assert r.n == 1 and r.convergio and r.separacion and np.isinf(r.se).all() and r.separacion_por_coef.all()


def test_logistica_es_determinista(datos_logit):
    X, y = datos_logit
    a, b = E.regresion_logistica(X, y), E.regresion_logistica(X, y)
    assert np.array_equal(a.coef, b.coef) and np.array_equal(a.se, b.se)


# ============================================================================================== VIF
def test_vif_a_mano_con_dos_variables_y_con_rango_completo_contra_la_inversa():
    vif, r2, exacta, rango = E.vif_desde_correlacion(np.array([[1.0, 0.9], [0.9, 1.0]]))
    assert np.allclose(vif, 1 / 0.19) and np.allclose(vif, 5.263158, atol=1e-6) and np.allclose(r2, 0.81) and not exacta.any() and rango == 2
    rng = np.random.default_rng(6)
    datos = rng.normal(size=(500, 5))
    datos[:, 1] += 0.7 * datos[:, 0]
    datos[:, 3] += 0.5 * datos[:, 2] - 0.3 * datos[:, 4]
    R = np.corrcoef(datos, rowvar=False)
    vif, _, exacta, rango = E.vif_desde_correlacion(R)
    assert np.allclose(vif, np.diag(np.linalg.inv(R)), rtol=1e-8) and rango == 5 and not exacta.any()


def test_vif_con_colinealidad_exacta_es_nulo_en_las_tres_variables_y_el_rango_baja():
    rng = np.random.default_rng(7)
    x1, x2 = rng.normal(size=400), rng.normal(size=400)
    R = np.corrcoef([x1, x2, x1 + x2])
    vif, r2, exacta, rango = E.vif_desde_correlacion(R)
    assert exacta.all() and np.isnan(vif).all() and np.all(r2 > 1 - 1e-9) and rango == 2
    # una cuarta variable independiente conserva su VIF finito
    R4 = np.corrcoef([x1, x2, x1 + x2, rng.normal(size=400)])
    vif4, _, exacta4, rango4 = E.vif_desde_correlacion(R4)
    assert exacta4.tolist() == [True, True, True, False] and np.isfinite(vif4[3]) and rango4 == 3


def test_vif_con_una_matriz_exactamente_singular_se_resuelve_con_minimos_cuadrados_sin_excepcion():
    """Dos variables idénticas hacen singular, de forma exacta, la matriz de las demás para la tercera: ``solve`` o ``inv`` lanzarían ``LinAlgError``."""
    R = np.array([[1.0, 1.0, 0.3], [1.0, 1.0, 0.3], [0.3, 0.3, 1.0]])
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.inv(R[np.ix_([0, 1], [0, 1])])                                                       # el caso es de verdad singular
    vif, r2, exacta, rango = E.vif_desde_correlacion(R)
    assert exacta.tolist() == [True, True, False] and rango == 2
    assert np.isnan(vif[:2]).all() and r2[:2] == pytest.approx([1.0, 1.0], abs=1e-9)
    assert r2[2] == pytest.approx(0.3**2, abs=1e-12) and vif[2] == pytest.approx(1 / (1 - 0.09), abs=1e-12)        # la proyección sobre dos copias es la de una sola


def test_vif_de_una_sola_variable_y_matriz_vacia():
    vif, r2, exacta, rango = E.vif_desde_correlacion(np.array([[1.0]]))
    assert vif.tolist() == [1.0] and r2.tolist() == [0.0] and not exacta.any() and rango == 1
    vif, _, _, rango = E.vif_desde_correlacion(np.zeros((0, 0)))
    assert vif.size == 0 and rango == 0


# ====================================================================================== agrupamiento
D4 = np.array([[0.0, 0.1, 0.8, 0.9], [0.1, 0.0, 0.7, 0.8], [0.8, 0.7, 0.0, 0.2], [0.9, 0.8, 0.2, 0.0]])


@pytest.mark.parametrize("metodo, altura", [("average", 0.8), ("single", 0.7), ("complete", 0.9)])
def test_alturas_de_enlace_del_ejemplo_a_mano(metodo, altura):
    orden, Z = E.orden_por_clusters(D4, metodo)
    assert Z.shape == (3, 4) and Z[-1, 2] == pytest.approx(altura, abs=1e-12)
    assert sorted(Z[:2, 2].tolist()) == pytest.approx([0.1, 0.2])
    assert sorted(orden.tolist()) == [0, 1, 2, 3]


def test_las_hojas_coinciden_con_el_dendrograma_de_scipy_y_los_pares_cercanos_quedan_juntos():
    rng = np.random.default_rng(9)
    A = rng.random((10, 10))
    D = 1 - (A + A.T) / 2
    np.fill_diagonal(D, 0)
    orden, Z = E.orden_por_clusters(D)
    assert orden.tolist() == hierarchy.dendrogram(Z, no_plot=True)["leaves"] and sorted(orden.tolist()) == list(range(10))
    orden4, _ = E.orden_por_clusters(D4)
    p = {int(v): i for i, v in enumerate(orden4)}
    assert abs(p[0] - p[1]) == 1 and abs(p[2] - p[3]) == 1


def test_nan_pasa_a_distancia_uno_se_simetriza_y_la_diagonal_queda_en_cero():
    con_nan = np.array([[0.0, np.nan, 0.4], [np.nan, 0.0, 0.3], [0.4, 0.3, 0.0]])
    _, Z = E.orden_por_clusters(con_nan, "single")
    assert Z[:, 2].tolist() == pytest.approx([0.3, 0.4])                                                    # el par sin medición no los une: pasó a distancia 1
    assert E.orden_por_clusters(np.array([[0.0, np.nan], [np.nan, 0.0]]), "single")[1][0, 2] == pytest.approx(1.0)
    medio_nan = np.array([[0.0, np.nan, 0.4], [0.2, 0.0, 0.3], [0.4, 0.3, 0.0]])                            # basta un NaN en una de las dos mitades
    assert E.orden_por_clusters(medio_nan, "single")[1][:, 2].tolist() == pytest.approx([0.3, 0.4])
    asimetrica = np.array([[5.0, 0.2, 0.9], [0.4, 5.0, 0.9], [0.9, 0.9, 5.0]])                              # diagonal ruidosa y mitades distintas
    _, Z2 = E.orden_por_clusters(asimetrica, "single")
    assert Z2[0, 2] == pytest.approx(0.3)                                                                   # (0.2 + 0.4) / 2
    recorte = E.orden_por_clusters(np.array([[0.0, 1.8], [0.4, 0.0]]), "single")[1]
    assert recorte[0, 2] == pytest.approx(1.0)                                                              # (1,8 + 0,4) / 2 = 1,1 se recorta a 1 después de promediar


def test_con_menos_de_dos_hojas_el_orden_es_trivial():
    for n in (0, 1):
        orden, Z = E.orden_por_clusters(np.zeros((n, n)))
        assert orden.tolist() == list(range(n)) and Z.shape == (0, 4)


def test_componentes_encadenan_y_el_nan_no_es_arista():
    A = np.array([[1.0, 0.9, 0.5], [0.9, 1.0, 0.85], [0.5, 0.85, 1.0]])
    assert len(set(E.componentes_redundantes(A, 0.8).tolist())) == 1                                         # A~B y B~C: un solo componente aunque A y C no estén asociadas
    B = np.array([[1.0, np.nan, 0.1], [np.nan, 1.0, 0.95], [0.1, 0.95, 1.0]])
    assert E.componentes_redundantes(B, 0.8).tolist() == [0, 1, 1]
    assert E.componentes_redundantes(np.eye(4), 0.8).tolist() == [0, 1, 2, 3]
    assert E.componentes_redundantes(np.full((3, 3), 0.8), 0.8).tolist() == [0, 0, 0]                        # la igualdad cuenta


def _grupos(A, umbral=0.8):
    """Grupos del dendrograma como los arma la Tarea 10: distancia con el valor publicado y corte con tolerancia."""
    D = 1.0 - E.redondear_publicado(A)
    _, Z = E.orden_por_clusters(D)
    return hierarchy.fcluster(Z, t=1.0 - umbral + 1e-12, criterion="distance"), Z


def test_componente_distinto_de_grupo_con_la_cadena_de_asociaciones_0_90_0_85_y_0_50():
    A = np.array([[1.0, 0.90, 0.50], [0.90, 1.0, 0.85], [0.50, 0.85, 1.0]])
    assert len(set(E.componentes_redundantes(A, 0.8).tolist())) == 1
    grupos, Z = _grupos(A)
    assert grupos[0] == grupos[1] != grupos[2]                                                              # {A, B} y {C}
    assert sorted(Z[:, 2].tolist()) == pytest.approx([0.10, 0.325])                                         # C se une a (0,5 + 0,15) / 2


def test_cada_grupo_del_dendrograma_esta_contenido_en_un_componente_en_200_matrices_aleatorias():
    rng = np.random.default_rng(10)
    for _ in range(200):
        m = int(rng.integers(3, 12))
        A = rng.random((m, m)) ** 0.35                                                                      # sesgada hacia valores altos para que haya aristas
        A = np.triu(A, 1)
        A = A + A.T + np.eye(m)
        comp = E.componentes_redundantes(A, 0.8)
        grupos, _ = _grupos(A)
        for g in set(grupos.tolist()):
            assert len(set(comp[grupos == g].tolist())) == 1


def test_un_par_en_0_7999999999999999_es_arista_y_queda_en_el_mismo_grupo():
    casi = 0.7999999999999999
    A = np.array([[1.0, casi, 0.1], [casi, 1.0, 0.1], [0.1, 0.1, 1.0]])
    assert E.componentes_redundantes(A, 0.8).tolist() == [0, 0, 1]
    grupos, _ = _grupos(A)
    assert grupos[0] == grupos[1] != grupos[2]
    # sin redondear, la distancia 0.19999999999999996 queda bajo 0.2 pero el corte exacto de 1 - 0.8 la dejaría fuera
    assert (1.0 - casi) <= (1.0 - 0.8) + 1e-12


# ====================================================================== puntaje, deciles y PSI
def test_puntaje_categorico_a_mano_y_nivel_sin_observaciones():
    codigos = np.r_[np.zeros(10, int), np.ones(10, int)]
    y = np.r_[[1, 1], np.zeros(8), np.zeros(10)]
    p = E.puntaje_categorico(codigos, y, 3)
    assert p[0] == pytest.approx(4 / 30) and p[1] == pytest.approx(2 / 30)                                  # p0 = 0,1 y m = 20
    assert p[2] == pytest.approx(0.1)                                                                       # nivel sin observaciones: la tasa global, sin tratamiento aparte
    assert E.SUAVIZADO_M == 20.0


def test_sin_suavizado_un_nivel_con_una_observacion_valdria_cero_o_uno():
    p = E.puntaje_categorico([0, 1, 1, 1], [1, 0, 0, 1], 2, m=0.0)
    assert p[0] == 1.0 and p[1] == pytest.approx(1 / 3)
    assert E.puntaje_categorico([0, 0], [1, 0], 2, m=0.0)[1] == pytest.approx(0.5)                          # sin observaciones y m = 0: sigue siendo la tasa global
    suave = E.puntaje_categorico([0, 1, 1, 1], [1, 0, 0, 1], 2)
    assert suave[0] < 1.0


def test_puntaje_categorico_descarta_codigos_negativos_y_y_nan():
    p = E.puntaje_categorico([0, 0, -1, 1, 1], [1, 0, 1, np.nan, 0], 2)
    # p0 = 1/3; nivel 0: n = 2, k = 1 -> (1 + 20/3) / 22; nivel 1: n = 1, k = 0 -> (20/3) / 21
    assert p[0] == pytest.approx((1 + 20 / 3) / 22) and p[1] == pytest.approx((20 / 3) / 21)


def test_bordes_de_deciles_unicos_con_empates_y_bin_de_nulos():
    b = E.bordes_deciles(np.arange(100.0))
    assert b.size == 9 and np.allclose(b, np.quantile(np.arange(100.0), np.arange(1, 10) / 10))
    # muchos empates: los cuantiles repetidos se funden en un borde (95 ceros: los nueve cuantiles valen 0)
    assert E.bordes_deciles([0.0] * 95 + [1.0] * 5).tolist() == [0.0]
    # 50 ceros, 30 unos y 20 doses: los cuantiles 0,1 a 0,4 valen 0, el 0,5 cae entre 0 y 1, el 0,6 y el 0,7 valen 1, el 0,8 cae entre 1 y 2 y el 0,9 vale 2
    assert E.bordes_deciles([0.0] * 50 + [1.0] * 30 + [2.0] * 20).tolist() == pytest.approx([0.0, 0.5, 1.0, 1.2, 2.0])
    assert np.all(np.diff(E.bordes_deciles(np.random.default_rng(1).normal(size=1000))) > 0)
    assert E.bordes_deciles([5.0] * 10).size == 0 and E.bordes_deciles([np.nan, np.nan]).size == 0
    assert E.bordes_deciles([1.0, 2.0, 3.0, 4.0], n_bins=2).tolist() == [2.5]


def test_asignar_bins_usa_intervalos_abiertos_por_la_izquierda_y_un_bin_propio_para_nulos():
    bordes = np.array([10.0, 50.0])
    c = E.asignar_bins([0.0, 10.0, 10.0001, 50.0, 50.1, np.nan], bordes)
    assert c.tolist() == [0, 0, 1, 1, 2, 3]                                                                 # v <= 10 al bin 0; 10 < v <= 50 al 1; el nulo a len(bordes) + 1
    assert E.asignar_bins([1.0, np.nan], np.array([])).tolist() == [0, 1]                                    # sin bordes: un solo bin y el de nulos


def test_psi_de_codigos_coincide_con_psi_y_el_mismo_reparto_da_cero():
    base = np.array([0, 0, 0, 1, 1, 2, -1])
    nuevo = np.array([0, 1, 1, 1, 2, 2, 2, -1, -1])
    assert E.psi_de_codigos(base, nuevo, 3) == pytest.approx(E.psi([3, 2, 1], [1, 3, 3]), abs=1e-15)
    assert E.psi_de_codigos(base, base, 3) == 0.0
    cb = E.asignar_bins(np.arange(200.0), E.bordes_deciles(np.arange(200.0)))
    assert E.psi_de_codigos(cb, cb, 11) == 0.0 and math.isnan(E.psi_de_codigos([-1], [-1], 2))
