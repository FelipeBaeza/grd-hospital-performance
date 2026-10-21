"""Tarea 10: asociación entre variables, redundancia y multicolinealidad (``eda/asociacion.py``).

Las cuentas a mano y los oráculos (scipy, scikit-learn, numpy) fijan cada medida; el panel artificial (``tests/panel_artificial.py``) aporta la verdad de diseño:
pares anidados, una cadena de tres variables, una constante y los agregados exactos de Elixhauser. Las pruebas numéricas exactas leen los resultados en memoria
(sin suprimir); las de formato y privacidad leen los CSV.
"""

from __future__ import annotations

import dataclasses
import shutil

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from scipy import stats
from scipy.cluster import hierarchy
from sklearn.metrics import matthews_corrcoef

from eda import asociacion as asoc
from eda import comorbilidad, config, descripcion, elixhauser, privacidad, rutas
from eda import estadistica as est
from eda import panel as pn
from eda.contexto import Contexto
from eda.informes import Informe
from panel_artificial import panel_artificial
from sintetico import crear_fuente_sintetica


@pytest.fixture(scope="module")
def u():
    """Umbrales del proyecto. Las cuentas a mano de este archivo suponen los valores propuestos; si cambian, la prueba lo dice."""
    u = config.cargar_umbrales()
    supuestos = {"redundancia_abs": 0.80, "vif_alerta": 5, "vif_critico": 10, "faltante_alerta": 0.20, "celda_minima": 10}
    for clave, valor in supuestos.items():
        assert u[clave] == valor, f"las cuentas a mano de este archivo suponen {clave} = {valor}"
    return u


@pytest.fixture(scope="module")
def contexto(tmp_path_factory, u):
    """Contexto sobre una fuente sintética mínima: ``calcular`` recibe el panel y nunca lee."""
    raiz = tmp_path_factory.mktemp("t10_asociacion")
    return Contexto.crear("extracto", fuente=crear_fuente_sintetica(raiz / "datos", n=60, semilla=29), dir_salida_base=raiz / "salidas")


@pytest.fixture(scope="module")
def panel():
    return panel_artificial(con_cadena=True)


@pytest.fixture(scope="module")
def resultado(contexto, panel):
    return asoc.calcular(contexto, panel)


def col(nombre, tipo, valores, n_niveles=None):
    return asoc.crear_columna(nombre, tipo, valores, n_niveles)


def _desde_tabla(tabla):
    """Códigos ``(a, b)`` cuya tabla de contingencia es ``tabla``."""
    t = np.asarray(tabla)
    a, b = [], []
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            a += [i] * int(t[i, j])
            b += [j] * int(t[i, j])
    return np.array(a), np.array(b)


def _con_variables(panel, nuevas):
    """Copia del panel con variables numéricas o binarias agregadas (``{nombre: (tipo, valores)}``; ``None`` es nulo)."""
    datos, variables = panel.datos, dict(panel.variables)
    for nombre, (tipo, valores) in nuevas.items():
        datos = datos.with_columns(pl.Series(nombre, list(valores), dtype=pl.Float64 if tipo == "num" else pl.Int8))
        variables[nombre] = pn.Variable(nombre=nombre, tipo=tipo, grupo="prueba", rol_fase1="FEATURE_BASAL", momento="basal", evaluable="si", uso="candidata", fuentes=())
    return dataclasses.replace(panel, datos=datos, variables=variables)


# ======================================================================================================== medida de cada tipo de par
def test_numerica_con_numerica_es_el_rho_de_spearman_con_signo_y_con_empates():
    rng = np.random.default_rng(5)
    x = rng.integers(0, 12, 300).astype(float)                                # con muchos empates
    y = -x + rng.integers(0, 8, 300)
    p = asoc.medida_par(col("x", "num", x), col("y", "num", y))
    rho = stats.spearmanr(x, y).statistic
    assert (p.medida, p.estado, p.n) == ("spearman", "ok", 300)
    assert p.valor == pytest.approx(abs(rho), abs=1e-12) and p.con_signo == pytest.approx(rho, abs=1e-12) and p.con_signo < 0


def test_spearman_a_mano_da_0_8():
    p = asoc.medida_par(col("x", "num", [1.0, 2.0, 3.0, 4.0]), col("y", "num", [1.0, 3.0, 2.0, 4.0]))
    assert p.valor == pytest.approx(0.8, abs=1e-12) and p.con_signo == pytest.approx(0.8, abs=1e-12) and p.n == 4


def test_binaria_con_binaria_es_phi_igual_al_mcc_y_0_8_a_mano():
    rng = np.random.default_rng(6)
    a = rng.integers(0, 2, 500)
    b = np.where(rng.random(500) < 0.7, a, 1 - a)
    p = asoc.medida_par(col("a", "bin", a), col("b", "bin", b))
    mcc = matthews_corrcoef(a, b)
    assert p.medida == "phi" and p.estado == "ok" and p.n == 500
    assert p.valor == pytest.approx(abs(mcc), abs=1e-12) and p.con_signo == pytest.approx(mcc, abs=1e-12)
    a8, b8 = _desde_tabla([[45, 5], [5, 45]])                                  # (ad - bc) / sqrt(...) = 2000 / 2500
    p8 = asoc.medida_par(col("a", "bin", a8), col("b", "bin", b8))
    assert p8.valor == pytest.approx(0.8, abs=1e-12) and p8.con_signo == pytest.approx(0.8, abs=1e-12)
    pn_ = asoc.medida_par(col("a", "bin", a8), col("b", "bin", 1 - b8))
    assert pn_.con_signo == pytest.approx(-0.8, abs=1e-12) and pn_.valor == pytest.approx(0.8, abs=1e-12)


def test_numerica_con_binaria_es_la_raiz_de_eta_cuadrado_con_el_signo_de_la_diferencia_de_medias():
    rng = np.random.default_rng(7)
    g = rng.integers(0, 2, 400)
    x = 2.0 * g + rng.normal(0, 1.5, 400)
    p = asoc.medida_par(col("x", "num", x), col("g", "bin", g))
    f = stats.f_oneway(x[g == 1], x[g == 0]).statistic                         # eta2 = F (k-1) / (F (k-1) + n - k) con k = 2
    assert p.medida == "eta" and p.estado == "ok" and p.n == 400
    assert p.valor == pytest.approx(np.sqrt(f / (f + 400 - 2)), abs=1e-10) and p.con_signo == pytest.approx(p.valor)
    invertida = asoc.medida_par(col("x", "num", -x), col("g", "bin", g))
    assert invertida.con_signo == pytest.approx(-p.valor) and invertida.valor == pytest.approx(p.valor)
    assert asoc.medida_par(col("g", "bin", g), col("x", "num", x)) == p                    # simétrica en el orden de los argumentos


def test_eta_a_mano_54_sobre_58():
    p = asoc.medida_par(col("x", "num", [1, 2, 3, 7, 8, 9]), col("g", "bin", [0, 0, 0, 1, 1, 1]))
    assert p.valor == pytest.approx(np.sqrt(54 / 58), abs=1e-12) and p.con_signo > 0 and p.n == 6


def test_numerica_con_categorica_es_la_raiz_de_eta_cuadrado_sin_signo():
    rng = np.random.default_rng(8)
    c = rng.integers(0, 4, 600)
    x = c * 1.0 + rng.normal(0, 2.0, 600)
    p = asoc.medida_par(col("x", "num", x), col("c", "cat", c, 4))
    f = stats.f_oneway(*[x[c == k] for k in range(4)]).statistic
    assert p.medida == "eta" and p.estado == "ok" and np.isnan(p.con_signo)
    assert p.valor == pytest.approx(np.sqrt(f * 3 / (f * 3 + 600 - 4)), abs=1e-10)


@pytest.mark.parametrize(
    "tabla, esperado",
    [([[30, 10], [10, 30]], 0.490290), ([[20, 10, 5], [5, 15, 10]], 0.381613), ([[8, 2, 1], [3, 9, 4], [2, 2, 10]], 0.459457), ([[10, 10], [10, 10]], 0.0), ([[50, 0], [0, 50]], 1.0)],
)
def test_categoricas_dan_la_v_de_bergsma_a_mano(tabla, esperado):
    a, b = _desde_tabla(tabla)
    p = asoc.medida_par(col("a", "cat", a, len(tabla)), col("b", "cat", b, len(tabla[0])))
    assert p.medida == "v_cramer_bergsma" and p.estado == "ok" and np.isnan(p.con_signo)
    assert p.valor == pytest.approx(esperado, abs=1e-6)


def test_binaria_con_categorica_usa_la_v_de_bergsma_y_descarta_el_nulo_de_la_binaria():
    a, b = _desde_tabla([[20, 10, 5], [5, 15, 10]])
    p = asoc.medida_par(col("a", "bin", a), col("b", "cat", b, 3))
    assert p.medida == "v_cramer_bergsma" and p.valor == pytest.approx(0.381613, abs=1e-6) and p.n == 65
    nulos = np.where(np.arange(a.size) % 5 == 0, -1, a)                          # la binaria con nulos dispersos: las filas nulas no entran a la tabla
    pn_ = asoc.medida_par(col("a", "bin", nulos), col("b", "cat", b, 3))
    ok = nulos >= 0
    esperado = est.v_cramer_sesgo(est.tabla_contingencia(nulos[ok], b[ok], 2, 3))
    assert pn_.n == int(ok.sum()) and pn_.valor == pytest.approx(esperado, abs=1e-12)


def test_la_medida_es_simetrica_en_valor_signo_n_y_estado_para_todos_los_tipos(panel):
    cols = asoc.preparar_columnas(panel, panel.anios_ranking)
    tomadas = {t: [c for c in cols.values() if c.tipo == t][:4] for t in ("num", "bin", "cat")}
    for ta in tomadas.values():
        for tb in tomadas.values():
            for a in ta:
                for b in tb:
                    if a is b:
                        continue
                    p, q = asoc.medida_par(a, b), asoc.medida_par(b, a)
                    assert (p.medida, p.n, p.estado) == (q.medida, q.n, q.estado)
                    assert (np.isnan(p.valor) and np.isnan(q.valor)) or p.valor == q.valor
                    assert (np.isnan(p.con_signo) and np.isnan(q.con_signo)) or p.con_signo == q.con_signo


def test_medida_de_tipos_cubre_las_seis_combinaciones_en_ambos_ordenes():
    esperado = {("num", "num"): "spearman", ("bin", "bin"): "phi", ("num", "bin"): "eta", ("num", "cat"): "eta", ("bin", "cat"): "v_cramer_bergsma", ("cat", "cat"): "v_cramer_bergsma"}
    for (ta, tb), medida in esperado.items():
        assert asoc.medida_de_tipos(ta, tb) == asoc.medida_de_tipos(tb, ta) == medida


# ========================================================================================================== faltantes y casos degenerados
def test_la_eliminacion_por_pares_usa_solo_las_filas_validas_y_no_rellena_con_cero():
    rng = np.random.default_rng(11)
    x = rng.normal(50, 10, 400)
    y = 0.5 * x + rng.normal(0, 8, 400)
    x[::7], y[::11] = np.nan, np.nan                                            # nulos dispersos y en filas distintas: los rangos se recalculan en el subconjunto
    ok = ~(np.isnan(x) | np.isnan(y))
    p = asoc.medida_par(col("x", "num", x), col("y", "num", y))
    esperado = stats.spearmanr(x[ok], y[ok]).statistic
    relleno = stats.spearmanr(np.nan_to_num(x), np.nan_to_num(y)).statistic
    assert p.n == int(ok.sum()) and p.valor == pytest.approx(esperado, abs=1e-12)
    assert abs(esperado - relleno) > 0.05                                       # la prueba distingue eliminar de rellenar con cero


def test_dos_numericas_sin_nulos_reutilizan_los_rangos_globales_y_coinciden_con_spearman():
    rng = np.random.default_rng(12)
    x, y = rng.normal(size=250), rng.normal(size=250)
    a, b = col("x", "num", x), col("y", "num", y)
    assert a.rangos is not None and b.rangos is not None
    assert col("z", "num", np.where(np.arange(250) == 3, np.nan, x)).rangos is None                          # con un nulo no hay rangos globales
    assert asoc.medida_par(a, b).con_signo == pytest.approx(est.spearman(x, y)[0], abs=1e-12)


def test_binaria_con_nulos_dispersos_coincide_con_el_mcc_de_las_filas_validas():
    rng = np.random.default_rng(13)
    a = rng.integers(0, 2, 600)
    b = np.where(rng.random(600) < 0.8, a, 1 - a)
    an, bn = a.copy(), b.copy()
    an[::6], bn[::9] = -1, -1
    ok = (an >= 0) & (bn >= 0)
    p = asoc.medida_par(col("a", "bin", an), col("b", "bin", bn))
    assert p.n == int(ok.sum()) and p.con_signo == pytest.approx(matthews_corrcoef(a[ok], b[ok]), abs=1e-12)


def test_una_variable_constante_da_valor_nulo_estado_constante_y_diagonal_uno():
    x = np.arange(50, dtype=float)
    cte = col("c", "num", np.full(50, 3.0))
    for otra in (col("x", "num", x), col("g", "bin", np.arange(50) % 2), col("k", "cat", np.arange(50) % 4, 4)):
        p = asoc.medida_par(cte, otra)
        assert np.isnan(p.valor) and np.isnan(p.con_signo) and p.estado == "constante" and p.n == 50
        assert asoc.medida_par(otra, cte).estado == "constante"
    d = asoc.medida_diagonal(cte)
    assert (d.medida, d.valor, d.estado, d.n) == ("identidad", 1.0, "constante", 50)
    assert asoc.medida_diagonal(col("x", "num", x)).estado == "ok" and asoc.medida_diagonal(col("k", "cat", np.arange(50) % 4, 4)).valor == 1.0
    assert asoc.medida_diagonal(col("b", "bin", np.zeros(50))).estado == "constante"                          # una binaria que solo vale 0 también
    assert asoc.medida_diagonal(col("k", "cat", np.zeros(50), 3)).estado == "constante"                         # una categórica con un solo nivel también


def test_menos_de_tres_pares_o_una_tabla_degenerada_dan_sin_datos_suficientes_no_constante():
    x = np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0])
    y = np.array([np.nan, np.nan, 3.0, 4.0, 5.0, 7.0])
    p = asoc.medida_par(col("x", "num", x), col("y", "num", y))                                                  # solo dos filas válidas en ambas
    assert p.n == 2 and np.isnan(p.valor) and p.estado == "sin datos suficientes"
    a = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    b = np.array([0, 1, 0, 1, -1, -1, -1, -1])                                                                   # en las filas donde b es válida, a vale siempre 0
    q = asoc.medida_par(col("a", "bin", a), col("b", "bin", b))
    assert q.n == 4 and np.isnan(q.valor) and q.estado == "sin datos suficientes"
    ca, cb = _desde_tabla([[1, 0], [0, 1]])
    assert asoc.medida_par(col("a", "cat", ca, 2), col("b", "cat", cb, 2)).estado == "sin datos suficientes"       # n = 2
    vacio = col("v", "bin", np.full(10, -1))
    assert asoc.medida_diagonal(vacio).estado == "sin datos suficientes" and asoc.medida_par(vacio, col("x", "num", np.arange(10.0))).n == 0


def test_largos_distintos_y_columnas_mal_formadas_son_errores():
    with pytest.raises(ValueError, match="largos distintos"):
        asoc.medida_par(col("x", "num", np.arange(5.0)), col("y", "num", np.arange(6.0)))
    with pytest.raises(ValueError, match="solo admite 0, 1 y -1"):
        asoc.crear_columna("b", "bin", [0, 1, 2])
    with pytest.raises(ValueError, match="tipo desconocido"):
        asoc.crear_columna("x", "otro", [1.0])
    with pytest.raises(ValueError, match="mayores o iguales"):
        asoc.crear_columna("c", "cat", [0, 1, 5], 3)


# ============================================================================================================ matriz completa
def test_la_matriz_vectorizada_coincide_con_medida_par_en_todos_los_pares(panel):
    cols = asoc.preparar_columnas(panel, panel.anios_ranking)
    med = asoc.matriz_de_medidas(cols)
    lista = list(cols.values())
    k = len(lista)
    for i in range(k):
        for j in range(i + 1, k):
            p = asoc.medida_par(lista[i], lista[j])
            assert med["medida"][i, j] == p.medida and med["estado"][i, j] == p.estado and med["n"][i, j] == p.n, (lista[i].nombre, lista[j].nombre)
            if np.isnan(p.valor):
                assert np.isnan(med["absoluto"][i, j])
            else:
                assert med["absoluto"][i, j] == pytest.approx(p.valor, abs=1e-12) and med["signo"][i, j] == pytest.approx(p.con_signo, abs=1e-12, nan_ok=True)


def test_la_matriz_es_simetrica_con_diagonal_uno_y_valores_en_0_1(resultado):
    A = resultado["A"]
    k = len(resultado["nombres"])
    assert A.shape == (k, k) and np.array_equal(A, A.T, equal_nan=True) and np.all(np.diag(A) == 1.0)
    finitos = A[np.isfinite(A)]
    assert finitos.min() >= 0.0 and finitos.max() <= 1.0
    m = resultado["matriz"]
    assert m.height == k * k and m.columns == asoc.COLUMNAS_MATRIZ
    assert m["variable_a"].to_list()[:k] == [resultado["nombres"][0]] * k and m["variable_b"].to_list()[:k] == resultado["nombres"]      # orden del registro en a y luego en b
    para = {(a, b): v for a, b, v in zip(m["variable_a"], m["variable_b"], m["asociacion_abs"])}
    for a, b in (("EDAD_ANIOS", "N_DX_SEC"), ("ELIX_01", "ELIX_02"), ("SEXO", "COMUNA")):
        assert para[(a, b)] == para[(b, a)]
    d = m.filter(pl.col("variable_a") == pl.col("variable_b"))
    assert d.height == k and (d["asociacion_abs"] == 1.0).all() and (d["medida"] == "identidad").all()
    assert m.filter(pl.col("variable_a") != pl.col("variable_b"))["medida"].is_in(["spearman", "phi", "eta", "v_cramer_bergsma"]).all()


def test_la_matriz_verifica_su_propia_simetria_diagonal_y_rango():
    A = np.array([[1.0, 0.5], [0.5, 1.0]])
    asoc._verificar_matriz(A, ["a", "b"])
    for mala, texto in ((np.array([[1.0, 0.5], [0.4, 1.0]]), "simétrica"), (np.array([[0.9, 0.5], [0.5, 1.0]]), "diagonal"), (np.array([[1.0, 1.2], [1.2, 1.0]]), r"\[0, 1\]")):
        with pytest.raises(ValueError, match=texto):
            asoc._verificar_matriz(mala, ["a", "b"])


def test_una_variable_constante_del_panel_queda_con_diagonal_uno_valores_nulos_y_estado_constante(resultado):
    m = resultado["matriz"]
    cte = m.filter(pl.col("variable_a") == "ELIX_28")                                  # constante en cero en el panel artificial
    assert cte.height == len(resultado["nombres"])
    d = cte.filter(pl.col("variable_b") == "ELIX_28").row(0, named=True)
    assert d["asociacion_abs"] == 1.0 and d["estado"] == "constante" and d["medida"] == "identidad"
    otros = cte.filter(pl.col("variable_b") != "ELIX_28")
    assert otros["asociacion_abs"].null_count() == otros.height and (otros["estado"] == "constante").all()


def test_la_referencia_se_analiza_aunque_su_evaluable_sea_no_pero_queda_fuera_de_clusters_pares_y_vif(panel, resultado):
    assert panel.variables["COD_HOSPITAL"].evaluable == "no" and "COD_HOSPITAL" in resultado["nombres"]
    m = resultado["matriz"]
    fila = m.filter((pl.col("variable_a") == "COD_HOSPITAL") & (pl.col("variable_b") == "ESPECIALIDAD_MEDICA")).row(0, named=True)
    assert fila["estado"] == "ok" and fila["asociacion_abs"] > 0.9 and fila["es_referencia_a"] == "si" and fila["es_referencia_b"] == "no"
    assert fila["posicion_a"] is None and fila["grupo_a"] is None and fila["posicion_b"] is not None and fila["grupo_b"] is not None
    assert set(m.filter(pl.col("posicion_a").is_null())["variable_a"]) == {"COD_HOSPITAL"} and set(m.filter(pl.col("grupo_b").is_null())["variable_b"]) == {"COD_HOSPITAL"}
    assert "COD_HOSPITAL" not in set(resultado["pares"]["variable_a"]) | set(resultado["pares"]["variable_b"])
    assert "COD_HOSPITAL" not in resultado["componentes"] and "COD_HOSPITAL" not in set(resultado["vif"]["variable"])
    assert "COD_HOSPITAL" not in set(resultado["dendrograma"]["variable"].drop_nulls())
    c = resultado["clusters"].filter(pl.col("variable") == "COD_HOSPITAL").row(0, named=True)
    assert c["es_referencia"] == "si" and c["cluster"] is None and c["grupo_dendrograma"] is None and c["indice_linkage"] is None and c["max_asociacion_otra"] > 0.9


def test_las_variables_analizadas_son_las_candidatas_evaluables_mas_la_referencia(panel, resultado):
    esperado = [n for n, v in panel.variables.items() if ((v.uso == "candidata" and v.evaluable != "no") or v.uso == "referencia") and n in panel.datos.columns]
    assert resultado["nombres"] == esperado == asoc.nombres_analizados(panel)
    assert len(esperado) == 62 and "IR_29301_MORTALIDAD" not in esperado and "ESTANCIA_DIAS" not in esperado and "HISTORIA_DISPONIBLE" not in esperado    # 58 + 3 de la cadena + la referencia
    r = resultado["resumen"].row(0, named=True)
    assert (r["n_variables"], r["n_numericas"], r["n_binarias"], r["n_categoricas"]) == (61, 8, 37, 16)


def test_preparar_columnas_usa_los_codigos_del_panel_en_la_ventana_pedida(panel):
    rank, conf = asoc.preparar_columnas(panel, panel.anios_ranking), asoc.preparar_columnas(panel, [2023])
    n_rank, n_conf = panel.ventana(panel.anios_ranking).height, panel.ventana([2023]).height
    assert all(c.valores.size == n_rank for c in rank.values()) and all(c.valores.size == n_conf for c in conf.values())
    assert conf["ES_EXTRANJERO"].n_validas == 4 and rank["ES_EXTRANJERO"].n_validas == n_rank                    # informada en 4 filas de 2023
    for nombre in ("SEXO", "COMUNA", "COD_HOSPITAL"):
        assert rank[nombre].tipo == "cat" and rank[nombre].n_niveles == conf[nombre].n_niveles == len(panel.niveles[nombre])
        assert np.array_equal(conf[nombre].valores, panel.codigos(nombre)[0][panel.datos["anio"].to_numpy() == 2023])           # la misma numeración
    assert rank["EDAD_ANIOS"].tipo == "num" and rank["EDAD_ANIOS"].rangos is not None and rank["ELIX_01"].tipo == "bin" and rank["ELIX_01"].n_niveles == 2


def test_calcular_no_lee_datos_cuando_recibe_el_panel(contexto, panel, monkeypatch):
    def prohibida(*args, **kwargs):
        raise AssertionError("asociacion no debe llamar a ctx.lector.leer")

    monkeypatch.setattr(contexto.lector, "leer", prohibida)
    r = asoc.calcular(contexto, panel)
    assert r["matriz"].height > 0


# ===================================================================================== redundancia, clusters y dendrograma
def test_los_pares_redundantes_y_los_componentes_son_los_de_la_verdad_de_diseno(resultado):
    pares = resultado["pares"]
    esperados = {
        ("ES_FIN_SEMANA", "DIA_SEMANA"), ("TIPO_PROCEDENCIA", "PROCEDENCIA_AGR"), ("ENTRANTE_TIPO_PROCEDENCIA", "PROCEDENCIA_AGR"), ("ENTRANTE_TIPO_PROCEDENCIA", "TIPO_PROCEDENCIA"),
        ("GRUPO_CLINICO", "CIE10_3C"), ("ESPECIALIDAD_MEDICA", "SERVICIO_SALUD"), ("ES_EXTRANJERO", "NACIONALIDAD"), ("PUEBLO_ORIGINARIO", "ETNIA"),
        ("PUNTAJE_VAN_WALRAVEN", "PUNTAJE_VAN_WALRAVEN_SIN_5"), ("CADENA_A", "CADENA_B"), ("CADENA_B", "CADENA_C"),
    }
    reales = {tuple(sorted(par, key=resultado["nombres"].index)) for par in zip(pares["variable_a"], pares["variable_b"])}
    assert reales == {tuple(sorted(par, key=resultado["nombres"].index)) for par in esperados}
    orden = {n: i for i, n in enumerate(resultado["nombres"])}
    assert all(orden[a] < orden[b] for a, b in zip(pares["variable_a"], pares["variable_b"]))                 # pares a < b en el orden del registro
    comp = resultado["componentes"]
    assert comp["TIPO_PROCEDENCIA"] == comp["PROCEDENCIA_AGR"] == comp["ENTRANTE_TIPO_PROCEDENCIA"]
    assert comp["ES_FIN_SEMANA"] == comp["DIA_SEMANA"] and comp["GRUPO_CLINICO"] == comp["CIE10_3C"] and comp["ESPECIALIDAD_MEDICA"] == comp["SERVICIO_SALUD"]
    assert comp["EDAD_ANIOS"] != comp["N_DX_SEC"] and comp["COMUNA"] != comp["ESPECIALIDAD_MEDICA"]
    for fila in pares.iter_rows(named=True):
        assert fila["cluster"] == comp[fila["variable_a"]] == comp[fila["variable_b"]] and fila["periodo"] == "2019-2022"
        assert fila["tamano_cluster"] == sum(1 for c in comp.values() if c == fila["cluster"])


def test_un_par_con_rho_de_menos_0_9_es_redundante_por_su_valor_absoluto(contexto, panel):
    n = panel.datos.height
    rng = np.random.default_rng(21)
    base = rng.normal(size=n)
    ruido = rng.normal(size=n) * 0.47                                         # rho de Spearman cercano a -0,9
    p2 = _con_variables(panel, {"NUM_X": ("num", base), "NUM_Y": ("num", -(base + ruido))})
    r = asoc.calcular(contexto, p2)
    fila = r["pares"].filter((pl.col("variable_a") == "NUM_X") & (pl.col("variable_b") == "NUM_Y")).row(0, named=True)
    assert fila["valor_con_signo"] < -0.8 and fila["asociacion_abs"] == pytest.approx(-fila["valor_con_signo"]) and fila["medida"] == "spearman"
    c = r["clusters"].filter(pl.col("variable").is_in(["NUM_X", "NUM_Y"]))
    assert c["cluster"].n_unique() == 1 and c["grupo_dendrograma"].n_unique() == 1 and c["altura_union"][0] < 0.2          # la distancia usa el valor absoluto, no el signo


def test_un_par_con_phi_exactamente_0_80_es_redundante_por_igualdad(contexto, panel):
    n, n_rank = panel.datos.height, panel.ventana(panel.anios_ranking).height
    rng = np.random.default_rng(22)
    a = np.concatenate([np.repeat([0, 1], n_rank // 2), rng.integers(0, 2, n - n_rank)])
    b = a.copy()
    cambiar = np.concatenate([np.arange(432, 480), np.arange(480, 528)])        # tabla (432, 48, 48, 432) en la ventana de ranking: phi = 184320 / 230400 = 0,8
    b[cambiar] = 1 - b[cambiar]
    p2 = _con_variables(panel, {"BIN_A": ("bin", a), "BIN_B": ("bin", b)})
    r = asoc.calcular(contexto, p2)
    fila = r["pares"].filter((pl.col("variable_a") == "BIN_A") & (pl.col("variable_b") == "BIN_B")).row(0, named=True)
    assert fila["asociacion_abs"] == pytest.approx(0.8, abs=1e-12) and fila["medida"] == "phi"
    assert est.redondear_publicado(fila["asociacion_abs"]) == 0.8
    c = r["clusters"].filter(pl.col("variable").is_in(["BIN_A", "BIN_B"]))                          # la misma igualdad en clusters_variables.csv: arista, cluster y grupo compartidos
    assert c["n_pares_redundantes"].to_list() == [1, 1] and c["cluster"].n_unique() == 1 and c["tamano_cluster"].to_list() == [2, 2] and c["grupo_dendrograma"].n_unique() == 1


def test_agrupar_variables_aplica_el_redondeo_publicado_en_el_umbral():
    def matriz(v):
        return np.array([[1.0, v, 0.1], [v, 1.0, 0.1], [0.1, 0.1, 1.0]])

    for v, mismo in ((0.8, True), (0.7999996, True), (0.7999999999999999, True), (0.7999994, False), (0.79, False)):
        e = asoc.agrupar_variables(matriz(v), 0.8)
        assert bool(e["cluster"][0] == e["cluster"][1]) is mismo, v
        assert bool(e["grupo"][0] == e["grupo"][1]) is mismo, v                # el corte usa la distancia redondeada: el par en 0,7999999999999999 queda en el mismo grupo
        assert e["cluster"][2] != e["cluster"][0]


def test_el_corte_del_dendrograma_no_depende_del_ruido_de_punto_flotante():
    e = asoc.agrupar_variables(np.array([[1.0, 0.7999999999999999], [0.7999999999999999, 1.0]]), 0.8)
    assert e["grupo"].tolist() == [1, 1] and e["cluster"].tolist() == [1, 1] and e["tamano_cluster"].tolist() == [2, 2]
    sin_redondear = 1.0 - np.array([[1.0, 0.7999999999999999], [0.7999999999999999, 1.0]])
    assert sin_redondear[0, 1] > 1.0 - 0.8                                        # sin el redondeo la distancia queda por encima del corte exacto y el par se separaría


def test_la_cadena_es_un_componente_pero_dos_grupos_del_dendrograma():
    A = np.array([[1.0, 0.90, 0.50], [0.90, 1.0, 0.85], [0.50, 0.85, 1.0]])
    e = asoc.agrupar_variables(A, 0.8)
    assert len(set(e["cluster"].tolist())) == 1 and len(set(e["grupo"].tolist())) == 2
    assert e["grupo"][0] == e["grupo"][1] != e["grupo"][2]
    assert sorted(e["Z"][:, 2].round(6).tolist()) == [0.1, 0.325]                  # uniones a 0,10 (A con B) y 0,325 (C con {A, B})
    assert e["altura_union"].round(6).tolist() == [0.1, 0.1, 0.325]


def test_la_cadena_del_panel_artificial_es_un_componente_y_dos_grupos(resultado):
    c = resultado["clusters"].filter(pl.col("variable").str.starts_with("CADENA_"))
    por = {r["variable"]: r for r in c.iter_rows(named=True)}
    assert por["CADENA_A"]["cluster"] == por["CADENA_B"]["cluster"] == por["CADENA_C"]["cluster"] and por["CADENA_A"]["tamano_cluster"] == 3
    assert por["CADENA_A"]["grupo_dendrograma"] == por["CADENA_B"]["grupo_dendrograma"] != por["CADENA_C"]["grupo_dendrograma"]
    assert por["CADENA_C"]["altura_union"] > 0.20 > por["CADENA_A"]["altura_union"]                   # el recíproco de la contención no vale


def test_cada_grupo_del_dendrograma_esta_contenido_en_un_componente_en_matrices_aleatorias():
    rng = np.random.default_rng(31)
    for _ in range(200):
        m = int(rng.integers(3, 14))
        A = rng.random((m, m)) ** 0.35                                              # muchas asociaciones altas para que haya aristas y grupos
        A = np.triu(A, 1)
        A = A + A.T + np.eye(m)
        e = asoc.agrupar_variables(A, 0.8)
        for g in set(e["grupo"].tolist()):
            assert len(set(e["cluster"][e["grupo"] == g].tolist())) == 1, (A, g)
        assert sorted(set(e["cluster"].tolist())) == list(range(1, int(e["cluster"].max()) + 1)) and sorted(set(e["grupo"].tolist())) == list(range(1, int(e["grupo"].max()) + 1))


def test_los_clusters_y_grupos_se_numeran_por_la_primera_hoja_del_dendrograma():
    rng = np.random.default_rng(32)
    A = rng.random((9, 9)) ** 0.3
    A = np.triu(A, 1)
    A = A + A.T + np.eye(9)
    e = asoc.agrupar_variables(A, 0.8)
    for etiqueta in ("cluster", "grupo"):
        vistos = []
        for hoja in e["orden"]:
            if e[etiqueta][hoja] not in vistos:
                vistos.append(int(e[etiqueta][hoja]))
        assert vistos == list(range(1, len(vistos) + 1)), etiqueta
    assert e["posicion"][e["orden"]].tolist() == list(range(1, 10))


def test_la_numeracion_sigue_el_orden_del_dendrograma_y_no_el_orden_del_registro():
    A = np.full((4, 4), 0.10)
    np.fill_diagonal(A, 1.0)
    A[0, 2] = A[2, 0] = 0.90                                                         # v0 con v2: distancia 0,10, se une segunda
    A[1, 3] = A[3, 1] = 0.95                                                         # v1 con v3: distancia 0,05, se une primera y queda arriba en el dendrograma
    e = asoc.agrupar_variables(A, 0.8)
    assert e["orden"].tolist() == [1, 3, 0, 2] and e["posicion"].tolist() == [3, 1, 4, 2]
    assert e["cluster"].tolist() == [2, 1, 2, 1] and e["grupo"].tolist() == [2, 1, 2, 1]               # el registro habría dado [1, 2, 1, 2]
    assert e["cluster"].tolist() != (est.componentes_redundantes(A, 0.8) + 1).tolist() and e["cluster"][e["orden"][0]] == 1


def test_una_variable_constante_no_mueve_el_dendrograma_de_las_demas(resultado):
    c = resultado["clusters"].filter(pl.col("variable") == "ELIX_28").row(0, named=True)
    assert c["tamano_cluster"] == 1 and c["n_pares_redundantes"] == 0 and c["max_asociacion_otra"] is None and c["variable_max_asociacion"] is None
    d = resultado["dendrograma"]
    hoja = d.filter(pl.col("variable") == "ELIX_28")["nodo"][0]
    union = d.filter((pl.col("nodo_a") == hoja) | (pl.col("nodo_b") == hoja)).row(0, named=True)
    assert union["altura"] == pytest.approx(1.0)                                  # distancia 1 con todas: se une al final y no altera las uniones bajo el corte
    sin = asoc.agrupar_variables(resultado["A"][np.ix_([i for i, n in enumerate(resultado["nombres"]) if n not in ("COD_HOSPITAL", "ELIX_28")],
                                                      [i for i, n in enumerate(resultado["nombres"]) if n not in ("COD_HOSPITAL", "ELIX_28")])], 0.8)
    assert sorted(sin["Z"][sin["Z"][:, 2] < 0.2][:, 2].round(9).tolist()) == sorted(resultado["Z"][resultado["Z"][:, 2] < 0.2][:, 2].round(9).tolist())


def test_clusters_variables_no_decide_representantes_y_trae_las_columnas_del_diseno(resultado, panel):
    c = resultado["clusters"]
    assert c.columns == asoc.COLUMNAS_CLUSTERS and not any("represent" in n for n in c.columns)
    assert c["variable"].to_list() == resultado["nombres"] and c.height == len(resultado["nombres"])
    resto = c.filter(pl.col("es_referencia") == "no")
    assert resto["indice_linkage"].to_list() == list(range(resto.height)) and sorted(resto["posicion_dendrograma"].to_list()) == list(range(1, resto.height + 1))
    pares = resultado["pares"]
    for fila in resto.iter_rows(named=True):
        n_aristas = pares.filter((pl.col("variable_a") == fila["variable"]) | (pl.col("variable_b") == fila["variable"])).height
        assert fila["n_pares_redundantes"] == n_aristas, fila["variable"]
        assert fila["tipo"] == panel.variables[fila["variable"]].tipo
    m = resultado["matriz"]
    for fila in resto.iter_rows(named=True):
        otras = m.filter((pl.col("variable_a") == fila["variable"]) & (pl.col("variable_b") != fila["variable"]) & (pl.col("es_referencia_b") == "no"))
        mejor = otras.sort("asociacion_abs", descending=True, nulls_last=True, maintain_order=True).row(0, named=True)
        if mejor["asociacion_abs"] is None:
            assert fila["max_asociacion_otra"] is None
        else:
            assert fila["max_asociacion_otra"] == pytest.approx(mejor["asociacion_abs"]) and fila["variable_max_asociacion"] == mejor["variable_b"]


def test_t23_dendrograma_reconstruye_z_y_el_orden_de_las_hojas(resultado):
    d = resultado["dendrograma"]
    assert d.columns == asoc.COLUMNAS_DENDROGRAMA
    hojas, uniones = d.filter(pl.col("tipo_fila") == "hoja"), d.filter(pl.col("tipo_fila") == "union")
    m = hojas.height
    assert d.height == 2 * m - 1 and hojas["nodo"].to_list() == list(range(m)) and uniones["nodo"].to_list() == list(range(m, 2 * m - 1))
    Z = np.column_stack([uniones["nodo_a"].to_numpy(), uniones["nodo_b"].to_numpy(), uniones["altura"].to_numpy(), uniones["n_hojas"].to_numpy()]).astype(float)
    assert np.array_equal(Z[:, [0, 1, 3]], resultado["Z"][:, [0, 1, 3]]) and np.allclose(Z[:, 2], resultado["Z"][:, 2], atol=0, rtol=0)
    assert (np.diff(Z[:, 2]) >= 0).all()                                            # el enlace promedio es monótono
    orden = hierarchy.leaves_list(Z).tolist()
    assert [hojas["variable"][i] for i in orden] == hojas.sort("posicion_dendrograma")["variable"].to_list()
    assert hierarchy.dendrogram(Z, no_plot=True)["leaves"] == orden
    assert sorted(hojas["posicion_dendrograma"].to_list()) == list(range(1, m + 1))
    arboles = d.filter(pl.col("tipo_fila") == "hoja")["variable"].to_list()
    assert arboles == [n for n in resultado["nombres"] if n != "COD_HOSPITAL"]            # la entrada del enlace es el registro sin la referencia


# ======================================================================================================================== VIF
def _elix_no_constantes():
    return {f"ELIX_{i:02d}" for i in range(1, 32) if i != 28}                             # ELIX_28 es constante en cero en el panel artificial


def test_el_vif_base_no_tiene_nulos_y_con_agregados_marca_la_colinealidad_exacta(resultado):
    v = resultado["vif"]
    assert v.columns == asoc.COLUMNAS_VIF
    base, con = v.filter(pl.col("conjunto") == "base"), v.filter(pl.col("conjunto") == "con_agregados")
    ok_base = base.filter(pl.col("estado") == "ok")
    assert ok_base["vif"].null_count() == 0 and (ok_base["colinealidad_exacta"] == "no").all() and (ok_base["clasificacion"] != "exacta").all()
    assert not set(asoc.AGREGADOS) & set(base["variable"]) and set(asoc.AGREGADOS) <= set(con["variable"])
    exactas = set(con.filter(pl.col("colinealidad_exacta") == "si")["variable"])
    assert exactas == _elix_no_constantes() | {"N_COMORB_ELIX", "PUNTAJE_VAN_WALRAVEN", "PUNTAJE_VAN_WALRAVEN_SIN_5"}          # las variables que participan en la relación quedan marcadas
    e = con.filter(pl.col("colinealidad_exacta") == "si")
    assert e["vif"].null_count() == e.height and (e["clasificacion"] == "exacta").all() and (e["r2"] > 1 - 1e-9).all()
    assert (con.filter(pl.col("variable") == "N_DX_SEC")["colinealidad_exacta"] == "no").all()                              # el cuarto agregado no es exacto
    assert con["n_columnas_conjunto"][0] - con["rango_conjunto"][0] == 3                                                     # tres relaciones exactas independientes
    assert base["n_columnas_conjunto"][0] == base["rango_conjunto"][0] == 40
    assert base["n_filas"].n_unique() == con["n_filas"].n_unique() == 1 and base["n_filas"][0] == con["n_filas"][0] == 960      # mismas filas en los dos conjuntos


def test_el_vif_coincide_con_la_inversa_de_la_matriz_de_correlacion(resultado, panel):
    cols = asoc.preparar_columnas(panel, panel.anios_ranking)
    base = resultado["vif"].filter((pl.col("conjunto") == "base") & (pl.col("estado") == "ok"))
    nombres = base["variable"].to_list()
    X = np.column_stack([cols[n].valores.astype(float) for n in nombres])
    esperado = np.diag(np.linalg.inv(np.corrcoef(X, rowvar=False)))
    assert np.allclose(base["vif"].to_numpy(), esperado, rtol=1e-6)
    assert np.allclose(base["r2"].to_numpy(), 1 - 1 / esperado, atol=1e-6)


def test_las_identidades_de_los_agregados_tienen_diferencia_cero_con_los_datos(resultado, panel):
    r = resultado["resumen"].row(0, named=True)
    assert r["identidad_n_comorb_max_abs_dif"] == 0.0 and r["identidad_puntaje_max_abs_dif"] == 0.0 and r["identidad_puntaje_sin5_max_abs_dif"] == 0.0
    assert asoc.identidades(panel, panel.anios_ranking) == {"identidad_n_comorb": 0.0, "identidad_puntaje": 0.0, "identidad_puntaje_sin5": 0.0}
    roto = panel.datos.with_columns((pl.col("N_COMORB_ELIX") + pl.when(pl.col("anio") == 2020).then(2.0).otherwise(0.0)).alias("N_COMORB_ELIX"))
    otro = dataclasses.replace(panel, datos=roto)
    assert asoc.identidades(otro, panel.anios_ranking)["identidad_n_comorb"] == 2.0                               # si la identidad dejara de valer, la diferencia lo muestra
    assert asoc.identidades(otro, [2023])["identidad_n_comorb"] == 0.0
    sin_elix = dataclasses.replace(panel, datos=panel.datos.drop("ELIX_07"))
    assert asoc.identidades(sin_elix, panel.anios_ranking) == {"identidad_n_comorb": None, "identidad_puntaje": None, "identidad_puntaje_sin5": None}


def test_una_variable_con_19_por_ciento_de_nulos_se_queda_y_una_con_20_exacto_sale(u):
    rng = np.random.default_rng(41)
    n = 1000

    def con_nulos(k):
        v = rng.normal(size=n)
        v[:k] = np.nan
        return v

    cols = {"V19": col("V19", "num", con_nulos(190)), "V20": col("V20", "num", con_nulos(200)), "V0": col("V0", "num", rng.normal(size=n)), "V1": col("V1", "num", rng.normal(size=n))}
    v = asoc.calcular_vif(cols, [], u).filter(pl.col("conjunto") == "con_agregados")
    por = {r["variable"]: r for r in v.iter_rows(named=True)}
    assert por["V19"]["estado"] == "ok" and por["V19"]["vif"] is not None and por["V19"]["n_filas"] == 810                       # 1000 menos los 190 nulos de V19
    assert por["V20"]["estado"] == "excluida" and por["V20"]["motivo"] == asoc.MOTIVO_FALTANTE and por["V20"]["vif"] is None and por["V20"]["clasificacion"] == "excluida"
    assert por["V0"]["motivo"] == "" and por["V0"]["n_columnas_conjunto"] == 3


def test_el_vif_excluye_las_constantes_con_su_fila_y_su_motivo_y_la_referencia_y_las_categoricas_no_entran(resultado):
    v = resultado["vif"]
    ex = v.filter(pl.col("estado") == "excluida")
    assert set(zip(ex["conjunto"], ex["variable"], ex["motivo"])) == {("base", "ELIX_28", "constante en la ventana"), ("con_agregados", "ELIX_28", "constante en la ventana")}
    assert set(v["tipo"]) == {"num", "bin"} and "COD_HOSPITAL" not in set(v["variable"])
    assert resultado["resumen"]["n_vif_excluidas_faltante"][0] == 0                                                              # la constante no cuenta como faltante


def test_una_variable_que_queda_constante_en_los_casos_completos_se_excluye_sin_romper_el_calculo(u):
    rng = np.random.default_rng(42)
    n = 100
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    x2[:10] = np.nan
    b = np.zeros(n, dtype=int)
    b[:5] = 1                                                                      # vale 1 solo en filas donde x2 es nulo: en los casos completos es constante
    cols = {"X1": col("X1", "num", x1), "X2": col("X2", "num", x2), "B": col("B", "bin", b)}
    v = asoc.calcular_vif(cols, [], u).filter(pl.col("conjunto") == "base")
    por = {r["variable"]: r for r in v.iter_rows(named=True)}
    assert por["B"]["estado"] == "excluida" and por["B"]["motivo"] == asoc.MOTIVO_CONSTANTE_COMPLETOS
    assert por["X1"]["estado"] == por["X2"]["estado"] == "ok" and por["X1"]["n_columnas_conjunto"] == 2 and por["X1"]["n_filas"] == 90


def test_con_menos_casos_completos_que_columnas_mas_dos_el_vif_dice_sin_datos_suficientes(u):
    rng = np.random.default_rng(43)
    n = 10
    cols = {}
    for j in range(5):
        v = rng.normal(size=n)
        v[j] = np.nan                                                              # un nulo por columna (10 % < 20 %) en filas distintas: quedan 5 casos completos
        cols[f"V{j}"] = col(f"V{j}", "num", v)
    v = asoc.calcular_vif(cols, [], u)
    assert (v["n_filas"] == 5).all() and (v["n_columnas_conjunto"] == 5).all()      # 5 < 5 + 2
    assert (v["estado"] == "sin datos suficientes").all() and (v["clasificacion"] == "no evaluable").all() and v["vif"].null_count() == v.height and v["rango_conjunto"].null_count() == v.height
    mas = {k: col(k, "num", rng.normal(size=12)) for k in ("A", "B", "C", "D", "E")}                            # 12 casos completos >= 5 + 2: sí se calcula
    assert (asoc.calcular_vif(mas, [], u)["estado"] == "ok").all()
    exacto = {k: col(k, "num", rng.normal(size=7)) for k in ("A", "B", "C", "D", "E")}                          # 7 = 5 + 2: el borde con igualdad
    assert (asoc.calcular_vif(exacto, [], u)["estado"] == "ok").all()


def test_la_clasificacion_del_vif_compara_el_valor_redondeado_con_igualdad_en_5_y_10(u, monkeypatch):
    vifs = np.array([4.999994, 5.0, 4.9999996, 9.999994, 10.0, 9.9999996])
    monkeypatch.setattr(asoc.est, "vif_desde_correlacion", lambda R: (vifs, 1 - 1 / vifs, np.zeros(vifs.size, dtype=bool), int(vifs.size)))
    rng = np.random.default_rng(44)
    cols = {f"V{j}": col(f"V{j}", "num", rng.normal(size=60)) for j in range(6)}
    v = asoc.calcular_vif(cols, [], u).filter(pl.col("conjunto") == "base")
    assert v["clasificacion"].to_list() == ["bajo", "alerta", "alerta", "alerta", "critico", "critico"]
    assert v["colinealidad_exacta"].to_list() == ["no"] * 6 and v["rango_conjunto"][0] == 6


def test_el_vif_de_dos_variables_con_correlacion_09_a_mano_es_5_263158(u):
    rng = np.random.default_rng(45)
    z = rng.normal(size=20000)
    x = z
    y = 0.9 * z + np.sqrt(1 - 0.81) * rng.normal(size=20000)
    v = asoc.calcular_vif({"X": col("X", "num", x), "Y": col("Y", "num", y)}, [], u).filter(pl.col("conjunto") == "base")
    r = np.corrcoef(x, y)[0, 1]
    assert v["vif"].to_list() == pytest.approx([1 / (1 - r * r)] * 2) and v["vif"][0] == pytest.approx(5.263158, rel=0.05)


def test_el_vif_sin_variables_numericas_ni_binarias_es_una_tabla_vacia_con_las_columnas(u):
    v = asoc.calcular_vif({"C": col("C", "cat", [0, 1, 2, 0, 1, 2], 3)}, [], u)
    assert v.height == 0 and v.columns == asoc.COLUMNAS_VIF


# ================================================================================================================ estabilidad de 2023
def test_la_estabilidad_reestima_cada_par_redundante_con_la_misma_medida_en_2023(resultado, panel):
    pares, d23 = resultado["pares"], panel.ventana([2023])
    f = pares.filter((pl.col("variable_a") == "CADENA_A") & (pl.col("variable_b") == "CADENA_B")).row(0, named=True)
    assert f["asociacion_2023"] == pytest.approx(abs(stats.spearmanr(d23["CADENA_A"], d23["CADENA_B"]).statistic), abs=1e-12)
    assert f["n_pares_2023"] == d23.height == 240 and f["se_mantiene_2023"] == "si"
    cod = {n: panel.codigos(n)[0][panel.datos["anio"].to_numpy() == 2023] for n in ("GRUPO_CLINICO", "CIE10_3C")}
    g = pares.filter((pl.col("variable_a") == "GRUPO_CLINICO") & (pl.col("variable_b") == "CIE10_3C")).row(0, named=True)
    esperado = est.v_cramer_sesgo(est.tabla_contingencia(cod["GRUPO_CLINICO"], cod["CIE10_3C"], len(panel.niveles["GRUPO_CLINICO"]), len(panel.niveles["CIE10_3C"])))
    assert g["asociacion_2023"] == pytest.approx(esperado, abs=1e-12) and g["n_pares_2023"] == 240
    for fila in pares.iter_rows(named=True):                                                            # la marca sigue al valor redondeado de 2023
        if fila["asociacion_2023"] is None:
            assert fila["se_mantiene_2023"] == "no evaluable"
        else:
            assert fila["se_mantiene_2023"] == ("si" if round(fila["asociacion_2023"], 6) >= 0.8 else "no")
    sin = pares.filter((pl.col("variable_a") == "ES_EXTRANJERO") & (pl.col("variable_b") == "NACIONALIDAD")).row(0, named=True)
    assert sin["n_pares_2023"] == 4 and sin["asociacion_2023"] is None and sin["se_mantiene_2023"] == "no evaluable"      # 4 filas informadas en 2023, todas de la misma clase


def test_un_par_redundante_que_deja_de_serlo_en_2023_queda_marcado_no(contexto, panel):
    n = panel.datos.height
    rng = np.random.default_rng(23)
    anio = panel.datos["anio"].to_numpy()
    base = rng.normal(size=n)
    q = np.where(anio == 2023, rng.normal(size=n), base + 0.3 * rng.normal(size=n))                      # asociado en la ventana de ranking e independiente en 2023
    r = asoc.calcular(contexto, _con_variables(panel, {"NUM_P": ("num", base), "NUM_Q": ("num", q)}))
    f = r["pares"].filter((pl.col("variable_a") == "NUM_P") & (pl.col("variable_b") == "NUM_Q")).row(0, named=True)
    assert f["asociacion_abs"] > 0.8 and f["asociacion_2023"] < 0.5 and f["se_mantiene_2023"] == "no"
    res = r["resumen"].row(0, named=True)
    assert res["n_pares_redundantes_evaluados_2023"] == res["n_pares_redundantes_estables_2023"] + 1


def test_sin_el_anio_de_confirmacion_todo_lo_de_2023_queda_nulo_sin_romper(contexto, panel):
    solo_ranking = dataclasses.replace(panel, datos=panel.datos.filter(pl.col("anio") <= 2022), anio_confirmacion=None)
    r = asoc.calcular(contexto, solo_ranking)
    p = r["pares"]
    assert p.height > 0 and p["asociacion_2023"].null_count() == p.height and p["n_pares_2023"].null_count() == p.height and (p["se_mantiene_2023"] == "no evaluable").all()
    res = r["resumen"].row(0, named=True)
    assert res["n_filas_2023"] == 0 and res["n_pares_redundantes_evaluados_2023"] == 0 and res["n_pares_redundantes_estables_2023"] == 0


def _con_par_40_4(panel):
    """Copia del panel en la que ES_EXTRANJERO está informada en 40 filas de 2019-2022 (20 chilenas y 20 no) y en las 4 de 2023: el par con NACIONALIDAD tiene (40, 4)."""
    nac = panel.datos["NACIONALIDAD"].to_numpy()
    anio = panel.datos["anio"].to_numpy()
    rank = np.flatnonzero(anio <= 2022)
    elegidas = set([i for i in rank if nac[i] == "CHILENA"][:20] + [i for i in rank if nac[i] != "CHILENA"][:20])
    original = panel.datos["ES_EXTRANJERO"].to_list()
    nuevo = [original[i] if (anio[i] == 2023 or i in elegidas) else None for i in range(panel.datos.height)]
    return dataclasses.replace(panel, datos=panel.datos.with_columns(pl.Series("ES_EXTRANJERO", nuevo, dtype=pl.Int8)))


def test_un_par_con_40_filas_en_2019_2022_y_4_en_2023_conserva_su_asociacion_y_anula_solo_el_bloque_de_2023(contexto, panel):
    p2 = _con_par_40_4(panel)
    r = asoc.calcular(contexto, p2)
    f = r["pares"].filter((pl.col("variable_a") == "ES_EXTRANJERO") & (pl.col("variable_b") == "NACIONALIDAD")).row(0, named=True)
    assert f["n_pares"] == 40 and f["n_pares_2023"] == 4 and f["asociacion_abs"] >= 0.8
    pub, celdas = asoc.supresion.suprimir_bloques(r["pares"], asoc.BLOQUES_PARES, minimo=10)
    g = pub.filter((pl.col("variable_a") == "ES_EXTRANJERO") & (pl.col("variable_b") == "NACIONALIDAD")).row(0, named=True)
    assert g["asociacion_abs"] == f["asociacion_abs"] and g["n_pares"] == 40 and g["valor_con_signo"] == f["valor_con_signo"]
    assert g["n_pares_2023"] is None and g["asociacion_2023"] is None and g["se_mantiene_2023"] is None and celdas == 1
    res = r["resumen"].row(0, named=True)
    assert res["n_pares_redundantes_evaluados_2023"] == r["pares"].filter(pl.col("se_mantiene_2023").is_in(["si", "no"])).height          # el resumen sale de la versión publicable


# =============================================================================================================== resumen y asociación con el desenlace
def test_el_resumen_es_coherente_con_las_tablas(resultado):
    r = resultado["resumen"].row(0, named=True)
    assert resultado["resumen"].columns == asoc.COLUMNAS_RESUMEN and resultado["resumen"].height == 1
    assert (r["periodo"], r["n_pares"], r["n_pares_no_evaluables"], r["n_pares_redundantes"]) == ("2019-2022", 61 * 60 // 2, 60, resultado["pares"].height) and r["n_pares_redundantes"] == 11
    assert (r["n_clusters_multiples"], r["tamano_cluster_max"], r["n_variables_en_clusters_multiples"]) == (8, 3, 18)
    m = resultado["matriz"].filter((pl.col("variable_a") != pl.col("variable_b")) & (pl.col("es_referencia_a") == "no") & (pl.col("es_referencia_b") == "no"))
    assert r["asociacion_max_valor"] == pytest.approx(m["asociacion_abs"].max()) and {r["asociacion_max_a"], r["asociacion_max_b"]} == {"ENTRANTE_TIPO_PROCEDENCIA", "PROCEDENCIA_AGR"}
    assert (r["n_filas_ranking"], r["n_filas_2023"], r["n_filas_vif"]) == (960, 240, 960)
    assert (r["n_pares_redundantes_evaluados_2023"], r["n_pares_redundantes_estables_2023"]) == (10, 10)
    assert (r["n_exactas_base"], r["n_exactas_con_agregados"], r["n_columnas_base"], r["rango_base"], r["n_columnas_con_agregados"], r["rango_con_agregados"]) == (0, 33, 40, 40, 44, 41)
    assert r["n_relaciones_exactas_con_agregados"] == 3 and r["vif_max_base_variable"] in {"CADENA_A", "CADENA_B", "CADENA_C"} and r["n_vif_critico_base"] == 3
    assert r["vif_max_con_agregados"] >= r["vif_max_base"] > 10
    assert (r["phi_elix_disponible"], r["figuras_omitidas"]) == ("no", "F10_04")                  # el contexto de prueba no tiene t23_elix_phi.csv


def test_el_resumen_no_publica_lo_que_la_tabla_fuente_anula_por_celda_pequena(contexto, panel):
    rng = np.random.default_rng(61)
    n, anio = panel.datos.height, panel.datos["anio"].to_numpy()
    y1 = rng.normal(size=n)
    extra = {"SOLO_A": ("num", [float(v) if (anio[i] <= 2022 and i < 6) else None for i, v in enumerate(y1)]),
             "SOLO_B": ("num", [float(v) * 2 if (anio[i] <= 2022 and i < 6) else None for i, v in enumerate(y1)])}
    r = asoc.calcular(contexto, _con_variables(panel, extra))
    f = r["matriz"].filter((pl.col("variable_a") == "SOLO_A") & (pl.col("variable_b") == "SOLO_B")).row(0, named=True)
    assert f["n_pares"] == 6 and f["asociacion_abs"] == pytest.approx(1.0)                              # el valor real existe (rho = 1)...
    res = r["resumen"].row(0, named=True)
    assert (res["asociacion_max_a"], res["asociacion_max_b"]) != ("SOLO_A", "SOLO_B")                    # ...pero el resumen no lo publica como máximo: la matriz lo anula (n entre 1 y 9)
    assert res["asociacion_max_valor"] < 1.0


def test_con_resultado_usa_eta_para_numericas_y_la_v_corregida_para_binarias_y_categoricas():
    rng = np.random.default_rng(51)
    y = rng.integers(0, 2, 500)
    x = 1.2 * y + rng.normal(size=500)
    v, medida, n = asoc.con_resultado(col("x", "num", x), y)
    f = stats.f_oneway(x[y == 1], x[y == 0]).statistic
    assert medida == "eta" and n == 500 and v == pytest.approx(np.sqrt(f / (f + 498)), abs=1e-10)
    a, b = _desde_tabla([[30, 10], [10, 30]])
    v, medida, n = asoc.con_resultado(col("c", "cat", a, 2), b)
    assert medida == "v_cramer_bergsma" and n == 80 and v == pytest.approx(0.490290, abs=1e-6)
    a, b = _desde_tabla([[20, 10, 5], [5, 15, 10]])
    v, medida, n = asoc.con_resultado(col("c", "cat", b, 3), a)                                           # la categórica de tres niveles contra el desenlace
    assert medida == "v_cramer_bergsma" and v == pytest.approx(0.381613, abs=1e-6)
    nulos = np.where(np.arange(y.size) % 4 == 0, -1, rng.integers(0, 2, y.size))
    v, _, n = asoc.con_resultado(col("b", "bin", nulos), y)
    assert n == int((nulos >= 0).sum()) and np.isfinite(v)                                                # el nulo de la binaria no entra a la tabla
    with pytest.raises(ValueError, match="solo admite 0 y 1"):
        asoc.con_resultado(col("x", "num", x), np.where(y == 1, 2, 0))
    with pytest.raises(ValueError, match="largos distintos"):
        asoc.con_resultado(col("x", "num", x), y[:-1])
    assert np.isnan(asoc.con_resultado(col("k", "num", np.full(500, 1.0)), y)[0])                         # constante: no hay asociación que medir


# =================================================================================================== escritura: tablas, métricas, figuras e informe
FUERTES_PHI = {frozenset(("ELIX_11", "ELIX_12")): 0.90, frozenset(("ELIX_19", "ELIX_20")): 0.85, frozenset(("ELIX_03", "ELIX_04")): -0.82}
TEXTOS_VIF = ["conjunto", "variable", "tipo", "es_agregado", "colinealidad_exacta", "clasificacion", "motivo", "estado"]


def _tabla_phi(fuertes=FUERTES_PHI) -> pl.DataFrame:
    """``t23_elix_phi.csv`` armado a mano (sin datos): phi casi cero salvo los pares pedidos; el par ELIX_01-ELIX_02 solo es fuerte en 2023."""
    e = elixhauser.cargar()
    filas = []
    for periodo in ("2019-2022", "2023"):
        for i, a in enumerate(e.ids):
            for j, b in enumerate(e.ids):
                clave = frozenset((a, b))
                phi = 1.0 if a == b else fuertes.get(clave, 0.01 * ((i + j) % 5 - 2))
                if periodo == "2023" and clave == frozenset(("ELIX_01", "ELIX_02")):
                    phi = 0.95                                                        # la figura de la ventana de ranking no debe usar el período de 2023
                filas.append({"periodo": periodo, "elix_a": a, "elix_b": b, "nombre_a": e.nombres_es[a], "nombre_b": e.nombres_es[b], "n_base": 1000, "n_a": 100, "n_b": 100, "n_ab": 50, "phi": phi})
    return pl.DataFrame(filas)


def _ctx_nuevo(contexto, raiz, nombre):
    return Contexto.crear("extracto", fuente=contexto.fuente, dir_salida_base=raiz / nombre)


@pytest.fixture(scope="module")
def ejecutada(tmp_path_factory, contexto, panel):
    """Corrida de ``ejecutar`` sobre el panel artificial con el par (40, 4) y la matriz phi de la Tarea 8 armada a mano."""
    raiz = tmp_path_factory.mktemp("t10_ejecutada")
    ctx = _ctx_nuevo(contexto, raiz, "salidas")
    _tabla_phi().write_csv(ctx.dir_tablas / "t23_elix_phi.csv")
    r = asoc.ejecutar(ctx, _con_par_40_4(panel))
    ctx.escribir_registros()
    return ctx, r


def _espia(ctx, monkeypatch):
    """Reemplaza ``guardar_figura`` por un espía que conserva la figura (para inspeccionarla) y no escribe el PNG."""
    capturas = {}

    def espia(fig, **kw):
        capturas.update(fig=fig, **kw)
        return None

    monkeypatch.setattr(ctx.figuras, "guardar_figura", espia)
    return capturas


def test_las_tablas_traen_exactamente_las_columnas_del_diseno_en_su_orden(ejecutada):
    ctx, r = ejecutada
    esperado = {
        "matriz_asociacion.csv": ["variable_a", "variable_b", "tipo_a", "tipo_b", "medida", "valor_con_signo", "asociacion_abs", "n_pares", "periodo", "estado", "es_referencia_a",
                                  "es_referencia_b", "posicion_a", "posicion_b", "grupo_a", "grupo_b"],
        "pares_redundantes.csv": ["variable_a", "variable_b", "medida", "valor_con_signo", "asociacion_abs", "n_pares", "cluster", "tamano_cluster", "periodo", "asociacion_2023",
                                  "n_pares_2023", "se_mantiene_2023"],
        "clusters_variables.csv": ["variable", "tipo", "es_referencia", "indice_linkage", "posicion_dendrograma", "altura_union", "cluster", "tamano_cluster", "grupo_dendrograma",
                                   "n_pares_redundantes", "max_asociacion_otra", "variable_max_asociacion"],
        "t23_dendrograma.csv": ["tipo_fila", "nodo", "nodo_a", "nodo_b", "altura", "n_hojas", "variable", "posicion_dendrograma", "grupo_dendrograma"],
        "t23_vif.csv": ["conjunto", "variable", "tipo", "es_agregado", "vif", "r2", "colinealidad_exacta", "clasificacion", "motivo", "estado", "n_filas", "n_columnas_conjunto",
                        "rango_conjunto"],
        "t23_resumen_asociacion.csv": ["periodo", "n_variables", "n_numericas", "n_binarias", "n_categoricas", "n_pares", "n_pares_no_evaluables", "asociacion_max_valor",
                                       "asociacion_max_a", "asociacion_max_b", "n_pares_redundantes", "n_clusters_multiples", "tamano_cluster_max", "n_variables_en_clusters_multiples",
                                       "n_pares_redundantes_evaluados_2023", "n_pares_redundantes_estables_2023", "n_filas_ranking", "n_filas_2023", "n_filas_vif",
                                       "n_vif_excluidas_faltante", "vif_max_base", "vif_max_base_variable", "vif_max_con_agregados", "vif_max_con_agregados_variable",
                                       "n_vif_alerta_base", "n_vif_critico_base", "n_exactas_base", "n_exactas_con_agregados", "n_columnas_base", "rango_base",
                                       "n_columnas_con_agregados", "rango_con_agregados", "n_relaciones_exactas_con_agregados", "identidad_n_comorb_max_abs_dif",
                                       "identidad_puntaje_max_abs_dif", "identidad_puntaje_sin5_max_abs_dif", "phi_elix_disponible", "figuras_omitidas"],
    }
    constantes = {"matriz_asociacion.csv": asoc.COLUMNAS_MATRIZ, "pares_redundantes.csv": asoc.COLUMNAS_PARES, "clusters_variables.csv": asoc.COLUMNAS_CLUSTERS,
                  "t23_dendrograma.csv": asoc.COLUMNAS_DENDROGRAMA, "t23_vif.csv": asoc.COLUMNAS_VIF, "t23_resumen_asociacion.csv": asoc.COLUMNAS_RESUMEN}
    claves = {"matriz_asociacion.csv": "matriz", "pares_redundantes.csv": "pares", "clusters_variables.csv": "clusters", "t23_dendrograma.csv": "dendrograma", "t23_vif.csv": "vif",
              "t23_resumen_asociacion.csv": "resumen"}
    for nombre, columnas in esperado.items():
        assert constantes[nombre] == columnas, nombre
        assert ctx.leer_tabla(nombre).columns == columnas, nombre
        assert r[claves[nombre]].columns == columnas, nombre
    for nombre in ("t23_panel_muestreo.csv", "t23_panel_niveles.csv"):                      # las dos del panel las escribe ``ejecutar`` por ``panel.escribir_tablas``
        assert (ctx.dir_tablas / nombre).is_file()
    indice = {f["nombre"]: f for f in ctx.indice_tablas().iter_rows(named=True)}
    assert {k: indice[k]["tarea"] for k in esperado} == {k: "T10" for k in esperado}
    assert {k: indice[k]["columnas_conteo"] for k in esperado} == {
        "matriz_asociacion.csv": "n_pares", "pares_redundantes.csv": "n_pares;n_pares_2023", "clusters_variables.csv": "", "t23_dendrograma.csv": "", "t23_vif.csv": "n_filas",
        "t23_resumen_asociacion.csv": "n_filas_ranking;n_filas_2023;n_filas_vif",
    }


def test_la_matriz_del_csv_trae_posicion_y_grupo_nulos_solo_para_la_referencia(ejecutada):
    ctx, _ = ejecutada
    m = ctx.leer_tabla("matriz_asociacion.csv", columnas_texto=["variable_a", "variable_b", "tipo_a", "tipo_b", "medida", "periodo", "estado", "es_referencia_a", "es_referencia_b"])
    assert set(m.filter(pl.col("posicion_a").is_null() | pl.col("grupo_a").is_null())["variable_a"]) == {"COD_HOSPITAL"}
    assert set(m.filter(pl.col("posicion_b").is_null() | pl.col("grupo_b").is_null())["variable_b"]) == {"COD_HOSPITAL"}
    assert m["periodo"].unique().to_list() == ["2019-2022"] and set(m["es_referencia_a"]) == {"si", "no"}
    por_var = m.select("variable_a", "posicion_a").unique().drop_nulls()
    assert sorted(por_var["posicion_a"].to_list()) == list(range(1, 62))                            # 61 variables sin la referencia, posiciones 1 a 61


def test_el_par_con_40_y_4_casos_conserva_su_asociacion_y_anula_solo_el_bloque_de_2023_en_el_csv(ejecutada):
    ctx, _ = ejecutada
    t = ctx.leer_tabla("pares_redundantes.csv", columnas_texto=["variable_a", "variable_b", "medida", "periodo", "se_mantiene_2023"])
    f = t.filter((pl.col("variable_a") == "ES_EXTRANJERO") & (pl.col("variable_b") == "NACIONALIDAD")).row(0, named=True)
    assert f["n_pares"] == 40 and f["asociacion_abs"] >= 0.8 and f["cluster"] is not None and f["tamano_cluster"] == 2
    assert f["n_pares_2023"] is None and f["asociacion_2023"] is None and f["se_mantiene_2023"] is None
    otro = t.filter((pl.col("variable_a") == "CADENA_A") & (pl.col("variable_b") == "CADENA_B")).row(0, named=True)
    assert otro["n_pares_2023"] == 240 and otro["asociacion_2023"] is not None and otro["se_mantiene_2023"] == "si"           # el resto de los pares conserva su bloque de 2023
    fila_indice = ctx.indice_tablas().filter(pl.col("nombre") == "pares_redundantes.csv").row(0, named=True)
    assert fila_indice["celdas_suprimidas"] == 1                                                    # la celda anulada en el paso previo cuenta en el índice (NIT 6)


def test_una_celda_pequena_en_la_matriz_se_anula_junto_con_sus_cifras_y_la_privacidad_queda_en_cero(contexto, panel, tmp_path):
    rng = np.random.default_rng(61)
    n, anio = panel.datos.height, panel.datos["anio"].to_numpy()
    valores = rng.normal(size=n)
    extra = {"SOLO_A": ("num", [float(v) if (anio[i] <= 2022 and i < 6) else None for i, v in enumerate(valores)]),
             "SOLO_B": ("num", [float(v) * 2 if (anio[i] <= 2022 and i < 6) else None for i, v in enumerate(valores)])}
    ctx = _ctx_nuevo(contexto, tmp_path, "salidas_pequena")
    r = asoc.ejecutar(ctx, _con_variables(panel, extra))
    ctx.escribir_registros()
    m = ctx.leer_tabla("matriz_asociacion.csv", columnas_texto=["variable_a", "variable_b", "medida", "periodo", "estado", "es_referencia_a", "es_referencia_b", "tipo_a", "tipo_b"])
    f = m.filter((pl.col("variable_a") == "SOLO_A") & (pl.col("variable_b") == "SOLO_B")).row(0, named=True)
    assert f["n_pares"] is None and f["asociacion_abs"] is None and f["valor_con_signo"] is None and f["medida"] == "spearman"
    d = m.filter((pl.col("variable_a") == "SOLO_A") & (pl.col("variable_b") == "SOLO_A")).row(0, named=True)
    assert d["n_pares"] is None and d["asociacion_abs"] is None                                         # la diagonal de una variable con 6 casos también se anula
    assert ctx.indice_tablas().filter(pl.col("nombre") == "matriz_asociacion.csv")["celdas_suprimidas"][0] >= 3
    res = r["resumen"].row(0, named=True)
    assert res["n_vif_excluidas_faltante"] == 2                                                         # con 6 casos informados, ambas superan el faltante de alerta
    esc = privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)
    assert privacidad.total_violaciones(esc) == 0, esc.filter(pl.col("n_violaciones") > 0)


def test_las_metricas_tienen_los_prefijos_y_el_alcance_explicitos_y_ninguna_cubre_2024(ejecutada):
    ctx, r = ejecutada
    m = ctx.metricas().filter(pl.col("clave").str.starts_with("asociacion.") | pl.col("clave").str.starts_with("vif."))
    assert set(m["clave"]) == {
        "asociacion.n_variables", "asociacion.n_pares_redundantes", "asociacion.n_clusters_multiples", "asociacion.max_abs", "asociacion.n_pares_redundantes_estables_2023", "vif.max_base",
        "vif.max_con_agregados", "vif.n_exactas_base", "vif.n_exactas_con_agregados", "vif.rango_base", "vif.rango_con_agregados",
    }
    for fila in m.iter_rows(named=True):
        assert fila["tarea"] == "T10" and fila["nivel"] == "extracto"
        assert fila["alcance"] == ("2023" if fila["clave"].endswith("_2023") else "2019-2022"), fila["clave"]
        assert "2024" not in fila["alcance"] + fila["clave"] + fila["nota"] + fila["tabla"]
    res = r["resumen"].row(0, named=True)
    por = {f["clave"]: f for f in m.iter_rows(named=True)}
    assert por["asociacion.max_abs"]["valor"] == pytest.approx(res["asociacion_max_valor"], abs=1e-6)
    assert por["asociacion.max_abs"]["nota"] == "ENTRANTE_TIPO_PROCEDENCIA con PROCEDENCIA_AGR"             # el par va en la nota
    # ES_EXTRANJERO tiene 40 casos informados en 960: su faltante supera el umbral de alerta y sale del VIF (43 columnas con agregados, rango 40, las mismas 3 relaciones exactas)
    assert por["vif.n_exactas_con_agregados"]["valor"] == 33 and por["vif.rango_con_agregados"]["valor"] == 40 and por["vif.n_exactas_base"]["valor"] == 0
    assert res["n_vif_excluidas_faltante"] == 1 and res["n_columnas_con_agregados"] - res["rango_con_agregados"] == res["n_relaciones_exactas_con_agregados"] == 3
    assert por["asociacion.n_variables"]["valor"] == 61 and por["asociacion.n_clusters_multiples"]["valor"] == 8
    assert por["asociacion.n_pares_redundantes_estables_2023"]["valor"] == 10                             # el par con 4 casos en 2023 no se evalúa y no cuenta como estable


def test_sin_el_anio_de_confirmacion_la_metrica_de_2023_se_registra_sin_valor_y_con_nota(contexto, panel, tmp_path):
    ctx = _ctx_nuevo(contexto, tmp_path, "salidas_sin_2023")
    asoc.ejecutar(ctx, dataclasses.replace(panel, datos=panel.datos.filter(pl.col("anio") <= 2022), anio_confirmacion=None))
    f = ctx.metricas().filter(pl.col("clave") == "asociacion.n_pares_redundantes_estables_2023").row(0, named=True)
    assert f["valor"] is None and f["nota"] == "confirmación no disponible" and f["alcance"] == "2023"


def test_la_privacidad_de_las_salidas_de_la_tarea_10_queda_en_cero(ejecutada):
    ctx, _ = ejecutada
    esc = privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)
    assert privacidad.total_violaciones(esc) == 0, esc.filter(pl.col("n_violaciones") > 0)
    assert ctx.cortafuegos.violaciones() == 0 and ctx.cortafuegos.n_bloqueos() == 0 and ctx.cortafuegos.registro().height == 0           # el panel recibido no lee nada


def test_las_cuatro_figuras_existen_con_su_csv_y_coinciden_con_el_plan_de_figuras(ejecutada):
    ctx, _ = ejecutada
    idx = {f["id_figura"]: f for f in ctx.figuras.indice().iter_rows(named=True)}
    assert sorted(idx) == ["F10_01", "F10_02", "F10_03", "F10_04"]
    plan = {f["id_figura"]: f for f in config.leer_csv_texto(rutas.DIR_CONFIG / "plan_figuras.csv").iter_rows(named=True) if f["id_figura"].startswith("F10_")}
    for id_figura, fila in idx.items():
        assert fila["csv"] == plan[id_figura]["tabla_csv"] and fila["decision_relacionada"] == (plan[id_figura]["decision_relacionada"] or ""), id_figura
        assert fila["tarea"] == "10" and fila["uso_interno"] == "no" and (ctx.dir_figuras / fila["archivo_png"]).is_file() and fila["descripcion_alternativa"].strip()
    assert idx["F10_02"]["csv"] == "t23_dendrograma.csv" and idx["F10_04"]["decision_relacionada"] == "8" and idx["F10_01"]["decision_relacionada"] == ""
    assert idx["F10_01"]["titulo"] == "Asociación entre variables, agrupada por similitud (2019 a 2022)"


def test_figura_mapa_se_arma_leyendo_solo_la_matriz_con_el_mismo_orden_y_los_mismos_recuadros(ejecutada, contexto, tmp_path, monkeypatch):
    ctx, r = ejecutada
    solo = _ctx_nuevo(contexto, tmp_path, "solo_matriz")
    shutil.copy(ctx.dir_tablas / "matriz_asociacion.csv", solo.dir_tablas / "matriz_asociacion.csv")
    assert [p.name for p in solo.dir_tablas.glob("*.csv")] == ["matriz_asociacion.csv"]              # se borran (no se copian) las demás tablas de la carpeta
    cap = _espia(solo, monkeypatch)
    assert asoc.figura_mapa(solo) is True
    fig = cap["fig"]
    try:
        ax = fig.axes[0]
        etiquetas = [t.get_text() for t in ax.get_yticklabels()]
        c = r["clusters"]
        esperado = c.filter(pl.col("es_referencia") == "no").sort("posicion_dendrograma")["variable"].to_list() + ["COD_HOSPITAL"]
        assert etiquetas == esperado == [t.get_text() for t in ax.get_xticklabels()]               # el mismo orden del dendrograma, con la referencia al final
        grupos = c.filter(pl.col("es_referencia") == "no").sort("posicion_dendrograma")["grupo_dendrograma"].to_list()
        cajas = sorted((int(round(p.get_x() + 0.5)), int(p.get_width())) for p in ax.patches)
        assert cajas == _bloques(grupos) and len(cajas) == 8                                      # los mismos recuadros que da clusters_variables.csv
        horizontales = [ln for ln in ax.lines if len(set(ln.get_ydata())) == 1]
        assert len(horizontales) == 1 and list(horizontales[0].get_ydata()) == [len(esperado) - 1.5, len(esperado) - 1.5]            # línea gruesa que separa la referencia
        assert (cap["id_figura"], cap["csv"], cap["decision"], cap["anios"], cap["uso_interno"], cap["tarea"]) == ("F10_01", "matriz_asociacion.csv", "", "2019 a 2022", False, 10)
        assert max(t.get_fontsize() for t in ax.get_yticklabels()) == min(t.get_fontsize() for t in ax.get_yticklabels()) == 6.5
        assert all(t.get_rotation() == 90 for t in ax.get_xticklabels())
    finally:
        plt.close(fig)


def _bloques(grupos):
    """Bloques ``(inicio, tamano)`` de más de un elemento con el mismo grupo contiguo."""
    out, i = [], 0
    while i < len(grupos):
        j = i
        while j + 1 < len(grupos) and grupos[j + 1] == grupos[i]:
            j += 1
        if j > i:
            out.append((i, j - i + 1))
        i = j + 1
    return out


def test_el_orden_de_hojas_de_f10_02_coincide_con_posicion_dendrograma_y_el_corte_se_ve(ejecutada, contexto, tmp_path, monkeypatch):
    ctx, r = ejecutada
    solo = _ctx_nuevo(contexto, tmp_path, "solo_dendrograma")
    shutil.copy(ctx.dir_tablas / "t23_dendrograma.csv", solo.dir_tablas / "t23_dendrograma.csv")
    cap = _espia(solo, monkeypatch)
    assert asoc.figura_dendrograma(solo) is True
    fig = cap["fig"]
    try:
        ax = fig.axes[0]
        por_altura = [t for _, t in sorted(zip(ax.get_yticks(), [x.get_text() for x in ax.get_yticklabels()]))]
        hojas = ctx.leer_tabla("t23_dendrograma.csv", columnas_texto=["tipo_fila", "variable"]).filter(pl.col("tipo_fila") == "hoja")
        assert ax.yaxis_inverted() and por_altura == hojas.sort("posicion_dendrograma")["variable"].to_list()              # posición 1 arriba
        cortes = [ln for ln in ax.lines if len(set(ln.get_xdata())) == 1 and abs(ln.get_xdata()[0] - 0.2) < 1e-9]
        assert len(cortes) == 1 and cortes[0].get_linestyle() == "--"
        assert any(t.get_text().strip() == "corte de redundancia" for t in ax.texts) and ax.get_xlabel() == "Distancia (1 menos la asociación absoluta)"
        colores = {tuple(np.round(c, 3)) for col in ax.collections for c in col.get_colors()}
        assert len(colores) >= 2                                                                    # hay enlaces de colores y enlaces grises
        assert (cap["id_figura"], cap["csv"], cap["anios"]) == ("F10_02", "t23_dendrograma.csv", "2019 a 2022")
    finally:
        plt.close(fig)


def test_f10_04_usa_solo_el_periodo_de_ranking_y_enmarca_los_grupos_al_corte(ejecutada, contexto, tmp_path, monkeypatch):
    ctx, _ = ejecutada
    solo = _ctx_nuevo(contexto, tmp_path, "solo_phi")
    shutil.copy(ctx.dir_tablas / "t23_elix_phi.csv", solo.dir_tablas / "t23_elix_phi.csv")
    cap = _espia(solo, monkeypatch)
    assert asoc.figura_phi_elix(solo) is True
    fig = cap["fig"]
    try:
        ax = fig.axes[0]
        etiquetas = [t.get_text() for t in ax.get_yticklabels()]
        ids = [e[:2] for e in etiquetas]
        assert len(etiquetas) == 31 and sorted(ids) == [f"{i:02d}" for i in range(1, 32)] and all(len(e) > 3 and e[2] == " " for e in etiquetas)       # "nn nombre"
        posicion = {i: k for k, i in enumerate(ids)}
        cajas = sorted((int(round(p.get_x() + 0.5)), int(p.get_width())) for p in ax.patches)
        assert len(cajas) == 3 and all(t == 2 for _, t in cajas)                                         # tres pares fuertes de 2019-2022; el de 2023 (01 con 02) no cuenta
        for a, b in (("11", "12"), ("19", "20"), ("03", "04")):
            assert abs(posicion[a] - posicion[b]) == 1 and (min(posicion[a], posicion[b]), 2) in cajas
        assert abs(posicion["01"] - posicion["02"]) != 1 or (min(posicion["01"], posicion["02"]), 2) not in cajas
        assert cap["decision"] == "8" and cap["csv"] == "t23_elix_phi.csv" and cap["id_figura"] == "F10_04"
        barra = fig.axes[1]
        fig.canvas.draw()
        marcas = [t.get_text() for t in barra.get_yticklabels() if t.get_text()]
        assert marcas and all("." not in t for t in marcas)                                              # coma decimal en la barra
        vmax = ax.images[0].norm.vmax
        assert vmax == pytest.approx(round(0.90 + 0.049, 1)) or vmax == pytest.approx(0.9)               # simétrico: el máximo fuera de la diagonal más 0,049, a un decimal
        assert ax.images[0].norm.vmin == -vmax
    finally:
        plt.close(fig)


def test_f10_03_dibuja_la_colinealidad_exacta_con_trama_y_rotulo_las_lineas_de_5_y_10_y_escala_logaritmica(ejecutada, contexto, tmp_path, monkeypatch):
    ctx, _ = ejecutada
    solo = _ctx_nuevo(contexto, tmp_path, "solo_vif")
    shutil.copy(ctx.dir_tablas / "t23_vif.csv", solo.dir_tablas / "t23_vif.csv")
    cap = _espia(solo, monkeypatch)
    assert asoc.figura_vif(solo) is True
    fig = cap["fig"]
    try:
        base, con = fig.axes[0], fig.axes[1]
        assert base.get_xscale() == con.get_xscale() == "log" and base.get_xlim()[0] == con.get_xlim()[0] == 1.0
        v = ctx.leer_tabla("t23_vif.csv", columnas_texto=["conjunto", "variable", "tipo", "es_agregado", "colinealidad_exacta", "clasificacion", "motivo", "estado"])
        exactas = v.filter((pl.col("conjunto") == "con_agregados") & (pl.col("colinealidad_exacta") == "si")).height
        assert exactas == 33
        assert sum(1 for p in base.patches if p.get_hatch()) == 0 and sum(1 for p in con.patches if p.get_hatch()) == exactas             # trama solo en las exactas
        assert [t.get_text() for t in con.texts].count("exacta") == exactas and "exacta" not in [t.get_text() for t in base.texts]            # nunca como un número
        for ax in (base, con):
            xs = sorted(round(ln.get_xdata()[0], 6) for ln in ax.lines if len(set(ln.get_xdata())) == 1)
            assert xs == [5.0, 10.0]                                                               # vif_alerta y vif_critico
            assert max(p.get_width() for p in ax.patches) == pytest.approx(ax.get_xlim()[1]) if ax is con else True       # la exacta llega hasta el borde del eje
        etiquetas = [t.get_text() for t in base.get_yticklabels()]
        top = v.filter((pl.col("conjunto") == "base") & (pl.col("estado") == "ok")).sort("vif", descending=True)["variable"][0]
        assert etiquetas[-1] == top and etiquetas[-1] in ("CADENA_B", "CADENA_A", "CADENA_C")           # ordenadas: el mayor VIF arriba
        assert {"VIF bajo", "VIF crítico", "colinealidad exacta", "umbral de alerta", "umbral crítico"} <= _textos_de_leyendas(fig)
        assert all(lg.get_in_layout() is False for a in fig.axes for lg in [a.get_legend()] if lg)       # la leyenda va en su propia banda, fuera de los datos
    finally:
        plt.close(fig)


def _textos_de_leyendas(fig) -> set[str]:
    return {t.get_text() for a in fig.axes for lg in [a.get_legend()] if lg for t in lg.get_texts()}


def test_f10_03_no_dibuja_como_exacta_una_celda_de_vif_suprimida(contexto, u, tmp_path, monkeypatch):
    rng = np.random.default_rng(91)
    cols = {n: col(n, "num", rng.normal(size=8)) for n in ("X", "Y", "Z")}                           # 8 casos completos: el VIF se calcula y la tabla lo anula (n entre 1 y 9)
    vif = asoc.calcular_vif(cols, [], u)
    assert vif["vif"].null_count() == 0 and vif["n_filas"][0] == 8
    ctx = _ctx_nuevo(contexto, tmp_path, "vif_suprimido")
    ctx.escribir_tabla("t23_vif.csv", vif, tarea="T10", descripcion="prueba", cols_n=["n_filas"], dependientes=["r2", "vif"])
    escrito = ctx.leer_tabla("t23_vif.csv", columnas_texto=["conjunto", "variable", "tipo", "es_agregado", "colinealidad_exacta", "clasificacion", "motivo", "estado"])
    assert escrito["vif"].null_count() == escrito.height and escrito["n_filas"].null_count() == escrito.height and (escrito["colinealidad_exacta"] == "no").all()
    cap = _espia(ctx, monkeypatch)
    assert asoc.figura_vif(ctx) is True
    fig = cap["fig"]
    try:
        for ax in fig.axes[:2]:
            assert [t.get_text() for t in ax.texts] == ["n/d: celda suprimida (menos de diez casos)"] and len(ax.patches) == 0         # ni barras ni «exacta»
        assert "colinealidad exacta" not in _textos_de_leyendas(fig)
    finally:
        plt.close(fig)


def test_las_figuras_no_dibujan_tablas_vacias_ni_fallan(contexto, tmp_path):
    ctx = _ctx_nuevo(contexto, tmp_path, "vacias")
    for nombre, columnas in (("matriz_asociacion.csv", asoc.COLUMNAS_MATRIZ), ("t23_dendrograma.csv", asoc.COLUMNAS_DENDROGRAMA), ("t23_vif.csv", asoc.COLUMNAS_VIF)):
        pl.DataFrame({c: [] for c in columnas}, schema={c: pl.String for c in columnas}).write_csv(ctx.dir_tablas / nombre)
    pl.DataFrame({c: [] for c in ("periodo", "elix_a", "elix_b", "nombre_a", "nombre_b", "phi")}, schema={c: pl.String for c in ("periodo", "elix_a", "elix_b", "nombre_a", "nombre_b", "phi")}).write_csv(
        ctx.dir_tablas / "t23_elix_phi.csv")
    assert (asoc.figura_mapa(ctx), asoc.figura_dendrograma(ctx), asoc.figura_vif(ctx), asoc.figura_phi_elix(ctx)) == (False, False, False, False)
    assert ctx.figuras.ids() == []
    sin_phi = _ctx_nuevo(contexto, tmp_path, "sin_phi")
    assert asoc.figura_phi_elix(sin_phi) is False and sin_phi.figuras.ids() == []                          # sin t23_elix_phi.csv no hay error


def test_sin_la_matriz_phi_de_la_tarea_8_f10_04_se_omite_sin_error_y_queda_anotada(contexto, panel, tmp_path):
    ctx = _ctx_nuevo(contexto, tmp_path, "sin_tarea_8")
    r = asoc.ejecutar(ctx, panel)
    res = r["resumen"].row(0, named=True)
    assert (res["phi_elix_disponible"], res["figuras_omitidas"]) == ("no", "F10_04")
    assert ctx.figuras.ids() == ["F10_01", "F10_02", "F10_03"]
    inf = Informe(ctx, "t23_prueba_sin_phi.md", "Informe de prueba")
    asoc.agregar_secciones(inf, ctx, n0=1)
    md = inf.guardar().read_text(encoding="utf8")
    assert "Figura F10_04." not in md and "no se generó" in md and "Figura F10_01." in md and "Figura F10_03." in md          # nunca incrusta una figura omitida


def test_sin_variables_numericas_ni_binarias_el_vif_queda_vacio_y_f10_03_se_omite(contexto, panel, tmp_path):
    solo_cat = dataclasses.replace(panel, variables={n: v for n, v in panel.variables.items() if v.tipo == "cat"})
    ctx = _ctx_nuevo(contexto, tmp_path, "solo_categoricas")
    _tabla_phi().write_csv(ctx.dir_tablas / "t23_elix_phi.csv")
    r = asoc.ejecutar(ctx, solo_cat)
    assert r["vif"].height == 0 and ctx.leer_tabla("t23_vif.csv").height == 0
    res = r["resumen"].row(0, named=True)
    assert res["figuras_omitidas"] == "F10_03" and res["n_filas_vif"] == 0 and res["vif_max_base"] is None and res["n_exactas_con_agregados"] is None
    assert ctx.figuras.ids() == ["F10_01", "F10_02", "F10_04"]
    inf = Informe(ctx, "t23_prueba_sin_vif.md", "Informe de prueba")
    asoc.agregar_secciones(inf, ctx, n0=1)
    md = inf.guardar().read_text(encoding="utf8")
    assert "Figura F10_03." not in md and "el VIF no se calculó" in md


def test_ejecutar_devuelve_lo_mismo_que_calcular_y_es_determinista(ejecutada, panel):
    ctx, r = ejecutada
    otra = asoc.calcular(ctx, _con_par_40_4(panel))                       # el mismo contexto: el resumen depende de si la carpeta de tablas trae t23_elix_phi.csv
    for clave in ("matriz", "pares", "clusters", "dendrograma", "vif", "resumen"):
        assert r[clave].equals(otra[clave]), clave
    assert np.array_equal(r["A"], otra["A"], equal_nan=True) and np.array_equal(r["Z"], otra["Z"]) and r["componentes"] == otra["componentes"] and r["nombres"] == otra["nombres"]


def test_el_informe_agrega_las_cinco_secciones_las_figuras_y_la_desviacion_declarada(ejecutada):
    ctx, _ = ejecutada
    inf = Informe(ctx, "t23_prueba_asociacion.md", "Informe de prueba")
    asoc.agregar_secciones(inf, ctx, n0=1)
    md = inf.guardar().read_text(encoding="utf8")
    for titulo in ("1. Qué se midió y con qué medidas", "2. La matriz de asociación y su lectura", "3. Pares y clusters redundantes", "4. VIF y colinealidad exacta", "5. Límites"):
        assert titulo in md, titulo
    assert asoc.N_SECCIONES == 5 and md.count("\n## ") == 5
    for fig in ("F10_01", "F10_02", "F10_03", "F10_04"):
        assert f"Figura {fig}." in md
    assert "{" not in md and "}" not in md and "extracto" in md.lower()
    assert "por bloques" in md and "regla general de privacidad" in md                                       # la desviación declarada (NIT 6), sin dígitos
    assert "Los quince pares más asociados" in md and "ENTRANTE_TIPO_PROCEDENCIA" in md


def test_el_informe_numera_las_secciones_desde_n0(ejecutada):
    ctx, _ = ejecutada
    inf = Informe(ctx, "t23_prueba_asociacion_n0.md", "Informe de prueba")
    asoc.agregar_secciones(inf, ctx, n0=7)
    md = inf.guardar().read_text(encoding="utf8")
    assert "## 7. Qué se midió y con qué medidas" in md and "## 11. Límites" in md and "## 1. " not in md


def test_las_cifras_del_informe_vienen_de_los_csv_y_el_resumen_publica_lo_que_ve_el_lector(ejecutada):
    ctx, _ = ejecutada
    inf = Informe(ctx, "t23_prueba_asociacion_cifras.md", "Informe de prueba")
    asoc.agregar_secciones(inf, ctx, n0=1)
    inf.guardar()
    c = ctx.cifras_informe().filter(pl.col("informe") == "t23_prueba_asociacion_cifras.md")
    fuentes = set(c["csv"])
    assert fuentes <= {"t23_resumen_asociacion.csv", "umbrales.csv", "clusters_variables.csv", "pares_redundantes.csv", "t23_vif.csv"} and "t23_resumen_asociacion.csv" in fuentes
    assert set(c.filter(pl.col("csv") == "umbrales.csv")["marcador"]) == {"umbral", "faltante", "alerta", "critico"}


# ============================================================================================ integración sintética (B5): extracto y modo completo
@pytest.fixture(scope="module")
def sintetica(tmp_path_factory, u):
    """Fuente sintética del generador existente e insumos de las Tareas 4 y 8 generados en un contexto aparte (esas tareas leen 2024 solo con propósito de estructura)."""
    raiz = tmp_path_factory.mktemp("t10_integracion")
    fuente = crear_fuente_sintetica(raiz / "datos", n=700, semilla=29)
    ins = Contexto.crear("extracto", fuente=fuente, dir_salida_base=raiz / "insumos")
    descripcion.ejecutar(ins)
    comorbilidad.ejecutar(ins)
    return raiz, fuente, ins.dir_tablas


def _correr(sintetica, nombre, *, completo=False, con_insumos=True):
    """Panel real, ``asociacion.ejecutar``, informe provisional y registros, en una carpeta propia (el contexto de la Tarea 10 no tiene ninguna lectura de 2024)."""
    raiz, fuente, insumos = sintetica
    if completo:
        ctx = Contexto.crear("completo", fuente=dataclasses.replace(fuente, etiqueta="completo"), dir_salida_base=raiz / nombre)
    else:
        ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=raiz / nombre)
    if con_insumos:
        for archivo in ("t22_perfil_columnas.csv", "t23_elix_phi.csv"):
            shutil.copy(insumos / archivo, ctx.dir_tablas / archivo)
    panel = pn.construir_panel(ctx)
    a = asoc.ejecutar(ctx, panel)
    inf = Informe(ctx, "t23_prueba_asociacion.md", "Informe de prueba")
    asoc.agregar_secciones(inf, ctx, n0=1)
    inf.guardar()
    ctx.escribir_registros()
    return ctx, panel, a


@pytest.fixture(scope="module")
def extracto(sintetica):
    return _correr(sintetica, "salidas_extracto")


@pytest.fixture(scope="module")
def completo(sintetica):
    return _correr(sintetica, "salidas_completo", completo=True)


def _plan_f10():
    plan = config.leer_csv_texto(rutas.DIR_CONFIG / "plan_figuras.csv")
    return {f["id_figura"]: (f["tabla_csv"], f["decision_relacionada"] or "") for f in plan.iter_rows(named=True) if f["id_figura"].startswith("F10_")}


def test_integracion_extracto_escribe_las_ocho_tablas_las_cuatro_figuras_y_el_indice_coincide_con_el_plan(extracto):
    ctx, panel, a = extracto
    for nombre in ("matriz_asociacion.csv", "pares_redundantes.csv", "clusters_variables.csv", "t23_dendrograma.csv", "t23_vif.csv", "t23_resumen_asociacion.csv",
                   "t23_panel_muestreo.csv", "t23_panel_niveles.csv"):
        assert (ctx.dir_tablas / nombre).is_file(), nombre
    figuras = pl.read_csv(ctx.dir_tablas / "figuras_indice.csv", infer_schema_length=1000)
    assert figuras["id_figura"].to_list() == ["F10_01", "F10_02", "F10_03", "F10_04"]
    plan = _plan_f10()
    assert {f["id_figura"]: (f["csv"], f["decision_relacionada"] or "") for f in figuras.iter_rows(named=True)} == plan
    assert plan["F10_02"] == ("t23_dendrograma.csv", "") and plan["F10_04"] == ("t23_elix_phi.csv", "8")
    assert all((ctx.dir_figuras / f"{i}.png").is_file() for i in plan) and (figuras["uso_interno"] == "no").all()
    r = a["resumen"].row(0, named=True)
    assert (r["phi_elix_disponible"], r["figuras_omitidas"]) == ("si", "ninguna") and r["n_variables"] == 58
    assert (r["n_numericas"], r["n_binarias"], r["n_categoricas"]) == (5, 37, 16)                     # las 58 de los extractos: 5 numéricas, 37 binarias y 16 categóricas
    assert panel.con_historial is False and "N_EGRESOS_12M" not in a["nombres"] and "COD_HOSPITAL" in a["nombres"] and len(a["nombres"]) == 59


def test_integracion_extracto_cumple_privacidad_cortafuegos_y_no_lee_2024(extracto):
    ctx, _, _ = extracto
    esc = privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)
    assert privacidad.total_violaciones(esc) == 0, esc.filter(pl.col("n_violaciones") > 0)
    assert ctx.cortafuegos.violaciones() == 0 and ctx.cortafuegos.n_bloqueos() == 0
    reg = ctx.cortafuegos.registro()
    assert 2024 not in set(reg["anio"]) and set(reg["tarea"]) == {"T10"} and set(reg["anio"]) == {2019, 2020, 2021, 2022, 2023}
    m = ctx.metricas()
    assert 2024 not in {a for a in m["alcance"]} and not any("2024" in c for c in m["clave"]) and set(m["nivel"]) == {"extracto"}
    claves = set(m.filter(pl.col("clave").str.starts_with("asociacion.") | pl.col("clave").str.starts_with("vif."))["clave"])
    assert len(claves) == 11


def test_integracion_extracto_pares_y_clusters_esperados_del_generador(extracto):
    _, _, a = extracto
    pares = {tuple(sorted(p)) for p in zip(a["pares"]["variable_a"], a["pares"]["variable_b"])}
    for par in (("PROCEDENCIA_AGR", "TIPO_PROCEDENCIA"), ("CIE10_3C", "GRUPO_CLINICO"), ("DIA_SEMANA", "ES_FIN_SEMANA")):
        assert par in pares, par
    comp = a["componentes"]
    assert comp["TIPO_PROCEDENCIA"] == comp["PROCEDENCIA_AGR"] and comp["GRUPO_CLINICO"] == comp["CIE10_3C"] and comp["DIA_SEMANA"] == comp["ES_FIN_SEMANA"]
    assert comp["PROVINCIA"] == comp["ESPECIALIDAD_MEDICA"] == comp["COMUNA"]                          # los proxies del hospital quedan en un mismo componente
    assert "COD_HOSPITAL" not in comp and a["clusters"].filter(pl.col("variable") == "COD_HOSPITAL")["cluster"][0] is None


def test_integracion_extracto_los_agregados_de_comorbilidad_tienen_vif_exacto_e_identidades_cero(extracto):
    _, _, a = extracto
    v = a["vif"]
    con = v.filter((pl.col("conjunto") == "con_agregados") & (pl.col("variable").is_in(["N_COMORB_ELIX", "PUNTAJE_VAN_WALRAVEN"])))
    assert (con["colinealidad_exacta"] == "si").all() and con["vif"].null_count() == con.height and (con["clasificacion"] == "exacta").all()
    base = v.filter((pl.col("conjunto") == "base") & (pl.col("estado") == "ok"))
    assert base["vif"].null_count() == 0 and (base["colinealidad_exacta"] == "no").all()
    r = a["resumen"].row(0, named=True)
    assert r["identidad_n_comorb_max_abs_dif"] == r["identidad_puntaje_max_abs_dif"] == r["identidad_puntaje_sin5_max_abs_dif"] == 0.0
    assert r["n_relaciones_exactas_con_agregados"] == 3 and r["n_exactas_base"] == 0
    constantes = {f"ELIX_{i:02d}" for i in (3, 4, 7, 9, 16, 17, 18, 21, 24, 26, 29, 30)}                # las 12 categorías constantes del generador salen del VIF con su motivo
    ex = v.filter((pl.col("estado") == "excluida") & (pl.col("conjunto") == "base"))
    assert set(ex["variable"]) >= constantes and set(ex.filter(pl.col("variable").is_in(sorted(constantes)))["motivo"]) == {"constante en la ventana"}


def test_integracion_extracto_el_informe_se_arma_sin_llaves_con_la_advertencia_las_cinco_secciones_y_las_figuras(extracto):
    ctx, _, _ = extracto
    md = (ctx.dir_informes / "t23_prueba_asociacion.md").read_text(encoding="utf8")
    assert "{" not in md and "}" not in md and "extracto" in md.lower() and "nada de lo anterior es concluyente" in md
    for titulo in ("1. Qué se midió y con qué medidas", "2. La matriz de asociación y su lectura", "3. Pares y clusters redundantes", "4. VIF y colinealidad exacta", "5. Límites"):
        assert titulo in md, titulo
    for fig in ("F10_01", "F10_02", "F10_03", "F10_04"):
        assert f"Figura {fig}." in md and f"../figuras/{fig}.png" in md
    assert ctx.cifras_informe().filter(pl.col("informe") == "t23_prueba_asociacion.md").height > 20


def test_integracion_modo_completo_incorpora_el_historial_y_lo_excluye_del_vif_por_faltante(completo):
    ctx, panel, a = completo
    assert panel.con_historial and panel.regimenes == [[2019, 2020], [2021, 2022, 2023]]
    historial = {"N_EGRESOS_12M", "DIAS_DESDE_EGRESO_PREVIO", "HISTORIA_DISPONIBLE"}
    assert historial <= set(a["nombres"]) and historial <= set(a["clusters"]["variable"]) and historial <= set(a["matriz"]["variable_a"])        # entran a la matriz y a los clusters
    r = a["resumen"].row(0, named=True)
    assert r["n_variables"] == 61 and r["n_vif_excluidas_faltante"] == 2 and r["figuras_omitidas"] == "ninguna"
    ex = a["vif"].filter(pl.col("motivo") == asoc.MOTIVO_FALTANTE)
    assert set(ex["variable"]) == {"N_EGRESOS_12M", "DIAS_DESDE_EGRESO_PREVIO"} and set(ex["conjunto"]) == {"base", "con_agregados"}
    assert (ex["estado"] == "excluida").all() and ex["vif"].null_count() == ex.height and ex.height == 4              # cada una con su fila en los dos conjuntos
    assert "HISTORIA_DISPONIBLE" in set(a["vif"].filter(pl.col("estado") == "ok")["variable"])                        # nunca es nula: se queda


def test_integracion_modo_completo_no_trae_la_advertencia_de_extracto_y_cumple_las_reglas(completo):
    ctx, _, _ = completo
    md = (ctx.dir_informes / "t23_prueba_asociacion.md").read_text(encoding="utf8")
    assert "extracto" not in md.lower() and "completo" in md and "{" not in md and "}" not in md
    assert set(ctx.metricas()["nivel"]) == {"completo"} and set(ctx.indice_tablas()["fuente"]) == {"completo"}
    esc = privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)
    assert privacidad.total_violaciones(esc) == 0, esc.filter(pl.col("n_violaciones") > 0)
    assert ctx.cortafuegos.violaciones() == 0 and ctx.cortafuegos.n_bloqueos() == 0 and 2024 not in set(ctx.cortafuegos.registro()["anio"])
    assert set(ctx.cortafuegos.registro()["tarea"]) == {"T10"}


def test_integracion_sin_los_insumos_de_las_tareas_4_y_8_f10_04_se_omite_sin_error(sintetica):
    ctx, _, a = _correr(sintetica, "salidas_sin_insumos", con_insumos=False)
    r = a["resumen"].row(0, named=True)
    assert (r["phi_elix_disponible"], r["figuras_omitidas"]) == ("no", "F10_04") and ctx.figuras.ids() == ["F10_01", "F10_02", "F10_03"]
    md = (ctx.dir_informes / "t23_prueba_asociacion.md").read_text(encoding="utf8")
    assert "Figura F10_04." not in md and "{" not in md


def test_determinismo_dos_corridas_en_carpetas_distintas_dan_las_mismas_sumas_de_verificacion(sintetica, extracto):
    ctx1, _, _ = extracto
    ctx2, _, _ = _correr(sintetica, "salidas_extracto_repetida")
    assert ctx1.dir_salida != ctx2.dir_salida
    t1, t2 = (pl.read_csv(c.dir_tablas / "tablas_indice.csv") for c in (ctx1, ctx2))
    assert t1.select("nombre", "tarea", "filas", "columnas_conteo", "celdas_suprimidas", "fuente", "sha256").equals(t2.select("nombre", "tarea", "filas", "columnas_conteo", "celdas_suprimidas", "fuente", "sha256"))
    f1, f2 = (pl.read_csv(c.dir_tablas / "figuras_indice.csv") for c in (ctx1, ctx2))
    assert f1.select("id_figura", "sha256_png", "sha256_csv").equals(f2.select("id_figura", "sha256_png", "sha256_csv")) and f1.height == 4
    for nombre in ("matriz_asociacion.csv", "pares_redundantes.csv", "clusters_variables.csv", "t23_dendrograma.csv", "t23_vif.csv", "t23_resumen_asociacion.csv"):
        assert (ctx1.dir_tablas / nombre).read_bytes() == (ctx2.dir_tablas / nombre).read_bytes(), nombre
    for i in ("F10_01", "F10_02", "F10_03", "F10_04"):
        assert (ctx1.dir_figuras / f"{i}.png").read_bytes() == (ctx2.dir_figuras / f"{i}.png").read_bytes(), i


# ================================================================================================= bordes que fijan mutantes de la etapa B
def test_el_corte_incluye_una_union_cuya_altura_cae_exactamente_en_el_umbral():
    A = np.array([[1.0, 0.95, 0.75], [0.95, 1.0, 0.85], [0.75, 0.85, 1.0]])             # C se une a {A, B} a (0,25 + 0,15) / 2 = 0,2 en aritmética real
    e = asoc.agrupar_variables(A, 0.8)
    assert e["Z"][1, 2] == pytest.approx(0.2, abs=1e-12) and e["Z"][1, 2] > 1.0 - 0.8       # pero el corte plano 1 - 0,8 vale 0,19999999999999996: sin la holgura de 1e-12 se separaría
    assert len(set(e["grupo"].tolist())) == 1 and len(set(e["cluster"].tolist())) == 1


def test_el_resumen_excluye_el_par_de_2023_que_la_tabla_publica_anula_aunque_su_valor_exista(contexto, panel):
    nac, anio = panel.datos["NACIONALIDAD"].to_numpy(), panel.datos["anio"].to_numpy()
    rank = np.flatnonzero(anio <= 2022)
    conf = np.flatnonzero(anio == 2023)
    elegidas = set([i for i in rank if nac[i] == "CHILENA"][:20] + [i for i in rank if nac[i] != "CHILENA"][:20]
                   + [i for i in conf if nac[i] == "CHILENA"][:3] + [i for i in conf if nac[i] != "CHILENA"][:3])          # (40, 6): ambas clases en las seis filas de 2023
    nuevo = [(0 if nac[i] == "CHILENA" else 1) if i in elegidas else None for i in range(panel.datos.height)]                   # el panel define ES_EXTRANJERO desde NACIONALIDAD
    r = asoc.calcular(contexto, dataclasses.replace(panel, datos=panel.datos.with_columns(pl.Series("ES_EXTRANJERO", nuevo, dtype=pl.Int8))))
    f = r["pares"].filter((pl.col("variable_a") == "ES_EXTRANJERO") & (pl.col("variable_b") == "NACIONALIDAD")).row(0, named=True)
    assert f["n_pares_2023"] == 6 and f["asociacion_2023"] is not None and f["se_mantiene_2023"] == "si"                       # el valor real de 2023 existe...
    res = r["resumen"].row(0, named=True)
    assert res["n_pares_redundantes_evaluados_2023"] == r["pares"].height - 1 == 10 and res["n_pares_redundantes_estables_2023"] == 10    # ...pero la tabla lo anula (n entre 1 y 9) y el resumen no lo cuenta


def test_se_mantiene_2023_compara_con_igualdad_cuando_la_asociacion_de_2023_vale_exactamente_0_80(contexto, panel):
    anio = panel.datos["anio"].to_numpy()
    n, n_conf = panel.datos.height, int((anio == 2023).sum())
    rng = np.random.default_rng(81)
    a = rng.integers(0, 2, n)
    b = np.where(rng.random(n) < 0.96, a, 1 - a)                                            # asociación fuerte (cercana a 0,92) en la ventana de ranking
    pos = np.flatnonzero(anio == 2023)
    a[pos] = np.repeat([0, 1], n_conf // 2)
    b[pos] = a[pos]
    voltear = np.concatenate([pos[: n_conf // 2][:12], pos[n_conf // 2:][:12]])              # tabla (108, 12, 12, 108) en 2023: phi = (108 * 108 - 12 * 12) / (120 * 120) = 0,8
    b[voltear] = 1 - b[voltear]
    r = asoc.calcular(contexto, _con_variables(panel, {"BIN_C": ("bin", a), "BIN_D": ("bin", b)}))
    f = r["pares"].filter((pl.col("variable_a") == "BIN_C") & (pl.col("variable_b") == "BIN_D")).row(0, named=True)
    assert f["asociacion_abs"] > 0.8 and f["n_pares_2023"] == n_conf == 240
    assert est.redondear_publicado(f["asociacion_2023"]) == 0.8 and f["se_mantiene_2023"] == "si"                         # el borde con igualdad


def test_dos_binarias_con_solo_dos_filas_validas_en_comun_dan_sin_datos_suficientes():
    a = np.array([0, 1, 0, 1, -1, -1])
    b = np.array([-1, -1, 0, 1, 0, 1])                                                       # solo las filas 3 y 4 son válidas en ambas, y ahí phi valdría 1
    p = asoc.medida_par(col("a", "bin", a), col("b", "bin", b))
    assert p.n == 2 and np.isnan(p.valor) and p.estado == "sin datos suficientes"
    assert asoc.matriz_de_medidas({"a": col("a", "bin", a), "b": col("b", "bin", b)})["estado"][0, 1] == "sin datos suficientes"


def test_la_referencia_nunca_entra_al_vif_aunque_fuera_numerica(u):
    rng = np.random.default_rng(71)
    cols = {n: col(n, "num", rng.normal(size=50)) for n in ("X", "Y", "REF")}
    v = asoc.calcular_vif(cols, ["REF"], u)
    assert set(v["variable"]) == {"X", "Y"} and v.height == 4                                # dos variables en cada uno de los dos conjuntos
    assert set(asoc.calcular_vif(cols, [], u)["variable"]) == {"X", "Y", "REF"}


def test_phi_elix_disponible_exige_filas_de_la_ventana_de_ranking(contexto, tmp_path):
    ctx = _ctx_nuevo(contexto, tmp_path, "phi_solo_2023")
    assert asoc._phi_elix_disponible(ctx) == "no"                                            # sin el archivo de la Tarea 8
    _tabla_phi().filter(pl.col("periodo") == "2023").write_csv(ctx.dir_tablas / "t23_elix_phi.csv")
    assert asoc._phi_elix_disponible(ctx) == "no" and asoc.figura_phi_elix(ctx) is False     # el archivo existe pero no trae el período de 2019-2022
    _tabla_phi().write_csv(ctx.dir_tablas / "t23_elix_phi.csv")
    assert asoc._phi_elix_disponible(ctx) == "si"


# FIN-test_t10_asociacion
