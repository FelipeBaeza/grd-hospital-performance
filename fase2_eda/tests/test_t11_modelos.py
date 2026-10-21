"""Tarea 11, ``eda/modelos.py``: diseños, ajustes logísticos, referencias, incremento de AUC, modelo L1, permutación y AUC univariado.

Las cuentas a mano y los oráculos (scikit-learn, scipy, numpy) fijan cada pieza; el panel artificial aporta la verdad de diseño. Las pruebas numéricas leen los resultados
en memoria. Cubre los mutantes M30, M37, M43 y M47.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import polars as pl
import pytest
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from eda import decision, elixhauser, modelos
from eda import estadistica as est
from eda import panel as pn
from panel_artificial import panel_artificial

SEMILLA = 20261007


@pytest.fixture(scope="module")
def panel():
    return panel_artificial()


@pytest.fixture(scope="module")
def base(panel):
    return modelos.ajuste_base(panel, SEMILLA)


def _con_columna(panel, nombre, valores, dtype=None):
    """Copia del panel con una columna reemplazada (``dataclasses.replace`` crea una caché de códigos nueva)."""
    return dataclasses.replace(panel, datos=panel.datos.with_columns(pl.Series(nombre, list(valores), dtype=dtype or panel.datos.schema[nombre])))


def _auc_oraculo(X, y, X2, y2, ridge):
    lr = LogisticRegression(C=1.0 / ridge, tol=1e-10, max_iter=2000).fit(X, y)
    return float(roc_auc_score(y2, lr.decision_function(X2)))


# ================================================================================================================ diseño
def test_espec_diseno_tiene_todos_los_campos_del_contrato(panel):
    espec = modelos.especificar_diseno(panel, ["EDAD_ANIOS", "SEXO", "ELIX_01"], panel.anios_ranking, SEMILLA)
    assert [f.name for f in dataclasses.fields(espec)] == [
        "variables", "medias", "desvios", "medianas", "tiene_nulos", "niveles", "nivel_base", "columnas", "filas_entrenamiento",
    ]
    assert espec.variables == ("EDAD_ANIOS", "ELIX_01", "SEXO") and espec.filas_entrenamiento is None         # el orden del registro, no el pedido
    assert espec.columnas[:2] == ("EDAD_ANIOS", "ELIX_01") and "SEXO=HOMBRE" in espec.columnas
    with pytest.raises(dataclasses.FrozenInstanceError):
        espec.columnas = ()                                                           # inmutable


def test_resultado_l1_tiene_los_campos_del_contrato_y_convergio():
    assert [f.name for f in dataclasses.fields(modelos.ResultadoL1)] == [
        "coef", "intercepto", "c_elegido", "columnas", "grupos", "sobrevive_por_variable", "estado", "convergio",
    ]


def test_la_numerica_se_estandariza_con_el_entrenamiento_tambien_en_2023(panel):
    espec = modelos.especificar_diseno(panel, ["EDAD_ANIOS"], panel.anios_ranking, SEMILLA)
    edad_rank = panel.ventana(panel.anios_ranking)["EDAD_ANIOS"].to_numpy()
    assert espec.medias["EDAD_ANIOS"] == pytest.approx(edad_rank.mean()) and espec.desvios["EDAD_ANIOS"] == pytest.approx(edad_rank.std())      # ddof=0
    X, y, grupos = modelos.aplicar_diseno(espec, panel, panel.anios_ranking)
    z = X[:, 0].toarray().ravel()
    assert z.mean() == pytest.approx(0.0, abs=1e-12) and z.std() == pytest.approx(1.0)
    X23, _, _ = modelos.aplicar_diseno(espec, panel, [2023])
    edad23 = panel.ventana([2023])["EDAD_ANIOS"].to_numpy()
    assert X23[:, 0].toarray().ravel() == pytest.approx((edad23 - edad_rank.mean()) / edad_rank.std())          # con la media y la desviación del ranking, no con las de 2023
    assert isinstance(X, sparse.csr_matrix) and X.dtype == np.float64 and y.dtype == np.int8 and grupos == {"EDAD_ANIOS": [0]}


def test_el_cuadrado_de_la_edad_se_estandariza_por_separado(panel):
    espec = modelos.especificar_diseno(panel, modelos.BASE_EDAD_SEXO, panel.anios_ranking, SEMILLA)
    edad = panel.ventana(panel.anios_ranking)["EDAD_ANIOS"].to_numpy()
    assert espec.medias["EDAD_ANIOS^2"] == pytest.approx((edad**2).mean()) and espec.desvios["EDAD_ANIOS^2"] == pytest.approx((edad**2).std())
    assert espec.variables == ("EDAD_ANIOS", "EDAD_ANIOS^2", "SEXO")
    X, _, g = modelos.aplicar_diseno(espec, panel, panel.anios_ranking)
    assert X[:, 1].toarray().ravel() == pytest.approx((edad**2 - (edad**2).mean()) / (edad**2).std())
    assert g["EDAD_ANIOS^2"] == [1] and g["SEXO"] == [2]


def test_nulos_solo_hay_indicador_si_el_entrenamiento_los_tuvo_y_el_relleno_usa_solo_el_entrenamiento(panel):
    anio = panel.datos["anio"].to_numpy()
    dx = panel.datos["N_DX_SEC"].to_numpy().astype(float)
    solo_2023 = dx.copy()
    solo_2023[np.flatnonzero(anio == 2023)[:7]] = np.nan                                    # nulos únicamente en 2023
    p1 = _con_columna(panel, "N_DX_SEC", [None if np.isnan(v) else v for v in solo_2023])
    e1 = modelos.especificar_diseno(p1, ["N_DX_SEC"], p1.anios_ranking, SEMILLA)
    assert e1.tiene_nulos["N_DX_SEC"] is False and e1.columnas == ("N_DX_SEC",)            # el entrenamiento no tuvo nulos: no hay columna indicadora
    X1, _, _ = modelos.aplicar_diseno(e1, p1, [2023])
    media_rank, sd_rank = e1.medias["N_DX_SEC"], e1.desvios["N_DX_SEC"]
    nulos = np.flatnonzero(np.isnan(solo_2023[anio == 2023]))
    mediana_rank = float(np.median(dx[anio <= 2022]))
    assert X1[nulos, 0].toarray().ravel() == pytest.approx(np.full(nulos.size, (mediana_rank - media_rank) / sd_rank))      # relleno = mediana del ENTRENAMIENTO
    con_nulos = dx.copy()
    con_nulos[np.flatnonzero(anio <= 2022)[:30]] = np.nan                                   # nulos en el entrenamiento
    p2 = _con_columna(panel, "N_DX_SEC", [None if np.isnan(v) else v for v in con_nulos])
    e2 = modelos.especificar_diseno(p2, ["N_DX_SEC"], p2.anios_ranking, SEMILLA)
    assert e2.tiene_nulos["N_DX_SEC"] is True and e2.columnas == ("N_DX_SEC", "N_DX_SEC__nulo")
    X2, _, g2 = modelos.aplicar_diseno(e2, p2, p2.anios_ranking)
    assert int(X2[:, 1].sum()) == 30 and g2["N_DX_SEC"] == [0, 1]                           # el grupo incluye la columna de nulos
    assert p2.datos["N_DX_SEC"].null_count() == 30                                          # el panel conserva los nulos (R5)


def test_binaria_nula_se_rellena_con_cero_y_agrega_el_indicador(panel):
    col = panel.datos["ELIX_01"].to_list()
    col[0], col[1] = None, None
    p = _con_columna(panel, "ELIX_01", col)
    espec = modelos.especificar_diseno(p, ["ELIX_01"], p.anios_ranking, SEMILLA)
    assert espec.columnas == ("ELIX_01", "ELIX_01__nulo") and espec.medianas["ELIX_01"] == 0.0
    X, _, _ = modelos.aplicar_diseno(espec, p, p.anios_ranking)
    assert X[0, 0] == 0.0 and X[0, 1] == 1.0 and X[1, 1] == 1.0


def test_omite_el_nivel_mas_frecuente_y_lo_guarda_como_nivel_base(panel):
    espec = modelos.especificar_diseno(panel, ["TIPO_PROCEDENCIA"], panel.anios_ranking, SEMILLA, omitir_nivel_base=True)
    codigos, niveles = panel.codigos("TIPO_PROCEDENCIA")
    mascara = panel.datos["anio"].is_in(panel.anios_ranking).to_numpy()
    mas_frecuente = niveles[int(np.argmax(np.bincount(codigos[mascara], minlength=len(niveles))))]
    assert espec.nivel_base["TIPO_PROCEDENCIA"] == mas_frecuente
    assert f"TIPO_PROCEDENCIA={mas_frecuente}" not in espec.columnas and len(espec.columnas) == len(niveles) - 1
    completo = modelos.especificar_diseno(panel, ["TIPO_PROCEDENCIA"], panel.anios_ranking, SEMILLA, omitir_nivel_base=False)
    assert completo.nivel_base["TIPO_PROCEDENCIA"] == "" and len(completo.columnas) == len(niveles)           # el L1 deja todos los niveles


def test_el_nivel_base_de_sexo_es_mujer_aunque_hombre_sea_el_mas_frecuente(panel):
    sexo = np.where(np.arange(panel.datos.height) % 10 < 7, "HOMBRE", "MUJER").tolist()
    p = _con_columna(panel, "SEXO", sexo)
    espec = modelos.especificar_diseno(p, ["SEXO"], p.anios_ranking, SEMILLA)
    assert espec.nivel_base["SEXO"] == "MUJER" and "SEXO=HOMBRE" in espec.columnas and "SEXO=MUJER" not in espec.columnas
    otra = modelos.especificar_diseno(p, ["TIPO_PROCEDENCIA"], p.anios_ranking, SEMILLA)
    assert otra.nivel_base["TIPO_PROCEDENCIA"] != ""                                          # la excepción es solo de SEXO


def test_las_columnas_constantes_se_omiten(panel):
    espec = modelos.especificar_diseno(panel, ["ELIX_28", "ELIX_01"], panel.anios_ranking, SEMILLA)
    assert espec.columnas == ("ELIX_01",) and "ELIX_28" in espec.variables
    X, _, grupos = modelos.aplicar_diseno(espec, panel, panel.anios_ranking)
    assert grupos == {"ELIX_01": [0], "ELIX_28": []} and X.shape[1] == 1
    const = _con_columna(panel, "EDAD_ANIOS", [50.0] * panel.datos.height, pl.Float64)
    assert modelos.especificar_diseno(const, ["EDAD_ANIOS"], const.anios_ranking, SEMILLA).columnas == ()           # numérica con desviación cero


def test_aplicar_diseno_da_los_mismos_coeficientes_que_la_version_densa(panel):
    espec = modelos.especificar_diseno(panel, ["EDAD_ANIOS", "SEXO", "ELIX_01", "TIPO_PROCEDENCIA"], panel.anios_ranking, SEMILLA)
    X, y, _ = modelos.aplicar_diseno(espec, panel, panel.anios_ranking)
    densa = est.regresion_logistica(X.toarray(), y, ridge=modelos.RIDGE_AUC)
    dispersa = est.regresion_logistica(X, y, ridge=modelos.RIDGE_AUC)
    assert dispersa.coef == pytest.approx(densa.coef, abs=1e-10)


def test_una_variable_inexistente_o_sin_columna_falla_con_un_mensaje_claro(panel):
    with pytest.raises(KeyError, match="NO_EXISTE"):
        modelos.especificar_diseno(panel, ["NO_EXISTE"], panel.anios_ranking, SEMILLA)
    sin_historial = pn.registro_variables(False)
    assert "N_EGRESOS_12M" in sin_historial and "N_EGRESOS_12M" not in panel.datos.columns
    with pytest.raises(ValueError, match="no tiene columna"):
        modelos.especificar_diseno(panel, ["N_EGRESOS_12M"], panel.anios_ranking, SEMILLA)
    with pytest.raises(ValueError, match="al cuadrado"):
        modelos.especificar_diseno(panel, ["SEXO^2"], panel.anios_ranking, SEMILLA)


# ======================================================================================================== muestra de entrenamiento (M47)
def test_indices_entrenamiento_es_la_identidad_si_cabe_y_el_oraculo_si_no(monkeypatch):
    assert np.array_equal(modelos.indices_entrenamiento(500, 7), np.arange(500))
    monkeypatch.setattr(modelos, "TOPE_FILAS_MODELOS", 100)
    assert np.array_equal(modelos.indices_entrenamiento(100, 7), np.arange(100))                 # igualdad con el tope: cabe
    oraculo = np.sort(np.random.default_rng(7 + 7).choice(250, size=100, replace=False))
    assert np.array_equal(modelos.indices_entrenamiento(250, 7), oraculo)
    assert not np.array_equal(modelos.indices_entrenamiento(250, 8), oraculo)                      # otra semilla, otra muestra
    assert np.array_equal(modelos.indices_entrenamiento(250, 7), modelos.indices_entrenamiento(250, 7))


def test_diseno_referencias_e_incremento_usan_la_muestra_de_entrenamiento(panel, monkeypatch):
    monkeypatch.setattr(modelos, "TOPE_FILAS_MODELOS", 400)
    n_rank = int(panel.datos["anio"].is_in(panel.anios_ranking).sum())
    oraculo = np.sort(np.random.default_rng(SEMILLA + 7).choice(n_rank, size=400, replace=False))
    espec = modelos.especificar_diseno(panel, modelos.BASE_EDAD_SEXO, panel.anios_ranking, SEMILLA)
    assert np.array_equal(espec.filas_entrenamiento, oraculo)
    otra = modelos.especificar_diseno(panel, modelos.BASE_EDAD_SEXO, panel.anios_ranking, SEMILLA + 1)
    assert not np.array_equal(otra.filas_entrenamiento, oraculo)
    X, y, _ = modelos.aplicar_diseno(espec, panel, panel.anios_ranking, entrenamiento=True)
    assert X.shape[0] == 400
    y_oraculo = panel.ventana(panel.anios_ranking)["y"].to_numpy()[oraculo]
    assert np.array_equal(y, y_oraculo)                                                             # las filas son exactamente las del oráculo
    Xt, _, _ = modelos.aplicar_diseno(espec, panel, panel.anios_ranking)
    assert Xt.shape[0] == n_rank                                                                    # sin entrenamiento=True se usa toda la ventana
    # las medias del diseño se aprenden con la muestra
    edad = panel.ventana(panel.anios_ranking)["EDAD_ANIOS"].to_numpy()[oraculo]
    assert espec.medias["EDAD_ANIOS"] == pytest.approx(edad.mean())
    ref = modelos.referencias(panel, SEMILLA)
    assert {v["n_entrenamiento"] for v in ref.values()} == {400}
    inc = modelos.incremento_auc(panel, "ELIX_01", SEMILLA)
    assert inc["n_entrenamiento"] == 400
    # la misma semilla da la misma muestra y otra semilla da otra
    otra_ref = modelos.referencias(panel, SEMILLA + 1)
    assert ref["MODELO_EDAD_SEXO"]["auc_ranking"] != otra_ref["MODELO_EDAD_SEXO"]["auc_ranking"]
    assert modelos.referencias(panel, SEMILLA) == ref


# ================================================================================================================ incremento (M43)
def test_incremento_ruido_cerca_de_cero_y_positivo_para_la_variable_que_genera_y(panel, base):
    ruido = modelos.incremento_auc(panel, "ELIX_03", SEMILLA, base=base)
    fuerte = modelos.incremento_auc(panel, "NIVEL_CUIDADO_INGRESO", SEMILLA, base=base)
    assert ruido["incremento_estado"] == "ok" and abs(ruido["incremento_auc"]) < 0.02
    assert fuerte["incremento_estado"] == "ok" and fuerte["incremento_auc"] > 0.05
    assert fuerte["incremento_auc"] == pytest.approx(fuerte["auc_con_variable_2023"] - fuerte["auc_base_2023"], abs=1e-15)
    assert fuerte["auc_base_2023"] == pytest.approx(base["auc_2023"], abs=1e-15) and fuerte["n_columnas_diseno"] == 1
    assert (fuerte["n_entrenamiento"], fuerte["n_2023"]) == (960, 240) and fuerte["n_eventos_2023"] == 30


def test_incremento_no_aplica_a_edad_ni_sexo_y_existe_para_variables_que_la_fase1_excluye(panel, base):
    for v in ("EDAD_ANIOS", "SEXO"):
        r = modelos.incremento_auc(panel, v, SEMILLA, base=base)
        assert r["incremento_estado"] == "no aplica: variable del modelo base" and r["incremento_auc"] is None
    l1 = set(modelos.variables_modelo(panel))
    for v in ("N_DX_SEC", "SERVICIO_SALUD"):
        assert v not in l1                                                                       # ninguna de sus columnas está en el diseño del L1...
        r = modelos.incremento_auc(panel, v, SEMILLA, base=base)
        assert r["incremento_estado"] == "ok" and r["incremento_auc"] is not None               # ...pero su incremento existe


def test_incremento_variable_constante_o_pocos_eventos(panel, base):
    r = modelos.incremento_auc(panel, "ELIX_28", SEMILLA, base=base)
    assert r["incremento_estado"] == "constante en el entrenamiento" and r["incremento_auc"] is None and r["n_columnas_diseno"] == 0
    y = np.zeros(panel.datos.height, dtype=int)
    y[np.flatnonzero(panel.datos["anio"].to_numpy() <= 2022)[:2]] = 1                           # solo dos eventos en el entrenamiento
    p = _con_columna(panel, "y", y.tolist(), pl.Int8)
    assert modelos.incremento_auc(p, "ELIX_01", SEMILLA)["incremento_estado"] == "pocos eventos"


def test_incremento_sin_2023_es_no_evaluable(panel):
    sin = dataclasses.replace(panel, datos=panel.datos.filter(pl.col("anio") <= 2022), anio_confirmacion=None)
    r = modelos.incremento_auc(sin, "ELIX_01", SEMILLA)
    assert r["incremento_estado"] == "no evaluable en 2023" and r["incremento_auc"] is None and r["n_2023"] == 0


def _panel_con_categorica_de_60_niveles(panel, nivel_sin_eventos=150):
    """Copia del panel con ``CAT60``: 60 niveles, uno de ellos (``CAT60_59``) con ``nivel_sin_eventos`` filas del ranking y ningún evento."""
    n = panel.datos.height
    y, anio = panel.datos["y"].to_numpy(), panel.datos["anio"].to_numpy()
    rng = np.random.default_rng(5)
    codigo = rng.integers(0, 59, size=n)
    sin_eventos = np.flatnonzero((anio <= 2022) & (y == 0))[:nivel_sin_eventos]
    codigo[sin_eventos] = 59
    codigo[(y == 1)] = np.where(codigo[(y == 1)] == 59, 0, codigo[(y == 1)])
    niveles = [f"CAT60_{i:02d}" for i in range(60)]
    datos = panel.datos.with_columns(pl.Series("CAT60", [niveles[c] for c in codigo], dtype=pl.String))
    variables = dict(panel.variables)
    variables["CAT60"] = pn.Variable(nombre="CAT60", tipo="cat", grupo="prueba", rol_fase1="FEATURE_BASAL", momento="basal", evaluable="si", uso="candidata", fuentes=())
    return dataclasses.replace(panel, datos=datos, variables=variables, niveles={**panel.niveles, "CAT60": niveles})


def test_una_categorica_de_60_niveles_con_un_nivel_separado_tiene_incremento_igual_al_del_oraculo(panel):
    p = _panel_con_categorica_de_60_niveles(panel)
    r = modelos.incremento_auc(p, "CAT60", SEMILLA)
    assert r["incremento_estado"] == "ok" and r["incremento_auc"] is not None and r["n_columnas_diseno"] >= 55
    espec = modelos.especificar_diseno(p, (*modelos.BASE_EDAD_SEXO, "CAT60"), p.anios_ranking, SEMILLA)
    X, y, _ = modelos.aplicar_diseno(espec, p, p.anios_ranking, entrenamiento=True)
    X2, y2, _ = modelos.aplicar_diseno(espec, p, [2023])
    auc_con = _auc_oraculo(X, y, X2, y2, modelos.RIDGE_AUC)
    eb = modelos.especificar_diseno(p, modelos.BASE_EDAD_SEXO, p.anios_ranking, SEMILLA)
    Xb, yb, _ = modelos.aplicar_diseno(eb, p, p.anios_ranking, entrenamiento=True)
    Xb2, yb2, _ = modelos.aplicar_diseno(eb, p, [2023])
    auc_base = _auc_oraculo(Xb, yb, Xb2, yb2, modelos.RIDGE_AUC)
    assert abs(r["incremento_auc"] - (auc_con - auc_base)) < 1e-6                                 # la prueba no mira la bandera ``separacion``: no interviene
    assert est.regresion_logistica(X, y, ridge=modelos.RIDGE_WALD).separacion                      # y sin ridge sí habría separación: el diseño es el de la advertencia


def test_con_max_iter_forzado_a_uno_el_estado_es_no_convergio_y_el_incremento_nulo(panel, monkeypatch):
    original = est.regresion_logistica
    monkeypatch.setattr(est, "regresion_logistica", lambda X, y, **kw: original(X, y, **{**kw, "max_iter": 1}))
    r = modelos.incremento_auc(panel, "NIVEL_CUIDADO_INGRESO", SEMILLA)
    assert r["incremento_estado"] == "no convergió" and r["incremento_auc"] is None and r["auc_con_variable_2023"] is None


def test_la_separacion_no_anula_el_incremento_aunque_la_bandera_este_activa(panel, base, monkeypatch):
    """Mutante M43: si el incremento mirara ``separacion`` quedaría nulo con una columna separada."""
    p = _panel_con_categorica_de_60_niveles(panel)
    original = est.regresion_logistica

    def con_bandera(X, y, **kw):
        r = original(X, y, **kw)
        return dataclasses.replace(r, separacion=True, separacion_por_coef=np.ones_like(r.separacion_por_coef, dtype=bool))

    monkeypatch.setattr(est, "regresion_logistica", con_bandera)
    r = modelos.incremento_auc(p, "CAT60", SEMILLA)
    assert r["incremento_estado"] == "ok" and r["incremento_auc"] is not None


# ============================================================================================================== referencias (M37)
def test_modelo_elix31_coincide_con_el_oraculo_solo_con_las_31_categorias_y_sin_edad_ni_sexo(panel):
    ref = modelos.referencias(panel, SEMILLA)["MODELO_ELIX31"]
    espec = modelos.especificar_diseno(panel, list(elixhauser.IDS), panel.anios_ranking, SEMILLA)
    assert len(espec.columnas) == 30 and all(c.startswith("ELIX_") for c in espec.columnas) and "ELIX_28" not in espec.columnas      # sin edad, sin sexo, sin la constante
    X, y, _ = modelos.aplicar_diseno(espec, panel, panel.anios_ranking, entrenamiento=True)
    X2, y2, _ = modelos.aplicar_diseno(espec, panel, [2023])
    assert ref["auc_2023"] == pytest.approx(_auc_oraculo(X, y, X2, y2, modelos.RIDGE_AUC), abs=1e-6)
    assert ref["estado"] == "ok" and ref["n_columnas"] == 30
    con_edad = modelos.especificar_diseno(panel, [*elixhauser.IDS, "EDAD_ANIOS", "SEXO"], panel.anios_ranking, SEMILLA)
    Xe, ye, _ = modelos.aplicar_diseno(con_edad, panel, panel.anios_ranking, entrenamiento=True)
    Xe2, ye2, _ = modelos.aplicar_diseno(con_edad, panel, [2023])
    assert abs(_auc_oraculo(Xe, ye, Xe2, ye2, modelos.RIDGE_AUC) - ref["auc_2023"]) > 0.02          # agregar edad y sexo cambia el AUC: la composición queda fijada


def test_referencias_con_y_que_depende_solo_de_la_edad_o_solo_de_las_categorias():
    solo_edad = modelos.referencias(panel_artificial(y_modo="solo_edad"), SEMILLA)
    assert solo_edad["MODELO_EDAD_SEXO"]["auc_2023"] > solo_edad["MODELO_ELIX31"]["auc_2023"] + 0.10 and abs(solo_edad["MODELO_ELIX31"]["auc_2023"] - 0.5) < 0.10
    solo_elix = modelos.referencias(panel_artificial(y_modo="solo_elix"), SEMILLA)
    assert solo_elix["MODELO_ELIX31"]["auc_2023"] > solo_elix["MODELO_EDAD_SEXO"]["auc_2023"] + 0.10


def test_modelo_edad_sexo_coincide_con_el_oraculo_y_con_el_ajuste_base(panel, base):
    ref = modelos.referencias(panel, SEMILLA)["MODELO_EDAD_SEXO"]
    espec = base["espec"]
    assert espec.columnas == ("EDAD_ANIOS", "EDAD_ANIOS^2", "SEXO=HOMBRE", "SEXO=(sin dato)") or espec.columnas[:3] == ("EDAD_ANIOS", "EDAD_ANIOS^2", "SEXO=HOMBRE")
    X, y, _ = modelos.aplicar_diseno(espec, panel, panel.anios_ranking, entrenamiento=True)
    X2, y2, _ = modelos.aplicar_diseno(espec, panel, [2023])
    assert ref["auc_2023"] == pytest.approx(_auc_oraculo(X, y, X2, y2, modelos.RIDGE_AUC), abs=1e-6) == pytest.approx(base["auc_2023"], abs=1e-6)
    assert ref["auc_ranking"] > ref["auc_2023"] - 0.2 and 0.5 < ref["auc_ranking"] < 1.0


# ===================================================================================================================== L1
def _diseno_l1(panel):
    espec = modelos.especificar_diseno(panel, modelos.variables_modelo(panel), panel.anios_ranking, SEMILLA, omitir_nivel_base=False)
    X, y, grupos = modelos.aplicar_diseno(espec, panel, panel.anios_ranking, entrenamiento=True)
    return espec, X, y, grupos


@pytest.fixture(scope="module")
def l1(panel):
    espec, X, y, grupos = _diseno_l1(panel)
    return espec, X, y, grupos, modelos.ajustar_l1(X, y, espec.columnas, grupos, SEMILLA)


def test_l1_es_determinista_y_sobreviven_las_variables_con_senal_y_no_el_ruido(l1):
    espec, X, y, grupos, r = l1
    otra = modelos.ajustar_l1(X, y, espec.columnas, grupos, SEMILLA)
    assert np.array_equal(r.coef, otra.coef) and r.c_elegido == otra.c_elegido and r.intercepto == otra.intercepto
    assert r.estado == "ok" and r.convergio is True and r.columnas == espec.columnas
    assert r.sobrevive_por_variable["EDAD_ANIOS"] and r.sobrevive_por_variable["NIVEL_CUIDADO_INGRESO"]
    ruido = [f"ELIX_{i:02d}" for i in (5, 6, 8, 10, 11, 12, 13, 14, 15, 19, 20, 23, 27, 30)]
    assert sum(r.sobrevive_por_variable[v] for v in ruido) <= 3                                      # el ruido casi nunca sobrevive
    assert r.c_elegido in set(modelos.CS_L1.tolist()) or any(abs(r.c_elegido - c) < 1e-12 for c in modelos.CS_L1)
    assert all(r.sobrevive_por_variable[v] == bool(np.any(r.coef[g] != 0)) for v, g in grupos.items() if g)


def test_l1_coincide_con_la_forma_literal_del_encargo_con_el_mismo_splitter(l1, panel):
    espec, X, y, grupos, r = l1
    cv = StratifiedKFold(modelos.PLIEGUES_CV, shuffle=True, random_state=SEMILLA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)               # la forma literal está deprecada: es la prueba que avisaría si una versión nueva de scikit-learn cambiara la equivalencia
        warnings.simplefilter("ignore", ConvergenceWarning)
        literal = LogisticRegressionCV(Cs=modelos.CS_L1, penalty="l1", solver="liblinear", scoring="roc_auc", cv=cv, random_state=SEMILLA, n_jobs=1, max_iter=200, refit=True).fit(X, y)
    assert np.allclose(r.coef, literal.coef_.ravel(), atol=1e-12) and r.c_elegido == pytest.approx(float(literal.C_[0]), rel=1e-12)
    assert r.intercepto == pytest.approx(float(literal.intercept_[0]), abs=1e-12)


def test_ajustar_l1_no_deja_escapar_avisos_de_scikit_learn(l1):
    espec, X, y, grupos, _ = l1
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                        # cualquier FutureWarning o ConvergenceWarning que escapara sería una excepción
        modelos.ajustar_l1(X, y, espec.columnas, grupos, SEMILLA)
        r = modelos.ajustar_l1(X, y, espec.columnas, grupos, SEMILLA, max_iter=1)
    assert r.convergio is False                                               # con una sola iteración el liblinear no puede converger


def test_l1_con_pocos_eventos_no_es_fatal(l1):
    espec, X, y, grupos, _ = l1
    chico = np.zeros(y.size, dtype=np.int8)
    chico[:5] = 1
    r = modelos.ajustar_l1(X, chico, espec.columnas, grupos, SEMILLA)
    assert r.estado == "no evaluable: pocos eventos" and not any(r.sobrevive_por_variable.values()) and np.isnan(r.coef).all() and np.isnan(r.c_elegido)
    assert modelos.auc_l1(r, X, chico) is None
    assert modelos.importancia_permutacion(X, chico, r, grupos, semilla=SEMILLA).height == 0


def test_los_modelos_multivariados_no_usan_el_hospital_ni_la_fuga_ni_roles_excluyentes(panel, l1):
    espec, *_ = l1
    permitidas = modelos.variables_modelo(panel)
    for v in ("COD_HOSPITAL", "IR_29301_MORTALIDAD", "IR_29301_SEVERIDAD", "ESTANCIA_DIAS", "ETNIA", "PUEBLO_ORIGINARIO"):
        assert v not in permitidas
    assert all(panel.variables[v].uso == "candidata" and panel.variables[v].rol_fase1 not in decision.ROLES_EXCLUYENTES for v in permitidas)
    assert not any(c.split("=")[0].split("__")[0] in {"COD_HOSPITAL", "IR_29301_MORTALIDAD", "N_DX_SEC", "SERVICIO_SALUD"} for c in espec.columnas)
    assert "ESPECIALIDAD_MEDICA" in permitidas and "SERVICIOINGRESO" in permitidas                 # las CONDICIONALES sí entran (y se advierte en el informe)


# ================================================================================================================ permutación (M30)
def _oraculo_permutacion(X, y, modelo, grupos, repeticiones, semilla):
    """Permuta explícitamente las columnas del grupo con el mismo ``p`` y usa ``predict_proba`` de scikit-learn."""
    lr = LogisticRegression()
    lr.classes_ = np.array([0, 1])
    lr.coef_ = modelo.coef[None, :]
    lr.intercept_ = np.array([modelo.intercepto])
    lr.n_features_in_ = X.shape[1]
    # el orden de las predicciones es el del logit de scikit-learn (``predict_proba`` es la sigmoide del mismo valor; con ella las filas muy separadas se saturarían y
    # empatarían), redondeado igual que el módulo para que el ruido de 1e-16 no desempate
    puntaje = lambda M: np.round(lr.decision_function(sparse.csr_matrix(M)), modelos.DECIMALES_LOGIT)  # noqa: E731
    assert lr.predict_proba(X).shape == (y.size, 2)                                          # el modelo de scikit-learn está bien armado
    auc_base = roc_auc_score(y, puntaje(X))
    rng = np.random.default_rng(semilla + 11)
    Xd = X.toarray()
    salida = {}
    for nombre, cols in grupos.items():
        if not cols:
            continue
        aucs = []
        for _ in range(repeticiones):
            p = rng.permutation(y.size)
            Xp = Xd.copy()
            Xp[:, cols] = Xd[p][:, cols]
            aucs.append(roc_auc_score(y, puntaje(Xp)))
        salida[nombre] = (auc_base - float(np.mean(aucs)), float(np.std(aucs, ddof=1)))
    return auc_base, salida


def test_la_permutacion_es_exacta_e_igual_a_la_permutacion_explicita_con_predict_proba(panel, l1):
    espec, X, y, grupos, r = l1
    X2, y2, g2 = modelos.aplicar_diseno(espec, panel, [2023])
    imp = modelos.importancia_permutacion(X2, y2, r, g2, repeticiones=4, semilla=SEMILLA)
    auc_base, oraculo = _oraculo_permutacion(X2, y2, r, g2, 4, SEMILLA)
    assert set(imp["variable"]) == set(oraculo) and imp["auc_base"].to_list()[0] == pytest.approx(auc_base, abs=1e-12)
    for f in imp.iter_rows(named=True):
        assert f["importancia"] == pytest.approx(oraculo[f["variable"]][0], abs=1e-9), f["variable"]
        assert f["desviacion"] == pytest.approx(oraculo[f["variable"]][1], abs=1e-9)


def test_la_permutacion_con_semilla_fija_repite_y_otra_semilla_cambia(panel, l1):
    espec, X, y, grupos, r = l1
    X2, y2, g2 = modelos.aplicar_diseno(espec, panel, [2023])
    a = modelos.importancia_permutacion(X2, y2, r, g2, semilla=SEMILLA)
    b = modelos.importancia_permutacion(X2, y2, r, g2, semilla=SEMILLA)
    c = modelos.importancia_permutacion(X2, y2, r, g2, semilla=SEMILLA + 1)
    assert a.equals(b) and not a.equals(c)


def test_la_permutacion_del_ruido_es_cercana_a_cero_y_la_de_la_senal_es_grande(panel, l1):
    espec, X, y, grupos, r = l1
    X2, y2, g2 = modelos.aplicar_diseno(espec, panel, [2023])
    imp = {f["variable"]: f for f in modelos.importancia_permutacion(X2, y2, r, g2, semilla=SEMILLA).iter_rows(named=True)}
    assert imp["EDAD_ANIOS"]["importancia"] > 0.05 and imp["NIVEL_CUIDADO_INGRESO"]["importancia"] > 0.02
    sin_coef = [v for v, ok in r.sobrevive_por_variable.items() if not ok and g2[v]]
    assert sin_coef and all(imp[v]["importancia"] == 0.0 for v in sin_coef)                          # sin coeficientes el logit no cambia: caída exactamente cero
    assert imp["EDAD_ANIOS"]["desviacion"] > 0


def test_la_permutacion_valida_las_repeticiones_y_con_una_sola_deja_la_desviacion_nula(panel, l1):
    espec, X, y, grupos, r = l1
    X2, y2, g2 = modelos.aplicar_diseno(espec, panel, [2023])
    with pytest.raises(ValueError, match="repeticiones"):
        modelos.importancia_permutacion(X2, y2, r, g2, repeticiones=0)
    una = modelos.importancia_permutacion(X2, y2, r, g2, repeticiones=1, semilla=SEMILLA)
    assert una["desviacion"].null_count() == una.height


# ======================================================================================================= AUC univariado compartido
def test_efecto_invertido_da_direccion_menos_y_auc_de_al_menos_un_medio(panel):
    ajuste = modelos.ajustar_univariado(panel, "EDAD_ANIOS")
    assert ajuste.direccion == "+" and modelos.auc_univariado(panel, ajuste, panel.anios_ranking).auc > 0.6
    invertida = _con_columna(panel, "EDAD_ANIOS", (-panel.datos["EDAD_ANIOS"]).to_list())
    a2 = modelos.ajustar_univariado(invertida, "EDAD_ANIOS")
    r = modelos.auc_univariado(invertida, a2, invertida.anios_ranking)
    assert a2.direccion == "-" and r.auc >= 0.5 and r.auc == pytest.approx(1.0 - r.auc_bruto) and r.auc_bruto < 0.5
    assert r.ic_bajo <= r.auc <= r.ic_alto


def test_en_2023_la_direccion_es_la_del_ranking_y_el_auc_puede_quedar_bajo_un_medio(panel):
    ajuste = modelos.ajustar_univariado(panel, "ELIX_02")                                            # OR mayor que 1 en 2019-2022 y menor que 1 en 2023
    rank = modelos.auc_univariado(panel, ajuste, panel.anios_ranking)
    conf = modelos.auc_univariado(panel, ajuste, [2023])
    assert ajuste.direccion == "+" and rank.auc > 0.5 and conf.auc < 0.5                              # no se reorienta con 2023: el cambio de signo queda visible


def test_el_auc_univariado_numerico_coincide_con_scikit_learn_y_el_ic_con_hanley(panel):
    ajuste = modelos.ajustar_univariado(panel, "EDAD_ANIOS")
    r = modelos.auc_univariado(panel, ajuste, panel.anios_ranking)
    v = panel.ventana(panel.anios_ranking)
    assert r.auc == pytest.approx(roc_auc_score(v["y"].to_numpy(), v["EDAD_ANIOS"].to_numpy()), abs=1e-12)
    assert (r.n, r.n_eventos, r.n_no_eventos) == (v.height, int(v["y"].sum()), v.height - int(v["y"].sum()))
    assert (r.ic_bajo, r.ic_alto) == pytest.approx(est.ic_hanley_mcneil(r.auc, r.n_eventos, r.n_no_eventos))


def test_la_categorica_usa_las_tasas_de_la_ventana_de_ranking(panel):
    ajuste = modelos.ajustar_univariado(panel, "GRUPO_CLINICO")
    codigos, niveles = panel.codigos("GRUPO_CLINICO")
    m = panel.datos["anio"].is_in(panel.anios_ranking).to_numpy()
    tasas = est.puntaje_categorico(codigos[m], panel.datos["y"].to_numpy()[m].astype(float), len(niveles))
    assert ajuste.tasas == pytest.approx(tasas) and ajuste.niveles == tuple(niveles) and ajuste.tipo == "cat"
    conf = modelos.auc_univariado(panel, ajuste, [2023])
    m23 = panel.datos["anio"].to_numpy() == 2023
    assert conf.auc_bruto == pytest.approx(roc_auc_score(panel.datos["y"].to_numpy()[m23], tasas[codigos[m23]]), abs=1e-12)        # puntajes del ranking aplicados a 2023


def _panel_con_nivel_solo_de_2023(panel, otros_en_ranking: bool):
    """``NIV3``: niveles A, B y OTROS; OTROS solo aparece en 2023 (o también en el ranking). El panel recodifica a OTROS el nivel nuevo de 2023."""
    n = panel.datos.height
    anio, y = panel.datos["anio"].to_numpy(), panel.datos["y"].to_numpy()
    valores = np.where(np.arange(n) % 2 == 0, "A", "B").astype(object)
    valores[anio == 2023] = np.where(np.arange((anio == 2023).sum()) % 3 == 0, "OTROS", valores[anio == 2023])
    if otros_en_ranking:
        idx = np.flatnonzero(anio <= 2022)[:60]
        valores[idx] = "OTROS"
    datos = panel.datos.with_columns(pl.Series("NIV3", valores.tolist(), dtype=pl.String))
    variables = dict(panel.variables)
    variables["NIV3"] = pn.Variable(nombre="NIV3", tipo="cat", grupo="prueba", rol_fase1="FEATURE_BASAL", momento="basal", evaluable="si", uso="candidata", fuentes=())
    return dataclasses.replace(panel, datos=datos, variables=variables, niveles={**panel.niveles, "NIV3": ["A", "B", "OTROS"]})


def test_un_nivel_solo_de_2023_recibe_la_tasa_global_y_si_otros_existe_en_el_ranking_la_suya(panel):
    sin = _panel_con_nivel_solo_de_2023(panel, otros_en_ranking=False)
    a_sin = modelos.ajustar_univariado(sin, "NIV3")
    y = sin.ventana(sin.anios_ranking)["y"].to_numpy()
    assert a_sin.tasas[2] == pytest.approx(y.mean())                                                  # sin observaciones en el ranking: la tasa global (suavizada vale lo mismo)
    con = _panel_con_nivel_solo_de_2023(panel, otros_en_ranking=True)
    a_con = modelos.ajustar_univariado(con, "NIV3")
    c = con.ventana(con.anios_ranking)
    codigos, _ = con.codigos("NIV3")
    m = con.datos["anio"].is_in(con.anios_ranking).to_numpy()
    k, n = int(c["y"].to_numpy()[codigos[m] == 2].sum()), int((codigos[m] == 2).sum())
    p0 = float(c["y"].to_numpy().mean())
    esperada = (k + est.SUAVIZADO_M * p0) / (n + est.SUAVIZADO_M)                                     # la tasa suavizada del propio nivel OTROS
    assert n == 60 and a_con.tasas[2] == pytest.approx(esperada, rel=1e-12)
    assert esperada != pytest.approx(p0, rel=1e-6)                                                    # y distinta de la global: la prueba distingue los dos casos


def test_auc_cv_es_nulo_con_un_solo_ano_y_coincide_con_la_validacion_cruzada_a_mano(panel):
    ajuste = modelos.ajustar_univariado(panel, "GRUPO_CLINICO")
    cv = modelos.auc_cv_univariado(panel, ajuste)
    codigos, niveles = panel.codigos("GRUPO_CLINICO")
    m = panel.datos["anio"].is_in(panel.anios_ranking).to_numpy()
    cod, y, anio = codigos[m], panel.datos["y"].to_numpy()[m].astype(float), panel.datos["anio"].to_numpy()[m]
    puntaje = np.empty(cod.size)
    for a in (2019, 2020, 2021, 2022):
        tasas = est.puntaje_categorico(cod[anio != a], y[anio != a], len(niveles))
        puntaje[anio == a] = tasas[cod[anio == a]]
    assert cv.auc_bruto == pytest.approx(roc_auc_score(y, puntaje), abs=1e-12) and (cv.n_eventos, cv.n_no_eventos) == (int(y.sum()), int(y.size - y.sum()))
    assert cv.ic_bajo <= cv.auc <= cv.ic_alto and (cv.ic_bajo, cv.ic_alto) == pytest.approx(est.ic_hanley_mcneil(cv.auc, cv.n_eventos, cv.n_no_eventos))
    un_ano = dataclasses.replace(panel, datos=panel.datos.filter(pl.col("anio").is_in([2019, 2023])), anios_ranking=[2019])
    assert modelos.auc_cv_univariado(un_ano, modelos.ajustar_univariado(un_ano, "GRUPO_CLINICO")) is None
    assert modelos.auc_cv_univariado(panel, modelos.ajustar_univariado(panel, "EDAD_ANIOS")) is None          # solo las categóricas


def test_la_categorica_optimista_es_al_menos_la_de_validacion_cruzada(panel):
    for v in ("CIE10_3C", "GRUPO_CLINICO", "TIPO_PROCEDENCIA", "COMUNA"):
        ajuste = modelos.ajustar_univariado(panel, v)
        assert modelos.auc_univariado(panel, ajuste, panel.anios_ranking).auc >= modelos.auc_cv_univariado(panel, ajuste).auc - 1e-12


def test_estado_de_la_ventana(panel):
    assert modelos.estado_ventana(panel, "EDAD_ANIOS", panel.anios_ranking) == "ok"
    assert modelos.estado_ventana(panel, "ELIX_28", panel.anios_ranking) == "constante"
    assert modelos.estado_ventana(panel, "ES_EXTRANJERO", [2023]) == "sin ambos desenlaces"          # 4 filas informadas sin ambos desenlaces
    assert modelos.estado_ventana(panel, "EDAD_ANIOS", [2030]) == "no evaluable en la ventana"
    nulos = _con_columna(panel, "ES_EXTRANJERO", [None] * panel.datos.height)
    assert modelos.estado_ventana(nulos, "ES_EXTRANJERO", nulos.anios_ranking) == "no evaluable en la ventana"


def test_or_es_relevante_compara_con_los_valores_publicados():
    assert modelos.or_es_relevante(1.2, 1.1, 1.3, 1.2) == "si" and modelos.or_es_relevante(1.19, 1.1, 1.3, 1.2) == "no"
    assert modelos.or_es_relevante(1.1999996, 1.1, 1.3, 1.2) == "si"                                  # redondea a 1,2
    assert modelos.or_es_relevante(1.0 / 1.2, 0.7, 0.99, 1.2) == "si" and modelos.or_es_relevante(0.9, 0.7, 0.99, 1.2) == "no"
    assert modelos.or_es_relevante(1.5, 0.9, 2.5, 1.2) == "no"                                        # el IC incluye 1
    assert modelos.or_es_relevante(None, 1.0, 2.0, 1.2) is None and modelos.or_es_relevante(float("nan"), 1.0, 2.0, 1.2) is None


def test_las_constantes_tecnicas_tienen_los_valores_del_diseno():
    assert (modelos.RIDGE_AUC, modelos.RIDGE_WALD, modelos.TOPE_FILAS_MODELOS, modelos.PLIEGUES_CV, modelos.REPETICIONES_PERMUTACION) == (1.0, 0.0, None, 3, 10)
    assert modelos.CS_L1.tolist() == pytest.approx(np.logspace(-3, 1, 7).tolist()) and len(modelos.CS_L1) == 7


# FIN-test_t11_modelos


def test_l1_se_ajusta_con_muestra_fija_y_el_resto_usa_todas_las_filas(monkeypatch):
    """``TOPE_FILAS_L1``: el L1 se ajusta con una muestra reproducible; sin tope usa todas las filas; ``TOPE_FILAS_MODELOS`` por defecto no muestrea."""
    from scipy import sparse

    assert modelos.TOPE_FILAS_MODELOS is None and modelos.TOPE_FILAS_L1 == 400_000
    assert modelos.indices_entrenamiento(1_000_000, 5).size == 1_000_000
    rng = np.random.default_rng(0)
    n = 3000
    X = sparse.csr_matrix(rng.normal(size=(n, 4)))
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * X[:, 0].toarray().ravel() - 1.0)))).astype(np.int8)
    columnas = ("a", "b", "c", "d")
    grupos = {c: [i] for i, c in enumerate(columnas)}
    monkeypatch.setattr(modelos, "TOPE_FILAS_L1", 1000)
    r1 = modelos.ajustar_l1(X, y, columnas, grupos, semilla=3)
    r2 = modelos.ajustar_l1(X, y, columnas, grupos, semilla=3)
    assert r1.estado == "ok" and np.array_equal(r1.coef, r2.coef) and r1.c_elegido == r2.c_elegido          # reproducible
    assert r1.sobrevive_por_variable["a"]
    monkeypatch.setattr(modelos, "TOPE_FILAS_L1", None)
    r3 = modelos.ajustar_l1(X, y, columnas, grupos, semilla=3)
    assert r3.estado == "ok" and r3.sobrevive_por_variable["a"]
    assert modelos.auc_l1(r1, X, y) > 0.7                                                                       # se evalúa con todas las filas
