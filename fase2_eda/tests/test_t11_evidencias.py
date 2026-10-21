"""Tarea 11, evidencias: proxy del hospital y fuga (``eda/sospechas.py``) y Elixhauser (``eda/sensibilidad_elix.py``).

Las cuentas a mano y los oráculos (scipy, scikit-learn, numpy) fijan cada pieza; el panel artificial (``tests/panel_artificial.py``) aporta la verdad de diseño. Las pruebas
numéricas leen los resultados en memoria (sin suprimir); las de formato y privacidad leen los CSV. Cubre los mutantes M28, M38, M49 (proxy y fuga) y M33, M44, M53 (Elixhauser).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl
import pytest
from scipy import stats

from ayudas_figuras import vigilar_savefig
from eda import asociacion as asoc
from eda import config, modelos, supresion
from eda import estadistica as est
from eda import panel as pn
from eda import sospechas as sp
from eda.contexto import Contexto
from panel_artificial import panel_artificial
from sintetico import crear_fuente_sintetica

SEMILLA = 20261007


@pytest.fixture(scope="module")
def u():
    """Umbrales del proyecto. Las cuentas a mano de este archivo suponen los valores propuestos; si cambian, la prueba lo dice."""
    u = config.cargar_umbrales()
    supuestos = {
        "proxy_hospital_alto": 0.30, "proxy_hospital_moderado": 0.15, "auc_sospecha_fuga": 0.85, "or_relevante": 1.2, "celda_minima": 10, "winsor": 0.10,
    }
    for clave, valor in supuestos.items():
        assert u[clave] == valor, f"las cuentas a mano de este archivo suponen {clave} = {valor}"
    return u


@pytest.fixture(scope="module")
def contexto(tmp_path_factory, u):
    """Contexto sobre una fuente sintética mínima: las funciones de la Tarea 11 reciben el panel y nunca leen."""
    raiz = tmp_path_factory.mktemp("t11_evidencias")
    return Contexto.crear("extracto", fuente=crear_fuente_sintetica(raiz / "datos", n=60, semilla=29), dir_salida_base=raiz / "salidas")


@pytest.fixture(scope="module")
def panel():
    return panel_artificial()


def _ranking_de_juguete(panel):
    """Fila de 2019-2022 por variable del ranking con ``v_resultado``, ``medida_resultado`` y ``auc_senal``: lo que ``tabla_proxy`` y ``tabla_fuga`` leen de él."""
    cols = asoc.preparar_columnas(panel, panel.anios_ranking)
    y = modelos.desenlace(panel, panel.anios_ranking)
    filas = []
    for n in modelos.variables_ranking(panel):
        v, medida, _ = asoc.con_resultado(cols[n], y)
        a = modelos.ajustar_univariado(panel, n)
        filas.append({"variable": n, "ventana": pn.V_RANK, "v_resultado": v, "medida_resultado": medida, "auc_senal": modelos.auc_univariado(panel, a, panel.anios_ranking).auc})
    return pl.DataFrame(filas)


@pytest.fixture(scope="module")
def ranking(panel):
    return _ranking_de_juguete(panel)


@pytest.fixture(scope="module")
def referencias(panel):
    return modelos.referencias(panel, SEMILLA)


@pytest.fixture(scope="module")
def proxy(contexto, panel, ranking):
    return sp.tabla_proxy(contexto, panel, ranking)


@pytest.fixture(scope="module")
def fuga(contexto, panel, ranking, referencias):
    return sp.tabla_fuga(contexto, panel, ranking, referencias)


def _con_columna(panel, nombre, valores, dtype=None):
    return dataclasses.replace(panel, datos=panel.datos.with_columns(pl.Series(nombre, list(valores), dtype=dtype or panel.datos.schema[nombre])))


def _fila(tabla, **filtro):
    sel = tabla
    for k, v in filtro.items():
        sel = sel.filter(pl.col(k) == v)
    assert sel.height == 1, (filtro, sel.height)
    return sel.row(0, named=True)


# ================================================================================================================== proxy: clases
def test_las_columnas_del_proxy_y_de_la_fuga_son_las_del_contrato(proxy, fuga):
    assert proxy.columns == sp.COLUMNAS_PROXY == [
        "tipo_fila", "variable", "tipo", "rol_fase1", "grupo", "medida", "n", "n_2023", "asociacion_hospital", "asociacion_hospital_2023", "clase_proxy", "analiza_resultado",
        "v_resultado", "auc", "estado", "n_hospitales", "prop_global", "prop_minima", "prop_p10", "prop_mediana", "prop_p90", "prop_maxima", "razon_p90_p10", "phi_sobredispersion",
        "card_minima", "card_mediana", "card_maxima", "n_niveles_totales", "n_niveles_en_un_hospital",
    ]
    assert fuga.columns == sp.COLUMNAS_FUGA and fuga.columns[:6] == ["tipo_fila", "variable", "categoria", "tipo", "rol_fase1", "prohibida_siempre"]
    assert len(sp.COLUMNAS_FUGA) == 35 and sp.COLUMNAS_FUGA[-3:] == ["coincide_mayoria", "estado", "nota"]


def test_clase_proxy_en_los_bordes_exactos_0_15_y_0_30(u):
    c = lambda v: sp.clasificar_proxy(v, u)  # noqa: E731
    assert c(0.30) == "alto" and c(0.2999996) == "alto" and c(0.299999) == "moderado"          # 0,2999996 redondea a 0,3
    assert c(0.15) == "moderado" and c(0.1499996) == "moderado" and c(0.149999) == "bajo"
    assert c(0.0) == "bajo" and c(1.0) == "alto"
    assert c(None) == "no evaluable" and c(float("nan")) == "no evaluable"


def test_la_asociacion_con_el_hospital_es_identica_a_la_de_la_matriz_de_la_tarea_10(contexto, panel, proxy):
    a = asoc.calcular(contexto, panel)
    nombres, A = a["nombres"], a["A"]
    j = nombres.index("COD_HOSPITAL")
    v = proxy.filter(pl.col("tipo_fila") == "variable")
    assert set(v["variable"]) == set(nombres) - {"COD_HOSPITAL"} and "COD_HOSPITAL" not in v["variable"].to_list()
    for f in v.iter_rows(named=True):
        esperado = A[nombres.index(f["variable"]), j]
        if f["asociacion_hospital"] is None:
            assert np.isnan(esperado), f["variable"]
        else:
            assert f["asociacion_hospital"] == pytest.approx(esperado, abs=1e-12), f["variable"]


def test_el_proxy_de_2023_usa_la_misma_medida_con_los_codigos_del_ranking(contexto, panel, proxy):
    conf = asoc.preparar_columnas(panel, [2023])
    for nombre in ("ESPECIALIDAD_MEDICA", "COMUNA", "EDAD_ANIOS", "ELIX_01"):
        f = _fila(proxy, tipo_fila="variable", variable=nombre)
        p = asoc.medida_par(conf[nombre], conf["COD_HOSPITAL"])
        assert f["asociacion_hospital_2023"] == pytest.approx(p.valor, abs=1e-12) and f["n_2023"] == p.n == 240


def test_el_proxy_clasifica_la_verdad_de_diseno_del_panel_artificial(proxy):
    clase = {f["variable"]: f["clase_proxy"] for f in proxy.filter(pl.col("tipo_fila") == "variable").iter_rows(named=True)}
    assert clase["ESPECIALIDAD_MEDICA"] == "alto" and clase["SERVICIO_SALUD"] == "alto"                # función del hospital
    assert clase["COMUNA"] == "moderado" and clase["NIVEL_CUIDADO_INGRESO"] == "moderado"
    assert clase["PREVISION"] == "bajo" and clase["EDAD_ANIOS"] == "bajo"
    assert clase["ELIX_28"] == "no evaluable"                                                             # constante: la medida no existe
    f = _fila(proxy, tipo_fila="variable", variable="ELIX_28")
    assert f["estado"] == "constante" and f["asociacion_hospital"] is None


def test_etnia_y_pueblo_originario_tienen_proxy_pero_ninguna_cifra_contra_el_resultado(proxy, ranking):
    assert "ETNIA" not in ranking["variable"].to_list() and "PUEBLO_ORIGINARIO" not in ranking["variable"].to_list()
    for nombre in ("ETNIA", "PUEBLO_ORIGINARIO"):
        f = _fila(proxy, tipo_fila="variable", variable=nombre)
        assert f["analiza_resultado"] == "no" and f["v_resultado"] is None and f["auc"] is None
        assert f["asociacion_hospital"] is not None and f["clase_proxy"] != "no evaluable"                 # pero sí se mide contra el hospital
    f = _fila(proxy, tipo_fila="variable", variable="EDAD_ANIOS")
    r = ranking.filter(pl.col("variable") == "EDAD_ANIOS").row(0, named=True)
    assert f["analiza_resultado"] == "si" and f["v_resultado"] == r["v_resultado"] and f["auc"] == r["auc_senal"]            # v_resultado y auc salen del ranking


def test_el_proxy_sin_la_referencia_lanza_un_error_claro(contexto, panel, ranking):
    sin = dataclasses.replace(panel, variables={k: v for k, v in panel.variables.items() if k != "COD_HOSPITAL"}, datos=panel.datos.drop("COD_HOSPITAL"))
    with pytest.raises(ValueError, match="COD_HOSPITAL"):
        sp.tabla_proxy(contexto, sin, ranking)


# ==================================================================================== proxy: variación entre hospitales y cardinalidad
def _hospitales(spec):
    """``spec`` = [(hospital, n, k)]: listas ``hospital`` y ``marca`` con ``k`` unos entre ``n`` filas de cada hospital."""
    hosp, marca = [], []
    for h, n, k in spec:
        hosp += [h] * n
        marca += [1.0] * k + [0.0] * (n - k)
    return hosp, np.array(marca)


def test_variacion_entre_tres_hospitales_de_proporciones_conocidas_a_mano():
    hosp, marca = _hospitales([("A", 200, 20), ("B", 200, 40), ("C", 200, 60), ("D", 50, 45)])          # D tiene menos de 100 filas: queda fuera
    r = sp.variacion_por_hospital(hosp, marca, 100, 0.10)
    assert r["n_hospitales"] == 3 and r["prop_global"] == pytest.approx(0.2)                              # 120 de 600
    assert (r["prop_minima"], r["prop_mediana"], r["prop_maxima"]) == pytest.approx((0.1, 0.2, 0.3))
    assert r["prop_p10"] == pytest.approx(0.12) and r["prop_p90"] == pytest.approx(0.28)                    # cuantiles lineales de [0,1; 0,2; 0,3]
    assert r["razon_p90_p10"] == pytest.approx(0.28 / 0.12)
    z = (0.1 - 0.2) / np.sqrt(0.2 * 0.8 / 200)                                                              # -3,5355; 0; 3,5355
    assert r["phi_sobredispersion"] == pytest.approx((z**2 + 0.0 + z**2) / 3.0)                            # promedio de z al cuadrado sin recorte (floor(0,1 * 3) = 0)


def test_la_razon_p90_p10_es_nula_si_el_p10_es_cero_y_los_nulos_no_cuentan():
    spec = [(f"H{i:02d}", 100, 0) for i in range(2)] + [(f"H{i:02d}", 100, 10 * (i - 1)) for i in range(2, 11)]            # 11 hospitales: p10 cae sobre un hospital con proporción 0
    hosp, marca = _hospitales(spec)
    r = sp.variacion_por_hospital(hosp, marca, 100, 0.10)
    assert r["n_hospitales"] == 11 and r["prop_p10"] == pytest.approx(0.0) and r["razon_p90_p10"] is None and r["prop_p90"] > 0
    marca_nula = marca.copy()
    marca_nula[:50] = np.nan                                                                                # 50 nulos en H00: le quedan 50 filas válidas (< 100) y sale
    assert sp.variacion_por_hospital(hosp, marca_nula, 100, 0.10)["n_hospitales"] == 10


def test_variacion_sin_hospitales_elegibles_deja_todo_nulo():
    r = sp.variacion_por_hospital(["A"] * 20, np.array([1.0, 0.0] * 10), 100, 0.10)
    assert r["n_hospitales"] == 0 and all(r[k] is None for k in ("prop_global", "prop_p10", "razon_p90_p10", "phi_sobredispersion"))


def test_la_fila_de_variacion_de_cuidados_especiales_sale_del_panel(contexto, panel, proxy):
    f = _fila(proxy, tipo_fila="variacion_uci")
    d = panel.ventana(panel.anios_ranking)
    oraculo = sp.variacion_por_hospital(d["COD_HOSPITAL"].to_list(), d["NIVEL_CUIDADO_INGRESO"].cast(pl.Float64).to_numpy(), pn.MIN_FILAS_HOSPITAL, 0.10)
    assert f["variable"] == "NIVEL_CUIDADO_INGRESO" and f["n_hospitales"] == oraculo["n_hospitales"] == 5
    assert f["phi_sobredispersion"] == pytest.approx(oraculo["phi_sobredispersion"]) and f["phi_sobredispersion"] > 1.0         # los hospitales difieren más que el azar
    assert f["prop_minima"] < f["prop_mediana"] < f["prop_maxima"] and f["asociacion_hospital"] is None and f["n"] is None


def test_la_fila_de_cardinalidad_copia_los_agregados_del_panel_y_no_los_recalcula_con_el_texto_recortado(contexto, panel, ranking):
    # 70 niveles de especialidad en 5 hospitales de 200 filas: 67 compartidos por todos (cada uno aparece al menos dos veces en cada hospital) y 3 raros, cada uno de un solo
    # hospital; esos 3 quedarían fuera de los 60 más frecuentes y se perderían en OTROS si la cardinalidad se calculara sobre la columna recortada
    texto, hosp = [], []
    for h in range(5):
        texto += [f"ESP_{i % 67:02d}" for i in range(200)] + ([f"RARA_{h}"] * 3 if h < 3 else [])
        hosp += [f"H{h}"] * (200 + (3 if h < 3 else 0))
    agregados = pn.cardinalidad_especialidad(pl.Series(texto, dtype=pl.String), pl.Series(hosp, dtype=pl.String), pn.MIN_FILAS_HOSPITAL)
    assert (agregados["n_niveles_totales"], agregados["n_niveles_en_un_hospital"], agregados["n_hospitales"]) == (70, 3, 5)      # el panel real la calcula sobre el texto sin tope
    con_agregados = dataclasses.replace(panel, cardinalidad_especialidad=agregados)
    f = _fila(sp.tabla_proxy(contexto, con_agregados, ranking), tipo_fila="cardinalidad_especialidad")
    assert f["variable"] == "ESPECIALIDAD_MEDICA" and f["n_niveles_totales"] == 70 and f["n_niveles_en_un_hospital"] == 3                # no 60 ni 15: se copian
    for clave in ("n_hospitales", "card_minima", "card_mediana", "card_maxima"):
        assert f[clave] == agregados[clave], clave
    assert len(panel.niveles["ESPECIALIDAD_MEDICA"]) == 15 and 70 not in (len(panel.niveles["ESPECIALIDAD_MEDICA"]), 60)         # la columna recortada del panel no podría dar esa cifra


def test_los_niveles_de_especialidad_del_panel_artificial_son_exclusivos_de_un_hospital(proxy):
    f = _fila(proxy, tipo_fila="cardinalidad_especialidad")
    assert f["n_niveles_totales"] == f["n_niveles_en_un_hospital"] == 15 and f["n_hospitales"] == 5 and f["card_minima"] == f["card_maxima"] == 3


def test_el_panel_sin_cardinalidad_no_agrega_la_fila(contexto, panel, ranking):
    t = sp.tabla_proxy(contexto, dataclasses.replace(panel, cardinalidad_especialidad={}), ranking)
    assert "cardinalidad_especialidad" not in t["tipo_fila"].to_list()


# ========================================================================================================== proxy: supresión por bloques
def test_es_extranjero_conserva_2019_2022_y_anula_el_conteo_de_2023(contexto, proxy):
    f = _fila(proxy, tipo_fila="variable", variable="ES_EXTRANJERO")
    assert f["n_2023"] == 4 and f["n"] > 100 and f["asociacion_hospital"] is not None and f["clase_proxy"] != "no evaluable"
    pub, n_celdas = supresion.suprimir_bloques(proxy, sp.BLOQUES_PROXY, minimo=contexto.celda_minima)
    g = _fila(pub, tipo_fila="variable", variable="ES_EXTRANJERO")
    assert g["n_2023"] is None and g["asociacion_hospital_2023"] is None
    assert g["asociacion_hospital"] == f["asociacion_hospital"] and g["clase_proxy"] == f["clase_proxy"] and g["n"] == f["n"]       # conserva las cifras de 2019-2022
    assert n_celdas == 1 and _fila(pub, tipo_fila="variable", variable="EDAD_ANIOS")["asociacion_hospital_2023"] is not None


def test_un_n_2023_entre_1_y_9_anula_solo_el_bloque_de_2023_aunque_el_valor_real_exista(contexto, panel, ranking):
    anio = panel.datos["anio"].to_numpy()
    pocos = set(np.flatnonzero(anio == 2023)[:8].tolist())                                                    # N_DX_SEC informada solo en 8 filas de 2023
    nuevo = [v if (a <= 2022 or i in pocos) else None for i, (v, a) in enumerate(zip(panel.datos["N_DX_SEC"].to_list(), anio))]
    tabla = sp.tabla_proxy(contexto, _con_columna(panel, "N_DX_SEC", nuevo), ranking)
    f = _fila(tabla, tipo_fila="variable", variable="N_DX_SEC")
    assert f["n_2023"] == 8 and f["asociacion_hospital_2023"] is not None and f["n"] == 960                  # el valor real de 2023 existe...
    pub, n_celdas = supresion.suprimir_bloques(tabla, sp.BLOQUES_PROXY, minimo=contexto.celda_minima)
    g = _fila(pub, tipo_fila="variable", variable="N_DX_SEC")
    assert g["n_2023"] is None and g["asociacion_hospital_2023"] is None                                      # ...pero la tabla publicada lo anula
    assert (g["asociacion_hospital"], g["clase_proxy"], g["v_resultado"], g["auc"], g["n"]) == (f["asociacion_hospital"], f["clase_proxy"], f["v_resultado"], f["auc"], f["n"])
    assert n_celdas == 2                                                                                       # N_DX_SEC y ES_EXTRANJERO


def test_los_conteos_de_hospitales_no_estan_en_las_columnas_de_conteo_del_proxy():
    cols_n = {c for cols, _ in sp.BLOQUES_PROXY for c in cols}
    assert cols_n == {"n", "n_2023"} and not cols_n & {"n_hospitales", "card_minima", "n_niveles_totales"}


# ================================================================================================================================= fuga
def test_fuga_la_variable_artificial_sobre_0_85_esta_marcada_y_es_prohibida_siempre(fuga):
    f = _fila(fuga, tipo_fila="variable", variable="IR_29301_MORTALIDAD")
    assert f["categoria"] == "solo_fuga" and f["prohibida_siempre"] == "si" and f["nota"] == sp.NOTA_R6
    assert f["auc_ranking"] > 0.85 and f["supera_umbral_fuga"] == "si" and f["supera_a_basales"] == "si" and f["estado"] == "ok" and f["direccion"] == "+"
    assert f["auc_optimista"] == "si" and 0.90 < f["auc_ranking"] < 0.95                                   # medido 0,9222 en el generador sintético; 0,925 en el panel artificial
    sev = _fila(fuga, tipo_fila="variable", variable="IR_29301_SEVERIDAD")
    assert sev["prohibida_siempre"] == "no" and sev["supera_umbral_fuga"] == "no" and sev["nota"] is None


def test_fuga_lista_las_siete_de_solo_fuga_y_las_cuatro_condicionales(fuga):
    v = fuga.filter(pl.col("tipo_fila") == "variable")
    solo = {"IR_29301_SEVERIDAD", "IR_29301_MORTALIDAD", "IR_29301_PESO", "SERVICIOALTA", "USOSPABELLON", "N_PROCEDIMIENTOS", "N_TRASLADOS_INTERNOS"}
    cond = {"SERVICIOINGRESO", "ESPECIALIDAD_MEDICA", "NIVEL_CUIDADO_INGRESO", "DERIVADO_OTRO_HOSPITAL"}
    assert set(v.filter(pl.col("categoria") == "solo_fuga")["variable"]) == solo and set(v.filter(pl.col("categoria") == "condicional")["variable"]) == cond
    assert v.height == 11 and set(fuga["tipo_fila"]) == {"variable", "referencia", "concordancia", "concordancia_con_traslado"}


def test_fuga_n_2023_y_n_eventos_2023_estan_presentes_y_el_ic_de_2023_tiene_su_base(fuga, panel):
    f = _fila(fuga, tipo_fila="variable", variable="IR_29301_MORTALIDAD")
    assert (f["n"], f["n_eventos"], f["n_2023"], f["n_eventos_2023"]) == (960, 154, 240, 30)
    bajo, alto = est.ic_hanley_mcneil(f["auc_confirmacion"], 30, 210)
    assert (f["auc_confirmacion_bajo"], f["auc_confirmacion_alto"]) == pytest.approx((bajo, alto))


def test_fuga_el_auc_de_la_variable_coincide_con_scikit_learn_y_se_orienta_con_el_ranking(fuga, panel):
    from sklearn.metrics import roc_auc_score

    f = _fila(fuga, tipo_fila="variable", variable="IR_29301_PESO")
    d, d23 = panel.ventana(panel.anios_ranking), panel.ventana([2023])
    assert f["auc_ranking"] == pytest.approx(roc_auc_score(d["y"], d["IR_29301_PESO"]), abs=1e-12)
    assert f["auc_confirmacion"] == pytest.approx(roc_auc_score(d23["y"], d23["IR_29301_PESO"]), abs=1e-12)


def test_fuga_estado_ok_en_una_variable_normal_y_constante_en_la_copia_con_usospabellon_forzada(contexto, panel, ranking, referencias):
    assert _fila(sp.tabla_fuga(contexto, panel, ranking, referencias), tipo_fila="variable", variable="USOSPABELLON")["estado"] == "ok"      # vale "1" en el 31 % y "(sin dato)" en el resto
    forzada = dataclasses.replace(panel, datos=panel.datos.with_columns(pl.lit("1").alias("USOSPABELLON")))
    f = _fila(sp.tabla_fuga(contexto, forzada, ranking, referencias), tipo_fila="variable", variable="USOSPABELLON")
    assert f["estado"] == "constante" and f["auc_ranking"] is None and f["auc_ranking_bajo"] is None and f["auc_ranking_alto"] is None
    assert f["auc_confirmacion"] is None and f["supera_umbral_fuga"] is None and f["supera_a_basales"] is None        # sin dirección del ranking no hay AUC de 2023 que comparar
    assert f["n"] == 960 and f["n_eventos"] == 154 and f["n_2023"] == 240                                    # los conteos se informan aunque no haya AUC


def test_fuga_sin_ambos_desenlaces_y_no_evaluable_en_la_ventana(contexto, panel, ranking, referencias):
    nulos = _con_columna(panel, "N_PROCEDIMIENTOS", [None] * panel.datos.height)
    assert _fila(sp.tabla_fuga(contexto, nulos, ranking, referencias), tipo_fila="variable", variable="N_PROCEDIMIENTOS")["estado"] == "no evaluable en la ventana"
    anio, y = panel.datos["anio"].to_numpy(), panel.datos["y"].to_numpy()
    solo_no_eventos = [float(i) if (a <= 2022 and yy == 0) else None for i, (a, yy) in enumerate(zip(anio, y))]      # las filas con valor del ranking no tienen ningún evento
    p = _con_columna(panel, "N_PROCEDIMIENTOS", solo_no_eventos)
    f = _fila(sp.tabla_fuga(contexto, p, ranking, referencias), tipo_fila="variable", variable="N_PROCEDIMIENTOS")
    assert f["estado"] == "sin ambos desenlaces" and f["auc_ranking"] is None and f["n_eventos"] == 0


def test_mortalidad_es_prohibida_siempre_aun_con_auc_de_un_medio(contexto, panel, ranking, referencias):
    rng = np.random.default_rng(9)
    ruido = _con_columna(panel, "IR_29301_MORTALIDAD", rng.choice(["0", "1", "2", "3"], size=panel.datos.height).tolist())
    ruido = dataclasses.replace(ruido, niveles={**ruido.niveles, "IR_29301_MORTALIDAD": ["0", "1", "2", "3"]})
    f = _fila(sp.tabla_fuga(contexto, ruido, ranking, referencias), tipo_fila="variable", variable="IR_29301_MORTALIDAD")
    assert f["auc_ranking"] < 0.60 and f["supera_umbral_fuga"] == "no" and f["prohibida_siempre"] == "si" and f["nota"] == sp.NOTA_R6


def _con_pocas_filas_en_2023(panel, nombre, por_clase=3):
    """Copia del panel donde ``nombre`` solo tiene valor en ``2 * por_clase`` filas de 2023 (la mitad con evento) y en todo el ranking."""
    anio, y = panel.datos["anio"].to_numpy(), panel.datos["y"].to_numpy()
    elegidas = set(np.flatnonzero((anio == 2023) & (y == 1))[:por_clase].tolist()) | set(np.flatnonzero((anio == 2023) & (y == 0))[:por_clase].tolist())
    originales = panel.datos[nombre].to_list()
    return _con_columna(panel, nombre, [v if (a <= 2022 or i in elegidas) else None for i, (v, a) in enumerate(zip(originales, anio))])


def test_el_bloque_de_2023_de_la_fuga_se_anula_con_n_2023_entre_1_y_9_sin_tocar_el_de_2019_2022(contexto, panel, ranking, referencias):
    p = _con_pocas_filas_en_2023(panel, "N_PROCEDIMIENTOS")
    tabla = sp.tabla_fuga(contexto, p, ranking, referencias)
    f = _fila(tabla, tipo_fila="variable", variable="N_PROCEDIMIENTOS")
    assert f["n_2023"] == 6 and f["n_eventos_2023"] == 3 and f["auc_confirmacion"] is not None                 # el AUC real de 2023 existe...
    pub, _ = supresion.suprimir_bloques(tabla, sp.BLOQUES_FUGA, minimo=contexto.celda_minima)
    g = _fila(pub, tipo_fila="variable", variable="N_PROCEDIMIENTOS")
    assert g["n_2023"] is None and g["auc_confirmacion"] is None and g["supera_a_basales"] is None and g["supera_umbral_fuga"] is None and g["auc_ref_edad_sexo"] is None
    assert g["auc_ranking"] == f["auc_ranking"] and g["n"] == f["n"] and g["n_eventos"] == f["n_eventos"] and g["v_resultado"] == f["v_resultado"]       # 2019-2022 intacto
    otra = _fila(pub, tipo_fila="variable", variable="IR_29301_PESO")
    assert otra["auc_confirmacion"] is not None and otra["supera_a_basales"] is not None


# ---------------------------------------------------------------------------------------------------------------------- marcas de fuga
def test_supera_umbral_fuga_en_los_bordes_con_igualdad_y_valores_redondeados():
    m = lambda rank, conf=None: sp.marcas_de_fuga(rank, conf, 0.6, 0.6, 0.85)["supera_umbral_fuga"]  # noqa: E731
    assert m(0.85) == "si" and m(0.8499996) == "si" and m(0.849999) == "no"                                 # 0,8499996 redondea a 0,85 (mutante M28: > en vez de >=)
    assert m(0.70, 0.85) == "si" and m(0.86, 0.70) == "si" and m(0.70, 0.70) == "no"                      # el mayor de las dos ventanas
    assert m(None, 0.85) == "si" and m(None, None) is None


def test_supera_a_basales_es_estricto_y_compara_con_el_maximo_de_las_dos_referencias():
    f = lambda conf, edad, elix: sp.marcas_de_fuga(0.5, conf, edad, elix, 0.85)  # noqa: E731
    assert f(0.70, 0.60, 0.65)["supera_a_basales"] == "si" and f(0.65, 0.60, 0.65)["supera_a_basales"] == "no"        # igual al máximo no supera
    assert f(0.6500004, 0.60, 0.65)["supera_a_basales"] == "no"                                             # redondea a 0,65: empate
    assert f(0.70, 0.60, 0.72)["supera_a_basales"] == "no"                                                  # supera a edad y sexo pero no a Elixhauser (mutante M38)
    assert f(0.70, 0.60, 0.65)["referencia_maxima"] == "MODELO_ELIX31" and f(0.70, 0.66, 0.65)["referencia_maxima"] == "MODELO_EDAD_SEXO"
    assert f(0.70, 0.65, 0.65)["referencia_maxima"] == "MODELO_EDAD_SEXO"                                   # en empate, la primera
    assert f(None, 0.60, 0.65)["supera_a_basales"] is None and f(0.70, None, None)["referencia_maxima"] is None
    assert f(0.70, None, 0.65)["referencia_maxima"] == "MODELO_ELIX31"


def test_referencia_maxima_con_y_que_depende_solo_de_la_edad_o_solo_de_las_categorias(contexto):
    for modo, esperada in (("solo_edad", "MODELO_EDAD_SEXO"), ("solo_elix", "MODELO_ELIX31")):
        p = panel_artificial(y_modo=modo)
        r = modelos.referencias(p, SEMILLA)
        t = sp.tabla_fuga(contexto, p, _ranking_de_juguete(p), r)
        assert set(t.filter(pl.col("tipo_fila") == "variable")["referencia_maxima"]) == {esperada}
        assert _fila(t, tipo_fila="referencia", variable="MODELO_EDAD_SEXO")["estado"] == "ok" and _fila(t, tipo_fila="referencia", variable="MODELO_ELIX31")["auc_confirmacion"] is not None


def test_una_variable_que_solo_depende_de_las_categorias_no_supera_a_basales_si_no_supera_a_elixhauser(contexto):
    p = panel_artificial(y_modo="solo_elix")
    r = modelos.referencias(p, SEMILLA)
    elix_max = max(r["MODELO_EDAD_SEXO"]["auc_2023"], r["MODELO_ELIX31"]["auc_2023"])
    t = sp.tabla_fuga(contexto, p, _ranking_de_juguete(p), r)
    for f in t.filter(pl.col("tipo_fila") == "variable").iter_rows(named=True):
        if f["auc_confirmacion"] is not None:
            assert (f["supera_a_basales"] == "si") == (round(f["auc_confirmacion"], 6) > round(elix_max, 6)), f["variable"]


def test_las_filas_de_referencia_traen_los_dos_auc_y_sus_conteos(fuga, referencias):
    for nombre in ("MODELO_EDAD_SEXO", "MODELO_ELIX31"):
        f = _fila(fuga, tipo_fila="referencia", variable=nombre)
        r = referencias[nombre]
        assert f["categoria"] == "referencia" and f["auc_ranking"] == pytest.approx(r["auc_ranking"]) and f["auc_confirmacion"] == pytest.approx(r["auc_2023"])
        assert (f["n"], f["n_eventos"], f["n_2023"], f["n_eventos_2023"]) == (960, 154, 240, 30) and f["estado"] == "ok" and f["prohibida_siempre"] is None
        assert f["auc_confirmacion_bajo"] < f["auc_confirmacion"] < f["auc_confirmacion_alto"]
    v = _fila(fuga, tipo_fila="variable", variable="IR_29301_PESO")
    assert v["auc_ref_edad_sexo"] == pytest.approx(referencias["MODELO_EDAD_SEXO"]["auc_2023"]) and v["auc_ref_elix31"] == pytest.approx(referencias["MODELO_ELIX31"]["auc_2023"])


# ------------------------------------------------------------------------------------------------------------------------- concordancia
def _diez_filas(con_nulo: bool = False):
    # filas 0 a 5 sin traslado interno (igual: 1, 1, 1, 1, 0, 0) y filas 6 a 9 con traslado (igual: 1, 0, 0, 0)
    igual = [1, 1, 1, 1, 0, 0, 1, 0, 0, 0]
    traslados = [0, 0, 0, 0, 0, 0, 2, 1, 3, 1]
    igual = np.array(igual, dtype=float)
    if con_nulo:
        igual[1] = np.nan
    return igual, np.array(traslados, dtype=float)


def _wilson(k, n, z=1.959964):
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    mitad = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centro - mitad, centro + mitad


def test_concordancia_con_diez_filas_a_mano_con_y_sin_traslado_y_wilson():
    sin, con = sp.concordancia(*_diez_filas())
    assert sin["tipo_fila"] == "concordancia" and con["tipo_fila"] == "concordancia_con_traslado"
    assert (sin["n_sin_traslado"], sin["n_iguales_sin_traslado"]) == (6, 4) and sin["prop_igualdad"] == pytest.approx(4 / 6)
    assert (sin["ic_bajo"], sin["ic_alto"]) == pytest.approx(_wilson(4, 6)) and sin["coincide_mayoria"] == "si" and sin["estado"] == "ok"
    assert (con["n_con_traslado"], con["n_iguales_con_traslado"]) == (4, 1) and con["prop_igualdad"] == pytest.approx(0.25)
    assert (con["ic_bajo"], con["ic_alto"]) == pytest.approx(_wilson(1, 4)) and con["coincide_mayoria"] == "no"
    assert "n_con_traslado" not in sin and "n_sin_traslado" not in con                                      # los conteos de cada tipo están nulos en la otra fila


def test_la_concordancia_no_cuenta_en_el_denominador_las_filas_con_el_indicador_nulo():
    sin, con = sp.concordancia(*_diez_filas(con_nulo=True))
    assert (sin["n_sin_traslado"], sin["n_iguales_sin_traslado"]) == (5, 3)                                  # 5 y no 6: la fila nula no cuenta (mutante M49)
    assert sin["prop_igualdad"] == pytest.approx(3 / 5) and (sin["ic_bajo"], sin["ic_alto"]) == pytest.approx(_wilson(3, 5))
    assert con["n_con_traslado"] == 4                                                                        # y la fila nula no es "no igual" tampoco en el otro grupo


def test_la_concordancia_solo_clasifica_filas_con_traslados_conocidos_y_sin_datos_si_no_hay_filas():
    igual = np.array([1.0, 1.0, 0.0, 1.0])
    traslados = np.array([0.0, np.nan, 0.0, 2.0])
    sin, con = sp.concordancia(igual, traslados)
    assert (sin["n_sin_traslado"], sin["n_iguales_sin_traslado"], con["n_con_traslado"]) == (2, 1, 1)
    vacio, _ = sp.concordancia(np.array([np.nan]), np.array([0.0]))
    assert vacio["estado"] == "sin datos" and vacio["prop_igualdad"] is None and vacio["coincide_mayoria"] is None and vacio["n_sin_traslado"] == 0
    with pytest.raises(ValueError, match="largos distintos"):
        sp.concordancia(np.array([1.0]), np.array([0.0, 1.0]))


def test_las_filas_de_concordancia_de_la_tabla_salen_de_la_ventana_de_ranking(fuga, panel):
    d = panel.ventana(panel.anios_ranking)
    esperado = sp.concordancia(d["SERVICIO_INGRESO_IGUAL_ALTA"].cast(pl.Float64).to_numpy(), d["N_TRASLADOS_INTERNOS"].cast(pl.Float64).to_numpy())
    f = _fila(fuga, tipo_fila="concordancia")
    assert f["variable"] == "SERVICIO_INGRESO_IGUAL_ALTA" and f["n_sin_traslado"] == esperado[0]["n_sin_traslado"] and f["prop_igualdad"] == pytest.approx(esperado[0]["prop_igualdad"])
    assert _fila(fuga, tipo_fila="concordancia_con_traslado")["n_con_traslado"] == esperado[1]["n_con_traslado"]


def test_un_conteo_de_concordancia_pequeno_anula_solo_su_bloque(contexto, panel, ranking, referencias):
    traslados = np.zeros(panel.datos.height)
    traslados[:6] = 2.0                                                                                         # solo 6 egresos del ranking con traslado interno
    p = _con_columna(panel, "N_TRASLADOS_INTERNOS", traslados.tolist())
    tabla = sp.tabla_fuga(contexto, p, ranking, referencias)
    con = _fila(tabla, tipo_fila="concordancia_con_traslado")
    assert con["n_con_traslado"] == 6 and con["prop_igualdad"] is not None                                     # el valor real existe...
    pub, _ = supresion.suprimir_bloques(tabla, sp.BLOQUES_FUGA, minimo=10)
    assert _fila(pub, tipo_fila="concordancia_con_traslado")["prop_igualdad"] is None and _fila(pub, tipo_fila="concordancia_con_traslado")["n_con_traslado"] is None
    sin = _fila(pub, tipo_fila="concordancia")
    assert sin["prop_igualdad"] is not None and sin["n_sin_traslado"] > 900                                      # ...y la otra fila de concordancia y las variables quedan intactas
    assert _fila(pub, tipo_fila="variable", variable="IR_29301_PESO")["auc_ranking"] is not None


# ------------------------------------------------------------------------------------------------------------------------------ métricas
def test_registrar_estadistica_anula_el_valor_y_anota_la_supresion_si_la_fuente_se_suprimio(contexto):
    sp.registrar_estadistica(contexto, "prueba.visible", 0.7, 0.7, tabla="t.csv", alcance="2019-2022", ic_bajo=0.6, ic_alto=0.8, nota="categórica: validación cruzada")
    sp.registrar_estadistica(contexto, "prueba.suprimida", 0.7, None, tabla="t.csv", alcance="2019-2022", ic_bajo=0.6, ic_alto=0.8)
    sp.registrar_estadistica(contexto, "prueba.sin_valor", None, None, tabla="t.csv", alcance="2023")
    m = {f["clave"]: f for f in contexto.metricas().iter_rows(named=True)}
    assert m["prueba.visible"]["valor"] == 0.7 and (m["prueba.visible"]["ic_bajo"], m["prueba.visible"]["ic_alto"]) == (0.6, 0.8) and m["prueba.visible"]["nota"] == "categórica: validación cruzada"
    s = m["prueba.suprimida"]
    assert s["valor"] is None and s["ic_bajo"] is None and s["ic_alto"] is None and "[suprimido: menos de diez casos en la fila de la tabla fuente]" in s["nota"]
    assert m["prueba.sin_valor"]["valor"] is None and m["prueba.sin_valor"]["nota"] == "" and m["prueba.visible"]["alcance"] == "2019-2022"


def test_las_metricas_de_proxy_y_fuga_tienen_prefijo_alcance_y_ninguna_cubre_2024(contexto, panel, ranking, referencias):
    p = _con_pocas_filas_en_2023(_con_pocas_filas_en_2023(panel, "N_PROCEDIMIENTOS"), "N_DX_SEC")                # dos variables con 6 filas en 2023: sus bloques de 2023 se suprimen
    proxy = sp.tabla_proxy(contexto, p, ranking)
    fuga = sp.tabla_fuga(contexto, p, ranking, referencias)
    pub_proxy = supresion.suprimir_bloques(proxy, sp.BLOQUES_PROXY, minimo=contexto.celda_minima)[0]
    pub_fuga = supresion.suprimir_bloques(fuga, sp.BLOQUES_FUGA, minimo=contexto.celda_minima)[0]
    sp.registrar_metricas(contexto, proxy, pub_proxy, fuga, pub_fuga)
    m = contexto.metricas().filter(pl.col("clave").str.starts_with("proxy.") | pl.col("clave").str.starts_with("fuga."))
    assert m.height > 100 and set(m["tarea"]) == {"T11"} and "" not in set(m["alcance"]) and not any("2024" in a for a in m["alcance"])
    f = {r["clave"]: r for r in m.iter_rows(named=True)}
    assert f["proxy.v.ESPECIALIDAD_MEDICA"]["alcance"] == "2019-2022" and f["proxy.v_2023.ESPECIALIDAD_MEDICA"]["alcance"] == "2023"
    assert f["proxy.v_2023.N_DX_SEC"]["valor"] is None and "[suprimido" in f["proxy.v_2023.N_DX_SEC"]["nota"]                  # la fuente suprimió n_2023 = 6
    assert f["proxy.v.N_DX_SEC"]["valor"] is not None and f["proxy.v_2023.ESPECIALIDAD_MEDICA"]["valor"] is not None
    assert f["fuga.auc_2023.N_PROCEDIMIENTOS"]["valor"] is None and "[suprimido" in f["fuga.auc_2023.N_PROCEDIMIENTOS"]["nota"]
    assert f["fuga.auc.N_PROCEDIMIENTOS"]["valor"] is not None and f["fuga.auc.N_PROCEDIMIENTOS"]["ic_bajo"] < f["fuga.auc.N_PROCEDIMIENTOS"]["valor"]
    assert f["proxy.n_alto"]["valor"] == 2 and f["proxy.n_moderado"]["valor"] == 2 and f["proxy.uci_phi"]["valor"] > 0
    n_sobre = int((fuga.filter(pl.col("tipo_fila") == "variable")["supera_umbral_fuga"] == "si").sum())              # cuenta variables con los valores reales (no son casos)
    assert f["fuga.auc.IR_29301_MORTALIDAD"]["valor"] > 0.85 and f["fuga.auc_2023.IR_29301_MORTALIDAD"]["alcance"] == "2023" and f["fuga.n_supera_umbral"]["valor"] == n_sobre >= 1
    assert f["fuga.auc.IR_29301_MORTALIDAD"]["ic_bajo"] < f["fuga.auc.IR_29301_MORTALIDAD"]["valor"] < f["fuga.auc.IR_29301_MORTALIDAD"]["ic_alto"]
    assert f["fuga.concordancia_pct_sin_traslado"]["valor"] == pytest.approx(100 * _fila(fuga, tipo_fila="concordancia")["prop_igualdad"])
    assert f["fuga.auc.USOSPABELLON"]["nota"] == "categórica: optimista" and f["fuga.n_supera_umbral"]["alcance"] == "2019-2023"


# ------------------------------------------------------------------------------------------------------------------------------- figuras
@pytest.fixture()
def contexto_con_tablas(tmp_path, panel, ranking, referencias, u):
    ctx = Contexto.crear("extracto", fuente=crear_fuente_sintetica(tmp_path / "datos", n=60, semilla=29), dir_salida_base=tmp_path / "salidas")
    pn.escribir_por_bloques(ctx, "t23_proxy_hospital.csv", sp.tabla_proxy(ctx, panel, ranking), sp.BLOQUES_PROXY, tarea="T11", descripcion="proxy")
    pn.escribir_por_bloques(ctx, "t23_fuga.csv", sp.tabla_fuga(ctx, panel, ranking, referencias), sp.BLOQUES_FUGA, tarea="T11", descripcion="fuga")
    return ctx


def test_figura_proxy_pasa_la_prueba_del_diseno_final_y_se_arma_solo_con_su_csv(contexto_con_tablas, monkeypatch):
    ctx = contexto_con_tablas
    resultados = vigilar_savefig(monkeypatch)
    for otro in ctx.dir_tablas.glob("*.csv"):
        if otro.name != "t23_proxy_hospital.csv":
            otro.unlink()                                                                                   # la figura lee únicamente su CSV
    assert sp.figura_proxy(ctx) is True
    assert len(resultados) == 1 and resultados[0] == [], resultados[0]                                      # ningún par de etiquetas (con zonas rotuladas) se solapa con el título y el pie puestos
    e = ctx.figuras.indice().row(0, named=True)
    assert (e["id_figura"], e["tarea"], e["csv"], e["decision_relacionada"], e["uso_interno"]) == ("F11_02", "11", "t23_proxy_hospital.csv", "2; 3", "no")
    assert e["titulo"] == "Mapa de proxies: asociación con el resultado y con el hospital (2019 a 2022)" and (ctx.dir_figuras / "F11_02.png").is_file()


def test_figura_proxy_rotula_los_puntos_pedidos_y_no_dibuja_etnia_ni_pueblo_originario(contexto_con_tablas, monkeypatch):
    from matplotlib.figure import Figure

    ctx = contexto_con_tablas
    capturadas = []
    original = Figure.savefig
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: (capturadas.append(self), original(self, *a, **k))[1])
    sp.figura_proxy(ctx)
    fig = capturadas[0]
    textos = [t.get_text() for t in fig.axes[0].texts]
    esperados = {"ESPECIALIDAD_MEDICA", "SERVICIO_SALUD", "COMUNA", "NIVEL_CUIDADO_INGRESO", "EDAD_ANIOS", "PREVISION", "SERVICIOINGRESO", "N_DX_SEC"}
    assert esperados <= set(textos) and {"moderado", "alto"} <= set(textos)
    assert "ETNIA" not in textos and "PUEBLO_ORIGINARIO" not in textos
    t = ctx.leer_tabla("t23_proxy_hospital.csv", columnas_texto=sp._TEXTO_PROXY).filter(pl.col("tipo_fila") == "variable")
    pedidos = t.filter((pl.col("asociacion_hospital") >= 0.15) | (pl.col("v_resultado") >= sp.ETIQUETA_V_RESULTADO))["variable"].to_list()
    assert {x for x in textos if x not in ("moderado", "alto")} == set(pedidos)                             # exactamente los puntos con x >= moderado o y >= la constante de visualización
    pie = " ".join(x.get_text() for x in fig.texts)
    assert "ETNIA y PUEBLO_ORIGINARIO no se dibujan" in pie


def test_figura_proxy_sin_puntos_se_omite(contexto_con_tablas):
    ctx = contexto_con_tablas
    t = ctx.leer_tabla("t23_proxy_hospital.csv", columnas_texto=sp._TEXTO_PROXY).with_columns(pl.lit(None, dtype=pl.Float64).alias("v_resultado"))
    t.write_csv(ctx.dir_tablas / "t23_proxy_hospital.csv")
    assert sp.figura_proxy(ctx) is False and ctx.figuras.ids() == []


def test_figura_fuga_se_arma_solo_con_su_csv_y_rotula_la_variable_prohibida(contexto_con_tablas, monkeypatch):
    from matplotlib.figure import Figure

    ctx = contexto_con_tablas
    for otro in ctx.dir_tablas.glob("*.csv"):
        if otro.name != "t23_fuga.csv":
            otro.unlink()
    capturadas = []
    original = Figure.savefig
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: (capturadas.append(self), original(self, *a, **k))[1])
    assert sp.figura_fuga(ctx) is True
    ax = capturadas[0].axes[0]
    etiquetas = [t.get_text() for t in ax.get_yticklabels()]
    assert any(e.startswith("IR_29301_MORTALIDAD") and "prohibida siempre (R6)" in e for e in etiquetas) and etiquetas[0].startswith("IR_29301_MORTALIDAD")        # la de mayor AUC de 2023 va arriba
    assert sum("(condicional)" in e for e in etiquetas) == 4 and len(etiquetas) == 11
    e = ctx.figuras.indice().row(0, named=True)
    assert (e["id_figura"], e["csv"], e["decision_relacionada"], e["titulo"]) == ("F11_06", "t23_fuga.csv", "2; 14", "Evidencia de fuga frente a las referencias basales (2019 a 2023)")


def test_figura_fuga_sin_variables_con_auc_se_omite(contexto_con_tablas):
    ctx = contexto_con_tablas
    t = ctx.leer_tabla("t23_fuga.csv", columnas_texto=sp._TEXTO_FUGA).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("auc_ranking"), pl.lit(None, dtype=pl.Float64).alias("auc_confirmacion"),
    )
    t.write_csv(ctx.dir_tablas / "t23_fuga.csv")
    assert sp.figura_fuga(ctx) is False


# ================================================================================================================================
# Elixhauser: OR de las 31 categorías y sensibilidad de estadía (``eda/sensibilidad_elix.py``)
# ================================================================================================================================
from sklearn.linear_model import LogisticRegression  # noqa: E402

from eda import elixhauser  # noqa: E402
from eda import sensibilidad_elix as se  # noqa: E402

COLUMNAS_SENSIBILIDAD = [
    "elix_id", "nombre_es", "corte_p25", "corte_p50", "corte_p75", "n_t1", "n_t2", "n_t3", "n_t4", "k_t1", "k_t2", "k_t3", "k_t4", "prev_t1", "prev_t2", "prev_t3", "prev_t4",
    "razon_prev_t4_t1", "razon_bajo", "razon_alto", "or_larga", "or_larga_bajo", "or_larga_alto", "p_larga", "q_larga", "indice_complicacion", "rango_indice", "sospechosa_fase1",
    "en_lista_repositorio", "en_top5", "estado",
]
COLUMNAS_OR = [
    "periodo", "elix_id", "nombre_es", "sospechosa_fase1", "n_expuestos", "n_eventos_expuestos", "n_no_expuestos", "n_eventos_no_expuestos", "prevalencia", "or_crudo", "or_crudo_bajo",
    "or_crudo_alto", "p_crudo", "q_crudo", "or_ajustado", "or_ajustado_bajo", "or_ajustado_alto", "p_ajustado", "q_ajustado", "or_relevante_crudo", "or_relevante_ajustado",
    "correccion_haldane", "convergio", "estado",
]


@pytest.fixture(scope="module")
def or_elix(contexto, panel):
    return se.tabla_or_elix(contexto, panel)


@pytest.fixture(scope="module")
def sensibilidad(contexto, panel, or_elix):
    return se.tabla_sensibilidad(contexto, panel, or_elix)


def _panel_con_cuadro(panel, a, b, c, d, categoria="ELIX_01"):
    """Panel de ``a + b + c + d`` egresos de 2019 a 2022 (sin 2023) cuyo cuadro de 2 por 2 de ``categoria`` contra el desenlace es (a, b, c, d) a mano."""
    n = a + b + c + d
    x = [1] * (a + b) + [0] * (c + d)
    y = [1] * a + [0] * b + [1] * c + [0] * d
    datos = panel.datos.head(n).with_columns(
        pl.Series("anio", [2019 + (i % 4) for i in range(n)], dtype=pl.Int32), pl.Series("y", y, dtype=pl.Int8), pl.Series(categoria, x, dtype=pl.Int8),
    )
    return dataclasses.replace(panel, datos=datos, anio_confirmacion=None)


def _oraculo_ajustado(panel, anios, categoria):
    """OR ajustado con scikit-learn (``C=np.inf`` equivale a ``penalty=None``) y su IC de Wald con la información de Fisher; SEXO contra MUJER, armado a mano."""
    d = panel.ventana(anios)
    y, x, edad = d["y"].to_numpy(), d[categoria].to_numpy().astype(float), d["EDAD_ANIOS"].to_numpy()
    sexo = np.array(d["SEXO"].to_list(), dtype=object)
    columnas = [x, (edad - edad.mean()) / edad.std(), (edad**2 - (edad**2).mean()) / (edad**2).std(), (sexo == "HOMBRE").astype(float)]
    if (sexo == "(sin dato)").any():
        columnas.append((sexo == "(sin dato)").astype(float))
    X = np.column_stack(columnas)
    lr = LogisticRegression(C=np.inf, tol=1e-10, max_iter=5000).fit(X, y)
    Xa = np.column_stack([np.ones(len(y)), X])
    p = lr.predict_proba(X)[:, 1]
    cov = np.linalg.inv(Xa.T @ (Xa * (p * (1 - p))[:, None]))
    b, s = float(lr.coef_[0][0]), float(np.sqrt(cov[1, 1]))
    return np.exp(b), np.exp(b - 1.959964 * s), np.exp(b + 1.959964 * s)


def test_las_columnas_de_las_dos_tablas_de_elixhauser_son_exactamente_las_del_diseno(or_elix, sensibilidad):
    assert or_elix.columns == se.COLUMNAS_OR_ELIX == COLUMNAS_OR and len(COLUMNAS_OR) == 24
    assert sensibilidad.columns == se.COLUMNAS_SENSIBILIDAD == COLUMNAS_SENSIBILIDAD and len(COLUMNAS_SENSIBILIDAD) == 31


def test_hay_31_categorias_por_periodo_y_solo_las_del_ranking_sin_confirmacion(contexto, panel, or_elix):
    assert or_elix.height == 62 and or_elix.group_by("periodo", maintain_order=True).len()["len"].to_list() == [31, 31]
    assert or_elix["periodo"].unique(maintain_order=True).to_list() == ["2019-2022", "2023"]
    sin_conf = se.tabla_or_elix(contexto, dataclasses.replace(panel, anio_confirmacion=None, datos=panel.datos.filter(pl.col("anio") <= 2022)))
    assert sin_conf.height == 31 and set(sin_conf["periodo"]) == {"2019-2022"}


def test_sospechosa_fase1_es_igual_a_las_sospechosas_del_catalogo(or_elix, sensibilidad):
    e = elixhauser.cargar()
    for t in (or_elix.filter(pl.col("periodo") == "2019-2022"), or_elix.filter(pl.col("periodo") == "2023"), sensibilidad):
        assert set(t.filter(pl.col("sospechosa_fase1") == "si")["elix_id"]) == set(e.sospechosas) == {"ELIX_04", "ELIX_22", "ELIX_24", "ELIX_25", "ELIX_26"}
        assert dict(zip(t["elix_id"], t["nombre_es"])) == e.nombres_es


def test_or_crudo_contra_la_tabla_a_mano_y_contra_scipy(contexto, panel):
    p = _panel_con_cuadro(panel, 20, 80, 10, 190)
    f = _fila(se.tabla_or_elix(contexto, p), periodo="2019-2022", elix_id="ELIX_01")
    assert (f["n_expuestos"], f["n_eventos_expuestos"], f["n_no_expuestos"], f["n_eventos_no_expuestos"]) == (100, 20, 200, 10) and f["prevalencia"] == pytest.approx(100 / 300)
    assert f["or_crudo"] == pytest.approx(4.75) and (f["or_crudo_bajo"], f["or_crudo_alto"]) == pytest.approx((2.128388, 10.600746), abs=1e-6) and f["correccion_haldane"] == "no"
    ic = stats.contingency.odds_ratio([[20, 80], [10, 190]], kind="sample").confidence_interval(0.95)
    assert (f["or_crudo_bajo"], f["or_crudo_alto"]) == pytest.approx((ic.low, ic.high), rel=1e-5)
    assert f["p_crudo"] == pytest.approx(2 * stats.norm.sf(abs(np.log(4.75)) / ((np.log(10.600746) - np.log(2.128388)) / (2 * 1.959964))), rel=1e-6)
    assert f["or_relevante_crudo"] == "si"


def test_haldane_con_una_celda_en_cero_y_la_separacion_de_la_categoria_sin_eventos_entre_los_expuestos(contexto, panel):
    p = _panel_con_cuadro(panel, 0, 50, 5, 45)
    f = _fila(se.tabla_or_elix(contexto, p), periodo="2019-2022", elix_id="ELIX_01")
    assert f["correccion_haldane"] == "si" and f["or_crudo"] == pytest.approx(0.081908, abs=1e-6) and (f["or_crudo_bajo"], f["or_crudo_alto"]) == pytest.approx((0.004406, 1.522707), abs=1e-5)
    assert f["estado"] == "separación" and f["or_ajustado"] is None and f["or_ajustado_bajo"] is None and f["or_ajustado_alto"] is None and f["p_ajustado"] is None
    assert f["q_ajustado"] is None and f["or_relevante_ajustado"] is None and f["or_crudo"] is not None and f["convergio"] in ("si", "no")             # el crudo con Haldane se publica


def test_or_ajustado_contra_logistic_regression_sin_penalizacion_y_su_ic_de_wald(contexto, panel, or_elix):
    for categoria in ("ELIX_01", "ELIX_20", "ELIX_02"):
        f = _fila(or_elix, periodo="2019-2022", elix_id=categoria)
        o, bajo, alto = _oraculo_ajustado(panel, panel.anios_ranking, categoria)
        assert f["estado"] == "ok" and f["convergio"] == "si"
        assert (f["or_ajustado"], f["or_ajustado_bajo"], f["or_ajustado_alto"]) == pytest.approx((o, bajo, alto), rel=1e-4), categoria
        b, s = np.log(o), (np.log(alto) - np.log(bajo)) / (2 * 1.959964)
        assert f["p_ajustado"] == pytest.approx(2 * stats.norm.sf(abs(b / s)), rel=1e-3)


def test_el_nivel_base_de_sexo_es_mujer_aunque_hombre_sea_el_mas_frecuente_en_el_or(contexto, panel):
    sexo = np.where(np.arange(panel.datos.height) % 10 < 7, "HOMBRE", "MUJER").tolist()
    p = dataclasses.replace(panel, datos=panel.datos.with_columns(pl.Series("SEXO", sexo, dtype=pl.String)))
    f = _fila(se.tabla_or_elix(contexto, p), periodo="2019-2022", elix_id="ELIX_01")
    o, bajo, alto = _oraculo_ajustado(p, p.anios_ranking, "ELIX_01")
    assert (f["or_ajustado"], f["or_ajustado_bajo"], f["or_ajustado_alto"]) == pytest.approx((o, bajo, alto), rel=1e-4)
    assert se._diseno_edad_sexo(p, p.anios_ranking, SEMILLA).shape[1] == 3                              # edad, su cuadrado y una sola columna de sexo (HOMBRE contra MUJER)


def _panel_con_sexo_sin_dato_separado(panel):
    """Una de cada cuatro filas con SEXO "(sin dato)" y ningún evento: la columna de ese nivel queda separada (su coeficiente se va a menos infinito)."""
    sexo = ["(sin dato)" if i % 4 == 0 else s for i, s in enumerate(panel.datos["SEXO"].to_list())]
    y = [0 if i % 4 == 0 else v for i, v in enumerate(panel.datos["y"].to_list())]
    datos = panel.datos.with_columns(pl.Series("SEXO", sexo, dtype=pl.String), pl.Series("y", y, dtype=pl.Int8))
    return dataclasses.replace(panel, datos=datos, niveles={**panel.niveles, "SEXO": [*panel.niveles["SEXO"], "(sin dato)"]})


def test_la_separacion_de_una_columna_ajena_no_anula_el_or_de_las_categorias(contexto, panel):
    """Mutante M44: con la bandera global ``separacion`` los 31 OR ajustados quedarían nulos por culpa de SEXO "(sin dato)"."""
    p = _panel_con_sexo_sin_dato_separado(panel)
    base = se._diseno_edad_sexo(p, p.anios_ranking, SEMILLA)
    d = p.ventana(p.anios_ranking)
    res = est.regresion_logistica(np.column_stack([d["ELIX_01"].to_numpy().astype(float), base]), d["y"].to_numpy(), ridge=0.0)
    assert res.separacion and not res.separacion_por_coef[1] and res.separacion_por_coef[1:].sum() >= 1       # hay una columna separada y NO es la de la categoría
    t = se.tabla_or_elix(contexto, p).filter(pl.col("periodo") == "2019-2022")
    ok = t.filter(pl.col("estado") == "ok")
    assert ok.height >= 15 and ok["or_ajustado"].null_count() == 0                                                # las categorías siguen con su OR ajustado
    sin_filas = dataclasses.replace(p, datos=p.datos.filter(pl.col("SEXO") != "(sin dato)"))                       # el oráculo se ajusta sin esas filas ni esa columna
    for categoria in ("ELIX_01", "ELIX_20"):
        f = _fila(t, elix_id=categoria)
        o, bajo, alto = _oraculo_ajustado(sin_filas, p.anios_ranking, categoria)
        assert (f["or_ajustado"], f["or_ajustado_bajo"], f["or_ajustado_alto"]) == pytest.approx((o, bajo, alto), rel=2e-3), categoria
        assert f["estado"] == "ok" and f["convergio"] == "si"


def test_una_categoria_sin_eventos_entre_los_expuestos_sale_con_separacion_y_el_resto_sigue_en_ok(contexto, panel):
    d = panel.datos
    x = d["ELIX_20"].to_numpy()
    y = np.where(x == 1, 0, d["y"].to_numpy())                                                                    # nadie expuesto a ELIX_20 muere
    p = dataclasses.replace(panel, datos=d.with_columns(pl.Series("y", y, dtype=pl.Int8)))
    t = se.tabla_or_elix(contexto, p).filter(pl.col("periodo") == "2019-2022")
    f = _fila(t, elix_id="ELIX_20")
    assert f["estado"] == "separación" and f["or_ajustado"] is None and f["or_crudo"] is not None and f["correccion_haldane"] == "si" and f["n_eventos_expuestos"] == 0
    assert _fila(t, elix_id="ELIX_01")["estado"] == "ok"


def test_categoria_sin_expuestos_y_periodo_sin_ambos_desenlaces(contexto, panel, or_elix):
    f = _fila(or_elix, periodo="2019-2022", elix_id="ELIX_28")
    assert f["estado"] == "sin expuestos" and f["n_expuestos"] == 0 and f["n_eventos_expuestos"] == 0 and f["n_no_expuestos"] == 960
    assert f["prevalencia"] == 0.0 and all(f[c] is None for c in ("or_crudo", "or_crudo_bajo", "p_crudo", "q_crudo", "or_ajustado", "p_ajustado", "or_relevante_crudo", "correccion_haldane", "convergio"))
    sin_eventos = dataclasses.replace(panel, datos=panel.datos.with_columns(pl.when(pl.col("anio") == 2023).then(0).otherwise(pl.col("y")).cast(pl.Int8).alias("y")))
    t = se.tabla_or_elix(contexto, sin_eventos)
    assert set(t.filter(pl.col("periodo") == "2023")["estado"]) == {"sin ambos desenlaces"} and set(t.filter(pl.col("periodo") == "2019-2022")["estado"]) != {"sin ambos desenlaces"}


def test_q_de_benjamini_y_hochberg_por_periodo_y_tipo_de_or_igual_al_de_scipy(or_elix):
    for periodo in ("2019-2022", "2023"):
        t = or_elix.filter(pl.col("periodo") == periodo)
        for p_col, q_col in (("p_crudo", "q_crudo"), ("p_ajustado", "q_ajustado")):
            validas = t.filter(pl.col(p_col).is_not_null())
            esperado = stats.false_discovery_control(validas[p_col].to_numpy(), method="bh")
            assert validas[q_col].to_numpy() == pytest.approx(esperado, abs=1e-9), (periodo, q_col)
            assert t.filter(pl.col(p_col).is_null())[q_col].null_count() == t.filter(pl.col(p_col).is_null()).height
            assert (validas[q_col] >= validas[p_col] - 1e-12).all() and (validas[q_col] <= 1.0).all()


def test_or_relevante_exige_el_umbral_y_que_el_ic_excluya_uno(or_elix):
    for f in or_elix.iter_rows(named=True):
        if f["or_crudo"] is not None:
            esperado = modelos.or_es_relevante(f["or_crudo"], f["or_crudo_bajo"], f["or_crudo_alto"], 1.2)
            assert f["or_relevante_crudo"] == esperado, f["elix_id"]
        else:
            assert f["or_relevante_crudo"] is None


def test_la_supresion_del_or_usa_los_cuatro_conteos_y_deja_las_marcas_y_el_estado(contexto, or_elix):
    pub, n_celdas = supresion.suprimir_bloques(or_elix, se.BLOQUES_OR_ELIX, minimo=contexto.celda_minima)
    chico = or_elix.filter((pl.col("n_eventos_expuestos") > 0) & (pl.col("n_eventos_expuestos") < 10))
    assert chico.height > 0 and n_celdas >= chico.height
    k = chico.row(0, named=True)
    g = _fila(pub, periodo=k["periodo"], elix_id=k["elix_id"])
    assert g["n_eventos_expuestos"] is None and all(g[c] is None for c in se.DEPENDIENTES_OR)                    # cifras anuladas
    assert g["estado"] == k["estado"] and g["sospechosa_fase1"] == k["sospechosa_fase1"] and g["nombre_es"] == k["nombre_es"]       # marcas y estado intactos
    assert se.COLS_N_OR == ["n_expuestos", "n_eventos_expuestos", "n_no_expuestos", "n_eventos_no_expuestos"]


# ------------------------------------------------------------------------------------------------------------------------ tramos
def test_asignar_tramos_con_el_ejemplo_a_mano_de_veinte_sobrevivientes():
    estancia = np.array([1] * 5 + [2] * 5 + [3] * 5 + [4] * 5, dtype=float)
    valores, conteos = np.unique(estancia, return_counts=True)
    cortes = [est.percentil_de_conteos(valores, conteos, q) for q in (0.25, 0.50, 0.75)]
    assert cortes == [1.0, 2.0, 3.0]
    tramo = se.asignar_tramos(estancia, *cortes)
    assert [int((tramo == k).sum()) for k in (1, 2, 3, 4)] == [5, 5, 0, 10]                                      # T3 queda vacío


def test_asignar_tramos_con_cortes_empatados_es_disjunto_y_exhaustivo_mutante_m33():
    estancia = np.array([1] * 8 + [2] * 10 + [3] * 2, dtype=float)
    valores, conteos = np.unique(estancia, return_counts=True)
    cortes = [est.percentil_de_conteos(valores, conteos, q) for q in (0.25, 0.50, 0.75)]
    assert cortes == [1.0, 2.0, 2.0]
    tramo = se.asignar_tramos(estancia, *cortes)
    cuentas = [int((tramo == k).sum()) for k in (1, 2, 3, 4)]
    assert cuentas == [8, 0, 0, 12] and sum(cuentas) == 20                                                       # la definición anterior daría (8, 10, 0, 12) con suma 30
    rng = np.random.default_rng(3)
    for _ in range(50):                                                                                          # propiedad: cada valor cae en exactamente un tramo para cualquier cortes ordenados
        c = np.sort(rng.integers(1, 20, 3)).astype(float)
        t = se.asignar_tramos(rng.integers(1, 40, 200).astype(float), *c)
        assert set(np.unique(t)) <= {1, 2, 3, 4} and t.size == 200
        assert (se.asignar_tramos(np.array([c[2]]), *c) == 4).all() and (se.asignar_tramos(np.array([c[0]]), *c) <= 2).all()


def _panel_con_estancias(panel, estancias, expuestos_por_tramo=None):
    """Panel de ``len(estancias)`` egresos de 2019 a 2022, todos sobrevivientes de la cohorte E, con las estadías dadas; ELIX_01 expuesta según ``expuestos_por_tramo``."""
    n = len(estancias)
    datos = panel.datos.head(n).with_columns(
        pl.Series("anio", [2019 + (i % 4) for i in range(n)], dtype=pl.Int32), pl.Series("ES_COHORTE_E", [1] * n, dtype=pl.Int8), pl.Series("ESTANCIA_DIAS", estancias, dtype=pl.Float64),
    )
    if expuestos_por_tramo is not None:
        datos = datos.with_columns(pl.Series("ELIX_01", expuestos_por_tramo, dtype=pl.Int8))
    return dataclasses.replace(panel, datos=datos, anio_confirmacion=None)


def test_sensibilidad_con_el_ejemplo_a_mano_cortes_tramos_prevalencias_y_razon_de_katz(contexto, panel):
    estancias = [1.0] * 5 + [2.0] * 5 + [3.0] * 5 + [4.0] * 5
    expuestos = [1, 1, 1, 0, 0] + [1, 1, 0, 0, 0] + [1, 1, 1, 0, 0] + [1] * 5                                      # estadías 1, 2, 3 y 4: k por tramo 3, 2, (T3 vacío) y 3 + 5 = 8 (T4 = estadías 3 y 4)
    p = _panel_con_estancias(panel, estancias, expuestos)
    t = se.tabla_sensibilidad(contexto, p, se.tabla_or_elix(contexto, p))
    f = _fila(t, elix_id="ELIX_01")
    assert (f["corte_p25"], f["corte_p50"], f["corte_p75"]) == (1.0, 2.0, 3.0) and t["corte_p75"].n_unique() == 1          # los cortes se repiten en las 31 filas
    assert [f[f"n_t{k}"] for k in (1, 2, 3, 4)] == [5, 5, 0, 10] and [f[f"k_t{k}"] for k in (1, 2, 3, 4)] == [3, 2, 0, 8]
    assert (f["prev_t1"], f["prev_t2"], f["prev_t3"], f["prev_t4"]) == (0.6, 0.4, None, 0.8)
    rr = 0.8 / 0.6
    s = np.sqrt(1 / 8 - 1 / 10 + 1 / 3 - 1 / 5)
    assert f["razon_prev_t4_t1"] == pytest.approx(rr) and (f["razon_bajo"], f["razon_alto"]) == pytest.approx((rr * np.exp(-1.959964 * s), rr * np.exp(1.959964 * s)))
    assert (f["razon_prev_t4_t1"], f["razon_bajo"], f["razon_alto"]) == pytest.approx(est.razon_de_proporciones(8, 10, 3, 5))
    assert f["indice_complicacion"] is None and f["en_top5"] is None                                              # con tan pocos expuestos en T1 no hay índice (celda mínima)


def test_los_tramos_suman_el_total_de_la_cohorte_e_y_los_expuestos(contexto, panel, sensibilidad):
    d = panel.ventana(panel.anios_ranking)
    en_e = d.filter(pl.col("ES_COHORTE_E") == 1)
    for f in sensibilidad.iter_rows(named=True):
        assert sum(f[f"n_t{k}"] for k in (1, 2, 3, 4)) == en_e.height, f["elix_id"]
        assert sum(f[f"k_t{k}"] for k in (1, 2, 3, 4)) == int(en_e[f["elix_id"]].sum()), f["elix_id"]
    cuentas = [sensibilidad["n_t1"][0], sensibilidad["n_t2"][0], sensibilidad["n_t3"][0], sensibilidad["n_t4"][0]]
    assert all(sensibilidad[f"n_t{k}"].n_unique() == 1 for k in (1, 2, 3, 4)) and sum(cuentas) == en_e.height
    estancia = en_e["ESTANCIA_DIAS"].to_numpy()
    assert sensibilidad["corte_p75"][0] == np.sort(estancia)[int(np.ceil(0.75 * estancia.size)) - 1]            # percentil por frecuencia acumulada


def test_la_cohorte_e_vacia_deja_todas_las_categorias_sin_expuestos(contexto, panel):
    p = dataclasses.replace(panel, datos=panel.datos.with_columns(pl.lit(0, dtype=pl.Int8).alias("ES_COHORTE_E")))
    t = se.tabla_sensibilidad(contexto, p, None)
    assert t.height == 31 and set(t["estado"]) == {"sin expuestos"} and set(t["n_t1"]) == {0} and t["corte_p25"].null_count() == 31 and t["indice_complicacion"].null_count() == 31


def test_estadia_sin_variacion_no_tiene_estadia_larga_que_comparar(contexto, panel):
    p = _panel_con_estancias(panel, [3.0] * 40)
    t = se.tabla_sensibilidad(contexto, p, None)
    f = _fila(t, elix_id="ELIX_01")
    assert f["estado"] in ("sin variación de estadía", "sin expuestos") and f["or_larga"] is None and (f["n_t1"], f["n_t2"], f["n_t3"], f["n_t4"]) == (0, 0, 0, 40)


def test_el_gradiente_mas_fuerte_queda_primero_en_el_indice_con_un_panel_mayor(contexto):
    p = panel_artificial(n=6000)
    t = se.tabla_sensibilidad(contexto, p, se.tabla_or_elix(contexto, p))
    primeras = t.filter(pl.col("rango_indice").is_not_null()).sort("rango_indice")
    assert primeras["elix_id"].to_list()[:2] == ["ELIX_22", "ELIX_25"]                                          # las dos con gradiente de estadía de diseño (multiplicadores 2 y 1,7)
    assert primeras["indice_complicacion"][0] == 1.0 and primeras["en_top5"].to_list()[:5] == ["si"] * 5 and set(primeras["en_top5"].to_list()[5:]) == {"no"}
    f = _fila(t, elix_id="ELIX_22")
    assert f["razon_prev_t4_t1"] > 5.0 and f["or_larga"] > 3.0 and f["razon_bajo"] > 1.0 and f["or_larga_bajo"] > 1.0 and f["sospechosa_fase1"] == "si" and f["en_lista_repositorio"] == "si"


def test_el_indice_solo_existe_con_al_menos_celda_minima_expuestos_en_t1_y_t4_y_estado_ok(sensibilidad):
    for f in sensibilidad.iter_rows(named=True):
        deberia = f["estado"] == "ok" and f["k_t1"] >= 10 and f["k_t4"] >= 10
        assert (f["indice_complicacion"] is not None) == deberia == (f["rango_indice"] is not None) == (f["en_top5"] is not None), f["elix_id"]
    con = sensibilidad.filter(pl.col("indice_complicacion").is_not_null())
    assert con.height >= 5 and con["rango_indice"].sort().to_list() == list(range(1, con.height + 1)) and (con["en_top5"] == "si").sum() == min(5, con.height)
    assert 0.0 <= con["indice_complicacion"].min() and con["indice_complicacion"].max() <= 1.0                  # reescalado a [0, 1]: los extremos se alcanzan solo si no hay empates en el rango promedio
    por_rango = con.sort("rango_indice")["indice_complicacion"].to_list()
    assert all(a >= b for a, b in zip(por_rango, por_rango[1:]))                                               # el rango sigue al índice, de mayor a menor


def test_el_resumen_de_elixhauser_cuenta_las_coincidencias_con_las_dos_listas(or_elix, sensibilidad):
    r = se.resumen_elixhauser(or_elix, sensibilidad)
    top = sensibilidad.filter(pl.col("en_top5") == "si").sort("rango_indice")["elix_id"].to_list()
    assert r["elix_top5"] == "; ".join(top) and len(top) == 5
    assert r["elix_n_coinciden_fase1"] == len(set(top) & {"ELIX_04", "ELIX_22", "ELIX_24", "ELIX_25", "ELIX_26"})
    assert r["elix_n_coinciden_repo"] == len(set(top) & set(se.LISTA_REPOSITORIO)) and set(se.LISTA_REPOSITORIO) == {"ELIX_02", "ELIX_22", "ELIX_24", "ELIX_25", "ELIX_26"}
    assert r["elix_n_or_relevante_ajustado"] == or_elix.filter((pl.col("periodo") == "2019-2022") & (pl.col("or_relevante_ajustado") == "si")).height
    vacio = se.resumen_elixhauser(or_elix, sensibilidad.with_columns(pl.lit(None, dtype=pl.String).alias("en_top5"), pl.lit(None, dtype=pl.Int64).alias("rango_indice")))
    assert vacio["elix_top5"] == "" and vacio["elix_n_coinciden_fase1"] == 0 and vacio["elix_n_con_indice"] == 0


# ---------------------------------------------------------------------------------------------------------- índice de complicación (MEDIUM 2)
def test_el_indice_a_mano_con_cuatro_categorias():
    ids = ["ELIX_01", "ELIX_02", "ELIX_03", "ELIX_04"]
    r = se.indice_complicacion([2.0, 1.5, 1.2, 1.0], [1.4, 1.6, 1.1, 1.0], ids, [True] * 4)
    # rangos de mayor a menor: razon_bajo 1, 2, 3, 4 y or_larga_bajo 2, 1, 3, 4; promedios 1,5; 1,5; 3; 4; índice 1 - (p - 1) / 3
    assert r["indice"] == pytest.approx([1 - 0.5 / 3, 1 - 0.5 / 3, 1 - 2 / 3, 0.0])
    assert r["rango"] == [2, 1, 3, 4]                                                                            # el empate 1-2 lo decide el rango de or_larga_bajo (ELIX_02 es mejor)
    assert r["en_top5"] == ["si"] * 4


def test_tres_categorias_empatadas_en_las_posiciones_4_a_6_dan_exactamente_cinco_en_top5_mutante_m53():
    ids = list("ABCDEFGH")
    valores = [10.0, 9.0, 8.0, 5.0, 5.0, 5.0, 2.0, 1.0]                                                          # D, E y F empatan en las dos cifras: rango promedio 5
    r = se.indice_complicacion(valores, valores, ids, [True] * 8)
    assert r["rango"] == [1, 2, 3, 4, 5, 6, 7, 8] and r["en_top5"] == ["si"] * 5 + ["no"] * 3          # con rango "mínimo" sin desempate saldrían seis marcadas
    assert r["indice"][3] == r["indice"][4] == r["indice"][5]
    for semilla in range(5):                                                                                     # la misma selección en cualquier orden de entrada
        orden = np.random.default_rng(semilla).permutation(8)
        r2 = se.indice_complicacion([valores[i] for i in orden], [valores[i] for i in orden], [ids[i] for i in orden], [True] * 8)
        elegidas = {ids[orden[j]] for j, m in enumerate(r2["en_top5"]) if m == "si"}
        assert elegidas == set("ABCDE") and sorted(r2["rango"]) == list(range(1, 9))
    assert se.indice_complicacion(valores, valores, ids, [True] * 8) == r                                       # y repetible


def test_el_desempate_usa_or_larga_bajo_luego_razon_bajo_y_luego_el_id():
    # mismos promedios: A (rangos 1 y 3), B (2 y 2), C (3 y 1) -> promedio 2 para los tres
    r = se.indice_complicacion([3.0, 2.0, 1.0], [1.0, 2.0, 3.0], ["A", "B", "C"], [True] * 3)
    assert r["indice"] == [0.5, 0.5, 0.5] and r["rango"] == [3, 2, 1]                                            # gana el mejor rango de or_larga_bajo (C), luego B, luego A
    r = se.indice_complicacion([3.0, 3.0], [3.0, 3.0], ["Z", "A"], [True, True])
    assert r["rango"] == [2, 1]                                                                                  # todo empatado: el id ascendente


def test_con_menos_de_dos_categorias_con_indice_los_campos_quedan_nulos_sin_excepcion():
    uno = se.indice_complicacion([2.0, 1.0], [2.0, 1.0], ["A", "B"], [True, False])
    assert uno == {"indice": [None, None], "rango": [None, None], "en_top5": [None, None]}                       # m = 1
    cero = se.indice_complicacion([2.0, 1.0], [2.0, 1.0], ["A", "B"], [False, False])
    assert cero == {"indice": [None, None], "rango": [None, None], "en_top5": [None, None]}                      # m = 0
    assert se.indice_complicacion([], [], [], []) == {"indice": [], "rango": [], "en_top5": []}
    nulos = se.indice_complicacion([None, 2.0, 1.0], [1.0, None, 3.0], ["A", "B", "C"], [True, True, True])
    assert nulos["en_top5"] == [None, None, None]                                                                # una cifra nula deja fuera de la escala
    with pytest.raises(ValueError, match="largos distintos"):
        se.indice_complicacion([1.0], [1.0, 2.0], ["A"], [True])


def test_con_menos_de_cinco_categorias_todas_van_al_top():
    r = se.indice_complicacion([5.0, 4.0, 3.0], [5.0, 4.0, 3.0], ["A", "B", "C"], [True] * 3)
    assert r["en_top5"] == ["si", "si", "si"] and r["rango"] == [1, 2, 3]


def test_el_top5_de_la_tabla_en_dos_corridas_es_el_mismo(contexto, panel, or_elix, sensibilidad):
    otra = se.tabla_sensibilidad(contexto, panel, or_elix)
    assert otra.equals(sensibilidad)


# ----------------------------------------------------------------------------------------------------------------------------- figura
@pytest.fixture()
def contexto_con_or(tmp_path, panel):
    ctx = Contexto.crear("extracto", fuente=crear_fuente_sintetica(tmp_path / "datos", n=60, semilla=29), dir_salida_base=tmp_path / "salidas")
    grande = panel_artificial(n=6000)
    pn.escribir_por_bloques(ctx, "t23_or_elix.csv", se.tabla_or_elix(ctx, grande), se.BLOQUES_OR_ELIX, tarea="T11", descripcion="or")
    return ctx


def test_figura_or_elix_se_arma_solo_con_su_csv_y_marca_las_cinco_sospechosas(contexto_con_or, monkeypatch):
    from matplotlib.figure import Figure

    ctx = contexto_con_or
    for otro in ctx.dir_tablas.glob("*.csv"):
        if otro.name != "t23_or_elix.csv":
            otro.unlink()
    capturadas = []
    original = Figure.savefig
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: (capturadas.append(self), original(self, *a, **k))[1])
    assert se.figura_or_elix(ctx) is True
    ax = capturadas[0].axes[0]
    etiquetas = [t.get_text() for t in ax.get_yticklabels()]
    assert len(etiquetas) == 31 and {e.split()[0] for e in etiquetas if "*" in e.split()[0]} == {"ELIX_04*", "ELIX_22*", "ELIX_24*", "ELIX_25*", "ELIX_26*"}
    tabla = ctx.leer_tabla("t23_or_elix.csv", columnas_texto=se._TEXTO_OR).filter(pl.col("periodo") == "2019-2022")
    con_or = tabla.filter(pl.col("or_ajustado").is_not_null()).sort("or_ajustado", descending=True)
    assert [e.split()[0].rstrip("*") for e in etiquetas[: con_or.height]] == con_or["elix_id"].to_list()          # de mayor a menor OR ajustado, de arriba abajo
    assert all(e.split()[0].rstrip("*") in set(tabla.filter(pl.col("or_ajustado").is_null())["elix_id"]) for e in etiquetas[con_or.height:])        # y abajo las que no lo tienen
    assert ax.get_xscale() == "log" and any(abs(v - 1.0) < 1e-12 for line in ax.lines for v in np.atleast_1d(line.get_xdata()))
    e = ctx.figuras.indice().row(0, named=True)
    assert (e["id_figura"], e["csv"], e["decision_relacionada"], e["titulo"]) == ("F11_04", "t23_or_elix.csv", "8", "Odds ratio de las categorías de Elixhauser, crudo y ajustado (2019 a 2022)")


def test_figura_or_elix_rotula_n_d_las_categorias_sin_cifras(contexto_con_or, monkeypatch):
    from matplotlib.figure import Figure

    ctx = contexto_con_or
    capturadas = []
    original = Figure.savefig
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: (capturadas.append(self), original(self, *a, **k))[1])
    se.figura_or_elix(ctx)
    t = ctx.leer_tabla("t23_or_elix.csv", columnas_texto=se._TEXTO_OR).filter(pl.col("periodo") == "2019-2022")
    sin_cifras = t.filter(pl.col("or_crudo").is_null() & pl.col("or_ajustado").is_null()).height
    assert sin_cifras >= 1 and sum(1 for x in capturadas[0].axes[0].texts if x.get_text() == "n/d") == sin_cifras


def test_figura_or_elix_sin_filas_se_omite(contexto_con_or):
    ctx = contexto_con_or
    t = ctx.leer_tabla("t23_or_elix.csv", columnas_texto=se._TEXTO_OR).filter(pl.col("periodo") == "2023")
    t.write_csv(ctx.dir_tablas / "t23_or_elix.csv")
    assert se.figura_or_elix(ctx) is False and ctx.figuras.ids() == []


# FIN-test_t11_evidencias
