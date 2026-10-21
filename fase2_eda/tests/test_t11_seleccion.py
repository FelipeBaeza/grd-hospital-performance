"""Tarea 11, ``eda/seleccion.py``: ranking, estabilidad, importancia, parámetros técnicos y sus figuras (parte 1); matriz, resumen, informe e integración (parte 2).

Las cuentas a mano y los oráculos (scikit-learn, scipy, numpy) fijan cada pieza; el panel artificial aporta la verdad de diseño. Las pruebas numéricas leen los resultados en
memoria (sin suprimir); las de formato y privacidad leen los CSV. Cubre los mutantes M25, M26, M27, M29, M30 y M46.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score

from eda import asociacion as asoc
from eda import config, decision, modelos, privacidad, seleccion, sospechas, supresion
from eda import estadistica as est
from eda import panel as pn
from eda.contexto import Contexto
from panel_artificial import panel_artificial
from sintetico import crear_fuente_sintetica

SEMILLA = 20261007
sel = seleccion


@pytest.fixture(scope="module")
def u():
    """Umbrales del proyecto. Las cuentas a mano de este archivo suponen los valores propuestos; si cambian, la prueba lo dice."""
    u = config.cargar_umbrales()
    supuestos = {
        "auc_senal_alta": 0.70, "auc_senal_media": 0.60, "auc_senal_baja": 0.55, "faltante_alerta": 0.20, "faltante_alto": 0.50, "faltante_informativo_min_prop": 0.01,
        "faltante_informativo_or": 1.5, "estabilidad_efecto_minimo": 0.01, "psi_moderado": 0.10, "psi_alto": 0.25, "importancia_permutacion_min": 0.001, "or_relevante": 1.2,
        "celda_minima": 10,
    }
    for clave, valor in supuestos.items():
        assert u[clave] == valor, f"las cuentas a mano de este archivo suponen {clave} = {valor}"
    return u


@pytest.fixture(scope="module")
def contexto(tmp_path_factory, u):
    """Contexto sobre una fuente sintética mínima: las funciones de la Tarea 11 reciben el panel y nunca leen."""
    raiz = tmp_path_factory.mktemp("t11_seleccion")
    return Contexto.crear("extracto", fuente=crear_fuente_sintetica(raiz / "datos", n=60, semilla=29), dir_salida_base=raiz / "salidas")


@pytest.fixture(scope="module")
def panel():
    return panel_artificial()


@pytest.fixture(scope="module")
def ranking(contexto, panel):
    return sel.tabla_ranking(contexto, panel)


@pytest.fixture(scope="module")
def estabilidad(contexto, panel, ranking):
    return sel.tabla_estabilidad(contexto, panel, ranking)


@pytest.fixture(scope="module")
def clase_proxy(contexto, panel, ranking):
    t = sospechas.tabla_proxy(contexto, panel, ranking).filter(pl.col("tipo_fila") == "variable")
    return dict(zip(t["variable"], t["clase_proxy"]))


@pytest.fixture(scope="module")
def referencias(panel):
    return modelos.referencias(panel, SEMILLA)


@pytest.fixture(scope="module")
def importancia(contexto, panel, referencias, clase_proxy):
    return sel.tabla_importancia(contexto, panel, referencias, clase_proxy)


def _con_columna(panel, nombre, valores, dtype=None):
    return dataclasses.replace(panel, datos=panel.datos.with_columns(pl.Series(nombre, list(valores), dtype=dtype or panel.datos.schema[nombre])))


def _fila(tabla, **filtro):
    sel_ = tabla
    for k, v in filtro.items():
        sel_ = sel_.filter(pl.col(k) == v)
    assert sel_.height == 1, (filtro, sel_.height)
    return sel_.row(0, named=True)


# =========================================================================================================================== ranking
def test_las_columnas_del_ranking_son_las_del_diseno_con_n_faltante_antes_de_prop_faltante(ranking):
    esperadas = [
        "variable", "tipo", "rol_fase1", "ventana", "n_filas_ventana", "n", "n_eventos", "n_no_eventos", "n_faltante", "prop_faltante", "clase_faltante", "auc", "auc_bajo", "auc_alto",
        "direccion", "auc_optimista", "auc_cv", "auc_senal", "auc_senal_bajo", "auc_senal_alto", "clase_senal", "posicion", "v_resultado", "medida_resultado", "info_mutua_norm",
        "n_expuestos", "n_eventos_expuestos", "n_no_expuestos", "n_eventos_no_expuestos", "or_crudo", "or_bajo", "or_alto", "or_relevante", "or_faltante", "or_faltante_bajo",
        "or_faltante_alto", "faltante_informativo", "estado",
    ]
    assert ranking.columns == sel.COLUMNAS_RANKING == esperadas and len(esperadas) == 38


def test_hay_una_fila_por_variable_y_ventana_y_etnia_y_pueblo_originario_no_estan(ranking, panel):
    variables = modelos.variables_ranking(panel)
    assert len(variables) == 56 and "ETNIA" not in variables and "PUEBLO_ORIGINARIO" not in variables
    assert ranking.height == 112 and set(ranking["ventana"]) == {"2019-2022", "2023"}
    assert ranking.group_by("variable", maintain_order=True).len()["len"].unique().to_list() == [2]
    assert ranking.filter(pl.col("ventana") == "2019-2022")["variable"].to_list() == variables
    assert "COD_HOSPITAL" not in ranking["variable"].to_list() and "IR_29301_MORTALIDAD" not in ranking["variable"].to_list()


def test_efecto_invertido_da_direccion_menos_y_auc_de_al_menos_un_medio(contexto, panel):
    invertida = _con_columna(panel, "EDAD_ANIOS", (-panel.datos["EDAD_ANIOS"]).to_list())
    f = _fila(sel.tabla_ranking(contexto, invertida), variable="EDAD_ANIOS", ventana="2019-2022")
    assert f["direccion"] == "-" and f["auc"] >= 0.5 and f["auc"] > 0.65 and f["auc_bajo"] <= f["auc"] <= f["auc_alto"]
    g = _fila(sel.tabla_ranking(contexto, invertida), variable="EDAD_ANIOS", ventana="2023")
    assert g["auc"] > 0.5                                                                                     # el efecto sigue invertido en 2023: con la dirección del ranking queda sobre un medio


def test_si_el_efecto_se_invierte_en_2023_el_auc_de_confirmacion_queda_bajo_un_medio_y_hay_cambio_de_signo_mutante_m25(ranking, estabilidad):
    r23 = _fila(ranking, variable="ELIX_02", ventana="2023")
    r = _fila(ranking, variable="ELIX_02", ventana="2019-2022")
    assert r["direccion"] == "+" and r["auc"] > 0.5 and r["auc_senal"] == r["auc"] and r["auc_optimista"] == "no"
    assert r23["auc"] < 0.5 and r23["direccion"] is None                                                         # no se reorienta con 2023 (orientar con su propio máximo lo ocultaría)
    assert _fila(estabilidad, tipo_fila="resumen", variable="ELIX_02")["cambio_signo"] == "si"
    anual = estabilidad.filter((pl.col("tipo_fila") == "auc_anual") & (pl.col("variable") == "ELIX_02")).sort("anio")
    assert anual["signo"].to_list()[-1] == "-" and anual["efecto_suficiente"].to_list()[-1] == "si"


def test_el_ranking_usa_solo_2019_a_2022_para_fijar_la_direccion_y_el_puntaje_mutante_m26(contexto, panel):
    """Mutante M26: ranking con 2019 a 2023. Si 2023 cambia la señal, la dirección y el AUC de la ventana de ranking no deben moverse."""
    anio = panel.datos["anio"].to_numpy()
    edad = panel.datos["EDAD_ANIOS"].to_numpy()
    cambiada = np.where(anio == 2023, 200.0 - edad, edad)                                                         # en 2023 la edad pasa a ser protectora
    p = _con_columna(panel, "EDAD_ANIOS", cambiada.tolist())
    a, b = sel.tabla_ranking(contexto, panel), sel.tabla_ranking(contexto, p)
    ra, rb = _fila(a, variable="EDAD_ANIOS", ventana="2019-2022"), _fila(b, variable="EDAD_ANIOS", ventana="2019-2022")
    assert (ra["auc"], ra["direccion"], ra["n"], ra["n_eventos"]) == (rb["auc"], rb["direccion"], rb["n"], rb["n_eventos"])
    assert _fila(b, variable="EDAD_ANIOS", ventana="2023")["auc"] < 0.45 and _fila(a, variable="EDAD_ANIOS", ventana="2023")["auc"] > 0.55      # 2023 sí se ve invertido, con la dirección del ranking


def test_las_categoricas_son_optimistas_en_la_ventana_de_ranking_y_el_auc_senal_es_el_de_validacion_cruzada(ranking):
    cat = ranking.filter((pl.col("tipo") == "cat") & (pl.col("ventana") == "2019-2022") & (pl.col("estado") == "ok"))
    assert cat.height >= 10
    for f in cat.iter_rows(named=True):
        assert f["auc_optimista"] == "si" and f["auc_cv"] is not None and f["auc"] >= f["auc_cv"] - 1e-12 and f["auc_senal"] == f["auc_cv"], f["variable"]
    nocat = ranking.filter((pl.col("tipo") != "cat") & (pl.col("ventana") == "2019-2022") & (pl.col("estado") == "ok"))
    assert (nocat["auc_optimista"] == "no").all() and (nocat["auc_cv"] == nocat["auc"]).all() and (nocat["auc_senal"] == nocat["auc"]).all()
    assert set(ranking.filter(pl.col("ventana") == "2023")["auc_optimista"].drop_nulls()) == set()


def test_auc_senal_bajo_y_alto_son_el_ic_de_hanley_y_mcneil_del_auc_cv_con_los_conteos_de_la_ventana_mutante_m46(ranking):
    cat = ranking.filter((pl.col("tipo") == "cat") & (pl.col("ventana") == "2019-2022") & (pl.col("estado") == "ok"))
    distintos = 0
    for f in cat.iter_rows(named=True):
        bajo, alto = est.ic_hanley_mcneil(f["auc_cv"], f["n_eventos"], f["n_no_eventos"])
        assert (f["auc_senal_bajo"], f["auc_senal_alto"]) == pytest.approx((bajo, alto), abs=1e-12), f["variable"]
        distintos += (f["auc_senal_bajo"], f["auc_senal_alto"]) != pytest.approx((f["auc_bajo"], f["auc_alto"]), abs=1e-9)           # no son los del AUC en muestra
    assert distintos >= 8


def test_auc_senal_bajo_menor_o_igual_que_auc_senal_menor_o_igual_que_auc_senal_alto_en_toda_fila(contexto, panel, ranking):
    completas = ranking.filter(pl.col("auc_senal").is_not_null() & pl.col("auc_senal_bajo").is_not_null() & pl.col("auc_senal_alto").is_not_null())
    assert completas.height >= 40 and (completas["auc_senal_bajo"] <= completas["auc_senal"]).all() and (completas["auc_senal"] <= completas["auc_senal_alto"]).all()
    un_ano = dataclasses.replace(panel, datos=panel.datos.filter(pl.col("anio").is_in([2019, 2023])), anios_ranking=[2019])
    t = sel.tabla_ranking(contexto, un_ano).filter(pl.col("ventana") == "2019-2022")
    cat = t.filter((pl.col("tipo") == "cat") & (pl.col("estado") == "ok"))
    assert cat.height >= 10 and cat["auc_cv"].null_count() == cat.height                                      # un solo año de ranking: sin validación cruzada
    assert (cat["auc_senal"] == cat["auc"]).all() and (cat["auc_senal_bajo"] == cat["auc_bajo"]).all() and (cat["auc_senal_alto"] == cat["auc_alto"]).all()
    assert (cat["auc_optimista"] == "si").all()                                                                 # y la fila lo dice


def test_el_auc_numerico_y_binario_coincide_con_scikit_learn_y_la_direccion_es_la_del_auc_bruto(ranking, panel):
    d = panel.ventana(panel.anios_ranking)
    for nombre in ("EDAD_ANIOS", "N_DX_SEC", "ELIX_01", "NIVEL_CUIDADO_INGRESO"):
        f = _fila(ranking, variable=nombre, ventana="2019-2022")
        bruto = roc_auc_score(d["y"], d[nombre].fill_null(0))
        esperado = bruto if bruto >= 0.5 else 1 - bruto
        assert f["auc"] == pytest.approx(esperado, abs=1e-12) and f["direccion"] == ("+" if bruto >= 0.5 else "-")
        assert (f["n_filas_ventana"], f["n"], f["n_eventos"], f["n_no_eventos"]) == (960, 960, int(d["y"].sum()), 960 - int(d["y"].sum()))


def test_clases_de_senal_en_los_bordes_exactos_0_55_0_60_y_0_70(u):
    c = lambda x: sel.clase_senal(x, u)  # noqa: E731
    assert (c(0.70), c(0.6999996), c(0.699999)) == ("alta", "alta", "media")
    assert (c(0.60), c(0.5999996), c(0.599999)) == ("media", "media", "débil")
    assert (c(0.55), c(0.5499996), c(0.549999)) == ("débil", "débil", "baja")
    assert c(0.5) == "baja" and c(None) is None and c(float("nan")) is None


def test_clase_de_faltante_en_los_bordes_0_20_y_0_50(u):
    c = lambda x: sel.clase_faltante(x, u)  # noqa: E731
    assert (c(0.50), c(0.499999), c(0.4999996)) == ("alto", "alerta", "alto")
    assert (c(0.20), c(0.199999), c(0.1999996)) == ("alerta", "bajo", "alerta")
    assert c(0.0) == "bajo" and c(None) == "no evaluable"


def test_prop_faltante_es_una_fraccion_y_no_un_porcentaje(contexto, panel):
    n = panel.datos.height
    nulos = [None if i % 5 == 0 else v for i, v in enumerate(panel.datos["N_DX_SEC"].to_list())]                  # 20 % exacto de nulos
    t = sel.tabla_ranking(contexto, _con_columna(panel, "N_DX_SEC", nulos))
    f = _fila(t, variable="N_DX_SEC", ventana="2019-2022")
    assert f["prop_faltante"] == pytest.approx(0.2) and f["n_faltante"] == 192 and f["n"] == 960 - 192 and f["n_filas_ventana"] == 960         # 0,2 y no 20
    assert f["clase_faltante"] == "alerta"
    assert 0.0 <= t["prop_faltante"].drop_nulls().min() and t["prop_faltante"].drop_nulls().max() <= 1.0
    cat = _fila(t, variable="SERVICIOINGRESO", ventana="2019-2022")
    assert cat["prop_faltante"] == pytest.approx(cat["n_faltante"] / 960) and 0.45 < cat["prop_faltante"] < 0.65 and cat["n"] == 960        # la categórica falta si su nivel es "(sin dato)"


def test_faltante_informativo_exige_proporcion_y_celdas_suficientes(contexto, panel):
    anio, y = panel.datos["anio"].to_numpy(), panel.datos["y"].to_numpy()
    edad = panel.datos["EDAD_ANIOS"].to_list()
    rng = np.random.default_rng(12)
    # la falta se asocia con el evento: 40 % de los eventos y 10 % de los no eventos tienen nulo -> OR de unas 6 veces
    falta = np.where(y == 1, rng.random(y.size) < 0.40, rng.random(y.size) < 0.10)
    t = sel.tabla_ranking(contexto, _con_columna(panel, "EDAD_ANIOS", [None if f else v for v, f in zip(edad, falta)]))
    f = _fila(t, variable="EDAD_ANIOS", ventana="2019-2022")
    assert f["faltante_informativo"] == "si" and f["or_faltante"] > 3.0 and f["or_faltante_bajo"] > 1.0 and f["or_faltante_bajo"] < f["or_faltante"] < f["or_faltante_alto"]
    sin_asociacion = np.random.default_rng(13).random(y.size) < 0.25
    g = _fila(sel.tabla_ranking(contexto, _con_columna(panel, "EDAD_ANIOS", [None if x else v for v, x in zip(edad, sin_asociacion)])), variable="EDAD_ANIOS", ventana="2019-2022")
    assert g["faltante_informativo"] == "no" and g["or_faltante"] is not None and g["or_faltante_bajo"] < 1.0 < g["or_faltante_alto"]
    muy_poco = np.zeros(y.size, dtype=bool)
    muy_poco[np.flatnonzero(anio <= 2022)[:5]] = True                                                           # 5 nulos en 960 filas: menos del 1 % y menos de 10 por celda
    h = _fila(sel.tabla_ranking(contexto, _con_columna(panel, "EDAD_ANIOS", [None if x else v for v, x in zip(edad, muy_poco)])), variable="EDAD_ANIOS", ventana="2019-2022")
    assert h["faltante_informativo"] == "no evaluable" and h["or_faltante"] is None and h["n_faltante"] == 5
    sin_nulos = _fila(t, variable="ELIX_01", ventana="2019-2022")
    assert sin_nulos["faltante_informativo"] == "no evaluable" and sin_nulos["n_faltante"] == 0


def test_el_conteo_de_faltantes_entre_1_y_9_anula_n_n_faltante_y_prop_faltante_pero_no_el_auc(contexto, panel, u):
    """NIT 7: con 5 nulos en 100.000 filas ``n_filas_ventana - n`` no debe revelar el conteo."""
    fila = {c: None for c in sel.COLUMNAS_RANKING}
    fila.update({
        "variable": "X", "tipo": "num", "rol_fase1": "FEATURE_BASAL", "ventana": "2019-2022", "n_filas_ventana": 100000, "n": 99995, "n_eventos": 9000, "n_no_eventos": 90995, "n_faltante": 5,
        "prop_faltante": 5e-5, "clase_faltante": "bajo", "auc": 0.7, "auc_bajo": 0.69, "auc_alto": 0.71, "auc_senal": 0.7, "faltante_informativo": "no evaluable", "estado": "ok",
    })
    t = pl.DataFrame([fila], schema=sel.ESQUEMA_RANKING)
    pub, n_celdas = supresion.suprimir_bloques(t, sel.BLOQUES_RANKING, minimo=10)
    g = pub.row(0, named=True)
    assert n_celdas == 1 and g["n_faltante"] is None and g["n"] is None and g["prop_faltante"] is None and g["clase_faltante"] is None and g["faltante_informativo"] is None
    assert g["n_filas_ventana"] == 100000 and g["n_eventos"] == 9000 and g["auc"] == 0.7 and g["auc_senal"] == 0.7                  # el resto de la fila se conserva
    grande = t.with_columns(pl.lit(500).alias("n_faltante"))
    assert supresion.suprimir_bloques(grande, sel.BLOQUES_RANKING, minimo=10)[0].row(0, named=True)["n"] == 99995


def test_un_conteo_pequeno_del_cuadro_de_una_binaria_anula_sus_cifras_pero_no_las_de_otras_filas(ranking):
    pub, _ = supresion.suprimir_bloques(ranking, sel.BLOQUES_RANKING, minimo=10)
    chicas = ranking.filter((pl.col("tipo") == "bin") & (pl.col("n_eventos_expuestos") > 0) & (pl.col("n_eventos_expuestos") < 10))
    assert chicas.height >= 5
    f = chicas.row(0, named=True)
    g = _fila(pub, variable=f["variable"], ventana=f["ventana"])
    assert g["n_eventos_expuestos"] is None and g["auc"] is None and g["or_crudo"] is None and g["posicion"] is None and g["estado"] == f["estado"] and g["tipo"] == f["tipo"]
    assert _fila(pub, variable="EDAD_ANIOS", ventana="2019-2022")["auc"] is not None


def test_la_fila_de_2023_trae_solo_los_conteos_el_faltante_el_auc_con_su_ic_y_v_resultado(ranking):
    f = _fila(ranking, variable="EDAD_ANIOS", ventana="2023")
    llenas = {k for k, v in f.items() if v is not None}
    assert llenas == {"variable", "tipo", "rol_fase1", "ventana", "n_filas_ventana", "n", "n_eventos", "n_no_eventos", "n_faltante", "prop_faltante", "auc", "auc_bajo", "auc_alto", "v_resultado", "medida_resultado", "estado"}
    assert (f["n_filas_ventana"], f["n"], f["n_eventos"]) == (240, 240, 30) and f["auc_bajo"] <= f["auc"] <= f["auc_alto"] and f["estado"] == "ok"


def test_posicion_ordena_por_auc_senal_descendente_y_en_empate_por_nombre(ranking):
    r = ranking.filter((pl.col("ventana") == "2019-2022") & pl.col("posicion").is_not_null()).sort("posicion")
    assert r["posicion"].to_list() == list(range(1, r.height + 1))
    claves = [(-round(a, 6), v) for a, v in zip(r["auc_senal"], r["variable"])]
    assert claves == sorted(claves)
    assert r["variable"][0] == "EDAD_ANIOS" and ranking.filter((pl.col("ventana") == "2023")).get_column("posicion").null_count() == 112 // 2
    assert _fila(ranking, variable="ELIX_28", ventana="2019-2022")["posicion"] is None


def test_una_variable_constante_queda_en_estado_constante_sin_cifras_de_asociacion(ranking):
    f = _fila(ranking, variable="ELIX_28", ventana="2019-2022")
    assert f["estado"] == "constante" and all(f[c] is None for c in ("auc", "auc_bajo", "auc_alto", "auc_senal", "clase_senal", "v_resultado", "info_mutua_norm", "or_crudo", "direccion"))
    assert f["n"] == 960 and f["n_expuestos"] == 0 and f["n_eventos_expuestos"] == 0 and f["clase_faltante"] == "bajo"
    assert _fila(ranking, variable="ELIX_28", ventana="2023")["estado"] == "constante"


def test_v_resultado_e_informacion_mutua_coinciden_con_los_oraculos(ranking, panel):
    d = panel.ventana(panel.anios_ranking)
    cols = asoc.preparar_columnas(panel, panel.anios_ranking)
    for nombre in ("EDAD_ANIOS", "ELIX_01", "GRUPO_CLINICO"):
        f = _fila(ranking, variable=nombre, ventana="2019-2022")
        v, medida, _ = asoc.con_resultado(cols[nombre], d["y"].to_numpy())
        assert f["v_resultado"] == pytest.approx(v, abs=1e-12) and f["medida_resultado"] == medida
        codigos = sel._codigos_info_mutua(panel, nombre, panel.anios_ranking)
        assert f["info_mutua_norm"] == pytest.approx(normalized_mutual_info_score(codigos, d["y"].to_numpy()), abs=1e-9)
    assert _fila(ranking, variable="EDAD_ANIOS", ventana="2019-2022")["medida_resultado"] == "eta" and _fila(ranking, variable="GRUPO_CLINICO", ventana="2019-2022")["medida_resultado"] == "v_cramer_bergsma"


def test_la_informacion_mutua_usa_deciles_de_la_ventana_de_ranking_y_el_nulo_como_bin_propio(panel):
    cod = sel._codigos_info_mutua(panel, "N_DX_SEC", panel.anios_ranking)
    x = panel.ventana(panel.anios_ranking)["N_DX_SEC"].to_numpy()
    bordes = est.bordes_deciles(x)
    assert cod.tolist() == np.searchsorted(bordes, x, side="left").tolist() and cod.max() <= bordes.size
    nulos = [None if i % 7 == 0 else v for i, v in enumerate(panel.datos["N_DX_SEC"].to_list())]
    con_nulos = sel._codigos_info_mutua(_con_columna(panel, "N_DX_SEC", nulos), "N_DX_SEC", panel.anios_ranking)
    assert (con_nulos == con_nulos.max()).sum() == int(np.ceil(960 / 7)) and con_nulos.max() == est.bordes_deciles(np.array([v for v in nulos[:960] if v is not None], dtype=float)).size + 1
    b = sel._codigos_info_mutua(panel, "ELIX_01", panel.anios_ranking)
    assert set(np.unique(b)) == {0, 1}


def test_las_binarias_traen_el_cuadro_de_dos_por_dos_y_el_or_crudo_con_woolf(ranking, panel, u):
    d = panel.ventana(panel.anios_ranking)
    f = _fila(ranking, variable="ELIX_01", ventana="2019-2022")
    x, y = d["ELIX_01"].to_numpy(), d["y"].to_numpy()
    a, b, c, dd = int(((x == 1) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum()), int(((x == 0) & (y == 0)).sum())
    assert (f["n_expuestos"], f["n_eventos_expuestos"], f["n_no_expuestos"], f["n_eventos_no_expuestos"]) == (a + b, a, c + dd, c)
    o, bajo, alto, _ = est.or_woolf(a, b, c, dd)
    assert (f["or_crudo"], f["or_bajo"], f["or_alto"]) == pytest.approx((o, bajo, alto)) and f["or_relevante"] == modelos.or_es_relevante(o, bajo, alto, 1.2)
    assert _fila(ranking, variable="EDAD_ANIOS", ventana="2019-2022")["n_expuestos"] is None and _fila(ranking, variable="EDAD_ANIOS", ventana="2019-2022")["or_crudo"] is None


def test_sin_confirmacion_el_ranking_no_trae_filas_de_2023(contexto, panel):
    sin = dataclasses.replace(panel, datos=panel.datos.filter(pl.col("anio") <= 2022), anio_confirmacion=None)
    t = sel.tabla_ranking(contexto, sin)
    assert set(t["ventana"]) == {"2019-2022"} and t.height == 56


# ======================================================================================================================== estabilidad
def test_las_columnas_de_la_estabilidad_son_las_del_diseno(estabilidad):
    assert estabilidad.columns == sel.COLUMNAS_ESTABILIDAD == [
        "tipo_fila", "variable", "tipo", "posicion", "anio", "comparacion", "n", "n_eventos", "n_base", "n_nuevo", "auc", "auc_bajo", "auc_alto", "auc_optimista", "efecto", "signo",
        "efecto_suficiente", "cambio_signo", "n_anios_efecto_suficiente", "psi", "clase_psi",
    ]
    assert set(estabilidad["tipo_fila"]) == {"auc_anual", "resumen", "psi"}


def test_etnia_y_pueblo_originario_no_tienen_auc_anual_ni_resumen_pero_si_psi(estabilidad):
    anual = estabilidad.filter(pl.col("tipo_fila").is_in(["auc_anual", "resumen"]))
    assert not anual["variable"].is_in(["ETNIA", "PUEBLO_ORIGINARIO"]).any()
    psi = estabilidad.filter((pl.col("tipo_fila") == "psi") & pl.col("variable").is_in(["ETNIA", "PUEBLO_ORIGINARIO"]))
    assert psi.height == 4 and psi["psi"].null_count() == 0                                                       # dos comparaciones por variable


def test_hay_cinco_anos_por_variable_dos_psi_y_un_resumen(estabilidad):
    anual = estabilidad.filter(pl.col("tipo_fila") == "auc_anual")
    assert anual.height == 56 * 5 and anual.group_by("variable").len()["len"].unique().to_list() == [5] and sorted(anual["anio"].unique().to_list()) == [2019, 2020, 2021, 2022, 2023]
    assert estabilidad.filter(pl.col("tipo_fila") == "resumen").height == 56
    psi = estabilidad.filter(pl.col("tipo_fila") == "psi")
    assert psi.height == 2 * 58 and set(psi["comparacion"]) == {"2019-2022 contra 2023", "2019 contra 2020-2022"}


def test_auc_optimista_solo_en_las_filas_anuales_de_categoricas_de_2019_a_2022(estabilidad):
    anual = estabilidad.filter(pl.col("tipo_fila") == "auc_anual")
    si = anual.filter(pl.col("auc_optimista") == "si")
    assert (si["tipo"] == "cat").all() and set(si["anio"].to_list()) == {2019, 2020, 2021, 2022}
    assert (anual.filter(pl.col("tipo") == "cat").filter(pl.col("anio") == 2023)["auc_optimista"] == "no").all()
    assert (anual.filter(pl.col("tipo") != "cat")["auc_optimista"] == "no").all()
    assert estabilidad.filter(pl.col("tipo_fila") != "auc_anual")["auc_optimista"].null_count() == estabilidad.filter(pl.col("tipo_fila") != "auc_anual").height


def test_el_auc_anual_usa_la_direccion_y_el_puntaje_fijos_del_ranking(estabilidad, panel):
    ajuste = modelos.ajustar_univariado(panel, "GRUPO_CLINICO")
    for anio in (2019, 2023):
        f = _fila(estabilidad, tipo_fila="auc_anual", variable="GRUPO_CLINICO", anio=anio)
        r = modelos.auc_univariado(panel, ajuste, [anio])
        assert f["auc"] == pytest.approx(r.auc, abs=1e-12) and (f["n"], f["n_eventos"]) == (r.n, r.n_eventos)
        assert f["efecto"] == pytest.approx(abs(r.auc - 0.5), abs=1e-6)


def _panel_con_efecto(panel, expuestos_eventos, expuestos_no_eventos, n_por_clase=100):
    """Panel de cinco años idénticos (``2 * n_por_clase`` filas cada uno) donde ELIX_01 tiene la proporción dada entre eventos y no eventos: AUC = 0,5 + 0,5 (p_e - p_n)."""
    n_anio = 2 * n_por_clase
    anio = np.repeat([2019, 2020, 2021, 2022, 2023], n_anio)
    y = np.tile(np.r_[np.ones(n_por_clase), np.zeros(n_por_clase)], 5).astype(np.int8)
    x = np.zeros(anio.size, dtype=np.int8)
    for k in range(5):
        base = k * n_anio
        x[base:base + int(round(expuestos_eventos * n_por_clase))] = 1
        x[base + n_por_clase:base + n_por_clase + int(round(expuestos_no_eventos * n_por_clase))] = 1
    datos = panel.datos.head(anio.size).with_columns(pl.Series("anio", anio, dtype=pl.Int32), pl.Series("y", y, dtype=pl.Int8), pl.Series("ELIX_01", x, dtype=pl.Int8))
    return dataclasses.replace(panel, datos=datos)


def test_efecto_minimo_de_0_01_exacto_cuenta_y_0_005_no_mutante_m27(contexto, panel):
    exacto = _panel_con_efecto(panel, 0.52, 0.50)                                                                  # AUC = 0,51: efecto 0,01 en los cinco años
    t = sel.tabla_estabilidad(contexto, exacto, sel.tabla_ranking(contexto, exacto))
    anual = t.filter((pl.col("tipo_fila") == "auc_anual") & (pl.col("variable") == "ELIX_01"))
    assert anual["auc"].to_list() == pytest.approx([0.51] * 5) and set(anual["efecto"]) == {0.01} and set(anual["efecto_suficiente"]) == {"si"} and set(anual["signo"]) == {"+"}
    r = _fila(t, tipo_fila="resumen", variable="ELIX_01")
    assert r["n_anios_efecto_suficiente"] == 5 and r["cambio_signo"] == "no"
    medio = _panel_con_efecto(panel, 0.51, 0.50)                                                                   # AUC = 0,505: efecto 0,005
    t2 = sel.tabla_estabilidad(contexto, medio, sel.tabla_ranking(contexto, medio))
    anual2 = t2.filter((pl.col("tipo_fila") == "auc_anual") & (pl.col("variable") == "ELIX_01"))
    assert set(anual2["efecto"]) == {0.005} and set(anual2["efecto_suficiente"]) == {"no"}
    r2 = _fila(t2, tipo_fila="resumen", variable="ELIX_01")
    assert r2["n_anios_efecto_suficiente"] == 0 and r2["cambio_signo"] == "no evaluable"                           # sin el efecto mínimo el cambio de signo no se declara


def test_cambio_de_signo_requiere_signos_opuestos_entre_anos_con_efecto_suficiente(contexto, panel):
    p = _panel_con_efecto(panel, 0.60, 0.50)                                                                       # efecto +0,05 en todos los años
    invertido = p.datos.with_columns(pl.when(pl.col("anio") >= 2022).then(1 - pl.col("ELIX_01")).otherwise(pl.col("ELIX_01")).cast(pl.Int8).alias("ELIX_01"))      # 2022 y 2023 al revés
    for datos, esperado, anios in ((p.datos, "no", 5), (invertido, "si", 5)):
        q = dataclasses.replace(p, datos=datos)
        t = sel.tabla_estabilidad(contexto, q, sel.tabla_ranking(contexto, q))
        r = _fila(t, tipo_fila="resumen", variable="ELIX_01")
        assert r["cambio_signo"] == esperado and r["n_anios_efecto_suficiente"] == anios


def test_psi_con_deciles_de_la_ventana_de_ranking_en_las_dos_comparaciones(estabilidad, panel):
    anio = panel.datos["anio"].to_numpy()
    for nombre in ("EDAD_ANIOS", "DIA_SEMANA", "ELIX_02"):
        tipo = panel.variables[nombre].tipo
        if tipo == "num":
            x = panel.datos[nombre].to_numpy()
            bordes = np.unique(np.quantile(x[anio <= 2022], np.arange(1, 10) / 10))
            cod = np.searchsorted(bordes, x, side="left")
            n_cod = bordes.size + 2
        elif tipo == "bin":
            cod, n_cod = panel.datos[nombre].fill_null(2).to_numpy(), 3
        else:
            cod, n_cod = panel.codigos(nombre)[0], len(panel.niveles[nombre])
        conteo = lambda m: np.bincount(cod[m], minlength=n_cod)  # noqa: E731
        esperado = {
            "2019-2022 contra 2023": est.psi(conteo(anio <= 2022), conteo(anio == 2023)),
            "2019 contra 2020-2022": est.psi(conteo(anio == 2019), conteo((anio >= 2020) & (anio <= 2022))),
        }
        for comp, valor in esperado.items():
            f = _fila(estabilidad, tipo_fila="psi", variable=nombre, comparacion=comp)
            assert f["psi"] == pytest.approx(valor, abs=1e-12), (nombre, comp)
        f1 = _fila(estabilidad, tipo_fila="psi", variable=nombre, comparacion="2019-2022 contra 2023")
        f2 = _fila(estabilidad, tipo_fila="psi", variable=nombre, comparacion="2019 contra 2020-2022")
        assert (f1["n_base"], f1["n_nuevo"], f2["n_base"], f2["n_nuevo"]) == (960, 240, 240, 720)


def test_la_deriva_fuerte_de_dia_semana_en_2023_es_psi_alto_y_la_h17a_no(estabilidad):
    f = _fila(estabilidad, tipo_fila="psi", variable="DIA_SEMANA", comparacion="2019-2022 contra 2023")
    assert f["psi"] > 0.25 and f["clase_psi"] == "alto"
    g = _fila(estabilidad, tipo_fila="psi", variable="DIA_SEMANA", comparacion="2019 contra 2020-2022")
    assert g["psi"] < 0.10 and g["clase_psi"] == "estable"


def test_clases_de_psi_en_los_bordes_exactos_0_10_y_0_25_y_con_valores_redondeados(contexto, panel, ranking, monkeypatch):
    for valor, esperada in ((0.10, "moderado"), (0.0999996, "moderado"), (0.099999, "estable"), (0.25, "alto"), (0.2499996, "alto"), (0.249999, "moderado")):
        monkeypatch.setattr(est, "psi_de_codigos", lambda *a, _v=valor, **k: _v)
        t = sel.tabla_estabilidad(contexto, panel, ranking).filter(pl.col("tipo_fila") == "psi")
        assert set(t["clase_psi"]) == {esperada}, valor


def test_psi_sin_2023_o_sin_anos_para_la_segunda_comparacion_queda_nulo(contexto, panel):
    sin23 = dataclasses.replace(panel, datos=panel.datos.filter(pl.col("anio") <= 2022), anio_confirmacion=None)
    t = sel.tabla_estabilidad(contexto, sin23, sel.tabla_ranking(contexto, sin23)).filter(pl.col("tipo_fila") == "psi")
    a = t.filter(pl.col("comparacion") == "2019-2022 contra 2023")
    assert a["psi"].null_count() == a.height and set(a["clase_psi"]) == {"no evaluable"} and set(a["n_nuevo"]) == {0}
    un_ano = dataclasses.replace(panel, datos=panel.datos.filter(pl.col("anio").is_in([2019, 2023])), anios_ranking=[2019])
    b = sel.tabla_estabilidad(contexto, un_ano, sel.tabla_ranking(contexto, un_ano)).filter((pl.col("tipo_fila") == "psi") & (pl.col("comparacion") == "2019 contra 2020-2022"))
    assert b["psi"].null_count() == b.height                                                                      # exige 2019 y al menos un año de 2020 a 2022


def test_un_nivel_nuevo_de_2023_recibe_la_tasa_de_otros_si_la_variable_pasa_de_60_niveles_y_la_global_si_no(panel):
    n = panel.datos.height
    anio = panel.datos["anio"].to_numpy()
    y = panel.datos["y"].to_numpy()
    rng = np.random.default_rng(8)
    for n_niveles, hay_otros_en_ranking in ((70, True), (20, False)):
        valores = np.array([f"NIV_{k:02d}" for k in rng.integers(0, n_niveles, n)], dtype=object)
        valores[anio == 2023] = np.where(np.arange((anio == 2023).sum()) % 4 == 0, "SOLO_2023", valores[anio == 2023])
        serie = pl.Series("CATX", valores.tolist(), dtype=pl.String)
        niv = pn.calcular_niveles(serie, panel.datos["anio"], [2019, 2020, 2021, 2022], 60)
        recodificada = pn.recodificar(serie, niv)
        variables = dict(panel.variables)
        variables["CATX"] = pn.Variable(nombre="CATX", tipo="cat", grupo="prueba", rol_fase1="FEATURE_BASAL", momento="basal", evaluable="si", uso="candidata", fuentes=())
        p = dataclasses.replace(panel, datos=panel.datos.with_columns(recodificada.alias("CATX")), variables=variables, niveles={**panel.niveles, "CATX": niv.niveles})
        a = modelos.ajustar_univariado(p, "CATX")
        idx = niv.niveles.index(niv.otros)
        en_ranking = (recodificada.to_numpy() == niv.otros) & (anio <= 2022)
        global_ = y[anio <= 2022].mean()
        if hay_otros_en_ranking:
            assert en_ranking.sum() > 0 and a.tasas[idx] == pytest.approx((y[en_ranking].sum() + est.SUAVIZADO_M * global_) / (en_ranking.sum() + est.SUAVIZADO_M))
            assert a.tasas[idx] != pytest.approx(global_, rel=1e-3)
        else:
            assert en_ranking.sum() == 0 and a.tasas[idx] == pytest.approx(global_)                               # nivel sin observaciones en el ranking: la tasa global


# ====================================================================================================================== importancia
def test_las_columnas_de_la_importancia_son_las_del_diseno(importancia):
    assert importancia.columns == sel.COLUMNAS_IMPORTANCIA == [
        "tipo_fila", "variable", "tipo", "clase_proxy", "n_columnas_diseno", "n_entrenamiento", "n_eventos_entrenamiento", "n_2023", "n_eventos_2023", "auc_base_2023",
        "auc_con_variable_2023", "incremento_auc", "incremento_estado", "en_modelo_l1", "l1_sobrevive", "l1_n_coef_no_cero", "l1_coef_max_abs", "importancia_permutacion",
        "importancia_sd", "importa", "auc_2023", "auc_ranking", "l1_c", "l1_convergio",
    ]


def test_hay_una_fila_por_candidata_y_tres_de_referencia_y_etnia_no_esta(importancia, panel):
    v = importancia.filter(pl.col("tipo_fila") == "variable")
    assert v["variable"].to_list() == modelos.variables_ranking(panel) and "ETNIA" not in v["variable"].to_list() and "PUEBLO_ORIGINARIO" not in v["variable"].to_list()
    assert importancia.filter(pl.col("tipo_fila") == "referencia")["variable"].to_list() == ["MODELO_EDAD_SEXO", "MODELO_ELIX31", "MODELO_L1"]


def test_el_incremento_existe_para_las_excluidas_por_la_fase1_que_no_entran_al_l1(importancia):
    for nombre in ("N_DX_SEC", "SERVICIO_SALUD", "PREVISION"):
        f = _fila(importancia, tipo_fila="variable", variable=nombre)
        assert f["en_modelo_l1"] == "no" and f["incremento_estado"] == "ok" and f["incremento_auc"] is not None and f["l1_sobrevive"] is None and f["importancia_permutacion"] is None
    for nombre in ("EDAD_ANIOS", "SEXO"):
        f = _fila(importancia, tipo_fila="variable", variable=nombre)
        assert f["incremento_estado"] == "no aplica: variable del modelo base" and f["incremento_auc"] is None and f["auc_con_variable_2023"] is None
    assert _fila(importancia, tipo_fila="variable", variable="ELIX_28")["incremento_estado"] == "constante en el entrenamiento"
    assert _fila(importancia, tipo_fila="variable", variable="ELIX_28")["en_modelo_l1"] == "no"


def test_la_clase_de_proxy_sale_de_la_tabla_de_proxy(importancia, clase_proxy):
    for f in importancia.filter(pl.col("tipo_fila") == "variable").iter_rows(named=True):
        assert f["clase_proxy"] == clase_proxy[f["variable"]]
    assert _fila(importancia, tipo_fila="variable", variable="NIVEL_CUIDADO_INGRESO")["clase_proxy"] == "moderado"
    assert _fila(importancia, tipo_fila="referencia", variable="MODELO_L1")["clase_proxy"] is None


def test_el_l1_la_permutacion_y_el_c_elegido_se_informan_para_las_variables_del_l1(importancia, panel):
    en_l1 = importancia.filter((pl.col("tipo_fila") == "variable") & (pl.col("en_modelo_l1") == "si"))
    assert set(en_l1["variable"]) <= set(modelos.variables_modelo(panel)) and en_l1.height >= 40
    sobreviven = en_l1.filter(pl.col("l1_sobrevive") == "si")
    assert {"EDAD_ANIOS", "NIVEL_CUIDADO_INGRESO"} <= set(sobreviven["variable"]) and (sobreviven["l1_n_coef_no_cero"] >= 1).all() and (sobreviven["l1_coef_max_abs"] > 0).all()
    no_sobreviven = en_l1.filter(pl.col("l1_sobrevive") == "no")
    assert no_sobreviven.height > 20 and (no_sobreviven["l1_n_coef_no_cero"] == 0).all() and (no_sobreviven["importancia_permutacion"] == 0.0).all() and (no_sobreviven["importa"] == "no").all()
    f = _fila(importancia, tipo_fila="variable", variable="EDAD_ANIOS")
    assert f["importancia_permutacion"] > 0.05 and f["importa"] == "si" and f["importancia_sd"] > 0
    m = _fila(importancia, tipo_fila="referencia", variable="MODELO_L1")
    assert m["l1_c"] in [pytest.approx(c) for c in modelos.CS_L1] and m["l1_convergio"] == "si" and m["auc_2023"] > 0.65 and m["auc_ranking"] > m["auc_2023"]


def test_importa_exige_que_el_valor_redondeado_alcance_el_minimo_de_0_001(importancia):
    for f in importancia.filter((pl.col("tipo_fila") == "variable") & pl.col("importancia_permutacion").is_not_null()).iter_rows(named=True):
        assert (f["importa"] == "si") == (round(f["importancia_permutacion"], 6) >= 0.001), f["variable"]


def test_la_permutacion_se_calcula_sobre_2023_con_el_modelo_del_ranking_mutante_m30(contexto, panel, importancia):
    espec = modelos.especificar_diseno(panel, modelos.variables_modelo(panel), panel.anios_ranking, SEMILLA, omitir_nivel_base=False)
    X, y, grupos = modelos.aplicar_diseno(espec, panel, panel.anios_ranking, entrenamiento=True)
    l1 = modelos.ajustar_l1(X, y, espec.columnas, grupos, SEMILLA)
    X23, y23, g23 = modelos.aplicar_diseno(espec, panel, [2023])
    en_2023 = {f["variable"]: f for f in modelos.importancia_permutacion(X23, y23, l1, g23, semilla=SEMILLA).iter_rows(named=True)}
    en_ranking = {f["variable"]: f for f in modelos.importancia_permutacion(X, y, l1, grupos, semilla=SEMILLA).iter_rows(named=True)}
    f = _fila(importancia, tipo_fila="variable", variable="EDAD_ANIOS")
    assert f["importancia_permutacion"] == pytest.approx(en_2023["EDAD_ANIOS"]["importancia"], abs=1e-12) and f["importancia_sd"] == pytest.approx(en_2023["EDAD_ANIOS"]["desviacion"], abs=1e-12)
    assert abs(f["importancia_permutacion"] - en_ranking["EDAD_ANIOS"]["importancia"]) > 1e-3                          # sobre la ventana de ranking daría otro valor
    assert _fila(importancia, tipo_fila="referencia", variable="MODELO_L1")["auc_2023"] == pytest.approx(modelos.auc_l1(l1, X23, y23), abs=1e-12)
    assert _fila(importancia, tipo_fila="variable", variable="EDAD_ANIOS")["n_2023"] == 240 and _fila(importancia, tipo_fila="variable", variable="EDAD_ANIOS")["n_entrenamiento"] == 960


def test_la_importancia_es_determinista_con_la_semilla_del_contexto(contexto, panel, referencias, clase_proxy, importancia):
    otra = sel.tabla_importancia(contexto, panel, referencias, clase_proxy)
    assert otra.equals(importancia)


def test_las_filas_de_referencia_traen_los_dos_auc_y_los_conteos(importancia, referencias):
    for nombre in ("MODELO_EDAD_SEXO", "MODELO_ELIX31"):
        f = _fila(importancia, tipo_fila="referencia", variable=nombre)
        assert f["auc_2023"] == pytest.approx(referencias[nombre]["auc_2023"]) and f["auc_ranking"] == pytest.approx(referencias[nombre]["auc_ranking"])
        assert (f["n_entrenamiento"], f["n_eventos_entrenamiento"], f["n_2023"], f["n_eventos_2023"]) == (960, 154, 240, 30) and f["n_columnas_diseno"] == referencias[nombre]["n_columnas"]
        assert f["tipo"] == "modelo" and f["incremento_auc"] is None and f["en_modelo_l1"] is None


def _con_eventos_en_2023(panel, n_eventos):
    anio, y = panel.datos["anio"].to_numpy(), panel.datos["y"].to_numpy().copy()
    en_2023 = np.flatnonzero(anio == 2023)
    y[en_2023] = 0
    y[en_2023[:n_eventos]] = 1
    return _con_columna(panel, "y", y.tolist(), pl.Int8)


def test_con_seis_eventos_en_2023_se_anula_el_bloque_de_2023_y_se_conserva_el_de_entrenamiento(contexto, panel):
    p = _con_eventos_en_2023(panel, 6)
    refs = modelos.referencias(p, SEMILLA)
    t = sel.tabla_importancia(contexto, p, refs, {})
    assert set(t["n_eventos_2023"]) == {6}
    pub, n_celdas = supresion.suprimir_bloques(t, sel.BLOQUES_IMPORTANCIA, minimo=contexto.celda_minima)
    assert n_celdas == t.height * 1 and pub["n_eventos_2023"].null_count() == t.height                             # una celda pequeña por fila: n_eventos_2023
    assert set(pub["n_2023"]) == {240}                                                                              # el conteo grande de ese bloque no se anula: solo la celda pequeña y las cifras que dependen
    for c in ("auc_base_2023", "auc_con_variable_2023", "incremento_auc", "importancia_permutacion", "importancia_sd", "importa", "auc_2023"):
        assert pub[c].null_count() == pub.height, c
    for c in ("n_entrenamiento", "n_eventos_entrenamiento", "l1_sobrevive", "l1_coef_max_abs", "auc_ranking", "l1_c"):
        assert pub[c].null_count() < pub.height, c                                                                  # el bloque de entrenamiento queda
    ref = pub.filter(pl.col("tipo_fila") == "referencia")
    assert ref["auc_ranking"].null_count() == 0 and ref["auc_2023"].null_count() == ref.height                       # incluidas las filas de referencia
    assert ref.filter(pl.col("variable") == "MODELO_L1")["l1_convergio"].to_list() == ["si"]                         # la marca de convergencia no se anula
    assert t.filter(pl.col("variable") == "MODELO_L1")["l1_convergio"].to_list() == ["si"]


def test_con_doce_eventos_en_2023_no_se_anula_nada(contexto, panel):
    p = _con_eventos_en_2023(panel, 12)
    t = sel.tabla_importancia(contexto, p, modelos.referencias(p, SEMILLA), {})
    pub, n_celdas = supresion.suprimir_bloques(t, sel.BLOQUES_IMPORTANCIA, minimo=contexto.celda_minima)
    assert n_celdas == 0 and pub.equals(t)


def test_el_l1_con_pocos_eventos_se_informa_sin_ser_fatal(contexto, panel):
    anio, y = panel.datos["anio"].to_numpy(), panel.datos["y"].to_numpy().copy()
    rank = np.flatnonzero(anio <= 2022)
    y[rank] = 0
    y[rank[:5]] = 1                                                                                             # solo 5 eventos en el ranking
    p = _con_columna(panel, "y", y.tolist(), pl.Int8)
    t = sel.tabla_importancia(contexto, p, modelos.referencias(p, SEMILLA), {})
    en_l1 = t.filter((pl.col("tipo_fila") == "variable") & (pl.col("en_modelo_l1") == "si"))
    assert set(en_l1["l1_sobrevive"]) == {"no evaluable: pocos eventos"} and en_l1["importancia_permutacion"].null_count() == en_l1.height
    m = _fila(t, tipo_fila="referencia", variable="MODELO_L1")
    assert m["l1_sobrevive"] == "no evaluable: pocos eventos" and m["auc_2023"] is None and m["l1_c"] is None and m["l1_convergio"] is None


def test_un_panel_sin_confirmacion_deja_nulas_las_cifras_de_2023(contexto, panel):
    sin = dataclasses.replace(panel, datos=panel.datos.filter(pl.col("anio") <= 2022), anio_confirmacion=None)
    t = sel.tabla_importancia(contexto, sin, modelos.referencias(sin, SEMILLA), {})
    v = t.filter((pl.col("tipo_fila") == "variable") & (pl.col("variable") != "ELIX_28"))
    assert v["incremento_auc"].null_count() == v.height and v["importancia_permutacion"].null_count() == v.height
    assert set(v.filter(~pl.col("incremento_estado").str.starts_with("no aplica"))["incremento_estado"]) == {"no evaluable en 2023"}


# ======================================================================================================================== parámetros
def test_los_parametros_traen_una_fila_por_constante_leida_del_modulo(monkeypatch):
    t = sel.tabla_parametros()
    assert t.columns == sel.COLUMNAS_PARAMETROS == ["modulo", "parametro", "valor", "para_que_sirve", "afecta"] and t.height == 15
    esperados = {
        "SUAVIZADO_M", "EPS_COLINEALIDAD", "DECIMALES_PUBLICADOS", "SEPARACION_COEF_MAX", "SEPARACION_SE_MAX", "RIDGE_AUC", "RIDGE_WALD", "TOPE_FILAS_MODELOS", "TOPE_FILAS_L1", "CS_L1", "PLIEGUES_CV",
        "REPETICIONES_PERMUTACION", "MIN_FILAS_HOSPITAL", "ETIQUETA_V_RESULTADO", "relleno_nulos_modelos",
    }
    assert set(t["parametro"]) == esperados and t["parametro"].n_unique() == 15
    assert "RIDGE_DISENO" not in t["parametro"].to_list()
    v = dict(zip(t["parametro"], t["valor"]))
    assert v["SUAVIZADO_M"] == "20" and v["EPS_COLINEALIDAD"] == "0,000000001" and v["DECIMALES_PUBLICADOS"] == "6" and v["SEPARACION_COEF_MAX"] == "12" and v["SEPARACION_SE_MAX"] == "1000"
    assert v["RIDGE_AUC"] == "1" and v["RIDGE_WALD"] == "0" and v["TOPE_FILAS_MODELOS"] == "sin tope" and v["TOPE_FILAS_L1"] == "400000" and v["PLIEGUES_CV"] == "3" and v["REPETICIONES_PERMUTACION"] == "10"
    assert v["MIN_FILAS_HOSPITAL"] == "100" and v["ETIQUETA_V_RESULTADO"] == "0,1" and v["CS_L1"] == "0,001; 0,004642; 0,02154; 0,1; 0,4642; 2,154; 10"
    assert v["relleno_nulos_modelos"] == "mediana del entrenamiento más indicador de nulos"
    assert dict(zip(t["parametro"], t["modulo"]))["SUAVIZADO_M"] == "estadistica" and dict(zip(t["parametro"], t["modulo"]))["RIDGE_AUC"] == "modelos"
    assert all("e+" not in x and "e-" not in x and "." not in x for x in t["valor"] if x != "mediana del entrenamiento más indicador de nulos")        # sin notación científica y con coma decimal
    monkeypatch.setattr(est, "SUAVIZADO_M", 35.5)
    monkeypatch.setattr(modelos, "RIDGE_AUC", 2.0)
    monkeypatch.setattr(sospechas, "ETIQUETA_V_RESULTADO", 0.25)
    cambiado = dict(zip(sel.tabla_parametros()["parametro"], sel.tabla_parametros()["valor"]))
    assert cambiado["SUAVIZADO_M"] == "35,5" and cambiado["RIDGE_AUC"] == "2" and cambiado["ETIQUETA_V_RESULTADO"] == "0,25"


def test_los_parametros_no_incluyen_la_semilla_y_pasan_el_escaneo_de_privacidad(contexto):
    t = sel.tabla_parametros()
    assert not any(str(contexto.semilla) in str(x) for fila in t.iter_rows() for x in fila)
    contexto.escribir_tabla("t23_parametros_seleccion.csv", t, tarea="T11", descripcion="parámetros técnicos")
    res = privacidad.escanear_tabla(contexto.dir_tablas / "t23_parametros_seleccion.csv", [], contexto.celda_minima)
    assert sum(res.values()) == 0, res
    assert contexto.indice_tablas().filter(pl.col("nombre") == "t23_parametros_seleccion.csv")["columnas_conteo"][0] == ""        # sin cols_n


# ========================================================================================================================== escritura
@pytest.fixture()
def contexto_con_tablas(tmp_path, panel, ranking, estabilidad, importancia):
    ctx = Contexto.crear("extracto", fuente=crear_fuente_sintetica(tmp_path / "datos", n=60, semilla=29), dir_salida_base=tmp_path / "salidas")
    for nombre, tabla, bloques in (
        ("t23_ranking_variables.csv", ranking, sel.BLOQUES_RANKING), ("t23_estabilidad.csv", estabilidad, sel.BLOQUES_ESTABILIDAD), ("t23_importancia.csv", importancia, sel.BLOQUES_IMPORTANCIA),
    ):
        pn.escribir_por_bloques(ctx, nombre, tabla, bloques, tarea="T11", descripcion=nombre)
    return ctx


def test_las_columnas_de_cada_csv_coinciden_con_las_constantes_del_modulo(contexto_con_tablas):
    ctx = contexto_con_tablas
    for nombre, columnas in (("t23_ranking_variables.csv", sel.COLUMNAS_RANKING), ("t23_estabilidad.csv", sel.COLUMNAS_ESTABILIDAD), ("t23_importancia.csv", sel.COLUMNAS_IMPORTANCIA)):
        assert pl.read_csv(ctx.dir_tablas / nombre, n_rows=0).columns == columnas, nombre
    indice = ctx.indice_tablas()
    assert indice.filter(pl.col("nombre") == "t23_ranking_variables.csv")["columnas_conteo"][0].split(";") == [*sel.COLS_N_RANKING, "n_faltante"]
    assert indice.filter(pl.col("nombre") == "t23_estabilidad.csv")["columnas_conteo"][0].split(";") == ["n", "n_eventos", "n_base", "n_nuevo"]
    assert indice.filter(pl.col("nombre") == "t23_importancia.csv")["columnas_conteo"][0].split(";") == ["n_entrenamiento", "n_eventos_entrenamiento", "n_2023", "n_eventos_2023"]
    esc = privacidad.escanear_salidas(ctx.dir_salida, indice, ctx.celda_minima)
    assert privacidad.total_violaciones(esc) == 0, esc.filter(pl.col("n_violaciones") > 0)


def test_el_ranking_escrito_anula_las_celdas_pequenas_de_todos_los_conteos_registrados(contexto_con_tablas):
    leido = contexto_con_tablas.leer_tabla("t23_ranking_variables.csv", columnas_texto=sel._TEXTO_RANKING)
    for c in (*sel.COLS_N_RANKING, "n_faltante"):
        sub = leido[c].drop_nulls()
        assert ((sub >= 1) & (sub < 10)).sum() == 0, c


# ============================================================================================================================ figuras
def _capturar(monkeypatch):
    from matplotlib.figure import Figure

    capturadas = []
    original = Figure.savefig
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: (capturadas.append(self), original(self, *a, **k))[1])
    return capturadas


def _solo(ctx, nombre):
    for otro in ctx.dir_tablas.glob("*.csv"):
        if otro.name != nombre:
            otro.unlink()


def test_figura_ranking_se_arma_solo_con_su_csv_con_las_variables_ordenadas_por_posicion(contexto_con_tablas, monkeypatch):
    ctx = contexto_con_tablas
    _solo(ctx, "t23_ranking_variables.csv")
    capturadas = _capturar(monkeypatch)
    assert sel.figura_ranking(ctx) is True
    ax = capturadas[0].axes[0]
    etiquetas = [t.get_text().rstrip("†") for t in ax.get_yticklabels()]
    t = ctx.leer_tabla("t23_ranking_variables.csv", columnas_texto=sel._TEXTO_RANKING).filter((pl.col("ventana") == "2019-2022") & pl.col("auc_senal").is_not_null()).sort("posicion")
    assert etiquetas == t["variable"].to_list() and etiquetas[0] == "EDAD_ANIOS"
    assert {"0,55", "0,60", "0,70", "0,85"} <= {x.get_text() for x in ax.texts}                                  # rótulos de los umbrales arriba
    assert ax.get_xlabel() == "AUC (0,5 = sin señal)" and ax.get_xlim()[0] <= 0.45
    e = ctx.figuras.indice().row(0, named=True)
    assert (e["id_figura"], e["csv"], e["decision_relacionada"], e["titulo"]) == ("F11_01", "t23_ranking_variables.csv", "2; 14", "Señal univariada frente a la mortalidad (2019 a 2022 y 2023)")
    assert sum(1 for p in ax.patches if p.get_hatch()) == t.filter(pl.col("rol_fase1").is_in(sorted(decision.ROLES_EXCLUYENTES))).height > 3          # trama en las excluidas por la Fase 1


def test_figura_ranking_sin_variables_con_senal_se_omite(contexto_con_tablas):
    ctx = contexto_con_tablas
    t = ctx.leer_tabla("t23_ranking_variables.csv", columnas_texto=sel._TEXTO_RANKING).with_columns(pl.lit(None, dtype=pl.Float64).alias("auc_senal"))
    t.write_csv(ctx.dir_tablas / "t23_ranking_variables.csv")
    assert sel.figura_ranking(ctx) is False and ctx.figuras.ids() == []


def test_figura_estabilidad_se_arma_solo_con_su_csv_y_marca_cambio_de_signo_y_celdas_optimistas(contexto_con_tablas, monkeypatch):
    ctx = contexto_con_tablas
    _solo(ctx, "t23_estabilidad.csv")                                                                           # se borran las demás tablas antes de llamarla
    capturadas = _capturar(monkeypatch)
    assert sel.figura_estabilidad(ctx) is True
    fig = capturadas[0]
    ax, ax_psi = fig.axes[0], fig.axes[1]                                                                       # el mapa de calor y el panel de PSI (la barra de color se agrega después)
    etiquetas = [t.get_text() for t in ax.get_yticklabels()]
    t = ctx.leer_tabla("t23_estabilidad.csv", columnas_texto=sel._TEXTO_ESTABILIDAD)
    con_cambio = set(t.filter((pl.col("tipo_fila") == "resumen") & (pl.col("cambio_signo") == "si"))["variable"])
    assert {e.rstrip("*") for e in etiquetas if e.endswith("*")} == con_cambio and len(con_cambio) >= 5
    optimistas = t.filter((pl.col("tipo_fila") == "auc_anual") & (pl.col("auc_optimista") == "si"))
    puntos = [ln for ln in ax.lines if ln.get_marker() == "o" and ln.get_markersize() < 3]
    assert len(puntos) == optimistas.height > 20                                                                # un punto por celda optimista
    assert [x.get_text() for x in ax.get_xticklabels()] == ["2019", "2020", "2021", "2022", "2023"] and len(ax_psi.patches) == 2 * len(etiquetas)
    e = ctx.figuras.indice().row(0, named=True)
    assert (e["id_figura"], e["csv"], e["decision_relacionada"], e["titulo"]) == ("F11_03", "t23_estabilidad.csv", "17", "Estabilidad por año de la señal de cada variable (2019 a 2023)")
    assert not ctx.dir_tablas.joinpath("t23_ranking_variables.csv").exists()


def test_figura_estabilidad_sin_filas_anuales_se_omite(contexto_con_tablas):
    ctx = contexto_con_tablas
    ctx.leer_tabla("t23_estabilidad.csv", columnas_texto=sel._TEXTO_ESTABILIDAD).filter(pl.col("tipo_fila") == "psi").write_csv(ctx.dir_tablas / "t23_estabilidad.csv")
    assert sel.figura_estabilidad(ctx) is False


def test_figura_importancia_se_arma_solo_con_su_csv_con_las_variables_del_l1_y_la_trama_de_proxy(contexto_con_tablas, monkeypatch):
    ctx = contexto_con_tablas
    _solo(ctx, "t23_importancia.csv")
    capturadas = _capturar(monkeypatch)
    assert sel.figura_importancia(ctx) is True
    ax = capturadas[0].axes[0]
    t = ctx.leer_tabla("t23_importancia.csv", columnas_texto=sel._TEXTO_IMPORTANCIA)
    en_l1 = t.filter((pl.col("tipo_fila") == "variable") & (pl.col("en_modelo_l1") == "si") & pl.col("importancia_permutacion").is_not_null())
    etiquetas = [x.get_text() for x in ax.get_yticklabels()]
    assert sorted(etiquetas) == sorted(en_l1["variable"].to_list()) and etiquetas[0] == "EDAD_ANIOS"
    assert "ESPECIALIDAD_MEDICA" in etiquetas and "N_DX_SEC" not in etiquetas and "SERVICIO_SALUD" not in etiquetas          # solo las del L1
    con_trama = en_l1.filter(pl.col("clase_proxy").is_in(["moderado", "alto"]))["variable"].to_list()
    assert "NIVEL_CUIDADO_INGRESO" in con_trama and sum(1 for p in ax.patches if p.get_hatch()) == len(con_trama)
    assert any(abs(float(v) - 0.001) < 1e-12 for ln in ax.lines for v in np.atleast_1d(ln.get_xdata()))               # la línea del umbral de importancia
    e = ctx.figuras.indice().row(0, named=True)
    assert (e["id_figura"], e["csv"], e["decision_relacionada"], e["titulo"]) == ("F11_05", "t23_importancia.csv", "", "Importancia por permutación de las variables en 2023")


def test_figura_importancia_sin_variables_del_l1_se_omite(contexto_con_tablas):
    ctx = contexto_con_tablas
    ctx.leer_tabla("t23_importancia.csv", columnas_texto=sel._TEXTO_IMPORTANCIA).with_columns(pl.lit("no").alias("en_modelo_l1")).write_csv(ctx.dir_tablas / "t23_importancia.csv")
    assert sel.figura_importancia(ctx) is False


# ============================================================================================ parte 2: calcular, resumen, métricas
from collections import Counter  # noqa: E402

from eda import sensibilidad_elix  # noqa: E402


@pytest.fixture(scope="module")
def calculado(contexto, panel):
    """``calcular`` sobre el panel artificial: toda la Tarea 11 (y la 10, que calcula por su cuenta) sin escribir nada."""
    return sel.calcular(contexto, panel)


def _publicable(ctx, tabla, bloques):
    return supresion.suprimir_bloques(tabla, bloques, minimo=ctx.celda_minima)[0]


def test_calcular_no_escribe_nada_y_devuelve_las_claves_del_contrato(ctx, panel):
    antes = sorted(p.name for p in ctx.dir_salida.rglob("*"))
    r = sel.calcular(ctx, panel)
    assert sorted(p.name for p in ctx.dir_salida.rglob("*")) == antes and ctx.indice_tablas().height == 0 and ctx.metricas().height == 0 and ctx.figuras.ids() == []
    reales = {"ranking", "estabilidad", "proxy", "importancia", "fuga", "or_elix", "sensibilidad"}
    assert {"panel", "asoc", "referencias", "disponibilidad", "matriz", "matriz_pub", "evidencia", "figuras_omitidas", "parametros", "resumen"} <= set(r)
    assert reales <= set(r) and {f"{k}_pub" for k in reales} <= set(r) and r["panel"] is panel
    assert r["parametros"].equals(sel.tabla_parametros()) and r["resumen"].columns == sel.COLUMNAS_RESUMEN


def test_las_versiones_publicables_son_las_reales_con_las_listas_de_bloques_aplicadas(contexto, calculado):
    bloques = {
        "ranking": sel.BLOQUES_RANKING, "estabilidad": sel.BLOQUES_ESTABILIDAD, "proxy": sospechas.BLOQUES_PROXY, "importancia": sel.BLOQUES_IMPORTANCIA, "fuga": sospechas.BLOQUES_FUGA,
        "or_elix": sensibilidad_elix.BLOQUES_OR_ELIX, "sensibilidad": sensibilidad_elix.BLOQUES_SENSIBILIDAD,
    }
    for clave, lista in bloques.items():
        assert calculado[f"{clave}_pub"].equals(_publicable(contexto, calculado[clave], lista)), clave
        assert calculado[f"{clave}_pub"].columns == calculado[clave].columns
    # en este panel las categorías raras de Elixhauser tienen celdas pequeñas: la publicable anula cifras que la real conserva
    assert calculado["ranking_pub"]["auc_senal"].null_count() > calculado["ranking"]["auc_senal"].null_count()


def test_la_matriz_se_arma_con_la_evidencia_real_y_se_publica_con_las_publicables(contexto, panel, calculado):
    ev = calculado["evidencia"]
    for clave in ("ranking", "estabilidad", "proxy", "importancia", "fuga"):
        assert ev[clave] is calculado[clave] and ev[f"{clave}_pub"] is calculado[f"{clave}_pub"]
    assert ev["clusters"] is calculado["asoc"]["clusters"]
    aparte = decision.armar_matriz(contexto, panel, ev)
    assert aparte["matriz"].equals(calculado["matriz"]) and aparte["matriz_pub"].equals(calculado["matriz_pub"]) and aparte["disponibilidad"].equals(calculado["disponibilidad"])
    assert (calculado["matriz_pub"]["valor_suprimido"] == "si").sum() > 0 and (calculado["matriz"]["valor_suprimido"] == "no").all()


def test_calcular_con_asoc_dado_toma_su_panel_no_recalcula_la_tarea_10_y_es_determinista(contexto, calculado, monkeypatch):
    monkeypatch.setattr(asoc, "calcular", lambda *a, **k: (_ for _ in ()).throw(AssertionError("la Tarea 10 no debe recalcularse")))
    r = sel.calcular(contexto, None, calculado["asoc"])
    assert r["panel"] is calculado["panel"] and r["asoc"] is calculado["asoc"]
    for clave in ("ranking", "estabilidad", "proxy", "importancia", "fuga", "or_elix", "sensibilidad", "ranking_pub", "importancia_pub", "disponibilidad", "matriz", "matriz_pub", "resumen"):
        assert r[clave].equals(calculado[clave]), clave                                                           # dos cálculos con la misma semilla dan lo mismo


# ------------------------------------------------------------------------------------------------------------------ resumen
def test_el_resumen_tiene_una_fila_con_las_columnas_del_diseno_y_los_tipos_declarados(calculado):
    r = calculado["resumen"]
    assert r.height == 1 and r.columns == sel.COLUMNAS_RESUMEN and dict(r.schema) == sel.ESQUEMA_RESUMEN and len(set(r.columns)) == len(r.columns)
    assert r.columns[:6] == ["n_universo", "n_crudas", "n_derivadas", "n_evaluables", "n_parciales", "n_no_evaluables"]
    assert r.columns[-4:] == ["confirmacion_disponible", "t22_disponible", "historial_evaluable", "figuras_omitidas"]
    assert r.columns[r.columns.index("n_matriz"):r.columns.index("n_restriccion_r3")] == [
        "n_matriz", *[f"n_recomendacion_{g}" for g in ("excluir_fase1", "excluir_fuga", "excluir_proxy", "condicional_proxy", "condicional_calidad", "redundante", "baja_senal",
                                                      "mantener_clinico", "mantener", "no_evaluable")],
        *[f"n_contraste_{e}" for e in decision.ETIQUETAS_CONTRASTE],
    ]
    assert all(c in r.columns for c in ("n_sin_medicion", "elix_n_con_indice", "n_filas_ranking_suprimidas"))


def test_el_resumen_cuenta_el_universo_el_ranking_y_el_faltante_con_cuentas_independientes(calculado):
    res = calculado["resumen"].row(0, named=True)
    d = calculado["disponibilidad"]
    origen, evaluable = Counter(d["origen"]), Counter(d["evaluable"])
    assert res["n_universo"] == 199 and res["n_crudas"] == origen["cruda"] == 129 and res["n_derivadas"] == origen["derivada"] + origen["regla_fase1"] == 70
    assert (res["n_evaluables"], res["n_parciales"], res["n_no_evaluables"]) == (evaluable["si"], evaluable["parcial"], evaluable["no"])
    rank = calculado["ranking"].filter(pl.col("ventana") == "2019-2022")
    clases, faltante = Counter(rank["clase_senal"]), Counter(rank["clase_faltante"])
    assert res["n_candidatas_ranking"] == rank.height == 56
    assert (res["n_senal_alta"], res["n_senal_media"], res["n_senal_debil"], res["n_senal_baja"]) == (clases["alta"], clases["media"], clases["débil"], clases["baja"])
    con_senal = rank.filter(pl.col("auc_senal").is_not_null()).height
    assert res["n_senal_alta"] + res["n_senal_media"] + res["n_senal_debil"] + res["n_senal_baja"] == con_senal == 55            # ELIX_28 es constante: no tiene señal ni clase
    assert (res["n_faltante_alerta"], res["n_faltante_alto"]) == (faltante["alerta"], faltante["alto"]) and res["n_faltante_alto"] >= 1      # SERVICIOINGRESO tiene más de la mitad de faltantes
    assert res["n_faltante_informativo"] == Counter(rank["faltante_informativo"])["si"]
    mejor = rank.filter(pl.col("posicion") == 1)
    assert res["mejor_variable"] == mejor["variable"][0] == "EDAD_ANIOS" and res["mejor_auc_senal"] == pytest.approx(mejor["auc_senal"][0])


def test_el_resumen_cuenta_estabilidad_proxy_modelos_y_fuga_con_cuentas_independientes(calculado, panel):
    res = calculado["resumen"].row(0, named=True)
    es = calculado["estabilidad"]
    cambios = Counter(es.filter(pl.col("tipo_fila") == "resumen")["cambio_signo"])
    psi23 = Counter(es.filter((pl.col("tipo_fila") == "psi") & (pl.col("comparacion") == decision.COMP_CONFIRMACION))["clase_psi"])
    assert res["n_cambio_signo"] == cambios["si"] and (res["n_psi_alto_2023"], res["n_psi_moderado_2023"]) == (psi23["alto"], psi23["moderado"])
    proxy = Counter(calculado["proxy"].filter(pl.col("tipo_fila") == "variable")["clase_proxy"])
    assert (res["n_proxy_alto"], res["n_proxy_moderado"]) == (proxy["alto"], proxy["moderado"]) and res["n_proxy_alto"] >= 2
    uci = calculado["proxy_pub"].filter(pl.col("tipo_fila") == "variacion_uci")
    assert res["uci_phi_sobredispersion"] == pytest.approx(uci["phi_sobredispersion"][0]) and res["uci_razon_p90_p10"] == pytest.approx(uci["razon_p90_p10"][0])
    card = panel.cardinalidad_especialidad
    assert (res["espec_n_niveles_en_un_hospital"], res["espec_n_niveles_totales"]) == (card["n_niveles_en_un_hospital"], card["n_niveles_totales"])
    imp = calculado["importancia"].filter(pl.col("tipo_fila") == "variable")
    refs = {f["variable"]: f for f in calculado["importancia_pub"].filter(pl.col("tipo_fila") == "referencia").iter_rows(named=True)}
    assert res["l1_n_variables_sobreviven"] == Counter(imp["l1_sobrevive"])["si"] and res["n_importan"] == Counter(imp["importa"])["si"]
    assert res["auc_ref_edad_sexo_2023"] == pytest.approx(refs["MODELO_EDAD_SEXO"]["auc_2023"]) and res["auc_ref_elix31_2023"] == pytest.approx(refs["MODELO_ELIX31"]["auc_2023"])
    assert res["auc_modelo_l1_2023"] == pytest.approx(refs["MODELO_L1"]["auc_2023"]) and res["l1_c"] == pytest.approx(refs["MODELO_L1"]["l1_c"])
    fuga = calculado["fuga"].filter(pl.col("tipo_fila") == "variable")
    assert res["fuga_n_supera_umbral"] == Counter(fuga["supera_umbral_fuga"])["si"] >= 1
    conc = calculado["fuga_pub"]
    assert res["concordancia_prop_sin_traslado"] == pytest.approx(conc.filter(pl.col("tipo_fila") == "concordancia")["prop_igualdad"][0])
    assert res["concordancia_prop_con_traslado"] == pytest.approx(conc.filter(pl.col("tipo_fila") == "concordancia_con_traslado")["prop_igualdad"][0])


def test_el_resumen_de_elixhauser_sale_de_las_versiones_publicables(calculado):
    res = calculado["resumen"].row(0, named=True)
    esperado = sensibilidad_elix.resumen_elixhauser(calculado["or_elix_pub"], calculado["sensibilidad_pub"])
    assert {k: res[k] for k in esperado} == esperado and res["elix_top5"].count(";") == 4


def test_el_resumen_de_la_matriz_es_el_de_decision_resumen_matriz(calculado):
    res = calculado["resumen"].row(0, named=True)
    esperado = decision.resumen_matriz(calculado["matriz"], calculado["matriz_pub"])
    assert {k: res[k] for k in esperado} == esperado and res["n_matriz"] == 199 and res["n_restriccion_r3"] == 31
    assert sum(res[f"n_recomendacion_{g}"] for g in decision.BINES_RECOMENDACION.values()) == 199
    assert res["n_valor_suprimido"] == (calculado["matriz_pub"]["valor_suprimido"] == "si").sum() > 0


def test_el_estado_del_resumen_dice_confirmacion_t22_historial_y_figuras(calculado):
    res = calculado["resumen"].row(0, named=True)
    assert (res["confirmacion_disponible"], res["t22_disponible"], res["historial_evaluable"], res["figuras_omitidas"]) == ("si", "no", "no", "ninguna")


def test_el_mejor_auc_senal_sale_de_la_version_publicable_y_las_filas_suprimidas_se_cuentan(contexto, calculado):
    r = dict(calculado)
    pub = r["ranking_pub"]
    suprimida = pub.with_columns(
        pl.when((pl.col("variable") == "EDAD_ANIOS") & (pl.col("ventana") == "2019-2022")).then(None).otherwise(pl.col("auc_senal")).alias("auc_senal")
    )
    antes = calculado["resumen"].row(0, named=True)
    despues = sel.tabla_resumen(contexto, {**r, "ranking_pub": suprimida}).row(0, named=True)
    assert antes["mejor_variable"] == "EDAD_ANIOS" and despues["mejor_variable"] == "NIVEL_CUIDADO_INGRESO" and despues["mejor_auc_senal"] < antes["mejor_auc_senal"]
    assert despues["n_filas_ranking_suprimidas"] == antes["n_filas_ranking_suprimidas"] + 1
    assert despues["n_senal_alta"] == antes["n_senal_alta"] >= 1                                              # los conteos por clase salen de los valores reales


def _con_psi(es: pl.DataFrame, valores: dict[str, float | None]) -> pl.DataFrame:
    """Reemplaza el PSI de 2019 contra 2020-2022 de las variables dadas (y su clase con los umbrales de 0,10 y 0,25)."""
    def clase(v):
        return None if v is None else ("alto" if v >= 0.25 else ("moderado" if v >= 0.10 else "estable"))

    filas = []
    for f in es.iter_rows(named=True):
        if f["tipo_fila"] == "psi" and f["comparacion"] == decision.COMP_RESTO and f["variable"] in valores:
            f = {**f, "psi": valores[f["variable"]], "clase_psi": clase(valores[f["variable"]])}
        filas.append(f)
    return pl.DataFrame(filas, schema=es.schema)


def test_h17a_cuenta_solo_las_variables_clave_y_la_mayoria_exige_mas_de_la_mitad(contexto, panel, calculado):
    clave = sorted(n for n, v in panel.variables.items() if v.uso == "candidata" and v.evaluable != "no" and v.rol_fase1 not in decision.ROLES_EXCLUYENTES)
    assert len(clave) == 47 and "SERVICIO_SALUD" not in clave and "ETNIA" not in clave and "COMUNA" not in clave           # los roles excluyentes (llave, auditoría) no son clave
    base = _con_psi(calculado["estabilidad"], {n: 0.01 for n in clave})
    base = _con_psi(base, {"SERVICIO_SALUD": 0.9, "COMUNA": 0.9, "ETNIA": 0.9, clave[0]: None})                          # PSI alto de variables no clave y un PSI no medido
    medidas = clave[1:]                                                                                                  # 46 variables clave con PSI
    resultados = {}
    for k in (22, 23, 24):
        es = _con_psi(base, {n: 0.30 for n in medidas[:k]})
        resultados[k] = sel.tabla_resumen(contexto, {**calculado, "estabilidad": es, "estabilidad_pub": es}).row(0, named=True)
    for k, r in resultados.items():
        assert r["h17a_n_clave"] == 46 and r["h17a_n_psi_alto"] == k and r["h17a_prop_psi_alto"] == pytest.approx(k / 46)
    assert [resultados[k]["h17a_mayoria_psi_alto"] for k in (22, 23, 24)] == ["no", "no", "si"]                          # 23 de 46 es exactamente la mitad: no es mayoría


def test_h17a_prop_sale_de_la_version_publicable_y_los_conteos_de_la_real(contexto, panel, calculado):
    clave = sorted(n for n, v in panel.variables.items() if v.uso == "candidata" and v.evaluable != "no" and v.rol_fase1 not in decision.ROLES_EXCLUYENTES)
    real = _con_psi(calculado["estabilidad"], {n: (0.30 if i < 30 else 0.01) for i, n in enumerate(clave)})
    pub = _con_psi(real, {clave[0]: None, clave[1]: None})                                                                 # dos PSI altos suprimidos en la publicable
    r = sel.tabla_resumen(contexto, {**calculado, "estabilidad": real, "estabilidad_pub": pub}).row(0, named=True)
    assert r["h17a_n_clave"] == 47 and r["h17a_n_psi_alto"] == 30
    assert r["h17a_prop_psi_alto"] == pytest.approx(28 / 45) and r["h17a_mayoria_psi_alto"] == "si"


def test_h17a_sin_variables_con_psi_queda_nulo_y_no_inventa_una_mayoria(contexto, panel, calculado):
    clave = [n for n, v in panel.variables.items() if v.uso == "candidata" and v.evaluable != "no" and v.rol_fase1 not in decision.ROLES_EXCLUYENTES]
    es = _con_psi(calculado["estabilidad"], {n: None for n in clave})
    r = sel.tabla_resumen(contexto, {**calculado, "estabilidad": es, "estabilidad_pub": es}).row(0, named=True)
    assert r["h17a_n_clave"] == 0 and r["h17a_n_psi_alto"] == 0 and r["h17a_prop_psi_alto"] is None and r["h17a_mayoria_psi_alto"] is None


def test_las_cifras_estadisticas_suprimidas_de_los_modelos_y_la_fuga_salen_nulas_del_resumen(contexto, calculado):
    imp = calculado["importancia_pub"].with_columns(
        pl.when(pl.col("tipo_fila") == "referencia").then(None).otherwise(pl.col("auc_2023")).alias("auc_2023"),
        pl.when(pl.col("tipo_fila") == "referencia").then(None).otherwise(pl.col("l1_c")).alias("l1_c"),
    )
    fuga = calculado["fuga_pub"].with_columns(pl.when(pl.col("tipo_fila").str.starts_with("concordancia")).then(None).otherwise(pl.col("prop_igualdad")).alias("prop_igualdad"))
    r = sel.tabla_resumen(contexto, {**calculado, "importancia_pub": imp, "fuga_pub": fuga}).row(0, named=True)
    for col in ("auc_ref_edad_sexo_2023", "auc_ref_elix31_2023", "auc_modelo_l1_2023", "l1_c", "concordancia_prop_sin_traslado", "concordancia_prop_con_traslado"):
        assert r[col] is None, col
    assert r["l1_n_variables_sobreviven"] == calculado["resumen"]["l1_n_variables_sobreviven"][0]                         # los conteos de variables salen de la tabla real


# ---------------------------------------------------------------------------------------------------- figuras omitidas
def _omitidas(calculado, **cambios):
    pub = {k: calculado[k] for k in ("ranking_pub", "proxy_pub", "estabilidad_pub", "or_elix_pub", "importancia_pub", "fuga_pub")}
    pub.update(cambios)
    return sel.figuras_omitidas(pub["ranking_pub"], pub["proxy_pub"], pub["estabilidad_pub"], pub["or_elix_pub"], pub["importancia_pub"], pub["fuga_pub"])


def test_con_todas_las_tablas_con_contenido_no_se_omite_ninguna_figura(calculado):
    assert _omitidas(calculado) == [] and calculado["figuras_omitidas"] == []


@pytest.mark.parametrize(
    "figura, tabla, vaciar",
    [
        ("F11_01", "ranking_pub", lambda t: t.with_columns(pl.lit(None, dtype=pl.Float64).alias("auc_senal"))),
        ("F11_02", "proxy_pub", lambda t: t.with_columns(pl.lit(None, dtype=pl.Float64).alias("v_resultado"))),
        ("F11_03", "estabilidad_pub", lambda t: t.filter(pl.col("tipo_fila") != "auc_anual")),
        ("F11_04", "or_elix_pub", lambda t: t.filter(pl.col("periodo") != "2019-2022")),
        ("F11_05", "importancia_pub", lambda t: t.with_columns(pl.lit(None, dtype=pl.Float64).alias("importancia_permutacion"))),
        ("F11_06", "fuga_pub", lambda t: t.with_columns(pl.lit(None, dtype=pl.Float64).alias("auc_ranking"), pl.lit(None, dtype=pl.Float64).alias("auc_confirmacion"))),
    ],
)
def test_cada_figura_se_omite_exactamente_cuando_su_tabla_publicable_no_tiene_nada_que_mostrar(calculado, figura, tabla, vaciar):
    assert _omitidas(calculado, **{tabla: vaciar(calculado[tabla])}) == [figura]


def test_figuras_omitidas_no_exige_que_la_tabla_de_las_otras_figuras_tenga_contenido(calculado):
    vacias = {k: calculado[k].clear() for k in ("ranking_pub", "proxy_pub", "estabilidad_pub", "or_elix_pub", "importancia_pub", "fuga_pub")}
    assert _omitidas(calculado, **vacias) == list(sel.IDS_FIGURAS)


# --------------------------------------------------------------------------------------------------------------- métricas
def _metricas(ctx, r):
    sel.registrar_metricas(ctx, r)
    return {f["clave"]: f for f in ctx.metricas().iter_rows(named=True)}


def test_las_metricas_tienen_tarea_alcance_y_nota_y_ninguna_cubre_2024(ctx, calculado):
    m = _metricas(ctx, calculado)
    t = ctx.metricas()
    assert set(t["tarea"]) == {"T11"} and set(t["nivel"]) == {"extracto"} and set(t["alcance"]) == {"2019-2022", "2023", "2019-2023"}
    assert not any("2024" in a for a in t["alcance"]) and not any("2024" in c for c in t["clave"]) and (t["alcance"] != "").all()
    assert Counter(c.split(".")[0] for c in t["clave"]) == Counter({"seleccion": sum(1 for c in m if c.startswith("seleccion.")), "proxy": sum(1 for c in m if c.startswith("proxy.")),
                                                                  "fuga": sum(1 for c in m if c.startswith("fuga."))})
    for familia, n in (("seleccion.auc.", 56), ("seleccion.auc_2023.", 56), ("seleccion.incremento_auc.", 56)):
        assert sum(1 for c in m if c.startswith(familia)) == n, familia
    en_l1 = calculado["importancia"].filter((pl.col("tipo_fila") == "variable") & (pl.col("en_modelo_l1") == "si")).height
    assert sum(1 for c in m if c.startswith("seleccion.importancia.")) == en_l1 > 0


def test_la_metrica_de_auc_es_el_auc_senal_publicable_con_su_ic_y_la_nota_de_categorica(ctx, calculado):
    m = _metricas(ctx, calculado)
    pub = {(f["variable"], f["ventana"]): f for f in calculado["ranking_pub"].iter_rows(named=True)}
    e = m["seleccion.auc.EDAD_ANIOS"]
    f = pub[("EDAD_ANIOS", "2019-2022")]
    assert (e["valor"], e["ic_bajo"], e["ic_alto"], e["tabla"], e["alcance"], e["nota"]) == (pytest.approx(f["auc_senal"]), pytest.approx(f["auc_senal_bajo"]), pytest.approx(f["auc_senal_alto"]),
                                                                                              "t23_ranking_variables.csv", "2019-2022", "")
    c = m["seleccion.auc.GRUPO_CLINICO"]
    g = pub[("GRUPO_CLINICO", "2019-2022")]
    assert c["valor"] == pytest.approx(g["auc_senal"]) and c["valor"] != pytest.approx(g["auc"]) and c["nota"] == "categórica: validación cruzada"           # el valor no es el AUC en muestra
    assert c["ic_bajo"] == pytest.approx(g["auc_senal_bajo"]) and c["ic_bajo"] != pytest.approx(g["auc_bajo"])                                              # M46: el IC es el de auc_senal
    d = m["seleccion.auc_2023.EDAD_ANIOS"]
    assert d["alcance"] == "2023" and d["valor"] == pytest.approx(pub[("EDAD_ANIOS", "2023")]["auc"]) and d["ic_bajo"] == pytest.approx(pub[("EDAD_ANIOS", "2023")]["auc_bajo"])


def test_una_categorica_sin_validacion_cruzada_se_registra_como_optimista(ctx, calculado):
    sin_cv = calculado["ranking"].with_columns(pl.when(pl.col("variable") == "GRUPO_CLINICO").then(None).otherwise(pl.col("auc_cv")).alias("auc_cv"))
    m = _metricas(ctx, {**calculado, "ranking": sin_cv})
    assert m["seleccion.auc.GRUPO_CLINICO"]["nota"] == "categórica: optimista" and m["seleccion.auc.CIE10_3C"]["nota"] == "categórica: validación cruzada"
    assert m["seleccion.auc.ELIX_01"]["nota"] == ""


def test_una_fila_fuente_suprimida_registra_la_metrica_con_valor_nulo_y_la_nota(ctx, calculado):
    m = _metricas(ctx, calculado)
    suprimida = calculado["ranking_pub"].filter((pl.col("ventana") == "2019-2022") & pl.col("auc_senal").is_null() & (pl.col("estado") == "ok"))["variable"].to_list()
    assert suprimida, "el panel artificial debe tener alguna fila del ranking suprimida"
    for nombre in suprimida:
        f = m[f"seleccion.auc.{nombre}"]
        assert f["valor"] is None and f["ic_bajo"] is None and f["ic_alto"] is None and sospechas.MOTIVO_SUPRIMIDO in f["nota"], nombre
    assert m["seleccion.auc.ELIX_28"]["valor"] is None and m["seleccion.auc.ELIX_28"]["nota"] == "estado: constante"            # una constante: nulo por su estado y no por supresión


def test_el_incremento_y_la_importancia_son_de_2023_y_el_modelo_base_no_aplica(ctx, calculado):
    m = _metricas(ctx, calculado)
    imp = {f["variable"]: f for f in calculado["importancia_pub"].filter(pl.col("tipo_fila") == "variable").iter_rows(named=True)}
    i = m["seleccion.incremento_auc.NIVEL_CUIDADO_INGRESO"]
    assert i["alcance"] == "2023" and i["tabla"] == "t23_importancia.csv" and i["valor"] == pytest.approx(imp["NIVEL_CUIDADO_INGRESO"]["incremento_auc"])
    base = m["seleccion.incremento_auc.EDAD_ANIOS"]
    assert base["valor"] is None and base["nota"] == "no aplica: variable del modelo base"
    p = m["seleccion.importancia.NIVEL_CUIDADO_INGRESO"]
    assert p["alcance"] == "2023" and p["valor"] == pytest.approx(imp["NIVEL_CUIDADO_INGRESO"]["importancia_permutacion"])
    assert "seleccion.importancia.CIE10_3C" in m and "seleccion.importancia.ESPECIALIDAD_MEDICA" in m                      # las condicionales sí entran al L1
    assert "seleccion.importancia.PROVINCIA" not in m and "seleccion.incremento_auc.PROVINCIA" in m                         # una excluida por la Fase 1 tiene incremento pero no entra al L1


def test_los_conteos_de_la_matriz_se_registran_como_variables_y_suman_el_total(ctx, calculado):
    m = _metricas(ctx, calculado)
    claves = ["n_mantener", "n_mantener_clinico", "n_excluir_fase1", "n_excluir_fuga", "n_excluir_proxy", "n_condicional", "n_redundante", "n_baja_senal", "n_no_evaluable"]
    assert m["seleccion.n_total"]["valor"] == 199 and sum(m[f"seleccion.{c}"]["valor"] for c in claves) == 199
    res = calculado["resumen"].row(0, named=True)
    assert m["seleccion.n_condicional"]["valor"] == res["n_recomendacion_condicional_proxy"] + res["n_recomendacion_condicional_calidad"]
    for c in ["n_total", *claves]:
        f = m[f"seleccion.{c}"]
        assert (f["alcance"], f["nota"], f["tabla"]) == ("2019-2023", "ranking 2019 a 2022 y confirmación 2023", "matriz_decision_variables.csv"), c


def test_las_metricas_de_elixhauser_y_de_los_modelos_de_referencia(ctx, calculado):
    m = _metricas(ctx, calculado)
    res = calculado["resumen"].row(0, named=True)
    assert m["seleccion.elix.n_coinciden_fase1"]["valor"] == res["elix_n_coinciden_fase1"] and m["seleccion.elix.n_coinciden_repo"]["valor"] == res["elix_n_coinciden_repo"]
    assert m["seleccion.elix.n_coinciden_repo"]["alcance"] == "2019-2022"
    for clave, col in (("seleccion.auc_ref_edad_sexo", "auc_ref_edad_sexo_2023"), ("seleccion.auc_ref_elix31", "auc_ref_elix31_2023"), ("seleccion.auc_modelo_l1", "auc_modelo_l1_2023")):
        assert m[clave]["alcance"] == "2023" and m[clave]["valor"] == pytest.approx(res[col]) and m[clave]["tabla"] == "t23_importancia.csv", clave


def test_tambien_se_registran_las_metricas_de_proxy_y_fuga_desde_las_versiones_publicables(ctx, calculado):
    m = _metricas(ctx, calculado)
    assert {"proxy.n_alto", "proxy.n_moderado", "proxy.uci_phi", "fuga.n_supera_umbral", "fuga.concordancia_pct_sin_traslado", "proxy.v.SERVICIO_SALUD", "fuga.auc.IR_29301_MORTALIDAD"} <= set(m)
    res = calculado["resumen"].row(0, named=True)
    assert m["proxy.n_alto"]["valor"] == res["n_proxy_alto"] and m["fuga.n_supera_umbral"]["valor"] == res["fuga_n_supera_umbral"]
    assert m["fuga.concordancia_pct_sin_traslado"]["valor"] == pytest.approx(100.0 * res["concordancia_prop_sin_traslado"])


# ------------------------------------------------------------------------------------------------------ ejecutar y el informe
TABLAS_T11 = {
    "t23_ranking_variables.csv": [*sel.COLS_N_RANKING, "n_faltante"], "t23_estabilidad.csv": ["n", "n_eventos", "n_base", "n_nuevo"], "t23_proxy_hospital.csv": ["n", "n_2023"],
    "t23_or_elix.csv": sensibilidad_elix.COLS_N_OR, "t23_sensibilidad_elix.csv": sensibilidad_elix.COLS_N_SENSIBILIDAD,
    "t23_importancia.csv": ["n_entrenamiento", "n_eventos_entrenamiento", "n_2023", "n_eventos_2023"],
    "t23_fuga.csv": ["n", "n_eventos", "n_2023", "n_eventos_2023", "n_sin_traslado", "n_iguales_sin_traslado", "n_con_traslado", "n_iguales_con_traslado"],
    "t23_resumen_seleccion.csv": [], "t23_parametros_seleccion.csv": [], "disponibilidad_variables.csv": [], "matriz_decision_variables.csv": [],
    "t23_panel_muestreo.csv": ["n_cohorte_d", "n_muestra", "n_eventos_muestra"], "t23_panel_niveles.csv": ["n"],
}


@pytest.fixture(scope="module")
def ejecutada(tmp_path_factory, panel):
    """``seleccion.ejecutar`` sobre el panel artificial en una carpeta propia (los tests de escritura y de informe la comparten; los de informe solo agregan informes)."""
    raiz = tmp_path_factory.mktemp("t11_ejecutada")
    ctx = Contexto.crear("extracto", fuente=crear_fuente_sintetica(raiz / "datos", n=60, semilla=29), dir_salida_base=raiz / "salidas")
    r = sel.ejecutar(ctx, panel)
    ctx.escribir_registros()
    return ctx, r


def test_ejecutar_escribe_las_tablas_con_sus_conteos_registrados_y_sus_columnas(ejecutada):
    ctx, r = ejecutada
    indice = ctx.indice_tablas()
    assert set(indice["nombre"]) == set(TABLAS_T11) and set(indice["tarea"]) == {"T11"}
    for nombre, conteos in TABLAS_T11.items():
        fila = indice.filter(pl.col("nombre") == nombre)
        assert [c for c in (fila["columnas_conteo"][0] or "").split(";") if c] == conteos, nombre
    esperadas = {
        "t23_ranking_variables.csv": sel.COLUMNAS_RANKING, "t23_estabilidad.csv": sel.COLUMNAS_ESTABILIDAD, "t23_proxy_hospital.csv": sospechas.COLUMNAS_PROXY,
        "t23_or_elix.csv": sensibilidad_elix.COLUMNAS_OR_ELIX, "t23_sensibilidad_elix.csv": sensibilidad_elix.COLUMNAS_SENSIBILIDAD, "t23_importancia.csv": sel.COLUMNAS_IMPORTANCIA,
        "t23_fuga.csv": sospechas.COLUMNAS_FUGA, "t23_resumen_seleccion.csv": sel.COLUMNAS_RESUMEN, "t23_parametros_seleccion.csv": sel.COLUMNAS_PARAMETROS,
        "matriz_decision_variables.csv": decision.COLUMNAS_MATRIZ,
    }
    for nombre, columnas in esperadas.items():
        assert pl.read_csv(ctx.dir_tablas / nombre, n_rows=0).columns == columnas, nombre
    filas = dict(zip(indice["nombre"], indice["filas"]))
    assert filas["disponibilidad_variables.csv"] == filas["matriz_decision_variables.csv"] == 199 and filas["t23_resumen_seleccion.csv"] == 1 and filas["t23_parametros_seleccion.csv"] == 15
    assert filas["t23_ranking_variables.csv"] == 112 and filas["t23_sensibilidad_elix.csv"] == 31 and filas["t23_or_elix.csv"] == 62


def test_ejecutar_dibuja_las_seis_figuras_con_el_csv_y_la_decision_del_plan_y_cumple_privacidad(ejecutada):
    ctx, r = ejecutada
    plan = config.leer_csv_texto(config.rutas.DIR_CONFIG / "plan_figuras.csv")
    esperado = {f["id_figura"]: (f["tabla_csv"], f["decision_relacionada"] or "") for f in plan.iter_rows(named=True) if f["id_figura"].startswith("F11_")}
    assert sorted(esperado) == list(sel.IDS_FIGURAS)
    figuras = pl.read_csv(ctx.dir_tablas / "figuras_indice.csv", infer_schema_length=1000)
    leidas = {f["id_figura"]: (f["csv"], f["decision_relacionada"] or "") for f in figuras.iter_rows(named=True) if f["id_figura"].startswith("F11_")}
    assert leidas == esperado and r["figuras_omitidas"] == [] and (figuras["uso_interno"] == "no").all()
    assert all((ctx.dir_figuras / f"{i}.png").is_file() for i in esperado)
    esc = privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)
    assert privacidad.total_violaciones(esc) == 0, esc.filter(pl.col("n_violaciones") > 0)


def test_ejecutar_registra_las_metricas_del_resumen_y_no_toca_2024(ejecutada):
    ctx, r = ejecutada
    m = ctx.metricas()
    assert {"seleccion.n_total", "seleccion.auc.EDAD_ANIOS", "proxy.n_alto", "fuga.n_supera_umbral", "panel.n_filas_ranking"} <= set(m["clave"])
    assert set(m["tarea"]) == {"T11"} and not any("2024" in a for a in m["alcance"])
    res = ctx.leer_tabla("t23_resumen_seleccion.csv", columnas_texto=sorted(sel._TEXTO_RESUMEN)).row(0, named=True)
    assert res["n_matriz"] == 199 and res["figuras_omitidas"] == "ninguna"


def test_ejecutar_no_exige_que_la_tarea_10_haya_escrito_nada(ejecutada):
    ctx, _ = ejecutada
    assert not (ctx.dir_tablas / "matriz_asociacion.csv").exists() and (ctx.dir_tablas / "matriz_decision_variables.csv").exists()               # las tablas de T10 las escribe asociacion.ejecutar


TITULOS = [
    "Universo de variables y disponibilidad", "Señal univariada", "Estabilidad y deriva", "Proxy del hospital", "Comorbilidad de Elixhauser", "Valor incremental e importancia", "Fuga",
    "Matriz de decisión y contraste con la Fase 1", "Advertencias y límites",
]


def _informe(ctx, n0=1, nombre="t23_prueba_seleccion.md"):
    from eda.informes import Informe

    inf = Informe(ctx, nombre, "Informe de prueba")
    sel.agregar_secciones(inf, ctx, n0=n0)
    return inf.guardar().read_text(encoding="utf8")


def test_el_informe_agrega_las_nueve_secciones_con_la_octava_de_decision(ejecutada):
    ctx, _ = ejecutada
    md = _informe(ctx)
    assert sel.N_SECCIONES == 9 and decision.N_SECCIONES == 1 and md.count("\n## ") == 9
    for i, titulo in enumerate(TITULOS, start=1):
        assert f"## {i}. {titulo}\n" in md, titulo
    posiciones = [md.index(f"## {i}. {t}") for i, t in enumerate(TITULOS, start=1)]
    assert posiciones == sorted(posiciones) and "{" not in md and "}" not in md


def test_el_informe_numera_las_secciones_desde_n0(ejecutada):
    ctx, _ = ejecutada
    md = _informe(ctx, n0=4, nombre="t23_prueba_seleccion_n0.md")
    for i, titulo in enumerate(TITULOS, start=4):
        assert f"## {i}. {titulo}\n" in md, titulo


def test_el_informe_incrusta_las_seis_figuras_y_dice_la_advertencia_de_extracto(ejecutada):
    ctx, _ = ejecutada
    md = _informe(ctx)
    for fig in sel.IDS_FIGURAS:
        assert f"Figura {fig}." in md and f"../figuras/{fig}.png" in md, fig
    assert "Los datos son un extracto y no una muestra aleatoria simple: nada de lo anterior es concluyente" in md
    assert "no se pueden evaluar y quedan como «NO EVALUABLE en extracto»" in md


def test_el_informe_trae_las_frases_obligatorias_de_la_tarea_11(ejecutada):
    ctx, _ = ejecutada
    md = _informe(ctx)
    for frase in (
        "La matriz recomienda; la decisión es humana", "pendientes de revisión clínica (decisión 12)", "hipótesis H17a", "Hipótesis H02a", "Hipótesis H02b", "(S09)", "queda prohibida siempre (R6)",
        "regla R3 de la Fase 1", "hipótesis H01c", "El relleno de los nulos con la mediana del entrenamiento más un indicador de nulos solo ocurre dentro de los modelos exploratorios",
        "la regla R5 de la Fase 1 lo admite para las líneas base y el panel conserva los nulos", "por bloques de supresión en lugar de separarlas en dos tablas",
        "sin edad ni sexo", "los intervalos de confianza ignoran el agrupamiento por paciente y por hospital".capitalize(), "El año 2024 no se lee", "decisión 4", "(decisión 8)",
    ):
        assert frase in md, frase
    assert "| estadistica | SUAVIZADO_M |" in md and "| modelos | relleno_nulos_modelos |" in md and md.count("| modelos | ") >= 7             # la tabla de parámetros técnicos


def test_las_cifras_del_informe_vienen_del_resumen_y_de_los_umbrales(ejecutada):
    ctx, _ = ejecutada
    _informe(ctx, nombre="t23_prueba_seleccion_cifras.md")
    c = ctx.cifras_informe().filter(pl.col("informe") == "t23_prueba_seleccion_cifras.md")
    numericas = c.filter(pl.col("formato") != "tabla")
    assert numericas.height > 60 and set(numericas["csv"]) <= {"t23_resumen_seleccion.csv", "umbrales.csv", "matriz_decision_variables.csv"} and "t23_resumen_seleccion.csv" in set(numericas["csv"])
    umbrales = set(numericas.filter(pl.col("csv") == "umbrales.csv")["filtro"].str.extract(r'"umbral": "(\w+)"', 1))
    assert {"auc_senal_alta", "auc_senal_media", "auc_senal_baja", "faltante_alerta", "faltante_alto", "psi_alto", "psi_moderado", "proxy_hospital_alto", "proxy_hospital_moderado", "or_relevante",
            "importancia_permutacion_min", "auc_sospecha_fuga", "estabilidad_efecto_minimo"} <= umbrales
    tablas = set(c.filter(pl.col("formato") == "tabla")["csv"])
    assert {"disponibilidad_variables.csv", "t23_ranking_variables.csv", "t23_estabilidad.csv", "t23_proxy_hospital.csv", "t23_sensibilidad_elix.csv", "t23_importancia.csv", "t23_fuga.csv",
            "t23_parametros_seleccion.csv", "matriz_decision_variables.csv"} <= tablas


def test_el_informe_formatea_el_faltante_de_la_tabla_de_las_doce_con_prop1(ejecutada):
    ctx, _ = ejecutada
    md = _informe(ctx, nombre="t23_prueba_seleccion_prop.md")
    fila = next(ln for ln in md.splitlines() if ln.startswith("| 5 | SERVICIOINGRESO |"))                    # SERVICIOINGRESO tiene más de la mitad de faltantes en el panel artificial
    assert fila.rstrip(" |").endswith(" %") and "," in fila.split("|")[-2]


def test_sin_confirmacion_se_omite_f11_05_se_anota_y_el_informe_no_la_incrusta(ctx, panel):
    sin_2023 = dataclasses.replace(panel, datos=panel.datos.filter(pl.col("anio") != 2023), anio_confirmacion=None)
    r = sel.ejecutar(ctx, sin_2023)
    res = r["resumen"].row(0, named=True)
    assert r["figuras_omitidas"] == ["F11_05"] and res["figuras_omitidas"] == "F11_05" and res["confirmacion_disponible"] == "no"
    assert ctx.figuras.ids() == ["F11_01", "F11_02", "F11_03", "F11_04", "F11_06"] and not (ctx.dir_figuras / "F11_05.png").exists()
    assert res["auc_ref_edad_sexo_2023"] is None and res["auc_modelo_l1_2023"] is None and res["n_importan"] == 0
    md = _informe(ctx)
    assert "Figura F11_05." not in md and "Figura F11_01." in md and md.count("\n## ") == 9 and "{" not in md
    m = {f["clave"]: f for f in ctx.metricas().iter_rows(named=True)}
    assert m["seleccion.auc_2023.EDAD_ANIOS"]["valor"] is None and m["seleccion.auc_2023.EDAD_ANIOS"]["nota"] == "confirmación no disponible"
    assert m["seleccion.auc_modelo_l1"]["valor"] is None and m["seleccion.incremento_auc.NIVEL_CUIDADO_INGRESO"]["nota"] == "confirmación no disponible"
    assert privacidad.total_violaciones(privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)) == 0


def test_la_figura_omitida_segun_el_resumen_es_la_que_no_se_dibuja_en_otros_escenarios(ctx, panel, monkeypatch):
    # sin hospitales elegibles ni asociación medible el mapa de proxies queda vacío: F11_02 se omite y las demás no
    original = sospechas.tabla_proxy

    def sin_asociacion(c, p, r):
        t = original(c, p, r)
        return t.with_columns(pl.when(pl.col("tipo_fila") == "variable").then(None).otherwise(pl.col("asociacion_hospital")).alias("asociacion_hospital"))

    monkeypatch.setattr(sospechas, "tabla_proxy", sin_asociacion)
    r = sel.ejecutar(ctx, panel)
    assert r["figuras_omitidas"] == ["F11_02"] and ctx.figuras.ids() == ["F11_01", "F11_03", "F11_04", "F11_05", "F11_06"]


# ============================================================================== publicación de punta a punta (hallazgo 17)
def test_con_una_celda_de_5_casos_en_el_ranking_el_resumen_las_metricas_y_la_matriz_no_traen_el_valor_pero_la_recomendacion_no_cambia(ctx, panel, ejecutada, monkeypatch):
    original = sel.tabla_ranking

    def forzada(c, p):
        t = original(c, p)
        return t.with_columns(pl.when((pl.col("variable") == "EDAD_ANIOS") & (pl.col("ventana") == "2019-2022")).then(5).otherwise(pl.col("n")).alias("n"))     # umbral forzado: 5 casos

    monkeypatch.setattr(sel, "tabla_ranking", forzada)
    r = sel.ejecutar(ctx, panel)
    base_ctx, base = ejecutada
    # el resumen: la mejor variable pasa a la siguiente y la fila suprimida se cuenta; los conteos por clase siguen saliendo de los valores reales
    res, res0 = r["resumen"].row(0, named=True), base["resumen"].row(0, named=True)
    assert res0["mejor_variable"] == "EDAD_ANIOS" and res["mejor_variable"] == "NIVEL_CUIDADO_INGRESO"
    assert res["n_filas_ranking_suprimidas"] == res0["n_filas_ranking_suprimidas"] + 1 and res["n_senal_alta"] == res0["n_senal_alta"]
    # el CSV: ninguna cifra de esa fila
    fila = ctx.leer_tabla("t23_ranking_variables.csv", columnas_texto=sel._TEXTO_RANKING).filter((pl.col("variable") == "EDAD_ANIOS") & (pl.col("ventana") == "2019-2022"))
    assert fila["n"][0] is None and fila["auc"][0] is None and fila["auc_senal"][0] is None
    # las métricas de esa variable: valor nulo y la nota de supresión (la de 2023 no depende de la fila suprimida)
    m = {f["clave"]: f for f in ctx.metricas().iter_rows(named=True)}
    assert m["seleccion.auc.EDAD_ANIOS"]["valor"] is None and sospechas.MOTIVO_SUPRIMIDO in m["seleccion.auc.EDAD_ANIOS"]["nota"]
    assert m["seleccion.auc_2023.EDAD_ANIOS"]["valor"] is not None
    # la matriz: cifras nulas, valor_suprimido = "si" y la misma recomendación que sin supresión
    e = _por_variable(ctx.leer_tabla("matriz_decision_variables.csv", columnas_texto=decision._TEXTOS_MATRIZ))["EDAD_ANIOS"]
    e0 = _por_variable(base_ctx.leer_tabla("matriz_decision_variables.csv", columnas_texto=decision._TEXTOS_MATRIZ))["EDAD_ANIOS"]
    assert e["auc_ranking"] is None and e["auc_senal"] is None and e["valor_suprimido"] == "si" and e0["valor_suprimido"] == "no" and e0["auc_senal"] is not None
    for col in ("regla_decisiva", "reglas_activadas", "recomendacion", "contraste_fase1", "medicion_completa"):
        assert e[col] == e0[col], col
    assert privacidad.total_violaciones(privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)) == 0


def _por_variable(t):
    return {f["variable"]: f for f in t.iter_rows(named=True)}


# =============================================================================== integración sintética (extracto y modo completo)
import shutil  # noqa: E402

from eda import comorbilidad, descripcion  # noqa: E402
from eda.informes import Informe  # noqa: E402


@pytest.fixture(scope="module")
def sintetica(tmp_path_factory):
    """Fuente sintética del generador existente e insumos de las Tareas 4 y 8 generados en un contexto aparte (esas tareas leen 2024 solo con propósito de estructura)."""
    raiz = tmp_path_factory.mktemp("t11_integracion")
    fuente = crear_fuente_sintetica(raiz / "datos", n=700, semilla=29)
    ins = Contexto.crear("extracto", fuente=fuente, dir_salida_base=raiz / "insumos")
    descripcion.ejecutar(ins)
    comorbilidad.ejecutar(ins)
    return raiz, fuente, ins.dir_tablas


def _correr(sintetica, nombre, *, completo=False):
    """Panel real, ``asociacion.ejecutar``, ``seleccion.ejecutar``, informe provisional y registros, en una carpeta propia (el contexto no tiene ninguna lectura de 2024)."""
    raiz, fuente, insumos = sintetica
    if completo:
        ctx = Contexto.crear("completo", fuente=dataclasses.replace(fuente, etiqueta="completo"), dir_salida_base=raiz / nombre)
    else:
        ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=raiz / nombre)
    for archivo in ("t22_perfil_columnas.csv", "t23_elix_phi.csv"):
        shutil.copy(insumos / archivo, ctx.dir_tablas / archivo)
    panel = pn.construir_panel(ctx, tarea="T11")
    a = asoc.ejecutar(ctx, panel)
    s = sel.ejecutar(ctx, panel, a)
    inf = Informe(ctx, "t23_prueba_seleccion.md", "Informe de prueba")
    sel.agregar_secciones(inf, ctx, n0=1)
    inf.guardar()
    ctx.escribir_registros()
    return ctx, panel, a, s


@pytest.fixture(scope="module")
def extracto(sintetica):
    return _correr(sintetica, "salidas_extracto")


@pytest.fixture(scope="module")
def completo(sintetica):
    return _correr(sintetica, "salidas_completo", completo=True)


T10_T11 = [
    "matriz_asociacion.csv", "pares_redundantes.csv", "clusters_variables.csv", "t23_dendrograma.csv", "t23_vif.csv", "t23_resumen_asociacion.csv", "t23_panel_muestreo.csv", "t23_panel_niveles.csv",
    "disponibilidad_variables.csv", "matriz_decision_variables.csv", "t23_ranking_variables.csv", "t23_estabilidad.csv", "t23_proxy_hospital.csv", "t23_or_elix.csv",
    "t23_sensibilidad_elix.csv", "t23_importancia.csv", "t23_fuga.csv", "t23_resumen_seleccion.csv", "t23_parametros_seleccion.csv",
]


def test_integracion_extracto_escribe_las_19_tablas_las_10_figuras_y_el_indice_coincide_con_el_plan(extracto):
    ctx, panel, a, s = extracto
    assert len(T10_T11) == 19 and all((ctx.dir_tablas / n).is_file() for n in T10_T11)
    plan = config.leer_csv_texto(config.rutas.DIR_CONFIG / "plan_figuras.csv")
    esperado = {f["id_figura"]: (f["tabla_csv"], f["decision_relacionada"] or "") for f in plan.iter_rows(named=True) if f["id_figura"].startswith(("F10_", "F11_"))}
    figuras = pl.read_csv(ctx.dir_tablas / "figuras_indice.csv", infer_schema_length=1000)
    leidas = {f["id_figura"]: (f["csv"], f["decision_relacionada"] or "") for f in figuras.iter_rows(named=True)}
    assert leidas == esperado and len(esperado) == 10 and esperado["F10_02"] == ("t23_dendrograma.csv", "") and esperado["F10_04"] == ("t23_elix_phi.csv", "8")
    assert esperado["F11_01"] == ("t23_ranking_variables.csv", "2; 14") and esperado["F11_05"] == ("t23_importancia.csv", "")
    assert all((ctx.dir_figuras / f"{i}.png").is_file() for i in esperado) and s["figuras_omitidas"] == []


def test_integracion_extracto_las_tablas_citadas_por_las_hipotesis_y_las_fichas_existen(extracto):
    import re

    ctx, _, _, _ = extracto
    citadas = set()
    for h in config.cargar_hipotesis().iter_rows(named=True):
        if h["tarea"] == "11":
            citadas |= set(re.findall(r"[\w.]+\.csv", h["tablas_clave"]))
    fichas = config.leer_csv_texto(config.rutas.DIR_CONFIG / "conocimiento_clinico.csv")
    citadas |= set(re.findall(r"[\w.]+\.csv", " ".join(fichas["evidencia_eda"])))
    nuestras = {n for n in citadas if n in T10_T11}
    assert {"t23_proxy_hospital.csv", "t23_fuga.csv", "t23_sensibilidad_elix.csv", "t23_or_elix.csv", "matriz_decision_variables.csv"} <= nuestras
    assert all((ctx.dir_tablas / n).is_file() for n in nuestras | {"t23_estabilidad.csv"})


def test_integracion_extracto_las_12_categorias_constantes_salen_no_evaluables_y_ninguna_es_mantener_sin_medicion(extracto):
    ctx, _, _, s = extracto
    m = ctx.leer_tabla("matriz_decision_variables.csv", columnas_texto=decision._TEXTOS_MATRIZ)
    constantes = [f"ELIX_{i:02d}" for i in (3, 4, 7, 9, 16, 17, 18, 21, 24, 26, 29, 30)]
    por = _por_variable(m)
    for n in constantes:
        f = por[n]
        assert (f["regla_decisiva"], f["recomendacion"], f["motivo_no_evaluable"], f["contraste_fase1"], f["medicion_completa"]) == (
            "(ninguna)", "NO EVALUABLE", "sin medición: constante", "fase1_exige_conservar_sin_evidencia", "no"), n
    assert s["resumen"]["n_sin_medicion"][0] == 12 and s["resumen"]["n_contraste_fase1_exige_conservar_sin_evidencia"][0] == 12
    assert (m.filter(pl.col("regla_decisiva") == "MANTENER")["medicion_completa"] == "si").all()
    elix = m.filter(pl.col("variable").str.starts_with("ELIX_"))
    assert elix.height == 31 and (elix["restriccion_fase1"] == decision.LEYENDA_R3).all()
    assert (m.filter(~pl.col("variable").str.starts_with("ELIX_"))["restriccion_fase1"].fill_null("") == "").all()
    for n in decision.HISTORIALES:
        assert por[n]["recomendacion"] == "NO EVALUABLE en extracto" and por[n]["evaluable"] == "no", n
    assert m["variable"].to_list() == ctx.leer_tabla("disponibilidad_variables.csv", columnas_texto=sel._TEXTO_DISPONIBILIDAD)["variable"].to_list()                  # DT09


def test_integracion_extracto_ninguna_no_evaluable_tiene_cifras_y_los_parametros_no_traen_la_semilla(extracto):
    ctx, _, _, _ = extracto
    m = ctx.leer_tabla("matriz_decision_variables.csv", columnas_texto=decision._TEXTOS_MATRIZ)
    no = m.filter(pl.col("evaluable") == "no")
    assert no.height > 100
    for c in ("auc_ranking", "auc_senal", "auc_confirmacion", "incremento_auc", "asociacion_hospital", "psi_vs_2023", "psi_2019_vs_resto", "psi_max"):
        assert no[c].null_count() == no.height, c
    p = ctx.leer_tabla("t23_parametros_seleccion.csv", columnas_texto=sel.COLUMNAS_PARAMETROS)
    assert p.height == 15 and str(ctx.semilla) not in (ctx.dir_tablas / "t23_parametros_seleccion.csv").read_text(encoding="utf8")
    assert {"SUAVIZADO_M", "RIDGE_AUC", "RIDGE_WALD", "CS_L1", "TOPE_FILAS_MODELOS", "REPETICIONES_PERMUTACION", "SEPARACION_COEF_MAX", "SEPARACION_SE_MAX", "MIN_FILAS_HOSPITAL",
            "ETIQUETA_V_RESULTADO", "PLIEGUES_CV", "relleno_nulos_modelos"} <= set(p["parametro"])


def test_integracion_extracto_las_columnas_nuevas_cumplen_sus_invariantes(extracto):
    ctx, _, _, _ = extracto
    r = ctx.leer_tabla("t23_ranking_variables.csv", columnas_texto=sel._TEXTO_RANKING)
    con = r.filter(pl.col("auc_senal").is_not_null() & pl.col("auc_senal_bajo").is_not_null() & pl.col("auc_senal_alto").is_not_null())
    assert con.height > 40 and (con["auc_senal_bajo"] <= con["auc_senal"]).all() and (con["auc_senal"] <= con["auc_senal_alto"]).all()
    assert r["prop_faltante"].drop_nulls().min() >= 0 and r["prop_faltante"].drop_nulls().max() <= 1
    e = ctx.leer_tabla("t23_estabilidad.csv", columnas_texto=sel._TEXTO_ESTABILIDAD)
    anual = e.filter(pl.col("tipo_fila") == "auc_anual")
    assert set(anual.filter(pl.col("tipo") == "cat")["auc_optimista"].to_list()) == {"si", "no"} and set(anual.filter(pl.col("tipo") != "cat")["auc_optimista"]) == {"no"}
    assert set(e.filter(pl.col("tipo_fila") != "auc_anual")["auc_optimista"].drop_nulls()) == set()
    f = ctx.leer_tabla("t23_fuga.csv", columnas_texto=sel._TEXTO_FUGA_LISTA)
    assert f["estado"].null_count() == 0 and f.filter(pl.col("tipo_fila") == "variable")["estado"].is_in(["ok", "constante", "sin ambos desenlaces", "no evaluable en la ventana"]).all()
    s_ = ctx.leer_tabla("t23_sensibilidad_elix.csv", columnas_texto=sel._TEXTO_SENSIBILIDAD)
    assert s_.columns == sensibilidad_elix.COLUMNAS_SENSIBILIDAD and s_.height == 31


def test_integracion_extracto_ninguna_cifra_de_la_matriz_ni_de_las_metricas_esta_donde_su_fila_fuente_quedo_nula(extracto):
    ctx, _, _, s = extracto
    matriz = _por_variable(ctx.leer_tabla("matriz_decision_variables.csv", columnas_texto=decision._TEXTOS_MATRIZ))
    metricas = {f["clave"]: f for f in ctx.metricas().iter_rows(named=True)}
    rank = ctx.leer_tabla("t23_ranking_variables.csv", columnas_texto=sel._TEXTO_RANKING)
    r22, r23 = (rank.filter(pl.col("ventana") == v) for v in ("2019-2022", "2023"))
    for f in r22.iter_rows(named=True):
        if f["auc_senal"] is None:
            assert matriz[f["variable"]]["auc_senal"] is None and metricas[f"seleccion.auc.{f['variable']}"]["valor"] is None, f["variable"]
        if f["auc"] is None:
            assert matriz[f["variable"]]["auc_ranking"] is None, f["variable"]
    for f in r23.iter_rows(named=True):
        if f["auc"] is None:
            assert matriz[f["variable"]]["auc_confirmacion"] is None and metricas[f"seleccion.auc_2023.{f['variable']}"]["valor"] is None, f["variable"]
    for f in ctx.leer_tabla("t23_proxy_hospital.csv", columnas_texto=sel._TEXTO_PROXY_LISTA).filter(pl.col("tipo_fila") == "variable").iter_rows(named=True):
        if f["asociacion_hospital"] is None:
            assert matriz[f["variable"]]["asociacion_hospital"] is None and metricas[f"proxy.v.{f['variable']}"]["valor"] is None, f["variable"]
    for f in ctx.leer_tabla("t23_importancia.csv", columnas_texto=sel._TEXTO_IMPORTANCIA).filter(pl.col("tipo_fila") == "variable").iter_rows(named=True):
        if f["incremento_auc"] is None:
            assert matriz[f["variable"]]["incremento_auc"] is None and metricas[f"seleccion.incremento_auc.{f['variable']}"]["valor"] is None, f["variable"]
    for f in ctx.leer_tabla("t23_estabilidad.csv", columnas_texto=sel._TEXTO_ESTABILIDAD).filter(pl.col("tipo_fila") == "psi").iter_rows(named=True):
        if f["psi"] is None:
            col = "psi_vs_2023" if f["comparacion"] == decision.COMP_CONFIRMACION else "psi_2019_vs_resto"
            assert matriz[f["variable"]][col] is None and matriz[f["variable"]]["psi_max"] is None, (f["variable"], col)
    mejor = r22.filter(pl.col("auc_senal").is_not_null()).sort("auc_senal", descending=True)
    assert s["resumen"]["mejor_auc_senal"][0] == pytest.approx(mejor["auc_senal"][0]) and s["resumen"]["mejor_variable"][0] == mejor["variable"][0]


def test_integracion_extracto_el_modelo_y_la_fuga_dan_lo_esperado_del_generador(extracto):
    ctx, _, _, s = extracto
    f = _por_variable(s["fuga"].filter(pl.col("tipo_fila") == "variable"))
    assert f["IR_29301_MORTALIDAD"]["prohibida_siempre"] == "si" and f["IR_29301_MORTALIDAD"]["auc_ranking"] > 0.85 and f["IR_29301_MORTALIDAD"]["supera_umbral_fuga"] == "si"
    assert 0.80 < f["IR_29301_SEVERIDAD"]["auc_ranking"] < 0.85 and f["IR_29301_SEVERIDAD"]["supera_umbral_fuga"] == "no"
    m = _por_variable(s["matriz"])
    assert "FUGA" in m["IR_29301_MORTALIDAD"]["reglas_activadas"].split("; ") and m["IR_29301_MORTALIDAD"]["contraste_fase1"] == "coinciden"
    assert m["ESPECIALIDAD_MEDICA"]["regla_decisiva"] == "PROXY" and m["PROVINCIA"]["rol_fase1"] == "AUDITORIA_SENSIBILIDAD"
    res = s["resumen"].row(0, named=True)
    assert res["n_proxy_alto"] >= 3 and res["fuga_n_supera_umbral"] == 1 and res["espec_n_niveles_en_un_hospital"] == res["espec_n_niveles_totales"] > 0
    assert res["n_sin_medicion"] == 12 and res["t22_disponible"] == "si" and res["historial_evaluable"] == "no" and res["confirmacion_disponible"] == "si"


def test_integracion_extracto_cumple_privacidad_cortafuegos_y_no_lee_2024(extracto):
    ctx, _, _, _ = extracto
    esc = privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)
    assert privacidad.total_violaciones(esc) == 0, esc.filter(pl.col("n_violaciones") > 0)
    assert ctx.cortafuegos.violaciones() == 0 and ctx.cortafuegos.n_bloqueos() == 0
    reg = ctx.cortafuegos.registro()
    assert 2024 not in set(reg["anio"]) and set(reg["tarea"]) == {"T11"} and set(reg["anio"]) == {2019, 2020, 2021, 2022, 2023}
    m = ctx.metricas()
    assert 2024 not in {a for a in m["alcance"]} and not any("2024" in c for c in m["clave"]) and set(m["nivel"]) == {"extracto"} and (m["alcance"] != "").all()
    assert set(c.split(".")[0] for c in m["clave"]) >= {"asociacion", "vif", "seleccion", "fuga", "proxy", "panel"}


def test_integracion_extracto_el_informe_se_arma_sin_llaves_con_la_palabra_extracto_las_nueve_secciones_y_las_seis_figuras(extracto):
    ctx, _, _, _ = extracto
    md = (ctx.dir_informes / "t23_prueba_seleccion.md").read_text(encoding="utf8")
    assert "{" not in md and "}" not in md and "extracto" in md.lower() and md.count("\n## ") == 9
    for i, titulo in enumerate(TITULOS, start=1):
        assert f"## {i}. {titulo}\n" in md, titulo
    for fig in sel.IDS_FIGURAS:
        assert f"Figura {fig}." in md and f"../figuras/{fig}.png" in md
    assert ctx.cifras_informe().filter(pl.col("informe") == "t23_prueba_seleccion.md").height > 60


def test_integracion_modo_completo_incorpora_los_historiales_al_ranking_la_estabilidad_y_la_matriz_sin_la_etiqueta_de_extracto(completo):
    ctx, panel, a, s = completo
    assert panel.con_historial and panel.regimenes == [[2019, 2020], [2021, 2022, 2023]]
    historiales = set(decision.HISTORIALES)
    assert historiales <= set(s["ranking"]["variable"]) and historiales <= set(s["estabilidad"]["variable"]) and historiales <= set(s["matriz"]["variable"])
    m = _por_variable(s["matriz"])
    for n in historiales:
        assert m[n]["evaluable"] != "no" and m[n]["recomendacion"] != "NO EVALUABLE en extracto" and "t23_historial_factibilidad.csv" in m[n]["evidencia"], n
    res = s["resumen"].row(0, named=True)
    assert res["historial_evaluable"] == "si" and set(ctx.metricas()["nivel"]) == {"completo"} and set(ctx.indice_tablas()["fuente"]) == {"completo"}
    assert not any("extracto" in str(x).lower() for x in s["matriz"]["recomendacion"].unique())


def test_integracion_modo_completo_el_informe_trae_la_tabla_de_historial_y_no_la_advertencia_de_extracto(completo):
    ctx, _, _, _ = completo
    md = (ctx.dir_informes / "t23_prueba_seleccion.md").read_text(encoding="utf8")
    assert "extracto" not in md.lower() and "completo" in md and "{" not in md and "}" not in md
    assert "Variables de historial: incremento, proxy y recomendación de la matriz" in md and "Las tres variables de historial" in md
    assert "El historial del paciente no es evaluable en esta corrida" not in md and md.count("\n## ") == 9
    esc = privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)
    assert privacidad.total_violaciones(esc) == 0, esc.filter(pl.col("n_violaciones") > 0)
    assert ctx.cortafuegos.violaciones() == 0 and ctx.cortafuegos.n_bloqueos() == 0 and 2024 not in set(ctx.cortafuegos.registro()["anio"])


def test_determinismo_dos_corridas_en_carpetas_distintas_dan_las_mismas_sumas_de_verificacion_las_mismas_metricas_y_el_mismo_informe(sintetica, extracto):
    ctx1, _, _, _ = extracto
    ctx2, _, _, _ = _correr(sintetica, "salidas_extracto_repetida")
    assert ctx1.dir_salida != ctx2.dir_salida
    cols = ["nombre", "tarea", "filas", "columnas_conteo", "celdas_suprimidas", "fuente", "sha256"]
    t1, t2 = (pl.read_csv(c.dir_tablas / "tablas_indice.csv") for c in (ctx1, ctx2))
    assert t1.select(cols).equals(t2.select(cols)) and set(T10_T11) <= set(t1["nombre"])
    assert ctx1.metricas().equals(ctx2.metricas())
    assert (ctx1.dir_informes / "t23_prueba_seleccion.md").read_text(encoding="utf8") == (ctx2.dir_informes / "t23_prueba_seleccion.md").read_text(encoding="utf8")


# ====================================================================== informe con tablas sin contenido y modo completo sin historial
def _vaciadas(calculado):
    """Las tablas publicables de ``calculado`` sin ninguna fila con contenido para las tablas del informe (sin señal, sin deriva, sin proxy, sin índice, sin importancia, sin fuga)."""
    nulo = lambda c: pl.lit(None, dtype=pl.Float64).alias(c)  # noqa: E731
    ranking = calculado["ranking_pub"].with_columns(nulo("auc_senal"))
    estabilidad = calculado["estabilidad_pub"].with_columns(pl.when(pl.col("tipo_fila") == "resumen").then(pl.lit("no")).otherwise(pl.col("cambio_signo")).alias("cambio_signo"), nulo("psi"))
    proxy = calculado["proxy_pub"].with_columns(nulo("asociacion_hospital"))
    sensibilidad = calculado["sensibilidad_pub"].with_columns(pl.lit(None, dtype=pl.Int64).alias("rango_indice"))
    importancia = calculado["importancia_pub"].with_columns(nulo("importancia_permutacion"))
    fuga = calculado["fuga_pub"].filter(pl.col("tipo_fila") != "variable")
    disponibilidad = calculado["disponibilidad"].with_columns(pl.lit("si").alias("evaluable"))
    return {
        "t23_ranking_variables.csv": ranking, "t23_estabilidad.csv": estabilidad, "t23_proxy_hospital.csv": proxy, "t23_sensibilidad_elix.csv": sensibilidad,
        "t23_importancia.csv": importancia, "t23_fuga.csv": fuga, "disponibilidad_variables.csv": disponibilidad,
    }


def test_el_informe_no_falla_con_tablas_sin_contenido_y_no_incrusta_tablas_ni_figuras_vacias(tmp_path, fuente_sintetica, calculado):
    ctx = Contexto.crear("completo", fuente=dataclasses.replace(fuente_sintetica, etiqueta="completo"), dir_salida_base=tmp_path)
    for nombre, tabla in _vaciadas(calculado).items():
        ctx.escribir_tabla(nombre, tabla, tarea="T11", descripcion=nombre)
    ctx.escribir_tabla("matriz_decision_variables.csv", calculado["matriz_pub"], tarea="T11", descripcion="matriz")
    ctx.escribir_tabla("t23_parametros_seleccion.csv", calculado["parametros"], tarea="T11", descripcion="parametros")
    resumen = calculado["resumen"].with_columns(pl.lit("no").alias("confirmacion_disponible"), pl.lit("no").alias("historial_evaluable"))
    ctx.escribir_tabla("t23_resumen_seleccion.csv", resumen, tarea="T11", descripcion="resumen")
    md = _informe(ctx, nombre="t23_prueba_vacia.md")
    assert md.count("\n## ") == 9 and "{" not in md and "}" not in md and ctx.figuras.ids() == [] and "Figura F11_" not in md
    for leyenda in ("Las doce variables con mayor señal univariada", "Variables con cambio de signo de la señal entre años", "Las diez variables con mayor PSI contra 2023", "Las doce variables más asociadas con el hospital",
                    "Las diez categorías con mayor índice de complicación probable", "Las doce variables del modelo L1 con mayor importancia", "AUC de las variables de solo fuga", "Variables no evaluables por motivo"):
        assert leyenda not in md, leyenda
    assert "Parámetros técnicos de la selección de variables" in md                                      # la tabla de parámetros siempre se incrusta
    assert "Las variables de historial no son evaluables en esta corrida" in md and "El historial del paciente no es evaluable en esta corrida" in md        # modo completo sin historial
    assert "no se pueden evaluar y quedan como" not in md and "Los datos son un extracto" not in md              # (el motivo de la matriz sí dice "extracto": es el de las tablas de un panel de extracto)


# FIN-test_t11_seleccion
