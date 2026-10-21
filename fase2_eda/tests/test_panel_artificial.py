"""Autoverificación del panel artificial: la verdad de diseño que prometen ``tests/panel_artificial.py`` y sus ayudas."""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
import pytest
from scipy.cluster import hierarchy

from eda import config, elixhauser, rutas
from eda import estadistica as E
from eda import panel as pn
from eda.contexto import Contexto
from panel_artificial import HOSPITALES, REGIMENES_HISTORIAL, SIN_EVENTOS_PREVISION, panel_artificial, umbrales_temporales


@pytest.fixture(scope="module")
def p():
    return panel_artificial()


RANK = [2019, 2020, 2021, 2022]


def _rank(panel):
    return panel.datos["anio"].to_numpy() <= 2022


def _y(panel):
    return panel.datos["y"].to_numpy().astype(float)


def _auc(panel, col, ventana="rank"):
    sel = _rank(panel) if ventana == "rank" else ~_rank(panel)
    return E.auc_por_rangos(panel.datos[col].cast(pl.Float64).to_numpy()[sel], _y(panel)[sel])[0]


def _auc_cat(panel, col):
    cod, niv = panel.codigos(col)
    ra = _rank(panel)
    puntaje = E.puntaje_categorico(cod[ra], _y(panel)[ra], len(niv))
    return E.auc_por_rangos(puntaje[cod[ra]], _y(panel)[ra])[0]


def _v(panel, a, b):
    ca, na = panel.codigos(a)
    cb, nb = panel.codigos(b)
    ra = _rank(panel)
    return E.v_cramer_sesgo(E.tabla_contingencia(ca[ra], cb[ra], len(na), len(nb)))


# ================================================================================ estructura e invariantes
def test_el_panel_cumple_los_invariantes_de_un_panel_real(p):
    d = p.datos
    assert d.height == 1200 and d["anio"].dtype == pl.Int32 and d["y"].dtype == pl.Int8 and set(d["y"].unique()) == {0, 1}
    assert d.columns[:2] == ["anio", "y"]
    prohibidas = {"id_num", "ing", "alta", "CIP_ENCRIPTADO", "FECHA_INGRESO", "FECHA_NACIMIENTO", "FECHAALTA", "HOSPPROCEDENCIA", "NOMBRE_HOSPITAL"}
    assert not prohibidas & set(d.columns)
    assert d["anio"].to_list() == sorted(d["anio"].to_list())                                         # ordenado por año
    conteo = dict(d.group_by("anio").len().iter_rows())
    assert conteo == {2019: 240, 2020: 240, 2021: 240, 2022: 240, 2023: 240} and _rank(p).sum() == 960     # 80 % en 2019-2022
    for nombre, v in p.variables.items():
        if nombre in d.columns:
            esperado = {"num": pl.Float64, "bin": pl.Int8, "cat": pl.String}[v.tipo]
            assert d[nombre].dtype == esperado, nombre
    construibles = [n for n, v in p.variables.items() if v.uso != "no_construible" and not (n in ("N_EGRESOS_12M", "DIAS_DESDE_EGRESO_PREVIO", "HISTORIA_DISPONIBLE"))]
    assert set(construibles) <= set(d.columns) and set(d.columns) - {"anio", "y"} <= set(p.variables)
    assert not {"ESPECIALIDAD_MACRO", "TRAMO_FONASA", "INDICE_PRIVACION"} & set(d.columns)                 # las que no se pueden construir no tienen columna
    assert [c for c in d.columns[2:]] == [n for n in p.variables if n in d.columns]                      # mismo orden que el registro
    assert p.variables == pn.registro_variables(False) and p.con_historial is False and p.regimenes == []
    assert p.anios_ranking == RANK and p.anio_confirmacion == 2023


def test_las_categoricas_tienen_niveles_de_las_funciones_del_panel_real_y_sus_codigos_son_validos(p):
    for nombre in p.nombres(tipo="cat"):
        if nombre not in p.datos.columns:
            continue
        niv = p.niveles[nombre]
        assert set(p.datos[nombre].unique()) <= set(niv) and len(set(niv)) == len(niv), nombre
        cod, _ = p.codigos(nombre)
        assert cod.min() >= 0 and cod.max() < len(niv), nombre
    assert p.niveles["COD_HOSPITAL"] == list(HOSPITALES) and p.nombres(uso="referencia") == ["COD_HOSPITAL"]
    assert p.datos["COD_HOSPITAL"].n_unique() == 5 and sorted(p.datos["COD_HOSPITAL"].unique()) == list(HOSPITALES)
    for nombre in p.nombres(tipo="bin"):
        if nombre in p.datos.columns:
            assert set(p.datos[nombre].drop_nulls().unique()) <= {0, 1}, nombre
    assert "COD_HOSPITAL" not in set(p.niveles_tabla["variable"])                                      # la referencia no entra a la tabla de niveles


def test_el_muestreo_y_la_cardinalidad_de_especialidad_vienen_de_las_funciones_puras(p):
    m = p.muestreo
    assert m["anio"].to_list() == [2019, 2020, 2021, 2022, 2023] and m["ventana"].to_list() == ["ranking"] * 4 + ["confirmacion"]
    assert m["n_cohorte_d"].to_list() == m["n_muestra"].to_list() == [240] * 5 and set(m["muestreado"]) == {"no"} and set(m["tope"]) == {200_000}
    assert m["n_eventos_muestra"].sum() == p.datos["y"].sum()
    c = p.cardinalidad_especialidad
    assert c["n_hospitales"] == 5 and c["card_minima"] == c["card_maxima"] == 3 and c["n_niveles_totales"] == 15 and c["n_niveles_en_un_hospital"] == 15


def test_dos_construcciones_con_la_misma_semilla_dan_el_mismo_panel_y_otra_semilla_otro():
    a, b, c = panel_artificial(), panel_artificial(), panel_artificial(semilla=4)
    assert a.datos.equals(b.datos) and a.datos.hash_rows().sum() == b.datos.hash_rows().sum() and a.niveles == b.niveles
    assert not a.datos.equals(c.datos)


def test_un_modo_de_y_desconocido_lanza_value_error():
    with pytest.raises(ValueError, match="y_modo"):
        panel_artificial(y_modo="otro")


# ====================================================================================== señal y asociaciones
def test_edad_tiene_senal_alta_y_n_dx_sec_se_correlaciona_con_ella_sin_ser_redundante(p):
    assert 0.68 <= _auc(p, "EDAD_ANIOS") <= 0.76
    ra = _rank(p)
    rho, n = E.spearman(p.datos["N_DX_SEC"].to_numpy()[ra], p.datos["EDAD_ANIOS"].to_numpy()[ra])
    assert 0.35 <= rho <= 0.65 and n == 960 and rho < 0.80


def test_los_pares_anidados_tienen_v_de_al_menos_0_99(p):
    for a, b in (("DIA_SEMANA", "ES_FIN_SEMANA"), ("GRUPO_CLINICO", "CIE10_3C"), ("TIPO_PROCEDENCIA", "PROCEDENCIA_AGR")):
        assert _v(p, a, b) >= 0.99, (a, b)
    d = p.datos
    assert (d["ES_FIN_SEMANA"].to_numpy() == (d["DIA_SEMANA"].cast(pl.Int32).to_numpy() >= 5)).all()
    assert len(p.niveles["CIE10_3C"]) == 8 and len(p.niveles["GRUPO_CLINICO"]) == 3 and len(p.niveles["TIPO_PROCEDENCIA"]) == 6 and len(p.niveles["PROCEDENCIA_AGR"]) == 3
    # CIE10_3C es más fina y tiene mayor AUC en muestra; y depende solo del grupo, así que la señal honesta es la misma
    assert _auc_cat(p, "CIE10_3C") > _auc_cat(p, "GRUPO_CLINICO")


def test_dia_de_la_semana_tiene_una_deriva_fuerte_en_2023(p):
    cod, niv = p.codigos("DIA_SEMANA")
    ra = _rank(p)
    assert E.psi_de_codigos(cod[ra], cod[~ra], len(niv)) > 0.25


def test_elix_03_es_ruido_de_auc_cercano_a_0_508_y_elix_28_es_constante(p):
    assert 0.49 <= _auc(p, "ELIX_03") <= 0.53
    assert p.datos["ELIX_28"].sum() == 0 and p.datos["ELIX_28"].n_unique() == 1
    ra = _rank(p)
    prevalencias = [p.datos[f"ELIX_{i:02d}"].to_numpy()[ra].mean() for i in range(4, 28)]
    assert all(0.02 <= x <= 0.09 for x in prevalencias)                                               # ruido de prevalencia cercana a 5 %


def test_elix_31_tiene_6_expuestos_en_la_ventana_de_ranking_y_dos_en_2023(p):
    ra = _rank(p)
    assert p.datos["ELIX_31"].to_numpy()[ra].sum() == 6 and p.datos["ELIX_31"].to_numpy()[~ra].sum() == 2


def test_elix_02_cambia_de_signo_en_2023_y_elix_01_conserva_su_efecto(p):
    d = p.datos

    def razon(col, ventana):
        sel = _rank(p) if ventana == "rank" else ~_rank(p)
        e, y = d[col].to_numpy()[sel], d["y"].to_numpy()[sel]
        return E.or_woolf(((e == 1) & (y == 1)).sum(), ((e == 1) & (y == 0)).sum(), ((e == 0) & (y == 1)).sum(), ((e == 0) & (y == 0)).sum())[0]

    assert razon("ELIX_02", "rank") > 1.0 > razon("ELIX_02", "conf")
    assert razon("ELIX_01", "rank") > 1.0 and razon("ELIX_01", "conf") > 1.0


def test_los_agregados_de_comorbilidad_son_exactamente_la_suma_y_las_combinaciones_de_los_pesos(p):
    d = p.datos
    e = elixhauser.cargar()
    m = np.column_stack([d[i].to_numpy() for i in e.ids]).astype(np.int64)
    pesos = np.array([e.pesos[i] for i in e.ids])
    sin5 = np.array([0 if i in e.sospechosas else e.pesos[i] for i in e.ids])
    assert (d["N_COMORB_ELIX"].to_numpy() == m.sum(axis=1)).all()
    assert (d["PUNTAJE_VAN_WALRAVEN"].to_numpy() == m @ pesos).all() and (d["PUNTAJE_VAN_WALRAVEN_SIN_5"].to_numpy() == m @ sin5).all()
    assert (d["PUNTAJE_VAN_WALRAVEN"].to_numpy() != d["PUNTAJE_VAN_WALRAVEN_SIN_5"].to_numpy()).any()      # las cinco sospechosas pesan


def test_sexo_es_ruido(p):                                                                               # AUC cercano a 0,5 por validación sobre la misma ventana
    assert abs(_auc_cat(p, "SEXO") - 0.5) < 0.06


def test_proxies_del_hospital_alto_moderado_y_ninguno(p):
    assert _v(p, "COD_HOSPITAL", "ESPECIALIDAD_MEDICA") >= 0.99 and _v(p, "COD_HOSPITAL", "SERVICIO_SALUD") >= 0.99
    assert 0.15 <= _v(p, "COD_HOSPITAL", "COMUNA") < 0.30
    assert 0.15 <= _v(p, "COD_HOSPITAL", "NIVEL_CUIDADO_INGRESO") < 0.30
    assert _v(p, "COD_HOSPITAL", "PREVISION") < 0.15 and _v(p, "COD_HOSPITAL", "SEXO") < 0.15 and _v(p, "COD_HOSPITAL", "DERIVADO_OTRO_HOSPITAL") < 0.15
    assert p.datos.group_by("ESPECIALIDAD_MEDICA").agg(pl.col("COD_HOSPITAL").n_unique())["COD_HOSPITAL"].max() == 1       # cada nivel es de un solo hospital


def test_nivel_cuidado_ingreso_tiene_senal_media(p):
    assert 0.60 <= _auc(p, "NIVEL_CUIDADO_INGRESO") <= 0.70


def test_prevision_tiene_un_nivel_de_150_filas_sin_eventos_y_es_independiente_del_hospital(p):
    d = p.datos.filter(pl.col("PREVISION") == "PRAIS")
    assert d.height == SIN_EVENTOS_PREVISION == 150 and d["y"].sum() == 0 and (d["anio"] <= 2022).all()


def test_etnia_y_pueblo_originario_no_se_analizan_frente_al_resultado_y_son_coherentes_entre_si(p):
    assert not p.variables["ETNIA"].analiza_resultado and not p.variables["PUEBLO_ORIGINARIO"].analiza_resultado
    d = p.datos
    assert {a for a in d["PUEBLO_ORIGINARIO"].to_list()} <= {0, 1}
    por_etnia = dict(d.group_by("ETNIA").agg(pl.col("PUEBLO_ORIGINARIO").min()).iter_rows())
    assert por_etnia["NINGUNO"] == 0 and por_etnia["OTRO"] == 0 and por_etnia["MAPUCHE"] == 1 and por_etnia["AYMARA"] == 1


def test_servicio_ingreso_tiene_alrededor_de_55_por_ciento_sin_dato(p):
    assert 0.50 <= (p.datos["SERVICIOINGRESO"] == pn.SIN_DATO).mean() <= 0.60
    assert pn.SIN_DATO in p.niveles["SERVICIOINGRESO"]


def test_ir_29301_mortalidad_es_fuga_de_auc_cercano_a_0_93_y_severidad_no_la_alcanza(p):
    assert 0.90 <= _auc_cat(p, "IR_29301_MORTALIDAD") <= 0.96
    assert 0.70 <= _auc_cat(p, "IR_29301_SEVERIDAD") < 0.85
    assert p.variables["IR_29301_MORTALIDAD"].uso == "solo_fuga"


def test_estancia_tiene_gradiente_en_elix_22_y_elix_25_y_la_cohorte_e_son_los_sobrevivientes_con_estadia(p):
    d = p.datos
    sobrevive = d["y"].to_numpy() == 0
    est = d["ESTANCIA_DIAS"].to_numpy()
    for col in ("ELIX_22", "ELIX_25"):
        e = d[col].to_numpy() == 1
        assert est[sobrevive & e].mean() > 1.3 * est[sobrevive & ~e].mean(), col
    assert (d["ES_COHORTE_E"].to_numpy() == (sobrevive & (est > 0))).all()
    assert (est >= 0).all() and (est == np.round(est)).all() and (est[d["y"].to_numpy() == 1].mean() < est[sobrevive].mean())


def test_es_extranjero_esta_informada_en_el_ranking_y_en_solo_cuatro_filas_de_2023(p):
    d = p.datos
    assert d.filter(pl.col("anio") <= 2022)["ES_EXTRANJERO"].null_count() == 0 and d.filter(pl.col("anio") == 2023)["ES_EXTRANJERO"].null_count() == 236
    assert d.filter(pl.col("anio") == 2023)["ES_EXTRANJERO"].drop_nulls().len() == 4
    assert set(d["ES_EXTRANJERO"].drop_nulls().unique()) == {0, 1}


def test_tipo_ingreso_tiene_un_nivel_de_40_casos_en_el_ranking_y_4_en_2023(p):
    t = p.niveles_tabla.filter((pl.col("variable") == "TIPO_INGRESO") & (pl.col("nivel") == "OBSTETRICA"))
    f = {r["ventana"]: r["n"] for r in t.iter_rows(named=True)}
    assert f == {pn.V_RANK: 40, pn.V_CONF: 4}
    assert t.filter(pl.col("ventana") == pn.V_RANK)["prop"].item() == pytest.approx(40 / 960)


def test_la_tabla_de_niveles_cubre_cada_nivel_en_las_dos_ventanas(p):
    t = p.niveles_tabla
    assert t.schema == pl.Schema(pn.ESQUEMA_NIVELES_TABLA)
    for nombre in p.nombres(tipo="cat"):
        if nombre in p.datos.columns and nombre != "COD_HOSPITAL":
            sub = t.filter(pl.col("variable") == nombre)
            assert sub.filter(pl.col("ventana") == pn.V_RANK)["n"].sum() == 960 and sub.filter(pl.col("ventana") == pn.V_CONF)["n"].sum() == 240, nombre
            assert sub.filter(pl.col("ventana") == pn.V_RANK)["nivel"].to_list() == p.niveles[nombre]


# ====================================================================================== modos de y
def test_y_solo_edad_depende_solo_de_la_edad():
    q = panel_artificial(y_modo="solo_edad")
    assert 0.65 <= _auc(q, "EDAD_ANIOS") <= 0.85 and abs(_auc(q, "ELIX_01") - 0.5) < 0.06 and abs(_auc(q, "ELIX_02") - 0.5) < 0.06


def test_y_solo_elix_depende_solo_de_las_categorias_de_elixhauser():
    q = panel_artificial(y_modo="solo_elix")
    assert abs(_auc(q, "EDAD_ANIOS") - 0.5) < 0.07
    assert _auc(q, "ELIX_01") > 0.60 and _auc(q, "ELIX_02") > 0.60
    # la mezcla de ELIX_01 y ELIX_02 separa bien: el AUC de su suma supera con holgura al de la edad
    d = q.datos
    suma = (d["ELIX_01"] + d["ELIX_02"]).to_numpy().astype(float)
    assert E.auc_por_rangos(suma[_rank(q)], _y(q)[_rank(q)])[0] > 0.68


# ====================================================================================== cadena e historial
def test_sin_la_cadena_no_hay_columnas_de_prueba_y_con_ella_estan_en_el_panel_y_en_su_registro():
    base, con = panel_artificial(), panel_artificial(con_cadena=True)
    assert not any(c.startswith("CADENA_") for c in base.datos.columns) and not any(n.startswith("CADENA_") for n in base.variables)
    assert [c for c in con.datos.columns if c.startswith("CADENA_")] == ["CADENA_A", "CADENA_B", "CADENA_C"]
    assert {n: con.variables[n].tipo for n in ("CADENA_A", "CADENA_B", "CADENA_C")} == {"CADENA_A": "num", "CADENA_B": "num", "CADENA_C": "num"}
    # agregar la cadena no altera el resto del panel (generador aparte)
    assert base.datos.equals(con.datos.drop("CADENA_A", "CADENA_B", "CADENA_C")) and pn.registro_variables(False).keys() <= con.variables.keys()


def test_la_cadena_da_un_componente_y_dos_grupos_al_corte_de_0_20():
    q = panel_artificial(con_cadena=True)
    ra = _rank(q)
    cols = ["CADENA_A", "CADENA_B", "CADENA_C"]
    x = {c: q.datos[c].to_numpy()[ra] for c in cols}
    AB, BC, AC = (E.spearman(x[a], x[b])[0] for a, b in (("CADENA_A", "CADENA_B"), ("CADENA_B", "CADENA_C"), ("CADENA_A", "CADENA_C")))
    # promedios medidos por el diseño: 0,892, 0,838 y 0,544 (mínimos 0,872, 0,807 y 0,481); con la semilla 3 dio 0,899, 0,851 y 0,572
    assert AB == pytest.approx(0.892, abs=0.05) and BC == pytest.approx(0.838, abs=0.05) and AC == pytest.approx(0.544, abs=0.06)
    A = np.array([[1.0, AB, AC], [AB, 1.0, BC], [AC, BC, 1.0]])
    assert len(set(E.componentes_redundantes(A, 0.80).tolist())) == 1                                  # un solo componente {A, B, C}
    D = 1.0 - E.redondear_publicado(A)
    _, Z = E.orden_por_clusters(D)
    grupos = hierarchy.fcluster(Z, t=1.0 - 0.80 + 1e-12, criterion="distance")
    assert grupos[0] == grupos[1] != grupos[2]                                                          # grupos {A, B} y {C}
    assert 0.20 < Z[:, 2].max() < 0.45                                                                  # C se une a una altura de 0,28 a 0,36 (el diseño midió ese rango)


def test_con_historial_existen_las_tres_variables_y_los_espacios_de_identificadores():
    q = panel_artificial(con_historial=True)
    assert q.con_historial and q.regimenes == REGIMENES_HISTORIAL == [[2019, 2020], [2021, 2022, 2023]]
    for n in ("N_EGRESOS_12M", "DIAS_DESDE_EGRESO_PREVIO", "HISTORIA_DISPONIBLE"):
        assert n in q.datos.columns
    d = q.datos
    assert dict(d.group_by("anio").agg(pl.col("HISTORIA_DISPONIBLE").max()).iter_rows()) == {2019: 0, 2020: 1, 2021: 0, 2022: 1, 2023: 1}
    assert d.filter(pl.col("HISTORIA_DISPONIBLE") == 0)["N_EGRESOS_12M"].null_count() == 480 and d.filter(pl.col("HISTORIA_DISPONIBLE") == 1)["N_EGRESOS_12M"].null_count() == 0
    con_previo = d.filter(pl.col("N_EGRESOS_12M") > 0)
    assert con_previo["DIAS_DESDE_EGRESO_PREVIO"].null_count() == 0 and d.filter(pl.col("N_EGRESOS_12M").fill_null(0) == 0)["DIAS_DESDE_EGRESO_PREVIO"].null_count() == (d.height - con_previo.height)
    candidatas = [n for n, v in q.variables.items() if v.uso == "candidata" and v.evaluable != "no"]
    assert len(candidatas) == 61 and all(n in d.columns for n in candidatas)


# ====================================================================================== umbrales temporales
def test_umbrales_temporales_cambian_solo_los_valores_pedidos_y_dejan_intacto_el_archivo_real(tmp_path):
    real = rutas.DIR_CONFIG / "umbrales.toml"
    antes = hashlib.sha256(real.read_bytes()).hexdigest()
    base = config.cargar_umbrales()
    u = umbrales_temporales(tmp_path, max_filas_asociacion_por_anio=150, cardinalidad_max_categorica=5, celda_minima=5, redundancia_abs=0.75, cola_estadia_percentiles=[0.9, 0.95])
    assert (u["max_filas_asociacion_por_anio"], u["cardinalidad_max_categorica"], u["celda_minima"], u["redundancia_abs"]) == (150, 5, 5, 0.75)
    assert u["cola_estadia_percentiles"] == [0.9, 0.95]
    cambiados = {"max_filas_asociacion_por_anio", "cardinalidad_max_categorica", "celda_minima", "redundancia_abs", "cola_estadia_percentiles"}
    for nombre in base.nombres():
        if nombre not in cambiados:
            assert u[nombre] == base[nombre], nombre
        assert u.justificacion(nombre) == base.justificacion(nombre)
    assert u.meta == base.meta and u.particion == base.particion and u.cortafuegos == base.cortafuegos
    assert hashlib.sha256(real.read_bytes()).hexdigest() == antes and (tmp_path / "umbrales_temporales.toml").is_file()
    assert config.cargar_umbrales()["celda_minima"] == 10


def test_umbrales_temporales_rechaza_un_umbral_inexistente_y_el_contexto_los_usa(tmp_path, fuente_sintetica):
    with pytest.raises(KeyError, match="no existe"):
        umbrales_temporales(tmp_path, umbral_inventado=3)
    u = umbrales_temporales(tmp_path, celda_minima=5)
    ctx = Contexto.crear("extracto", fuente=fuente_sintetica, dir_salida_base=tmp_path / "salidas", umbrales=u)
    assert ctx.celda_minima == 5 and ctx.semilla == 20261007
