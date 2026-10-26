"""Supresión por bloques (``supresion.suprimir_bloques``) y escritura por bloques (``panel.escribir_por_bloques``)."""

from __future__ import annotations

import itertools

import polars as pl
import pytest

from eda import panel, privacidad, supresion
from eda.contexto import Contexto
from panel_artificial import panel_artificial, umbrales_temporales

BLOQUES_DOS_VENTANAS = [(["n_ranking"], ["prop_ranking"]), (["n_2023"], ["prop_2023"])]


def _dos_ventanas() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "nivel": ["a", "b", "c", "d"],
            "n_ranking": [40, 40, 6, 0],
            "prop_ranking": [0.4, 0.4, 0.06, 0.0],
            "n_2023": [4, 120, 120, 0],
            "prop_2023": [0.04, 0.5, 0.5, 0.0],
        }
    )


def test_cada_bloque_se_suprime_por_separado_a_mano():
    df, n_celdas = supresion.suprimir_bloques(_dos_ventanas(), BLOQUES_DOS_VENTANAS)
    f = {r["nivel"]: r for r in df.iter_rows(named=True)}
    # (40, 4): el bloque de 2019-2022 conserva sus cifras y el de 2023 se anula completo
    assert (f["a"]["n_ranking"], f["a"]["prop_ranking"], f["a"]["n_2023"], f["a"]["prop_2023"]) == (40, 0.4, None, None)
    # (40, 120): no cambia
    assert (f["b"]["n_ranking"], f["b"]["prop_ranking"], f["b"]["n_2023"], f["b"]["prop_2023"]) == (40, 0.4, 120, 0.5)
    # (6, 120): se anula el bloque de 2019-2022 y se conserva el de 2023
    assert (f["c"]["n_ranking"], f["c"]["prop_ranking"], f["c"]["n_2023"], f["c"]["prop_2023"]) == (None, None, 120, 0.5)
    # (0, 0): el cero se informa
    assert (f["d"]["n_ranking"], f["d"]["prop_ranking"], f["d"]["n_2023"], f["d"]["prop_2023"]) == (0, 0.0, 0, 0.0)
    assert n_celdas == 2                                                                  # las celdas 4 y 6; las dependientes no cuentan
    assert df["nivel"].to_list() == ["a", "b", "c", "d"] and df.columns == _dos_ventanas().columns


def test_suprimir_no_hace_lo_mismo_con_los_dos_conteos_en_un_solo_bloque():
    """Lo que evita el paso por bloques: con un único bloque la fila (40, 4) pierde las cifras de las dos ventanas."""
    unico = [(["n_ranking", "n_2023"], ["prop_ranking", "prop_2023"])]
    df, n_celdas = supresion.suprimir_bloques(_dos_ventanas(), unico)
    a = df.filter(pl.col("nivel") == "a").row(0, named=True)
    assert a["n_2023"] is None and a["prop_2023"] is None and a["prop_ranking"] is None       # la proporción de 2019-2022 se pierde por un conteo ajeno
    assert a["n_ranking"] == 40 and n_celdas == 2


def test_el_resultado_no_depende_del_orden_de_los_bloques():
    df = _dos_ventanas()
    base, n0 = supresion.suprimir_bloques(df, BLOQUES_DOS_VENTANAS)
    for orden in itertools.permutations(BLOQUES_DOS_VENTANAS):
        otro, n1 = supresion.suprimir_bloques(df, list(orden))
        assert otro.equals(base) and n1 == n0


def test_una_dependiente_de_un_bloque_puede_ser_el_conteo_de_otro_sin_depender_del_orden():
    # bloque A: n1 protege a n2 y f1; bloque B: n2 protege a f2. Con (5, 7) las dos celdas son pequeñas.
    a = (["n1"], ["n2", "f1"])
    b = (["n2"], ["f2"])
    df = pl.DataFrame({"n1": [5, 50, 5], "n2": [7, 5, 50], "f1": [0.1, 0.2, 0.3], "f2": [0.4, 0.5, 0.6]})
    r1, c1 = supresion.suprimir_bloques(df, [a, b])
    r2, c2 = supresion.suprimir_bloques(df, [b, a])
    assert r1.equals(r2) and c1 == c2 == 4                                          # cuatro celdas pequeñas: (5, 7), 5 y 5
    f = r1.to_dicts()
    assert f[0] == {"n1": None, "n2": None, "f1": None, "f2": None}              # un esquema secuencial dejaría f2 visible cuando A va primero (n2 ya nulo)
    assert f[1] == {"n1": 50, "n2": None, "f1": 0.2, "f2": None}                  # n2 = 5 es pequeña por sí misma
    assert f[2] == {"n1": None, "n2": None, "f1": None, "f2": 0.6}                # n1 = 5 anula n2 (dependiente de A) aunque n2 = 50 no sea pequeña


def test_con_un_solo_bloque_coincide_con_suprimir():
    df = pl.DataFrame({"n": [3, 40, 9, 10, 0, None], "k": [1, 20, 5, 4, 0, 2], "p": [0.1, 0.5, 0.2, 0.3, 0.0, 0.4], "q": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    esperado, n_esperado = supresion.suprimir(df, ["n", "k"], dependientes=["p", "q"])
    obtenido, n_obtenido = supresion.suprimir_bloques(df, [(["n", "k"], ["p", "q"])])
    assert obtenido.equals(esperado) and n_obtenido == n_esperado
    for minimo in (5, 10, 20):
        a, na = supresion.suprimir(df, ["n", "k"], minimo=minimo, dependientes=["p"])
        b, nb = supresion.suprimir_bloques(df, [(["n", "k"], ["p"])], minimo=minimo)
        assert a.equals(b) and na == nb


def test_el_minimo_es_explicito_y_por_defecto_vale_diez():
    df = pl.DataFrame({"n": [4, 7, 10, 9], "p": [0.1, 0.2, 0.3, 0.4]})
    por_defecto, _ = supresion.suprimir_bloques(df, [(["n"], ["p"])])
    assert por_defecto["n"].to_list() == [None, None, 10, None]                                   # 9 se suprime, 10 no
    con_cinco, n5 = supresion.suprimir_bloques(df, [(["n"], ["p"])], minimo=5)
    assert con_cinco["n"].to_list() == [None, 7, 10, 9] and con_cinco["p"].to_list() == [None, 0.2, 0.3, 0.4] and n5 == 1
    con_diez, n10 = supresion.suprimir_bloques(df, [(["n"], ["p"])], minimo=10)
    assert con_diez["n"].to_list() == [None, None, 10, None] and n10 == 3


def test_columnas_ausentes_se_ignoran_y_sin_celdas_pequenas_la_tabla_no_cambia():
    df = pl.DataFrame({"n": [50, 60], "p": [0.5, 0.6]})
    r, n = supresion.suprimir_bloques(df, [(["n", "no_existe"], ["p", "tampoco"]), (["otra"], ["p"])])
    assert r.equals(df) and n == 0
    r, n = supresion.suprimir_bloques(df, [])
    assert r.equals(df) and n == 0
    df2 = pl.DataFrame({"n": [3, 60], "p": [0.5, 0.6]})
    r2, n2 = supresion.suprimir_bloques(df2, [(["n", "no_existe"], ["p", "tampoco"])])
    assert r2["n"].to_list() == [None, 60] and r2["p"].to_list() == [None, 0.6] and n2 == 1


def test_no_modifica_la_tabla_original_ni_los_tipos_y_respeta_los_nulos():
    df = pl.DataFrame({"n": [3, None, 80], "p": [0.5, 0.7, 0.6], "m": [1, 2, 3]}, schema={"n": pl.Int64, "p": pl.Float64, "m": pl.Int32})
    copia = df.clone()
    r, n = supresion.suprimir_bloques(df, [(["n"], ["p", "m"])])
    assert df.equals(copia)                                                                         # inmutable
    assert r.schema == df.schema                                                                    # los tipos se conservan
    assert r["n"].to_list() == [None, None, 80] and r["p"].to_list() == [None, 0.7, 0.6] and r["m"].to_list() == [None, 2, 3] and n == 1


def test_una_columna_de_conteo_en_dos_bloques_cuenta_su_celda_una_sola_vez():
    df = pl.DataFrame({"n": [4, 50], "a": [1.0, 2.0], "b": [3.0, 4.0]})
    r, n = supresion.suprimir_bloques(df, [(["n"], ["a"]), (["n"], ["b"])])
    assert n == 1 and r["a"].to_list() == [None, 2.0] and r["b"].to_list() == [None, 4.0] and r["n"].to_list() == [None, 50]


# ==================================================================================== panel.escribir_por_bloques
@pytest.fixture()
def ctx_bloques(tmp_path, fuente_sintetica):
    return Contexto.crear("extracto", fuente=fuente_sintetica, dir_salida_base=tmp_path)


def _tabla_doble() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "nivel": ["a", "b", "c", "d"], "n": [40, 40, 6, 0], "prop": [0.4, 0.4, 0.06, 0.0], "n_2023": [4, 120, 120, 0], "prop_2023": [0.04, 0.5, 0.5, 0.0],
        }
    )


BLOQUES = [(["n"], ["prop"]), (["n_2023"], ["prop_2023"])]


def _leer(ctx, nombre):
    return {r["nivel"]: r for r in ctx.leer_tabla(nombre, columnas_texto=["nivel"]).iter_rows(named=True)}


def test_escribir_por_bloques_conserva_las_cifras_de_2019_2022_y_anula_solo_las_de_2023_en_la_fila_40_4(ctx_bloques):
    ruta = panel.escribir_por_bloques(ctx_bloques, "t99_dos_ventanas.csv", _tabla_doble(), BLOQUES, tarea="T99", descripcion="tabla de juguete con dos ventanas")
    assert ruta == ctx_bloques.dir_tablas / "t99_dos_ventanas.csv" and ruta.is_file()
    f = _leer(ctx_bloques, "t99_dos_ventanas.csv")
    assert (f["a"]["n"], f["a"]["prop"], f["a"]["n_2023"], f["a"]["prop_2023"]) == (40, 0.4, None, None)         # la proporción de 2019-2022 queda y la de 2023 es nula
    assert (f["b"]["n"], f["b"]["prop"], f["b"]["n_2023"], f["b"]["prop_2023"]) == (40, 0.4, 120, 0.5)
    assert (f["c"]["n"], f["c"]["prop"], f["c"]["n_2023"], f["c"]["prop_2023"]) == (None, None, 120, 0.5)
    assert (f["d"]["n"], f["d"]["prop"], f["d"]["n_2023"], f["d"]["prop_2023"]) == (0, 0.0, 0, 0.0)


def test_el_indice_registra_todos_los_conteos_y_suma_las_celdas_anuladas_en_el_paso_previo(ctx_bloques):
    panel.escribir_por_bloques(ctx_bloques, "t99_dos_ventanas.csv", _tabla_doble(), BLOQUES, tarea="T99", descripcion="juguete")
    ind = {r["nombre"]: r for r in ctx_bloques.indice_tablas().iter_rows(named=True)}["t99_dos_ventanas.csv"]
    assert ind["columnas_conteo"] == "n;n_2023" and ind["tarea"] == "T99" and ind["filas"] == 4 and ind["fuente"] == "extracto"
    assert ind["celdas_suprimidas"] == 2                                                                           # las celdas de 4 y de 6 casos, anuladas antes de llamar a escribir_tabla
    assert len(ind["sha256"]) == 64


def test_la_red_de_seguridad_sin_el_paso_por_bloques_anula_las_cifras_de_las_dos_ventanas(ctx_bloques):
    ctx_bloques.escribir_tabla("t99_sin_bloques.csv", _tabla_doble(), tarea="T99", descripcion="sin bloques", cols_n=["n", "n_2023"], dependientes=["prop", "prop_2023"])
    f = _leer(ctx_bloques, "t99_sin_bloques.csv")
    assert (f["a"]["n"], f["a"]["prop"], f["a"]["n_2023"], f["a"]["prop_2023"]) == (40, None, None, None)           # lo que evita el paso por bloques: se pierde la proporción de 2019-2022
    assert (f["c"]["n"], f["c"]["prop"], f["c"]["n_2023"], f["c"]["prop_2023"]) == (None, None, 120, None)


def test_el_escaner_da_cero_violaciones_y_detecta_una_celda_pequena_de_2023_introducida_a_mano(ctx_bloques):
    panel.escribir_por_bloques(ctx_bloques, "t99_dos_ventanas.csv", _tabla_doble(), BLOQUES, tarea="T99", descripcion="juguete")
    esc = privacidad.escanear_salidas(ctx_bloques.dir_salida, ctx_bloques.indice_tablas(), ctx_bloques.celda_minima)
    assert privacidad.total_violaciones(esc) == 0
    ruta = ctx_bloques.dir_tablas / "t99_dos_ventanas.csv"
    editado = ruta.read_text(encoding="utf8").replace("c,,,120,0.5", "c,,,7,0.5")                                       # n_2023 = 7 introducido a mano en la fila c
    assert editado != ruta.read_text(encoding="utf8")
    ruta.write_text(editado, encoding="utf8")
    esc = privacidad.escanear_salidas(ctx_bloques.dir_salida, ctx_bloques.indice_tablas(), ctx_bloques.celda_minima)
    malas = esc.filter(pl.col("n_violaciones") > 0)
    assert malas.height == 1 and malas["chequeo"].item() == "celda_pequena" and privacidad.total_violaciones(esc) == 1          # el escáner cubre n_2023 porque está registrada


def test_una_columna_ausente_en_un_bloque_es_un_error_que_la_nombra(ctx_bloques):
    with pytest.raises(ValueError, match="prop_2024"):
        panel.escribir_por_bloques(ctx_bloques, "t99_mal.csv", _tabla_doble(), [(["n"], ["prop"]), (["n_2023"], ["prop_2024"])], tarea="T99", descripcion="x")
    with pytest.raises(ValueError, match="n_dos"):
        panel.escribir_por_bloques(ctx_bloques, "t99_mal.csv", _tabla_doble(), [(["n_dos"], ["prop"])], tarea="T99", descripcion="x")
    assert not (ctx_bloques.dir_tablas / "t99_mal.csv").exists() and ctx_bloques.indice_tablas().height == 0


def test_el_umbral_del_contexto_manda_con_celda_minima_5_una_celda_de_7_no_se_suprime_y_una_de_4_si(tmp_path, fuente_sintetica):
    df = pl.DataFrame({"nivel": ["a", "b"], "n": [50, 50], "prop": [0.5, 0.5], "n_2023": [7, 4], "prop_2023": [0.07, 0.04]})
    con5 = Contexto.crear("extracto", fuente=fuente_sintetica, dir_salida_base=tmp_path / "c5", umbrales=umbrales_temporales(tmp_path, celda_minima=5))
    panel.escribir_por_bloques(con5, "t99.csv", df, BLOQUES, tarea="T99", descripcion="x")
    f = _leer(con5, "t99.csv")
    assert (f["a"]["n_2023"], f["a"]["prop_2023"]) == (7, 0.07) and (f["b"]["n_2023"], f["b"]["prop_2023"]) == (None, None)
    assert con5.indice_tablas()["celdas_suprimidas"].item() == 1
    con10 = Contexto.crear("extracto", fuente=fuente_sintetica, dir_salida_base=tmp_path / "c10")
    panel.escribir_por_bloques(con10, "t99.csv", df, BLOQUES, tarea="T99", descripcion="x")
    g = _leer(con10, "t99.csv")
    assert (g["a"]["n_2023"], g["b"]["n_2023"]) == (None, None) and g["a"]["prop"] == 0.5 and con10.indice_tablas()["celdas_suprimidas"].item() == 2


def test_las_dependientes_que_son_el_conteo_de_otro_bloque_no_rompen_la_escritura(ctx_bloques):
    df = pl.DataFrame({"nivel": ["a", "b"], "n": [5, 50], "n2": [7, 5], "f1": [0.1, 0.2], "f2": [0.3, 0.4]})
    panel.escribir_por_bloques(ctx_bloques, "t99_cruzada.csv", df, [(["n"], ["n2", "f1"]), (["n2"], ["f2"])], tarea="T99", descripcion="x")
    f = _leer(ctx_bloques, "t99_cruzada.csv")
    assert (f["a"]["n"], f["a"]["n2"], f["a"]["f1"], f["a"]["f2"]) == (None, None, None, None) and (f["b"]["n"], f["b"]["n2"], f["b"]["f1"], f["b"]["f2"]) == (50, None, 0.2, None)
    ind = ctx_bloques.indice_tablas().row(0, named=True)
    assert ind["columnas_conteo"] == "n;n2" and ind["celdas_suprimidas"] == 3


# ==================================================================================== t23_panel_niveles.csv en formato largo
def test_el_nivel_de_40_casos_en_2019_2022_y_4_en_2023_conserva_la_proporcion_de_2019_2022_y_anula_la_de_2023(ctx_bloques):
    p = panel_artificial()
    panel.escribir_tablas(ctx_bloques, p)
    t = ctx_bloques.leer_tabla("t23_panel_niveles.csv", columnas_texto=["variable", "nivel", "tipo_nivel", "ventana"])
    fila = t.filter((pl.col("variable") == "TIPO_INGRESO") & (pl.col("nivel") == "OBSTETRICA"))
    f = {r["ventana"]: r for r in fila.iter_rows(named=True)}
    assert f[panel.V_RANK]["n"] == 40 and f[panel.V_RANK]["prop"] == pytest.approx(40 / 960, abs=1e-6)
    assert f[panel.V_CONF]["n"] is None and f[panel.V_CONF]["prop"] is None                                         # 4 casos: se anula la fila de 2023, no la de 2019-2022
    # 2019-2022 no pierde nada; en 2023 se oculta además el nivel más chico (supresión complementaria, cambio 1.6): con el total de la
    # cohorte de 2023 y los demás niveles a la vista, los 4 casos se despejarían restando
    otros = t.filter((pl.col("variable") == "TIPO_INGRESO") & (pl.col("nivel") != "OBSTETRICA"))
    rank, conf = otros.filter(pl.col("ventana") == panel.V_RANK), otros.filter(pl.col("ventana") == panel.V_CONF)
    assert rank["n"].null_count() == 0 and rank["prop"].null_count() == 0
    assert conf["n"].null_count() == 1 and conf["prop"].null_count() == 1
    assert conf.filter(pl.col("n").is_null())["nivel"][0] != "OBSTETRICA"
    ind = {r["nombre"]: r for r in ctx_bloques.indice_tablas().iter_rows(named=True)}["t23_panel_niveles.csv"]
    assert ind["columnas_conteo"] == "n" and ind["celdas_suprimidas"] >= 1
    assert privacidad.total_violaciones(privacidad.escanear_salidas(ctx_bloques.dir_salida, ctx_bloques.indice_tablas(), ctx_bloques.celda_minima)) == 0
