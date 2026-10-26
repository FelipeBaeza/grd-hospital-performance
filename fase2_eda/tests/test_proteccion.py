"""Supresión complementaria (cambio 1.6): ninguna suma conocida queda con una sola celda oculta. Canarios hechos a mano."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from eda import proteccion as pr
from eda import proteccion_tablas as pt
from eda.contexto import Contexto
from eda.proteccion import Fila, Grupo, Proteccion, donde


def _nulos(t: pl.DataFrame, col: str) -> list[int]:
    return [i for i, v in enumerate(t[col].to_list()) if v is None]


def test_la_tabla_de_rut_ya_no_deja_despejar_la_columna_pequena():
    """El caso que motivó el cambio: 22 valores en 2019 (publicados en t24_reglas) = 15 + 7."""
    df = pl.DataFrame({"columna": ["FECHAPROCEDIMIENTO1", "USOSPABELLON"], "anio": [2019, 2019], "n": [7, 15]})
    res = pr.complementar(df, ["n"], pt.REGISTRO["t24_rut_por_columna.csv"], 10)
    assert res.tabla["n"].to_list() == [None, None] and (res.primarias, res.complementarias) == (1, 1)


def test_un_total_de_grupo_obliga_a_ocultar_la_celda_menor_y_no_otra():
    df = pl.DataFrame({"anio": [2020] * 4, "cat": ["a", "b", "c", "d"], "n": [500, 40, 3, 12], "prop": [0.9, 0.07, 0.005, 0.02]})
    res = pr.complementar(df, ["n"], Proteccion(grupos=(Grupo(("anio",)),)), 10, dependientes=["prop"])
    assert res.tabla["n"].to_list() == [500, 40, None, None] and res.tabla["prop"].to_list() == [0.9, 0.07, None, None]


def test_dos_celdas_pequenas_cuya_suma_es_diez_o_mas_no_necesitan_complemento():
    df = pl.DataFrame({"anio": [2020] * 3, "n": [500, 6, 7]})
    res = pr.complementar(df, ["n"], Proteccion(grupos=(Grupo(("anio",)),)), 10)
    assert res.complementarias == 0 and _nulos(res.tabla, "n") == [1, 2]


def test_dos_celdas_pequenas_cuya_suma_es_pequena_necesitan_complemento():
    """Con el total conocido, 4 y 3 ocultos dejan ver que las dos categorías juntas suman 7: se oculta una celda más."""
    df = pl.DataFrame({"anio": [2020] * 4, "n": [500, 4, 3, 800]})
    res = pr.complementar(df, ["n"], Proteccion(grupos=(Grupo(("anio",)),)), 10)
    assert res.complementarias == 1 and _nulos(res.tabla, "n") == [0, 1, 2]


def test_el_complemento_se_propaga_a_las_otras_identidades_hasta_cerrar():
    """Tabla 2 x 2 con totales conocidos por año y por categoría: una celda pequeña obliga a ocultar el rectángulo completo."""
    df = pl.DataFrame({"anio": [2019, 2019, 2020, 2020], "cat": ["a", "b", "a", "b"], "n": [5, 100, 200, 300]})
    res = pr.complementar(df, ["n"], Proteccion(grupos=(Grupo(("anio",)), Grupo(("cat",)))), 10)
    assert res.tabla["n"].null_count() == 4 and pr.una_sola_oculta(df, res.tabla, ["n"], Proteccion(grupos=(Grupo(("anio",)), Grupo(("cat",)))), 10) == 0


def test_periodo_agregado_suma_sus_anios():
    df = pl.DataFrame({"periodo": ["2019", "2020", "2021", "2019-2021"], "n": [4, 30, 50, 84]})
    prot = Proteccion(grupos=(Grupo((), donde=donde(periodo=("2019", "2020", "2021", "2019-2021"))),))
    res = pr.complementar(df, ["n"], prot, 10)
    assert res.tabla["n"].to_list() == [None, None, 50, 84]


def test_identidad_de_fila_y_resto_implicito():
    """n_filas = n_nulos + n_informados; y el resto implícito informados - modal (valores no modales) también se protege."""
    df = pl.DataFrame({"n_filas": [1000, 1000], "n_nulos": [3, 600], "n_informados": [997, 400], "n_modal": [500, 395]})
    prot = Proteccion(filas=(Fila(("n_filas", "n_nulos", "n_informados")), Fila(("n_informados", "n_modal"), resto_de="n_informados")))
    res = pr.complementar(df, ["n_nulos", "n_informados", "n_modal"], prot, 10)
    t = res.tabla
    assert t["n_nulos"][0] is None and t["n_informados"][0] is None          # 1000 - 997 dejaría ver los 3 nulos
    assert t["n_modal"][1] is None and t["n_informados"][1] == 400            # 400 - 395 dejaría ver 5 valores no modales


def test_grupo_con_total_en_columna_y_lista_truncada():
    """Lista de los valores más frecuentes: el resto (los no listados) es una celda implícita."""
    df = pl.DataFrame({"col": ["X"] * 3, "valor": ["a", "b", "c"], "n": [900, 60, 35], "_total": [1000] * 3})
    res = pr.complementar(df, ["n"], Proteccion(grupos=(Grupo(("col",), total="_total"),)), 10)
    assert res.tabla["n"].to_list() == [900, 60, None]                         # 1000 - 995 = 5 valores fuera de la lista
    df2 = df.with_columns(pl.Series("n", [900, 60, 20]))
    assert pr.complementar(df2, ["n"], Proteccion(grupos=(Grupo(("col",), total="_total"),)), 10).complementarias == 0   # resto de 20


def test_las_celdas_nulas_no_entran_en_la_identidad():
    """Un año bloqueado (nulo) no está en la suma: si otro año es el único oculto, se despeja igual y hay que complementar."""
    df = pl.DataFrame({"ambito": ["2019", "2020", "2024", "total"], "n": [5, 300, None, 305]})
    prot = Proteccion(grupos=(Grupo((), donde=donde(ambito=("2019", "2020", "2024", "total"))),))
    res = pr.complementar(df, ["n"], prot, 10)
    assert res.tabla["n"].to_list() == [None, None, None, 305]


def test_preferir_oculta_la_celda_propia_aunque_valga_cero():
    df = pl.DataFrame({"n_fecha": [333365], "n_servicio": [333372], "n_solo_fecha": [0], "n_solo_servicio": [7], "diferencia": [7]})
    res = pr.complementar(df, ["n_fecha", "n_servicio", "n_solo_fecha", "n_solo_servicio"], pt.REGISTRO["t22_pares_traslado.csv"], 10)
    r = res.tabla.row(0, named=True)
    assert r["n_solo_fecha"] is None and r["n_solo_servicio"] is None and r["diferencia"] is None
    # las dos ocultas suman 7: se oculta además una de las dos columnas grandes, para que la diferencia no deje acotar el conteo
    assert (r["n_fecha"] is None) != (r["n_servicio"] is None)


def test_mortalidad_del_resto_es_dependiente_en_los_estratos():
    assert "mortalidad_resto" in pt.REGISTRO["t23_resultado_estratos.csv"].dependientes


def test_familia_hereda_las_celdas_ocultas_de_la_primera_tabla(tmp_path):
    from sintetico import crear_fuente_sintetica

    fuente = crear_fuente_sintetica(tmp_path / "d", n=200, semilla=3)
    ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=tmp_path / "s")
    a = pl.DataFrame({"anio": [2019] * 3, "cod_hospital": ["1", "2", "3"], "n_D": [5, 40, 900], "esperados_D": [1.0, 2.0, 30.0]})
    ctx.escribir_tabla("t23_hospitales_volumen.csv", a, tarea="T09", descripcion="d", cols_n=["n_D"])
    assert ctx.leer_tabla("t23_hospitales_volumen.csv")["n_D"].to_list() == [None, None, 900]
    b = pl.DataFrame({"periodo": ["2019"] * 3, "cod_hospital": ["1", "2", "3"], "n": [5, 40, 900], "k": [0, 0, 50]})
    ctx.escribir_tabla("t23_embudo.csv", b, tarea="T09", descripcion="d", cols_n=["n", "k"])
    assert ctx.leer_tabla("t23_embudo.csv")["n"].to_list() == [None, None, 900]
    assert ctx.indice_tablas().filter(pl.col("nombre") == "t23_embudo.csv")["celdas_suprimidas"][0] == 2


def test_las_columnas_auxiliares_no_se_publican(tmp_path):
    from sintetico import crear_fuente_sintetica

    fuente = crear_fuente_sintetica(tmp_path / "d", n=200, semilla=3)
    ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=tmp_path / "s")
    t = pl.DataFrame({"columna": ["X", "X"], "anio": [2019, 2019], "valor": ["a", "b"], "n": [995, 5], "prop": [0.995, 0.005],
                      "_clave": ["a", "b"], "_rol": ["", ""], "_total": [1000, 1000]})
    ctx.escribir_tabla("t22_top_valores.csv", t, tarea="T04", descripcion="d", cols_n=["n"], dependientes=["prop"], auxiliares=["_clave", "_rol", "_total"])
    pub = ctx.leer_tabla("t22_top_valores.csv")
    assert pub.columns == ["columna", "anio", "valor", "n", "prop"] and pub["n"].to_list() == [None, None]


@pytest.mark.parametrize("nombre", sorted(pt.REGISTRO))
def test_cada_identidad_declarada_usa_columnas_validas(nombre):
    prot = pt.REGISTRO[nombre]
    for g in prot.grupos:
        assert isinstance(g.por, tuple) and all(isinstance(c, str) for c in g.por)
    for f in prot.filas:
        assert len(f.columnas) >= 2 and (f.resto_de is None or f.resto_de in f.columnas)
        assert set(f.preferir) <= set(f.columnas)


def test_verificacion_independiente_detecta_una_fuga_plantada():
    df = pl.DataFrame({"anio": [2020] * 3, "n": [500, 40, 3]})
    publicada = df.with_columns(pl.Series("n", [500, 40, None]))                  # solo supresión primaria: se despeja 3
    assert pr.una_sola_oculta(df, publicada, ["n"], Proteccion(grupos=(Grupo(("anio",)),)), 10) == 1
    np.testing.assert_equal(pr.complementar(df, ["n"], Proteccion(grupos=(Grupo(("anio",)),)), 10).tabla["n"].null_count(), 2)


def test_una_derivada_con_origen_declarado_solo_se_anula_si_su_conteo_se_oculta():
    df = pl.DataFrame({"n_filas": [1000], "n_nulos": [400], "n_centinela": [4], "n_informados": [600], "pct_nulos": [40.0], "pct_centinela": [0.4]})
    prot = Proteccion(dependencias=(("pct_nulos", ("n_nulos",)), ("pct_centinela", ("n_centinela",))))
    t = pr.complementar(df, ["n_nulos", "n_centinela", "n_informados"], prot, 10, dependientes=["pct_nulos", "pct_centinela"]).tabla
    assert t["pct_nulos"][0] == 40.0 and t["pct_centinela"][0] is None and t["n_centinela"][0] is None


def test_los_dos_periodos_agregados_no_dejan_despejar_2023():
    """2019-2023 menos 2019-2022 es 2023: con los dos períodos a la vista, un 2023 oculto se despejaría."""
    df = pl.DataFrame({"periodo": ["2019", "2020", "2021", "2022", "2023", "2019-2022", "2019-2023"], "cat": ["x"] * 7,
                       "n": [5, 7, 300, 400, 4, 712, 716]})
    prot = Proteccion(grupos=pt._agregados("periodo", ("cat",)))
    t = pr.complementar(df, ["n"], prot, 10).tabla
    vis = dict(zip(t["periodo"].to_list(), t["n"].to_list()))
    assert vis["2023"] is None and (vis["2019-2022"] is None or vis["2019-2023"] is None)


def test_total_secreto_obliga_a_ocultar_una_celda_en_cada_grupo():
    df = pl.DataFrame({"periodo": ["2023"] * 5, "variable": ["a", "a", "b", "b", "b"], "n_fallecidos": [24235, 4, 20000, 4000, 239],
                       "_T_oculto": [True] * 5})
    prot = Proteccion(grupos=(Grupo(("periodo", "variable"), total_oculto="_T_oculto", columnas=("n_fallecidos",)),))
    t = pr.complementar(df, ["n_fallecidos"], prot, 10).tabla
    assert t["n_fallecidos"].to_list() == [24235, None, 20000, 4000, None]       # cada grupo con al menos una oculta: el total no se deduce


def test_la_resta_de_dos_variantes_vecinas_se_protege():
    from eda import cohorte

    var = pl.DataFrame({"periodo": ["2023"] * 4, "variante": ["A", "B", "C", "D"], "n": [948573, 791069, 623137, 568737],
                        "n_fallecidos": [24624, 24239, 24235, 24235], "n_no_evento": [1, 1, 1, 1], "n_censurados": [10, 10, 10, 0], "n_sin_dato": [0, 0, 0, 0],
                        "prevalencia": [0.026, 0.031, 0.039, 0.043]})
    aux = cohorte.con_variante_siguiente(var)
    t = pr.complementar(aux, list(cohorte.CONTEOS_VARIANTE), pt.REGISTRO["t23_cascada_variantes.csv"], 10, dependientes=["prevalencia"]).tabla
    k = dict(zip(t["variante"].to_list(), t["n_fallecidos"].to_list()))
    assert k["B"] is None and k["A"] == 24624                                                # B - C = 4 fallecidos en actividad ambulatoria


def test_la_celda_de_la_variante_no_se_despeja_con_la_suma_de_su_fila():
    """El n de cada variante es la suma de sus cuatro desenlaces: ocultar solo los fallecidos de B no basta (se despejaban restando)."""
    from eda import cohorte

    falle, cens = [24624, 24239, 24235, 24235], [54900, 54797, 54790, 0]
    no_ev = [868000, 712033, 544112, 544502]
    n = [f + c + e for f, c, e in zip(falle, cens, no_ev)]
    var = pl.DataFrame({"periodo": ["2023"] * 4, "variante": ["A", "B", "C", "D"], "n": n, "n_fallecidos": falle, "n_no_evento": no_ev,
                        "n_censurados": cens, "n_sin_dato": [0, 0, 0, 0], "prevalencia": [0.026, 0.031, 0.039, 0.043]})
    aux = cohorte.con_variante_siguiente(var)
    t = pr.complementar(aux, list(cohorte.CONTEOS_VARIANTE), pt.REGISTRO["t23_cascada_variantes.csv"], 10, dependientes=["prevalencia"]).tabla
    assert t.filter(pl.col("variante") == "B")["n_fallecidos"].item() is None
    for fila in t.iter_rows(named=True):
        ocultas = sum(fila[c] is None for c in cohorte.CONTEOS_VARIANTE)
        assert ocultas != 1, fila


def test_una_exclusion_pequena_de_la_cascada_no_se_despeja_con_la_cadena():
    """TOTAL - EX01 - EX02 - EX03 = A: con EX01 de 1 a 9 oculto y lo demás a la vista, la resta lo daba (pasaba en los extractos)."""
    pasos = ["TOTAL", "EX01", "EX02", "EX03", "A", "EX04", "B"]
    df = pl.DataFrame({"periodo": ["2020"] * 7, "paso": pasos, "n": [49999, 3, 1582, 5283, 43131, 7383, 35748], "prop_del_total": [1.0] * 7})
    t = pr.complementar(df, ["n"], pt.REGISTRO["t23_cascada.csv"], 10).tabla
    v = dict(zip(t["paso"].to_list(), t["n"].to_list()))
    assert v["EX01"] is None and sum(v[p] is None for p in ("TOTAL", "EX02", "EX03", "A")) >= 1
    assert pr.una_sola_oculta(df, t, ["n"], pt.REGISTRO["t23_cascada.csv"], 10) == 0


def test_el_resto_implicito_de_los_enlaces_de_traslado_se_protege():
    """n_total = inválidos + elegibles + otros: con «otros» de 1 a 9, se oculta una celda (los elegibles) y su tasa."""
    df = pl.DataFrame({"periodo": ["2020"], "direccion": ["entrante"], "ventana_dias": [30], "n_total": [59787], "n_id_invalido": [510],
                       "n_elegibles": [59270], "n_enlazados": [400], "tasa_enlace": [0.0067], "ic_bajo": [0.006], "ic_alto": [0.0074]})
    t = pr.complementar(df, ["n_total", "n_id_invalido", "n_elegibles", "n_enlazados"], pt.REGISTRO["t23_enlaces_traslado.csv"], 10).tabla
    r = t.row(0, named=True)
    assert r["n_elegibles"] is None and r["tasa_enlace"] is None and r["n_total"] == 59787


def test_las_metricas_de_reglas_salen_de_la_tabla_publicada(tmp_path):
    """Una celda oculta como complemento no vuelve a salir por el ledger: las métricas se leen de la tabla publicada."""
    from eda import calidad
    from sintetico import crear_fuente_sintetica

    ctx = Contexto.crear("extracto", fuente=crear_fuente_sintetica(tmp_path / "d", n=700, semilla=21), dir_salida_base=tmp_path / "s")
    calidad.ejecutar(ctx)
    pub = ctx.leer_tabla("t24_reglas.csv", columnas_texto=["regla_id", "ambito"])
    m = {r["clave"]: r["valor"] for r in ctx.metricas().iter_rows(named=True)}
    for rid, clave in (("Q05", "calidad.alta_antes_ingreso"), ("Q08", "calidad.edad_mayor_110"), ("Q17", "calidad.valores_rut")):
        for f in pub.filter(pl.col("regla_id") == rid).iter_rows(named=True):
            k = f"{clave}.{f['ambito']}"
            if k in m:
                assert (m[k] is None) == (f["n_violaciones"] is None), k
