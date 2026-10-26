"""Análisis posteriores pedidos por el profesor guía (cambio 1.6): R1 a R5 sobre datos sintéticos y reglas con tablas hechas a mano."""

from __future__ import annotations

from collections import Counter

import polars as pl
import pytest

from eda import catalogos, cohorte, config, elixhauser, privacidad
from eda import revision_profesor as rp
from eda.contexto import Contexto
from sintetico import HOSP_PEDIATRICO, HOSPITALES, crear_fuente_sintetica

DEIS_PRUEBA = {h: "General de alta complejidad" for h in HOSPITALES[:3]} | {h: "General de mediana complejidad" for h in HOSPITALES[3:6]} | {HOSP_PEDIATRICO: "Institutos y pediátricos"}


@pytest.fixture(scope="module")
def corrida(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("revision")
    fuente = crear_fuente_sintetica(raiz / "datos", n=3000, semilla=41)
    ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=raiz / "salidas")
    out = rp.ejecutar(ctx, deis=DEIS_PRUEBA)
    ctx.escribir_registros()
    return ctx, out


def test_escribe_las_cinco_tablas_y_las_tres_figuras(corrida):
    ctx, out = corrida
    for t in ("r1_resultado_anio_diagnostico.csv", "r1_resumen_pandemia.csv", "r2_faltantes_variables.csv", "r3_vocabulario_diagnostico.csv",
              "r4_traslados_tipo_hospital.csv", "r5_volumen_cohortes.csv"):
        assert (ctx.dir_tablas / t).is_file(), t
    idx = ctx.figuras.indice()
    assert set(idx["id_figura"].to_list()) == {"R01", "R02", "R03"} and set(idx["tarea"].to_list()) == {"R"}


def test_ninguna_salida_publica_celdas_de_1_a_9_ni_fechas_completas(corrida):
    ctx, _ = corrida
    esc = privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)
    assert privacidad.total_violaciones(esc) == 0, esc.filter(pl.col("n_violaciones") > 0)


def test_el_desenlace_de_2024_no_se_lee(corrida):
    ctx, _ = corrida
    reg = ctx.cortafuegos.registro()
    a2024 = reg.filter(pl.col("anio") == 2024)
    assert a2024.height > 0 and set(a2024["proposito"].to_list()) == {"estructura"} and (reg["decision"] != "BLOQUEADO").all()
    assert set(reg["tarea"].to_list()) == {"R"}


def test_la_cohorte_d_y_e_de_r1_cuadra_con_la_cascada_de_la_tarea_7(corrida):
    ctx, out = corrida
    cat = catalogos.cargar_catalogos()
    e, e3 = elixhauser.cargar(), elixhauser.cargar(solo_3c=True)
    tot = out["r1"].filter(pl.col("nivel") == "total")
    for anio in ctx.umbrales.particion["anios_resultado"]:
        b = cohorte.leer_base(ctx, anio, cat, e, e3)
        fila = tot.filter(pl.col("anio") == anio).row(0, named=True)
        assert fila["n_d"] == int(b["D"].sum()) and fila["n_e"] == int(b["E"].sum())
        assert fila["n_fallecidos"] == int((b["D"] & (b["rol"] == "EVENTO")).sum())


def test_r2_cuenta_los_faltantes_desde_las_columnas_crudas(corrida):
    ctx, out = corrida
    df = ctx.lector.leer(2021, ["SEXO", "TIPO_PROCEDENCIA"], proposito="estructura", tarea="R")
    cat = catalogos.cargar_catalogos()
    b = rp.leer_anio(ctx, 2021, cat, elixhauser.cargar(), elixhauser.cargar(solo_3c=True))
    c = b["C"].to_numpy()
    sexo_falta = (~df["SEXO"].str.strip_chars().is_in(["MUJER", "HOMBRE"]).fill_null(False)).to_numpy()
    fila = out["r2"].filter((pl.col("anio") == 2021) & (pl.col("variable") == "SEXO")).row(0, named=True)
    assert fila["n"] == int(c.sum()) and fila["n_faltante"] == int(sexo_falta[c].sum())
    assert set(out["r2"]["anio"].to_list()) == {2019, 2020, 2021, 2022, 2023, 2024}


def test_r3_raros_y_nuevos_con_contadores_hechos_a_mano():
    u = config.cargar_umbrales()
    r = {"vocab": {2019: Counter({"A00": 80, "B01": 30}), 2020: Counter({"A00": 40}), 2021: Counter(), 2022: Counter({"B01": 20}),
                   2023: Counter({"A00": 50, "C02": 12, "B01": 15}), 2024: Counter({"D03": 11, "A00": 60})}}
    t = rp.tabla_r3(r, u)
    base = t.filter(pl.col("periodo") == "2019-2022").row(0, named=True)
    assert (base["n_codigos"], base["n_codigos_conservados"], base["n_codigos_raros"], base["egresos_codigos_raros"]) == (2, 1, 1, 50)
    a23 = t.filter(pl.col("periodo") == "2023").row(0, named=True)
    assert (a23["n_codigos_nuevos"], a23["egresos_codigos_nuevos"], a23["egresos_codigos_raros"]) == (1, 12, 15)
    a24 = t.filter(pl.col("periodo") == "2024").row(0, named=True)
    assert (a24["n_codigos_nuevos"], a24["egresos_codigos_nuevos"]) == (1, 11)


def _d_e(muertes_2020: int):
    filas_d, filas_e = [], []
    for anio, k in ((2019, 20), (2020, muertes_2020), (2021, 30), (2022, 20), (2023, 20)):
        filas_d += [{"anio": anio, "hospital": "1", "capitulo": "IX", "cat3": "I21", "evento": i < k} for i in range(500)]
        filas_e += [{"anio": anio, "hospital": "1", "capitulo": "IX", "cat3": "I21", "estancia": 5 + (anio == 2020)} for _ in range(400)]
    return {"D": pl.DataFrame(filas_d, schema_overrides={"anio": pl.Int32}), "E": pl.DataFrame(filas_e, schema_overrides={"anio": pl.Int32})}


def test_r1_la_razon_de_pandemia_compara_2020_2021_con_2019_2022_2023():
    u = config.cargar_umbrales()
    _, res, tope = rp.tabla_r1(_d_e(40), u, {})
    tot = res.filter(pl.col("nivel") == "total").row(0, named=True)
    assert tot["razon_mortalidad_pandemia"] == pytest.approx(((40 + 30) / 2 / 500) / (20 / 500)) and tot["cambio_mortalidad_relevante"] == "si"
    assert tot["razon_estancia_pandemia"] == pytest.approx(5.5 / 5.0) and tot["cambio_estancia_relevante"] == "no" and tope == 6


def test_r1_no_publica_la_razon_si_un_anio_tiene_menos_de_diez_muertes():
    """Con un año suprimido, la razón y las tasas de los demás años permitirían despejar el conteo suprimido."""
    u = config.cargar_umbrales()
    _, res, _ = rp.tabla_r1(_d_e(7), u, {})
    assert res.filter(pl.col("nivel") == "total")["razon_mortalidad_pandemia"][0] is None


def test_cargar_deis_junta_institutos_y_pediatricos(tmp_path):
    ruta = tmp_path / "deis.csv"
    pl.DataFrame({"cod_hospital": ["1", "2", "3", "4"], "tipo": ["general de alta complejidad", "general de mediana complejidad", "instituto monográfico", "pediátrico"]}).write_csv(ruta)
    m = rp.cargar_deis(ruta)
    assert m == {"1": "General de alta complejidad", "2": "General de mediana complejidad", "3": "Institutos y pediátricos", "4": "Institutos y pediátricos"}
    assert rp.cargar_deis(tmp_path / "no_existe.csv") == {}


def test_r4_reparte_los_egresos_de_la_cohorte_c_entre_los_grupos(corrida):
    ctx, out = corrida
    r4 = out["r4"]
    assert set(r4["grupo"].to_list()) <= set(rp.ORDEN_GRUPOS) | {"Sin clasificar"}
    c_total = sum(int(r["n_c"]) for r in r4.iter_rows(named=True))
    cascada = 0
    cat = catalogos.cargar_catalogos()
    e, e3 = elixhauser.cargar(), elixhauser.cargar(solo_3c=True)
    for anio in ctx.umbrales.particion["anios_resultado"]:
        cascada += int(cohorte.leer_base(ctx, anio, cat, e, e3)["C"].sum())
    assert c_total == cascada


def test_la_razon_de_pandemia_se_anula_si_la_tabla_publicada_oculta_un_anio():
    res = pl.DataFrame({"nivel": ["total", "capitulo"], "grupo": ["Todos", "IX"], "etiqueta": ["Todos", "Circulatorio"],
                        "razon_mortalidad_pandemia": [1.3, 1.2], "razon_estancia_pandemia": [1.1, 1.0],
                        "cambio_mortalidad_relevante": ["si", "no"], "cambio_estancia_relevante": ["no", "no"]})
    filas = [{"nivel": "total", "grupo": "Todos", "anio": a, "mortalidad": 0.05, "estancia_media_tope": 7.0} for a in range(2019, 2024)]
    filas += [{"nivel": "capitulo", "grupo": "IX", "anio": a, "mortalidad": None if a == 2021 else 0.08, "estancia_media_tope": 8.0} for a in range(2019, 2024)]
    out = rp.sin_razon_con_anios_ocultos(res, pl.DataFrame(filas))
    assert out["razon_mortalidad_pandemia"].to_list() == [1.3, None] and out["cambio_mortalidad_relevante"].to_list() == ["si", "no evaluable"]
    assert out["razon_estancia_pandemia"].to_list() == [1.1, 1.0]
