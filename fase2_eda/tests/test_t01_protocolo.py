"""Tarea 1: el protocolo (hipótesis, umbrales, plan de figuras) está completo y es coherente."""

from __future__ import annotations

import polars as pl
import pytest

from eda import config, protocolo


@pytest.fixture(scope="module")
def decisiones() -> pl.DataFrame:
    return config.cargar_decisiones_fase1()


@pytest.fixture(scope="module")
def hipotesis() -> pl.DataFrame:
    return config.cargar_hipotesis()


def test_las_17_decisiones_de_la_fase1_estan_registradas(decisiones):
    assert decisiones.height == 17
    assert sorted(int(x) for x in decisiones["n"].to_list()) == list(range(1, 18))


def test_cada_decision_tiene_hipotesis_o_marca_no_depende_de_datos(hipotesis, decisiones):
    assert protocolo.validar_hipotesis(hipotesis, decisiones) == []
    cob = protocolo.tabla_cobertura(hipotesis, decisiones)
    assert cob.height == 17
    assert cob["n_hipotesis"].min() >= 1
    # las decisiones que no dependen de datos están marcadas explícitamente
    no_dep = set(cob.filter(pl.col("depende_de_datos") == "no")["decision"].to_list())
    assert {4, 10, 12}.issubset(no_dep)


def test_cada_hipotesis_dependiente_de_datos_apunta_a_una_tarea_3_a_12(hipotesis):
    dep = hipotesis.filter(pl.col("depende_de_datos").is_in(["si", "parcial"]))
    assert dep.height > 0
    tareas = {int(t) for t in dep["tarea"].to_list()}
    assert tareas.issubset(set(range(3, 13)))


def test_hipotesis_sin_tarea_valida_se_detecta(hipotesis, decisiones):
    roto = hipotesis.with_columns(
        pl.when(pl.col("id") == "H07").then(pl.lit("99")).otherwise(pl.col("tarea")).alias("tarea")
    )
    problemas = protocolo.validar_hipotesis(roto, decisiones)
    assert any("H07" in p for p in problemas)


def test_decision_sin_hipotesis_se_detecta(hipotesis, decisiones):
    sin_14 = hipotesis.filter(pl.col("decision") != "14")
    problemas = protocolo.validar_hipotesis(sin_14, decisiones)
    assert any("decisión 14" in p for p in problemas)


def test_todos_los_umbrales_tienen_valor_y_justificacion():
    u = config.cargar_umbrales()
    assert config.validar_umbrales(u) == []
    assert len(u.nombres()) >= 30
    for n in u.nombres():
        assert u[n] is not None
        assert len(u.justificacion(n)) > 20


def test_umbral_sin_justificacion_se_detecta(tmp_path):
    p = tmp_path / "u.toml"
    p.write_text('[umbral.a]\nvalor = 1\njustificacion = ""\n[umbral.b]\njustificacion = "x"\n', encoding="utf8")
    problemas = config.validar_umbrales(config.cargar_umbrales(p))
    assert any("a: falta 'justificacion'" in x for x in problemas)
    assert any("b: falta 'valor'" in x for x in problemas)


def test_umbrales_de_la_fase1_coinciden_con_la_spec():
    u = config.cargar_umbrales()
    spec = config.cargar_spec_fase1()
    assert u["z_95"] == spec["indicador"]["z_95"]
    assert u["z_998"] == spec["indicador"]["z_998"]
    assert u["winsor"] == spec["indicador"]["winsor"]
    assert u["celda_minima"] == spec["indicador"]["celda_minima_publicable"]
    assert u["volumen_minimo_egresos"] == spec["indicador"]["volumen_minimo"]["egresos"]
    assert u["muertes_esperadas_minimas"] == spec["indicador"]["volumen_minimo"]["muertes_esperadas"]


def test_particion_respeta_el_cortafuegos_de_2024():
    u = config.cargar_umbrales()
    assert 2024 not in u.particion["anios_resultado"]
    assert 2024 not in u.particion["anios_ranking"]
    assert u.particion["anio_confirmacion"] == 2023
    assert u.particion["anio_expuesto"] == 2024
    assert set(u.cortafuegos["propositos_permitidos"]) == {"estructura", "calidad"}
    assert "TIPOALTA" in u.cortafuegos["columnas_resultado"]


def test_plan_de_figuras_y_definicion_de_terminado_son_validos():
    assert protocolo.validar_plan_figuras(config.cargar_plan_figuras()) == []
    assert protocolo.validar_definicion_terminado(config.cargar_definicion_terminado()) == []
    plan = config.cargar_plan_figuras()
    assert plan.height >= 30


def test_arbol_de_decision_tiene_el_orden_del_plan():
    arbol = config.cargar_arbol_decision()
    codigos = arbol.sort(pl.col("orden").cast(pl.Int64))["codigo"].to_list()
    assert codigos[0] == "R_FASE1"
    assert codigos.index("FUGA") < codigos.index("PROXY") < codigos.index("CALIDAD")
    assert codigos.index("CALIDAD") < codigos.index("REDUNDANTE") < codigos.index("BAJA_SENAL")
    fila = arbol.filter(pl.col("codigo") == "R_FASE1").row(0, named=True)
    assert fila["prevalece_sobre_estadistica"] == "si"


def test_validar_todo_no_reporta_problemas():
    res = protocolo.validar_todo()
    assert all(v == [] for v in res.values()), res
