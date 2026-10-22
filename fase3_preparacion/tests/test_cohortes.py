import polars as pl

from prep import cohortes, elixhauser


def base(**cols):
    n = max(len(v) for v in cols.values())
    d = {"MDC": ["04"] * n, "GRD_ESTADO": ["agrupable"] * n, "EDAD_ANIOS": [60] * n, "TIPO_ACTIVIDAD": ["HOSPITALIZACIÓN"] * n,
         "N_COMORB_MDC14": [None] * n, "ROL": ["NO_EVENTO"] * n, "ESTANCIA_DIAS": [5] * n}
    d.update(cols)
    return pl.DataFrame(d, schema_overrides={"N_COMORB_MDC14": pl.Int32, "EDAD_ANIOS": pl.Int32, "ESTANCIA_DIAS": pl.Int32})


def marcado(**cols):
    return cohortes.marcar(base(**cols))


def test_exclusiones_de_la_cohorte_dura():
    m = marcado(MDC=["04", "15", "14", "14", "99"], GRD_ESTADO=["agrupable", "agrupable", "agrupable", "agrupable", "mdc_99"], N_COMORB_MDC14=[None, None, 0, 2, None])
    assert m["A"].to_list() == [True, False, False, True, False]
    assert m["EX01"].to_list() == [False, False, False, False, True]
    assert m["EX02"].to_list() == [False, True, False, False, False]
    assert m["EX03"].to_list() == [False, False, True, False, False]


def test_solo_adultos_y_edad_desconocida_sale():
    m = marcado(EDAD_ANIOS=[17, 18, None, 40])
    assert m["B"].to_list() == [False, True, False, True]


def test_solo_hospitalizacion():
    m = marcado(TIPO_ACTIVIDAD=["HOSPITALIZACIÓN", "CONSULTA", "(sin dato)"])
    assert m["C"].to_list() == [True, False, False]


def test_cohorte_de_mortalidad_excluye_censurados_y_sin_dato():
    m = marcado(ROL=["NO_EVENTO", "EVENTO", "CENSURADO", "SIN_DATO"])
    assert m["D"].to_list() == [True, True, False, False]


def test_cohorte_de_estadia_solo_sobrevivientes_con_estadia_positiva():
    m = marcado(ROL=["NO_EVENTO", "EVENTO", "NO_EVENTO", "NO_EVENTO"], ESTANCIA_DIAS=[3, 3, 0, None])
    assert m["E"].to_list() == [True, False, False, False]


def test_la_cascada_cuadra_y_suma_el_total():
    b = base(MDC=["04", "15", "14", "04", "04", "04", "04", "04"], N_COMORB_MDC14=[None, None, 0] + [None] * 5,
             EDAD_ANIOS=[60, 60, 60, 10, 60, 60, 60, 60], TIPO_ACTIVIDAD=["HOSPITALIZACIÓN"] * 4 + ["CONSULTA"] + ["HOSPITALIZACIÓN"] * 3,
             ROL=["NO_EVENTO"] * 5 + ["CENSURADO", "EVENTO", "NO_EVENTO"], ESTANCIA_DIAS=[5] * 7 + [0])
    c = cohortes.cascada_anio(cohortes.marcar(b), 2020)
    v = dict(zip(c["paso"], c["n"]))
    assert cohortes.cuadra(c) and v["TOTAL"] == 8 and v["EX02"] == 1 and v["EX03"] == 1 and v["A"] == 6 and v["B"] == 5 and v["C"] == 4
    assert v["D"] == 3 and v["EX06"] == 1 and v["EX07"] == 1 and v["EX08"] == 1 and v["E"] == 1


def test_sin_desenlace_la_cascada_llega_solo_hasta_C():
    b = base(MDC=["04", "04"]).drop("ROL", "ESTANCIA_DIAS")
    c = cohortes.cascada_anio(cohortes.marcar(b), 2024)
    assert "D" not in set(c["paso"]) and "E" not in set(c["paso"]) and cohortes.cuadra(c)


def test_comparacion_con_la_fase_2_detecta_diferencias(tmp_path):
    c = cohortes.cascada_anio(cohortes.marcar(base(MDC=["04", "04"])), 2020)
    ref = c.select("periodo", "paso", pl.col("n")).with_columns(pl.when(pl.col("paso") == "A").then(pl.col("n") + 1).otherwise(pl.col("n")).alias("n"))
    ruta = tmp_path / "ref.csv"
    ref.write_csv(ruta)
    d = cohortes.comparar_con_referencia(c, ruta)
    assert d.filter(pl.col("diferencia") != 0)["paso"].to_list() == ["A"]


def test_comorbilidad_de_obstetricos_usa_solo_diagnosticos_secundarios(tmp_path):
    # un parto (MDC 14) con insuficiencia cardíaca como diagnóstico SECUNDARIO tiene comorbilidad; con ella solo como principal no
    e = elixhauser.cargar()
    from prep import esquema
    dx = pl.DataFrame({c: ["", ""] for c in esquema.DIAG}).with_columns(pl.Series("DIAGNOSTICO2", ["I500", ""]), pl.Series("DIAGNOSTICO1", ["O800", "I500"]))
    flags = elixhauser.calcular_flags(dx, esquema.DIAG_SEC, e)
    assert flags["N_COMORB_ELIX"].to_list() == [1, 0]


# ------------------------------------------------------------------ enmienda 02 (borrador): opciones apagadas por defecto
def _con_tipoalta(**cols):
    b = base(**cols)
    return b.with_columns(pl.Series("ETIQUETA_TIPOALTA", cols.get("ETIQUETA_TIPOALTA", ["DOMICILIO"] * b.height)))


def test_sin_opciones_todo_queda_igual():
    b = _con_tipoalta(MDC=["04", "14", "14"], N_COMORB_MDC14=[None, 0, 2], ETIQUETA_TIPOALTA=["ALTA VOLUNTARIA", "DOMICILIO", "FUGA DEL PACIENTE"])
    assert cohortes.marcar(b).equals(cohortes.marcar(b, opciones=cohortes.Opciones()))
    assert "EX09" not in cohortes.marcar(b).columns
    assert not cohortes.Opciones().activas


def test_mdc14_completo_saca_cesarea_y_preeclampsia_con_comorbilidades():
    # cesárea con hipertensión y preeclampsia con insuficiencia renal: MDC 14 con comorbilidades, hoy en la cohorte
    b = base(MDC=["14", "14", "14", "04"], N_COMORB_MDC14=[0, 1, 2, None], ROL=["NO_EVENTO", "NO_EVENTO", "EVENTO", "EVENTO"])
    hoy = cohortes.marcar(b)
    assert hoy["EX03"].to_list() == [True, False, False, False] and hoy["D"].to_list() == [False, True, True, True]
    op = cohortes.Opciones(mdc14_completo=True)
    m = cohortes.marcar(b, opciones=op)
    assert m["EX03"].to_list() == [True, True, True, False] and m["D"].to_list() == [False, False, False, True]
    muertes_maternas = m.filter(pl.col("D") & (pl.col("MDC") == "14") & (pl.col("ROL") == "EVENTO")).height
    assert muertes_maternas == 0
    c = cohortes.cascada_anio(m, 2020, op)
    assert cohortes.cuadra(c)
    assert c.filter(pl.col("paso") == "EX03")["descripcion"][0] == cohortes.DESC_EX03_MDC14_COMPLETO


def test_alta_voluntaria_y_fuga_salen_de_e_pero_no_de_d():
    b = _con_tipoalta(ROL=["NO_EVENTO"] * 4 + ["EVENTO"], ESTANCIA_DIAS=[3, 3, 3, 0, 3],
                      ETIQUETA_TIPOALTA=["DOMICILIO", "ALTA VOLUNTARIA", "FUGA DEL PACIENTE", "ALTA VOLUNTARIA", "FALLECIDO"])
    op = cohortes.Opciones(excluir_tipoalta_e=("ALTA VOLUNTARIA", "FUGA DEL PACIENTE"))
    m = cohortes.marcar(b, opciones=op)
    assert m["D"].to_list() == [True] * 5
    assert m["E"].to_list() == [True, False, False, False, False]
    assert m["EX09"].to_list() == [False, True, True, False, False]     # la de 0 días ya salía por EX08
    c = cohortes.cascada_anio(m, 2021, op)
    v = dict(zip(c["paso"], c["n"]))
    assert cohortes.cuadra(c) and v["EX09"] == 2 and v["EX08"] == 1 and v["EX07"] == 1 and v["E"] == 1
    assert c["paso"].to_list()[-1] == "EX09"                            # al final: no mueve el orden de los pasos existentes


def test_la_opcion_de_tipo_de_alta_exige_la_etiqueta():
    import pytest
    with pytest.raises(ValueError, match="ETIQUETA_TIPOALTA"):
        cohortes.marcar(base(MDC=["04"]), opciones=cohortes.Opciones(excluir_tipoalta_e=("ALTA VOLUNTARIA",)))


def test_tipos_de_alta_fuera_de_e_salen_de_la_configuracion():
    from prep.config import DIR_CONFIG
    roles = pl.read_csv(DIR_CONFIG / "tipoalta_roles.csv")
    assert cohortes.tipoalta_fuera_de_e(roles) == ("ALTA VOLUNTARIA", "FUGA DEL PACIENTE")
    op = cohortes.opciones_enmienda02(roles)
    assert op.activas and op.mdc14_completo


# ------------------------------------------------------------------ catálogo DEIS (plan 1.3)
def test_reglas_del_tipo_de_establecimiento():
    import pytest
    import catalogo_deis as cd
    assert cd.tipo("110110", "Instituto Traumatológico Dr. Teodoro Gebauer", "Alta Complejidad")[0] == "instituto monográfico"
    assert cd.tipo("112102", "Hospital de Niños Dr. Luis Calvo Mackenna", "Alta Complejidad")[0] == "pediátrico"
    assert cd.tipo("113130", "Hospital Dr. Exequiel González Cortés", "Alta Complejidad")[0] == "pediátrico"
    assert cd.tipo("116110", "Hospital San José (Parral)", "Mediana Complejidad")[0] == "general de mediana complejidad"
    with pytest.raises(ValueError):
        cd.tipo("999999", "Hospital X", "No Aplica")


def test_el_catalogo_publicado_sale_de_la_fuente_deis():
    import pandas as pd
    import catalogo_deis as cd
    nuevo = cd.construir()
    guardado = pd.read_csv(cd.DESTINO, dtype={"cod_hospital": str})
    pd.testing.assert_frame_equal(nuevo.reset_index(drop=True), guardado, check_dtype=False)
    assert len(guardado) == 72 and set(guardado.loc[guardado["tipo"] == "instituto monográfico", "cod_hospital"]) == {"110110", "112103", "112104"}
