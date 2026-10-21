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
