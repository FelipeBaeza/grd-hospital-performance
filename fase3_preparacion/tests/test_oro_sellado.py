import datetime as dt

import numpy as np
import polars as pl
import pytest

from prep import catalogo, oro, privacidad, sellado
from prep.config import anio_split, cargar_umbrales
from prep import definiciones as dfn


def test_particion_solo_por_anio():
    assert [anio_split(a) for a in range(2019, 2025)] == ["entrenamiento"] * 4 + ["calibracion", "prueba"]
    with pytest.raises(ValueError):
        anio_split(2018)


def test_vocabulario_agrupa_los_raros_y_protege_sin_dato():
    pred = pl.DataFrame({v: ["A"] * 150 + ["B"] * 5 + [dfn.SIN_DATO] * 2 for v in oro.CATEGORICAS})
    voc = oro.calcular_vocabularios(pred, 100)
    assert voc["CIE10_3C"] == ["A", dfn.SIN_DATO]
    r = oro.aplicar_vocabularios(pred, voc)
    assert set(r["CIE10_3C"].to_list()) == {"A", "RARO", dfn.SIN_DATO}


def test_un_nivel_que_solo_existe_despues_de_2022_cae_en_raro():
    entrenamiento = pl.DataFrame({v: ["A"] * 200 for v in oro.CATEGORICAS})
    voc = oro.calcular_vocabularios(entrenamiento, 100)          # el vocabulario se calcula solo con los años de entrenamiento
    futuro = pl.DataFrame({v: ["A"] * 10 + ["NUEVO"] * 5000 for v in oro.CATEGORICAS})
    assert set(oro.aplicar_vocabularios(futuro, voc)["CIE10_3C"].to_list()) == {"A", "RARO"}


def test_percentil_por_rango_mas_cercano():
    datos = np.arange(1, 101)
    assert oro.percentil_estancia(datos, 99) == 99 and oro.percentil_estancia(datos, 50) == 50 and oro.percentil_estancia(np.array([7]), 99) == 7


def test_percentil_de_estancia_usa_solo_entrenamiento():
    # si el tope se calculara con los años posteriores cambiaría; aquí se comprueba que el valor depende solo de lo que se le entrega
    entren = np.array([1] * 98 + [10, 50])
    todos = np.concatenate([entren, np.array([500] * 50)])
    assert oro.percentil_estancia(entren, 99) != oro.percentil_estancia(todos, 99)


def test_columnas_prohibidas_nunca_son_predictores():
    assert not (set(oro.PREDICTORES) & set(oro.PROHIBIDAS))
    assert "ESPECIALIDAD_MEDICA" not in oro.PREDICTORES and "SERVICIOINGRESO" not in oro.PREDICTORES
    assert "MES_INGRESO" not in oro.PREDICTORES and "DIA_SEMANA" not in oro.PREDICTORES
    assert len([c for c in oro.PREDICTORES if c.startswith("ELIX_")]) == 31


def test_predictores_crudos_calcula_fin_de_semana_y_cuidado_especial():
    from prep import elixhauser, esquema
    n = 3
    df = pl.DataFrame({
        "ID_EPISODIO": ["a", "b", "c"], "ANIO_EGRESO": [2020] * n, "COD_HOSPITAL": ["1"] * n, "SEXO": ["HOMBRE"] * n,
        "FECHA_INGRESO": [dt.date(2020, 10, 17), dt.date(2020, 10, 19), dt.date(2020, 10, 18)], "EDAD_ANIOS": [50, 60, 70],
        "TIPO_INGRESO": ["URGENCIA"] * n, "TIPO_PROCEDENCIA": ["CONSULTA PRIVADA"] * n,
        "SERVICIOINGRESO": ["UCI ADULTO", "MEDICINA", ""], "N_DX_SECUNDARIOS": [0, 0, 0],
        **{c: ["J189", "I10", "E119"] if c == "DIAGNOSTICO1" else [""] * n for c in esquema.DIAG},
    })
    pred, base = oro.predictores_crudos(df, {"J18": "X", "I10": "IX", "E11": "IV"}, elixhauser.cargar())
    assert pred["ES_FIN_SEMANA"].to_list() == [1, 0, 1]                    # sábado, lunes, domingo
    assert pred["NIVEL_CUIDADO_INGRESO"].to_list() == [1, 0, None]
    assert pred["GRUPO_CLINICO"].to_list() == ["X", "IX", "IV"] and pred["CIE10_3C"].to_list() == ["J18", "I10", "E11"]
    assert "N_COMORB_ELIX" not in pred.columns and "N_COMORB_ELIX" in base.columns        # los agregados no van junto a los 31 indicadores


def test_el_desenlace_2024_no_se_puede_leer_sin_autorizacion_y_queda_registro(tmp_path):
    (tmp_path / "sellado").mkdir()
    pl.DataFrame({"ID_EPISODIO": ["2024-1"], "ROL": ["EVENTO"]}).write_parquet(tmp_path / "sellado" / "desenlace_2024.parquet")
    with pytest.raises(sellado.SelloRoto):
        sellado.leer_desenlace(tmp_path / "sellado", 2024, proposito="prueba")
    reg = pl.read_csv(sellado.ruta_registro())
    assert reg["concedido"].to_list() == ["no"] and reg["evento"][0] == "lectura_desenlace"


def test_con_autorizacion_la_lectura_se_concede_y_se_registra(tmp_path, monkeypatch):
    (tmp_path / "sellado").mkdir()
    pl.DataFrame({"ID_EPISODIO": ["2024-1"], "ROL": ["EVENTO"]}).write_parquet(tmp_path / "sellado" / "desenlace_2024.parquet")
    monkeypatch.setenv(sellado.VARIABLE_AUTORIZACION, sellado.FRASE)
    assert sellado.leer_desenlace(tmp_path / "sellado", 2024).height == 1
    assert pl.read_csv(sellado.ruta_registro())["concedido"].to_list() == ["si"]


def test_la_lectura_enmascarada_nunca_devuelve_la_mortalidad(tmp_path, monkeypatch):
    (tmp_path / "sellado").mkdir()
    pl.DataFrame({"ID_EPISODIO": ["a", "b", "c"], "ROL": ["EVENTO", "NO_EVENTO", "CENSURADO"]}).write_parquet(tmp_path / "sellado" / "desenlace_2024.parquet")
    with pytest.raises(sellado.SelloRoto):
        sellado.marca_cohorte_d_2024(tmp_path / "sellado", pl.Series(["a", "b", "c"]))
    monkeypatch.setenv(sellado.VARIABLE_AUTORIZACION, sellado.FRASE)
    r = sellado.marca_cohorte_d_2024(tmp_path / "sellado", pl.Series(["a", "b", "c"]))
    assert r.columns == ["ID_EPISODIO", "ENTRA_A_D"] and r.sort("ID_EPISODIO")["ENTRA_A_D"].to_list() == [True, True, False]


def test_los_anios_libres_se_leen_sin_registro_de_sellado(tmp_path):
    (tmp_path / "plata").mkdir()
    (tmp_path / "sellado").mkdir()
    pl.DataFrame({"ID_EPISODIO": ["2023-1"], "ROL": ["EVENTO"]}).write_parquet(tmp_path / "plata" / "desenlace_2023.parquet")
    assert sellado.leer_desenlace(tmp_path / "sellado", 2023).height == 1
    assert not sellado.ruta_registro().exists()


def test_escaneo_de_privacidad_detecta_rut_fechas_y_celdas_pequenas(tmp_path):
    (tmp_path / "a.csv").write_text("anio,n\n2020,5\n2021,500\n", encoding="utf8")
    (tmp_path / "b.md").write_text("el paciente 12.345.678-9 ingresó el 2020-03-02", encoding="utf8")
    (tmp_path / "c.csv").write_text("anio,n\n2020,500\n", encoding="utf8")
    h = {x["archivo"]: x for x in privacidad.escanear_carpeta(tmp_path)}
    assert h["a.csv"]["celdas_pequenas"] == 1 and h["b.md"]["rut"] == 1 and h["b.md"]["fechas"] == 1 and "c.csv" not in h


def test_supresion_de_celdas_del_catalogo():
    d = pl.DataFrame({"n": [0, 3, 9, 10, 500]}).select(catalogo.suprimir(pl.col("n")).alias("n"))
    assert d["n"].to_list() == ["0", "<10", "<10", "10", "500"]


def test_script_sql_no_lleva_nombres_ni_celdas_pequenas():
    cat = pl.DataFrame({"anio": [2023], "cod_hospital": ["101100"], "n_egresos": ["1000"], "n_cohorte_c": ["800"], "n_cohorte_d": ["700"], "n_fallecidos_d": ["<10"],
                        "en_catalogo_oficial": [True], "cumple_volumen_observado": [False]})
    s = catalogo.script_sql(cat)
    assert "CREATE TABLE" in s and "(2023, '101100', 1000, 800, 700, NULL, TRUE, FALSE)" in s


def test_umbrales_heredados_de_la_fase_2():
    u = cargar_umbrales()
    assert u["edad_adulto"] == 18 and u["edad_maxima"] == 110 and u["id_largo_minimo"] == 5 and u["percentil_estancia"] == 99
    assert u.particion["anios_entrenamiento"] == [2019, 2020, 2021, 2022] and u.particion["anio_prueba"] == 2024
