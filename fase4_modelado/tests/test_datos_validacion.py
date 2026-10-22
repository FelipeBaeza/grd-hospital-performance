import numpy as np
import pytest

from modelado import datos, validacion
from modelado.config import cargar_reglas
from tests.sintetico import escribir_capas, hacer_df


def test_los_40_predictores_coinciden_con_el_diccionario_de_la_fase_3():
    assert len(datos.PREDICTORES) == 40
    assert set(datos.PREDICTORES) == set(datos.predictores_del_diccionario())


def test_lista_blanca_rechaza_hospital_y_desenlace():
    for prohibida in ("COD_HOSPITAL", "ANIO_EGRESO", "MORTALIDAD_BINARIA", "ESTANCIA_DIAS", "N_COMORB_ELIX", "PUNTAJE_VAN_WALRAVEN"):
        with pytest.raises(datos.DatoProhibido):
            datos.verificar_lista_blanca(datos.PREDICTORES + [prohibida])


def test_las_columnas_que_ve_un_modelo_no_incluyen_claves():
    df = hacer_df(200)
    assert list(datos.X(df).columns) == datos.PREDICTORES
    assert not set(datos.PROHIBIDAS) & set(datos.X(df).columns)


def test_una_matriz_con_2024_se_rechaza(tmp_path):
    df = hacer_df(300, anios=(2019, 2024))
    df["SPLIT"] = np.where(df["ANIO_EGRESO"] == 2024, "prueba", "entrenamiento")
    escribir_capas(tmp_path, df)
    with pytest.raises(datos.DatoProhibido):
        datos.cargar_matriz(tmp_path, "mortalidad")


def test_la_carga_normal_agrega_el_objetivo_y(tmp_path):
    df = hacer_df(300)
    escribir_capas(tmp_path, df)
    c = datos.cargar_matriz(tmp_path, "mortalidad")
    assert "y" in c and len(c) == 300 and set(c["SPLIT"].unique()) <= datos.SPLITS_PERMITIDOS


def test_el_desenlace_incompleto_se_detecta(tmp_path):
    import polars as pl
    df = hacer_df(100)
    escribir_capas(tmp_path, df)
    pl.DataFrame({"ID_EPISODIO": df["ID_EPISODIO"].tolist()[:50], "MORTALIDAD_BINARIA": df["y"].tolist()[:50]}).write_parquet(tmp_path / "oro" / "desenlace_mortalidad.parquet")
    with pytest.raises(ValueError):
        datos.cargar_matriz(tmp_path, "mortalidad")


def test_los_pliegues_siempre_validan_en_un_anio_posterior():
    df = hacer_df(3000)
    r = cargar_reglas()
    for tr, va in validacion.pliegues(df, r):
        assert df.loc[tr, "ANIO_EGRESO"].max() < df.loc[va, "ANIO_EGRESO"].min()
        assert not (tr & va).any()
        assert df.loc[tr | va, "ANIO_EGRESO"].max() <= 2022


def test_un_pliegue_con_fuga_temporal_o_con_2023_se_rechaza():
    with pytest.raises(ValueError):
        validacion.comprobar_pliegue([2019, 2021], 2021, [2019, 2020, 2021, 2022])
    with pytest.raises(ValueError):
        validacion.comprobar_pliegue([2019, 2020, 2021, 2022], 2023, [2019, 2020, 2021, 2022])


def test_mascaras_finales_separan_entrenamiento_y_calibracion():
    df = hacer_df(2000)
    tr, ca = validacion.mascaras_finales(df, cargar_reglas())
    assert df.loc[tr, "ANIO_EGRESO"].max() == 2022 and set(df.loc[ca, "ANIO_EGRESO"]) == {2023}


def test_submuestra_es_determinista():
    a, b = datos.submuestra(1000, 0.3, 5), datos.submuestra(1000, 0.3, 5)
    assert (a == b).all() and 200 < a.sum() < 400 and datos.submuestra(10, 1.0, 1).all()


def test_canario_copia_del_desenlace_no_entra_al_modelo():
    """Auditoría §3.5, prueba 6: una columna igual al desenlace, con un nombre inocente, no pasa la lista blanca ni
    llega al modelo: el modelo entrenado con ella da exactamente las mismas predicciones que sin ella."""
    from modelado.candidatos import entrenar
    with pytest.raises(datos.DatoProhibido):
        datos.verificar_lista_blanca(datos.PREDICTORES + ["MARCA_ADMINISTRATIVA"])
    df = hacer_df(3000, semilla=4)
    df["MARCA_ADMINISTRATIVA"] = df["y"]
    params = {"learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 20, "colsample": 1.0, "subsample": 1.0, "reg_alpha": 0.0, "reg_lambda": 0.0}
    m = entrenar("lightgbm", "mortalidad", df, params, 0, 1, n_arboles=30)
    limpio = entrenar("lightgbm", "mortalidad", df.drop(columns="MARCA_ADMINISTRATIVA"), params, 0, 1, n_arboles=30)
    assert "MARCA_ADMINISTRATIVA" not in m.objeto.booster_.feature_name()
    assert np.array_equal(m.predecir(df), limpio.predecir(df))
