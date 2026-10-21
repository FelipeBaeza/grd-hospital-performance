import hashlib

import polars as pl
import pyarrow.parquet as pq
import pytest

from prep import bronce, esquema
from tests.sintetico import escribir, fila


def test_esquema_tiene_129_columnas_unicas():
    assert esquema.N_COLUMNAS == 129 and esquema.nombre_unico(list(esquema.COLUMNAS))


def test_bronce_conserva_filas_y_cuadra(tmp_path):
    ruta = escribir(tmp_path / "GRD_PUBLICO_2020.txt", [fila(CIP_ENCRIPTADO=str(100000 + i)) for i in range(25)], bom=True, crlf=True)
    rep = bronce.construir_bronce(2020, ruta, tmp_path / "b")
    assert rep["filas_leidas"] == 25 and rep["filas_rechazadas"] == 0 and rep["cuadra"]
    assert pq.ParquetFile(tmp_path / "b" / "bronce_2020.parquet").metadata.num_rows == 25


def test_linea_rechazada_se_guarda_aparte_y_la_cuenta_cuadra(tmp_path):
    ruta = escribir(tmp_path / "GRD_PUBLICO_2022.txt", [fila(CIP_ENCRIPTADO="100001"), "a|b|c", fila(CIP_ENCRIPTADO="100002")])
    rep = bronce.construir_bronce(2022, ruta, tmp_path / "b")
    assert rep["lineas_datos"] == 3 and rep["filas_leidas"] == 2 and rep["filas_rechazadas"] == 1 and rep["cuadra"]
    rech = (tmp_path / "b" / "rechazadas_2022.csv").read_text(encoding="utf8")
    assert rech.splitlines()[1] == "3,3"          # número de línea y de campos; nunca el contenido
    assert "a|b|c" not in rech


def test_id_episodio_usa_la_linea_original_aunque_haya_rechazadas(tmp_path):
    ruta = escribir(tmp_path / "GRD_PUBLICO_2022.txt", [fila(CIP_ENCRIPTADO="100001"), "x|y", fila(CIP_ENCRIPTADO="100002")])
    bronce.construir_bronce(2022, ruta, tmp_path / "b")
    ids = pl.read_parquet(tmp_path / "b" / "bronce_2022.parquet", columns=["ID_EPISODIO", "LINEA_ORIGEN"])
    assert ids["LINEA_ORIGEN"].to_list() == [2, 4]
    assert all(i.startswith(f"2022-{l:08d}-") for i, l in zip(ids["ID_EPISODIO"], ids["LINEA_ORIGEN"]))


def test_id_episodio_es_determinista_y_unico(tmp_path):
    filas = [fila(CIP_ENCRIPTADO=str(100000 + i)) for i in range(40)]
    r1 = escribir(tmp_path / "a" / "GRD_PUBLICO_2021.txt", filas)
    r2 = escribir(tmp_path / "b" / "GRD_PUBLICO_2021.txt", filas)
    bronce.construir_bronce(2021, r1, tmp_path / "ba")
    bronce.construir_bronce(2021, r2, tmp_path / "bb")
    a = pl.read_parquet(tmp_path / "ba" / "bronce_2021.parquet")["ID_EPISODIO"]
    b = pl.read_parquet(tmp_path / "bb" / "bronce_2021.parquet")["ID_EPISODIO"]
    assert a.to_list() == b.to_list() and a.n_unique() == 40


def test_alias_historico_del_identificador_se_acepta(tmp_path):
    ruta = escribir(tmp_path / "GRD_PUBLICO_2019.txt", [fila()], alias_id=True)
    bronce.construir_bronce(2019, ruta, tmp_path / "b")
    assert "CIP_ENCRIPTADO" in pq.ParquetFile(tmp_path / "b" / "bronce_2019.parquet").schema_arrow.names


def test_encabezado_con_otro_numero_de_columnas_detiene_el_proceso(tmp_path):
    ruta = tmp_path / "GRD_PUBLICO_2020.txt"
    ruta.write_text("a|b|c\n1|2|3\n", encoding="utf8")
    with pytest.raises(bronce.EsquemaInvalido):
        bronce.construir_bronce(2020, ruta, tmp_path / "b")


def test_columna_con_nombre_distinto_detiene_el_proceso(tmp_path):
    ruta = escribir(tmp_path / "GRD_PUBLICO_2020.txt", [fila()])
    texto = ruta.read_text(encoding="utf8").replace("SEXO", "GENERO", 1)
    ruta.write_text(texto, encoding="utf8")
    with pytest.raises(bronce.EsquemaInvalido):
        bronce.construir_bronce(2020, ruta, tmp_path / "b")


def test_rut_en_columna_sensible_se_cuenta_y_la_columna_se_descarta(tmp_path):
    ruta = escribir(tmp_path / "GRD_PUBLICO_2019.txt", [fila(USOSPABELLON="12.345.678-9"), fila(USOSPABELLON="12345678-K"), fila()])
    rep = bronce.construir_bronce(2019, ruta, tmp_path / "b")
    assert rep["hallazgos_rut"] == {"USOSPABELLON": 2}
    esquema_b = pq.ParquetFile(tmp_path / "b" / "bronce_2019.parquet").schema_arrow.names
    assert "USOSPABELLON" not in esquema_b and "FECHAPROCEDIMIENTO1" not in esquema_b
    assert "12.345.678-9" not in str(rep)


def test_rut_en_otra_columna_detiene_el_proceso(tmp_path):
    ruta = escribir(tmp_path / "GRD_PUBLICO_2019.txt", [fila(COMUNA="12.345.678-9")])
    with pytest.raises(bronce.HallazgoSensibleNuevo):
        bronce.construir_bronce(2019, ruta, tmp_path / "b")
    assert not (tmp_path / "b" / "bronce_2019.parquet").exists()


def test_el_archivo_original_no_se_modifica(tmp_path):
    ruta = escribir(tmp_path / "GRD_PUBLICO_2020.txt", [fila(CIP_ENCRIPTADO=str(100000 + i)) for i in range(10)])
    antes = hashlib.sha256(ruta.read_bytes()).hexdigest()
    bronce.construir_bronce(2020, ruta, tmp_path / "b")
    assert hashlib.sha256(ruta.read_bytes()).hexdigest() == antes


def test_filas_repetidas_exactas_se_documentan_sin_eliminarlas(tmp_path):
    ruta = escribir(tmp_path / "GRD_PUBLICO_2020.txt", [fila(), fila(), fila(CIP_ENCRIPTADO="999999")])
    rep = bronce.construir_bronce(2020, ruta, tmp_path / "b")
    assert rep["filas_repetidas_exactas"] == 1 and rep["filas_leidas"] == 3
