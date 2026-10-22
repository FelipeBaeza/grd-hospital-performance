import datetime as dt

import polars as pl
import pyarrow.parquet as pq
import pytest

from prep import bronce, plata
from prep.config import cargar_umbrales
from tests.sintetico import escribir, fila

ROLES = pl.read_csv(__import__("pathlib").Path(__file__).resolve().parents[1] / "config" / "tipoalta_roles.csv")


def construir(tmp_path, anio, filas, **kw):
    ruta = escribir(tmp_path / f"GRD_PUBLICO_{anio}.txt", filas, **kw)
    bronce.construir_bronce(anio, ruta, tmp_path / "b")
    rep = plata.construir_plata(anio, tmp_path / "b", tmp_path / "p", tmp_path / "s", ROLES)
    return rep, pl.read_parquet(tmp_path / "p" / f"plata_{anio}.parquet")


def desenlace(tmp_path, anio):
    carpeta = "s" if anio == 2024 else "p"
    return pl.read_parquet(tmp_path / carpeta / f"desenlace_{anio}.parquet")


def test_plata_conserva_todas_las_filas(tmp_path):
    rep, p = construir(tmp_path, 2020, [fila(CIP_ENCRIPTADO=str(100000 + i)) for i in range(30)])
    assert p.height == 30 and rep["cuadra"] and rep["filas_bronce"] == 30


def test_los_dos_formatos_de_fecha_se_unifican(tmp_path):
    _, p = construir(tmp_path, 2020, [fila(FECHA_INGRESO="2020-03-02"), fila(FECHA_INGRESO="02-03-2020")])
    assert p["FECHA_INGRESO"].to_list() == [dt.date(2020, 3, 2)] * 2


def test_fecha_imposible_se_anula_y_el_egreso_se_conserva(tmp_path):
    rep, p = construir(tmp_path, 2020, [fila(FECHA_INGRESO="2020-13-45"), fila()])
    assert p.height == 2 and p["FECHA_INGRESO"][0] is None and p["EDAD_ANIOS"][0] is None
    assert rep["anulados"]["fecha_ingreso_ilegible"] == 1


def test_edad_imposible_se_anula(tmp_path):
    rep, p = construir(tmp_path, 2020, [fila(FECHA_NACIMIENTO="1850-01-01"), fila(FECHA_NACIMIENTO="2025-01-01"), fila()])
    assert p["EDAD_ANIOS"].to_list()[:2] == [None, None] and p["EDAD_ANIOS"][2] == 60
    assert rep["anulados"]["edad_fuera_de_rango"] == 2


def test_edad_se_calcula_en_anios_cumplidos(tmp_path):
    _, p = construir(tmp_path, 2020, [fila(FECHA_NACIMIENTO="2002-03-03", FECHA_INGRESO="2020-03-02"), fila(FECHA_NACIMIENTO="2002-03-02", FECHA_INGRESO="2020-03-02")])
    assert p["EDAD_ANIOS"].to_list() == [17, 18]


def test_estancia_negativa_se_anula(tmp_path):
    rep, _ = construir(tmp_path, 2020, [fila(FECHAALTA="2020-03-01"), fila()])
    d = desenlace(tmp_path, 2020)
    assert d["ESTANCIA_DIAS"].to_list() == [None, 5] and rep["anulados"]["estancia_negativa"] == 1


def test_diagnosticos_se_normalizan(tmp_path):
    _, p = construir(tmp_path, 2020, [fila(DIAGNOSTICO1=" j18.9 ", DIAGNOSTICO2="i50.0", DIAGNOSTICO3=" ")])
    assert p["DIAGNOSTICO1"][0] == "J189" and p["DIAGNOSTICO2"][0] == "I500" and p["DIAGNOSTICO3"][0] == ""


def test_roles_del_tipo_de_alta(tmp_path):
    filas = [fila(TIPOALTA="DOMICILIO"), fila(TIPOALTA="FALLECIDO"), fila(TIPOALTA="DERIVACIÓN OTRO HOSPITAL DE LA RED NACIONAL"), fila(TIPOALTA="DESCONOCIDO"), fila(TIPOALTA="")]
    construir(tmp_path, 2020, filas)
    d = desenlace(tmp_path, 2020)
    assert d["ROL"].to_list() == ["NO_EVENTO", "EVENTO", "CENSURADO", "CENSURADO", "SIN_DATO"]
    assert d["ES_TRASLADO_SALIENTE"].to_list() == [0, 0, 1, 0, 0]


def test_etiqueta_cortada_en_el_origen_se_asigna_por_prefijo(tmp_path):
    construir(tmp_path, 2020, [fila(TIPOALTA="DERIVACIÓN OTRO HOSPITAL DE LA RED NACI")])
    d = desenlace(tmp_path, 2020)
    assert d["ROL"][0] == "CENSURADO" and d["ES_TRASLADO_SALIENTE"][0] == 1


def test_cuentas_de_procedimientos_y_traslados_sin_guardar_las_columnas(tmp_path):
    _, p = construir(tmp_path, 2020, [fila(PROCEDIMIENTO1="1", PROCEDIMIENTO2="2", FECHATRASLADO1="2020-03-03")])
    assert p["N_PROCEDIMIENTOS"][0] == 2 and p["N_TRASLADOS_INTERNOS"][0] == 1
    assert not any(c.startswith(("PROCEDIMIENTO", "FECHATRASLADO", "SERVICIOTRASLADO")) for c in p.columns)


def test_columnas_prohibidas_no_pasan_a_plata(tmp_path):
    _, p = construir(tmp_path, 2020, [fila()])
    for c in ("IR_29301_MORTALIDAD", "IR_29301_SEVERIDAD", "IR_29301_PESO", "MEDICOINTERV1_ENCRIPTADO", "MEDICOALTA_ENCRIPTADO", "PESORN1", "TIPOALTA", "FECHAALTA", "USOSPABELLON"):
        assert c not in p.columns


def test_el_desenlace_de_2024_queda_sellado_y_fuera_de_plata(tmp_path):
    construir(tmp_path, 2024, [fila(FECHA_INGRESO="2024-02-02", FECHAALTA="2024-02-05", TIPOALTA="FALLECIDO")])
    assert (tmp_path / "s" / "desenlace_2024.parquet").exists()
    assert not (tmp_path / "p" / "desenlace_2024.parquet").exists()
    cols = pq.ParquetFile(tmp_path / "p" / "plata_2024.parquet").schema_arrow.names
    assert not {"ROL", "TIPOALTA", "FECHAALTA", "ESTANCIA_DIAS"} & set(cols)


def test_en_2024_no_se_calcula_ningun_agregado_del_desenlace(tmp_path):
    rep, _ = construir(tmp_path, 2024, [fila(FECHA_INGRESO="2024-02-02", FECHAALTA="2024-02-05"), fila(FECHA_INGRESO="2024-02-02", FECHAALTA="2024-02-01")])
    assert "TIPOALTA" not in rep["faltantes"] and "FECHAALTA" not in rep["faltantes"]
    assert "estancia_negativa" not in rep["anulados"] and "fecha_alta_ilegible" not in rep["anulados"]


def test_identificador_atipico_no_se_usa_para_enlazar(tmp_path):
    _, p = construir(tmp_path, 2020, [fila(CIP_ENCRIPTADO="123"), fila(CIP_ENCRIPTADO="1234567"), fila(CIP_ENCRIPTADO="abc1234")])
    assert p["ID_PACIENTE_NUM"].to_list() == [None, 1234567, None]


def test_traslado_entrante_se_marca_con_tipo_de_procedencia(tmp_path):
    _, p = construir(tmp_path, 2020, [fila(TIPO_PROCEDENCIA="OTROS HOSPITALES DE LA RED"), fila()])
    assert p["ES_TRASLADO_ENTRANTE"].to_list() == [1, 0]


def test_grd_no_agrupable_se_clasifica(tmp_path):
    _, p = construir(tmp_path, 2020, [fila(IR_29301_COD_GRD="041101"), fila(IR_29301_COD_GRD="991101"), fila(IR_29301_COD_GRD="DESCONOCIDO"), fila(IR_29301_COD_GRD="12"), fila(IR_29301_COD_GRD="")])
    assert p["GRD_ESTADO"].to_list() == ["agrupable", "mdc_99", "centinela", "formato_invalido", "formato_invalido"][:4] + [p["GRD_ESTADO"][4]]
    assert p["MDC"][0] == "04"


def test_faltantes_se_cuentan_y_nunca_se_imputan(tmp_path):
    rep, p = construir(tmp_path, 2020, [fila(SEXO=""), fila(TIPO_INGRESO="DESCONOCIDO"), fila()])
    assert p["SEXO"].to_list()[0] == "SIN_DATO" and p["TIPO_INGRESO"][1] == "(sin dato)"
    assert rep["faltantes"]["SEXO"] == 1 and rep["faltantes"]["TIPO_INGRESO"] == 1


def test_etiqueta_cortada_ambigua_no_se_asigna():
    """Auditoría §3.5, prueba 9: si la etiqueta cortada calza con dos tipos de alta, no se adivina el rol."""
    from prep import definiciones as dfn
    assert dfn.rol_de_etiqueta("DERIVACIÓN OTRO HOSPITAL DE", ROLES)[1] == "sin_rol"
    assert dfn.rol_de_etiqueta("DERIVACIÓN INST. PRIVADA (", ROLES)[1] == "sin_rol"
    nombre, via, rol = dfn.rol_de_etiqueta("DERIVACIÓN INST. PRIVADA (COMPRA DE SERVIC", ROLES)
    assert (nombre, via, rol) == ("DERIVACIÓN INST. PRIVADA (COMPRA DE SERVICIOS)", "por_prefijo", "CENSURADO")
