"""Opciones de la segunda revisión del profesor guía (02-11-2026): apagadas por defecto, cada una con su paso en la cascada."""
import datetime as dt

import polars as pl
import pytest

from prep import cohortes
from prep.config import cargar_umbrales
from tests.test_cohortes import base

HD = "HOSPITALIZACIÓN DOMICILIARIA"


def _b(**cols):
    b = base(**cols)
    n = b.height
    extra = {"ETIQUETA_TIPOALTA": cols.get("ETIQUETA_TIPOALTA", ["DOMICILIO"] * n), "DUP_Q15": cols.get("DUP_Q15", [False] * n),
             "ENTRANTE_ENLAZADO": cols.get("ENTRANTE_ENLAZADO", [False] * n)}
    return b.with_columns(*[pl.Series(k, v) for k, v in extra.items() if k not in b.columns])


def test_sin_opciones_las_columnas_nuevas_no_cambian_nada():
    b = _b(ROL=["NO_EVENTO", "CENSURADO", "EVENTO"], ETIQUETA_TIPOALTA=["DOMICILIO", HD, "FALLECIDO"], DUP_Q15=[True, False, False],
           ENTRANTE_ENLAZADO=[False, False, True])
    m = cohortes.marcar(b)
    assert m["D"].to_list() == [True, False, True]
    assert not {"EX10", "EX11", "EX12", "ROL_MODELO"} & set(m.columns)
    assert not cohortes.Opciones().activas


def test_hospitalizacion_domiciliaria_entra_a_d_como_viva_y_no_a_e():
    b = _b(ROL=["NO_EVENTO", "CENSURADO", "CENSURADO", "EVENTO"], ETIQUETA_TIPOALTA=["DOMICILIO", HD, "DERIVACIÓN OTRO HOSPITAL DE LA RED NACIONAL", "FALLECIDO"])
    op = cohortes.Opciones(tipoalta_vivos=(HD,))
    m = cohortes.marcar(b, opciones=op)
    assert m["D"].to_list() == [True, True, False, True]
    assert m["ROL_MODELO"].to_list()[1] == "NO_EVENTO"
    assert m["E"].to_list() == [True, False, False, False]
    assert m["EX11"].to_list() == [False, True, False, False]
    c = cohortes.cascada_anio(m, 2022, op)
    v = dict(zip(c["paso"], c["n"]))
    assert cohortes.cuadra(c) and v["EX06"] == 1 and v["EX11"] == 1 and v["EX07"] == 1


def test_filas_repetidas_y_entrantes_enlazados_salen_de_d():
    b = _b(ROL=["NO_EVENTO", "NO_EVENTO", "EVENTO", "CENSURADO", "NO_EVENTO"], DUP_Q15=[False, True, False, True, True],
           ENTRANTE_ENLAZADO=[False, False, True, False, True])
    op = cohortes.Opciones(deduplicar_q15=True, excluir_entrantes_enlazados=True)
    m = cohortes.marcar(b, opciones=op)
    assert m["D"].to_list() == [True, False, False, False, False]
    assert m["EX10"].to_list() == [False, True, False, False, True]       # la repetida censurada ya salía por EX06
    assert m["EX12"].to_list() == [False, False, True, False, False]      # repetida y enlazada cuenta una sola vez, en EX10
    c = cohortes.cascada_anio(m, 2021, op)
    v = dict(zip(c["paso"], c["n"]))
    assert cohortes.cuadra(c) and (v["EX06"], v["EX10"], v["EX12"], v["D"]) == (1, 2, 1, 1)
    assert c["paso"].to_list()[-2:] == ["EX10", "EX12"]


def test_las_opciones_exigen_sus_columnas():
    with pytest.raises(ValueError, match="DUP_Q15"):
        cohortes.marcar(base(MDC=["04"]), opciones=cohortes.Opciones(deduplicar_q15=True))
    with pytest.raises(ValueError, match="ETIQUETA_TIPOALTA"):
        cohortes.marcar(base(MDC=["04"]), opciones=cohortes.Opciones(tipoalta_vivos=(HD,)))


def test_q15_conserva_la_primera_fila_y_no_marca_ids_nulos():
    f = dt.date(2021, 3, 1)
    plata = pl.DataFrame({"ID_PACIENTE_NUM": [11, 11, 11, None, None], "COD_HOSPITAL": ["A"] * 5, "FECHA_INGRESO": [f] * 5,
                          "DIAGNOSTICO1": ["I21"] * 5, "FECHA_NACIMIENTO": [f] * 5, "SEXO": ["MUJER"] * 5})
    des = pl.DataFrame({"FECHAALTA": [f, f, dt.date(2021, 3, 9), f, f]})
    assert cohortes.marcar_q15(plata, des).to_list() == [False, True, False, False, False]


def test_entrante_enlazado_con_saliente_del_anio_anterior_del_mismo_regimen(tmp_path):
    u = cargar_umbrales()
    def escribir(anio, plata, des):
        pl.DataFrame(plata).write_parquet(tmp_path / f"plata_{anio}.parquet")
        pl.DataFrame(des).write_parquet(tmp_path / f"desenlace_{anio}.parquet")
    # 2021: paciente 7 sale derivado del hospital A el 28-12; paciente 8 sale derivado pero es menor de edad
    escribir(2021, {"ID_EPISODIO": ["a", "b"], "ID_PACIENTE_NUM": [7, 8], "COD_HOSPITAL": ["A", "A"], "EDAD_ANIOS": [70, 10],
                    "TIPO_ACTIVIDAD": ["HOSPITALIZACIÓN"] * 2},
             {"ES_TRASLADO_SALIENTE": [1, 1], "FECHAALTA": [dt.date(2021, 12, 28), dt.date(2021, 12, 28)]})
    # 2022: entrantes 7 (B, 5 días después: enlaza), 7 (A, mismo hospital: no), 8 (saliente menor: no), 9 (sin saliente: no), 7 a los 40 días: no
    escribir(2022, {"ID_EPISODIO": list("cdefg"), "ID_PACIENTE_NUM": [7, 7, 8, 9, 7], "COD_HOSPITAL": ["B", "A", "B", "B", "C"],
                    "FECHA_INGRESO": [dt.date(2022, 1, 2), dt.date(2022, 1, 2), dt.date(2022, 1, 2), dt.date(2022, 1, 2), dt.date(2022, 2, 6)],
                    "ES_TRASLADO_ENTRANTE": [1, 1, 1, 1, 1], "EDAD_ANIOS": [70] * 5, "TIPO_ACTIVIDAD": ["HOSPITALIZACIÓN"] * 5},
             {"ES_TRASLADO_SALIENTE": [0] * 5, "FECHAALTA": [dt.date(2022, 1, 9)] * 5})
    assert cohortes.entrantes_enlazados(2022, tmp_path, u).to_list() == [True, False, False, False, False]


def test_el_enlace_no_cruza_el_quiebre_del_identificador(tmp_path):
    u = cargar_umbrales()
    pl.DataFrame({"ID_EPISODIO": ["a"], "ID_PACIENTE_NUM": [7], "COD_HOSPITAL": ["A"], "EDAD_ANIOS": [70], "TIPO_ACTIVIDAD": ["HOSPITALIZACIÓN"]}
                 ).write_parquet(tmp_path / "plata_2020.parquet")
    pl.DataFrame({"ES_TRASLADO_SALIENTE": [1], "FECHAALTA": [dt.date(2020, 12, 30)]}).write_parquet(tmp_path / "desenlace_2020.parquet")
    pl.DataFrame({"ID_EPISODIO": ["b"], "ID_PACIENTE_NUM": [7], "COD_HOSPITAL": ["B"], "FECHA_INGRESO": [dt.date(2021, 1, 2)],
                  "ES_TRASLADO_ENTRANTE": [1], "EDAD_ANIOS": [70], "TIPO_ACTIVIDAD": ["HOSPITALIZACIÓN"]}).write_parquet(tmp_path / "plata_2021.parquet")
    pl.DataFrame({"ES_TRASLADO_SALIENTE": [0], "FECHAALTA": [dt.date(2021, 1, 5)]}).write_parquet(tmp_path / "desenlace_2021.parquet")
    assert cohortes.entrantes_enlazados(2021, tmp_path, u).to_list() == [False]
    with pytest.raises(ValueError, match="sellado"):
        cohortes.entrantes_enlazados(2024, tmp_path, u)


def test_opciones_de_la_revision():
    op = cohortes.opciones_revision()
    assert op.activas and op.mdc14_completo and op.deduplicar_q15 and op.excluir_entrantes_enlazados and op.tipoalta_vivos == (HD,)
    assert not cohortes.opciones_revision(False, False, False, False).activas
