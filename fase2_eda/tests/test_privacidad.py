"""Escáner de privacidad de las salidas (se usa en el cierre, criterio DT04)."""

from __future__ import annotations

import polars as pl

from eda import privacidad


def test_escanear_texto_detecta_fechas_ruts_y_numeros_largos():
    t = "Nació el 1950-03-04 y el 04-03-1950; RUT 12.345.678-9 y 12345678-K; id 123456789 en el texto; fecha 99-99-9999 es máscara."
    r = privacidad.escanear_texto(t)
    assert r["fecha_completa"] == 2 and r["rut"] == 2 and r["numero_largo_en_texto"] >= 1


def test_mascaras_y_cifras_formateadas_no_se_marcan():
    t = "Patrón 9999-99-99 y 99-99-9999; total 5.808.536 casos; tasa 12,5 %; hash 4456ae03697fc7a4; versión 2.5.3"
    r = privacidad.escanear_texto(t)
    assert r == {"fecha_completa": 0, "rut": 0, "numero_largo_en_texto": 0}


def test_escanear_salidas_marca_encabezados_celdas_pequenas_y_fechas(tmp_path):
    (tmp_path / "tablas").mkdir()
    (tmp_path / "informes").mkdir()
    pl.DataFrame({"CIP_ENCRIPTADO": ["a"], "n": [5]}).write_csv(tmp_path / "tablas" / "mala.csv")
    pl.DataFrame({"g": ["x", "y"], "n": [0, 50], "nota": ["ok", "nacido 1999-12-31"]}).write_csv(tmp_path / "tablas" / "regular.csv")
    pl.DataFrame({"g": ["x"], "n": [None]}, schema={"g": pl.String, "n": pl.Int64}).write_csv(tmp_path / "tablas" / "buena.csv")
    (tmp_path / "informes" / "i.md").write_text("Texto con 31-01-2020 y nada más.", encoding="utf8")
    indice = pl.DataFrame(
        {
            "nombre": ["mala.csv", "regular.csv", "buena.csv"],
            "columnas_conteo": ["n", "n", "n"],
        }
    )
    e = privacidad.escanear_salidas(tmp_path, indice, 10)
    por = {(r["archivo"], r["chequeo"]): r["n_violaciones"] for r in e.iter_rows(named=True)}
    assert por[("tablas/mala.csv", "encabezado_sensible")] == 1
    assert por[("tablas/mala.csv", "celda_pequena")] == 1
    assert por[("tablas/regular.csv", "celda_pequena")] == 0  # el cero se informa
    assert por[("tablas/regular.csv", "fecha_completa")] == 1
    assert por[("informes/i.md", "fecha_completa")] == 1
    assert privacidad.total_violaciones(e.filter(pl.col("archivo") == "tablas/buena.csv")) == 0
    assert privacidad.total_violaciones(e) == 4


MANIFIESTO_REAL = (
    '{\n  "datos": {\n    "2019": {\n      "archivo": "GRD_PUBLICO_2019.txt",\n      "lineas_datos": 49999,\n      "sha256": "56d6b5fde2c447d1a98a041a0cd990e98f6f29e0013709a26e5d41f049280ba2",\n'
    '      "tamano_bytes": 24503085\n    }\n  },\n  "semilla": 20261007\n}\n'
)


def test_los_tamanos_y_conteos_numericos_del_manifiesto_no_son_identificadores():
    """Con los archivos reales (de más de diez millones de bytes) el último campo de cada objeto del JSON se marcaba como número largo."""
    assert privacidad.escanear_texto(MANIFIESTO_REAL)["numero_largo_en_texto"] == 2          # escaneado como texto plano, los dos números de 8 dígitos sin coma final
    assert privacidad.escanear_manifiesto(MANIFIESTO_REAL) == {"fecha_completa": 0, "rut": 0, "numero_largo_en_texto": 0}


def test_el_manifiesto_se_sigue_escaneando_en_sus_cadenas(tmp_path):
    malo = '{"nota": "nació el 1950-03-04", "hash": "ab12", "ids": ["12.345.678-9", "123456789"], "123456789": 7, "n": 123456789}'
    assert privacidad.escanear_manifiesto(malo) == {"fecha_completa": 1, "rut": 1, "numero_largo_en_texto": 2}        # una cadena larga y una clave larga; el número 123456789 no cuenta
    assert privacidad.escanear_manifiesto("no es json 1950-03-04")["fecha_completa"] == 1                              # sin JSON válido se escanea como texto
    (tmp_path / "tablas").mkdir()
    (tmp_path / "informes").mkdir()
    (tmp_path / "manifiesto.json").write_text(MANIFIESTO_REAL, encoding="utf8")
    sin_indice = pl.DataFrame(schema={"nombre": pl.String, "columnas_conteo": pl.String})
    assert privacidad.total_violaciones(privacidad.escanear_salidas(tmp_path, sin_indice, 10)) == 0
    (tmp_path / "manifiesto.json").write_text(malo, encoding="utf8")
    e = privacidad.escanear_salidas(tmp_path, sin_indice, 10)
    assert privacidad.total_violaciones(e) == 4 and set(e["archivo"]) == {"manifiesto.json"}
