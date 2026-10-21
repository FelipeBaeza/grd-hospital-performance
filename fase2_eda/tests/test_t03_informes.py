"""Tarea 3: el generador de informes falla si una cifra no viene de un CSV."""

from __future__ import annotations

import polars as pl
import pytest

from eda import informes
from eda.informes import CifraSinRespaldo, Informe, cifra, escanear_cifras_md, leer_cifra, validar_narrativa


def test_narrativa_acepta_anios_ids_y_referencias():
    validar_narrativa("Entre 2019 y 2022 se aplica la regla R1 a ELIX_04 según la decisión 15 de la Fase 1 (sección 5.2).")
    validar_narrativa("Los códigos CIE-10 y MDC 15 se leen como texto en `t21_archivos.csv`.")


@pytest.mark.parametrize(
    "texto",
    ["La mortalidad fue 2,5 % en el período", "Hubo 15 hospitales", "Se leyeron 1.234 filas", "Un 7 por ciento"],
)
def test_narrativa_rechaza_cifras_escritas_a_mano(texto):
    with pytest.raises(CifraSinRespaldo):
        validar_narrativa(texto)


def test_leer_cifra_exige_una_sola_fila():
    df = pl.DataFrame({"anio": [2019, 2020], "n": [10, 20]})
    assert leer_cifra(df, "t.csv", "n", anio=2020).valor == 20
    with pytest.raises(CifraSinRespaldo):
        leer_cifra(df, "t.csv", "n")
    with pytest.raises(CifraSinRespaldo):
        leer_cifra(df, "t.csv", "n", anio=1999)
    with pytest.raises(CifraSinRespaldo):
        leer_cifra(df, "t.csv", "no_existe", anio=2019)


def test_formatos_en_estilo_chileno():
    assert informes.fmt_entero(1151475) == "1.151.475"
    assert informes.fmt_dec(2.945, 2) in ("2,94", "2,95")
    assert informes.fmt_dec(1234.5, 1) == "1.234,5"
    assert informes.FORMATOS["pct1"](12.34) == "12,3 %"
    assert informes.FORMATOS["prop2"](0.0294) == "2,94 %"


def test_informe_resuelve_marcadores_desde_csv_y_los_registra(ctx):
    ctx.escribir_tabla("t_x.csv", pl.DataFrame({"anio": [2019, 2020], "n": [500, 700]}), tarea=3, descripcion="x", cols_n=["n"])
    inf = Informe(ctx, "prueba.md", "Informe de prueba")
    inf.seccion("1. Resumen")
    inf.parrafo("En {a} hubo {n} egresos.", a=cifra(ctx, "t_x.csv", "anio", anio=2020), n=cifra(ctx, "t_x.csv", "n", anio=2020))
    inf.tabla("t_x.csv", formatos={"anio": "texto"})
    ruta = inf.guardar()
    md = ruta.read_text(encoding="utf8")
    assert "En 2.020 hubo 700 egresos." in md  # el año sale como entero: el formato debe pedirse con |texto
    assert ctx.cifras_informe().height >= 3


def test_marcador_sin_cifra_o_con_digitos_a_mano_falla(ctx):
    ctx.escribir_tabla("t_x.csv", pl.DataFrame({"anio": [2019], "n": [500]}), tarea=3, descripcion="x")
    inf = Informe(ctx, "prueba.md", "Informe de prueba")
    with pytest.raises(CifraSinRespaldo):
        inf.parrafo("Hubo {n} egresos.")  # el marcador no tiene cifra
    with pytest.raises(CifraSinRespaldo):
        inf.parrafo("Hubo 500 egresos.")  # cifra escrita a mano
    with pytest.raises(CifraSinRespaldo):
        inf.parrafo("Hubo {n|formato_raro} egresos.", n=cifra(ctx, "t_x.csv", "n"))


def test_tabla_marca_celdas_suprimidas(ctx):
    ctx.escribir_tabla("t_y.csv", pl.DataFrame({"g": ["a", "b"], "n": [5, 50]}), tarea=3, descripcion="y", cols_n=["n"])
    inf = Informe(ctx, "prueba.md", "Informe de prueba")
    inf.tabla("t_y.csv")
    md = inf.a_markdown()
    assert "| a | <10 |" in md and "| b | 50 |" in md


def test_escaneo_posterior_detecta_cifras_sueltas():
    md = "# 1. Título\n\nTexto sin cifras.\n\nHubo 12 casos.\n\n| a | 12 |\n|---|---|\n\nUna cifra 1.234 registrada.\n"
    malas = escanear_cifras_md(md, ["1.234"])
    assert malas == ["Hubo 12 casos."]
    # un token corto no debe borrar los años de la prosa
    assert escanear_cifras_md("Entre 2019 y 2024 hubo 2 grupos.", ["2"]) == []


def test_informe_falla_al_guardar_si_la_prosa_tiene_cifras(ctx):
    inf = Informe(ctx, "prueba.md", "Informe de prueba")
    inf.partes.append("Se registraron 33 casos.\n")  # se colocó a mano, saltando el validador
    with pytest.raises(CifraSinRespaldo):
        inf.guardar()


def test_una_tabla_con_columnas_calculadas_se_registra_como_derivada_de_su_csv(ctx):
    """DT02 exige las columnas de una tabla directa en el encabezado de su CSV; una tabla con conteos por grupo o uniones se marca como derivada."""
    ctx.escribir_tabla("t_z.csv", pl.DataFrame({"g": ["a", "a", "b"], "n": [50, 60, 70]}), tarea=3, descripcion="z")
    inf = Informe(ctx, "prueba.md", "Informe de prueba")
    inf.tabla("t_z.csv", columnas=["g", "n"])                                                               # directa: sus dos columnas están en el CSV
    inf.tabla("t_z.csv", df=pl.DataFrame({"g": ["a", "b"], "n": [2, 1]}))                                   # con df propio pero con columnas del CSV: sigue siendo directa
    inf.tabla("t_z.csv", df=pl.DataFrame({"g": ["a", "b"], "n_filas": [2, 1]}), columnas=["g", "n_filas"])  # n_filas se calculó: derivada
    inf.tabla("t_inexistente.csv", df=pl.DataFrame({"x": [1]}))                                             # sin CSV en disco no hay con qué comparar: directa
    registro = [(c["marcador"], c["columna"], c["formato"]) for c in inf.cifras]
    assert registro == [
        (informes.MARCADOR_TABLA, "g;n", "tabla"), (informes.MARCADOR_TABLA, "g;n", "tabla"),
        (informes.MARCADOR_TABLA_DERIVADA, "g;n_filas", "tabla"), (informes.MARCADOR_TABLA, "x", "tabla"),
    ]
    assert informes.MARCADOR_TABLA == "(tabla)" and informes.MARCADOR_TABLA_DERIVADA == "(tabla derivada)"
