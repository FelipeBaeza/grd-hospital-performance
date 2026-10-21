"""Tarea 13, parte B: los doce verificadores de la definición de terminado (DT01 a DT12). Cada uno con una prueba positiva y otra que lo hace fallar a propósito."""

from __future__ import annotations

import json
import subprocess
import sys

import polars as pl
import pytest

from ayudas_t13 import LanzadorCorrida, cifras_informe, clonar, etiquetar_completo, registro
from eda import config, cortafuegos, decision, ledger, rutas, verificadores
from eda.contexto import Contexto
from eda.informes import fmt_entero
from eda.verificadores import Banderas, ResultadoDT

MANIFIESTO_REAL = (
    '{\n  "datos": {\n    "2019": {\n      "archivo": "GRD_PUBLICO_2019.txt",\n      "sha256": "56d6b5fde2c447d1a98a041a0cd990e98f6f29e0013709a26e5d41f049280ba2",\n'
    '      "tamano_bytes": 24503085\n    }\n  }\n}\n'
)


@pytest.fixture(scope="module")
def ctx(corrida_t13):
    """La carpeta de salidas de la corrida sintética (solo lectura)."""
    return corrida_t13.ctx


def nombres_informes(ctx: Contexto) -> tuple[str, ...]:
    """Los informes que hay en la carpeta (para las pruebas de DT02 que solo miran el registro de cifras)."""
    return tuple(sorted(p.name for p in ctx.dir_informes.glob("*.md")))


# ============================================================================================================================ el resultado y el conjunto
def test_un_resultado_solo_admite_los_tres_estados():
    assert ResultadoDT("DT99", "criterio", "PASS", "detalle").evidencia == ""
    for estado in ("PASS", "FAIL", "PENDIENTE"):
        assert ResultadoDT("DT99", "c", estado, "d").estado == estado
    with pytest.raises(ValueError, match="estado inválido"):
        ResultadoDT("DT99", "c", "OK", "d")


def test_cada_verificador_de_la_definicion_de_terminado_existe_y_son_doce():
    dt = config.cargar_definicion_terminado()
    assert dt.height == 12 and dt["id"].to_list() == [f"DT{i:02d}" for i in range(1, 13)]
    nombres = dt["verificador"].to_list()
    assert nombres == ["decisiones_cubiertas", "cifras_con_csv", "cero_lecturas_2024", "escaneo_privacidad", "cascada_cuadra", "diccionario_conciliado", "ledger_completo",
                       "figuras_indexadas", "matriz_cubre_candidatas", "pytest_verde", "corrida_determinista", "completo_corrido"]
    for nombre in nombres:
        assert callable(getattr(verificadores, nombre)), nombre


def test_los_resultados_toman_el_id_y_el_criterio_de_la_definicion_de_terminado(ctx):
    r = verificadores.completo_corrido(ctx)
    fila = config.cargar_definicion_terminado().filter(pl.col("verificador") == "completo_corrido").row(0, named=True)
    assert (r.id, r.criterio) == (fila["id"], fila["criterio"]) == ("DT12", "Corrida con datos completos")


def test_el_detalle_larga_de_problemas_se_resume_con_los_primeros_cinco():
    assert verificadores._lista(["b", "a", "b"]) == "a; b"
    assert verificadores._lista([f"p{i}" for i in range(5)]) == "p0; p1; p2; p3; p4"
    assert verificadores._lista([f"p{i}" for i in range(8)]) == "p0; p1; p2; p3; p4; y 3 más"
    assert verificadores._lista([]) == ""


# ============================================================================================================================ DT01
def test_dt01_las_17_decisiones_estan_cubiertas(ctx):
    r = verificadores.decisiones_cubiertas(ctx)
    assert r.id == "DT01" and r.estado == "PASS"
    assert r.detalle == "17 de 17 decisiones cubiertas; el protocolo trae 26 hipótesis y 15 supuestos"


def test_dt01_falla_si_una_decision_se_queda_sin_hipotesis(ctx):
    hip = config.cargar_hipotesis()
    sin_h07 = hip.filter(pl.col("id") != "H07")          # la decisión 7 solo tiene H07 como hipótesis
    r = verificadores.decisiones_cubiertas(ctx, hipotesis=sin_h07)
    assert r.estado == "FAIL" and "la decisión 7 no tiene hipótesis" in r.detalle


def test_dt01_falla_si_la_tabla_de_evidencia_no_trae_exactamente_las_17(ctx):
    ev = ctx.leer_tabla("decisiones_evidencia.csv", columnas_texto=["n", "tipo_evidencia"])
    r = verificadores.decisiones_cubiertas(ctx, tabla_evidencia=ev.filter(pl.col("n") != "5"))
    assert r.estado == "FAIL" and "faltan las decisiones 5" in r.detalle
    extra = pl.concat([ev, ev.filter(pl.col("n") == "3")])
    assert "repetidas: 3" in verificadores.decisiones_cubiertas(ctx, tabla_evidencia=extra).detalle
    ajena = pl.concat([ev, ev.filter(pl.col("n") == "3").with_columns(pl.lit("18").alias("n"))])
    assert "sobran las decisiones 18" in verificadores.decisiones_cubiertas(ctx, tabla_evidencia=ajena).detalle


def test_dt01_el_protocolo_exige_una_hipotesis_por_decision_aunque_sea_institucional(ctx):
    hip = config.cargar_hipotesis().filter(pl.col("id") != "H10")           # la 10 es institucional y solo tiene H10
    r = verificadores.decisiones_cubiertas(ctx, hipotesis=hip)
    assert r.estado == "FAIL" and "la decisión 10 no tiene hipótesis" in r.detalle


def test_dt01_cada_decision_de_la_tabla_de_evidencia_trae_una_hipotesis_salvo_las_institucionales_o_que_no_dependen_de_datos(ctx):
    ev = ctx.leer_tabla("decisiones_evidencia.csv", columnas_texto=["n", "hipotesis", "tipo_evidencia"])
    sin = lambda *numeros: ev.with_columns(pl.when(pl.col("n").is_in(list(numeros))).then(pl.lit("  ")).otherwise(pl.col("hipotesis")).alias("hipotesis"))   # noqa: E731
    # la 7 es cuantitativa: sin hipótesis en su fila falla
    r = verificadores.decisiones_cubiertas(ctx, tabla_evidencia=sin("7"))
    assert r.estado == "FAIL" and r.detalle == "la decisión 7 no tiene ninguna hipótesis en la tabla de evidencia ni es institucional o independiente de los datos"
    # la 4 no depende de datos y la 10 y la 12 son institucionales: no las necesitan
    assert verificadores.decisiones_cubiertas(ctx, tabla_evidencia=sin("4", "10", "12")).estado == "PASS"
    assert "la decisión 7" in verificadores.decisiones_cubiertas(ctx, tabla_evidencia=sin("4", "7", "10")).detalle


# ============================================================================================================================ DT02
def test_dt02_las_cifras_de_los_seis_informes_previos_se_reproducen_desde_sus_tablas(ctx, corrida_t13):
    r = verificadores.cifras_con_csv(ctx, informes_esperados=corrida_t13.previos)
    assert r.id == "DT02" and r.estado == "PASS", r.detalle
    cifras = config.leer_csv_texto(ctx.dir_tablas / "cifras_informe.csv")
    n_derivadas = cifras.filter(pl.col("marcador") == "(tabla derivada)").height
    n_escalares = cifras.filter(pl.col("formato") != "tabla").height
    assert f"{cifras.height:,}".replace(",", ".") + " cifras de 6 informes" in r.detalle and n_derivadas >= 4 and n_escalares > 100
    assert f"{n_escalares} cifras escalares" in r.detalle and f"{n_derivadas} tablas derivadas" in r.detalle


def test_dt02_exige_los_siete_informes_salvo_que_se_indique_otra_cosa(ctx):
    r = verificadores.cifras_con_csv(ctx)                       # por defecto espera t26_cierre.md, que aún no existe
    assert r.estado == "FAIL" and "faltan los informes t26_cierre.md" in r.detalle


def test_dt02_falla_con_una_cifra_sin_csv_o_con_un_csv_inexistente(ctx):
    base = {"informe": "t24_informe.md", "marcador": "n", "csv": "t24_resumen.csv", "columna": "n_reglas", "filtro": "{}", "formato": "entero", "texto": "x"}
    inexistente = cifras_informe({**base, "csv": "no_existe.csv"})
    r = verificadores.cifras_con_csv(ctx, cifras=inexistente, informes_esperados=nombres_informes(ctx))
    assert r.estado == "FAIL" and "t24_informe.md n: el CSV no_existe.csv no existe en tablas" in r.detalle
    sin_csv = cifras_informe({**base, "csv": ""})
    assert "sin CSV o sin columna de origen" in verificadores.cifras_con_csv(ctx, cifras=sin_csv, informes_esperados=nombres_informes(ctx)).detalle
    sin_columna = cifras_informe({**base, "columna": ""})
    assert "sin CSV o sin columna de origen" in verificadores.cifras_con_csv(ctx, cifras=sin_columna, informes_esperados=nombres_informes(ctx)).detalle


def test_dt02_falla_si_la_columna_no_esta_en_el_encabezado_salvo_en_una_tabla_derivada(ctx):
    base = {"informe": "t24_informe.md", "marcador": "(tabla)", "csv": "t24_resumen.csv", "columna": "n_reglas;columna_inventada", "filtro": "{}", "formato": "tabla", "texto": ""}
    r = verificadores.cifras_con_csv(ctx, cifras=cifras_informe(base), informes_esperados=nombres_informes(ctx))
    assert r.estado == "FAIL" and "t24_resumen.csv no tiene la columna columna_inventada" in r.detalle
    derivada = {**base, "marcador": "(tabla derivada)"}
    r = verificadores.cifras_con_csv(ctx, cifras=cifras_informe(derivada), informes_esperados=nombres_informes(ctx))
    assert r.estado == "PASS" and "1 tablas derivadas" in r.detalle
    # una derivada también necesita que su CSV exista
    r = verificadores.cifras_con_csv(ctx, cifras=cifras_informe({**derivada, "csv": "no_existe.csv"}), informes_esperados=nombres_informes(ctx))
    assert r.estado == "FAIL" and "no existe en tablas" in r.detalle


def test_dt02_vuelve_a_leer_cada_cifra_escalar_y_exige_el_mismo_texto(ctx):
    valor = ctx.leer_tabla("t24_resumen.csv")["n_reglas"][0]
    fila = {"informe": "t24_informe.md", "marcador": "n", "csv": "t24_resumen.csv", "columna": "n_reglas", "filtro": "{}", "formato": "entero", "texto": str(valor)}
    assert verificadores.cifras_con_csv(ctx, cifras=cifras_informe(fila), informes_esperados=nombres_informes(ctx)).estado == "PASS"
    mala = verificadores.cifras_con_csv(ctx, cifras=cifras_informe({**fila, "texto": "999"}), informes_esperados=nombres_informes(ctx))
    assert mala.estado == "FAIL" and f"el informe dice 999 y t24_resumen.csv da {valor}" in mala.detalle
    # un filtro que no devuelve exactamente una fila
    sin_fila = verificadores.cifras_con_csv(ctx, cifras=cifras_informe({**fila, "filtro": json.dumps({"n_reglas": -1})}), informes_esperados=nombres_informes(ctx))
    assert sin_fila.estado == "FAIL" and "no se pudo volver a leer la cifra (CifraSinRespaldo)" in sin_fila.detalle


def test_dt02_acepta_una_columna_que_el_informe_leyo_como_texto(ctx):
    # "1.50" leído con el tipo inferido da 1.5; el informe lo leyó como texto (``columnas_texto``) y registró "1.50"
    pl.DataFrame({"clave": ["a", "b"], "v": ["1.50", "2.25"]}).write_csv(ctx.dir_tablas / "t99_texto_numerico.csv")
    try:
        fila = {"informe": "t24_informe.md", "marcador": "v", "csv": "t99_texto_numerico.csv", "columna": "v", "filtro": json.dumps({"clave": "a"}), "formato": "texto", "texto": "1.50"}
        assert verificadores.cifras_con_csv(ctx, cifras=cifras_informe(fila), informes_esperados=nombres_informes(ctx)).estado == "PASS"
        assert verificadores.cifras_con_csv(ctx, cifras=cifras_informe({**fila, "texto": "1.5"}), informes_esperados=nombres_informes(ctx)).estado == "PASS"      # y la lectura con tipo inferido también vale
        assert verificadores.cifras_con_csv(ctx, cifras=cifras_informe({**fila, "texto": "9.99"}), informes_esperados=nombres_informes(ctx)).estado == "FAIL"
    finally:
        (ctx.dir_tablas / "t99_texto_numerico.csv").unlink()


def test_dt02_un_registro_vacio_es_un_fallo(ctx):
    r = verificadores.cifras_con_csv(ctx, cifras=cifras_informe(), informes_esperados=nombres_informes(ctx))
    assert r.estado == "FAIL" and "el registro de cifras de los informes está vacío" in r.detalle


def test_dt02_detecta_llaves_sobrantes_y_informes_de_mas_o_de_menos(corrida_t13, tmp_path):
    c = clonar(corrida_t13.ctx, tmp_path)
    (c.dir_informes / "t24_informe.md").write_text((c.dir_informes / "t24_informe.md").read_text(encoding="utf8") + "\nsobra {llave}\n", encoding="utf8")
    r = verificadores.cifras_con_csv(c, informes_esperados=corrida_t13.previos)
    assert r.estado == "FAIL" and "t24_informe.md: tiene llaves sobrantes" in r.detalle
    (c.dir_informes / "t24_informe.md").write_text("# sin llaves\n", encoding="utf8")
    (c.dir_informes / "t99_extra.md").write_text("# informe de más\n", encoding="utf8")
    (c.dir_informes / "t25_evidencia_decisiones.md").unlink()
    r = verificadores.cifras_con_csv(c, informes_esperados=corrida_t13.previos)
    assert r.estado == "FAIL" and "sobran los informes t99_extra.md" in r.detalle and "faltan los informes t25_evidencia_decisiones.md" in r.detalle


# ============================================================================================================================ DT03
def test_dt03_la_corrida_sintetica_no_leyo_resultados_de_2024(ctx):
    r = verificadores.cero_lecturas_2024(ctx)
    assert r.id == "DT03" and r.estado == "PASS", r.detalle
    reg = ctx.cortafuegos.registro()
    n_2024 = reg.filter(pl.col("anio") == 2024).height
    assert f"{reg.height} accesos registrados, {n_2024} del año 2024: ninguna lectura de resultado y cero violaciones" in r.detalle
    assert "bloqueos 1" in r.detalle                                   # la autoprueba de 2024 de la Tarea 7


def test_dt03_un_registro_vacio_no_tiene_lecturas(ctx):
    r = verificadores.cero_lecturas_2024(ctx, registro=registro())
    assert r.estado == "PASS" and r.detalle.startswith("0 accesos registrados, 0 del año 2024")


def test_dt03_falla_con_una_lectura_inyectada_de_resultado_de_2024(ctx):
    malo = registro({"anio": 2024, "proposito": "estructura", "columnas_resultado": "TIPOALTA", "decision": cortafuegos.PERMITIDO})
    r = verificadores.cero_lecturas_2024(ctx, registro=malo)
    assert r.estado == "FAIL" and "1 lecturas permitidas de columnas de resultado de 2024" in r.detalle
    # enmascarada o por una regla de calidad autorizada: no es una lectura de resultado
    regla = sorted(ctx.cortafuegos.reglas_autorizadas)[0]
    bien = registro(
        {"anio": 2024, "columnas_resultado": "TIPOALTA", "decision": cortafuegos.ENMASCARADO},
        {"anio": 2024, "proposito": "calidad", "columnas_resultado": "TIPOALTA", "reglas": regla, "decision": cortafuegos.REGLA_CALIDAD},
    )
    r = verificadores.cero_lecturas_2024(ctx, registro=bien)
    assert r.estado == "PASS" and "enmascaradas 1 y reglas de calidad autorizadas 1" in r.detalle


def test_dt03_un_proposito_no_permitido_en_2024_solo_vale_si_se_bloqueo(ctx):
    r = verificadores.cero_lecturas_2024(ctx, registro=registro({"anio": 2024, "proposito": "resultado", "decision": cortafuegos.BLOQUEADO}))
    assert r.estado == "PASS" and "bloqueos 1" in r.detalle
    r = verificadores.cero_lecturas_2024(ctx, registro=registro({"anio": 2024, "proposito": "resultado", "decision": cortafuegos.PERMITIDO}))
    assert r.estado == "FAIL" and "1 lecturas de 2024 con un propósito no permitido que no se bloquearon" in r.detalle


def test_dt03_las_tareas_10_a_13_no_leen_2024_y_la_12_y_la_13_no_leen_nada(ctx):
    r = verificadores.cero_lecturas_2024(ctx, registro=registro({"tarea": "T10", "anio": 2024, "decision": cortafuegos.ENMASCARADO}))
    assert r.estado == "FAIL" and "1 accesos de 2024 de las Tareas 10 a 13" in r.detalle
    assert verificadores.cero_lecturas_2024(ctx, registro=registro({"tarea": "T10", "anio": 2023})).estado == "PASS"        # el panel lee 2019 a 2023
    for tarea in ("T12", "T13"):
        r = verificadores.cero_lecturas_2024(ctx, registro=registro({"tarea": tarea, "anio": 2019}))
        assert r.estado == "FAIL" and "1 accesos de las Tareas 12 y 13, que no leen datos" in r.detalle


def test_dt03_una_regla_de_calidad_no_autorizada_es_una_violacion(ctx):
    r = verificadores.cero_lecturas_2024(ctx, registro=registro(
        {"anio": 2024, "proposito": "calidad", "columnas_resultado": "TIPOALTA", "reglas": "R_NO_AUTORIZADA", "decision": cortafuegos.REGLA_CALIDAD}))
    assert r.estado == "FAIL" and "el cortafuegos cuenta 1 violaciones de la política" in r.detalle


def test_dt03_las_tareas_se_comparan_sin_ceros_a_la_izquierda():
    assert [verificadores._tarea_normalizada(t) for t in ("T03", "T3", "t10", " T012 ", "", None)] == ["T3", "T3", "T10", "T12", "", ""]


# ============================================================================================================================ DT04
def test_dt04_las_salidas_de_la_corrida_sintetica_no_tienen_nada_que_proteger(ctx):
    r = verificadores.escaneo_privacidad(ctx)
    assert r.id == "DT04" and r.estado == "PASS", r.detalle
    assert "(tablas e informes)" in r.detalle and r.detalle.endswith("sin identificadores, fechas completas ni celdas de 1 a 9 casos")


def test_dt04_falla_con_una_celda_de_cinco_casos_en_una_columna_de_conteo(corrida_t13, tmp_path):
    c = clonar(corrida_t13.ctx, tmp_path)
    pl.DataFrame({"g": ["x", "y"], "n": [5, 50]}).write_csv(c.dir_tablas / "t99_mala.csv")
    c._tablas["t99_mala.csv"] = {"nombre": "t99_mala.csv", "tarea": "T99", "descripcion": "d", "filas": 2, "columnas_conteo": "n", "celdas_suprimidas": 0, "fuente": "extracto", "sha256": "x"}
    r = verificadores.escaneo_privacidad(c)
    assert r.estado == "FAIL" and r.detalle.startswith("1 violaciones: tablas/t99_mala.csv celda_pequena 1")


def test_dt04_falla_con_una_fecha_completa_o_un_rut_en_un_informe_o_en_el_manifiesto(corrida_t13, tmp_path):
    c = clonar(corrida_t13.ctx, tmp_path)
    (c.dir_informes / "t24_informe.md").write_text("Nació el 1950-03-04.\n", encoding="utf8")
    r = verificadores.escaneo_privacidad(c)
    assert r.estado == "FAIL" and "informes/t24_informe.md fecha_completa 1" in r.detalle
    (c.dir_informes / "t24_informe.md").write_text("limpio\n", encoding="utf8")
    (c.dir_salida / "manifiesto.json").write_text('{"nota": "12.345.678-9"}', encoding="utf8")
    r = verificadores.escaneo_privacidad(c)
    assert r.estado == "FAIL" and "manifiesto.json rut 1" in r.detalle


def test_dt04_un_tamano_de_archivo_del_manifiesto_no_es_un_identificador(corrida_t13, tmp_path):
    c = clonar(corrida_t13.ctx, tmp_path)
    (c.dir_salida / "manifiesto.json").write_text(MANIFIESTO_REAL, encoding="utf8")
    r = verificadores.escaneo_privacidad(c)
    assert r.estado == "PASS" and "(tablas, informes y manifiesto)" in r.detalle


def test_dt04_acepta_un_escaneo_ya_hecho(ctx):
    escaneo = pl.DataFrame({"archivo": ["tablas/a.csv", "tablas/b.csv"], "chequeo": ["rut", "rut"], "n_violaciones": [0, 2]})
    r = verificadores.escaneo_privacidad(ctx, escaneo=escaneo)
    assert r.estado == "FAIL" and r.detalle == "2 violaciones: tablas/b.csv rut 2"
    assert verificadores.escaneo_privacidad(ctx, escaneo=escaneo.filter(pl.col("n_violaciones") == 0)).detalle.startswith("1 archivos escaneados")


# ============================================================================================================================ DT05
ANIO_A = {"TOTAL": 100, "EX01": 5, "EX02": 10, "EX03": 5, "A": 80, "EX04": 10, "B": 70, "EX05": 20, "C": 50, "EX06": 5, "D": 45, "EX07": 3, "EX08": 2, "E": 40}
ANIO_B = {"TOTAL": 200, "EX01": 10, "EX02": 20, "EX03": 10, "A": 160, "EX04": 20, "B": 140, "EX05": 40, "C": 100, "EX06": 10, "D": 90, "EX07": 6, "EX08": 4, "E": 80}
PIPE_A = {"TOTAL": 100, "EX01": 5, "EX02": 10, "EX03_PIPELINE": 15, "COHORTE_DURA_PIPELINE": 70}
PIPE_B = {"TOTAL": 200, "EX01": 10, "EX02": 20, "EX03_PIPELINE": 30, "COHORTE_DURA_PIPELINE": 140}


def casc(periodos: dict[str, dict[str, int | None]]) -> pl.DataFrame:
    filas = [{"periodo": p, "paso": paso, "n": n} for p, pasos in periodos.items() for paso, n in pasos.items()]
    return pl.DataFrame(filas, schema={"periodo": pl.String, "paso": pl.String, "n": pl.Int64})


def suma(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {k: a[k] + b[k] for k in a}


def datos_cascada():
    cascada = casc({"2019": ANIO_A, "2020": ANIO_B, "2019-2020": suma(ANIO_A, ANIO_B)})
    pipe = casc({"2019": PIPE_A, "2020": PIPE_B, "2019-2020": suma(PIPE_A, PIPE_B)})
    archivos = pl.DataFrame({"anio": [2019, 2020], "filas_leidas": [100, 200]})
    return cascada, pipe, archivos


def test_dt05_la_cascada_de_la_corrida_sintetica_cuadra_con_los_valores_sin_suprimir(ctx, corrida_t13):
    r = verificadores.cascada_cuadra(ctx, cascada=corrida_t13.cascada, cascada_pipeline=corrida_t13.cascada_pipeline)
    assert r.id == "DT05" and r.estado == "PASS", r.detalle
    publicada = verificadores.cascada_cuadra(ctx)               # sin los DataFrames usa las tablas publicadas
    assert publicada.estado == "PASS"


def test_dt05_cuenta_a_mano_las_identidades_los_totales_y_las_sumas(ctx):
    cascada, pipe, archivos = datos_cascada()
    r = verificadores.cascada_cuadra(ctx, cascada=cascada, cascada_pipeline=pipe, archivos=archivos)
    # cinco identidades en cada uno de los tres períodos, dos totales anuales, catorce pasos sumados en el período agregado y una identidad del pipeline en cada período
    assert r.estado == "PASS"
    assert r.detalle == ("15 identidades secuenciales, 2 totales anuales contra las filas leídas, 14 sumas de períodos agregados y 3 identidades de la cascada del pipeline cuadran")


def test_dt05_falla_si_un_paso_se_desplaza_en_uno(ctx):
    cascada, pipe, archivos = datos_cascada()
    malo = cascada.with_columns(pl.when((pl.col("periodo") == "2019") & (pl.col("paso") == "A")).then(81).otherwise(pl.col("n")).alias("n"))
    r = verificadores.cascada_cuadra(ctx, cascada=malo, cascada_pipeline=pipe, archivos=archivos)
    assert r.estado == "FAIL" and "2019: A es 81 y TOTAL menos EX01 y EX02 y EX03 da 80" in r.detalle
    assert "2019: B es 70 y A menos EX04 da 71" in r.detalle and "2019-2020: A es 240 y los años suman 241" in r.detalle        # el desplazamiento también rompe la identidad siguiente y el agregado


def test_dt05_una_celda_suprimida_no_hace_fallar_la_cascada(ctx):
    cascada, pipe, archivos = datos_cascada()
    suprimida = cascada.with_columns(pl.when((pl.col("periodo") == "2019") & (pl.col("paso") == "EX02")).then(None).otherwise(pl.col("n")).alias("n"))
    r = verificadores.cascada_cuadra(ctx, cascada=suprimida, cascada_pipeline=pipe, archivos=archivos)
    assert r.estado == "PASS"
    assert r.detalle.startswith("14 identidades secuenciales") and "13 sumas de períodos agregados" in r.detalle           # la identidad de A en 2019 y la suma de EX02 no se verifican


def test_dt05_falla_si_el_total_no_es_el_de_las_filas_leidas(ctx):
    cascada, pipe, archivos = datos_cascada()
    r = verificadores.cascada_cuadra(ctx, cascada=cascada, cascada_pipeline=pipe, archivos=archivos.with_columns(pl.Series("filas_leidas", [99, 200])))
    assert r.estado == "FAIL" and "2019: el total de la cascada es 100 y el archivo trae 99 filas leídas" in r.detalle


def test_dt05_falla_si_un_periodo_agregado_no_suma_sus_anios(ctx):
    cascada, pipe, archivos = datos_cascada()
    malo = cascada.with_columns(pl.when((pl.col("periodo") == "2019-2020") & (pl.col("paso") == "EX04")).then(31).otherwise(pl.col("n")).alias("n"))
    r = verificadores.cascada_cuadra(ctx, cascada=malo, cascada_pipeline=pipe, archivos=archivos)
    assert r.estado == "FAIL" and "2019-2020: EX04 es 31 y los años suman 30" in r.detalle
    # un agregado cuyos años no están todos presentes no se verifica (no hay con qué comparar)
    incompleto = pl.concat([cascada, casc({"2019-2021": ANIO_A})])
    assert verificadores.cascada_cuadra(ctx, cascada=incompleto, cascada_pipeline=pipe, archivos=archivos).estado == "PASS"


def test_dt05_falla_si_la_cascada_del_pipeline_no_cuadra(ctx):
    cascada, pipe, archivos = datos_cascada()
    malo = pipe.with_columns(pl.when((pl.col("periodo") == "2019") & (pl.col("paso") == "COHORTE_DURA_PIPELINE")).then(71).otherwise(pl.col("n")).alias("n"))
    r = verificadores.cascada_cuadra(ctx, cascada=cascada, cascada_pipeline=malo, archivos=archivos)
    assert r.estado == "FAIL" and "2019 (pipeline): COHORTE_DURA_PIPELINE es 71 y TOTAL menos EX01 y EX02 y EX03_PIPELINE da 70" in r.detalle


def test_dt05_las_sumas_de_periodos_y_la_identidad_del_pipeline_son_funciones_puras():
    cascada, pipe, _ = datos_cascada()
    pasos = verificadores._por_periodo(cascada)
    assert verificadores.problemas_agregados(pasos) == ([], 14)
    assert verificadores.problemas_cascada_pipeline(pipe) == ([], 3)
    solo_pipeline_sin_total = pipe.filter(pl.col("paso") != "TOTAL")
    assert verificadores.problemas_cascada_pipeline(solo_pipeline_sin_total) == ([], 0)


# ============================================================================================================================ DT06
def test_dt06_las_129_variables_del_diccionario_estan_conciliadas(ctx):
    r = verificadores.diccionario_conciliado(ctx)
    assert r.id == "DT06" and r.estado == "PASS", r.detalle
    estados = ctx.leer_tabla("diccionario_medido.csv", columnas_texto=["estado"])["estado"].to_list()
    esperado = f"129 de 129 variables del diccionario conciliadas: {estados.count('COINCIDE')} coinciden, {estados.count('DISCREPANCIA')} discrepan y {estados.count('NO EVALUABLE')} no se evalúan"
    assert r.detalle == esperado


def test_dt06_cuenta_a_mano_y_falla_con_128_filas_o_un_estado_ajeno(ctx):
    diccionario = pl.DataFrame({"Variable": ["A", "B", "C"]})
    medido = pl.DataFrame({"columna": ["A", "B", "C"], "estado": ["COINCIDE", "DISCREPANCIA", "NO EVALUABLE"]})
    r = verificadores.diccionario_conciliado(ctx, medido=medido, diccionario=diccionario)
    assert r.estado == "PASS" and r.detalle == "3 de 3 variables del diccionario conciliadas: 1 coinciden, 1 discrepan y 1 no se evalúan"
    corta = verificadores.diccionario_conciliado(ctx, medido=medido.head(2), diccionario=diccionario)
    assert corta.estado == "FAIL" and "el diccionario tiene 3 variables y la conciliación 2 filas" in corta.detalle and "faltan en la conciliación: C" in corta.detalle
    ajeno = verificadores.diccionario_conciliado(ctx, medido=medido.with_columns(pl.Series("estado", ["COINCIDE", "OK", "NO EVALUABLE"])), diccionario=diccionario)
    assert ajeno.estado == "FAIL" and "estados fuera de COINCIDE, DISCREPANCIA y NO EVALUABLE: OK" in ajeno.detalle


def test_dt06_falla_con_columnas_repetidas_o_de_mas(ctx):
    diccionario = pl.DataFrame({"Variable": ["A", "B", "C"]})
    repetida = pl.DataFrame({"columna": ["A", "B", "B"], "estado": ["COINCIDE"] * 3})
    r = verificadores.diccionario_conciliado(ctx, medido=repetida, diccionario=diccionario)
    assert r.estado == "FAIL" and "columnas repetidas: B" in r.detalle and "faltan en la conciliación: C" in r.detalle
    de_mas = pl.DataFrame({"columna": ["A", "B", "C", "Z"], "estado": ["COINCIDE"] * 4})
    r = verificadores.diccionario_conciliado(ctx, medido=de_mas, diccionario=diccionario)
    assert r.estado == "FAIL" and "sobran en la conciliación: Z" in r.detalle


# ============================================================================================================================ DT07
def fila_ledger(**cambios):
    base = {"id": "C001", "tipo": "proporcion", "unidad": "porcentaje", "estado": "CONFIRMA", "motivo": "", "valor_medido": 1.0, "tabla_origen": "t21_archivos.csv", "n_doc": None, "n_medido": None}
    return {**base, **cambios}


def ledger_a_mano(*filas):
    return pl.DataFrame(list(filas), schema={"id": pl.String, "tipo": pl.String, "unidad": pl.String, "estado": pl.String, "motivo": pl.String, "valor_medido": pl.Float64,
                                              "tabla_origen": pl.String, "n_doc": pl.Int64, "n_medido": pl.Int64})


DOC2 = pl.DataFrame({"id": ["C001", "C002"]})


def test_dt07_el_ledger_de_la_corrida_sintetica_esta_completo(ctx):
    r = verificadores.ledger_completo(ctx)
    assert r.id == "DT07" and r.estado == "PASS", r.detalle
    led = ctx.leer_tabla("ledger_cifras.csv", columnas_texto=list(ledger.COLUMNAS_TEXTO_LEDGER))
    c = {e: int((led["estado"] == e).sum()) for e in ledger.ESTADOS}
    assert r.detalle == (f"129 de 129 cifras documentadas con estado: {c['CONFIRMA']} confirman, {c['DISCREPA']} discrepan y {c['NO VERIFICABLE']} no se pueden verificar, "
                         "cada una con su motivo o su cifra medida")


def test_dt07_cuenta_a_mano_el_ledger_valido(ctx):
    led = ledger_a_mano(fila_ledger(), fila_ledger(id="C002", estado="NO VERIFICABLE", motivo="no hay métrica", valor_medido=None, tabla_origen=None))
    r = verificadores.ledger_completo(ctx, tabla_ledger=led, cifras=DOC2)
    assert r.estado == "PASS" and r.detalle.startswith("2 de 2 cifras documentadas con estado: 1 confirman, 0 discrepan y 1 no se pueden verificar")


def test_dt07_falla_con_un_estado_invalido_o_con_128_filas(ctx):
    invalido = ledger_a_mano(fila_ledger(), fila_ledger(id="C002", estado="TAL VEZ"))
    r = verificadores.ledger_completo(ctx, tabla_ledger=invalido, cifras=DOC2)
    assert r.estado == "FAIL" and "C002: estado inválido 'TAL VEZ'" in r.detalle
    corto = ledger_a_mano(fila_ledger())
    r = verificadores.ledger_completo(ctx, tabla_ledger=corto, cifras=DOC2)
    assert r.estado == "FAIL" and "hay 2 cifras documentadas y 1 filas en el ledger" in r.detalle and "faltan en el ledger: C002" in r.detalle


def test_dt07_falla_con_ids_repetidos_o_de_mas(ctx):
    repetido = ledger_a_mano(fila_ledger(), fila_ledger())
    r = verificadores.ledger_completo(ctx, tabla_ledger=repetido, cifras=DOC2)
    assert r.estado == "FAIL" and "ids repetidos: C001" in r.detalle
    de_mas = ledger_a_mano(fila_ledger(), fila_ledger(id="C002"), fila_ledger(id="C003"))
    assert "sobran en el ledger: C003" in verificadores.ledger_completo(ctx, tabla_ledger=de_mas, cifras=DOC2).detalle


def test_dt07_exige_motivo_en_discrepa_y_no_verificable(ctx):
    for estado in ("DISCREPA", "NO VERIFICABLE"):
        sin_motivo = ledger_a_mano(fila_ledger(), fila_ledger(id="C002", estado=estado, motivo="  "))
        r = verificadores.ledger_completo(ctx, tabla_ledger=sin_motivo, cifras=DOC2)
        assert r.estado == "FAIL" and f"C002: {estado} sin motivo" in r.detalle
    confirma_sin_motivo = ledger_a_mano(fila_ledger(), fila_ledger(id="C002", motivo=""))
    assert verificadores.ledger_completo(ctx, tabla_ledger=confirma_sin_motivo, cifras=DOC2).estado == "PASS"        # una cifra confirmada no necesita motivo


def test_dt07_confirma_y_discrepa_traen_el_valor_medido_y_una_tabla_de_origen_que_existe(ctx):
    for estado in ("CONFIRMA", "DISCREPA"):
        motivo = "diferencia" if estado == "DISCREPA" else ""
        sin_valor = ledger_a_mano(fila_ledger(), fila_ledger(id="C002", estado=estado, motivo=motivo, valor_medido=None))
        assert f"C002: {estado} sin valor medido" in verificadores.ledger_completo(ctx, tabla_ledger=sin_valor, cifras=DOC2).detalle
        sin_tabla = ledger_a_mano(fila_ledger(), fila_ledger(id="C002", estado=estado, motivo=motivo, tabla_origen=""))
        assert f"C002: {estado} sin tabla de origen" in verificadores.ledger_completo(ctx, tabla_ledger=sin_tabla, cifras=DOC2).detalle
        fantasma = ledger_a_mano(fila_ledger(), fila_ledger(id="C002", estado=estado, motivo=motivo, tabla_origen="no_existe.csv"))
        assert f"C002: la tabla de origen no_existe.csv no existe" in verificadores.ledger_completo(ctx, tabla_ledger=fantasma, cifras=DOC2).detalle
    # la aritmética sobre cifras documentadas cita la tabla de configuración
    documental = ledger_a_mano(fila_ledger(), fila_ledger(id="C002", estado="DISCREPA", motivo="razón", tabla_origen="cifras_documentadas.csv"))
    assert verificadores.ledger_completo(ctx, tabla_ledger=documental, cifras=DOC2).estado == "PASS"


def test_dt07_los_valores_protegidos_no_se_exigen(ctx):
    # un conteo de 1 a 9 casos queda sin valores en la tabla publicada: sin n no se pide el valor medido
    protegido = ledger_a_mano(fila_ledger(), fila_ledger(id="C002", tipo="absoluto", unidad="conteo", valor_medido=None, tabla_origen=None, n_doc=None, n_medido=None))
    assert verificadores.ledger_completo(ctx, tabla_ledger=protegido, cifras=DOC2).estado == "PASS"
    no_protegido = ledger_a_mano(fila_ledger(), fila_ledger(id="C002", tipo="absoluto", unidad="conteo", valor_medido=None, tabla_origen=None, n_doc=50, n_medido=60))
    assert verificadores.ledger_completo(ctx, tabla_ledger=no_protegido, cifras=DOC2).estado == "FAIL"


# ============================================================================================================================ DT08
def test_dt08_las_33_figuras_del_plan_estan_indexadas_incrustadas_y_con_las_huellas_del_indice(ctx):
    r = verificadores.figuras_indexadas(ctx)
    assert r.id == "DT08" and r.estado == "PASS", r.detalle
    assert r.detalle.startswith("33 de 33 figuras del plan indexadas con su PNG y su CSV de respaldo (huellas iguales) e incrustadas")


def test_dt08_falla_si_se_borra_un_png(corrida_t13, tmp_path):
    c = clonar(corrida_t13.ctx, tmp_path)
    (c.dir_figuras / "F02_01.png").unlink()
    r = verificadores.figuras_indexadas(c)
    assert r.estado == "FAIL" and "F02_01: no existe el PNG F02_01.png" in r.detalle


def test_dt08_falla_si_un_png_o_un_csv_no_coinciden_con_su_huella(corrida_t13, tmp_path):
    c = clonar(corrida_t13.ctx, tmp_path)
    with open(c.dir_figuras / "F03_01.png", "ab") as f:
        f.write(b"x")
    r = verificadores.figuras_indexadas(c)
    assert r.estado == "FAIL" and "F03_01: el PNG no coincide con su huella del índice" in r.detalle
    with open(c.dir_tablas / "t21_archivos.csv", "a", encoding="utf8") as f:
        f.write("\n")
    r = verificadores.figuras_indexadas(c)
    assert "F02_01: el CSV t21_archivos.csv cambió después de dibujar la figura" in r.detalle


def test_dt08_falla_si_el_indice_pierde_una_figura_del_plan_o_trae_una_ajena(ctx):
    idx = config.leer_csv_texto(ctx.dir_tablas / "figuras_indice.csv")
    r = verificadores.figuras_indexadas(ctx, indice=idx.filter(pl.col("id_figura") != "F05_01"))
    assert r.estado == "FAIL" and "faltan en el índice: F05_01" in r.detalle
    ajena = pl.concat([idx, idx.filter(pl.col("id_figura") == "F05_01").with_columns(pl.lit("F99_01").alias("id_figura"))])
    r = verificadores.figuras_indexadas(ctx, indice=ajena)
    assert r.estado == "FAIL" and "figuras ajenas al plan: F99_01" in r.detalle
    repetida = pl.concat([idx, idx.filter(pl.col("id_figura") == "F05_01")])
    assert "figuras repetidas en el índice: F05_01" in verificadores.figuras_indexadas(ctx, indice=repetida).detalle


def test_dt08_el_csv_del_indice_debe_ser_el_del_plan(ctx):
    idx = config.leer_csv_texto(ctx.dir_tablas / "figuras_indice.csv")
    otro = idx.with_columns(pl.when(pl.col("id_figura") == "F05_01").then(pl.lit("t21_archivos.csv")).otherwise(pl.col("csv")).alias("csv"))
    r = verificadores.figuras_indexadas(ctx, indice=otro)
    plan = config.cargar_plan_figuras().filter(pl.col("id_figura") == "F05_01")["tabla_csv"][0]
    assert r.estado == "FAIL" and f"F05_01: el CSV del índice (t21_archivos.csv) no es el del plan ({plan})" in r.detalle


def test_dt08_falla_si_una_figura_no_esta_incrustada_en_ningun_informe(ctx, corrida_t13, tmp_path):
    r = verificadores.figuras_indexadas(ctx, sin_incrustar=["F07_01", "F12_02"])
    assert r.estado == "FAIL" and "sin incrustar en ningún informe: F07_01, F12_02" in r.detalle
    c = clonar(corrida_t13.ctx, tmp_path)
    (c.dir_informes / "t24_informe.md").unlink()                       # las figuras de calidad quedan sin informe
    r = verificadores.figuras_indexadas(c)
    assert r.estado == "FAIL" and "sin incrustar en ningún informe: F05_01, F05_02, F05_03, F06_01, F06_02" in r.detalle


# ============================================================================================================================ DT09
def test_dt09_la_matriz_cubre_las_199_candidatas(ctx):
    r = verificadores.matriz_cubre_candidatas(ctx)
    assert r.id == "DT09" and r.estado == "PASS", r.detalle
    n_rank = ctx.leer_tabla("t23_ranking_variables.csv", columnas_texto=["variable"])["variable"].n_unique()
    assert r.detalle == f"199 variables en la matriz, igual al universo documentado y a la tabla de disponibilidad; las {n_rank} variables del ranking están en la matriz"
    assert decision.cargar_universo().height == 199


def test_dt09_falla_si_a_la_matriz_le_falta_una_variable_o_le_sobra_otra(ctx):
    uni = pl.DataFrame({"variable": ["A", "B", "C"]})
    mat = pl.DataFrame({"variable": ["A", "B", "C"]})
    ok = verificadores.matriz_cubre_candidatas(ctx, matriz=mat, disponibilidad=mat, universo=uni, ranking=pl.DataFrame({"variable": ["A", "B", "A"]}))
    assert ok.estado == "PASS" and ok.detalle.startswith("3 variables en la matriz") and "las 2 variables del ranking" in ok.detalle
    falta = verificadores.matriz_cubre_candidatas(ctx, matriz=mat.head(2), disponibilidad=mat, universo=uni, ranking=pl.DataFrame({"variable": ["A"]}))
    assert falta.estado == "FAIL" and "faltan en la matriz: C" in falta.detalle
    sobra = verificadores.matriz_cubre_candidatas(ctx, matriz=pl.DataFrame({"variable": ["A", "B", "C", "Z"]}), disponibilidad=mat, universo=uni, ranking=pl.DataFrame({"variable": ["A"]}))
    assert sobra.estado == "FAIL" and "sobran en la matriz: Z" in sobra.detalle


def test_dt09_falla_si_la_disponibilidad_no_es_el_universo_o_el_ranking_se_sale_de_la_matriz(ctx):
    uni = pl.DataFrame({"variable": ["A", "B", "C"]})
    mat = pl.DataFrame({"variable": ["A", "B", "C"]})
    disp = verificadores.matriz_cubre_candidatas(ctx, matriz=mat, disponibilidad=mat.head(2), universo=uni, ranking=pl.DataFrame({"variable": ["A"]}))
    assert disp.estado == "FAIL" and "faltan en la disponibilidad: C" in disp.detalle
    fuera = verificadores.matriz_cubre_candidatas(ctx, matriz=mat, disponibilidad=mat, universo=uni, ranking=pl.DataFrame({"variable": ["A", "Z"]}))
    assert fuera.estado == "FAIL" and "están en el ranking y no en la matriz: Z" in fuera.detalle
    repetida = verificadores.matriz_cubre_candidatas(ctx, matriz=pl.DataFrame({"variable": ["A", "B", "C", "C"]}), disponibilidad=mat, universo=uni, ranking=pl.DataFrame({"variable": ["A"]}))
    assert repetida.estado == "FAIL" and "variables repetidas en la matriz: C" in repetida.detalle


# ============================================================================================================================ DT10
SALIDA_VERDE = "........ [100%]\n1587 passed in 131.31s (0:02:11)\n"


@pytest.fixture()
def fuera_de_la_guarda(monkeypatch):
    """Quita la variable de recursión para probar el lanzador inyectado (estas pruebas corren dentro de pytest, no dentro del subproceso de DT10)."""
    monkeypatch.delenv(verificadores.VAR_DENTRO_DE_PYTEST, raising=False)


def test_dt10_pasa_si_el_lanzador_termina_en_cero_y_cuenta_las_pruebas(ctx, fuera_de_la_guarda):
    r = verificadores.pytest_verde(ctx, lanzador=lambda: (0, SALIDA_VERDE))
    assert r.id == "DT10" and r.estado == "PASS" and r.detalle == "1.587 pruebas pasaron"
    con_advertencias = verificadores.pytest_verde(ctx, lanzador=lambda: (0, "12 passed, 3 warnings in 1.2s\n"))
    assert con_advertencias.estado == "PASS" and con_advertencias.detalle == "12 pruebas pasaron"


def test_dt10_falla_con_un_lanzador_de_codigo_uno(ctx, fuera_de_la_guarda):
    r = verificadores.pytest_verde(ctx, lanzador=lambda: (1, "FAILED tests/test_x.py::test_a\n2 failed, 1585 passed in 130.1s\n"))
    assert r.estado == "FAIL" and r.detalle == "pytest terminó con código 1; pruebas con fallo o error: 2"
    con_errores = verificadores.pytest_verde(ctx, lanzador=lambda: (2, "1 error, 5 passed in 1.0s\n"))
    assert con_errores.estado == "FAIL" and "pruebas con fallo o error: 1" in con_errores.detalle
    assert verificadores.pytest_verde(ctx, lanzador=lambda: (2, "interrumpido")).detalle == "pytest terminó con código 2"


def test_dt10_un_codigo_cero_sin_resumen_o_con_fallos_no_es_un_pase(ctx, fuera_de_la_guarda):
    sin_resumen = verificadores.pytest_verde(ctx, lanzador=lambda: (0, "no tests ran"))
    assert sin_resumen.estado == "FAIL" and "no se pudo leer el resumen" in sin_resumen.detalle
    contradictorio = verificadores.pytest_verde(ctx, lanzador=lambda: (0, "1 failed, 5 passed in 1.0s"))
    assert contradictorio.estado == "FAIL"
    # un resumen que dice "pasaron" no basta si el proceso terminó con un código distinto de cero (por ejemplo, interrumpido)
    for codigo in (1, 2, 5):
        r = verificadores.pytest_verde(ctx, lanzador=lambda c=codigo: (c, "5 passed in 1.0s\n"))
        assert r.estado == "FAIL" and r.detalle == f"pytest terminó con código {codigo}", codigo


def test_dt10_queda_pendiente_con_la_bandera_o_dentro_de_pytest_y_no_lanza_nada(ctx, monkeypatch):
    def no_debe_llamarse():
        raise AssertionError("el lanzador no debe llamarse")

    monkeypatch.delenv(verificadores.VAR_DENTRO_DE_PYTEST, raising=False)
    r = verificadores.pytest_verde(ctx, omitir=True, lanzador=no_debe_llamarse)
    assert r.estado == "PENDIENTE" and r.detalle == "no se ejecutó: la corrida usó --sin-pytest"
    monkeypatch.setenv(verificadores.VAR_DENTRO_DE_PYTEST, "1")
    r = verificadores.pytest_verde(ctx, lanzador=no_debe_llamarse)
    assert r.estado == "PENDIENTE" and "ya está dentro de una ejecución de pytest" in r.detalle
    # sin lanzador inyectado y dentro de pytest (esta misma prueba) tampoco se anida otra suite
    monkeypatch.delenv(verificadores.VAR_DENTRO_DE_PYTEST, raising=False)
    r = verificadores.pytest_verde(ctx)
    assert r.estado == "PENDIENTE" and "dentro de pytest" in r.detalle


def test_el_lanzador_de_pytest_corre_la_suite_en_un_subproceso_con_la_guarda(monkeypatch):
    llamadas = []

    def falso(cmd, **kw):
        llamadas.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout="3 passed in 0.1s\n", stderr="aviso\n")

    monkeypatch.setattr(subprocess, "run", falso)
    codigo, salida = verificadores.lanzar_pytest()
    (cmd, kw), = llamadas
    assert cmd == [sys.executable, "-B", "-m", "pytest", "tests", "-o", "addopts=-p no:cacheprovider", "-q"]
    assert kw["cwd"] == rutas.RAIZ_EDA and kw["env"]["PYTHONDONTWRITEBYTECODE"] == "1" and kw["env"][verificadores.VAR_DENTRO_DE_PYTEST] == "1"
    assert kw["capture_output"] is True and kw["text"] is True and kw["timeout"] == verificadores.TIMEOUT_SUBPROCESO_S
    assert (codigo, salida) == (0, "3 passed in 0.1s\naviso\n")


def test_el_lanzador_real_de_pytest_corre_un_subproceso_con_la_guarda_y_dt10_lee_su_resumen(ctx, tmp_path, fuera_de_la_guarda):
    """La ruta real de DT10 sin correr toda la suite: un archivo de prueba mínimo en un subproceso de pytest."""
    ok = tmp_path / "test_minimo_ok.py"
    ok.write_text(
        "import os\n\ndef test_la_guarda_y_el_entorno_llegan_al_subproceso():\n    assert os.environ['FASE2_EDA_DENTRO_DE_PYTEST'] == '1' and os.environ['PYTHONDONTWRITEBYTECODE'] == '1'\n\n"
        "def test_otra():\n    assert True\n", encoding="utf8")
    codigo, salida = verificadores.lanzar_pytest([str(ok)])
    assert codigo == 0 and "2 passed" in salida, salida
    r = verificadores.pytest_verde(ctx, lanzador=lambda: verificadores.lanzar_pytest([str(ok)]))
    assert r.estado == "PASS" and r.detalle == "2 pruebas pasaron"
    malo = tmp_path / "test_minimo_malo.py"
    malo.write_text("def test_falla():\n    assert False\n\ndef test_pasa():\n    assert True\n", encoding="utf8")
    r = verificadores.pytest_verde(ctx, lanzador=lambda: verificadores.lanzar_pytest([str(malo)]))
    assert r.estado == "FAIL" and r.detalle == "pytest terminó con código 1; pruebas con fallo o error: 1"


# ============================================================================================================================ DT11
IGUALES = {"tablas/a.csv": b"a,b\n1,2\n", "figuras/F01.png": b"\x89PNG\r\n", "manifiesto.json": b"{}"}


def test_las_huellas_de_un_arbol_son_rutas_relativas_con_su_sha256(tmp_path):
    import hashlib

    (tmp_path / "sub").mkdir()
    (tmp_path / "x.txt").write_bytes(b"uno")
    (tmp_path / "sub" / "y.txt").write_bytes(b"dos")
    (tmp_path / "vacia").mkdir()
    h = verificadores.huellas_arbol(tmp_path)
    assert list(h) == ["sub/y.txt", "x.txt"]
    assert h["x.txt"] == hashlib.sha256(b"uno").hexdigest() and h["sub/y.txt"] == hashlib.sha256(b"dos").hexdigest()
    assert list(verificadores.huellas_arbol(tmp_path, excluir=("x.txt",))) == ["sub/y.txt"]
    assert verificadores.huellas_arbol(tmp_path / "vacia") == {}


def test_dt11_dos_corridas_con_los_mismos_hashes_pasan_y_las_carpetas_temporales_se_borran(ctx):
    lanzador = LanzadorCorrida([IGUALES, IGUALES])
    r = verificadores.corrida_determinista(ctx, lanzador=lanzador)
    assert r.id == "DT11" and r.estado == "PASS"
    assert r.detalle == "dos corridas completas dieron los mismos 3 archivos con los mismos SHA-256"
    assert len(lanzador.bases) == 2 and lanzador.bases[0] != lanzador.bases[1] and not any(b.exists() for b in lanzador.bases)


def test_dt11_falla_si_una_figura_difiere_en_un_byte(ctx):
    segunda = {**IGUALES, "figuras/F01.png": b"\x89PNG\r\x00"}
    lanzador = LanzadorCorrida([IGUALES, segunda])
    r = verificadores.corrida_determinista(ctx, lanzador=lanzador)
    assert r.estado == "FAIL" and r.detalle == "1 archivos no coinciden entre dos corridas: figuras/F01.png difiere"
    assert not any(b.exists() for b in lanzador.bases)


def test_dt11_falla_si_un_archivo_solo_esta_en_una_de_las_corridas(ctx):
    r = verificadores.corrida_determinista(ctx, lanzador=LanzadorCorrida([IGUALES, {**IGUALES, "tablas/extra.csv": b"x"}]))
    assert r.estado == "FAIL" and "tablas/extra.csv solo está en la segunda corrida" in r.detalle
    sin_manifiesto = {k: v for k, v in IGUALES.items() if k != "manifiesto.json"}
    r = verificadores.corrida_determinista(ctx, lanzador=LanzadorCorrida([IGUALES, sin_manifiesto]))
    assert r.estado == "FAIL" and "manifiesto.json solo está en la primera corrida" in r.detalle


def test_dt11_el_codigo_de_salida_de_cada_corrida_importa_pero_uno_no_es_un_fallo_de_la_corrida(ctx):
    lanzador = LanzadorCorrida([IGUALES, IGUALES], codigos=(2, 0))
    r = verificadores.corrida_determinista(ctx, lanzador=lanzador)
    assert r.estado == "FAIL" and r.detalle == "la corrida 1 terminó con código 2" and len(lanzador.bases) == 1 and not lanzador.bases[0].exists()
    r = verificadores.corrida_determinista(ctx, lanzador=LanzadorCorrida([IGUALES, IGUALES], codigos=(0, 3)))
    assert r.estado == "FAIL" and r.detalle == "la corrida 2 terminó con código 3"
    # un criterio de terminado que falla dentro de la corrida (código 1) no impide comparar sus huellas
    assert verificadores.corrida_determinista(ctx, lanzador=LanzadorCorrida([IGUALES, IGUALES], codigos=(1, 1))).estado == "PASS"


def test_dt11_las_carpetas_temporales_se_borran_aunque_el_lanzador_falle(ctx):
    lanzador = LanzadorCorrida([IGUALES, IGUALES], falla_en=1)
    with pytest.raises(RuntimeError, match="el lanzador falló"):
        verificadores.corrida_determinista(ctx, lanzador=lanzador)
    assert len(lanzador.bases) == 2 and not any(b.exists() for b in lanzador.bases)


def test_dt11_sin_archivos_en_la_salida_es_un_fallo(ctx):
    r = verificadores.corrida_determinista(ctx, lanzador=LanzadorCorrida([{}, {}]))
    assert r.estado == "FAIL" and r.detalle == "las corridas no produjeron ningún archivo"


def test_dt11_con_la_bandera_queda_pendiente_y_sin_lanzador_es_un_error_de_uso(ctx):
    lanzador = LanzadorCorrida([IGUALES, IGUALES])
    r = verificadores.corrida_determinista(ctx, omitir=True, lanzador=lanzador)
    assert r.estado == "PENDIENTE" and r.detalle == "no se ejecutó: la corrida usó --sin-determinismo" and lanzador.bases == []
    with pytest.raises(ValueError, match="necesita un lanzador"):
        verificadores.corrida_determinista(ctx)


def test_dt11_no_excluye_ningun_archivo_del_cotejo_por_defecto(ctx):
    assert verificadores.EXCLUIR_DEL_COTEJO == ()
    distinta = {**IGUALES, "manifiesto.json": b'{"otra": 1}'}
    assert verificadores.corrida_determinista(ctx, lanzador=LanzadorCorrida([IGUALES, distinta])).estado == "FAIL"
    r = verificadores.corrida_determinista(ctx, lanzador=LanzadorCorrida([IGUALES, distinta]), excluir=("manifiesto.json",))
    assert r.estado == "PASS" and "los mismos 2 archivos" in r.detalle


# ============================================================================================================================ DT12
def test_dt12_con_extractos_queda_pendiente(ctx):
    r = verificadores.completo_corrido(ctx)
    assert r.id == "DT12" and r.estado == "PENDIENTE" and r.detalle == "solo se corrió sobre extractos: falta la corrida con los datos completos"


def test_dt12_una_tabla_o_una_metrica_extracto_en_una_corrida_completo_es_un_fallo(corrida_t13, tmp_path):
    c = clonar(corrida_t13.ctx, tmp_path, etiqueta="completo")
    r = verificadores.completo_corrido(c)
    n_tablas, n_metricas = c.indice_tablas().height, c.metricas().height
    assert r.estado == "FAIL" and f"{n_tablas} tablas sin la etiqueta completo" in r.detalle and f"{n_metricas} métricas sin la etiqueta completo" in r.detalle
    etiquetar_completo(c)
    assert verificadores.completo_corrido(c).estado == "PASS"
    clave = c._metricas[0]["clave"]
    c._metricas[0]["nivel"] = "extracto"
    r = verificadores.completo_corrido(c)
    assert r.estado == "FAIL" and f"1 métricas sin la etiqueta completo: {clave}" in r.detalle and "tablas sin la etiqueta" not in r.detalle
    c._metricas[0]["nivel"] = "completo"
    c._tablas["t21_archivos.csv"]["fuente"] = "extracto"
    r = verificadores.completo_corrido(c)
    assert r.estado == "FAIL" and "1 tablas sin la etiqueta completo: t21_archivos.csv" in r.detalle


def test_dt12_con_los_datos_completos_pasa_si_se_leyeron_los_seis_anios(corrida_t13, tmp_path):
    c = clonar(corrida_t13.ctx, tmp_path, etiqueta="completo")
    etiquetar_completo(c)
    r = verificadores.completo_corrido(c)
    n_tablas, n_metricas = fmt_entero(c.indice_tablas().height), fmt_entero(c.metricas().height)
    assert r.estado == "PASS" and r.detalle == f"{n_tablas} tablas y {n_metricas} métricas con la etiqueta completo; se leyeron los 6 años"
    faltan = verificadores.completo_corrido(c, anios_esperados=[2019, 2020, 2021, 2022, 2023, 2024, 2025])
    assert faltan.estado == "FAIL" and "no se leyeron los años 2025" in faltan.detalle
    c.cortafuegos._entradas = [e for e in c.cortafuegos._entradas if e["anio"] != 2022]
    assert "no se leyeron los años 2022" in verificadores.completo_corrido(c).detalle


# ============================================================================================================================ el conjunto
def banderas_de(corrida_t13, **cambios):
    base = dict(sin_pytest=True, sin_determinismo=True, informes_esperados=corrida_t13.previos, cascada=corrida_t13.cascada, cascada_pipeline=corrida_t13.cascada_pipeline)
    return Banderas(**{**base, **cambios})


def test_verificar_todo_entrega_doce_filas_en_el_orden_del_csv(ctx, corrida_t13):
    resultados = verificadores.verificar_todo(ctx, banderas_de(corrida_t13))
    assert [r.id for r in resultados] == [f"DT{i:02d}" for i in range(1, 13)]
    assert [r.estado for r in resultados] == ["PASS"] * 9 + ["PENDIENTE"] * 3
    dt = config.cargar_definicion_terminado()
    assert [r.criterio for r in resultados] == dt["criterio"].to_list()
    for r in resultados:                                                       # el detalle es reproducible: ni rutas absolutas ni segundos
        assert str(ctx.dir_salida) not in r.detalle and "/tmp" not in r.detalle and " s " not in r.detalle and r.evidencia


def test_verificar_todo_puede_correr_solo_algunos_criterios(ctx, corrida_t13):
    resultados = verificadores.verificar_todo(ctx, banderas_de(corrida_t13), ids=("DT08", "DT01"))
    assert [r.id for r in resultados] == ["DT01", "DT08"]                      # en el orden del CSV, no en el pedido
    assert verificadores.verificar_todo(ctx, banderas_de(corrida_t13), ids=()) == []


def test_un_verificador_que_lanza_una_excepcion_queda_en_fail_sin_rutas_absolutas(ctx, corrida_t13, monkeypatch):
    def explota(c, **kw):
        raise RuntimeError(f"no pude leer {c.dir_salida}/tablas/x.csv\ncon salto de línea")

    monkeypatch.setattr(verificadores, "ledger_completo", explota)
    resultados = verificadores.verificar_todo(ctx, banderas_de(corrida_t13))
    dt07 = next(r for r in resultados if r.id == "DT07")
    assert dt07.estado == "FAIL" and dt07.detalle == "el verificador falló: RuntimeError: no pude leer <ruta>/tablas/x.csv con salto de línea"
    assert [r.estado for r in resultados if r.id != "DT07"] == ["PASS"] * 8 + ["PENDIENTE"] * 3         # los demás siguen


def test_un_verificador_que_no_existe_en_el_csv_es_un_fallo(ctx, corrida_t13, monkeypatch):
    dt = config.cargar_definicion_terminado()
    extra = pl.concat([dt, dt.head(1).with_columns(pl.lit("DT13").alias("id"), pl.lit("no_existe").alias("verificador"))])
    monkeypatch.setattr(config, "cargar_definicion_terminado", lambda *a, **k: extra)
    r = verificadores.verificar_todo(ctx, banderas_de(corrida_t13), ids=("DT13",))
    assert len(r) == 1 and r[0].estado == "FAIL" and r[0].detalle == "no existe el verificador no_existe"


def test_sin_banderas_la_suite_no_se_anida_y_la_corrida_determinista_pide_su_lanzador(ctx, corrida_t13):
    resultados = verificadores.verificar_todo(ctx, Banderas(informes_esperados=corrida_t13.previos), ids=("DT10", "DT11"))
    assert resultados[0].estado == "PENDIENTE" and ("dentro de pytest" in resultados[0].detalle or "ejecución de pytest" in resultados[0].detalle)
    assert resultados[1].estado == "FAIL" and resultados[1].detalle.startswith("el verificador falló: ValueError: corrida_determinista necesita un lanzador")


def test_la_tabla_de_verificacion_tiene_las_cinco_columnas_y_los_conteos_salen_de_los_estados():
    rs = [ResultadoDT("DT01", "c1", "PASS", "d1", "e1"), ResultadoDT("DT02", "c2", "FAIL", "d2"), ResultadoDT("DT03", "c3", "PENDIENTE", "d3"), ResultadoDT("DT04", "c4", "PASS", "d4")]
    t = verificadores.tabla_verificacion(rs)
    assert t.columns == ["id", "criterio", "estado", "detalle", "evidencia"] and t.height == 4 and t.schema == pl.Schema(verificadores.ESQUEMA_VERIFICACION)
    assert t.row(1) == ("DT02", "c2", "FAIL", "d2", "")
    assert verificadores.resumen_verificacion(rs) == {"n_criterios": 4, "n_pass": 2, "n_fail": 1, "n_pendiente": 1}
    assert verificadores.resumen_verificacion([]) == {"n_criterios": 0, "n_pass": 0, "n_fail": 0, "n_pendiente": 0}


def test_escribir_la_verificacion_publica_las_dos_tablas_de_la_tarea_13(corrida_t13, tmp_path):
    c = clonar(corrida_t13.ctx, tmp_path)
    antes = c.indice_tablas().height
    resultados = verificadores.verificar_todo(c, banderas_de(corrida_t13))
    tabla = verificadores.escribir_verificacion(c, resultados)
    assert tabla.height == 12
    v = c.leer_tabla("verificacion_dt.csv", columnas_texto=list(verificadores.COLUMNAS_VERIFICACION))
    assert v["id"].to_list() == [f"DT{i:02d}" for i in range(1, 13)] and v["estado"].to_list() == ["PASS"] * 9 + ["PENDIENTE"] * 3
    resumen = c.leer_tabla("t26_resumen_cierre.csv").row(0, named=True)
    assert resumen == {"n_criterios": 12, "n_pass": 9, "n_fail": 0, "n_pendiente": 3, "n_tablas": antes, "n_figuras": 33, "n_informes": 7}
    indice = c.indice_tablas()
    assert set(indice.filter(pl.col("tarea") == "T13")["nombre"].to_list()) == {"verificacion_dt.csv", "t26_resumen_cierre.csv"} and indice.height == antes + 2
    # al repetirla las dos tablas propias no se cuentan: el resumen no cambia
    verificadores.escribir_verificacion(c, resultados)
    assert c.leer_tabla("t26_resumen_cierre.csv").row(0, named=True) == resumen


# FIN-test_t13_verificadores
