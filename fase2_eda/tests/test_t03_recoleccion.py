"""Tarea 3: recolección (CRISP-DM 2.1) con datos sintéticos."""

from __future__ import annotations

import polars as pl

from eda import recoleccion
from eda.contexto import Contexto
from eda.cortafuegos import Cortafuegos
from eda.lector import Fuente, Lector
from sintetico import escribir_archivo, generar_anio


def test_archivos_129_columnas_cuadratura_y_bom(ctx):
    t = recoleccion.tabla_archivos(ctx.lector)
    assert t["anio"].to_list() == [2019, 2020, 2021, 2022, 2023, 2024]
    assert set(t["n_columnas_encabezado"].to_list()) == {129}
    assert set(t["filas_leidas"].to_list()) == {600}
    assert set(t["suma_cuadra"].to_list()) == {"si"}
    assert t.filter(pl.col("bom") == "si")["anio"].to_list() == [2021, 2022, 2023]
    assert set(t["fin_de_linea"].to_list()) == {"CRLF"}
    assert t["sha256"].str.len_chars().unique().to_list() == [64]
    assert set(t["codificacion"].to_list()) == {"UTF-8 válido"}


def test_id_beneficiario_solo_en_2024(ctx):
    es = recoleccion.deriva_esquema(ctx.lector)
    assert es.height == 129 * 6
    dif = es.filter(pl.col("coincide") == "no")
    assert dif.height == 1
    fila = dif.row(0, named=True)
    assert (fila["posicion"], fila["anio"], fila["nombre_encabezado"]) == (2, 2024, "ID_BENEFICIARIO")


def test_fecha_dmy_solo_en_2023_y_nacimiento_siempre_iso(ctx):
    f = recoleccion.deriva_formato(ctx.lector)
    ing = f.filter(pl.col("columna") == "FECHA_INGRESO")
    dmy = ing.filter(pl.col("clase") == "DD-MM-AAAA")
    assert dmy.filter(pl.col("n_clase") > 0)["anio"].to_list() == [2023]
    iso = ing.filter(pl.col("clase") == "ISO")
    assert iso.filter(pl.col("n_clase") > 0)["anio"].to_list() == [2019, 2020, 2021, 2022, 2024]
    nac = f.filter((pl.col("columna") == "FECHA_NACIMIENTO") & (pl.col("clase") == "ISO"))
    assert (nac["n_clase"] == nac["n_valores"]).all()
    peso = f.filter((pl.col("columna") == "IR_29301_PESO") & (pl.col("clase") == "COMA_DECIMAL"))
    assert (peso["n_clase"] == peso["n_valores"]).all()
    # las fechas de alta de 2024 llegan enmascaradas pero conservan la forma ISO
    alta24 = f.filter((pl.col("columna") == "FECHAALTA") & (pl.col("clase") == "ISO") & (pl.col("anio") == 2024))
    assert alta24["n_clase"][0] == alta24["n_valores"][0]
    # sin lecturas de resultado en claro sobre 2024
    assert ctx.cortafuegos.violaciones() == 0
    assert ctx.cortafuegos.n_lecturas_en_claro_expuestas() == 0


def test_verificaciones_marcan_el_hallazgo_del_id_en_2024(ctx):
    a = recoleccion.tabla_archivos(ctx.lector)
    es = recoleccion.deriva_esquema(ctx.lector).filter(pl.col("coincide") == "no")
    f = recoleccion.deriva_formato(ctx.lector)
    v = recoleccion.verificaciones(a, es, f, "extracto", filas_extracto_esperadas=600).to_dicts()
    por_id = {r["id"]: r for r in v}
    assert por_id["V01"]["resultado"] == "COINCIDE"
    assert por_id["V02"]["resultado"] == "COINCIDE"
    assert por_id["V03"]["resultado"] == "COINCIDE"
    # la Fase 2 dice 2019; el encabezado real dice 2024
    assert por_id["V04"]["observado"] == "2024" and por_id["V04"]["resultado"] == "DISCREPA"
    assert por_id["V05"]["resultado"] == "COINCIDE"
    assert por_id["V07"]["resultado"] == "COINCIDE"
    v2 = recoleccion.verificaciones(a, es, f, "extracto", filas_extracto_esperadas=599).to_dicts()
    assert {r["id"]: r for r in v2}["V03"]["resultado"] == "DISCREPA"
    # en modo completo no existe la verificación de filas del extracto
    v3 = recoleccion.verificaciones(a, es, f, "completo", filas_extracto_esperadas=600)
    assert "V03" not in v3["id"].to_list()


def test_procedencia_detecta_presencia_y_ausencia_del_oficio(tmp_path, ctx):
    a = recoleccion.tabla_archivos(ctx.lector)
    vacia = recoleccion.procedencia(tmp_path, a)
    est = {r["item"]: r["estado"] for r in vacia.iter_rows(named=True)}
    assert est["Oficio FONASA 23364/2026 (respuesta a la solicitud AO004T0010901)"] == "AUSENTE"
    assert est["Respuesta de FONASA por los microdatos de egresos 2019 a 2024"] == "AUSENTE"
    assert est["Condiciones de uso y de publicación de resultados por hospital"] == "NO DOCUMENTADAS"
    (tmp_path / "Oficio_Respuesta_N\ufffd23364_T10901.pdf").write_bytes(b"%PDF-1.4 prueba")
    con = recoleccion.procedencia(tmp_path, a)
    est2 = {r["item"]: r["estado"] for r in con.iter_rows(named=True)}
    assert est2["Oficio FONASA 23364/2026 (respuesta a la solicitud AO004T0010901)"] == "PRESENTE"
    # el oficio presente no cuenta como respuesta por los microdatos
    assert est2["Respuesta de FONASA por los microdatos de egresos 2019 a 2024"] == "AUSENTE"
    assert (con["item"].str.starts_with("Archivo de egresos")).sum() == 6


def test_bitacora_registra_lineas_rechazadas_y_formatos(tmp_path):
    df = generar_anio(2022, n=40, semilla=2)
    ruta = escribir_archivo(df, tmp_path / "GRD_PUBLICO_2022.txt", lineas_extra=["a|b|c"])
    lector = Lector(Fuente("extracto", tmp_path, {2022: ruta}), Cortafuegos(2024))
    a = recoleccion.tabla_archivos(lector)
    es = recoleccion.deriva_esquema(lector).filter(pl.col("coincide") == "no")
    f = recoleccion.deriva_formato(lector)
    b = recoleccion.bitacora_problemas(a, es, f)
    assert a["filas_rechazadas"][0] == 1 and a["suma_cuadra"][0] == "si"
    fila = b.filter(pl.col("problema").str.starts_with("Líneas con número de campos"))
    assert fila.height == 1 and fila["n"][0] == 1 and fila["severidad"][0] == "alta"


def test_ejecutar_genera_tablas_figuras_e_informe(tmp_path, fuente_sintetica):
    ctx = Contexto.crear("extracto", fuente=fuente_sintetica, dir_salida_base=tmp_path / "salidas")
    res = recoleccion.ejecutar(ctx, raiz_proyecto=tmp_path)
    for nombre in (
        "t21_archivos.csv", "t21_deriva_esquema.csv", "t21_esquema_diferencias.csv", "t21_deriva_formato.csv",
        "t21_problemas_carga.csv", "t21_verificaciones.csv", "t21_procedencia.csv", "t21_resumen.csv",
    ):
        assert (ctx.dir_tablas / nombre).is_file(), nombre
    assert sorted(ctx.figuras.ids()) == ["F02_01", "F03_01"]
    for fid in ctx.figuras.ids():
        assert (ctx.dir_figuras / f"{fid}.png").stat().st_size > 5000
    md = res["informe"].read_text(encoding="utf8")
    assert "Informe de recolección inicial de los datos" in md
    assert "![" in md and "F03_01" in md
    assert "Fuente de los datos: extracto" in md
    # el informe cita los valores desde los CSV
    cifras = ctx.cifras_informe()
    assert cifras.filter(pl.col("marcador") == "n_archivos")["texto"][0] == "6"
    assert cifras.filter(pl.col("marcador") == "n_columnas")["texto"][0] == "129"
    metricas = ctx.metricas()
    assert metricas.filter(pl.col("clave") == "estructura.primer_anio_id_beneficiario")["valor"][0] == 2024
    assert metricas.filter(pl.col("clave") == "estructura.primer_anio_fecha_dmy")["valor"][0] == 2023
    # años sin separador de miles
    assert "2.019" not in md
