"""Tarea 2: lector seguro, supresión, cortafuegos de 2024, manifiesto y gráficos deterministas."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import polars as pl
import pytest

from eda import esquema, graficos, manifiesto
from eda.contexto import Contexto
from eda.cortafuegos import AccesoBloqueado, Cortafuegos
from eda.lector import ArchivoNoEncontrado, EsquemaInvalido, Lector, resolver_fuente
from eda.supresion import contar_pequenas, suprimir
from sintetico import escribir_archivo, generar_anio

COLS_RES = ["TIPOALTA", "FECHAALTA", "SERVICIOALTA", "IR_29301_MORTALIDAD", "IR_29301_SEVERIDAD", "IR_29301_PESO"]


def _cf(reglas=()):
    return Cortafuegos(2024, ("estructura", "calidad"), COLS_RES, reglas_autorizadas=reglas)


# ------------------------------------------------------------------------ lector
def test_esquema_tiene_129_columnas_unicas():
    assert esquema.N_COLUMNAS == 129
    assert len(set(esquema.COLUMNAS)) == 129
    assert esquema.COLUMNAS[1] == "CIP_ENCRIPTADO"
    assert esquema.COLUMNAS[-1] == "HOSPPROCEDENCIA"
    assert len(esquema.DIAG) == 35 and len(esquema.PROC) == 30


def test_bom_y_crlf_no_contaminan_las_columnas(fuente_sintetica):
    lector = Lector(fuente_sintetica, _cf())
    # 2021 tiene BOM y CRLF
    esc = lector.escaneo(2021)
    assert esc.tiene_bom and esc.fin_de_linea == "CRLF"
    df = lector.leer(2021, proposito="estructura")
    assert df.columns[0] == "COD_HOSPITAL"
    assert df.width == 129
    ultima = df[esquema.COLUMNAS[-1]].drop_nulls()
    assert ultima.len() > 0
    assert not ultima.str.contains("\r").any()
    # el primer valor del primer campo no trae BOM
    assert not df["COD_HOSPITAL"].str.contains("\ufeff").any()


def test_alias_de_la_columna_2_por_posicion(fuente_sintetica):
    lector = Lector(fuente_sintetica, _cf())
    assert "ID_BENEFICIARIO" in lector.escaneo(2024).encabezado
    df24 = lector.leer(2024, ["COD_HOSPITAL", "CIP_ENCRIPTADO"], proposito="estructura")
    df23 = lector.leer(2023, ["COD_HOSPITAL", "CIP_ENCRIPTADO"], proposito="estructura")
    assert df24.columns == df23.columns == ["COD_HOSPITAL", "CIP_ENCRIPTADO"]
    assert df24["CIP_ENCRIPTADO"].null_count() < df24.height


def test_lineas_con_campos_de_mas_o_de_menos_se_rechazan_y_cuadran(tmp_path):
    df = generar_anio(2022, n=50, semilla=3)
    de_mas = "|".join(["x"] * 130)
    de_menos = "|".join(["y"] * 100)
    ruta = escribir_archivo(df, tmp_path / "GRD_PUBLICO_2022.txt", lineas_extra=[de_mas, de_menos, ""])
    from eda.lector import Fuente

    lector = Lector(Fuente("extracto", tmp_path, {2022: ruta}), _cf())
    res = lector.leer(2022, ["COD_HOSPITAL", "TIPO_INGRESO"], proposito="estructura")
    esc = lector.escaneo(2022)
    assert esc.lineas_datos == 53  # 50 filas + 3 líneas extra (una vacía)
    assert esc.rechazadas == 3
    assert res.height == 50
    assert res.height + esc.rechazadas == esc.lineas_total - 1
    assert lector.contabilidad[2022] == {"lineas_datos": 53, "rechazadas": 3, "leidas": 50}
    assert esc.lineas_vacias == 1


def test_ceros_a_la_izquierda_y_comas_decimales_se_conservan(fuente_sintetica):
    lector = Lector(fuente_sintetica, _cf())
    df = lector.leer(2020, ["PROCEDIMIENTO1", "IR_29301_COD_GRD", "IR_29301_PESO"], proposito="estructura")
    proc = df["PROCEDIMIENTO1"].drop_nulls()
    assert proc.is_in(["00.17", "09.1", "47.01", "81.51"]).all()
    assert (proc == "00.17").any() and (proc == "09.1").any()
    grd = df["IR_29301_COD_GRD"]
    assert grd.str.contains(r"^0\d{5}$").any()  # códigos del MDC 01 conservan el cero inicial
    assert df["IR_29301_PESO"].str.contains(",").all()
    assert df.dtypes == [pl.String] * 3


def test_encabezado_con_otro_numero_de_columnas_se_rechaza(tmp_path):
    ruta = tmp_path / "GRD_PUBLICO_2020.txt"
    ruta.write_bytes(b"A|B|C\r\n1|2|3\r\n")
    from eda.lector import Fuente

    lector = Lector(Fuente("extracto", tmp_path, {2020: ruta}), _cf())
    with pytest.raises(EsquemaInvalido):
        lector.leer(2020, ["COD_HOSPITAL"], proposito="estructura")


def test_muestra_es_reproducible(fuente_sintetica):
    lector = Lector(fuente_sintetica, _cf())
    a = lector.leer(2020, ["COD_HOSPITAL", "SEXO"], proposito="estructura", muestra=100, semilla=5)
    b = lector.leer(2020, ["COD_HOSPITAL", "SEXO"], proposito="estructura", muestra=100, semilla=5)
    c = lector.leer(2020, ["COD_HOSPITAL", "SEXO"], proposito="estructura", muestra=100, semilla=6)
    assert a.height == 100 and a.equals(b) and not a.equals(c)


def test_resolver_fuente_completo_exige_raiz_y_detecta_faltantes(tmp_path, fuente_sintetica):
    with pytest.raises(ValueError):
        resolver_fuente("completo")
    with pytest.raises(ArchivoNoEncontrado):
        resolver_fuente("completo", data_root=tmp_path)
    # con los archivos sintéticos, la búsqueda por patrón *<año>*.txt los encuentra
    f = resolver_fuente("completo", data_root=fuente_sintetica.raiz)
    assert f.anios() == [2019, 2020, 2021, 2022, 2023, 2024]
    assert f.etiqueta == "completo"


# ------------------------------------------------------------------- supresión
def test_supresion_celdas_pequenas_conserva_cero_y_mayores():
    df = pl.DataFrame({"g": ["a", "b", "c", "d"], "n": [0, 5, 10, 250], "tasa": [0.0, 0.4, 0.1, 0.02]})
    prot, n_sup = suprimir(df, ["n"], minimo=10, dependientes=["tasa"])
    assert n_sup == 1
    assert prot["n"].to_list() == [0, None, 10, 250]
    assert prot["tasa"].to_list() == [0.0, None, 0.1, 0.02]
    assert contar_pequenas(prot, ["n"], 10) == 0
    assert contar_pequenas(df, ["n"], 10) == 1
    # con suprime_cero también se anula el cero
    prot0, n0 = suprimir(df, ["n"], minimo=10, suprime_cero=True)
    assert n0 == 2 and prot0["n"].to_list() == [None, None, 10, 250]


def test_escribir_tabla_suprime_y_registra(ctx):
    df = pl.DataFrame({"anio": [2019, 2020], "n": [7, 120], "tasa": [0.5, 0.01]})
    ctx.escribir_tabla("t_prueba.csv", df, tarea=2, descripcion="prueba", cols_n=["n"], dependientes=["tasa"])
    leida = ctx.leer_tabla("t_prueba.csv")
    assert leida["n"].to_list() == [None, 120]
    assert leida["tasa"].to_list() == [None, 0.01]
    idx = ctx.indice_tablas()
    assert idx["celdas_suprimidas"].to_list() == [1]
    assert idx["fuente"].to_list() == ["extracto"]
    with pytest.raises(ValueError):
        ctx.escribir_tabla("sin_extension", df, tarea=2, descripcion="x")


# ------------------------------------------------------------------ cortafuegos
def test_2024_bloquea_resultado_y_lo_registra(fuente_sintetica):
    lector = Lector(fuente_sintetica, _cf())
    with pytest.raises(AccesoBloqueado):
        lector.leer(2024, ["TIPOALTA"], proposito="resultado", tarea="t07")
    with pytest.raises(AccesoBloqueado):
        lector.leer(2024, ["SEXO"], proposito="seleccion", tarea="t11")
    reg = lector.cortafuegos.registro()
    assert reg.height == 2
    assert reg["decision"].to_list() == ["BLOQUEADO", "BLOQUEADO"]
    assert lector.cortafuegos.n_bloqueos() == 2
    assert lector.cortafuegos.violaciones() == 0


def test_2024_entrega_resultado_enmascarado_para_estructura(fuente_sintetica):
    lector = Lector(fuente_sintetica, _cf())
    df = lector.leer(2024, ["TIPOALTA", "FECHAALTA", "SEXO"], proposito="estructura", tarea="t03")
    # TIPOALTA solo informa presencia y FECHAALTA solo el patrón: no se pueden contar fallecidos
    assert set(df["TIPOALTA"].drop_nulls().unique().to_list()) == {"A"}
    assert set(df["FECHAALTA"].drop_nulls().unique().to_list()) == {"9999-99-99"}
    lector2 = Lector(fuente_sintetica, _cf())
    sa = lector2.leer(2024, ["SERVICIOALTA"], proposito="estructura")["SERVICIOALTA"].drop_nulls()
    assert set(sa.unique().to_list()) == {"A"}
    assert df["SEXO"].is_in(["MUJER", "HOMBRE", "DESCONOCIDO"]).all()  # no protegida: en claro
    reg = lector.cortafuegos.registro()
    assert reg["decision"].to_list() == ["PERMITIDO_ENMASCARADO"]


def test_2024_calidad_con_regla_autorizada_entrega_en_claro(fuente_sintetica):
    lector = Lector(fuente_sintetica, _cf(reglas=["F01"]))
    df = lector.leer(2024, ["FECHAALTA"], proposito="calidad", tarea="t05", reglas=["F01"])
    assert df["FECHAALTA"].drop_nulls().str.contains(r"^\d{4}-\d{2}-\d{2}$").all()
    # una regla no autorizada recibe la columna enmascarada
    df2 = lector.leer(2024, ["FECHAALTA"], proposito="calidad", tarea="t05", reglas=["X99"])
    assert set(df2["FECHAALTA"].drop_nulls().unique().to_list()) == {"9999-99-99"}
    reg = lector.cortafuegos.registro()
    assert reg["decision"].to_list() == ["PERMITIDO_REGLA_CALIDAD", "PERMITIDO_ENMASCARADO"]
    assert lector.cortafuegos.n_lecturas_en_claro_expuestas() == 1
    assert lector.cortafuegos.violaciones() == 0


def test_otros_anios_permiten_resultado(fuente_sintetica):
    lector = Lector(fuente_sintetica, _cf())
    df = lector.leer(2023, ["TIPOALTA"], proposito="resultado", tarea="t07")
    assert (df["TIPOALTA"] == "FALLECIDO").any()
    assert lector.cortafuegos.registro()["decision"].to_list() == ["PERMITIDO"]


def test_violaciones_detecta_registro_inconsistente():
    cf = _cf()
    cf._entradas.append(
        {"secuencia": 1, "tarea": "x", "anio": 2024, "proposito": "estructura", "n_columnas": 1,
         "columnas_resultado": "TIPOALTA", "reglas": "", "decision": "PERMITIDO", "motivo": ""}
    )
    assert cf.violaciones() == 1


# --------------------------------------------------------------------- gráficos
def _tabla_para_figura(ctx: Contexto) -> None:
    df = pl.DataFrame({"anio": ["2019", "2020", "2021"], "filas": [600, 600, 600]})
    ctx.escribir_tabla("t_fig.csv", df, tarea=2, descripcion="figura de prueba", cols_n=["filas"])


def _hacer_figura(ctx: Contexto, id_figura: str = "F99_01"):
    t = ctx.leer_tabla("t_fig.csv", columnas_texto=["anio"])
    fig, ax = graficos.nueva_figura()
    graficos.barras_verticales(ax, t["anio"].to_list(), t["filas"].to_list(), etiqueta_y="Filas", etiqueta_x="Año")
    return ctx.figuras.guardar_figura(
        fig,
        id_figura=id_figura,
        tarea=2,
        titulo="Filas por archivo y año",
        descripcion_alt="Barras con 600 filas en cada año",
        csv="t_fig.csv",
        anios="2019 a 2021",
    )


def test_png_identico_en_dos_corridas(tmp_path, fuente_sintetica):
    hashes = []
    for k in range(2):
        c = Contexto.crear("extracto", fuente=fuente_sintetica, dir_salida_base=tmp_path / f"corrida{k}")
        _tabla_para_figura(c)
        ruta = _hacer_figura(c)
        hashes.append(manifiesto.sha256_archivo(ruta))
    assert hashes[0] == hashes[1]


def test_figura_sin_csv_falla_y_se_indexa_con_csv(ctx):
    fig, ax = graficos.nueva_figura()
    ax.plot([1, 2], [1, 2])
    with pytest.raises(graficos.FiguraSinCSV):
        ctx.figuras.guardar_figura(
            fig, id_figura="F99_02", tarea=2, titulo="x", descripcion_alt="x", csv="no_existe.csv", anios="2019"
        )
    plt.close("all")
    _tabla_para_figura(ctx)
    _hacer_figura(ctx)
    idx = ctx.figuras.indice()
    assert idx.height == 1
    fila = idx.row(0, named=True)
    assert fila["csv"] == "t_fig.csv" and fila["fuente"] == "extracto" and fila["archivo_png"] == "F99_01.png"
    assert len(fila["sha256_png"]) == 64 and len(fila["sha256_csv"]) == 64


def test_figura_exige_descripcion_alternativa(ctx):
    _tabla_para_figura(ctx)
    fig, ax = graficos.nueva_figura()
    with pytest.raises(ValueError):
        ctx.figuras.guardar_figura(fig, id_figura="F99_03", tarea=2, titulo="x", descripcion_alt=" ", csv="t_fig.csv", anios="2019")
    plt.close("all")


# -------------------------------------------------------------------- manifiesto
def test_manifiesto_es_determinista_y_sin_rutas_absolutas(ctx, fuente_sintetica):
    lector = ctx.lector
    escaneos = {a: lector.escaneo(a) for a in (2019, 2020)}
    _tabla_para_figura(ctx)
    salida = list(ctx.dir_tablas.glob("*.csv"))
    kw = dict(
        fuente="extracto",
        escaneos=escaneos,
        semilla=1,
        archivos_config=[],
        archivos_codigo=[],
        archivos_salida=salida,
    )
    m1 = manifiesto.a_json(manifiesto.construir_manifiesto(**kw))
    m2 = manifiesto.a_json(manifiesto.construir_manifiesto(**kw))
    assert m1 == m2
    datos = json.loads(m1)
    assert set(datos) >= {"fuente", "semilla", "versiones", "datos", "config", "codigo", "salidas", "accesos"}
    assert "/home" not in m1 and "/tmp" not in m1
    assert len(datos["datos"]["2019"]["sha256"]) == 64
    assert datos["versiones"]["polars"]


def test_mascara_y_centinela():
    s = pl.DataFrame({"x": ["AB-12", "12.3", "é9", None, "DESCONOCIDO", " desconocido ", "NO IDENTIFICADA", "NINGUNO"]})
    m = s.select(esquema.mascara_patron(pl.col("x")))["x"].to_list()
    assert m[:3] == ["AA-99", "99.9", "A9"] and m[3] is None
    # idempotente
    m2 = s.select(esquema.mascara_patron(esquema.mascara_patron(pl.col("x"))))["x"].to_list()
    assert m == m2
    c = s.select(esquema.es_centinela(pl.col("x")))["x"].to_list()
    assert c == [False, False, False, False, True, True, True, False]
