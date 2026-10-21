"""Tarea 4: descripción (CRISP-DM 2.2) y conciliación con el diccionario, con datos sintéticos."""

from __future__ import annotations

import polars as pl
import pytest

from eda import descripcion, esquema, privacidad, transformaciones as tr
from eda.contexto import Contexto
from sintetico import crear_fuente_sintetica


# ------------------------------------------------------------------- transformaciones
def test_fechas_en_dos_formatos_y_edad():
    df = pl.DataFrame(
        {
            "nac": ["2000-03-15", "2000-03-15", "1990-12-31", "2020-01-01"],
            "ing": ["2020-03-14", "15-03-2020", "2020-12-30", "2019-06-01"],
        }
    )
    r = df.select(
        tr.edad_anios(tr.a_fecha(pl.col("nac")), tr.a_fecha(pl.col("ing"))).alias("edad"),
        tr.estancia_dias(tr.a_fecha(pl.col("ing")), tr.a_fecha(pl.col("nac"))).alias("dias"),
    )
    # 1) el cumpleaños aún no ocurre; 2) ya ocurrió; 3) fin de año; 4) nacimiento posterior al ingreso da edad negativa
    assert r["edad"].to_list() == [19, 20, 29, -1]
    assert df.select(tr.a_fecha(pl.col("ing")))["ing"].null_count() == 0
    assert pl.DataFrame({"x": ["DESCONOCIDO", None, "2020-02-30"]}).select(tr.a_fecha(pl.col("x")))["x"].null_count() == 3


def test_decimales_con_coma_incluyen_sin_cero_inicial():
    r = pl.DataFrame({"p": ["1,2345", ",5", "0,75", "3.5", "x", None]}).select(tr.a_decimal(pl.col("p")))["p"].to_list()
    assert r[:4] == [1.2345, 0.5, 0.75, 3.5] and r[4] is None and r[5] is None


# ------------------------------------------------------------------ etiquetas y perfil
def test_etiquetas_truncadas_se_detectan():
    etiquetas = [
        "DOMICILIO",
        "FALLECIDO",
        "DERIVACIÓN OTRO HOSPITAL DE LA RED NACIONAL",
        "DERIVACIÓN INST. PRIVADA (VOLUNTARIO)",
        "DERIVACIÓN A OTROS CENTROS (CÁRCEL, HOGAR DE",
        "DERIVACIÓN INST. PRIVADA (COMPRA DE SERVICIOS",
    ]
    assert descripcion.etiquetas_truncadas(etiquetas) == [
        "DERIVACIÓN A OTROS CENTROS (CÁRCEL, HOGAR DE",
        "DERIVACIÓN INST. PRIVADA (COMPRA DE SERVICIOS",
    ]
    assert descripcion.etiquetas_truncadas(["DOMICILIO", "FALLECIDO (SIN CIERRE"]) == ["FALLECIDO (SIN CIERRE"]
    assert descripcion.etiquetas_truncadas(["A (B)", "C"]) == []


def test_inferir_tipo():
    assert descripcion.inferir_tipo(pl.Series("x", ["2020-01-02"] * 20 + ["DESCONOCIDO"])) == "fecha_iso"
    assert descripcion.inferir_tipo(pl.Series("x", ["02-01-2020"] * 5)) == "fecha_dmy"
    assert descripcion.inferir_tipo(pl.Series("x", ["12", "300", "7"])) == "entero"
    assert descripcion.inferir_tipo(pl.Series("x", ["1,5", ",25", "3"])) == "decimal"
    assert descripcion.inferir_tipo(pl.Series("x", ["MUJER", "HOMBRE"])) == "texto"
    assert descripcion.inferir_tipo(pl.Series("x", [None, None], dtype=pl.String)) == "vacia"


def test_perfil_cuenta_nulos_y_centinelas_sinteticos():
    valores = ["A"] * 60 + ["B"] * 25 + ["DESCONOCIDO"] * 5 + [None] * 10
    df = pl.DataFrame({"X": valores})
    fila, top, conteo = descripcion.perfil_columna(df, "X", 2020, protegida=False)
    assert (fila["n_filas"], fila["n_nulos"], fila["n_centinela"], fila["n_informados"]) == (100, 10, 5, 90)
    assert fila["n_distintos"] == 3
    assert fila["valor_modal"] == "A" and fila["n_modal"] == 60
    assert fila["tipo_inferido"] == "texto"
    assert [(t["valor"], t["n"]) for t in top] == [("A", 60), ("B", 25), ("DESCONOCIDO", 5)]
    assert conteo["A"] == 60


def test_perfil_de_columna_protegida_solo_informa_nulos_y_patron():
    df = pl.DataFrame({"TIPOALTA": ["A"] * 8 + [None] * 2})
    fila, top, conteo = descripcion.perfil_columna(df, "TIPOALTA", 2024, protegida=True)
    assert fila["n_nulos"] == 2 and fila["enmascarada"] == "si"
    assert fila["n_distintos"] is None and fila["valor_modal"] is None and fila["chao1"] is None
    assert top == [] and conteo is None


def test_perfil_no_expone_valores_de_fechas_ni_ruts():
    df = pl.DataFrame({"FECHAPROCEDIMIENTO1": ["12345678-9"] * 12, "FECHA_NACIMIENTO": ["1950-01-01"] * 12})
    f1, top1, c1 = descripcion.perfil_columna(df, "FECHAPROCEDIMIENTO1", 2019, False)
    f2, top2, c2 = descripcion.perfil_columna(df, "FECHA_NACIMIENTO", 2019, False)
    assert f1["patron_moda"] == "99999999-9" and top1 == [] and c1 is None and f1["valor_modal"] is None
    assert f2["tipo_inferido"] == "fecha_iso" and f2["anio_min"] == 1950 and top2 == [] and f2["valor_modal"] is None


# ------------------------------------------------------------------------ conciliación
def test_conciliacion_exacta_en_modo_completo():
    assert descripcion.conciliar(95, 5, 5, 100, "completo")[0] == "COINCIDE"
    assert descripcion.conciliar(95, 5, 6, 100, "completo")[0] == "DISCREPANCIA"
    assert descripcion.conciliar(95, 5, 5, 99, "completo")[0] == "DISCREPANCIA"


def test_conciliacion_por_proporcion_en_extracto():
    n_doc, n = 5_808_536, 50_000
    sin_doc = int(0.10 * n_doc)
    e, z = descripcion.conciliar(n_doc - sin_doc, sin_doc, 5_000, n, "extracto")
    assert e == "COINCIDE" and abs(z) < 3
    e, z = descripcion.conciliar(n_doc - sin_doc, sin_doc, 6_000, n, "extracto")
    assert e == "DISCREPANCIA" and z > 3
    # afirmación determinista: columna sin ningún vacío
    assert descripcion.conciliar(n_doc, 0, 0, n, "extracto")[0] == "COINCIDE"
    assert descripcion.conciliar(n_doc, 0, 1, n, "extracto")[0] == "DISCREPANCIA"
    # columna completamente vacía según el diccionario
    assert descripcion.conciliar(0, n_doc, n, n, "extracto")[0] == "COINCIDE"
    assert descripcion.conciliar(0, n_doc, n - 5, n, "extracto")[0] == "DISCREPANCIA"
    # evento raro: nueve valores en 5,8 millones no se evalúan con 50.000 filas
    assert descripcion.conciliar(9, n_doc - 9, n, n, "extracto")[0] in ("NO EVALUABLE", "DISCREPANCIA")
    assert descripcion.conciliar(n_doc - 9, 9, 0, n, "extracto")[0] == "NO EVALUABLE"
    assert descripcion.conciliar(n_doc - 9, 9, 40, n, "extracto")[0] == "DISCREPANCIA"


def test_conciliacion_estratificada_pondera_los_anios_por_volumen():
    n_doc = 5_000_000
    # dos años del extracto con el mismo tamaño pero poblaciones de 80 % y 20 %
    estratos = [(1000, 100, 0.8), (1000, 300, 0.2)]  # p_h = 0,1 y 0,3 -> p estimado 0,14
    sin_doc = round(0.14 * n_doc)
    estado, z, p_hat = descripcion.conciliar_estratificado(n_doc - sin_doc, sin_doc, estratos)
    assert p_hat == pytest.approx(0.14) and estado == "COINCIDE" and abs(z) < 0.1
    # la proporción simple del extracto (0,20) habría marcado una discrepancia falsa
    e_simple, z_simple = descripcion.conciliar(n_doc - sin_doc, sin_doc, 400, 2000, "extracto")
    assert e_simple == "DISCREPANCIA" and z_simple > 3
    # con un diccionario que dice 0,20 sí hay discrepancia contra el estimado ponderado
    sin_doc2 = round(0.20 * n_doc)
    estado2, z2, _ = descripcion.conciliar_estratificado(n_doc - sin_doc2, sin_doc2, estratos)
    assert estado2 == "DISCREPANCIA" and z2 < -3
    # afirmación determinista y evento raro
    assert descripcion.conciliar_estratificado(n_doc, 0, [(1000, 0, 1.0)])[0] == "COINCIDE"
    assert descripcion.conciliar_estratificado(n_doc, 0, [(1000, 2, 1.0)])[0] == "DISCREPANCIA"
    assert descripcion.conciliar_estratificado(n_doc - 9, 9, [(1000, 0, 1.0)])[0] == "NO EVALUABLE"


# -------------------------------------------------------------- recorrido de punta a punta
@pytest.fixture(scope="module")
def sintetico_t4(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("t4")
    fuente = crear_fuente_sintetica(
        raiz / "datos", n=700, semilla=11, defectos_por_anio={2019: {"rut_en_fecha": 12}}
    )
    ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=raiz / "salidas")
    return ctx, descripcion.calcular(ctx)


def test_calcular_perfila_las_129_columnas_en_los_6_anios(sintetico_t4):
    ctx, r = sintetico_t4
    p = r["perfil"]
    assert p.height == 129 * 6
    assert p["columna"].n_unique() == 129
    # columnas de resultado de 2024: enmascaradas y sin estadísticas de valores
    m = p.filter((pl.col("anio") == 2024) & (pl.col("columna").is_in(["TIPOALTA", "FECHAALTA", "SERVICIOALTA", "IR_29301_MORTALIDAD"])))
    assert m.height == 4 and set(m["enmascarada"].to_list()) == {"si"}
    assert m["n_distintos"].null_count() == 4 and m["valor_modal"].null_count() == 4
    # el resto de 2024 y los otros años sí traen estadísticas
    assert p.filter((pl.col("anio") == 2023) & (pl.col("columna") == "TIPOALTA"))["valor_modal"][0] == "DOMICILIO"
    assert ctx.cortafuegos.violaciones() == 0 and ctx.cortafuegos.n_bloqueos() == 0


def test_no_se_tabula_el_resultado_de_2024(sintetico_t4):
    _, r = sintetico_t4
    assert r["cat"].filter(pl.col("columna") == "TIPOALTA")["anio"].max() == 2023
    assert r["est"]["anio"].max() == 2023
    assert 2024 in r["edad"]["anio"].to_list()  # la edad no es resultado
    assert r["cat"].filter(pl.col("columna") == "TIPO_ACTIVIDAD")["anio"].max() == 2024
    assert r["card"].filter((pl.col("columna") == "TIPOALTA") & (pl.col("periodo") == "2019-2024")).height == 0
    assert r["card"].filter((pl.col("columna") == "TIPOALTA") & (pl.col("periodo") == "2019-2023")).height == 1


def test_etiquetas_truncadas_de_tipoalta_se_detectan_en_los_datos(sintetico_t4):
    _, r = sintetico_t4
    t = r["trunc"].filter(pl.col("columna") == "TIPOALTA")
    assert sorted(t["etiqueta"].to_list()) == [
        "DERIVACIÓN A OTROS CENTROS (CÁRCEL, HOGAR DE",
        "DERIVACIÓN INST. PRIVADA (COMPRA DE SERVICIOS",
    ]
    assert sorted(t["longitud"].to_list()) == [44, 45]


def test_deriva_etnia_y_psi(sintetico_t4):
    _, r = sintetico_t4
    et = r["cat"].filter(pl.col("columna") == "ETNIA")
    # 'OTRO ' (con espacio) en 2019-2021 y 'OTRO' después: recodificación por año
    con_esp = et.filter(pl.col("categoria") == "OTRO ")["anio"].unique().sort().to_list()
    sin_esp = et.filter(pl.col("categoria") == "OTRO")["anio"].unique().sort().to_list()
    assert con_esp == [2019, 2020, 2021] and sin_esp == [2022, 2023, 2024]
    psi = r["psi"].filter((pl.col("columna") == "ETNIA") & (pl.col("anio_base") == 2021) & (pl.col("anio_comparado") == 2022))
    assert psi["psi"][0] > 0.25 and psi["clasificacion"][0] == "alto"
    psi_ing = r["psi"].filter((pl.col("columna") == "TIPO_INGRESO") & (pl.col("anio_base") == 2019) & (pl.col("anio_comparado") == 2020))
    assert psi_ing["psi"][0] < 0.10


def test_pares_de_traslado_y_bloque_rn(sintetico_t4):
    _, r = sintetico_t4
    assert r["pares"].height == 9
    p1 = r["pares"].filter(pl.col("traslado") == 1).row(0, named=True)
    assert p1["n_fecha"] == p1["n_servicio"] and p1["diferencia"] == 0 and p1["n_fecha"] > 0
    rn2 = r["rn"].filter(pl.col("columna") == "RN2ESTADO").row(0, named=True)
    assert rn2["valor_modal"] == "0" and rn2["prop_modal"] > 0.9  # relleno constante


def test_histogramas_desde_conteos(sintetico_t4):
    _, r = sintetico_t4
    e = r["edad"].filter(pl.col("anio") == 2020)
    assert e["edad"].min() == 0 and e["edad"].max() <= 110
    assert int(e["n"].sum()) <= 700
    h = r["est"].filter(pl.col("anio") == 2021)
    assert "0" in h["tramo"].to_list()
    assert int(h["n"].sum()) == 700


def test_ejecutar_escribe_tablas_figuras_e_informe(tmp_path):
    fuente = crear_fuente_sintetica(tmp_path / "datos", n=700, semilla=5, defectos_por_anio={2019: {"rut_en_fecha": 12}})
    ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=tmp_path / "salidas")
    res = descripcion.ejecutar(ctx)
    for nombre in (
        "t22_perfil_columnas.csv", "t22_top_valores.csv", "t22_cardinalidades.csv", "t22_deriva_categorias.csv", "t22_psi.csv",
        "t22_hist_edad.csv", "t22_hist_estancia.csv", "t22_pares_traslado.csv", "t22_bloque_rn.csv",
        "t22_etiquetas_truncadas.csv", "t22_conciliacion.csv", "diccionario_medido.csv", "t22_resumen.csv",
    ):
        assert (ctx.dir_tablas / nombre).is_file(), nombre
    assert sorted(ctx.figuras.ids()) == ["F04_01", "F04_02", "F04_03"]
    # 129 filas conciliadas, cada una con un estado válido
    dm = ctx.leer_tabla("diccionario_medido.csv", columnas_texto=["estado"])
    assert dm.height == 129
    assert set(dm["estado"].to_list()) <= {"COINCIDE", "DISCREPANCIA", "NO EVALUABLE"}
    md = res["informe"].read_text(encoding="utf8")
    assert "Conciliación con el diccionario" in md
    # privacidad: ni el valor con forma de RUT ni fechas completas en ninguna salida
    for p in ctx.dir_tablas.glob("*.csv"):
        texto = p.read_text(encoding="utf8")
        assert "12345678-9" not in texto, p.name
    ctx.escribir_registros()
    escaneo = privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)
    assert privacidad.total_violaciones(escaneo) == 0, escaneo.filter(pl.col("n_violaciones") > 0)
    assert ctx.cortafuegos.violaciones() == 0


def test_discrepancia_plantada_en_el_diccionario_se_detecta(tmp_path):
    fuente = crear_fuente_sintetica(tmp_path / "datos", n=700, semilla=9)
    ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=tmp_path / "salidas")
    r = descripcion.calcular(ctx)
    perfil = r["perfil"]
    por_col = perfil.group_by("columna").agg(pl.col("n_nulos").sum().alias("sin"), pl.col("n_filas").sum().alias("n"))
    n_doc = 5_808_536
    filas = []
    for row in por_col.iter_rows(named=True):
        p_sin = row["sin"] / row["n"]
        sin_doc = round(p_sin * n_doc)
        if row["columna"] == "SERVICIOINGRESO":  # discrepancia plantada: el diccionario dice 40 % de vacíos
            sin_doc = round(0.40 * n_doc)
        filas.append({"columna": row["columna"], "que_representa": "", "tipo_doc": "Texto", "con_datos_doc": n_doc - sin_doc,
                      "sin_datos_doc": sin_doc, "uso": "", "categoria_fase1": "X", "nota": ""})
    dic = pl.DataFrame(filas)
    conc = descripcion.tabla_conciliacion(dic, perfil, "extracto", 3.0)
    discrepan = conc.filter(pl.col("estado") == "DISCREPANCIA")["columna"].to_list()
    assert discrepan == ["SERVICIOINGRESO"]
    assert conc.height == 129


# ------------------------------------------------------------ magnitud de la discrepancia (Tarea 12)
def test_magnitud_no_aplica_si_el_estado_no_es_discrepancia():
    for estado in ("COINCIDE", "NO EVALUABLE"):
        for dif in (None, 0.0, 0.5, 40.0):
            assert descripcion.magnitud_discrepancia(estado, dif, 0.5) == "no aplica"


def test_magnitud_con_la_diferencia_exacta_del_umbral_es_relevante():
    assert descripcion.magnitud_discrepancia("DISCREPANCIA", 0.5, 0.5) == "relevante"
    assert descripcion.magnitud_discrepancia("DISCREPANCIA", 0.5000001, 0.5) == "relevante"
    assert descripcion.magnitud_discrepancia("DISCREPANCIA", 0.4999999, 0.5) == "despreciable"


def test_magnitud_sin_diferencia_o_con_diferencia_nula_es_despreciable():
    assert descripcion.magnitud_discrepancia("DISCREPANCIA", 0.0, 0.5) == "despreciable"
    assert descripcion.magnitud_discrepancia("DISCREPANCIA", None, 0.5) == "despreciable"


def test_magnitud_usa_el_umbral_que_se_le_entrega():
    assert descripcion.magnitud_discrepancia("DISCREPANCIA", 1.0, 2.0) == "despreciable"
    assert descripcion.magnitud_discrepancia("DISCREPANCIA", 2.0, 2.0) == "relevante"


def test_tabla_conciliacion_rotula_la_magnitud_con_la_misma_funcion():
    # modo completo: 1000 filas y 100 vacías documentadas (10,0 %). Medido: 100 (coincide), 104 (0,4 pp), 105 (0,5 pp), 130 (3,0 pp)
    medidos = {"A": 100, "B": 104, "C": 105, "D": 130}
    dic = pl.DataFrame(
        [
            {"columna": c, "que_representa": "", "tipo_doc": "Texto", "con_datos_doc": 900, "sin_datos_doc": 100,
             "uso": "", "categoria_fase1": "X", "nota": ""}
            for c in medidos
        ]
    )
    perfil = pl.DataFrame(
        [
            {"columna": c, "anio": 2020, "n_filas": 1000, "n_nulos": n, "n_centinela": 0, "n_distintos": 5, "tipo_inferido": "texto"}
            for c, n in medidos.items()
        ]
    )
    conc = descripcion.tabla_conciliacion(dic, perfil, "completo", 3.0, dif_relevante_pp=0.5)
    fila = {r["columna"]: (r["estado"], r["magnitud"]) for r in conc.iter_rows(named=True)}
    assert fila == {
        "A": ("COINCIDE", "no aplica"),
        "B": ("DISCREPANCIA", "despreciable"),
        "C": ("DISCREPANCIA", "relevante"),
        "D": ("DISCREPANCIA", "relevante"),
    }


def test_las_siete_categorias_de_f04_02_se_eligen_con_desempate_por_nombre_aunque_cambie_el_orden_de_los_grupos(ctx, monkeypatch):
    """Con categorías empatadas en el corte de las siete más frecuentes, dos corridas dibujaban categorías y colores distintos (DT11)."""
    import matplotlib.axes

    from ayudas_t13 import grupos_al_reves

    filas = [("A", 100), ("B", 90), ("C", 80), ("D", 70), ("E", 60), ("F", 50), ("G", 40), ("H", 40), ("I", 40)]
    tabla = pl.DataFrame([{"columna": "TIPO_ACTIVIDAD", "anio": a, "categoria": c, "n": n, "prop": n / 570} for a in (2019, 2020) for c, n in filas])
    ctx.escribir_tabla("t22_deriva_categorias.csv", tabla, tarea=4, descripcion="deriva a mano")
    original = matplotlib.axes.Axes.bar

    def dibujar() -> list[str]:
        etiquetas: list[str] = []

        def espia(self, *args, **kwargs):
            etiquetas.append(kwargs.get("label"))
            return original(self, *args, **kwargs)

        with monkeypatch.context() as m:
            m.setattr(matplotlib.axes.Axes, "bar", espia)
            descripcion.figura_deriva_categorias(ctx)
        return etiquetas

    normal = dibujar()
    with grupos_al_reves():
        al_reves = dibujar()
    # a mano: las seis primeras y, entre G, H e I empatadas con 40, la primera por nombre; el resto va en "(otras)"
    assert normal == ["A", "B", "C", "D", "E", "F", "G", "(otras)"]
    assert al_reves == normal
