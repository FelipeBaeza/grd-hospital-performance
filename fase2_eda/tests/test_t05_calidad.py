"""Tarea 5: reglas de calidad (fechas, identificador, duplicados) con datos sintéticos."""

from __future__ import annotations

import polars as pl
import pytest

from eda import calidad, config, esquema, privacidad
from eda.contexto import Contexto
from sintetico import crear_fuente_sintetica


@pytest.fixture(scope="module")
def reglas():
    return {r.id: r for r in calidad.cargar_reglas()}


@pytest.fixture(scope="module")
def umbrales():
    return config.cargar_umbrales()


# --------------------------------------------------------------------- definición de reglas
def test_reglas_toml_son_validas_y_declaran_su_accion(reglas):
    lista = calidad.cargar_reglas()
    assert calidad.validar_reglas(lista) == []
    assert len(lista) == 32  # 19 del lote 1 y 13 del lote 2
    assert len(calidad.reglas_lote1()) == 19
    assert {r.accion for r in lista} <= set(calidad.ACCIONES)
    assert {r.severidad for r in lista} <= set(calidad.SEVERIDADES)
    assert reglas["Q17"].accion == "informar_fonasa"
    assert reglas["Q05"].accion == "anular"  # R5: los errores se anulan, no se imputan
    # todas las reglas del lote 1 pueden leer el año expuesto (solo devuelven violaciones)
    assert all(r.permite_expuesto for r in calidad.reglas_lote1())


def test_regla_invalida_se_detecta():
    mala = calidad.Regla("X1", "x", "no_existe", (), "grave", "borrar", True, "global", "", "")
    problemas = calidad.validar_reglas([mala, mala])
    assert any("repetidos" in p for p in problemas)
    assert any("tipo desconocido" in p for p in problemas)
    assert any("severidad inválida" in p for p in problemas)
    assert any("acción inválida" in p for p in problemas)
    assert any("alcance inválido" in p for p in problemas)
    assert any("sin descripción" in p for p in problemas)


# ------------------------------------------------ una prueba positiva y una negativa por regla
def _marco_con_defectos() -> pl.DataFrame:
    filas = [
        # ID, NAC, SEXO, ING, ALTA, HOSP, DX1
        ("123456", "1980-05-10", "MUJER", "2020-03-01", "2020-03-05", "101100", "J189"),  # normal
        (None, "1980-05-10", "HOMBRE", "2020-03-01", "2020-03-02", "101100", "J189"),  # ID vacío
        ("DESCONOCIDO", "1980-05-10", "HOMBRE", "2020-03-01", "2020-03-02", "101100", "J189"),  # ID centinela
        ("12", None, "MUJER", "2020-03-10", "2020-03-09", "101100", "J189"),  # ID corto, nacimiento vacío, alta antes del ingreso
        ("AB12", "2030-01-01", "MUJER", "2020-03-10", None, "101100", "J189"),  # ID no numérico, edad negativa, alta vacía
        ("7654321", "1890-01-01", "HOMBRE", "10-03-2020", "2021-06-01", "101100", "J189"),  # edad > 110, estadía > 365, alta en otro año
        ("7654321", "1990-01-01", "HOMBRE", "ayer", "2020-03-12", "101100", "J189"),  # fecha de ingreso no convertible
        ("123456", "1980-05-10", "MUJER", "2020-03-01", "2020-03-05", "101100", "J189"),  # duplicado de la fila 0
    ]
    cols = ["CIP_ENCRIPTADO", "FECHA_NACIMIENTO", "SEXO", "FECHA_INGRESO", "FECHAALTA", "COD_HOSPITAL", "DIAGNOSTICO1"]
    return pl.DataFrame({c: [f[i] for f in filas] for i, c in enumerate(cols)}, schema={c: pl.String for c in cols})


def _marco_limpio() -> pl.DataFrame:
    cols = ["CIP_ENCRIPTADO", "FECHA_NACIMIENTO", "SEXO", "FECHA_INGRESO", "FECHAALTA", "COD_HOSPITAL", "DIAGNOSTICO1"]
    filas = [(str(100000 + i), "1980-05-10", "MUJER", "2020-03-01", "2020-03-05", "101100", "J189") for i in range(6)]
    return pl.DataFrame({c: [f[i] for f in filas] for i, c in enumerate(cols)}, schema={c: pl.String for c in cols})


# regla -> (violaciones, evaluados) esperados sobre el marco con defectos (año 2020)
ESPERADO = {
    "Q01": (1, 8),
    "Q02": (0, 8),
    "Q03": (1, 8),
    "Q04": (1, 22),
    "Q05": (1, 6),
    "Q06": (1, 6),
    "Q07": (1, 6),
    "Q08": (1, 6),
    "Q09": (1, 8),
    "Q10": (1, 8),
    "Q11": (1, 5),
    "Q12": (1, 8),
    "Q15": (1, 5),
    "Q18": (1, 7),
    "Q19": (0, 7),
}


@pytest.mark.parametrize("regla_id", sorted(ESPERADO))
def test_regla_detecta_el_caso_positivo(regla_id, reglas, umbrales):
    dfp = calidad.preparar(_marco_con_defectos())
    assert calidad.evaluar_regla(dfp, reglas[regla_id], umbrales, 2020) == ESPERADO[regla_id]


@pytest.mark.parametrize("regla_id", sorted(ESPERADO))
def test_regla_no_marca_datos_limpios(regla_id, reglas, umbrales):
    dfp = calidad.preparar(_marco_limpio())
    v, e = calidad.evaluar_regla(dfp, reglas[regla_id], umbrales, 2020)
    assert v == 0 and e > 0


def test_las_reglas_no_modifican_las_columnas_originales():
    df = _marco_con_defectos()
    dfp = calidad.preparar(df)
    assert dfp.select(df.columns).equals(df)  # R4/R5: las columnas originales no se tocan
    assert {"_nac", "_ing", "_alta", "_edad", "_est", "_id_clase", "_id_largo"} <= set(dfp.columns)


# -------------------------------------------------------------------------- regímenes de ID
def test_quiebre_de_identificador_entre_2020_y_2021():
    consecutivos = {(2019, 2020): 0.012, (2020, 2021): 0.0, (2021, 2022): 0.014, (2022, 2023): 0.013, (2023, 2024): 0.015}
    q = calidad.detectar_quiebres(consecutivos, 0.10)
    assert q == [(2020, 2021)]
    assert calidad.regimenes([2019, 2020, 2021, 2022, 2023, 2024], q) == [[2019, 2020], [2021, 2022, 2023, 2024]]
    assert calidad.etiqueta_regimen([2019, 2020]) == "2019 a 2020"
    # sin quiebre cuando todos los pares son parecidos o cuando todo es cero
    assert calidad.detectar_quiebres({(2019, 2020): 0.5, (2020, 2021): 0.45, (2021, 2022): 0.55}, 0.10) == []
    assert calidad.detectar_quiebres({(2019, 2020): 0.0, (2020, 2021): 0.0, (2021, 2022): 0.0}, 0.10) == []
    assert calidad.regimenes([2019, 2020, 2021], []) == [[2019, 2020, 2021]]


def test_traslape_de_ids_valor_conocido():
    a = pl.Series("id", ["1", "2", "3", "4"])
    b = pl.Series("id", ["3", "4", "5"])
    t = calidad.traslape_ids({2019: a, 2020: b, 2021: pl.Series("id", ["9"])})
    ab = t.filter((pl.col("anio_a") == 2019) & (pl.col("anio_b") == 2020)).row(0, named=True)
    assert ab["n_interseccion"] == 2 and ab["prop_min"] == pytest.approx(2 / 3) and ab["consecutivo"] == "si"
    ac = t.filter((pl.col("anio_a") == 2019) & (pl.col("anio_b") == 2021)).row(0, named=True)
    assert ac["n_interseccion"] == 0 and ac["consecutivo"] == "no"


def test_consistencia_de_id_nacimiento_y_sexo():
    d0 = pl.Series([None], dtype=pl.Date)
    f = pl.DataFrame(
        {
            "id": ["a", "a", "b", "b", "c", "c", "d", "d", "e"],
            "nac": ["2000-01-01", "2000-01-01", "2000-01-01", "2001-01-01", "2000-01-01", "2000-01-01", "2000-01-01", "2000-01-01", "2000-01-01"],
            "sexo": ["MUJER", "MUJER", "HOMBRE", "HOMBRE", "MUJER", "HOMBRE", "DESCONOCIDO", "MUJER", "MUJER"],
        }
    ).with_columns(pl.col("nac").str.to_date())
    n_multi, n_nac, n_sexo, n_multi_sexo = calidad._consistencia_id(f)
    assert n_multi == 4  # a, b, c, d tienen 2 episodios
    assert n_nac == 1  # solo b cambia de nacimiento
    assert n_sexo == 1  # solo c cambia de sexo
    assert n_multi_sexo == 3  # a, b y c tienen sexo conocido en dos episodios; d solo en uno
    assert d0.len() == 1


def test_solapes_excluyen_duplicados_exactos():
    d = lambda n: pl.Series([f"2020-01-{n:02d}"]).str.to_date()[0]  # noqa: E731
    filas = [
        ("a", 1, 5), ("a", 3, 4),  # a: un solape
        ("b", 1, 5), ("b", 1, 5),  # b: duplicado exacto (no es solape)
        ("c", 1, 2), ("c", 5, 6),  # c: secuencial
        ("d", 1, 1), ("d", 1, 1),  # d: duplicado exacto de estadía cero
        ("e", 1, 10), ("e", 2, 3), ("e", 5, 6),  # e: dos solapes con el primer episodio
    ]
    f = pl.DataFrame(
        {"id": [x[0] for x in filas], "nac": [None] * len(filas), "sexo": ["MUJER"] * len(filas),
         "ing": [d(x[1]) for x in filas], "alta": [d(x[2]) for x in filas], "anio": [2020] * len(filas)},
        schema={"id": pl.String, "nac": pl.Date, "sexo": pl.String, "ing": pl.Date, "alta": pl.Date, "anio": pl.Int64},
    )
    s = calidad.solapes(f).row(0, named=True)
    assert s["n_solapes"] == 3 and s["n_duplicados_exactos"] == 2 and s["n_evaluados"] == len(filas)


# ----------------------------------------------------------------- punta a punta (sintético)
DEFECTOS = {
    2020: {"alta_antes_ingreso": 3, "estancia_larga": 2, "edad_negativa": 2, "edad_mayor_110": 3, "alta_vacia": 2, "rut_en_fecha": 12, "duplicados": 15},
    2024: {"alta_antes_ingreso": 4},
}


@pytest.fixture(scope="module")
def corrida(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("t5")
    fuente = crear_fuente_sintetica(raiz / "datos", n=700, semilla=21, defectos_por_anio=DEFECTOS)
    ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=raiz / "salidas")
    return ctx, calidad.calcular(ctx)


def _v(r, regla_id, ambito):
    t = r["reglas"].filter((pl.col("regla_id") == regla_id) & (pl.col("ambito") == ambito))
    assert t.height == 1, (regla_id, ambito)
    return t["n_violaciones"][0]


def test_defectos_plantados_se_cuentan_exactamente(corrida):
    _, r = corrida
    assert _v(r, "Q05", "2020") == 3
    assert _v(r, "Q06", "2020") == 2
    assert _v(r, "Q07", "2020") == 2
    assert _v(r, "Q08", "2020") == 3
    assert _v(r, "Q03", "2020") == 2
    assert 12 <= _v(r, "Q15", "2020") <= 15  # los duplicados con ID válido
    for anio in ("2019", "2021", "2022", "2023"):
        assert _v(r, "Q05", anio) == 0 and _v(r, "Q07", anio) == 0 and _v(r, "Q08", anio) == 0
    assert _v(r, "Q05", "total") == 7  # 3 de 2020 y 4 de 2024


def test_2024_se_evalua_con_reglas_autorizadas_sin_violar_el_cortafuegos(corrida):
    ctx, r = corrida
    assert _v(r, "Q05", "2024") == 4  # FECHAALTA de 2024 leída en claro solo por una regla autorizada
    t = r["reglas"].filter(pl.col("ambito") == "2024")
    assert set(t["estado"].to_list()) == {"EVALUADA"}
    cf = ctx.cortafuegos
    assert cf.violaciones() == 0 and cf.n_bloqueos() == 0
    assert cf.n_lecturas_en_claro_expuestas() >= 1
    # ninguna regla devuelve una distribución de resultado: solo conteos de violación
    assert set(r["reglas"].columns) == set(calidad.ESQUEMA_REGLAS)


def test_sin_reglas_autorizadas_las_reglas_con_resultado_quedan_bloqueadas_en_2024(tmp_path):
    fuente = crear_fuente_sintetica(tmp_path / "datos", n=300, semilla=3, defectos_por_anio={2024: {"alta_antes_ingreso": 4}})
    ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=tmp_path / "salidas", reglas_autorizadas=[])
    r = calidad.calcular(ctx)
    t = r["reglas"].filter(pl.col("ambito") == "2024")
    bloq = t.filter(pl.col("estado") == "BLOQUEADA_2024")["regla_id"].to_list()
    assert {"Q03", "Q04", "Q05", "Q06", "Q15", "Q16", "Q18"}.issubset(set(bloq) | {"Q16"})
    assert t.filter(pl.col("regla_id") == "Q05")["n_violaciones"][0] is None
    # las reglas que no tocan columnas de resultado se evalúan igual
    assert t.filter(pl.col("regla_id") == "Q09")["estado"][0] == "EVALUADA"
    assert ctx.cortafuegos.violaciones() == 0


def test_quiebre_de_id_2020_2021_y_regimenes_en_los_datos(corrida):
    _, r = corrida
    assert r["quiebres"] == [(2020, 2021)]
    assert r["regimenes"]["etiqueta"].to_list() == ["2019 a 2020", "2021 a 2024"]
    t = r["traslape"].filter(pl.col("consecutivo") == "si")
    por = {(x["anio_a"], x["anio_b"]): x["n_interseccion"] for x in t.iter_rows(named=True)}
    assert por[(2020, 2021)] == 0
    assert all(por[p] > 0 for p in por if p != (2020, 2021))


def test_largo_del_id_cambia_de_regimen(corrida):
    _, r = corrida
    largo = r["largo"]
    for anio in (2019, 2020):
        cats = set(largo.filter((pl.col("anio") == anio) & (pl.col("n") > 0) & pl.col("categoria").str.contains(r"^\d+$"))["categoria"].to_list())
        assert cats <= {"6", "7"}
    for anio in (2021, 2022, 2023, 2024):
        cats = set(largo.filter((pl.col("anio") == anio) & (pl.col("n") > 0) & pl.col("categoria").str.contains(r"^\d+$"))["categoria"].to_list())
        assert cats <= {"8", "9"}


def test_rut_se_cuenta_por_columna_sin_guardar_el_valor(corrida):
    _, r = corrida
    t = r["rut"]
    assert t.columns == ["columna", "anio", "n"]
    assert t.to_dicts() == [{"columna": "FECHAPROCEDIMIENTO1", "anio": 2020, "n": 12}]


def test_consistencia_detecta_cambios_de_sexo_por_id(tmp_path):
    fuente = crear_fuente_sintetica(tmp_path / "datos", n=700, semilla=5, defectos_por_anio={2021: {"cambio_sexo": 60}})
    ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=tmp_path / "salidas")
    r = calidad.calcular(ctx)
    cons = r["consistencia"]
    todos = cons.filter(pl.col("grupo_id") == "todos los IDs")
    reg_b = todos.filter(pl.col("regimen") == "régimen 2021 a 2024").row(0, named=True)
    reg_a = todos.filter(pl.col("regimen") == "régimen 2019 a 2020").row(0, named=True)
    # los tres grupos de largo del ID suman al total
    suma = cons.filter((pl.col("regimen") == "régimen 2021 a 2024") & (pl.col("grupo_id") != "todos los IDs"))
    assert int(suma["n_ids_con_dos_o_mas_episodios"].sum()) == reg_b["n_ids_con_dos_o_mas_episodios"]
    assert reg_b["n_ids_sexo_inconsistente"] > 0
    assert reg_a["n_ids_sexo_inconsistente"] == 0 and reg_a["n_ids_nacimiento_inconsistente"] == 0


def test_ejecutar_escribe_tablas_figuras_y_cumple_privacidad(tmp_path):
    fuente = crear_fuente_sintetica(tmp_path / "datos", n=700, semilla=21, defectos_por_anio=DEFECTOS)
    ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=tmp_path / "salidas")
    calidad.ejecutar(ctx)
    for nombre in ("t24_reglas.csv", "t24_id_largo.csv", "t24_id_traslape.csv", "t24_id_regimenes.csv", "t24_id_consistencia.csv",
                   "t24_duplicados.csv", "t24_rut_por_columna.csv", "t24_resumen.csv"):
        assert (ctx.dir_tablas / nombre).is_file(), nombre
    assert sorted(ctx.figuras.ids()) == ["F05_01", "F05_02", "F05_03"]
    # los conteos menores a 10 salen suprimidos en la tabla escrita (3 alteraciones de Q05 en 2020)
    t = ctx.leer_tabla("t24_reglas.csv", columnas_texto=["ambito"])
    q05 = t.filter((pl.col("regla_id") == "Q05") & (pl.col("ambito") == "2020"))
    assert q05["n_violaciones"][0] is None and q05["prop_violaciones"][0] is None
    # el total de 7 también es menor a 10: suprimido; el RUT (12) se informa
    assert t.filter((pl.col("regla_id") == "Q05") & (pl.col("ambito") == "total"))["n_violaciones"][0] is None
    assert ctx.leer_tabla("t24_rut_por_columna.csv")["n"].to_list() == [12]
    # métricas del ledger: la suprimida queda sin valor, la grande con valor
    m = ctx.metricas()
    assert m.filter(pl.col("clave") == "calidad.alta_antes_ingreso.total")["valor"][0] is None
    assert m.filter(pl.col("clave") == "calidad.valores_rut.total")["valor"][0] == 12
    assert m.filter(pl.col("clave") == "calidad.primer_quiebre_id_anio_posterior")["valor"][0] == 2021
    ctx.escribir_registros()
    escaneo = privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)
    assert privacidad.total_violaciones(escaneo) == 0, escaneo.filter(pl.col("n_violaciones") > 0)
    for p in ctx.dir_tablas.glob("*.csv"):
        assert "12345678-9" not in p.read_text(encoding="utf8"), p.name


def test_informe_del_lote_1_se_genera_sin_cifras_sueltas(tmp_path):
    from eda.informes import Informe

    fuente = crear_fuente_sintetica(tmp_path / "datos", n=400, semilla=8, defectos_por_anio={2020: {"rut_en_fecha": 12}})
    ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=tmp_path / "salidas")
    calidad.ejecutar(ctx)
    inf = Informe(ctx, "t24_parcial.md", "Informe de calidad de los datos, primer lote")
    calidad.agregar_secciones(inf, ctx)
    md = inf.guardar().read_text(encoding="utf8")
    assert "Regímenes de IDs: 2019 a 2020, 2021 a 2024" in md or "2019 a 2020; 2021 a 2024" in md
    assert "Quiebres encontrados: 2020 a 2021" in md
    assert "![" in md
