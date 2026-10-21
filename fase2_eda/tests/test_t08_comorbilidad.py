"""Tarea 8, parte B: prevalencia de Elixhauser, mapeo de tres caracteres, diagnóstico principal, phi, deriva de la codificación y conocimiento clínico."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from eda import comorbilidad as cm
from eda import config, conocimiento, elixhauser, esquema, privacidad
from eda.contexto import Contexto
from eda.informes import Informe
from sintetico import crear_fuente_sintetica


@pytest.fixture(scope="module")
def e():
    return elixhauser.cargar()


@pytest.fixture(scope="module")
def e3():
    return elixhauser.cargar(solo_3c=True)


@pytest.fixture(scope="module")
def u():
    return config.cargar_umbrales()


# ------------------------------------------------------------------------- phi
def test_phi_valores_conocidos():
    # asociación perfecta, independencia y asociación perfecta negativa con 100 egresos y dos variables de 50
    for n11, esperado in ((50, 1.0), (25, 0.0), (0, -1.0)):
        co = np.array([[50, n11], [n11, 50]])
        assert cm.phi_desde_conteos(co, 100)[0, 1] == pytest.approx(esperado)
    # una variable constante no tiene phi
    co = np.array([[0, 0], [0, 40]])
    phi = cm.phi_desde_conteos(co, 100)
    assert np.isnan(phi[0, 1]) and np.isnan(phi[0, 0]) and phi[1, 1] == 1.0


def test_phi_coincide_con_la_correlacion_de_pearson_de_variables_binarias():
    rng = np.random.default_rng(3)
    base = rng.random((5000, 1)) < 0.3
    x = np.hstack([base, base ^ (rng.random((5000, 1)) < 0.2), rng.random((5000, 2)) < 0.1]).astype(np.int64)
    co = x.T @ x
    phi = cm.phi_desde_conteos(co, x.shape[0])
    assert np.allclose(phi, np.corrcoef(x.T), atol=1e-12)
    assert np.allclose(phi, phi.T)


# ------------------------------------------------------------------ largos del prefijo
def test_largos_de_prefijo_por_categoria(e):
    largos = cm.largos_por_categoria(e)
    assert largos["ELIX_11"] == "4" and largos["ELIX_12"] == "4"          # la diabetes solo se define con cuatro caracteres
    assert largos["ELIX_06"] == "3" and largos["ELIX_19"] == "3" and largos["ELIX_23"] == "3"
    assert largos["ELIX_01"] == "3 y 4" and largos["ELIX_02"] == "3 y 4"


# --------------------------------------------------------------- conteos de un año
def _meta(filas: list[tuple[str, str | None, bool]]) -> pl.DataFrame:
    return pl.DataFrame({"hospital": [f[0] for f in filas], "DIAGNOSTICO1": [f[1] for f in filas], "es_ah": [f[2] for f in filas]},
                        schema={"hospital": pl.String, "DIAGNOSTICO1": pl.String, "es_ah": pl.Boolean})


def _dx(filas: list[list[str]]) -> pl.DataFrame:
    return pl.DataFrame({c: [(f[i] if i < len(f) else None) for f in filas] for i, c in enumerate(esquema.DIAG_SEC)}, schema={c: pl.String for c in esquema.DIAG_SEC})


@pytest.fixture(scope="module")
def anio_a_mano(e, e3):
    meta = _meta([("H1", "I50.0", True), ("H1", "J18.9", True), ("H2", "A09", False), ("H2", "C34.9", True)])
    dx = _dx([["E11.9", "I10"], ["E11.2", "I13.0"], ["I10"], []])
    return cm.resumir_anio(meta, dx, e, e3)


def test_resumir_anio_cuenta_categorias_con_los_dos_mapeos_y_en_las_dos_bases(anio_a_mano):
    r = anio_a_mano
    assert r["n_todos"] == 4 and r["n"] == 3
    # mapeo completo, adultos hospitalizados: fila 0 tiene diabetes no complicada e hipertensión; fila 1, diabetes complicada, insuficiencia cardíaca e hipertensión complicada
    esperado = {"ELIX_06": 1, "ELIX_11": 1, "ELIX_12": 1, "ELIX_01": 1, "ELIX_07": 1}
    assert {k: v for k, v in r["sec"].items() if v} == esperado
    # con tres caracteres no se ve la diabetes ni I13.0 como insuficiencia cardíaca, pero sí la hipertensión
    assert {k: v for k, v in r["tres"].items() if v} == {"ELIX_06": 1, "ELIX_07": 1}
    # en todos los egresos entra también el menor con hipertensión
    assert r["sec_todos"]["ELIX_06"] == 2 and r["tres_todos"]["ELIX_06"] == 2 and r["sec_todos"]["ELIX_11"] == 1
    # el diagnóstico principal agrega la insuficiencia cardíaca de la fila 0 y el tumor sólido de la fila 3
    assert {k: v for k, v in r["prin"].items() if v} == {"ELIX_01": 1, "ELIX_20": 1}
    assert r["uni"]["ELIX_01"] == 2 and r["uni"]["ELIX_20"] == 1 and r["uni"]["ELIX_11"] == 1


def test_coocurrencia_y_codificacion_por_hospital(anio_a_mano, e):
    r = anio_a_mano
    co = r["co"]
    ids = list(e.ids)
    assert co.shape == (31, 31) and (co == co.T).all()
    assert [int(co[k, k]) for k in range(31)] == [r["sec"][i] for i in ids]
    assert int(co[ids.index("ELIX_06"), ids.index("ELIX_11")]) == 1 and int(co[ids.index("ELIX_01"), ids.index("ELIX_07")]) == 1
    h = {row["hospital"]: row for row in r["hospitales"].iter_rows(named=True)}
    assert set(h) == {"H1", "H2"}                                       # el menor no cuenta entre los adultos hospitalizados
    assert (h["H1"]["n"], h["H1"]["suma"], h["H1"]["suma2"]) == (2, 4, 8)
    assert (h["H2"]["n"], h["H2"]["suma"], h["H2"]["suma2"]) == (1, 0, 0)
    assert r["suma"] == 4 and r["suma2"] == 8


def test_tablas_de_un_anio_a_mano(anio_a_mano, e, u):
    por_anio = {2019: anio_a_mano}
    v = cm.tabla_vs_3c(por_anio, e, u).filter(pl.col("elix_id") == "ELIX_11").row(0, named=True)
    assert v["prev_completo_ah"] == pytest.approx(1 / 3) and v["prev_3c_ah"] == 0.0
    assert v["dif_relativa_ah"] == pytest.approx(-1.0) and v["supera_umbral"] == "si" and v["largos_definicion"] == "4"
    assert cm.tabla_vs_3c(por_anio, e, u).filter(pl.col("elix_id") == "ELIX_06").row(0, named=True)["supera_umbral"] == "no"
    p = cm.tabla_principal(por_anio, e).filter(pl.col("elix_id") == "ELIX_01").row(0, named=True)
    assert (p["n_secundario"], p["n_con_principal"], p["n_principal"]) == (1, 2, 1) and p["aumento_relativo"] == pytest.approx(1.0)
    assert cm.tabla_principal(por_anio, e).filter(pl.col("elix_id") == "ELIX_20").row(0, named=True)["aumento_relativo"] is None   # sin casos secundarios


# --------------------------------------------------------------- deriva de la codificación
def test_deriva_calcula_media_varianza_ic_y_cambio_relativo_como_numpy(u):
    k0 = np.array([1, 2, 3, 4, 5, 5, 6])
    k1 = np.array([2, 3, 4, 5, 6, 7, 8, 9])

    def anio(k: np.ndarray) -> dict:
        h = pl.DataFrame({"hospital": ["H1"], "n": [len(k)], "suma": [int(k.sum())], "suma2": [int((k**2).sum())]},
                         schema={"hospital": pl.String, "n": pl.UInt32, "suma": pl.Int64, "suma2": pl.Int64})
        return {"n": len(k), "suma": int(k.sum()), "suma2": int((k**2).sum()), "hospitales": h}

    d = cm.tabla_deriva({2019: anio(k0), 2024: anio(k1)}, u)
    z = float(u["z_95"])
    fila = d.filter((pl.col("cod_hospital") == cm.TODOS) & (pl.col("anio") == 2024)).row(0, named=True)
    assert fila["media"] == pytest.approx(k1.mean()) and fila["desv"] == pytest.approx(k1.std(ddof=1))
    assert fila["ic_bajo"] == pytest.approx(k1.mean() - z * k1.std(ddof=1) / math.sqrt(len(k1)))
    razon = k1.mean() / k0.mean()
    se = math.sqrt(k1.var(ddof=1) / (len(k1) * k1.mean() ** 2) + k0.var(ddof=1) / (len(k0) * k0.mean() ** 2))
    assert fila["cambio_relativo"] == pytest.approx(razon - 1)
    assert fila["cambio_bajo"] == pytest.approx(razon * math.exp(-z * se) - 1) and fila["cambio_alto"] == pytest.approx(razon * math.exp(z * se) - 1)
    primero = d.filter((pl.col("cod_hospital") == cm.TODOS) & (pl.col("anio") == 2019)).row(0, named=True)
    assert primero["cambio_relativo"] is None                         # el primer año es la referencia
    assert d["cod_hospital"].to_list()[:2] == [cm.TODOS, cm.TODOS]    # el total nacional va primero


# --------------------------------------------------------------- conocimiento clínico
def test_conocimiento_clinico_cumple_las_reglas_de_fuentes_y_certeza(e):
    df = conocimiento.cargar()
    assert conocimiento.validar(df) == []
    assert set(elixhauser.IDS) <= set(df["variable"].to_list())
    for v in ("EDAD", "SEXO", "TIPO_INGRESO", "TIPO_ACTIVIDAD", "ESTANCIA_CERO", "IR_29301_MORTALIDAD", "IR_29301_SEVERIDAD", "IR_29301_PESO",
              "N_EGRESOS_12M", "N_DX_SEC", "VOLUMEN_HOSPITAL", "PUNTAJE_VAN_WALRAVEN"):
        assert v in df["variable"].to_list(), v
    por = {r["variable"]: r for r in df.iter_rows(named=True)}
    # los pesos de van Walraven dan la dirección: positivo aumenta el riesgo, negativo queda pendiente de interpretar
    assert por["ELIX_01"]["direccion_esperada"].startswith("aumenta") and por["ELIX_01"]["certeza"] == conocimiento.CERTEZA_VERIFICADA
    assert por["ELIX_29"]["certeza"] == conocimiento.CERTEZA_PENDIENTE                # peso negativo
    for sospechosa in e.sospechosas:                                                   # las cinco de la Fase 1 quedan pendientes
        assert por[sospechosa]["certeza"] == conocimiento.CERTEZA_PENDIENTE
    assert por["VOLUMEN_HOSPITAL"]["referencia"] == "Spiegelhalter 2005"
    assert por["EDAD"]["referencia"] == conocimiento.SIN_REFERENCIA and por["EDAD"]["certeza"] == conocimiento.CERTEZA_PENDIENTE


def test_el_validador_rechaza_fuentes_no_reconocidas_y_certeza_inventada():
    df = conocimiento.cargar()
    malo = df.with_columns(
        pl.when(pl.col("variable") == "EDAD").then(pl.lit("Un blog")).otherwise(pl.col("referencia")).alias("referencia"),
        pl.when(pl.col("variable") == "SEXO").then(pl.lit("verificada en fuente"))
        .when(pl.col("variable") == "TIPO_INGRESO").then(pl.lit("segura"))
        .otherwise(pl.col("certeza")).alias("certeza"),
    )
    problemas = " | ".join(conocimiento.validar(malo))
    assert "EDAD: fuente no reconocida" in problemas
    assert "SEXO: sin referencia verificada exige certeza pendiente" in problemas
    assert "TIPO_INGRESO: certeza inválida" in problemas
    assert conocimiento.validar(df.drop("evidencia_eda"))[0].startswith("las columnas deben ser")
    assert any("faltan categorías" in p for p in conocimiento.validar(df.filter(~pl.col("variable").str.starts_with("ELIX_0"))))


# ------------------------------------------------------------ corrida con datos sintéticos
@pytest.fixture(scope="module")
def corrida(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("t8")
    fuente = crear_fuente_sintetica(raiz / "datos", n=700, semilla=29)
    ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=raiz / "salidas")
    return ctx, cm.ejecutar(ctx)


def test_el_mapeo_de_tres_caracteres_es_un_subconjunto_del_completo_y_no_ve_la_diabetes(corrida):
    _, r = corrida
    v = r["vs_3c"]
    assert (v["n_3c_ah"] <= v["n_completo_ah"]).all() and (v["n_3c_todos"] <= v["n_completo_todos"]).all()
    dm = v.filter(pl.col("elix_id") == "ELIX_11").row(0, named=True)
    assert dm["n_completo_ah"] > 0 and dm["n_3c_ah"] == 0 and dm["supera_umbral"] == "si"     # los códigos sintéticos de diabetes tienen cuatro caracteres
    ht = v.filter(pl.col("elix_id") == "ELIX_06").row(0, named=True)
    assert ht["n_completo_ah"] == ht["n_3c_ah"] > 0                                             # la hipertensión se define con tres caracteres


def test_la_prevalencia_agrupada_es_la_suma_de_los_anios(corrida):
    _, r = corrida
    p = r["prevalencia"]
    fila = p.filter((pl.col("periodo") == "2019-2024") & (pl.col("elix_id") == "ELIX_06")).row(0, named=True)
    por_anio = r["por_anio"]
    assert fila["n"] == sum(d["sec"]["ELIX_06"] for d in por_anio.values()) and fila["n_base"] == sum(d["n"] for d in por_anio.values())
    assert fila["prevalencia"] == pytest.approx(fila["n"] / fila["n_base"]) and fila["ic_bajo"] < fila["prevalencia"] < fila["ic_alto"]
    assert set(p["periodo"].unique()) >= {"2019", "2024", "2019-2024", "2019-2022", "2019-2023"}
    assert sorted(p.filter(pl.col("periodo") == "2019-2024")["rango"].to_list()) == list(range(1, 32))


def test_phi_de_la_corrida_es_simetrico_y_esta_acotado(corrida):
    _, r = corrida
    ph = r["phi"]
    assert set(ph["periodo"].unique()) == {"2019-2022", "2023"}
    s = ph.filter(pl.col("periodo") == "2019-2022")
    assert s.height == 31 * 31
    m = {(a, b): p for a, b, p in zip(s["elix_a"], s["elix_b"], s["phi"])}
    assert all(abs((m[(a, b)] or 0) - (m[(b, a)] or 0)) < 1e-12 for a, b in m)
    assert all(p is None or -1.0 - 1e-9 <= p <= 1.0 + 1e-9 for p in m.values())
    # la diagonal es la prevalencia en ese período (n_a)
    diag = s.filter(pl.col("elix_a") == pl.col("elix_b"))
    esperado = {i: sum(r["por_anio"][a]["sec"][i] for a in (2019, 2020, 2021, 2022)) for i in r["e"].ids}
    assert dict(zip(diag["elix_a"], diag["n_a"])) == esperado


def test_el_total_nacional_de_la_deriva_suma_los_hospitales(corrida):
    _, r = corrida
    d = r["deriva"]
    for anio, g in d.group_by("anio"):
        nac = g.filter(pl.col("cod_hospital") == cm.TODOS)
        hosp = g.filter(pl.col("cod_hospital") != cm.TODOS)
        assert int(nac["n"][0]) == int(hosp["n"].sum()) and int(nac["suma"][0]) == int(hosp["suma"].sum()), anio
        assert nac["media"][0] == pytest.approx(hosp["suma"].sum() / hosp["n"].sum())


def test_esta_tarea_no_toca_el_desenlace_y_usa_los_seis_anios_como_estructura(corrida):
    ctx, _ = corrida
    reg = ctx.cortafuegos.registro().filter(pl.col("tarea") == "T08")
    assert set(reg["anio"].to_list()) == {2019, 2020, 2021, 2022, 2023, 2024}
    assert set(reg["proposito"].to_list()) == {"estructura"} and reg["columnas_resultado"].fill_null("").str.len_chars().sum() == 0
    assert ctx.cortafuegos.violaciones() == 0 and ctx.cortafuegos.n_bloqueos() == 0


def test_ejecutar_escribe_tablas_figuras_metricas_y_cumple_privacidad(corrida):
    ctx, _ = corrida
    for nombre in ("t23_elix_prevalencia", "t23_elix_vs_3c", "t23_elix_principal", "t23_elix_phi", "t23_deriva_codificacion", "t23_resumen_comorbilidad"):
        assert (ctx.dir_tablas / f"{nombre}.csv").is_file(), nombre
    assert ctx.figuras.ids() == ["F08_01", "F08_02", "F08_03"]
    idx = ctx.figuras.indice()
    assert idx.filter(pl.col("id_figura") == "F08_02")["uso_interno"][0] == "si" and idx.filter(pl.col("id_figura") == "F08_01")["uso_interno"][0] == "no"
    m = ctx.metricas().filter(pl.col("tarea") == "T08")
    claves = set(m["clave"].to_list())
    for k in ("elix.prev_pct.ELIX_06", "elix.prev_pct_todos.ELIX_06", "elix.prev_pct_3c_todos.ELIX_06", "elix.prev_pct.diabetes", "elix.prev_pct_3c_todos.diabetes",
              "elix.n_categorias", "codificacion.media_dx_sec_adultos_hosp.2019", "codificacion.media_dx_sec_adultos_hosp.2024", "codificacion.aumento_pct_2019_2024"):
        assert k in claves, k
    val = dict(zip(m["clave"], m["valor"]))
    assert val["elix.n_categorias"] == 31 and val["elix.prev_pct_3c_todos.diabetes"] == 0.0 and val["elix.prev_pct.diabetes"] > 0
    ctx.escribir_registros()
    esc = privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)
    assert privacidad.total_violaciones(esc) == 0, esc.filter(pl.col("n_violaciones") > 0)


def test_el_informe_agrega_las_secciones_de_comorbilidad(corrida):
    ctx, _ = corrida
    inf = Informe(ctx, "t23_prueba_comorbilidad.md", "Informe de prueba")
    cm.agregar_secciones(inf, ctx, n0=3)
    md = inf.guardar().read_text(encoding="utf8")
    for titulo in ("3. Prevalencia de las categorías de Elixhauser", "4. Mapeo completo frente al de tres caracteres", "5. Diagnóstico principal y regla R3",
                   "6. Asociación entre categorías", "7. Deriva de la codificación"):
        assert titulo in md, titulo
    for fig in ("F08_01", "F08_02", "F08_03"):
        assert f"Figura {fig}." in md
    assert "{" not in md and "}" not in md


# ------------------------------------------------------------ B0: orden estable de las líneas de F08_02 (DT11)
COLS_DERIVA = ("anio", "cod_hospital", "n", "suma", "media", "desv", "ic_bajo", "ic_alto", "cambio_relativo", "cambio_bajo", "cambio_alto")
ANIOS_DERIVA = (2019, 2020, 2021)
GRIS = "#9A9A9A"


def _deriva_a_mano(hospitales: dict[str, tuple[int, float]]) -> pl.DataFrame:
    """Tabla ``t23_deriva_codificacion.csv`` hecha a mano: el total nacional y un hospital por entrada ``código: (egresos por año, media del primer año)``.

    La media de cada hospital sube 0,1 por año, de modo que la serie de cada línea identifica al hospital.
    """
    filas = []
    for k, anio in enumerate(ANIOS_DERIVA):
        filas.append({"anio": anio, "cod_hospital": cm.TODOS, "n": 5000, "suma": 20000, "media": 4.0 + 0.1 * k, "desv": 3.0, "ic_bajo": 3.9 + 0.1 * k, "ic_alto": 4.1 + 0.1 * k,
                      "cambio_relativo": None if k == 0 else 0.1 * k, "cambio_bajo": None if k == 0 else 0.05 * k, "cambio_alto": None if k == 0 else 0.15 * k})
        for cod, (n, media0) in hospitales.items():
            media = media0 + 0.1 * k
            filas.append({"anio": anio, "cod_hospital": cod, "n": n, "suma": round(n * media), "media": media, "desv": 2.0, "ic_bajo": media - 0.1, "ic_alto": media + 0.1,
                          "cambio_relativo": None if k == 0 else 0.1 * k, "cambio_bajo": None if k == 0 else 0.05 * k, "cambio_alto": None if k == 0 else 0.15 * k})
    return pl.DataFrame(filas, schema={c: (pl.String if c == "cod_hospital" else pl.Int64 if c in ("anio", "n", "suma") else pl.Float64) for c in COLS_DERIVA})


def _lineas_grises(ctx, tabla: pl.DataFrame, monkeypatch, *, invertir_grupos: bool = False) -> list[tuple[float, ...]]:
    """Dibuja F08_02 con la tabla dada y devuelve, en orden, la serie de cada trazo gris (un espía de ``Axes.plot``).

    Con ``invertir_grupos`` el ``group_by`` de polars entrega los grupos al revés cuando no se le pide ``maintain_order=True``: así se reproduce el orden arbitrario de otra corrida.
    """
    import matplotlib.axes

    ctx.escribir_tabla("t23_deriva_codificacion.csv", tabla, tarea=8, descripcion="deriva a mano", cols_n=["n"],
                       dependientes=["suma", "media", "desv", "ic_bajo", "ic_alto", "cambio_relativo", "cambio_bajo", "cambio_alto"])
    trazos: list[tuple[float, ...]] = []
    original = matplotlib.axes.Axes.plot

    def espia(self, *args, **kwargs):
        if kwargs.get("color") == GRIS:
            trazos.append(tuple(float(v) for v in args[1]))
        return original(self, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(matplotlib.axes.Axes, "plot", espia)
        if invertir_grupos:
            agrupar = pl.DataFrame.group_by

            class Invertido:
                def __init__(self, grupos):
                    self._grupos = list(grupos)

                def __iter__(self):
                    return iter(reversed(self._grupos))

            def group_by_invertido(self, *by, maintain_order=False, **kw):
                gb = agrupar(self, *by, maintain_order=maintain_order, **kw)
                return gb if maintain_order else Invertido(gb)

            m.setattr(pl.DataFrame, "group_by", group_by_invertido)
        cm.figura_codificacion(ctx)
    return trazos


def test_las_lineas_grises_de_f08_02_siguen_el_orden_de_la_tabla_y_no_el_de_los_grupos(fuente_sintetica, tmp_path, monkeypatch):
    hospitales = {"103100": (150, 5.0), "101100": (150, 3.0), "104100": (400, 4.5), "102100": (120, 6.0)}
    tabla = _deriva_a_mano(hospitales)
    normal = _lineas_grises(Contexto.crear("extracto", fuente=fuente_sintetica, dir_salida_base=tmp_path / "a"), tabla, monkeypatch)
    al_reves = _lineas_grises(Contexto.crear("extracto", fuente=fuente_sintetica, dir_salida_base=tmp_path / "b"), tabla, monkeypatch, invertir_grupos=True)
    esperado = [tuple(round(m0 + 0.1 * k, 10) for k in range(3)) for m0 in (5.0, 3.0, 4.5, 6.0)]        # el orden en que aparecen los hospitales en la tabla
    assert [tuple(round(v, 10) for v in t) for t in normal] == esperado
    assert al_reves == normal


def test_f08_02_solo_dibuja_los_hospitales_con_al_menos_el_minimo_de_egresos_cada_anio(fuente_sintetica, tmp_path, monkeypatch):
    minimo = cm.MIN_EPISODIOS_LINEA
    # justo en el mínimo se dibuja; un egreso menos, no; y dos hospitales con la misma media dan dos trazos iguales
    hospitales = {"101100": (minimo, 3.0), "102100": (minimo - 1, 4.0), "103100": (minimo, 3.0)}
    trazos = _lineas_grises(Contexto.crear("extracto", fuente=fuente_sintetica, dir_salida_base=tmp_path / "a"), _deriva_a_mano(hospitales), monkeypatch)
    assert len(trazos) == 2 and trazos[0] == trazos[1]
    # sin hospitales sobre el mínimo la figura se guarda igual, con cero líneas
    ctx = Contexto.crear("extracto", fuente=fuente_sintetica, dir_salida_base=tmp_path / "b")
    assert _lineas_grises(ctx, _deriva_a_mano({"101100": (minimo - 1, 3.0)}), monkeypatch) == []
    assert (ctx.dir_figuras / "F08_02.png").is_file() and "F08_02" in ctx.figuras.ids()


def test_el_rango_de_la_prevalencia_desempata_por_categoria_aunque_cambie_el_orden_de_las_filas(e, u):
    """Las filas de los períodos agregados salen de un group_by: con empates en n el rango dependía de ese orden y cambiaba entre corridas (DT11)."""
    from ayudas_t13 import grupos_al_reves

    sec = {i: 0 for i in e.ids} | {"ELIX_01": 30, "ELIX_02": 20, "ELIX_03": 20, "ELIX_04": 10}
    por_anio = {a: {"n": 100, "sec": dict(sec)} for a in range(2019, 2024)}
    normal = cm.tabla_prevalencia(por_anio, e, u)
    with grupos_al_reves():
        al_reves = cm.tabla_prevalencia(por_anio, e, u)
    assert al_reves.equals(normal)
    # a mano, en el período agregado: 150, 100, 100 y 50 egresos y 27 categorías sin egresos; los empates siguen el orden de la categoría
    agregado = normal.filter(pl.col("periodo") == "2019-2023").sort("rango")
    assert agregado["rango"].to_list() == list(range(1, 32))
    assert agregado["elix_id"].to_list() == ["ELIX_01", "ELIX_02", "ELIX_03", "ELIX_04"] + [f"ELIX_{i:02d}" for i in range(5, 32)]
    assert agregado["n"].to_list()[:5] == [150, 100, 100, 50, 0]
    # en un año suelto vale lo mismo
    anual = normal.filter(pl.col("periodo") == "2020").sort("rango")
    assert anual["elix_id"].to_list()[:5] == ["ELIX_01", "ELIX_02", "ELIX_03", "ELIX_04", "ELIX_05"]
