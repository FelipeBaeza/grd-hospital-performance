"""Tarea 9: hospitales, identificador del paciente, historial de doce meses, traslados, procedencia y embudo crudo."""

from __future__ import annotations

import dataclasses
import datetime as dt
import math

import polars as pl
import pytest

from eda import catalogos, cohorte, config, privacidad
from eda import hospitales as hp
from eda.contexto import Contexto
from eda.informes import Informe
from sintetico import crear_fuente_sintetica


@pytest.fixture(scope="module")
def u():
    """Umbrales del proyecto. Las cuentas a mano de este archivo suponen los valores propuestos; si cambian, la prueba lo dice."""
    u = config.cargar_umbrales()
    supuestos = {
        "celda_minima": 10, "edad_adulto": 18, "volumen_minimo_egresos": 500, "muertes_esperadas_minimas": 10, "volumen_grid_egresos": [200, 500, 1000],
        "volumen_grid_muertes": [5, 10, 20], "hospital_pediatrico_prop": 0.5, "historial_factibilidad_minima": 0.8, "ventana_historial_dias": 365,
        "ventanas_traslado_dias": [1, 2, 7, 30], "winsor": 0.10, "autorreferencia_procedencia_max": 0.05, "enlace_traslado_minimo": 0.30,
    }
    for clave, valor in supuestos.items():
        assert u[clave] == valor, f"las cuentas a mano de esta prueba suponen {clave} = {valor}"
    return u


@pytest.fixture(scope="module")
def cat_chico():
    """Catálogo reducido con tres hospitales, para no depender del contenido de Hospitales.csv."""
    base = catalogos.cargar_catalogos()
    hosp = pl.DataFrame({
        "cod_hospital": ["100001", "100002", "100003"],
        "nombre": ["Hospital Clínico Regional de Valdivia Dr. Mario Muñoz", "Hospital de Linares", "Clínica San José"],
        "tipo_por_nombre": ["publico_probable", "publico_probable", "privado_probable"],
    })
    return dataclasses.replace(base, hospitales=hosp, hospitales_set=frozenset(hosp["cod_hospital"].to_list()))


def _d(texto: str | None) -> dt.date | None:
    return dt.date.fromisoformat(texto) if texto else None


def _wilson(k: int, n: int, z: float) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    mitad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centro - mitad, centro + mitad


# ------------------------------------------------------------------------------ nombres
def test_normalizar_nombre_quita_tildes_signos_y_espacios_repetidos():
    assert hp.normalizar_nombre("Hospital Clínico  San-José (Sur)") == "HOSPITAL CLINICO SAN JOSE SUR"
    assert hp.normalizar_nombre("Hospital de Ñuble") == "HOSPITAL DE NUBLE"
    assert hp.normalizar_nombre("  ") == ""


def test_similitud_acepta_una_palabra_de_diferencia_pero_no_otro_hospital():
    assert hp.similitud_nombres("HOSPITAL DE TALCA", "HOSPITAL DE TALCA") == 1.0
    assert hp.similitud_nombres("HOSPITAL REGIONAL DE TALCA", "HOSPITAL REGIONAL TALCA") >= hp.SIMILITUD_MINIMA
    assert hp.similitud_nombres("HOSPITAL DE TALCA", "HOSPITAL DE LINARES") < hp.SIMILITUD_MINIMA


def test_el_mapa_de_nombres_va_por_codigo_y_con_nombres_normalizados(cat_chico):
    m = hp.mapa_nombres(cat_chico)
    assert m == {"100001": "HOSPITAL CLINICO REGIONAL DE VALDIVIA DR MARIO MUNOZ", "100002": "HOSPITAL DE LINARES", "100003": "CLINICA SAN JOSE"}


# ------------------------------------------------------------------ agregados de un año
def _base_a_mano() -> pl.DataFrame:
    filas = [
        # hospital, edad, sexo, actividad, procedencia, hospproc, id, ingreso, alta, D, B, conocido, rol, etiqueta
        ("H1", 30, "MUJER", "HOSPITALIZACIÓN", "OTROS HOSPITALES DE LA RED", "HOSP X", 1, "2022-03-01", "2022-03-05", True, True, True, "EVENTO", "FALLECIDO"),
        ("H1", 40, "HOMBRE", "HOSPITALIZACIÓN", "SERVICIO EMERGENCIA (DOMICILIO)", None, 2, "2022-03-02", "2022-03-06", True, True, True, "NO_EVENTO", "DOMICILIO"),
        ("H1", 10, "HOMBRE", "HOSPITALIZACIÓN", "SERVICIO EMERGENCIA (DOMICILIO)", None, 3, "2022-03-03", "2022-03-07", False, False, True, "NO_EVENTO", "DOMICILIO"),
        ("H2", 50, "MUJER", "CIRUGÍA MAYOR AMBULATORIA (CMA)", "OTROS HOSPITALES RED NACIONAL", "HOSP Y", None, "2022-04-01", "2022-04-01", False, True, True, "NO_EVENTO", "DOMICILIO"),
        ("H2", 60, "MUJER", "HOSPITALIZACIÓN", "OTROS HOSPITALES RED NACIONAL", "HOSP Y", 5, "2022-04-02", "2022-04-09", True, True, True, "NO_EVENTO",
         "DERIVACIÓN OTRO HOSPITAL DE LA RED NACIONAL"),
        ("H2", None, "HOMBRE", "HOSPITALIZACIÓN", "DESCONOCIDO", None, 6, "2022-04-03", "2022-04-10", False, False, False, "SIN_DATO", "(sin rol)"),
        ("H2", 18, "HOMBRE", "HOSPITALIZACIÓN", "SERVICIO EMERGENCIA (DOMICILIO)", None, 7, "2022-04-04", "2022-04-11", True, True, True, "NO_EVENTO",
         "DERIVACIÓN OTRO HOSPITAL DEL SERVICIO"),
    ]
    return pl.DataFrame(
        {
            "hospital": [f[0] for f in filas], "edad": [f[1] for f in filas], "sexo": [f[2] for f in filas], "actividad": [f[3] for f in filas],
            "procedencia": [f[4] for f in filas], "hospproc": [f[5] for f in filas], "id_num": [f[6] for f in filas], "ing": [_d(f[7]) for f in filas],
            "alta": [_d(f[8]) for f in filas], "anio": [2022] * len(filas), "D": [f[9] for f in filas], "B": [f[10] for f in filas],
            "conocido": [f[11] for f in filas], "rol": [f[12] for f in filas], "etiqueta_roles": [f[13] for f in filas],
        },
        schema={
            "hospital": pl.String, "edad": pl.Int64, "sexo": pl.String, "actividad": pl.String, "procedencia": pl.String, "hospproc": pl.String, "id_num": pl.Int64,
            "ing": pl.Date, "alta": pl.Date, "anio": pl.Int64, "D": pl.Boolean, "B": pl.Boolean, "conocido": pl.Boolean, "rol": pl.String, "etiqueta_roles": pl.String,
        },
    )


def test_procesar_anio_cuenta_hospitales_procedencia_y_cohorte_a_mano():
    d = hp.procesar_anio(_base_a_mano(), 18)
    assert d["n"] == 7
    h = {r["hospital"]: r for r in d["hospitales"].iter_rows(named=True)}
    assert (h["H1"]["n_egresos"], h["H1"]["n_adultos"], h["H1"]["n_menores"], h["H1"]["n_edad_desconocida"]) == (3, 2, 1, 0)
    assert (h["H2"]["n_egresos"], h["H2"]["n_adultos"], h["H2"]["n_menores"], h["H2"]["n_edad_desconocida"]) == (4, 3, 0, 1)   # los 18 años cuentan como adultos
    assert dict(d["procedencia_total"].iter_rows()) == {
        "OTROS HOSPITALES DE LA RED": 1, "SERVICIO EMERGENCIA (DOMICILIO)": 3, "OTROS HOSPITALES RED NACIONAL": 2, "DESCONOCIDO": 1,
    }
    nombres = {(r["procedencia"], r["hospital"], r["hospproc"]): r["n"] for r in d["procedencia_nombres"].iter_rows(named=True)}
    assert nombres == {("OTROS HOSPITALES DE LA RED", "H1", "HOSP X"): 1, ("OTROS HOSPITALES RED NACIONAL", "H2", "HOSP Y"): 2}
    assert sorted(d["id_validos"].to_list()) == [1, 2, 3, 5, 6, 7]
    # cohorte del modelo (D): solo adultos hospitalizados con desenlace conocido
    vol = {r["hospital"]: (r["n_D"], r["k_D"]) for r in d["hospitales_d"].iter_rows(named=True)}
    assert vol == {"H1": (2, 1), "H2": (2, 0)}
    estratos = {(r["hospital"], r["grupo_edad"], r["sexo"]): (r["n"], r["k"]) for r in d["estratos_d"].iter_rows(named=True)}
    assert estratos == {("H1", "25 a 34", "MUJER"): (1, 1), ("H1", "35 a 44", "HOMBRE"): (1, 0), ("H2", "55 a 64", "MUJER"): (1, 0), ("H2", "15 a 24", "HOMBRE"): (1, 0)}
    # entrantes: procedencia que contiene "OTROS HOSPITALES", adultos de la cohorte dura con desenlace conocido
    ent = {r["entrante"]: (r["n"], r["k"]) for r in d["entrantes"].iter_rows(named=True)}
    assert ent == {True: (3, 1), False: (2, 0)}
    # episodios para el historial y los traslados
    ep = d["episodios"]
    assert ep["es_ah"].to_list() == [True, True, False, False, True, False, True]
    assert ep["salida"].to_list() == [False, False, False, False, True, False, True]            # derivación a otro hospital de un adulto hospitalizado
    assert ep["entrante_ah"].to_list() == [True, False, False, False, True, False, False]       # el entrante ambulatorio no cuenta
    assert ep["anio"].dtype == pl.Int32


def test_un_anio_sin_desenlace_solo_aporta_estructura():
    sin = _base_a_mano().drop("rol", "D", "B", "conocido", "etiqueta_roles")
    d = hp.procesar_anio(sin, 18)
    assert set(d) == {"n", "hospitales", "procedencia_total", "procedencia_nombres", "id_validos"}


# --------------------------------------------------------------------------- hospitales
def test_el_perfil_marca_pediatricos_con_umbral_estricto_y_hospitales_fuera_del_catalogo(cat_chico, u):
    def hosp(filas):
        return pl.DataFrame(filas, schema={"hospital": pl.String, "n_egresos": pl.UInt32, "n_adultos": pl.UInt32, "n_menores": pl.UInt32, "n_edad_desconocida": pl.UInt32},
                            orient="row")

    por_anio = {
        2024: {"hospitales": hosp([("100001", 80, 10, 70, 0)])},
        2021: {"hospitales": hosp([("100001", 100, 40, 60, 0), ("100002", 100, 50, 50, 0), ("999999", 10, 5, 5, 0)])},
    }
    t = hp.tabla_hospitales_anio(por_anio, cat_chico, u)
    assert list(zip(t["anio"], t["cod_hospital"])) == [(2021, "100001"), (2021, "100002"), (2021, "999999"), (2024, "100001")]
    f = {(r["anio"], r["cod_hospital"]): r for r in t.iter_rows(named=True)}
    assert f[(2021, "100001")]["pediatrico"] == "si" and f[(2021, "100001")]["prop_menores"] == pytest.approx(0.6)
    assert f[(2021, "100002")]["pediatrico"] == "no"          # la mitad exacta no supera el umbral
    assert f[(2021, "999999")]["en_catalogo"] == "no" and f[(2021, "100001")]["en_catalogo"] == "si"
    assert f[(2024, "100001")]["pediatrico"] == "si"


def test_muertes_esperadas_por_estandarizacion_indirecta_a_mano():
    est = pl.DataFrame({
        "hospital": ["H1", "H1", "H2", "H2"], "grupo_edad": ["25 a 34", "65 a 74", "25 a 34", "65 a 74"], "sexo": ["MUJER", "HOMBRE", "MUJER", "HOMBRE"],
        "n": [10, 10, 30, 10], "k": [1, 4, 2, 1],
    })
    esp = {r["hospital"]: r["esperadas"] for r in hp.esperadas_por_hospital(est).iter_rows(named=True)}
    # tasas nacionales: 3 de 40 (0,075) en mujeres de 25 a 34 y 5 de 20 (0,25) en hombres de 65 a 74
    assert esp["H1"] == pytest.approx(10 * 0.075 + 10 * 0.25)
    assert esp["H2"] == pytest.approx(30 * 0.075 + 10 * 0.25)
    assert sum(esp.values()) == pytest.approx(est["k"].sum())        # la suma de lo esperado es lo observado


def _anio_con_volumen() -> dict:
    # un solo estrato, así que lo esperado es el volumen por la tasa nacional (39 de 1.005)
    return {
        "hospitales_d": pl.DataFrame({"hospital": ["H1", "H2", "H3"], "n_D": [600, 400, 5], "k_D": [30, 8, 1]}),
        "estratos_d": pl.DataFrame({"hospital": ["H1", "H2", "H3"], "grupo_edad": ["65 a 74"] * 3, "sexo": ["HOMBRE"] * 3, "n": [600, 400, 5], "k": [30, 8, 1]}),
    }


def test_volumen_elegible_y_cumplimiento_del_minimo_propuesto(u):
    vol = hp.tabla_hospitales_volumen({2022: _anio_con_volumen(), 2024: {"hospitales": pl.DataFrame()}}, u)
    assert vol["anio"].unique().to_list() == [2022]                    # el año expuesto no tiene cohorte de resultado
    f = {r["cod_hospital"]: r for r in vol.iter_rows(named=True)}
    tasa = 39 / 1005
    assert f["H1"]["esperados_D"] == pytest.approx(600 * tasa) and f["H3"]["esperados_D"] == pytest.approx(5 * tasa)
    assert (f["H1"]["cumple_egresos"], f["H1"]["cumple_muertes"], f["H1"]["cumple_volumen"]) == ("si", "si", "si")
    assert (f["H2"]["cumple_egresos"], f["H2"]["cumple_muertes"], f["H2"]["cumple_volumen"]) == ("no", "si", "no")
    assert (f["H3"]["cumple_egresos"], f["H3"]["cumple_muertes"], f["H3"]["cumple_volumen"]) == ("no", "no", "no")
    sin = hp.tabla_hospitales_volumen({2024: {"hospitales": pl.DataFrame()}}, u)
    assert sin.height == 0 and "n_D" in sin.columns


def test_los_minimos_de_volumen_se_cumplen_con_igualdad(u):
    # tasa nacional de 20 en 1.280 (1/64, exacta en binario): A tiene justo 500 egresos y 7,8 esperadas; B tiene 640 egresos y justo 10 esperadas
    anio = {
        "hospitales_d": pl.DataFrame({"hospital": ["A", "B", "C"], "n_D": [500, 640, 140], "k_D": [5, 10, 5]}),
        "estratos_d": pl.DataFrame({"hospital": ["A", "B", "C"], "grupo_edad": ["65 a 74"] * 3, "sexo": ["HOMBRE"] * 3, "n": [500, 640, 140], "k": [5, 10, 5]}),
    }
    f = {r["cod_hospital"]: r for r in hp.tabla_hospitales_volumen({2022: anio}, u).iter_rows(named=True)}
    assert f["B"]["esperados_D"] == 10.0 and f["A"]["esperados_D"] == pytest.approx(7.8125)
    assert (f["A"]["cumple_egresos"], f["A"]["cumple_muertes"], f["A"]["cumple_volumen"]) == ("si", "no", "no")
    assert (f["B"]["cumple_egresos"], f["B"]["cumple_muertes"], f["B"]["cumple_volumen"]) == ("si", "si", "si")


def test_cuadricula_de_volumen_y_resumen_por_anio(u):
    vol = hp.tabla_hospitales_volumen({2022: _anio_con_volumen()}, u)
    hosp = pl.DataFrame({
        "anio": [2022, 2022, 2022, 2024], "cod_hospital": ["H1", "H2", "H3", "H1"], "en_catalogo": ["si", "si", "no", "si"], "pediatrico": ["si", "no", "no", "si"],
    })
    g = hp.tabla_volumen_umbrales(vol, hosp, u)
    assert g.height == 9 and set(g["anio"]) == {2022}

    def fila(egresos: int, muertes: int):
        r = g.filter((pl.col("egresos_min") == egresos) & (pl.col("muertes_esperadas_min") == muertes)).row(0, named=True)
        return tuple(r[c] for c in ("n_hospitales", "n_cumplen_egresos", "n_cumplen_muertes", "n_cumplen_ambos", "n_cumplen_ambos_no_pediatricos"))

    # esperadas: 23,3 (H1), 15,5 (H2) y 0,2 (H3); H1 es pediátrico
    assert fila(500, 10) == (3, 1, 2, 1, 0)
    assert fila(200, 5) == (3, 2, 2, 2, 1)
    assert fila(200, 20) == (3, 2, 1, 1, 0)
    assert fila(1000, 20) == (3, 0, 1, 0, 0)
    res = {r["anio"]: r for r in hp.tabla_hospitales_resumen(hosp, vol, u).iter_rows(named=True)}
    assert (res[2022]["n_hospitales"], res[2022]["n_fuera_catalogo"], res[2022]["n_pediatricos"], res[2022]["n_hospitales_con_cohorte"], res[2022]["n_cumplen_volumen"]) == (3, 1, 1, 3, 1)
    assert res[2022]["prop_celdas_bajo_minimo"] == pytest.approx(1 / 3)                 # H3 tiene menos de diez egresos elegibles
    assert res[2024]["n_hospitales"] == 1 and res[2024]["n_hospitales_con_cohorte"] is None and res[2024]["n_cumplen_volumen"] is None


# ------------------------------------------------------------------------------- embudo
def _anio_embudo(ns: list[int], ks: list[int]) -> dict:
    return {"hospitales_d": pl.DataFrame({"hospital": [f"H{i:02d}" for i in range(len(ns))], "n_D": ns, "k_D": ks})}


def test_embudo_marca_los_hospitales_fuera_de_los_limites_sin_sobredispersion(u):
    e = hp.tabla_embudo({2022: _anio_embudo([100] * 12, [5] * 10 + [0, 20])}, u)
    assert e["periodo"].unique().to_list() == ["2022"]                   # con un solo año no hay período agrupado
    f = {r["cod_hospital"]: r for r in e.iter_rows(named=True)}
    p0 = 70 / 1200
    z95 = float(u["z_95"])
    assert f["H00"]["p0"] == pytest.approx(p0) and f["H00"]["phi"] == 1.0   # los dos extremos winsorizados dejan z de -0,36 en todos: sin sobredispersión
    assert f["H11"]["z"] == pytest.approx((0.20 - p0) / math.sqrt(p0 * (1 - p0) / 100))
    assert (f["H11"]["fuera_95"], f["H11"]["fuera_998"]) == ("alto", "alto")
    assert (f["H10"]["fuera_95"], f["H10"]["fuera_998"]) == ("bajo", "no")   # sin muertes: bajo el 95 %, pero el límite de 99,8 % se corta en cero
    assert all(f[f"H{i:02d}"]["fuera_95"] == "no" for i in range(10))
    assert f["H11"]["lim95_alto"] == pytest.approx(p0 + z95 * math.sqrt(p0 * (1 - p0) / 100))
    assert f["H10"]["lim998_bajo"] == 0.0


def test_la_sobredispersion_ensancha_los_limites_y_evita_falsas_alarmas(u):
    e = hp.tabla_embudo({2022: _anio_embudo([200] * 10, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])}, u)
    phi = float(e["phi"][0])
    assert phi == pytest.approx(2.5589, abs=1e-3)                      # cuenta a mano: 10 puntajes z con los dos extremos reemplazados por sus vecinos
    z95 = float(u["z_95"])
    assert int((e["z"].abs() > z95).sum()) == 4                         # sin ajustar, cuatro hospitales quedarían fuera del 95 %
    assert set(e["fuera_95"]) == {"no"} and set(e["fuera_998"]) == {"no"}
    assert bool((e["z_ajustado"].abs() < z95).all())
    assert e["lim95_alto"][0] == pytest.approx(0.055 + z95 * math.sqrt(phi * 0.055 * 0.945 / 200))
    assert e["p0"][0] == pytest.approx(0.055)


def test_el_embudo_deja_sin_evaluar_las_celdas_pequenas_y_agrupa_los_anios(u):
    anio = _anio_embudo([300, 250, 200, 150, 5], [30, 5, 25, 3, 1])
    e = hp.tabla_embudo({2022: anio, 2023: anio, 2024: {"hospitales": pl.DataFrame()}}, u)
    assert sorted(e["periodo"].unique().to_list()) == ["2022", "2022-2023", "2023"]       # el año expuesto no entra
    h = lambda periodo, cod: e.filter((pl.col("periodo") == periodo) & (pl.col("cod_hospital") == cod)).row(0, named=True)  # noqa: E731
    chico = h("2022", "H04")
    assert (chico["fuera_95"], chico["fuera_998"]) == ("no evaluable", "no evaluable") and chico["p"] == pytest.approx(0.2)
    # la tasa nacional cuenta los cinco hospitales, pero la sobredispersión sale solo de los cuatro evaluables
    p0 = 64 / 905
    z = [(k / n - p0) / math.sqrt(p0 * (1 - p0) / n) for n, k in zip([300, 250, 200, 150], [30, 5, 25, 3])]
    phi = sum(v * v for v in z) / 4
    assert phi > 5 and h("2022", "H00")["p0"] == pytest.approx(p0) and h("2022", "H00")["phi"] == pytest.approx(phi)
    # agrupados, el hospital chico suma diez egresos y pasa a ser evaluable
    pool = h("2022-2023", "H04")
    assert (pool["n"], pool["k"]) == (10, 2) and pool["fuera_95"] != "no evaluable"
    assert (h("2022-2023", "H00")["n"], h("2022-2023", "H00")["k"]) == (600, 60)


# ----------------------------------------------------------------- espacios de identificadores
def test_los_espacios_de_identificadores_se_separan_donde_se_rompe_el_traslape(u):
    ids = lambda a, b: pl.Series(list(range(a, b)), dtype=pl.Int64)  # noqa: E731
    dos = {2019: ids(1, 801), 2020: ids(1, 801), 2021: ids(5001, 5801), 2022: ids(5001, 5801), 2023: ids(5001, 5801)}
    assert hp.regimenes_de(dos, u) == [[2019, 2020], [2021, 2022, 2023]]
    assert hp.regimenes_de({a: ids(1, 801) for a in (2019, 2020, 2021)}, u) == [[2019, 2020, 2021]]
    assert hp.regimenes_de({2019: ids(1, 10)}, u) == [[2019]] and hp.regimenes_de({}, u) == []


# ---------------------------------------------------------------------------- historial
FILAS_HISTORIAL = [
    # año, identificador, ingreso, alta
    (2019, 1001, "2019-03-01", "2019-03-05"),   # 0  paciente 1: primer egreso
    (2020, 1001, "2020-01-10", "2020-01-12"),   # 1  egreso previo a 311 días: dentro de la ventana
    (2020, 2002, "2020-05-01", "2020-05-03"),   # 2  paciente 2: sin egreso previo
    (2020, 2002, "2020-05-03", "2020-05-06"),   # 3  reingresa el día del alta: el alta previa no es estrictamente anterior
    (2019, 3003, "2019-01-05", "2019-01-06"),   # 4
    (2020, 3003, "2020-02-01", "2020-02-02"),   # 5  egreso previo a 391 días: fuera de la ventana
    (2020, None, "2020-06-01", "2020-06-03"),   # 6  identificador inválido
    (2020, 5005, "2020-03-01", None),           # 7  sin fecha de alta
    (2020, 5005, "2020-09-01", "2020-09-02"),   # 8  el egreso previo no tiene alta: no cuenta
    (2020, 6006, "2020-03-10", "2020-03-05"),   # 9  alta anterior al ingreso (fechas invertidas)
    (2020, 6006, "2020-04-01", "2020-04-02"),   # 10 el egreso previo con fechas invertidas no cuenta
    (2019, 7007, "2019-07-01", "2019-07-02"),   # 11
    (2020, 7007, "2020-07-01", "2020-07-03"),   # 12 egreso previo a 365 días exactos: dentro de la ventana
    (2019, 8008, "2019-07-01", "2019-07-01"),   # 13 estadía de cero días
    (2020, 8008, "2020-07-01", "2020-07-02"),   # 14 egreso previo a 366 días: fuera de la ventana
]


def _episodios_historial(filas) -> pl.DataFrame:
    return pl.DataFrame(
        {"idx": list(range(len(filas))), "anio": [f[0] for f in filas], "id_num": [f[1] for f in filas], "ing": [_d(f[2]) for f in filas], "alta": [_d(f[3]) for f in filas]},
        schema={"idx": pl.UInt32, "anio": pl.Int32, "id_num": pl.Int64, "ing": pl.Date, "alta": pl.Date},
    )


def test_historial_de_doce_meses_a_mano():
    h = hp.historial_regimen(_episodios_historial(FILAS_HISTORIAL), dt.date(2019, 1, 1), 365)
    f = {r["anio"]: r for r in h.iter_rows(named=True)}
    cols = ("n_episodios", "n_id_valido", "n_ventana_completa", "n_viable", "n_con_previo")
    # 2019 es el primer año del espacio: ningún egreso tiene la ventana completa por construcción
    assert tuple(f[2019][c] for c in cols) == (4, 4, 0, 0, 0)
    # 2020: once egresos, uno sin identificador; los viables con egreso previo estricto dentro de 365 días son solo el 1 y el 12
    assert tuple(f[2020][c] for c in cols) == (11, 10, 11, 10, 2)
    assert h["anio"].to_list() == [2019, 2020]


def test_la_ventana_completa_incluye_el_dia_exacto_del_inicio_del_espacio():
    h = hp.historial_regimen(_episodios_historial(FILAS_HISTORIAL), dt.date(2019, 7, 2), 365)
    f = {r["anio"]: r for r in h.iter_rows(named=True)}
    # con el espacio comenzando el 2 de julio de 2019, solo tienen doce meses completos hacia atrás los ingresos desde el 1 de julio de 2020 (365 días después)
    assert (f[2020]["n_ventana_completa"], f[2020]["n_viable"], f[2020]["n_con_previo"]) == (3, 3, 1)
    assert f[2019]["n_ventana_completa"] == 0


def test_tabla_de_historial_calcula_proporciones_y_el_veredicto_de_la_decision_1(u):
    hist = pl.DataFrame({
        "anio": [2019, 2020, 2021, 2022], "regimen": ["2019-2020", "2019-2020", "2021-2022", "2021-2022"], "n_episodios": [4, 11, 10, 10],
        "n_id_valido": [4, 10, 10, 10], "n_ventana_completa": [0, 11, 0, 10], "n_viable": [0, 10, 0, 8], "n_con_previo": [0, 2, 0, 4],
    })
    f = {r["anio"]: r for r in hp.tabla_historial(hist, u).iter_rows(named=True)}
    assert f[2020]["factibilidad"] == pytest.approx(10 / 11) and f[2020]["cumple_minimo"] == "si"
    assert f[2020]["prop_con_previo"] == pytest.approx(0.2) and f[2020]["prop_id_valido"] == pytest.approx(10 / 11)
    assert f[2019]["factibilidad"] == 0.0 and f[2019]["cumple_minimo"] == "no"
    assert f[2019]["prop_con_previo"] is None or math.isnan(f[2019]["prop_con_previo"])      # nadie es viable
    assert f[2022]["factibilidad"] == pytest.approx(0.8) and f[2022]["cumple_minimo"] == "si"   # el mínimo se alcanza con igualdad


# ----------------------------------------------------------------------------- traslados
FILAS_ENLACES = [
    # año, identificador, ingreso, alta, hospital, salida, entrante
    (2022, 1, "2022-03-05", "2022-03-10", "H1", True, False),         # S1 se enlaza al día siguiente con un ingreso en H2
    (2022, 2, "2022-03-25", "2022-04-01", "H1", True, False),         # S2 se enlaza a los 8 días
    (2022, 3, "2022-04-25", "2022-05-01", "H1", True, False),         # S3 reingresa en el mismo hospital: no es un traslado enlazado
    (2022, 4, "2022-05-25", "2022-06-01", "H1", True, False),         # S4 no vuelve a ingresar
    (2022, None, "2022-06-20", "2022-07-01", "H1", True, False),      # S5 sin identificador
    (2023, 6, "2023-12-15", "2023-12-20", "H1", True, False),         # S6 se da de alta cerca del fin del período observado
    (2022, 7, "2022-07-25", "2022-08-01", "H1", True, False),         # S7 su único otro ingreso es anterior al alta
    (2022, 8, "2022-08-25", "2022-09-01", "H1", True, False),         # S8 ingresa en H2 el mismo día del alta
    (2022, 1, "2022-03-11", "2022-03-15", "H2", False, True),         # D1 entrante que sigue a S1
    (2022, 2, "2022-04-09", "2022-04-12", "H2", False, True),         # D2 entrante que sigue a S2 a los 8 días
    (2022, 3, "2022-05-02", "2022-05-04", "H1", False, True),         # D3 entrante en el mismo hospital que S3
    (2021, 9, "2021-01-05", "2021-01-08", "H2", False, True),         # D4 entrante cerca del inicio del espacio
    (2022, 8, "2022-09-01", "2022-09-03", "H2", False, True),         # D5 entrante que sigue a S8 el mismo día
    (2022, None, "2022-10-01", "2022-10-03", "H2", False, True),      # D6 entrante sin identificador
    (2022, 7, "2022-07-31", "2022-08-02", "H2", False, False),        # D7 ingreso común antes del alta de S7
]


def _episodios_enlaces(filas) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "idx": list(range(len(filas))), "anio": [f[0] for f in filas], "id_num": [f[1] for f in filas], "ing": [_d(f[2]) for f in filas],
            "alta": [_d(f[3]) for f in filas], "hospital": [f[4] for f in filas], "salida": [f[5] for f in filas], "entrante_ah": [f[6] for f in filas],
        },
        schema={"idx": pl.UInt32, "anio": pl.Int32, "id_num": pl.Int64, "ing": pl.Date, "alta": pl.Date, "hospital": pl.String, "salida": pl.Boolean, "entrante_ah": pl.Boolean},
    )


def _cuentas(e: pl.DataFrame, direccion: str, anio: int) -> list[tuple[int, int, int, int]]:
    out = []
    for w in (1, 2, 7, 30):
        f = e.filter((pl.col("direccion") == direccion) & (pl.col("ventana_dias") == w) & (pl.col("anio") == anio))
        assert f.height == 1, (direccion, w, anio)
        r = f.row(0, named=True)
        out.append((r["n_total"], r["n_id_invalido"], r["n_elegibles"], r["n_enlazados"]))
    return out


def test_enlace_de_traslados_a_mano():
    e = hp.enlaces_regimen(_episodios_enlaces(FILAS_ENLACES), dt.date(2021, 1, 1), dt.date(2023, 12, 31), [1, 2, 7, 30])
    # cuentas: (traslados, sin identificador, elegibles, enlazados) para las ventanas de 1, 2, 7 y 30 días
    # salientes de 2022: S1 y S8 se enlazan en cualquier ventana, S2 solo a los 30 días; S3, S4 y S7 no; S5 no tiene identificador
    assert _cuentas(e, "saliente", 2022) == [(7, 1, 6, 2), (7, 1, 6, 2), (7, 1, 6, 2), (7, 1, 6, 3)]
    # el alta del 20 de diciembre de 2023 no se alcanza a observar a 30 días: sale del denominador en lugar de contarse como no enlazada
    assert _cuentas(e, "saliente", 2023) == [(1, 0, 1, 0), (1, 0, 1, 0), (1, 0, 1, 0), (1, 0, 0, 0)]
    # entrantes de 2022: D1 y D5 se enlazan en cualquier ventana, D2 a los 30 días; D3 es del mismo hospital; D6 no tiene identificador
    assert _cuentas(e, "entrante", 2022) == [(5, 1, 4, 2), (5, 1, 4, 2), (5, 1, 4, 2), (5, 1, 4, 3)]
    # el ingreso del 5 de enero de 2021 no tiene siete días de historia observable dentro del espacio
    assert _cuentas(e, "entrante", 2021) == [(1, 0, 1, 0), (1, 0, 1, 0), (1, 0, 0, 0), (1, 0, 0, 0)]


def test_los_identificadores_no_se_enlazan_entre_espacios_distintos(u):
    ep = pl.DataFrame(
        {
            "anio": [2020, 2021], "id_num": [77, 77], "ing": [dt.date(2020, 12, 25), dt.date(2021, 1, 1)], "alta": [dt.date(2020, 12, 30), dt.date(2021, 1, 4)],
            "hospital": ["H1", "H2"], "es_ah": [True, True], "salida": [True, False], "entrante_ah": [False, True],
        },
        schema={"anio": pl.Int32, "id_num": pl.Int64, "ing": pl.Date, "alta": pl.Date, "hospital": pl.String, "es_ah": pl.Boolean, "salida": pl.Boolean, "entrante_ah": pl.Boolean},
    )
    hist, separados = hp.historial_y_enlaces(ep, [[2019, 2020], [2021, 2022]], u)
    assert set(hist["regimen"]) == {"2019-2020", "2021-2022"} and hist["n_episodios"].to_list() == [1, 1]
    sal = separados.filter((pl.col("direccion") == "saliente") & (pl.col("regimen") == "2019-2020")).sort("ventana_dias")
    # el alta del 30 de diciembre solo se observa un día completo dentro de su espacio; el ingreso del otro espacio no se busca
    assert sal["n_elegibles"].to_list() == [1, 0, 0, 0] and sal["n_enlazados"].to_list() == [0, 0, 0, 0]
    ent = separados.filter((pl.col("direccion") == "entrante") & (pl.col("regimen") == "2021-2022")).sort("ventana_dias")
    assert ent["n_elegibles"].to_list() == [0, 0, 0, 0] and ent["n_enlazados"].to_list() == [0, 0, 0, 0]
    _, junto = hp.historial_y_enlaces(ep, [[2019, 2020, 2021, 2022]], u)
    sal_j = junto.filter(pl.col("direccion") == "saliente").sort("ventana_dias")
    assert sal_j["n_elegibles"].to_list() == [1, 1, 1, 1] and sal_j["n_enlazados"].to_list() == [0, 1, 1, 1]      # con un solo espacio habría enlace a los 2 días


def test_tabla_de_enlaces_agrupa_los_anios_y_calcula_el_intervalo_de_wilson(u):
    enl = pl.DataFrame({
        "direccion": ["saliente", "saliente"], "ventana_dias": [30, 30], "anio": [2022, 2023], "n_total": [120, 60], "n_id_invalido": [5, 2], "n_elegibles": [100, 50],
        "n_enlazados": [30, 10], "regimen": ["2021-2023", "2021-2023"],
    })
    t = hp.tabla_enlaces(enl, u)
    assert sorted(t["periodo"].to_list()) == ["2022", "2022-2023", "2023"]
    pool = t.filter(pl.col("periodo") == "2022-2023").row(0, named=True)
    assert (pool["n_total"], pool["n_id_invalido"], pool["n_elegibles"], pool["n_enlazados"]) == (180, 7, 150, 40)
    lo, hi = _wilson(40, 150, float(u["z_95"]))
    assert pool["tasa_enlace"] == pytest.approx(40 / 150) and pool["ic_bajo"] == pytest.approx(lo) and pool["ic_alto"] == pytest.approx(hi)


# --------------------------------------------------------------------- procedencia y entrantes
def test_autorreferencia_compara_nombres_normalizados_y_aproximados(cat_chico):
    nombres = pl.DataFrame(
        [
            ("OTROS HOSPITALES DE LA RED", "100001", "HOSPITAL CLINICO REGIONAL DE VALDIVIA DR MARIO MUNOZ", 5),
            ("OTROS HOSPITALES DE LA RED", "100001", "HOSPITAL DE LINARES", 20),
            ("SERVICIO EMERGENCIA (DOMICILIO)", "100001", "Hospital Clínico Regional de Valdivia Dr. Mario Muñoz  ", 3),
            ("SERVICIO EMERGENCIA (DOMICILIO)", "100001", "HOSPITAL CLINICO REGIONAL DE VALDIVIA DR MARIO MUNOS", 2),
            ("DESCONOCIDO", "999999", "ALGUN HOSPITAL", 4),
        ],
        schema={"procedencia": pl.String, "hospital": pl.String, "hospproc": pl.String, "n": pl.Int64}, orient="row",
    )
    total = pl.DataFrame([("OTROS HOSPITALES DE LA RED", 100), ("SERVICIO EMERGENCIA (DOMICILIO)", 50), ("DESCONOCIDO", 10)], schema={"procedencia": pl.String, "n": pl.Int64}, orient="row")
    t = hp.tabla_autorref({2023: {"procedencia_total": total, "procedencia_nombres": nombres}}, cat_chico)
    f = {r["tipo_procedencia"]: r for r in t.iter_rows(named=True)}
    cols = ("n_episodios", "n_con_hospproc", "n_evaluables", "n_autorref_exacta", "n_autorref_aprox")
    assert tuple(f["OTROS HOSPITALES DE LA RED"][c] for c in cols) == (100, 25, 25, 5, 5)
    assert tuple(f["SERVICIO EMERGENCIA (DOMICILIO)"][c] for c in cols) == (50, 5, 5, 3, 5)         # una letra distinta es aproximada pero no exacta
    assert tuple(f["DESCONOCIDO"][c] for c in cols) == (10, 4, 0, 0, 0)                           # hospital fuera del catálogo: no evaluable
    assert tuple(f[hp.TODAS][c] for c in cols) == (160, 34, 30, 8, 10)
    assert f[hp.TODAS]["prop_autorref_exacta"] == pytest.approx(8 / 30) and f[hp.TODAS]["prop_autorref_aprox"] == pytest.approx(10 / 30)
    assert f[hp.TODAS]["prop_con_hospproc"] == pytest.approx(34 / 160)


def test_mortalidad_de_entrantes_y_razon_contra_los_demas_ingresos(u):
    def anio(n_si, k_si, n_no, k_no):
        return {
            "hospitales_d": pl.DataFrame(schema={"hospital": pl.String, "n_D": pl.Int64, "k_D": pl.Int64}),
            "entrantes": pl.DataFrame({"entrante": [True, False], "n": [n_si, n_no], "k": [k_si, k_no]}),
        }

    t = hp.tabla_entrantes({2022: anio(200, 30, 1800, 100), 2023: anio(100, 10, 900, 50)}, u)
    f = {(r["periodo"], r["entrante"]): r for r in t.iter_rows(named=True)}
    assert set(f) == {(p, e) for p in ("2022", "2023", "2022-2023") for e in ("si", "no")}
    p, q = f[("2022-2023", "si")], f[("2022-2023", "no")]
    assert (p["n"], p["n_fallecidos"], q["n"], q["n_fallecidos"]) == (300, 40, 2700, 150)
    rr = (40 / 300) / (150 / 2700)
    se = math.sqrt(1 / 40 - 1 / 300 + 1 / 150 - 1 / 2700)
    z = float(u["z_95"])
    assert p["mortalidad"] == pytest.approx(40 / 300) and p["rr"] == pytest.approx(rr)
    assert p["rr_bajo"] == pytest.approx(rr * math.exp(-z * se)) and p["rr_alto"] == pytest.approx(rr * math.exp(z * se))
    assert q["rr"] is None and f[("2022", "si")]["rr"] == pytest.approx((30 / 200) / (100 / 1800))


def test_el_resumen_aplica_los_criterios_de_las_decisiones_1_2_y_13(u):
    hist = hp.tabla_historial(pl.DataFrame({
        "anio": [2021, 2022], "regimen": ["2021-2022", "2021-2022"], "n_episodios": [10, 10], "n_id_valido": [10, 10], "n_ventana_completa": [0, 10], "n_viable": [0, 9],
        "n_con_previo": [0, 3],
    }), u)
    enl = hp.tabla_enlaces(pl.DataFrame({
        "direccion": ["saliente"] * 3, "ventana_dias": [7, 30, 30], "anio": [2023, 2022, 2023], "n_total": [60, 120, 60], "n_id_invalido": [2, 5, 2],
        "n_elegibles": [50, 100, 50], "n_enlazados": [10, 30, 10], "regimen": ["R"] * 3,
    }), u)

    def con_autorref(aprox: int) -> pl.DataFrame:
        return pl.DataFrame({"anio": [2023], "tipo_procedencia": [hp.TODAS], "n_evaluables": [100], "n_autorref_exacta": [3], "n_autorref_aprox": [aprox]})

    base = {"historial": hist, "enlaces": enl, "embudo": hp.tabla_embudo({}, u), "por_anio": {}, "regimenes": [[2019, 2020], [2021, 2022, 2023]]}
    s = hp.resumen({**base, "autorref": con_autorref(6)}, u).row(0, named=True)
    assert s["n_regimenes"] == 2 and s["regimenes"] == "2019-2020; 2021-2023"
    assert s["autorref_supera_maximo"] == "si" and s["autorref_prop_aprox"] == pytest.approx(0.06) and s["autorref_prop_exacta"] == pytest.approx(0.03)
    assert hp.resumen({**base, "autorref": con_autorref(5)}, u)["autorref_supera_maximo"][0] == "no"           # el 5 % exacto no supera el máximo
    assert (s["historial_anio"], s["historial_cumple_minimo"]) == (2022, "si") and s["historial_prop_con_previo"] == pytest.approx(3 / 9)
    assert (s["enlace_anio"], s["enlace_ventana_dias"], s["enlace_n_elegibles"], s["enlace_cumple_minimo"]) == (2023, 30, 50, "no")
    assert s["enlace_tasa"] == pytest.approx(0.2) and "embudo_periodo" not in s


# ------------------------------------------------------------ corrida con datos sintéticos
@pytest.fixture(scope="module")
def corrida(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("t9")
    fuente = crear_fuente_sintetica(raiz / "datos", n=700, semilla=29)
    ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=raiz / "salidas")
    return ctx, hp.ejecutar(ctx)


def test_la_corrida_encuentra_los_dos_espacios_de_identificadores_del_generador(corrida):
    _, r = corrida
    assert r["regimenes"] == [[2019, 2020], [2021, 2022, 2023]]
    assert r["resumen"]["regimenes"][0] == "2019-2020; 2021-2023" and r["resumen"]["n_regimenes"][0] == 2


def test_los_regimenes_que_llegan_de_la_tarea_5_se_restringen_a_los_anios_con_desenlace(corrida, tmp_path):
    ctx, _ = corrida
    otro = Contexto.crear("extracto", fuente=ctx.fuente, dir_salida_base=tmp_path)
    r = hp.calcular(otro, regimenes=[[2019, 2020], [2021, 2022, 2023, 2024]])
    assert r["regimenes"] == [[2019, 2020], [2021, 2022, 2023]]
    assert set(r["historial"]["anio"]) == {2019, 2020, 2021, 2022, 2023}


def test_el_anio_expuesto_solo_aporta_perfil_de_hospitales_y_procedencia(corrida):
    ctx, r = corrida
    d24 = r["por_anio"][2024]
    assert set(d24) == {"n", "hospitales", "procedencia_total", "procedencia_nombres", "id_validos"}
    assert set(r["hospitales"]["anio"]) == {2019, 2020, 2021, 2022, 2023, 2024}
    assert set(r["autorref"]["anio"]) == {2019, 2020, 2021, 2022, 2023, 2024}
    for nombre in ("volumen", "umbrales", "historial"):
        assert 2024 not in set(r[nombre]["anio"]), nombre
    assert all(p != "2024" for p in r["embudo"]["periodo"]) and all(p != "2024" for p in r["entrantes"]["periodo"]) and all(p != "2024" for p in r["enlaces"]["periodo"])
    reg = ctx.cortafuegos.registro()
    r24 = reg.filter(pl.col("anio") == 2024)
    assert r24.height > 0 and set(r24["proposito"]) == {"estructura"} and r24["columnas_resultado"].fill_null("").str.len_chars().sum() == 0
    antes = reg.filter(pl.col("anio") < 2024)
    assert "resultado" in set(antes["proposito"]) and antes["columnas_resultado"].fill_null("").str.contains("TIPOALTA").any()
    assert ctx.cortafuegos.violaciones() == 0 and ctx.cortafuegos.n_bloqueos() == 0


def test_el_volumen_y_los_entrantes_suman_las_cohortes_de_la_tarea_7(corrida, tmp_path):
    ctx, r = corrida
    ctx7 = Contexto.crear("extracto", fuente=ctx.fuente, dir_salida_base=tmp_path)
    var = cohorte.calcular(ctx7)["variantes"]
    for anio in range(2019, 2024):
        d = var.filter((pl.col("periodo") == str(anio)) & (pl.col("variante") == "D")).row(0, named=True)
        v = r["volumen"].filter(pl.col("anio") == anio)
        assert (int(v["n_D"].sum()), int(v["k_D"].sum())) == (d["n"], d["n_fallecidos"]), anio
        b = var.filter((pl.col("periodo") == str(anio)) & (pl.col("variante") == "B")).row(0, named=True)
        ent = r["entrantes"].filter(pl.col("periodo") == str(anio))
        assert int(ent["n"].sum()) == b["n_fallecidos"] + b["n_no_evento"] and int(ent["n_fallecidos"].sum()) == b["n_fallecidos"], anio


def test_el_hospital_pediatrico_del_generador_es_el_unico_marcado(corrida):
    _, r = corrida
    ped = r["hospitales"].filter(pl.col("pediatrico") == "si")
    assert set(ped["cod_hospital"]) == {"107100"} and ped["anio"].n_unique() == 6
    todos = r["hospitales"].filter(pl.col("cod_hospital") == "107100")
    assert todos["n_menores"].sum() > todos["n_adultos"].sum()


def test_la_factibilidad_del_historial_respeta_el_primer_anio_de_cada_espacio(corrida):
    _, r = corrida
    h = {row["anio"]: row for row in r["historial"].iter_rows(named=True)}
    assert h[2019]["n_ventana_completa"] == 0 and h[2021]["n_ventana_completa"] == 0
    assert h[2019]["cumple_minimo"] == "no" and h[2021]["cumple_minimo"] == "no"
    for anio in (2020, 2022, 2023):
        assert h[anio]["n_ventana_completa"] > 0 and h[anio]["n_viable"] <= min(h[anio]["n_id_valido"], h[anio]["n_ventana_completa"]), anio
        assert h[anio]["n_con_previo"] <= h[anio]["n_viable"]
    assert h[2020]["regimen"] == "2019-2020" and h[2022]["regimen"] == "2021-2023"


def test_las_cuentas_de_enlace_son_coherentes(corrida):
    _, r = corrida
    e = r["enlaces"]
    assert bool((e["n_enlazados"] <= e["n_elegibles"]).all()) and bool((e["n_elegibles"] <= e["n_total"]).all()) and bool((e["n_id_invalido"] <= e["n_total"]).all())
    for (direccion, periodo), g in e.group_by(["direccion", "periodo"]):
        g = g.sort("ventana_dias")
        assert g["n_elegibles"].to_list() == sorted(g["n_elegibles"].to_list(), reverse=True), (direccion, periodo)   # una ventana más larga nunca admite más traslados
        assert g["n_total"].n_unique() == 1


def test_la_autorreferencia_del_generador_supera_el_maximo_de_la_decision_2(corrida):
    _, r = corrida
    s = r["resumen"].row(0, named=True)
    assert s["autorref_supera_maximo"] == "si" and s["autorref_prop_aprox"] >= s["autorref_prop_exacta"] > 0.05


def test_ejecutar_escribe_tablas_figuras_metricas_y_cumple_privacidad(corrida):
    ctx, _ = corrida
    for nombre in ("t23_hospitales_anio", "t23_procedencia_autorref", "t23_hospitales_volumen", "t23_volumen_umbrales", "t23_hospitales_resumen", "t23_embudo",
                   "t23_resultado_entrantes", "t23_historial_factibilidad", "t23_enlaces_traslado", "t23_resumen_hospitales"):
        assert (ctx.dir_tablas / f"{nombre}.csv").is_file(), nombre
    assert ctx.figuras.ids() == ["F09_01", "F09_02", "F09_03"]
    idx = {row["id_figura"]: row["uso_interno"] for row in ctx.figuras.indice().iter_rows(named=True)}
    assert idx == {"F09_01": "si", "F09_02": "si", "F09_03": "no"}                # las figuras por hospital son de uso interno
    m = ctx.metricas().filter(pl.col("tarea") == "T09")
    claves = set(m["clave"].to_list())
    for k in ("hospitales.n_pediatricos.2019", "hospitales.n_cumplen_volumen.2023", "historial.factibilidad_pct.2020", "procedencia.autorref_pct_exacta", "procedencia.autorref_pct_aprox",
              "entrantes.mortalidad_pct.si.2019_2023", "entrantes.mortalidad_pct.no.2019_2023"):
        assert k in claves, k
    assert {k for k in claves if k.endswith(".2024")} == {"hospitales.n_pediatricos.2024"}          # lo demás usa el desenlace y no mira el año expuesto
    for fila in m.iter_rows(named=True):
        sufijo = fila["clave"].rsplit(".", 1)[-1]
        if sufijo.isdigit():
            assert fila["alcance"] == sufijo, fila["clave"]
    ctx.escribir_registros()
    esc = privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)
    assert privacidad.total_violaciones(esc) == 0, esc.filter(pl.col("n_violaciones") > 0)


def test_las_salidas_no_contienen_nombres_de_hospital(corrida):
    ctx, _ = corrida
    cat = catalogos.cargar_catalogos()
    nombres = {n for n in hp.mapa_nombres(cat).values() if len(n) >= 12}
    assert nombres
    for ruta in list(ctx.dir_tablas.glob("*.csv")):
        texto = hp.normalizar_nombre(ruta.read_text(encoding="utf8"))
        halladas = [n for n in nombres if n in texto]
        assert not halladas, (ruta.name, halladas[:3])


def test_el_informe_agrega_las_secciones_de_hospitales(corrida):
    ctx, _ = corrida
    inf = Informe(ctx, "t23_prueba_hospitales.md", "Informe de prueba")
    hp.agregar_secciones(inf, ctx, n0=1)
    md = inf.guardar().read_text(encoding="utf8")
    for titulo in ("1. Hospitales: perfil y volumen", "2. Embudo crudo exploratorio", "3. Identificador, historial y traslados", "4. Hospital de procedencia",
                   "5. Mortalidad de los traslados entrantes", "6. Límites"):
        assert titulo in md, titulo
    for fig in ("F09_01", "F09_02", "F09_03"):
        assert f"Figura {fig}." in md
    assert "{" not in md and "}" not in md
    assert "extracto" in md.lower()
