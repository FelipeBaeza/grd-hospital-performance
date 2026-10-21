"""Elixhauser (Quan 2005, CIE-10): mapeo contrastado con una segunda fuente, casos por categoría y jerarquías."""

from __future__ import annotations

import polars as pl
import pytest

from eda import config, elixhauser as ex


@pytest.fixture(scope="module")
def e() -> ex.Elixhauser:
    return ex.cargar()


def _rng(letra: str, a: int, b: int) -> list[str]:
    return [f"{letra}{n:02d}" for n in range(a, b + 1)]


def _lista(txt: str) -> set[str]:
    return {t.strip().upper().replace(".", "") for t in txt.replace("\n", "").split(",") if t.strip()}


# Segunda fuente: Tabla S1 del suplemento de PLOS ONE (2023), transcrita de forma independiente al mapeo del CSV.
SEGUNDA_FUENTE: dict[str, set[str]] = {
    "ELIX_01": _lista("I43,I50,I09.9,I11.0,I13.0,I13.2,I25.5,I42.0,I42.5,I42.6,I42.7,I42.8,I42.9,P29.0"),
    "ELIX_02": _lista("I47,I48,I49,I44.1,I44.2,I44.3,I45.6,I45.9,R00.0,R00.1,R00.8,T82.1,Z45.0,Z95.0"),
    "ELIX_03": _lista("I05,I06,I07,I08,I34,I35,I36,I37,I38,I39,A52.0,I09.1,I09.8,Q23.0,Q23.1,Q23.2,Q23.3,Z95.2,Z95.4"),
    "ELIX_04": _lista("I26,I27,I28.0,I28.8,I28.9"),
    "ELIX_05": _lista("I70,I71,I73.1,I73.8,I73.9,I77.1,I79.0,I79.2,K55.1,K55.8,K55.9,Z95.8,Z95.9"),
    "ELIX_06": _lista("I10"),
    "ELIX_07": _lista("I11,I12,I13,I15"),
    "ELIX_08": _lista("G81,G82,G04.1,G11.4,G80.1,G80.2,G83.0,G83.1,G83.2,G83.3,G83.4,G83.9"),
    "ELIX_09": _lista("G10,G11,G12,G13,G20,G21,G22,G32,G35,G36,G37,G40,G41,R56,G25.4,G25.5,G31.2,G31.8,G31.9,G93.1,G93.4,R47.0"),
    "ELIX_10": _lista("J40,J41,J42,J43,J44,J45,J46,J47,J60,J61,J62,J63,J64,J65,J66,J67,I27.8,I27.9,J68.4,J70.1,J70.3"),
    "ELIX_11": _lista("E10.0,E10.1,E10.9,E11.0,E11.1,E11.9,E12.0,E12.1,E12.9,E13.0,E13.1,E13.9,E14.0,E14.1,E14.9"),
    "ELIX_12": _lista(
        "E10.2,E10.3,E10.4,E10.5,E10.6,E10.7,E10.8,E11.2,E11.3,E11.4,E11.5,E11.6,E11.7,E11.8,E12.2,E12.3,E12.4,E12.5,E12.6,E12.7,E12.8,"
        "E13.2,E13.3,E13.4,E13.5,E13.6,E13.7,E13.8,E14.2,E14.3,E14.4,E14.5,E14.6,E14.7,E14.8"
    ),
    "ELIX_13": _lista("E00,E01,E02,E03,E89.0"),
    "ELIX_14": _lista("N18,N19,I12.0,I13.1,N25.0,Z49.0,Z49.1,Z49.2,Z94.0,Z99.2"),
    "ELIX_15": _lista("B18,I85,K70,K72,K73,K74,I86.4,I98.2,K71.1,K71.3,K71.4,K71.5,K71.7,K76.0,K76.2,K76.3,K76.4,K76.5,K76.6,K76.7,K76.8,K76.9,Z94.4"),
    "ELIX_16": _lista("K25.7,K25.9,K26.7,K26.9,K27.7,K27.9,K28.7,K28.9"),
    "ELIX_17": _lista("B20,B21,B22,B24"),
    "ELIX_18": _lista("C81,C82,C83,C84,C85,C88,C96,C90.0,C90.2"),
    "ELIX_19": _lista("C77,C78,C79,C80"),
    "ELIX_20": set(_rng("C", 0, 26) + _rng("C", 30, 34) + _rng("C", 37, 41) + ["C43"] + _rng("C", 45, 58) + _rng("C", 60, 76) + ["C97"]),
    "ELIX_21": _lista("M05,M06,M08,M30,M32,M33,M34,M35,M45,L94.0,L94.1,L94.3,M12.0,M12.3,M31.0,M31.1,M31.2,M31.3,M46.1,M46.8,M46.9"),
    "ELIX_22": _lista("D65,D66,D67,D68,D69.1,D69.3,D69.4,D69.5,D69.6"),
    "ELIX_23": _lista("E66"),
    "ELIX_24": _lista("E40,E41,E42,E43,E44,E45,E46,R64,R63.4"),
    "ELIX_25": _lista("E86,E87,E22.2"),
    "ELIX_26": _lista("D50.0"),
    "ELIX_27": _lista("D51,D52,D53,D50.8,D50.9"),
    "ELIX_28": _lista("F10,E52,T51,G62.1,I42.6,K29.2,K70.0,K70.3,K70.9,Z50.2,Z71.4,Z72.1"),
    "ELIX_29": _lista("F11,F12,F13,F14,F15,F16,F18,F19,Z71.5,Z72.2"),
    "ELIX_30": _lista("F20,F22,F23,F24,F25,F28,F29,F30.2,F31.2,F31.5"),
    "ELIX_31": _lista("F32,F33,F20.4,F31.3,F31.4,F31.5,F34.1,F41.2,F43.2"),
}


def test_el_mapeo_tiene_31_categorias_en_el_orden_de_quan(e):
    assert e.ids == tuple(f"ELIX_{i:02d}" for i in range(1, 32))
    assert e.nombres_en["ELIX_01"] == "Congestive heart failure" and e.nombres_en["ELIX_31"] == "Depression"
    # las cinco sospechosas de la Fase 1 (circulación pulmonar, coagulopatía, pérdida de peso, electrolitos y anemia por sangrado)
    assert e.sospechosas == {"ELIX_04", "ELIX_22", "ELIX_24", "ELIX_25", "ELIX_26"}
    assert e.nombres_en["ELIX_04"].startswith("Pulmonary") and e.nombres_en["ELIX_22"] == "Coagulopathy"
    assert e.nombres_en["ELIX_24"] == "Weight loss" and e.nombres_en["ELIX_25"].startswith("Fluid") and e.nombres_en["ELIX_26"] == "Blood loss anemia"
    assert e.nombres_en["ELIX_11"] == "Diabetes, uncomplicated" and e.nombres_en["ELIX_12"] == "Diabetes, complicated"
    assert e.domina == {"ELIX_07": "ELIX_06", "ELIX_12": "ELIX_11", "ELIX_19": "ELIX_20"}


def test_mapeo_coincide_con_la_segunda_fuente_categoria_por_categoria():
    m = config.leer_csv_texto(ex.rutas.DIR_CONFIG / "elixhauser_quan2005.csv")
    assert set(m["contrastado_con_dos_fuentes"].to_list()) == {"si"}
    por_cat = {k: set(g["patron"].to_list()) for k, g in ((x[0], d) for x, d in m.group_by("elix_id"))}
    assert set(por_cat) == set(SEGUNDA_FUENTE)
    for k in SEGUNDA_FUENTE:
        assert por_cat[k] == SEGUNDA_FUENTE[k], (k, por_cat[k] ^ SEGUNDA_FUENTE[k])
    assert sum(len(v) for v in por_cat.values()) == m.height  # sin patrones repetidos dentro de una categoría


def test_pesos_de_van_walraven_estan_en_su_rango_publicado(e):
    assert min(e.pesos.values()) == -7 and max(e.pesos.values()) == 12  # "los pesos van de -7 a 12"
    assert e.pesos["ELIX_01"] == 7 and e.pesos["ELIX_19"] == 12 and e.pesos["ELIX_29"] == -7 and e.pesos["ELIX_23"] == -4
    assert sum(1 for v in e.pesos.values() if v == 0) == 10  # hipertensión (2), diabetes (2), hipotiroidismo, úlcera, VIH, artritis, alcohol y psicosis


# (positivo con punto, positivo sin punto o con quinto carácter, negativo cercano)
CASOS = {
    "ELIX_01": ("I50.0", "I509", "I51.9"),
    "ELIX_02": ("I48.9", "I490", "I46.9"),
    "ELIX_03": ("I34.0", "I350", "I09.0"),
    "ELIX_04": ("I26.9", "I270", "I24.9"),
    "ELIX_05": ("I70.2", "I739", "I72.4"),
    "ELIX_06": ("I10", "I10X", "I20.0"),
    "ELIX_07": ("I11.9", "I129", "I14.9"),
    "ELIX_08": ("G81.9", "G820", "G83.5"),
    "ELIX_09": ("G40.9", "G200", "G14"),
    "ELIX_10": ("J44.9", "J450", "J48.9"),
    "ELIX_11": ("E11.9", "E109", "E10.2"),
    "ELIX_12": ("E11.2", "E145", "E10.0"),
    "ELIX_13": ("E03.9", "E890", "E04.9"),
    "ELIX_14": ("N18.9", "N185", "N17.9"),
    "ELIX_15": ("K70.3", "K746", "K75.9"),
    "ELIX_16": ("K25.7", "K289", "K25.0"),
    "ELIX_17": ("B20.1", "B240", "B23.0"),
    "ELIX_18": ("C81.9", "C902", "C90.1"),
    "ELIX_19": ("C78.0", "C799", "C76.0"),
    "ELIX_20": ("C34.9", "C509", "C77.0"),
    "ELIX_21": ("M05.8", "M3130", "M07.0"),
    "ELIX_22": ("D68.9", "D696", "D69.2"),
    "ELIX_23": ("E66.9", "E660", "E65"),
    "ELIX_24": ("E43", "R634", "E50.9"),
    "ELIX_25": ("E87.1", "E861", "E88.9"),
    "ELIX_26": ("D50.0", "D500", "D50.1"),
    "ELIX_27": ("D50.9", "D531", "D54"),
    "ELIX_28": ("F10.2", "K703", "F11.2"),
    "ELIX_29": ("F11.2", "F199", "F17.2"),
    "ELIX_30": ("F20.0", "F2900", "F21"),
    "ELIX_31": ("F32.9", "F412", "F34.0"),
}


@pytest.mark.parametrize("elix_id", sorted(CASOS))
def test_cada_categoria_reconoce_codigos_con_punto_sin_punto_y_rechaza_un_cercano(e, elix_id):
    con_punto, sin_punto, negativo = CASOS[elix_id]
    assert elix_id in ex.categorias_de(con_punto, e), (elix_id, con_punto)
    assert elix_id in ex.categorias_de(sin_punto, e), (elix_id, sin_punto)
    assert elix_id in ex.categorias_de(" " + con_punto.lower() + " ", e)  # minúsculas y espacios
    assert elix_id not in ex.categorias_de(negativo, e), (elix_id, negativo)


def test_un_codigo_puede_pertenecer_a_dos_categorias(e):
    assert ex.categorias_de("I42.6", e) == {"ELIX_01", "ELIX_28"}  # insuficiencia cardíaca y abuso de alcohol
    assert ex.categorias_de("I27.8", e) == {"ELIX_04", "ELIX_10"}
    assert ex.categorias_de("I13.0", e) == {"ELIX_01", "ELIX_07"}
    assert ex.categorias_de("F20.4", e) == {"ELIX_30", "ELIX_31"}


def test_jerarquias_de_complicada_sobre_no_complicada_y_metastasico_sobre_tumor(e):
    assert ex.aplicar_jerarquia_ids({"ELIX_11", "ELIX_12"}, e) == {"ELIX_12"}
    assert ex.aplicar_jerarquia_ids({"ELIX_06", "ELIX_07"}, e) == {"ELIX_07"}
    assert ex.aplicar_jerarquia_ids({"ELIX_19", "ELIX_20"}, e) == {"ELIX_19"}
    assert ex.aplicar_jerarquia_ids({"ELIX_11"}, e) == {"ELIX_11"}


def _marco(*filas: list[str | None]) -> pl.DataFrame:
    n = max(len(f) for f in filas)
    return pl.DataFrame({f"D{i}": [f[i] if i < len(f) else None for f in filas] for i in range(n)}, schema={f"D{i}": pl.String for i in range(n)})


def test_flags_por_episodio_aplican_jerarquia_conteo_y_puntaje(e):
    df = _marco(
        ["E11.9", "E11.2", "I10", "C34.9"],   # diabetes complicada (anula no complicada), HTN no complicada, tumor sólido
        ["C78.0", "C34.9", "I13.0"],           # metastásico anula tumor; I13.0 = ICC + HTN complicada
        ["J18.9", "A09"],                      # sin comorbilidades
        [None, None],                          # sin diagnósticos secundarios
        ["e66.9", " F32.9 ", "Z99.2"],         # obesidad, depresión, insuficiencia renal (Z99.2), con minúsculas y espacios
    )
    f = ex.calcular_flags(df, df.columns, e)
    fila = lambda i: {k: v for k, v in f.row(i, named=True).items() if k.startswith("ELIX_") and v == 1}  # noqa: E731
    assert set(fila(0)) == {"ELIX_12", "ELIX_06", "ELIX_20"}  # ELIX_11 anulada por ELIX_12
    assert set(fila(1)) == {"ELIX_19", "ELIX_01", "ELIX_07"}  # ELIX_20 anulada; ELIX_06 no se activa
    assert fila(2) == {} and fila(3) == {}
    assert set(fila(4)) == {"ELIX_23", "ELIX_31", "ELIX_14"}
    assert f["N_COMORB_ELIX"].to_list() == [3, 3, 0, 0, 3]
    # puntaje de van Walraven: 0 + 0 + 4 = 4 ; 12 + 7 + 0 = 19 ; -4 -3 +5 = -2
    assert f["SCORE_VANWALRAVEN"].to_list() == [4, 19, 0, 0, -2]
    # sin las cinco sospechosas el puntaje no cambia en estos casos (ninguna activa); para ELIX_25 sí cambia
    assert f["PUNTAJE_VAN_WALRAVEN_SIN_5"].to_list() == [4, 19, 0, 0, -2]
    g = ex.calcular_flags(_marco(["E87.1", "I50.0"]), ["D0", "D1"], e)
    assert g["SCORE_VANWALRAVEN"][0] == 12 and g["PUNTAJE_VAN_WALRAVEN_SIN_5"][0] == 7


def test_solo_se_usan_las_columnas_indicadas_regla_r3(e):
    df = pl.DataFrame({"DIAGNOSTICO1": ["I50.0", "E11.9"], "DIAGNOSTICO2": ["J18.9", "I10"]})
    secundarios = ex.calcular_flags(df, ["DIAGNOSTICO2"], e)
    con_principal = ex.calcular_flags(df, ["DIAGNOSTICO1", "DIAGNOSTICO2"], e)
    assert secundarios["ELIX_01"].to_list() == [0, 0] and secundarios["ELIX_11"].to_list() == [0, 0]
    assert con_principal["ELIX_01"].to_list() == [1, 0] and con_principal["ELIX_11"].to_list() == [0, 1]


def test_la_diabetes_no_es_cero_y_la_version_de_tres_caracteres_si_la_pierde(e):
    df = _marco(["E11.9"], ["E10.2"], ["E14.0"], ["I10"], ["I50.0"], ["J44.9"])
    completo = ex.calcular_flags(df, ["D0"], e)
    assert int(completo["ELIX_11"].sum()) == 2 and int(completo["ELIX_12"].sum()) == 1  # diabetes distinta de cero
    e3 = ex.cargar(solo_3c=True)
    assert e3.largos == (3,)
    tres = ex.calcular_flags(df, ["D0"], e3)
    assert int(tres["ELIX_11"].sum()) == 0 and int(tres["ELIX_12"].sum()) == 0  # el mapeo de 3 caracteres no ve la diabetes
    # las categorías definidas con tres caracteres siguen apareciendo
    assert int(tres["ELIX_06"].sum()) == 1 and int(tres["ELIX_01"].sum()) == 1 and int(tres["ELIX_10"].sum()) == 1


def test_codigos_con_quinto_caracter_y_tres_caracteres(e):
    assert "ELIX_11" in ex.categorias_de("E11.90", e)  # extensión chilena de cinco caracteres
    assert ex.categorias_de("E11", e) == set()  # E11 sin cuarto carácter no se asigna (Quan exige el cuarto)
    assert "ELIX_01" in ex.categorias_de("I50", e) and "ELIX_06" in ex.categorias_de("I10", e)
    assert ex.categorias_de("", e) == set()
