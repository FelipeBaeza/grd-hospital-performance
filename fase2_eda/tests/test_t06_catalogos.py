"""Tarea 6: calidad contra catálogos y consistencia cruzada (datos sintéticos y catálogos públicos)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl
import pytest

from eda import calidad, calidad_catalogos as cc, catalogos, config, privacidad, rutas
from eda.contexto import Contexto
from sintetico import crear_fuente_sintetica


# ------------------------------------------------------------------ catálogos mínimos de prueba
def _xlsx_minimo(ruta: Path, codigos: list[tuple[str | int, str]]) -> Path:
    """Crea un .xlsx mínimo con título en la fila 1, encabezado en la fila 2 y los códigos desde la fila 3."""
    compartidas = ["COD_GRD", "Etiqueta"]
    filas_xml = ['<row r="1"><c r="A1"/></row>', '<row r="2"><c r="A2" t="s"><v>0</v></c><c r="B2" t="s"><v>1</v></c></row>']
    for i, (cod, et) in enumerate(codigos, start=3):
        compartidas.append(et)
        idx_et = len(compartidas) - 1
        if isinstance(cod, int):
            celda_a = f'<c r="A{i}"><v>{cod}</v></c>'  # número: pierde el cero inicial
        else:
            compartidas.append(cod)
            celda_a = f'<c r="A{i}" t="s"><v>{len(compartidas) - 1}</v></c>'
        filas_xml.append(f'<row r="{i}">{celda_a}<c r="B{i}" t="s"><v>{idx_et}</v></c></row>')
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    hoja = f'<?xml version="1.0" encoding="UTF-8"?><worksheet {ns}><sheetData>{"".join(filas_xml)}</sheetData></worksheet>'
    ss = f'<?xml version="1.0" encoding="UTF-8"?><sst {ns}>' + "".join(f"<si><t>{t}</t></si>" for t in compartidas) + "</sst>"
    with zipfile.ZipFile(ruta, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", hoja)
        z.writestr("xl/sharedStrings.xml", ss)
    return ruta


def test_xlsx_del_grd_se_lee_como_texto_con_ceros_y_estructura(tmp_path):
    ruta = _xlsx_minimo(tmp_path / "grd.xlsx", [("011011", "PH PROCEDIMIENTOS VASCULARES"), (11012, "PH PROCEDIMIENTOS W/CC"), ("146101", "PH CESÁREA")])
    d = catalogos.cargar_grd_xlsx(ruta)
    assert d["cod_grd"].to_list() == ["011011", "011012", "146101"]  # el número se rellena a seis dígitos
    fila = d.row(0, named=True)
    assert (fila["mdc"], fila["tipo_grd"], fila["grd_base"], fila["severidad_digito"], fila["prefijo_etiqueta"]) == ("01", "1", "01101", "1", "PH")
    assert d.row(2, named=True)["mdc"] == "14"


def test_ir_grd_csv_se_rellena_a_seis_digitos(tmp_path):
    p = tmp_path / "IR-GRD.csv"
    p.write_text("IR- GRD,\n11011,PH PROCEDIMIENTOS VASCULARES\n11012,PH CON CC\n235440,MA VISITA\n", encoding="utf-8")
    d = catalogos.cargar_ir_grd_csv(p)
    assert d["cod_grd"].to_list() == ["011011", "011012", "235440"]
    assert d["mdc"].to_list() == ["01", "01", "23"]


def test_hospitales_distingue_clinicas_por_el_nombre(tmp_path):
    p = tmp_path / "Hospitales.csv"
    p.write_text("HOSPITALES,\n101100,Hospital Dr. Juan Noé Crevanni (Arica)\n101203,Clínica Hebe\n101212,Sanatorio Marítimo\n", encoding="utf-8")
    d = catalogos.cargar_hospitales(p)
    assert d["tipo_por_nombre"].to_list() == ["publico_probable", "privado_probable", "privado_probable"]
    assert catalogos.clasificar_hospital("101203", d) == "en_catalogo_privado_probable"
    assert catalogos.clasificar_hospital("999999", d) == "fuera_de_catalogo"


def test_cie10_y_cie9_se_normalizan_y_traen_capitulo(tmp_path):
    p = tmp_path / "CIE-10.csv"
    p.write_text(
        "Versión,Código,Descripción,Categoría,Sección,Capítulo\n"
        "CIE-v2013,A00,Cólera,A00 CÓLERA,A00-A09 X,Cap.01  CIERTAS (A00-B99)\n"
        'CIE-v2013,A00.0,"Cólera, biotipo",A00 CÓLERA,A00-A09 X,Cap.01  CIERTAS (A00-B99)\n'
        "CIE-v2013,O80,Parto,O80 PARTO,O80-O84 X,Cap.15  EMBARAZO (O00-O99)\n",
        encoding="utf-8",
    )
    d = catalogos.cargar_cie10(p)
    assert d["norm"].to_list() == ["A00", "A000", "O80"]
    assert d["capitulo"].to_list() == ["I", "I", "XV"]
    q = tmp_path / "CIE-9 .csv"
    q.write_text("Código,Descripción,Categoría,Sección,Capítulo\n00.17,X,c,s,k\n99.04,Y,c,s,k\n", encoding="utf-8")
    d9 = catalogos.cargar_cie9(q)
    assert d9["norm"].to_list() == ["0017", "9904"] and d9["cat2"].to_list() == ["00", "99"]


# ---------------------------------------------------------------------------- clasificadores
EX10, CAT3 = frozenset({"A00", "A000", "I10", "J189", "O80"}), frozenset({"A00", "I10", "J18", "O80"})


@pytest.mark.parametrize(
    "valor,esperado",
    [("A00.0", "valido"), ("a00.0", "valido"), (" I10 ", "valido"), ("J18.9", "valido"), ("A00.9", "solo_categoria"), ("J18.99", "solo_categoria"),
     ("Z99.9", "invalido"), ("XYZ", "invalido"), ("DESCONOCIDO", "centinela"), ("NO IDENTIFICADA", "centinela")],
)
def test_clasificar_cie10(valor, esperado):
    assert catalogos.clasificar_cie10(valor, EX10, CAT3) == esperado


def test_clasificar_cie9_y_conjuntos():
    ex9, c2 = frozenset({"0017", "9904"}), frozenset({"00", "99"})
    assert catalogos.clasificar_cie9("00.17", ex9, c2) == "valido"
    assert catalogos.clasificar_cie9("00.99", ex9, c2) == "solo_categoria"
    assert catalogos.clasificar_cie9("55.55", ex9, c2) == "invalido"
    assert catalogos.clasificar_en_conjunto("2", frozenset({"1", "2"})) == "valido"
    assert catalogos.clasificar_en_conjunto("212", frozenset({"1", "2"})) == "invalido"
    assert catalogos.clasificar_en_conjunto("DESCONOCIDO", frozenset({"1"})) == "centinela"


@pytest.mark.parametrize(
    "valor,esperado",
    [("011011", "valido"), ("011099", "solo_en_ir_grd_csv"), ("999991", "no_agrupable_98_99"), ("980001", "no_agrupable_98_99"),
     ("DESCONOCIDO", "centinela"), ("011555", "no_en_catalogo"), ("11011", "formato_invalido"), ("ABC123", "formato_invalido")],
)
def test_clasificar_grd(valor, esperado):
    assert catalogos.clasificar_grd(valor, frozenset({"011011"}), frozenset({"011011", "011099"})) == esperado


def test_rol_de_etiqueta_exacta_por_prefijo_y_sin_rol():
    roles = config.cargar_tipoalta_roles()
    assert catalogos.rol_de_etiqueta("DOMICILIO", roles) == ("DOMICILIO", "exacta", "NO_EVENTO")
    assert catalogos.rol_de_etiqueta("FALLECIDO", roles)[2] == "EVENTO"
    # etiquetas cortadas en los datos (44 y 45 caracteres)
    n1, m1, r1 = catalogos.rol_de_etiqueta("DERIVACIÓN A OTROS CENTROS (CÁRCEL, HOGAR DE", roles)
    n2, m2, r2 = catalogos.rol_de_etiqueta("DERIVACIÓN INST. PRIVADA (COMPRA DE SERVICIOS", roles)
    assert (m1, r1) == ("por_prefijo", "CENSURADO") and n1.startswith("DERIVACIÓN A OTROS CENTROS")
    assert (m2, r2) == ("por_prefijo", "CENSURADO") and n2.endswith("(COMPRA DE SERVICIOS)")
    assert catalogos.rol_de_etiqueta("CAMBIO DE SERVICIO", roles) == (None, "sin_rol", None)
    # una etiqueta que apenas comparte el comienzo no se asigna por prefijo
    assert catalogos.rol_de_etiqueta("DERIVACIÓN INST. PRIVADA (OTRA COSA DISTINTA)", roles)[1] == "sin_rol"


# ------------------------------------------------------------- reglas cruzadas (positivo y negativo)
@pytest.fixture(scope="module")
def reglas2():
    return {r.id: r for r in calidad.reglas_lote2()}


@pytest.fixture(scope="module")
def umbrales():
    return config.cargar_umbrales()


def _marco(filas: list[dict]) -> pl.DataFrame:
    cols = ["SEXO", "DIAGNOSTICO1", "FECHA_NACIMIENTO", "FECHA_INGRESO", "IR_29301_COD_GRD", "TIPO_ACTIVIDAD", "TIPOALTA", "IR_29301_MORTALIDAD"]
    base = {c: None for c in cols}
    return pl.DataFrame([{**base, **f} for f in filas], schema={c: pl.String for c in cols})


def test_sexo_incompatible_con_el_diagnostico(reglas2, umbrales):
    df = cc.preparar_cruzadas(_marco([
        {"SEXO": "HOMBRE", "DIAGNOSTICO1": "O80"},       # hombre con diagnóstico obstétrico: viola
        {"SEXO": "HOMBRE", "DIAGNOSTICO1": "N80.1"},     # hombre con genital femenino: viola
        {"SEXO": "MUJER", "DIAGNOSTICO1": "N40"},        # mujer con próstata: viola
        {"SEXO": "MUJER", "DIAGNOSTICO1": "C61"},        # mujer con cáncer de próstata: viola
        {"SEXO": "MUJER", "DIAGNOSTICO1": "O80"},        # coherente
        {"SEXO": "HOMBRE", "DIAGNOSTICO1": "N40"},       # coherente
        {"SEXO": "HOMBRE", "DIAGNOSTICO1": "J189"},      # coherente
        {"SEXO": "DESCONOCIDO", "DIAGNOSTICO1": "O80"},  # sexo desconocido: no se evalúa
    ]))
    assert cc.eval_sexo_dx(df, reglas2["Q25"], umbrales) == (4, 7)


def test_mdc15_con_edad_mayor_y_mdc14_con_hombre(reglas2, umbrales):
    df = cc.preparar_cruzadas(_marco([
        {"IR_29301_COD_GRD": "151011", "FECHA_NACIMIENTO": "2020-03-01", "FECHA_INGRESO": "2020-03-05", "SEXO": "HOMBRE"},   # neonato: coherente
        {"IR_29301_COD_GRD": "151011", "FECHA_NACIMIENTO": "1980-01-01", "FECHA_INGRESO": "2020-03-05", "SEXO": "HOMBRE"},   # adulto con MDC 15: viola
        {"IR_29301_COD_GRD": "141011", "FECHA_NACIMIENTO": "1990-01-01", "FECHA_INGRESO": "2020-03-05", "SEXO": "HOMBRE"},   # parto en hombre: viola
        {"IR_29301_COD_GRD": "141011", "FECHA_NACIMIENTO": "1990-01-01", "FECHA_INGRESO": "2020-03-05", "SEXO": "MUJER"},    # coherente
        {"IR_29301_COD_GRD": "DESCONOCIDO", "FECHA_NACIMIENTO": "1990-01-01", "FECHA_INGRESO": "2020-03-05", "SEXO": "HOMBRE"},
    ]))
    assert cc.eval_mdc15_edad(df, reglas2["Q26"], umbrales) == (1, 2)
    assert cc.eval_mdc14_hombre(df, reglas2["Q27"], umbrales) == (1, 2)


def test_tipo_de_grd_frente_a_la_actividad(reglas2, umbrales):
    df = cc.preparar_cruzadas(_marco([
        {"IR_29301_COD_GRD": "012511", "TIPO_ACTIVIDAD": "HOSPITALIZACIÓN"},                   # tipo 2 (ambulatorio) en hospitalización: viola Q28
        {"IR_29301_COD_GRD": "014511", "TIPO_ACTIVIDAD": "HOSPITALIZACIÓN"},                   # tipo 4 (médico hospitalizado): coherente
        {"IR_29301_COD_GRD": "011511", "TIPO_ACTIVIDAD": "CIRUGÍA MAYOR AMBULATORIA (CMA)"},   # tipo 1 (hospitalizado) en CMA: viola Q29
        {"IR_29301_COD_GRD": "013511", "TIPO_ACTIVIDAD": "CIRUGÍA MAYOR AMBULATORIA (CMA)"},   # tipo 3 (ambulatorio) en CMA: coherente
        {"IR_29301_COD_GRD": "014511", "TIPO_ACTIVIDAD": "HOSPITALIZACIÓN DIURNA"},            # tipo 4 en actividad diurna: viola Q29
        {"IR_29301_COD_GRD": "010511", "TIPO_ACTIVIDAD": "HOSPITALIZACIÓN"},                   # tipo 0 (error): fuera del denominador
        {"IR_29301_COD_GRD": "014511", "TIPO_ACTIVIDAD": "HOSPITALIZACIÓN EN URGENCIA"},        # urgencia: ambigua, no viola
    ]))
    assert cc.eval_grd_ambulatorio_en_hosp(df, reglas2["Q28"], umbrales) == (1, 6)
    assert cc.eval_grd_hosp_en_ambulatoria(df, reglas2["Q29"], umbrales) == (2, 6)
    assert cc.grupo_grd("4") == "hospitalizado" and cc.grupo_grd("5") == "ambulatorio" and cc.grupo_grd("0") == "otro" and cc.grupo_grd(None) == "sin_tipo"


def test_fallecido_con_riesgo_de_mortalidad_del_grd_en_cero(reglas2, umbrales):
    df = _marco([
        {"TIPOALTA": "FALLECIDO", "IR_29301_MORTALIDAD": "0"},       # viola
        {"TIPOALTA": "FALLECIDO", "IR_29301_MORTALIDAD": "3"},       # coherente
        {"TIPOALTA": "FALLECIDO", "IR_29301_MORTALIDAD": "DESCONOCIDO"},
        {"TIPOALTA": "DOMICILIO", "IR_29301_MORTALIDAD": "0"},       # no es fallecido
    ])
    assert cc.eval_fallecido_riesgo_cero(df, reglas2["Q24"], umbrales) == (1, 2)


def test_etiquetas_sin_rol_se_cuentan(reglas2):
    roles = config.cargar_tipoalta_roles()
    df = pl.DataFrame({"TIPOALTA": ["DOMICILIO"] * 5 + ["DERIVACIÓN INST. PRIVADA (COMPRA DE SERVICIOS"] * 2 + ["OTRO MOTIVO NUEVO"] * 3 + [None]})
    v, e, metodos = cc.eval_tipoalta_sin_rol(df, roles)
    assert (v, e) == (3, 10) and metodos == {"exacta": 5, "por_prefijo": 2, "sin_rol": 3}


def test_contar_clases_usa_solo_valores_distintos():
    df = pl.DataFrame({"a": ["A00.0", "A00.0", "Z99", None, "DESCONOCIDO"], "b": ["I10", "XYZ", None, None, None]})
    c = cc.contar_clases(df, ("a", "b"), lambda v: catalogos.clasificar_cie10(v, EX10, CAT3))
    assert dict(c) == {"valido": 3, "invalido": 2, "centinela": 1}


def test_formato_cie10_cuenta_rasgos():
    df = pl.DataFrame({"D1": ["J18.9", "J189", "j18.9", " I10", "Ó30", "DESCONOCIDO", None]})
    f = cc.formato_cie10(df, ("D1",))
    assert f["valores"] == 6 and f["centinela"] == 1
    assert f["con_punto"] == 2 and f["sin_punto"] == 3
    assert f["con_minuscula"] == 1 and f["con_espacio_extremo"] == 1 and f["con_caracter_no_ascii"] == 1


# ------------------------------------------------------------------- reglas del lote 2 en el TOML
def test_reglas_del_lote_2_y_autorizacion_de_2024():
    lote2 = calidad.reglas_lote2()
    assert [r.id for r in lote2] == [f"Q{n}" for n in range(20, 33)]
    assert calidad.validar_reglas(calidad.cargar_reglas()) == []
    no_permitidas = {r.id for r in calidad.cargar_reglas() if not r.permite_expuesto}
    assert no_permitidas == {"Q24", "Q30"}  # las que cruzan el desenlace no se evalúan en 2024
    assert {r.id for r in calidad.reglas_lote1()} == {f"Q{n:02d}" for n in range(1, 20)}
    q21 = next(r for r in lote2 if r.id == "Q21")
    assert len(q21.columnas) == 30 and q21.columnas[0] == "PROCEDIMIENTO1"


# ------------------------------------------------------------------ punta a punta (sintético)
@pytest.fixture(scope="module")
def cat():
    return catalogos.cargar_catalogos()


@pytest.fixture(scope="module")
def corrida(tmp_path_factory, cat):
    raiz = tmp_path_factory.mktemp("t6")
    fuente = crear_fuente_sintetica(raiz / "datos", n=500, semilla=31, defectos_por_anio={2020: {"hospital_fuera_catalogo": 7}})
    ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=raiz / "salidas")
    return ctx, cc.calcular(ctx, cat)


def test_catalogos_reales_cargan_con_los_conteos_esperados(cat):
    assert cat.hospitales.height == 377
    assert cat.grd_xlsx.height == 1058 and all(len(c) == 6 for c in cat.grd_xlsx_set)
    assert cat.cie10.height > 39000 and "U07" in cat.cie10_cat3
    assert cat.severidad == frozenset({"0", "1", "2", "3"}) and cat.mortalidad == frozenset({"0", "1", "2", "3"})
    assert len(cat.pabellon) == 8
    assert len(cat.roles_tipoalta) == 12
    assert cat.cie10_capitulo_por_cat3["O80"] == "XV" and cat.cie10_capitulo_por_cat3["J18"] == "X"
    # los ceros iniciales del CSV se restituyen: ninguna llave de IR-GRD.csv queda con menos de seis dígitos
    assert all(len(c) == 6 for c in cat.grd_csv_set)


def test_hospital_fuera_de_catalogo_se_detecta(corrida):
    _, r = corrida
    h = r["hospitales"]
    fuera = h.filter((pl.col("estado") == "fuera_de_catalogo") & (pl.col("cod_hospital") == "999999") & (pl.col("anio") == 2020))
    assert fuera["n_episodios"].to_list() == [7]
    q23 = r["reglas"].filter((pl.col("regla_id") == "Q23") & (pl.col("ambito") == "2020")).row(0, named=True)
    assert q23["n_violaciones"] >= 7


def test_validez_de_cie10_secundarios_y_formato_de_diagnosticos(corrida):
    _, r = corrida
    v = r["validez"].filter((pl.col("familia") == "CIE-10 diagnósticos secundarios") & (pl.col("anio") == 2021))
    assert v.filter(pl.col("estado") == "valido")["prop"][0] > 0.9
    f = r["formato"].filter(pl.col("anio") == 2021)
    por = dict(zip(f["rasgo"], f["n"]))
    assert por["con_punto"] > por["sin_punto"] > 0  # la mayoría con punto, algunos sin punto
    assert por["con_minuscula"] > 0 and por["con_espacio_extremo"] > 0


def test_2024_bloquea_las_reglas_que_cruzan_el_desenlace_y_no_viola_el_cortafuegos(corrida):
    ctx, r = corrida
    t = r["reglas"].filter(pl.col("ambito") == "2024")
    assert set(t.filter(pl.col("estado") == "BLOQUEADA_2024")["regla_id"].to_list()) == {"Q24", "Q30"}
    assert t.filter(pl.col("regla_id") == "Q32")["estado"][0] == "EVALUADA"  # validez de severidad y mortalidad: solo conteos
    # las etiquetas de TIPOALTA no incluyen 2024
    assert r["etiquetas"].filter(pl.col("n") > 0).height >= 8
    assert ctx.cortafuegos.violaciones() == 0 and ctx.cortafuegos.n_bloqueos() == 0
    # las columnas de resultado de 2024 nunca llegaron en claro a una regla no autorizada
    reg = ctx.cortafuegos.registro().filter(pl.col("anio") == 2024)
    assert "TIPOALTA" not in ";".join(reg["columnas_resultado"].to_list())


def test_etiquetas_truncadas_de_tipoalta_se_asignan_por_prefijo(corrida):
    _, r = corrida
    e = r["etiquetas"]
    pref = e.filter(pl.col("metodo") == "por_prefijo")
    assert sorted(pref["longitud"].to_list()) == [44, 45]
    assert e.filter(pl.col("metodo") == "sin_rol").height == 0
    assert 0 < r["pct_perdido_union_exacta"] < 5
    # el rol DESCONOCIDO del archivo no aparece en los datos sintéticos
    assert e.filter(pl.col("metodo") == "sin_datos").height >= 1


def test_comparacion_de_catalogos_grd_de_la_raiz(cat):
    comp = cc.comparar_catalogos_grd(cat)
    assert set(comp["donde"].to_list()) <= {"solo_en_xlsx", "solo_en_ir_grd_csv"}
    assert comp.height > 0 and all(len(c) == 6 for c in comp["cod_grd"].to_list())


def test_ejecutar_escribe_tablas_figuras_y_cumple_privacidad(tmp_path, cat):
    fuente = crear_fuente_sintetica(tmp_path / "datos", n=400, semilla=14)
    ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=tmp_path / "salidas")
    cc.ejecutar(ctx, cat)
    for nombre in ("t24_validez_codigos.csv", "t24_formato_cie10.csv", "t24_capitulos_cie10.csv", "t24_hospitales_catalogo.csv",
                   "t24_hospitales_resumen.csv", "t24_catalogo_grd_comparacion.csv", "t24_grd_tipo_actividad.csv",
                   "t24_tipoalta_etiquetas.csv", "t24_consistencia_cruzada.csv", "t24_resumen_catalogos.csv"):
        assert (ctx.dir_tablas / nombre).is_file(), nombre
    assert sorted(ctx.figuras.ids()) == ["F06_01", "F06_02"]
    m = ctx.metricas()
    assert m.filter(pl.col("clave") == "hospitales.n_distintos.maximo")["valor"][0] >= 6
    assert m.filter(pl.col("clave") == "calidad.cie10_principal_valido_pct")["valor"][0] > 90
    ctx.escribir_registros()
    escaneo = privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)
    assert privacidad.total_violaciones(escaneo) == 0, escaneo.filter(pl.col("n_violaciones") > 0)
    # el catálogo de hospitales se informa con códigos, nunca con nombres
    h = (ctx.dir_tablas / "t24_hospitales_catalogo.csv").read_text(encoding="utf8")
    assert "Hospital" not in h and "Clínica" not in h


def test_informe_agrega_secciones_de_catalogos(tmp_path, cat):
    from eda.informes import Informe

    fuente = crear_fuente_sintetica(tmp_path / "datos", n=300, semilla=2)
    ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=tmp_path / "salidas")
    cc.ejecutar(ctx, cat)
    inf = Informe(ctx, "t24_catalogos.md", "Informe de catálogos y consistencia cruzada")
    cc.agregar_secciones(inf, ctx)
    md = inf.guardar().read_text(encoding="utf8")
    assert "Etiquetas de TIPOALTA frente a los roles" in md and "![" in md
    assert "Una unión exacta pierde" in md
