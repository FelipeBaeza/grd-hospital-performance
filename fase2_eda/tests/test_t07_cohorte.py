"""Tarea 7: base tipada, cascada de exclusiones, variantes de cohorte, resultado, estadía y partición temporal."""

from __future__ import annotations

import datetime as dt
import re

import polars as pl
import pytest

from eda import catalogos, cohorte, config, elixhauser, esquema, privacidad
from eda.contexto import Contexto
from eda.cortafuegos import BLOQUEADO
from eda.informes import Informe
from sintetico import crear_fuente_sintetica

ANIOS_RES = [2019, 2020, 2021, 2022, 2023]


@pytest.fixture(scope="module")
def cat():
    return catalogos.cargar_catalogos()


@pytest.fixture(scope="module")
def u():
    return config.cargar_umbrales()


@pytest.fixture(scope="module")
def e():
    return elixhauser.cargar()


@pytest.fixture(scope="module")
def e3():
    return elixhauser.cargar(solo_3c=True)


# --------------------------------------------------------------------------- marcos a mano
def _f(**kw):
    base = {
        "SEXO": "MUJER", "FECHA_NACIMIENTO": "1980-01-01", "FECHA_INGRESO": "2021-03-01", "FECHAALTA": "2021-03-05", "TIPOALTA": "DOMICILIO",
        "TIPO_ACTIVIDAD": "HOSPITALIZACIÓN", "TIPO_INGRESO": "URGENCIA", "TIPO_PROCEDENCIA": "SERVICIO EMERGENCIA",
        "IR_29301_COD_GRD": "011011", "DIAGNOSTICO1": "I21.9",
    }
    base.update(kw)
    return base


def _marco(filas: list[dict]) -> pl.DataFrame:
    cols = list(cohorte.COLS_BASE) + list(cohorte.COLS_RESULTADO)
    return pl.DataFrame({c: [f.get(c) for f in filas] for c in cols}, schema={c: pl.String for c in cols})


def _dx(filas: list[list[str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {c: [(f[i] if i < len(f) else None) for f in filas] for i, c in enumerate(esquema.DIAG_SEC)}, schema={c: pl.String for c in esquema.DIAG_SEC}
    )


# ------------------------------------------------------------------------------ tipado
def test_tipar_convierte_edad_estancia_rol_capitulo_y_estado_del_grd(cat, u):
    df = _marco([
        _f(FECHA_NACIMIENTO="1980-05-10", IR_29301_COD_GRD="146101", DIAGNOSTICO1="O80"),                          # 0: adulta, parto, vuelve a domicilio
        _f(SEXO="HOMBRE", FECHA_NACIMIENTO="1950-01-01", FECHAALTA="2021-03-01", TIPOALTA="FALLECIDO"),            # 1: fallece el mismo día
        _f(TIPOALTA="DERIVACIÓN A OTROS CENTROS (CÁRCEL, HOGAR DE", IR_29301_COD_GRD="990019", DIAGNOSTICO1=None),  # 2: etiqueta cortada, GRD 99
        _f(SEXO="DESCONOCIDO", FECHA_NACIMIENTO="2030-01-01", FECHAALTA="2021-02-20", TIPOALTA=None, IR_29301_COD_GRD=None),  # 3: edad negativa, alta antes del ingreso
        _f(IR_29301_COD_GRD="DESCONOCIDO", FECHA_INGRESO="2021-12-30", FECHAALTA="2022-01-02", DIAGNOSTICO1="DESCONOCIDO"),   # 4: alta en otro año
        _f(IR_29301_COD_GRD="12345", FECHA_NACIMIENTO="1890-01-01"),                                               # 5: GRD de largo inválido, edad sobre el máximo
    ])
    b = cohorte.tipar(df, 2021, cat, u)
    assert b["anio"].to_list() == [2021] * 6
    assert b["edad"].to_list() == [40, 71, 41, None, 41, None]            # 0: cumpleaños posterior al ingreso; 3 y 5 fuera de rango
    assert b["sexo"].to_list() == ["MUJER", "HOMBRE", "MUJER", "SIN_DATO", "MUJER", "MUJER"]
    assert b["estancia"].to_list() == [4, 0, 4, None, 3, 4]                # 3: alta anterior al ingreso, se anula
    assert b["mes_alta"].to_list() == [3, 3, 3, 2, None, 3]                # 4: el alta cae en 2022, fuera del año del archivo
    assert b["rol"].to_list() == ["NO_EVENTO", "EVENTO", "CENSURADO", "SIN_DATO", "NO_EVENTO", "NO_EVENTO"]
    assert b["etiqueta_roles"][2] == "DERIVACIÓN A OTROS CENTROS (CÁRCEL, HOGAR)"   # la etiqueta cortada se asigna por prefijo
    assert b["capitulo"].to_list() == ["XV", "IX", SIN_DATO_ESPERADO, "IX", "sin capítulo", "IX"]
    assert b["grd_estado"].to_list() == ["agrupable", "agrupable", "mdc_99", "nulo", "centinela", "formato_invalido"]
    assert b["mdc"].to_list() == ["14", "01", "99", None, None, None]


SIN_DATO_ESPERADO = cohorte.SIN_DATO


def test_tipar_sin_columnas_de_resultado_no_agrega_desenlace_ni_estancia(cat, u):
    df = _marco([_f()]).drop(list(cohorte.COLS_RESULTADO))
    b = cohorte.tipar(df, 2024, cat, u)
    assert not {"rol", "estancia", "mes_alta", "etiqueta_roles"} & set(b.columns)
    marcado = cohorte.marcar(cohorte.agregar_comorbilidad(b, _dx([[]]), elixhauser.cargar(), elixhauser.cargar(solo_3c=True)))
    assert {"A", "B", "C"} <= set(marcado.columns) and not {"D", "E", "ex03_pipe", "est_pipe"} & set(marcado.columns)


# ------------------------------------------------------------------------- exclusiones
def test_exclusiones_en_secuencia_y_diabetes_distingue_el_mapeo_completo_del_de_tres_caracteres(cat, u, e, e3):
    filas = [
        _f(IR_29301_COD_GRD="146101"),                       # 0 MDC 14 sin comorbilidades
        _f(IR_29301_COD_GRD="146101"),                       # 1 MDC 14 con diabetes no complicada: el mapeo de tres caracteres no la ve
        _f(IR_29301_COD_GRD="146101"),                       # 2 MDC 14 con hipertensión: la ven los dos mapeos
        _f(IR_29301_COD_GRD="151011"),                       # 3 recién nacido
        _f(IR_29301_COD_GRD=None),                           # 4 sin GRD
        _f(IR_29301_COD_GRD="990019"),                       # 5 GRD no agrupable
        _f(IR_29301_COD_GRD="011011"),                       # 6 MDC 01, entra en la cohorte dura
        _f(IR_29301_COD_GRD="146101", TIPOALTA="FALLECIDO"), # 7 MDC 14 sin comorbilidades que fallece
    ]
    dx = _dx([[], ["E11.9"], ["I10"], [], [], [], ["I50.0"], []])
    b = cohorte.marcar(cohorte.agregar_comorbilidad(cohorte.tipar(_marco(filas), 2021, cat, u), dx, e, e3))
    assert b["n_comorb"].to_list() == [0, 1, 1, None, None, None, None, 0]
    assert b["n_comorb_3c"].to_list() == [0, 0, 1, None, None, None, None, 0]
    assert b["ex01"].to_list() == [False, False, False, False, True, True, False, False]
    assert b["ex02"].to_list() == [False, False, False, True, False, False, False, False]
    assert b["ex03"].to_list() == [True, False, False, False, False, False, False, True]
    assert b["A"].to_list() == [False, True, True, False, False, False, True, False]
    # cada egreso cae en exactamente una de las cuatro cajas: EX01, EX02, EX03 o la cohorte dura
    suma = b.select((pl.col("ex01").cast(pl.Int8) + pl.col("ex02").cast(pl.Int8) + pl.col("ex03").cast(pl.Int8) + pl.col("A").cast(pl.Int8)).alias("s"))["s"]
    assert suma.to_list() == [1] * 8
    # la versión del pipeline: el mapeo de tres caracteres deja pasar la diabetes y solo excluye a los sobrevivientes
    assert b["ex03_pipe_sd"].to_list() == [True, True, False, False, False, False, False, True]
    assert b["ex03_pipe"].to_list() == [True, True, False, False, False, False, False, False]   # la fallecida no cuenta como EX03 del pipeline


def test_variantes_estan_anidadas_y_cada_una_exige_su_condicion(cat, u, e, e3):
    filas = [
        _f(),                                                                                   # 0 adulta hospitalizada, vuelve a domicilio: E
        _f(FECHA_NACIMIENTO="2015-01-01"),                                                      # 1 menor: A pero no B
        _f(TIPO_ACTIVIDAD="CIRUGÍA MAYOR AMBULATORIA (CMA)"),                                   # 2 adulta ambulatoria: B pero no C
        _f(TIPOALTA="HOSPITALIZACIÓN DOMICILIARIA"),                                            # 3 censurada: C pero no D
        _f(TIPOALTA="FALLECIDO"),                                                               # 4 fallecida: D pero no E
        _f(FECHAALTA="2021-03-01"),                                                             # 5 sobreviviente con estadía cero: D pero no E
        _f(FECHAALTA=None),                                                                     # 6 sin fecha de alta: estadía desconocida, D pero no E
        _f(FECHA_NACIMIENTO=None),                                                              # 7 edad desconocida: A pero no B
    ]
    b = cohorte.marcar(cohorte.agregar_comorbilidad(cohorte.tipar(_marco(filas), 2021, cat, u), _dx([[] for _ in filas]), e, e3))
    for v, esperado in (
        ("A", [True] * 8),
        ("B", [True, False, True, True, True, True, True, False]),
        ("C", [True, False, False, True, True, True, True, False]),
        ("D", [True, False, False, False, True, True, True, False]),
        ("E", [True, False, False, False, False, False, False, False]),
    ):
        assert b[v].to_list() == esperado, v
    # anidamiento: E dentro de D dentro de C dentro de B dentro de A
    for chico, grande in (("E", "D"), ("D", "C"), ("C", "B"), ("B", "A")):
        assert b.filter(pl.col(chico) & ~pl.col(grande)).height == 0
    assert b["ambulatoria"].to_list()[2] is True and b["ambulatoria"].to_list()[0] is False
    # la lectura alternativa de la cohorte de estadía del pipeline también excluye a los censurados
    assert b["est_pipe"].to_list()[3] is True and b["est_pipe_sc"].to_list()[3] is False


def test_filas_de_cascada_rechazan_pasos_que_no_cuadran():
    d = {"total": 100, "ex01": 5, "ex02": 10, "ex03": 5, "A": 80, "B": 60, "C": 50}
    filas = cohorte.filas_cascada("2021", d, con_res=False)
    assert [f["paso"] for f in filas] == ["TOTAL", "EX01", "EX02", "EX03", "A", "EX04", "B", "EX05", "C"]
    assert [f["n"] for f in filas] == [100, 5, 10, 5, 80, 20, 60, 10, 50]
    with pytest.raises(ValueError, match="no cuadra"):
        cohorte.filas_cascada("2021", {**d, "A": 79}, con_res=False)
    with pytest.raises(ValueError, match="negativos"):
        cohorte.filas_cascada("2021", {**d, "D": 60, "E": 0, "D_fallecidos": 0}, con_res=True)


# -------------------------------------------------------------- corrida con datos sintéticos
@pytest.fixture(scope="module")
def corrida(tmp_path_factory, cat):
    raiz = tmp_path_factory.mktemp("t7")
    fuente = crear_fuente_sintetica(raiz / "datos", n=700, semilla=23)
    ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=raiz / "salidas")
    return ctx, cohorte.ejecutar(ctx, cat)


def test_la_cascada_suma_el_total_en_cada_periodo_y_los_periodos_suman_los_anios(corrida):
    _, r = corrida
    c = r["cascada"]
    por_periodo = {p[0]: dict(zip(g["paso"], g["n"])) for p, g in c.group_by("periodo")}
    assert set(por_periodo) == {"2019", "2020", "2021", "2022", "2023", "2024", "2019-2023", "2019-2024"}
    for periodo, d in por_periodo.items():
        assert d["TOTAL"] - d["EX01"] - d["EX02"] - d["EX03"] == d["A"], periodo
        assert d["A"] - d["EX04"] == d["B"] and d["B"] - d["EX05"] == d["C"], periodo
        if "D" in d:
            assert d["C"] - d["EX06"] == d["D"] and d["D"] - d["EX07"] - d["EX08"] == d["E"], periodo
    for paso in ("TOTAL", "EX01", "EX03", "A", "C", "D", "E"):
        assert por_periodo["2019-2023"][paso] == sum(por_periodo[str(a)][paso] for a in ANIOS_RES), paso
    for paso in ("TOTAL", "EX02", "B", "C"):
        assert por_periodo["2019-2024"][paso] == sum(por_periodo[str(a)][paso] for a in [*ANIOS_RES, 2024]), paso
    assert c["cuadra"].unique().to_list() == ["si"]


def _fecha(txt: str | None) -> dt.date | None:
    if not txt:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y"):
        try:
            return dt.datetime.strptime(txt, fmt).date()
        except ValueError:
            pass
    return None


def _oraculo(raw: pl.DataFrame, cat, e, anio: int) -> dict[str, int]:
    """Cascada de un año calculada fila a fila en Python puro (sin polars) como comprobación independiente."""
    cuentas = {k: 0 for k in ("total", "ex01", "ex02", "ex03", "A", "B", "C", "D", "E")}
    for fila in raw.iter_rows(named=True):
        cuentas["total"] += 1
        grd = (fila["IR_29301_COD_GRD"] or "").strip()
        if not re.fullmatch(r"\d{6}", grd) or grd[:2] == "99":
            cuentas["ex01"] += 1
            continue
        mdc = grd[:2]
        if mdc == "15":
            cuentas["ex02"] += 1
            continue
        if mdc == "14" and not any(elixhauser.categorias_de(fila[c], e) for c in esquema.DIAG_SEC if fila[c]):
            cuentas["ex03"] += 1
            continue
        cuentas["A"] += 1
        nac, ing, alta = _fecha(fila["FECHA_NACIMIENTO"]), _fecha(fila["FECHA_INGRESO"]), _fecha(fila["FECHAALTA"])
        edad = None
        if nac and ing:
            edad = ing.year - nac.year - ((ing.month, ing.day) < (nac.month, nac.day))
            edad = edad if 0 <= edad <= 110 else None
        _, _, rol = catalogos.rol_de_etiqueta(fila["TIPOALTA"], cat.roles_tipoalta) if fila["TIPOALTA"] else (None, None, None)
        if edad is None or edad < 18:
            continue
        cuentas["B"] += 1
        if fila["TIPO_ACTIVIDAD"] != "HOSPITALIZACIÓN":
            continue
        cuentas["C"] += 1
        if rol not in ("EVENTO", "NO_EVENTO"):
            continue
        cuentas["D"] += 1
        estancia = (alta - ing).days if (alta and ing and alta >= ing) else None
        if rol == "NO_EVENTO" and estancia is not None and estancia > 0:
            cuentas["E"] += 1
    return cuentas


@pytest.mark.parametrize("anio", [2021, 2023])  # 2021 con fechas ISO y 2023 con fechas día-mes-año
def test_la_cascada_coincide_con_un_calculo_independiente_en_python_puro(corrida, cat, e, anio):
    ctx, r = corrida
    raw = pl.read_csv(ctx.fuente.archivos[anio], separator="|", infer_schema=False, quote_char=None, encoding="utf8")
    raw = raw.rename({raw.columns[1]: "CIP_ENCRIPTADO"})
    esperado = _oraculo(raw, cat, e, anio)
    obtenido = r["por_anio"][anio]["cas"]
    for k in ("total", "ex01", "ex02", "ex03", "A", "B", "C", "D", "E"):
        assert obtenido[k] == esperado[k], (anio, k, obtenido[k], esperado[k])
    todos = r["por_anio"][anio]["roles"]["todos"]
    sobre_todos = {"EVENTO": 0, "CENSURADO": 0}
    for et in raw["TIPOALTA"].drop_nulls():
        _, _, rol = catalogos.rol_de_etiqueta(et, cat.roles_tipoalta)
        if rol in sobre_todos:
            sobre_todos[rol] += 1
    assert todos["EVENTO"] == sobre_todos["EVENTO"] and todos["CENSURADO"] == sobre_todos["CENSURADO"]


def test_mortalidad_por_anio_coincide_con_el_conteo_directo_del_archivo(corrida):
    ctx, r = corrida
    ra = r["resultado_anio"].filter(pl.col("base") == "todos")
    for anio in ANIOS_RES:
        raw = pl.read_csv(ctx.fuente.archivos[anio], separator="|", infer_schema=False, quote_char=None, encoding="utf8", columns=["TIPOALTA"])
        fila = ra.filter(pl.col("periodo") == str(anio)).row(0, named=True)
        assert fila["n"] == raw.height
        assert fila["n_fallecidos"] == int((raw["TIPOALTA"] == "FALLECIDO").sum())
        assert fila["ic_bajo"] < fila["mortalidad"] < fila["ic_alto"]
    agrupado = ra.filter(pl.col("periodo") == "2019-2023").row(0, named=True)
    assert agrupado["n"] == int(ra.filter(pl.col("periodo").str.contains(r"^\d{4}$"))["n"].sum())
    # la razón contra el año previo existe desde el segundo año y no en el primero
    assert ra.filter(pl.col("periodo") == "2019")["rr_previo"][0] is None and ra.filter(pl.col("periodo") == "2020")["rr_previo"][0] is not None


def test_2024_solo_aporta_estructura_y_el_intento_de_leer_su_desenlace_se_rechaza(corrida):
    ctx, r = corrida
    assert r["intento_2024"] == "rechazado"
    reg = ctx.cortafuegos.registro().filter(pl.col("anio") == 2024)
    bloqueados = reg.filter(pl.col("decision") == BLOQUEADO)
    assert bloqueados.height == 1 and bloqueados["columnas_resultado"][0] == "TIPOALTA" and bloqueados["proposito"][0] == "resultado"
    permitidos = reg.filter(pl.col("decision") != BLOQUEADO)
    assert permitidos.height >= 2 and set(permitidos["proposito"].to_list()) == {"estructura"}
    assert permitidos["columnas_resultado"].fill_null("").str.len_chars().sum() == 0   # ni en claro ni enmascarada
    assert ctx.cortafuegos.violaciones() == 0 and ctx.cortafuegos.n_bloqueos() == 1
    # ninguna tabla de resultado o de estadía contiene 2024
    for nombre in ("resultado_anio", "tipoalta", "estratos", "edad_sexo", "estancia_cero", "hist_estancia", "dispersion"):
        assert "2024" not in r[nombre]["periodo"].to_list(), nombre
    assert 2024 not in r["serie_mensual"]["anio"].to_list()
    assert r["cascada"].filter(pl.col("periodo") == "2024")["paso"].to_list() == ["TOTAL", "EX01", "EX02", "EX03", "A", "EX04", "B", "EX05", "C"]
    assert "2019-2024" not in r["cascada_pipeline"].filter(pl.col("paso") == "FALLECIDOS")["periodo"].to_list()
    v = r["variantes"].filter(pl.col("periodo") == "2024")
    assert v["variante"].to_list() == ["A", "B", "C"] and v["n_fallecidos"].null_count() == 3


def test_el_cortafuegos_rechaza_el_desenlace_de_2024_si_se_pide_con_otro_proposito(corrida, tmp_path):
    from eda.cortafuegos import AccesoBloqueado

    ctx = Contexto.crear("extracto", fuente=corrida[0].fuente, dir_salida_base=tmp_path)   # contexto nuevo: no altera el registro de la corrida
    for prop in ("resultado", "seleccion"):
        with pytest.raises(AccesoBloqueado):
            ctx.lector.leer(2024, ["FECHAALTA"], proposito=prop, tarea="T07")


def test_la_estadia_sin_dato_va_con_un_dia_o_mas_para_no_ocultar_el_estrato_de_cero_dias(corrida):
    """Cambio 1.6: la estadía sin dato no es una categoría propia de ``estancia_cero`` (con menos de diez egresos obligaba a ocultar «0 días»)."""
    _, r = corrida
    cats = set(r["estratos"].filter(pl.col("variable") == "estancia_cero")["categoria"].to_list())
    assert cats <= {"0 días", cohorte.CATEGORIA_NO_CERO} and "0 días" in cats and cohorte.SIN_DATO not in cats


def test_los_estratos_de_la_decision_15_marcan_el_umbral_y_las_razones_son_coherentes(corrida, u):
    _, r = corrida
    e = r["estratos"]
    amb = e.filter((pl.col("periodo") == "2019-2022") & (pl.col("variable") == "actividad_grupo"))
    assert set(amb["categoria"]) == {"hospitalización", "ambulatoria"}
    fila = amb.filter(pl.col("categoria") == "ambulatoria").row(0, named=True)
    hosp = amb.filter(pl.col("categoria") == "hospitalización").row(0, named=True)
    assert fila["bajo_umbral"] in ("si", "no") and hosp["bajo_umbral"] == ""
    assert fila["razon_vs_resto"] == pytest.approx(fila["mortalidad"] / hosp["mortalidad"], rel=1e-9)   # el resto de ambulatoria es hospitalización
    assert (fila["razon_vs_resto"] < float(u["estrato_mortalidad_relativa_max"])) == (fila["bajo_umbral"] == "si")
    assert fila["razon_bajo"] <= fila["razon_vs_resto"] <= fila["razon_alto"]
    # cada variable suma el mismo total de egresos en un período
    totales = e.filter(pl.col("periodo") == "2019-2022").group_by("variable").agg(pl.col("n").sum())
    assert totales["n"].n_unique() == 1


def test_estadia_cola_dispersion_e_histograma_son_coherentes(corrida, u):
    _, r = corrida
    cola = r["cola"].filter(pl.col("base") == "E")
    assert cola["regla"].to_list() == ["p99", "p99,5", "p99,9", "365 días"]
    assert cola["propuesta"].to_list().count("si") == 1
    assert cola["corte_dias"].to_list() == sorted(cola["corte_dias"].to_list())            # los percentiles más altos cortan más tarde
    assert cola["prop_afectados_def"].to_list() == sorted(cola["prop_afectados_def"].to_list(), reverse=True)
    assert cola.filter(pl.col("propuesta") == "si")["prop_afectados_def"][0] <= float(u["cola_estadia_perdida_max"])
    d = r["dispersion"].filter((pl.col("base") == "E") & (pl.col("periodo") == "2019-2022")).row(0, named=True)
    assert d["indice_dispersion"] == pytest.approx(d["varianza"] / d["media"])
    h = r["hist_estancia"].filter(pl.col("periodo") == "2019-2022")
    assert int(h["n"].sum()) == r["dispersion"].filter((pl.col("base") == "D_sobrevivientes") & (pl.col("periodo") == "2019-2022"))["n"][0]
    assert h["tramo"].to_list() == [t[0] for t in cohorte.TRAMOS_ESTANCIA]


def test_tipo_de_alta_suma_los_egresos_y_asigna_roles(corrida):
    _, r = corrida
    t = r["tipoalta"].filter(pl.col("periodo") == "2019-2023")
    assert int(t["n"].sum()) == int(r["resultado_anio"].filter((pl.col("periodo") == "2019-2023") & (pl.col("base") == "todos"))["n"][0])
    assert dict(zip(t["codigo"], t["rol"]))["FALLECIDO"] == "EVENTO" and dict(zip(t["codigo"], t["rol"]))["DOMICILIO"] == "NO_EVENTO"
    assert t["prop"].sum() == pytest.approx(1.0)
    assert "(sin rol)" not in t["etiqueta"].to_list()


def test_ejecutar_escribe_tablas_figuras_metricas_y_cumple_privacidad(corrida):
    ctx, _ = corrida
    for nombre in ("t23_cascada", "t23_cascada_pipeline", "t23_cascada_variantes", "t23_tipoalta", "t23_resultado_anio", "t23_resultado_estratos",
                   "t23_resultado_edad_sexo", "t23_serie_mensual", "t23_estancia_cero", "t23_estancia_hist", "t23_estancia_dispersion",
                   "t23_cola_estancia", "t23_resumen"):
        assert (ctx.dir_tablas / f"{nombre}.csv").is_file(), nombre
    assert ctx.figuras.ids() == ["F07_01", "F07_02", "F07_03", "F07_04", "F07_05"]
    m = ctx.metricas().filter(pl.col("tarea") == "T07")
    claves = set(m["clave"].to_list())
    for k in ("resultado.mortalidad_pct.2019", "resultado.mortalidad_pct.total", "cascada.n_ex01", "cascada.n_ex03_pipeline", "cascada.n_ex03_pipeline.cota_sup_2024",
              "tipoalta.pct.FALLECIDO", "tipoalta.pct.sobrevivientes", "cohorte.prevalencia_pct.C.2023", "estancia.mediana_dias", "estancia.pct_cero_dias",
              "actividad.pct_no_hospitalizacion", "cohorte.pct_estancia_total", "cortafuegos.intento_2024_rechazado"):
        assert k in claves, k
    alcance = dict(zip(m["clave"], m["alcance"]))
    assert alcance["cascada.n_ex01"] == "2019-2024" and alcance["cascada.n_ex03_pipeline"] == "2019-2023" and alcance["cascada.n_ex03_pipeline.cota_sup_2024"] == "2024"
    assert m.filter(pl.col("clave") == "cortafuegos.intento_2024_rechazado")["valor"][0] == 1.0
    ctx.escribir_registros()
    esc = privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)
    assert privacidad.total_violaciones(esc) == 0, esc.filter(pl.col("n_violaciones") > 0)


def test_el_informe_agrega_las_secciones_y_no_deja_marcadores_sin_resolver(corrida):
    ctx, _ = corrida
    inf = Informe(ctx, "t23_prueba.md", "Informe de prueba")
    cohorte.agregar_secciones(inf, ctx)
    md = inf.guardar().read_text(encoding="utf8")
    for titulo in ("Cohorte y cascada de exclusiones", "Prevalencia de la mortalidad por variante de cohorte", "Desenlace: tipo de alta y mortalidad",
                   "Estadía de los sobrevivientes", "Estadía cero y actividad ambulatoria", "Límites"):
        assert titulo in md, titulo
    for fig in ("F07_01", "F07_02", "F07_03", "F07_04", "F07_05"):
        assert f"Figura {fig}." in md
    assert "{" not in md and "}" not in md and re.search(r"\bnan\b", md.lower()) is None


# ---------------------------------------------------------------------- funciones sueltas
def test_deteccion_de_cambios_compara_con_el_mismo_mes_del_anio_anterior(u):
    anios = [2020] * 12 + [2021] * 12
    meses = list(range(1, 13)) * 2
    n = [1000] * 24
    k = [30] * 12 + [30, 30, 30, 30, 30, 90, 90, 90, 30, 30, 30, 30]
    q = cohorte.detectar_quiebres(anios, meses, n, k, 1.959964, 0.20)
    assert all(x["quiebre"] is None for x in q[:12])                                  # el primer año no tiene referencia
    assert [i - 12 + 1 for i, x in enumerate(q) if x["quiebre"] == "alza"] == [6, 7, 8]
    assert not any(x["quiebre"] == "baja" for x in q)
    assert q[18]["rr"] == pytest.approx(3.0) and q[18]["cambio"] == pytest.approx(2.0)
    # la estacionalidad no genera cambios: un ciclo repetido igual cada año no marca nada
    k_ciclo = [20, 20, 25, 30, 40, 50, 50, 40, 30, 25, 20, 20] * 2
    assert not any(x["quiebre"] in ("alza", "baja") for x in cohorte.detectar_quiebres(anios, meses, n, k_ciclo, 1.959964, 0.20))
    # una baja grande también se detecta
    k_baja = [60] * 12 + [20] * 12
    assert {x["quiebre"] for x in cohorte.detectar_quiebres(anios, meses, n, k_baja, 1.959964, 0.20)[12:]} == {"baja"}


def test_deteccion_exige_cambio_relevante_aunque_el_intervalo_excluya_la_igualdad():
    # 10 % de diferencia con muchísimos casos: el IC excluye 1 pero el cambio no alcanza el umbral
    q = cohorte.detectar_quiebres([2020, 2021], [1, 1], [1_000_000, 1_000_000], [30_000, 33_000], 1.959964, 0.20)
    assert q[1]["bajo"] > 1.0 and q[1]["quiebre"] == "no"


def test_histograma_y_estadisticas_de_estancia_desde_conteos():
    dias = {0: 5, 1: 3, 7: 2, 9: 4, 400: 1}
    h = dict((t, n) for _, t, n in cohorte.histograma_estancia(dias))
    assert h["0"] == 5 and h["1"] == 3 and h["7"] == 2 and h["8 a 9"] == 4 and h["365 o más"] == 1
    assert sum(h.values()) == sum(dias.values())
    est = cohorte.estadisticas_dias(dias)
    assert est["n"] == 15 and est["maximo"] == 400 and est["mediana"] == 1
    assert cohorte.estadisticas_dias({})["n"] == 0


def test_cola_de_estancia_propone_el_percentil_mas_bajo_que_cumple_y_aplica_el_corte_a_2023(u):
    def anio(n_largas: int) -> dict:
        dias = {d: 100 for d in range(1, 61)}      # 6.000 sobrevivientes con 1 a 60 días
        dias[200] = n_largas                        # y unos pocos con estadía muy larga
        return {"roles": {}, "dias_E": dias, "dias_D": dias}

    por_anio = {a: anio(2) for a in range(2019, 2023)} | {2023: anio(100)}   # en 2023 hay muchas más estadías largas
    cola = cohorte.tabla_cola_estancia(por_anio, u).filter(pl.col("base") == "E")
    p99 = cola.filter(pl.col("regla") == "p99").row(0, named=True)
    assert p99["corte_dias"] == 60 and p99["n_afectados_def"] == 8 and p99["cumple_def"] == "si" and p99["propuesta"] == "si"
    assert p99["n_afectados_conf"] == 100 and p99["cumple_conf"] == "no"                        # la confirmación en 2023 falla
    assert cola.filter(pl.col("regla") == "365 días")["n_afectados_def"][0] == 0


def test_las_tablas_con_filas_empatadas_en_n_salen_en_un_orden_que_no_depende_del_orden_de_entrada(u):
    """B0 (DT11): ``tabla_estratos`` y ``tabla_tipoalta`` desempatan por la categoría; sin eso dos corridas podían intercambiar filas empatadas."""
    import itertools

    estratos = [("tipo_procedencia", "OTRAS INSTITUCIONES", 119, 7), ("tipo_procedencia", "CMA", 119, 5), ("tipo_procedencia", "AMBULATORIA", 119, 3),
                ("tipo_procedencia", "URGENCIA", 400, 20)]
    tipoalta = [("A", "EVENTO", 3), ("B", "EVENTO", 3), ("C", "NO_EVENTO", 3), ("D", "NO_EVENTO", 50)]
    ordenes_estratos, ordenes_tipoalta = set(), set()
    for perm in itertools.permutations(range(4)):
        de = pl.DataFrame([estratos[i] for i in perm], schema=["variable", "categoria", "n", "n_fallecidos"], orient="row")
        t = cohorte.tabla_estratos({a: {"roles": {}, "estratos": de} for a in (2019, 2020)}, u)
        ordenes_estratos.add(tuple(t.filter(pl.col("periodo") == "2019")["categoria"].to_list()))
        dt_ = pl.DataFrame([tipoalta[i] for i in perm], schema=["etiqueta_roles", "rol", "n"], orient="row")
        t = cohorte.tabla_tipoalta({a: {"roles": {}, "tipoalta": dt_} for a in (2019, 2020)}, u)
        ordenes_tipoalta.add(tuple(t.filter(pl.col("periodo") == "2019")["etiqueta"].to_list()))
    # primero la categoría con más egresos y, entre empatadas, el orden alfabético
    assert ordenes_estratos == {("URGENCIA", "AMBULATORIA", "CMA", "OTRAS INSTITUCIONES")}
    assert ordenes_tipoalta == {("D", "A", "B", "C")}
