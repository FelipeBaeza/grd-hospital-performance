"""Panel compartido de las Tareas 10 y 11: registro de variables, funciones puras, historial, cardinalidad y construcción sobre datos sintéticos."""

from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np
import polars as pl
import pytest

from eda import catalogos, cohorte, config, elixhauser, esquema, hospitales, privacidad
from eda import decision as dc
from eda import panel as pn
from eda.contexto import Contexto
from panel_artificial import umbrales_temporales
from sintetico import crear_fuente_sintetica, escribir_archivo, generar_anio


@pytest.fixture(scope="module")
def u():
    """Umbrales del proyecto. Las cuentas a mano de este archivo suponen los valores propuestos; si cambian, la prueba lo dice."""
    u = config.cargar_umbrales()
    supuestos = {"celda_minima": 10, "edad_adulto": 18, "cardinalidad_max_categorica": 60, "max_filas_asociacion_por_anio": 0, "ventana_historial_dias": 365}
    for clave, valor in supuestos.items():
        assert u[clave] == valor, f"las cuentas a mano de este archivo suponen {clave} = {valor}"
    assert u.meta["semilla"] == 20261007
    return u


def _s(valores, nombre="x"):
    return pl.Series(nombre, valores, dtype=pl.String)


# =============================================================================================== etiquetas y columnas
def test_las_etiquetas_de_ventana_son_las_nominales_de_la_particion():
    assert (pn.V_RANK, pn.V_CONF) == ("2019-2022", "2023")
    assert pn.MIN_FILAS_HOSPITAL == 100


def test_las_columnas_que_lee_el_panel_no_incluyen_las_prohibidas_ni_el_identificador():
    cols = pn.COLUMNAS_PANEL
    assert len(cols) == len(set(cols)) and set(cols) <= set(esquema.COLUMNAS)
    prohibidas = {"FECHAPROCEDIMIENTO1", "MEDICOINTERV1_ENCRIPTADO", "MEDICOALTA_ENCRIPTADO", "CIP_ENCRIPTADO", *esquema.BLOQUE_RN}
    assert not prohibidas & set(cols)
    assert {"FECHA_NACIMIENTO", "TIPOALTA", "FECHAALTA", "COD_HOSPITAL", "HOSPPROCEDENCIA", "USOSPABELLON", "IR_29301_PESO", *esquema.DIAG_SEC, *esquema.PROC, *esquema.FECHAS_TRASLADO} <= set(cols)
    assert pn.columnas_panel(False) == list(cols)
    con = pn.columnas_panel(True)
    assert con.index("CIP_ENCRIPTADO") == con.index("HOSPPROCEDENCIA") + 1 and len(con) == len(cols) + 1 and [c for c in con if c != "CIP_ENCRIPTADO"] == list(cols)
    assert cols[: len(cohorte.COLS_BASE)] == cohorte.COLS_BASE and cols[len(cohorte.COLS_BASE): len(cohorte.COLS_BASE) + 2] == cohorte.COLS_RESULTADO      # el orden del diseño


# =============================================================================================== registro de variables
ROLES_ESPERADOS = {
    "EDAD_ANIOS": "FEATURE_BASAL", "N_DX_SEC": "AUDITORIA_SENSIBILIDAD", "N_COMORB_ELIX": "FEATURE_BASAL", "PUNTAJE_VAN_WALRAVEN": "FEATURE_BASAL",
    "PUNTAJE_VAN_WALRAVEN_SIN_5": "AUDITORIA_SENSIBILIDAD", "N_EGRESOS_12M": "FEATURE_BASAL", "DIAS_DESDE_EGRESO_PREVIO": "FEATURE_BASAL",
    **{f"ELIX_{i:02d}": "FEATURE_BASAL" for i in range(1, 32)},
    "ES_FIN_SEMANA": "FEATURE_BASAL", "NIVEL_CUIDADO_INGRESO": "CONDICIONAL", "DERIVADO_OTRO_HOSPITAL": "CONDICIONAL", "ENTRANTE_TIPO_PROCEDENCIA": "FILTRO_COHORTE",
    "HISTORIA_DISPONIBLE": "FEATURE_BASAL", "ES_EXTRANJERO": "AUDITORIA_SENSIBILIDAD", "PUEBLO_ORIGINARIO": "AUDITORIA_SENSIBILIDAD",
    "SEXO": "FEATURE_BASAL", "TIPO_INGRESO": "FEATURE_BASAL", "PROCEDENCIA_AGR": "FEATURE_BASAL", "TIPO_PROCEDENCIA": "FEATURE_BASAL",
    "GRUPO_CLINICO": "FEATURE_BASAL", "CIE10_3C": "FEATURE_BASAL", "MES_INGRESO": "FEATURE_BASAL", "DIA_SEMANA": "FEATURE_BASAL",
    "ESPECIALIDAD_MEDICA": "CONDICIONAL", "SERVICIOINGRESO": "CONDICIONAL", "PREVISION": "AUDITORIA_SENSIBILIDAD", "NACIONALIDAD": "AUDITORIA_SENSIBILIDAD",
    "PROVINCIA": "AUDITORIA_SENSIBILIDAD", "COMUNA": "AUDITORIA_SENSIBILIDAD", "ETNIA": "AUDITORIA_SENSIBILIDAD", "SERVICIO_SALUD": "LLAVE_AGREGACION",
    "COD_HOSPITAL": "LLAVE_AGREGACION", "IR_29301_SEVERIDAD": "PROHIBIDA_FUGA", "IR_29301_MORTALIDAD": "PROHIBIDA_FUGA", "IR_29301_PESO": "PROHIBIDA_FUGA",
    "SERVICIOALTA": "PROHIBIDA_FUGA", "USOSPABELLON": "PROHIBIDA_FUGA", "N_PROCEDIMIENTOS": "PROHIBIDA_FUGA", "N_TRASLADOS_INTERNOS": "PROHIBIDA_FUGA",
    "ESTANCIA_DIAS": "TARGET", "ES_COHORTE_E": "TARGET", "SERVICIO_INGRESO_IGUAL_ALTA": "PROHIBIDA_FUGA", "ESPECIALIDAD_MACRO": "CONDICIONAL",
    "TRAMO_FONASA": "AUDITORIA_SENSIBILIDAD", "INDICE_PRIVACION": "AUDITORIA_SENSIBILIDAD",
}


def test_cada_rol_del_registro_coincide_con_los_roles_esperados_del_diseno_y_con_la_precedencia_de_documentos():
    for con_historial in (False, True):
        reg = pn.registro_variables(con_historial)
        assert {n: v.rol_fase1 for n, v in reg.items()} == ROLES_ESPERADOS
        for n, v in reg.items():
            assert v.rol_fase1 == dc.rol_fase1(n)[0], n                                    # precedencia: reglas_variables, diccionario, derivadas, herencia


def test_el_registro_tiene_las_variables_del_diseno_en_orden_estable_y_sin_repetidos():
    reg = pn.registro_variables(False)
    nombres = list(reg)
    assert len(nombres) == len(set(nombres)) == 75
    assert nombres[:7] == ["EDAD_ANIOS", "N_DX_SEC", "N_COMORB_ELIX", "PUNTAJE_VAN_WALRAVEN", "PUNTAJE_VAN_WALRAVEN_SIN_5", "N_EGRESOS_12M", "DIAS_DESDE_EGRESO_PREVIO"]
    assert [f"ELIX_{i:02d}" for i in range(1, 32)] == nombres[7:38]
    assert nombres[-5:] == ["ES_COHORTE_E", "SERVICIO_INGRESO_IGUAL_ALTA", "ESPECIALIDAD_MACRO", "TRAMO_FONASA", "INDICE_PRIVACION"]
    assert list(pn.registro_variables(False)) == list(pn.registro_variables(True)) == nombres
    assert all(v.tipo in {"num", "bin", "cat"} and v.momento in {"basal", "alta", "durante"} and v.evaluable in {"si", "no", "parcial"} for v in reg.values())
    assert all(v.uso in {"candidata", "solo_fuga", "referencia", "resultado", "auxiliar", "no_construible"} for v in reg.values())


def test_variables_candidatas_evaluables_en_extracto_y_en_modo_completo():
    ext, comp = pn.registro_variables(False), pn.registro_variables(True)

    def cuenta(reg, tipo):
        return sum(1 for v in reg.values() if v.uso == "candidata" and v.evaluable != "no" and v.tipo == tipo)

    assert (cuenta(ext, "num"), cuenta(ext, "bin"), cuenta(ext, "cat")) == (5, 37, 16)           # 58 en un extracto
    assert (cuenta(comp, "num"), cuenta(comp, "bin"), cuenta(comp, "cat")) == (7, 38, 16)        # 61 en modo completo
    for h in ("N_EGRESOS_12M", "DIAS_DESDE_EGRESO_PREVIO", "HISTORIA_DISPONIBLE"):
        assert ext[h].evaluable == "no" and ext[h].motivo == "no evaluable en extracto: el ingreso previo casi nunca está en el mismo extracto"
    assert [comp[h].evaluable for h in ("N_EGRESOS_12M", "DIAS_DESDE_EGRESO_PREVIO", "HISTORIA_DISPONIBLE")] == ["parcial", "parcial", "si"]
    assert comp["N_EGRESOS_12M"].motivo == "solo vale donde HISTORIA_DISPONIBLE es 1" and comp["HISTORIA_DISPONIBLE"].motivo == ""


def test_toda_variable_no_evaluable_o_parcial_tiene_motivo_y_las_evaluables_no():
    for con_historial in (False, True):
        for n, v in pn.registro_variables(con_historial).items():
            assert (v.evaluable == "si") == (v.motivo == ""), n


def test_etnia_pueblo_originario_cod_hospital_y_las_que_no_se_pueden_construir():
    reg = pn.registro_variables(False)
    for n in ("ETNIA", "PUEBLO_ORIGINARIO"):
        assert reg[n].evaluable == "parcial" and reg[n].analiza_resultado is False and reg[n].motivo == "decisión 4: no se analiza frente a resultados" and reg[n].uso == "candidata"
    assert [n for n, v in reg.items() if not v.analiza_resultado] == ["PUEBLO_ORIGINARIO", "ETNIA"]
    h = reg["COD_HOSPITAL"]
    assert (h.uso, h.es_referencia, h.evaluable, h.motivo) == ("referencia", True, "no", "referencia del análisis de proxy, no es candidata")
    assert [n for n, v in reg.items() if v.es_referencia] == ["COD_HOSPITAL"]
    assert [n for n, v in reg.items() if v.uso == "no_construible"] == ["ESPECIALIDAD_MACRO", "TRAMO_FONASA", "INDICE_PRIVACION"]
    assert all(reg[n].evaluable == "no" and reg[n].motivo for n in ("ESPECIALIDAD_MACRO", "TRAMO_FONASA", "INDICE_PRIVACION"))
    assert [n for n, v in reg.items() if v.uso == "solo_fuga"] == [
        "IR_29301_SEVERIDAD", "IR_29301_MORTALIDAD", "IR_29301_PESO", "SERVICIOALTA", "USOSPABELLON", "N_PROCEDIMIENTOS", "N_TRASLADOS_INTERNOS"]
    assert [n for n, v in reg.items() if v.uso == "resultado"] == ["ESTANCIA_DIAS"] and [n for n, v in reg.items() if v.uso == "auxiliar"] == ["ES_COHORTE_E", "SERVICIO_INGRESO_IGUAL_ALTA"]
    assert [n for n, v in reg.items() if v.vacio_es_ausencia] == ["USOSPABELLON"]


def test_el_registro_declara_las_fuentes_crudas_de_cada_variable():
    reg = pn.registro_variables(False)
    assert reg["EDAD_ANIOS"].fuentes == ("FECHA_INGRESO", "FECHA_NACIMIENTO") and reg["ELIX_09"].fuentes == tuple(esquema.DIAG_SEC)
    assert reg["COMUNA"].fuentes == ("COMUNA",) and reg["ES_COHORTE_E"].fuentes == ("TIPOALTA", "FECHAALTA", "FECHA_INGRESO")
    for v in reg.values():
        assert set(v.fuentes) <= set(esquema.COLUMNAS), v.nombre


def test_el_registro_es_inmutable_y_el_universo_de_decision_contiene_las_variables_no_auxiliares():
    reg = pn.registro_variables(False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        reg["EDAD_ANIOS"].tipo = "cat"
    universo = set(dc.cargar_universo()["variable"])
    no_auxiliares = {n for n, v in reg.items() if v.uso != "auxiliar"}
    assert no_auxiliares <= universo and not (set(n for n, v in reg.items() if v.uso == "auxiliar") & universo)


def test_la_disponibilidad_se_arma_con_el_registro_real(ctx):
    for con_historial in (False, True):
        t = dc.construir_disponibilidad(ctx, pn.registro_variables(con_historial))
        assert t.height == 199 and (t.filter(pl.col("evaluable") != "si")["motivo_no_evaluable"].str.len_chars() > 0).all()
        f = {r["variable"]: r for r in t.iter_rows(named=True)}
        assert f["COD_HOSPITAL"]["evaluable"] == "no" and f["ESTANCIA_DIAS"]["motivo_no_evaluable"] == "variable de resultado: no es candidata"
        assert f["N_EGRESOS_12M"]["evaluable"] == ("parcial" if con_historial else "no") and f["USOSPABELLON"]["vacio_es_ausencia"] == "si"
        assert f["EDAD_ANIOS"]["tipo"] == "num" and f["SEXO"]["tipo"] == "cat" and f["ELIX_12"]["tipo"] == "bin" and f["DIAGNOSTICO4"]["equivalente_panel"] != ""


# =============================================================================================== procedencia
REALES_MAPEADOS = {
    "SERVICIO EMERGENCIA (DOMICILIO)": "URGENCIA", "APS URGENCIA (SAPU, SUR, SUC)": "URGENCIA",
    "OTROS HOSPITALES DE LA RED": "DERIVADO_OTRO_HOSPITAL", "OTROS HOSPITALES RED NACIONAL": "DERIVADO_OTRO_HOSPITAL",
    "CENTRO ESPECIALIDADES (CDT, CRS, CONSULTORIO ADOS. ESP)": "PROGRAMADO", "APS CONSULTORIO (CESFAM)": "PROGRAMADO", "CONSULTA PRIVADA": "PROGRAMADO",
    "PLAN DE RESOLUCIÓN LE": "PROGRAMADO", "CIRUGÍA MAYOR AMBULATORIA (CMA)": "PROGRAMADO", "CARDIOCIRUGÍA PAGO GRD": "PROGRAMADO",
}
REALES_SIN_MAPEO = [
    "OTRAS INSTITUCIONES SALUD (CLÍNICAS PRIVADAS, DE REHABILITAC", "OTRAS INSTITUCIONES (CÁRCEL, HOGARES DE ANCIANOS, SENAME, EC", "ESTRATEGIA CRR",
    "HOSPITALIZACIÓN DOMICILIARIA", "HOSPITALIZACIÓN DIURNA", "POSTA RURAL", "UGCC",
]
REALES_SIN_DATO = ["DESCONOCIDO", "NO IDENTIFICADO"]


def test_agrupar_procedencia_asigna_cada_uno_de_los_19_valores_reales():
    assert len(REALES_MAPEADOS) + len(REALES_SIN_MAPEO) + len(REALES_SIN_DATO) == 19
    todos = [*REALES_MAPEADOS, *REALES_SIN_MAPEO, *REALES_SIN_DATO]
    esperado = [*REALES_MAPEADOS.values(), *[pn.SIN_MAPEO] * 7, *[pn.SIN_DATO] * 2]
    assert pn.agrupar_procedencia(_s(todos)).to_list() == esperado
    assert pn.agrupar_procedencia(_s(["UN VALOR NUEVO DEL AGRUPADOR"])).to_list() == [pn.SIN_MAPEO]             # un valor nuevo no se adivina
    assert pn.agrupar_procedencia(_s([None, "", "  ", "no identificado", " desconocido "])).to_list() == [pn.SIN_DATO] * 5
    assert pn.agrupar_procedencia(_s(["servicio emergencia (domicilio)", "Cirugía mayor ambulatoria (CMA)"])).to_list() == ["URGENCIA", "PROGRAMADO"]
    assert pn.agrupar_procedencia(_s([])).len() == 0


def test_las_claves_del_mapa_estan_normalizadas_y_cirugia_mayor_ambulatoria_cae_en_programado():
    for clave in pn.MAPA_PROCEDENCIA_AGR:
        assert clave == hospitales.normalizar_nombre(clave), clave
    assert pn.MAPA_PROCEDENCIA_AGR["CIRUGIA MAYOR AMBULATORIA CMA"] == "PROGRAMADO" and len(pn.MAPA_PROCEDENCIA_AGR) == 10
    assert pn.agrupar_procedencia(_s(["CIRUGÍA MAYOR AMBULATORIA (CMA)"])).to_list() == ["PROGRAMADO"]
    assert set(pn.MAPA_PROCEDENCIA_AGR.values()) == {"URGENCIA", "DERIVADO_OTRO_HOSPITAL", "PROGRAMADO"}
    assert pn.construir_mapa_procedencia(pn._PROCEDENCIA_CRUDA) == pn.MAPA_PROCEDENCIA_AGR


def test_el_mapa_de_procedencia_rechaza_dos_textos_que_se_normalizan_igual_con_grupos_distintos():
    with pytest.raises(ValueError, match="se normalizan igual"):
        pn.construir_mapa_procedencia({"Consulta privada": "PROGRAMADO", "CONSULTA  PRIVADA!": "URGENCIA"})
    # con el mismo grupo no hay conflicto
    assert pn.construir_mapa_procedencia({"Consulta privada": "PROGRAMADO", "CONSULTA  PRIVADA!": "PROGRAMADO"}) == {"CONSULTA PRIVADA": "PROGRAMADO"}


def test_entrante_tipo_procedencia_sigue_la_definicion_de_la_tarea_9_y_nunca_es_nulo():
    vals = ["OTROS HOSPITALES DE LA RED", "OTROS HOSPITALES RED NACIONAL", "SERVICIO EMERGENCIA (DOMICILIO)", None, "DESCONOCIDO", "Otros Hospitales (red)"]
    assert pn.entrante_tipo_procedencia(_s(vals)).to_list() == [1, 1, 0, 0, 0, 1]
    assert pn.entrante_tipo_procedencia(_s(vals)).dtype == pl.Int8
    assert hospitales.TEXTO_ENTRANTE == "OTROS HOSPITALES"


# =============================================================================================== binarias derivadas
def test_nivel_cuidado_ingreso_reconoce_uci_y_tratamiento_intermedio_y_deja_nulo_lo_no_informado():
    uno = ["UNIDAD DE CUIDADOS INTENSIVOS ADULTO", "UNIDAD DE TRATAMIENTO INTERMEDIO (UTI) (INDIFERENCIADO) ADULTO", "UTI ADULTOS", "UCI",
           "UNIDAD DE CUIDADO INTENSIVO PEDIÁTRICO", "tratamientos intermedios", "Unidad de Cuidados Intensivos (U.C.I.)"]
    cero = ["UNIDAD DE RECUPERACIÓN DE PABELLONES (CENTRAL Y CMA)", "AREA MEDICO-QUIRURGICO CUIDADOS MEDIOS", "UNIDAD EMERGENCIA ADULTOS", "MEDICINA", "AGUDOS", "UCIA", "SUCIO"]
    assert pn.nivel_cuidado_ingreso(_s(uno)).to_list() == [1] * len(uno)
    assert pn.nivel_cuidado_ingreso(_s(cero)).to_list() == [0] * len(cero)               # "UCIA" y "SUCIO" no son la sigla: el patrón exige palabra completa
    assert pn.nivel_cuidado_ingreso(_s([None, "", "  ", "DESCONOCIDO", "no identificado"])).to_list() == [None] * 5
    assert pn.nivel_cuidado_ingreso(_s(uno + cero)).dtype == pl.Int8 and pn.PATRONES_CUIDADO_ESPECIAL == (r"\bUCI\b", r"\bUTI\b", r"CUIDADOS? INTENSIVOS?", r"TRATAMIENTOS? INTERMEDIOS?")


def test_es_extranjero_y_es_pueblo_originario():
    assert pn.es_extranjero(_s(["CHILE", "CHILENA", "chileno", "PERÚ", "VENEZOLANA", "DESCONOCIDO", None, ""])).to_list() == [0, 0, 0, 1, 1, None, None, None]
    assert pn.es_pueblo_originario(_s(["NINGUNO", "NINGUNA", "OTRO", "OTRO ", "otro", "MAPUCHE", "AYMARA", "DESCONOCIDO", None])).to_list() == [0, 0, 0, 0, 0, 1, 1, None, None]
    assert pn.es_extranjero(_s(["PERÚ"])).dtype == pl.Int8 and pn.es_pueblo_originario(_s(["MAPUCHE"])).dtype == pl.Int8


def test_limpiar_categorica_recorta_y_lleva_nulo_vacio_y_centinela_a_sin_dato_sin_cambiar_mayusculas():
    r = pn.limpiar_categorica(_s(["OTRO ", "OTRO", " Mapuche ", None, "", "  ", "DESCONOCIDO", "No identificada", "N/A", "Chilena"], "ETNIA"))
    assert r.to_list() == ["OTRO", "OTRO", "Mapuche", pn.SIN_DATO, pn.SIN_DATO, pn.SIN_DATO, pn.SIN_DATO, pn.SIN_DATO, pn.SIN_DATO, "Chilena"] and r.name == "ETNIA"
    s = pn.limpiar_categorica(_s(["HOMBRE", "SIN_DATO", "MUJER"], "sexo"), extra_sin_dato=("SIN_DATO",))
    assert s.to_list() == ["HOMBRE", pn.SIN_DATO, "MUJER"]
    assert pn.limpiar_categorica(_s(["SIN_DATO"], "sexo")).to_list() == ["SIN_DATO"]            # sin el extra, SIN_DATO no es centinela
    assert pn.limpiar_categorica(_s([None, None], "x")).to_list() == [pn.SIN_DATO] * 2


# =============================================================================================== tope de niveles
def _frecuencias(spec: dict[str, tuple[int, int]]) -> tuple[pl.Series, pl.Series]:
    """Serie de valores y de años a partir de {nivel: (n en 2019-2022, n en 2023)}."""
    v, a = [], []
    for nivel, (n_rank, n_conf) in spec.items():
        v += [nivel] * (n_rank + n_conf)
        a += [2019 + (i % 4) for i in range(n_rank)] + [2023] * n_conf
    return pl.Series("v", v, dtype=pl.String), pl.Series("a", a, dtype=pl.Int64)


RANK = [2019, 2020, 2021, 2022]


def test_setenta_niveles_conservan_los_60_mas_frecuentes_de_la_ventana_de_ranking():
    spec = {f"N{i:02d}": (200 - i, 0) for i in range(70)}                       # frecuencias estrictamente decrecientes
    v, a = _frecuencias(spec)
    n = pn.calcular_niveles(v, a, RANK, 60)
    assert n.niveles[:60] == [f"N{i:02d}" for i in range(60)] and n.niveles[60:] == [pn.OTROS] and n.otros == pn.OTROS and len(n.niveles) == 61
    r = pn.recodificar(v, n)
    assert set(r.unique()) == set(n.niveles) and int((r == pn.OTROS).sum()) == sum(200 - i for i in range(60, 70))


def test_el_empate_en_el_corte_de_60_se_decide_por_texto_ascendente():
    spec = {f"A{i:02d}": (500 - i, 0) for i in range(58)} | {"Z": (10, 0), "B": (10, 0), "M": (10, 0)}
    v, a = _frecuencias(spec)
    n = pn.calcular_niveles(v, a, RANK, 60)
    assert n.niveles[58:60] == ["B", "M"] and "Z" not in n.niveles and n.niveles[-1] == pn.OTROS               # empatan en 10: gana el texto menor


def test_un_nivel_que_solo_existe_en_2023_pasa_a_otros_aunque_haya_menos_de_60_niveles():
    v, a = _frecuencias({"A": (100, 30), "B": (50, 10), "NUEVO": (0, 25)})
    n = pn.calcular_niveles(v, a, RANK, 60)
    assert n.niveles == ["A", "B", pn.OTROS] and n.otros == pn.OTROS
    r = pn.recodificar(v, n)
    assert int((r == pn.OTROS).sum()) == 25 and "NUEVO" not in set(r)


def test_sin_dato_y_sin_mapeo_no_ocupan_cupo_ni_se_fusionan_y_van_al_final_en_su_orden():
    spec = {"A": (100, 0), "B": (90, 0), pn.SIN_DATO: (500, 5), pn.SIN_MAPEO: (300, 2), "C": (80, 0), "D": (70, 0)}
    v, a = _frecuencias(spec)
    n = pn.calcular_niveles(v, a, RANK, 2)
    assert n.niveles == ["A", "B", pn.SIN_MAPEO, pn.OTROS, pn.SIN_DATO]                                         # (sin dato) con 500 casos no desplaza a A ni a B
    r = pn.recodificar(v, n)
    assert int((r == pn.SIN_DATO).sum()) == 505 and int((r == pn.SIN_MAPEO).sum()) == 302 and int((r == pn.OTROS).sum()) == 150
    # sin nada que agrupar y sin protegidos, el nivel OTROS no existe
    v2, a2 = _frecuencias({"A": (10, 0), "B": (5, 3)})
    assert pn.calcular_niveles(v2, a2, RANK, 60).niveles == ["A", "B"] and pn.calcular_niveles(v2, a2, RANK, 60).otros is None


def test_un_nivel_real_llamado_otros_fuerza_otros_agrupados():
    v, a = _frecuencias({"A": (100, 0), "B": (90, 0), "OTROS": (80, 0), "C": (5, 0), "D": (4, 0)})
    n = pn.calcular_niveles(v, a, RANK, 3)
    assert n.niveles == ["A", "B", "OTROS", pn.OTROS_AGRUPADOS] and n.otros == pn.OTROS_AGRUPADOS
    r = pn.recodificar(v, n)
    assert int((r == "OTROS").sum()) == 80 and int((r == pn.OTROS_AGRUPADOS).sum()) == 9                       # el real y el agrupado no se mezclan
    assert pn._tipo_nivel("OTROS", n) == "real" and pn._tipo_nivel(pn.OTROS_AGRUPADOS, n) == "otros"
    # si el nivel real OTROS no entra al cupo, se agrupa con los demás bajo el nombre nuevo
    n2 = pn.calcular_niveles(v, a, RANK, 2)
    assert n2.niveles == ["A", "B", pn.OTROS_AGRUPADOS] and int((pn.recodificar(v, n2) == pn.OTROS_AGRUPADOS).sum()) == 89


def test_la_referencia_cod_hospital_no_tiene_tope():
    hosp = pl.Series("COD_HOSPITAL", [f"{101100 + 1000 * (i % 68):06d}" for i in range(680)] + ["(sin dato)"])
    n = pn.calcular_niveles_referencia(hosp)
    assert len(n.niveles) == 69 and n.otros is None and n.niveles == sorted(set(hosp.to_list()))
    assert pn.recodificar(hosp, n).to_list() == hosp.to_list()                                                # nada se agrupa


def test_los_niveles_se_fijan_con_la_ventana_de_ranking_cambiar_la_ventana_cambia_la_lista():
    v, a = _frecuencias({"A": (10, 0), "B": (5, 100), "C": (3, 100)})
    con_ranking = pn.calcular_niveles(v, a, RANK, 2)
    con_todo = pn.calcular_niveles(v, a, RANK + [2023], 2)
    assert con_ranking.niveles == ["A", "B", pn.OTROS]                                                        # A 10, B 5 y C 3 en 2019-2022: C queda en OTROS
    assert pn.calcular_niveles(v, a, [2023], 2).niveles == ["B", "C", pn.OTROS]                               # con 2023 como ventana saldrían otros niveles
    assert con_todo.niveles == ["B", "C", pn.OTROS]                                                           # y con 2019 a 2023 también: usaría información de 2023 para fijarlos
    assert pn.calcular_niveles(v, a, RANK, 2).niveles == con_ranking.niveles                                  # determinista


def test_la_tabla_de_niveles_es_larga_por_nivel_y_ventana_con_ceros_informados_y_prop_sobre_la_ventana():
    v, a = _frecuencias({"A": (40, 4), "B": (60, 0), "C": (0, 6), pn.SIN_DATO: (10, 0)})
    n = pn.calcular_niveles(v, a, RANK, 2)
    r = pn.recodificar(v, n)
    t = pn.tabla_de_niveles("TIPO_INGRESO", r, a, n, [(pn.V_RANK, RANK), (pn.V_CONF, [2023])])
    assert t.columns == ["variable", "nivel", "orden", "tipo_nivel", "ventana", "n", "prop"] and t.schema == pl.Schema(pn.ESQUEMA_NIVELES_TABLA)
    f = {(r_["ventana"], r_["nivel"]): r_ for r_ in t.iter_rows(named=True)}
    assert n.niveles == ["B", "A", pn.OTROS, pn.SIN_DATO]
    assert f[(pn.V_RANK, "A")]["n"] == 40 and f[(pn.V_RANK, "A")]["prop"] == pytest.approx(40 / 110) and f[(pn.V_CONF, "A")]["n"] == 4 and f[(pn.V_CONF, "A")]["prop"] == pytest.approx(0.4)
    assert f[(pn.V_RANK, pn.OTROS)]["n"] == 0 and f[(pn.V_RANK, pn.OTROS)]["prop"] == 0.0 and f[(pn.V_CONF, pn.OTROS)]["n"] == 6      # el cero se informa
    assert f[(pn.V_CONF, "B")]["n"] == 0 and f[(pn.V_CONF, pn.SIN_DATO)]["n"] == 0
    assert [f[(pn.V_RANK, x)]["tipo_nivel"] for x in ("B", pn.OTROS, pn.SIN_DATO)] == ["real", "otros", "sin_dato"]
    assert [f[(pn.V_RANK, x)]["orden"] for x in ("B", "A", pn.OTROS, pn.SIN_DATO)] == [1, 2, 3, 4]
    assert t.filter(pl.col("ventana") == pn.V_RANK)["prop"].sum() == pytest.approx(1.0) and t.filter(pl.col("ventana") == pn.V_CONF)["prop"].sum() == pytest.approx(1.0)
    solo_rank = pn.tabla_de_niveles("TIPO_INGRESO", r.filter(a != 2023), a.filter(a != 2023), n, [(pn.V_RANK, RANK), (pn.V_CONF, [2023])])
    assert set(solo_rank["ventana"]) == {pn.V_RANK}                                                                    # sin filas en la ventana no hay proporción que informar


# =============================================================================================== muestreo
def test_muestrear_indices_es_la_identidad_si_no_supera_el_tope_y_si_no_sigue_el_oraculo():
    assert pn.muestrear_indices(10, 10, 20261007, 2019).tolist() == list(range(10)) and pn.muestrear_indices(0, 5, 1, 2019).size == 0
    n, tope, semilla = 1000, 150, 20261007
    for anio in (2019, 2020, 2023):
        esperado = np.sort(np.random.default_rng(semilla + anio).choice(n, 150, replace=False))
        obtenido = pn.muestrear_indices(n, tope, semilla, anio)
        assert np.array_equal(obtenido, esperado) and np.all(np.diff(obtenido) > 0) and obtenido.size == tope and obtenido.max() < n
        assert np.array_equal(obtenido, pn.muestrear_indices(n, tope, semilla, anio))                                  # reproducible
    assert not np.array_equal(pn.muestrear_indices(n, tope, semilla, 2019), pn.muestrear_indices(n, tope, semilla, 2020))   # distinta entre años
    assert not np.array_equal(pn.muestrear_indices(n, tope, semilla, 2019), pn.muestrear_indices(n, tope, semilla + 1, 2019))


def test_la_tabla_de_muestreo_tiene_una_fila_por_anio_con_la_marca_de_muestreo_y_ceros_en_el_anio_sin_cohorte():
    filas = [
        {"anio": 2023, "ventana": "confirmacion", "n_cohorte_d": 120, "n_muestra": 120, "n_eventos_muestra": 12, "tope": 150},
        {"anio": 2019, "ventana": "ranking", "n_cohorte_d": 1000, "n_muestra": 150, "n_eventos_muestra": 20, "tope": 150},
        {"anio": 2020, "ventana": "ranking", "n_cohorte_d": 0, "n_muestra": 0, "n_eventos_muestra": 0, "tope": 150},
        {"anio": 2021, "ventana": "ranking", "n_cohorte_d": 150, "n_muestra": 150, "n_eventos_muestra": 30, "tope": 150},
    ]
    t = pn.tabla_muestreo(filas)
    assert t.columns == ["anio", "ventana", "n_cohorte_d", "n_muestra", "n_eventos_muestra", "muestreado", "tope", "fraccion_muestreada"] and t["anio"].to_list() == [2019, 2020, 2021, 2023]
    assert t["muestreado"].to_list() == ["si", "no", "no", "no"]                                                      # con n igual al tope no se muestrea
    assert t["fraccion_muestreada"].to_list() == [0.15, None, 1.0, 1.0] and t.filter(pl.col("anio") == 2020)["n_cohorte_d"].item() == 0


# =============================================================================================== historial
FILAS_HISTORIAL = [
    # año, identificador, ingreso, alta  (los 15 episodios de la Tarea 9)
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


def _d(texto):
    return dt.date.fromisoformat(texto) if texto else None


def _episodios(filas, desde=0):
    return pl.DataFrame(
        {"fila": list(range(desde, desde + len(filas))), "anio": [f[0] for f in filas], "id_num": [f[1] for f in filas], "ing": [_d(f[2]) for f in filas], "alta": [_d(f[3]) for f in filas]},
        schema={"fila": pl.Int64, "anio": pl.Int32, "id_num": pl.Int64, "ing": pl.Date, "alta": pl.Date},
    )


def test_historial_por_episodio_a_mano_con_los_15_episodios_de_la_tarea_9():
    h = pn.historial_por_episodio(_episodios(FILAS_HISTORIAL), [[2019, 2020]], 365)
    assert h.columns == ["fila", "viable", "n_previos", "dias_previo"] and h["fila"].to_list() == list(range(15))
    eps = _episodios(FILAS_HISTORIAL)
    por_anio = {}
    for anio, viable, n_prev in zip(eps["anio"], h["viable"], h["n_previos"]):
        d = por_anio.setdefault(anio, [0, 0])
        d[0] += int(viable)
        d[1] += int(viable and n_prev > 0)
    assert por_anio[2019] == [0, 0]                                                          # 2019 es el primer año del espacio: nadie tiene la ventana completa
    assert por_anio[2020] == [10, 2]                                                         # diez viables y dos con egreso previo
    f = {i: r for i, r in enumerate(h.iter_rows(named=True))}
    assert (f[1]["n_previos"], f[1]["dias_previo"]) == (1, 311) and (f[12]["n_previos"], f[12]["dias_previo"]) == (1, 365)
    for i in (2, 3, 5, 7, 8, 9, 10, 14):                                                     # viables sin previo: 391 y 366 días quedan fuera, el reingreso del mismo día, el previo sin alta y el invertido no cuentan
        assert f[i]["viable"] is True and f[i]["n_previos"] == 0 and f[i]["dias_previo"] is None, i
    assert f[6]["viable"] is False and f[6]["n_previos"] is None and f[6]["dias_previo"] is None            # identificador inválido
    for i in (0, 4, 11, 13):
        assert f[i]["viable"] is False and f[i]["n_previos"] is None, i                      # episodios de 2019


def test_el_reingreso_del_mismo_dia_no_cuenta_como_previo():
    h = pn.historial_por_episodio(_episodios([(2019, 1, "2019-02-01", "2019-02-03"), (2020, 1, "2020-02-01", "2020-02-01"), (2020, 1, "2020-02-01", "2020-02-04")]), [[2019, 2020]], 365)
    assert h["n_previos"].to_list() == [None, 1, 1] and h["dias_previo"].to_list() == [None, 363, 363]     # la estadía de 0 días del mismo día no es previa del segundo


def test_un_episodio_con_dos_previos_dentro_de_la_ventana_da_dos_y_los_dias_hasta_el_mas_reciente():
    eps = _episodios([(2019, 9009, "2019-03-01", "2019-03-02"), (2019, 9009, "2019-09-01", "2019-09-05"), (2019, 9009, "2018-12-30", "2019-01-02"), (2020, 9009, "2020-02-01", "2020-02-03")])
    h = pn.historial_por_episodio(eps, [[2019, 2020]], 365)
    assert h["viable"].to_list() == [False, False, False, True]
    # previos de 2020-02-01 en [2019-02-01, 2020-01-31]: los dos primeros; el tercero (alta 2019-01-02) queda fuera
    assert h["n_previos"].to_list() == [None, None, None, 2] and h["dias_previo"].to_list() == [None, None, None, (dt.date(2020, 2, 1) - dt.date(2019, 9, 5)).days]


def test_los_espacios_de_identificadores_distintos_no_se_cruzan():
    filas = [(2020, 77, "2020-12-20", "2020-12-22"), (2021, 77, "2021-01-05", "2021-01-07"), (2022, 77, "2022-01-04", "2022-01-06")]
    dos = pn.historial_por_episodio(_episodios(filas), [[2019, 2020], [2021, 2022]], 365)
    # 2021 abre el segundo espacio: su ventana de 365 días llega al espacio anterior, así que no es viable y el egreso de 2020 no cuenta
    assert dos["viable"].to_list() == [True, False, True] and dos["n_previos"].to_list() == [0, None, 1]
    assert dos["dias_previo"][2] == (dt.date(2022, 1, 4) - dt.date(2021, 1, 7)).days
    uno = pn.historial_por_episodio(_episodios(filas), [[2019, 2020, 2021, 2022]], 365)
    assert uno["viable"].to_list() == [True, True, True] and uno["n_previos"].to_list() == [0, 1, 1]            # con un solo espacio el egreso de 2020 sí cuenta para el de 2021
    assert uno["dias_previo"][1] == (dt.date(2021, 1, 5) - dt.date(2020, 12, 22)).days


def test_un_identificador_de_15_digitos_no_desborda_y_los_episodios_de_otro_anio_fuera_de_los_espacios_no_son_viables():
    grande = 123_456_789_012_345
    filas = [(2019, grande, "2019-02-01", "2019-02-03"), (2020, grande, "2020-01-15", "2020-01-17"), (2020, grande - 1, "2020-01-15", "2020-01-17"), (2024, grande, "2024-05-01", "2024-05-03")]
    h = pn.historial_por_episodio(_episodios(filas), [[2019, 2020]], 365)
    assert h["viable"].to_list() == [False, True, True, False] and h["n_previos"].to_list() == [None, 1, 0, None]
    assert h["dias_previo"].to_list() == [None, (dt.date(2020, 1, 15) - dt.date(2019, 2, 3)).days, None, None]


def test_una_fecha_fuera_del_rango_de_la_clave_se_trata_como_invalida():
    filas = [
        (2019, 5, "2019-08-01", "2019-08-03"), (2020, 5, "2020-05-01", "2020-05-03"), (2020, 6, "2020-05-01", "2020-05-03"),
        (2019, 6, "1901-02-01", "2019-06-03"), (2020, 8, "2999-01-01", "2999-01-03"),
    ]
    h = pn.historial_por_episodio(_episodios(filas), [[2019, 2020]], 365)
    # el egreso de 1901 no es un previo válido del paciente 6 aunque su alta caiga dentro de la ventana; el ingreso del año 2999 tampoco es viable
    assert h["viable"].to_list() == [False, True, True, False, False] and h["n_previos"].to_list() == [None, 1, 0, None, None]


def test_sin_espacios_nadie_es_viable():
    h = pn.historial_por_episodio(_episodios(FILAS_HISTORIAL), [], 365)
    assert not any(h["viable"]) and h["n_previos"].null_count() == 15
    assert pn.historial_por_episodio(_episodios([]), [[2019]], 365).height == 0


# =============================================================================================== cardinalidad de especialidad (MEDIUM 1)
def _especialidades_70():
    """70 niveles: los 60 más frecuentes los usan varios hospitales; de los 10 raros, 3 pertenecen a un solo hospital."""
    texto, hosp = [], []
    hospitales_ = ["H1", "H2", "H3", "H4"]
    for i in range(60):                                         # frecuentes: 40 filas en un hospital y 20 en el siguiente
        texto += [f"E{i:02d}"] * 60
        hosp += [hospitales_[i % 4]] * 40 + [hospitales_[(i + 1) % 4]] * 20
    for j in range(10):                                         # raros: 3 exclusivos de un hospital y 7 repartidos entre dos
        nombre = f"R{j:02d}"
        if j < 3:
            texto += [nombre] * 2
            hosp += [hospitales_[j % 4]] * 2
        else:
            texto += [nombre] * 2
            hosp += [hospitales_[j % 4], hospitales_[(j + 1) % 4]]
    return texto, hosp


def test_la_cardinalidad_de_especialidad_se_calcula_sobre_el_texto_sin_tope_de_60_niveles():
    texto, hosp = _especialidades_70()
    c = pn.cardinalidad_especialidad(texto, hosp, 100)
    assert c["n_niveles_totales"] == 70 and c["n_niveles_en_un_hospital"] == 3 and c["n_hospitales"] == 4
    # con el tope aplicado antes (el error del diseño original) saldría 60 + OTROS y ningún nivel exclusivo de un hospital
    anios = pl.Series([2019] * len(texto))
    n = pn.calcular_niveles(pl.Series("t", texto), anios, RANK, 60)
    recortado = pn.recodificar(pl.Series("t", texto), n)
    assert recortado.n_unique() == 61
    c2 = pn.cardinalidad_especialidad(recortado, hosp, 100)
    assert c2["n_niveles_totales"] == 61 and c2["n_niveles_en_un_hospital"] == 0 and c2["n_niveles_totales"] != c["n_niveles_totales"]


def test_cardinalidad_a_mano_por_hospital_con_minimo_de_filas_y_niveles_no_reales():
    texto = ["a", "b", "a", "c", "b", "d", None, "", "DESCONOCIDO", " a ", "x", "x", "y"]
    hosp = ["H1", "H1", "H1", "H1", "H1", "H1", "H1", "H1", "H1", "H1", "H2", "H2", "H2"]
    c = pn.cardinalidad_especialidad(texto, hosp, 3)
    # H1 tiene 10 filas: niveles reales a, b, c, d (el nulo, el vacío y el centinela no cuentan; " a " se recorta); H2 tiene 3 filas: x, y
    assert c == {"n_hospitales": 2, "card_minima": 2, "card_mediana": 3.0, "card_maxima": 4, "n_niveles_totales": 6, "n_niveles_en_un_hospital": 6}
    # con un mínimo de 4 filas H2 queda fuera de todas las cifras (DIS 10.4: todo es "por hospital con al menos MIN_FILAS_HOSPITAL filas"), también de los niveles totales
    c4 = pn.cardinalidad_especialidad(texto, hosp, 4)
    assert (c4["n_hospitales"], c4["card_minima"], c4["card_maxima"]) == (1, 4, 4) and (c4["n_niveles_totales"], c4["n_niveles_en_un_hospital"]) == (4, 4)
    vacio = pn.cardinalidad_especialidad(texto, hosp, 11)
    assert vacio["n_hospitales"] == 0 and vacio["card_mediana"] is None and vacio["card_minima"] is None and (vacio["n_niveles_totales"], vacio["n_niveles_en_un_hospital"]) == (0, 0)
    # la base son los hospitales grandes: un nivel que usan uno grande y uno chico cuenta como de un solo hospital y uno que solo usa el chico no entra
    compartido = pn.cardinalidad_especialidad(["a"] * 4 + ["b"] * 4 + ["a", "z"], ["H1"] * 8 + ["H2"] * 2, 3)
    assert compartido["n_hospitales"] == 1 and compartido["n_niveles_totales"] == 2 and compartido["n_niveles_en_un_hospital"] == 2      # a y b; z solo lo usa H2, que es chico
    comparten = pn.cardinalidad_especialidad(["a", "b", "a", "b"], ["H1", "H1", "H2", "H2"], 2)
    assert comparten["n_niveles_en_un_hospital"] == 0 and comparten["card_mediana"] == 2.0 and comparten["card_minima"] == comparten["card_maxima"] == 2


def test_cardinalidad_de_un_hospital_sin_niveles_reales_vale_cero_y_valida_los_argumentos():
    c = pn.cardinalidad_especialidad([None, "", "NA", "x", "x", "x"], ["H1", "H1", "H1", "H2", "H2", "H2"], 3)
    assert c["card_minima"] == 0 and c["card_maxima"] == 1 and c["n_niveles_totales"] == 1 and c["n_hospitales"] == 2
    with pytest.raises(ValueError, match="'min_filas'"):
        pn.cardinalidad_especialidad(["a"], ["H1"], 0)
    with pytest.raises(ValueError, match="largos distintos"):
        pn.cardinalidad_especialidad(["a", "b"], ["H1"], 1)


# =============================================================================================== contrato de Panel
def _panel_chico(**cambios):
    datos = pl.DataFrame({
        "anio": pl.Series([2019, 2019, 2023, 2023], dtype=pl.Int32), "y": pl.Series([0, 1, 0, 1], dtype=pl.Int8),
        "ELIX_01": pl.Series([1, 0, None, 1], dtype=pl.Int8), "SEXO": ["MUJER", "HOMBRE", pn.SIN_DATO, "MUJER"], "EDAD_ANIOS": [30.0, 60.0, 45.0, 80.0],
        "COD_HOSPITAL": ["101100", "101100", "102100", "102100"],
    })
    reg = pn.registro_variables(False)
    base = dict(
        datos=datos, variables=reg, niveles={"SEXO": ["MUJER", "HOMBRE", pn.SIN_DATO], "COD_HOSPITAL": ["101100", "102100"]}, muestreo=pn.tabla_muestreo([]),
        niveles_tabla=pl.DataFrame(schema=pn.ESQUEMA_NIVELES_TABLA), regimenes=[], con_historial=False, anios_ranking=[2019, 2020, 2021, 2022], anio_confirmacion=2023,
    )
    base.update(cambios)
    return pn.Panel(**base)


def test_panel_ventana_nombres_y_hospital():
    p = _panel_chico()
    assert p.ventana([2019])["EDAD_ANIOS"].to_list() == [30.0, 60.0] and p.ventana((2019, 2023)).height == 4 and p.ventana([2021]).height == 0
    assert p.hospital.to_list() == ["101100", "101100", "102100", "102100"] and p.hospital.name == "COD_HOSPITAL"
    assert p.nombres(uso="referencia") == ["COD_HOSPITAL"]
    assert p.nombres(uso="candidata", tipo="num", evaluable="si") == ["EDAD_ANIOS", "N_DX_SEC", "N_COMORB_ELIX", "PUNTAJE_VAN_WALRAVEN", "PUNTAJE_VAN_WALRAVEN_SIN_5"]
    assert len(p.nombres(uso="candidata", evaluable="parcial")) == 2 and p.nombres(tipo="bin", uso="auxiliar") == ["ES_COHORTE_E", "SERVICIO_INGRESO_IGUAL_ALTA"]
    assert "ESPECIALIDAD_MACRO" in p.nombres(uso="no_construible") and p.nombres() == list(p.variables)


def test_panel_codigos_binarias_y_categoricas_se_cachean_y_el_nulo_vale_menos_uno():
    p = _panel_chico()
    cod, niv = p.codigos("ELIX_01")
    assert cod.tolist() == [1, 0, -1, 1] and niv == ["0", "1"] and cod.dtype == np.int32
    cod, niv = p.codigos("SEXO")
    assert cod.tolist() == [0, 1, 2, 0] and niv == ["MUJER", "HOMBRE", pn.SIN_DATO] and cod.dtype == np.int32
    assert p.codigos("SEXO")[0] is cod                                                                        # caché
    assert p.codigos("COD_HOSPITAL")[0].tolist() == [0, 0, 1, 1]


def test_panel_codigos_un_nivel_desconocido_vale_menos_uno_y_los_casos_de_error():
    p = _panel_chico(datos=_panel_chico().datos.with_columns(pl.Series("SEXO", ["MUJER", "NUEVO", pn.SIN_DATO, "HOMBRE"])))
    assert p.codigos("SEXO")[0].tolist() == [0, -1, 2, 1]
    with pytest.raises(ValueError, match="numérica"):
        p.codigos("EDAD_ANIOS")
    with pytest.raises(KeyError, match="no está en el registro"):
        p.codigos("NO_EXISTE")
    with pytest.raises(ValueError, match="no tiene columna"):
        p.codigos("ESPECIALIDAD_MACRO")


def test_una_copia_del_panel_con_otros_datos_no_hereda_la_cache_de_codigos():
    p = _panel_chico()
    assert p.codigos("SEXO")[0].tolist() == [0, 1, 2, 0]
    q = dataclasses.replace(p, datos=p.datos.with_columns(pl.Series("SEXO", [pn.SIN_DATO, "MUJER", "MUJER", "HOMBRE"])))
    assert q.codigos("SEXO")[0].tolist() == [2, 0, 0, 1] and p.codigos("SEXO")[0].tolist() == [0, 1, 2, 0]


# =============================================================================================== construcción sobre datos sintéticos
@pytest.fixture(scope="module")
def sintetica(tmp_path_factory, u):                                                    # pide ``u``: las pruebas de construcción suponen esos umbrales
    raiz = tmp_path_factory.mktemp("panel")
    return raiz, crear_fuente_sintetica(raiz / "datos", n=700, semilla=29)


@pytest.fixture(scope="module")
def corrida(sintetica):
    raiz, fuente = sintetica
    ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=raiz / "salidas_panel")
    return ctx, pn.construir_panel(ctx)


@pytest.fixture(scope="module")
def oraculo(sintetica):
    """Cohorte D de cada año leída por la Tarea 7 (``cohorte.leer_base``) en un contexto aparte: la referencia independiente del panel."""
    raiz, fuente = sintetica
    ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=raiz / "salidas_oraculo")
    cat, e, e3 = catalogos.cargar_catalogos(), elixhauser.cargar(), elixhauser.cargar(solo_3c=True)
    bases = {a: cohorte.leer_base(ctx, a, cat, e, e3, extras=True) for a in range(2019, 2024)}
    return {a: b.filter(pl.col("D")) for a, b in bases.items()}, {a: b for a, b in bases.items()}


def _ctx(sintetica, nombre, *, anios=None, fuente=None, **kw):
    """Contexto de extracto sobre la fuente sintética (o la dada) con salidas propias; ``anios`` restringe la fuente a esos años."""
    raiz, base = sintetica
    fuente = fuente or base
    if anios is not None:
        fuente = dataclasses.replace(fuente, archivos={a: r for a, r in fuente.archivos.items() if a in anios})
    return Contexto.crear("extracto", fuente=fuente, dir_salida_base=raiz / nombre, **kw)


def test_la_cohorte_d_por_anio_es_igual_a_la_de_la_tarea_7_y_y_suma_los_fallecidos(corrida, sintetica):
    ctx, panel = corrida
    ctx7 = _ctx(sintetica, "salidas_t7")                                               # contexto aparte: la autoprueba de 2024 de la Tarea 7 deja un bloqueo en su registro
    var = cohorte.calcular(ctx7)["variantes"]
    m = panel.muestreo
    assert m["anio"].to_list() == [2019, 2020, 2021, 2022, 2023]
    for anio in range(2019, 2024):
        d = var.filter((pl.col("periodo") == str(anio)) & (pl.col("variante") == "D")).row(0, named=True)
        f = m.filter(pl.col("anio") == anio).row(0, named=True)
        assert (f["n_cohorte_d"], f["n_muestra"], f["n_eventos_muestra"]) == (d["n"], d["n"], d["n_fallecidos"]), anio
        assert f["muestreado"] == "no" and f["tope"] == 0 and f["fraccion_muestreada"] == 1.0
    assert [r["n_cohorte_d"] for r in m.iter_rows(named=True)] == [421, 442, 429, 440, 415] and m["n_eventos_muestra"].to_list() == [49, 58, 47, 66, 46]
    por_anio = dict(panel.datos.group_by("anio").agg(pl.len().alias("n"), pl.col("y").sum().alias("k")).select("anio", pl.struct("n", "k")).iter_rows())
    assert {a: (v["n"], v["k"]) for a, v in por_anio.items()} == {2019: (421, 49), 2020: (442, 58), 2021: (429, 47), 2022: (440, 66), 2023: (415, 46)}
    assert panel.datos["y"].dtype == pl.Int8 and set(panel.datos["y"].unique()) == {0, 1} and panel.datos["anio"].dtype == pl.Int32


def test_las_filas_del_panel_son_las_de_la_cohorte_d_en_orden_de_archivo_con_los_mismos_valores(corrida, oraculo):
    _, panel = corrida
    d_por_anio, _ = oraculo
    for anio, d in d_por_anio.items():
        p = panel.datos.filter(pl.col("anio") == anio)
        assert p.height == d.height
        assert p["EDAD_ANIOS"].to_list() == d["edad"].cast(pl.Float64).to_list()
        assert p["y"].to_list() == (d["rol"] == "EVENTO").cast(pl.Int8).to_list()
        assert p["ESTANCIA_DIAS"].to_list() == d["estancia"].cast(pl.Float64).to_list() and p["ES_COHORTE_E"].to_list() == d["E"].cast(pl.Int8).to_list()
        assert p["COD_HOSPITAL"].to_list() == d["hospital"].to_list() and p["GRUPO_CLINICO"].to_list() == d["capitulo"].to_list()
        assert p["SEXO"].to_list() == [pn.SIN_DATO if s == "SIN_DATO" else s for s in d["sexo"]]
        assert p["ENTRANTE_TIPO_PROCEDENCIA"].to_list() == d.select(pl.col("procedencia").str.to_uppercase().str.contains(hospitales.TEXTO_ENTRANTE).cast(pl.Int8))["procedencia"].to_list()
        informadas = d.select((pl.col("hospproc").is_not_null() & (pl.col("hospproc") != "")).cast(pl.Int8).alias("x"))["x"].to_list()
        assert p["DERIVADO_OTRO_HOSPITAL"].to_list() == informadas, anio                                     # recuento independiente de HOSPPROCEDENCIA informada


def test_el_panel_no_conserva_identificadores_fechas_ni_nombres_de_hospital_y_el_puntaje_esta_renombrado(corrida):
    _, panel = corrida
    prohibidas = {"id_num", "ing", "alta", "CIP_ENCRIPTADO", "FECHA_INGRESO", "FECHA_NACIMIENTO", "FECHAALTA", "HOSPPROCEDENCIA", "hospproc", "hospital", "SCORE_VANWALRAVEN", "fila", "__fila"}
    assert not prohibidas & set(panel.datos.columns)
    assert "PUNTAJE_VAN_WALRAVEN" in panel.datos.columns and "PUNTAJE_VAN_WALRAVEN_SIN_5" in panel.datos.columns
    assert panel.datos.columns == ["anio", "y"] + [n for n in panel.variables if n in panel.datos.columns]
    assert all(set(panel.datos[c].drop_nulls().unique()) <= {0, 1} for c in panel.datos.columns if panel.variables.get(c) and panel.variables[c].tipo == "bin")
    categoricas = [n for n in panel.nombres(tipo="cat") if n in panel.datos.columns]
    assert len(categoricas) == 21 and "ESPECIALIDAD_MACRO" not in categoricas and set(categoricas) == set(panel.niveles)       # 16 candidatas, COD_HOSPITAL y 4 de solo fuga
    for nombre in categoricas:
        assert set(panel.datos[nombre].unique()) <= set(panel.niveles[nombre]), nombre


def test_los_agregados_de_comorbilidad_son_exactamente_la_suma_y_las_combinaciones_con_los_pesos_reales(corrida):
    _, panel = corrida
    d, e = panel.datos, elixhauser.cargar()
    m = np.column_stack([d[i].to_numpy() for i in e.ids]).astype(np.int64)
    assert (d["N_COMORB_ELIX"].to_numpy() == m.sum(axis=1)).all()
    assert (d["PUNTAJE_VAN_WALRAVEN"].to_numpy() == m @ np.array([e.pesos[i] for i in e.ids])).all()
    assert (d["PUNTAJE_VAN_WALRAVEN_SIN_5"].to_numpy() == m @ np.array([0 if i in e.sospechosas else e.pesos[i] for i in e.ids])).all()
    assert d["ELIX_01"].sum() > 0 and all(d[c].dtype == pl.Int8 for c in e.ids) and d["N_COMORB_ELIX"].dtype == pl.Float64


def test_dia_de_la_semana_mes_y_fin_de_semana_son_coherentes_y_la_procedencia_agrupa_a_mano(corrida):
    _, panel = corrida
    d = panel.datos
    dias = d["DIA_SEMANA"].cast(pl.Int32)
    assert set(d["DIA_SEMANA"].unique()) <= {str(i) for i in range(7)} and (d["ES_FIN_SEMANA"] == (dias >= 5).cast(pl.Int8)).all()
    assert set(d["MES_INGRESO"].unique()) <= {str(i) for i in range(1, 13)} and d["MES_INGRESO"].n_unique() > 6
    assert d["DIA_SEMANA"].n_unique() == 7 and pn.SIN_DATO not in set(d["DIA_SEMANA"]) | set(d["MES_INGRESO"])        # el generador siempre trae fecha de ingreso
    grupos = d.group_by("TIPO_PROCEDENCIA").agg(pl.col("PROCEDENCIA_AGR").unique().alias("g"))
    esperado = {"SERVICIO EMERGENCIA (DOMICILIO)": "URGENCIA", "APS URGENCIA (SAPU, SUR, SUC)": "URGENCIA", "OTROS HOSPITALES DE LA RED": "DERIVADO_OTRO_HOSPITAL",
                "CENTRO ESPECIALIDADES (CDT, CRS, CONSULTORIO ADOS. ESP)": "PROGRAMADO", "CONSULTA PRIVADA": "PROGRAMADO", "APS CONSULTORIO (CESFAM)": "PROGRAMADO",
                pn.SIN_DATO: pn.SIN_DATO}                                                              # "DESCONOCIDO" es un centinela: ya llega como "(sin dato)"
    assert set(d["TIPO_PROCEDENCIA"].unique()) == set(esperado)
    for r in grupos.iter_rows(named=True):
        assert r["g"] == [esperado[r["TIPO_PROCEDENCIA"]]], r


def test_nada_se_imputa_los_faltantes_se_conservan_como_nulo_o_como_sin_dato(corrida):
    _, panel = corrida
    d = panel.datos
    assert d["ES_EXTRANJERO"].null_count() == 0                                                             # en el generador siempre hay nacionalidad informada
    assert (d["TIPO_PROCEDENCIA"] == pn.SIN_DATO).sum() > 0                                                  # "DESCONOCIDO" es un centinela: nivel propio, no imputado
    assert (d["SERVICIOALTA"] == pn.SIN_DATO).sum() == 0 and (d["USOSPABELLON"] == pn.SIN_DATO).sum() > 0   # sin pabellón: el vacío se conserva como "(sin dato)"
    assert d["IR_29301_PESO"].null_count() < d.height and d["ESTANCIA_DIAS"].dtype == pl.Float64
    assert pn.SIN_DATO in panel.niveles["TIPO_PROCEDENCIA"] and panel.niveles["TIPO_PROCEDENCIA"][-1] == pn.SIN_DATO


def test_un_panel_de_extracto_no_tiene_variables_de_historial_y_su_registro_dice_por_que(corrida):
    _, panel = corrida
    assert panel.con_historial is False and panel.regimenes == []
    for n in ("N_EGRESOS_12M", "DIAS_DESDE_EGRESO_PREVIO", "HISTORIA_DISPONIBLE"):
        assert n not in panel.datos.columns and panel.variables[n].evaluable == "no" and "extracto" in panel.variables[n].motivo
    assert panel.variables == pn.registro_variables(False) and panel.anios_ranking == RANK and panel.anio_confirmacion == 2023


def test_dos_construcciones_dan_el_mismo_hash_de_los_datos(corrida, sintetica):
    _, panel = corrida
    otra = pn.construir_panel(_ctx(sintetica, "salidas_otra"))
    assert otra.datos.equals(panel.datos) and otra.datos.hash_rows().sum() == panel.datos.hash_rows().sum()
    assert otra.niveles == panel.niveles and otra.muestreo.equals(panel.muestreo) and otra.niveles_tabla.equals(panel.niveles_tabla)
    assert otra.cardinalidad_especialidad == panel.cardinalidad_especialidad


def test_el_registro_de_accesos_tiene_una_lectura_por_anio_de_2019_a_2023_y_ninguna_de_2024(corrida):
    ctx, _ = corrida
    reg = ctx.cortafuegos.registro()
    assert sorted(reg["anio"].to_list()) == [2019, 2020, 2021, 2022, 2023] and set(reg["proposito"]) == {"seleccion"} and set(reg["tarea"]) == {"T10"}
    assert 2024 not in set(reg["anio"]) and reg["n_columnas"].to_list() == [len(pn.COLUMNAS_PANEL)] * 5
    assert ctx.cortafuegos.violaciones() == 0 and ctx.cortafuegos.n_bloqueos() == 0
    assert set(reg["decision"]) == {"PERMITIDO"}


def test_las_tablas_del_panel_pasan_el_escaneo_de_privacidad_y_las_metricas_tienen_alcance(corrida, sintetica):
    _, panel = corrida
    ctx = _ctx(sintetica, "salidas_escritura")
    pn.escribir_tablas(ctx, panel)
    ctx.escribir_registros()
    esc = privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)
    assert privacidad.total_violaciones(esc) == 0, esc.filter(pl.col("n_violaciones") > 0)
    ind = {r["nombre"]: r for r in ctx.indice_tablas().iter_rows(named=True)}
    assert set(ind) == {"t23_panel_muestreo.csv", "t23_panel_niveles.csv"} and {r["tarea"] for r in ind.values()} == {"T10"}
    assert ind["t23_panel_muestreo.csv"]["columnas_conteo"] == "n_cohorte_d;n_muestra;n_eventos_muestra" and ind["t23_panel_niveles.csv"]["columnas_conteo"] == "n"
    assert ind["t23_panel_muestreo.csv"]["filas"] == 5 and ind["t23_panel_niveles.csv"]["filas"] == panel.niveles_tabla.height
    metricas = {r["clave"]: r for r in ctx.metricas().iter_rows(named=True)}
    assert set(metricas) == {"panel.n_filas_ranking", "panel.n_filas_2023", "panel.n_eventos_ranking", "panel.n_eventos_2023"}
    assert metricas["panel.n_filas_ranking"]["valor"] == 421 + 442 + 429 + 440 and metricas["panel.n_eventos_ranking"]["valor"] == 49 + 58 + 47 + 66
    assert metricas["panel.n_filas_2023"]["valor"] == 415 and metricas["panel.n_eventos_2023"]["valor"] == 46
    for r in metricas.values():
        assert r["alcance"] in {"2019-2022", "2023"} and r["tabla"] == "t23_panel_muestreo.csv" and r["tarea"] == "T10" and r["nivel"] == "extracto" and "2024" not in r["alcance"]
    assert privacidad.escanear_texto(ctx.dir_tablas.joinpath("t23_panel_niveles.csv").read_text(encoding="utf8"))["numero_largo_en_texto"] == 0


def test_un_conteo_de_1_a_9_se_suprime_en_la_tabla_y_la_metrica_correspondiente_queda_sin_valor(corrida, sintetica):
    _, panel = corrida
    chico = dataclasses.replace(panel, muestreo=pl.DataFrame({
        "anio": [2019, 2020, 2021, 2022, 2023], "ventana": ["ranking"] * 4 + ["confirmacion"], "n_cohorte_d": [100, 100, 100, 100, 100], "n_muestra": [100, 100, 100, 100, 100],
        "n_eventos_muestra": [2, 1, 0, 1, 5], "muestreado": ["no"] * 5, "tope": [200_000] * 5, "fraccion_muestreada": [1.0] * 5,
    }, schema=pn.ESQUEMA_MUESTREO))
    ctx = _ctx(sintetica, "salidas_conteo_chico")
    pn.escribir_tablas(ctx, chico)
    t = ctx.leer_tabla("t23_panel_muestreo.csv")
    assert t["n_eventos_muestra"].to_list() == [None, None, 0, None, None] and t["fraccion_muestreada"].to_list() == [None, None, 1.0, None, None]     # el cero se informa
    assert t["n_muestra"].to_list() == [100] * 5 and t["n_cohorte_d"].to_list() == [100] * 5                                                              # solo se anula la celda pequeña y sus dependientes
    m = {r["clave"]: r for r in ctx.metricas().iter_rows(named=True)}
    assert m["panel.n_eventos_ranking"]["valor"] is None and "suprimido" in m["panel.n_eventos_ranking"]["nota"]                     # 4 eventos en total: de 1 a 9
    assert m["panel.n_eventos_2023"]["valor"] is None and "suprimido" in m["panel.n_eventos_2023"]["nota"]                           # 5 eventos
    assert m["panel.n_filas_ranking"]["valor"] == 400 and m["panel.n_filas_2023"]["valor"] == 100 and m["panel.n_filas_ranking"]["nota"] == ""
    assert privacidad.total_violaciones(privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)) == 0


def test_escribir_tablas_es_idempotente_y_conserva_la_tarea_del_primer_escritor(corrida, sintetica):
    _, panel = corrida
    ctx = _ctx(sintetica, "salidas_idempotente")
    pn.escribir_tablas(ctx, panel)
    antes = ctx.indice_tablas()
    pn.escribir_tablas(ctx, panel, tarea="T11")
    assert ctx.indice_tablas().equals(antes) and set(ctx.indice_tablas()["tarea"]) == {"T10"}
    solo_t11 = _ctx(sintetica, "salidas_t11")
    pn.escribir_tablas(solo_t11, panel, tarea="T11")
    assert set(solo_t11.indice_tablas()["tarea"]) == {"T11"} and set(solo_t11.metricas()["tarea"]) == {"T11"}


# --------------------------------------------------------------------------------------------- muestreo reproducible
def test_con_un_tope_bajo_el_panel_muestrea_150_filas_por_anio_y_son_las_que_elige_el_oraculo(sintetica, oraculo, tmp_path):
    u150 = umbrales_temporales(tmp_path, max_filas_asociacion_por_anio=150)
    panel = pn.construir_panel(_ctx(sintetica, "salidas_tope", umbrales=u150))
    m = panel.muestreo
    assert m["n_muestra"].to_list() == [150] * 5 and m["muestreado"].to_list() == ["si"] * 5 and m["tope"].to_list() == [150] * 5
    assert m["n_cohorte_d"].to_list() == [421, 442, 429, 440, 415] and m["fraccion_muestreada"].to_list() == pytest.approx([150 / n for n in (421, 442, 429, 440, 415)])
    d_por_anio, _ = oraculo
    for anio, d in d_por_anio.items():
        elegidas = np.sort(np.random.default_rng(u150.meta["semilla"] + anio).choice(d.height, 150, replace=False))      # el oráculo del diseño
        p = panel.datos.filter(pl.col("anio") == anio)
        assert p["EDAD_ANIOS"].to_list() == d["edad"].cast(pl.Float64).gather(elegidas).to_list(), anio
        assert p["y"].to_list() == (d["rol"] == "EVENTO").cast(pl.Int8).gather(elegidas).to_list() and p["COD_HOSPITAL"].to_list() == d["hospital"].gather(elegidas).to_list()
        assert m.filter(pl.col("anio") == anio)["n_eventos_muestra"].item() == int(p["y"].sum())
    assert panel.datos.height == 750


# --------------------------------------------------------------------------------------------- MEDIUM 1: cardinalidad de especialidad
def test_la_cardinalidad_del_panel_coincide_con_un_oraculo_en_polars_sobre_el_texto_de_la_cohorte_d(corrida, sintetica, oraculo):
    ctx, panel = corrida
    _, bases = oraculo
    ctx_o = _ctx(sintetica, "salidas_oraculo_esp")
    partes = []
    for anio in RANK:
        esp = ctx_o.lector.leer(anio, ["ESPECIALIDAD_MEDICA"], proposito="seleccion", tarea="T10")["ESPECIALIDAD_MEDICA"]
        d = bases[anio]
        partes.append(pl.DataFrame({"esp": esp.filter(d["D"]), "h": d.filter(pl.col("D"))["hospital"]}))
    t = pl.concat(partes)
    grandes = t.group_by("h").agg(pl.len().alias("n")).filter(pl.col("n") >= pn.MIN_FILAS_HOSPITAL)["h"].to_list()
    en_grandes = t.filter(pl.col("h").is_in(grandes) & pl.col("esp").is_not_null())                       # DIS 10.4: todas las cifras salen de los hospitales con al menos MIN_FILAS_HOSPITAL filas
    por_h = en_grandes.group_by("h").agg(pl.col("esp").n_unique().alias("k"))
    por_nivel = en_grandes.group_by("esp").agg(pl.col("h").n_unique().alias("nh"))
    c = panel.cardinalidad_especialidad
    assert c["n_hospitales"] == len(grandes) == 6
    assert (c["card_minima"], c["card_maxima"], c["card_mediana"]) == (por_h["k"].min(), por_h["k"].max(), por_h["k"].median()) == (3, 3, 3.0)
    assert c["n_niveles_totales"] == por_nivel.height and c["n_niveles_en_un_hospital"] == (por_nivel["nh"] == 1).sum()
    assert c["n_niveles_totales"] >= 18 and c["n_niveles_en_un_hospital"] == c["n_niveles_totales"]       # en el generador cada nivel ESP_h_k es de un solo hospital


def test_la_cardinalidad_se_calcula_antes_del_tope_de_niveles_aunque_la_columna_quede_recortada(corrida, sintetica, tmp_path):
    _, sin_tope = corrida
    u5 = umbrales_temporales(tmp_path, cardinalidad_max_categorica=5)
    panel = pn.construir_panel(_ctx(sintetica, "salidas_tope5", umbrales=u5))
    assert len(panel.niveles["ESPECIALIDAD_MEDICA"]) == 6 and panel.niveles["ESPECIALIDAD_MEDICA"][-1] == pn.OTROS          # 5 niveles y OTROS
    assert panel.datos["ESPECIALIDAD_MEDICA"].n_unique() == 6
    # la cardinalidad no cambia con el tope: se calculó sobre el texto, antes de recortar (con la columna recortada serían 6 niveles y solo 5 exclusivos de un hospital)
    assert panel.cardinalidad_especialidad == sin_tope.cardinalidad_especialidad and panel.cardinalidad_especialidad["n_niveles_totales"] >= 18
    recortada = pn.cardinalidad_especialidad(panel.datos.filter(pl.col("anio") <= 2022)["ESPECIALIDAD_MEDICA"], panel.datos.filter(pl.col("anio") <= 2022)["COD_HOSPITAL"], pn.MIN_FILAS_HOSPITAL)
    assert (recortada["n_niveles_totales"], recortada["n_niveles_en_un_hospital"]) == (6, 5)
    assert recortada["n_niveles_totales"] < panel.cardinalidad_especialidad["n_niveles_totales"] and recortada["n_niveles_en_un_hospital"] < panel.cardinalidad_especialidad["n_niveles_en_un_hospital"]


# --------------------------------------------------------------------------------------------- niveles nuevos en 2023 y faltantes
@pytest.fixture(scope="module")
def fuente_con_2023_modificado(sintetica):
    """La fuente sintética con 2023 alterado: un valor de PREVISION que solo existe en 2023 y DIAGNOSTICO1 nulo en las filas con índice múltiplo de 9."""
    raiz, fuente = sintetica
    df = generar_anio(2023, n=700, semilla=29)
    previsiones = df["PREVISION"].to_list()
    marca = [("PREVISION_SOLO_2023" if i % 5 == 0 else v) for i, v in enumerate(previsiones)]
    dx = [None if i % 9 == 0 else v for i, v in enumerate(df["DIAGNOSTICO1"].to_list())]
    df = df.with_columns(pl.Series("PREVISION", marca, dtype=pl.String), pl.Series("DIAGNOSTICO1", dx, dtype=pl.String))
    ruta = escribir_archivo(df, raiz / "datos_alterados" / "GRD_PUBLICO_2023.txt", bom=True, crlf=True, dmy=True)
    archivos = dict(fuente.archivos)
    archivos[2023] = ruta
    return dataclasses.replace(fuente, archivos=archivos)


def test_un_nivel_que_solo_aparece_en_2023_pasa_a_otros_y_un_diagnostico_nulo_queda_en_sin_dato(fuente_con_2023_modificado, sintetica):
    panel = pn.construir_panel(_ctx(sintetica, "salidas_2023_alterado", fuente=fuente_con_2023_modificado))
    niv = panel.niveles["PREVISION"]
    assert "PREVISION_SOLO_2023" not in niv and pn.OTROS in niv and niv.index(pn.OTROS) < niv.index(pn.SIN_DATO) and "PREVISION_SOLO_2023" not in set(panel.datos["PREVISION"])
    p23 = panel.datos.filter(pl.col("anio") == 2023)
    ranking = panel.datos.filter(pl.col("anio") <= 2022)
    assert (ranking["PREVISION"] == pn.OTROS).sum() == 0                                                   # en 2019-2022 no hay nada que agrupar
    assert (p23["PREVISION"] == pn.OTROS).sum() > 50 and set(p23["PREVISION"].unique()) <= set(panel.niveles["PREVISION"])
    assert (p23["CIE10_3C"] == pn.SIN_DATO).sum() > 20 and (p23["GRUPO_CLINICO"] == pn.SIN_DATO).sum() > 20 and (ranking["CIE10_3C"] == pn.SIN_DATO).sum() == 0
    t = panel.niveles_tabla.filter((pl.col("variable") == "PREVISION") & (pl.col("nivel") == pn.OTROS))
    f = {r["ventana"]: r for r in t.iter_rows(named=True)}
    assert f[pn.V_RANK]["n"] == 0 and f[pn.V_CONF]["n"] == (p23["PREVISION"] == pn.OTROS).sum() and f[pn.V_CONF]["tipo_nivel"] == "otros"


# --------------------------------------------------------------------------------------------- casos de borde y errores
def test_una_particion_rota_lanza_runtime_error_antes_de_leer_cualquier_cosa(sintetica):
    raiz, fuente = sintetica
    u = config.cargar_umbrales()
    rota = dataclasses.replace(u, particion={**u.particion, "anios_resultado": [2019, 2020, 2021, 2022, 2023, 2024]})
    ctx = Contexto.crear("extracto", fuente=fuente, dir_salida_base=raiz / "salidas_rota", umbrales=rota)
    with pytest.raises(RuntimeError, match="la partición temporal está rota"):
        pn.construir_panel(ctx)
    assert ctx.cortafuegos.registro().height == 0                                                            # no se registró ninguna lectura


def test_el_tipo_de_historial_y_los_regimenes_se_validan(sintetica):
    ctx = _ctx(sintetica, "salidas_validacion")
    with pytest.raises(TypeError, match="'historial'"):
        pn.construir_panel(ctx, historial="si")
    with pytest.raises(TypeError, match="'historial'"):
        pn.construir_panel(ctx, historial=1)
    for malos, patron in (([[2019, 2024]], "no son de resultado"), ([[2020, 2019]], "ordenados"), ([[2019, 2019]], "ordenados"), ("2019", "lista de listas"), ([2019], "lista de años"), ([["a"]], "enteros")):
        with pytest.raises(ValueError, match=patron):
            pn.construir_panel(ctx, historial=True, regimenes=malos)
    assert ctx.cortafuegos.registro().height == 0                                                            # los argumentos inválidos se rechazan antes de leer


def test_sin_filas_de_ranking_o_sin_ambos_desenlaces_es_un_error(sintetica, monkeypatch):
    solo_2023 = _ctx(sintetica, "salidas_solo_2023", anios=[2023, 2024])
    with pytest.raises(ValueError, match="ventana de ranking no tiene filas"):
        pn.construir_panel(solo_2023)
    original = pn.cohorte.tipar
    monkeypatch.setattr(pn.cohorte, "tipar", lambda *a, **k: original(*a, **k).with_columns(pl.col("rol").replace({"EVENTO": "NO_EVENTO"})))
    with pytest.raises(ValueError, match="la ventana de ranking no tiene ambos desenlaces"):
        pn.construir_panel(_ctx(sintetica, "salidas_sin_eventos"))


def test_si_falta_el_anio_de_confirmacion_todo_lo_de_2023_queda_vacio_sin_error(sintetica):
    ctx = _ctx(sintetica, "salidas_sin_2023", anios=[2019, 2020, 2021, 2022, 2024])
    panel = pn.construir_panel(ctx)
    assert panel.anio_confirmacion is None and panel.anios_ranking == RANK and set(panel.datos["anio"]) == set(RANK)
    assert panel.muestreo["anio"].to_list() == RANK and set(panel.niveles_tabla["ventana"]) == {pn.V_RANK}
    pn.escribir_tablas(ctx, panel)
    m = {r["clave"]: r for r in ctx.metricas().iter_rows(named=True)}
    assert m["panel.n_filas_2023"]["valor"] is None and m["panel.n_filas_2023"]["nota"] == "confirmación no disponible" and m["panel.n_filas_ranking"]["valor"] == 1732
    assert ctx.cortafuegos.violaciones() == 0 and 2024 not in set(ctx.cortafuegos.registro()["anio"])


def test_faltan_anios_de_ranking_se_usan_los_disponibles(sintetica):
    panel = pn.construir_panel(_ctx(sintetica, "salidas_faltan", anios=[2021, 2022, 2023]))
    assert panel.anios_ranking == [2021, 2022] and panel.anio_confirmacion == 2023 and sorted(set(panel.datos["anio"])) == [2021, 2022, 2023]
    assert set(panel.niveles_tabla["ventana"]) == {pn.V_RANK, pn.V_CONF}


def test_un_anio_sin_cohorte_d_deja_una_fila_de_muestreo_con_ceros_y_no_aporta_filas(sintetica, monkeypatch):
    original = pn.cohorte.marcar

    def sin_d_en_2020(base, edad_adulto=18):
        b = original(base, edad_adulto)
        return b.with_columns(pl.lit(False).alias("D")) if int(base["anio"][0]) == 2020 else b

    monkeypatch.setattr(pn.cohorte, "marcar", sin_d_en_2020)
    panel = pn.construir_panel(_ctx(sintetica, "salidas_sin_d"))
    f = panel.muestreo.filter(pl.col("anio") == 2020).row(0, named=True)
    assert (f["n_cohorte_d"], f["n_muestra"], f["n_eventos_muestra"], f["muestreado"], f["fraccion_muestreada"]) == (0, 0, 0, "no", None)
    assert panel.anios_ranking == [2019, 2021, 2022] and 2020 not in set(panel.datos["anio"]) and panel.muestreo.height == 5


# --------------------------------------------------------------------------------------------- modo completo
@pytest.fixture(scope="module")
def corrida_completa(sintetica):
    raiz, fuente = sintetica
    completa = dataclasses.replace(fuente, etiqueta="completo")
    ctx = Contexto.crear("completo", fuente=completa, dir_salida_base=raiz / "salidas_completo")
    return ctx, pn.construir_panel(ctx)


def test_en_modo_completo_el_panel_trae_el_historial_con_los_dos_espacios_de_identificadores(corrida_completa):
    ctx, panel = corrida_completa
    d = panel.datos
    assert panel.con_historial and panel.regimenes == [[2019, 2020], [2021, 2022, 2023]]
    for n in ("N_EGRESOS_12M", "DIAS_DESDE_EGRESO_PREVIO", "HISTORIA_DISPONIBLE"):
        assert n in d.columns and "__fila" not in d.columns
    assert d.schema["HISTORIA_DISPONIBLE"] == pl.Int8 and d.schema["N_EGRESOS_12M"] == pl.Float64 and d.schema["DIAS_DESDE_EGRESO_PREVIO"] == pl.Float64
    por_anio = {r["anio"]: r for r in d.group_by("anio").agg(pl.col("HISTORIA_DISPONIBLE").sum().alias("viables"), pl.len().alias("n")).iter_rows(named=True)}
    assert por_anio[2019]["viables"] == 0 and por_anio[2021]["viables"] == 0                                 # el primer año de cada espacio de identificadores nunca tiene la ventana completa
    assert por_anio[2020]["viables"] > 0 and por_anio[2022]["viables"] > 0 and por_anio[2023]["viables"] > 0
    nulos = d.filter(pl.col("HISTORIA_DISPONIBLE") == 0)
    assert nulos["N_EGRESOS_12M"].null_count() == nulos.height and nulos["DIAS_DESDE_EGRESO_PREVIO"].null_count() == nulos.height
    con = d.filter(pl.col("HISTORIA_DISPONIBLE") == 1)
    assert con["N_EGRESOS_12M"].null_count() == 0 and (con["N_EGRESOS_12M"] >= 0).all()
    assert (con.filter(pl.col("N_EGRESOS_12M") == 0)["DIAS_DESDE_EGRESO_PREVIO"].null_count() == con.filter(pl.col("N_EGRESOS_12M") == 0).height) and (con.filter(pl.col("N_EGRESOS_12M") > 0)["DIAS_DESDE_EGRESO_PREVIO"] > 0).all()
    assert (con.filter(pl.col("N_EGRESOS_12M") > 0)["DIAS_DESDE_EGRESO_PREVIO"] <= 365).all()
    candidatas = [n for n, v in panel.variables.items() if v.uso == "candidata" and v.evaluable != "no"]
    assert len(candidatas) == 61 and all(n in d.columns for n in candidatas)
    assert ctx.cortafuegos.violaciones() == 0 and ctx.cortafuegos.n_bloqueos() == 0 and 2024 not in set(ctx.cortafuegos.registro()["anio"])
    assert set(ctx.cortafuegos.registro()["tarea"]) == {"T10"}
    assert ctx.cortafuegos.registro()["n_columnas"].to_list() == [len(pn.COLUMNAS_PANEL) + 1] * 5             # con historial se pide además el identificador


def test_en_modo_completo_las_tablas_y_las_metricas_llevan_la_etiqueta_completo_y_pasan_la_privacidad(corrida_completa):
    ctx, panel = corrida_completa
    pn.escribir_tablas(ctx, panel)
    ctx.escribir_registros()
    esc = privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)
    assert privacidad.total_violaciones(esc) == 0, esc.filter(pl.col("n_violaciones") > 0)
    assert set(ctx.indice_tablas()["fuente"]) == {"completo"} and set(ctx.metricas()["nivel"]) == {"completo"}
    assert ctx.cortafuegos.violaciones() == 0 and ctx.cortafuegos.n_bloqueos() == 0
    assert 2024 not in set(ctx.cortafuegos.registro().filter(pl.col("tarea") == "T10")["anio"])


def test_la_paridad_del_historial_con_la_factibilidad_de_la_tarea_9(corrida_completa, sintetica):
    _, panel = corrida_completa
    raiz, fuente = sintetica
    ctx9 = Contexto.crear("extracto", fuente=fuente, dir_salida_base=raiz / "salidas_t9")
    r = hospitales.calcular(ctx9)
    assert r["regimenes"] == panel.regimenes
    # los mismos episodios (todos los egresos de 2019 a 2023, no solo la cohorte D) pasados por la implementación del panel
    cat, e, e3 = catalogos.cargar_catalogos(), elixhauser.cargar(), elixhauser.cargar(solo_3c=True)
    ctx_e = Contexto.crear("extracto", fuente=fuente, dir_salida_base=raiz / "salidas_episodios")
    eps = pl.concat([cohorte.leer_base(ctx_e, a, cat, e, e3, extras=True).select(pl.col("anio").cast(pl.Int32), "id_num", "ing", "alta") for a in range(2019, 2024)])
    eps = eps.with_columns(pl.Series("fila", np.arange(eps.height), dtype=pl.Int64))
    h = pn.historial_por_episodio(eps, panel.regimenes, 365).with_columns(eps["anio"].alias("anio"))
    mio = {r_["anio"]: (r_["viables"], r_["con_previo"]) for r_ in h.group_by("anio").agg(
        pl.col("viable").sum().alias("viables"), (pl.col("viable") & (pl.col("n_previos") > 0)).sum().alias("con_previo")).iter_rows(named=True)}
    t9 = {r_["anio"]: (r_["n_viable"], r_["n_con_previo"]) for r_ in r["historial"].iter_rows(named=True)}
    assert mio == {a: t9[a] for a in range(2019, 2024)} and sum(v[1] for v in mio.values()) > 0


def test_con_historial_forzado_en_un_extracto_solo_se_prueba_la_mecanica(sintetica):
    panel = pn.construir_panel(_ctx(sintetica, "salidas_forzado"), historial=True)
    assert panel.con_historial and "HISTORIA_DISPONIBLE" in panel.datos.columns and panel.variables["HISTORIA_DISPONIBLE"].evaluable == "si"
    completa = dataclasses.replace(sintetica[1], etiqueta="completo")
    sin = pn.construir_panel(Contexto.crear("completo", fuente=completa, dir_salida_base=sintetica[0] / "salidas_completo_sin"), historial=False)     # lo fuerza aunque la etiqueta sea "completo"
    assert not sin.con_historial and "HISTORIA_DISPONIBLE" not in sin.datos.columns and sin.variables["HISTORIA_DISPONIBLE"].evaluable == "no"
    dados = pn.construir_panel(_ctx(sintetica, "salidas_dados"), historial=True, regimenes=[[2019, 2020], [2021, 2022]])
    assert dados.regimenes == [[2019, 2020], [2021, 2022]]
    assert dados.datos.filter(pl.col("anio") == 2023)["HISTORIA_DISPONIBLE"].sum() == 0                          # 2023 ya no pertenece a ningún espacio: nadie es viable


# FIN-test_panel
