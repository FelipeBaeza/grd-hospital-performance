"""Tarea 11, ``decision``: universo de variables, rol de la Fase 1, disponibilidad, árbol de decisión (lógica pura con tablas armadas a mano), matriz y publicación."""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import polars as pl
import pytest

from eda import config, esquema
from eda import decision as dc


@pytest.fixture(autouse=True)
def _documentos_limpios():
    """Cada prueba parte de los documentos reales (las que los alteran limpian la caché antes y después)."""
    dc.limpiar_cache()
    yield
    dc.limpiar_cache()


def _v(**kw):
    base = dict(tipo="num", evaluable="si", motivo="", uso="candidata", vacio_es_ausencia=False, es_referencia=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _registro(**especiales):
    """Registro de juguete con todas las variables que el panel construye (evaluables) y los cambios que pida cada prueba.

    Una variable del universo que no está en el registro y no tiene regla de motivo es un error de configuración, así que las pruebas deben
    entregar un registro que cubra las del panel (la comparación con el registro real vive en ``test_panel.py``).
    """
    reg = {n: _v() for n in PANEL_EJEMPLO}
    reg.update(especiales)
    return reg


# ------------------------------------------------------------------------------------------- roles
def test_la_precedencia_de_documentos_prevalece_reglas_variables_sobre_el_diccionario():
    # el diccionario dice FEATURE_BASAL para estas tres; reglas_variables.csv las declara CONDICIONAL y manda
    for nombre in ("ESPECIALIDAD_MEDICA", "SERVICIOINGRESO", "HOSPPROCEDENCIA"):
        assert dc.rol_fase1(nombre) == ("CONDICIONAL", "reglas_variables"), nombre
        assert dc._diccionario()[nombre]["categoria"] == "FEATURE_BASAL"
    assert dc.rol_fase1("COD_HOSPITAL") == ("LLAVE_AGREGACION", "reglas_variables")
    assert dc.rol_fase1("IR_29301_MORTALIDAD") == ("PROHIBIDA_FUGA", "reglas_variables")
    assert dc.rol_fase1("ETNIA") == ("AUDITORIA_SENSIBILIDAD", "reglas_variables")
    # solo en el diccionario
    assert dc.rol_fase1("SEXO") == ("FEATURE_BASAL", "diccionario")
    assert dc.rol_fase1("TIPO_PROCEDENCIA") == ("FEATURE_BASAL", "diccionario")
    assert dc.rol_fase1("DIAGNOSTICO7") == ("FEATURE_BASAL", "diccionario")
    assert dc.rol_fase1("PROCEDIMIENTO12") == ("PROHIBIDA_FUGA", "diccionario")
    assert dc.rol_fase1("SERVICIO_SALUD") == ("LLAVE_AGREGACION", "reglas_variables")


def test_una_derivada_declara_su_rol_antes_de_heredar_el_de_su_fuente():
    assert dc.rol_fase1("FECHA_INGRESO")[0] == "FUENTE_DERIVADA"
    assert dc.rol_fase1("MES_INGRESO") == ("FEATURE_BASAL", "derivadas")
    assert dc.rol_fase1("DIA_SEMANA") == ("FEATURE_BASAL", "derivadas")
    assert dc.rol_fase1("EDAD_ANIOS") == ("FEATURE_BASAL", "derivadas")
    assert dc.rol_fase1("HISTORIA_DISPONIBLE") == ("FEATURE_BASAL", "derivadas")
    assert dc.rol_fase1("PUEBLO_ORIGINARIO")[0] == "AUDITORIA_SENSIBILIDAD" and dc.rol_fase1("ES_EXTRANJERO")[0] == "AUDITORIA_SENSIBILIDAD"
    assert dc.rol_fase1("PUNTAJE_VAN_WALRAVEN_SIN_5")[0] == "AUDITORIA_SENSIBILIDAD"
    assert dc.rol_fase1("HISTORIAL_INCOMPLETO")[0] == "CONDICIONAL" and dc.rol_fase1("ESPECIALIDAD_MACRO")[0] == "CONDICIONAL"
    assert dc.rol_fase1("ID_EPISODIO")[0] == "LLAVE_AGREGACION" and dc.rol_fase1("ANIO_EGRESO")[0] == "LLAVE_AGREGACION"


def test_el_puntaje_de_van_walraven_acepta_su_nombre_del_documento_y_el_del_codigo():
    assert dc.rol_fase1("SCORE_VANWALRAVEN") == dc.rol_fase1("PUNTAJE_VAN_WALRAVEN") == ("FEATURE_BASAL", "derivadas")


def test_herencias_explicitas():
    assert dc.rol_fase1("N_DX_SEC") == ("AUDITORIA_SENSIBILIDAD", "hereda:N_DX_TOTAL")
    assert dc.rol_fase1("ENTRANTE_TIPO_PROCEDENCIA") == ("FILTRO_COHORTE", "hereda:ES_TRASLADO_ENTRANTE")
    assert dc.rol_fase1("N_PROCEDIMIENTOS") == ("PROHIBIDA_FUGA", "hereda:PROCEDIMIENTO1")
    assert dc.rol_fase1("N_TRASLADOS_INTERNOS") == ("PROHIBIDA_FUGA", "hereda:FECHATRASLADO1")
    assert dc.rol_fase1("ES_COHORTE_E") == ("TARGET", "hereda:ESTANCIA_DIAS")
    assert dc.rol_fase1("SERVICIO_INGRESO_IGUAL_ALTA") == ("PROHIBIDA_FUGA", "hereda:SERVICIOALTA")
    assert set(dc.HEREDA_ROL) == {"N_DX_SEC", "ENTRANTE_TIPO_PROCEDENCIA", "N_PROCEDIMIENTOS", "N_TRASLADOS_INTERNOS", "ES_COHORTE_E", "SERVICIO_INGRESO_IGUAL_ALTA"}


def test_un_nombre_sin_rol_o_un_prefijo_desconocido_lanza_value_error():
    with pytest.raises(ValueError, match="no hay rol de la Fase 1 para 'VARIABLE_INEXISTENTE'"):
        dc.rol_fase1("VARIABLE_INEXISTENTE")
    with pytest.raises(ValueError, match="ningún prefijo conocido"):
        dc.traducir_prefijo("QUIZÁ — a veces")


def test_un_documento_con_una_categoria_desconocida_es_un_error_fatal(monkeypatch):
    malo = config.cargar_reglas_variables().with_columns(pl.when(pl.col("variable") == "COD_HOSPITAL").then(pl.lit("INVENTADA")).otherwise(pl.col("categoria")).alias("categoria"))
    monkeypatch.setattr(config, "cargar_reglas_variables", lambda: malo)
    dc.limpiar_cache()
    with pytest.raises(ValueError, match="categorías desconocidas"):
        dc.rol_fase1("SEXO")


# ----------------------------------------------------------------- prefijos de "¿Entra al modelo?"
@pytest.mark.parametrize(
    "texto, rol",
    [
        ("SÍ — es el resultado a predecir por el modelo de estadía", "TARGET"),
        ("SÍ — es el resultado a predecir por el modelo de mortalidad", "TARGET"),
        ("SÍ — determina qué episodios se incluyen en el modelo de mortalidad", "TARGET"),
        ("SÍ", "FEATURE_BASAL"),
        ("SÍ — las 29 categorías restantes", "FEATURE_BASAL"),
        ("CONDICIONAL — requiere verificación", "CONDICIONAL"),
        ("PENDIENTE — depende de la aprobación del historial", "CONDICIONAL"),
        ("NO — solo para uso interno", "LLAVE_AGREGACION"),
        ("NO — llave de partición temporal", "LLAVE_AGREGACION"),
        ("NO — solo análisis de sensibilidad", "AUDITORIA_SENSIBILIDAD"),
        ("NO — solo análisis de auditoría", "AUDITORIA_SENSIBILIDAD"),
        ("NO — solo análisis de equidad", "AUDITORIA_SENSIBILIDAD"),
        ("NO — solo filtro de cohorte", "FILTRO_COHORTE"),
        ("NO — marca de cohorte (decisión pendiente)", "FILTRO_COHORTE"),
    ],
)
def test_cada_prefijo_de_entra_al_modelo_se_traduce_a_su_rol(texto, rol):
    assert dc.traducir_prefijo(texto) == rol


def test_los_prefijos_largos_ganan_a_los_cortos_estancia_es_target_y_elix_03_a_31_son_feature_basal():
    # si se probara el prefijo corto "SÍ" primero, ESTANCIA saldría FEATURE_BASAL
    assert dc.rol_fase1("ESTANCIA") == ("TARGET", "derivadas")
    for e in ("ELIX_03", "ELIX_17", "ELIX_31"):
        assert dc.rol_fase1(e) == ("FEATURE_BASAL", "derivadas")


def test_las_34_filas_del_documento_de_derivadas_se_traducen_a_un_rol_conocido():
    t = config.leer_csv_texto(dc.rutas.DIR_FASE2 / "Fase_2__Variables_Derivadas.csv")
    assert t.height == 34
    for uso in t["¿Entra al modelo?"]:
        assert dc.traducir_prefijo(uso.strip()) in dc.ROLES_CONOCIDOS
    assert len(dc._derivadas()) == 62                                   # 31 filas simples + ELIX_01 + ELIX_02 + 29 expandidas


def test_roles_excluyentes_son_los_siete_del_diseno():
    assert dc.ROLES_EXCLUYENTES == {"TARGET", "PROHIBIDA_FUGA", "LLAVE_AGREGACION", "FILTRO_COHORTE", "FUENTE_DERIVADA", "DESCARTAR", "AUDITORIA_SENSIBILIDAD"}
    assert "FEATURE_BASAL" not in dc.ROLES_EXCLUYENTES and "CONDICIONAL" not in dc.ROLES_EXCLUYENTES


# ------------------------------------------------------------------------------------------ universo
def test_el_universo_tiene_199_variables_sin_duplicados_y_en_el_orden_esperado():
    u = dc.cargar_universo()
    assert u.height == 199 and u["variable"].n_unique() == 199
    assert u["variable"].to_list()[:129] == list(esquema.COLUMNAS)
    assert u["origen"].to_list() == ["cruda"] * 129 + ["derivada"] * 66 + ["regla_fase1"] * 4


def test_el_universo_contiene_los_62_nombres_de_derivadas_expandidos_los_de_reglas_y_los_4_del_panel():
    nombres = set(dc.cargar_universo()["variable"])
    derivadas = {n for n, _ in dc._derivadas()}
    assert len(derivadas) == 62 and derivadas <= nombres
    assert {f"ELIX_{i:02d}" for i in range(1, 32)} <= derivadas and "SCORE_VANWALRAVEN" not in nombres and "PUNTAJE_VAN_WALRAVEN" in nombres
    assert set(config.cargar_reglas_variables()["variable"]) <= nombres
    assert set(dc.VARIABLES_PANEL_SIN_DOCUMENTO) == {"N_DX_SEC", "ENTRANTE_TIPO_PROCEDENCIA", "N_PROCEDIMIENTOS", "N_TRASLADOS_INTERNOS"} and set(dc.VARIABLES_PANEL_SIN_DOCUMENTO) <= nombres
    u = dc.cargar_universo()
    assert set(u.filter(pl.col("origen") == "regla_fase1")["variable"]) == {"ANIO_INGRESO", "IR_29301_PESO_NUM", "ESTANCIA_DIAS", "INGRESO_CRITICO"}
    assert not {"ES_COHORTE_E", "SERVICIO_INGRESO_IGUAL_ALTA"} & nombres                  # las columnas auxiliares del panel no son parte del universo


def test_cada_variable_del_universo_tiene_un_rol_conocido():
    for nombre in dc.cargar_universo()["variable"]:
        rol, fuente = dc.rol_fase1(nombre)
        assert rol in dc.ROLES_CONOCIDOS and fuente.split(":")[0] in {"reglas_variables", "diccionario", "derivadas", "hereda"}, nombre


def test_el_uso_del_modelo_es_textual_del_documento_de_origen():
    u = {r["variable"]: r for r in dc.cargar_universo().iter_rows(named=True)}
    assert u["SEXO"]["tipo_doc"] == "Texto" and u["FECHA_NACIMIENTO"]["tipo_doc"] == "Fecha" and u["PESORN1"]["tipo_doc"] == "Número"
    assert u["EDAD_ANIOS"]["uso_modelo_doc"] == "SÍ" and u["ELIX_20"]["uso_modelo_doc"] == "SÍ — las 29 categorías restantes"
    assert u["EDAD_ANIOS"]["tipo_doc"] == "" and u["N_DX_SEC"]["uso_modelo_doc"] == ""


def test_fuentes_derivadas_es_completo_y_solo_usa_columnas_crudas():
    u = dc.cargar_universo()
    no_crudas = u.filter(pl.col("origen") != "cruda")["variable"].to_list()
    assert set(no_crudas) <= set(dc.FUENTES_DERIVADAS) and set(dc.FUENTES_DERIVADAS) <= set(no_crudas) | {"ES_COHORTE_E"}
    for nombre, fuentes in dc.FUENTES_DERIVADAS.items():
        assert set(fuentes) <= set(esquema.COLUMNAS), nombre
    assert dc.FUENTES_DERIVADAS["ELIX_05"] == tuple(esquema.DIAG_SEC) and dc.FUENTES_DERIVADAS["N_DX_TOTAL"] == tuple(esquema.DIAG)


# ------------------------------------------------------------------------ motivos de no evaluable
PANEL_EJEMPLO = {
    "EDAD_ANIOS", "N_DX_SEC", "N_COMORB_ELIX", "PUNTAJE_VAN_WALRAVEN", "PUNTAJE_VAN_WALRAVEN_SIN_5", "N_EGRESOS_12M", "DIAS_DESDE_EGRESO_PREVIO", "HISTORIA_DISPONIBLE",
    "ES_FIN_SEMANA", "NIVEL_CUIDADO_INGRESO", "DERIVADO_OTRO_HOSPITAL", "ENTRANTE_TIPO_PROCEDENCIA", "ES_EXTRANJERO", "PUEBLO_ORIGINARIO", "SEXO", "TIPO_INGRESO",
    "PROCEDENCIA_AGR", "TIPO_PROCEDENCIA", "GRUPO_CLINICO", "CIE10_3C", "MES_INGRESO", "DIA_SEMANA", "ESPECIALIDAD_MEDICA", "SERVICIOINGRESO", "PREVISION",
    "NACIONALIDAD", "PROVINCIA", "COMUNA", "ETNIA", "SERVICIO_SALUD", "COD_HOSPITAL", "IR_29301_SEVERIDAD", "IR_29301_MORTALIDAD", "IR_29301_PESO", "SERVICIOALTA",
    "USOSPABELLON", "N_PROCEDIMIENTOS", "N_TRASLADOS_INTERNOS", "ESTANCIA_DIAS", *(f"ELIX_{i:02d}" for i in range(1, 32)),
}


def test_toda_variable_del_universo_que_el_panel_no_construye_tiene_motivo_y_ninguna_cae_en_un_valor_generico():
    u = dc.cargar_universo()
    restantes = [(r["variable"], r["origen"]) for r in u.iter_rows(named=True) if r["variable"] not in PANEL_EJEMPLO]
    assert len(restantes) == 199 - len(PANEL_EJEMPLO)
    for nombre, origen in restantes:
        motivo = dc.motivo_no_evaluable(nombre, origen)
        assert motivo.strip() and "sin regla" not in motivo, nombre


def test_motivos_concretos_del_diseno():
    assert dc.motivo_no_evaluable("DIAGNOSTICO9", "cruda") == "se evalúa mediante las categorías de Elixhauser y N_DX_SEC"
    assert dc.motivo_no_evaluable("FECHA_INGRESO", "cruda") == "se evalúa como EDAD_ANIOS, MES_INGRESO, DIA_SEMANA y ES_FIN_SEMANA"
    assert dc.motivo_no_evaluable("PROCEDIMIENTO30", "cruda") == "se resume en N_PROCEDIMIENTOS (solo fuga)"
    assert dc.motivo_no_evaluable("FECHATRASLADO9", "cruda") == dc.motivo_no_evaluable("SERVICIOTRASLADO1", "cruda") == "se resume en N_TRASLADOS_INTERNOS (solo fuga)"
    assert dc.motivo_no_evaluable("HOSPPROCEDENCIA", "cruda") == "se evalúa como DERIVADO_OTRO_HOSPITAL"
    assert dc.motivo_no_evaluable("TIPO_ACTIVIDAD", "cruda") == "constante en la cohorte D: filtro de cohorte"
    assert dc.motivo_no_evaluable("CIP_ENCRIPTADO", "cruda") == "identificador: solo para el historial"
    assert dc.motivo_no_evaluable("ESPECIALIDAD_MACRO", "derivada") == "no existe un mapeo de las especialidades a macro grupos"
    assert dc.motivo_no_evaluable("INDICE_PRIVACION", "derivada") == "exige una fuente externa de vulnerabilidad por comuna"
    assert dc.motivo_no_evaluable("RN3ESTADO", "cruda").startswith("bloque de recién nacidos") and dc.motivo_no_evaluable("PESORN4", "cruda").startswith("bloque de recién nacidos")
    assert dc.equivalente_panel("DIAGNOSTICO3") == "ELIX_01 a ELIX_31; N_DX_SEC" and dc.equivalente_panel("N_DX_TOTAL") == "N_DX_SEC" and dc.equivalente_panel("SEXO") == ""


def test_un_nombre_sin_motivo_lanza_value_error():
    with pytest.raises(ValueError, match="no hay motivo definido para 'UNA_VARIABLE_NUEVA'"):
        dc.motivo_no_evaluable("UNA_VARIABLE_NUEVA", "derivada")


def test_vacio_es_ausencia_en_las_columnas_crudas_que_dice_el_diseno():
    verdaderas = ["HOSPPROCEDENCIA", "DIAGNOSTICO2", "DIAGNOSTICO35", "PROCEDIMIENTO1", "PROCEDIMIENTO30", "FECHATRASLADO1", "SERVICIOTRASLADO9", "FECHAINTERV1",
                  "ESPECIALIDADINTERVENCION", "MEDICOINTERV1_ENCRIPTADO", "USOSPABELLON", "CONDICIONDEALTANEONATO1", "PESORN3", "SEXORN2", "RN4ESTADO"]
    falsas = ["DIAGNOSTICO1", "SERVICIOINGRESO", "SEXO", "FECHAPROCEDIMIENTO1", "MEDICOALTA_ENCRIPTADO", "COMUNA", "FECHAALTA", "IR_29301_SEVERIDAD"]
    assert all(dc.vacio_es_ausencia_cruda(c) for c in verdaderas) and not any(dc.vacio_es_ausencia_cruda(c) for c in falsas)


# ----------------------------------------------------------------------------------- disponibilidad
COLUMNAS_ESPERADAS = [
    "variable", "origen", "tipo", "tipo_doc", "rol_fase1", "fuente_rol", "uso_modelo_doc", "evaluable", "motivo_no_evaluable", "uso_panel", "equivalente_panel",
    "vacio_es_ausencia", "faltante_2019", "faltante_2020", "faltante_2021", "faltante_2022", "faltante_2023", "faltante_2024", "faltante_max_t22", "fuente_faltante",
]


def _perfil(filas: dict[tuple[str, int], float | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {"columna": [k[0] for k in filas], "anio": [k[1] for k in filas], "pct_faltante": [float(v) if v is not None else None for v in filas.values()]},
        schema={"columna": pl.String, "anio": pl.Int64, "pct_faltante": pl.Float64},
    )


def test_la_disponibilidad_cubre_el_universo_con_las_columnas_del_diseno_y_sin_perfil_deja_el_faltante_nulo(ctx):
    registro = _registro(ETNIA=_v(tipo="cat", evaluable="parcial", motivo="decisión 4: no se analiza frente a resultados"))
    t = dc.construir_disponibilidad(ctx, registro)
    assert t.columns == COLUMNAS_ESPERADAS and t.height == 199
    assert t["variable"].to_list() == dc.cargar_universo()["variable"].to_list()
    assert set(t["fuente_faltante"]) == {"no disponible"}
    assert all(t[c].null_count() == 199 for c in t.columns if c.startswith("faltante_"))
    f = {r["variable"]: r for r in t.iter_rows(named=True)}
    assert f["EDAD_ANIOS"]["evaluable"] == "si" and f["EDAD_ANIOS"]["motivo_no_evaluable"] == "" and f["EDAD_ANIOS"]["tipo"] == "num" and f["EDAD_ANIOS"]["uso_panel"] == "candidata"
    assert f["ETNIA"]["evaluable"] == "parcial" and f["ETNIA"]["motivo_no_evaluable"].startswith("decisión 4") and f["ETNIA"]["equivalente_panel"] == ""
    assert f["DIAGNOSTICO5"]["evaluable"] == "no" and f["DIAGNOSTICO5"]["equivalente_panel"] == "ELIX_01 a ELIX_31; N_DX_SEC" and f["DIAGNOSTICO5"]["tipo"] == ""
    assert f["SEXO"]["fuente_rol"] == "diccionario" and f["ESPECIALIDAD_MEDICA"]["rol_fase1"] == "CONDICIONAL"


def test_toda_variable_no_evaluable_de_la_disponibilidad_tiene_motivo(ctx):
    registro = {n: _v(evaluable="no", motivo="motivo de prueba") for n in PANEL_EJEMPLO}
    t = dc.construir_disponibilidad(ctx, registro)
    no = t.filter(pl.col("evaluable") != "si")
    assert no.height == 199 and (no["motivo_no_evaluable"].str.len_chars() > 0).all()
    assert (t.filter(pl.col("variable").is_in(sorted(PANEL_EJEMPLO)))["motivo_no_evaluable"] == "motivo de prueba").all()     # las del panel traen el del registro
    assert not (t.filter(~pl.col("variable").is_in(sorted(PANEL_EJEMPLO)))["motivo_no_evaluable"] == "motivo de prueba").any()   # las demás, el de sus reglas


def test_una_variable_del_panel_que_falta_en_el_registro_es_un_error_de_configuracion(ctx):
    faltante = {n: _v() for n in PANEL_EJEMPLO if n != "COD_HOSPITAL"}
    with pytest.raises(ValueError, match="no hay motivo definido para 'COD_HOSPITAL'"):
        dc.construir_disponibilidad(ctx, faltante)


def test_el_registro_decide_resultado_y_referencia_y_las_columnas_no_evaluables_llevan_su_motivo(ctx):
    registro = _registro(
        ESTANCIA_DIAS=_v(uso="resultado", evaluable="si"),
        COD_HOSPITAL=_v(tipo="cat", uso="referencia", evaluable="no", motivo="referencia del análisis de proxy, no es candidata", es_referencia=True),
        N_EGRESOS_12M=_v(evaluable="no", motivo="no evaluable en extracto: el ingreso previo casi nunca está en el mismo extracto"),
        USOSPABELLON=_v(tipo="cat", uso="solo_fuga", vacio_es_ausencia=True),
    )
    f = {r["variable"]: r for r in dc.construir_disponibilidad(ctx, registro).iter_rows(named=True)}
    assert (f["ESTANCIA_DIAS"]["evaluable"], f["ESTANCIA_DIAS"]["motivo_no_evaluable"]) == ("no", "variable de resultado: no es candidata")
    assert (f["COD_HOSPITAL"]["evaluable"], f["COD_HOSPITAL"]["motivo_no_evaluable"]) == ("no", "referencia del análisis de proxy, no es candidata")
    assert f["N_EGRESOS_12M"]["motivo_no_evaluable"].startswith("no evaluable en extracto")
    assert f["USOSPABELLON"]["evaluable"] == "si" and f["USOSPABELLON"]["vacio_es_ausencia"] == "si" and f["USOSPABELLON"]["motivo_no_evaluable"] == ""


def test_un_registro_con_variable_no_evaluable_sin_motivo_es_un_error(ctx):
    with pytest.raises(ValueError, match="no es evaluable y no tiene motivo"):
        dc.construir_disponibilidad(ctx, _registro(EDAD_ANIOS=_v(evaluable="no", motivo="")))


def test_vacio_es_ausencia_se_marca_en_las_crudas_del_diseno(ctx):
    f = {r["variable"]: r["vacio_es_ausencia"] for r in dc.construir_disponibilidad(ctx, _registro()).iter_rows(named=True)}
    assert f["HOSPPROCEDENCIA"] == f["DIAGNOSTICO20"] == f["PROCEDIMIENTO3"] == f["FECHATRASLADO2"] == f["USOSPABELLON"] == f["PESORN2"] == "si"
    assert f["DIAGNOSTICO1"] == f["SEXO"] == f["FECHAPROCEDIMIENTO1"] == "no"


def test_el_faltante_de_las_crudas_es_el_porcentaje_de_t22_dividido_por_100(ctx):
    perfil = _perfil({("COMUNA", 2019): 12.5, ("COMUNA", 2020): 20.0, ("COMUNA", 2021): 0.0, ("COMUNA", 2022): None, ("COMUNA", 2023): 5.0, ("COMUNA", 2024): 99.0})
    f = {r["variable"]: r for r in dc.construir_disponibilidad(ctx, _registro(), perfil).iter_rows(named=True)}
    c = f["COMUNA"]
    assert (c["faltante_2019"], c["faltante_2020"], c["faltante_2021"], c["faltante_2022"], c["faltante_2023"], c["faltante_2024"]) == (0.125, 0.2, 0.0, None, 0.05, 0.99)
    assert c["fuente_faltante"] == "t22"
    # el máximo de 2019 a 2023 no cuenta 2024 (aunque sea más alto) ni los años suprimidos
    assert c["faltante_max_t22"] == 0.2
    assert f["SEXO"]["faltante_2019"] is None and f["SEXO"]["faltante_max_t22"] is None          # la columna no está en el perfil


def test_el_faltante_de_las_derivadas_es_el_de_sus_fuentes_sin_contar_las_de_vacio_informativo(ctx):
    perfil = _perfil({
        ("FECHA_NACIMIENTO", 2019): 1.0, ("FECHA_INGRESO", 2019): 3.0, ("DIAGNOSTICO1", 2019): 0.5, ("DIAGNOSTICO2", 2019): 80.0, ("HOSPPROCEDENCIA", 2019): 89.5,
        ("TIPO_PROCEDENCIA", 2019): 2.0, ("ETNIA", 2019): 40.0,
    })
    f = {r["variable"]: r for r in dc.construir_disponibilidad(ctx, _registro(), perfil).iter_rows(named=True)}
    assert f["EDAD_ANIOS"]["faltante_2019"] == 0.03                                                # el máximo de FECHA_INGRESO y FECHA_NACIMIENTO
    assert f["N_DX_TOTAL"]["faltante_2019"] == 0.005                                               # DIAGNOSTICO2 a 35 tienen vacío informativo: no cuentan
    assert f["ELIX_07"]["faltante_2019"] == 0.0 and f["N_COMORB_ELIX"]["faltante_2019"] == 0.0     # todas sus fuentes son de vacío informativo: 0
    assert f["DERIVADO_OTRO_HOSPITAL"]["faltante_2019"] == 0.0                                     # HOSPPROCEDENCIA (89,5 %) es ausencia del evento
    assert f["PROCEDENCIA_AGR"]["faltante_2019"] == 0.02 and f["PUEBLO_ORIGINARIO"]["faltante_2019"] == 0.4
    assert f["ID_EPISODIO"]["faltante_2019"] == 0.0
    assert f["EDAD_ANIOS"]["faltante_2020"] is None                                                # el perfil no trae 2020
    assert f["DERIVADO_OTRO_HOSPITAL"]["vacio_es_ausencia"] == "no" and f["HOSPPROCEDENCIA"]["vacio_es_ausencia"] == "si"


def test_el_faltante_de_2024_solo_se_muestra_y_no_entra_al_maximo(ctx):
    perfil = _perfil({("PROVINCIA", 2019): 1.0, ("PROVINCIA", 2023): 2.0, ("PROVINCIA", 2024): 70.0, ("MES_INGRESO", 2024): 1.0, ("FECHA_INGRESO", 2022): 4.0, ("FECHA_INGRESO", 2024): 90.0})
    f = {r["variable"]: r for r in dc.construir_disponibilidad(ctx, _registro(), perfil).iter_rows(named=True)}
    assert f["PROVINCIA"]["faltante_2024"] == 0.7 and f["PROVINCIA"]["faltante_max_t22"] == 0.02
    assert f["MES_INGRESO"]["faltante_2024"] == 0.9 and f["MES_INGRESO"]["faltante_max_t22"] == 0.04


def test_sin_el_archivo_de_t22_el_faltante_queda_nulo_y_con_el_archivo_en_la_carpeta_de_tablas_se_lee_solo(ctx):
    t = dc.construir_disponibilidad(ctx, _registro())
    assert set(t["fuente_faltante"]) == {"no disponible"} and t["faltante_max_t22"].null_count() == 199
    _perfil({("COMUNA", 2021): 30.0, ("COMUNA", 2024): 50.0}).write_csv(ctx.dir_tablas / "t22_perfil_columnas.csv")
    f = {r["variable"]: r for r in dc.construir_disponibilidad(ctx, _registro()).iter_rows(named=True)}
    assert f["COMUNA"]["faltante_2021"] == 0.3 and f["COMUNA"]["faltante_max_t22"] == 0.3 and f["COMUNA"]["fuente_faltante"] == "t22"


# ====================================================================================== árbol de decisión (lógica pura)
ESQUEMA_BASE = {
    "variable": pl.String, "origen": pl.String, "tipo": pl.String, "rol_fase1": pl.String, "fuente_rol": pl.String, "evaluable": pl.String, "motivo_no_evaluable": pl.String,
    "vacio_es_ausencia": pl.String, "uso_panel": pl.String, "analiza_resultado": pl.String, "estado_medicion": pl.String, "auc_ranking": pl.Float64, "auc_senal": pl.Float64,
    "auc_confirmacion": pl.Float64, "incremento_auc": pl.Float64, "incremento_estado": pl.String, "asociacion_hospital": pl.Float64, "psi_vs_2023": pl.Float64,
    "psi_2019_vs_resto": pl.Float64, "cambio_signo": pl.String, "faltante_max": pl.Float64, "fuente_faltante": pl.String, "faltante_informativo": pl.String, "cluster": pl.Int64,
    "tamano_cluster": pl.Int64, "evidencia": pl.String,
}
VERIFICADA = "verificada en fuente"
PENDIENTE = "pendiente de revisión clínica (decisión 12)"
UMBRALES = config.cargar_umbrales()
RECOMENDACION = {r["codigo"]: r["recomendacion"] for r in config.cargar_arbol_decision().iter_rows(named=True)}
ETIQUETAS = (
    "coinciden", "fase1_excluye_estadistica_no_objeta", "fase1_excluye_sin_evidencia", "fase1_exige_conservar_estadistica_objeta", "fase1_exige_conservar_estadistica_no_objeta",
    "fase1_exige_conservar_sin_evidencia", "fase1_permite_estadistica_objeta", "fase1_permite_estadistica_no_objeta", "fase1_permite_no_evaluable",
)


def _fila(variable: str = "VAR", **kw) -> dict:
    """Una candidata sana: medida por completo y sin ninguna regla activa. Cada prueba cambia solo lo que necesita."""
    f = dict(
        variable=variable, origen="derivada", tipo="num", rol_fase1="FEATURE_BASAL", fuente_rol="diccionario", evaluable="si", motivo_no_evaluable="", vacio_es_ausencia="no",
        uso_panel="candidata", analiza_resultado="si", estado_medicion="ok", auc_ranking=0.60, auc_senal=0.60, auc_confirmacion=0.60, incremento_auc=0.02, incremento_estado="ok",
        asociacion_hospital=0.05, psi_vs_2023=0.01, psi_2019_vs_resto=0.01, cambio_signo="no", faltante_max=0.01, fuente_faltante="panel", faltante_informativo="no",
        cluster=None, tamano_cluster=None, evidencia="x.csv",
    )
    f.update(kw)
    return f


def _sin_evidencia(**kw) -> dict:
    """Los estadísticos de una variable que no se midió (todo nulo)."""
    f = dict(
        auc_ranking=None, auc_senal=None, auc_confirmacion=None, incremento_auc=None, incremento_estado=None, asociacion_hospital=None, psi_vs_2023=None, psi_2019_vs_resto=None,
        cambio_signo=None, faltante_max=None,
    )
    f.update(kw)
    return f


def _conocimiento(**certezas: str) -> pl.DataFrame:
    return pl.DataFrame({"variable": list(certezas), "certeza": list(certezas.values())}, schema={"variable": pl.String, "certeza": pl.String})


def _correr(*filas: dict, conocimiento=None, arbol=None) -> dict[str, dict]:
    """Aplica el árbol a filas armadas a mano y devuelve la matriz por variable. Sin ``conocimiento`` ninguna variable tiene ficha."""
    base = pl.DataFrame(list(filas), schema=ESQUEMA_BASE)
    out = dc.aplicar_arbol(base, UMBRALES, arbol=arbol, conocimiento=_conocimiento() if conocimiento is None else conocimiento)
    return {r["variable"]: r for r in out.iter_rows(named=True)}


def _una(**kw) -> dict:
    """La matriz de una sola variable (``VAR``) con los cambios pedidos."""
    return _correr(_fila(**kw))["VAR"]


def _activas(f: dict) -> list[str]:
    return [x for x in f["reglas_activadas"].split("; ") if x]


# ---------------------------------------------------------------------------------------- el árbol y la salida
def test_el_arbol_del_csv_tiene_los_nueve_codigos_en_el_orden_implementado():
    t = config.cargar_arbol_decision().sort(pl.col("orden").cast(pl.Int64))
    assert tuple(t["codigo"]) == dc.CODIGOS_ARBOL == ("R_FASE1", "FUGA", "PROXY", "PROXY_MOD", "CALIDAD", "REDUNDANTE", "BAJA_SENAL", "CLINICO", "MANTENER")
    assert t["prevalece_sobre_estadistica"].to_list() == ["si"] + ["no"] * 8


def test_un_arbol_con_el_orden_alterado_lanza_value_error():
    t = config.cargar_arbol_decision()
    alterado = t.with_columns(
        pl.when(pl.col("codigo") == "CALIDAD").then(pl.lit("6")).when(pl.col("codigo") == "REDUNDANTE").then(pl.lit("5")).otherwise(pl.col("orden")).alias("orden")
    )
    with pytest.raises(ValueError, match="el árbol de decisión del CSV no coincide con las reglas implementadas"):
        _correr(_fila(), arbol=alterado)


def test_un_arbol_sin_una_regla_con_una_de_mas_o_con_otro_codigo_lanza_value_error():
    t = config.cargar_arbol_decision()
    extra = pl.concat([t, t.filter(pl.col("codigo") == "MANTENER").with_columns(pl.lit("10").alias("orden"), pl.lit("OTRA").alias("codigo"))])
    for malo in (t.filter(pl.col("codigo") != "CLINICO"), extra, t.with_columns(pl.col("codigo").str.replace("^FUGA$", "SOSPECHA_FUGA"))):
        with pytest.raises(ValueError, match="no coincide con las reglas implementadas"):
            _correr(_fila(), arbol=malo)


def test_el_arbol_se_ordena_por_la_columna_orden_y_no_por_la_posicion_de_las_filas():
    t = config.cargar_arbol_decision()
    revuelto = t.sample(fraction=1.0, shuffle=True, seed=3)
    assert _correr(_fila(), arbol=revuelto)["VAR"]["recomendacion"] == RECOMENDACION["MANTENER"]


def test_los_textos_de_recomendacion_salen_del_csv_y_no_del_codigo():
    t = config.cargar_arbol_decision()
    cambiado = t.with_columns(pl.when(pl.col("codigo") == "MANTENER").then(pl.lit("TEXTO DE PRUEBA")).otherwise(pl.col("recomendacion")).alias("recomendacion"))
    assert _una()["recomendacion"] == RECOMENDACION["MANTENER"] and RECOMENDACION["MANTENER"] != "TEXTO DE PRUEBA"
    assert _correr(_fila(), arbol=cambiado)["VAR"]["recomendacion"] == "TEXTO DE PRUEBA"


def test_la_salida_trae_las_34_columnas_en_el_orden_del_diseno():
    base = pl.DataFrame([_fila()], schema=ESQUEMA_BASE)
    out = dc.aplicar_arbol(base, UMBRALES, conocimiento=_conocimiento())
    assert out.columns == dc.COLUMNAS_MATRIZ and len(out.columns) == 34
    assert out.columns == [
        "variable", "origen", "tipo", "rol_fase1", "fuente_rol", "restriccion_fase1", "evaluable", "motivo_no_evaluable", "estado_medicion", "medicion_completa", "auc_ranking", "auc_senal",
        "auc_confirmacion", "incremento_auc", "asociacion_hospital", "psi_vs_2023", "psi_2019_vs_resto", "psi_max", "cambio_signo", "faltante_max", "fuente_faltante", "faltante_informativo",
        "cluster", "tamano_cluster", "es_representante", "representante_cluster", "margen_auc_vs_representante", "certeza_clinica", "regla_decisiva", "reglas_activadas", "recomendacion",
        "contraste_fase1", "valor_suprimido", "evidencia",
    ]
    assert dict(out.schema) == dc.ESQUEMA_MATRIZ and out["valor_suprimido"].to_list() == ["no"]


def test_a_la_tabla_de_entrada_le_pueden_faltar_las_columnas_opcionales_pero_no_las_de_evidencia():
    base = pl.DataFrame([_fila()], schema=ESQUEMA_BASE)
    opcionales = base.drop(*dc._OPCIONALES_ARBOL)
    assert dc.aplicar_arbol(opcionales, UMBRALES, conocimiento=_conocimiento()).columns == dc.COLUMNAS_MATRIZ
    for obligatoria in ("asociacion_hospital", "incremento_estado", "tamano_cluster"):
        with pytest.raises(ValueError, match=f"le faltan las columnas .*{obligatoria}"):
            dc.aplicar_arbol(base.drop(obligatoria), UMBRALES, conocimiento=_conocimiento())


def test_decision_no_importa_ni_el_panel_ni_seleccion():
    arbol = ast.parse(inspect.getsource(dc))
    importados = {a.module for a in ast.walk(arbol) if isinstance(a, ast.ImportFrom) and a.level == 1 and a.module} | {
        n.name for a in ast.walk(arbol) if isinstance(a, ast.ImportFrom) and a.level == 1 and not a.module for n in a.names
    }
    assert not {"panel", "seleccion", "sospechas", "sensibilidad_elix", "modelos"} & importados


# ----------------------------------------------------------------------------------------------- reglas una a una
@pytest.mark.parametrize("rol", sorted(dc.ROLES_EXCLUYENTES))
@pytest.mark.parametrize("evaluable", ["si", "parcial", "no"])
def test_r_fase1_se_activa_con_cada_uno_de_los_siete_roles_excluyentes_tambien_si_no_es_evaluable(rol, evaluable):
    f = _una(rol_fase1=rol, evaluable=evaluable, motivo_no_evaluable="" if evaluable == "si" else "motivo de prueba")
    assert _activas(f)[0] == "R_FASE1" and f["regla_decisiva"] == "R_FASE1" and f["recomendacion"] == RECOMENDACION["R_FASE1"]


@pytest.mark.parametrize("rol", ["FEATURE_BASAL", "CONDICIONAL"])
def test_r_fase1_no_se_activa_con_feature_basal_ni_condicional(rol):
    f = _una(rol_fase1=rol)
    assert "R_FASE1" not in _activas(f) and f["regla_decisiva"] == "MANTENER"


@pytest.mark.parametrize("auc, activa", [(0.85, True), (0.8499996, True), (0.849999, False), (0.90, True), (0.60, False)])
def test_fuga_se_activa_desde_el_umbral_con_igualdad_y_sobre_el_valor_redondeado(auc, activa):
    f = _una(auc_ranking=auc, auc_confirmacion=0.60)
    assert ("FUGA" in _activas(f)) is activa and (f["regla_decisiva"] == "FUGA") is activa


def test_fuga_se_activa_solo_con_la_confirmacion_o_solo_con_el_ranking():
    solo_confirmacion = _una(auc_ranking=0.60, auc_confirmacion=0.86)
    solo_ranking = _una(auc_ranking=0.86, auc_confirmacion=None)
    assert _activas(solo_confirmacion) == ["FUGA"] and _activas(solo_ranking) == ["FUGA"]
    assert "FUGA" not in _activas(_una(auc_ranking=None, auc_confirmacion=None))
    assert _una(auc_ranking=0.86)["recomendacion"] == RECOMENDACION["FUGA"]


@pytest.mark.parametrize(
    "ah, esperada",
    [(0.30, "PROXY"), (0.2999996, "PROXY"), (0.299999, "PROXY_MOD"), (0.45, "PROXY"), (0.20, "PROXY_MOD"), (0.15, "PROXY_MOD"), (0.1499996, "PROXY_MOD"), (0.149999, None), (0.0, None)],
)
def test_proxy_y_proxy_moderado_en_sus_bordes(ah, esperada):
    f = _una(asociacion_hospital=ah)
    assert [r for r in _activas(f) if r in ("PROXY", "PROXY_MOD")] == ([esperada] if esperada else [])     # nunca las dos a la vez
    assert f["regla_decisiva"] == (esperada or "MANTENER")


@pytest.mark.parametrize("faltante, activa", [(0.50, True), (0.4999996, True), (0.499999, False), (0.90, True), (0.0, False)])
def test_calidad_por_faltante_desde_el_umbral(faltante, activa):
    assert ("CALIDAD" in _activas(_una(faltante_max=faltante))) is activa


def test_calidad_no_se_activa_por_faltante_cuando_el_vacio_es_ausencia_pero_si_por_psi():
    assert "CALIDAD" not in _activas(_una(faltante_max=0.90, vacio_es_ausencia="si"))
    assert "CALIDAD" in _activas(_una(faltante_max=0.90, vacio_es_ausencia="si", psi_vs_2023=0.30))
    assert "CALIDAD" in _activas(_una(faltante_max=0.0, vacio_es_ausencia="si", cambio_signo="si"))


@pytest.mark.parametrize("psi, activa", [(0.25, True), (0.2499996, True), (0.249999, False), (1.5, True), (0.0, False)])
def test_calidad_por_psi_contra_2023_desde_el_umbral(psi, activa):
    assert ("CALIDAD" in _activas(_una(psi_vs_2023=psi))) is activa


def test_calidad_usa_solo_el_psi_contra_2023_y_no_el_de_2019_contra_el_resto():
    f = _una(psi_vs_2023=0.01, psi_2019_vs_resto=0.90)
    assert "CALIDAD" not in _activas(f) and f["psi_max"] == 0.90 and f["psi_vs_2023"] == 0.01 and f["psi_2019_vs_resto"] == 0.90


@pytest.mark.parametrize("cambio, activa", [("si", True), ("no", False), (None, False)])
def test_calidad_por_cambio_de_signo(cambio, activa):
    assert ("CALIDAD" in _activas(_una(cambio_signo=cambio))) is activa


@pytest.mark.parametrize(
    "auc_senal, incremento, activa",
    [
        (0.55, 0.0, False),               # el AUC igual al umbral no es baja señal: desigualdad estricta
        (0.5499996, 0.0, False),          # redondea a 0,55
        (0.549999, 0.0, True),
        (0.52, 0.005, False),             # el incremento igual al mínimo tampoco
        (0.52, 0.0049996, False),
        (0.52, 0.004999, True),
        (0.52, -0.010, True),
        (0.52, None, False),              # sin incremento no se declara baja señal lo que no se pudo medir
        (None, 0.0, False),
        (0.70, 0.0, False),
        (0.52, 0.020, False),
    ],
)
def test_baja_senal_exige_las_dos_condiciones_estrictas_y_ambas_presentes(auc_senal, incremento, activa):
    f = _una(auc_senal=auc_senal, incremento_auc=incremento, incremento_estado="ok" if incremento is not None else "no convergió")
    assert ("BAJA_SENAL" in _activas(f)) is activa


def test_clinico_solo_si_baja_senal_y_certeza_verificada_en_fuente():
    baja = dict(auc_senal=0.52, incremento_auc=0.0)
    f = _correr(
        _fila("A", **baja), _fila("B", **baja), _fila("C", **baja), _fila("D", auc_senal=0.70), conocimiento=_conocimiento(A=VERIFICADA, B=PENDIENTE, D=VERIFICADA)
    )
    assert _activas(f["A"]) == ["BAJA_SENAL", "CLINICO"] and f["A"]["regla_decisiva"] == "CLINICO" and f["A"]["recomendacion"] == RECOMENDACION["CLINICO"]       # M23
    assert _activas(f["B"]) == ["BAJA_SENAL"] and f["B"]["certeza_clinica"] == PENDIENTE and f["B"]["regla_decisiva"] == "BAJA_SENAL"
    assert _activas(f["C"]) == ["BAJA_SENAL"] and f["C"]["certeza_clinica"] == "sin ficha"
    assert _activas(f["D"]) == ["MANTENER"] and f["D"]["certeza_clinica"] == VERIFICADA                    # certeza verificada pero sin baja señal: no hay nada que compensar


def test_mantener_si_no_se_activa_ninguna_regla_y_la_medicion_esta_completa():
    f = _una()
    assert _activas(f) == ["MANTENER"] and f["regla_decisiva"] == "MANTENER" and f["recomendacion"] == RECOMENDACION["MANTENER"]
    assert f["medicion_completa"] == "si" and f["motivo_no_evaluable"] == "" and f["contraste_fase1"] == "fase1_permite_estadistica_no_objeta"
    assert f["es_representante"] == "no aplica" and f["representante_cluster"] == "" and f["margen_auc_vs_representante"] is None


def test_las_reglas_2_a_9_no_se_activan_en_una_variable_con_evaluable_no():
    f = _una(
        evaluable="no", motivo_no_evaluable="motivo de prueba", auc_ranking=0.95, auc_confirmacion=0.95, asociacion_hospital=0.50, faltante_max=0.90, psi_vs_2023=0.90, cambio_signo="si",
        auc_senal=0.50, incremento_auc=0.0,
    )
    assert _activas(f) == [] and f["regla_decisiva"] == "(ninguna)" and f["recomendacion"] == "NO EVALUABLE" and f["medicion_completa"] == "no"


# ----------------------------------------------------------------------------------------- orden de las reglas
def test_proxy_moderado_y_calidad_juntos_deciden_proxy_moderado():
    f = _una(asociacion_hospital=0.20, faltante_max=0.60)
    assert _activas(f) == ["PROXY_MOD", "CALIDAD"] and f["regla_decisiva"] == "PROXY_MOD" and f["recomendacion"] == RECOMENDACION["PROXY_MOD"]


def test_fuga_y_proxy_juntos_deciden_fuga():
    f = _una(auc_ranking=0.90, asociacion_hospital=0.40)
    assert _activas(f) == ["FUGA", "PROXY"] and f["regla_decisiva"] == "FUGA"


def test_calidad_y_redundante_juntos_deciden_calidad():                                                  # M21
    f = _correr(_fila("A", cluster=1, tamano_cluster=2, auc_senal=0.65), _fila("B", cluster=1, tamano_cluster=2, auc_senal=0.60, faltante_max=0.70))
    assert _activas(f["B"]) == ["CALIDAD", "REDUNDANTE"] and f["B"]["regla_decisiva"] == "CALIDAD" and f["B"]["recomendacion"] == RECOMENDACION["CALIDAD"]


def test_redundante_y_baja_senal_juntos_deciden_redundante():
    f = _correr(_fila("A", cluster=1, tamano_cluster=2, auc_senal=0.65), _fila("B", cluster=1, tamano_cluster=2, auc_senal=0.52, incremento_auc=0.0))
    assert _activas(f["B"]) == ["REDUNDANTE", "BAJA_SENAL"] and f["B"]["regla_decisiva"] == "REDUNDANTE"
    assert _activas(f["A"]) == ["MANTENER"]


def test_con_todas_las_reglas_posibles_activadas_decide_r_fase1():
    f = _correr(
        _fila(
            "V", rol_fase1="PROHIBIDA_FUGA", auc_ranking=0.90, asociacion_hospital=0.40, faltante_max=0.90, psi_vs_2023=0.90, cambio_signo="si", auc_senal=0.50, incremento_auc=0.0,
            cluster=1, tamano_cluster=2,
        ),
        _fila("W", cluster=1, tamano_cluster=2),
        conocimiento=_conocimiento(V=VERIFICADA),
    )["V"]
    # REDUNDANTE no puede activarse con R_FASE1, FUGA ni PROXY: esa variable no es elegible
    assert _activas(f) == ["R_FASE1", "FUGA", "PROXY", "CALIDAD", "BAJA_SENAL", "CLINICO"] and f["regla_decisiva"] == "R_FASE1"
    assert f["recomendacion"] == RECOMENDACION["R_FASE1"]


def test_baja_senal_con_certeza_verificada_decide_clinico_y_detras_de_otra_regla_solo_queda_listada():
    baja = dict(auc_senal=0.52, incremento_auc=0.0)
    f = _correr(_fila("A", **baja), _fila("B", faltante_max=0.80, **baja), conocimiento=_conocimiento(A=VERIFICADA, B=VERIFICADA))
    assert f["A"]["regla_decisiva"] == "CLINICO"
    assert _activas(f["B"]) == ["CALIDAD", "BAJA_SENAL", "CLINICO"] and f["B"]["regla_decisiva"] == "CALIDAD" and f["B"]["recomendacion"] == RECOMENDACION["CALIDAD"]


def test_las_reglas_activadas_siempre_salen_en_el_orden_del_arbol():
    f = _una(auc_ranking=0.9, asociacion_hospital=0.2, faltante_max=0.7, auc_senal=0.5, incremento_auc=0.0, rol_fase1="CONDICIONAL")
    posiciones = [dc.CODIGOS_ARBOL.index(r) for r in _activas(f)]
    assert _activas(f) == ["FUGA", "PROXY_MOD", "CALIDAD", "BAJA_SENAL"] and posiciones == sorted(posiciones)


# --------------------------------------------------------------------------------------------- representantes
def _cluster(nombre: str, auc_senal: float | None, tam: int = 3, cid: int = 1, **kw) -> dict:
    return _fila(nombre, cluster=cid, tamano_cluster=tam, auc_senal=auc_senal, **kw)


def test_el_representante_es_el_de_mayor_auc_senal_y_los_demas_quedan_redundantes():
    f = _correr(_cluster("A", 0.62), _cluster("B", 0.60), _cluster("C", 0.58))
    assert [f[n]["es_representante"] for n in "ABC"] == ["si", "no", "no"]
    assert {f[n]["representante_cluster"] for n in "ABC"} == {"A"}
    assert [_activas(f[n]) for n in "ABC"] == [["MANTENER"], ["REDUNDANTE"], ["REDUNDANTE"]]
    assert f["A"]["margen_auc_vs_representante"] == 0.0
    assert f["B"]["margen_auc_vs_representante"] == pytest.approx(-0.02) and f["C"]["margen_auc_vs_representante"] == pytest.approx(-0.04)
    assert f["B"]["recomendacion"] == RECOMENDACION["REDUNDANTE"]


def test_el_mejor_miembro_excluido_por_la_fase1_no_es_representante_y_los_demas_no_quedan_eliminados():       # M24
    f = _correr(_cluster("A", 0.70, rol_fase1="PROHIBIDA_FUGA"), _cluster("B", 0.60), _cluster("C", 0.58))
    assert [f[n]["es_representante"] for n in "ABC"] == ["no elegible", "si", "no"]
    assert {f[n]["representante_cluster"] for n in "ABC"} == {"B"}
    assert _activas(f["A"]) == ["R_FASE1"] and _activas(f["B"]) == ["MANTENER"] and _activas(f["C"]) == ["REDUNDANTE"]
    assert f["A"]["margen_auc_vs_representante"] == pytest.approx(0.10)                                       # la diferencia se informa aunque A no sea elegible


@pytest.mark.parametrize(
    "cambio, elegible",
    [
        ({"rol_fase1": "LLAVE_AGREGACION"}, False),
        ({"auc_ranking": 0.90}, False),                                         # FUGA
        ({"asociacion_hospital": 0.35}, False),                                  # PROXY
        ({"evaluable": "no", "motivo_no_evaluable": "motivo de prueba"}, False),
        ({"asociacion_hospital": 0.20}, True),                                   # PROXY_MOD sigue siendo elegible
        ({"faltante_max": 0.80}, True),                                          # CALIDAD también
        ({"cambio_signo": "si"}, True),
        ({"rol_fase1": "CONDICIONAL"}, True),
    ],
)
def test_elegibilidad_para_ser_representante(cambio, elegible):
    f = _correr(_cluster("A", 0.70, tam=2, **cambio), _cluster("B", 0.60, tam=2))
    assert (f["A"]["es_representante"] == "si") is elegible
    assert f["A"]["es_representante"] == ("si" if elegible else "no elegible") and f["B"]["es_representante"] == ("no" if elegible else "si")
    assert ("REDUNDANTE" in _activas(f["B"])) is elegible and "REDUNDANTE" not in _activas(f["A"])


def test_en_empate_de_auc_senal_publicado_gana_el_menor_faltante_y_luego_el_nombre():
    # 0,6200004 y 0,62 son el mismo valor publicado: el empate se resuelve por faltante y no por la cuarta cifra
    f = _correr(_cluster("A", 0.62, faltante_max=0.20), _cluster("B", 0.6200004, faltante_max=0.10), _cluster("C", 0.62, faltante_max=0.10))
    assert f["B"]["es_representante"] == "si" and f["A"]["es_representante"] == f["C"]["es_representante"] == "no"
    solo_faltante = _correr(_cluster("A", 0.62, faltante_max=0.20), _cluster("C", 0.62, faltante_max=0.10), _cluster("B", 0.62, faltante_max=0.1000004))
    assert {solo_faltante[n]["representante_cluster"] for n in "ABC"} == {"B"}                                  # faltantes iguales al publicado: gana el nombre
    por_faltante = _correr(_cluster("A", 0.62, faltante_max=0.20), _cluster("C", 0.62, faltante_max=0.10), _cluster("B", 0.60, faltante_max=0.0))
    assert {por_faltante[n]["representante_cluster"] for n in "ABC"} == {"C"}


def test_una_variable_sin_auc_senal_solo_es_representante_si_ningun_otro_elegible_tiene_valor():
    con_valor = _correr(_cluster("A", None, tam=2, estado_medicion="constante"), _cluster("B", 0.51, tam=2))
    assert con_valor["B"]["es_representante"] == "si" and con_valor["A"]["es_representante"] == "no"
    assert con_valor["A"]["margen_auc_vs_representante"] is None
    sin_valor = _correr(_cluster("B", None, tam=2, estado_medicion="constante"), _cluster("A", None, tam=2, estado_medicion="constante"))
    assert sin_valor["A"]["es_representante"] == "si" and sin_valor["B"]["es_representante"] == "no"          # empate total: el nombre
    assert sin_valor["B"]["margen_auc_vs_representante"] is None


def test_un_cluster_sin_elegibles_no_activa_redundante_en_ningun_miembro():
    f = _correr(_cluster("A", 0.70, tam=2, rol_fase1="TARGET"), _cluster("B", 0.60, tam=2, rol_fase1="PROHIBIDA_FUGA"))
    assert f["A"]["es_representante"] == f["B"]["es_representante"] == "no elegible"
    assert f["A"]["representante_cluster"] == f["B"]["representante_cluster"] == "(ninguno elegible)"
    assert all("REDUNDANTE" not in _activas(f[n]) for n in "AB") and f["A"]["margen_auc_vs_representante"] is None


def test_un_cluster_de_un_miembro_o_sin_cluster_es_no_aplica():
    f = _correr(_fila("A", cluster=5, tamano_cluster=1), _fila("B", cluster=None, tamano_cluster=None))
    for n in "AB":
        assert f[n]["es_representante"] == "no aplica" and f[n]["representante_cluster"] == "" and f[n]["margen_auc_vs_representante"] is None
        assert _activas(f[n]) == ["MANTENER"]
    assert f["A"]["cluster"] == 5 and f["A"]["tamano_cluster"] == 1


def test_los_clusters_no_se_mezclan_entre_si():
    f = _correr(_cluster("A", 0.70, tam=2, cid=1), _cluster("B", 0.60, tam=2, cid=1), _cluster("C", 0.52, tam=2, cid=2), _cluster("D", 0.65, tam=2, cid=2))
    assert [f[n]["representante_cluster"] for n in "ABCD"] == ["A", "A", "D", "D"]
    assert [f[n]["es_representante"] for n in "ABCD"] == ["si", "no", "no", "si"]


# ------------------------------------------------------------------------------- no evaluables y sin medición
def test_una_variable_no_evaluable_del_registro_queda_no_evaluable_con_su_motivo():
    f = _una(evaluable="no", motivo_no_evaluable="se evalúa como GRUPO_CLINICO", uso_panel="", analiza_resultado="", estado_medicion="sin fila de medición", **_sin_evidencia())
    assert f["regla_decisiva"] == "(ninguna)" and f["recomendacion"] == "NO EVALUABLE" and f["reglas_activadas"] == "" and f["medicion_completa"] == "no"
    assert f["motivo_no_evaluable"] == "se evalúa como GRUPO_CLINICO" and f["contraste_fase1"] == "fase1_permite_no_evaluable"


def test_no_evaluable_en_extracto_tiene_su_propia_recomendacion():
    motivo = "no evaluable en extracto: el ingreso previo casi nunca está en el mismo extracto"
    f = _una(evaluable="no", motivo_no_evaluable=motivo, **_sin_evidencia())
    assert f["recomendacion"] == "NO EVALUABLE en extracto" and f["regla_decisiva"] == "(ninguna)" and f["motivo_no_evaluable"] == motivo
    assert _una(evaluable="no", motivo_no_evaluable="otro motivo", **_sin_evidencia())["recomendacion"] == "NO EVALUABLE"


def test_con_r_fase1_gana_la_exclusion_aunque_la_variable_no_sea_evaluable():
    f = _una(rol_fase1="PROHIBIDA_FUGA", evaluable="no", motivo_no_evaluable="se resume en N_PROCEDIMIENTOS (solo fuga)", **_sin_evidencia())
    assert f["regla_decisiva"] == "R_FASE1" and f["recomendacion"] == RECOMENDACION["R_FASE1"] and f["contraste_fase1"] == "fase1_excluye_sin_evidencia"
    assert f["motivo_no_evaluable"] == "se resume en N_PROCEDIMIENTOS (solo fuga)"


@pytest.mark.parametrize("estado", ["constante", "sin ambos desenlaces", "no evaluable en la ventana"])
def test_una_candidata_sin_medicion_no_inventa_conclusiones(estado):
    f = _una(estado_medicion=estado, **_sin_evidencia(asociacion_hospital=0.05, psi_vs_2023=0.01, faltante_max=0.0))
    assert f["regla_decisiva"] == "(ninguna)" and f["recomendacion"] == "NO EVALUABLE" and f["medicion_completa"] == "no"
    assert f["motivo_no_evaluable"] == f"sin medición: {estado}" and f["reglas_activadas"] == "" and f["contraste_fase1"] == "fase1_permite_no_evaluable"      # M34


def test_una_candidata_sin_auc_en_la_ventana_tampoco_tiene_medicion_completa():
    f = _una(estado_medicion="ok", auc_senal=None)
    assert f["medicion_completa"] == "no" and f["motivo_no_evaluable"] == "sin medición: sin AUC" and f["regla_decisiva"] == "(ninguna)"


def test_una_candidata_con_auc_pero_sin_incremento_no_es_mantener():
    f = _una(auc_senal=0.52, incremento_auc=None, incremento_estado="no convergió")
    assert f["regla_decisiva"] == "(ninguna)" and f["medicion_completa"] == "no" and f["motivo_no_evaluable"] == "sin medición: incremento no convergió"
    assert "BAJA_SENAL" not in _activas(f)                                                                  # sin incremento no se declara baja señal
    pocos = _una(incremento_auc=None, incremento_estado="pocos eventos")
    assert pocos["motivo_no_evaluable"] == "sin medición: incremento pocos eventos" and pocos["regla_decisiva"] == "(ninguna)"


@pytest.mark.parametrize("nombre", ["EDAD_ANIOS", "SEXO"])
def test_las_variables_del_modelo_base_tienen_medicion_completa_sin_incremento_y_pueden_ser_mantener(nombre):
    f = _correr(_fila(nombre, incremento_auc=None, incremento_estado="no aplica: variable del modelo base"))[nombre]
    assert f["medicion_completa"] == "si" and f["regla_decisiva"] == "MANTENER" and f["motivo_no_evaluable"] == ""
    # con baja señal tampoco se activa BAJA_SENAL: no se declara baja señal lo que no se pudo medir
    baja = _correr(_fila(nombre, auc_senal=0.50, incremento_auc=None, incremento_estado="no aplica: variable del modelo base"))[nombre]
    assert baja["regla_decisiva"] == "MANTENER" and "BAJA_SENAL" not in _activas(baja)


def test_una_candidata_sin_medicion_completa_pero_con_proxy_activo_decide_proxy():
    f = _una(estado_medicion="constante", **_sin_evidencia(asociacion_hospital=0.40, psi_vs_2023=0.01))
    assert f["regla_decisiva"] == "PROXY" and f["recomendacion"] == RECOMENDACION["PROXY"] and f["medicion_completa"] == "no"
    assert f["motivo_no_evaluable"] == "sin medición: constante" and f["contraste_fase1"] == "fase1_permite_estadistica_objeta"


def test_una_candidata_sin_medicion_completa_pero_con_calidad_activa_decide_calidad():
    f = _una(estado_medicion="sin ambos desenlaces", **_sin_evidencia(faltante_max=0.70))
    assert f["regla_decisiva"] == "CALIDAD" and f["medicion_completa"] == "no"


def test_nit_11b_una_candidata_con_auc_e_incremento_pero_sin_proxy_o_sin_psi_reales_no_sale_mantener():
    sin_proxy = _una(asociacion_hospital=None)
    assert sin_proxy["regla_decisiva"] == "(ninguna)" and sin_proxy["medicion_completa"] == "no" and sin_proxy["motivo_no_evaluable"] == "sin medición: sin asociación con el hospital"
    sin_psi = _una(psi_vs_2023=None)
    assert sin_psi["regla_decisiva"] == "(ninguna)" and sin_psi["medicion_completa"] == "no" and sin_psi["motivo_no_evaluable"] == "sin medición: sin PSI contra 2023"
    assert _una()["regla_decisiva"] == "MANTENER"                                                            # con los dos reales sí


def test_nit_11b_el_psi_de_2019_contra_el_resto_no_es_parte_de_la_medicion_completa():
    assert _una(psi_2019_vs_resto=None)["regla_decisiva"] == "MANTENER"


def test_una_variable_de_solo_fuga_con_auc_tiene_medicion_completa_aunque_no_analice_el_resto():
    solo_ranking = _una(uso_panel="solo_fuga", rol_fase1="PROHIBIDA_FUGA", analiza_resultado="", **_sin_evidencia(auc_ranking=0.70))
    solo_confirmacion = _una(uso_panel="solo_fuga", rol_fase1="PROHIBIDA_FUGA", analiza_resultado="", **_sin_evidencia(auc_confirmacion=0.70))
    for f in (solo_ranking, solo_confirmacion):
        assert f["medicion_completa"] == "si" and f["motivo_no_evaluable"] == "" and f["contraste_fase1"] == "fase1_excluye_estadistica_no_objeta"
    fuga = _una(uso_panel="solo_fuga", rol_fase1="PROHIBIDA_FUGA", analiza_resultado="", **_sin_evidencia(auc_ranking=0.90))
    assert _activas(fuga) == ["R_FASE1", "FUGA"] and fuga["contraste_fase1"] == "coinciden" and fuga["regla_decisiva"] == "R_FASE1"
    sin_auc = _una(uso_panel="solo_fuga", rol_fase1="PROHIBIDA_FUGA", analiza_resultado="", estado_medicion="sin dato", **_sin_evidencia())
    assert sin_auc["medicion_completa"] == "no" and sin_auc["motivo_no_evaluable"] == "sin medición: sin dato" and sin_auc["contraste_fase1"] == "fase1_excluye_sin_evidencia"


def test_los_conteos_de_la_matriz_incluyen_las_candidatas_sin_medicion_en_no_evaluable():
    filas = [
        _fila("A"), _fila("B", estado_medicion="constante", **_sin_evidencia()), _fila("C", evaluable="no", motivo_no_evaluable="se evalúa como X", **_sin_evidencia()),
        _fila("D", rol_fase1="TARGET"),
    ]
    base = pl.DataFrame(filas, schema=ESQUEMA_BASE)
    m = dc.aplicar_arbol(base, UMBRALES, conocimiento=_conocimiento())
    r = dc.resumen_matriz(m)
    assert r["n_matriz"] == 4 and r["n_recomendacion_mantener"] == 1 and r["n_recomendacion_excluir_fase1"] == 1
    assert r["n_recomendacion_no_evaluable"] == 2 and r["n_sin_medicion"] == 1                                # B (sin medición) y C (del registro)
    assert sum(v for k, v in r.items() if k.startswith("n_recomendacion_")) == r["n_matriz"]
    assert sum(v for k, v in r.items() if k.startswith("n_contraste_")) == r["n_matriz"]


# --------------------------------------------------------------------------------------------- restricción R3
def test_la_restriccion_r3_cubre_exactamente_las_31_categorias_de_elixhauser():
    assert dc.REGLA_FASE1_CONSERVAR == {f"ELIX_{i:02d}": "R3" for i in range(1, 32)} and len(dc.REGLA_FASE1_CONSERVAR) == 31
    assert dc.LEYENDA_R3 == "R3: las 31 categorías deben estar en el modelo principal"


def test_r3_elix_03_con_baja_senal_y_certeza_pendiente_da_baja_senal_con_el_texto_del_csv_y_etiqueta_de_objeta():      # M36
    f = _correr(_fila("ELIX_03", auc_senal=0.508, incremento_auc=0.0), conocimiento=_conocimiento(ELIX_03=PENDIENTE))["ELIX_03"]
    assert f["regla_decisiva"] == "BAJA_SENAL" and f["recomendacion"] == RECOMENDACION["BAJA_SENAL"] == "BAJA SEÑAL: candidata a eliminar"
    assert f["restriccion_fase1"] == dc.LEYENDA_R3 and f["contraste_fase1"] == "fase1_exige_conservar_estadistica_objeta"
    assert f["rol_fase1"] == "FEATURE_BASAL" and _activas(f) == ["BAJA_SENAL"]                               # la restricción no cambia el orden del árbol ni agrega R_FASE1


def test_r3_con_certeza_verificada_decide_clinico_y_la_etiqueta_sigue_siendo_la_de_objeta():
    f = _correr(_fila("ELIX_03", auc_senal=0.508, incremento_auc=0.0), conocimiento=_conocimiento(ELIX_03=VERIFICADA))["ELIX_03"]
    assert f["regla_decisiva"] == "CLINICO" and f["contraste_fase1"] == "fase1_exige_conservar_estadistica_objeta" and f["restriccion_fase1"] == dc.LEYENDA_R3


def test_r3_una_categoria_constante_da_exige_conservar_sin_evidencia():
    f = _correr(_fila("ELIX_28", estado_medicion="constante", **_sin_evidencia(psi_vs_2023=0.0, faltante_max=0.0)))["ELIX_28"]
    assert f["contraste_fase1"] == "fase1_exige_conservar_sin_evidencia" and f["regla_decisiva"] == "(ninguna)" and f["recomendacion"] == "NO EVALUABLE"
    assert f["restriccion_fase1"] == dc.LEYENDA_R3


def test_r3_una_categoria_con_medicion_completa_y_sin_reglas_da_exige_conservar_no_objeta():
    f = _correr(_fila("ELIX_20"))["ELIX_20"]
    assert f["contraste_fase1"] == "fase1_exige_conservar_estadistica_no_objeta" and f["regla_decisiva"] == "MANTENER" and f["restriccion_fase1"] == dc.LEYENDA_R3


def test_r3_las_31_categorias_reciben_la_restriccion_y_ninguna_otra_variable():
    filas = [_fila(f"ELIX_{i:02d}") for i in range(1, 32)] + [_fila("EDAD_ANIOS"), _fila("PUNTAJE_VAN_WALRAVEN"), _fila("N_COMORB_ELIX"), _fila("ELIX_32")]
    f = _correr(*filas)
    assert [n for n, r in f.items() if r["restriccion_fase1"]] == [f"ELIX_{i:02d}" for i in range(1, 32)]
    assert all(f[n]["restriccion_fase1"] == "" and f[n]["contraste_fase1"] == "fase1_permite_estadistica_no_objeta" for n in ("EDAD_ANIOS", "PUNTAJE_VAN_WALRAVEN", "N_COMORB_ELIX", "ELIX_32"))


def test_r3_con_un_rol_excluyente_prevalece_la_exclusion_sobre_la_exigencia():
    f = _correr(_fila("ELIX_05", rol_fase1="AUDITORIA_SENSIBILIDAD"))["ELIX_05"]
    assert f["regla_decisiva"] == "R_FASE1" and f["contraste_fase1"] == "fase1_excluye_estadistica_no_objeta" and f["restriccion_fase1"] == dc.LEYENDA_R3


# ----------------------------------------------------------------------------------------- contraste con la Fase 1
ETNIA = dict(
    rol_fase1="AUDITORIA_SENSIBILIDAD", evaluable="parcial", motivo_no_evaluable="decisión 4: no se analiza frente a resultados", analiza_resultado="no",
    estado_medicion="no analiza resultado",
)
COD_HOSPITAL = dict(
    rol_fase1="LLAVE_AGREGACION", evaluable="no", motivo_no_evaluable="referencia del análisis de proxy, no es candidata", uso_panel="referencia", analiza_resultado="",
    estado_medicion="referencia",
)


def test_las_nueve_etiquetas_de_contraste_cada_una_con_un_caso_armado_a_mano():
    casos = {
        "coinciden": _fila("V1", rol_fase1="PROHIBIDA_FUGA", auc_ranking=0.90),
        "fase1_excluye_estadistica_no_objeta": _fila("V2", rol_fase1="LLAVE_AGREGACION"),
        "fase1_excluye_sin_evidencia": _fila("V3", **ETNIA, **_sin_evidencia(asociacion_hospital=0.05, psi_vs_2023=0.01)),
        "fase1_exige_conservar_estadistica_objeta": _fila("ELIX_03", auc_senal=0.508, incremento_auc=0.0),
        "fase1_exige_conservar_estadistica_no_objeta": _fila("ELIX_20"),
        "fase1_exige_conservar_sin_evidencia": _fila("ELIX_28", estado_medicion="constante", **_sin_evidencia()),
        "fase1_permite_estadistica_objeta": _fila("V7", faltante_max=0.80),
        "fase1_permite_estadistica_no_objeta": _fila("V8"),
        "fase1_permite_no_evaluable": _fila("V9", estado_medicion="constante", **_sin_evidencia()),
    }
    f = _correr(*casos.values())
    assert set(casos) == set(ETIQUETAS) == set(dc.ETIQUETAS_CONTRASTE) and len(dc.ETIQUETAS_CONTRASTE) == 9
    for etiqueta, fila in casos.items():
        assert f[fila["variable"]]["contraste_fase1"] == etiqueta, fila["variable"]


def test_las_etiquetas_son_exhaustivas_y_excluyentes_en_toda_combinacion_de_rol_restriccion_y_medicion():
    medidas = {
        "sana": {}, "constante": dict(estado_medicion="constante", **_sin_evidencia()), "proxy": dict(asociacion_hospital=0.40), "baja": dict(auc_senal=0.50, incremento_auc=0.0),
        "no_evaluable": dict(evaluable="no", motivo_no_evaluable="motivo", **_sin_evidencia()),
    }
    filas, esperadas = [], {}
    for rol in ("FEATURE_BASAL", "CONDICIONAL", "PROHIBIDA_FUGA", "AUDITORIA_SENSIBILIDAD", "FILTRO_COHORTE"):
        for nombre in ("ELIX_05", "VARIABLE"):
            for medida, cambios in medidas.items():
                clave = f"{nombre}|{rol}|{medida}"
                filas.append(_fila(clave, rol_fase1=rol, **cambios))
                esperadas[clave] = (rol in dc.ROLES_EXCLUYENTES, nombre.startswith("ELIX_"), medida)
    # el árbol conoce la restricción por el nombre de la variable: se renombra después de armar el caso
    base = pl.DataFrame(filas, schema=ESQUEMA_BASE)
    base = base.with_columns(pl.col("variable").str.split("|").list.get(0).alias("variable"), pl.col("variable").alias("_clave"))
    out = dc.aplicar_arbol(base.drop("_clave"), UMBRALES, conocimiento=_conocimiento())
    assert out.height == len(esperadas) and out["contraste_fase1"].null_count() == 0 and set(out["contraste_fase1"]) <= set(ETIQUETAS)
    vistas = set()
    for (clave, (excluyente, elix, medida)), r in zip(esperadas.items(), out.iter_rows(named=True)):
        etiqueta = r["contraste_fase1"]
        vistas.add(etiqueta)
        familia = "fase1_excluye" if excluyente else ("fase1_exige_conservar" if elix else "fase1_permite")
        assert etiqueta == "coinciden" if (excluyente and r["reglas_activadas"].count(";") >= 1) else etiqueta.startswith(familia), clave
        if etiqueta.endswith("_no_objeta"):
            assert r["medicion_completa"] == "si" and not (set(_activas(r)) - {"R_FASE1", "MANTENER"}), clave
        if r["medicion_completa"] == "no":
            assert not etiqueta.endswith("_no_objeta"), clave
    assert vistas == set(ETIQUETAS)


def test_etnia_y_cod_hospital_nunca_reciben_una_etiqueta_no_objeta():                                      # M35
    f = _correr(
        _fila("ETNIA", **ETNIA, **_sin_evidencia(asociacion_hospital=0.05, psi_vs_2023=0.01, faltante_max=0.20)),
        _fila("PUEBLO_ORIGINARIO", **ETNIA, **_sin_evidencia(asociacion_hospital=0.20, psi_vs_2023=0.01, faltante_max=0.20)),
        _fila("COD_HOSPITAL", **COD_HOSPITAL, **_sin_evidencia()),
    )
    assert f["ETNIA"]["contraste_fase1"] == "fase1_excluye_sin_evidencia" and f["ETNIA"]["medicion_completa"] == "no"
    assert f["PUEBLO_ORIGINARIO"]["contraste_fase1"] == "coinciden" and _activas(f["PUEBLO_ORIGINARIO"]) == ["R_FASE1", "PROXY_MOD"]      # el proxy sí se mide en ETNIA
    assert f["COD_HOSPITAL"]["contraste_fase1"] == "fase1_excluye_sin_evidencia" and f["COD_HOSPITAL"]["motivo_no_evaluable"] == COD_HOSPITAL["motivo_no_evaluable"]
    assert all(not r["contraste_fase1"].endswith("_no_objeta") for r in f.values())


def test_la_etnia_conserva_el_motivo_del_registro_y_no_se_le_inventa_una_falta_de_medicion():
    f = _correr(_fila("ETNIA", **ETNIA, **_sin_evidencia(faltante_max=0.20)))["ETNIA"]
    assert f["motivo_no_evaluable"] == ETNIA["motivo_no_evaluable"] and f["estado_medicion"] == "no analiza resultado" and f["evaluable"] == "parcial"


# ----------------------------------------------------------------------------------------------- otras columnas
def test_psi_max_es_el_maximo_de_los_dos_psi_y_nulo_si_faltan_ambos():
    f = _correr(
        _fila("A", psi_vs_2023=0.10, psi_2019_vs_resto=0.30), _fila("B", psi_vs_2023=0.40, psi_2019_vs_resto=None), _fila("C", psi_vs_2023=None, psi_2019_vs_resto=None),
    )
    assert f["A"]["psi_max"] == 0.30 and f["B"]["psi_max"] == 0.40 and f["C"]["psi_max"] is None


def test_la_certeza_clinica_usa_los_alias_y_dice_sin_ficha_si_no_hay_fila():
    ficha = _conocimiento(EDAD=VERIFICADA, CAPITULO_CIE10=PENDIENTE, ESTANCIA_DIAS=VERIFICADA, ELIX_01=VERIFICADA)
    f = _correr(_fila("EDAD_ANIOS"), _fila("GRUPO_CLINICO"), _fila("ESTANCIA"), _fila("ELIX_01"), _fila("OTRA"), conocimiento=ficha)
    assert [f[n]["certeza_clinica"] for n in ("EDAD_ANIOS", "GRUPO_CLINICO", "ESTANCIA", "ELIX_01", "OTRA")] == [VERIFICADA, PENDIENTE, VERIFICADA, VERIFICADA, "sin ficha"]
    assert dc.ALIAS_CONOCIMIENTO == {"EDAD_ANIOS": "EDAD", "GRUPO_CLINICO": "CAPITULO_CIE10", "ESTANCIA": "ESTANCIA_DIAS"}


def test_con_el_conocimiento_real_las_fichas_de_elixhauser_estan_y_las_categorias_pendientes_no_son_clinico():
    base = pl.DataFrame([_fila("ELIX_01", auc_senal=0.50, incremento_auc=0.0), _fila("ELIX_03", auc_senal=0.50, incremento_auc=0.0)], schema=ESQUEMA_BASE)
    out = {r["variable"]: r for r in dc.aplicar_arbol(base, UMBRALES).iter_rows(named=True)}              # sin ``conocimiento``: lee el archivo real
    assert out["ELIX_01"]["certeza_clinica"] == VERIFICADA and out["ELIX_01"]["regla_decisiva"] == "CLINICO"
    assert out["ELIX_03"]["certeza_clinica"] == PENDIENTE and out["ELIX_03"]["regla_decisiva"] == "BAJA_SENAL"


def test_los_valores_de_la_entrada_pasan_tal_cual_a_la_salida_y_las_opcionales_tambien():
    f = _una(origen="cruda", tipo="cat", fuente_rol="reglas_variables", fuente_faltante="t22", faltante_informativo="si", evidencia="a.csv; b.csv", auc_ranking=0.61234567, faltante_max=0.123)
    assert (f["origen"], f["tipo"], f["fuente_rol"], f["fuente_faltante"], f["faltante_informativo"], f["evidencia"]) == ("cruda", "cat", "reglas_variables", "t22", "si", "a.csv; b.csv")
    assert f["auc_ranking"] == 0.61234567 and f["faltante_max"] == 0.123 and f["estado_medicion"] == "ok"      # la matriz real conserva el valor sin redondear


# ------------------------------------------------------------------------------------------------------ publicación
ESTADISTICOS = [
    "auc_ranking", "auc_senal", "auc_confirmacion", "incremento_auc", "asociacion_hospital", "psi_vs_2023", "psi_2019_vs_resto", "psi_max", "faltante_max", "margen_auc_vs_representante",
]


def _matriz_ejemplo() -> pl.DataFrame:
    """V1 y V2 (mismo cluster, V1 representante), F1 (solo fuga) y T (su faltante sale de t22)."""
    filas = [
        _fila("V1", cluster=1, tamano_cluster=2, auc_ranking=0.62, auc_senal=0.62, auc_confirmacion=0.60, incremento_auc=0.02, asociacion_hospital=0.05, psi_vs_2023=0.01,
              psi_2019_vs_resto=0.02, faltante_max=0.12),
        _fila("V2", cluster=1, tamano_cluster=2, auc_ranking=0.58, auc_senal=0.58, auc_confirmacion=0.57, incremento_auc=0.01, asociacion_hospital=0.06, psi_vs_2023=0.03,
              psi_2019_vs_resto=0.04, faltante_max=0.06),
        _fila("F1", uso_panel="solo_fuga", rol_fase1="PROHIBIDA_FUGA", analiza_resultado="", **_sin_evidencia(auc_ranking=0.70, auc_confirmacion=0.68)),
        _fila("T", fuente_faltante="t22", auc_ranking=0.61, auc_senal=0.61, auc_confirmacion=0.60, incremento_auc=0.03, asociacion_hospital=0.07, psi_vs_2023=0.05,
              psi_2019_vs_resto=0.06, faltante_max=0.30),
    ]
    return dc.aplicar_arbol(pl.DataFrame(filas, schema=ESQUEMA_BASE), UMBRALES, conocimiento=_conocimiento())


def _fuentes() -> dict[str, pl.DataFrame]:
    """Versiones publicables mínimas (solo las columnas que lee ``publicar_matriz``) de la evidencia de ``_matriz_ejemplo``."""
    nombres = ["V1", "V2", "T"]
    return {
        "ranking": pl.DataFrame({
            "variable": [n for n in nombres for _ in (0, 1)], "ventana": ["2019-2022", "2023"] * 3, "auc": [0.62, 0.60, 0.58, 0.57, 0.61, 0.60],
            "auc_senal": [0.62, 0.60, 0.58, 0.57, 0.61, 0.60], "prop_faltante": [0.10, 0.12, 0.05, 0.06, 0.01, 0.02],
        }),
        "estabilidad": pl.DataFrame({
            "variable": [n for n in nombres for _ in (0, 1)], "tipo_fila": ["psi"] * 6, "comparacion": [dc.COMP_CONFIRMACION, dc.COMP_RESTO] * 3, "psi": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        }),
        "proxy": pl.DataFrame({"variable": nombres, "tipo_fila": ["variable"] * 3, "asociacion_hospital": [0.05, 0.06, 0.07]}),
        "importancia": pl.DataFrame({"variable": nombres, "tipo_fila": ["variable"] * 3, "incremento_auc": [0.02, 0.01, 0.03]}),
        "fuga": pl.DataFrame({"variable": ["F1"], "tipo_fila": ["variable"], "categoria": ["solo_fuga"], "auc_ranking": [0.70], "auc_confirmacion": [0.68]}),
    }


def _anular(t: pl.DataFrame, columna: str, **filtro) -> pl.DataFrame:
    """Anula una celda de la tabla fuente, como lo hace la supresión de una fila de menos de diez casos."""
    cond = pl.lit(True)
    for k, v in filtro.items():
        cond = cond & (pl.col(k) == v)
    return t.with_columns(pl.when(cond).then(None).otherwise(pl.col(columna)).alias(columna))


def _por_variable(t: pl.DataFrame) -> dict[str, dict]:
    return {r["variable"]: r for r in t.iter_rows(named=True)}


def test_la_matriz_de_ejemplo_tiene_los_valores_esperados_antes_de_publicar():
    m = _por_variable(_matriz_ejemplo())
    assert m["V1"]["es_representante"] == "si" and m["V2"]["margen_auc_vs_representante"] == pytest.approx(-0.04) and m["V1"]["margen_auc_vs_representante"] == 0.0
    assert m["F1"]["regla_decisiva"] == "R_FASE1" and m["F1"]["auc_ranking"] == 0.70 and m["T"]["fuente_faltante"] == "t22"


def test_sin_supresion_la_matriz_publicada_es_igual_a_la_real_y_valor_suprimido_es_no():
    m = _matriz_ejemplo()
    p = dc.publicar_matriz(m, _fuentes())
    assert p.equals(m) and set(p["valor_suprimido"]) == {"no"}
    assert dc.publicar_matriz(m, {}).equals(m)                                       # sin tablas fuente no hay nada que anular


@pytest.mark.parametrize(
    "fuente, columna, filtro, anulados",
    [
        ("ranking", "auc", {"variable": "V1", "ventana": "2019-2022"}, {"auc_ranking"}),
        ("ranking", "auc_senal", {"variable": "V1", "ventana": "2019-2022"}, {"auc_senal", "margen_auc_vs_representante"}),
        ("ranking", "auc", {"variable": "V1", "ventana": "2023"}, {"auc_confirmacion"}),
        ("importancia", "incremento_auc", {"variable": "V1"}, {"incremento_auc"}),
        ("proxy", "asociacion_hospital", {"variable": "V1"}, {"asociacion_hospital"}),
        ("estabilidad", "psi", {"variable": "V1", "comparacion": dc.COMP_CONFIRMACION}, {"psi_vs_2023", "psi_max"}),
        ("estabilidad", "psi", {"variable": "V1", "comparacion": dc.COMP_RESTO}, {"psi_2019_vs_resto", "psi_max"}),
        ("ranking", "prop_faltante", {"variable": "V1", "ventana": "2019-2022"}, {"faltante_max"}),
        ("ranking", "prop_faltante", {"variable": "V1", "ventana": "2023"}, {"faltante_max"}),
    ],
)
def test_publicar_anula_solo_el_estadistico_cuya_fila_fuente_quedo_suprimida(fuente, columna, filtro, anulados):           # M40
    m = _matriz_ejemplo()
    fuentes = _fuentes()
    fuentes[fuente] = _anular(fuentes[fuente], columna, **filtro)
    real, pub = _por_variable(m), _por_variable(dc.publicar_matriz(m, fuentes))
    for col in ESTADISTICOS:
        assert pub["V1"][col] == (None if col in anulados else real["V1"][col]), col
    assert pub["V1"]["valor_suprimido"] == "si"
    for n in ("F1", "T"):
        assert pub[n] == real[n], n
    resto = {k: v for k, v in pub["V2"].items() if k not in ("margen_auc_vs_representante", "valor_suprimido")}
    assert resto == {k: v for k, v in real["V2"].items() if k not in ("margen_auc_vs_representante", "valor_suprimido")}


def test_la_recomendacion_y_la_regla_decisiva_son_las_de_los_valores_reales_aunque_se_anule_todo():
    m = _matriz_ejemplo()
    anuladas = {k: t.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c, d in t.schema.items() if d.is_float()]) for k, t in _fuentes().items()}
    p = dc.publicar_matriz(m, anuladas)
    decision = [c for c in dc.COLUMNAS_MATRIZ if c not in ESTADISTICOS and c != "valor_suprimido"]
    assert p.select(decision).equals(m.select(decision))
    real, pub = _por_variable(m), _por_variable(p)
    for n in ("V1", "V2", "F1", "T"):
        assert pub[n]["valor_suprimido"] == "si", n
    assert all(pub["V1"][c] is None for c in ESTADISTICOS) and all(pub["V2"][c] is None for c in ESTADISTICOS)
    assert pub["F1"]["auc_ranking"] is None and pub["F1"]["auc_confirmacion"] is None
    assert pub["T"]["faltante_max"] == real["T"]["faltante_max"] == 0.30            # el faltante de t22 ya viene suprimido de su propia tabla: no depende del ranking
    assert real["V1"]["recomendacion"] == pub["V1"]["recomendacion"] and real["V1"]["regla_decisiva"] == pub["V1"]["regla_decisiva"]


def test_el_margen_se_anula_si_se_suprimio_la_fila_de_la_variable_o_la_de_su_representante():
    m = _matriz_ejemplo()
    sin_representante = _fuentes()
    sin_representante["ranking"] = _anular(sin_representante["ranking"], "auc_senal", variable="V1", ventana="2019-2022")
    pub = _por_variable(dc.publicar_matriz(m, sin_representante))
    assert pub["V2"]["margen_auc_vs_representante"] is None and pub["V2"]["valor_suprimido"] == "si" and pub["V2"]["auc_senal"] == 0.58      # su propia cifra sigue publicada
    sin_miembro = _fuentes()
    sin_miembro["ranking"] = _anular(sin_miembro["ranking"], "auc_senal", variable="V2", ventana="2019-2022")
    pub = _por_variable(dc.publicar_matriz(m, sin_miembro))
    assert pub["V2"]["margen_auc_vs_representante"] is None and pub["V2"]["auc_senal"] is None
    assert pub["V1"]["margen_auc_vs_representante"] == 0.0 and pub["V1"]["valor_suprimido"] == "no"


def test_una_variable_de_solo_fuga_se_publica_con_la_tabla_de_fuga_y_no_con_el_ranking():
    m = _matriz_ejemplo()
    fuentes = _fuentes()
    filas_ranking_f1 = pl.DataFrame({"variable": ["F1", "F1"], "ventana": ["2019-2022", "2023"], "auc": [None, None], "auc_senal": [None, None], "prop_faltante": [None, None]},
                                    schema=fuentes["ranking"].schema)
    fuentes["ranking"] = pl.concat([fuentes["ranking"], filas_ranking_f1])
    assert _por_variable(dc.publicar_matriz(m, fuentes))["F1"]["valor_suprimido"] == "no"                  # el ranking no manda sobre una variable de solo fuga
    fuentes["fuga"] = _anular(fuentes["fuga"], "auc_ranking", variable="F1")
    pub = _por_variable(dc.publicar_matriz(m, fuentes))["F1"]
    assert pub["auc_ranking"] is None and pub["auc_confirmacion"] == 0.68 and pub["valor_suprimido"] == "si"
    fuentes["fuga"] = _anular(_fuentes()["fuga"], "auc_confirmacion", variable="F1")
    pub = _por_variable(dc.publicar_matriz(m, fuentes))["F1"]
    assert pub["auc_ranking"] == 0.70 and pub["auc_confirmacion"] is None


def test_una_condicional_de_la_tabla_de_fuga_se_publica_con_el_ranking():
    m = _matriz_ejemplo()
    fuentes = _fuentes()
    fuentes["fuga"] = pl.DataFrame({"variable": ["F1", "V1"], "tipo_fila": ["variable"] * 2, "categoria": ["solo_fuga", "condicional"], "auc_ranking": [0.70, None],
                                    "auc_confirmacion": [0.68, None]})
    pub = _por_variable(dc.publicar_matriz(m, fuentes))
    assert pub["V1"]["auc_ranking"] == 0.62 and pub["V1"]["auc_confirmacion"] == 0.60 and pub["V1"]["valor_suprimido"] == "no"


def test_un_estadistico_que_ya_era_nulo_en_la_matriz_real_no_marca_valor_suprimido():
    base = pl.DataFrame([_fila("V1", incremento_auc=None, incremento_estado="no aplica: variable del modelo base")], schema=ESQUEMA_BASE)
    m = dc.aplicar_arbol(base, UMBRALES, conocimiento=_conocimiento())
    fuentes = _fuentes()
    fuentes["importancia"] = _anular(fuentes["importancia"], "incremento_auc", variable="V1")
    p = dc.publicar_matriz(m, fuentes)
    assert p["valor_suprimido"].to_list() == ["no"] and p["incremento_auc"].to_list() == [None]


def test_publicar_exige_que_las_columnas_de_la_matriz_sean_las_del_esquema():
    m = _matriz_ejemplo()
    assert dict(dc.publicar_matriz(m, _fuentes()).schema) == dc.ESQUEMA_MATRIZ and dc.publicar_matriz(m, _fuentes()).columns == dc.COLUMNAS_MATRIZ


# ============================================================================== armado, escritura e informe (panel artificial)
@pytest.fixture(scope="module")
def artificial(tmp_path_factory, fuente_sintetica):
    """La evidencia real (y su versión publicable) del panel artificial: lo que ``seleccion.calcular`` le entregará a ``decision.armar_matriz``."""
    from panel_artificial import panel_artificial

    from eda import asociacion as asoc
    from eda import modelos, sospechas, supresion
    from eda import seleccion as sel
    from eda.contexto import Contexto

    ctx = Contexto.crear("extracto", fuente=fuente_sintetica, dir_salida_base=tmp_path_factory.mktemp("decision_artificial"))
    panel = panel_artificial()
    ranking = sel.tabla_ranking(ctx, panel)
    estabilidad = sel.tabla_estabilidad(ctx, panel, ranking)
    proxy = sospechas.tabla_proxy(ctx, panel, ranking)
    clase = {f["variable"]: f["clase_proxy"] for f in proxy.filter(pl.col("tipo_fila") == "variable").iter_rows(named=True)}
    referencias = modelos.referencias(panel, ctx.semilla)
    importancia = sel.tabla_importancia(ctx, panel, referencias, clase)
    fuga = sospechas.tabla_fuga(ctx, panel, ranking, referencias)
    clusters = asoc.calcular(ctx, panel)["clusters"]

    def publicable(t, bloques):
        return supresion.suprimir_bloques(t, bloques, minimo=ctx.celda_minima)[0]

    ev = {
        "ranking": ranking, "estabilidad": estabilidad, "proxy": proxy, "importancia": importancia, "fuga": fuga, "clusters": clusters,
        "ranking_pub": publicable(ranking, sel.BLOQUES_RANKING), "estabilidad_pub": publicable(estabilidad, sel.BLOQUES_ESTABILIDAD),
        "proxy_pub": publicable(proxy, sospechas.BLOQUES_PROXY), "importancia_pub": publicable(importancia, sel.BLOQUES_IMPORTANCIA), "fuga_pub": publicable(fuga, sospechas.BLOQUES_FUGA),
    }
    return SimpleNamespace(ctx=ctx, panel=panel, ev=ev)


def _armar(art, **cambios):
    return dc.armar_matriz(art.ctx, art.panel, {**art.ev, **cambios})


def _igual(a, b) -> bool:
    return (a is None and b is None) or (a is not None and b is not None and a == pytest.approx(b))


def test_la_matriz_cubre_exactamente_el_universo_de_disponibilidad_con_el_panel_artificial(artificial):         # DT09
    r = _armar(artificial)
    universo = dc.cargar_universo()["variable"].to_list()
    assert set(r) == {"disponibilidad", "matriz", "matriz_pub"} and r["disponibilidad"]["variable"].to_list() == universo
    for clave in ("matriz", "matriz_pub"):
        assert r[clave]["variable"].to_list() == universo and r[clave].height == 199 and r[clave]["variable"].n_unique() == 199
        assert r[clave].columns == dc.COLUMNAS_MATRIZ and dict(r[clave].schema) == dc.ESQUEMA_MATRIZ


def test_las_etiquetas_de_ventana_y_de_comparacion_son_las_del_panel_y_las_de_la_estabilidad(artificial):
    from eda import panel as modulo_panel

    assert dc.V_RANKING == modulo_panel.V_RANK and dc.V_CONFIRMACION == modulo_panel.V_CONF
    assert set(artificial.ev["ranking"]["ventana"]) == {dc.V_RANKING, dc.V_CONFIRMACION}
    assert set(artificial.ev["estabilidad"].filter(pl.col("tipo_fila") == "psi")["comparacion"]) == {dc.COMP_CONFIRMACION, dc.COMP_RESTO}


def test_las_columnas_estadisticas_de_las_candidatas_son_las_de_sus_tablas_fuente(artificial):
    ev = artificial.ev
    m = _por_variable(_armar(artificial)["matriz"])
    rk = {(f["variable"], f["ventana"]): f for f in ev["ranking"].iter_rows(named=True)}
    im = _por_variable(ev["importancia"].filter(pl.col("tipo_fila") == "variable"))
    px = _por_variable(ev["proxy"].filter(pl.col("tipo_fila") == "variable"))
    psi = {(f["variable"], f["comparacion"]): f["psi"] for f in ev["estabilidad"].filter(pl.col("tipo_fila") == "psi").iter_rows(named=True)}
    resumen = _por_variable(ev["estabilidad"].filter(pl.col("tipo_fila") == "resumen"))
    cl = _por_variable(ev["clusters"])
    candidatas = [n for (n, v) in rk if v == dc.V_RANKING]
    assert len(candidatas) == 56
    for n in candidatas:
        a, b, r = rk[(n, dc.V_RANKING)], rk[(n, dc.V_CONFIRMACION)], m[n]
        assert _igual(r["auc_ranking"], a["auc"]) and _igual(r["auc_senal"], a["auc_senal"]) and _igual(r["auc_confirmacion"], b["auc"]), n
        assert _igual(r["incremento_auc"], im[n]["incremento_auc"]) and _igual(r["asociacion_hospital"], px[n]["asociacion_hospital"]), n
        assert _igual(r["psi_vs_2023"], psi[(n, dc.COMP_CONFIRMACION)]) and _igual(r["psi_2019_vs_resto"], psi[(n, dc.COMP_RESTO)]), n
        assert r["cambio_signo"] == resumen[n]["cambio_signo"] and (r["cluster"], r["tamano_cluster"]) == (cl[n]["cluster"], cl[n]["tamano_cluster"]), n
        faltantes = [x for x in (a["prop_faltante"], b["prop_faltante"]) if x is not None]
        assert _igual(r["faltante_max"], max(faltantes) if faltantes else None) and r["fuente_faltante"] == "panel" and r["faltante_informativo"] == a["faltante_informativo"], n


def test_el_estado_de_medicion_sale_del_ranking_de_la_fuga_o_de_la_naturaleza_de_la_variable(artificial):
    m = _por_variable(_armar(artificial)["matriz"])
    ranking = artificial.ev["ranking"].filter(pl.col("ventana") == dc.V_RANKING)
    for n, estado in zip(ranking["variable"], ranking["estado"]):
        assert m[n]["estado_medicion"] == estado, n
    assert "constante" in set(ranking["estado"]) and m["ELIX_28"]["estado_medicion"] == "constante"
    assert m["ETNIA"]["estado_medicion"] == m["PUEBLO_ORIGINARIO"]["estado_medicion"] == "no analiza resultado"
    assert m["COD_HOSPITAL"]["estado_medicion"] == "referencia"
    fuga = _por_variable(artificial.ev["fuga"].filter((pl.col("tipo_fila") == "variable") & (pl.col("categoria") == "solo_fuga")))
    for n, f in fuga.items():
        assert m[n]["estado_medicion"] == f["estado"], n
    for n in ("DIAGNOSTICO7", "N_EGRESOS_12M", "ESTANCIA_DIAS", "ID_EPISODIO"):
        assert m[n]["estado_medicion"] == "sin fila de medición" and m[n]["medicion_completa"] == "no", n


def test_las_variables_de_solo_fuga_toman_el_auc_de_la_tabla_de_fuga_y_no_tienen_auc_senal(artificial):
    ev = artificial.ev
    m = _por_variable(_armar(artificial)["matriz"])
    fuga = _por_variable(ev["fuga"].filter(pl.col("tipo_fila") == "variable"))
    solo = [n for n, f in fuga.items() if f["categoria"] == "solo_fuga"]
    assert {"SERVICIOALTA", "USOSPABELLON", "IR_29301_PESO", "IR_29301_SEVERIDAD", "IR_29301_MORTALIDAD", "N_PROCEDIMIENTOS", "N_TRASLADOS_INTERNOS"} <= set(solo)
    for n in solo:
        assert _igual(m[n]["auc_ranking"], fuga[n]["auc_ranking"]) and _igual(m[n]["auc_confirmacion"], fuga[n]["auc_confirmacion"]), n
        assert m[n]["auc_senal"] is None and m[n]["incremento_auc"] is None and m[n]["regla_decisiva"] == "R_FASE1" and m[n]["medicion_completa"] == "si", n
    assert "FUGA" in m["IR_29301_MORTALIDAD"]["reglas_activadas"].split("; ") and m["IR_29301_MORTALIDAD"]["contraste_fase1"] == "coinciden"
    for n in ("SERVICIOINGRESO", "ESPECIALIDAD_MEDICA", "NIVEL_CUIDADO_INGRESO", "DERIVADO_OTRO_HOSPITAL"):                                # las condicionales se miden con el ranking
        assert fuga[n]["categoria"] == "condicional" and m[n]["auc_senal"] is not None, n


def test_nit_11a_el_faltante_de_etnia_y_pueblo_originario_sale_de_t22_y_el_de_las_candidatas_del_panel_del_ranking(artificial):
    perfil = pl.DataFrame({
        "columna": ["ETNIA"] * 6 + ["SEXO"] * 2, "anio": [2019, 2020, 2021, 2022, 2023, 2024, 2019, 2020],
        "pct_faltante": [40.0, 42.0, 45.0, 41.0, 43.0, 99.0, 99.0, 98.0],
    }, schema={"columna": pl.String, "anio": pl.Int64, "pct_faltante": pl.Float64})
    disponibilidad = dc.construir_disponibilidad(artificial.ctx, artificial.panel.variables, perfil)
    m = _por_variable(_armar(artificial, disponibilidad=disponibilidad)["matriz"])
    for n in ("ETNIA", "PUEBLO_ORIGINARIO"):
        assert m[n]["faltante_max"] == pytest.approx(0.45) and m[n]["fuente_faltante"] == "t22", n            # 2024 no cuenta
    ranking = artificial.ev["ranking"].filter(pl.col("variable") == "SEXO")
    assert m["SEXO"]["fuente_faltante"] == "panel" and m["SEXO"]["faltante_max"] == pytest.approx(max(ranking["prop_faltante"].drop_nulls()))        # t22 no pisa al panel
    assert m["COD_HOSPITAL"]["fuente_faltante"] == "t22" and m["COD_HOSPITAL"]["faltante_max"] is None                                             # el perfil no trae la columna
    assert m["ETNIA"]["medicion_completa"] == "no" and m["ETNIA"]["contraste_fase1"] in ("coinciden", "fase1_excluye_sin_evidencia")


def test_sin_t22_el_faltante_de_etnia_queda_nulo_y_el_de_las_candidatas_sigue_siendo_el_del_panel(artificial):
    m = _por_variable(_armar(artificial)["matriz"])
    assert m["ETNIA"]["faltante_max"] is None and m["ETNIA"]["fuente_faltante"] == "no disponible" and m["SEXO"]["fuente_faltante"] == "panel"


def test_la_evidencia_lista_las_tablas_con_filas_de_la_variable(artificial):
    m = _por_variable(_armar(artificial)["matriz"])
    e = {n: r["evidencia"].split("; ") for n, r in m.items()}
    comunes = ["t23_ranking_variables.csv", "t23_estabilidad.csv", "t23_proxy_hospital.csv", "matriz_asociacion.csv", "clusters_variables.csv", "t23_importancia.csv"]
    assert e["EDAD_ANIOS"] == ["disponibilidad_variables.csv", "Fase_2__Variables_Derivadas.csv", *comunes]
    assert e["SEXO"] == ["disponibilidad_variables.csv", "Fase_2__Diccionario_de_Base_de_Datos.csv", *comunes]
    assert e["ELIX_05"] == ["disponibilidad_variables.csv", "Fase_2__Variables_Derivadas.csv", *comunes, "t23_or_elix.csv", "t23_sensibilidad_elix.csv"]
    assert e["SERVICIOALTA"] == ["disponibilidad_variables.csv", "reglas_variables.csv", "t23_fuga.csv"]
    assert e["DERIVADO_OTRO_HOSPITAL"] == ["disponibilidad_variables.csv", "reglas_variables.csv", *comunes, "t23_fuga.csv", "t23_procedencia_autorref.csv"]
    assert e["SERVICIOINGRESO"] == ["disponibilidad_variables.csv", "reglas_variables.csv", *comunes, "t23_fuga.csv"]
    assert e["ETNIA"] == ["disponibilidad_variables.csv", "reglas_variables.csv", "t23_estabilidad.csv", "t23_proxy_hospital.csv", "matriz_asociacion.csv", "clusters_variables.csv"]
    assert e["N_PROCEDIMIENTOS"] == ["disponibilidad_variables.csv", "Fase_2__Diccionario_de_Base_de_Datos.csv", "t23_fuga.csv"]            # el rol lo heredó de PROCEDIMIENTO1
    assert e["DIAGNOSTICO7"] == ["disponibilidad_variables.csv", "Fase_2__Diccionario_de_Base_de_Datos.csv"]
    assert all(x[0] == "disponibilidad_variables.csv" and len(x) == len(set(x)) and "" not in x for x in e.values())


def test_los_historiales_agregan_su_tabla_de_factibilidad_solo_en_modo_completo_y_si_son_evaluables(artificial):
    import dataclasses

    historiales = ("N_EGRESOS_12M", "DIAS_DESDE_EGRESO_PREVIO", "HISTORIA_DISPONIBLE")
    evaluables = {n: dataclasses.replace(artificial.panel.variables[n], evaluable="si", motivo="") for n in historiales}
    panel = dataclasses.replace(artificial.panel, variables={**artificial.panel.variables, **evaluables})
    extracto = dc.armar_matriz(artificial.ctx, panel, artificial.ev)["matriz"]
    completo = dc.armar_matriz(SimpleNamespace(etiqueta="completo", umbrales=artificial.ctx.umbrales, dir_tablas=artificial.ctx.dir_tablas), panel, artificial.ev)["matriz"]
    no_evaluable = dc.armar_matriz(SimpleNamespace(etiqueta="completo", umbrales=artificial.ctx.umbrales, dir_tablas=artificial.ctx.dir_tablas), artificial.panel, artificial.ev)["matriz"]
    for n in historiales:
        assert "t23_historial_factibilidad.csv" not in _por_variable(extracto)[n]["evidencia"]
        assert "t23_historial_factibilidad.csv" in _por_variable(completo)[n]["evidencia"].split("; ")
        assert "t23_historial_factibilidad.csv" not in _por_variable(no_evaluable)[n]["evidencia"]
    assert "t23_historial_factibilidad.csv" not in "".join(r["evidencia"] for n, r in _por_variable(completo).items() if n not in historiales)


def test_armar_matriz_es_determinista_y_no_modifica_la_evidencia_que_recibe(artificial):
    antes = {k: t.clone() for k, t in artificial.ev.items()}
    a, b = _armar(artificial), _armar(artificial)
    assert a["matriz"].equals(b["matriz"]) and a["matriz_pub"].equals(b["matriz_pub"]) and a["disponibilidad"].equals(b["disponibilidad"])
    assert all(artificial.ev[k].equals(antes[k]) for k in antes)


def test_con_una_fila_fuente_suprimida_la_matriz_publica_nulos_pero_decide_igual(artificial):                 # M40 con el flujo real de supresión
    from eda import seleccion as sel
    from eda import supresion

    objetivo = "ELIX_20"
    ranking = artificial.ev["ranking"].with_columns(
        pl.when((pl.col("variable") == objetivo) & (pl.col("ventana") == dc.V_RANKING)).then(5).otherwise(pl.col("n")).alias("n")                      # una celda de 5 casos
    )
    ranking_pub = supresion.suprimir_bloques(ranking, sel.BLOQUES_RANKING, minimo=artificial.ctx.celda_minima)[0]
    fila = ranking_pub.filter((pl.col("variable") == objetivo) & (pl.col("ventana") == dc.V_RANKING))
    assert fila["auc"][0] is None and fila["auc_senal"][0] is None                                          # la fuente sí quedó suprimida
    base, suprimida = _armar(artificial), _armar(artificial, ranking=ranking, ranking_pub=ranking_pub)
    assert suprimida["matriz"].equals(base["matriz"])                                                       # la decisión usa los valores reales
    real, pub = _por_variable(suprimida["matriz"]), _por_variable(suprimida["matriz_pub"])
    assert real[objetivo]["auc_ranking"] is not None and real[objetivo]["auc_senal"] is not None
    assert pub[objetivo]["auc_ranking"] is None and pub[objetivo]["auc_senal"] is None and pub[objetivo]["valor_suprimido"] == "si"
    for col in ("regla_decisiva", "reglas_activadas", "recomendacion", "contraste_fase1", "es_representante", "medicion_completa"):
        assert pub[objetivo][col] == real[objetivo][col] == _por_variable(base["matriz"])[objetivo][col], col
    otras = [n for n in pub if n != objetivo]
    assert all(pub[n] == _por_variable(base["matriz_pub"])[n] for n in otras)                               # ninguna otra variable cambia


def test_sin_supresion_la_matriz_publicable_del_panel_artificial_solo_difiere_de_la_real_en_las_celdas_pequenas(artificial):
    r = _armar(artificial)
    decision = [c for c in dc.COLUMNAS_MATRIZ if c not in ESTADISTICOS and c != "valor_suprimido"]
    assert r["matriz_pub"].select(decision).equals(r["matriz"].select(decision))
    real, pub = _por_variable(r["matriz"]), _por_variable(r["matriz_pub"])
    for n, p in pub.items():
        for col in ESTADISTICOS:
            assert p[col] is None or _igual(p[col], real[n][col]), (n, col)                                  # lo publicado es el valor real o nulo, nunca otro valor
        assert (p["valor_suprimido"] == "si") == any(p[c] is None and real[n][c] is not None for c in ESTADISTICOS), n


def test_ejecutar_escribe_la_disponibilidad_y_la_matriz_publicable_sin_conteos(ctx, artificial):
    from eda import privacidad

    r = dc.ejecutar(ctx, artificial.panel, artificial.ev)
    assert set(r) == {"disponibilidad", "matriz", "matriz_pub"}
    indice = ctx.indice_tablas()
    assert set(indice["nombre"]) == {"disponibilidad_variables.csv", "matriz_decision_variables.csv"} and set(indice["tarea"]) == {"T11"}
    assert indice["filas"].to_list() == [199, 199] and all(not c for c in indice["columnas_conteo"]) and indice["celdas_suprimidas"].to_list() == [0, 0]
    leida = ctx.leer_tabla("matriz_decision_variables.csv", columnas_texto=dc._TEXTOS_MATRIZ)
    assert leida.columns == dc.COLUMNAS_MATRIZ and leida["variable"].to_list() == r["matriz_pub"]["variable"].to_list()
    for col in dc.COLUMNAS_MATRIZ:
        esperado, obtenido = r["matriz_pub"][col].to_list(), leida[col].to_list()
        if dc.ESQUEMA_MATRIZ[col] == pl.Float64:
            assert all(_igual(e, o) or (e is not None and o is not None and abs(e - o) < 2e-6) for e, o in zip(esperado, obtenido)), col
        else:
            assert [e or None for e in esperado] == [o or None for o in obtenido], col
    disp = ctx.leer_tabla("disponibilidad_variables.csv", columnas_texto=[c for c, t in dc.ESQUEMA_DISPONIBILIDAD_BASE.items() if t == pl.String] + ["fuente_faltante"])
    assert disp.columns == dc.columnas_disponibilidad([2019, 2020, 2021, 2022, 2023, 2024]) and disp["variable"].to_list() == leida["variable"].to_list()
    esc = privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima)
    assert privacidad.total_violaciones(esc) == 0, esc.filter(pl.col("n_violaciones") > 0)


def test_la_matriz_escrita_nunca_trae_un_nombre_de_columna_llamado_decision():
    assert not [c for c in dc.COLUMNAS_MATRIZ if c.lower() in ("decision", "decisión", "decision_final")]


# ========================================================================================= informe: sección 8 de la Tarea 11
@pytest.fixture()
def ctx_completo(tmp_path, fuente_sintetica):
    import dataclasses

    from eda.contexto import Contexto

    return Contexto.crear("completo", fuente=dataclasses.replace(fuente_sintetica, etiqueta="completo"), dir_salida_base=tmp_path)


def _escribir_matriz_y_resumen(ctx, matriz: pl.DataFrame, matriz_pub: pl.DataFrame | None = None) -> None:
    ctx.escribir_tabla("matriz_decision_variables.csv", matriz if matriz_pub is None else matriz_pub, tarea="T11", descripcion="matriz de prueba")
    ctx.escribir_tabla("t23_resumen_seleccion.csv", pl.DataFrame([dc.resumen_matriz(matriz, matriz_pub)]), tarea="T11", descripcion="resumen de prueba")


def _informe(ctx, n0: int = 1, nombre: str = "t23_prueba_decision.md") -> str:
    from eda.informes import Informe

    inf = Informe(ctx, nombre, "Informe de prueba")
    dc.agregar_secciones(inf, ctx, n0=n0)
    return inf.guardar().read_text(encoding="utf8")


def test_la_seccion_8_trae_el_conteo_las_tres_tablas_de_contraste_la_frase_de_r3_y_h01c_no_evaluable_en_extracto(ctx, artificial):
    from eda.informes import fmt_entero

    r = dc.ejecutar(ctx, artificial.panel, artificial.ev)
    ctx.escribir_tabla("t23_resumen_seleccion.csv", pl.DataFrame([dc.resumen_matriz(r["matriz"], r["matriz_pub"])]), tarea="T11", descripcion="resumen de prueba")
    md = _informe(ctx, n0=8)
    res = ctx.leer_tabla("t23_resumen_seleccion.csv").row(0, named=True)
    assert "## 8. Matriz de decisión y contraste con la Fase 1" in md and dc.N_SECCIONES == 1 and md.count("\n## ") == 1
    assert "la matriz recomienda; la decisión es humana" in md.lower() and "pendiente de revisión clínica (decisión 12)" in md
    for leyenda in ("Variables por recomendación de la matriz", "La Fase 1 excluye y la estadística no objeta (solo variables medidas por completo)",
                    "La Fase 1 permite y la estadística objeta", "La Fase 1 exige conservar y la estadística objeta", "Candidatas sin medición completa, por causa"):
        assert f"*{leyenda}.*" in md, leyenda
    assert "regla R3 de la Fase 1" in md and f"las {fmt_entero(res['n_restriccion_r3'])} deben estar en el modelo principal" in md and res["n_restriccion_r3"] == 31
    assert "H01c" in md and "no es evaluable" in md and "no evaluable en extracto" in md and "Variables de historial" not in md          # en extracto el historial no se evalúa
    assert f"{fmt_entero(res['n_matriz'])} variables del universo" in md and res["n_matriz"] == 199
    assert "{" not in md and "}" not in md
    for texto in RECOMENDACION.values():
        assert texto in md, texto                                                                           # la tabla de recomendaciones usa los textos del CSV
    assert "| NO EVALUABLE |" in md


def test_las_cifras_de_la_seccion_8_vienen_del_resumen_y_de_los_umbrales(ctx, artificial):
    r = dc.ejecutar(ctx, artificial.panel, artificial.ev)
    ctx.escribir_tabla("t23_resumen_seleccion.csv", pl.DataFrame([dc.resumen_matriz(r["matriz"], r["matriz_pub"])]), tarea="T11", descripcion="resumen de prueba")
    _informe(ctx)
    c = ctx.cifras_informe().filter(pl.col("informe") == "t23_prueba_decision.md")
    assert c.height > 15 and set(c["csv"]) <= {"t23_resumen_seleccion.csv", "umbrales.csv", "matriz_decision_variables.csv"} and "t23_resumen_seleccion.csv" in set(c["csv"])
    columnas = set(c.filter(pl.col("csv") == "t23_resumen_seleccion.csv")["columna"])
    assert {"n_matriz", "n_sin_medicion", "n_restriccion_r3", "n_contraste_coinciden"} <= columnas
    assert {"incremento_auc_minimo", "proxy_hospital_moderado"} <= {x for x in c.filter(pl.col("csv") == "umbrales.csv")["filtro"].str.extract(r'"umbral": "(\w+)"', 1)}


def test_la_seccion_8_numera_desde_n0(ctx, artificial):
    r = dc.ejecutar(ctx, artificial.panel, artificial.ev)
    ctx.escribir_tabla("t23_resumen_seleccion.csv", pl.DataFrame([dc.resumen_matriz(r["matriz"], r["matriz_pub"])]), tarea="T11", descripcion="resumen de prueba")
    assert "## 3. Matriz de decisión y contraste con la Fase 1" in _informe(ctx, n0=3)


def test_las_tablas_de_contraste_vacias_no_se_incrustan_y_sin_falta_de_medicion_lo_dice(ctx):
    base = pl.DataFrame([_fila(n) for n in ("N_EGRESOS_12M", "DIAS_DESDE_EGRESO_PREVIO")], schema=ESQUEMA_BASE)
    m = dc.aplicar_arbol(base, UMBRALES, conocimiento=_conocimiento())
    _escribir_matriz_y_resumen(ctx, m)
    md = _informe(ctx)
    assert "La Fase 1 excluye y la estadística no objeta" not in md and "La Fase 1 permite y la estadística objeta" not in md and "La Fase 1 exige conservar" not in md
    assert "Candidatas sin medición completa" not in md and "Todas las candidatas evaluables del panel se midieron por completo en esta corrida." in md


def _historiales_en_extracto() -> list[dict]:
    motivo = "no evaluable en extracto: el ingreso previo casi nunca está en el mismo extracto"
    return [_fila(n, evaluable="no", motivo_no_evaluable=motivo, estado_medicion="sin fila de medición", **_sin_evidencia()) for n in dc.HISTORIALES]


def test_una_tabla_de_contraste_larga_se_trunca_y_dice_cuantas_filas_hay(ctx):
    base = pl.DataFrame([_fila(f"FUGA_{i:02d}", rol_fase1="PROHIBIDA_FUGA") for i in range(35)] + _historiales_en_extracto(), schema=ESQUEMA_BASE)
    m = dc.aplicar_arbol(base, UMBRALES, conocimiento=_conocimiento())
    _escribir_matriz_y_resumen(ctx, m)
    md = _informe(ctx)
    assert "Se muestran las primeras 30 de 35 filas" in md and md.count("| FUGA_") == 30


def test_la_seccion_8_en_modo_completo_trae_la_tabla_de_los_historiales_con_su_factibilidad(ctx_completo):
    historiales = [
        _fila("N_EGRESOS_12M", incremento_auc=0.012, asociacion_hospital=0.05), _fila("DIAS_DESDE_EGRESO_PREVIO", incremento_auc=0.001, asociacion_hospital=0.20),
        _fila("HISTORIA_DISPONIBLE", incremento_auc=0.03, asociacion_hospital=0.02),
    ]
    m = dc.aplicar_arbol(pl.DataFrame(historiales, schema=ESQUEMA_BASE), UMBRALES, conocimiento=_conocimiento())
    _escribir_matriz_y_resumen(ctx_completo, m)
    ctx_completo.escribir_tabla(
        "t23_historial_factibilidad.csv",
        pl.DataFrame({
            "anio": [2019, 2023], "regimen": ["A", "B"], "n_episodios": [500, 520], "n_id_valido": [480, 515], "n_ventana_completa": [300, 505], "n_viable": [290, 500], "n_con_previo": [60, 90],
            "prop_id_valido": [0.96, 0.99], "prop_ventana_completa": [0.60, 0.97], "factibilidad": [0.58, 0.96], "prop_con_previo": [0.2, 0.18], "cumple_minimo": ["no", "si"],
        }),
        tarea="T09", descripcion="factibilidad de prueba", cols_n=["n_episodios", "n_id_valido", "n_ventana_completa", "n_viable", "n_con_previo"],
        dependientes=["prop_id_valido", "prop_ventana_completa", "factibilidad", "prop_con_previo", "cumple_minimo"],
    )
    md = _informe(ctx_completo)
    assert "Variables de historial: incremento, proxy y recomendación de la matriz" in md and "no es evaluable" not in md and "extracto" not in md.lower()
    assert "La factibilidad del historial en 2023 es 96,0 %" in md and "| N_EGRESOS_12M | 0,012 | 0,050 |" in md and "| DIAS_DESDE_EGRESO_PREVIO | 0,001 | 0,200 |" in md
    assert "el veredicto lo calcula la Tarea 12" in md and md.count("| HISTORIA_DISPONIBLE |") == 1


def test_en_modo_completo_sin_la_tabla_de_factibilidad_se_muestra_la_tabla_sin_esa_frase(ctx_completo):
    m = dc.aplicar_arbol(pl.DataFrame([_fila(n) for n in dc.HISTORIALES], schema=ESQUEMA_BASE), UMBRALES, conocimiento=_conocimiento())
    _escribir_matriz_y_resumen(ctx_completo, m)
    md = _informe(ctx_completo)
    assert "Variables de historial" in md and "La factibilidad del historial" not in md


def test_la_seccion_8_dice_cuando_una_cifra_de_la_matriz_publicada_se_anulo(ctx):
    historiales = dc.aplicar_arbol(pl.DataFrame(_historiales_en_extracto(), schema=ESQUEMA_BASE), UMBRALES, conocimiento=_conocimiento())
    m = pl.concat([_matriz_ejemplo(), historiales])
    p = dc.publicar_matriz(m, {**_fuentes(), "importancia": _anular(_fuentes()["importancia"], "incremento_auc", variable="V1")})
    _escribir_matriz_y_resumen(ctx, m, p)
    assert p["valor_suprimido"].to_list().count("si") == 1
    md = _informe(ctx)
    assert "Filas con alguna cifra anulada en la matriz publicada: 1." in md and "se calcularon con los valores reales" in md
    _escribir_matriz_y_resumen(ctx, m)                                                                      # la misma matriz sin ninguna cifra anulada
    assert "Filas con alguna cifra anulada" not in _informe(ctx, nombre="t23_prueba_decision_sin.md")


# FIN-test_t11_decision
