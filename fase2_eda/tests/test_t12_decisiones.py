"""Tarea 12: evidencia para las 17 decisiones y supuestos S01 a S15 (reglas con tablas hechas a mano, bordes, semáforo e integración sintética)."""

from __future__ import annotations

import dataclasses
import re

import polars as pl
import pytest

from ayudas_t12 import CIFRAS_BASE, FuenteEvidenciaFalsa, copiar_tablas, df, fuente_base, metrica, metricas_base, tablas_base
from eda import config, decisiones, ledger, privacidad, protocolo
from eda.contexto import Contexto
from eda.decisiones import POR_ID, Hipotesis, Resultado
from eda.informes import Informe


def ev(hid, fuente=None, previos=None):
    """Evalúa una hipótesis sobre la fuente base (o la dada)."""
    return decisiones.evaluar_hipotesis(fuente or fuente_base(), POR_ID[hid], previos)


def fuente(etiqueta="extracto", **tablas):
    """Fuente base con tablas reemplazadas: ``fuente(**{"t24_resumen.csv": df(...)})``."""
    return fuente_base(etiqueta, tablas=tablas)


def sin(f, *nombres):
    """La fuente sin las tablas dadas."""
    for n in nombres:
        f._tablas.pop(n, None)
    return f


class FuenteEspia(FuenteEvidenciaFalsa):
    """Registra las tablas que se piden (para probar que una hipótesis que requiere los datos completos no lee nada en un extracto)."""

    def __init__(self, etiqueta="extracto"):
        super().__init__(tablas_base(), metricas=metricas_base(etiqueta), etiqueta=etiqueta, cifras=CIFRAS_BASE)
        self.leidas = []

    def tabla(self, nombre, columnas_texto=()):
        self.leidas.append(nombre)
        return super().tabla(nombre, columnas_texto)


CLASES = {
    "H01a": "decisiva", "H01b": "decisiva", "H01c": "decisiva", "H02a": "decisiva", "H02b": "decisiva", "H02c": "decisiva", "H03": "decisiva", "H04": "informativa",
    "H05a": "informativa", "H05b": "informativa", "H06a": "informativa", "H06b": "informativa", "H07": "decisiva", "H08": "decisiva", "H09": "informativa", "H10": "informativa",
    "H11": "decisiva", "H12": "informativa", "H13": "decisiva", "H14": "decisiva", "H15a": "informativa", "H15b": "decisiva", "H16a": "informativa", "H16b": "informativa",
    "H17a": "decisiva", "H17b": "decisiva",
}
REQUIEREN_COMPLETO = {"H01b", "H01c", "H06a", "H06b", "H13", "H16a"}
PRIMARIAS = {
    1: ["H01a", "H01b", "H01c"], 2: ["H02a", "H02b", "H02c"], 3: ["H03"], 4: ["H04"], 5: ["H05a", "H05b"], 6: ["H06a", "H06b"], 7: ["H07"], 8: ["H08"], 9: ["H09"],
    10: ["H10"], 11: ["H11"], 12: ["H12"], 13: ["H13"], 14: ["H14"], 15: ["H15a", "H15b"], 16: ["H16a", "H16b"], 17: ["H17a", "H17b"],
}


# ================================================================================================================================ registro y protocolo
def test_el_registro_tiene_las_26_hipotesis_en_el_orden_de_hipotesis_csv():
    cfg, dec = config.cargar_hipotesis(), config.cargar_decisiones_fase1()
    assert [h.id for h in decisiones.HIPOTESIS] == [r["id"] for r in cfg.iter_rows(named=True) if r["id"].startswith("H")]
    assert len(decisiones.HIPOTESIS) == 26 and decisiones.validar_registro(cfg, dec) == []
    assert {h.id: h.clase for h in decisiones.HIPOTESIS} == CLASES
    assert {h.id for h in decisiones.HIPOTESIS if h.requiere_completo} == REQUIEREN_COMPLETO
    assert all(h.tablas for h in decisiones.HIPOTESIS) and all(callable(h.evaluador) for h in decisiones.HIPOTESIS)


def test_validar_registro_detecta_diferencias_con_el_protocolo():
    cfg, dec = config.cargar_hipotesis(), config.cargar_decisiones_fase1()
    sin_h07 = cfg.filter(pl.col("id") != "H07")
    assert any("H07" in p for p in decisiones.validar_registro(sin_h07, dec))
    cambiada = cfg.with_columns(pl.when(pl.col("id") == "H07").then(pl.lit("8")).otherwise(pl.col("decision")).alias("decision"))
    assert any("H07" in p and "decisión" in p for p in decisiones.validar_registro(cambiada, dec))
    sin_decision_17 = dec.filter(pl.col("n") != "17")
    assert any("decisión 17" in p for p in decisiones.validar_registro(cfg, sin_decision_17))


def test_el_mapa_de_decisiones_a_hipotesis_primarias_coincide_con_el_protocolo():
    cfg, dec = config.cargar_hipotesis(), config.cargar_decisiones_fase1()
    por_decision = {n: [h.id for h in decisiones.HIPOTESIS if h.decision == n] for n in range(1, 18)}
    assert por_decision == PRIMARIAS
    cobertura = {int(r["decision"]): set(r["hipotesis"].split("; ")) for r in protocolo.tabla_cobertura(cfg, dec).iter_rows(named=True)}
    for n, ids in PRIMARIAS.items():
        assert set(ids) <= cobertura[n], n                                                # las primarias están entre las hipótesis que cubren la decisión
        assert {r["id"] for r in cfg.iter_rows(named=True) if r["decision"].strip() == str(n)} == set(ids)


def test_tipo_de_evidencia_de_las_17_decisiones_sale_de_hipotesis_csv():
    cfg = config.cargar_hipotesis()
    esperado = {n: "cuantitativa" for n in range(1, 18)}
    esperado.update({3: "parcial", 9: "parcial", 4: "no depende de datos", 10: "institucional", 12: "institucional"})
    assert {n: decisiones.tipo_evidencia(n, cfg) for n in range(1, 18)} == esperado
    # las dos decisiones institucionales son las que dicen "Decisión institucional" en su criterio; la 4 es metodológica
    institucionales = {int(r["decision"]) for r in cfg.iter_rows(named=True) if r["id"].startswith("H") and r["criterio_decision"].startswith("Decisión institucional")}
    assert institucionales == set(decisiones.INSTITUCIONALES) == {10, 12}
    assert [r["depende_de_datos"] for r in cfg.iter_rows(named=True) if r["decision"] in ("4", "10", "12")] == ["no", "no", "no"]


def test_la_fuente_del_contexto_lee_tablas_umbrales_metricas_y_cifras(ctx):
    f = decisiones.FuenteContexto(ctx)
    assert f.etiqueta == "extracto" and f.umbral("proxy_hospital_alto") == 0.30 and list(f.umbral("anios_resultado")) == [2019, 2020, 2021, 2022, 2023]
    assert f.umbral("anio_expuesto") == 2024
    assert f.tabla("no_existe.csv") is None and f.metrica("no.existe") is None
    assert f.cifra_documentada("C077") == 65.0 and f.cifra_documentada("C078") == 68.0 and f.cifra_documentada("C999") is None
    ctx.escribir_tabla("prueba.csv", pl.DataFrame({"a": ["1", "2"], "b": [3, 4]}), tarea="T12", descripcion="prueba")
    assert f.tabla("prueba.csv", ["a"])["a"].to_list() == ["1", "2"]                          # las columnas de texto se leen como texto
    ctx.registrar_metrica("m.x", 5.0, tarea="T12", tabla="t.csv")
    assert decisiones.FuenteContexto(ctx).metrica("m.x")["valor"] == 5.0


def test_una_tabla_ausente_deja_la_hipotesis_no_evaluable_sin_excepcion():
    f = sin(fuente(), "t24_resumen.csv")
    r = ev("H01a", f)
    assert r.resultado == "no_evaluable" and "falta la tabla t24_resumen.csv" in r.motivo and r.tabla == "t24_resumen.csv" and r.cifras_clave == ""
    assert r.veredicto.startswith("no evaluable")


def test_una_columna_ausente_deja_la_hipotesis_no_evaluable_sin_excepcion():
    r = ev("H01a", fuente(**{"t24_resumen.csv": df({"otra": 1})}))
    assert r.resultado == "no_evaluable" and "columna" in r.motivo
    r = ev("H07", fuente(**{"t23_cola_estancia.csv": df({"base": "E", "regla": "p99", "cumple_def": "si", "cumple_conf": "si", "propuesta": "si"})}))   # falta corte_dias
    assert r.resultado == "no_evaluable"


def test_una_hipotesis_que_requiere_los_datos_completos_no_lee_nada_en_un_extracto():
    for hid in sorted(REQUIEREN_COMPLETO):
        f = FuenteEspia("extracto")
        r = decisiones.evaluar_hipotesis(f, POR_ID[hid])
        assert r.resultado == "no_evaluable" and r.motivo == decisiones.MOTIVO_REQUIERE_COMPLETO and f.leidas == [], hid
        assert r.cifras_clave == "" and r.tabla == ", ".join(POR_ID[hid].tablas)


def test_las_mismas_hipotesis_si_leen_sus_tablas_con_los_datos_completos():
    f = FuenteEspia("completo")
    r = decisiones.evaluar_hipotesis(f, POR_ID["H01b"])
    assert f.leidas == ["t23_historial_factibilidad.csv"] and r.resultado == "respalda"


# ================================================================================================================================= decisión 1
def test_h01a_sin_quiebres_respalda_y_con_quiebres_matiza():
    r0 = ev("H01a", fuente(**{"t24_resumen.csv": df({"n_quiebres": 0, "quiebres": "ninguno", "regimenes": "2019 a 2024"})}))
    assert r0.resultado == "respalda" and "= 0" in r0.cifras_clave and "(ninguno)" not in r0.cifras_clave and "2019 a 2024" in r0.cifras_clave
    r1 = ev("H01a")
    assert r1.resultado == "matiza" and "= 1 (2020 a 2021)" in r1.cifras_clave and "el historial no cruza el quiebre" in r1.veredicto and r1.tabla == "t24_resumen.csv"
    r2 = ev("H01a", fuente(**{"t24_resumen.csv": df({"n_quiebres": 3, "quiebres": "a; b; c", "regimenes": "x"})}))
    assert r2.resultado == "matiza" and "3 quiebre" in r2.veredicto


def test_h01a_con_numero_de_quiebres_nulo_no_es_evaluable():
    r = ev("H01a", fuente(**{"t24_resumen.csv": df({"n_quiebres": None, "quiebres": "x", "regimenes": "y"})}))
    assert r.resultado == "no_evaluable" and "n_quiebres" in r.motivo


def _historial(*filas):
    return df(*({"anio": a, "regimen": rg, "factibilidad": fa, "cumple_minimo": c} for a, rg, fa, c in filas))


def test_h01b_excluye_el_primer_anio_de_cada_espacio():
    # el 2019 y el 2021 son el primer año de su espacio ("no" por construcción) y no cuentan
    r = ev("H01b", fuente("completo"))
    assert r.resultado == "respalda" and "años evaluables = 3; cumplen el mínimo = 3" in r.cifras_clave and "2020: 97,1 %" in r.cifras_clave and "2019:" not in r.cifras_clave
    assert "mínimo = 80 %" in r.cifras_clave


@pytest.mark.parametrize(
    "filas, esperado",
    [
        ([(2019, "a", 0.0, "no"), (2020, "a", 0.9, "si"), (2021, "a", 0.5, "no")], "matiza"),
        ([(2019, "a", 0.0, "no"), (2020, "a", 0.5, "no"), (2021, "a", 0.4, "no")], "contradice"),
        ([(2019, "a", 0.9, "si"), (2020, "a", 0.5, "no")], "contradice"),                     # el "si" del primer año no cuenta
        ([(2019, "a", 0.0, "no"), (2020, "a", 0.9, "si")], "respalda"),
        ([(2019, "a", 0.0, "no"), (2020, "a", 0.9, "si"), (2022, "b", 0.0, "no"), (2023, "b", 0.4, "no")], "matiza"),
    ],
)
def test_h01b_agrega_los_anios_evaluables(filas, esperado):
    assert ev("H01b", fuente("completo", **{"t23_historial_factibilidad.csv": _historial(*filas)})).resultado == esperado


def test_h01b_sin_anios_evaluables_o_con_valor_invalido_no_es_evaluable():
    solo_primeros = _historial((2019, "a", 0.0, "no"), (2021, "b", 0.0, "no"))
    r = ev("H01b", fuente("completo", **{"t23_historial_factibilidad.csv": solo_primeros}))
    assert r.resultado == "no_evaluable" and "ventana" in r.motivo
    r = ev("H01b", fuente("completo", **{"t23_historial_factibilidad.csv": _historial((2019, "a", 0.0, "no"), (2020, "a", 0.9, "quizá"))}))
    assert r.resultado == "no_evaluable"


def _matriz(inc1, aso1, inc2, aso2, completa="si"):
    return df({"variable": "N_EGRESOS_12M", "medicion_completa": completa, "incremento_auc": inc1, "asociacion_hospital": aso1},
              {"variable": "DIAS_DESDE_EGRESO_PREVIO", "medicion_completa": completa, "incremento_auc": inc2, "asociacion_hospital": aso2})


H01B_RESPALDA = Resultado("respalda", "v", "t", "c")


def test_h01c_respalda_si_las_dos_variables_cumplen_y_la_factibilidad_se_cumple():
    r = ev("H01c", fuente("completo"), {"H01b": H01B_RESPALDA})
    assert r.resultado == "respalda" and "N_EGRESOS_12M: incremento de AUC = 0,020" in r.cifras_clave and "DIAS_DESDE_EGRESO_PREVIO" in r.cifras_clave
    assert "incremento mínimo = 0,005" in r.cifras_clave and "proxy moderado desde 0,15" in r.cifras_clave


@pytest.mark.parametrize(
    "inc1, aso1, inc2, aso2, esperado",
    [
        (0.0050001, 0.1499, 0.02, 0.05, "respalda"),
        (0.005, 0.05, 0.02, 0.05, "matiza"),                  # el incremento exactamente igual al mínimo no lo supera
        (0.02, 0.15, 0.02, 0.05, "matiza"),                   # la asociación exactamente igual al proxy moderado no es menor
        (0.005, 0.15, 0.005, 0.15, "contradice"),             # ninguna cumple ambas condiciones
        (0.02, 0.30, 0.001, 0.05, "contradice"),
    ],
)
def test_h01c_bordes_de_incremento_y_proxy(inc1, aso1, inc2, aso2, esperado):
    f = fuente("completo", **{"matriz_decision_variables.csv": _matriz(inc1, aso1, inc2, aso2)})
    assert ev("H01c", f, {"H01b": H01B_RESPALDA}).resultado == esperado


def test_h01c_depende_de_h01b_y_de_la_medicion_completa():
    f = fuente("completo")
    assert ev("H01c", f, {"H01b": Resultado("matiza", "v", "t", "c")}).resultado == "matiza"         # las dos cumplen pero la factibilidad solo matiza
    assert ev("H01c", f, {"H01b": Resultado("contradice", "v", "t", "c")}).resultado == "matiza"     # cumple alguna: no contradice del todo
    assert ev("H01c", f, {"H01b": Resultado("no_evaluable", "v", "t", "c")}).resultado == "no_evaluable"
    assert ev("H01c", f, {}).resultado == "no_evaluable"
    r = ev("H01c", fuente("completo", **{"matriz_decision_variables.csv": _matriz(0.02, 0.05, 0.02, 0.05, completa="no")}), {"H01b": H01B_RESPALDA})
    assert r.resultado == "no_evaluable" and "medición completa" in r.motivo
    r = ev("H01c", fuente("completo", **{"matriz_decision_variables.csv": _matriz(None, 0.05, 0.02, 0.05)}), {"H01b": H01B_RESPALDA})
    assert r.resultado == "no_evaluable"


# ================================================================================================================================= decisión 2
def _proxy(variable, asoc, **extra):
    base = {"tipo_fila": "variable", "variable": variable, "asociacion_hospital": asoc, "clase_proxy": "x", "n_niveles_totales": None, "n_niveles_en_un_hospital": None}
    return {**base, **extra}


def _fuga(variable="SERVICIOINGRESO", fuga="no"):
    return {"tipo_fila": "variable", "variable": variable, "supera_a_basales": "no", "supera_umbral_fuga": fuga, "prohibida_siempre": "no", "auc_ranking": 0.7, "auc_confirmacion": 0.7,
            "prop_igualdad": None, "ic_bajo": None, "ic_alto": None, "coincide_mayoria": None}


@pytest.mark.parametrize(
    "asoc, fuga, esperado",
    [
        (0.30, "no", "contradice"),             # igual al umbral alto
        (0.299999, "no", "matiza"),
        (0.15, "no", "matiza"),                 # igual al umbral moderado
        (0.149999, "no", "respalda"),
        (0.149999, "si", "contradice"),         # la fuga contradice aunque el proxy sea bajo
        (0.50, "si", "contradice"),
        (0.20, "si", "contradice"),
    ],
)
def test_h02a_bordes_de_proxy_y_fuga(asoc, fuga, esperado):
    f = fuente(**{"t23_proxy_hospital.csv": df(_proxy("SERVICIOINGRESO", asoc)), "t23_fuga.csv": df(_fuga(fuga=fuga))})
    r = ev("H02a", f)
    assert r.resultado == esperado
    assert f"asociación con el hospital = {decisiones._d(asoc, 3)}" in r.cifras_clave and ("supera el umbral de fuga de 0,85" if fuga == "si" else "no supera el umbral de fuga") in r.cifras_clave


def test_h02a_sin_datos_no_es_evaluable():
    assert "falta la tabla t23_fuga.csv" in ev("H02a", sin(fuente(), "t23_fuga.csv")).motivo
    assert ev("H02a", sin(fuente(), "t23_proxy_hospital.csv")).resultado == "no_evaluable"
    assert ev("H02a", fuente(**{"t23_proxy_hospital.csv": df(_proxy("SERVICIOINGRESO", None)), "t23_fuga.csv": df(_fuga())})).resultado == "no_evaluable"
    assert ev("H02a", fuente(**{"t23_proxy_hospital.csv": df(_proxy("OTRA", 0.5)), "t23_fuga.csv": df(_fuga())})).resultado == "no_evaluable"
    assert ev("H02a", fuente(**{"t23_proxy_hospital.csv": df(_proxy("SERVICIOINGRESO", 0.1)), "t23_fuga.csv": df(_fuga(fuga="quizá"))})).resultado == "no_evaluable"


@pytest.mark.parametrize("asoc, esperado", [(0.30, "contradice"), (0.2999, "matiza"), (0.15, "matiza"), (0.1499, "respalda"), (0.0, "respalda")])
def test_h02b_bordes_de_proxy(asoc, esperado):
    assert ev("H02b", fuente(**{"t23_proxy_hospital.csv": df(_proxy("ESPECIALIDAD_MEDICA", asoc))})).resultado == esperado


def test_h02b_informa_la_cardinalidad_pero_no_decide_con_ella():
    r = ev("H02b")
    assert "niveles en total = 75" in r.cifras_clave and "niveles usados por un solo hospital = 12" in r.cifras_clave and r.resultado == "matiza"
    sin_card = ev("H02b", fuente(**{"t23_proxy_hospital.csv": df(_proxy("ESPECIALIDAD_MEDICA", 0.10))}))
    assert sin_card.resultado == "respalda" and "niveles" not in sin_card.cifras_clave


def test_h02c_autorreferencia_sobre_el_maximo_matiza():
    r = ev("H02c")
    assert r.resultado == "matiza" and "30,6 %" in r.cifras_clave and "máximo aceptado 5 %" in r.cifras_clave and "TIPO_PROCEDENCIA" in r.veredicto
    ok = ev("H02c", fuente(**{"t23_resumen_hospitales.csv": df({"autorref_prop_exacta": 0.02, "autorref_supera_maximo": "no"})}))
    assert ok.resultado == "respalda" and "2,0 %" in ok.cifras_clave
    raro = ev("H02c", fuente(**{"t23_resumen_hospitales.csv": df({"autorref_prop_exacta": 0.02, "autorref_supera_maximo": "tal vez"})}))
    assert raro.resultado == "no_evaluable"


# ================================================================================================================================= decisión 3
def _equidad(**asoc):
    return df(*(_proxy(v, a) for v, a in asoc.items()))


def test_h03_alguna_alta_o_moderada_respalda_y_todas_bajas_matiza_sin_contradecir_nunca():
    bajas = ev("H03", fuente(**{"t23_proxy_hospital.csv": _equidad(PREVISION=0.09, NACIONALIDAD=0.07, ES_EXTRANJERO=0.05, PROVINCIA=0.10, COMUNA=0.149999)}))
    assert bajas.resultado == "matiza" and "la reserva se justifica por equidad" in bajas.veredicto
    moderada = ev("H03", fuente(**{"t23_proxy_hospital.csv": _equidad(PREVISION=0.09, COMUNA=0.15)}))
    assert moderada.resultado == "respalda"
    altas = ev("H03", fuente(**{"t23_proxy_hospital.csv": _equidad(PREVISION=0.9, NACIONALIDAD=0.8, ES_EXTRANJERO=0.9, PROVINCIA=0.7, COMUNA=0.65)}))
    assert altas.resultado == "respalda"
    assert "PROVINCIA 0,725 (alto)" in ev("H03").cifras_clave and "PREVISION 0,094 (bajo)" in ev("H03").cifras_clave


def test_h03_usa_las_variables_que_hay_y_sin_ninguna_no_es_evaluable():
    r = ev("H03", fuente(**{"t23_proxy_hospital.csv": _equidad(PREVISION=0.5, OTRA=0.9)}))
    assert r.resultado == "respalda" and "OTRA" not in r.cifras_clave and "COMUNA" not in r.cifras_clave
    sin_equidad = ev("H03", fuente(**{"t23_proxy_hospital.csv": _equidad(OTRA=0.9)}))
    assert sin_equidad.resultado == "no_evaluable" and "equidad" in sin_equidad.motivo
    nula = ev("H03", fuente(**{"t23_proxy_hospital.csv": df(_proxy("PREVISION", None))}))
    assert nula.resultado == "no_evaluable"


# ================================================================================================================================= decisión 4
def test_h04_nunca_es_evaluable_y_cita_la_calidad_de_la_etnia_si_la_tabla_existe():
    r = ev("H04")
    assert r.resultado == "no_evaluable" and r.motivo == "no depende de datos" and "categorías distintas de ETNIA en los archivos = 2" in r.cifras_clave
    sin_tabla = ev("H04", sin(fuente(), "t22_deriva_categorias.csv"))
    assert sin_tabla.resultado == "no_evaluable" and sin_tabla.cifras_clave == ""
    sin_etnia = ev("H04", fuente(**{"t22_deriva_categorias.csv": df({"columna": "PREVISION", "anio": 2019, "categoria": "X"})}))
    assert sin_etnia.cifras_clave == ""


# ================================================================================================================================= decisión 5
def _variantes(a, b, dif, *, todos_a=0.04, todos_b=0.07):
    """``t23_cascada_variantes.csv`` mínima: ``prevalencia`` es sobre todos los egresos y ``prevalencia_conocida`` sobre los de desenlace conocido (la base de la diferencia)."""
    return {"t23_cascada_variantes.csv": df({"periodo": "2019-2023", "variante": "A", "prevalencia": todos_a, "prevalencia_conocida": a, "dif_prevalencia_pp": None},
                                            {"periodo": "2019-2023", "variante": "B", "prevalencia": todos_b, "prevalencia_conocida": b, "dif_prevalencia_pp": dif})}


def test_h05a_informa_la_prevalencia_de_adultos_frente_a_la_cohorte_dura():
    r = ev("H05a")
    assert r.resultado == "matiza"
    assert r.cifras_clave == ("2019-2023: prevalencia entre los egresos con desenlace conocido de la cohorte dura (A) = 3,64 %; solo adultos (B) = 4,31 %; "
                              "diferencia = 0,67 puntos porcentuales (calculada con los valores sin redondear)")
    cambiada = ev("H05a", fuente(**_variantes(0.05, 0.06, 1.5)))
    assert "= 5,00 %" in cambiada.cifras_clave and "= 6,00 %" in cambiada.cifras_clave and "1,50 puntos" in cambiada.cifras_clave       # la evidencia sale de las celdas
    falta_b = ev("H05a", fuente(**{"t23_cascada_variantes.csv": df({"periodo": "2019-2023", "variante": "A", "prevalencia": 0.05, "prevalencia_conocida": 0.05,
                                                                   "dif_prevalencia_pp": None})}))
    assert falta_b.resultado == "no_evaluable"
    sin_base = ev("H05a", fuente(**{"t23_cascada_variantes.csv": df({"periodo": "2019-2023", "variante": "A", "prevalencia": 0.05, "dif_prevalencia_pp": None},
                                                                   {"periodo": "2019-2023", "variante": "B", "prevalencia": 0.06, "dif_prevalencia_pp": 1.0})}))
    assert sin_base.resultado == "no_evaluable" and "prevalencia_conocida" in sin_base.motivo


def test_h05a_cita_las_dos_prevalencias_y_la_diferencia_sobre_la_misma_base_y_las_tres_concilian():
    # la diferencia publicada es B menos A sobre los egresos de desenlace conocido: con la prevalencia sobre todos los egresos (4 % y 7 %) la resta sería 3 puntos y no concilia
    r = ev("H05a", fuente(**_variantes(0.0500, 0.0625, 1.25, todos_a=0.04, todos_b=0.07)))
    a, b = (float(x.replace(",", ".")) for x in re.findall(r"= (\d+,\d+) %", r.cifras_clave))
    dif = float(re.search(r"diferencia = (\d+,\d+) puntos", r.cifras_clave).group(1).replace(",", "."))
    assert (a, b, dif) == (5.0, 6.25, 1.25) and round(b - a, 2) == dif
    assert "4,00 %" not in r.cifras_clave and "7,00 %" not in r.cifras_clave
    # lo mismo con las cifras con que se corrió la Tarea 7 en el extracto (3,64 % y 4,31 %): la resta redondeada de los dos porcentajes citados concilia con la diferencia citada
    real = ev("H05a")
    pa, pb = (float(x.replace(",", ".")) for x in re.findall(r"= (\d+,\d+) %", real.cifras_clave))
    assert round(pb - pa, 2) == float(re.search(r"diferencia = (\d+,\d+) puntos", real.cifras_clave).group(1).replace(",", "."))


def test_h05b_usa_el_ultimo_anio_con_cohorte():
    r = ev("H05b")
    assert r.resultado == "matiza" and r.cifras_clave == "2023: hospitales pediátricos o monográficos = 3 de 68"                   # 2024 no tiene cohorte
    sin_cohorte = ev("H05b", fuente(**{"t23_hospitales_resumen.csv": df({"anio": 2024, "n_hospitales": 72, "n_pediatricos": 3, "n_hospitales_con_cohorte": None})}))
    assert sin_cohorte.resultado == "no_evaluable"


# ================================================================================================================================= decisión 6
def test_h06a_y_h06b_con_datos_completos_informan_por_anio():
    a = ev("H06a", fuente("completo"))
    assert a.resultado == "matiza" and a.cifras_clave == "hospitales que cumplen 500 egresos y 10 muertes esperadas: 2022: 22 de 64; 2023: 23 de 67"      # no mezcla la fila de 200 y 5
    b = ev("H06b", fuente("completo"))
    assert b.resultado == "matiza" and b.cifras_clave == "celdas de hospital bajo el mínimo publicable: 2022: 3,1 %; 2023: 3,0 %"
    fuera = ev("H06a", fuente("completo", **{"t23_volumen_umbrales.csv": df({"anio": 2023, "egresos_min": 200, "muertes_esperadas_min": 5, "n_hospitales": 1, "n_cumplen_ambos": 1})}))
    assert fuera.resultado == "no_evaluable" and "combinación" in fuera.motivo
    todo_nulo = ev("H06b", fuente("completo", **{"t23_hospitales_resumen.csv": df({"anio": 2024, "prop_celdas_bajo_minimo": None})}))
    assert todo_nulo.resultado == "no_evaluable"


# ================================================================================================================================= decisión 7
def _cola(*filas):
    return df(*({"base": b, "regla": rg, "corte_dias": c, "cumple_def": d, "cumple_conf": cf, "propuesta": p, "prop_afectados_def": 0.01} for b, rg, c, d, cf, p in filas))


@pytest.mark.parametrize(
    "filas, esperado",
    [
        ([("E", "p99", 51, "si", "si", "si")], "respalda"),
        ([("E", "p99", 51, "si", "no", "si")], "matiza"),
        ([("E", "p99", 51, "si", "no", "si"), ("D", "p99", 50, "si", "si", "si")], "respalda"),         # basta una con ambas
        ([("E", "p99", 51, "si", "si", "no"), ("E", "365 días", 365, "si", "si", "no")], "contradice"),  # ninguna fila es la propuesta
    ],
)
def test_h07_regla_de_corte_propuesta(filas, esperado):
    assert ev("H07", fuente(**{"t23_cola_estancia.csv": _cola(*filas)})).resultado == esperado


def test_h07_texto_de_evidencia_y_ninguna_propuesta():
    r = ev("H07")
    assert r.cifras_clave == "regla p99 (base E): corte en 51 días, definición si, confirmación si; pérdida máxima = 1,0 %"
    ninguna = ev("H07", fuente(**{"t23_cola_estancia.csv": _cola(("E", "p99", 51, "si", "si", "no"))}))
    assert ninguna.resultado == "contradice" and "reglas candidatas = 1" in ninguna.cifras_clave and "1,0 %" in ninguna.cifras_clave


# ================================================================================================================================= decisión 8
def _top5(n, repo=2):
    return df({"elix_top5": "ELIX_24; ELIX_04; ELIX_25; ELIX_30; ELIX_27", "elix_n_coinciden_fase1": n, "elix_n_coinciden_repo": repo, "h17a_n_clave": 47, "h17a_n_psi_alto": 4})


@pytest.mark.parametrize("n, esperado", [(5, "respalda"), (4, "matiza"), (3, "matiza"), (1, "matiza"), (0, "contradice")])
def test_h08_coincidencias_con_la_lista_de_la_fase_1(n, esperado):
    r = ev("H08", fuente(**{"t23_resumen_seleccion.csv": _top5(n)}))
    assert r.resultado == esperado and f"coinciden con la lista de la Fase 1 = {n}" in r.cifras_clave


def test_h08_cita_las_cinco_categorias_y_la_lista_del_repositorio():
    r = ev("H08")
    assert "ELIX_24; ELIX_04; ELIX_25; ELIX_30; ELIX_27" in r.cifras_clave and "con la del repositorio = 2" in r.cifras_clave and "revisión clínica" in r.veredicto
    assert ev("H08", fuente(**{"t23_resumen_seleccion.csv": _top5(3, repo=4)})).cifras_clave.endswith("con la del repositorio = 4")
    assert ev("H08", fuente(**{"t23_resumen_seleccion.csv": df({"elix_top5": None, "elix_n_coinciden_fase1": None, "elix_n_coinciden_repo": None})})).resultado == "no_evaluable"


# ================================================================================================================================= decisión 9
def test_h09_informa_el_indice_de_dispersion_del_periodo_agregado():
    r = ev("H09")
    assert r.resultado == "matiza" and r.cifras_clave == "2019-2023: índice de dispersión de la estadía de los sobrevivientes (varianza sobre media) = 21,07; media = 7,31 días"
    sin_media = ev("H09", fuente(**{"t23_estancia_dispersion.csv": df({"periodo": "2019-2023", "base": "E", "indice_dispersion": 3.0, "media": None})}))
    assert sin_media.resultado == "matiza" and "media = sin dato días" in sin_media.cifras_clave
    otra_base = ev("H09", fuente(**{"t23_estancia_dispersion.csv": df({"periodo": "2019-2023", "base": "D", "indice_dispersion": 3.0, "media": 1.0})}))
    assert otra_base.resultado == "no_evaluable"
    assert ev("H09", fuente(**{"t23_estancia_dispersion.csv": df({"periodo": "2019-2023", "base": "E", "indice_dispersion": None, "media": 1.0})})).resultado == "no_evaluable"


# ======================================================================================================================== decisiones 10 y 12
def test_h10_y_h12_son_institucionales_y_no_evaluables():
    for hid, tabla in (("H10", "figuras_indice.csv"), ("H12", "conocimiento_clinico.csv")):
        r = ev(hid, FuenteEvidenciaFalsa({}))
        assert r.resultado == "no_evaluable" and r.motivo == "no depende de datos" and r.tabla == tabla and r.cifras_clave == "" and "institucional" in r.veredicto


# ================================================================================================================================ decisión 11
def test_h11_respalda_con_rut_y_matiza_con_cero():
    assert ev("H11", fuente(), None).resultado == "matiza"
    f = fuente_base("extracto", metricas={"calidad.valores_rut.total": metrica(5)})
    r = ev("H11", f)
    assert r.resultado == "respalda" and r.cifras_clave == "valores con estructura de RUT = 5"
    cero = ev("H11")
    assert cero.cifras_clave == "valores con estructura de RUT = 0" and "no se encontraron" in cero.veredicto


def test_h11_un_valor_suprimido_cuenta_como_presente_y_no_como_cero():
    f = fuente_base("extracto", metricas={"calidad.valores_rut.total": metrica(None, nota="[suprimido: menos de diez casos]")})
    r = ev("H11", f)
    assert r.resultado == "respalda" and "entre 1 y 9 (suprimido)" in r.cifras_clave and "= 0" not in r.cifras_clave
    sin_nota = fuente_base("extracto", metricas={"calidad.valores_rut.total": metrica(None)})
    assert ev("H11", sin_nota).resultado == "no_evaluable"
    ausente = FuenteEvidenciaFalsa(tablas_base(), metricas={}, cifras=CIFRAS_BASE)
    assert ev("H11", ausente).resultado == "no_evaluable" and "calidad.valores_rut.total" in ev("H11", ausente).motivo


# ================================================================================================================================ decisión 13
def test_h13_respalda_o_contradice_segun_el_enlace_y_solo_con_datos_completos():
    asi = lambda v: fuente("completo", **{"t23_resumen_hospitales.csv": df({"autorref_prop_exacta": 0.3, "autorref_supera_maximo": "si", "enlace_anio": 2023, "enlace_ventana_dias": 30,
                                                                       "enlace_tasa": 0.45, "enlace_cumple_minimo": v})})
    r = ev("H13", asi("si"))
    assert r.resultado == "respalda" and r.cifras_clave == "tasa de enlace de los traslados salientes a 30 días en 2023 = 45,0 %; mínimo = 30 %"
    assert ev("H13", asi("no")).resultado == "contradice"
    assert ev("H13", fuente("extracto")).resultado == "no_evaluable"
    assert ev("H13", fuente("completo")).resultado == "contradice"                           # la fila base trae "no"


# ================================================================================================================================ decisión 14
def _fuga_df(basales, fuga, mortalidad=True):
    filas = [{"tipo_fila": "variable", "variable": "IR_29301_SEVERIDAD", "supera_a_basales": basales, "supera_umbral_fuga": fuga, "prohibida_siempre": "no", "auc_ranking": 0.842, "auc_confirmacion": 0.858}]
    if mortalidad:
        filas.append({"tipo_fila": "variable", "variable": "IR_29301_MORTALIDAD", "supera_a_basales": "si", "supera_umbral_fuga": "si", "prohibida_siempre": "si", "auc_ranking": 0.860, "auc_confirmacion": 0.871})
    return df(*filas)


@pytest.mark.parametrize("basales, fuga, esperado", [("si", "si", "respalda"), ("si", "no", "matiza"), ("no", "si", "matiza"), ("no", "no", "matiza")])
def test_h14_exige_las_dos_condiciones_a_la_vez(basales, fuga, esperado):
    assert ev("H14", fuente(**{"t23_fuga.csv": _fuga_df(basales, fuga)})).resultado == esperado


def test_h14_cita_los_auc_de_ranking_y_confirmacion_de_ambas_variables():
    r = ev("H14")
    assert "IR_29301_SEVERIDAD: AUC de ranking = 0,842 y de confirmación = 0,858" in r.cifras_clave and "supera el umbral de fuga de 0,85" in r.cifras_clave
    assert "IR_29301_MORTALIDAD (prohibida siempre por R6): AUC de ranking = 0,860 y de confirmación = 0,871" in r.cifras_clave
    assert ev("H14", fuente(**{"t23_fuga.csv": _fuga_df("si", "si", mortalidad=False)})).resultado == "no_evaluable"


def _fuga_ventanas(ranking, confirmacion, fuga="si"):
    return df({"tipo_fila": "variable", "variable": "IR_29301_SEVERIDAD", "supera_a_basales": "si", "supera_umbral_fuga": fuga, "prohibida_siempre": "no", "auc_ranking": ranking,
               "auc_confirmacion": confirmacion},
              {"tipo_fila": "variable", "variable": "IR_29301_MORTALIDAD", "supera_a_basales": "si", "supera_umbral_fuga": "si", "prohibida_siempre": "si", "auc_ranking": 0.86,
               "auc_confirmacion": 0.87})


@pytest.mark.parametrize("ranking, confirmacion, fuga, donde", [
    (0.842, 0.858, "si", " en la confirmación de 2023;"),                                  # los AUC de la corrida real: solo la confirmación alcanza el umbral
    (0.860, 0.840, "si", " en el ranking de 2019 a 2022;"),
    (0.860, 0.870, "si", " en el ranking de 2019 a 2022 y la confirmación de 2023;"),
    (0.850, 0.840, "si", " en el ranking de 2019 a 2022;"),                                # el umbral se alcanza con igualdad
    (0.842, 0.858, "no", "el umbral de fuga de 0,85;"),                                     # si la marca de la tabla dice que no supera, no se nombra ninguna ventana
])
def test_h14_dice_en_que_ventana_se_supera_el_umbral_de_fuga(ranking, confirmacion, fuga, donde):
    r = ev("H14", fuente(**{"t23_fuga.csv": _fuga_ventanas(ranking, confirmacion, fuga)}))
    if fuga == "si":
        assert f"supera el umbral de fuga de 0,85{donde}" in r.cifras_clave
    else:
        assert f"no supera {donde}" in r.cifras_clave and " en la confirmación" not in r.cifras_clave.split("IR_29301_MORTALIDAD")[0]


def test_h14_con_la_marca_de_fuga_pero_sin_los_auc_no_nombra_una_ventana_inventada():
    r = ev("H14", fuente(**{"t23_fuga.csv": _fuga_ventanas(None, None, "si")}))
    assert "supera el umbral de fuga de 0,85;" in r.cifras_clave and "sin dato" in r.cifras_clave and r.resultado == "respalda"


# ================================================================================================================================ decisión 15
def test_h15a_describe_la_composicion_y_omite_lo_suprimido():
    r = ev("H15a")
    assert r.resultado == "matiza" and r.cifras_clave == "estadías de cero días = 43.354; de actividad ambulatoria = 75,7 %; con fallecimiento = 1,3 %; censuradas = 2,7 %"
    parcial = ev("H15a", fuente(**{"t23_resumen.csv": df({"n_cero_total": 500, "prop_cero_ambulatoria": None, "prop_cero_fallecidos": None, "prop_cero_censurados": 0.1, "intento_2024": "rechazado"})}))
    assert parcial.cifras_clave == "estadías de cero días = 500; censuradas = 10,0 %"
    assert ev("H15a", fuente(**{"t23_resumen.csv": df({"n_cero_total": None, "intento_2024": "rechazado"})})).resultado == "no_evaluable"


def _estratos(amb, cero):
    return df(
        {"periodo": "2019-2023", "variable": "actividad_grupo", "categoria": "ambulatoria", "razon_vs_resto": 0.109, "bajo_umbral": amb},
        {"periodo": "2019-2023", "variable": "estancia_cero", "categoria": "0 días", "razon_vs_resto": 0.32, "bajo_umbral": cero},
        {"periodo": "2019", "variable": "estancia_cero", "categoria": "0 días", "razon_vs_resto": 0.9, "bajo_umbral": "no"},
    )


@pytest.mark.parametrize("amb, cero, esperado", [("no", "no", "respalda"), ("si", "no", "matiza"), ("no", "si", "matiza"), ("si", "si", "matiza")])
def test_h15b_algun_estrato_bajo_el_umbral_matiza(amb, cero, esperado):
    r = ev("H15b", fuente(**{"t23_resultado_estratos.csv": _estratos(amb, cero)}))
    assert r.resultado == esperado


def test_h15b_cita_las_razones_del_periodo_agregado_y_el_umbral():
    r = ev("H15b")
    assert r.cifras_clave == "2019-2023: mortalidad del estrato ambulatorio = 0,11 veces la del resto; del estrato de estadía cero = 0,32 veces; umbral = 0,25 veces"
    assert ev("H15b", fuente(**{"t23_resultado_estratos.csv": df({"periodo": "2019-2023", "variable": "actividad_grupo", "categoria": "ambulatoria", "razon_vs_resto": None,
                                                                 "bajo_umbral": None})})).resultado == "no_evaluable"


# ================================================================================================================================ decisión 16
def test_h16a_con_datos_completos_informa_el_enlace_de_entrantes_a_la_ventana_mas_larga():
    r = ev("H16a", fuente("completo"))
    assert r.resultado == "matiza" and r.cifras_clave == "2019-2023: entrantes enlazados con un traslado saliente previo a 30 días = 0,7 % de 14.939 elegibles"
    assert ev("H16a", fuente("extracto")).resultado == "no_evaluable"
    sin_entrante = ev("H16a", fuente("completo", **{"t23_enlaces_traslado.csv": df({"periodo": "2019-2023", "direccion": "saliente", "ventana_dias": 30, "n_elegibles": 5, "tasa_enlace": 0.1})}))
    assert sin_entrante.resultado == "no_evaluable"


def test_h16b_informa_la_mortalidad_de_entrantes_con_su_intervalo():
    r = ev("H16b")
    assert r.resultado == "matiza"
    assert r.cifras_clave == "2019-2023: mortalidad de los entrantes = 7,53 %; de los demás ingresos = 4,06 %; razón = 1,86 (intervalo de 1,74 a 1,98)"
    sin_rr = ev("H16b", fuente(**{"t23_resultado_entrantes.csv": df({"periodo": "2019-2023", "entrante": "si", "mortalidad": 0.07, "rr": None, "rr_bajo": None, "rr_alto": None},
                                                                   {"periodo": "2019-2023", "entrante": "no", "mortalidad": 0.04, "rr": None, "rr_bajo": None, "rr_alto": None})}))
    assert sin_rr.resultado == "no_evaluable"


# ================================================================================================================================ decisión 17
def _resumen_17(n_alto, n_clave=46):
    return df({"elix_top5": "x", "elix_n_coinciden_fase1": 3, "elix_n_coinciden_repo": 2, "h17a_n_clave": n_clave, "h17a_n_psi_alto": n_alto})


H01A_RESPALDA = Resultado("respalda", "v", "t", "c")
H01A_MATIZA = Resultado("matiza", "v", "t", "c")


@pytest.mark.parametrize("n_alto, previos, esperado", [
    (23, {"H01a": H01A_RESPALDA}, "respalda"),                # la mitad exacta (23 de 46) no es mayoría
    (24, {"H01a": H01A_RESPALDA}, "matiza"),                  # 24 de 46 sí
    (0, {"H01a": H01A_MATIZA}, "matiza"),                     # un quiebre del identificador basta
    (0, {"H01a": H01A_RESPALDA}, "respalda"),
    (0, {}, "respalda"),
    (0, {"H01a": Resultado("no_evaluable", "v", "t", "c")}, "respalda"),
])
def test_h17a_mayoria_de_variables_con_psi_alto_o_quiebre_del_identificador(n_alto, previos, esperado):
    assert ev("H17a", fuente(**{"t23_resumen_seleccion.csv": _resumen_17(n_alto)}), previos).resultado == esperado


def test_h17a_texto_y_casos_sin_dato():
    r = ev("H17a", None, {"H01a": H01A_MATIZA})
    assert r.cifras_clave == "variables clave con PSI alto contra 2023 = 4 de 47; quiebre del identificador = sí; PSI alto desde 0,25" and "2020 a 2022" in r.veredicto
    assert "quiebre del identificador = no" in ev("H17a", None, {"H01a": H01A_RESPALDA}).cifras_clave
    assert ev("H17a", fuente(**{"t23_resumen_seleccion.csv": _resumen_17(0, n_clave=0)}), {}).resultado == "no_evaluable"
    assert ev("H17a", fuente(**{"t23_resumen_seleccion.csv": _resumen_17(None)}), {}).resultado == "no_evaluable"


def test_h17b_cualquier_quiebre_mensual_matiza_y_ninguno_respalda():
    def resumen(alza, baja):
        return df({"n_cero_total": 1, "n_meses_evaluados": 48, "n_meses_alza": alza, "n_meses_baja": baja, "intento_2024": "rechazado"})

    r = ev("H17b")
    assert r.resultado == "matiza" and r.cifras_clave == "meses con quiebre al alza = 11; a la baja = 11 de 48 meses evaluados"
    assert ev("H17b", fuente(**{"t23_resumen.csv": resumen(0, 0)})).resultado == "respalda"
    assert ev("H17b", fuente(**{"t23_resumen.csv": resumen(1, 0)})).resultado == "matiza"
    assert ev("H17b", fuente(**{"t23_resumen.csv": resumen(0, 2)})).resultado == "matiza"


def test_h17b_sin_resumen_cuenta_los_quiebres_de_la_serie_mensual():
    sin_resumen = sin(fuente(), "t23_resumen.csv")
    r = ev("H17b", sin_resumen)
    assert r.resultado == "matiza" and r.cifras_clave == "meses con quiebre al alza = 1; a la baja = 1"                       # 'no' y vacío no cuentan
    ninguno = fuente(**{"t23_serie_mensual.csv": df({"mes_id": "2020-01", "quiebre": "no"}, {"mes_id": "2020-02", "quiebre": None})})
    assert ev("H17b", sin(ninguno, "t23_resumen.csv")).resultado == "respalda"
    assert ev("H17b", sin(fuente(), "t23_resumen.csv", "t23_serie_mensual.csv")).resultado == "no_evaluable"


# ================================================================================================================================ cascada (S10 y DT05)
PASOS = ("TOTAL", "EX01", "EX02", "EX03", "A", "EX04", "B", "EX05", "C", "EX06", "D", "EX07", "EX08", "E")
N_2019 = (49999, 17, 1142, 3809, 45031, 8039, 36992, 8953, 28039, 1351, 26688, 1059, 974, 24655)


def _cascada(periodo="2019", valores=N_2019, **cambios):
    return df(*({"periodo": periodo, "paso": p, "n": cambios.get(p, n)} for p, n in zip(PASOS, valores)))


def test_la_cascada_de_cuentas_a_mano_cuadra():
    # 49.999 - 17 - 1.142 - 3.809 = 45.031; 45.031 - 8.039 = 36.992; 36.992 - 8.953 = 28.039; 28.039 - 1.351 = 26.688; 26.688 - 1.059 - 974 = 24.655
    assert decisiones.revisar_cascada(_cascada()) == ([], 5)
    assert decisiones.problemas_cascada(_cascada()) == []
    assert decisiones.revisar_cascada(df({"periodo": "x", "paso": "TOTAL", "n": 1}).head(0)) == ([], 0)


@pytest.mark.parametrize("paso, idx, n_problemas", [("A", 4, 2), ("B", 6, 2), ("C", 8, 2), ("D", 10, 2), ("E", 13, 1)])
def test_cada_identidad_de_la_cascada_detecta_un_paso_desplazado_en_uno(paso, idx, n_problemas):
    # un paso desplazado rompe su propia identidad y la del paso siguiente, que lo usa como base (el último paso, E, solo la suya)
    problemas, verificadas = decisiones.revisar_cascada(_cascada(**{paso: N_2019[idx] + 1}))
    assert verificadas == 5 and len(problemas) == n_problemas and problemas[0].startswith(f"2019: {paso} es ")


def test_la_cascada_detecta_una_exclusion_distinta_y_cita_el_esperado():
    problemas = decisiones.problemas_cascada(_cascada(EX02=1143))
    assert problemas == ["2019: A es 45.031 y TOTAL menos EX01 y EX02 y EX03 da 45.030"]
    assert len(decisiones.problemas_cascada(_cascada(EX07=1060))) == 1


def test_la_cascada_no_falla_por_celdas_suprimidas_pero_si_las_cuenta_como_no_verificadas():
    suprimida = _cascada()
    suprimida = suprimida.with_columns(pl.when(pl.col("paso") == "EX03").then(None).otherwise(pl.col("n")).alias("n"))
    assert decisiones.revisar_cascada(suprimida) == ([], 4)                                   # la identidad de A no se puede verificar
    todo = _cascada().with_columns(pl.lit(None, dtype=pl.Int64).alias("n"))
    assert decisiones.revisar_cascada(todo) == ([], 0)


def test_la_cascada_se_revisa_en_cada_periodo():
    dos = pl.concat([_cascada("2019"), _cascada("2020", EX01=18)])
    problemas, verificadas = decisiones.revisar_cascada(dos)
    assert verificadas == 10 and len(problemas) == 1 and problemas[0].startswith("2020: A")


# ================================================================================================================================== semáforo
def _h(id_, clase="decisiva", completo=False):
    return Hipotesis(id_, 1, clase, completo, lambda f, p: None, ())


def _res(r):
    return Resultado(r, "v", "t", "c")


SEMAFORO_DOS_DECISIVAS = {
    ("respalda", "respalda"): "verde", ("respalda", "matiza"): "amarillo", ("respalda", "contradice"): "rojo", ("respalda", "no_evaluable"): "amarillo",
    ("matiza", "respalda"): "amarillo", ("matiza", "matiza"): "amarillo", ("matiza", "contradice"): "rojo", ("matiza", "no_evaluable"): "amarillo",
    ("contradice", "respalda"): "rojo", ("contradice", "matiza"): "rojo", ("contradice", "contradice"): "rojo", ("contradice", "no_evaluable"): "rojo",
    ("no_evaluable", "respalda"): "amarillo", ("no_evaluable", "matiza"): "amarillo", ("no_evaluable", "contradice"): "rojo", ("no_evaluable", "no_evaluable"): "gris",
}


@pytest.mark.parametrize("combinacion, color", sorted(SEMAFORO_DOS_DECISIVAS.items()))
def test_semaforo_de_dos_hipotesis_decisivas(combinacion, color):
    pares = [(_h("Ha"), _res(combinacion[0])), (_h("Hb"), _res(combinacion[1]))]
    assert decisiones.agregar_semaforo(pares, "cuantitativa") == color


def test_semaforo_gris_si_la_decision_es_institucional_o_no_depende_de_datos():
    pares = [(_h("Ha"), _res("contradice")), (_h("Hb"), _res("respalda"))]
    assert decisiones.agregar_semaforo(pares, "institucional") == "gris" and decisiones.agregar_semaforo(pares, "no depende de datos") == "gris"


def test_semaforo_gris_si_una_hipotesis_que_requiere_los_datos_completos_no_es_evaluable_aunque_otra_contradiga():
    pares = [(_h("Ha"), _res("contradice")), (_h("Hb", completo=True), _res("no_evaluable"))]
    assert decisiones.agregar_semaforo(pares, "cuantitativa") == "gris"                       # el gris va antes que el rojo
    ok = [(_h("Ha"), _res("contradice")), (_h("Hb", completo=True), _res("respalda"))]
    assert decisiones.agregar_semaforo(ok, "cuantitativa") == "rojo"
    solo_completo_noev = [(_h("Hb", completo=True), _res("no_evaluable")), (_h("Hc", "informativa", completo=True), _res("matiza"))]
    assert decisiones.agregar_semaforo(solo_completo_noev, "cuantitativa") == "gris"


def test_semaforo_con_informativas():
    assert decisiones.agregar_semaforo([(_h("Ha", "informativa"), _res("matiza")), (_h("Hb", "informativa"), _res("no_evaluable"))], "cuantitativa") == "amarillo"
    assert decisiones.agregar_semaforo([(_h("Ha", "informativa"), _res("no_evaluable"))], "parcial") == "gris"
    # una decisiva que respalda y una informativa sin evaluar siguen en verde; la informativa no frena el verde
    assert decisiones.agregar_semaforo([(_h("Ha"), _res("respalda")), (_h("Hb", "informativa"), _res("no_evaluable"))], "cuantitativa") == "verde"
    # una informativa evaluable no cambia el rojo de una decisiva
    assert decisiones.agregar_semaforo([(_h("Ha"), _res("contradice")), (_h("Hb", "informativa"), _res("matiza"))], "cuantitativa") == "rojo"
    assert decisiones.agregar_semaforo([], "cuantitativa") == "gris"


# =================================================================================================================================== las 17 filas
def _construir(f, supuestos=True):
    cfg, dec = config.cargar_hipotesis(), config.cargar_decisiones_fase1()
    res = decisiones.evaluar_todas(f)
    sup = decisiones.construir_supuestos(f, cfg) if supuestos else None
    return decisiones.construir_decisiones(f.etiqueta, cfg, dec, res, sup), res


def test_las_17_decisiones_con_sus_columnas_y_el_estado_pendiente():
    d, _ = _construir(fuente_base())
    assert d.height == 17 and d["n"].to_list() == list(range(1, 18)) and d.columns == list(decisiones.ESQUEMA_DECISIONES)
    assert set(d["estado"].to_list()) == {"pendiente de decisión humana"}
    fase1 = {int(r["n"]): r for r in config.cargar_decisiones_fase1().iter_rows(named=True)}
    for r in d.iter_rows(named=True):
        assert r["tema"] == decisiones.normalizar_texto_fase1(fase1[r["n"]]["tema"])
        assert r["propuesta_fase1"] == decisiones.normalizar_texto_fase1(fase1[r["n"]]["propuesta"])
    cfg, dec = config.cargar_hipotesis(), config.cargar_decisiones_fase1()
    cobertura = {int(r["decision"]): r["hipotesis"] for r in protocolo.tabla_cobertura(cfg, dec).iter_rows(named=True)}
    assert dict(zip(d["n"], d["hipotesis"])) == cobertura


def test_el_texto_de_las_decisiones_de_la_fase_1_se_publica_con_tildes_y_con_el_decimal_con_coma():
    # decisiones_pendientes.csv (docs/, solo lectura) viene sin tildes y escribe 0;95 porque la coma separa las columnas
    n = decisiones.normalizar_texto_fase1
    assert n("Rango entre 0;95 y 1;05") == "Rango entre 0,95 y 1,05"
    assert n("Servicio de ingreso; especialidad medica; paciente derivado") == "Servicio de ingreso; especialidad médica; paciente derivado"          # el ';' que separa no cambia
    assert n("Estadia de 0 dias y actividad ambulatoria") == "Estadía de 0 días y actividad ambulatoria" and n("Que traslados entrantes se excluyen") == "Qué traslados entrantes se excluyen"
    assert n("Rastrear con el identificador encriptado; atribuir al hospital que traslado") == "Rastrear con el identificador encriptado; atribuir al hospital que trasladó"
    assert n("Excluir solo los que se enlazan con un traslado saliente") == "Excluir solo los que se enlazan con un traslado saliente"              # 'traslado' sustantivo no cambia
    textos = [n(t) for t in ("Meta del O/E de estadia", "Definir quien revisa las decisiones del dominio", "Definir como informar los valores con estructura de RUT")]
    assert textos == ["Meta del O/E de estadía", "Definir quién revisa las decisiones del dominio", "Definir cómo informar los valores con estructura de RUT"]
    assert n(n("Usar hospitalizaciones de los ultimos 12 meses y dias desde el egreso previo")) == "Usar hospitalizaciones de los últimos 12 meses y días desde el egreso previo"
    d, _ = _construir(fuente_base())                                                                                                                  # idempotente y completo en las 17
    publicado = " ".join(d["tema"].to_list() + d["propuesta_fase1"].to_list())
    sin_tilde = r"\b(ultimos|dias|medica|Analisis|analisis|prevision|socioeconomico|Poblacion|anos|mas|minimo|pequenas|Estadias|Estadia|estadia|categorias|academico|Revision|clinica|quien)\b"
    assert re.search(sin_tilde, publicado) is None and re.search(r"\d;\d", publicado) is None and decisiones.normalizar_texto_fase1(publicado) == publicado


def test_semaforo_esperado_de_las_17_decisiones_en_un_extracto():
    d, _ = _construir(fuente_base("extracto"))
    sem = dict(zip(d["n"], d["semaforo"]))
    assert {n for n, s in sem.items() if s == "gris"} == {1, 4, 6, 10, 12, 13, 16}
    assert {n for n, s in sem.items() if s == "verde"} == {3, 7, 14}
    assert {n for n, s in sem.items() if s == "amarillo"} == {2, 5, 8, 9, 11, 15, 17} and "rojo" not in sem.values()
    assert dict(zip(d["n"], d["tipo_evidencia"]))[3] == "parcial" and dict(zip(d["n"], d["tipo_evidencia"]))[10] == "institucional"


def test_semaforo_esperado_de_las_17_decisiones_con_datos_completos():
    f = fuente_base("completo", tablas={"t23_resumen_hospitales.csv": df({"autorref_prop_exacta": 0.3, "autorref_supera_maximo": "si", "enlace_anio": 2023, "enlace_ventana_dias": 30,
                                                                         "enlace_tasa": 0.45, "enlace_cumple_minimo": "si"})})
    d, _ = _construir(f)
    sem = dict(zip(d["n"], d["semaforo"]))
    assert {n for n, s in sem.items() if s == "gris"} == {4, 10, 12}
    assert {n for n, s in sem.items() if s == "verde"} == {3, 7, 13, 14}
    assert {n for n, s in sem.items() if s == "amarillo"} == {1, 2, 5, 6, 8, 9, 11, 15, 16, 17}


def test_un_contradicho_pone_la_decision_en_rojo():
    f = fuente_base("extracto", tablas={"t23_proxy_hospital.csv": df(_proxy("SERVICIOINGRESO", 0.5), _proxy("ESPECIALIDAD_MEDICA", 0.1))})
    d, _ = _construir(f)
    assert dict(zip(d["n"], d["semaforo"]))[2] == "rojo"


def test_la_evidencia_se_lee_de_las_celdas_y_cambia_con_ellas():
    d, res = _construir(fuente_base())
    ev8 = d.filter(pl.col("n") == 8)["evidencia"][0]
    assert ev8.startswith("H08: cinco categorías con mayor índice de complicación = ELIX_24; ELIX_04; ELIX_25; ELIX_30; ELIX_27; coinciden con la lista de la Fase 1 = 3")
    assert "S02 RECHAZADO: categorías cuya prevalencia con tres caracteres difiere más del umbral = 11 de 31" in ev8 and "S03 RECHAZADO" in ev8
    otro, _ = _construir(fuente_base(tablas={"t23_resumen_seleccion.csv": _top5(4, repo=1)}))
    ev8b = otro.filter(pl.col("n") == 8)["evidencia"][0]
    assert "coinciden con la lista de la Fase 1 = 4; con la del repositorio = 1" in ev8b and ev8b != ev8


def test_la_evidencia_incluye_las_hipotesis_y_supuestos_que_afectan_a_la_decision():
    d, _ = _construir(fuente_base())
    ev13 = d.filter(pl.col("n") == 13)["evidencia"][0]
    assert ev13.startswith("H13: ") is False                                                   # H13 no es evaluable en un extracto: no aporta cifras
    assert "H01a (también afecta a esta decisión): quiebres del identificador" in ev13 and "S01 RECHAZADO: IDs con fecha de nacimiento inconsistente = 1,82 %" in ev13
    ev6 = d.filter(pl.col("n") == 6)["evidencia"][0]
    assert "H05b (también afecta a esta decisión): 2023: hospitales pediátricos" in ev6 and "S08 RECHAZADO" in ev6
    sin_sup, _ = _construir(fuente_base(), supuestos=False)
    assert "S01" not in sin_sup.filter(pl.col("n") == 13)["evidencia"][0]


def test_la_evidencia_de_la_decision_2_cita_las_dos_proporciones_de_concordancia_y_el_estado_de_s09():
    d, _ = _construir(fuente_base())
    ev2 = d.filter(pl.col("n") == 2)["evidencia"][0]
    assert ("S09 ACEPTADO: el servicio de ingreso coincide con el de alta en 99,2 % de los egresos sin traslado interno (intervalo de 99,1 % a 99,3 %) "
            "y en 15,2 % de los egresos con traslado interno (intervalo de 14,8 % a 15,7 %)") in ev2
    # sin la fila con traslado la evidencia cita solo la fila sin traslado y el supuesto no se evalúa
    fuga = tablas_base()["t23_fuga.csv"].filter(pl.col("tipo_fila") != "concordancia_con_traslado")
    sin_fila, _ = _construir(fuente_base(tablas={"t23_fuga.csv": fuga}))
    ev2s = sin_fila.filter(pl.col("n") == 2)["evidencia"][0]
    assert "S09 NO EVALUABLE: el servicio de ingreso coincide con el de alta en 99,2 % de los egresos sin traslado interno" in ev2s and "con traslado interno" not in ev2s
    # con la igualdad también mayoritaria con traslado el supuesto se rechaza y la evidencia lo dice con las dos cifras
    a_favor = _fuga9(con_traslado=(1000, 800, 0.77, 0.83))
    rechazado, _ = _construir(fuente_base(tablas={"t23_fuga.csv": pl.concat([tablas_base()["t23_fuga.csv"].filter(pl.col("tipo_fila") == "variable"), a_favor], how="diagonal")}))
    assert "S09 RECHAZADO: el servicio de ingreso coincide con el de alta en 99,2 % de los egresos sin traslado interno" in rechazado.filter(pl.col("n") == 2)["evidencia"][0]


def test_la_evidencia_de_la_decision_5_cita_la_prevalencia_sobre_la_misma_base_que_la_diferencia():
    d, _ = _construir(fuente_base())
    ev5 = d.filter(pl.col("n") == 5)["evidencia"][0]
    assert ("H05a: 2019-2023: prevalencia entre los egresos con desenlace conocido de la cohorte dura (A) = 3,64 %; solo adultos (B) = 4,31 %; "
            "diferencia = 0,67 puntos porcentuales (calculada con los valores sin redondear)") in ev5
    assert "3,41 %" not in ev5 and "4,02 %" not in ev5                  # las prevalencias sobre todos los egresos no se mezclan con la diferencia de la otra base


def test_las_decisiones_que_no_dependen_de_datos_dicen_que_no_hay_cifras():
    d, _ = _construir(fuente_base())
    for n in (4, 10, 12):
        fila = d.filter(pl.col("n") == n).row(0, named=True)
        assert fila["semaforo"] == "gris"
    assert d.filter(pl.col("n") == 10)["evidencia"][0] == "sin cifras: la decisión no depende de datos"
    assert d.filter(pl.col("n") == 4)["evidencia"][0].startswith("H04: categorías distintas de ETNIA")


def test_el_veredicto_resume_los_resultados_y_da_una_frase_por_hipotesis():
    d, _ = _construir(fuente_base())
    v1 = d.filter(pl.col("n") == 1)["veredicto"][0]
    assert v1.startswith("0 respalda, 1 matiza, 0 contradice, 2 no evaluable | H01a: el identificador tiene 1 quiebre")
    assert "H01b: no evaluable en un extracto" in v1 and "H01c: no evaluable en un extracto" in v1
    v5 = d.filter(pl.col("n") == 5)["veredicto"][0]
    assert v5.startswith("0 respalda, 2 matiza, 0 contradice, 0 no evaluable") and "H05a (solo informa):" in v5 and "H05b (solo informa):" in v5
    v2 = d.filter(pl.col("n") == 2)["veredicto"][0]
    assert "H02c: HOSPPROCEDENCIA no define el traslado entrante" in v2


def test_que_falta_con_datos_completos_junta_frases_fijas_y_cierra_con_el_aviso_del_extracto():
    d, _ = _construir(fuente_base("extracto"))
    f = lambda n: d.filter(pl.col("n") == n)["que_falta_con_datos_completos"][0]  # noqa: E731
    cierre = "las cifras del extracto son indicativas: repetir con los datos completos antes de decidir"
    assert f(1) == ("la factibilidad del historial de doce meses con el identificador y la ventana completa de cada egreso; "
                    "la señal, el valor incremental y el proxy del hospital de las variables de historial, que dependen del ingreso previo del paciente; " + cierre)
    assert f(3) == cierre and f(4).startswith("decisión metodológica de la Fase 1") and f(4).endswith(cierre)
    assert f(10).startswith("decisión institucional sobre si el prototipo muestra los nombres")
    completo, _ = _construir(fuente_base("completo"))
    g = lambda n: completo.filter(pl.col("n") == n)["que_falta_con_datos_completos"][0]  # noqa: E731
    assert g(3) == "nada pendiente de datos; falta la decisión humana" and g(1) == "nada pendiente de datos; falta la decisión humana" and "cifras del extracto" not in g(12)
    # las frases fijas no llevan cifras de resultados: solo referencias a las fases y a los años de los datos
    for k, frase in decisiones.FALTA_CON_DATOS_COMPLETOS.items():
        assert not re.search(r"\d", re.sub(r"\bFase \d\b|\b20(?:19|2[0-4])\b", "", frase)), k


def test_la_frase_de_lo_que_falta_cubre_todas_las_hipotesis():
    assert set(decisiones.FALTA_CON_DATOS_COMPLETOS) == set(CLASES)


def test_las_tablas_de_evidencia_son_las_que_realmente_se_leyeron():
    d, res = _construir(fuente_base())
    t2 = d.filter(pl.col("n") == 2)["tablas_evidencia"][0]
    assert t2 == "t23_proxy_hospital.csv; t23_fuga.csv; t23_resumen_hospitales.csv"
    assert d.filter(pl.col("n") == 17)["tablas_evidencia"][0] == "t23_resumen_seleccion.csv; t23_resumen.csv; t23_serie_mensual.csv"


def test_construir_hipotesis_arma_las_26_filas():
    res = decisiones.evaluar_todas(fuente_base())
    h = decisiones.construir_hipotesis(res)
    assert h.height == 26 and h.columns == list(decisiones.ESQUEMA_HIPOTESIS) and h["id_hipotesis"].to_list() == [x.id for x in decisiones.HIPOTESIS]
    assert dict(zip(h["id_hipotesis"], h["requiere_completo"]))["H01b"] == "si" and dict(zip(h["id_hipotesis"], h["requiere_completo"]))["H01a"] == "no"
    assert set(h["resultado"].unique()) <= set(decisiones.RESULTADOS) and set(h["clase"].unique()) == {"decisiva", "informativa"}


# ================================================================================================================ complementos de hipótesis y semáforo
@pytest.mark.parametrize("valor, esperado, texto", [(0.0, "matiza", "= 0"), (5.0, "respalda", "= 5"), (1_000_000.0, "respalda", "= 1.000.000"), (12.0, "respalda", "= 12")])
def test_h11_con_distintos_conteos_de_rut(valor, esperado, texto):
    r = ev("H11", fuente_base("extracto", metricas={"calidad.valores_rut.total": metrica(valor)}))
    assert r.resultado == esperado and texto in r.cifras_clave


@pytest.mark.parametrize("combinacion", sorted(SEMAFORO_DOS_DECISIVAS))
def test_semaforo_gris_si_falta_una_hipotesis_que_requiere_los_datos_completos_con_cualquier_otro_resultado(combinacion):
    # la regla 2 va antes que el rojo y el verde: sin los datos completos la decisión queda gris con cualquier combinación de las otras decisivas
    pares = [(_h("Ha"), _res(combinacion[0])), (_h("Hb"), _res(combinacion[1])), (_h("Hc", completo=True), _res("no_evaluable"))]
    assert decisiones.agregar_semaforo(pares, "cuantitativa") == "gris"
    evaluada = [(_h("Ha"), _res(combinacion[0])), (_h("Hb"), _res(combinacion[1])), (_h("Hc", completo=True), _res("respalda"))]
    assert decisiones.agregar_semaforo(evaluada, "cuantitativa") == SEMAFORO_TRES[combinacion]


# con una tercera decisiva que respalda el color es el de las otras dos, salvo que ninguna se pueda evaluar: entonces hay evidencia que matiza (amarillo) y no gris
SEMAFORO_TRES = {k: ("amarillo" if v == "gris" else v) for k, v in SEMAFORO_DOS_DECISIVAS.items()}


def test_una_hipotesis_informativa_que_contradice_no_da_rojo():
    assert decisiones.agregar_semaforo([(_h("Ha", "informativa"), _res("contradice"))], "cuantitativa") == "amarillo"


# ================================================================================================================================= supuestos
def sup(sid, f=None, cascada=None):
    return decisiones.evaluar_supuesto(f or fuente_base(), sid, cascada)


def con_metricas(**m):
    """Fuente base cuyas métricas SOLO son las dadas (para probar métricas ausentes)."""
    return FuenteEvidenciaFalsa(tablas_base(), metricas={k.replace("__", "."): v for k, v in m.items()}, cifras=CIFRAS_BASE)


def test_los_15_supuestos_estan_en_el_registro_y_concuerdan_con_hipotesis_csv():
    cfg = config.cargar_hipotesis()
    ids = [r["id"] for r in cfg.iter_rows(named=True) if r["id"].startswith("S")]
    assert list(decisiones.SUPUESTOS) == ids == [f"S{i:02d}" for i in range(1, 16)]
    # las constantes del módulo salen del texto del criterio
    por_id = {r["id"]: r for r in cfg.iter_rows(named=True)}
    assert "menores a 1 %" in por_id["S01"]["criterio_decision"] and decisiones.UMBRAL_S01 == 0.01
    assert "solo en 2019" in por_id["S07"]["pregunta"] and decisiones.ANIO_S07 == 2019
    assert "100 %" in por_id["S03"]["criterio_decision"] and "20 %" in por_id["S02"]["criterio_decision"]
    # el criterio de S09 quedó ampliado por el cambio 1.5 del protocolo: contrasta los egresos sin traslado con los que tienen traslado interno y dice cómo se decide
    criterio_s09 = por_id["S09"]["criterio_decision"]
    for frase in ("mayoría", "más de la mitad", "con traslado interno", "celda_minima", "se acepta", "se rechaza", "no se puede evaluar", "cambio 1.5 del protocolo",
                  "no levanta las sospechas de proxy ni de fuga"):
        assert frase in criterio_s09, frase


@pytest.mark.parametrize(
    "nac, sexo, esperado",
    [(0.0099, 0.0001, "ACEPTADO"), (0.01, 0.0, "RECHAZADO"), (0.0, 0.01, "RECHAZADO"), (0.018162, 0.007242, "RECHAZADO"), (0.5, None, "RECHAZADO"), (0.0, None, "NO EVALUABLE"),
     (None, 0.0, "NO EVALUABLE"), (None, None, "NO EVALUABLE")],
)
def test_s01_inconsistencias_del_identificador_contra_uno_por_ciento(nac, sexo, esperado):
    m = {}
    if nac is not None:
        m["id.consistencia.nacimiento_inconsistente_prop"] = metrica(nac)
    if sexo is not None:
        m["id.consistencia.sexo_inconsistente_prop"] = metrica(sexo)
    r = sup("S01", FuenteEvidenciaFalsa(tablas_base(), metricas=m, cifras=CIFRAS_BASE))
    assert r.estado == esperado


def test_s01_cita_las_proporciones_leidas_y_declara_la_base_de_ids_con_dos_o_mas_episodios():
    r = sup("S01")
    assert r.tabla == "t24_id_consistencia.csv"
    assert r.cifra == ("IDs con fecha de nacimiento inconsistente = 1,82 %; con sexo inconsistente = 0,72 %; máximo aceptado = 1 %; "
                       "base: IDs con dos o más episodios = 8.975 (nacimiento) y 8.970 (sexo conocido en dos o más episodios)")
    # la base es el grupo "todos los IDs" (el que usa la métrica): el otro grupo de la tabla es un subconjunto y no se suma
    assert "9.075" not in r.cifra and "9.070" not in r.cifra


def test_s01_sin_los_denominadores_conserva_las_proporciones_y_no_inventa_la_base():
    sin_tabla = sup("S01", sin(fuente(), "t24_id_consistencia.csv"))
    assert sin_tabla.cifra == "IDs con fecha de nacimiento inconsistente = 1,82 %; con sexo inconsistente = 0,72 %; máximo aceptado = 1 %" and sin_tabla.estado == "RECHAZADO"
    sin_columna = fuente(**{"t24_id_consistencia.csv": df({"regimen": "2019", "grupo_id": "todos los IDs", "n_ids_nacimiento_inconsistente": 3})})
    assert "base:" not in sup("S01", sin_columna).cifra


def _elix(n_si, n_no=20, otros=0):
    return df(*([{"elix_id": f"ELIX_{i:02d}", "supera_umbral": "si"} for i in range(n_si)] + [{"elix_id": f"ELIX_{i:02d}", "supera_umbral": "no"} for i in range(n_no)]
                + [{"elix_id": "X", "supera_umbral": "no evaluable"}] * otros))


def test_s02_mapeo_de_tres_caracteres_contra_el_completo():
    assert sup("S02", fuente(**{"t23_elix_vs_3c.csv": _elix(0)})).estado == "ACEPTADO"
    r = sup("S02", fuente(**{"t23_elix_vs_3c.csv": _elix(1)}))
    assert r.estado == "RECHAZADO" and r.cifra == "categorías cuya prevalencia con tres caracteres difiere más del umbral = 1 de 21"
    assert sup("S02").cifra.endswith("= 11 de 31")
    todas_no_evaluables = df(*({"elix_id": f"E{i}", "supera_umbral": "no evaluable"} for i in range(5)))
    assert sup("S02", fuente(**{"t23_elix_vs_3c.csv": todas_no_evaluables})).estado == "NO EVALUABLE"


def test_s02_sin_la_tabla_usa_la_metrica_de_la_tarea_8():
    sin_tabla = lambda v: FuenteEvidenciaFalsa({}, metricas={"elix.n_categorias_sobre_umbral_3c": metrica(v)})  # noqa: E731
    assert sup("S02", sin_tabla(3)).estado == "RECHAZADO" and sup("S02", sin_tabla(0)).estado == "ACEPTADO"
    assert sup("S02", FuenteEvidenciaFalsa({}, metricas={})).estado == "NO EVALUABLE"


@pytest.mark.parametrize("valor, esperado", [(100.0, "ACEPTADO"), (99.999332, "RECHAZADO"), (99.9995, "RECHAZADO"), (99.9999999, "RECHAZADO"), (None, "NO EVALUABLE")])
def test_s03_exige_exactamente_cien_por_ciento(valor, esperado):
    m = {} if valor is None else {"calidad.cie10_principal_valido_pct": metrica(valor)}
    r = sup("S03", FuenteEvidenciaFalsa(tablas_base(), metricas=m, cifras=CIFRAS_BASE))
    assert r.estado == esperado
    if valor == 99.999332:
        assert r.cifra == "diagnósticos principales normalizados que son válidos en CIE-10 = 99,9993 %" and "100 %" in r.motivo
    if valor is not None:                                                                  # la regla aplicada (100 % exacto, sin tolerancia) se declara en los dos estados
        assert r.motivo.startswith(decisiones.REGLA_S03) and "sin tolerancia" in r.motivo


@pytest.mark.parametrize("filas, esperado", [
    ([("exacta", 224254), ("por_prefijo", 120), ("sin_rol", 5)], "ACEPTADO"),                 # el sin rol de 5 casos queda bajo el mínimo publicable
    ([("exacta", 224254), ("sin_rol", 10)], "RECHAZADO"),                                        # exactamente el mínimo cuenta
    ([("exacta", 224254), ("sin_rol", None)], "ACEPTADO"),                                       # suprimido: no cuenta
    ([("exacta", 1)], "ACEPTADO"),
])
def test_s04_toda_etiqueta_con_muestra_suficiente_debe_tener_rol(filas, esperado):
    t = df(*({"etiqueta": f"E{i}", "metodo": m, "n": n} for i, (m, n) in enumerate(filas)))
    assert sup("S04", fuente(**{"t24_tipoalta_etiquetas.csv": t})).estado == esperado


def test_s04_declara_el_denominador_solo_con_las_etiquetas_con_casos_suficientes():
    # 10 etiquetas con casos suficientes, una suprimida (menos de diez casos) y la fila "(no aparece en los datos)" de las etiquetas del archivo de roles que no se observan
    filas = [{"etiqueta": f"E{i}", "metodo": "exacta", "n": 1000 + i} for i in range(10)]
    filas += [{"etiqueta": "RARA", "metodo": "exacta", "n": None}, {"etiqueta": "(no aparece en los datos)", "metodo": "sin_datos", "n": 0}]
    r = sup("S04", fuente(**{"t24_tipoalta_etiquetas.csv": df(*filas)}))
    assert r.estado == "ACEPTADO" and r.cifra == ("etiquetas de TIPOALTA con 10 casos o más y sin rol asignado = 0 de 10 con esos casos; "
                                                  "no cuentan 2 etiquetas con menos casos, suprimidas o ausentes de los datos")
    con_sin_rol = sup("S04", fuente(**{"t24_tipoalta_etiquetas.csv": df(*filas, {"etiqueta": "X", "metodo": "sin_rol", "n": 50})}))
    assert con_sin_rol.estado == "RECHAZADO" and "= 1 de 11 con esos casos; no cuentan 2 etiquetas" in con_sin_rol.cifra


def _card(**sobre):
    """Cardinalidades de las cinco columnas del supuesto S05: por defecto observado 10, Chao1 10,5 y documentado 10 (se acepta)."""
    filas = []
    for col in decisiones.COLUMNAS_S05:
        o, ch, d_ = sobre.get(col, (10, 10.5, 10))
        filas.append({"columna": col, "periodo": "2019-2024", "n_observado": o, "chao1": ch, "documentado": d_})
    return df(*filas)


@pytest.mark.parametrize("fila, esperado", [
    ((10, 10.5, 10), "ACEPTADO"),            # observado <= documentado <= Chao1
    ((11, 12.0, 10), "RECHAZADO"),           # el extracto ya supera lo documentado
    ((90, 90.33, 158), "NO EVALUABLE"),      # lo documentado queda sobre el Chao1: el extracto no puede confirmarlo
    ((10, 10.0, 10), "ACEPTADO"),            # documentado igual al observado y al Chao1
    ((10, 9.99, 10), "NO EVALUABLE"),        # documentado igual al observado pero mayor que el Chao1
])
def test_s05_cardinalidades_en_el_extracto(fila, esperado):
    assert sup("S05", fuente(**{"t22_cardinalidades.csv": _card(PREVISION=fila)})).estado == esperado if fila != (10, 10.5, 10) else True
    r = sup("S05", fuente(**{"t22_cardinalidades.csv": _card(PREVISION=fila, TIPOALTA=fila, DIAGNOSTICO1=fila, ESPECIALIDAD_MEDICA=fila, TIPO_PROCEDENCIA=fila)}))
    assert r.estado == esperado


def test_s05_el_rechazo_manda_sobre_lo_no_evaluable_y_se_necesitan_las_cinco_columnas():
    mezcla = sup("S05", fuente(**{"t22_cardinalidades.csv": _card(PREVISION=(90, 90.33, 158), COMUNA=(1, 1, 1), TIPOALTA=(11, 12.0, 10))}))
    assert mezcla.estado == "RECHAZADO"
    solo_no_ev = sup("S05", fuente(**{"t22_cardinalidades.csv": _card(PREVISION=(90, 90.33, 158))}))
    assert solo_no_ev.estado == "NO EVALUABLE" and "Chao1" in solo_no_ev.motivo
    sin_una = _card().filter(pl.col("columna") != "PREVISION")
    assert sup("S05", fuente(**{"t22_cardinalidades.csv": sin_una})).estado == "NO EVALUABLE"
    sin_doc = _card().with_columns(pl.when(pl.col("columna") == "PREVISION").then(None).otherwise(pl.col("documentado")).alias("documentado"))
    assert sup("S05", fuente(**{"t22_cardinalidades.csv": sin_doc})).estado == "NO EVALUABLE"


def test_s05_usa_el_periodo_agregado_mas_largo_y_en_completo_exige_igualdad():
    filas = [{"columna": c, "periodo": p, "n_observado": o, "chao1": 10.5, "documentado": 10}
             for c in decisiones.COLUMNAS_S05 for p, o in (("2019", 99), ("2019-2023", 99), ("2019-2024", 10))]
    t = df(*filas)
    assert sup("S05", fuente(**{"t22_cardinalidades.csv": t})).estado == "ACEPTADO"                         # no toma el año suelto ni el período más corto
    assert sup("S05", fuente("completo", **{"t22_cardinalidades.csv": t})).estado == "ACEPTADO"
    otra = _card(PREVISION=(9, 10.5, 10))
    assert sup("S05", fuente("completo", **{"t22_cardinalidades.csv": otra})).estado == "RECHAZADO"        # en completo 9 contra 10 no es igualdad
    assert sup("S05", fuente("extracto", **{"t22_cardinalidades.csv": otra})).estado == "ACEPTADO"


def _fmt(*filas):
    return df(*({"anio": a, "columna": c, "clase": cl, "n_clase": n} for a, c, cl, n in filas))


@pytest.mark.parametrize("filas, esperado", [
    ([(2022, "F", "ISO", 100), (2022, "F", "DD-MM-AAAA", 0)], "ACEPTADO"),
    ([(2022, "F", "ISO", 100), (2022, "F", "DD-MM-AAAA", 5)], "RECHAZADO"),                          # las dos clases presentes
    ([(2022, "F", "ISO", 100), (2022, "F", "DD-MM-AAAA", None)], "RECHAZADO"),                       # un conteo suprimido cuenta como presente
    ([(2022, "F", "ISO", None), (2022, "F", "DD-MM-AAAA", 0)], "ACEPTADO"),
    ([(2022, "F", "ISO", 100), (2023, "F", "DD-MM-AAAA", 100)], "ACEPTADO"),                         # formatos distintos en años distintos no es mezcla
    ([(2022, "F", "ISO", 100), (2022, "G", "DD-MM-AAAA", 100)], "ACEPTADO"),                         # ni en columnas distintas
    ([(2022, "F", "SOLO_DIGITOS", 100), (2022, "F", "TEXTO", 100)], "ACEPTADO"),
])
def test_s06_ninguna_columna_de_fecha_mezcla_formatos_en_un_anio(filas, esperado):
    assert sup("S06", fuente(**{"t21_deriva_formato.csv": _fmt(*filas)})).estado == esperado


def test_s06_cita_cuantas_combinaciones_mezclan():
    r = sup("S06", fuente(**{"t21_deriva_formato.csv": _fmt((2022, "F", "ISO", 100), (2022, "F", "DD-MM-AAAA", 5), (2023, "F", "ISO", 100))}))
    assert r.cifra == "combinaciones de año y columna de fecha que mezclan formatos = 1 de 2 con fechas"


@pytest.mark.parametrize("n, primero, esperado", [(1, 2019, "ACEPTADO"), (1, 2024, "RECHAZADO"), (2, 2019, "RECHAZADO"), (0, None, "RECHAZADO"), (1, None, "RECHAZADO")])
def test_s07_el_identificador_cambia_de_nombre_solo_en_2019(n, primero, esperado):
    m = {"estructura.n_anios_id_beneficiario": metrica(n), "estructura.primer_anio_id_beneficiario": metrica(primero)}
    r = sup("S07", FuenteEvidenciaFalsa(tablas_base(), metricas=m, cifras=CIFRAS_BASE))
    assert r.estado == esperado
    assert sup("S07", FuenteEvidenciaFalsa(tablas_base(), metricas={}, cifras=CIFRAS_BASE)).estado == "NO EVALUABLE"


def test_s07_cita_el_primer_anio():
    assert sup("S07").cifra == "años con la columna del identificador llamada ID_BENEFICIARIO = 1 (el primero es 2024)"


@pytest.mark.parametrize("maximo, fuera, etiqueta, esperado", [
    (65, 0, "extracto", "ACEPTADO"), (68, 0, "extracto", "ACEPTADO"), (66, 0, "completo", "ACEPTADO"), (69, 0, "extracto", "RECHAZADO"), (72, 1, "extracto", "RECHAZADO"),
    (66, 1, "extracto", "RECHAZADO"), (64, 0, "extracto", "NO EVALUABLE"), (64, 0, "completo", "RECHAZADO"),
])
def test_s08_hospitales_entre_las_dos_cifras_documentadas(maximo, fuera, etiqueta, esperado):
    f = fuente_base(etiqueta, tablas={"t24_resumen_catalogos.csv": df({"n_codigos_hospital_fuera_catalogo": fuera})}, metricas={"hospitales.n_distintos.maximo": metrica(maximo)})
    assert sup("S08", f).estado == esperado


def test_s08_sin_la_metrica_o_sin_las_cifras_documentadas_no_es_evaluable():
    assert sup("S08", FuenteEvidenciaFalsa(tablas_base(), metricas={}, cifras=CIFRAS_BASE)).estado == "NO EVALUABLE"
    sin_cifras = FuenteEvidenciaFalsa(tablas_base(), metricas={"hospitales.n_distintos.maximo": metrica(66)}, cifras={})
    assert sup("S08", sin_cifras).estado == "NO EVALUABLE"
    # las cifras documentadas pueden venir en cualquier orden
    invertidas = FuenteEvidenciaFalsa(tablas_base(), metricas={"hospitales.n_distintos.maximo": metrica(66)}, cifras={"C077": 68.0, "C078": 65.0})
    assert sup("S08", invertidas).estado == "RECHAZADO"                                               # la base trae 1 código fuera del catálogo
    r = sup("S08")
    assert r.cifra == "máximo de hospitales distintos en un año = 72; documentados = 65 y 68; códigos de hospital fuera del catálogo = 1"


SIN_TRASLADO_REAL = (92445, 91706, 0.9914, 0.9926)       # egresos, iguales e intervalo de la corrida real sobre los extractos
CON_TRASLADO_REAL = (22696, 3454, 0.148, 0.157)
GRUPO_SUPRIMIDO = (None, None, None, None)               # menos de diez casos: la Tarea 11 publica las cifras del grupo vacías
CIFRA_SIN = "el servicio de ingreso coincide con el de alta en 99,2 % de los egresos sin traslado interno (intervalo de 99,1 % a 99,3 %)"
CIFRA_CON = "15,2 % de los egresos con traslado interno (intervalo de 14,8 % a 15,7 %)"


def _fuga9(sin_traslado=SIN_TRASLADO_REAL, con_traslado=CON_TRASLADO_REAL):
    """``t23_fuga.csv`` mínima con las dos filas de concordancia de la Tarea 11 (``None`` omite la fila). Cada grupo es ``(egresos, iguales, ic_bajo, ic_alto)``; la proporción y la
    marca ``coincide_mayoria`` salen de los conteos, como las calcula la Tarea 11 (mayoría es estrictamente más de la mitad)."""
    filas = []
    for tipo, (col_n, col_k), v in (("concordancia", ("n_sin_traslado", "n_iguales_sin_traslado"), sin_traslado),
                                    ("concordancia_con_traslado", ("n_con_traslado", "n_iguales_con_traslado"), con_traslado)):
        if v is None:
            continue
        n, k, lo, hi = v
        sin_proporcion = n is None or n == 0                 # suprimido (menos de diez casos) o grupo vacío ("sin datos"): la Tarea 11 deja la proporción y la marca vacías
        filas.append({"tipo_fila": tipo, "variable": "SERVICIO_INGRESO_IGUAL_ALTA", col_n: n, col_k: k, "prop_igualdad": None if sin_proporcion else k / n, "ic_bajo": lo,
                      "ic_alto": hi, "coincide_mayoria": None if sin_proporcion else ("si" if 2 * k > n else "no")})
    return df(*filas)


def _concordancia9(sin_traslado=SIN_TRASLADO_REAL, con_traslado=CON_TRASLADO_REAL):
    return fuente(**{"t23_fuga.csv": _fuga9(sin_traslado, con_traslado)})


def test_s09_acepta_si_coincide_en_la_mayoria_sin_traslado_y_en_la_minoria_con_traslado_interno():
    # sin traslado interno la coincidencia es esperable por construcción y no discrimina; la prueba informativa es el grupo con traslado (cambio 1.5 del protocolo)
    r = sup("S09")                                           # tablas de la corrida real: 91.706 de 92.445 sin traslado y 3.454 de 22.696 con traslado
    assert r.estado == "ACEPTADO" and r.tabla == "t23_fuga.csv" and r.motivo == decisiones.MOTIVO_S09_ACEPTADO
    assert r.cifra == f"{CIFRA_SIN} y en {CIFRA_CON}"
    assert "registra el servicio de ingreso" in r.motivo and "no levanta las sospechas de proxy del hospital ni de fuga" in r.motivo and "refleja la estadía" not in r.motivo
    assert sup("S09", _concordancia9()).cifra == r.cifra                                  # la tabla hecha a mano con los mismos conteos da lo mismo


def test_s09_rechaza_si_coincide_en_la_mayoria_tambien_con_traslado_interno():
    r = sup("S09", _concordancia9(con_traslado=(1000, 800, 0.77, 0.83)))
    assert r.estado == "RECHAZADO" and r.motivo == decisiones.MOTIVO_S09_RECHAZADO and "refleja el servicio de la estadía" in r.motivo
    assert r.cifra == f"{CIFRA_SIN} y en 80,0 % de los egresos con traslado interno (intervalo de 77,0 % a 83,0 %)"


@pytest.mark.parametrize("sin_traslado, con_traslado, esperado", [
    ((100, 51, 0.4, 0.6), (100, 49, 0.4, 0.6), "ACEPTADO"),            # mayoría sin traslado y minoría con traslado
    ((100, 51, 0.4, 0.6), (100, 50, 0.4, 0.6), "ACEPTADO"),            # la mitad exacta con traslado no es mayoría
    ((100, 51, 0.4, 0.6), (100, 51, 0.4, 0.6), "RECHAZADO"),           # apenas más de la mitad con traslado
    ((100, 50, 0.4, 0.6), (100, 10, 0.4, 0.6), "NO EVALUABLE"),        # la mitad exacta sin traslado no es mayoría: inconsistente
    ((100, 49, 0.4, 0.6), (100, 10, 0.4, 0.6), "NO EVALUABLE"),
    ((101, 50, 0.4, 0.6), (101, 10, 0.4, 0.6), "NO EVALUABLE"),        # 50 de 101 no es más de la mitad
    ((101, 51, 0.4, 0.6), (101, 51, 0.4, 0.6), "RECHAZADO"),           # 51 de 101 sí lo es, en los dos grupos
    ((101, 51, 0.4, 0.6), (101, 50, 0.4, 0.6), "ACEPTADO"),
    ((100, 100, 0.9, 1.0), (100, 0, 0.0, 0.1), "ACEPTADO"),            # los extremos
    ((100, 100, 0.9, 1.0), (100, 100, 0.9, 1.0), "RECHAZADO"),
])
def test_s09_mayoria_es_estrictamente_mas_de_la_mitad_en_cada_grupo(sin_traslado, con_traslado, esperado):
    assert sup("S09", _concordancia9(sin_traslado, con_traslado)).estado == esperado


def test_s09_calcula_la_mayoria_con_los_conteos_y_no_con_la_marca_de_la_tabla():
    # la marca coincide_mayoria dice "si" en los dos grupos, pero 50 de 100 sin traslado no es más de la mitad y 50 de 100 con traslado tampoco
    marcada = _fuga9((100, 50, 0.4, 0.6), (100, 50, 0.4, 0.6)).with_columns(pl.lit("si").alias("coincide_mayoria"))
    assert sup("S09", fuente(**{"t23_fuga.csv": marcada})).estado == "NO EVALUABLE"
    # y al revés: la marca dice "no", pero 51 de 100 es mayoría en los dos grupos
    desmarcada = _fuga9((100, 51, 0.4, 0.6), (100, 51, 0.4, 0.6)).with_columns(pl.lit("no").alias("coincide_mayoria"))
    assert sup("S09", fuente(**{"t23_fuga.csv": desmarcada})).estado == "RECHAZADO"


@pytest.mark.parametrize("sin_traslado, con_traslado, esperado", [
    ((10, 8, 0.5, 0.9), (10, 3, 0.1, 0.5), "ACEPTADO"),                # exactamente el mínimo publicable (celda_minima = 10) se evalúa
    ((9, 8, 0.5, 0.9), (100, 3, 0.1, 0.5), "NO EVALUABLE"),            # un grupo con menos del mínimo
    ((100, 80, 0.5, 0.9), (9, 3, 0.1, 0.5), "NO EVALUABLE"),
    ((9, 8, 0.5, 0.9), (9, 3, 0.1, 0.5), "NO EVALUABLE"),
    ((0, 0, None, None), (100, 3, 0.1, 0.5), "NO EVALUABLE"),          # un grupo vacío
])
def test_s09_cada_grupo_necesita_al_menos_el_minimo_publicable_de_egresos(sin_traslado, con_traslado, esperado):
    r = sup("S09", _concordancia9(sin_traslado, con_traslado))
    assert r.estado == esperado
    if esperado == "NO EVALUABLE":
        assert r.motivo == decisiones.MOTIVO_S09_GRUPO_PEQUENO


def test_s09_toma_el_minimo_de_egresos_del_umbral_celda_minima_y_no_de_una_constante():
    tabla = {"t23_fuga.csv": _fuga9((40, 30, 0.6, 0.9), (40, 10, 0.1, 0.4))}
    base = FuenteEvidenciaFalsa(tabla, etiqueta="extracto")
    assert decisiones.evaluar_supuesto(base, "S09").estado == "ACEPTADO"                     # con el umbral real (10), 40 egresos por grupo alcanzan
    exigente = FuenteEvidenciaFalsa(tabla, umbrales={"celda_minima": 50}, etiqueta="extracto")
    r = decisiones.evaluar_supuesto(exigente, "S09")
    assert r.estado == "NO EVALUABLE" and r.motivo == decisiones.MOTIVO_S09_GRUPO_PEQUENO       # con un mínimo de 50, 40 egresos no alcanzan


def test_s09_con_un_grupo_ausente_o_suprimido_no_es_evaluable_y_cita_el_otro_grupo():
    for con in (None, GRUPO_SUPRIMIDO):                      # sin la fila con traslado o con sus cifras suprimidas
        r = sup("S09", _concordancia9(con_traslado=con))
        assert r.estado == "NO EVALUABLE" and r.motivo == decisiones.MOTIVO_S09_GRUPO_PEQUENO and r.cifra == CIFRA_SIN, con
    for sin_ in (None, GRUPO_SUPRIMIDO):                     # sin la fila sin traslado o con sus cifras suprimidas
        r = sup("S09", _concordancia9(sin_traslado=sin_))
        assert r.estado == "NO EVALUABLE" and r.motivo == decisiones.MOTIVO_S09_GRUPO_PEQUENO
        assert r.cifra == f"el servicio de ingreso coincide con el de alta en {CIFRA_CON}", sin_
    ninguno = sup("S09", _concordancia9(sin_traslado=None, con_traslado=None))
    assert ninguno.estado == "NO EVALUABLE" and ninguno.cifra == "" and ninguno.motivo == decisiones.MOTIVO_S09_GRUPO_PEQUENO
    sin_intervalo = sup("S09", _concordancia9(con_traslado=(22696, 3454, None, None)))                  # un intervalo suprimido tampoco se cita a medias
    assert sin_intervalo.estado == "NO EVALUABLE" and sin_intervalo.cifra == CIFRA_SIN


def test_s09_sin_coincidir_en_la_mayoria_sin_traslado_la_medicion_es_inconsistente_y_no_se_evalua():
    # donde no hay traslado el servicio de ingreso es el de alta por construcción: si no coincide en la mayoría, la medición no permite concluir (antes quedaba aceptado)
    r = sup("S09", _concordancia9(sin_traslado=(1000, 300, 0.27, 0.33)))
    assert r.estado == "NO EVALUABLE" and r.motivo == decisiones.MOTIVO_S09_INCONSISTENTE and "inconsistente" in r.motivo
    assert r.cifra == f"el servicio de ingreso coincide con el de alta en 30,0 % de los egresos sin traslado interno (intervalo de 27,0 % a 33,0 %) y en {CIFRA_CON}"


def test_s09_sin_los_conteos_o_sin_la_tabla_no_es_evaluable():
    sin_conteos = tablas_base()["t23_fuga.csv"].drop(["n_sin_traslado", "n_iguales_sin_traslado", "n_con_traslado", "n_iguales_con_traslado"])
    r = sup("S09", fuente(**{"t23_fuga.csv": sin_conteos}))                               # sin los conteos no se puede comprobar el mínimo ni calcular la mayoría
    assert r.estado == "NO EVALUABLE" and r.motivo == decisiones.MOTIVO_S09_GRUPO_PEQUENO
    assert sup("S09", fuente(**{"t23_fuga.csv": _fuga9(sin_traslado=None, con_traslado=CON_TRASLADO_REAL)})).estado == "NO EVALUABLE"
    assert sup("S09", sin(fuente(), "t23_fuga.csv")).estado == "NO EVALUABLE"


def test_s10_cascada_con_el_dataframe_sin_suprimir_o_con_la_tabla_publicada():
    assert sup("S10", cascada=_cascada()).estado == "ACEPTADO"
    mala = sup("S10", cascada=_cascada(EX02=1143))
    assert mala.estado == "RECHAZADO" and "2019: A es 45.031" in mala.motivo and mala.cifra == "identidades de la cascada verificadas = 5; que no cuadran = 1"      # solo la de A usa EX02
    publicada = sup("S10")                                                                          # sin DataFrame, lee t23_cascada.csv de la base
    assert publicada.estado == "ACEPTADO" and publicada.cifra == "identidades de la cascada verificadas = 5; que no cuadran = 0"
    suprimida = fuente(**{"t23_cascada.csv": _cascada().with_columns(pl.lit(None, dtype=pl.Int64).alias("n"))})
    assert sup("S10", suprimida).estado == "NO EVALUABLE"
    assert sup("S10", sin(fuente(), "t23_cascada.csv")).estado == "NO EVALUABLE"
    # la cascada sin suprimir manda sobre la publicada
    assert sup("S10", suprimida, cascada=_cascada()).estado == "ACEPTADO"


@pytest.mark.parametrize("ic, esperado", [((38.66, 42.24), "ACEPTADO"), ((0.0, 5.0), "NO EVALUABLE"), ((-1.0, 2.0), "NO EVALUABLE"), ((-5.0, 0.0), "NO EVALUABLE"),
                                          ((-5.0, -0.1), "RECHAZADO"), ((0.0001, 3.0), "ACEPTADO")])
def test_s11_la_codificacion_crece_si_el_intervalo_queda_sobre_cero(ic, esperado):
    f = fuente_base("extracto", metricas={"codificacion.aumento_pct_2019_2024": metrica(10.0, ic=ic)})
    assert sup("S11", f).estado == esperado
    sin_ic = fuente_base("extracto", metricas={"codificacion.aumento_pct_2019_2024": metrica(10.0)})
    assert sup("S11", sin_ic).estado == "NO EVALUABLE"


def test_s11_cita_el_aumento_y_su_intervalo():
    assert sup("S11").cifra == "aumento relativo del promedio de diagnósticos secundarios entre el primer y el último año = 40,4 % (intervalo de 38,7 % a 42,2 %)"


def _rn(rn2=(100, 0.99), rn3=(100, 1.0), rn4=(100, 1.0)):
    return df(*({"columna": c, "n_informados": n, "prop_modal": p} for c, (n, p) in zip(decisiones.COLUMNAS_S12, (rn2, rn3, rn4))))


@pytest.mark.parametrize("filas, esperado", [
    (dict(rn2=(100, 1.0)), "ACEPTADO"),                      # todas constantes
    (dict(), "RECHAZADO"),                                   # una con proporción modal menor que uno trae información
    (dict(rn2=(0, None), rn3=(0, None), rn4=(0, None)), "NO EVALUABLE"),
    (dict(rn2=(0, None), rn3=(100, 1.0)), "ACEPTADO"),       # sin valores o constantes: relleno
    (dict(rn2=(100, None)), "NO EVALUABLE"),                 # proporción suprimida
    (dict(rn2=(100, None), rn3=(100, 0.5)), "RECHAZADO"),    # basta una con información
    (dict(rn2=(None, None)), "NO EVALUABLE"),                # conteo suprimido: hay valores de proporción desconocida
])
def test_s12_bloque_de_recien_nacidos_relleno_o_informacion(filas, esperado):
    r = sup("S12", fuente(**{"t22_bloque_rn.csv": _rn(**filas)}))
    assert r.estado == esperado
    if esperado != "NO EVALUABLE":                                           # la regla aplicada (proporción modal menor que uno = información) se declara en los dos estados
        assert r.motivo.startswith(decisiones.REGLA_S12) and "proporción menor cuenta como información" in r.motivo


def test_s12_sin_una_de_las_columnas_no_es_evaluable():
    sin_rn4 = _rn().filter(pl.col("columna") != "RN4ESTADO")
    assert sup("S12", fuente(**{"t22_bloque_rn.csv": sin_rn4})).estado == "NO EVALUABLE"


def _conc(*filas):
    return df(*({"columna": f"C{i}", "estado": e, "n_filas_med": 1000, "n_sin_med": 100, "n_centinela_med": c, "pct_sin_doc": d, "pct_sin_med": 10.0}
                for i, (e, c, d) in enumerate(filas)))


@pytest.mark.parametrize("filas, esperado", [
    ([("COINCIDE", 50, 10.0), ("COINCIDE", 50, 10.0), ("DISCREPANCIA", 50, 15.0)], "ACEPTADO"),         # 2 de 3 siguen el conteo sin centinelas
    ([("COINCIDE", 50, 10.0), ("DISCREPANCIA", 50, 15.0)], "RECHAZADO"),                                 # la mitad exacta no es "más de la mitad"
    ([("COINCIDE", 50, 15.0), ("COINCIDE", 50, 15.0), ("COINCIDE", 50, 10.0)], "RECHAZADO"),             # 1 de 3
    ([("COINCIDE", 0, 10.0)], "NO EVALUABLE"),                                                           # sin centinelas no cuenta
    ([("NO EVALUABLE", 50, 10.0)], "NO EVALUABLE"),                                                      # estado no evaluable
    ([("COINCIDE", 50, 12.5)], "ACEPTADO"),                                                              # equidistante: sigue el supuesto (igualdad)
])
def test_s13_el_diccionario_sigue_el_conteo_sin_centinelas(filas, esperado):
    r = sup("S13", fuente(**{"t22_conciliacion.csv": _conc(*filas)}))
    assert r.estado == esperado
    if esperado != "NO EVALUABLE":                                           # la regla aplicada (mayoría simple de las columnas con centinelas) se declara en los dos estados
        assert r.motivo == decisiones.REGLA_S13 and "mayoría simple" in r.motivo


def test_s13_cita_cuantas_columnas_siguen_el_supuesto():
    assert sup("S13").cifra == "columnas con centinelas donde el diccionario sigue el conteo sin centinelas = 2 de 3"
    columna_sin_dato = _conc(("COINCIDE", 50, 10.0)).with_columns(pl.lit(None, dtype=pl.Float64).alias("pct_sin_doc"))
    assert sup("S13", fuente(**{"t22_conciliacion.csv": columna_sin_dato})).estado == "NO EVALUABLE"


def test_s14_nunca_es_evaluable_y_cita_la_suma_si_no_hay_celdas_suprimidas():
    r = sup("S14")
    assert r.estado == "NO EVALUABLE" and "suprimida" in r.cifra and "pocas" in r.motivo
    completas = {k: metrica(v) for k, v in (("calidad.alta_antes_ingreso.total", 12), ("calidad.edad_negativa.total", 10), ("calidad.edad_mayor_110.total", 26))}
    r2 = sup("S14", fuente_base("extracto", metricas=completas))
    assert r2.estado == "NO EVALUABLE" and r2.cifra.endswith("= 48")


def _dup(*filas):
    return df(*({"anio": 2019 + i, "n_filas_duplicadas_clave": a, "n_solapes": b, "n_duplicados_exactos_por_id_y_fechas": c} for i, (a, b, c) in enumerate(filas)))


@pytest.mark.parametrize("filas, esperado", [([(0, 0, 0), (0, 0, 0)], "ACEPTADO"), ([(0, 0, 0), (0, 1, 0)], "NO EVALUABLE"), ([(0, 0, None)], "NO EVALUABLE"),
                                             ([(5000, 7000, 9000)], "NO EVALUABLE")])
def test_s15_duplicados_y_solapes_nunca_se_rechaza(filas, esperado):
    r = sup("S15", fuente(**{"t24_duplicados.csv": _dup(*filas)}))
    assert r.estado == esperado and r.estado != "RECHAZADO"


def test_s15_cita_el_total_y_las_celdas_suprimidas():
    assert sup("S15").cifra == ("suma de las celdas visibles de 3 medidas de duplicados y solapes, que pueden contar los mismos episodios más de una vez, = 62; "
                                "celdas suprimidas por tener menos de diez casos = 3")
    assert sup("S15", sin(fuente(), "t24_duplicados.csv")).estado == "NO EVALUABLE"


def test_un_supuesto_con_la_tabla_ausente_o_sin_columnas_no_lanza_excepcion():
    vacia = FuenteEvidenciaFalsa({}, metricas={}, cifras={})
    for sid in decisiones.SUPUESTOS:
        r = decisiones.evaluar_supuesto(vacia, sid)
        assert r.estado in ("NO EVALUABLE",) and r.motivo, sid
    r = sup("S04", fuente(**{"t24_tipoalta_etiquetas.csv": df({"otra": 1})}))
    assert r.estado == "NO EVALUABLE" and "columna" in r.motivo


def test_construir_supuestos_arma_las_15_filas_con_texto_del_protocolo():
    cfg = config.cargar_hipotesis()
    s = decisiones.construir_supuestos(fuente_base(), cfg, cascada=_cascada())
    assert s.height == 15 and s.columns == list(decisiones.ESQUEMA_SUPUESTOS) and s["id"].to_list() == [f"S{i:02d}" for i in range(1, 16)]
    por_id = {r["id"]: r for r in cfg.iter_rows(named=True)}
    for r in s.iter_rows(named=True):
        assert r["supuesto"] == por_id[r["id"]]["pregunta"] and r["criterio"] == por_id[r["id"]]["criterio_decision"] and r["nivel"] == "extracto"
        assert r["estado"] in decisiones.ESTADOS_SUPUESTO
    estados = dict(zip(s["id"], s["estado"]))
    assert estados == {"S01": "RECHAZADO", "S02": "RECHAZADO", "S03": "RECHAZADO", "S04": "ACEPTADO", "S05": "NO EVALUABLE", "S06": "ACEPTADO", "S07": "RECHAZADO", "S08": "RECHAZADO",
                       "S09": "ACEPTADO", "S10": "ACEPTADO", "S11": "ACEPTADO", "S12": "RECHAZADO", "S13": "ACEPTADO", "S14": "NO EVALUABLE", "S15": "NO EVALUABLE"}
    # el supuesto no evaluable por tabla ausente conserva las tablas del protocolo para que el lector sepa dónde buscar
    sin_tabla = decisiones.construir_supuestos(sin(fuente_base(), "t24_duplicados.csv"), cfg)
    assert sin_tabla.filter(pl.col("id") == "S15")["tabla"][0] == "t24_duplicados.csv" and sin_tabla.filter(pl.col("id") == "S15")["estado"][0] == "NO EVALUABLE"


def test_el_nivel_de_los_supuestos_es_el_de_la_fuente():
    s = decisiones.construir_supuestos(fuente_base("completo"), config.cargar_hipotesis())
    assert set(s["nivel"].to_list()) == {"completo"}


# ================================================================================================================================== resumen y métricas
def test_resumen_de_decisiones_cuenta_semaforo_supuestos_e_hipotesis():
    f = fuente_base()
    res = decisiones.evaluar_todas(f)
    d, _ = _construir(f)
    s = decisiones.construir_supuestos(f, config.cargar_hipotesis(), cascada=_cascada())
    r = decisiones.resumen_decisiones(d, s, decisiones.construir_hipotesis(res))
    assert r["decisiones_n_total"] == 17 and r["decisiones_n_verde"] == 3 and r["decisiones_n_amarillo"] == 7 and r["decisiones_n_rojo"] == 0 and r["decisiones_n_gris"] == 7
    assert r["decisiones_n_supuestos"] == 15 and r["decisiones_n_supuestos_aceptados"] == 6 and r["decisiones_n_supuestos_rechazados"] == 6 and r["decisiones_n_supuestos_no_evaluables"] == 3
    assert r["decisiones_n_hipotesis"] == 26 and sum(r[f"decisiones_n_hipotesis_{k}"] for k in decisiones.RESULTADOS) == 26
    # respaldan H03, H07 y H14; no se evalúan H01b, H01c, H06a, H06b, H13 y H16a (requieren datos completos) y H04, H10 y H12 (no dependen de datos)
    assert r["decisiones_n_hipotesis_respalda"] == 3 and r["decisiones_n_hipotesis_contradice"] == 0 and r["decisiones_n_hipotesis_no_evaluable"] == 9
    assert r["decisiones_n_hipotesis_matiza"] == 14


# =============================================================================================================== corrida sobre la fuente sintética
@pytest.fixture(scope="module")
def decisiones_corridas(pipeline_sintetico, tmp_path_factory):
    ctx_d = copiar_tablas(pipeline_sintetico.ctx, tmp_path_factory.mktemp("t12_dec"), con_figuras=True)
    r = decisiones.ejecutar(ctx_d, cascada=pipeline_sintetico.r7["cascada"])
    ctx_d.escribir_registros()
    return ctx_d, r


def test_ejecutar_escribe_las_tres_tablas_con_sus_filas(decisiones_corridas):
    ctx_d, r = decisiones_corridas
    assert (r["hipotesis"].height, r["decisiones"].height, r["supuestos"].height) == (26, 17, 15)
    for nombre, filas in (("t25_hipotesis_evaluadas.csv", 26), ("decisiones_evidencia.csv", 17), ("supuestos_verificados.csv", 15)):
        t = ctx_d.indice_tablas().filter(pl.col("nombre") == nombre)
        assert t.height == 1 and t["filas"][0] == filas and t["tarea"][0] == "T12" and t["columnas_conteo"][0] == ""
    d = ctx_d.leer_tabla("decisiones_evidencia.csv", columnas_texto=decisiones.COLUMNAS_TEXTO_DECISIONES)
    assert d["n"].to_list() == list(range(1, 18)) and set(d["estado"].to_list()) == {"pendiente de decisión humana"}
    assert set(d["semaforo"].to_list()) <= set(decisiones.SEMAFOROS) and set(d["tipo_evidencia"].to_list()) <= set(decisiones.TIPOS_EVIDENCIA)


def test_ninguna_decision_sale_de_una_hipotesis_primaria_inexistente(decisiones_corridas):
    ctx_d, _ = decisiones_corridas
    h = ctx_d.leer_tabla("t25_hipotesis_evaluadas.csv", columnas_texto=["id_hipotesis", "clase", "requiere_completo", "resultado", "veredicto", "tabla", "cifras_clave", "motivo"])
    assert h["id_hipotesis"].to_list() == [x.id for x in decisiones.HIPOTESIS]
    no_ev = h.filter(pl.col("resultado") == "no_evaluable")
    assert no_ev["motivo"].null_count() == 0 and (no_ev["motivo"] == "").sum() == 0                 # todo no evaluable dice por qué
    assert set(h.filter(pl.col("requiere_completo") == "si")["resultado"].to_list()) == {"no_evaluable"}


def test_con_datos_sinteticos_de_extracto_los_grises_incluyen_las_decisiones_que_requieren_datos_completos(decisiones_corridas):
    ctx_d, _ = decisiones_corridas
    d = ctx_d.leer_tabla("decisiones_evidencia.csv", columnas_texto=decisiones.COLUMNAS_TEXTO_DECISIONES)
    sem = dict(zip(d["n"], d["semaforo"]))
    assert all(sem[n] == "gris" for n in (1, 4, 6, 10, 12, 13, 16))


def test_la_cascada_sin_suprimir_alimenta_s10(pipeline_sintetico, tmp_path):
    ctx_d = copiar_tablas(pipeline_sintetico.ctx, tmp_path)
    r = decisiones.ejecutar(ctx_d, cascada=pipeline_sintetico.r7["cascada"])
    s10 = r["supuestos"].filter(pl.col("id") == "S10").row(0, named=True)
    assert s10["estado"] == "ACEPTADO" and "que no cuadran = 0" in s10["cifra"]
    # una cascada adulterada se rechaza
    mala = pipeline_sintetico.r7["cascada"].with_columns(pl.when(pl.col("paso") == "A").then(pl.col("n") + 1).otherwise(pl.col("n")).alias("n"))
    r2 = decisiones.ejecutar(copiar_tablas(pipeline_sintetico.ctx, tmp_path / "otra"), cascada=mala)
    assert r2["supuestos"].filter(pl.col("id") == "S10")["estado"][0] == "RECHAZADO"


def test_las_metricas_de_decisiones_cuentan_decisiones_y_supuestos(decisiones_corridas):
    ctx_d, r = decisiones_corridas
    m = ctx_d.metricas().filter(pl.col("clave").str.starts_with("decisiones."))
    val = dict(zip(m["clave"], m["valor"]))
    assert set(val) == {"decisiones.n_verde", "decisiones.n_amarillo", "decisiones.n_rojo", "decisiones.n_gris", "decisiones.n_supuestos_aceptados", "decisiones.n_supuestos_rechazados",
                        "decisiones.n_supuestos_no_evaluables"}
    assert val["decisiones.n_verde"] + val["decisiones.n_amarillo"] + val["decisiones.n_rojo"] + val["decisiones.n_gris"] == 17
    assert val["decisiones.n_supuestos_aceptados"] + val["decisiones.n_supuestos_rechazados"] + val["decisiones.n_supuestos_no_evaluables"] == 15
    assert set(m["tarea"]) == {"T12"} and all("no casos" in n for n in m["nota"].to_list())


def test_el_resumen_de_evidencia_junta_las_columnas_de_decisiones_y_las_del_ledger(pipeline_sintetico, tmp_path):
    ctx_d = copiar_tablas(pipeline_sintetico.ctx, tmp_path)
    ledger.ejecutar(ctx_d)
    decisiones.ejecutar(ctx_d)
    fila = ctx_d.leer_tabla("t25_resumen_evidencia.csv").row(0, named=True)
    assert fila["ledger_n_cifras"] == 129 and fila["decisiones_n_total"] == 17 and fila["decisiones_n_supuestos"] == 15
    assert ctx_d.leer_tabla("t25_resumen_evidencia.csv").height == 1


def test_la_privacidad_queda_en_cero_y_la_tarea_12_no_deja_filas_en_el_registro_de_accesos(pipeline_sintetico, tmp_path, monkeypatch):
    ctx_d = copiar_tablas(pipeline_sintetico.ctx, tmp_path, con_figuras=True)
    monkeypatch.setattr(ctx_d.lector, "leer", lambda *a, **k: (_ for _ in ()).throw(AssertionError("la Tarea 12 no lee datos")))
    decisiones.ejecutar(ctx_d)
    ctx_d.escribir_registros()
    esc = privacidad.escanear_salidas(ctx_d.dir_salida, ctx_d.indice_tablas(), ctx_d.celda_minima)
    assert privacidad.total_violaciones(esc) == 0, esc.filter(pl.col("n_violaciones") > 0)
    assert ctx_d.cortafuegos.registro().height == 0 and ctx_d.cortafuegos.violaciones() == 0


def test_f12_01_existe_con_su_csv_de_respaldo_y_es_identica_en_otra_carpeta(decisiones_corridas, pipeline_sintetico, tmp_path):
    ctx_d, _ = decisiones_corridas
    idx = ctx_d.figuras.indice().filter(pl.col("id_figura") == "F12_01")
    assert idx.height == 1 and idx["csv"][0] == "decisiones_evidencia.csv" and idx["tarea"][0] == "12" and idx["uso_interno"][0] == "no"
    png = ctx_d.dir_figuras / "F12_01.png"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and "verde, amarillo, rojo y gris" in idx["descripcion_alternativa"][0]
    otro = copiar_tablas(pipeline_sintetico.ctx, tmp_path, con_figuras=True)
    decisiones.ejecutar(otro, cascada=pipeline_sintetico.r7["cascada"])
    for nombre in ("t25_hipotesis_evaluadas.csv", "decisiones_evidencia.csv", "supuestos_verificados.csv"):
        assert (ctx_d.dir_tablas / nombre).read_bytes() == (otro.dir_tablas / nombre).read_bytes(), nombre
    assert png.read_bytes() == (otro.dir_figuras / "F12_01.png").read_bytes()


def test_ejecutar_en_modo_completo_evalua_las_hipotesis_que_requieren_datos_completos(pipeline_sintetico, tmp_path):
    ctx_c = copiar_tablas(pipeline_sintetico.ctx, tmp_path, etiqueta="completo")
    r = decisiones.ejecutar(ctx_c)
    h = r["hipotesis"]
    assert ctx_c.etiqueta == "completo" and decisiones.MOTIVO_REQUIERE_COMPLETO not in h["motivo"].to_list()
    assert set(r["supuestos"]["nivel"].to_list()) == {"completo"}
    assert all(decisiones.CIERRE_EXTRACTO not in x for x in r["decisiones"]["que_falta_con_datos_completos"].to_list())


def test_ejecutar_rechaza_un_registro_que_no_concuerda_con_el_protocolo(monkeypatch, pipeline_sintetico, tmp_path):
    ctx_d = copiar_tablas(pipeline_sintetico.ctx, tmp_path)
    original = config.cargar_hipotesis
    monkeypatch.setattr(config, "cargar_hipotesis", lambda ruta=None: original().filter(pl.col("id") != "H07"))
    with pytest.raises(ValueError, match="no es válido"):
        decisiones.ejecutar(ctx_d)


# ======================================================================================================================================== informe
def test_el_informe_de_decisiones_tiene_tres_secciones_la_figura_y_un_parrafo_por_decision(decisiones_corridas):
    ctx_d, _ = decisiones_corridas
    inf = Informe(ctx_d, "t25_prueba_decisiones.md", "Prueba de las decisiones")
    decisiones.agregar_secciones(inf, ctx_d, 3)
    md = inf.guardar().read_text(encoding="utf8")
    assert md.count("\n## ") == decisiones.N_SECCIONES == 3
    for titulo in ("## 3. Evidencia por decisión", "## 4. Detalle por decisión", "## 5. Supuestos verificados"):
        assert titulo in md
    assert "![" in md and "F12_01" in md and "decisiones_evidencia.csv" in md and "supuestos_verificados.csv" in md and "{" not in md and "}" not in md
    for n in range(1, 18):
        assert f"Decisión {n}, " in md
    d = ctx_d.leer_tabla("decisiones_evidencia.csv", columnas_texto=decisiones.COLUMNAS_TEXTO_DECISIONES)
    assert d.filter(pl.col("n") == 8)["veredicto"][0] in md and d.filter(pl.col("n") == 8)["evidencia"][0] in md and "pendiente de decisión humana" in md
    cifras = ctx_d.cifras_informe().filter(pl.col("informe") == "t25_prueba_decisiones.md")
    assert cifras.height > 17 * 8 and (cifras["csv"] == "").sum() == 0 and (cifras["columna"] == "").sum() == 0
    escalares = cifras.filter(pl.col("formato") != "tabla")
    assert set(escalares["csv"].to_list()) <= {"t25_resumen_evidencia.csv", "decisiones_evidencia.csv"}


def test_cada_cifra_escalar_del_informe_se_reproduce_desde_su_csv(decisiones_corridas):
    from eda.informes import leer_cifra

    ctx_d, _ = decisiones_corridas
    inf = Informe(ctx_d, "t25_prueba_decisiones2.md", "Prueba")
    decisiones.agregar_secciones(inf, ctx_d, 1)
    inf.guardar()
    import json

    for c in ctx_d.cifras_informe().filter((pl.col("informe") == "t25_prueba_decisiones2.md") & (pl.col("formato") != "tabla")).iter_rows(named=True):
        t = ctx_d.leer_tabla(c["csv"], columnas_texto=decisiones.COLUMNAS_TEXTO_DECISIONES + ["id"])
        cifra_ = leer_cifra(t, c["csv"], c["columna"], **json.loads(c["filtro"]))
        assert cifra_.texto(c["formato"]) == c["texto"], c["marcador"]


def test_el_informe_de_decisiones_con_la_numeracion_desplazada(decisiones_corridas):
    ctx_d, _ = decisiones_corridas
    inf = Informe(ctx_d, "t25_prueba_decisiones3.md", "Prueba")
    decisiones.agregar_secciones(inf, ctx_d, 7)
    md = inf.guardar().read_text(encoding="utf8")
    assert "## 7. Evidencia por decisión" in md and "## 8. Detalle por decisión" in md and "## 9. Supuestos verificados" in md


# FIN-t12dec-v2
