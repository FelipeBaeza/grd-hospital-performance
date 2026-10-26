"""Tarea 12: ledger de cifras documentadas frente a lo medido (reglas por tipo de cifra con cuentas a mano, bordes, oráculos e integración sintética)."""

from __future__ import annotations

import dataclasses
import math

import polars as pl
import pytest
from scipy import stats as sps

from ayudas_t12 import FILAS_POR_ANIO, cifra_fila, cifras_df, copiar_tablas, metrica, metricas_df, reglas
from eda import descripcion, ledger, privacidad
from eda.contexto import Contexto
from eda.informes import Informe


def _ev(valor, medido, *, tipo="proporcion", unidad="porcentaje", d=2, alcance="2023", req="no", clave="m.a", ic=None, alc_m="", nivel="extracto", id="C900",
        doc="Fase 2", filas=None, **kw):
    """Evalúa una cifra con UNA métrica de clave ``m.a`` (atajo de las pruebas)."""
    f = cifra_fila(id, valor=valor, tipo=tipo, unidad=unidad, decimales=d, alcance=alcance, requiere_resultado=req, clave=clave, documento=doc)
    m = {clave: metrica(medido, ic=ic, alcance=alc_m)}
    return ledger.evaluar_cifra(f, m, reglas(nivel=nivel, **kw), FILAS_POR_ANIO if filas is None else filas)


def _conteo(valor, medido, **kw):
    kw.setdefault("tipo", "absoluto")
    kw.setdefault("unidad", "conteo")
    kw.setdefault("d", 0)
    return _ev(valor, medido, **kw)


# ============================================================================================================================ redondeo y alcances
def test_redondear_es_de_mitad_hacia_arriba_sobre_la_representacion_decimal():
    assert ledger.redondear(2.575, 2) == 2.58
    assert ledger.redondear(2.675, 2) == 2.68                       # round() de Python da 2,67: aquí el empate sube
    assert ledger.redondear(2.585, 2) == 2.59
    assert ledger.redondear(-2.575, 2) == -2.58                     # lejos de cero
    assert ledger.redondear(0.5, 0) == 1.0 and ledger.redondear(1.5, 0) == 2.0 and ledger.redondear(2.5, 0) == 3.0
    assert ledger.redondear(5808536.0, 0) == 5808536.0
    assert ledger.redondear(2.574999, 2) == 2.57


def test_cifras_significativas_no_cuentan_los_ceros_finales():
    assert [ledger.cifras_significativas(x) for x in (970000, 22, 1400, 0.5, 0, 1234, 100, 9500)] == [2, 2, 2, 1, 1, 4, 1, 2]


def test_redondear_a_cifras_significativas():
    assert ledger.redondear_cifras(968089, 2) == 970000
    assert ledger.redondear_cifras(965000, 2) == 970000             # empate hacia arriba
    assert ledger.redondear_cifras(1449, 2) == 1400 and ledger.redondear_cifras(1450, 2) == 1500
    assert ledger.redondear_cifras(21.6, 2) == 22 and ledger.redondear_cifras(0, 2) == 0


def test_anios_de_alcance_numerico_y_no_numerico():
    assert ledger.anios_de_alcance("2019") == (2019, 2019)
    assert ledger.anios_de_alcance("2019-2023") == (2019, 2023)
    assert ledger.anios_de_alcance(" 2019 - 2023 ") == (2019, 2023) and ledger.anios_de_alcance("2019_2023") == (2019, 2023)
    for texto in ("diccionario", "documental", "cohorte final", "mapeo", "", None, "2023-2019", "20-23"):
        assert ledger.anios_de_alcance(texto) is None, texto


def test_etiqueta_alcance_usa_el_primero_y_el_ultimo():
    assert ledger.etiqueta_alcance([2019]) == "2019"
    assert ledger.etiqueta_alcance([2023, 2019, 2021]) == "2019-2023"


def test_incluye_anio():
    assert ledger.incluye_anio("2019-2024", 2024) and ledger.incluye_anio("2024", 2024) and ledger.incluye_anio("2019-2024", 2019)
    assert not ledger.incluye_anio("2019-2023", 2024) and not ledger.incluye_anio("diccionario", 2024) and not ledger.incluye_anio(None, 2024)


def test_n_del_alcance_suma_los_anios_y_exige_todos():
    assert ledger.n_del_alcance("2019-2020", {2019: 10, 2020: 20, 2021: 30}) == 30
    assert ledger.n_del_alcance("2021", {2019: 10, 2020: 20, 2021: 30}) == 30
    assert ledger.n_del_alcance("2019-2022", {2019: 10, 2020: 20, 2021: 30}) is None      # falta 2022
    assert ledger.n_del_alcance("diccionario", FILAS_POR_ANIO) is None


def test_alternativas_de_clave_eda():
    assert ledger.alternativas_de("a;b") == ["a", "b"] and ledger.alternativas_de("") == [] and ledger.alternativas_de(" a ; ") == ["a"]
    assert ledger.alternativas_de(None) == []


def test_claves_de_valores_distintos():
    for k in ("cardinalidad.PREVISION", "hospitales.n_distintos.2023", "hospitales.n_distintos.maximo", "tipoalta.n_valores_distintos", "cardinalidad.COD_HOSPITAL"):
        assert ledger.es_clave_de_distintos(k), k
    for k in ("cardinalidad.PREVISION.chao1", "volumen.filas_leidas.2019", "hospitales.n_cumplen_volumen.2019"):
        assert not ledger.es_clave_de_distintos(k), k


def test_alcance_medido_prefiere_el_de_la_metrica_y_recorta_el_anio_expuesto_en_resultados():
    r = reglas()
    assert ledger._alcance_medido("", "2019-2024", "si", r) == "2019-2023"          # el cortafuegos garantiza que 2024 no aportó el desenlace
    assert ledger._alcance_medido("", "2019-2024", "no", r) == "2019-2024"
    assert ledger._alcance_medido("2019-2022", "2019-2024", "si", r) == "2019-2022"
    assert ledger._alcance_medido("", "2021-2023", "si", r) == "2021-2023"          # el documentado no llega a 2024: se queda
    assert ledger._alcance_medido("", "diccionario", "si", r) == "diccionario"


def test_las_reglas_salen_de_los_umbrales_del_contexto(ctx):
    r = ledger.Reglas.desde_contexto(ctx)
    assert (r.z_umbral, r.dif_relevante_pp, r.celda_minima, r.anio_expuesto, r.nivel) == (3.0, 0.5, 10, 2024, "extracto")
    assert r.z_95 == pytest.approx(1.959964) and r.anios_resultado == (2019, 2020, 2021, 2022, 2023)


# =============================================================================================================== proporciones en extracto (con intervalo)
def test_z_exactamente_igual_al_umbral_no_discrepa():
    r = _ev(2.0, 5.0, ic=(4.0, 6.0), z_95=1.0)                       # error estándar 1: z = 3 exacto
    assert r["z"] == 3.0 and r["estado"] == "CONFIRMA" and r["magnitud"] == "no aplica"


def test_z_sobre_el_umbral_discrepa_con_magnitud_relevante():
    r = _ev(2.0, 5.5, ic=(4.5, 6.5), z_95=1.0)                       # error estándar 1: z = 3,5
    assert r["estado"] == "DISCREPA" and r["z"] == 3.5 and r["magnitud"] == "relevante"
    assert r["dif_abs"] == 3.5 and r["dif_rel"] == 1.75              # relativa al documentado y no al medido
    assert r["valor_medido"] == 5.5 and r["tabla_origen"] == "t.csv" and (r["ic_bajo"], r["ic_alto"]) == (4.5, 6.5)


def test_el_error_estandar_sale_del_intervalo_entre_dos_z_95():
    # IC de 4,0 a 6,0 con z_95 real: se = 2 / (2 * 1,959964); z = 3 / se
    r = _ev(2.0, 5.0, ic=(4.0, 6.0))
    assert r["z"] == pytest.approx(3.0 * 2 * 1.959964 / 2.0)
    assert r["z"] == pytest.approx(5.879892)                          # con el factor 2 omitido saldría 2,94 y no discreparía


def test_z_negativo_se_evalua_por_valor_absoluto():
    assert _ev(8.0, 5.0, ic=(4.0, 6.0), z_95=1.0)["estado"] == "CONFIRMA"                        # z = -3 exacto
    r = _ev(8.0, 4.9, ic=(3.9, 5.9), z_95=1.0)
    assert r["estado"] == "DISCREPA" and r["z"] < -3.0 and r["z"] == pytest.approx(-3.1)


def test_magnitud_relevante_desde_medio_punto_porcentual_inclusive():
    exacto = _ev(5.0, 5.5, ic=(5.49, 5.51), z_95=1.0)               # z enorme, diferencia de 0,5 puntos exactos
    menor = _ev(5.0, 5.4, ic=(5.39, 5.41), z_95=1.0)
    assert exacto["estado"] == "DISCREPA" and exacto["magnitud"] == "relevante"
    assert menor["estado"] == "DISCREPA" and menor["magnitud"] == "despreciable"


NOTA_REDONDEO = "con la tolerancia de redondeo de la cifra documentada (media unidad del último decimal)"


def test_una_discrepancia_que_la_tolerancia_de_redondeo_absorbe_lo_dice_en_el_motivo_sin_cambiar_el_estado():
    # documentado 4,08 (dos decimales: el valor verdadero puede estar a 0,005), medido 4,12 y error estándar 0,0125: z = 3,2 discrepa; con 0,035 de diferencia z = 2,8
    r = _ev(4.08, 4.12, unidad="dx", d=2, ic=(4.1075, 4.1325), z_95=1.0)
    assert r["estado"] == "DISCREPA" and r["z"] == pytest.approx(3.2)
    assert f"; {NOTA_REDONDEO} z = 2,80 y no discreparía" in r["motivo"] and r["motivo"].endswith("resultado indicativo")
    # una discrepancia que sobrevive a la tolerancia no la lleva, ni tampoco una confirmación
    assert NOTA_REDONDEO not in _ev(4.08, 4.20, unidad="dx", d=2, ic=(4.1875, 4.2125), z_95=1.0)["motivo"]                         # z = 9,6 y con la tolerancia 9,2
    assert NOTA_REDONDEO not in _ev(4.08, 4.10, unidad="dx", d=2, ic=(4.0875, 4.1125), z_95=1.0)["motivo"]                         # z = 1,6: confirma
    # sin decimales la media unidad es 0,5: documentado 38, medido 39,2, error estándar 0,35 -> z = 3,43 y con la tolerancia 2,0
    r0 = _ev(38.0, 39.2, d=0, unidad="porcentaje", ic=(38.85, 39.55), z_95=1.0)
    assert r0["estado"] == "DISCREPA" and f"{NOTA_REDONDEO} z = 2,00 y no discreparía" in r0["motivo"]


def test_la_nota_de_redondeo_tambien_sale_con_el_error_estandar_binomial_y_no_con_afirmaciones_deterministas():
    # 10.000 egresos y documentado 10,0 %: z = 91 / 30 = 3,033; con 0,05 puntos de tolerancia la diferencia baja de 0,91 a 0,86 y z a 2,87
    r = _ev(10.0, 10.91, d=1, filas={2023: 10_000})
    assert r["estado"] == "DISCREPA" and r["z"] == pytest.approx(91 / 30)
    assert f"{NOTA_REDONDEO} z = 2,87 y no discreparía" in r["motivo"]
    assert NOTA_REDONDEO not in _ev(10.0, 10.2, d=1, filas={2023: 1_000_000})["motivo"]                                           # z = 6,67 y con la tolerancia 5,0
    determinista = _ev(0.0, 0.5, d=1, filas={2023: 10_000})                                                                        # documentado 0,0: afirmación determinista, sin z
    assert determinista["estado"] == "DISCREPA" and determinista["z"] is None and NOTA_REDONDEO not in determinista["motivo"]


def test_la_nota_de_redondeo_es_una_funcion_pura_con_sus_bordes():
    nota = ledger._nota_redondeo
    assert nota("DISCREPA", 3.2, 4.12, 4.08, 2, 3.0) == f"; {NOTA_REDONDEO} z = 2,80 y no discreparía"
    assert nota("CONFIRMA", 3.2, 4.12, 4.08, 2, 3.0) == "" and nota("DISCREPA", None, 4.12, 4.08, 2, 3.0) == "" and nota("DISCREPA", 3.2, 4.08, 4.08, 2, 3.0) == ""
    assert nota("DISCREPA", -3.2, 4.04, 4.08, 2, 3.0).endswith("z = 2,80 y no discreparía")                                       # el signo de z no importa
    assert nota("DISCREPA", 3.2, 4.12, 4.08, 2, 2.81).endswith("z = 2,80 y no discreparía")                                         # z con tolerancia apenas bajo el umbral: no discreparía
    assert nota("DISCREPA", 3.2, 4.12, 4.08, 2, 2.79) == ""                                                                         # apenas sobre el umbral: la discrepancia sobrevive
    assert nota("DISCREPA", 3.2, 4.082, 4.08, 2, 3.0).endswith("z = 0,00 y no discreparía")                                         # la diferencia cabe dentro de la media unidad


def test_unidad_dias_y_dx_con_intervalo_usa_z_sin_magnitud():
    for unidad in ("dias", "dx"):
        r = _ev(4.0, 5.5, unidad=unidad, ic=(5.4, 5.6), z_95=1.0)
        assert r["estado"] == "DISCREPA" and r["magnitud"] == "no aplica" and r["z"] == pytest.approx(15.0)       # error estándar 0,1 y diferencia 1,5
        assert _ev(4.0, 4.05, unidad=unidad, ic=(3.95, 4.15), z_95=1.0)["estado"] == "CONFIRMA"


def test_unidad_dias_sin_intervalo_es_no_verificable():
    r = _ev(3.0, 4.0, unidad="dias", d=1)
    assert r["estado"] == "NO VERIFICABLE" and ledger.motivo_base(r["motivo"]) == ledger.MOTIVOS["sin_error_estandar"] and r["valor_medido"] == 4.0
    assert r["dif_abs"] is None and r["z"] is None


def test_clave_sin_error_estandar_es_no_verificable_aunque_traiga_intervalo():
    r = _ev(38, 40.8, d=0, clave="resultado.caida_mortalidad_2021_2023_pct", alcance="2021-2023", req="si", ic=(39.0, 42.0))
    assert r["estado"] == "NO VERIFICABLE" and ledger.motivo_base(r["motivo"]) == ledger.MOTIVOS["sin_error_estandar"]


# ============================================================================================================== proporciones en extracto (binomial)
def test_binomial_con_z_exactamente_tres_no_discrepa():
    # 10.000 egresos, documentado 10,0 % (p = 0,1): error estándar 30 casos; 10,9 % son 1.090 casos, 90 sobre lo esperado: z = 3 exacto
    r = _ev(10.0, 10.9, d=1, filas={2023: 10_000})
    assert r["z"] == 3.0 and r["estado"] == "CONFIRMA" and r["magnitud"] == "no aplica"
    assert "10.000" in r["motivo"]                                               # el n usado queda a la vista, rotulado como cota superior
    assert "cota superior" in r["motivo"]


def test_binomial_con_z_apenas_sobre_tres_discrepa_y_es_relevante():
    r = _ev(10.0, 10.91, d=1, filas={2023: 10_000})                              # 1.091 casos: z = 91 / 30 = 3,033
    assert r["estado"] == "DISCREPA" and r["z"] == pytest.approx(91 / 30) and r["magnitud"] == "relevante"
    assert r["dif_abs"] == pytest.approx(0.91) and r["dif_rel"] == pytest.approx(0.091)


def test_binomial_con_muestra_grande_discrepa_con_diferencia_despreciable():
    r = _ev(10.0, 10.2, d=1, filas={2023: 1_000_000})                           # error estándar de 300 casos (0,03 puntos); 2.000 casos de diferencia: z = 6,67
    assert r["estado"] == "DISCREPA" and r["z"] == pytest.approx(2000 / 300) and r["magnitud"] == "despreciable"


def test_los_conteos_equivalentes_reproducen_el_z_directo_de_la_formula():
    casos = [(89.62, 2, 89.70, 250_000), (2.94, 2, 3.10, 250_000), (87.9, 1, 88.09, 299_994), (15.32, 2, 15.14, 299_994), (0.76, 2, 0.755, 250_000)]
    for doc, d, med, n in casos:
        r = _ev(doc, med, d=d, filas={2023: n})
        p = round(doc * 10**d) / 10 ** (d + 2)
        sin_med = round(med / 100.0 * n)
        z_directo = (sin_med - p * n) / math.sqrt(n * p * (1 - p))
        assert abs(r["z"] - z_directo) < 1e-9, (doc, med)
        # la decisión coincide con el valor p de la normal (oráculo de scipy): discrepa si 2 * sf(|z|) < 2 * sf(3)
        assert (r["estado"] == "DISCREPA") == (2 * sps.norm.sf(abs(z_directo)) < 2 * sps.norm.sf(3.0)), (doc, med)


def test_documentado_cero_es_determinista_en_el_extracto():
    assert _ev(0.0, 0.0, d=1)["estado"] == "CONFIRMA"
    r = _ev(0.0, 0.0004, d=1, filas={2023: 50_000})                              # 0,2 casos se redondean a cero
    assert r["estado"] == "CONFIRMA"
    r = _ev(0.0, 0.01, d=1, filas={2023: 50_000})                                # cinco casos donde el documento afirma ninguno
    assert r["estado"] == "DISCREPA" and r["z"] is None and "determinista" in r["motivo"]


def test_documentado_cien_es_determinista_en_el_extracto():
    assert _ev(100.0, 100.0, d=1, filas={2023: 5_000})["estado"] == "CONFIRMA"
    assert _ev(100.0, 99.9, d=1, filas={2023: 5_000})["estado"] == "DISCREPA"


def test_evento_raro_en_el_extracto_no_se_evalua_si_el_valor_p_no_es_pequeno():
    # documentado 0,01 % (p = 1e-4) con 10.000 filas: se esperan 1 caso, menos de cinco
    r = _ev(0.01, 0.01, filas={2023: 10_000})
    assert r["estado"] == "NO VERIFICABLE" and ledger.motivo_base(r["motivo"]) == ledger.MOTIVOS["evento_raro"]
    r = _ev(0.01, 1.0, filas={2023: 10_000})                                     # 100 casos contra 1 esperado: valor p mínimo
    assert r["estado"] == "DISCREPA"


def test_alcance_sin_anios_o_sin_tamano_es_no_verificable():
    r = _ev(10.0, 10.0, d=1, alcance="cohorte final", filas={2023: 10_000})
    assert r["estado"] == "NO VERIFICABLE" and ledger.motivo_base(r["motivo"]) == ledger.MOTIVOS["sin_tamano"]
    r = _ev(10.0, 10.0, d=1, filas={2022: 10_000})                               # falta el año 2023
    assert ledger.motivo_base(r["motivo"]) == ledger.MOTIVOS["sin_tamano"]


def test_intervalo_de_ancho_cero_cae_al_error_estandar_binomial():
    r = _ev(10.0, 10.9, d=1, ic=(10.9, 10.9), filas={2023: 10_000})
    assert r["estado"] == "CONFIRMA" and r["z"] == 3.0 and "cota superior" in r["motivo"]
    r = _ev(10.0, 10.9, d=1, ic=(10.9, 10.9), filas={})
    assert ledger.motivo_base(r["motivo"]) == ledger.MOTIVOS["sin_tamano"]


# ======================================================================================================================== proporciones con datos completos
def test_completo_proporcion_exige_igualdad_al_redondeo_de_los_decimales():
    assert _ev(2.58, 2.575, nivel="completo")["estado"] == "CONFIRMA"            # 2,575 sube a 2,58
    assert _ev(2.68, 2.675, nivel="completo")["estado"] == "CONFIRMA"            # el round de Python lo bajaría a 2,67
    r = _ev(2.57, 2.575, nivel="completo")
    assert r["estado"] == "DISCREPA" and r["dif_abs"] == pytest.approx(0.005) and r["magnitud"] == "despreciable" and r["z"] is None


def test_completo_proporcion_magnitud_relevante_y_unidades_sin_magnitud():
    assert _ev(2.0, 2.6, nivel="completo")["magnitud"] == "relevante"
    assert _ev(2.0, 2.5, nivel="completo")["magnitud"] == "relevante"           # diferencia exacta de medio punto
    assert _ev(2.0, 2.49, nivel="completo")["magnitud"] == "despreciable"
    r = _ev(3.0, 4.0, unidad="dias", d=1, nivel="completo")
    assert r["estado"] == "DISCREPA" and r["magnitud"] == "no aplica"
    assert _ev(4.7, 4.74, unidad="dias", d=1, nivel="completo")["estado"] == "CONFIRMA"


def test_completo_no_usa_el_error_estandar():
    r = _ev(2.0, 2.0001, nivel="completo", ic=(1.0, 3.0))
    assert r["estado"] == "CONFIRMA" and r["z"] is None


# ===================================================================================================================================== conteos absolutos
def test_conteo_de_la_poblacion_completa_no_es_comparable_en_el_extracto():
    r = _conteo(1151475, 49999, alcance="2019")
    assert r["estado"] == "NO VERIFICABLE" and ledger.motivo_base(r["motivo"]) == ledger.MOTIVOS["poblacion_completa"]
    assert r["valor_medido"] == 49999 and r["n_medido"] == 49999 and r["n_doc"] == 1151475 and r["dif_abs"] is None and r["tabla_origen"] == "t.csv"


def test_conteo_completo_exige_igualdad_al_redondeo():
    assert _conteo(5808536, 5808536, nivel="completo")["estado"] == "CONFIRMA"
    r = _conteo(5808536, 5808537, nivel="completo")
    assert r["estado"] == "DISCREPA" and r["dif_abs"] == 1 and r["dif_rel"] == pytest.approx(1 / 5808536)
    assert _conteo(970000, 970000.4, nivel="completo")["estado"] == "CONFIRMA"      # el redondeo a cero decimales lo iguala


def test_dif_rel_se_divide_por_el_documentado_y_es_nula_si_el_documentado_es_cero():
    r = _conteo(100, 125, nivel="completo")
    assert r["dif_abs"] == 25 and r["dif_rel"] == 0.25                               # 25 / 100 y no 25 / 125
    assert _conteo(0, 12, nivel="completo")["dif_rel"] is None


def test_hipotesis_de_origen_en_extracto_exige_igualdad_exacta():
    f = dict(id="C113", doc="Fase 1", alcance="2019-2024", clave="cascada.n_ex02")
    igual = _conteo(7760, 7760, **f)
    assert igual["estado"] == "CONFIRMA" and "hipótesis" in igual["motivo"] and igual["dif_abs"] == 0
    distinto = _conteo(75, 215, **{**f, "id": "C112", "clave": "cascada.n_ex01"})
    assert distinto["estado"] == "DISCREPA" and distinto["dif_abs"] == 140
    # una cifra de la Fase 2 con el mismo conteo NO recibe la hipótesis: es un conteo de la población
    assert _conteo(5073, 215, id="C103", alcance="2019-2024", clave="cascada.n_ex01")["estado"] == "NO VERIFICABLE"


def test_claves_de_valores_distintos_son_cota_inferior_en_el_extracto():
    mayor = _conteo(65, 68, alcance="2023", clave="hospitales.n_distintos.2023")
    assert mayor["estado"] == "DISCREPA" and "cota inferior" in mayor["motivo"]
    assert _conteo(68, 68, alcance="2023", clave="hospitales.n_distintos.2023")["estado"] == "CONFIRMA"
    menor = _conteo(158, 90, alcance="2019-2024", clave="cardinalidad.ESPECIALIDAD_MEDICA")
    assert menor["estado"] == "NO VERIFICABLE" and ledger.motivo_base(menor["motivo"]) == ledger.MOTIVOS["valores_distintos"]
    # la clave del estimador Chao1 no es un conteo de valores distintos
    assert _conteo(158, 90, clave="cardinalidad.ESPECIALIDAD_MEDICA.chao1")["estado"] == "NO VERIFICABLE"
    assert ledger.motivo_base(_conteo(158, 90, clave="cardinalidad.ESPECIALIDAD_MEDICA.chao1")["motivo"]) == ledger.MOTIVOS["poblacion_completa"]


def test_estructura_exige_igualdad_redondeada_tambien_en_extracto():
    assert _conteo(129, 129, tipo="estructura", alcance="2019-2024", clave="estructura.columnas")["estado"] == "CONFIRMA"
    assert _conteo(6, 6.4, tipo="estructura", alcance="2019-2024", clave="estructura.n_archivos")["estado"] == "CONFIRMA"      # igualdad tras redondear a cero decimales
    r = _conteo(2019, 2024, tipo="estructura", unidad="anio", alcance="2019-2024", clave="estructura.primer_anio_id_beneficiario")
    assert r["estado"] == "DISCREPA" and "2024" in r["motivo"] and "2.024" not in r["motivo"]
    assert r["n_doc"] is None                                                       # una estructura no cuenta casos: no se protege


def test_estructura_en_modo_completo_tambien_es_igualdad():
    assert _conteo(41, 41, tipo="estructura", alcance="diccionario", nivel="completo")["estado"] == "CONFIRMA"
    assert _conteo(41, 42, tipo="estructura", alcance="diccionario", nivel="completo")["estado"] == "DISCREPA"


# ============================================================================================================================== comparaciones especiales
def test_mas_de_en_completo_exige_estrictamente_mas():
    base = dict(id="C125", alcance="2019-2024", clave="cardinalidad.DIAGNOSTICO1", nivel="completo")
    assert _conteo(9500, 9501, **base)["estado"] == "CONFIRMA"
    assert _conteo(9500, 9500, **base)["estado"] == "DISCREPA"                       # "más de 9.500" no incluye 9.500
    assert _conteo(9500, 9000, **base)["estado"] == "DISCREPA"


def test_mas_de_en_extracto_solo_confirma_si_ya_lo_supera():
    base = dict(id="C125", alcance="2019-2024", clave="cardinalidad.DIAGNOSTICO1")
    r = _conteo(9500, 9600, **base)
    assert r["estado"] == "CONFIRMA" and r["dif_abs"] == 100
    r = _conteo(9500, 6178, **base)
    assert r["estado"] == "NO VERIFICABLE" and ledger.motivo_base(r["motivo"]) == ledger.MOTIVOS["mas_de_extracto"]
    assert _conteo(9500, 9500, **base)["estado"] == "NO VERIFICABLE"                  # igual no supera a la cota: no confirma


def test_aproximado_en_completo_compara_a_las_cifras_significativas_del_documentado():
    base = dict(id="C015", alcance="2019-2024", clave="volumen.filas_leidas.promedio", nivel="completo")
    assert _conteo(970000, 968089, **base)["estado"] == "CONFIRMA"
    assert _conteo(970000, 965000, **base)["estado"] == "CONFIRMA"                    # 965.000 se redondea hacia arriba a 970.000
    assert _conteo(970000, 964999, **base)["estado"] == "DISCREPA"
    assert _conteo(970000, 960000, **base)["estado"] == "DISCREPA"
    r = _conteo(22, 22, id="C127", tipo="estructura", alcance="2019-2024", clave="cardinalidad.GRUPO_CLINICO", nivel="completo")
    assert r["estado"] == "CONFIRMA" and "2 cifras significativas" in r["motivo"]


def test_aproximado_en_extracto_es_no_verificable():
    r = _conteo(970000, 49999, id="C015", alcance="2019-2024", clave="volumen.filas_leidas.promedio")
    assert r["estado"] == "NO VERIFICABLE" and ledger.motivo_base(r["motivo"]) == ledger.MOTIVOS["aproximada_poblacion"]
    r = _conteo(1400, 1400, id="C126", alcance="2019-2024", clave="cardinalidad.CIE10_3C")
    assert r["estado"] == "NO VERIFICABLE"                                            # ni siquiera una coincidencia confirma una aproximación en el extracto


# ======================================================================================================================================= año expuesto
def _fila_cota(doc, id="C114", **kw):
    return cifra_fila(id, valor=doc, tipo="absoluto", unidad="conteo", decimales=0, alcance="2019-2024", requiere_resultado="si", clave="cascada.n_ex03_pipeline",
                      documento=kw.pop("documento", "Fase 1"))


def _metricas_cota(lo=19899, cota=2670):
    return {
        "cascada.n_ex03_pipeline": metrica(lo, alcance="2019-2023", tabla="t23_cascada_pipeline.csv"),
        "cascada.n_ex03_pipeline.cota_sup_2024": metrica(cota, alcance="2024", tabla="t23_cascada_pipeline.csv"),
    }


# ``dif_abs`` es siempre |medido - documentado| (19.899 contra el documentado); el estado se juzga contra el intervalo [19.899, 22.569]
@pytest.mark.parametrize("doc, estado, dif", [(19899, "CONFIRMA", 0), (22569, "CONFIRMA", 2670), (22303, "CONFIRMA", 2404), (22570, "DISCREPA", 2671), (19898, "DISCREPA", 1),
                                              (89438, "DISCREPA", 69539)])
def test_la_cota_del_anio_expuesto_incluye_los_bordes(doc, estado, dif):
    r = ledger.evaluar_cifra(_fila_cota(doc), _metricas_cota(), reglas(), FILAS_POR_ANIO)
    assert r["estado"] == estado and r["dif_abs"] == dif and r["dif_rel"] == pytest.approx(dif / doc)
    assert r["valor_medido"] == 19899 and r["alcance_medido"] == "2019-2023" and r["clave_usada"] == "cascada.n_ex03_pipeline"
    assert "19.899" in r["motivo"] and "22.569" in r["motivo"]


def test_una_cifra_dentro_de_la_cota_publica_la_diferencia_real_y_no_la_distancia_al_intervalo():
    # C114 de la corrida real: documentado 22.303, medido 19.899 en 2019 a 2023 y cota de 2.670; está dentro del intervalo pero difiere en 2.404 y no en cero
    r = ledger.evaluar_cifra(_fila_cota(22303), _metricas_cota(), reglas(), FILAS_POR_ANIO)
    assert r["estado"] == "CONFIRMA" and r["dif_abs"] == 22303 - 19899 == 2404 and r["dif_rel"] == pytest.approx(2404 / 22303)
    assert "cae dentro" in r["motivo"]


def test_cota_en_extracto_solo_aplica_a_la_hipotesis_de_origen():
    r = ledger.evaluar_cifra(_fila_cota(431832, id="C105", documento="Fase 2"), _metricas_cota(), reglas(), FILAS_POR_ANIO)
    assert r["estado"] == "NO VERIFICABLE" and ledger.motivo_base(r["motivo"]) == ledger.MOTIVOS["poblacion_completa"]
    assert r["valor_medido"] == 19899                                                  # el valor medido de apoyo se informa igual
    # con los datos completos la misma cifra sí se compara con el intervalo
    comp = ledger.evaluar_cifra(_fila_cota(431832, id="C105", documento="Fase 2"), _metricas_cota(lo=431000, cota=1000), reglas(nivel="completo"), FILAS_POR_ANIO)
    assert comp["estado"] == "CONFIRMA" and "hipótesis" not in comp["motivo"]


def test_sin_cota_o_con_cota_suprimida_es_no_verificable():
    solo = {"cascada.n_censurados": metrica(14450, alcance="2019-2023")}
    f = cifra_fila("C115", valor=89438, tipo="absoluto", unidad="conteo", decimales=0, alcance="2019-2024", requiere_resultado="si", clave="cascada.n_censurados", documento="Fase 1")
    r = ledger.evaluar_cifra(f, solo, reglas(), FILAS_POR_ANIO)
    assert r["estado"] == "NO VERIFICABLE" and ledger.motivo_base(r["motivo"]) == ledger.MOTIVOS["sin_cota"] and r["valor_medido"] == 14450
    suprimida = {**_metricas_cota(), "cascada.n_ex03_pipeline.cota_sup_2024": metrica(None, alcance="2024", nota="[suprimido: menos de diez casos]")}
    r = ledger.evaluar_cifra(_fila_cota(22303), suprimida, reglas(), FILAS_POR_ANIO)
    assert r["estado"] == "NO VERIFICABLE" and ledger.motivo_base(r["motivo"]) == ledger.MOTIVOS["sin_cota"]


def test_proporcion_que_requiere_el_desenlace_de_2024_no_se_compara_nunca():
    r = _ev(2.94, 3.10, d=2, alcance="2019-2024", req="si", ic=(3.03, 3.17), alc_m="2019-2023")
    assert r["estado"] == "NO VERIFICABLE" and ledger.motivo_base(r["motivo"]) == ledger.MOTIVOS["sin_cota"]
    assert r["valor_medido"] == 3.10 and r["alcance_medido"] == "2019-2023" and r["z"] is None and r["dif_abs"] is None
    # una cota con el mismo nombre no cambia nada: la cota solo aplica a conteos absolutos
    f = cifra_fila("C900", valor=2.94, alcance="2019-2024", requiere_resultado="si", clave="m.a")
    m = {"m.a": metrica(3.10, alcance="2019-2023"), "m.a.cota_sup_2024": metrica(100)}
    assert ledger.evaluar_cifra(f, m, reglas(), FILAS_POR_ANIO)["estado"] == "NO VERIFICABLE"


def test_el_desenlace_de_2024_no_se_mide_aunque_la_corrida_sea_completa():
    r = _ev(2.94, 3.10, d=2, alcance="2019-2024", req="si", alc_m="2019-2023", nivel="completo")
    assert r["estado"] == "NO VERIFICABLE"


def test_cifra_sin_desenlace_con_alcance_de_2024_si_se_compara():
    r = _ev(15.32, 15.14, alcance="2019-2024", req="no", alc_m="2019-2024", filas={a: 50_000 for a in range(2019, 2025)})
    assert r["estado"] in ("CONFIRMA", "DISCREPA") and r["alcance_medido"] == "2019-2024"


def test_cifra_que_requiere_desenlace_y_no_llega_a_2024_se_compara_normalmente():
    r = _ev(3.83, 3.65, alcance="2020", req="si", ic=(3.49, 3.82), alc_m="")
    assert r["estado"] in ("CONFIRMA", "DISCREPA") and r["alcance_medido"] == "2020"


def test_alcances_distintos_son_no_verificables():
    r = _ev(15.32, 15.14, alcance="2019-2024", req="no", alc_m="2019-2023")
    assert r["estado"] == "NO VERIFICABLE" and ledger.motivo_base(r["motivo"]) == ledger.MOTIVOS["alcances_distintos"]
    assert "2019-2023" in r["motivo"] and "2019-2024" in r["motivo"] and r["valor_medido"] == 15.14
    # alcances no numéricos no se comparan por años: sigue a la regla del tipo
    f = cifra_fila("C069", valor=31, tipo="estructura", unidad="conteo", decimales=0, alcance="diccionario", clave="elix.n_categorias")
    assert ledger.evaluar_cifra(f, {"elix.n_categorias": metrica(31, alcance="mapeo")}, reglas(), FILAS_POR_ANIO)["estado"] == "CONFIRMA"


# ================================================================================================================================== privacidad (1 a 9 casos)
def test_conteo_documentado_de_1_a_9_casos_no_se_compara_ni_se_publica():
    for doc in (1, 9):
        r = _conteo(doc, 0, alcance="2019-2024")
        assert r["estado"] == "NO VERIFICABLE" and r["motivo"] == ledger.MOTIVOS["pocos_casos"] and r["n_doc"] == doc
    assert _conteo(10, 0, alcance="2019-2024")["motivo"] == ledger.MOTIVOS["poblacion_completa"]        # diez casos ya se publican
    assert _conteo(0, 0, alcance="2019-2024", nivel="completo")["estado"] == "CONFIRMA"                  # el cero se informa


def test_conteo_medido_de_1_a_9_casos_tampoco_se_compara():
    r = _conteo(100, 9, alcance="2019-2024", nivel="completo")
    assert r["estado"] == "NO VERIFICABLE" and r["motivo"] == ledger.MOTIVOS["pocos_casos"] and r["n_medido"] == 9
    assert _conteo(100, 10, alcance="2019-2024", nivel="completo")["estado"] == "DISCREPA"


def test_la_proteccion_de_1_a_9_solo_aplica_a_conteos_absolutos():
    assert _conteo(1, 1, tipo="estructura", alcance="diccionario")["estado"] == "CONFIRMA"                 # una columna TARGET no es un caso
    assert _ev(5.0, 5.0, tipo="proporcion", unidad="porcentaje", ic=(4.9, 5.1))["n_doc"] is None


# ============================================================================================================================== cifras sin métrica
def test_cifra_sin_clave_o_sin_metrica():
    f = cifra_fila("C900", valor=5, tipo="absoluto", unidad="conteo", decimales=0, alcance="2019-2024", clave="")
    r = ledger.evaluar_cifra(f, {}, reglas(), FILAS_POR_ANIO)
    assert r["estado"] == "NO VERIFICABLE" and r["motivo"] == ledger.MOTIVOS["sin_clave"] and r["valor_medido"] is None and r["tabla_origen"] is None
    f2 = cifra_fila("C901", valor=50, tipo="absoluto", unidad="conteo", decimales=0, alcance="2019-2024", clave="no.existe")
    r2 = ledger.evaluar_cifra(f2, {}, reglas(), FILAS_POR_ANIO)
    assert r2["motivo"] == ledger.MOTIVOS["sin_metrica"] and r2["clave_usada"] is None


def test_cifra_de_resultado_de_2024_sin_metrica_explica_que_el_desenlace_no_se_lee():
    f = cifra_fila("C013", valor=2.46, alcance="2024", requiere_resultado="si", clave="resultado.mortalidad_pct.2024")
    r = ledger.evaluar_cifra(f, {}, reglas(), FILAS_POR_ANIO)
    assert r["estado"] == "NO VERIFICABLE" and r["motivo"] == ledger.MOTIVOS["sin_cota"]


def test_metrica_suprimida_o_nula_es_no_verificable_con_su_motivo():
    f = cifra_fila("C900", valor=80, tipo="absoluto", unidad="conteo", decimales=0, alcance="2019-2024", clave="m.a")
    r = ledger.evaluar_cifra(f, {"m.a": metrica(None, nota="[suprimido: menos de diez casos]", tabla="t22.csv")}, reglas(), FILAS_POR_ANIO)
    assert r["motivo"] == ledger.MOTIVOS["suprimida"] and r["tabla_origen"] == "t22.csv" and r["valor_medido"] is None and r["clave_usada"] == "m.a"
    r = ledger.evaluar_cifra(f, {"m.a": metrica(None, nota="sin datos en el año")}, reglas(), FILAS_POR_ANIO)
    assert ledger.motivo_base(r["motivo"]) == ledger.MOTIVOS["nula"] and "sin datos en el año" in r["motivo"]
    assert r["n_doc"] == 80                                                              # el conteo documentado queda registrado para protegerlo


# =================================================================================================================== no comparables y subestimadas
@pytest.mark.parametrize("cid, clave", [("C116", "cohorte.prevalencia_pct.fase1"), ("C117", "cohorte.prevalencia_pct.C.2023"), ("C089", "calidad.diagnostico_con_punto_pct")])
def test_las_cifras_con_definiciones_distintas_son_no_verificables_aunque_exista_la_metrica(cid, clave):
    f = cifra_fila(cid, valor=4.35, alcance="2023", requiere_resultado="si", clave=clave)
    sin = ledger.evaluar_cifra(f, {}, reglas(), FILAS_POR_ANIO)
    con = ledger.evaluar_cifra(f, {clave: metrica(4.35, ic=(4.30, 4.40), alcance="2023", tabla="t.csv")}, reglas(nivel="completo"), FILAS_POR_ANIO)
    for r in (sin, con):
        assert r["estado"] == "NO VERIFICABLE" and r["motivo"] == ledger.NO_COMPARABLES[cid]
    assert sin["valor_medido"] is None and con["valor_medido"] == 4.35 and con["tabla_origen"] == "t.csv" and con["dif_abs"] is None


def test_enlace_de_traslados_se_subestima_en_el_extracto_pero_se_compara_en_completo():
    f = cifra_fila("C119", valor=45.2, decimales=1, alcance="2023", requiere_resultado="si", clave="traslados.enlace_pct_30d.2023")
    m = {"traslados.enlace_pct_30d.2023": metrica(2.83, ic=(1.94, 4.12), alcance="2023")}
    ext = ledger.evaluar_cifra(f, m, reglas(), FILAS_POR_ANIO)
    assert ext["estado"] == "NO VERIFICABLE" and ext["motivo"] == ledger.SUBESTIMADAS_EN_EXTRACTO["C119"] and ext["valor_medido"] == 2.83 and ext["z"] is None
    comp = ledger.evaluar_cifra(f, {"traslados.enlace_pct_30d.2023": metrica(45.2, alcance="2023")}, reglas(nivel="completo"), FILAS_POR_ANIO)
    assert comp["estado"] == "CONFIRMA"


# ==================================================================================================================================== alternativas de clave
def _alt(a, b, doc=61.31):
    f = cifra_fila("C051", valor=doc, alcance="2019-2023", requiere_resultado="no", clave="a;b")
    m = {}
    if a is not None:
        m["a"] = metrica(a, ic=(a - 0.5, a + 0.5), alcance="2019-2023")
    if b is not None:
        m["b"] = metrica(b, ic=(b - 0.5, b + 0.5), alcance="2019-2023")
    return ledger.evaluar_cifra(f, m, reglas(), FILAS_POR_ANIO)


def test_gana_la_alternativa_que_confirma_aunque_no_sea_la_primera():
    r = _alt(67.84, 61.30)
    assert r["estado"] == "CONFIRMA" and r["clave_usada"] == "b" and r["valor_medido"] == 61.30
    assert "otras lecturas: a = 67,84" in r["motivo"]


def test_gana_la_primera_alternativa_si_confirma():
    r = _alt(61.30, 61.31)
    assert r["estado"] == "CONFIRMA" and r["clave_usada"] == "a"
    assert "otras lecturas: b = 61,31" in r["motivo"]


def test_si_ninguna_alternativa_confirma_discrepa_e_informa_las_demas():
    r = _alt(67.84, 62.84)
    assert r["estado"] == "DISCREPA" and r["clave_usada"] == "a" and r["valor_medido"] == 67.84
    assert "otras lecturas: b = 62,84" in r["motivo"]


def test_con_una_sola_alternativa_con_metrica_se_usa_esa():
    r = _alt(None, 61.30)
    assert r["estado"] == "CONFIRMA" and r["clave_usada"] == "b" and "otras lecturas" not in r["motivo"]
    assert _alt(None, None)["motivo"] == ledger.MOTIVOS["sin_metrica"]


def test_las_alternativas_no_verificables_informan_la_otra_lectura():
    f = cifra_fila("C051", valor=61.31, alcance="2019-2024", requiere_resultado="si", clave="a;b")
    m = {"a": metrica(67.84, alcance="2019-2023"), "b": metrica(62.84, alcance="2019-2023")}
    r = ledger.evaluar_cifra(f, m, reglas(), FILAS_POR_ANIO)
    assert r["estado"] == "NO VERIFICABLE" and r["clave_usada"] == "a" and r["valor_medido"] == 67.84
    assert ledger.motivo_base(r["motivo"]) == ledger.MOTIVOS["sin_cota"] and "otras lecturas: b = 62,84" in r["motivo"]


# =========================================================================================================================================== aritmética
def _pares_documentados():
    def c(cid, valor, clave, documento):
        return cifra_fila(cid, valor=valor, tipo="absoluto", unidad="conteo", decimales=0, alcance="2019-2024", clave=clave, documento=documento)

    return cifras_df(
        c("C103", 5073, "k.a", "Fase 2"), c("C112", 75, "k.a", "Fase 1"),
        c("C104", 146928, "k.b", "Fase 2"), c("C113", 7760, "k.b", "Fase 1"),
        c("C111", 354853, "k.c", "Fase 2"), c("C115", 89438, "k.c", "Fase 1"),
        c("C001", 1151475, "k.solo2", "Fase 2"),                                                          # sin par
        c("C078", 68, "k.solo1", "Fase 1"),                                                               # sin par
        c("C201", 1, "k.dos", "Fase 2"), c("C202", 2, "k.dos", "Fase 2"), c("C203", 3, "k.dos", "Fase 1"),    # par ambiguo: dos de la Fase 2
        c("C204", 100, "k.cero", "Fase 2"), c("C205", 0, "k.cero", "Fase 1"),                             # divisor cero
        cifra_fila("C128", valor=10, tipo="aritmetica", unidad="razon", decimales=0, alcance="documental", clave="doc.razon_minima_cascada"),
    )


def test_razones_fase2_sobre_fase1_con_cuentas_a_mano():
    pares = ledger.razones_fase2_fase1(_pares_documentados())
    assert [(p["clave"], p["id_fase2"], p["id_fase1"]) for p in pares] == [("k.a", "C103", "C112"), ("k.b", "C104", "C113"), ("k.c", "C111", "C115")]
    esperadas = {"k.a": 5073 / 75, "k.b": 146928 / 7760, "k.c": 354853 / 89438}
    assert {p["clave"]: p["razon"] for p in pares} == pytest.approx(esperadas)
    assert esperadas["k.a"] == pytest.approx(67.64) and esperadas["k.b"] == pytest.approx(18.934, abs=1e-3) and esperadas["k.c"] == pytest.approx(3.9676, abs=1e-4)


def test_aritmetica_compara_el_minimo_y_el_maximo_con_lo_documentado():
    pares = ledger.razones_fase2_fase1(_pares_documentados())
    fila_min = cifra_fila("C128", valor=10, tipo="aritmetica", unidad="razon", decimales=0, alcance="documental", clave="doc.razon_minima_cascada")
    fila_max = cifra_fila("C129", valor=20, tipo="aritmetica", unidad="razon", decimales=0, alcance="documental", clave="doc.razon_maxima_cascada")
    mn = ledger.evaluar_aritmetica(fila_min, pares, reglas())
    mx = ledger.evaluar_aritmetica(fila_max, pares, reglas())
    assert mn["valor_medido"] == pytest.approx(354853 / 89438) and mn["estado"] == "DISCREPA"
    assert mx["valor_medido"] == pytest.approx(5073 / 75) and mx["estado"] == "DISCREPA"
    assert mn["nivel"] == "documental" and mn["alcance_medido"] == "documental" and mn["tabla_origen"] == "cifras_documentadas.csv" and mn["z"] is None
    assert mn["dif_abs"] == pytest.approx(10 - 354853 / 89438) and mn["dif_rel"] == pytest.approx((10 - 354853 / 89438) / 10)
    for clave in ("k.a", "k.b", "k.c"):
        assert clave in mn["motivo"]
    assert "3,97" in mn["motivo"] and "67,64" in mn["motivo"] and "18,93" in mn["motivo"]


def test_aritmetica_confirma_cuando_las_razones_coinciden_al_redondeo():
    pares = [{"clave": "k.a", "razon": 10.4}, {"clave": "k.b", "razon": 15.0}, {"clave": "k.c", "razon": 19.6}]
    mn = cifra_fila("C128", valor=10, tipo="aritmetica", unidad="razon", decimales=0, alcance="documental", clave="doc.razon_minima_cascada")
    mx = cifra_fila("C129", valor=20, tipo="aritmetica", unidad="razon", decimales=0, alcance="documental", clave="doc.razon_maxima_cascada")
    assert ledger.evaluar_aritmetica(mn, pares, reglas())["estado"] == "CONFIRMA" and ledger.evaluar_aritmetica(mx, pares, reglas())["estado"] == "CONFIRMA"
    # el mínimo y el máximo no se intercambian: con el máximo en C128 no confirmaría
    pares2 = [{"clave": "k.a", "razon": 10.5}, {"clave": "k.b", "razon": 19.5}]
    assert ledger.evaluar_aritmetica(mn, pares2, reglas())["estado"] == "DISCREPA"        # 10,5 se redondea hacia arriba a 11
    assert ledger.evaluar_aritmetica(mx, pares2, reglas())["estado"] == "CONFIRMA"        # 19,5 se redondea hacia arriba a 20
    assert ledger.evaluar_aritmetica(mx, [{"clave": "k.a", "razon": 19.4}, {"clave": "k.b", "razon": 10.0}], reglas())["estado"] == "DISCREPA"


def test_aritmetica_sin_pares_o_con_clave_desconocida_es_no_verificable():
    mn = cifra_fila("C128", valor=10, tipo="aritmetica", unidad="razon", decimales=0, alcance="documental", clave="doc.razon_minima_cascada")
    assert ledger.evaluar_aritmetica(mn, [], reglas())["motivo"] == ledger.MOTIVOS["sin_pares"]
    raro = {**mn, "clave_eda": "doc.otra"}
    assert ledger.evaluar_aritmetica(raro, [{"clave": "k", "razon": 1.0}], reglas())["motivo"] == ledger.MOTIVOS["clave_aritmetica"]


def test_aritmetica_agrega_el_inverso_de_la_fraccion_de_muestreo_si_se_conoce():
    mn = cifra_fila("C128", valor=10, tipo="aritmetica", unidad="razon", decimales=0, alcance="documental", clave="doc.razon_minima_cascada")
    r = ledger.evaluar_aritmetica(mn, [{"clave": "k", "razon": 19.0}], reglas(), 19.362)
    assert "inverso de la fracción de muestreo del extracto: 19,36" in r["motivo"] and ledger.motivo_base(r["motivo"]).startswith("razones de la Fase 2")


def test_factor_de_muestreo_es_el_documentado_sobre_lo_leido():
    cifras = cifras_df(cifra_fila("C007", valor=5808536, tipo="absoluto", unidad="conteo", decimales=0, clave="volumen.filas_leidas.total", alcance="2019-2024"))
    assert ledger.factor_muestreo(cifras, {"volumen.filas_leidas.total": metrica(299994)}) == pytest.approx(5808536 / 299994)
    assert ledger.factor_muestreo(cifras, {"volumen.filas_leidas.total": metrica(299994)}) == pytest.approx(19.362, abs=1e-3)
    assert ledger.factor_muestreo(cifras, {}) is None and ledger.factor_muestreo(cifras, {"volumen.filas_leidas.total": metrica(0)}) is None
    assert ledger.factor_muestreo(cifras.head(0), {"volumen.filas_leidas.total": metrica(5)}) is None


# ======================================================================================================== la tabla de cifras documentadas (configuración)
def test_cifras_documentadas_son_129_con_ids_unicos_y_sin_problemas():
    df = ledger.cargar_cifras()
    assert df.height == 129 and df["id"].n_unique() == 129
    assert df["id"].to_list() == [f"C{i:03d}" for i in range(1, 130)]
    assert ledger.validar_cifras(df) == []
    assert set(df["tipo"].to_list()) == {"absoluto", "proporcion", "estructura", "aritmetica"}
    assert df["tipo"].value_counts().sort("tipo").rows() == [("absoluto", 56), ("aritmetica", 2), ("estructura", 15), ("proporcion", 56)]


def test_las_constantes_por_id_concuerdan_con_el_csv():
    por_id = {f["id"]: f for f in ledger.cargar_cifras().iter_rows(named=True)}
    assert set(ledger.NO_COMPARABLES) == {"C089", "C116", "C117"} and set(ledger.SUBESTIMADAS_EN_EXTRACTO) == {"C119"}
    assert set(ledger.COMPARACION_ESPECIAL) == {"C015", "C125", "C126", "C127"} and ledger.HIPOTESIS_ORIGEN_EXTRACTO == frozenset({"C112", "C113", "C114", "C115"})
    assert {por_id[c]["tipo"] for c in ledger.NO_COMPARABLES} == {"proporcion"}
    assert por_id["C116"]["alcance"] == "cohorte final" and por_id["C117"]["alcance"] == "2023" and por_id["C089"]["clave_eda"] == "calidad.diagnostico_con_punto_pct"
    assert por_id["C119"]["clave_eda"] == "traslados.enlace_pct_30d.2023" and por_id["C119"]["unidad"] == "porcentaje"
    assert [por_id[c]["documento"] for c in sorted(ledger.HIPOTESIS_ORIGEN_EXTRACTO)] == ["Fase 1"] * 4
    assert {c: ("más de" in por_id[c]["descripcion"] or "aproximadamente" in por_id[c]["nota"] or "aproximadamente" in por_id[c]["descripcion"]) for c in ledger.COMPARACION_ESPECIAL} == {
        "C015": True, "C125": True, "C126": True, "C127": True}
    assert ledger.COMPARACION_ESPECIAL["C125"] == "mas_de" and por_id["C125"]["valor"] == "9500"
    assert {k for k, f in por_id.items() if f["tipo"] == "aritmetica"} == {"C128", "C129"}
    assert {por_id[c]["clave_eda"] for c in ("C128", "C129")} == set(ledger.ARITMETICA)
    assert ledger.CLAVES_SIN_ERROR_ESTANDAR == frozenset({"resultado.caida_mortalidad_2021_2023_pct"}) and por_id["C048"]["clave_eda"] in ledger.CLAVES_SIN_ERROR_ESTANDAR


def test_las_claves_corregidas_apuntan_a_las_metricas_de_tres_caracteres_sin_cambiar_lo_documentado():
    por_id = {f["id"]: f for f in ledger.cargar_cifras().iter_rows(named=True)}
    esperadas = {"C062": ("27.8", "ELIX_06"), "C063": ("6.2", "ELIX_10"), "C064": ("5.9", "ELIX_25"), "C065": ("5.6", "ELIX_14"), "C066": ("3.6", "ELIX_02"), "C067": ("3.3", "ELIX_01")}
    for cid, (valor, elix) in esperadas.items():
        assert por_id[cid]["clave_eda"] == f"elix.prev_pct_3c_todos.{elix}" and por_id[cid]["valor"] == valor and por_id[cid]["documento"] == "Fase 2", cid
    assert por_id["C068"]["clave_eda"] == "elix.prev_pct_3c_todos.diabetes" and por_id["C068"]["valor"] == "0.0"
    assert por_id["C051"]["clave_eda"] == "cohorte.pct_estancia_total;cohorte.pct_estancia_total.sin_censurados" and por_id["C051"]["valor"] == "61.31"
    assert por_id["C100"]["clave_eda"] == "faltantes.n_nulos.IR_29301_COD_GRD" and por_id["C100"]["valor"] == "90" and por_id["C030"]["clave_eda"] == "faltantes.n_nulos.IR_29301_COD_GRD"
    assert por_id["C116"]["clave_eda"] == "cohorte.prevalencia_pct.fase1"                  # sin cambios: no es comparable


def test_los_conteos_documentados_de_menos_de_diez_no_se_copian_en_la_configuracion():
    """Cambio 1.6: el registro dice «<10» en vez de copiar del documento un conteo de 1 a 9 casos (C031, C080, C082, C085 y C097)."""
    df = ledger.cargar_cifras()
    pequenos = df.filter(pl.col("valor") == ledger.VALOR_PEQUENO)
    assert sorted(pequenos["id"].to_list()) == ["C031", "C080", "C082", "C085", "C097"]
    conteos = df.filter((pl.col("tipo") == "absoluto") & (pl.col("unidad") == "conteo") & (pl.col("valor") != ledger.VALOR_PEQUENO))
    assert all(not (0 < float(v) < 10) for v in conteos["valor"].to_list())


def test_un_conteo_documentado_como_menos_de_diez_queda_no_verificable_sin_valor():
    f = cifra_fila("C901", valor="<10", tipo="absoluto", unidad="conteo", decimales=0, alcance="2019", clave="calidad.x")
    out = ledger.evaluar_cifra(f, {"calidad.x": metrica(7, tabla="t24_reglas.csv")}, reglas())
    assert out["estado"] == "NO VERIFICABLE" and out["motivo"] == ledger.MOTIVOS["pocos_casos"]
    assert out["valor_doc"] is None and out["n_doc"] is None and out["clave_usada"] == "calidad.x"
    sin_metrica = ledger.evaluar_cifra(f, {}, reglas())
    assert sin_metrica["estado"] == "NO VERIFICABLE" and sin_metrica["valor_doc"] is None
    propios = lambda df: [p for p in ledger.validar_cifras(df, esperadas=None) if p.startswith("C9")]   # sin las constantes por id del CSV real
    assert propios(cifras_df(f)) == []
    malo = cifra_fila("C902", valor="<10", tipo="proporcion", unidad="porcentaje")
    assert any("solo vale para un conteo absoluto" in p for p in propios(cifras_df(malo)))
    assert ledger.razones_fase2_fase1(cifras_df(f, cifra_fila("C903", valor=12, tipo="absoluto", unidad="conteo", documento="Fase 1", clave="calidad.x"))) == []


def test_validar_cifras_detecta_los_problemas():
    base = ledger.cargar_cifras()
    assert ledger.validar_cifras(base.head(128))[0] == "hay 128 cifras y se esperaban 129"
    assert any("repetidos" in p for p in ledger.validar_cifras(pl.concat([base, base.head(1)])))
    assert any("tipo inválido" in p for p in ledger.validar_cifras(base.with_columns(pl.when(pl.col("id") == "C001").then(pl.lit("raro")).otherwise(pl.col("tipo")).alias("tipo"))))
    assert any("requiere_resultado" in p for p in ledger.validar_cifras(base.with_columns(pl.when(pl.col("id") == "C001").then(pl.lit("quizá")).otherwise(pl.col("requiere_resultado")).alias("requiere_resultado"))))
    assert any("valor no es numérico" in p for p in ledger.validar_cifras(base.with_columns(pl.when(pl.col("id") == "C001").then(pl.lit("x")).otherwise(pl.col("valor")).alias("valor"))))
    assert any("mal formada" in p for p in ledger.validar_cifras(base.with_columns(pl.when(pl.col("id") == "C001").then(pl.lit("a; b")).otherwise(pl.col("clave_eda")).alias("clave_eda"))))
    assert any("mal formada" in p for p in ledger.validar_cifras(base.with_columns(pl.when(pl.col("id") == "C001").then(pl.lit("a;;b")).otherwise(pl.col("clave_eda")).alias("clave_eda"))))
    assert ledger.validar_cifras(base.drop("unidad"))[0].startswith("faltan columnas")
    sin_c119 = base.filter(pl.col("id") != "C119")
    assert any("C119" in p for p in ledger.validar_cifras(sin_c119, esperadas=128))
    de_fase2 = base.with_columns(pl.when(pl.col("id") == "C113").then(pl.lit("Fase 2")).otherwise(pl.col("documento")).alias("documento"))
    assert any("C113" in p and "Fase 1" in p for p in ledger.validar_cifras(de_fase2))


# ============================================================================================================================ métricas disponibles
def test_metricas_disponibles_prefieren_la_memoria_y_completan_con_el_disco(tmp_path, fuente_sintetica):
    ctx1 = Contexto.crear("extracto", fuente=fuente_sintetica, dir_salida_base=tmp_path)
    ctx1.registrar_metrica("a.x", 1.0, tarea="T00", tabla="t1.csv", ic_bajo=0.5, ic_alto=1.5, alcance="2019-2023", nota="nota a")
    ctx1.registrar_metrica("b.y", 2.0, tarea="T00", tabla="t2.csv")
    ctx1.escribir_registros()
    ctx2 = Contexto.crear("extracto", fuente=fuente_sintetica, dir_salida_base=tmp_path)           # memoria vacía y el mismo directorio
    solo_disco = ledger.metricas_disponibles(ctx2)
    assert solo_disco["a.x"] == {"valor": 1.0, "ic_bajo": 0.5, "ic_alto": 1.5, "tabla": "t1.csv", "nivel": "extracto", "alcance": "2019-2023", "nota": "nota a"}
    assert solo_disco["b.y"]["valor"] == 2.0 and solo_disco["b.y"]["alcance"] == "" and solo_disco["b.y"]["ic_bajo"] is None
    ctx2.registrar_metrica("a.x", 99.0, tarea="T00", tabla="t1.csv")
    ctx2.registrar_metrica("c.z", 3.0, tarea="T00", tabla="t3.csv")
    mezcla = ledger.metricas_disponibles(ctx2)
    assert mezcla["a.x"]["valor"] == 99.0 and mezcla["b.y"]["valor"] == 2.0 and mezcla["c.z"]["valor"] == 3.0       # la memoria manda


def test_metricas_disponibles_sin_memoria_ni_disco_es_vacio(tmp_path, fuente_sintetica):
    assert ledger.metricas_disponibles(Contexto.crear("extracto", fuente=fuente_sintetica, dir_salida_base=tmp_path)) == {}


def test_indexar_metricas_conserva_valores_nulos_y_textos_vacios():
    m = ledger.indexar_metricas(metricas_df({"clave": "k.a", "valor": None, "nota": "[suprimido: menos de diez casos]"}, {"clave": "k.b", "valor": 5.0, "alcance": "2019"}))
    assert m["k.a"]["valor"] is None and "[suprimido" in m["k.a"]["nota"] and m["k.b"] == metrica(5.0, alcance="2019")


def test_filas_leidas_por_anio_solo_toma_los_anios():
    m = {"volumen.filas_leidas.2019": metrica(10), "volumen.filas_leidas.total": metrica(30), "volumen.filas_leidas.promedio": metrica(10), "volumen.filas_leidas.2020": metrica(None),
         "volumen.filas_leidas.2021": metrica(20.0)}
    assert ledger.filas_leidas_por_anio(m) == {2019: 10, 2021: 20}


# ======================================================================================================================================== calcular
def _metricas_mini():
    return {
        "k.a": metrica(215, alcance="2019-2024"), "k.b": metrica(7760, alcance="2019-2024"), "k.c": metrica(14450, alcance="2019-2023"),
        "volumen.filas_leidas.2023": metrica(50_000),
    }


def test_calcular_arma_una_fila_por_cifra_ordenada_por_id_y_con_el_esquema(ctx):
    cifras = _pares_documentados()
    df = ledger.calcular(ctx, cifras, _metricas_mini())
    assert df.height == cifras.height and df["id"].to_list() == sorted(cifras["id"].to_list())
    assert df.columns == list(ledger.COLUMNAS_LEDGER) and dict(df.schema) == ledger.ESQUEMA_LEDGER
    por_id = {r["id"]: r for r in df.iter_rows(named=True)}
    assert por_id["C113"]["estado"] == "CONFIRMA" and por_id["C112"]["estado"] == "DISCREPA"            # hipótesis de origen: 7.760 igual y 75 contra 215
    assert por_id["C103"]["estado"] == "NO VERIFICABLE"                                                  # la misma clave de la Fase 2 es un conteo de la población
    assert por_id["C128"]["estado"] == "DISCREPA" and por_id["C128"]["valor_medido"] == pytest.approx(354853 / 89438)
    assert set(df["estado"].unique()) <= set(ledger.ESTADOS) and set(df["nivel"].unique()) == {"extracto", "documental"}


def test_calcular_agrega_el_factor_de_muestreo_solo_en_extracto(tmp_path, fuente_sintetica):
    cifras = pl.concat([_pares_documentados(), cifras_df(cifra_fila("C007", valor=5808536, tipo="absoluto", unidad="conteo", decimales=0, alcance="2019-2024",
                                                                      clave="volumen.filas_leidas.total"))])
    metricas = {**_metricas_mini(), "volumen.filas_leidas.total": metrica(299994)}
    ext = ledger.calcular(Contexto.crear("extracto", fuente=fuente_sintetica, dir_salida_base=tmp_path / "a"), cifras, metricas)
    assert "inverso de la fracción de muestreo del extracto: 19,36" in ext.filter(pl.col("id") == "C128")["motivo"][0]
    completo = dataclasses.replace(fuente_sintetica, etiqueta="completo")
    comp = ledger.calcular(Contexto.crear("completo", fuente=completo, dir_salida_base=tmp_path / "b"), cifras, metricas)
    assert "fracción de muestreo" not in comp.filter(pl.col("id") == "C128")["motivo"][0]
    assert set(comp["nivel"].unique()) == {"completo", "documental"}


def test_calcular_en_modo_completo_aplica_la_igualdad_al_redondeo(tmp_path, fuente_sintetica):
    completo = dataclasses.replace(fuente_sintetica, etiqueta="completo")
    ctx_c = Contexto.crear("completo", fuente=completo, dir_salida_base=tmp_path)
    cifras = cifras_df(
        cifra_fila("C001", valor=1151475, tipo="absoluto", unidad="conteo", decimales=0, alcance="2019", clave="v.2019"),
        cifra_fila("C002", valor=781912, tipo="absoluto", unidad="conteo", decimales=0, alcance="2020", clave="v.2020"),
        cifra_fila("C008", valor=2.57, alcance="2019", clave="p.2019", requiere_resultado="si"),
        cifra_fila("C009", valor=3.83, alcance="2020", clave="p.2020", requiere_resultado="si"),
    )
    metricas = {"v.2019": metrica(1151475), "v.2020": metrica(781913), "p.2019": metrica(2.574999), "p.2020": metrica(3.80)}
    df = ledger.calcular(ctx_c, cifras, metricas)
    estados = dict(zip(df["id"], df["estado"]))
    assert estados == {"C001": "CONFIRMA", "C002": "DISCREPA", "C008": "CONFIRMA", "C009": "DISCREPA"}
    assert set(df["nivel"].unique()) == {"completo"} and df["z"].null_count() == df.height


def test_calcular_es_determinista(ctx):
    a = ledger.calcular(ctx, _pares_documentados(), _metricas_mini())
    b = ledger.calcular(ctx, _pares_documentados(), _metricas_mini())
    assert a.equals(b)


# ============================================================================================================== motivos, contradicciones y resumen
def test_motivo_base_corta_el_detalle():
    assert ledger.motivo_base("causa fija | detalle 1 | detalle 2") == "causa fija" and ledger.motivo_base("sin detalle") == "sin detalle" and ledger.motivo_base(None) == ""


def test_tabla_motivos_cuenta_solo_las_no_verificables_y_ordena():
    df = pl.DataFrame(
        {"id": ["C001", "C002", "C003", "C004", "C005", "C006"], "estado": ["NO VERIFICABLE", "NO VERIFICABLE", "CONFIRMA", "NO VERIFICABLE", "DISCREPA", "NO VERIFICABLE"],
         "motivo": ["causa b | detalle uno", "causa a", "ok", "causa b | detalle dos", "mal", "causa b"]})
    t = ledger.tabla_motivos(df)
    assert t.rows() == [("causa b", 3, "C001; C004; C006"), ("causa a", 1, "C002")]
    assert ledger.tabla_motivos(df.head(0)).height == 0


def test_resumen_ledger_cuenta_los_estados_y_las_discrepancias_relevantes():
    df = pl.DataFrame({"estado": ["CONFIRMA", "DISCREPA", "DISCREPA", "NO VERIFICABLE", "NO VERIFICABLE"], "magnitud": ["no aplica", "relevante", "despreciable", "no aplica", "relevante"]})
    assert ledger.resumen_ledger(df) == {"ledger_n_cifras": 5, "ledger_n_confirma": 1, "ledger_n_discrepa": 2, "ledger_n_no_verificable": 2, "ledger_n_discrepa_relevante": 1}


def _ledger_minimo(*filas):
    base = {"id": "C001", "documento": "Fase 2", "clave_usada": "k", "alcance_doc": "2019-2024", "valor_doc": 1.0}
    return pl.DataFrame([{**base, **f} for f in filas], schema={"id": pl.String, "documento": pl.String, "clave_usada": pl.String, "alcance_doc": pl.String, "valor_doc": pl.Float64})


def test_contradicciones_documentales_son_pares_del_mismo_documento_con_la_misma_clave_y_alcance():
    df = _ledger_minimo(
        {"id": "C030", "valor_doc": 75.0, "clave_usada": "faltantes.n_nulos.IR_29301_COD_GRD"},
        {"id": "C100", "valor_doc": 90.0, "clave_usada": "faltantes.n_nulos.IR_29301_COD_GRD"},
        {"id": "C103", "valor_doc": 5073.0, "clave_usada": "cascada.n_ex01"},
        {"id": "C112", "valor_doc": 75.0, "clave_usada": "cascada.n_ex01", "documento": "Fase 1"},                       # otro documento: no es contradicción
        {"id": "C201", "valor_doc": 5.0, "clave_usada": "z.misma"}, {"id": "C202", "valor_doc": 5.0, "clave_usada": "z.misma"},   # mismo valor
        {"id": "C203", "valor_doc": 1.0, "clave_usada": "z.otro", "alcance_doc": "2019"}, {"id": "C204", "valor_doc": 2.0, "clave_usada": "z.otro", "alcance_doc": "2020"},
    )
    c = ledger.contradicciones_documentales(df)
    assert c == [{"clave": "faltantes.n_nulos.IR_29301_COD_GRD", "alcance": "2019-2024", "documento": "Fase 2", "id_a": "C030", "id_b": "C100", "valor_a": 75.0, "valor_b": 90.0}]


def test_contradicciones_ignoran_valores_suprimidos_y_claves_vacias():
    df = _ledger_minimo({"id": "C080", "valor_doc": None, "clave_usada": "a"}, {"id": "C082", "valor_doc": 3.0, "clave_usada": "a"},
                        {"id": "C300", "valor_doc": 1.0, "clave_usada": None}, {"id": "C301", "valor_doc": 2.0, "clave_usada": None},
                        {"id": "C302", "valor_doc": 1.0, "clave_usada": ""}, {"id": "C303", "valor_doc": 2.0, "clave_usada": ""})
    assert ledger.contradicciones_documentales(df) == []


def test_fusionar_resumen_agrega_columnas_sin_perder_las_anteriores(tmp_path, fuente_sintetica):
    ctx = Contexto.crear("extracto", fuente=fuente_sintetica, dir_salida_base=tmp_path)
    ledger.fusionar_resumen(ctx, {"ledger_n_cifras": 129, "ledger_n_confirma": 33})
    ledger.fusionar_resumen(ctx, {"decisiones_n_verde": 3, "ledger_n_confirma": 34})
    fila = ctx.leer_tabla("t25_resumen_evidencia.csv").row(0, named=True)
    assert fila == {"ledger_n_cifras": 129, "ledger_n_confirma": 34, "decisiones_n_verde": 3}
    assert ctx.indice_tablas().filter(pl.col("nombre") == "t25_resumen_evidencia.csv")["tarea"][0] == "T12"
    assert ctx.leer_tabla("t25_resumen_evidencia.csv").height == 1


# ============================================================================================================== corrida sobre la fuente sintética
@pytest.fixture(scope="module")
def ledger_corrido(pipeline_sintetico, tmp_path_factory):
    ctx_l = copiar_tablas(pipeline_sintetico.ctx, tmp_path_factory.mktemp("t12_ledger"), con_figuras=True)
    df = ledger.ejecutar(ctx_l)
    ctx_l.escribir_registros()
    return ctx_l, df


def test_el_ledger_tiene_129_filas_con_estado_valido_y_orden_por_id(ledger_corrido):
    ctx_l, df = ledger_corrido
    assert df.height == 129 and df["id"].to_list() == [f"C{i:03d}" for i in range(1, 130)]
    assert set(df["estado"].unique()) <= set(ledger.ESTADOS) and df["estado"].null_count() == 0 and df["motivo"].null_count() == 0
    assert df.columns == list(ledger.COLUMNAS_LEDGER)
    csv = ctx_l.leer_tabla("ledger_cifras.csv", columnas_texto=list(ledger.COLUMNAS_TEXTO_LEDGER))
    assert csv.height == 129 and csv["id"].to_list() == df["id"].to_list()


def test_cada_estado_esta_justificado_con_el_valor_medido_y_su_tabla(ledger_corrido):
    _, df = ledger_corrido
    comparadas = df.filter(pl.col("estado").is_in(["CONFIRMA", "DISCREPA"]))
    assert comparadas.height > 0
    assert comparadas["valor_medido"].null_count() == 0 and comparadas["tabla_origen"].null_count() == 0 and comparadas["clave_usada"].null_count() == 0
    # una fila comparada trae su diferencia; una no verificable nunca la trae (no se comparó nada)
    assert comparadas.filter(pl.col("tipo") != "aritmetica")["dif_abs"].null_count() == 0
    assert df.filter(pl.col("estado") == "NO VERIFICABLE")["dif_abs"].null_count() == df.filter(pl.col("estado") == "NO VERIFICABLE").height
    assert df.filter(pl.col("estado") != "DISCREPA")["magnitud"].unique().to_list() == ["no aplica"]
    assert set(df["magnitud"].unique()) <= {"no aplica", "relevante", "despreciable"}


def test_las_cifras_aritmeticas_se_recalculan_con_las_cifras_documentadas(ledger_corrido):
    _, df = ledger_corrido
    f = {r["id"]: r for r in df.iter_rows(named=True)}
    # razones de la Fase 2 sobre la Fase 1 con los valores de config/cifras_documentadas.csv
    assert f["C128"]["valor_medido"] == pytest.approx(354853 / 89438) and f["C129"]["valor_medido"] == pytest.approx(5073 / 75)
    assert f["C128"]["estado"] == "DISCREPA" and f["C129"]["estado"] == "DISCREPA" and f["C128"]["nivel"] == "documental"
    assert f["C128"]["tabla_origen"] == "cifras_documentadas.csv"


def test_los_conteos_de_1_a_9_casos_quedan_protegidos_en_el_csv(ledger_corrido):
    ctx_l, _ = ledger_corrido
    csv = ctx_l.leer_tabla("ledger_cifras.csv", columnas_texto=list(ledger.COLUMNAS_TEXTO_LEDGER))
    fila = {r["id"]: r for r in csv.iter_rows(named=True)}
    for cid in ("C031", "C080", "C082", "C085", "C097"):                                     # documentados con menos de diez casos (en la configuración, «<10»)
        assert fila[cid]["valor_doc"] is None and fila[cid]["n_doc"] is None and fila[cid]["estado"] == "NO VERIFICABLE", cid
        assert fila[cid]["valor_medido"] is None and fila[cid]["dif_abs"] is None and fila[cid]["z"] is None, cid
    assert fila["C081"]["valor_doc"] == 10.0 and fila["C081"]["n_doc"] == 10                    # diez casos ya se publican
    conteos = csv.filter((pl.col("tipo") == "absoluto") & (pl.col("unidad") == "conteo"))
    for col in ("valor_doc", "valor_medido", "n_doc", "n_medido"):
        pequenos = conteos.filter((pl.col(col) > 0) & (pl.col(col) < 10))
        assert pequenos.height == 0, (col, pequenos["id"].to_list())


def test_la_tabla_se_registra_con_sus_columnas_de_conteo_y_la_privacidad_queda_en_cero(ledger_corrido):
    ctx_l, _ = ledger_corrido
    fila = ctx_l.indice_tablas().filter(pl.col("nombre") == "ledger_cifras.csv").row(0, named=True)
    assert fila["columnas_conteo"] == "n_doc;n_medido" and fila["tarea"] == "T12" and fila["celdas_suprimidas"] >= 5
    esc = privacidad.escanear_salidas(ctx_l.dir_salida, ctx_l.indice_tablas(), ctx_l.celda_minima)
    assert privacidad.total_violaciones(esc) == 0, esc.filter(pl.col("n_violaciones") > 0)


def test_las_columnas_de_conteo_solo_existen_en_los_conteos_absolutos(ledger_corrido):
    _, df = ledger_corrido
    no_conteo = df.filter(~((pl.col("tipo") == "absoluto") & (pl.col("unidad") == "conteo")))
    assert no_conteo["n_doc"].null_count() == no_conteo.height and no_conteo["n_medido"].null_count() == no_conteo.height
    conteo = df.filter((pl.col("tipo") == "absoluto") & (pl.col("unidad") == "conteo"))
    assert conteo["n_doc"].null_count() == 0


def test_la_tarea_12_no_lee_datos_ni_deja_filas_en_el_registro_de_accesos(pipeline_sintetico, tmp_path, monkeypatch):
    ctx_l = copiar_tablas(pipeline_sintetico.ctx, tmp_path, con_figuras=True)

    def leer_prohibido(*a, **k):
        raise AssertionError("la Tarea 12 no lee datos")

    monkeypatch.setattr(ctx_l.lector, "leer", leer_prohibido)
    ledger.ejecutar(ctx_l)
    assert ctx_l.cortafuegos.registro().height == 0 and ctx_l.cortafuegos.violaciones() == 0 and ctx_l.cortafuegos.n_bloqueos() == 0


def test_las_metricas_del_ledger_cuentan_cifras_y_no_casos(ledger_corrido):
    ctx_l, df = ledger_corrido
    m = ctx_l.metricas().filter(pl.col("clave").str.starts_with("ledger."))
    val = dict(zip(m["clave"], m["valor"]))
    assert set(val) == {"ledger.n_cifras", "ledger.n_confirma", "ledger.n_discrepa", "ledger.n_no_verificable", "ledger.n_discrepa_relevante"}
    assert val["ledger.n_cifras"] == 129 and val["ledger.n_confirma"] + val["ledger.n_discrepa"] + val["ledger.n_no_verificable"] == 129
    assert val["ledger.n_confirma"] == (df["estado"] == "CONFIRMA").sum() and val["ledger.n_discrepa"] == (df["estado"] == "DISCREPA").sum()
    assert set(m["tarea"]) == {"T12"} and set(m["tabla"]) == {"ledger_cifras.csv"} and "no casos" in m["nota"][0]


def test_la_tabla_de_motivos_suma_las_no_verificables(ledger_corrido):
    ctx_l, df = ledger_corrido
    t = ctx_l.leer_tabla("ledger_motivos.csv", columnas_texto=["motivo", "ids"])
    nv = df.filter(pl.col("estado") == "NO VERIFICABLE")
    assert t["n_cifras"].sum() == nv.height
    assert sorted("; ".join(t["ids"].to_list()).split("; ")) == sorted(nv["id"].to_list())
    assert all(ledger.SEPARADOR_DETALLE not in m for m in t["motivo"].to_list())


def test_el_resumen_de_evidencia_trae_los_conteos_del_ledger(ledger_corrido):
    ctx_l, df = ledger_corrido
    fila = ctx_l.leer_tabla("t25_resumen_evidencia.csv").row(0, named=True)
    assert fila == ledger.resumen_ledger(df)


def test_f12_02_existe_con_su_csv_de_respaldo(ledger_corrido):
    ctx_l, _ = ledger_corrido
    idx = ctx_l.figuras.indice().filter(pl.col("id_figura") == "F12_02")
    assert idx.height == 1 and idx["csv"][0] == "ledger_cifras.csv" and idx["tarea"][0] == "12" and idx["uso_interno"][0] == "no"
    png = ctx_l.dir_figuras / "F12_02.png"
    assert png.is_file() and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "ledger_cifras.csv" in idx["csv"].to_list()[0] and (ctx_l.dir_tablas / "ledger_cifras.csv").is_file()
    assert "Fase 1 y Fase 2" in idx["descripcion_alternativa"][0]


def test_la_corrida_es_determinista_en_otra_carpeta(ledger_corrido, pipeline_sintetico, tmp_path):
    ctx_l, _ = ledger_corrido
    otro = copiar_tablas(pipeline_sintetico.ctx, tmp_path, con_figuras=True)
    ledger.ejecutar(otro)
    for nombre in ("ledger_cifras.csv", "ledger_motivos.csv", "t25_resumen_evidencia.csv"):
        assert (ctx_l.dir_tablas / nombre).read_bytes() == (otro.dir_tablas / nombre).read_bytes(), nombre
    assert (ctx_l.dir_figuras / "F12_02.png").read_bytes() == (otro.dir_figuras / "F12_02.png").read_bytes()


def test_ejecutar_rechaza_un_csv_de_cifras_invalido(monkeypatch, pipeline_sintetico, tmp_path):
    ctx_l = copiar_tablas(pipeline_sintetico.ctx, tmp_path)
    original = ledger.cargar_cifras
    monkeypatch.setattr(ledger, "cargar_cifras", lambda ruta=None: original().head(100))
    with pytest.raises(ValueError, match="no es válido"):
        ledger.ejecutar(ctx_l)


def test_el_informe_del_ledger_tiene_dos_secciones_y_la_figura(ledger_corrido):
    ctx_l, df = ledger_corrido
    inf = Informe(ctx_l, "t25_prueba_ledger.md", "Prueba del ledger")
    ledger.agregar_secciones(inf, ctx_l, 1)
    ruta = inf.guardar()
    md = ruta.read_text(encoding="utf8")
    assert md.count("\n## ") == ledger.N_SECCIONES == 2
    assert "## 1. Cifras documentadas frente a lo medido" in md and "## 2. Discrepancias y cifras no verificables" in md
    assert "![" in md and "F12_02" in md and "ledger_cifras.csv" in md and "ledger_motivos.csv" in md
    assert "{" not in md and "}" not in md
    assert f"compara {df.height} cifras" in md
    cifras = ctx_l.cifras_informe().filter(pl.col("informe") == "t25_prueba_ledger.md")
    assert cifras.height > 0 and cifras["csv"].null_count() == 0 and (cifras["csv"] == "").sum() == 0 and (cifras["columna"] == "").sum() == 0
    assert set(cifras["csv"].to_list()) <= {"t25_resumen_evidencia.csv", "umbrales.csv", "ledger_cifras.csv", "ledger_motivos.csv"}


def test_el_informe_con_numeracion_desplazada(ledger_corrido):
    ctx_l, _ = ledger_corrido
    inf = Informe(ctx_l, "t25_prueba_ledger2.md", "Prueba del ledger")
    ledger.agregar_secciones(inf, ctx_l, 5)
    md = inf.guardar().read_text(encoding="utf8")
    assert "## 5. Cifras documentadas frente a lo medido" in md and "## 6. Discrepancias y cifras no verificables" in md


def test_valor_con_unidad_da_a_cada_unidad_su_formato():
    v = ledger.valor_con_unidad
    assert v("anio", 0, 2024.0) == "2024" and v("anio", 0, 2019.4) == "2019"                       # un año no lleva separador de miles ni decimales
    assert v("conteo", 0, 1151475.0) == "1.151.475" and v("conteo", 0, 65.0) == "65"
    assert v("porcentaje", 1, 87.9) == "87,9 %" and v("porcentaje", 2, 3.0) == "3,00 %" and v("porcentaje", 0, 38.0) == "38 %"
    assert v("dias", 1, 3.0) == "3,0" and v("dx", 2, 4.08) == "4,08" and v("veces", 0, 10.0) == "10"


def test_la_tabla_de_discrepancias_del_informe_da_a_cada_cifra_el_formato_de_su_unidad(ledger_corrido, tmp_path):
    ctx_o = copiar_tablas(ledger_corrido[0], tmp_path, con_figuras=True)

    def fila(id_, unidad, d, doc, med, tipo="absoluto"):
        f = dict.fromkeys(ledger.COLUMNAS_LEDGER)
        f.update(id=id_, documento="Fase 2", ubicacion="1", descripcion=f"cifra {id_}", tipo=tipo, unidad=unidad, alcance_doc="2019", requiere_resultado="no", valor_doc=doc,
                 decimales=d, clave_eda="k", clave_usada="k", valor_medido=med, nivel="extracto", alcance_medido="2019", tabla_origen="t.csv", dif_abs=abs(med - doc),
                 dif_rel=abs(med - doc) / doc, magnitud="relevante", estado="DISCREPA", motivo="m", nota="")
        return f

    led = pl.DataFrame([fila("C901", "anio", 0, 2019.0, 2024.0, "estructura"), fila("C902", "conteo", 0, 65.0, 72.0), fila("C903", "conteo", 0, 1151475.0, 1300000.0),
                        fila("C904", "porcentaje", 1, 87.9, 88.4, "proporcion"), fila("C905", "dias", 1, 3.0, 4.5, "proporcion")], schema=ledger.ESQUEMA_LEDGER)
    ctx_o.escribir_tabla("ledger_cifras.csv", led, tarea="T12", cols_n=ledger.COLS_N_LEDGER, dependientes=ledger.DEPENDIENTES_LEDGER, descripcion="prueba de formatos")
    ctx_o.escribir_tabla("ledger_motivos.csv", ledger.tabla_motivos(led), tarea="T12", descripcion="prueba de formatos")
    inf = Informe(ctx_o, "t25_prueba_formatos.md", "Prueba de formatos")
    ledger.agregar_secciones(inf, ctx_o, 1)
    md = inf.guardar().read_text(encoding="utf8")
    assert "| 2019 | 2024 |" in md and "| 65 | 72 |" in md and "| 1.151.475 | 1.300.000 |" in md and "| 87,9 % | 88,4 % |" in md and "| 3,0 | 4,5 |" in md
    assert "2.019,00" not in md and "65,00" not in md and "72,00" not in md and "2.019 " not in md                                 # ni años con separador ni conteos con decimales


# FIN-test_t12_ledger
