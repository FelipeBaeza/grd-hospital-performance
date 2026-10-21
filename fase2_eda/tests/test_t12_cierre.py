"""Tarea 12: cambios sugeridos a las Fases 1 y 2, cambios al protocolo, resumen y cierre (cada hallazgo con su condición cumplida y no cumplida sobre tablas hechas a mano,
más la corrida sintética de punta a punta)."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest

from ayudas_t12 import FuenteEvidenciaFalsa, cifra_fila, cifras_df, copiar_tablas, df, metrica
from eda import cierre, config, decisiones, ledger, privacidad
from eda.cierre import Evidencia
from eda.informes import Informe

HAL = {h.id: h for h in cierre.HALLAZGOS}
CIFRAS = ledger.cargar_cifras()
HIPOTESIS = config.cargar_hipotesis()


# ============================================================================================================================ ayudas
def led(id_, *, estado="DISCREPA", doc=100.0, med=200.0, documento="Fase 2", ubicacion="2.1", descripcion="cifra de prueba", tipo="absoluto", unidad="conteo", d=0,
        alcance_doc="2019-2023", req="no", clave=None, alcance_med="2019-2023", tabla="t_prueba.csv", magnitud=None, ic=(None, None)):
    """Una fila del ledger con todas sus columnas (lo que no se pasa queda nulo). Cada cifra mide una clave propia (``k.<id>``) salvo que se pida otra: dos cifras con la misma
    clave, el mismo documento y el mismo alcance y valores distintos son una contradicción documental."""
    clave = clave or f"k.{id_.lower()}"
    if magnitud is None:
        magnitud = "relevante" if estado == "DISCREPA" else "no aplica"
    f = dict.fromkeys(ledger.COLUMNAS_LEDGER)
    f.update(
        id=id_, documento=documento, ubicacion=ubicacion, descripcion=descripcion, tipo=tipo, unidad=unidad, alcance_doc=alcance_doc, requiere_resultado=req,
        valor_doc=None if doc is None else float(doc), decimales=d, clave_eda=clave, clave_usada=clave, valor_medido=None if med is None else float(med),
        ic_bajo=ic[0], ic_alto=ic[1], nivel="extracto", alcance_medido=alcance_med, tabla_origen=tabla, magnitud=magnitud, estado=estado, motivo="", nota="",
    )
    return f


def ledger_df(*filas):
    return pl.DataFrame(list(filas), schema=ledger.ESQUEMA_LEDGER)


def sup(id_, estado="RECHAZADO", *, cifra="cifra del supuesto", motivo="motivo del supuesto", tabla="t_s.csv"):
    """Una fila de ``supuestos_verificados.csv``."""
    return {"id": id_, "supuesto": f"Texto del supuesto {id_}", "criterio": f"Criterio del supuesto {id_}", "estado": estado, "cifra": cifra, "tabla": tabla, "nivel": "extracto",
            "motivo": motivo}


def sup_df(*filas):
    return pl.DataFrame(list(filas), schema=decisiones.ESQUEMA_SUPUESTOS)


def dec_df(semaforos=None):
    """Las 17 decisiones (amarillas salvo las que se pidan)."""
    semaforos = semaforos or {}
    filas = [
        {"n": n, "tema": f"Tema {n}", "propuesta_fase1": f"Propuesta de la decisión {n}", "hipotesis": "H", "tablas_evidencia": "t.csv", "evidencia": "e", "veredicto": "v",
         "tipo_evidencia": "cuantitativa", "semaforo": semaforos.get(n, "amarillo"), "que_falta_con_datos_completos": "", "estado": decisiones.ESTADO_DECISION}
        for n in range(1, 18)
    ]
    return pl.DataFrame(filas, schema=decisiones.ESQUEMA_DECISIONES)


def evid(*, etiqueta="extracto", tablas=None, metricas=None, doc=None, filas=(), supuestos=(), semaforos=None, cifras=None):
    """``Evidencia`` hecha a mano: tablas publicadas, métricas, cifras documentadas, filas del ledger, supuestos y semáforos."""
    fuente = FuenteEvidenciaFalsa(tablas or {}, metricas=metricas or {}, etiqueta=etiqueta, cifras=doc or {})
    return Evidencia(fuente=fuente, ledger=ledger_df(*filas), supuestos=sup_df(*supuestos), decisiones=dec_df(semaforos), cifras=CIFRAS if cifras is None else cifras,
                     hipotesis=HIPOTESIS)


def valido(c):
    """Un cambio bien formado: documento, prioridad y tipo del vocabulario y ningún campo vacío."""
    assert c.documento in cierre.DOCUMENTOS and c.prioridad in cierre.PRIORIDADES and c.tipo in cierre.TIPOS
    for campo in ("ubicacion", "texto_actual", "hallazgo", "texto_propuesto", "evidencia_tabla", "evidencia_clave"):
        assert getattr(c, campo).strip(), campo
    return c


def emitir(hid, ev):
    return HAL[hid].funcion(ev)


def una(hid, ev):
    """El único cambio que emite el hallazgo (la prueba falla si emite otra cantidad)."""
    cs = emitir(hid, ev)
    assert len(cs) == 1, cs
    return valido(cs[0])


def ninguna(hid, ev):
    assert emitir(hid, ev) == [], hid


# ============================================================================================================================ catálogo y reglas generales
def test_el_catalogo_tiene_17_hallazgos_y_cada_id_que_cubre_existe():
    assert len(cierre.HALLAZGOS) == 17 and len(HAL) == 17
    ids_cifras = set(CIFRAS["id"].to_list())
    for h in cierre.HALLAZGOS:
        assert h.ledger_ids <= ids_cifras, h.id
        assert all(s in {f"S{i:02d}" for i in range(1, 16)} for s in h.supuestos), h.id


@pytest.mark.parametrize(
    "base, extracto, completo",
    [
        (cierre.TIPO_ESTADISTICA, "pendiente de datos completos", "corrección"),
        (cierre.TIPO_LISTA, "pendiente de datos completos", "aclaración"),
        (cierre.TIPO_ESTRUCTURAL, "corrección", "corrección"),
        (cierre.TIPO_DEFINICION, "aclaración", "aclaración"),
    ],
)
def test_el_tipo_del_cambio_depende_de_la_base_y_del_modo(base, extracto, completo):
    assert cierre._tipo(evid(), base) == extracto
    assert cierre._tipo(evid(etiqueta="completo"), base) == completo


def test_cada_supuesto_tiene_una_base_de_tipo_valida():
    assert set(cierre.TIPO_SUPUESTO) == {f"S{i:02d}" for i in range(1, 16)}
    assert set(cierre.TIPO_SUPUESTO.values()) <= {cierre.TIPO_ESTADISTICA, cierre.TIPO_ESTRUCTURAL}


@pytest.mark.parametrize(
    "tipo, prioridad, esperada",
    [
        ("aclaración", "media", "baja"), ("aclaración", "alta", "alta"), ("aclaración", "baja", "baja"),
        ("corrección", "alta", "alta"), ("corrección", "media", "media"), ("corrección", "baja", "baja"),
        ("pendiente de datos completos", "alta", "alta"), ("pendiente de datos completos", "media", "media"),
    ],
)
def test_una_aclaracion_no_pasa_de_prioridad_baja_salvo_que_ya_sea_alta(tipo, prioridad, esperada):
    c = cierre.Cambio("Fase 1", "1.1", "actual", "hallazgo", "propuesto", "t.csv", "k", prioridad, tipo)
    assert cierre.prioridad_final(c) == esperada


def test_la_tabla_escribe_la_prioridad_final_y_ordena_con_ella():
    media = cierre.Cambio("Fase 1", "1", "a", "aclaración media", "p", "t", "k", "media", "aclaración")
    corr = cierre.Cambio("Fase 1", "2", "a", "corrección media", "p", "t", "k", "media", "corrección")
    alta = cierre.Cambio("Fase 1", "3", "a", "aclaración alta", "p", "t", "k", "alta", "aclaración")
    t = cierre._a_tabla("Fase 1", [media, corr, alta])
    assert t["hallazgo"].to_list() == ["aclaración alta", "corrección media", "aclaración media"]
    assert t["prioridad"].to_list() == ["alta", "media", "baja"] and t["id"].to_list() == ["CF1-001", "CF1-002", "CF1-003"]
    assert cierre._a_tabla("Fase 2", [corr])["id"].to_list() == ["CF2-001"]


def test_una_aclaracion_en_modo_completo_sale_baja_y_con_la_decision_en_rojo_sale_alta():
    baja = cierre.construir_cambios(evid(etiqueta="completo", tablas=_seleccion()))["Fase 1"].row(0, named=True)
    assert (baja["tipo"], baja["prioridad"]) == ("aclaración", "baja")
    alta = cierre.construir_cambios(evid(etiqueta="completo", tablas=_seleccion(), semaforos={8: "rojo"}))["Fase 1"].row(0, named=True)
    assert (alta["tipo"], alta["prioridad"]) == ("aclaración", "alta")


# ============================================================================================================================ hallazgo: estimados de la Fase 1
def _estimados():
    f1 = dict(documento="Fase 1", ubicacion="9.3")
    return [
        led("C112", doc=75, med=215, descripcion="Estimado de la Fase 1: GRD no agrupable", alcance_med="2019-2024", **f1),
        led("C113", estado="CONFIRMA", doc=7760, med=7760, descripcion="Estimado de la Fase 1: recién nacidos MDC 15", **f1),
        led("C114", estado="CONFIRMA", doc=22303, med=19899, descripcion="Estimado de la Fase 1: obstétricos sin comorbilidades", req="si", **f1),
        led("C115", estado="NO VERIFICABLE", doc=89438, med=None, descripcion="Estimado de la Fase 1: censurados", **f1),
    ]


def test_los_estimados_de_la_fase_1_citan_lo_comparable_y_piden_los_datos_completos():
    c = una("estimados_fase1", evid(filas=_estimados()))
    assert (c.documento, c.ubicacion, c.prioridad, c.tipo) == ("Fase 1", "9.3", "alta", "pendiente de datos completos")
    # el texto actual cita los rótulos de la tabla 9.3 de la Fase 1 (C115 no se cita: la Fase 1 lo da redondeado)
    assert c.texto_actual == ("Cascada de exclusiones, volumen estimado: GRD no asignable, unos 75 casos; recién nacidos sanos (MDC 15), unos 7.760 casos; "
                              "partos normales sin complicación (MDC 14), unos 22.303 casos")
    assert c.hallazgo == (
        "Bajo la hipótesis de que los estimados de la Fase 1 provienen de un extracto, coinciden 2 de 3 comparables: GRD no agrupable: el estimado 75 difiere del conteo medido 215; "
        "recién nacidos MDC 15: el estimado 7.760 es igual al conteo medido 7.760; obstétricos sin comorbilidades: el estimado 22.303 es compatible con el conteo medido de "
        "2019-2023 (19.899) más la cota del año expuesto"
    )
    assert c.texto_propuesto == ("Reemplazar los estimados por los valores medidos con los datos completos (Fase 2, sección 5.2) y dejar constancia de que algunos estimados "
                                 "provienen de un extracto")
    assert (c.evidencia_tabla, c.evidencia_clave) == ("ledger_cifras.csv", "C112; C113; C114; C115")


def test_los_estimados_de_la_fase_1_en_modo_completo_son_una_correccion():
    assert una("estimados_fase1", evid(etiqueta="completo", filas=_estimados())).tipo == "corrección"


def test_los_estimados_de_la_fase_1_sin_discrepancias_tienen_prioridad_baja():
    filas = [led("C112", estado="CONFIRMA", doc=75, med=75, documento="Fase 1", ubicacion="9.3", descripcion="Estimado de la Fase 1: GRD no agrupable")] + _estimados()[1:]
    c = una("estimados_fase1", evid(filas=filas))
    assert c.prioridad == "baja" and "coinciden 3 de 3 comparables" in c.hallazgo and "difiere" not in c.hallazgo


def test_los_estimados_de_la_fase_1_sin_filas_comparables_no_se_emiten():
    ninguna("estimados_fase1", evid())
    ninguna("estimados_fase1", evid(filas=[led(c, estado="NO VERIFICABLE", med=None, documento="Fase 1") for c in ("C112", "C113", "C114", "C115")]))


# ============================================================================================================================ hallazgo: entre 10 y 20 veces
def _pares():
    """Cuatro pares (Fase 2, Fase 1) con la misma clave y razones de 4, 66,67, 16,67 y 15."""
    filas = []
    for k, (v2, v1, nombre) in enumerate(
        ((400, 100, "censurados por tipo de alta"), (1000, 15, "GRD no agrupable (EX01)"), (5000, 300, "recién nacidos MDC 15 (EX02)"),
         (3000, 200, "obstétricos sin comorbilidades MDC 14 (EX03)"))
    ):
        filas.append(cifra_fila(f"C3{k}0", valor=v2, tipo="absoluto", unidad="conteo", decimales=0, clave=f"k.{k}", documento="Fase 2", descripcion=f"Cascada: {nombre}"))
        filas.append(cifra_fila(f"C3{k}1", valor=v1, tipo="absoluto", unidad="conteo", decimales=0, clave=f"k.{k}", documento="Fase 1", descripcion=f"Estimado de la Fase 1: {nombre}"))
    return cifras_df(*filas)


def _razones(estado="DISCREPA"):
    return [
        led("C128", estado=estado, doc=10, med=4, tipo="aritmetica", unidad="razon", d=2),
        led("C129", estado=estado, doc=20, med=66.67, tipo="aritmetica", unidad="razon", d=2),
        led("C113", estado="CONFIRMA", doc=7760, med=7760, documento="Fase 1", descripcion="Estimado de la Fase 1: recién nacidos MDC 15"),
        led("C114", estado="CONFIRMA", doc=22303, med=19899, documento="Fase 1", descripcion="Estimado de la Fase 1: obstétricos sin comorbilidades MDC 14"),
    ]


def test_la_frase_de_entre_10_y_20_veces_se_contrasta_con_las_razones_recalculadas():
    ev = evid(filas=_razones(), cifras=_pares(), metricas={"volumen.filas_leidas.total": metrica(1000)}, doc={"C007": 19360})
    c = una("entre_10_y_20", ev)
    assert (c.documento, c.ubicacion, c.prioridad, c.tipo) == ("Fase 2", "5.2", "media", "corrección")
    assert c.texto_actual == "Los estimados de la Fase 1 estaban entre 10 y 20 veces más bajos que lo medido."
    assert c.hallazgo == (
        "Razón de lo medido sobre el estimado de la Fase 1: censurados por tipo de alta 4,00; GRD no agrupable (EX01) 66,67; recién nacidos MDC 15 (EX02) 16,67; "
        "obstétricos sin comorbilidades MDC 14 (EX03) 15,00. Caen entre 10 y 20 veces: recién nacidos MDC 15 (EX02) y obstétricos sin comorbilidades MDC 14 (EX03). "
        "El inverso de la fracción de muestreo del extracto es 19,36. Recién nacidos MDC 15 y obstétricos sin comorbilidades MDC 14: el estimado de la Fase 1 es compatible con "
        "el conteo del extracto, de modo que la razón refleja el muestreo"
    )
    assert c.texto_propuesto.startswith("Reemplazar la frase por las razones medidas") and (c.evidencia_tabla, c.evidencia_clave) == ("ledger_cifras.csv", "C128; C129")


def test_en_modo_completo_la_frase_de_entre_10_y_20_no_habla_del_muestreo_del_extracto():
    ev = evid(etiqueta="completo", filas=_razones(), cifras=_pares(), metricas={"volumen.filas_leidas.total": metrica(1000)}, doc={"C007": 19360})
    h = una("entre_10_y_20", ev).hallazgo
    assert "muestreo" not in h and "extracto" not in h and h.endswith("(EX03)") and "Caen entre 10 y 20 veces" in h


def test_entre_10_y_20_con_el_rango_documentado_distinto_y_ninguna_razon_dentro():
    filas = [led("C128", doc=2, med=4, tipo="aritmetica"), led("C129", estado="CONFIRMA", doc=30, med=66.67, tipo="aritmetica")]
    h = una("entre_10_y_20", evid(filas=filas, cifras=_pares())).hallazgo
    assert "Caen entre 2 y 30 veces: censurados por tipo de alta" in h                                     # 4,00; 16,67 y 15,00 caen en [2, 30]; 66,67 no
    filas = [led("C128", doc=100, med=4, tipo="aritmetica"), led("C129", doc=200, med=66.67, tipo="aritmetica")]
    assert una("entre_10_y_20", evid(filas=filas, cifras=_pares())).hallazgo.endswith("Caen entre 100 y 200 veces: ninguna")


def test_las_razones_exactamente_en_los_extremos_del_rango_documentado_caen_dentro():
    filas = []
    for k, (v2, v1, nombre) in enumerate(((1000, 100, "diez veces (EX01)"), (2000, 100, "veinte veces (EX02)"), (2100, 100, "veintiuna veces (EX03)"), (990, 100, "casi diez (EX04)"))):
        filas.append(cifra_fila(f"C4{k}0", valor=v2, tipo="absoluto", unidad="conteo", decimales=0, clave=f"k.{k}", documento="Fase 2", descripcion=f"Cascada: {nombre}"))
        filas.append(cifra_fila(f"C4{k}1", valor=v1, tipo="absoluto", unidad="conteo", decimales=0, clave=f"k.{k}", documento="Fase 1", descripcion=f"Estimado de la Fase 1: {nombre}"))
    h = una("entre_10_y_20", evid(filas=_razones(), cifras=cifras_df(*filas))).hallazgo
    assert "diez veces (EX01) 10,00; veinte veces (EX02) 20,00; veintiuna veces (EX03) 21,00; casi diez (EX04) 9,90" in h
    assert "Caen entre 10 y 20 veces: diez veces (EX01) y veinte veces (EX02)" in h            # 10,00 y 20,00 están dentro; 9,90 y 21,00 no


def test_entre_10_y_20_no_se_emite_si_el_documento_se_confirma_o_no_hay_pares():
    ninguna("entre_10_y_20", evid(filas=_razones("CONFIRMA"), cifras=_pares()))
    ninguna("entre_10_y_20", evid(filas=_razones(), cifras=cifras_df()))
    ninguna("entre_10_y_20", evid(cifras=_pares()))


# ============================================================================================================================ hallazgo: cinco categorías de Elixhauser
def _seleccion(n_f1=3, n_repo=2, top5="ELIX_24; ELIX_04; ELIX_25; ELIX_30; ELIX_27"):
    return {"t23_resumen_seleccion.csv": df({"elix_top5": top5, "elix_n_coinciden_fase1": n_f1, "elix_n_coinciden_repo": n_repo})}


def test_las_cinco_categorias_de_la_fase_1_se_comparan_con_el_indice_de_complicacion():
    c = una("cinco_categorias", evid(tablas=_seleccion()))
    assert (c.documento, c.ubicacion, c.prioridad, c.tipo) == ("Fase 1", "8.2 y decisión 8", "media", "pendiente de datos completos")
    assert c.texto_actual == cierre.TEXTO_CINCO_FASE1
    assert c.hallazgo == ("El índice de complicación probable ordena ELIX_24, ELIX_04, ELIX_25, ELIX_30, ELIX_27 como las cinco categorías más propensas a ser complicaciones "
                          "de la hospitalización; coinciden 3 con la lista de la Fase 1 y 2 con la del repositorio (Fase 2, sección 6.1)")
    assert c.texto_propuesto == ("Dejar una sola lista de cinco categorías para la sensibilidad, a decidir con revisión clínica (decisión 12); la medición propone ELIX_24, ELIX_04, "
                                 "ELIX_25, ELIX_30, ELIX_27")
    assert (c.evidencia_tabla, c.evidencia_clave) == ("t23_resumen_seleccion.csv", "elix_top5; elix_n_coinciden_fase1; elix_n_coinciden_repo")


def test_las_cinco_categorias_cambian_con_la_tabla_el_modo_y_el_semaforo():
    assert "coinciden 4 con la lista de la Fase 1 y 0 con la del repositorio" in una("cinco_categorias", evid(tablas=_seleccion(4, 0))).hallazgo
    assert una("cinco_categorias", evid(etiqueta="completo", tablas=_seleccion())).tipo == "aclaración"          # una lista distinta se aclara con los datos completos
    assert una("cinco_categorias", evid(tablas=_seleccion(), semaforos={8: "rojo"})).prioridad == "alta"
    assert una("cinco_categorias", evid(tablas=_seleccion(), semaforos={7: "rojo"})).prioridad == "media"      # el rojo de otra decisión no cuenta


def test_las_cinco_categorias_no_se_emiten_si_coinciden_las_cinco_o_falta_el_dato():
    ninguna("cinco_categorias", evid(tablas=_seleccion(5, 5)))
    ninguna("cinco_categorias", evid())
    ninguna("cinco_categorias", evid(tablas=_seleccion(n_f1=None)))
    ninguna("cinco_categorias", evid(tablas={"t23_resumen_seleccion.csv": df({"elix_top5": "x", "elix_n_coinciden_fase1": 3, "elix_n_coinciden_repo": 2},
                                                                        {"elix_top5": "y", "elix_n_coinciden_fase1": 3, "elix_n_coinciden_repo": 2})}))      # dos filas: ambiguo


# ============================================================================================================================ hallazgo: 28 categorías del repositorio
def test_las_28_categorias_del_repositorio_se_aclaran_cuando_la_suite_implementa_otro_numero():
    c = una("categorias_del_repositorio", evid(metricas={"elix.n_categorias": metrica(31)}))
    assert (c.documento, c.ubicacion, c.prioridad, c.tipo, c.evidencia_clave) == ("Fase 2", "6.1", "baja", "aclaración", "elix.n_categorias")
    assert c.hallazgo == ("La suite implementa 31 categorías de Elixhauser con el mapeo de Quan; la cifra de 28 categorías del repositorio no se puede medir aquí porque su código "
                          "no forma parte de la suite")
    assert una("categorias_del_repositorio", evid(etiqueta="completo", metricas={"elix.n_categorias": metrica(31)})).tipo == "aclaración"


def test_las_28_categorias_no_se_emiten_si_la_suite_implementa_28_o_falta_la_metrica():
    ninguna("categorias_del_repositorio", evid(metricas={"elix.n_categorias": metrica(28)}))
    ninguna("categorias_del_repositorio", evid())
    ninguna("categorias_del_repositorio", evid(metricas={"elix.n_categorias": metrica(None)}))


# ============================================================================================================================ hallazgo: HOSPPROCEDENCIA
def _hospproc(supera="si", prop=0.3064):
    return {"t23_resumen_hospitales.csv": df({"autorref_prop_exacta": prop, "autorref_supera_maximo": supera})}


def test_hospprocedencia_no_sirve_para_definir_el_traslado_entrante_si_supera_el_maximo():
    filas = [led("C120", doc=703854, documento="Fase 2"), led("C121", doc=12.1, unidad="porcentaje", d=1)]
    c = una("hospproc", evid(tablas=_hospproc(), filas=filas))
    assert (c.documento, c.ubicacion, c.prioridad, c.tipo) == ("Fase 2", "5.4", "media", "pendiente de datos completos")
    assert c.texto_actual == ("Entrantes. Hay 703.854 episodios con hospital de procedencia (12,1 %). Hay una contradicción que resolver: la Fase 1 excluye los traslados "
                              "entrantes de los modelos, pero DERIVADO_OTRO_HOSPITAL figura como predictor condicional.")
    assert c.hallazgo == ("HOSPPROCEDENCIA trae el nombre del hospital y coincide con el hospital del egreso en 30,6 % de los egresos que lo informan, sobre el máximo aceptado "
                          "de 5 %: no sirve para definir el traslado entrante")
    assert c.texto_propuesto == "Definir el traslado entrante con TIPO_PROCEDENCIA (otro hospital de la red) y no con HOSPPROCEDENCIA"
    assert c.evidencia_tabla == "t23_resumen_hospitales.csv; t23_procedencia_autorref.csv"


def test_hospprocedencia_sin_las_cifras_documentadas_conserva_la_frase_fija_y_sube_con_el_rojo():
    c = una("hospproc", evid(tablas=_hospproc()))
    assert c.texto_actual == ("Entrantes. Hay una contradicción que resolver: la Fase 1 excluye los traslados entrantes de los modelos, pero DERIVADO_OTRO_HOSPITAL figura como "
                              "predictor condicional.")
    assert una("hospproc", evid(tablas=_hospproc(), semaforos={16: "rojo"})).prioridad == "alta"
    assert una("hospproc", evid(tablas=_hospproc(), semaforos={2: "rojo"})).prioridad == "alta"
    assert una("hospproc", evid(tablas=_hospproc(), semaforos={3: "rojo"})).prioridad == "media"


def test_hospprocedencia_no_se_emite_bajo_el_maximo_ni_sin_tabla():
    ninguna("hospproc", evid(tablas=_hospproc("no", 0.01)))
    ninguna("hospproc", evid())
    ninguna("hospproc", evid(tablas=_hospproc(prop=None)))


# ============================================================================================================================ hallazgo: estadías muy largas
def _cola(*filas):
    base = {"base": "E", "regla": "p99", "corte_dias": 51, "cumple_def": "si", "cumple_conf": "si", "propuesta": "si", "prop_afectados_def": 0.0097, "prop_afectados_conf": 0.0078}
    return {"t23_cola_estancia.csv": df(*({**base, **f} for f in filas))}


def test_la_regla_de_estadias_largas_prefiere_la_base_e_y_cita_a_quienes_afecta():
    t = _cola({"base": "A", "regla": "p99", "corte_dias": 40, "prop_afectados_def": 0.01}, {"regla": "p99,5", "corte_dias": 66, "prop_afectados_def": 0.0049, "prop_afectados_conf": 0.0051})
    c = una("cola_estancia", evid(tablas=t))
    assert (c.documento, c.ubicacion, c.prioridad, c.tipo) == ("Fase 1", "decisión 7", "media", "pendiente de datos completos")
    assert c.texto_actual == "Estadías muy largas. Definir la regla en la Fase 2 con los datos de 2019 a 2022 solamente."
    assert c.hallazgo == "La regla p99,5 (corte en 66 días, base E) afecta a 0,49 % de los sobrevivientes de 2019 a 2022 y a 0,51 % en 2023; se confirma en 2023"
    assert c.texto_propuesto == "Fijar la regla de estadías largas en p99,5 (66 días), confirmada en 2023"
    assert (c.evidencia_tabla, c.evidencia_clave) == ("t23_cola_estancia.csv", "regla; corte_dias; propuesta; cumple_conf")


def test_la_regla_de_estadias_largas_es_una_correccion_solo_si_es_completo_y_se_confirma():
    assert una("cola_estancia", evid(etiqueta="completo", tablas=_cola({}))).tipo == "corrección"
    no_conf = _cola({"cumple_conf": "no"})
    c = una("cola_estancia", evid(etiqueta="completo", tablas=no_conf))
    assert c.tipo == "pendiente de datos completos" and c.hallazgo.endswith("; no se confirma en 2023")
    assert c.texto_propuesto == "Fijar la regla de estadías largas en p99 (51 días) y confirmarla con los datos completos"


def test_la_regla_de_estadias_largas_sin_fracciones_y_con_prioridad_por_el_rojo():
    t = _cola({"prop_afectados_def": None, "prop_afectados_conf": None})
    assert una("cola_estancia", evid(tablas=t)).hallazgo == "La regla p99 (corte en 51 días, base E) afecta a una fracción de los sobrevivientes de 2019 a 2022; se confirma en 2023"
    assert una("cola_estancia", evid(tablas=_cola({}), semaforos={7: "rojo"})).prioridad == "alta"


def test_la_regla_de_estadias_largas_no_se_emite_sin_propuesta_ni_datos():
    ninguna("cola_estancia", evid(tablas=_cola({"propuesta": "no"})))
    ninguna("cola_estancia", evid())
    ninguna("cola_estancia", evid(tablas=_cola({"cumple_conf": None})))
    ninguna("cola_estancia", evid(tablas={"t23_cola_estancia.csv": df({"base": "E", "regla": "p99"})}))          # sin la columna propuesta


# ============================================================================================================================ hallazgo: mapeo de tres caracteres
def _elix3c(*filas):
    return {"t23_elix_vs_3c.csv": df(*({"elix_id": i, "supera_umbral": s, "prev_completo_ah": c, "prev_3c_ah": t} for i, s, c, t in filas))}


TRES_C = _elix3c(("ELIX_11", "si", 0.148, 0.0), ("ELIX_12", "si", 0.042, 0.0), ("ELIX_01", "no", 0.2, 0.19))


def test_el_mapeo_de_tres_caracteres_cita_las_categorias_que_difieren_y_la_diabetes():
    c = una("mapeo_tres_caracteres", evid(tablas=TRES_C))
    assert (c.documento, c.ubicacion, c.prioridad, c.tipo) == ("Fase 2", "3.5", "media", "corrección")
    assert c.hallazgo == (
        "Con el mapeo de tres caracteres en vez del mapeo completo, 2 de 3 categorías difieren más del umbral de la prevalencia relativa en adultos hospitalizados "
        "(ELIX_11 y ELIX_12); diabetes no complicada (ELIX_11) 14,8 % con el mapeo completo y 0,0 % con tres caracteres; diabetes complicada (ELIX_12) 4,2 % con el mapeo "
        "completo y 0,0 % con tres caracteres"
    )
    assert c.texto_propuesto == "Usar el mapeo completo de Quan (códigos de tres y de cuatro caracteres sin puntos) en la suite y en el repositorio"
    assert c.evidencia_tabla == "t23_elix_vs_3c.csv; t23_resumen_comorbilidad.csv"


def test_el_mapeo_de_tres_caracteres_sube_a_prioridad_alta_si_el_supuesto_s02_se_rechaza():
    assert una("mapeo_tres_caracteres", evid(tablas=TRES_C, supuestos=[sup("S02")])).prioridad == "alta"
    assert una("mapeo_tres_caracteres", evid(tablas=TRES_C, supuestos=[sup("S02", "ACEPTADO")])).prioridad == "media"
    # rechazado el supuesto aunque ninguna categoría supere el umbral: se emite sin paréntesis vacíos
    sin_sobre = _elix3c(("ELIX_11", "no", 0.148, 0.14), ("ELIX_01", "no", 0.2, 0.19))
    h = una("mapeo_tres_caracteres", evid(tablas=sin_sobre, supuestos=[sup("S02")])).hallazgo
    assert h.startswith("Con el mapeo de tres caracteres en vez del mapeo completo, 0 de 2 categorías") and "()" not in h


def test_el_mapeo_de_tres_caracteres_no_se_emite_sin_categorias_sobre_el_umbral_ni_tabla():
    ninguna("mapeo_tres_caracteres", evid(tablas=_elix3c(("ELIX_11", "no", 0.148, 0.14))))
    ninguna("mapeo_tres_caracteres", evid())
    ninguna("mapeo_tres_caracteres", evid(tablas={"t23_elix_vs_3c.csv": df({"elix_id": "ELIX_11", "prev_completo_ah": 0.1, "prev_3c_ah": 0.1})}))      # sin supera_umbral


# ============================================================================================================================ hallazgo: colinealidad exacta
def _vif(*filas):
    return {"t23_vif.csv": df(*({"conjunto": c, "variable": v, "es_agregado": a, "colinealidad_exacta": e} for c, v, a, e in filas))}


def test_la_colinealidad_exacta_nombra_los_agregados_y_cuenta_las_categorias():
    t = _vif(("con_agregados", "N_COMORB_ELIX", "si", "si"), ("con_agregados", "PUNTAJE_VAN_WALRAVEN", "si", "si"), ("con_agregados", "PUNTAJE_VAN_WALRAVEN_SIN_5", "si", "no"),
             ("con_agregados", "ELIX_01", "no", "si"), ("con_agregados", "ELIX_02", "no", "si"), ("con_agregados", "ELIX_03", "no", "no"), ("sin_agregados", "ELIX_01", "no", "si"))
    c = una("colinealidad_exacta", evid(tablas=t))
    assert (c.documento, c.ubicacion, c.prioridad, c.tipo) == ("Fase 1", "8.3", "media", "corrección")
    assert c.hallazgo == ("Con las categorías de Elixhauser en el mismo modelo, N_COMORB_ELIX y PUNTAJE_VAN_WALRAVEN son combinaciones lineales exactas de ellas: la colinealidad "
                          "exacta alcanza a 2 categorías del conjunto con agregados")
    assert c.texto_propuesto == ("Usar el conteo de comorbilidades o el puntaje de van Walraven solo en la línea base simple y nunca junto con las 31 categorías en el mismo modelo")
    assert (c.evidencia_tabla, c.evidencia_clave) == ("t23_vif.csv", "conjunto = con_agregados; colinealidad_exacta = si")


def test_la_colinealidad_exacta_no_se_emite_si_ningun_agregado_es_exacto_o_solo_lo_es_sin_agregados():
    ninguna("colinealidad_exacta", evid(tablas=_vif(("con_agregados", "N_COMORB_ELIX", "si", "no"), ("con_agregados", "ELIX_01", "no", "si"))))
    ninguna("colinealidad_exacta", evid(tablas=_vif(("sin_agregados", "N_COMORB_ELIX", "si", "si"))))
    ninguna("colinealidad_exacta", evid())


# ============================================================================================================================ hallazgo: AUC del modelo L1
def _l1(*variables, auc=0.925):
    t = df({"elix_top5": "x", "auc_modelo_l1_2023": auc, "auc_ref_edad_sexo_2023": 0.767, "auc_ref_elix31_2023": 0.821})
    imp = df(*({"tipo_fila": "variable", "variable": v, "clase_proxy": cl, "en_modelo_l1": en, "importa": im} for v, cl, en, im in variables))
    return {"t23_resumen_seleccion.csv": t, "t23_importancia.csv": imp}


VARIABLES_L1 = (("SERVICIOINGRESO", "alto", "si", "si"), ("TIPO_INGRESO", "moderado", "si", "si"), ("EDAD", "bajo", "si", "si"), ("PROVINCIA", "alto", "no", "si"),
                ("SEXO", "moderado", "si", "no"))


def test_el_auc_del_modelo_l1_no_se_presenta_como_desempeno_esperable_si_incluye_variables_cercanas_a_proxy():
    c = una("auc_l1", evid(tablas=_l1(*VARIABLES_L1)))
    assert (c.documento, c.ubicacion, c.prioridad, c.tipo) == ("Fase 1", "Requisito: control de variables prohibidas en el modelo", "media", "pendiente de datos completos")
    assert c.hallazgo == (
        "El modelo L1 llega a un AUC de 0,925 en 2023, frente a 0,767 del modelo de edad y sexo y 0,821 del de las 31 categorías. Incluye variables con asociación moderada o alta "
        "con el hospital que importan al permutarlas (SERVICIOINGRESO y TIPO_INGRESO): están cerca de ser proxy o fuga y no hay un L1 sin ellas, de modo que ese AUC no es un "
        "desempeño esperable"
    )
    assert c.texto_propuesto.startswith("No presentar el AUC del L1 como desempeño esperable") and c.evidencia_tabla == "t23_resumen_seleccion.csv; t23_importancia.csv"
    assert "0,931" in una("auc_l1", evid(tablas=_l1(*VARIABLES_L1, auc=0.931))).hallazgo


def test_el_auc_del_modelo_l1_no_se_emite_sin_variables_cercanas_a_proxy_ni_tablas():
    ninguna("auc_l1", evid(tablas=_l1(("EDAD", "bajo", "si", "si"), ("PROVINCIA", "alto", "no", "si"), ("SEXO", "moderado", "si", "no"))))
    ninguna("auc_l1", evid(tablas={"t23_resumen_seleccion.csv": _l1()["t23_resumen_seleccion.csv"]}))
    ninguna("auc_l1", evid())
    ninguna("auc_l1", evid(tablas={**_l1(*VARIABLES_L1), "t23_resumen_seleccion.csv": df({"elix_top5": "x", "auc_modelo_l1_2023": None, "auc_ref_edad_sexo_2023": 0.7,
                                                                                         "auc_ref_elix31_2023": 0.8})}))


# ============================================================================================================================ hallazgo: variables que superan el umbral de fuga
def _fuga(*filas, concordancia=None, con_traslado=None):
    base = {"tipo_fila": "variable", "supera_a_basales": "si", "supera_umbral_fuga": "si", "prohibida_siempre": "no", "auc_ranking": 0.9, "auc_confirmacion": 0.9, "prop_igualdad": None,
            "ic_bajo": None, "ic_alto": None, "coincide_mayoria": None}
    todas = [{**base, **f} for f in filas]
    for tipo, valores in (("concordancia", concordancia), ("concordancia_con_traslado", con_traslado)):
        if valores is not None:
            todas.append({**base, "tipo_fila": tipo, "variable": "SERVICIO_INGRESO_IGUAL_ALTA", "supera_a_basales": None, "supera_umbral_fuga": None,
                          "prohibida_siempre": None, "auc_ranking": None, "auc_confirmacion": None, **valores})
    return {"t23_fuga.csv": df(*todas)}


SEVERIDAD = {"variable": "IR_29301_SEVERIDAD", "auc_ranking": 0.842, "auc_confirmacion": 0.858}
MORTALIDAD = {"variable": "IR_29301_MORTALIDAD", "auc_ranking": 0.860, "auc_confirmacion": 0.871, "prohibida_siempre": "si"}
SERVICIO = {"variable": "SERVICIOINGRESO", "supera_a_basales": "no", "supera_umbral_fuga": "no", "auc_ranking": 0.797, "auc_confirmacion": 0.796}


def test_la_fuga_cita_el_auc_de_cada_variable_y_las_ventanas_que_superan_el_umbral():
    c = una("fuga", evid(tablas=_fuga(SEVERIDAD, MORTALIDAD, SERVICIO)))
    assert (c.documento, c.ubicacion, c.prioridad, c.tipo) == ("Fase 1", "R6 y decisión 14", "alta", "pendiente de datos completos")
    assert c.hallazgo == (
        "Variables que superan el umbral de fuga: IR_29301_SEVERIDAD: AUC de 0,842 en el ranking y de 0,858 en la confirmación, sobre 0,85 en la confirmación de 2023; "
        "IR_29301_MORTALIDAD: AUC de 0,860 en el ranking y de 0,871 en la confirmación, sobre 0,85 en el ranking de 2019 a 2022 y la confirmación de 2023"
    )
    assert c.texto_propuesto == ("Mantener estas variables fuera de los modelos principales; IR_29301_MORTALIDAD queda prohibida siempre (R6) y la severidad del GRD solo en una "
                                 "sensibilidad aparte (decisión 14)")
    assert c.texto_actual.startswith("R6:") and "Decisión 14" in c.texto_actual
    assert (c.evidencia_tabla, c.evidencia_clave) == ("t23_fuga.csv", "supera_umbral_fuga; auc_ranking; auc_confirmacion")


def test_la_fuga_solo_sube_a_prioridad_alta_si_la_variable_prohibida_siempre_la_supera():
    assert una("fuga", evid(tablas=_fuga(SEVERIDAD, SERVICIO))).prioridad == "media"                 # R6 no está en juego
    assert una("fuga", evid(tablas=_fuga(MORTALIDAD))).prioridad == "alta"
    # una variable prohibida que NO supera el umbral no cuenta
    assert una("fuga", evid(tablas=_fuga(SEVERIDAD, {**MORTALIDAD, "supera_umbral_fuga": "no"}))).prioridad == "media"
    assert una("fuga", evid(etiqueta="completo", tablas=_fuga(SEVERIDAD))).tipo == "corrección"


def test_la_fuga_incluye_el_valor_igual_al_umbral_y_omite_la_ventana_sin_dato():
    c = una("fuga", evid(tablas=_fuga({**SEVERIDAD, "auc_ranking": 0.85, "auc_confirmacion": 0.84})))
    assert "sobre 0,85 en el ranking de 2019 a 2022" in c.hallazgo and "la confirmación de 2023" not in c.hallazgo                       # 0,85 exacto cuenta
    c = una("fuga", evid(tablas=_fuga({**SEVERIDAD, "auc_ranking": None, "auc_confirmacion": 0.9})))
    assert "sobre 0,85 en la confirmación de 2023" in c.hallazgo and "ranking de 2019 a 2022" not in c.hallazgo.split("sobre")[1]


def test_la_fuga_no_se_emite_si_ninguna_variable_supera_el_umbral():
    ninguna("fuga", evid(tablas=_fuga({**SEVERIDAD, "supera_umbral_fuga": "no"}, SERVICIO)))
    ninguna("fuga", evid(tablas=_fuga(concordancia={"prop_igualdad": 0.99, "ic_bajo": 0.98, "ic_alto": 0.995, "coincide_mayoria": "si"})))
    ninguna("fuga", evid())
    ninguna("fuga", evid(tablas={"t23_fuga.csv": df({"tipo_fila": "variable", "variable": "X"})}))              # sin la columna supera_umbral_fuga


# ============================================================================================================================ hallazgo: SERVICIOINGRESO
CONCORDANCIA = {"prop_igualdad": 0.992, "ic_bajo": 0.9914, "ic_alto": 0.9926, "coincide_mayoria": "si"}


CON_TRASLADO_A_FAVOR = {"prop_igualdad": 0.80, "ic_bajo": 0.79, "ic_alto": 0.81, "coincide_mayoria": "si"}
CON_TRASLADO_EN_CONTRA = {"prop_igualdad": 0.152, "ic_bajo": 0.148, "ic_alto": 0.157, "coincide_mayoria": "no"}
SIN_TRASLADO_TXT = "SERVICIOINGRESO coincide con SERVICIOALTA en 99,2 % de los egresos sin traslado interno (intervalo de 99,1 % a 99,3 %)"


def test_servicio_ingreso_que_coincide_con_el_de_alta_incluso_con_traslado_se_trata_como_reflejo_de_la_estadia():
    c = una("servicio_ingreso", evid(tablas=_fuga(concordancia=CONCORDANCIA, con_traslado=CON_TRASLADO_A_FAVOR)))
    assert (c.documento, c.ubicacion, c.prioridad, c.tipo) == ("Fase 1", "decisión 2", "media", "pendiente de datos completos")
    assert c.texto_actual == cierre.TEXTO_DECISION2_FASE1 and c.texto_actual.startswith("Variables condicionales. Decidir sobre servicio de ingreso") and c.texto_actual.endswith("se descarta.")
    assert c.hallazgo == (SIN_TRASLADO_TXT + " y en 80,0 % de los egresos con traslado interno (intervalo de 79,0 % a 81,0 %): en la práctica refleja el servicio de la estadía y no el del ingreso")
    assert c.texto_propuesto == ("Aplicar la regla de la propia decisión: SERVICIOINGRESO coincide en su mayoría con el servicio de alta, también en los egresos con traslado interno, "
                                 "de modo que se descartaría del modelo principal; confirmar qué registra antes de usarla")
    assert (c.evidencia_tabla, c.evidencia_clave) == ("t23_fuga.csv", "tipo_fila = concordancia; tipo_fila = concordancia_con_traslado; prop_igualdad; coincide_mayoria")


def test_servicio_ingreso_con_la_fila_con_traslado_en_contra_compara_las_dos_proporciones_y_no_dice_que_refleja_la_estadia():
    # sin traslado interno la igualdad es esperable por construcción y no distingue; con traslado interno no coincide en la mayoría (15,2 %): S09 queda aceptado (cambio 1.5)
    c = una("servicio_ingreso", evid(tablas=_fuga(concordancia=CONCORDANCIA, con_traslado=CON_TRASLADO_EN_CONTRA)))
    assert c.hallazgo == (SIN_TRASLADO_TXT + " y en 15,2 % de los egresos con traslado interno (intervalo de 14,8 % a 15,7 %): sin traslado interno la igualdad es esperable por "
                          "construcción y no distingue el ingreso de la estadía, y con traslado interno no coincide en la mayoría, lo que se espera si registra el ingreso")
    assert c.texto_propuesto == ("Dejar constancia de que, con esta evidencia, SERVICIOINGRESO registra el servicio de ingreso y no el de la estadía, de modo que la regla de descarte de "
                                 "la decisión 2 no se activa por esa razón; esto no levanta las sospechas de proxy del hospital ni de fuga de la Tarea 11 sobre la variable, que la matriz "
                                 "de decisión sigue evaluando; confirmar con los datos completos")
    assert "refleja el servicio de la estadía" not in c.hallazgo and "se descartaría" not in c.texto_propuesto
    assert "no levanta las sospechas de proxy del hospital ni de fuga" in c.texto_propuesto and "confirmar con los datos completos" in c.texto_propuesto
    assert (c.documento, c.ubicacion, c.prioridad, c.tipo) == ("Fase 1", "decisión 2", "media", "pendiente de datos completos")
    assert c.evidencia_clave == "tipo_fila = concordancia; tipo_fila = concordancia_con_traslado; prop_igualdad; coincide_mayoria"


@pytest.mark.parametrize("con_traslado", [None, {"prop_igualdad": None, "ic_bajo": None, "ic_alto": None, "coincide_mayoria": None}])
def test_servicio_ingreso_sin_la_fila_con_traslado_o_con_sus_cifras_suprimidas_no_concluye(con_traslado):
    c = una("servicio_ingreso", evid(tablas=_fuga(concordancia=CONCORDANCIA, con_traslado=con_traslado)))
    assert c.hallazgo == (SIN_TRASLADO_TXT + ": esa igualdad es esperable por construcción y no distingue el ingreso de la estadía; falta la fila con traslado interno, que es la prueba "
                          "informativa")
    assert c.texto_propuesto == "Medir también la coincidencia en los egresos con traslado interno antes de aplicar la regla de la propia decisión sobre SERVICIOINGRESO"
    assert "refleja el servicio de la estadía" not in c.hallazgo and c.evidencia_clave == "tipo_fila = concordancia; prop_igualdad; coincide_mayoria"


def test_servicio_ingreso_sube_a_prioridad_alta_con_el_supuesto_s09_rechazado_o_la_decision_2_en_rojo():
    tablas = _fuga(concordancia=CONCORDANCIA, con_traslado=CON_TRASLADO_A_FAVOR)
    assert una("servicio_ingreso", evid(tablas=tablas, supuestos=[sup("S09")])).prioridad == "alta"
    assert una("servicio_ingreso", evid(tablas=tablas, semaforos={2: "rojo"})).prioridad == "alta"
    assert una("servicio_ingreso", evid(tablas=tablas, supuestos=[sup("S09", "ACEPTADO")], semaforos={3: "rojo"})).prioridad == "media"
    # con la fila con traslado en contra el supuesto queda aceptado (cambio 1.5 del protocolo) y la prioridad baja a media, salvo que la decisión 2 esté en rojo; sin evaluar
    # el supuesto tampoco sube (la regla es la de siempre: S09 rechazado o la decisión 2 en rojo)
    en_contra = _fuga(concordancia=CONCORDANCIA, con_traslado=CON_TRASLADO_EN_CONTRA)
    assert una("servicio_ingreso", evid(tablas=en_contra, supuestos=[sup("S09", "ACEPTADO")])).prioridad == "media"
    assert una("servicio_ingreso", evid(tablas=en_contra, supuestos=[sup("S09", "NO EVALUABLE")])).prioridad == "media"
    assert una("servicio_ingreso", evid(tablas=en_contra, semaforos={2: "rojo"})).prioridad == "alta"


def test_servicio_ingreso_no_se_emite_si_la_coincidencia_no_es_mayoritaria_o_falta_el_dato():
    ninguna("servicio_ingreso", evid(tablas=_fuga(concordancia={**CONCORDANCIA, "coincide_mayoria": "no"})))
    ninguna("servicio_ingreso", evid(tablas=_fuga(concordancia={**CONCORDANCIA, "coincide_mayoria": "no"}, con_traslado=CON_TRASLADO_A_FAVOR)))
    ninguna("servicio_ingreso", evid(tablas=_fuga(concordancia={**CONCORDANCIA, "prop_igualdad": None})))
    ninguna("servicio_ingreso", evid(tablas=_fuga(SEVERIDAD)))
    ninguna("servicio_ingreso", evid(tablas=_fuga(con_traslado=CON_TRASLADO_A_FAVOR)))
    ninguna("servicio_ingreso", evid())


# ============================================================================================================================ hallazgo: diagnósticos secundarios
def _dx(*, estados=("DISCREPA", "DISCREPA", "CONFIRMA", "DISCREPA", "CONFIRMA", "CONFIRMA", "DISCREPA"), mag76="relevante", sin_medido=()):
    docs = (4.08, 4.84, 5.37, 5.33, 5.56, 5.88)
    meds = (4.14, 4.80, 5.30, 5.40, 5.50, 5.82)
    filas = []
    for i, (cid, d_, m_) in enumerate(zip(("C070", "C071", "C072", "C073", "C074", "C075"), docs, meds)):
        filas.append(led(cid, estado=estados[i], doc=d_, med=None if cid in sin_medido else m_, tipo="proporcion", unidad="razon", d=2, ubicacion="3.6"))
    filas.append(led("C076", estado=estados[6], doc=44, med=40.44, tipo="proporcion", unidad="porcentaje", d=0, ubicacion="3.6", magnitud=mag76, ic=(38.66, 42.24)))
    return filas


def test_la_codificacion_creciente_se_corrige_con_los_valores_medidos():
    c = una("diagnosticos_secundarios", evid(filas=_dx(), supuestos=[sup("S11", "ACEPTADO")]))
    assert (c.documento, c.ubicacion, c.prioridad, c.tipo) == ("Fase 2", "3.6", "media", "pendiente de datos completos")
    assert c.texto_actual == ("El repositorio reporta que el promedio de diagnósticos secundarios en adultos hospitalizados sube de 4,08 en 2019 a 4,84, 5,37, 5,33, 5,56 y 5,88 en "
                              "2024, un 44 % más.")
    assert c.hallazgo == ("Medido: de 4,14 en 2019 a 5,82 en 2024, un 40,4 % más (intervalo de 38,7 % a 42,2 %); 4 de las siete cifras difieren de lo documentado por más de tres "
                          "errores estándar")
    assert c.texto_propuesto == "Reemplazar por los valores medidos con los datos completos; la tendencia creciente se confirma"
    assert c.evidencia_tabla == "t23_deriva_codificacion.csv" and c.evidencia_clave.endswith("codificacion.aumento_pct_2019_2024")


def test_la_codificacion_creciente_cambia_con_el_supuesto_la_magnitud_y_el_modo():
    assert una("diagnosticos_secundarios", evid(filas=_dx())).texto_propuesto == "Reemplazar por los valores medidos con los datos completos"         # sin S11 aceptado
    assert una("diagnosticos_secundarios", evid(filas=_dx(mag76="despreciable"))).prioridad == "baja"
    assert una("diagnosticos_secundarios", evid(etiqueta="completo", filas=_dx())).tipo == "corrección"


def test_la_codificacion_creciente_no_se_emite_sin_discrepancias_ni_con_cifras_incompletas():
    ninguna("diagnosticos_secundarios", evid(filas=_dx(estados=("CONFIRMA",) * 7)))
    ninguna("diagnosticos_secundarios", evid(filas=_dx(sin_medido=("C073",))))                              # una de las siete sin valor medido
    ninguna("diagnosticos_secundarios", evid(filas=_dx()[:-1]))                                              # falta C076
    ninguna("diagnosticos_secundarios", evid())


# ============================================================================================================================ hallazgo: cohorte de estadía
def _cohorte(literal=67.84, sin_cens=62.84, doc=61.31):
    metricas = {"cohorte.pct_estancia_total": metrica(literal, alcance="2019-2023"), "cohorte.pct_estancia_total.sin_censurados": metrica(sin_cens, alcance="2019-2023")}
    filas = [led("C051", doc=doc, med=literal, tipo="proporcion", unidad="porcentaje", d=2, ubicacion="3.3"), led("C110", doc=3561516, ubicacion="3.3")]
    return {"metricas": metricas, "filas": filas}


def test_la_cohorte_de_estadia_documentada_no_coincide_con_ninguna_lectura_de_la_suite():
    c = una("cohorte_estadia", evid(**_cohorte()))
    assert (c.documento, c.ubicacion, c.prioridad, c.tipo) == ("Fase 2", "3.3 y 5.2", "media", "pendiente de datos completos")
    assert c.texto_actual == "Cohorte de estadía: 3.561.516 episodios, 61,31 % del total."
    assert c.hallazgo == (
        "En 2019-2023 la definición literal del pipeline (mapeo de tres caracteres) da 67,84 % del total y la que además excluye a los censurados da 62,84 %; ninguna coincide "
        "con el 61,31 % documentado, que abarca 2019 a 2024 (el desenlace de 2024 no se lee)"
    )
    assert c.texto_propuesto == "Indicar si la cifra incluye o no a los censurados y recalcularla con los datos completos"
    assert c.evidencia_clave == "cohorte.pct_estancia_total; cohorte.pct_estancia_total.sin_censurados"


def test_la_cohorte_de_estadia_sin_el_conteo_documentado_cita_solo_el_porcentaje():
    p = _cohorte()
    p["filas"] = p["filas"][:1]
    assert una("cohorte_estadia", evid(**p)).texto_actual == "Cohorte de estadía: 61,31 % del total."


def test_la_cohorte_de_estadia_respeta_el_margen_de_la_magnitud_relevante():
    # margen 0,5 puntos: una lectura a 0,5 exactos del documentado no basta para dejar de emitir; a menos de 0,5 la cifra se considera reproducida
    assert len(emitir("cohorte_estadia", evid(**_cohorte(literal=62.0, sin_cens=70.0, doc=61.5)))) == 1
    ninguna("cohorte_estadia", evid(**_cohorte(literal=61.75, sin_cens=70.0, doc=61.5)))
    ninguna("cohorte_estadia", evid(**_cohorte(literal=70.0, sin_cens=61.25, doc=61.5)))                    # basta con que una de las dos lecturas coincida


def test_la_cohorte_de_estadia_no_se_emite_sin_metricas_ni_cifra_documentada():
    p = _cohorte()
    ninguna("cohorte_estadia", evid(filas=p["filas"]))
    ninguna("cohorte_estadia", evid(metricas={"cohorte.pct_estancia_total": metrica(67.84)}, filas=p["filas"]))
    ninguna("cohorte_estadia", evid(metricas={**p["metricas"], "cohorte.pct_estancia_total": metrica(None)}, filas=p["filas"]))
    ninguna("cohorte_estadia", evid(metricas=p["metricas"]))


# ============================================================================================================================ hallazgo: prevalencia de la Fase 1
def _variantes(periodo="2019-2023", **prev):
    return {"t23_cascada_variantes.csv": df(*({"periodo": periodo, "variante": v, "prevalencia": p} for v, p in prev.items()))}


COHORTES = _variantes(A=0.0341, B=0.0402, C=0.0472, D=0.0511)
C116_C117 = [led("C116", estado="NO VERIFICABLE", doc=2.78, med=None, documento="Fase 1", unidad="porcentaje", d=2), led("C117", estado="NO VERIFICABLE", doc=4.35, med=None,
             documento="Fase 1", unidad="porcentaje", d=2)]


def test_la_prevalencia_de_la_fase_1_se_reemplaza_por_la_de_la_variante_de_cohorte():
    c = una("prevalencia_fase1", evid(tablas=COHORTES, filas=C116_C117))
    assert (c.documento, c.ubicacion, c.prioridad, c.tipo) == ("Fase 1", "9.5", "media", "pendiente de datos completos")
    assert c.texto_actual == "El 2,78 % de mortalidad corresponde a la base completa. Después de las exclusiones el valor cambia, así que se recalcula en la Fase 2."
    assert c.hallazgo == (
        "Prevalencia de la mortalidad (fallecidos sobre todos los egresos de la variante) en 2019-2023 según la variante de cohorte: A 3,41 %, B 4,02 %, C 4,72 %, D 5,11 %. "
        "La cifra de 4,35 % de adultos hospitalizados de 2023 "
        "usa otra definición de cohorte y no es comparable (C116 y C117)"
    )
    assert c.texto_propuesto == "Reemplazar la cifra por la prevalencia de la variante de cohorte que se decida (decisiones 5 y 15)"
    assert (c.evidencia_tabla, c.evidencia_clave) == ("t23_cascada_variantes.csv; ledger_cifras.csv", "periodo; variante; prevalencia; C116; C117")


def test_la_prevalencia_de_la_fase_1_usa_el_periodo_de_resultado_y_las_variantes_que_hay():
    tablas = {"t23_cascada_variantes.csv": df({"periodo": "2019", "variante": "A", "prevalencia": 0.5}, {"periodo": "2019-2023", "variante": "A", "prevalencia": 0.0341},
                                               {"periodo": "2019-2023", "variante": "C", "prevalencia": 0.0472})}
    h = una("prevalencia_fase1", evid(tablas=tablas)).hallazgo
    assert h == ("Prevalencia de la mortalidad (fallecidos sobre todos los egresos de la variante) en 2019-2023 según la variante de cohorte: A 3,41 %, C 4,72 %")    # sin C116 ni C117 no hay frase final
    assert una("prevalencia_fase1", evid(tablas=tablas)).texto_actual.startswith("La mortalidad corresponde a la base completa.")


def test_la_prevalencia_de_la_fase_1_no_se_emite_sin_la_variante_a_ni_tabla():
    ninguna("prevalencia_fase1", evid(tablas=_variantes(B=0.04, C=0.05)))
    ninguna("prevalencia_fase1", evid(tablas=_variantes(periodo="2019", A=0.03)))
    ninguna("prevalencia_fase1", evid(tablas=_variantes(A=None)))
    ninguna("prevalencia_fase1", evid())


# ============================================================================================================================ hallazgo: tipos de alta
def _alta(n=11, prefijo=2, pct=0.72):
    return {"t24_resumen_catalogos.csv": df({"n_etiquetas_tipoalta": n, "n_etiquetas_tipoalta_por_prefijo": prefijo, "pct_perdido_union_exacta": pct})}


def test_los_tipos_de_alta_reconocidos_solo_por_prefijo_se_listan_con_su_rol():
    c = una("tipos_de_alta", evid(tablas=_alta()))
    assert (c.documento, c.ubicacion, cierre.prioridad_final(c), c.tipo) == ("Fase 1", "9.2", "baja", "aclaración")                   # solo cambia una lista: se aclara
    assert una("tipos_de_alta", evid(etiqueta="completo", tablas=_alta())).tipo == "aclaración"
    assert c.hallazgo == ("2 de las 11 etiquetas de TIPOALTA solo se reconocen por prefijo porque el archivo las trae cortadas; una unión exacta con el archivo de roles dejaría "
                          "sin rol a 0,72 % de los egresos")
    assert c.texto_propuesto == "Listar los valores de TIPOALTA con su rol, indicar cuáles vienen cortados y fijar la regla de prefijo para asignarlos"
    assert (c.evidencia_tabla, c.evidencia_clave) == ("t24_resumen_catalogos.csv; t24_tipoalta_etiquetas.csv", "n_etiquetas_tipoalta_por_prefijo; pct_perdido_union_exacta")
    assert "3 de las 12 etiquetas" in una("tipos_de_alta", evid(tablas=_alta(12, 3))).hallazgo


def test_los_tipos_de_alta_no_se_emiten_si_ninguna_etiqueta_llega_cortada():
    ninguna("tipos_de_alta", evid(tablas=_alta(prefijo=0)))
    ninguna("tipos_de_alta", evid(tablas=_alta(pct=None)))
    ninguna("tipos_de_alta", evid())


# ============================================================================================================================ hallazgo: contradicciones entre cifras del mismo documento
def _contradiccion(id_, doc, *, ubicacion="2.2", descripcion="cifra", clave="faltantes.n_nulos.X", alcance="2019-2024", documento="Fase 2"):
    return led(id_, doc=doc, ubicacion=ubicacion, descripcion=descripcion, clave=clave, alcance_doc=alcance, documento=documento, estado="CONFIRMA", med=doc)


def test_dos_cifras_del_mismo_documento_con_la_misma_medicion_y_valores_distintos_se_unifican():
    filas = [_contradiccion("C230", 75, descripcion="IR_29301_COD_GRD sin dato", clave="faltantes.n_nulos.IR_29301_COD_GRD"),
             _contradiccion("C231", 90, ubicacion="4.1", descripcion="GRD sin código (seis años)", clave="faltantes.n_nulos.IR_29301_COD_GRD")]
    c = una("contradicciones_documentales", evid(filas=filas))
    assert (c.documento, c.ubicacion, c.prioridad, c.tipo) == ("Fase 2", "2.2 y 4.1", "media", "corrección")
    assert c.texto_actual == "IR_29301_COD_GRD sin dato: 75 (sección 2.2); GRD sin código (seis años): 90 (sección 4.1)."
    assert c.hallazgo == "Las dos cifras se miden con la misma columna (faltantes.n_nulos.IR_29301_COD_GRD) y el mismo alcance (2019-2024) pero se documentan distintas"
    assert c.texto_propuesto == "Unificar la definición y la cifra, o aclarar por qué difieren"
    assert (c.evidencia_tabla, c.evidencia_clave) == ("ledger_cifras.csv", "C230; C231")


def test_el_par_de_cifras_que_el_propio_documento_reconcilia_se_aclara_y_no_se_llama_contradiccion():
    # Fase 2, sección 4.1: 75 se mide en cinco años (2019 a 2023) y 90 en seis, con definiciones distintas; el cuadro de la sección 2.2 no declara el alcance
    filas = [_contradiccion("C030", 75, descripcion="IR_29301_COD_GRD sin dato", clave="faltantes.n_nulos.IR_29301_COD_GRD"),
             _contradiccion("C100", 90, ubicacion="4.1", descripcion="GRD sin código (seis años)", clave="faltantes.n_nulos.IR_29301_COD_GRD")]
    c = una("contradicciones_documentales", evid(filas=filas))
    assert c.hallazgo == ("Las dos cifras se miden con la misma columna (faltantes.n_nulos.IR_29301_COD_GRD) y el cuadro de datos las declara con el mismo alcance (2019-2024), "
                          "pero la sección 4.1 de la Fase 2 ya las reconcilia: la primera se mide en cinco años y la segunda en seis, con definiciones distintas")
    assert c.texto_propuesto == "Declarar en cada cuadro el alcance y la definición de la cifra"
    assert (c.tipo, c.evidencia_clave) == ("aclaración", "C030; C100") and cierre.prioridad_final(c) == "baja"
    assert "contradic" not in c.hallazgo.lower() and "pero se documentan distintas" not in c.hallazgo
    # el orden de las cifras no importa y un par con las mismas claves pero otros ids sigue siendo una contradicción
    assert frozenset({"C100", "C030"}) in cierre.CONCILIADAS_EN_EL_DOCUMENTO
    otras = [_contradiccion("C030", 75, clave="k.x"), _contradiccion("C231", 90, ubicacion="4.1", clave="k.x")]
    assert una("contradicciones_documentales", evid(filas=otras)).tipo == "corrección"


def test_las_contradicciones_con_la_misma_seccion_la_citan_una_vez_y_cada_par_es_un_cambio():
    filas = [_contradiccion("C200", 1), _contradiccion("C201", 2), _contradiccion("C202", 3)]
    cs = emitir("contradicciones_documentales", evid(filas=filas))
    assert len(cs) == 3 and all(valido(c).ubicacion == "2.2" for c in cs)                                   # tres valores distintos: tres pares
    assert [c.evidencia_clave for c in cs] == ["C200; C201", "C200; C202", "C201; C202"]


def test_no_hay_contradiccion_si_cambia_el_valor_el_alcance_el_documento_o_la_clave():
    ninguna("contradicciones_documentales", evid(filas=[_contradiccion("C300", 75), _contradiccion("C301", 75)]))                                         # mismo valor
    ninguna("contradicciones_documentales", evid(filas=[_contradiccion("C300", 75), _contradiccion("C301", 90, alcance="2019-2023")]))                    # otro alcance
    ninguna("contradicciones_documentales", evid(filas=[_contradiccion("C300", 75), _contradiccion("C301", 90, documento="Fase 1")]))                     # otro documento
    ninguna("contradicciones_documentales", evid(filas=[_contradiccion("C300", 75), _contradiccion("C301", 90, clave="faltantes.n_nulos.Y")]))            # otra clave
    ninguna("contradicciones_documentales", evid())


# ============================================================================================================================ hallazgo: diabetes en cero
def _diabetes(prev11=0.148, prev12=0.042):
    return {"t23_resumen_comorbilidad.csv": df({"prev_completo_ELIX_11": prev11, "prev_completo_ELIX_12": prev12})}


def _c068(estado="CONFIRMA"):
    return [led("C068", estado=estado, doc=0.0, med=0.0, tipo="proporcion", unidad="porcentaje", d=1, ubicacion="3.5")]


def test_la_diabetes_en_cero_se_explica_por_el_mapeo_de_tres_caracteres():
    c = una("diabetes_cero", evid(tablas=_diabetes(), filas=_c068()))
    assert (c.documento, c.ubicacion, c.prioridad, c.tipo) == ("Fase 2", "3.5", "media", "corrección")
    assert c.texto_actual == "Pero muestra diabetes complicada y no complicada con 0,0 %, algo muy improbable en pacientes hospitalizados."
    assert c.hallazgo == ("El mapeo de tres caracteres reproduce la cifra de la figura (0,0 %), mientras el mapeo completo da 14,8 % de diabetes no complicada y 4,2 % de diabetes "
                          "complicada en adultos hospitalizados")
    assert c.texto_propuesto == "Confirmar que la causa es el mapeo de tres caracteres y corregirlo con el mapeo completo antes de usar ELIX_11 y ELIX_12"
    assert (c.evidencia_tabla, c.evidencia_clave) == ("t23_resumen_comorbilidad.csv; ledger_cifras.csv", "prev_completo_ELIX_11; prev_completo_ELIX_12; C068")
    assert una("diabetes_cero", evid(tablas=_diabetes(), filas=_c068(), supuestos=[sup("S02")])).prioridad == "alta"


def test_la_diabetes_en_cero_no_se_emite_si_la_cifra_no_se_reproduce_o_el_mapeo_completo_tambien_da_cero():
    ninguna("diabetes_cero", evid(tablas=_diabetes(), filas=_c068("DISCREPA")))
    ninguna("diabetes_cero", evid(tablas=_diabetes(prev11=0.0), filas=_c068()))
    ninguna("diabetes_cero", evid(tablas=_diabetes(prev11=None), filas=_c068()))
    ninguna("diabetes_cero", evid(tablas=_diabetes()))
    ninguna("diabetes_cero", evid(filas=_c068()))


# ============================================================================================================================ construir_cambios: estructura
def _todo():
    """Evidencia que dispara hallazgos de las dos fases con distintas prioridades, más una discrepancia del ledger y un supuesto rechazado que no cubre ningún hallazgo."""
    tablas = {**_seleccion(), **_hospproc(), **_cola({}), **TRES_C, **COHORTES, **_alta(), **_diabetes(), **_fuga(SEVERIDAD, MORTALIDAD, concordancia=CONCORDANCIA)}
    filas = [*_estimados(), *_c068(), led("C050", doc=10.5, med=12.25, tipo="proporcion", unidad="porcentaje", d=2, clave="k.p", tabla="t_prop.csv")]
    return evid(tablas=tablas, filas=filas, supuestos=[sup("S05"), sup("S02", "ACEPTADO")], semaforos={})


def test_sin_hallazgos_ni_discrepancias_las_dos_tablas_salen_vacias_con_encabezado():
    t = cierre.construir_cambios(evid())
    assert list(t) == ["Fase 1", "Fase 2"]
    for doc in cierre.DOCUMENTOS:
        assert t[doc].height == 0 and t[doc].columns == list(cierre.COLUMNAS_CAMBIOS)
        assert set(t[doc].dtypes) == {pl.String}


def test_los_ids_son_consecutivos_por_documento_y_el_orden_es_por_prioridad_y_luego_de_emision():
    ev = _todo()
    t = cierre.construir_cambios(ev)
    assert t["Fase 1"].height >= 5 and t["Fase 2"].height >= 4
    todas = set(t["Fase 1"]["prioridad"].to_list()) | set(t["Fase 2"]["prioridad"].to_list())
    assert {"alta", "media"} <= todas                                                           # hay más de una prioridad: el orden se ejercita
    for doc, prefijo in (("Fase 1", "CF1"), ("Fase 2", "CF2")):
        tabla = t[doc]
        assert tabla["id"].to_list() == [f"{prefijo}-{i:03d}" for i in range(1, tabla.height + 1)] and set(tabla["documento"].to_list()) == {doc}
        orden = [cierre.PRIORIDADES.index(p) for p in tabla["prioridad"].to_list()]
        assert orden == sorted(orden)
        # dentro de una misma prioridad, los hallazgos del catálogo salen en el orden del catálogo
        catalogo = [c for h in cierre.HALLAZGOS for c in h.funcion(ev) if c.documento == doc]
        for prioridad in cierre.PRIORIDADES:
            esperados = [c.hallazgo for c in catalogo if cierre.prioridad_final(c) == prioridad]
            obtenidos = [h for h, p in zip(tabla["hallazgo"], tabla["prioridad"]) if p == prioridad and h in set(esperados)]
            assert obtenidos == esperados, (doc, prioridad)


def test_cada_cambio_va_a_la_tabla_de_su_documento_y_ninguno_se_repite():
    t = cierre.construir_cambios(_todo())
    vistos = []
    for doc in cierre.DOCUMENTOS:
        assert set(t[doc]["documento"].to_list()) <= {doc}
        vistos += [(r["ubicacion"], r["hallazgo"]) for r in t[doc].iter_rows(named=True)]
    assert len(vistos) == len(set(vistos))
    assert any("umbral de fuga" in h for _, h in vistos) and any("coincide con SERVICIOALTA" in h for _, h in vistos)


def test_los_prefijos_de_id_distinguen_el_documento():
    assert cierre.PREFIJO_ID == {"Fase 1": "CF1", "Fase 2": "CF2"}
    assert cierre.ARCHIVO_CAMBIOS == {"Fase 1": "cambios_fase1.csv", "Fase 2": "cambios_fase2.csv"}


# ============================================================================================================================ fuente (a): cifras del ledger que discrepan
def _discrepa(id_="C050", **kw):
    base = dict(doc=10.5, med=12.25, tipo="proporcion", unidad="porcentaje", d=2, clave="k.p", tabla="t_prop.csv", descripcion="Porcentaje de prueba")
    return led(id_, **{**base, **kw})


def test_una_discrepancia_relevante_sin_hallazgo_se_emite_desde_el_ledger_con_su_medicion():
    t = cierre.construir_cambios(evid(filas=[_discrepa()]))
    assert t["Fase 1"].height == 0 and t["Fase 2"].height == 1
    c = t["Fase 2"].row(0, named=True)
    assert (c["id"], c["ubicacion"], c["prioridad"], c["tipo"]) == ("CF2-001", "2.1", "media", "pendiente de datos completos")
    assert c["texto_actual"] == "Porcentaje de prueba: 10,50 %"
    assert c["hallazgo"] == "La suite midió 12,25 % (alcance 2019-2023) en t_prop.csv; con un extracto el resultado es indicativo"
    assert c["texto_propuesto"] == "Reemplazar 10,50 % por 12,25 % tras repetir la medición con los datos completos"
    assert (c["evidencia_tabla"], c["evidencia_clave"]) == ("t_prop.csv", "k.p")


def test_en_modo_completo_la_discrepancia_del_ledger_es_una_correccion_sin_salvedad_de_extracto():
    c = cierre.construir_cambios(evid(etiqueta="completo", filas=[_discrepa()]))["Fase 2"].row(0, named=True)
    assert c["tipo"] == "corrección" and "indicativo" not in c["hallazgo"] and c["texto_propuesto"] == "Reemplazar 10,50 % por 12,25 %"


def test_las_discrepancias_despreciables_solo_se_emiten_en_modo_completo_y_con_prioridad_baja():
    fila = _discrepa(magnitud="despreciable")
    assert cierre.construir_cambios(evid(filas=[fila]))["Fase 2"].height == 0
    c = cierre.construir_cambios(evid(etiqueta="completo", filas=[fila]))["Fase 2"].row(0, named=True)
    assert c["prioridad"] == "baja"


def test_una_cifra_de_la_fase_1_que_discrepa_va_a_la_tabla_de_la_fase_1_con_prioridad_alta():
    t = cierre.construir_cambios(evid(filas=[_discrepa("C060", documento="Fase 1", ubicacion="7")]))
    assert t["Fase 2"].height == 0 and t["Fase 1"].height == 1
    c = t["Fase 1"].row(0, named=True)
    assert (c["id"], c["documento"], c["ubicacion"], c["prioridad"]) == ("CF1-001", "Fase 1", "7", "alta")


def test_una_cota_de_valores_distintos_se_corrige_con_al_menos_el_valor_medido():
    fila = led("C061", doc=65, med=68, unidad="conteo", d=0, clave="hospitales.n_distintos.2023", alcance_med="2023", tabla="t24_hospitales_catalogo.csv",
               descripcion="Hospitales del repositorio en 2023")
    c = cierre.construir_cambios(evid(filas=[fila]))["Fase 2"].row(0, named=True)
    assert c["texto_actual"] == "Hospitales del repositorio en 2023: 65"
    assert c["hallazgo"] == "La suite midió 68 (alcance 2023) en t24_hospitales_catalogo.csv; el extracto es una cota inferior de la población"
    assert c["texto_propuesto"] == "Reemplazar 65 por al menos 68" and c["tipo"] == "corrección"                       # una cota ya contradice el documento: no espera los datos completos
    completo = cierre.construir_cambios(evid(etiqueta="completo", filas=[fila]))["Fase 2"].row(0, named=True)
    assert completo["texto_propuesto"] == "Reemplazar 65 por 68" and "cota" not in completo["hallazgo"]


def test_una_cifra_de_estructura_se_corrige_y_el_anio_no_lleva_separador_de_miles():
    fila = led("C018", doc=2019, med=2024, tipo="estructura", unidad="anio", d=0, clave="estructura.primer_anio_id_beneficiario", alcance_med="2019-2024",
               tabla="t21_esquema_diferencias.csv", descripcion="El identificador se llamaba ID_BENEFICIARIO en este año")
    c = cierre.construir_cambios(evid(filas=[fila]))["Fase 2"].row(0, named=True)
    assert c["texto_actual"] == "El identificador se llamaba ID_BENEFICIARIO en este año: 2019" and c["texto_propuesto"] == "Corregir 2019 por 2024"
    assert "2024" in c["hallazgo"] and "2.024" not in c["hallazgo"] and c["tipo"] == "corrección"


def test_el_ledger_no_repite_lo_que_ya_cubre_un_hallazgo_emitido():
    filas = [led("C128", doc=10, med=4, tipo="aritmetica", unidad="razon", d=2), led("C129", doc=20, med=66.67, tipo="aritmetica", unidad="razon", d=2)]
    con_pares = cierre.construir_cambios(evid(filas=filas, cifras=_pares()))["Fase 2"]
    assert con_pares.height == 1 and con_pares["evidencia_clave"][0] == "C128; C129"                  # solo el hallazgo de las razones
    sin_pares = cierre.construir_cambios(evid(filas=filas, cifras=cifras_df()))["Fase 2"]                     # sin pares el hallazgo no se emite: las dos cifras salen sueltas
    assert sin_pares.height == 2 and sorted(sin_pares["evidencia_clave"].to_list()) == ["k.c128", "k.c129"]


def test_el_ledger_omite_la_cifra_que_discrepa_sin_valor_medido_o_documentado_y_las_que_no_discrepan():
    filas = [_discrepa("C070", med=None), _discrepa("C071", doc=None), _discrepa("C072", estado="CONFIRMA"), _discrepa("C073", estado="NO VERIFICABLE", med=None)]
    assert cierre.construir_cambios(evid(filas=filas))["Fase 2"].height == 0


# ============================================================================================================================ fuente (b): supuestos rechazados
def test_un_supuesto_rechazado_sin_hallazgo_se_emite_con_su_cifra_y_su_motivo():
    t = cierre.construir_cambios(evid(supuestos=[sup("S05", cifra="cifra cinco", motivo="motivo cinco", tabla="t22_cardinalidades.csv")]))
    assert t["Fase 1"].height == 0 and t["Fase 2"].height == 1
    c = t["Fase 2"].row(0, named=True)
    assert (c["ubicacion"], c["prioridad"], c["tipo"]) == ("Supuesto S05", "media", "pendiente de datos completos")        # S05 no afecta ninguna decisión
    assert c["texto_actual"] == "Texto del supuesto S05" and c["hallazgo"] == "cifra cinco; motivo cinco"
    assert c["texto_propuesto"] == "Reemplazar la afirmación del documento por lo medido (cifra cinco) e indicar la regla aplicada; criterio del supuesto: Criterio del supuesto S05"
    assert "Aplicar el criterio" not in c["texto_propuesto"]                                             # un texto de reemplazo con lo medido y no solo el criterio copiado
    assert (c["evidencia_tabla"], c["evidencia_clave"]) == ("t22_cardinalidades.csv", "S05")


def test_el_supuesto_rechazado_que_afecta_una_decision_tiene_prioridad_alta():
    c = cierre.construir_cambios(evid(supuestos=[sup("S01")]))["Fase 2"].row(0, named=True)               # S01 afecta las decisiones 1 y 13
    assert c["prioridad"] == "alta" and c["ubicacion"] == "Supuesto S01"


def test_el_tipo_del_supuesto_rechazado_sigue_su_base_estructural_o_estadistica():
    estructural = cierre.construir_cambios(evid(supuestos=[sup("S06")]))["Fase 2"].row(0, named=True)
    assert estructural["tipo"] == "corrección"                                                          # S06 es estructural: siempre corrección
    completo = cierre.construir_cambios(evid(etiqueta="completo", supuestos=[sup("S05")]))["Fase 2"].row(0, named=True)
    assert completo["tipo"] == "corrección"                                                             # estadística en modo completo


def test_solo_los_supuestos_rechazados_generan_cambios():
    ev = evid(supuestos=[sup("S05", "ACEPTADO"), sup("S06", "NO EVALUABLE"), sup("S12", "ACEPTADO")])
    assert cierre.construir_cambios(ev)["Fase 2"].height == 0


def test_un_supuesto_sin_cifra_ni_motivo_dice_que_el_criterio_no_se_cumple():
    c = cierre.construir_cambios(evid(supuestos=[sup("S05", cifra="", motivo="")]))["Fase 2"].row(0, named=True)
    assert c["hallazgo"] == "El criterio del supuesto no se cumple"
    assert c["texto_propuesto"] == "Aplicar el criterio del supuesto: Criterio del supuesto S05"             # sin cifra medida no hay nada que citar en el reemplazo


def test_un_supuesto_cubierto_por_un_hallazgo_emitido_no_se_repite_pero_sin_el_hallazgo_si():
    # S02 lo cubre el hallazgo del mapeo de tres caracteres, S09 el de SERVICIOINGRESO y S11 el de los diagnósticos secundarios
    for tablas, filas, sid in ((TRES_C, [], "S02"), (_fuga(concordancia=CONCORDANCIA), [], "S09"), ({}, _dx(), "S11")):
        con = pl.concat(cierre.construir_cambios(evid(tablas=tablas, filas=filas, supuestos=[sup(sid)])).values())
        assert con.height >= 1 and f"Supuesto {sid}" not in con["ubicacion"].to_list(), sid
        sin = cierre.construir_cambios(evid(supuestos=[sup(sid)]))["Fase 2"]                              # sin la tabla del hallazgo, el supuesto sale por su cuenta
        assert sin["ubicacion"].to_list() == [f"Supuesto {sid}"], sid


def test_un_supuesto_con_cifra_en_el_ledger_no_se_repite_si_esa_cifra_ya_salio():
    s07 = sup("S07")
    emitida = cierre.construir_cambios(evid(filas=[led("C018", doc=2019, med=2024, tipo="estructura", unidad="anio")], supuestos=[s07]))["Fase 2"]
    assert emitida["ubicacion"].to_list() == ["2.1"]                                                    # solo la cifra del ledger
    no_emitida = cierre.construir_cambios(evid(filas=[led("C018", estado="CONFIRMA")], supuestos=[s07]))["Fase 2"]
    assert no_emitida["ubicacion"].to_list() == ["Supuesto S07"]
    # S08 se contrasta con C077 y C078: basta que una de las dos salga
    s08 = sup("S08")
    una_sola = cierre.construir_cambios(evid(filas=[led("C077", estado="NO VERIFICABLE", med=None), led("C078", doc=68, med=72)], supuestos=[s08]))["Fase 2"]
    assert una_sola["ubicacion"].to_list() == ["9.1"] and una_sola["evidencia_clave"].to_list() == ["k.c078"]            # C078 se cita en la sección 9.1 de la Fase 1, no en la del CSV
    ninguna_sale = cierre.construir_cambios(evid(filas=[led("C077", estado="CONFIRMA"), led("C078", estado="CONFIRMA")], supuestos=[s08]))["Fase 2"]
    assert ninguna_sale["ubicacion"].to_list() == ["Supuesto S08"]


def test_el_cambio_de_c078_cita_la_seccion_9_1_de_la_fase_1_y_las_demas_cifras_conservan_la_ubicacion_del_csv():
    # cifras_documentadas.csv dice 8 para C078 (solo se corrige su columna clave_eda); los 68 hospitales de la Fase 1 están en la sección 9.1
    filas = [led("C078", documento="Fase 1", ubicacion="8", doc=68, med=72), led("C040", documento="Fase 2", ubicacion="3.1", doc=1.0, med=2.0)]
    t = cierre.construir_cambios(evid(filas=filas))
    assert t["Fase 1"]["ubicacion"].to_list() == ["9.1"] and t["Fase 2"]["ubicacion"].to_list() == ["3.1"]
    assert cierre.UBICACION_CORREGIDA == {"C078": "9.1"} and CIFRAS.filter(pl.col("id") == "C078")["ubicacion"][0] == "8"


# ============================================================================================================================ cambios al protocolo y resumen
def test_los_cambios_al_protocolo_salen_del_archivo_de_umbrales_con_su_id_como_texto(ctx):
    t = cierre.tabla_cambios_protocolo(ctx)
    meta = ctx.umbrales.meta["cambios_protocolo"]
    assert t.columns == ["id", "descripcion"] and t.dtypes == [pl.String, pl.String]
    assert t.height == len(meta) >= 4 and t["id"].to_list() == [m.split(":")[0] for m in meta]
    assert t["id"].to_list()[:4] == ["1.1", "1.2", "1.3", "1.4"]
    assert t["descripcion"][0].startswith("se agrega conciliacion_dif_relevante_pp") and "H17b" in " ".join(t["descripcion"].to_list())
    assert all(not d.startswith(" ") for d in t["descripcion"].to_list())


def test_el_ultimo_cambio_al_protocolo_no_se_pierde_ni_se_duplica(ctx):
    meta = ctx.umbrales.meta["cambios_protocolo"]
    t = cierre.tabla_cambios_protocolo(ctx)
    assert t["id"][-1] == meta[-1].split(":")[0] and t["descripcion"][-1] == meta[-1].partition(":")[2].strip() and t["id"].n_unique() == len(meta)


def test_el_cambio_1_5_declara_el_criterio_ampliado_de_s09_y_la_version_del_protocolo_pasa_a_1_5(ctx):
    meta = ctx.umbrales.meta
    assert meta["version"] == "1.5" and "cinco cambios declarados" in meta["estado"] and len(meta["cambios_protocolo"]) == 5
    t = cierre.tabla_cambios_protocolo(ctx)
    assert t["id"].to_list() == ["1.1", "1.2", "1.3", "1.4", "1.5"]
    quinto = t.row(4, named=True)["descripcion"]
    assert quinto.startswith("el criterio de S09 se amplía con el contraste de los egresos con traslado interno")
    assert "esperable por construcción y no discrimina" in quinto and "Se advirtió después de ver" in quinto
    assert "99,2 % de coincidencia sin traslado y 15,2 % con traslado" in quinto and "no levanta las sospechas de proxy ni de fuga" in quinto
    assert "No cambia ningún valor de umbral" in quinto and "celda_minima" in quinto


def test_un_cambio_al_protocolo_sin_dos_puntos_usa_su_posicion_como_id_y_sin_cambios_la_tabla_sale_vacia():
    falso = SimpleNamespace(umbrales=SimpleNamespace(meta={"cambios_protocolo": ["2.1: se agrega un umbral", "un cambio sin identificador"]}))
    t = cierre.tabla_cambios_protocolo(falso)
    assert t.to_dicts() == [{"id": "2.1", "descripcion": "se agrega un umbral"}, {"id": "2", "descripcion": "un cambio sin identificador"}]
    vacio = cierre.tabla_cambios_protocolo(SimpleNamespace(umbrales=SimpleNamespace(meta={})))
    assert vacio.height == 0 and vacio.columns == ["id", "descripcion"]


def _cambios_df(*filas):
    base = dict.fromkeys(cierre.COLUMNAS_CAMBIOS, "x")
    return pl.DataFrame([{**base, **f} for f in filas], schema=cierre.ESQUEMA_CAMBIOS)


def test_el_resumen_de_cambios_cuenta_por_documento_prioridad_y_tipo():
    f1 = _cambios_df({"prioridad": "alta", "tipo": "corrección"}, {"prioridad": "media", "tipo": "pendiente de datos completos"})
    f2 = _cambios_df({"prioridad": "baja", "tipo": "aclaración"})
    protocolo = pl.DataFrame({"id": ["1.1", "1.2"], "descripcion": ["a", "b"]})
    assert cierre.resumen_cambios(f1, f2, protocolo, "1.4") == {
        "cambios_n_total": 3, "cambios_n_fase1": 2, "cambios_n_fase2": 1, "cambios_n_alta": 1, "cambios_n_media": 1, "cambios_n_baja": 1, "cambios_n_correccion": 1,
        "cambios_n_aclaracion": 1, "cambios_n_pendiente": 1, "cambios_n_protocolo": 2, "cambios_protocolo_version": "v1.4",
    }
    vacio = cierre.resumen_cambios(_cambios_df(), _cambios_df(), protocolo.head(0), "2.0")
    assert vacio["cambios_n_total"] == 0 and vacio["cambios_n_protocolo"] == 0 and vacio["cambios_protocolo_version"] == "v2.0"


# ============================================================================================================================ corrida de punta a punta sobre la fuente sintética
TABLAS_T12 = {
    "ledger_cifras.csv": 129, "ledger_motivos.csv": None, "t25_hipotesis_evaluadas.csv": 26, "decisiones_evidencia.csv": 17, "supuestos_verificados.csv": 15,
    "cambios_fase1.csv": None, "cambios_fase2.csv": None, "cambios_protocolo.csv": None, "t25_resumen_evidencia.csv": 1,
}


@pytest.fixture(scope="module")
def cierre_corrido(pipeline_sintetico, tmp_path_factory):
    ctx_c = copiar_tablas(pipeline_sintetico.ctx, tmp_path_factory.mktemp("t12_cierre"), con_figuras=True)
    r = cierre.ejecutar(ctx_c, cascada=pipeline_sintetico.r7["cascada"])
    ctx_c.escribir_registros()
    return ctx_c, r


@pytest.fixture(scope="module")
def cierre_completo(pipeline_sintetico, tmp_path_factory):
    ctx_c = copiar_tablas(pipeline_sintetico.ctx, tmp_path_factory.mktemp("t12_cierre_completo"), con_figuras=True, etiqueta="completo")
    r = cierre.ejecutar(ctx_c, cascada=pipeline_sintetico.r7["cascada"])
    ctx_c.escribir_registros()
    return ctx_c, r


def test_ejecutar_escribe_las_tablas_de_la_tarea_12_con_su_tarea_y_sin_columnas_de_conteo_nuevas(cierre_corrido):
    ctx_c, r = cierre_corrido
    assert set(r) == {"ledger", "hipotesis", "decisiones", "supuestos", "cambios_fase1", "cambios_fase2", "cambios_protocolo"}
    idx = ctx_c.indice_tablas()
    for nombre, filas in TABLAS_T12.items():
        t = idx.filter(pl.col("nombre") == nombre)
        assert t.height == 1 and t["tarea"][0] == "T12" and (ctx_c.dir_tablas / nombre).is_file(), nombre
        if filas is not None:
            assert t["filas"][0] == filas, nombre
        if nombre != "ledger_cifras.csv":
            assert t["columnas_conteo"][0] == "", nombre


def test_las_tablas_de_cambios_escritas_son_las_que_construye_la_evidencia_publicada(cierre_corrido):
    ctx_c, r = cierre_corrido
    esperado = cierre.construir_cambios(cierre.armar_evidencia(ctx_c))
    for doc, archivo in cierre.ARCHIVO_CAMBIOS.items():
        leida = ctx_c.leer_tabla(archivo, columnas_texto=cierre.COLUMNAS_TEXTO_CAMBIOS)
        assert leida.equals(esperado[doc]), archivo
        assert r["cambios_fase1" if doc == "Fase 1" else "cambios_fase2"].equals(esperado[doc])
        assert leida.columns == list(cierre.COLUMNAS_CAMBIOS) and set(leida["documento"].to_list()) <= {doc}
    protocolo = ctx_c.leer_tabla("cambios_protocolo.csv", columnas_texto=["id", "descripcion"])
    assert protocolo["id"].to_list() == ["1.1", "1.2", "1.3", "1.4", "1.5"]                                # los ids se leen como texto (1.1 no es un número)
    assert protocolo["descripcion"][4].startswith("el criterio de S09 se amplía")                          # cambios_protocolo.csv lista el cambio 1.5


def test_el_resumen_de_evidencia_junta_las_columnas_del_ledger_las_decisiones_y_los_cambios(cierre_corrido):
    ctx_c, r = cierre_corrido
    t = ctx_c.leer_tabla("t25_resumen_evidencia.csv", columnas_texto=["cambios_protocolo_version"])
    assert t.height == 1
    fila = t.row(0, named=True)
    f1, f2 = r["cambios_fase1"], r["cambios_fase2"]
    assert fila["ledger_n_cifras"] == 129 and fila["decisiones_n_total"] == 17 and fila["decisiones_n_supuestos"] == 15
    assert (fila["cambios_n_fase1"], fila["cambios_n_fase2"], fila["cambios_n_total"]) == (f1.height, f2.height, f1.height + f2.height)
    assert fila["cambios_n_alta"] + fila["cambios_n_media"] + fila["cambios_n_baja"] == fila["cambios_n_total"]
    assert fila["cambios_n_correccion"] + fila["cambios_n_aclaracion"] + fila["cambios_n_pendiente"] == fila["cambios_n_total"]
    assert fila["cambios_n_protocolo"] == len(ctx_c.umbrales.meta["cambios_protocolo"]) == r["cambios_protocolo"].height
    assert fila["cambios_protocolo_version"] == f"v{ctx_c.umbrales.meta['version']}" == "v1.5" and fila["cambios_n_protocolo"] == 5


def test_las_figuras_f12_01_y_f12_02_quedan_en_el_indice_con_su_csv(cierre_corrido):
    ctx_c, _ = cierre_corrido
    idx = ctx_c.figuras.indice()
    for fid, csv in (("F12_01", "decisiones_evidencia.csv"), ("F12_02", "ledger_cifras.csv")):
        f = idx.filter(pl.col("id_figura") == fid)
        assert f.height == 1 and f["csv"][0] == csv and (ctx_c.dir_figuras / f"{fid}.png").is_file() and (ctx_c.dir_tablas / csv).is_file(), fid


def test_las_metricas_de_la_tarea_12_son_solo_las_del_ledger_y_las_de_decisiones(cierre_corrido):
    ctx_c, _ = cierre_corrido
    claves = ctx_c.metricas().filter(pl.col("tarea") == "T12")["clave"].to_list()
    assert len(claves) == 12 and sorted(c for c in claves if c.startswith("ledger.")) == sorted(f"ledger.{k}" for k in ("n_cifras", "n_confirma", "n_discrepa", "n_no_verificable",
                                                                                                                       "n_discrepa_relevante"))
    assert sum(c.startswith("decisiones.") for c in claves) == 7


CONTRATO_TABLAS = {
    "t23_resumen_seleccion.csv": ["elix_top5", "elix_n_coinciden_fase1", "elix_n_coinciden_repo", "auc_modelo_l1_2023", "auc_ref_edad_sexo_2023", "auc_ref_elix31_2023"],
    "t23_resumen_hospitales.csv": ["autorref_prop_exacta", "autorref_supera_maximo"],
    "t23_cola_estancia.csv": ["base", "regla", "corte_dias", "cumple_def", "cumple_conf", "propuesta", "prop_afectados_def", "prop_afectados_conf"],
    "t23_elix_vs_3c.csv": ["elix_id", "supera_umbral", "prev_completo_ah", "prev_3c_ah"],
    "t23_vif.csv": ["conjunto", "variable", "es_agregado", "colinealidad_exacta"],
    "t23_importancia.csv": ["tipo_fila", "variable", "clase_proxy", "en_modelo_l1", "importa"],
    "t23_fuga.csv": ["tipo_fila", "variable", "supera_umbral_fuga", "prohibida_siempre", "auc_ranking", "auc_confirmacion", "prop_igualdad", "ic_bajo", "ic_alto", "coincide_mayoria"],
    "t23_cascada_variantes.csv": ["periodo", "variante", "prevalencia"],
    "t24_resumen_catalogos.csv": ["n_etiquetas_tipoalta", "n_etiquetas_tipoalta_por_prefijo", "pct_perdido_union_exacta"],
    "t23_resumen_comorbilidad.csv": ["prev_completo_ELIX_11", "prev_completo_ELIX_12"],
}
CONTRATO_METRICAS = ("elix.n_categorias", "cohorte.pct_estancia_total", "cohorte.pct_estancia_total.sin_censurados", "volumen.filas_leidas.total")


def test_las_tablas_y_metricas_que_leen_los_hallazgos_existen_con_esas_columnas_en_la_corrida(cierre_corrido):
    """Contrato entre las Tareas 7 a 11 (productoras) y los hallazgos: si una tabla cambia de columnas, el hallazgo dejaría de emitirse en silencio."""
    ctx_c, _ = cierre_corrido
    for nombre, columnas in CONTRATO_TABLAS.items():
        ruta = ctx_c.dir_tablas / nombre
        assert ruta.is_file(), nombre
        cabecera = ruta.read_text(encoding="utf8").splitlines()[0].split(",")
        faltan = [c for c in columnas if c not in cabecera]
        assert not faltan, (nombre, faltan)
    metricas = set(ctx_c.metricas()["clave"].to_list())
    assert [m for m in CONTRATO_METRICAS if m not in metricas] == []


def test_ningun_texto_de_los_cambios_trae_llaves_sin_resolver_ni_valores_nulos(cierre_corrido):
    ctx_c, r = cierre_corrido
    textos = []
    for t in (r["cambios_fase1"], r["cambios_fase2"], cierre.construir_cambios(_todo())["Fase 1"], cierre.construir_cambios(_todo())["Fase 2"]):
        textos += [str(v) for c in ("ubicacion", "texto_actual", "hallazgo", "texto_propuesto", "evidencia_tabla", "evidencia_clave") for v in t[c].to_list()]
    assert textos and all(t.strip() for t in textos)
    for t in textos:
        assert "{" not in t and "}" not in t and "None" not in t and "nan" not in t.lower().split() and "null" not in t.lower().split(), t


def test_la_privacidad_queda_en_cero_y_la_tarea_12_no_deja_filas_en_el_registro_de_accesos(cierre_corrido):
    ctx_c, _ = cierre_corrido
    esc = privacidad.escanear_salidas(ctx_c.dir_salida, ctx_c.indice_tablas(), ctx_c.celda_minima)
    assert privacidad.total_violaciones(esc) == 0, esc.filter(pl.col("n_violaciones") > 0)
    assert ctx_c.cortafuegos.registro().height == 0 and ctx_c.cortafuegos.violaciones() == 0 and ctx_c.cortafuegos.n_bloqueos() == 0


def test_la_tarea_12_no_lee_datos_y_su_resultado_es_identico_en_otra_carpeta(cierre_corrido, pipeline_sintetico, tmp_path, monkeypatch):
    ctx_c, _ = cierre_corrido
    otro = copiar_tablas(pipeline_sintetico.ctx, tmp_path, con_figuras=True)

    def leer_prohibido(*a, **k):
        raise AssertionError("la Tarea 12 no lee datos")

    monkeypatch.setattr(otro.lector, "leer", leer_prohibido)
    cierre.ejecutar(otro, cascada=pipeline_sintetico.r7["cascada"])
    assert otro.cortafuegos.registro().height == 0 and otro.cortafuegos.violaciones() == 0
    for nombre in TABLAS_T12:
        assert (ctx_c.dir_tablas / nombre).read_bytes() == (otro.dir_tablas / nombre).read_bytes(), nombre
    for png in ("F12_01.png", "F12_02.png"):
        assert (ctx_c.dir_figuras / png).read_bytes() == (otro.dir_figuras / png).read_bytes(), png


def test_en_modo_completo_los_cambios_no_hablan_del_extracto_ni_esperan_datos_completos_por_estadistica(cierre_completo):
    ctx_c, r = cierre_completo
    assert ctx_c.etiqueta == "completo"
    todos = pl.concat([r["cambios_fase1"], r["cambios_fase2"]])
    assert "indicativo" not in " ".join(todos["hallazgo"].to_list()) and "cota inferior" not in " ".join(todos["hallazgo"].to_list())
    # en modo completo solo queda pendiente la regla de estadías largas cuando no se confirma en 2023
    pendientes = todos.filter(pl.col("tipo") == "pendiente de datos completos")
    assert all("no se confirma" in h for h in pendientes["hallazgo"].to_list())


# ============================================================================================================================ informe
def test_el_informe_de_cambios_tiene_una_seccion_y_una_tabla_por_documento_con_cambios(cierre_corrido):
    ctx_c, r = cierre_corrido
    inf = Informe(ctx_c, "t25_prueba_cambios.md", "Prueba de los cambios")
    cierre.agregar_secciones_cambios(inf, ctx_c, 6)
    md = inf.guardar().read_text(encoding="utf8")
    assert md.count("\n## ") == cierre.N_SECCIONES_CAMBIOS == 1
    assert "## 6. Cambios sugeridos a los documentos de las fases anteriores" in md and "{" not in md and "}" not in md
    f1, f2 = r["cambios_fase1"], r["cambios_fase2"]
    assert f"Se sugieren {f1.height + f2.height} cambios a los documentos: {f1.height} a la Fase 1 y {f2.height} a la Fase 2" in md
    for doc, t in (("Fase 1", f1), ("Fase 2", f2)):
        if t.height:
            assert f"Cambios sugeridos a la {doc}" in md and t["id"][0] in md and t["hallazgo"][0] in md
        else:
            assert f"Con esta evidencia no hay cambios sugeridos a la {doc}." in md
    cifras = ctx_c.cifras_informe().filter(pl.col("informe") == "t25_prueba_cambios.md")
    assert cifras.height > 0 and (cifras["csv"] == "").sum() == 0 and (cifras["columna"] == "").sum() == 0


def test_el_informe_de_cambios_sin_cambios_lo_dice_para_cada_documento(pipeline_sintetico, tmp_path, monkeypatch):
    ctx_v = copiar_tablas(pipeline_sintetico.ctx, tmp_path, con_figuras=True)
    monkeypatch.setattr(cierre, "construir_cambios", lambda ev: {d: cierre._a_tabla(d, []) for d in cierre.DOCUMENTOS})
    cierre.ejecutar(ctx_v, cascada=pipeline_sintetico.r7["cascada"])
    inf = Informe(ctx_v, "t25_prueba_sin_cambios.md", "Prueba")
    cierre.agregar_secciones_cambios(inf, ctx_v, 1)
    md = inf.guardar().read_text(encoding="utf8")
    assert "Se sugieren 0 cambios a los documentos: 0 a la Fase 1 y 0 a la Fase 2" in md
    assert "Con esta evidencia no hay cambios sugeridos a la Fase 1." in md and "Con esta evidencia no hay cambios sugeridos a la Fase 2." in md


def _tabla_con_leyenda(md, leyenda):
    """Las filas de la tabla del informe cuya leyenda es ``leyenda`` (el bloque que la precede); vacío si la leyenda no está."""
    if f"*{leyenda}.*" not in md:
        return ""
    return md.split(f"*{leyenda}.*")[0].rstrip().rsplit("\n\n", 1)[-1]


def test_el_informe_de_cierre_tiene_tres_secciones_con_el_protocolo_las_limitaciones_y_lo_que_falta(cierre_corrido):
    ctx_c, _ = cierre_corrido
    inf = Informe(ctx_c, "t26_prueba_cierre.md", "Prueba del cierre")
    cierre.agregar_secciones(inf, ctx_c, 2)
    md = inf.guardar().read_text(encoding="utf8")
    assert md.count("\n## ") == cierre.N_SECCIONES == 3
    for titulo in ("## 2. Cambios al protocolo", "## 3. Limitaciones", "## 4. Qué falta con los datos completos"):
        assert titulo in md
    assert "{" not in md and "}" not in md
    assert f"versión v{ctx_c.umbrales.meta['version']}" in md and "cambios_protocolo.csv" in md
    assert "versión v1.5" in md and "| 1.5 |" in md and "el criterio de S09 se amplía con el contraste de los egresos con traslado interno" in md      # el informe lista el cambio 1.5
    assert md.count("\n| 1.") == 5 and "se hicieron 5 cambios al protocolo congelado" in md                                  # los cinco cambios, cada uno en su fila
    assert "Con extractos las cifras son indicativas" in md and "El desenlace de 2024 nunca se leyó" in md and "toda decisión sigue pendiente de decisión humana" in md
    assert "t25_hipotesis_evaluadas.csv" in md                                                         # la tabla de hipótesis no evaluables
    # H04, H10 y H12 no dependen de datos: ninguna corrida las resuelve y van aparte de las que se repiten con los datos completos
    con_datos, sin_datos = _tabla_con_leyenda(md, "Hipótesis que se repiten con los datos completos"), _tabla_con_leyenda(md, "Hipótesis que no dependen de datos")
    for h in ("H04", "H10", "H12"):
        assert f"| {h} |" in sin_datos and f"| {h} |" not in con_datos, h
    for h in ("H01b", "H01c", "H06a", "H06b", "H13", "H16a"):
        assert f"| {h} |" in con_datos and f"| {h} |" not in sin_datos, h
    assert "otras no dependen de datos" in md and "se repiten con los datos completos antes de decidir" in md
    cifras = ctx_c.cifras_informe().filter(pl.col("informe") == "t26_prueba_cierre.md")
    assert cifras.height > 0 and (cifras["csv"] == "").sum() == 0 and (cifras["columna"] == "").sum() == 0
    assert set(cifras.filter(pl.col("formato") != "tabla")["csv"].to_list()) <= {"t25_resumen_evidencia.csv"}


def test_el_informe_de_cierre_en_modo_completo_no_trae_la_salvedad_del_extracto(cierre_completo):
    ctx_c, _ = cierre_completo
    inf = Informe(ctx_c, "t26_prueba_cierre_completo.md", "Prueba del cierre")
    cierre.agregar_secciones(inf, ctx_c, 1)
    md = inf.guardar().read_text(encoding="utf8")
    assert "Con extractos las cifras son indicativas" not in md and "El desenlace de 2024 nunca se leyó" in md and md.count("\n## ") == 3
    assert "## 1. Cambios al protocolo" in md and "## 3. Qué falta con los datos completos" in md
    # con los datos completos quedan solo las hipótesis que no dependen de datos: la tabla de las que se repiten con los datos completos no tiene H04, H10 ni H12
    sin_datos, con_datos = _tabla_con_leyenda(md, "Hipótesis que no dependen de datos"), _tabla_con_leyenda(md, "Hipótesis que se repiten con los datos completos")
    assert all(f"| {h} |" in sin_datos for h in ("H04", "H10", "H12")) and all(f"| {h} |" not in con_datos for h in ("H04", "H10", "H12"))


# FIN-test_t12_cierre
