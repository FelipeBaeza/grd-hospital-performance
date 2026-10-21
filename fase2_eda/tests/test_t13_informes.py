"""Tarea 13, parte C: composición de los siete informes, numeración consecutiva, figuras incrustadas y el informe de cierre."""

from __future__ import annotations

import re

import polars as pl
import pytest

from ayudas_t13 import clonar, etiquetar_completo
from eda import informes_finales, privacidad, verificadores
from eda.informes import Informe, validar_narrativa
from eda.informes_finales import Bloque
from eda.verificadores import Banderas, ResultadoDT

NUMERACION = {"t24_informe.md": 10, "t23_informe.md": 17, "t23_seleccion_variables.md": 14, "t25_evidencia_decisiones.md": 6, "t26_cierre.md": 4}
TITULOS = {
    "t24_informe.md": "Informe de calidad de los datos (CRISP-DM 2.4)",
    "t23_informe.md": "Informe de exploración de los datos (CRISP-DM 2.3)",
    "t23_seleccion_variables.md": "Informe de selección estadística de variables (CRISP-DM 2.3)",
    "t25_evidencia_decisiones.md": "Informe de evidencia para las decisiones de la Fase 1 (CRISP-DM 2.3)",
    "t26_cierre.md": "Informe de cierre de la Fase 2 (CRISP-DM 2.4)",
}


def leer(ctx, nombre: str) -> str:
    return (ctx.dir_informes / nombre).read_text(encoding="utf8")


@pytest.fixture(scope="module")
def ctx(corrida_t13):
    return corrida_t13.ctx


@pytest.fixture(scope="module")
def cierre_compuesto(corrida_t13, tmp_path_factory):
    """Un clon de la corrida con el informe de cierre ya compuesto desde los doce criterios."""
    c = clonar(corrida_t13.ctx, tmp_path_factory.mktemp("t26"))
    banderas = Banderas(sin_pytest=True, sin_determinismo=True, informes_esperados=corrida_t13.previos, cascada=corrida_t13.cascada, cascada_pipeline=corrida_t13.cascada_pipeline)
    resultados = verificadores.verificar_todo(c, banderas)
    informes_finales.componer_cierre(c, resultados)
    c.escribir_registros()
    return c, resultados


# ============================================================================================================================ lista de informes
def test_los_siete_informes_esperados_estan_en_el_orden_de_produccion():
    assert informes_finales.informes_esperados() == (
        "t21_informe.md", "t22_informe.md", "t24_informe.md", "t23_informe.md", "t23_seleccion_variables.md", "t25_evidencia_decisiones.md", "t26_cierre.md",
    )
    assert informes_finales.INFORME_CIERRE == "t26_cierre.md"


def test_las_tareas_se_normalizan_sin_ceros_a_la_izquierda():
    assert [informes_finales.normalizar_tarea(t) for t in ("T3", "t03", " T12 ", "t010")] == ["T3", "T3", "T12", "T10"]
    for malo in ("", "3", "X3", "T", "T3,T4"):
        with pytest.raises(ValueError, match="tarea inválida"):
            informes_finales.normalizar_tarea(malo)


def test_la_numeracion_de_un_informe_se_lee_de_sus_encabezados():
    texto = "# Título\n\n## 1. Uno\n\ntexto\n\n## 2. Dos\n\n### 2.1 Subsección\n\n## 3. Tres\n"
    assert informes_finales.numeracion_secciones(texto) == [1, 2, 3]
    assert informes_finales.numeracion_secciones("sin secciones") == []


# ============================================================================================================================ composición
def test_se_componen_los_cuatro_informes_con_su_titulo_la_nota_de_la_fuente_y_el_parrafo_inicial(ctx):
    for nombre, titulo in TITULOS.items():
        if nombre == "t26_cierre.md":
            continue
        md = leer(ctx, nombre)
        lineas = md.splitlines()
        assert lineas[0] == f"# {titulo}", nombre
        assert lineas[2].startswith("**Fuente de los datos: extracto.**"), nombre
        assert lineas[4].startswith("Este informe "), nombre


def test_los_encabezados_de_cada_informe_son_consecutivos_desde_uno(ctx):
    for nombre, n in NUMERACION.items():
        if nombre == "t26_cierre.md":
            continue
        assert informes_finales.numeracion_secciones(leer(ctx, nombre)) == list(range(1, n + 1)), nombre


def test_ningun_informe_tiene_llaves_sobrantes_ni_cifras_sin_respaldo_en_la_prosa(ctx):
    for nombre in (n for n in informes_finales.informes_esperados() if n != "t26_cierre.md"):
        md = leer(ctx, nombre)
        assert "{" not in md and "}" not in md, nombre
    # los informes compuestos pasaron por Informe.guardar, que rechaza una línea de prosa con dígitos que no sean una cifra registrada: aquí se comprueba que la prosa nueva
    # de esta tarea (párrafos iniciales y de cierre) no trae ningún dígito suelto
    for c in informes_finales.COMPUESTOS:
        validar_narrativa(c.introduccion)                                    # no lanza: solo nombres de tareas y de fases permitidos, ningún dígito suelto


def test_las_cinco_composiciones_dejan_sus_cifras_registradas_con_csv_y_columna(ctx):
    cifras = ctx.cifras_informe()
    for nombre in ("t24_informe.md", "t23_informe.md", "t23_seleccion_variables.md", "t25_evidencia_decisiones.md"):
        sub = cifras.filter(pl.col("informe") == nombre)
        assert sub.height > 20 and (sub["csv"] == "").sum() == 0 and (sub["columna"] == "").sum() == 0, nombre


def test_la_numeracion_de_cada_bloque_empieza_donde_termino_el_anterior(corrida_t13, tmp_path):
    """Los desfases se calculan contando los encabezados que agregó cada llamada, no con una constante (MI01)."""
    c = clonar(corrida_t13.ctx, tmp_path)
    recibidos = []

    def bloque(cantidad):
        def agregar(inf, ctx, n0=1):
            recibidos.append(n0)
            for k in range(cantidad):
                inf.seccion(f"{n0 + k}. Sección de prueba")
        return agregar

    inf = Informe(c, "t99_prueba.md", "Informe de prueba")
    siguiente = informes_finales._agregar_bloques(inf, c, (Bloque(bloque(2)), Bloque(bloque(5)), Bloque(bloque(1)), Bloque(bloque(3))))
    assert recibidos == [1, 3, 8, 9] and siguiente == 12
    informes_finales._validar_numeracion(inf)                                # no lanza
    # un bloque de numeración fija (sin n0) cuenta igual sus encabezados
    fija = Informe(c, "t99_prueba_fija.md", "Informe de prueba")

    def sin_n0(inf_, ctx_):
        for k in range(1, 4):
            inf_.seccion(f"{k}. Fija")

    assert informes_finales._agregar_bloques(fija, c, (Bloque(sin_n0, usa_n0=False), Bloque(bloque(2)))) == 6
    assert recibidos[-1] == 4


def test_una_numeracion_que_no_es_consecutiva_lanza_un_error_en_vez_de_publicarse(corrida_t13, tmp_path):
    c = clonar(corrida_t13.ctx, tmp_path)
    inf = Informe(c, "t99_prueba.md", "Informe de prueba")
    inf.seccion("1. Primera")
    inf.seccion("3. Tercera")
    with pytest.raises(ValueError, match=r"la numeración de las secciones no es consecutiva: \[1, 3\]"):
        informes_finales._validar_numeracion(inf)


def test_con_tareas_parciales_solo_se_componen_los_informes_cuyas_tareas_corrieron(corrida_t13, tmp_path):
    c = clonar(corrida_t13.ctx, tmp_path)
    for archivo in ("t24_informe.md", "t23_informe.md", "t23_seleccion_variables.md", "t25_evidencia_decisiones.md"):
        (c.dir_informes / archivo).unlink()
    nombres = lambda tareas: [p.name for p in informes_finales.componer_informes(c, tareas)]    # noqa: E731
    assert nombres(["T5"]) == [] and nombres([]) == [] and nombres(["T3", "T4"]) == []
    assert nombres(["T5", "T6"]) == ["t24_informe.md"]
    assert nombres(["t07", "T8", "T9"]) == ["t23_informe.md"] and nombres(["T7", "T8"]) == []
    assert nombres(["T10", "T11"]) == ["t23_seleccion_variables.md"] and nombres(["T11"]) == []
    assert nombres(["T12"]) == ["t25_evidencia_decisiones.md"]
    assert sorted(p.name for p in c.dir_informes.glob("*.md")) == sorted(
        ["t21_informe.md", "t22_informe.md", "t24_informe.md", "t23_informe.md", "t23_seleccion_variables.md", "t25_evidencia_decisiones.md"])
    assert [p.name for p in informes_finales.componer_informes(c)] == ["t24_informe.md", "t23_informe.md", "t23_seleccion_variables.md", "t25_evidencia_decisiones.md"]


FRASES_DEL_EXTRACTO = {
    "t23_informe.md": ("Los extractos no son una muestra aleatoria simple: las prevalencias", "En el extracto los conteos absolutos no se comparan con los datos completos"),
    "t23_seleccion_variables.md": ("Los datos son un extracto y no una muestra aleatoria simple",),
    "t25_evidencia_decisiones.md": ("Con extractos la evidencia es indicativa", "Con extractos el resultado de las proporciones es indicativo"),
}


def test_las_frases_que_mencionan_el_extracto_solo_van_si_la_corrida_uso_extractos(ctx, corrida_t13, tmp_path):
    for nombre, frases in FRASES_DEL_EXTRACTO.items():
        assert all(f in leer(ctx, nombre) for f in frases), nombre                 # con extractos están
    c = clonar(corrida_t13.ctx, tmp_path, etiqueta="completo")
    etiquetar_completo(c)
    informes_finales.componer_informes(c)
    for nombre in ("t24_informe.md", "t23_informe.md", "t23_seleccion_variables.md", "t25_evidencia_decisiones.md"):
        md = leer(c, nombre)
        assert md.splitlines()[2].startswith("**Fuente de los datos: completo.**"), nombre
        assert not any(f in md for f in FRASES_DEL_EXTRACTO.get(nombre, ())), nombre   # con los datos completos no


def pie_de(ctx, dibujar) -> str:
    """El pie extra con el que una función dibuja su figura (un espía de ``guardar_figura``)."""
    capturado = {}
    original = ctx.figuras.guardar_figura

    def espia(fig, **kw):
        capturado.update(kw)
        return original(fig, **kw)

    ctx.figuras.guardar_figura = espia
    dibujar(ctx)
    return capturado["extra_pie"]


def test_los_pies_de_figura_que_mencionan_el_extracto_solo_van_con_extractos(corrida_t13, tmp_path):
    from eda import cohorte, hospitales

    con_extractos = clonar(corrida_t13.ctx, tmp_path / "a")
    completo = clonar(corrida_t13.ctx, tmp_path / "b", etiqueta="completo")
    assert pie_de(con_extractos, cohorte.figura_serie_mensual).endswith("En el extracto el volumen anual es fijo: solo la mortalidad es comparable entre años.")
    assert pie_de(con_extractos, hospitales.figura_historial).endswith("En un extracto el egreso previo se subestima.")
    for dibujar in (cohorte.figura_serie_mensual, hospitales.figura_historial):
        assert "extracto" not in pie_de(completo, dibujar), dibujar.__name__


# ============================================================================================================================ figuras
def test_las_33_figuras_del_indice_estan_incrustadas_con_su_leyenda_y_su_csv(ctx):
    assert informes_finales.figuras_sin_incrustar(ctx) == []
    idx = ctx.figuras.indice()
    assert idx.height == 33
    por_informe = {"t21_informe.md": {"F02_01", "F03_01"}, "t22_informe.md": {"F04_01", "F04_02", "F04_03"}, "t24_informe.md": {"F05_01", "F05_02", "F05_03", "F06_01", "F06_02"},
                   "t25_evidencia_decisiones.md": {"F12_01", "F12_02"}}
    for nombre, ids in por_informe.items():
        md = leer(ctx, nombre)
        assert all(f"*Figura {i}. " in md for i in ids), nombre
    en_t23 = leer(ctx, "t23_informe.md")
    assert all(f"*Figura F{n}_" in en_t23 for n in ("07", "08", "09"))
    en_seleccion = leer(ctx, "t23_seleccion_variables.md")
    assert all(f"*Figura F{n}_" in en_seleccion for n in ("10", "11"))
    # cada leyenda cita la tabla de origen y cada enlace apunta a un PNG que existe
    for f in idx.iter_rows(named=True):
        md = "\n".join(leer(ctx, n) for n in ctx.dir_informes.glob("*.md") and sorted(p.name for p in ctx.dir_informes.glob("*.md")))
        assert f"](../figuras/{f['archivo_png']})" in md and f"Datos de origen: `tablas/{f['csv']}`" in md
        assert (ctx.dir_figuras / f["archivo_png"]).is_file()


def test_una_figura_cuyo_informe_falta_se_detecta_como_sin_incrustar(corrida_t13, tmp_path):
    c = clonar(corrida_t13.ctx, tmp_path)
    (c.dir_informes / "t25_evidencia_decisiones.md").unlink()
    assert informes_finales.figuras_sin_incrustar(c) == ["F12_01", "F12_02"]
    (c.dir_informes / "t24_informe.md").unlink()
    assert informes_finales.figuras_sin_incrustar(c) == ["F05_01", "F05_02", "F05_03", "F06_01", "F06_02", "F12_01", "F12_02"]


def test_una_figura_necesita_el_enlace_la_leyenda_y_la_tabla_de_origen_para_contar_como_incrustada(corrida_t13, tmp_path):
    c = clonar(corrida_t13.ctx, tmp_path)
    ruta = c.dir_informes / "t24_informe.md"
    original = ruta.read_text(encoding="utf8")
    sin_enlace = original.replace("](../figuras/F05_01.png)", "](../figuras/otra.png)")
    ruta.write_text(sin_enlace, encoding="utf8")
    assert informes_finales.figuras_sin_incrustar(c) == ["F05_01"]
    sin_leyenda = original.replace("*Figura F05_02. ", "*Figura F05_99. ")
    ruta.write_text(sin_leyenda, encoding="utf8")
    assert informes_finales.figuras_sin_incrustar(c) == ["F05_02"]
    otra_tabla = original.replace("Datos de origen: `tablas/t24_id_largo.csv`", "Datos de origen: `tablas/t24_reglas.csv`")
    assert otra_tabla != original
    ruta.write_text(otra_tabla, encoding="utf8")
    assert informes_finales.figuras_sin_incrustar(c) == ["F05_03"]
    ruta.write_text(original, encoding="utf8")
    assert informes_finales.figuras_sin_incrustar(c) == []


def test_una_carpeta_sin_informes_deja_todas_las_figuras_sin_incrustar(corrida_t13, tmp_path):
    c = clonar(corrida_t13.ctx, tmp_path)
    for p in c.dir_informes.glob("*.md"):
        p.unlink()
    assert len(informes_finales.figuras_sin_incrustar(c)) == 33


# ============================================================================================================================ cierre
def test_el_informe_de_cierre_trae_cuatro_secciones_con_los_doce_criterios(cierre_compuesto):
    c, resultados = cierre_compuesto
    md = leer(c, "t26_cierre.md")
    assert md.splitlines()[0] == f"# {TITULOS['t26_cierre.md']}"
    assert informes_finales.numeracion_secciones(md) == [1, 2, 3, 4]
    for titulo in ("## 1. Resultado de la verificación", "## 2. Cambios al protocolo", "## 3. Limitaciones", "## 4. Qué falta con los datos completos"):
        assert titulo in md
    for r in resultados:
        assert f"| {r.id} | {r.criterio} | {r.estado} | {r.detalle} |" in md
    assert "{" not in md and "}" not in md


def test_el_informe_de_cierre_dice_los_conteos_desde_la_tabla_de_resumen(cierre_compuesto):
    c, resultados = cierre_compuesto
    md = leer(c, "t26_cierre.md")
    resumen = c.leer_tabla("t26_resumen_cierre.csv").row(0, named=True)
    assert resumen["n_pass"] == 9 and resumen["n_pendiente"] == 3 and resumen["n_fail"] == 0
    assert "La definición de terminado de la Fase 2 tiene 12 criterios. En esta corrida 9 pasan, 0 fallan y 3 quedan pendientes." in md
    assert f"Las Tareas 3 a 12 produjeron {resumen['n_tablas']} tablas y 33 figuras, reunidas en 7 informes." in md
    assert "Hay criterios que fallan" not in md
    assert "Con extractos el criterio de los datos completos queda pendiente" in md
    cifras = c.cifras_informe().filter(pl.col("informe") == "t26_cierre.md")
    assert "t26_resumen_cierre.csv" in cifras["csv"].to_list() and (cifras["csv"] == "").sum() == 0 and (cifras["columna"] == "").sum() == 0


def test_si_un_criterio_falla_el_informe_de_cierre_lo_dice(corrida_t13, tmp_path):
    c = clonar(corrida_t13.ctx, tmp_path)
    resultados = [ResultadoDT("DT01", "Criterio uno", "FAIL", "detalle del fallo", "x"), ResultadoDT("DT02", "Criterio dos", "PASS", "bien", "y")]
    informes_finales.componer_cierre(c, resultados)
    md = leer(c, "t26_cierre.md")
    assert "En esta corrida 1 pasan, 1 fallan y 0 quedan pendientes." in md and "Hay criterios que fallan" in md
    assert "| DT01 | Criterio uno | FAIL | detalle del fallo |" in md


def test_componer_el_cierre_sin_resultados_usa_las_tablas_ya_escritas(cierre_compuesto, corrida_t13, tmp_path):
    c, resultados = cierre_compuesto
    otro = clonar(c, tmp_path)
    antes = leer(otro, "t26_cierre.md")
    (otro.dir_informes / "t26_cierre.md").unlink()
    informes_finales.componer_cierre(otro)
    assert leer(otro, "t26_cierre.md") == antes


def test_en_modo_completo_el_cierre_no_dice_que_las_cifras_son_de_extractos(cierre_compuesto, corrida_t13, tmp_path):
    con_extractos, _ = cierre_compuesto
    md = leer(con_extractos, "t26_cierre.md")
    assert "Con extractos el criterio de los datos completos queda pendiente" in md and "Con extractos las cifras son indicativas" in md
    c = clonar(corrida_t13.ctx, tmp_path, etiqueta="completo")
    etiquetar_completo(c)
    informes_finales.componer_cierre(c, [ResultadoDT("DT12", "Corrida con datos completos", "PASS", "bien", "x")])
    md = leer(c, "t26_cierre.md")
    assert md.splitlines()[2].startswith("**Fuente de los datos: completo.**")
    assert "Con extractos el criterio" not in md and "Con extractos las cifras son indicativas" not in md


def test_el_cierre_pasa_el_escaneo_de_privacidad(cierre_compuesto):
    c, _ = cierre_compuesto
    esc = privacidad.escanear_salidas(c.dir_salida, c.indice_tablas(), c.celda_minima)
    assert privacidad.total_violaciones(esc) == 0


def test_los_siete_informes_quedan_en_la_carpeta_y_solo_esos(cierre_compuesto):
    c, _ = cierre_compuesto
    assert sorted(p.name for p in c.dir_informes.glob("*.md")) == sorted(informes_finales.informes_esperados())
    assert verificadores.cifras_con_csv(c).estado == "PASS" and verificadores.figuras_indexadas(c).estado == "PASS"
    assert re.fullmatch(r"\d+", str(c.leer_tabla("t26_resumen_cierre.csv")["n_figuras"][0]))
