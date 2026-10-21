"""Ayudas gráficas de las Tareas 10 y 11 (tanda 1, etapa B): barra de color con coma decimal, recuadros de grupo, línea gruesa y alto por filas."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colorbar import Colorbar
from matplotlib.patches import Rectangle

from ayudas_figuras import pares_solapados, vigilar_savefig
from eda import graficos


def _mapa(n: int = 4, **kw):
    fig, ax = graficos.nueva_figura()
    m = np.linspace(0.0, 1.0, n * n).reshape(n, n)
    etiquetas = [f"v{i}" for i in range(n)]
    barra = graficos.mapa_calor(fig, ax, m, etiquetas, etiquetas, vmin=0.0, vmax=1.0, anotar=False, **kw)
    return fig, ax, barra


# ------------------------------------------------------------------------------------------------ mapa_calor y la barra de color
def test_mapa_calor_devuelve_la_barra_de_color_que_crea():
    fig, ax, barra = _mapa()
    try:
        assert isinstance(barra, Colorbar)
        assert barra.ax is not ax and barra.ax in fig.axes                      # es la barra de la figura, no los ejes del mapa
        assert barra.ax is fig.axes[-1]                                         # la que el mapa agregó, y se entrega explícitamente
    finally:
        plt.close(fig)


def test_mapa_calor_con_anotaciones_y_con_etiqueta_de_barra_tambien_devuelve_la_barra():
    fig, ax = graficos.nueva_figura()
    barra = graficos.mapa_calor(fig, ax, np.array([[1.0, 0.5], [0.5, 1.0]]), ["a", "b"], ["a", "b"], etiqueta_barra="Asociación", vmin=0.0, vmax=1.0)
    try:
        assert isinstance(barra, Colorbar) and barra.ax.get_ylabel() == "Asociación"
        assert len([t for t in ax.texts]) == 4                                  # las anotaciones siguen saliendo con coma
        assert {t.get_text() for t in ax.texts} == {"1,00", "0,50"}
    finally:
        plt.close(fig)


def test_formato_barra_decimal_pone_coma_decimal_en_las_marcas_de_la_barra():
    fig, _, barra = _mapa()
    try:
        barra.set_ticks([0.0, 0.5, 1.0])
        fig.canvas.draw()
        antes = [t.get_text() for t in barra.ax.get_yticklabels()]
        assert "0.5" in antes                                                    # sin la ayuda, la marca sale con punto: la prueba distingue
        graficos.formato_barra_decimal(barra)
        fig.canvas.draw()
        despues = [t.get_text() for t in barra.ax.get_yticklabels()]
        assert "0,5" in despues and not any("." in t for t in despues), despues
    finally:
        plt.close(fig)


def test_formato_barra_decimal_resiste_un_set_ticks_posterior():
    fig, _, barra = _mapa()
    try:
        graficos.formato_barra_decimal(barra)
        barra.set_ticks([0.0, 0.25, 1.0])
        fig.canvas.draw()
        assert [t.get_text() for t in barra.ax.get_yticklabels()] == ["0", "0,25", "1"]
    finally:
        plt.close(fig)


# ------------------------------------------------------------------------------------------------ recuadros y línea gruesa
def test_recuadros_de_grupo_dibujan_un_rectangulo_por_bloque_con_las_coordenadas_de_las_celdas():
    fig, ax, _ = _mapa(6)
    try:
        antes = len(ax.patches)
        creados = graficos.recuadros_de_grupo(ax, [(0, 2), (3, 3)], color="red", ancho=2.0)
        assert len(creados) == 2 and len(ax.patches) == antes + 2
        assert all(isinstance(r, Rectangle) for r in creados)
        r0, r1 = creados
        assert (r0.get_x(), r0.get_y(), r0.get_width(), r0.get_height()) == (-0.5, -0.5, 2, 2)           # la celda i se centra en i: el bloque (0, 2) va de -0.5 a 1.5
        assert (r1.get_x(), r1.get_y(), r1.get_width(), r1.get_height()) == (2.5, 2.5, 3, 3)
        assert not r0.get_fill() and r0.get_linewidth() == 2.0 and r0.get_edgecolor()[:3] == (1.0, 0.0, 0.0)
    finally:
        plt.close(fig)


def test_recuadros_de_grupo_sin_bloques_o_con_bloques_vacios_no_dibujan_nada():
    fig, ax, _ = _mapa()
    try:
        antes = len(ax.patches)
        assert graficos.recuadros_de_grupo(ax, []) == [] and graficos.recuadros_de_grupo(ax, [(1, 0)]) == []
        assert len(ax.patches) == antes
        r = graficos.recuadros_de_grupo(ax, [(1, 1)])                                                      # un bloque de un miembro es válido
        assert len(r) == 1 and len(ax.patches) == antes + 1
    finally:
        plt.close(fig)


def test_linea_gruesa_dibuja_una_linea_horizontal_y_una_vertical_entre_dos_celdas():
    fig, ax, _ = _mapa(5)
    try:
        antes = len(ax.lines)
        h, v = graficos.linea_gruesa(ax, 4)
        assert len(ax.lines) == antes + 2
        assert list(h.get_ydata()) == [3.5, 3.5] and list(v.get_xdata()) == [3.5, 3.5]                    # entre la celda 3 y la 4
        assert h.get_linewidth() == 1.6 and v.get_linewidth() == 1.6 and h.get_color() == "black"
        h2, v2 = graficos.linea_gruesa(ax, 2, color="red", ancho=3.0)
        assert h2.get_linewidth() == 3.0 and v2.get_color() == "red"
    finally:
        plt.close(fig)


# ------------------------------------------------------------------------------------------------ altura_filas
def test_altura_filas_es_monotona_y_respeta_el_minimo():
    alturas = [graficos.altura_filas(n) for n in range(0, 400)]
    assert all(b >= a for a, b in zip(alturas, alturas[1:]))                                              # nunca decrece
    assert min(alturas) == 4.0 and graficos.altura_filas(0) == 4.0 and graficos.altura_filas(10) == 4.0   # 10 filas serían 3,5: manda el mínimo
    assert alturas[-1] > alturas[20]                                                                       # y crece pasado el codo
    assert graficos.altura_filas(59) == pytest.approx(59 * 0.17 + 1.8)
    assert graficos.altura_filas(30, por_fila=0.5, minimo=2.0, extra=1.0) == 16.0
    assert graficos.altura_filas(1, por_fila=0.5, minimo=2.0, extra=1.0) == 2.0


# ================================================================================================================ tanda 2: etiquetas y gancho (etapa C)
def _dispersion(n: int, semilla: int = 3, **kw):
    """Figura con ``n`` puntos dispersos y sus nombres; devuelve ``(fig, ax, xs, ys, textos)``."""
    rng = np.random.default_rng(semilla)
    xs, ys = rng.uniform(0.0, 1.0, n), rng.uniform(0.0, 1.0, n)
    fig, ax = graficos.nueva_figura(**kw)
    ax.scatter(xs, ys, s=18)
    return fig, ax, xs, ys, [f"VARIABLE_{i:02d}" for i in range(n)]


def _registro(tmp_path, nombre_csv="respaldo.csv"):
    (tmp_path / "tablas").mkdir(exist_ok=True)
    (tmp_path / "tablas" / nombre_csv).write_text("a\n1\n", encoding="utf8")
    return graficos.RegistroFiguras(tmp_path / "figuras", tmp_path / "tablas", "extracto"), nombre_csv


def test_etiquetas_sin_solape_con_25_puntos_separados_no_deja_cuadros_solapados():
    fig, ax, xs, ys, textos = _dispersion(25)
    try:
        anotaciones = graficos.etiquetas_sin_solape(ax, xs, ys, textos)
        assert len(anotaciones) == 25 and [a.get_text() for a in anotaciones] == textos        # una por punto y en el orden de entrada
        assert pares_solapados(fig) == []
    finally:
        plt.close(fig)


def test_etiquetas_sin_solape_separa_las_etiquetas_de_dos_puntos_identicos():
    fig, ax = graficos.nueva_figura()
    ax.scatter([0.5, 0.5], [0.5, 0.5], s=18)
    try:
        a, b = graficos.etiquetas_sin_solape(ax, [0.5, 0.5], [0.5, 0.5], ["PRIMERA", "SEGUNDA"])
        assert pares_solapados(fig) == []
        assert (a.xyann, a.get_ha(), a.get_va()) != (b.xyann, b.get_ha(), b.get_va())              # la segunda tuvo que buscar otro sitio
    finally:
        plt.close(fig)


def test_etiquetas_sin_solape_es_determinista_con_los_mismos_datos():
    posiciones = []
    for _ in range(2):
        fig, ax, xs, ys, textos = _dispersion(30, semilla=11)
        try:
            posiciones.append([(a.xyann, a.get_ha(), a.get_va()) for a in graficos.etiquetas_sin_solape(ax, xs, ys, textos)])
        finally:
            plt.close(fig)
    assert posiciones[0] == posiciones[1]


def test_las_anotaciones_se_crean_fuera_del_diseno_y_el_primer_candidato_es_a_la_derecha():
    fig, ax, xs, ys, textos = _dispersion(1)
    try:
        (a,) = graficos.etiquetas_sin_solape(ax, xs, ys, textos, radios=(6,))
        assert a.get_in_layout() is False and a in ax.texts
        assert a.get_ha() == "left" and a.get_va() == "center" and a.xyann == (6.0, 0.0)            # sin estorbos gana la primera dirección con el menor radio
    finally:
        plt.close(fig)


def test_etiquetas_sin_solape_sin_puntos_no_hace_nada_y_largos_distintos_fallan():
    fig, ax = graficos.nueva_figura()
    try:
        assert graficos.etiquetas_sin_solape(ax, [], [], []) == []
        with pytest.raises(ValueError, match="largos distintos"):
            graficos.etiquetas_sin_solape(ax, [0.1, 0.2], [0.1], ["a", "b"])
    finally:
        plt.close(fig)


def test_si_ninguna_posicion_esta_libre_elige_la_de_menor_solape():
    fig, ax = graficos.nueva_figura(figsize=(3.0, 2.0))
    xs, ys = np.linspace(0.45, 0.55, 12), np.full(12, 0.5)                # doce puntos casi encimados: no hay lugar sin solape
    ax.scatter(xs, ys, s=18)
    try:
        anotaciones = graficos.etiquetas_sin_solape(ax, xs, ys, [f"punto {i}" for i in range(12)])
        assert len(anotaciones) == 12                                          # no falla ni deja puntos sin etiqueta
        assert all(a.get_window_extent().width > 0 for a in anotaciones)
    finally:
        plt.close(fig)


def test_el_gancho_se_llama_una_vez_con_el_titulo_y_el_pie_ya_puestos(tmp_path):
    registro, csv = _registro(tmp_path)
    fig, ax, xs, ys, textos = _dispersion(3)
    vistos = []

    def gancho(f):
        vistos.append((f._suptitle.get_text() if f._suptitle is not None else None, [t.get_text() for t in f.texts]))

    registro.guardar_figura(fig, id_figura="F99_01", tarea=11, titulo="Título de prueba", descripcion_alt="Descripción", csv=csv, anios="2023", antes_de_guardar=gancho)
    assert len(vistos) == 1
    titulo, textos_figura = vistos[0]
    assert titulo == "Título de prueba" and any(t.startswith("Fuente: egresos GRD de FONASA (extracto)") for t in textos_figura)
    assert (tmp_path / "figuras" / "F99_01.png").is_file()


def test_sin_gancho_no_hay_ningun_draw_adicional_y_con_gancho_hay_exactamente_uno(tmp_path):
    registro, csv = _registro(tmp_path)
    llamadas = []
    for con_gancho in (False, True):
        fig, ax, xs, ys, textos = _dispersion(3)
        original = fig.canvas.draw
        fig.canvas.draw = lambda *a, _o=original, **k: (llamadas.append(con_gancho), _o(*a, **k))[1]           # lienzo espía
        registro.guardar_figura(
            fig, id_figura="F99_02", tarea=11, titulo="Título", descripcion_alt="Descripción", csv=csv, anios="2023", antes_de_guardar=(lambda f: None) if con_gancho else None,
        )
    assert llamadas == [True]                                                  # sin gancho: ninguno; con gancho: uno


def test_el_gancho_no_cambia_el_resultado_de_una_figura_sin_gancho(tmp_path):
    registro, csv = _registro(tmp_path)
    hashes = []
    for i in range(2):
        fig, ax, xs, ys, textos = _dispersion(5, semilla=2)
        registro.guardar_figura(fig, id_figura=f"F99_0{i}", tarea=11, titulo="T", descripcion_alt="D", csv=csv, anios="2023")
        hashes.append((tmp_path / "figuras" / f"F99_0{i}.png").read_bytes())
    assert hashes[0] == hashes[1]


def test_prueba_del_diseno_final_las_etiquetas_no_se_solapan_al_guardar(tmp_path, monkeypatch):
    registro, csv = _registro(tmp_path)
    resultados = vigilar_savefig(monkeypatch)
    fig, ax, xs, ys, textos = _dispersion(25, semilla=5)
    registro.guardar_figura(
        fig, id_figura="F99_03", tarea=11, titulo="Etiquetas sobre el diseño final", descripcion_alt="Dispersión", csv=csv, anios="2023",
        extra_pie="Pie largo para que la reserva inferior crezca y mueva los ejes. " * 3,
        antes_de_guardar=lambda f: graficos.etiquetas_sin_solape(f.axes[0], xs, ys, textos),
    )
    assert len(resultados) == 1 and resultados[0] == []


def test_la_prueba_del_diseno_final_detecta_un_solape_si_las_etiquetas_se_ponen_antes_del_titulo_y_el_pie(tmp_path, monkeypatch):
    """Si se miden antes del título y el pie (mutante M50), el diseño final las mueve y la vigilancia de ``savefig`` lo ve o, al menos, la medición previa no coincide."""
    registro, csv = _registro(tmp_path)
    fig, ax, xs, ys, textos = _dispersion(40, semilla=8, figsize=(4.0, 3.0))
    graficos.etiquetas_sin_solape(ax, xs, ys, textos)                                   # medidas ANTES del título y del pie (como en la mutación)
    antes = [a.get_window_extent() for a in ax.texts]
    fig.suptitle("Un título largo " * 4, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.012, "Un pie de varias líneas. " * 12, ha="left", va="bottom", fontsize=6.5, wrap=True)
    fig.canvas.draw()
    despues = [a.get_window_extent() for a in ax.texts]
    desplazamiento = max(abs(b.y0 - a.y0) for a, b in zip(antes, despues))
    assert desplazamiento > 0.5                                                              # el título y el pie mueven los ejes: medir antes no es medir el diseño final
    plt.close(fig)


def test_mapa_calor_anota_un_negativo_que_se_redondea_a_cero_sin_signo_pero_conserva_el_de_los_demas():
    fig, ax = graficos.nueva_figura()
    try:
        m = np.array([[-0.0004, 0.0004, -0.004], [-0.05, 0.25, -0.1]])
        graficos.mapa_calor(fig, ax, m, ["a", "b"], ["a", "b", "c"], vmin=-0.3, vmax=0.3)
        assert [t.get_text() for t in ax.texts] == ["0,00", "0,00", "0,00", "-0,05", "0,25", "-0,10"]                 # "-0,00" no se anota
    finally:
        plt.close(fig)


# FIN-test_graficos_t10t11
