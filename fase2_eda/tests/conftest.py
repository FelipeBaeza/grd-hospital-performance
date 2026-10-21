"""Fixtures comunes: datos sintéticos y contexto de ejecución en carpetas temporales."""

from __future__ import annotations

import pytest

from eda.contexto import Contexto
from sintetico import crear_fuente_sintetica


@pytest.fixture(scope="session")
def fuente_sintetica(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("datos_sinteticos")
    return crear_fuente_sintetica(raiz, n=600)


@pytest.fixture()
def ctx(tmp_path, fuente_sintetica):
    """Contexto sobre la fuente sintética (etiquetada 'extracto') con salidas en tmp_path."""
    return Contexto.crear("extracto", fuente=fuente_sintetica, dir_salida_base=tmp_path)


@pytest.fixture(scope="session")
def pipeline_sintetico(tmp_path_factory):
    """Tareas 3 a 11 sobre la fuente sintética (700 filas por año, semilla 29), una sola vez por sesión, con los registros escritos.

    Es de SOLO LECTURA por contrato: ninguna prueba debe escribir tablas ni métricas en este contexto. Quien necesite escribir copia las tablas
    (``ayudas_t12.copiar_tablas``) a un contexto propio. Devuelve un espacio de nombres con ``ctx``, ``cat``, ``r5`` (resultado de la Tarea 5), ``r7`` (de la Tarea 7,
    con la cascada sin suprimir), ``raiz`` y ``fuente``.
    """
    from types import SimpleNamespace

    from eda import (asociacion, calidad, calidad_catalogos, catalogos, cohorte, comorbilidad, descripcion, hospitales, recoleccion, seleccion)
    from eda import panel as modulo_panel

    raiz = tmp_path_factory.mktemp("pipeline_sintetico")
    fuente = crear_fuente_sintetica(raiz / "datos", n=700, semilla=29)
    ctx_p = Contexto.crear("extracto", fuente=fuente, dir_salida_base=raiz / "salidas")
    cat = catalogos.cargar_catalogos()
    recoleccion.ejecutar(ctx_p, raiz_proyecto=raiz)
    descripcion.ejecutar(ctx_p)
    r5 = calidad.ejecutar(ctx_p)
    calidad_catalogos.ejecutar(ctx_p, cat)
    r7 = cohorte.ejecutar(ctx_p, cat)
    comorbilidad.ejecutar(ctx_p, cat)
    regimenes = [[int(a) for a in s.split(",")] for s in r5["regimenes"]["anios"].to_list()]
    hospitales.ejecutar(ctx_p, cat, regimenes=regimenes)
    panel = modulo_panel.construir_panel(ctx_p, cat, tarea="T10")
    asoc = asociacion.ejecutar(ctx_p, panel, cat)
    seleccion.ejecutar(ctx_p, panel, asoc, cat)
    ctx_p.escribir_registros()
    return SimpleNamespace(ctx=ctx_p, cat=cat, r5=r5, r7=r7, raiz=raiz, fuente=fuente)


@pytest.fixture(scope="session")
def corrida_t13(pipeline_sintetico, tmp_path_factory):
    """Tareas 3 a 12 y los cinco informes compuestos (sin el de cierre) sobre la fuente sintética: la carpeta de salidas que reciben los verificadores.

    Es de SOLO LECTURA por contrato; quien necesite dañar una salida la clona (``ayudas_t13.clonar``). Devuelve un espacio de nombres con ``ctx``, ``cascada`` y ``cascada_pipeline``
    (las de la Tarea 7 sin suprimir) y ``previos`` (los seis informes que existen antes del de cierre).
    """
    from types import SimpleNamespace

    from ayudas_t13 import clonar
    from eda import cierre, informes_finales

    ctx_c = clonar(pipeline_sintetico.ctx, tmp_path_factory.mktemp("corrida_t13"))
    cierre.ejecutar(ctx_c, cascada=pipeline_sintetico.r7["cascada"])
    informes_finales.componer_informes(ctx_c)
    ctx_c.escribir_registros()
    previos = tuple(n for n in informes_finales.informes_esperados() if n != informes_finales.INFORME_CIERRE)
    return SimpleNamespace(ctx=ctx_c, cascada=pipeline_sintetico.r7["cascada"], cascada_pipeline=pipeline_sintetico.r7["cascada_pipeline"], previos=previos)
