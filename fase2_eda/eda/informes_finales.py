"""Composición de los siete informes Markdown de la Fase 2 y chequeos de que cada figura queda incrustada (Tarea 13).

Los informes de las Tareas 3 y 4 (``t21_informe.md`` y ``t22_informe.md``) los escriben sus propias tareas. Aquí se componen los otros cinco con la clase ``Informe`` a partir de
las ``agregar_secciones`` de cada tarea: ``t24_informe.md`` (calidad), ``t23_informe.md`` (exploración), ``t23_seleccion_variables.md`` (Tareas 10 y 11),
``t25_evidencia_decisiones.md`` (Tarea 12) y ``t26_cierre.md`` (verificación y cierre). La numeración de las secciones no es una constante: cada bloque empieza donde terminó el
anterior, contando los encabezados que agregó, de modo que si una tarea cambia su número de secciones los informes se ajustan solos (y una numeración que no sea
consecutiva lanza un error en vez de publicarse). Toda cifra viaja como marcador leído de un CSV; los textos fijos no llevan dígitos sueltos.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import asociacion, calidad, calidad_catalogos, cierre, cohorte, comorbilidad, decisiones, hospitales, ledger, seleccion, verificadores
from .contexto import Contexto
from .informes import Informe, cifra

INFORMES_ESPERADOS = (
    "t21_informe.md",
    "t22_informe.md",
    "t24_informe.md",
    "t23_informe.md",
    "t23_seleccion_variables.md",
    "t25_evidencia_decisiones.md",
    "t26_cierre.md",
)
INFORME_CIERRE = "t26_cierre.md"
TITULO_CIERRE = "Informe de cierre de la Fase 2 (CRISP-DM 2.4)"
N_SECCIONES_VERIFICACION = 1
ENCABEZADO_SECCION = re.compile(r"^## (\d+)\. ", re.MULTILINE)


@dataclass(frozen=True)
class Bloque:
    """Una llamada a ``agregar_secciones``: algunas (calidad) tienen numeración fija y no reciben ``n0``."""

    funcion: Callable[..., None]
    usa_n0: bool = True


@dataclass(frozen=True)
class Compuesto:
    """Un informe compuesto: su archivo, su título, el párrafo inicial, las tareas que necesita y sus bloques de secciones."""

    archivo: str
    titulo: str
    introduccion: str
    tareas: tuple[str, ...]
    bloques: tuple[Bloque, ...]


COMPUESTOS: tuple[Compuesto, ...] = (
    Compuesto(
        "t24_informe.md", "Informe de calidad de los datos (CRISP-DM 2.4)",
        "Este informe reúne la calidad de los datos: fechas, identificadores y duplicados (Tarea 5) y catálogos y consistencia cruzada (Tarea 6). Cada cifra sale de las tablas de la corrida.",
        ("T5", "T6"), (Bloque(calidad.agregar_secciones, False), Bloque(calidad_catalogos.agregar_secciones, False)),
    ),
    Compuesto(
        "t23_informe.md", "Informe de exploración de los datos (CRISP-DM 2.3)",
        "Este informe reúne la exploración de los datos: la cohorte, el resultado y la estadía (Tarea 7), la comorbilidad (Tarea 8) y los hospitales (Tarea 9).",
        ("T7", "T8", "T9"), (Bloque(cohorte.agregar_secciones), Bloque(comorbilidad.agregar_secciones), Bloque(hospitales.agregar_secciones)),
    ),
    Compuesto(
        "t23_seleccion_variables.md", "Informe de selección estadística de variables (CRISP-DM 2.3)",
        "Este informe reúne la evidencia estadística para elegir variables: la asociación, la redundancia y la multicolinealidad (Tarea 10) y la selección, los proxies del hospital, "
        "la fuga y la matriz de decisión (Tarea 11). La evidencia estadística se contrasta con el criterio clínico y no lo reemplaza.",
        ("T10", "T11"), (Bloque(asociacion.agregar_secciones), Bloque(seleccion.agregar_secciones)),
    ),
    Compuesto(
        "t25_evidencia_decisiones.md", "Informe de evidencia para las decisiones de la Fase 1 (CRISP-DM 2.3)",
        "Este informe contrasta las cifras documentadas de la Fase 1 y de la Fase 2 con lo medido, resume la evidencia de cada decisión pendiente de la Fase 1 y sugiere cambios a "
        "los documentos. La evidencia recomienda y no decide.",
        ("T12",), (Bloque(ledger.agregar_secciones), Bloque(decisiones.agregar_secciones), Bloque(cierre.agregar_secciones_cambios)),
    ),
)


def informes_esperados() -> tuple[str, ...]:
    """Los siete informes de la Fase 2, en el orden en que se producen."""
    return INFORMES_ESPERADOS


def normalizar_tarea(tarea: str) -> str:
    """``t03`` y ``T3`` son la tarea ``T3``."""
    m = re.fullmatch(r"\s*[Tt]0*(\d+)\s*", tarea)
    if not m:
        raise ValueError(f"tarea inválida: {tarea!r}")
    return f"T{int(m.group(1))}"


# ============================================================================================================================ numeración
def _n_secciones(inf: Informe) -> int:
    return sum(1 for p in inf.partes if p.startswith("## "))


def numeracion_secciones(texto: str) -> list[int]:
    """Los números de los encabezados de sección (``## 3. Título``) de un informe, en el orden en que aparecen."""
    return [int(n) for n in ENCABEZADO_SECCION.findall(texto)]


def _validar_numeracion(inf: Informe) -> None:
    numeros = numeracion_secciones(inf.a_markdown())
    if numeros != list(range(1, len(numeros) + 1)):
        raise ValueError(f"{inf.nombre_archivo}: la numeración de las secciones no es consecutiva: {numeros}")


def _agregar_bloques(inf: Informe, ctx: Contexto, bloques: Iterable[Bloque], n0: int = 1) -> int:
    """Agrega los bloques uno tras otro; cada uno empieza donde terminó el anterior. Devuelve el número siguiente."""
    for b in bloques:
        antes = _n_secciones(inf)
        if b.usa_n0:
            b.funcion(inf, ctx, n0)
        else:
            b.funcion(inf, ctx)
        n0 += _n_secciones(inf) - antes
    return n0


# ============================================================================================================================ informes compuestos
def componer_informes(ctx: Contexto, tareas: Iterable[str] | None = None) -> list[Path]:
    """Compone y guarda los informes de calidad, exploración, selección de variables y evidencia cuyas tareas corrieron (todas, si ``tareas`` es ``None``)."""
    corridas = None if tareas is None else {normalizar_tarea(t) for t in tareas}
    guardados: list[Path] = []
    for c in COMPUESTOS:
        if corridas is not None and not set(c.tareas) <= corridas:
            continue
        inf = Informe(ctx, c.archivo, c.titulo)
        inf.texto_libre_sin_cifras(c.introduccion)
        _agregar_bloques(inf, ctx, c.bloques)
        _validar_numeracion(inf)
        guardados.append(inf.guardar())
    return guardados


# ============================================================================================================================ cierre
def seccion_verificacion(inf: Informe, ctx: Contexto, n0: int = 1) -> None:
    """Agrega al informe de cierre (``N_SECCIONES_VERIFICACION = 1``) el resultado de los criterios de terminado.

    Necesita ``verificacion_dt.csv`` y ``t26_resumen_cierre.csv`` (los escribe ``verificadores.escribir_verificacion``).
    """
    r = lambda col: cifra(ctx, "t26_resumen_cierre.csv", col)  # noqa: E731
    resumen = ctx.leer_tabla("t26_resumen_cierre.csv").row(0, named=True)
    inf.seccion(f"{n0}. Resultado de la verificación")
    inf.parrafo(
        "La definición de terminado de la Fase 2 tiene {n_crit} criterios. En esta corrida {n_pass} pasan, {n_fail} fallan y {n_pend} quedan pendientes. Un criterio pendiente no es un "
        "fallo: depende de una opción de la corrida (las pruebas automáticas y la corrida repetida que comprueba el determinismo) o de que se corra con los datos completos.",
        n_crit=r("n_criterios"), n_pass=r("n_pass"), n_fail=r("n_fail"), n_pend=r("n_pendiente"),
    )
    inf.parrafo(
        "Las Tareas 3 a 12 produjeron {n_tablas} tablas y {n_figuras} figuras, reunidas en {n_informes} informes. Cada cifra de los informes sale de una tabla y cada figura "
        "lleva su leyenda y su tabla de origen.",
        n_tablas=r("n_tablas"), n_figuras=r("n_figuras"), n_informes=r("n_informes"),
    )
    if int(resumen["n_fail"]) > 0:
        inf.texto_libre_sin_cifras("Hay criterios que fallan: el detalle de cada uno está en la tabla y la corrida termina con un código de salida distinto de cero.")
    if ctx.etiqueta == "extracto":
        inf.texto_libre_sin_cifras(
            "Con extractos el criterio de los datos completos queda pendiente: las conclusiones son indicativas y se repiten con los datos completos antes de decidir."
        )
    inf.tabla(
        "verificacion_dt.csv", df=ctx.leer_tabla("verificacion_dt.csv", columnas_texto=list(verificadores.COLUMNAS_VERIFICACION)),
        columnas=["id", "criterio", "estado", "detalle"], encabezados={"id": "Criterio", "criterio": "Descripción", "estado": "Estado", "detalle": "Detalle"},
        leyenda="Resultado de los criterios de terminado",
    )


def componer_cierre(ctx: Contexto, verificacion: Sequence[verificadores.ResultadoDT] | None = None) -> Path:
    """Compone y guarda ``t26_cierre.md``: el resultado de la verificación (sección 1) y los cambios al protocolo, las limitaciones y lo que falta con los datos completos.

    Con ``verificacion`` escribe antes ``verificacion_dt.csv`` y ``t26_resumen_cierre.csv``; sin ella usa los que ya están en la carpeta de tablas.
    """
    if verificacion is not None:
        verificadores.escribir_verificacion(ctx, verificacion)
    inf = Informe(ctx, INFORME_CIERRE, TITULO_CIERRE)
    inf.texto_libre_sin_cifras(
        "Este informe cierra la Fase 2: resume el resultado de los criterios de terminado, los cambios que se hicieron al protocolo después de ver resultados, las limitaciones "
        "y lo que falta con los datos completos."
    )
    _agregar_bloques(inf, ctx, (Bloque(seccion_verificacion), Bloque(cierre.agregar_secciones)))
    _validar_numeracion(inf)
    return inf.guardar()


# ============================================================================================================================ chequeo de figuras
def figuras_sin_incrustar(ctx: Contexto) -> list[str]:
    """Identificadores del índice de figuras que ningún informe incrusta con su PNG, su leyenda (``Figura Fxx_yy``) y su CSV de origen, ordenados."""
    textos = [p.read_text(encoding="utf8") for p in sorted(ctx.dir_informes.glob("*.md"))] if ctx.dir_informes.is_dir() else []
    faltan = []
    for f in ctx.figuras.indice().iter_rows(named=True):
        enlace = f"](../figuras/{f['archivo_png']})"
        inicio = f"*Figura {f['id_figura']}. "
        origen = f"`tablas/{f['csv']}`"
        incrustada = any(
            enlace in t and any(linea.startswith(inicio) and origen in linea for linea in t.splitlines())
            for t in textos
        )
        if not incrustada:
            faltan.append(str(f["id_figura"]))
    return sorted(faltan)


# FIN-informes_finales
