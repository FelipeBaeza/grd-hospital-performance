"""Orquestador del runner de la Fase 2 (Tarea 13): argumentos, limpieza protegida de la carpeta de salidas y orden de ejecución de las Tareas 3 a 12.

``run_fase2_eda.py`` solo llama a ``main``. Aquí vive la lógica, para poder importarla y probarla: ``analizar_argumentos`` (la línea de comandos), ``limpiar_salida`` (borra solo
``<base>/<fuente>`` y solo lo que una corrida produce), ``ejecutar_corrida`` (valida el protocolo, ejecuta T3 a T12, compone los informes, corre los verificadores DT01 a DT12, escribe
el manifiesto y sella el resultado) e ``imprimir_resumen``. Códigos de salida: 0 si ningún criterio falla (un criterio PENDIENTE no es un fallo), 1 si alguno falla (incluido el
sello final) y 2 por un error de uso o de protocolo.

El panel se construye UNA vez para las Tareas 10 y 11, y se libera junto con los resultados pesados antes de los verificadores. ``--solo`` ejecuta solo las tareas pedidas sobre la
carpeta actual, sin limpiarla, sin verificadores ni manifiesto: antes de correr carga los registros que ya había (tablas, métricas, cifras, figuras y accesos) para fusionarlos con los
nuevos en vez de pisarlos.
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import polars as pl

from . import (
    asociacion, calidad, calidad_catalogos, catalogos, cierre, cohorte, comorbilidad, config, descripcion, ejecucion, hospitales, informes_finales, ledger, lector, protocolo,
    recoleccion, rutas, seleccion, verificadores,
)
from . import panel as modulo_panel
from .contexto import Contexto
from .graficos import _Entrada

TAREAS = tuple(f"T{n}" for n in range(3, 13))
NOMBRES_TAREA = {
    "T3": "recolección inicial", "T4": "descripción de los datos", "T5": "calidad: fechas, identificadores y duplicados", "T6": "calidad: catálogos y consistencia",
    "T7": "cohorte, resultado y estadía", "T8": "comorbilidad", "T9": "hospitales", "T10": "asociación entre variables", "T11": "selección de variables",
    "T12": "evidencia para decidir",
}
ENTRADAS_DE_SALIDA = ("tablas", "figuras", "informes", "manifiesto.json")
"""Lo único que una corrida escribe dentro de ``<base>/<fuente>``: es lo único que la limpieza borra."""
PROTEGIDAS_DEL_PROYECTO = (".git", "docs", "data seminario", ".agents")
PROTEGIDAS_DE_LA_SUITE = ("config", "eda", "tests", ".agents")
SELLO = ("DT02", "DT04", "DT08")
"""Criterios que se vuelven a evaluar sobre el árbol final, cuando ya existen t26_cierre.md y el manifiesto."""
NOMBRE_SCRIPT = "run_fase2_eda.py"
TAREAS_DEL_PANEL = ("T10", "T11")
MENSAJES_ARGPARSE = (
    (r"^argument ", "argumento "), (r"expected one argument", "se esperaba un valor"), (r"unrecognized arguments: ", "argumentos no reconocidos: "),
    (r"ambiguous option: (.*) could match", r"opción ambigua: \1 puede ser"),
)


class ErrorDeProtocolo(Exception):
    """El protocolo congelado o las cifras documentadas no son válidos: la corrida no empieza (código de salida 2)."""


class SalidaProtegida(Exception):
    """La carpeta de salidas pedida no se puede limpiar sin riesgo (código de salida 2)."""


# ============================================================================================================================ argumentos
@dataclass(frozen=True)
class Parametros:
    """Lo que pide la línea de comandos, ya validado."""

    fuente: str = "extracto"
    data_root: Path | None = None
    archivos: Mapping[int, Path] = field(default_factory=dict)
    salida_base: Path = rutas.DIR_SALIDAS
    sin_pytest: bool = False
    sin_determinismo: bool = False
    solo: tuple[str, ...] | None = None


class _Formato(argparse.HelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        super().add_usage(usage, actions, groups, "uso: " if prefix is None else prefix)


class _Parser(argparse.ArgumentParser):
    """``argparse`` con los mensajes de uso en español y código de salida 2."""

    def error(self, message: str) -> None:  # type: ignore[override]
        for patron, reemplazo in MENSAJES_ARGPARSE:
            message = re.sub(patron, reemplazo, message)
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error de uso: {message}\n")


def anios_validos() -> tuple[int, ...]:
    return tuple(int(a) for a in config.cargar_fuentes_toml()["meta"]["anios"])


def _tipo_fuente(valor: str) -> str:
    if valor not in rutas.FUENTES_VALIDAS:
        raise argparse.ArgumentTypeError(f"debe ser {' o '.join(rutas.FUENTES_VALIDAS)}, no {valor!r}")
    return valor


def _tipo_archivo(valor: str) -> tuple[int, Path]:
    anio_txt, separador, ruta = valor.partition("=")
    if not separador or not anio_txt.strip().isdigit() or not ruta.strip():
        raise argparse.ArgumentTypeError(f"debe tener la forma ANIO=RUTA, no {valor!r}")
    anio, validos = int(anio_txt), anios_validos()
    if anio not in validos:
        raise argparse.ArgumentTypeError(f"el año {anio} no está entre {min(validos)} y {max(validos)}")
    destino = Path(ruta.strip()).expanduser()
    if not destino.is_file():
        raise argparse.ArgumentTypeError(f"el archivo del año {anio} no existe: {ruta.strip()}")
    return anio, destino


def normalizar_solo(texto: str) -> tuple[str, ...]:
    """Las tareas de ``--solo`` (``T3,T4``, ``t03``...) en el orden canónico y sin repetir. T13 es el propio runner y no se pide."""
    pedidas: set[str] = set()
    for trozo in texto.split(","):
        try:
            tarea = informes_finales.normalizar_tarea(trozo)
        except ValueError:
            raise argparse.ArgumentTypeError(f"tarea desconocida {trozo.strip()!r}: use T3 a T12 separadas por comas") from None
        if tarea == "T13":
            raise argparse.ArgumentTypeError("T13 es el propio runner y no se pide con --solo")
        if tarea not in TAREAS:
            raise argparse.ArgumentTypeError(f"tarea desconocida {trozo.strip()!r}: use T3 a T12 separadas por comas")
        pedidas.add(tarea)
    return tuple(t for t in TAREAS if t in pedidas)


def _construir_parser() -> _Parser:
    p = _Parser(
        prog=NOMBRE_SCRIPT, add_help=False, formatter_class=_Formato,
        description="Corre la Fase 2 (comprensión de los datos) de la tesis: tareas 3 a 12, informes, verificadores y manifiesto.",
        epilog="Códigos de salida: 0 todo en orden, 1 algún criterio de terminado falla, 2 error de uso o de protocolo.",
    )
    p._optionals.title = "opciones"
    p.add_argument("-h", "--help", action="help", help="muestra esta ayuda y termina")
    p.add_argument("--fuente", type=_tipo_fuente, default="extracto", metavar="{extracto,completo}", help="origen de los datos (por defecto extracto)")
    p.add_argument("--data-root", type=Path, default=None, metavar="RUTA", help="carpeta con los archivos de datos; obligatoria con --fuente completo")
    p.add_argument("--archivo", type=_tipo_archivo, action="append", default=None, metavar="ANIO=RUTA", help="archivo explícito de un año (repetible); gana sobre fuentes.toml")
    p.add_argument("--salida-base", type=Path, default=rutas.DIR_SALIDAS, metavar="RUTA", help="carpeta de salidas (por defecto fase2_eda/salidas); se escribe en <RUTA>/<fuente>")
    p.add_argument("--sin-pytest", action="store_true", help="no corre la suite de pruebas: el criterio DT10 queda pendiente")
    p.add_argument("--sin-determinismo", action="store_true", help="no repite la corrida para comparar huellas: el criterio DT11 queda pendiente")
    p.add_argument("--solo", type=normalizar_solo, default=None, metavar="T3,T4,...", help="corre solo esas tareas (T3 a T12) sobre la carpeta actual, sin limpiarla ni verificar")
    return p


def analizar_argumentos(argv: Sequence[str] | None = None) -> Parametros:
    """Lee y valida la línea de comandos. Un error de uso imprime el mensaje en español y sale con código 2 (``SystemExit``)."""
    parser = _construir_parser()
    a = parser.parse_args(argv)
    if a.fuente == "completo" and a.data_root is None:
        parser.error("con --fuente completo hace falta --data-root con la carpeta de los datos completos")
    if a.data_root is not None and not a.data_root.is_dir():
        parser.error(f"la carpeta de --data-root no existe: {a.data_root}")
    archivos: dict[int, Path] = {}
    for anio, ruta in a.archivo or []:
        if anio in archivos:
            parser.error(f"el año {anio} aparece más de una vez en --archivo")
        archivos[anio] = ruta
    return Parametros(
        fuente=a.fuente, data_root=a.data_root, archivos=archivos, salida_base=a.salida_base, sin_pytest=a.sin_pytest, sin_determinismo=a.sin_determinismo, solo=a.solo
    )


# ============================================================================================================================ limpieza protegida
def _es_ancestro_o_igual(ancestro: Path, ruta: Path) -> bool:
    return ancestro == ruta or ancestro in ruta.parents


def limpiar_salida(base: Path | str, fuente: str) -> Path:
    """Vacía ``<base>/<fuente>`` y recrea sus subcarpetas; devuelve esa carpeta. NUNCA toca otra cosa.

    Se rechaza con ``SalidaProtegida`` (y no se borra nada) si la fuente no es válida, si el destino es un enlace simbólico, si ``base`` es la raíz del sistema, el directorio
    personal, la raíz del proyecto, la raíz de la suite o un ancestro de ellos, si el destino coincide con, está dentro de o contiene una carpeta protegida (``.git``, ``docs``,
    ``data seminario``, ``config``, ``eda``, ``tests`` o ``.agents``), o si trae algo distinto de ``tablas``, ``figuras``, ``informes`` y ``manifiesto.json``. Solo esas cuatro
    entradas se borran, de modo que ``salidas/completo`` no se toca al limpiar ``extracto`` ni al revés.
    """
    if fuente not in rutas.FUENTES_VALIDAS:
        raise SalidaProtegida(f"la fuente {fuente!r} no es válida: debe ser {' o '.join(rutas.FUENTES_VALIDAS)}")
    base = Path(base)
    destino = base / fuente
    if destino.is_symlink():
        raise SalidaProtegida(f"la carpeta de salidas {destino.name!r} es un enlace simbólico: no se limpia")
    base_res, destino_res = base.resolve(), destino.resolve()
    if destino_res.parent != base_res:
        raise SalidaProtegida("la carpeta de salidas no está directamente dentro de la base pedida: no se limpia")
    proyecto, suite = rutas.RAIZ_PROYECTO.resolve(), rutas.RAIZ_EDA.resolve()
    for sensible, nombre in ((Path(base_res.anchor), "la raíz del sistema"), (Path.home().resolve(), "el directorio personal"), (proyecto, "la raíz del proyecto"), (suite, "la raíz de la suite")):
        if base_res == sensible or _es_ancestro_o_igual(base_res, sensible):
            raise SalidaProtegida(f"la base de salidas es {nombre} o un ancestro suyo: no se limpia")
    protegidas = [proyecto / n for n in PROTEGIDAS_DEL_PROYECTO] + [suite / n for n in PROTEGIDAS_DE_LA_SUITE]
    for prot in protegidas:
        if _es_ancestro_o_igual(prot, destino_res) or _es_ancestro_o_igual(destino_res, prot):
            raise SalidaProtegida(f"la carpeta de salidas coincide con, contiene o está dentro de una carpeta protegida ({prot.name}): no se limpia")
    if ".git" in destino_res.parts:
        raise SalidaProtegida("la carpeta de salidas está dentro de un repositorio git: no se limpia")
    if destino.exists():
        if not destino.is_dir():
            raise SalidaProtegida(f"{destino.name!r} existe y no es una carpeta: no se limpia")
        ajenas = sorted(p.name for p in destino.iterdir() if p.name not in ENTRADAS_DE_SALIDA)
        if ajenas:
            raise SalidaProtegida("la carpeta de salidas trae archivos ajenos a una corrida (" + ", ".join(ajenas[:5]) + "): no se borra nada")
        enlaces = sorted(p.name for p in destino.iterdir() if p.is_symlink())
        if enlaces:
            raise SalidaProtegida("la carpeta de salidas trae enlaces simbólicos (" + ", ".join(enlaces) + "): no se borra nada")
        for nombre in ENTRADAS_DE_SALIDA:
            entrada = destino / nombre
            if entrada.is_dir():
                _borrar_arbol(entrada)
            elif entrada.exists():
                entrada.unlink()
    destino.mkdir(parents=True, exist_ok=True)
    rutas.subdirs_salida(destino)
    return destino


def _borrar_arbol(carpeta: Path) -> None:
    """Borra una carpeta sin seguir enlaces simbólicos (un enlace dentro se borra como enlace, no su destino)."""
    import shutil

    shutil.rmtree(carpeta)


# ============================================================================================================================ registros de una corrida anterior (--solo)
def _numero(texto: str | None, tipo: Callable[[str], Any] = float) -> Any:
    return None if texto is None or texto == "" else tipo(texto)


def _etiqueta_tarea(tarea: str) -> str:
    """``T5`` es ``T05`` en el registro de accesos (las tareas escriben su etiqueta con dos dígitos)."""
    n = int(tarea[1:])
    return f"T{n:02d}"


def hidratar_registros(ctx: Contexto, tareas_a_reemplazar: Iterable[str] = ()) -> None:
    """Carga en ``ctx`` los registros que una corrida anterior dejó en la carpeta: el índice de tablas, las métricas, las cifras de los informes, el índice de figuras y el registro
    de accesos. Así una corrida parcial (``--solo``) fusiona sus registros con los anteriores en vez de reemplazarlos por los suyos.

    Del registro de accesos se descartan las filas de las tareas que se van a volver a ejecutar (``tareas_a_reemplazar``, como ``T5``); las demás se conservan y se renumeran.
    """
    t = ctx.dir_tablas

    def leer(nombre: str) -> pl.DataFrame | None:
        return config.leer_csv_texto(t / nombre) if (t / nombre).is_file() else None

    indice = leer("tablas_indice.csv")
    if indice is not None:
        for r in indice.iter_rows(named=True):
            ctx._tablas[r["nombre"]] = {
                "nombre": r["nombre"], "tarea": r["tarea"], "descripcion": r["descripcion"], "filas": int(r["filas"]), "columnas_conteo": r["columnas_conteo"],
                "celdas_suprimidas": int(r["celdas_suprimidas"]), "fuente": r["fuente"], "sha256": r["sha256"],
            }
    metricas = leer("metricas_eda.csv")
    if metricas is not None:
        ctx._metricas = [
            {
                "clave": r["clave"], "valor": _numero(r["valor"]), "ic_bajo": _numero(r["ic_bajo"]), "ic_alto": _numero(r["ic_alto"]), "tarea": r["tarea"], "tabla": r["tabla"],
                "nivel": r["nivel"], "alcance": r["alcance"], "nota": r["nota"],
            }
            for r in metricas.iter_rows(named=True)
        ]
    cifras = leer("cifras_informe.csv")
    if cifras is not None:
        ctx._cifras_informe = [dict(r) for r in cifras.iter_rows(named=True)]
    figuras = leer("figuras_indice.csv")
    if figuras is not None:
        ctx.figuras._entradas = {r["id_figura"]: _Entrada(**r) for r in figuras.iter_rows(named=True)}
    accesos = leer("registro_accesos.csv")
    if accesos is not None:
        reemplazar = {_etiqueta_tarea(x) for x in tareas_a_reemplazar}
        conservados = [r for r in accesos.iter_rows(named=True) if r["tarea"] not in reemplazar]
        ctx.cortafuegos._entradas = [
            {
                "secuencia": i, "tarea": r["tarea"], "anio": int(r["anio"]), "proposito": r["proposito"], "n_columnas": int(r["n_columnas"]),
                "columnas_resultado": r["columnas_resultado"], "reglas": r["reglas"], "decision": r["decision"], "motivo": r["motivo"],
            }
            for i, r in enumerate(conservados, start=1)
        ]


# ============================================================================================================================ corrida
@dataclass
class ResultadoCorrida:
    """Lo que dejó una corrida: el código de salida, los pasos con su tiempo (solo para la consola), los criterios y el sello."""

    codigo: int
    parametros: Parametros
    dir_salida: Path | None = None
    pasos: list[tuple[str, float]] = field(default_factory=list)
    resultados: list[verificadores.ResultadoDT] = field(default_factory=list)
    sello: list[verificadores.ResultadoDT] = field(default_factory=list)
    tareas: tuple[str, ...] = ()
    parcial: bool = False


def validar_protocolo() -> list[str]:
    """Problemas del protocolo congelado (umbrales, hipótesis, plan de figuras, definición de terminado) y de las cifras documentadas."""
    problemas = [f"{archivo}: {p}" for archivo, lista in protocolo.validar_todo().items() for p in lista]
    problemas += [f"cifras_documentadas.csv: {p}" for p in ledger.validar_cifras(ledger.cargar_cifras())]
    return problemas


def _regimenes_de_t5(ctx: Contexto, r5: Mapping[str, Any] | None) -> list[list[int]] | None:
    """Los espacios de identificadores de la Tarea 5 como listas de años (la columna ``anios`` es texto: ``2019, 2020``). Sin la Tarea 5 en esta corrida, los de su tabla publicada."""
    if r5 is not None:
        textos = r5["regimenes"]["anios"].to_list()
    elif (ctx.dir_tablas / "t24_id_regimenes.csv").is_file():
        textos = ctx.leer_tabla("t24_id_regimenes.csv", columnas_texto=["anios"])["anios"].to_list()
    else:
        return None
    return [[int(a) for a in s.split(",")] for s in textos]


def lanzador_corrida(p: Parametros, script: Path | None = None, timeout: int = verificadores.TIMEOUT_SUBPROCESO_S) -> Callable[[Path], int]:
    """El lanzador de DT11: corre este mismo runner en un subproceso con los mismos datos, ``--sin-pytest --sin-determinismo`` y otra carpeta de salidas, y devuelve su código de salida.

    Cada subproceso tiene su propia semilla de hash de Python (no se fija ``PYTHONHASHSEED``): así se detecta un orden que depende de ella.
    """
    script = script or (rutas.RAIZ_EDA / NOMBRE_SCRIPT)

    def lanzar(base: Path) -> int:
        cmd = [sys.executable, "-B", str(script), "--fuente", p.fuente, "--salida-base", str(base), "--sin-pytest", "--sin-determinismo"]
        if p.data_root is not None:
            cmd += ["--data-root", str(p.data_root)]
        for anio, ruta in sorted(p.archivos.items()):
            cmd += ["--archivo", f"{anio}={ruta}"]
        entorno = dict(os.environ)
        entorno["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(cmd, cwd=rutas.RAIZ_EDA, env=entorno, capture_output=True, text=True, timeout=timeout)
        if proc.returncode not in (0, 1):
            print("\n".join(proc.stderr.splitlines()[-15:]), file=sys.stderr)
        return proc.returncode

    return lanzar


class _Registro:
    """Imprime cada paso al terminar (con su tiempo, solo en la consola) y los guarda."""

    def __init__(self, salida: TextIO) -> None:
        self.salida = salida
        self.pasos: list[tuple[str, float]] = []

    def __call__(self, nombre: str, funcion: Callable[[], Any]) -> Any:
        inicio = time.perf_counter()
        r = funcion()
        duracion = time.perf_counter() - inicio
        self.pasos.append((nombre, duracion))
        print(f"  {nombre}: {duracion:.1f} s".replace(".", ","), file=self.salida, flush=True)
        return r


def ejecutar_corrida(p: Parametros, *, raiz_proyecto: Path | None = None, salida: TextIO | None = None) -> ResultadoCorrida:
    """Ejecuta la corrida pedida y devuelve su resultado. ``raiz_proyecto`` (solo para las pruebas) es la carpeta de los documentos de procedencia y de los insumos del manifiesto."""
    salida = salida or sys.stdout
    parcial = p.solo is not None
    activas = tuple(p.solo) if parcial else TAREAS
    paso = _Registro(salida)

    problemas = validar_protocolo()
    if problemas:
        raise ErrorDeProtocolo("el protocolo no es válido: " + "; ".join(problemas[:5]))
    fuente = lector.resolver_fuente(p.fuente, data_root=p.data_root, archivos_explicitos=dict(p.archivos) or None)
    print(f"Fase 2, fuente {p.fuente}" + (f", corrida parcial: {', '.join(activas)}" if parcial else ""), file=salida, flush=True)
    if not parcial:
        limpiar_salida(p.salida_base, p.fuente)
    ctx = Contexto.crear(p.fuente, dir_salida_base=p.salida_base, fuente=fuente)
    cat = paso("catálogos", catalogos.cargar_catalogos)
    if parcial:
        reemplazar = list(activas) + (list(TAREAS_DEL_PANEL) if set(activas) & set(TAREAS_DEL_PANEL) else [])
        hidratar_registros(ctx, reemplazar)

    r5: Mapping[str, Any] | None = None
    cascada = cascada_pipeline = None
    regimenes_t9: list[list[int]] | None = None
    panel = asoc = None

    def tarea(nombre: str, funcion: Callable[[], Any]) -> Any:
        return paso(f"{nombre} {NOMBRES_TAREA[nombre]}", funcion) if nombre in activas else None

    tarea("T3", lambda: recoleccion.ejecutar(ctx, raiz_proyecto=raiz_proyecto))
    tarea("T4", lambda: descripcion.ejecutar(ctx))
    r5 = tarea("T5", lambda: calidad.ejecutar(ctx))
    tarea("T6", lambda: calidad_catalogos.ejecutar(ctx, cat))
    r7 = tarea("T7", lambda: cohorte.ejecutar(ctx, cat))
    if r7 is not None:
        cascada, cascada_pipeline = r7["cascada"], r7["cascada_pipeline"]
        del r7
    tarea("T8", lambda: comorbilidad.ejecutar(ctx, cat))
    r9 = tarea("T9", lambda: hospitales.ejecutar(ctx, cat, regimenes=_regimenes_de_t5(ctx, r5)))
    if r9 is not None:
        regimenes_t9 = r9["regimenes"]
        del r9
    r5 = None
    if set(activas) & set(TAREAS_DEL_PANEL):
        panel = paso(
            "panel de las Tareas 10 y 11",
            lambda: modulo_panel.construir_panel(ctx, cat, regimenes=regimenes_t9 if ctx.etiqueta == "completo" else None, tarea="T10"),
        )
    asoc = tarea("T10", lambda: asociacion.ejecutar(ctx, panel, cat))
    tarea("T11", lambda: seleccion.ejecutar(ctx, panel, asoc, cat))
    panel = asoc = None
    gc.collect()
    tarea("T12", lambda: cierre.ejecutar(ctx, cascada=cascada))
    gc.collect()

    paso("informes", lambda: informes_finales.componer_informes(ctx, activas if parcial else None))
    paso("registros", ctx.escribir_registros)
    if parcial:
        return ResultadoCorrida(codigo=0, parametros=p, dir_salida=ctx.dir_salida, pasos=paso.pasos, tareas=activas, parcial=True)

    previos = tuple(n for n in informes_finales.informes_esperados() if n != informes_finales.INFORME_CIERRE)
    banderas = verificadores.Banderas(
        sin_pytest=p.sin_pytest, sin_determinismo=p.sin_determinismo, cascada=cascada, cascada_pipeline=cascada_pipeline, informes_esperados=previos,
        lanzador_determinismo=lanzador_corrida(p),
    )
    resultados = paso("criterios de terminado", lambda: verificadores.verificar_todo(ctx, banderas))
    cascada = cascada_pipeline = None
    paso("informe de cierre", lambda: informes_finales.componer_cierre(ctx, resultados))
    paso("registros finales", ctx.escribir_registros)
    resumen = verificadores.resumen_verificacion(resultados)
    paso(
        "manifiesto",
        lambda: ejecucion.escribir_manifiesto_corrida(
            ctx, extras={"tareas": f"{TAREAS[0]} a {TAREAS[-1]}", "criterios": {"fail": resumen["n_fail"], "pass": resumen["n_pass"], "pendiente": resumen["n_pendiente"]}},
            raiz_proyecto=raiz_proyecto,
        ),
    )
    sello = paso("sello", lambda: verificadores.verificar_todo(ctx, verificadores.Banderas(informes_esperados=informes_finales.informes_esperados()), ids=SELLO))
    falla = any(r.estado == "FAIL" for r in resultados) or any(r.estado == "FAIL" for r in sello)
    return ResultadoCorrida(
        codigo=1 if falla else 0, parametros=p, dir_salida=ctx.dir_salida, pasos=paso.pasos, resultados=resultados, sello=sello, tareas=activas, parcial=False
    )


def linea_resultado(resultados: Sequence[verificadores.ResultadoDT]) -> str:
    r = verificadores.resumen_verificacion(resultados)
    return f"Resultado: {r['n_pass']} PASS, {r['n_pendiente']} PENDIENTE, {r['n_fail']} FAIL"


def imprimir_resumen(res: ResultadoCorrida, salida: TextIO | None = None) -> None:
    """Resumen en español: qué se ejecutó y, en una corrida completa, la tabla de los criterios, el sello y la línea de resultado."""
    salida = salida or sys.stdout
    total = sum(d for _, d in res.pasos)
    if res.parcial:
        print(f"\nCorrida parcial ({', '.join(res.tareas)}): no se limpió la carpeta, no se corrieron los criterios de terminado ni se escribió el manifiesto.", file=salida)
        print(f"Tiempo total: {total:.1f} s".replace(".", ","), file=salida, flush=True)
        return
    print("\nCriterios de terminado", file=salida)
    for r in res.resultados:
        print(f"  {r.id} {r.estado:<10} {r.criterio}. {r.detalle}", file=salida)
    fallan_sello = [r for r in res.sello if r.estado == "FAIL"]
    if fallan_sello:
        print("\nEl sello final sobre el árbol de salidas encontró fallos:", file=salida)
        for r in fallan_sello:
            print(f"  {r.id} {r.estado:<10} {r.criterio}. {r.detalle}", file=salida)
    else:
        print(f"\nSello final sobre el árbol de salidas: {', '.join(r.id for r in res.sello)} sin fallos.", file=salida)
    print(f"Tiempo total: {total:.1f} s".replace(".", ","), file=salida)
    print(linea_resultado(res.resultados), file=salida, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada de ``run_fase2_eda.py``. Devuelve el código de salida."""
    parametros = analizar_argumentos(argv)
    try:
        resultado = ejecutar_corrida(parametros)
    except (ErrorDeProtocolo, SalidaProtegida, lector.ArchivoNoEncontrado) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    imprimir_resumen(resultado)
    return resultado.codigo


# FIN-orquestador
