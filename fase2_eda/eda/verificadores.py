"""Verificadores de la definición de terminado de la Fase 2 (DT01 a DT12, Tarea 13).

Hay una función por cada valor de la columna ``verificador`` de ``config/definicion_terminado.csv``, con ese nombre exacto. Cada una recibe el ``Contexto`` y, solo para poder
probarla, insumos opcionales (tablas, listas o un lanzador) con los que se la hace fallar a propósito sin tocar archivos reales. Devuelve un ``ResultadoDT`` con el estado
``PASS``, ``FAIL`` o ``PENDIENTE`` (un criterio que la corrida no pidió o que no corresponde a la fuente: no es un fallo).

El ``detalle`` no lleva tiempos ni rutas absolutas, de modo que dos corridas iguales producen el mismo ``verificacion_dt.csv`` (criterio DT11). Los verificadores no leen
datos: solo las tablas, los informes y los registros que la corrida dejó en la carpeta de salidas, más los archivos de configuración y de las fases anteriores.

``verificar_todo`` recorre las filas del CSV en su orden y entrega una ``ResultadoDT`` por criterio; ``escribir_verificacion`` publica ``verificacion_dt.csv`` y
``t26_resumen_cierre.csv`` (tarea ``T13``), de donde sale el informe de cierre.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from . import config, cortafuegos, decision, decisiones, ledger, manifiesto, privacidad, protocolo, rutas
from .contexto import Contexto
from .informes import MARCADOR_TABLA_DERIVADA, Cifra, CifraSinRespaldo, fmt_entero, leer_cifra

TAREA = "T13"
ESTADOS = ("PASS", "FAIL", "PENDIENTE")
COLUMNAS_VERIFICACION = ("id", "criterio", "estado", "detalle", "evidencia")
ESQUEMA_VERIFICACION = {c: pl.String for c in COLUMNAS_VERIFICACION}
ESQUEMA_RESUMEN_CIERRE = {
    "n_criterios": pl.Int64, "n_pass": pl.Int64, "n_fail": pl.Int64, "n_pendiente": pl.Int64, "n_tablas": pl.Int64, "n_figuras": pl.Int64, "n_informes": pl.Int64,
}
VAR_DENTRO_DE_PYTEST = "FASE2_EDA_DENTRO_DE_PYTEST"
"""Variable de entorno que el lanzador de DT10 pone en el subproceso de pytest: una corrida del runner dentro de esa suite no vuelve a lanzar pytest."""
EXCLUIR_DEL_COTEJO: tuple[str, ...] = ()
"""Rutas relativas (dentro de ``<base>/<fuente>``) que DT11 no compara. Va vacía: ningún archivo se excluye (si una corrida no es determinista se corrige la causa)."""
MAX_PROBLEMAS = 5
ESTADOS_DICCIONARIO = ("COINCIDE", "DISCREPANCIA", "NO EVALUABLE")
TAREAS_SIN_2024 = frozenset({"T10", "T11", "T12", "T13"})
TAREAS_SIN_LECTURAS = frozenset({"T12", "T13"})
TIMEOUT_SUBPROCESO_S = 3 * 3600


@dataclass(frozen=True)
class ResultadoDT:
    """Resultado de un criterio de terminado."""

    id: str
    criterio: str
    estado: str
    detalle: str
    evidencia: str = ""

    def __post_init__(self) -> None:
        if self.estado not in ESTADOS:
            raise ValueError(f"estado inválido {self.estado!r}: debe ser uno de {ESTADOS}")


@dataclass(frozen=True)
class Banderas:
    """Lo que el runner entrega a los verificadores; todo es opcional para poder probarlos por separado."""

    sin_pytest: bool = False
    sin_determinismo: bool = False
    cascada: pl.DataFrame | None = None
    cascada_pipeline: pl.DataFrame | None = None
    informes_esperados: tuple[str, ...] | None = None
    lanzador_pytest: Callable[[], tuple[int, str]] | None = None
    lanzador_determinismo: Callable[[Path], int] | None = None


# ============================================================================================================================ ayudantes
def _fila_dt(nombre: str) -> tuple[str, str]:
    """``(id, criterio)`` del verificador ``nombre`` según ``config/definicion_terminado.csv``."""
    t = config.cargar_definicion_terminado().filter(pl.col("verificador") == nombre)
    if t.height != 1:
        raise KeyError(f"definicion_terminado.csv debe tener exactamente una fila para el verificador {nombre!r} (tiene {t.height})")
    return str(t["id"][0]), str(t["criterio"][0])


def _resultado(nombre: str, estado: str, detalle: str, evidencia: str = "") -> ResultadoDT:
    id_, criterio = _fila_dt(nombre)
    return ResultadoDT(id_, criterio, estado, detalle, evidencia)


def _lista(problemas: Iterable[str], maximo: int = MAX_PROBLEMAS) -> str:
    """Los problemas ordenados y sin repetir; después de ``maximo`` solo se cuenta cuántos más hay."""
    unicos = sorted(set(problemas))
    if len(unicos) <= maximo:
        return "; ".join(unicos)
    return "; ".join(unicos[:maximo]) + f"; y {len(unicos) - maximo} más"


def _texto(df: pl.DataFrame, columna: str) -> list[str]:
    return [("" if v is None else str(v).strip()) for v in df[columna].to_list()]


def _leer_tabla(ctx: Contexto, nombre: str, columnas_texto: Sequence[str] = ()) -> pl.DataFrame:
    """Una tabla publicada del contexto; si falta, el mensaje la nombra (``FileNotFoundError``)."""
    return ctx.leer_tabla(nombre, columnas_texto=list(columnas_texto))


def _leer_todo_texto(ctx: Contexto, nombre: str) -> pl.DataFrame:
    """Una tabla publicada con todas las columnas como texto (los vacíos quedan como cadena vacía)."""
    ruta = ctx.dir_tablas / nombre
    if not ruta.is_file():
        raise FileNotFoundError(f"no existe la tabla {nombre}")
    return config.leer_csv_texto(ruta)


# ============================================================================================================================ DT01
def decisiones_cubiertas(
    ctx: Contexto, *, hipotesis: pl.DataFrame | None = None, decisiones_fase1: pl.DataFrame | None = None, tabla_evidencia: pl.DataFrame | None = None
) -> ResultadoDT:
    """DT01: las 17 decisiones tienen hipótesis (o son institucionales o no dependen de datos) y ``decisiones_evidencia.csv`` las trae todas, una vez cada una."""
    hip = config.cargar_hipotesis() if hipotesis is None else hipotesis
    dec = config.cargar_decisiones_fase1() if decisiones_fase1 is None else decisiones_fase1
    ev = _leer_tabla(ctx, "decisiones_evidencia.csv", decisiones.COLUMNAS_TEXTO_DECISIONES) if tabla_evidencia is None else tabla_evidencia
    problemas = list(protocolo.validar_hipotesis(hip, dec))
    esperadas = sorted(int(x) for x in dec["n"].to_list())
    halladas = [int(x) for x in ev["n"].to_list()]
    if sorted(halladas) != esperadas:
        faltan = sorted(set(esperadas) - set(halladas))
        sobran = sorted(set(halladas) - set(esperadas))
        repetidas = sorted({n for n in halladas if halladas.count(n) > 1})
        partes = []
        if faltan:
            partes.append("faltan las decisiones " + ", ".join(str(n) for n in faltan))
        if sobran:
            partes.append("sobran las decisiones " + ", ".join(str(n) for n in sobran))
        if repetidas:
            partes.append("repetidas: " + ", ".join(str(n) for n in repetidas))
        problemas.append("decisiones_evidencia.csv no coincide con decisiones_pendientes.csv (" + "; ".join(partes) + ")")
    hipotesis_de = {int(r["n"]): str(r["hipotesis"] or "").strip() for r in ev.iter_rows(named=True)}
    tipo_de = {int(r["n"]): str(r["tipo_evidencia"]) for r in ev.iter_rows(named=True)}
    for n in esperadas:
        if not hipotesis_de.get(n) and tipo_de.get(n) not in ("institucional", "no depende de datos"):
            problemas.append(f"la decisión {n} no tiene ninguna hipótesis en la tabla de evidencia ni es institucional o independiente de los datos")
    ids = hip["id"].to_list()
    n_h, n_s = sum(1 for i in ids if str(i).startswith("H")), sum(1 for i in ids if str(i).startswith("S"))
    evidencia = "tablas/decisiones_evidencia.csv; config/hipotesis.csv"
    if problemas:
        return _resultado("decisiones_cubiertas", "FAIL", _lista(problemas), evidencia)
    return _resultado("decisiones_cubiertas", "PASS", f"{len(esperadas)} de {len(esperadas)} decisiones cubiertas; el protocolo trae {n_h} hipótesis y {n_s} supuestos", evidencia)


# ============================================================================================================================ DT02
def _texto_cifra(ruta: Path, columna: str, filtro: Mapping[str, Any], formato: str, esperado: str) -> str:
    """Vuelve a leer una cifra de su CSV y la formatea; devuelve el texto que coincide con ``esperado`` o, si ninguna lectura coincide, la primera.

    El informe pudo leer la columna como texto (``columnas_texto``) o con el tipo inferido, y el CSV no dice cuál: se prueban las dos lecturas.
    """
    obtenido: str | None = None
    try:
        obtenido = leer_cifra(pl.read_csv(ruta, infer_schema_length=100000), ruta.name, columna, **filtro).texto(formato)
    except CifraSinRespaldo:
        pass
    if obtenido == esperado:
        return obtenido
    c = leer_cifra(pl.read_csv(ruta, infer_schema=False), ruta.name, columna, **filtro)
    valor = c.valor
    if valor is not None and formato != "texto":
        valor = float(valor)
    alterno = Cifra(c.nombre, valor, c.csv, c.columna, c.filtro).texto(formato)
    return alterno if alterno == esperado or obtenido is None else obtenido


def cifras_con_csv(ctx: Contexto, *, cifras: pl.DataFrame | None = None, informes_esperados: Sequence[str] | None = None) -> ResultadoDT:
    """DT02: toda cifra de todos los informes tiene CSV y columna, el CSV existe con esa columna y la cifra se reproduce; no hay llaves sobrantes ni informes de más o de menos."""
    from . import informes_finales

    if cifras is None:
        ruta = ctx.dir_tablas / "cifras_informe.csv"
        cifras = config.leer_csv_texto(ruta) if ruta.is_file() else ctx.cifras_informe()
    esperados = tuple(informes_esperados) if informes_esperados is not None else informes_finales.informes_esperados()
    problemas: list[str] = []
    if cifras.height == 0:
        problemas.append("el registro de cifras de los informes está vacío")
    encabezados: dict[str, list[str] | None] = {}

    def columnas_de(csv: str) -> list[str] | None:
        if csv not in encabezados:
            ruta = ctx.dir_tablas / csv
            encabezados[csv] = pl.read_csv(ruta, n_rows=0, infer_schema=False).columns if ruta.is_file() else None
        return encabezados[csv]

    n_escalares = n_derivadas = 0
    for r in cifras.iter_rows(named=True):
        csv, columna = str(r["csv"] or "").strip(), str(r["columna"] or "").strip()
        quien = f"{r['informe']} {r['marcador']}"
        if not csv or not columna:
            problemas.append(f"{quien}: sin CSV o sin columna de origen")
            continue
        cols = columnas_de(csv)
        if cols is None:
            problemas.append(f"{quien}: el CSV {csv} no existe en tablas")
            continue
        if r["marcador"] == MARCADOR_TABLA_DERIVADA:
            n_derivadas += 1                      # columnas calculadas a partir del CSV citado: basta con que el CSV exista
            continue
        faltan = [c for c in columna.split(";") if c not in cols]
        if faltan:
            problemas.append(f"{quien}: {csv} no tiene la columna {', '.join(faltan)}")
            continue
        if r["formato"] == "tabla":
            continue
        n_escalares += 1
        try:
            filtro = json.loads(r["filtro"] or "{}")
            obtenido = _texto_cifra(ctx.dir_tablas / csv, columna, filtro, r["formato"], r["texto"])
        except (CifraSinRespaldo, ValueError, KeyError) as e:
            problemas.append(f"{quien}: no se pudo volver a leer la cifra ({type(e).__name__})")
            continue
        if obtenido != r["texto"]:
            problemas.append(f"{quien}: el informe dice {r['texto']} y {csv} da {obtenido}")
    presentes = sorted(p.name for p in ctx.dir_informes.glob("*.md")) if ctx.dir_informes.is_dir() else []
    for nombre in presentes:
        texto = (ctx.dir_informes / nombre).read_text(encoding="utf8")
        if "{" in texto or "}" in texto:
            problemas.append(f"{nombre}: tiene llaves sobrantes")
    faltan, sobran = sorted(set(esperados) - set(presentes)), sorted(set(presentes) - set(esperados))
    if faltan:
        problemas.append("faltan los informes " + ", ".join(faltan))
    if sobran:
        problemas.append("sobran los informes " + ", ".join(sobran))
    evidencia = "tablas/cifras_informe.csv; informes/"
    if problemas:
        return _resultado("cifras_con_csv", "FAIL", _lista(problemas), evidencia)
    return _resultado(
        "cifras_con_csv", "PASS",
        f"{fmt_entero(cifras.height)} cifras de {len(presentes)} informes con su CSV de origen; {fmt_entero(n_escalares)} cifras escalares se reproducen desde su columna y "
        f"{fmt_entero(n_derivadas)} tablas derivadas tienen columnas calculadas desde su CSV; sin llaves sobrantes",
        evidencia,
    )


# ============================================================================================================================ DT03
def _tarea_normalizada(tarea: Any) -> str:
    """``T03`` y ``T3`` son la misma tarea."""
    m = re.fullmatch(r"\s*[Tt]0*(\d+)\s*", str(tarea or ""))
    return f"T{int(m.group(1))}" if m else str(tarea or "").strip().upper()


def _violaciones(cf: cortafuegos.Cortafuegos, registro: pl.DataFrame) -> int:
    """Violaciones de la política de un registro dado, con la misma regla del cortafuegos de la corrida."""
    clon = cortafuegos.Cortafuegos(
        anio_expuesto=cf.anio_expuesto, propositos_permitidos=cf.propositos_permitidos, columnas_resultado=cf.columnas_resultado, reglas_autorizadas=cf.reglas_autorizadas
    )
    clon._entradas = [dict(f) for f in registro.iter_rows(named=True)]
    return clon.violaciones()


def cero_lecturas_2024(ctx: Contexto, *, registro: pl.DataFrame | None = None) -> ResultadoDT:
    """DT03: el registro de accesos no tiene lecturas del año expuesto con columnas de resultado ni con propósitos no permitidos, ni filas de las Tareas 10 a 13 en ese año."""
    cf = ctx.cortafuegos
    reg = cf.registro() if registro is None else registro
    exp = int(cf.anio_expuesto)
    problemas: list[str] = []
    if reg.height:
        del_exp = reg.filter(pl.col("anio") == exp)
        tarea_n = [_tarea_normalizada(t) for t in reg["tarea"].to_list()]
        en_claro = del_exp.filter((pl.col("decision") == cortafuegos.PERMITIDO) & (pl.col("columnas_resultado").fill_null("") != ""))
        if en_claro.height:
            problemas.append(f"{en_claro.height} lecturas permitidas de columnas de resultado de {exp}")
        fuera = del_exp.filter(~pl.col("proposito").is_in(sorted(cf.propositos_permitidos)) & (pl.col("decision") != cortafuegos.BLOQUEADO))
        if fuera.height:
            problemas.append(f"{fuera.height} lecturas de {exp} con un propósito no permitido que no se bloquearon")
        prohibidas = sum(1 for t, a in zip(tarea_n, reg["anio"].to_list()) if a == exp and t in TAREAS_SIN_2024)
        if prohibidas:
            problemas.append(f"{prohibidas} accesos de {exp} de las Tareas 10 a 13, que nunca leen ese año")
        sin_datos = sum(1 for t in tarea_n if t in TAREAS_SIN_LECTURAS)
        if sin_datos:
            problemas.append(f"{sin_datos} accesos de las Tareas 12 y 13, que no leen datos")
        n_exp, n_bloq = del_exp.height, del_exp.filter(pl.col("decision") == cortafuegos.BLOQUEADO).height
        n_masc = del_exp.filter(pl.col("decision") == cortafuegos.ENMASCARADO).height
        n_regla = del_exp.filter(pl.col("decision") == cortafuegos.REGLA_CALIDAD).height
    else:
        n_exp = n_bloq = n_masc = n_regla = 0
    violaciones = _violaciones(cf, reg)
    if violaciones:
        problemas.append(f"el cortafuegos cuenta {violaciones} violaciones de la política")
    evidencia = "tablas/registro_accesos.csv"
    if problemas:
        return _resultado("cero_lecturas_2024", "FAIL", _lista(problemas), evidencia)
    return _resultado(
        "cero_lecturas_2024", "PASS",
        f"{fmt_entero(reg.height)} accesos registrados, {fmt_entero(n_exp)} del año {exp}: ninguna lectura de resultado y cero violaciones; bloqueos {n_bloq}, "
        f"enmascaradas {n_masc} y reglas de calidad autorizadas {n_regla}",
        evidencia,
    )


# ============================================================================================================================ DT04
def escaneo_privacidad(ctx: Contexto, *, escaneo: pl.DataFrame | None = None) -> ResultadoDT:
    """DT04: ni identificadores, ni fechas completas, ni celdas de 1 a 9 casos en tablas, informes y manifiesto (si ya existe)."""
    esc = privacidad.escanear_salidas(ctx.dir_salida, ctx.indice_tablas(), ctx.celda_minima) if escaneo is None else escaneo
    total = privacidad.total_violaciones(esc)
    evidencia = "tablas/; informes/; manifiesto.json"
    if total:
        malos = esc.filter(pl.col("n_violaciones") > 0)
        return _resultado("escaneo_privacidad", "FAIL", f"{total} violaciones: " + _lista(f"{r['archivo']} {r['chequeo']} {r['n_violaciones']}" for r in malos.iter_rows(named=True)), evidencia)
    n_archivos = esc["archivo"].n_unique() if esc.height else 0
    return _resultado(
        "escaneo_privacidad", "PASS",
        f"{fmt_entero(n_archivos)} archivos escaneados ({'tablas, informes y manifiesto' if 'manifiesto.json' in esc['archivo'].to_list() else 'tablas e informes'}) sin identificadores, "
        "fechas completas ni celdas de 1 a 9 casos",
        evidencia,
    )


# ============================================================================================================================ DT05
def _por_periodo(df: pl.DataFrame) -> dict[str, dict[str, int | None]]:
    out: dict[str, dict[str, int | None]] = {}
    for periodo, paso, n in df.select(["periodo", "paso", "n"]).iter_rows():
        out.setdefault(str(periodo), {})[str(paso)] = None if n is None else int(n)
    return out


def problemas_agregados(pasos_por_periodo: Mapping[str, Mapping[str, int | None]]) -> tuple[list[str], int]:
    """Los períodos agregados (``2019-2023``) deben sumar los años que abarcan, paso por paso. Devuelve los problemas y cuántas sumas se pudieron verificar."""
    problemas: list[str] = []
    verificadas = 0
    for periodo in sorted(pasos_por_periodo):
        m = re.fullmatch(r"(\d{4})-(\d{4})", periodo)
        if not m:
            continue
        anios = [str(a) for a in range(int(m.group(1)), int(m.group(2)) + 1)]
        if not all(a in pasos_por_periodo for a in anios):
            continue
        for paso, n in pasos_por_periodo[periodo].items():
            partes = [pasos_por_periodo[a].get(paso) for a in anios]
            if n is None or any(p is None for p in partes):
                continue
            verificadas += 1
            if n != sum(partes):
                problemas.append(f"{periodo}: {paso} es {fmt_entero(n)} y los años suman {fmt_entero(sum(partes))}")
    return problemas, verificadas


IDENTIDAD_PIPELINE = ("COHORTE_DURA_PIPELINE", "TOTAL", ("EX01", "EX02", "EX03_PIPELINE"))


def problemas_cascada_pipeline(df: pl.DataFrame) -> tuple[list[str], int]:
    """TOTAL menos EX01, EX02 y EX03_PIPELINE debe dar COHORTE_DURA_PIPELINE en cada período. Una identidad con alguna celda suprimida no se verifica."""
    resultado, base, restas = IDENTIDAD_PIPELINE
    problemas: list[str] = []
    verificadas = 0
    por_periodo = _por_periodo(df)
    for periodo in sorted(por_periodo):
        pasos = por_periodo[periodo]
        if any(pasos.get(p) is None for p in (resultado, base, *restas)):
            continue
        verificadas += 1
        esperado = int(pasos[base]) - sum(int(pasos[p]) for p in restas)
        if int(pasos[resultado]) != esperado:
            problemas.append(f"{periodo} (pipeline): {resultado} es {fmt_entero(pasos[resultado])} y {base} menos {' y '.join(restas)} da {fmt_entero(esperado)}")
    return problemas, verificadas


def cascada_cuadra(
    ctx: Contexto, *, cascada: pl.DataFrame | None = None, cascada_pipeline: pl.DataFrame | None = None, archivos: pl.DataFrame | None = None
) -> ResultadoDT:
    """DT05: la cascada de exclusiones cuadra con el total en cada año y en cada período, con los valores SIN suprimir que entrega la Tarea 7.

    Sin esos DataFrames se usan las tablas publicadas y solo se verifican las identidades sin celdas suprimidas.
    """
    casc = _leer_tabla(ctx, "t23_cascada.csv", ["periodo", "paso"]) if cascada is None else cascada
    pipe = _leer_tabla(ctx, "t23_cascada_pipeline.csv", ["periodo", "paso"]) if cascada_pipeline is None else cascada_pipeline
    arch = _leer_tabla(ctx, "t21_archivos.csv") if archivos is None else archivos
    problemas, n_secuenciales = decisiones.revisar_cascada(casc)
    pasos = _por_periodo(casc)
    leidas = {str(int(r["anio"])): int(r["filas_leidas"]) for r in arch.select(["anio", "filas_leidas"]).iter_rows(named=True) if r["filas_leidas"] is not None}
    n_totales = 0
    for anio, filas in sorted(leidas.items()):
        total = pasos.get(anio, {}).get("TOTAL")
        if total is None:
            continue
        n_totales += 1
        if int(total) != filas:
            problemas.append(f"{anio}: el total de la cascada es {fmt_entero(total)} y el archivo trae {fmt_entero(filas)} filas leídas")
    p_agregados, n_agregados = problemas_agregados(pasos)
    p_pipeline, n_pipeline = problemas_cascada_pipeline(pipe)
    problemas += p_agregados + p_pipeline
    evidencia = "tablas/t23_cascada.csv; tablas/t23_cascada_pipeline.csv; tablas/t21_archivos.csv"
    if problemas:
        return _resultado("cascada_cuadra", "FAIL", _lista(problemas), evidencia)
    return _resultado(
        "cascada_cuadra", "PASS",
        f"{fmt_entero(n_secuenciales)} identidades secuenciales, {fmt_entero(n_totales)} totales anuales contra las filas leídas, {fmt_entero(n_agregados)} sumas de períodos agregados y "
        f"{fmt_entero(n_pipeline)} identidades de la cascada del pipeline cuadran",
        evidencia,
    )


# ============================================================================================================================ DT06
def diccionario_conciliado(ctx: Contexto, *, medido: pl.DataFrame | None = None, diccionario: pl.DataFrame | None = None) -> ResultadoDT:
    """DT06: ``diccionario_medido.csv`` tiene una fila por cada variable del diccionario de la Fase 2 y un estado COINCIDE, DISCREPANCIA o NO EVALUABLE."""
    dic = config.leer_csv_texto(rutas.DIR_FASE2 / "Fase_2__Diccionario_de_Base_de_Datos.csv") if diccionario is None else diccionario
    med = _leer_todo_texto(ctx, "diccionario_medido.csv") if medido is None else medido
    variables = _texto(dic, "Variable") if "Variable" in dic.columns else _texto(dic, "columna")
    columnas = _texto(med, "columna")
    problemas: list[str] = []
    if len(columnas) != len(variables):
        problemas.append(f"el diccionario tiene {len(variables)} variables y la conciliación {len(columnas)} filas")
    repetidas = sorted({c for c in columnas if columnas.count(c) > 1})
    if repetidas:
        problemas.append("columnas repetidas: " + ", ".join(repetidas))
    faltan, sobran = sorted(set(variables) - set(columnas)), sorted(set(columnas) - set(variables))
    if faltan:
        problemas.append("faltan en la conciliación: " + ", ".join(faltan))
    if sobran:
        problemas.append("sobran en la conciliación: " + ", ".join(sobran))
    estados = _texto(med, "estado")
    ajenos = sorted({e for e in estados if e not in ESTADOS_DICCIONARIO})
    if ajenos:
        problemas.append("estados fuera de COINCIDE, DISCREPANCIA y NO EVALUABLE: " + ", ".join(ajenos))
    evidencia = "tablas/diccionario_medido.csv"
    if problemas:
        return _resultado("diccionario_conciliado", "FAIL", _lista(problemas), evidencia)
    cuenta = {e: estados.count(e) for e in ESTADOS_DICCIONARIO}
    return _resultado(
        "diccionario_conciliado", "PASS",
        f"{len(variables)} de {len(variables)} variables del diccionario conciliadas: {cuenta['COINCIDE']} coinciden, {cuenta['DISCREPANCIA']} discrepan y {cuenta['NO EVALUABLE']} no se evalúan",
        evidencia,
    )


# ============================================================================================================================ DT07
def ledger_completo(ctx: Contexto, *, tabla_ledger: pl.DataFrame | None = None, cifras: pl.DataFrame | None = None) -> ResultadoDT:
    """DT07: el ledger trae las cifras documentadas una vez cada una, con estado válido, motivo donde corresponde y valor medido con su tabla de origen en CONFIRMA y DISCREPA."""
    doc = ledger.cargar_cifras() if cifras is None else cifras
    led = _leer_tabla(ctx, "ledger_cifras.csv", ledger.COLUMNAS_TEXTO_LEDGER) if tabla_ledger is None else tabla_ledger
    ids_doc, ids_led = _texto(doc, "id"), _texto(led, "id")
    problemas: list[str] = []
    if len(ids_led) != len(ids_doc):
        problemas.append(f"hay {len(doc)} cifras documentadas y {len(led)} filas en el ledger")
    repetidos = sorted({i for i in ids_led if ids_led.count(i) > 1})
    if repetidos:
        problemas.append("ids repetidos: " + ", ".join(repetidos))
    faltan, sobran = sorted(set(ids_doc) - set(ids_led)), sorted(set(ids_led) - set(ids_doc))
    if faltan:
        problemas.append("faltan en el ledger: " + ", ".join(faltan))
    if sobran:
        problemas.append("sobran en el ledger: " + ", ".join(sobran))
    for r in led.iter_rows(named=True):
        id_, estado = r["id"], r["estado"]
        if estado not in ledger.ESTADOS:
            problemas.append(f"{id_}: estado inválido {estado!r}")
            continue
        if estado in ("DISCREPA", "NO VERIFICABLE") and not str(r["motivo"] or "").strip():
            problemas.append(f"{id_}: {estado} sin motivo")
        protegido = r["tipo"] == "absoluto" and r["unidad"] == "conteo" and (r["n_doc"] is None or r["n_medido"] is None)
        if estado in ("CONFIRMA", "DISCREPA") and not protegido:
            if r["valor_medido"] is None:
                problemas.append(f"{id_}: {estado} sin valor medido")
            origen = str(r["tabla_origen"] or "").strip()
            if not origen:
                problemas.append(f"{id_}: {estado} sin tabla de origen")
            elif not (ctx.dir_tablas / origen).is_file() and not (rutas.DIR_CONFIG / origen).is_file():
                problemas.append(f"{id_}: la tabla de origen {origen} no existe")
    evidencia = "tablas/ledger_cifras.csv; config/cifras_documentadas.csv"
    if problemas:
        return _resultado("ledger_completo", "FAIL", _lista(problemas), evidencia)
    estados = [str(e) for e in led["estado"].to_list()]
    return _resultado(
        "ledger_completo", "PASS",
        f"{len(ids_doc)} de {len(ids_doc)} cifras documentadas con estado: {estados.count('CONFIRMA')} confirman, {estados.count('DISCREPA')} discrepan y "
        f"{estados.count('NO VERIFICABLE')} no se pueden verificar, cada una con su motivo o su cifra medida",
        evidencia,
    )


# ============================================================================================================================ DT08
def _figuras_sin_incrustar(ctx: Contexto) -> list[str]:
    from . import informes_finales

    return informes_finales.figuras_sin_incrustar(ctx)


def figuras_indexadas(
    ctx: Contexto, *, indice: pl.DataFrame | None = None, plan: pl.DataFrame | None = None, sin_incrustar: Sequence[str] | None = None
) -> ResultadoDT:
    """DT08: el índice trae todas las figuras del plan y ninguna ajena, con PNG y CSV existentes y con las huellas del índice, y cada una está incrustada en un informe."""
    idx = _leer_todo_texto(ctx, "figuras_indice.csv") if indice is None else indice
    pl_figs = config.cargar_plan_figuras() if plan is None else plan
    ids_idx, ids_plan = _texto(idx, "id_figura"), _texto(pl_figs, "id_figura")
    problemas: list[str] = []
    repetidas = sorted({i for i in ids_idx if ids_idx.count(i) > 1})
    if repetidas:
        problemas.append("figuras repetidas en el índice: " + ", ".join(repetidas))
    faltan, sobran = sorted(set(ids_plan) - set(ids_idx)), sorted(set(ids_idx) - set(ids_plan))
    if faltan:
        problemas.append("faltan en el índice: " + ", ".join(faltan))
    if sobran:
        problemas.append("figuras ajenas al plan: " + ", ".join(sobran))
    csv_plan = dict(zip(ids_plan, _texto(pl_figs, "tabla_csv")))
    for r in idx.iter_rows(named=True):
        id_ = r["id_figura"]
        png, csv = ctx.dir_figuras / str(r["archivo_png"]), ctx.dir_tablas / str(r["csv"])
        if not png.is_file():
            problemas.append(f"{id_}: no existe el PNG {r['archivo_png']}")
        elif manifiesto.sha256_archivo(png) != r["sha256_png"]:
            problemas.append(f"{id_}: el PNG no coincide con su huella del índice")
        if not csv.is_file():
            problemas.append(f"{id_}: no existe el CSV {r['csv']}")
        elif manifiesto.sha256_archivo(csv) != r["sha256_csv"]:
            problemas.append(f"{id_}: el CSV {r['csv']} cambió después de dibujar la figura")
        if id_ in csv_plan and csv_plan[id_] != r["csv"]:
            problemas.append(f"{id_}: el CSV del índice ({r['csv']}) no es el del plan ({csv_plan[id_]})")
    faltantes = list(sin_incrustar) if sin_incrustar is not None else _figuras_sin_incrustar(ctx)
    if faltantes:
        problemas.append("sin incrustar en ningún informe: " + ", ".join(sorted(faltantes)))
    evidencia = "tablas/figuras_indice.csv; figuras/; config/plan_figuras.csv"
    if problemas:
        return _resultado("figuras_indexadas", "FAIL", _lista(problemas), evidencia)
    return _resultado(
        "figuras_indexadas", "PASS",
        f"{len(ids_plan)} de {len(ids_plan)} figuras del plan indexadas con su PNG y su CSV de respaldo (huellas iguales) e incrustadas en un informe con su leyenda y su CSV de origen",
        evidencia,
    )


# ============================================================================================================================ DT09
def matriz_cubre_candidatas(
    ctx: Contexto, *, matriz: pl.DataFrame | None = None, disponibilidad: pl.DataFrame | None = None, universo: pl.DataFrame | None = None, ranking: pl.DataFrame | None = None
) -> ResultadoDT:
    """DT09: la matriz de decisión y la tabla de disponibilidad cubren exactamente el universo de variables, y todo lo que está en el ranking está en la matriz."""
    mat = _leer_tabla(ctx, "matriz_decision_variables.csv", ["variable"]) if matriz is None else matriz
    disp = _leer_tabla(ctx, "disponibilidad_variables.csv", ["variable"]) if disponibilidad is None else disponibilidad
    uni = decision.cargar_universo() if universo is None else universo
    rank = _leer_tabla(ctx, "t23_ranking_variables.csv", ["variable"]) if ranking is None else ranking
    v_mat, v_disp, v_uni, v_rank = (_texto(t, "variable") for t in (mat, disp, uni, rank))
    problemas: list[str] = []
    repetidas = sorted({v for v in v_mat if v_mat.count(v) > 1})
    if repetidas:
        problemas.append("variables repetidas en la matriz: " + ", ".join(repetidas))
    for nombre, vs in (("la matriz", v_mat), ("la disponibilidad", v_disp)):
        faltan, sobran = sorted(set(v_uni) - set(vs)), sorted(set(vs) - set(v_uni))
        if faltan:
            problemas.append(f"faltan en {nombre}: " + ", ".join(faltan))
        if sobran:
            problemas.append(f"sobran en {nombre}: " + ", ".join(sobran))
    fuera = sorted(set(v_rank) - set(v_mat))
    if fuera:
        problemas.append("están en el ranking y no en la matriz: " + ", ".join(fuera))
    evidencia = "tablas/matriz_decision_variables.csv; tablas/disponibilidad_variables.csv; tablas/t23_ranking_variables.csv"
    if problemas:
        return _resultado("matriz_cubre_candidatas", "FAIL", _lista(problemas), evidencia)
    return _resultado(
        "matriz_cubre_candidatas", "PASS",
        f"{len(v_uni)} variables en la matriz, igual al universo documentado y a la tabla de disponibilidad; las {len(set(v_rank))} variables del ranking están en la matriz",
        evidencia,
    )


# ============================================================================================================================ DT10
_RE_PASARON = re.compile(r"(\d+) passed")
_RE_FALLARON = re.compile(r"(\d+) (?:failed|errors?)")


def lanzar_pytest(objetivos: Sequence[str] = ("tests",)) -> tuple[int, str]:
    """Corre la suite en un subproceso (``python -B -m pytest tests``) y devuelve ``(código de salida, salida)``.

    El subproceso lleva ``FASE2_EDA_DENTRO_DE_PYTEST`` para que una corrida del runner dentro de esas pruebas no vuelva a lanzar pytest. Se pasa ``-o addopts`` porque
    ``pytest.ini`` ya trae ``-q`` y un segundo ``-q`` oculta la línea de resumen. ``objetivos`` solo cambia en las pruebas del propio lanzador (un archivo de prueba mínimo).
    """
    entorno = dict(os.environ)
    entorno["PYTHONDONTWRITEBYTECODE"] = "1"
    entorno[VAR_DENTRO_DE_PYTEST] = "1"
    p = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", *objetivos, "-o", "addopts=-p no:cacheprovider", "-q"],
        cwd=rutas.RAIZ_EDA, env=entorno, capture_output=True, text=True, timeout=TIMEOUT_SUBPROCESO_S,
    )
    return p.returncode, p.stdout + p.stderr


def pytest_verde(ctx: Contexto, *, omitir: bool = False, lanzador: Callable[[], tuple[int, str]] | None = None) -> ResultadoDT:
    """DT10: la suite de pruebas pasa en un subproceso. PENDIENTE con ``--sin-pytest`` o dentro de pytest (guarda contra la recursión)."""
    evidencia = "pytest tests"
    if omitir:
        return _resultado("pytest_verde", "PENDIENTE", "no se ejecutó: la corrida usó --sin-pytest", evidencia)
    if os.environ.get(VAR_DENTRO_DE_PYTEST):
        return _resultado("pytest_verde", "PENDIENTE", "no se ejecutó: la corrida ya está dentro de una ejecución de pytest (dentro de pytest)", evidencia)
    if lanzador is None:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return _resultado("pytest_verde", "PENDIENTE", "no se ejecutó: la corrida está dentro de pytest y no se anida otra suite", evidencia)
        lanzador = lanzar_pytest
    codigo, salida = lanzador()
    pasaron = _RE_PASARON.findall(salida)
    fallaron = _RE_FALLARON.findall(salida)
    if codigo == 0 and pasaron and not fallaron:
        return _resultado("pytest_verde", "PASS", f"{fmt_entero(int(pasaron[-1]))} pruebas pasaron", evidencia)
    if fallaron:
        return _resultado("pytest_verde", "FAIL", f"pytest terminó con código {codigo}; pruebas con fallo o error: {fmt_entero(sum(int(n) for n in fallaron))}", evidencia)
    if codigo == 0:
        return _resultado("pytest_verde", "FAIL", "pytest terminó con código 0 pero no se pudo leer el resumen de pruebas", evidencia)
    return _resultado("pytest_verde", "FAIL", f"pytest terminó con código {codigo}", evidencia)


# ============================================================================================================================ DT11
def huellas_arbol(carpeta: Path | str, excluir: Iterable[str] = ()) -> dict[str, str]:
    """Mapa ruta relativa (con barras) -> SHA-256 de todos los archivos bajo ``carpeta``, ordenado por ruta."""
    carpeta = Path(carpeta)
    excluidos = set(excluir)
    out = {}
    for p in sorted(carpeta.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(carpeta).as_posix()
        if rel not in excluidos:
            out[rel] = manifiesto.sha256_archivo(p)
    return out


def corrida_determinista(
    ctx: Contexto, *, omitir: bool = False, lanzador: Callable[[Path], int] | None = None, excluir: Iterable[str] = EXCLUIR_DEL_COTEJO
) -> ResultadoDT:
    """DT11: dos corridas completas en carpetas temporales distintas producen los mismos SHA-256 en todos los archivos de ``<base>/<fuente>``.

    ``lanzador(base)`` corre el runner con ``--salida-base base`` y devuelve su código de salida (0 o 1: la corrida terminó). Las carpetas temporales se borran siempre.
    """
    evidencia = "dos corridas en carpetas temporales"
    if omitir:
        return _resultado("corrida_determinista", "PENDIENTE", "no se ejecutó: la corrida usó --sin-determinismo", evidencia)
    if lanzador is None:
        raise ValueError("corrida_determinista necesita un lanzador de la corrida")
    bases: list[Path] = []
    try:
        for i in (1, 2):
            base = Path(tempfile.mkdtemp(prefix=f"fase2_det{i}_"))
            bases.append(base)
            codigo = lanzador(base)
            if codigo not in (0, 1):
                return _resultado("corrida_determinista", "FAIL", f"la corrida {i} terminó con código {codigo}", evidencia)
        h1, h2 = (huellas_arbol(b / ctx.etiqueta, excluir) for b in bases)
    finally:
        for b in bases:
            shutil.rmtree(b, ignore_errors=True)
    solo1, solo2 = sorted(set(h1) - set(h2)), sorted(set(h2) - set(h1))
    distintos = sorted(r for r in set(h1) & set(h2) if h1[r] != h2[r])
    if not h1:
        return _resultado("corrida_determinista", "FAIL", "las corridas no produjeron ningún archivo", evidencia)
    if solo1 or solo2 or distintos:
        problemas = [f"{r} difiere" for r in distintos] + [f"{r} solo está en la primera corrida" for r in solo1] + [f"{r} solo está en la segunda corrida" for r in solo2]
        return _resultado("corrida_determinista", "FAIL", f"{len(problemas)} archivos no coinciden entre dos corridas: " + _lista(problemas), evidencia)
    return _resultado("corrida_determinista", "PASS", f"dos corridas completas dieron los mismos {fmt_entero(len(h1))} archivos con los mismos SHA-256", evidencia)


# ============================================================================================================================ DT12
def completo_corrido(ctx: Contexto, *, anios_esperados: Sequence[int] | None = None) -> ResultadoDT:
    """DT12: PENDIENTE con extractos; con datos completos exige que todas las tablas y métricas lleven la etiqueta ``completo`` y que se hayan leído los seis años."""
    evidencia = "fuente de la corrida"
    if ctx.etiqueta != "completo":
        return _resultado("completo_corrido", "PENDIENTE", "solo se corrió sobre extractos: falta la corrida con los datos completos", evidencia)
    esperados = sorted(int(a) for a in (anios_esperados if anios_esperados is not None else config.cargar_fuentes_toml()["meta"]["anios"]))
    indice, mets = ctx.indice_tablas(), ctx.metricas()
    problemas: list[str] = []
    tablas_ajenas = sorted(str(n) for n, f in zip(indice["nombre"].to_list(), indice["fuente"].to_list()) if f != "completo")
    if tablas_ajenas:
        problemas.append(f"{len(tablas_ajenas)} tablas sin la etiqueta completo: " + _lista(tablas_ajenas, 3))
    metricas_ajenas = sorted(str(k) for k, n in zip(mets["clave"].to_list(), mets["nivel"].to_list()) if n != "completo")
    if metricas_ajenas:
        problemas.append(f"{len(metricas_ajenas)} métricas sin la etiqueta completo: " + _lista(metricas_ajenas, 3))
    leidos = set(int(a) for a in ctx.cortafuegos.registro()["anio"].to_list())
    sin_leer = [a for a in esperados if a not in leidos]
    if sin_leer:
        problemas.append("no se leyeron los años " + ", ".join(str(a) for a in sin_leer))
    if problemas:
        return _resultado("completo_corrido", "FAIL", _lista(problemas), evidencia)
    return _resultado(
        "completo_corrido", "PASS",
        f"{fmt_entero(indice.height)} tablas y {fmt_entero(mets.height)} métricas con la etiqueta completo; se leyeron los {len(esperados)} años",
        evidencia,
    )


# ============================================================================================================================ conjunto
def _sanear(texto: str, ctx: Contexto) -> str:
    """Un mensaje de error sin rutas absolutas y de largo acotado, para que el detalle sea reproducible."""
    for ruta in sorted({str(ctx.dir_salida), str(rutas.RAIZ_PROYECTO), str(Path.home())}, key=len, reverse=True):
        texto = texto.replace(ruta, "<ruta>")
    return " ".join(texto.split())[:200]


def _argumentos(nombre: str, b: Banderas) -> dict[str, Any]:
    """Los insumos que el runner entrega a cada verificador, por nombre de verificador."""
    return {
        "cifras_con_csv": {"informes_esperados": b.informes_esperados},
        "cascada_cuadra": {"cascada": b.cascada, "cascada_pipeline": b.cascada_pipeline},
        "pytest_verde": {"omitir": b.sin_pytest, "lanzador": b.lanzador_pytest},
        "corrida_determinista": {"omitir": b.sin_determinismo, "lanzador": b.lanzador_determinismo},
    }.get(nombre, {})


def verificar_todo(ctx: Contexto, banderas: Banderas | None = None, *, ids: Iterable[str] | None = None) -> list[ResultadoDT]:
    """Corre los verificadores en el orden de ``config/definicion_terminado.csv`` (o solo los ``ids`` pedidos) y devuelve un ``ResultadoDT`` por criterio.

    Un verificador que lanza una excepción (por ejemplo, una tabla que falta) queda en FAIL con el motivo; no detiene a los demás.
    """
    b = banderas or Banderas()
    pedidos = None if ids is None else set(ids)
    modulo = sys.modules[__name__]
    resultados: list[ResultadoDT] = []
    for fila in config.cargar_definicion_terminado().iter_rows(named=True):
        if pedidos is not None and fila["id"] not in pedidos:
            continue
        funcion = getattr(modulo, fila["verificador"], None)
        if not callable(funcion):
            resultados.append(ResultadoDT(fila["id"], fila["criterio"], "FAIL", f"no existe el verificador {fila['verificador']}", ""))
            continue
        try:
            resultados.append(funcion(ctx, **_argumentos(fila["verificador"], b)))
        except Exception as e:  # noqa: BLE001 - un fallo del verificador es un FAIL del criterio, no una caída de la corrida
            resultados.append(ResultadoDT(fila["id"], fila["criterio"], "FAIL", f"el verificador falló: {type(e).__name__}: {_sanear(str(e), ctx)}", ""))
    return resultados


def tabla_verificacion(resultados: Sequence[ResultadoDT]) -> pl.DataFrame:
    """``verificacion_dt.csv``: una fila por criterio con ``id``, ``criterio``, ``estado``, ``detalle`` y ``evidencia``."""
    return pl.DataFrame([{c: getattr(r, c) for c in COLUMNAS_VERIFICACION} for r in resultados], schema=ESQUEMA_VERIFICACION)


def resumen_verificacion(resultados: Sequence[ResultadoDT]) -> dict[str, int]:
    """Conteos de los estados."""
    estados = [r.estado for r in resultados]
    return {"n_criterios": len(estados), "n_pass": estados.count("PASS"), "n_fail": estados.count("FAIL"), "n_pendiente": estados.count("PENDIENTE")}


def escribir_verificacion(ctx: Contexto, resultados: Sequence[ResultadoDT]) -> pl.DataFrame:
    """Escribe ``verificacion_dt.csv`` y ``t26_resumen_cierre.csv`` (tarea ``T13``) y devuelve la tabla de verificación.

    El resumen suma los conteos de estados a los de las tablas y figuras de las tareas anteriores (las dos tablas de esta tarea no se cuentan) y a los siete informes.
    """
    from . import informes_finales

    tabla = tabla_verificacion(resultados)
    n_antes = ctx.indice_tablas().filter(pl.col("tarea") != TAREA).height
    ctx.escribir_tabla("verificacion_dt.csv", tabla, tarea=TAREA, descripcion="Resultado de los doce criterios de terminado de la Fase 2 (PASS, FAIL o PENDIENTE)")
    resumen = {**resumen_verificacion(resultados), "n_tablas": n_antes, "n_figuras": ctx.figuras.indice().height, "n_informes": len(informes_finales.informes_esperados())}
    ctx.escribir_tabla("t26_resumen_cierre.csv", pl.DataFrame([resumen], schema=ESQUEMA_RESUMEN_CIERRE), tarea=TAREA, descripcion="Conteos de la verificación y de los productos de la corrida")
    return tabla


# FIN-verificadores
