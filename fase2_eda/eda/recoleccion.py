"""CRISP-DM 2.1: recolección inicial de los datos.

Tablas (``salidas/<fuente>/tablas``):
- ``t21_archivos.csv``: tamaño, líneas, codificación, fin de línea, hash y cuadratura de lectura por archivo.
- ``t21_deriva_esquema.csv`` y ``t21_esquema_diferencias.csv``: nombre de cada columna por año contra el canónico.
- ``t21_deriva_formato.csv``: indicadores de formato (fechas, decimales, puntos en códigos, IDs) por año.
- ``t21_problemas_carga.csv``: bitácora de problemas de carga.
- ``t21_verificaciones.csv``: contraste de lo observado con lo esperado/documentado.
- ``t21_procedencia.csv``: hoja de procedencia de los datos y de los documentos.
- ``t21_resumen.csv``: cifras globales para el informe.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from . import config, esquema, rutas
from .contexto import Contexto
from .informes import Informe, cifra
from .lector import Lector
from . import graficos

TAREA = "T03"
ANIOS_TXT = "2019 a 2024"


# ---------------------------------------------------------------------------- archivos
def tabla_archivos(lector: Lector, tarea: str = TAREA) -> pl.DataFrame:
    """Una fila por archivo. Verifica leyendo que ``leídas + rechazadas = líneas de datos``."""
    filas = []
    for anio in lector.fuente.anios():
        esc = lector.escaneo(anio)
        df = lector.leer(anio, ["COD_HOSPITAL"], proposito="estructura", tarea=tarea)
        invalidas = esc.lineas_utf8_invalidas
        filas.append(
            {
                "anio": anio,
                "archivo": esc.nombre,
                "tamano_bytes": esc.tamano_bytes,
                "lineas_total": esc.lineas_total,
                "lineas_datos": esc.lineas_datos,
                "n_columnas_encabezado": esc.n_campos_encabezado,
                "bom": "si" if esc.tiene_bom else "no",
                "fin_de_linea": esc.fin_de_linea,
                "codificacion": "UTF-8 válido" if invalidas == 0 else "UTF-8 con líneas inválidas",
                "lineas_utf8_invalidas": invalidas,
                "lineas_vacias": esc.lineas_vacias,
                "filas_leidas": df.height,
                "filas_rechazadas": esc.rechazadas,
                "suma_cuadra": "si" if df.height + esc.rechazadas == esc.lineas_datos else "no",
                "sha256": esc.sha256,
                "fuente": lector.fuente.etiqueta,
            }
        )
    return pl.DataFrame(filas).sort("anio")


# --------------------------------------------------------------------------- esquema
def deriva_esquema(lector: Lector) -> pl.DataFrame:
    filas = []
    for anio in lector.fuente.anios():
        enc = lector.escaneo(anio).encabezado
        for i, canon in enumerate(esquema.COLUMNAS):
            nombre = enc[i] if i < len(enc) else None
            filas.append(
                {
                    "posicion": i + 1,
                    "nombre_canonico": canon,
                    "anio": anio,
                    "nombre_encabezado": nombre,
                    "coincide": "si" if nombre == canon else "no",
                }
            )
    return pl.DataFrame(filas).sort(["posicion", "anio"])


# ---------------------------------------------------------------------------- formato
_ISO = r"^\d{4}-\d{2}-\d{2}$"
_DMY = r"^\d{2}-\d{2}-\d{4}$"
_TEXTO = r"^[A-Za-zÁÉÍÓÚÑáéíóúñ ]+$"

Predicado = Callable[[pl.Expr], pl.Expr]


def _indicadores() -> list[tuple[str, str, str, Predicado]]:
    """(columna, clase, descripción, predicado) sobre los valores NO nulos."""
    ind: list[tuple[str, str, str, Predicado]] = []
    for col in ("FECHA_NACIMIENTO", "FECHA_INGRESO", "FECHAALTA", "FECHAINTERV1", "FECHATRASLADO1"):
        ind.append((col, "ISO", "AAAA-MM-DD", lambda c: c.str.contains(_ISO)))
        ind.append((col, "DD-MM-AAAA", "DD-MM-AAAA", lambda c: c.str.contains(_DMY)))
        ind.append((col, "TEXTO", "solo letras (centinela)", lambda c: c.str.contains(_TEXTO)))
    ind += [
        ("IR_29301_PESO", "COMA_DECIMAL", "coma decimal", lambda c: c.str.contains(r"^\d*,\d+$")),
        ("IR_29301_PESO", "SIN_CERO_INICIAL", "coma sin cero inicial", lambda c: c.str.contains(r"^,\d+$")),
        ("IR_29301_PESO", "PUNTO_DECIMAL", "punto decimal", lambda c: c.str.contains(r"^\d*\.\d+$")),
        ("DIAGNOSTICO1", "CON_PUNTO", "con punto", lambda c: c.str.contains(r"\.")),
        ("DIAGNOSTICO1", "MINUSCULA", "con minúsculas", lambda c: c.str.contains(r"[a-z]")),
        ("DIAGNOSTICO1", "ESPACIO_EXTREMO", "espacio al inicio o al final", lambda c: c.str.contains(r"^\s|\s$")),
        ("PROCEDIMIENTO1", "CON_PUNTO", "con punto", lambda c: c.str.contains(r"\.")),
        ("PROCEDIMIENTO1", "CERO_INICIAL", "comienza con cero", lambda c: c.str.starts_with("0")),
        ("IR_29301_COD_GRD", "SEIS_DIGITOS", "seis dígitos", lambda c: c.str.contains(r"^\d{6}$")),
        ("IR_29301_COD_GRD", "TEXTO", "solo letras (centinela)", lambda c: c.str.contains(_TEXTO)),
        ("CIP_ENCRIPTADO", "SOLO_DIGITOS", "solo dígitos", lambda c: c.str.contains(r"^\d+$")),
        ("CIP_ENCRIPTADO", "TEXTO", "solo letras (centinela)", lambda c: c.str.contains(_TEXTO)),
        ("COD_HOSPITAL", "SEIS_DIGITOS", "seis dígitos", lambda c: c.str.contains(r"^\d{6}$")),
    ]
    return ind


def deriva_formato(lector: Lector, tarea: str = TAREA) -> pl.DataFrame:
    ind = _indicadores()
    columnas = sorted({c for c, _, _, _ in ind})
    filas = []
    for anio in lector.fuente.anios():
        df = lector.leer(anio, columnas, proposito="estructura", tarea=tarea)
        exprs = []
        for k, (col, clase, _, pred) in enumerate(ind):
            exprs.append(pred(pl.col(col)).fill_null(False).sum().alias(f"c{k}"))
        for col in columnas:
            exprs.append(pl.col(col).is_not_null().sum().alias(f"v_{col}"))
        r = df.select(exprs).row(0, named=True)
        for k, (col, clase, desc, _) in enumerate(ind):
            nv = int(r[f"v_{col}"])
            nc = int(r[f"c{k}"])
            filas.append(
                {
                    "anio": anio,
                    "indicador": f"{col}: {desc}",
                    "columna": col,
                    "clase": clase,
                    "n_valores": nv,
                    "n_clase": nc,
                    "prop_clase": (nc / nv) if nv else None,
                }
            )
        # clase OTRO para las fechas: lo que no es ISO, DMY ni TEXTO
        for col in ("FECHA_NACIMIENTO", "FECHA_INGRESO", "FECHAALTA", "FECHAINTERV1", "FECHATRASLADO1"):
            nv = int(r[f"v_{col}"])
            nc_ok = sum(
                int(r[f"c{k}"]) for k, (c, _, _, _) in enumerate(ind) if c == col
            )
            otro = nv - nc_ok
            filas.append(
                {
                    "anio": anio,
                    "indicador": f"{col}: otro formato",
                    "columna": col,
                    "clase": "OTRO",
                    "n_valores": nv,
                    "n_clase": max(otro, 0),
                    "prop_clase": (max(otro, 0) / nv) if nv else None,
                }
            )
    return pl.DataFrame(filas).sort(["columna", "clase", "anio"])


# ------------------------------------------------------------------------- bitácora
def bitacora_problemas(archivos: pl.DataFrame, esquema_dif: pl.DataFrame, formato: pl.DataFrame) -> pl.DataFrame:
    filas = []

    def agregar(anio, problema, severidad, n, detalle, accion):
        filas.append(
            {"anio": anio, "problema": problema, "severidad": severidad, "n": int(n), "detalle": detalle, "accion": accion}
        )

    for r in archivos.iter_rows(named=True):
        a = r["anio"]
        if r["bom"] == "si":
            agregar(a, "BOM UTF-8 al inicio del archivo", "baja", 1, "El primer nombre de columna llega con BOM si no se decodifica como utf-8-sig", "El lector lo retira")
        if r["fin_de_linea"] != "CRLF":
            agregar(a, "Fin de línea distinto de CRLF", "media", 1, r["fin_de_linea"], "Se acepta CRLF y LF")
        if r["lineas_utf8_invalidas"]:
            agregar(a, "Líneas con bytes no UTF-8", "alta", r["lineas_utf8_invalidas"], "No decodifican como UTF-8", "Revisar la codificación de origen")
        if r["lineas_vacias"]:
            agregar(a, "Líneas vacías", "media", r["lineas_vacias"], "Líneas sin contenido", "Se rechazan y se cuentan")
        if r["filas_rechazadas"]:
            agregar(a, "Líneas con número de campos distinto del esperado", "alta", r["filas_rechazadas"], "Se rechazan con su número de línea", "Informar a FONASA")
        if r["n_columnas_encabezado"] != esquema.N_COLUMNAS:
            agregar(a, "Número de columnas distinto del esperado", "alta", abs(r["n_columnas_encabezado"] - esquema.N_COLUMNAS), "El encabezado no tiene el ancho canónico", "Detener la carga")
        if r["suma_cuadra"] != "si":
            agregar(a, "Las filas leídas más las rechazadas no suman las líneas de datos", "alta", 1, "Cuadratura rota", "Detener la carga")
    for r in esquema_dif.iter_rows(named=True):
        if r["posicion"] == esquema.POS_ID + 1:
            agregar(r["anio"], "Nombre alternativo de la segunda columna (identificador)", "media", 1, f"El encabezado dice {r['nombre_encabezado']} en lugar de {r['nombre_canonico']}", "Alias por posición")
        else:
            agregar(r["anio"], "Nombre de columna distinto del canónico", "media", 1, f"posición {r['posicion']}", "Revisar el esquema")
    dmy = formato.filter((pl.col("clase") == "DD-MM-AAAA") & (pl.col("n_clase") > 0))
    for r in dmy.group_by(["anio", "columna"]).agg(pl.col("n_clase").sum().alias("n")).sort(["anio", "columna"]).iter_rows(named=True):
        agregar(r["anio"], "Fechas en formato DD-MM-AAAA", "media", r["n"], f"Columna {r['columna']}", "El lector de fechas acepta ISO y DD-MM-AAAA")
    for r in formato.filter(pl.col("clase") == "SIN_CERO_INICIAL").filter(pl.col("n_clase") > 0).sort("anio").iter_rows(named=True):
        agregar(r["anio"], "Peso GRD con coma y sin cero inicial", "baja", r["n_clase"], r["columna"], "Convertir coma a punto y aceptar ,xxxx")
    for r in formato.filter(pl.col("clase") == "TEXTO").filter(pl.col("n_clase") > 0).sort(["columna", "anio"]).iter_rows(named=True):
        agregar(r["anio"], "Texto (centinela) en columna de formato definido", "media", r["n_clase"], r["columna"], "Contar como faltante disfrazado")
    if not filas:
        return pl.DataFrame(schema={"anio": pl.Int64, "problema": pl.String, "severidad": pl.String, "n": pl.Int64, "detalle": pl.String, "accion": pl.String})
    return pl.DataFrame(filas).sort(["anio", "problema", "detalle"])


# -------------------------------------------------------------------- verificaciones
def verificaciones(
    archivos: pl.DataFrame,
    esquema_dif: pl.DataFrame,
    formato: pl.DataFrame,
    fuente: str,
    filas_extracto_esperadas: int | None = None,
) -> pl.DataFrame:
    """Contrasta hechos de la carga con lo esperado. 'esperado' viene de la Fase 2 o de los extractos."""
    filas = []

    def agregar(id_, descripcion, esperado, observado, base):
        filas.append(
            {
                "id": id_,
                "descripcion": descripcion,
                "esperado": esperado,
                "observado": observado,
                "base_del_esperado": base,
                "resultado": "COINCIDE" if esperado == observado else "DISCREPA",
                "fuente": fuente,
            }
        )

    anios = archivos["anio"].to_list()
    ncol = sorted(set(archivos["n_columnas_encabezado"].to_list()))
    agregar("V01", "Todos los archivos tienen el mismo número de columnas", str(esquema.N_COLUMNAS), ";".join(str(x) for x in ncol), "Fase 2, sección 1.1")
    agregar("V02", "Las filas leídas más las rechazadas suman las líneas de datos", "si", "si" if (archivos["suma_cuadra"] == "si").all() else "no", "Fase 1, requisito de lectura sin modificaciones")
    if fuente == "extracto" and filas_extracto_esperadas is not None:
        obs = sorted(set(archivos["filas_leidas"].to_list()))
        agregar("V03", "Filas por año en el extracto", str(filas_extracto_esperadas), ";".join(str(x) for x in obs), "Inspección de los extractos")
    ids_alt = sorted(
        esquema_dif.filter(pl.col("posicion") == esquema.POS_ID + 1)["anio"].to_list()
    )
    agregar(
        "V04",
        "Años en que la segunda columna se llama ID_BENEFICIARIO",
        "2019",
        ";".join(str(a) for a in ids_alt) if ids_alt else "ninguno",
        "Fase 2, sección 1.3 y problemas de calidad (afirma que ocurre en 2019)",
    )
    dmy_anios = sorted(
        formato.filter((pl.col("columna") == "FECHA_INGRESO") & (pl.col("clase") == "DD-MM-AAAA") & (pl.col("n_clase") > 0))["anio"].to_list()
    )
    agregar("V05", "Años con FECHA_INGRESO en DD-MM-AAAA", "2023", ";".join(str(a) for a in dmy_anios) if dmy_anios else "ninguno", "Inspección de los extractos")
    iso_nac = formato.filter((pl.col("columna") == "FECHA_NACIMIENTO") & (pl.col("clase") == "ISO"))
    ok_nac = bool((iso_nac["n_clase"] == iso_nac["n_valores"]).all()) if iso_nac.height else False
    agregar("V06", "FECHA_NACIMIENTO viene en formato ISO en todos los años", "si", "si" if ok_nac else "no", "Fase 2, problemas de calidad")
    bom_anios = sorted(archivos.filter(pl.col("bom") == "si")["anio"].to_list())
    agregar("V07", "Años con BOM", "2021;2022;2023", ";".join(str(a) for a in bom_anios) if bom_anios else "ninguno", "Inspección de los extractos")
    agregar("V08", "Cobertura de años", "2019;2020;2021;2022;2023;2024", ";".join(str(a) for a in anios), "Fase 2, sección 1.1")
    return pl.DataFrame(filas)


# ----------------------------------------------------------------------- procedencia
def procedencia(raiz_proyecto: Path, archivos: pl.DataFrame) -> pl.DataFrame:
    """Hoja de procedencia de los datos y de los documentos de respaldo."""
    from .manifiesto import sha256_archivo

    oficios = sorted(p for p in raiz_proyecto.glob("Oficio_Respuesta_N*23364_T10901.pdf") if p.is_file())
    otros = sorted(
        p for p in raiz_proyecto.glob("Oficio*.pdf") if p.is_file() and p not in oficios
    )
    filas = []

    def agregar(item, estado, evidencia, accion):
        filas.append({"item": item, "estado": estado, "evidencia": evidencia, "accion": accion})

    if oficios:
        agregar(
            "Oficio FONASA 23364/2026 (respuesta a la solicitud AO004T0010901)",
            "PRESENTE",
            f"archivo con SHA-256 {sha256_archivo(oficios[0])[:16]}; entrega el catálogo de GRD (código y glosa) y explica la estructura del código IR-GRD",
            "Citar como fuente del catálogo GRD; no es respuesta por los microdatos",
        )
    else:
        agregar("Oficio FONASA 23364/2026 (respuesta a la solicitud AO004T0010901)", "AUSENTE", "no se encontró el archivo en la carpeta del proyecto", "Agregar el oficio al expediente")
    agregar(
        "Respuesta de FONASA por los microdatos de egresos 2019 a 2024",
        "AUSENTE" if not otros else "PRESENTE",
        "no hay en la carpeta un documento de respuesta que acompañe los archivos de egresos"
        if not otros
        else "hay otros oficios en la carpeta; revisar si corresponde",
        "Incorporar la respuesta (con su número de oficio y fecha) al expediente",
    )
    agregar(
        "Condiciones de uso y de publicación de resultados por hospital",
        "NO DOCUMENTADAS",
        "el oficio 23364 es un catálogo técnico, sin registros de pacientes, y no se pronuncia sobre publicación por hospital",
        "Consultar por escrito a FONASA (Fase 1, sección 14.4); decisión 10 sigue abierta",
    )
    agregar(
        "Diccionario oficial de todas las versiones del agrupador",
        "NO EXISTE SEGÚN EL OFICIO",
        "el oficio advierte que identificar el CDM y el grupo base por la estructura del código no equivale a un diccionario oficial de las versiones históricas",
        "Usar el catálogo GRDs_T10901.xlsx solo como referencia de códigos vigentes",
    )
    for r in archivos.sort("anio").iter_rows(named=True):
        agregar(
            f"Archivo de egresos {r['anio']}",
            "PRESENTE",
            f"{r['archivo']}; SHA-256 {r['sha256'][:16]}; {r['lineas_datos']} líneas de datos",
            "Conservar sin modificar (regla R4)",
        )
    return pl.DataFrame(filas)


# ------------------------------------------------------------------------------ figuras
def figura_filas_por_archivo(ctx: Contexto) -> None:
    t = ctx.leer_tabla("t21_archivos.csv")
    fig, ax = graficos.nueva_figura()
    graficos.barras_verticales(
        ax,
        [str(a) for a in t["anio"].to_list()],
        t["filas_leidas"].to_list(),
        etiqueta_y="Filas leídas",
        etiqueta_x="Año del archivo",
    )
    ctx.figuras.guardar_figura(
        fig,
        id_figura="F02_01",
        tarea=2,
        titulo="Filas leídas por archivo y año",
        descripcion_alt="Barras con el número de filas leídas de cada archivo anual de egresos, de 2019 a 2024.",
        csv="t21_archivos.csv",
        anios=ANIOS_TXT,
    )


def figura_deriva_formato(ctx: Contexto) -> None:
    t = ctx.leer_tabla("t21_deriva_formato.csv")
    claves = (
        ("FECHA_INGRESO", "DD-MM-AAAA"),
        ("FECHAALTA", "DD-MM-AAAA"),
        ("FECHAINTERV1", "DD-MM-AAAA"),
        ("FECHAINTERV1", "TEXTO"),
        ("FECHAINTERV1", "ISO"),
        ("FECHA_INGRESO", "ISO"),
        ("FECHAALTA", "ISO"),
        ("FECHA_NACIMIENTO", "ISO"),
        ("IR_29301_PESO", "SIN_CERO_INICIAL"),
        ("IR_29301_PESO", "COMA_DECIMAL"),
        ("DIAGNOSTICO1", "CON_PUNTO"),
        ("DIAGNOSTICO1", "MINUSCULA"),
        ("PROCEDIMIENTO1", "CERO_INICIAL"),
        ("IR_29301_COD_GRD", "SEIS_DIGITOS"),
        ("IR_29301_COD_GRD", "TEXTO"),
        ("CIP_ENCRIPTADO", "SOLO_DIGITOS"),
        ("CIP_ENCRIPTADO", "TEXTO"),
    )
    sel = t.with_columns((pl.col("columna") + "|" + pl.col("clase")).alias("clave"))
    orden = [f"{c}|{k}" for c, k in claves]
    sel = sel.filter(pl.col("clave").is_in(orden)).with_columns((pl.col("prop_clase") * 100).alias("pct"))
    nombres = {f"{c}|{k}": t.filter((pl.col("columna") == c) & (pl.col("clase") == k))["indicador"][0] for c, k in claves if t.filter((pl.col("columna") == c) & (pl.col("clase") == k)).height}
    orden = [o for o in orden if o in nombres]
    m, filas, cols = graficos.tabla_a_matriz(sel, "clave", "anio", "pct", orden_filas=orden)
    fig, ax = graficos.nueva_figura(figsize=(9.0, 6.0))
    graficos.mapa_calor(
        fig, ax, m, [nombres[o] for o in filas], cols, vmin=0, vmax=100, etiqueta_barra="% de los valores informados",
        fmt="{:.1f}",
    )
    ax.set_xlabel("Año del archivo")
    ctx.figuras.guardar_figura(
        fig,
        id_figura="F03_01",
        tarea=3,
        titulo="Deriva de formato por año",
        descripcion_alt="Mapa de calor con el porcentaje de valores de cada formato (fechas, decimales, puntos en códigos, identificadores) por año.",
        csv="t21_deriva_formato.csv",
        anios=ANIOS_TXT,
    )


# ------------------------------------------------------------------------------ informe
def construir_informe(ctx: Contexto) -> Path:
    inf = Informe(ctx, "t21_informe.md", "Informe de recolección inicial de los datos (CRISP-DM 2.1)")
    c = lambda col, **kw: cifra(ctx, "t21_resumen.csv", col)  # noqa: E731
    inf.seccion("1. Qué se recolectó")
    inf.parrafo(
        "Se recolectaron {n_archivos} archivos anuales de egresos hospitalarios de FONASA (2019 a 2024). "
        "Cada archivo se leyó completo como texto, por posición de columna, con {n_columnas} columnas por archivo. "
        "En total se leyeron {n_filas} filas y se rechazaron {n_rechazadas}; en cada archivo las filas leídas más "
        "las rechazadas suman las líneas de datos.",
        n_archivos=c("n_archivos"),
        n_columnas=c("n_columnas"),
        n_filas=c("filas_leidas_total"),
        n_rechazadas=c("filas_rechazadas_total"),
    )
    inf.tabla(
        "t21_archivos.csv",
        columnas=["anio", "archivo", "tamano_bytes", "lineas_datos", "bom", "fin_de_linea", "codificacion", "filas_leidas", "filas_rechazadas"],
        encabezados={"anio": "Año", "archivo": "Archivo", "tamano_bytes": "Bytes", "lineas_datos": "Líneas de datos", "bom": "BOM", "fin_de_linea": "Fin de línea", "codificacion": "Codificación", "filas_leidas": "Filas leídas", "filas_rechazadas": "Rechazadas"},
        formatos={"anio": "texto"},
        leyenda="Archivos anuales y cuadratura de lectura",
    )
    inf.figura("F02_01")
    inf.seccion("2. Deriva de esquema")
    inf.parrafo(
        "Se comparó el nombre de cada una de las {n_columnas} columnas de cada año con el esquema canónico. "
        "Número total de diferencias de nombre: {n_dif_esquema}. La segunda columna (identificador del paciente) se llama "
        "ID_BENEFICIARIO en {anios_id_alt|texto} y CIP_ENCRIPTADO en el resto; el lector la trata por posición.",
        n_columnas=c("n_columnas"),
        n_dif_esquema=c("n_diferencias_esquema"),
        anios_id_alt=c("anios_id_beneficiario"),
    )
    inf.tabla(
        "t21_esquema_diferencias.csv",
        encabezados={"posicion": "Posición", "nombre_canonico": "Nombre canónico", "anio": "Año", "nombre_encabezado": "Nombre en el encabezado"},
        leyenda="Diferencias de nombre de columna respecto del esquema canónico",
        columnas_texto=["nombre_encabezado"],
    )
    inf.seccion("3. Deriva de formato")
    inf.parrafo(
        "Se midió, para columnas clave, la proporción de valores informados en cada formato. "
        "Las fechas de ingreso, alta e intervención vienen en AAAA-MM-DD salvo en {anios_dmy|texto}, donde vienen en DD-MM-AAAA; "
        "la fecha de nacimiento mantiene el formato ISO todos los años.",
        anios_dmy=c("anios_fecha_dmy"),
    )
    inf.figura("F03_01")
    inf.seccion("4. Bitácora de problemas de carga")
    inf.parrafo(
        "La bitácora lista los problemas de carga por año con su severidad y la acción aplicada. "
        "Los conteos de líneas del archivo no se consideran casos de pacientes; los conteos de valores sí se suprimen si tienen menos de diez casos."
    )
    inf.tabla(
        "t21_problemas_carga.csv",
        max_filas=40,
        encabezados={"anio": "Año", "problema": "Problema", "severidad": "Severidad", "n": "N", "detalle": "Detalle", "accion": "Acción"},
        leyenda="Problemas de carga detectados",
    )
    inf.seccion("5. Verificaciones contra lo documentado")
    inf.parrafo(
        "Cada fila contrasta un hecho de la carga con lo que decían los documentos de la Fase 2 o con lo observado en los extractos. "
        "Una discrepancia es un hallazgo, no un error de la suite."
    )
    inf.tabla(
        "t21_verificaciones.csv",
        columnas=["id", "descripcion", "esperado", "observado", "resultado"],
        encabezados={"id": "Id", "descripcion": "Descripción", "esperado": "Esperado", "observado": "Observado", "resultado": "Resultado"},
        leyenda="Verificaciones de la carga",
    )
    inf.seccion("6. Procedencia de los datos y de los documentos")
    inf.tabla(
        "t21_procedencia.csv",
        columnas=["item", "estado", "evidencia", "accion"],
        encabezados={"item": "Elemento", "estado": "Estado", "evidencia": "Evidencia", "accion": "Acción"},
        leyenda="Hoja de procedencia",
    )
    inf.seccion("7. Límites")
    inf.texto_libre_sin_cifras(
        "- Los hashes SHA-256 completos están en `tablas/t21_archivos.csv` y en el manifiesto de la corrida.\n"
        "- Falta la respuesta de FONASA por los microdatos y la confirmación de las condiciones de publicación por hospital.\n"
        "- Los datos de 2024 se usan solo para estructura y calidad; las columnas de resultado de 2024 llegan enmascaradas."
    )
    return inf.guardar()


# --------------------------------------------------------------------------- ejecución
def ejecutar(ctx: Contexto, raiz_proyecto: Path | None = None) -> dict[str, Path]:
    """Ejecuta la Tarea 3: tablas, figuras, métricas e informe 2.1."""
    raiz_proyecto = raiz_proyecto or rutas.RAIZ_PROYECTO
    lector = ctx.lector
    fuentes_cfg = config.cargar_fuentes_toml()

    archivos = tabla_archivos(lector)
    ctx.escribir_tabla("t21_archivos.csv", archivos, tarea=TAREA, descripcion="Archivos recolectados y cuadratura de lectura")
    es_dif = deriva_esquema(lector)
    ctx.escribir_tabla("t21_deriva_esquema.csv", es_dif, tarea=TAREA, descripcion="Nombre de cada columna por año contra el canónico")
    diferencias = es_dif.filter(pl.col("coincide") == "no")
    ctx.escribir_tabla("t21_esquema_diferencias.csv", diferencias.select(["posicion", "nombre_canonico", "anio", "nombre_encabezado"]), tarea=TAREA, descripcion="Diferencias de esquema")
    fmt = deriva_formato(lector)
    ctx.escribir_tabla(
        "t21_deriva_formato.csv", fmt, tarea=TAREA, descripcion="Indicadores de formato por año",
        cols_n=["n_clase"], dependientes=["prop_clase"],
    )
    ctx.escribir_tabla("t21_problemas_carga.csv", bitacora_problemas(archivos, diferencias, ctx.leer_tabla("t21_deriva_formato.csv")), tarea=TAREA, descripcion="Bitácora de problemas de carga")
    ver = verificaciones(
        archivos, diferencias, ctx.leer_tabla("t21_deriva_formato.csv"), ctx.etiqueta,
        fuentes_cfg["meta"].get("filas_extracto_por_anio"),
    )
    ctx.escribir_tabla("t21_verificaciones.csv", ver, tarea=TAREA, descripcion="Verificaciones de la carga")
    ctx.escribir_tabla("t21_procedencia.csv", procedencia(raiz_proyecto, archivos), tarea=TAREA, descripcion="Hoja de procedencia")

    tf = ctx.leer_tabla("t21_deriva_formato.csv")
    anios_dmy = sorted(
        tf.filter((pl.col("columna") == "FECHA_INGRESO") & (pl.col("clase") == "DD-MM-AAAA") & (pl.col("n_clase") > 0))["anio"].unique().to_list()
    )
    ids_alt = sorted(diferencias.filter(pl.col("posicion") == esquema.POS_ID + 1)["anio"].to_list())
    resumen = pl.DataFrame(
        [
            {
                "n_archivos": archivos.height,
                "n_columnas": int(archivos["n_columnas_encabezado"].max()),
                "filas_leidas_total": int(archivos["filas_leidas"].sum()),
                "filas_rechazadas_total": int(archivos["filas_rechazadas"].sum()),
                "n_diferencias_esquema": diferencias.height,
                "anios_id_beneficiario": ", ".join(str(a) for a in ids_alt) if ids_alt else "ningún año",
                "anios_fecha_dmy": ", ".join(str(a) for a in anios_dmy) if anios_dmy else "ningún año",
            }
        ]
    )
    ctx.escribir_tabla("t21_resumen.csv", resumen, tarea=TAREA, descripcion="Cifras globales de la recolección")

    for r in archivos.iter_rows(named=True):
        ctx.registrar_metrica(f"volumen.filas_leidas.{r['anio']}", r["filas_leidas"], tarea=TAREA, tabla="t21_archivos.csv", nota="filas del archivo leído")
    ctx.registrar_metrica("volumen.filas_leidas.total", int(archivos["filas_leidas"].sum()), tarea=TAREA, tabla="t21_archivos.csv")
    ctx.registrar_metrica("volumen.filas_leidas.promedio", float(archivos["filas_leidas"].mean()), tarea=TAREA, tabla="t21_archivos.csv")
    ctx.registrar_metrica("estructura.n_archivos", archivos.height, tarea=TAREA, tabla="t21_archivos.csv")
    ctx.registrar_metrica("estructura.columnas", int(archivos["n_columnas_encabezado"].max()), tarea=TAREA, tabla="t21_archivos.csv")
    ctx.registrar_metrica("estructura.primer_anio_id_beneficiario", min(ids_alt) if ids_alt else None, tarea=TAREA, tabla="t21_esquema_diferencias.csv", nota="año más antiguo con la columna 2 llamada ID_BENEFICIARIO")
    ctx.registrar_metrica("estructura.n_anios_id_beneficiario", len(ids_alt), tarea=TAREA, tabla="t21_esquema_diferencias.csv")
    ctx.registrar_metrica("estructura.primer_anio_fecha_dmy", min(anios_dmy) if anios_dmy else None, tarea=TAREA, tabla="t21_deriva_formato.csv")
    ctx.registrar_metrica("estructura.n_anios_fecha_dmy", len(anios_dmy), tarea=TAREA, tabla="t21_deriva_formato.csv")

    figura_filas_por_archivo(ctx)
    figura_deriva_formato(ctx)
    informe = construir_informe(ctx)
    return {"informe": informe}
