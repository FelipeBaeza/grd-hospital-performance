"""CRISP-DM 2.3 (selección de variables): universo de variables, rol de la Fase 1 y disponibilidad.

Esta parte del módulo fija QUÉ variables existen y qué se puede decir de cada una antes de mirar el desenlace:

- ``cargar_universo``: las 199 variables que la matriz de decisión debe cubrir (DT09): las 129 columnas crudas, las 62 derivadas del
  documento de variables derivadas, 4 variables del panel que ese documento no trae y 4 nombres que solo menciona ``reglas_variables.csv``.
- ``rol_fase1``: el rol de cada variable con la precedencia ``reglas_variables.csv``, luego la columna "Categoría Fase 1" del diccionario, luego
  la columna "¿Entra al modelo?" del documento de derivadas traducida por prefijo y por último la herencia explícita ``HEREDA_ROL``.
- ``construir_disponibilidad``: ``disponibilidad_variables.csv`` (una fila por variable del universo) con el rol, si es evaluable y por qué no, y
  el faltante por año tomado de ``t22_perfil_columnas.csv``.

No importa ``panel.py`` (recibe el registro de variables como argumento) ni ``seleccion.py`` (recibe la evidencia ya calculada) para evitar ciclos.

La segunda parte del módulo (Tarea 11) aplica el árbol de decisión de ``config/arbol_decision.csv`` (``aplicar_arbol``, pura), arma la matriz con su contraste con la Fase 1
(``armar_matriz``), publica la versión que hereda la supresión de sus fuentes (``publicar_matriz``), escribe las dos tablas (``ejecutar``) y agrega la sección 8 del informe
(``agregar_secciones``). La matriz recomienda, la decisión es humana y todo lo clínico queda "pendiente de revisión clínica (decisión 12)".
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import polars as pl

from . import config, esquema, rutas
from . import conocimiento as modulo_conocimiento
from . import estadistica as est
from .informes import cifra

N_SECCIONES = 1

ROLES_CONOCIDOS = frozenset(
    {"FEATURE_BASAL", "TARGET", "FUENTE_DERIVADA", "FILTRO_COHORTE", "LLAVE_AGREGACION", "AUDITORIA_SENSIBILIDAD", "PROHIBIDA_FUGA", "DESCARTAR", "CONDICIONAL"}
)
ROLES_EXCLUYENTES = frozenset({"TARGET", "PROHIBIDA_FUGA", "LLAVE_AGREGACION", "FILTRO_COHORTE", "FUENTE_DERIVADA", "DESCARTAR", "AUDITORIA_SENSIBILIDAD"})
"""Roles que la regla R_FASE1 del árbol de decisión hace prevalecer sobre la estadística."""

ALIAS_NOMBRE = {"SCORE_VANWALRAVEN": "PUNTAJE_VAN_WALRAVEN"}
"""El documento de variables derivadas llama SCORE_VANWALRAVEN al puntaje que el código y ``conocimiento_clinico.csv`` llaman PUNTAJE_VAN_WALRAVEN."""

HEREDA_ROL = {
    "N_DX_SEC": "N_DX_TOTAL",
    "ENTRANTE_TIPO_PROCEDENCIA": "ES_TRASLADO_ENTRANTE",
    "N_PROCEDIMIENTOS": "PROCEDIMIENTO1",
    "N_TRASLADOS_INTERNOS": "FECHATRASLADO1",
    "ES_COHORTE_E": "ESTANCIA_DIAS",
    "SERVICIO_INGRESO_IGUAL_ALTA": "SERVICIOALTA",
}
"""Variables del panel que ningún documento declara: heredan el rol de la variable de la que salen. Son decisiones metodológicas, no clínicas;
el informe de la Tarea 11 las lista para que el equipo las confirme."""

VARIABLES_PANEL_SIN_DOCUMENTO = ("N_DX_SEC", "ENTRANTE_TIPO_PROCEDENCIA", "N_PROCEDIMIENTOS", "N_TRASLADOS_INTERNOS")
"""Variables del panel que el documento de derivadas no trae y que forman parte del universo (``origen = "derivada"``)."""

# prefijo de "¿Entra al modelo?" -> rol. Se prueban del más largo al más corto: "SÍ — es el resultado" debe ganarle a "SÍ".
PREFIJOS_ENTRA_AL_MODELO = {
    "SÍ — es el resultado": "TARGET",
    "SÍ — determina": "TARGET",
    "SÍ": "FEATURE_BASAL",
    "CONDICIONAL": "CONDICIONAL",
    "PENDIENTE": "CONDICIONAL",
    "NO — solo para uso interno": "LLAVE_AGREGACION",
    "NO — llave": "LLAVE_AGREGACION",
    "NO — solo análisis de sensibilidad": "AUDITORIA_SENSIBILIDAD",
    "NO — solo análisis de auditoría": "AUDITORIA_SENSIBILIDAD",
    "NO — solo análisis de equidad": "AUDITORIA_SENSIBILIDAD",
    "NO — solo filtro de cohorte": "FILTRO_COHORTE",
    "NO — marca de cohorte": "FILTRO_COHORTE",
}

# Fuentes crudas de cada variable derivada (explícito y con prueba de completitud: el faltante de una derivada es el de sus fuentes).
_DIAG = tuple(esquema.DIAG)
_ELIX = tuple(f"ELIX_{i:02d}" for i in range(1, 32))
FUENTES_DERIVADAS: dict[str, tuple[str, ...]] = {
    "ID_EPISODIO": (),
    "EDAD_ANIOS": ("FECHA_INGRESO", "FECHA_NACIMIENTO"),
    "ANIO_EGRESO": ("FECHAALTA",),
    "MES_INGRESO": ("FECHA_INGRESO",),
    "DIA_SEMANA": ("FECHA_INGRESO",),
    "ES_FIN_SEMANA": ("FECHA_INGRESO",),
    "FLAG_PANDEMIA": ("FECHAALTA",),
    "N_EGRESOS_12M": ("CIP_ENCRIPTADO", "FECHA_INGRESO", "FECHAALTA"),
    "DIAS_DESDE_EGRESO_PREVIO": ("CIP_ENCRIPTADO", "FECHA_INGRESO", "FECHAALTA"),
    "HISTORIA_DISPONIBLE": ("CIP_ENCRIPTADO", "FECHA_INGRESO"),
    "ESTANCIA": ("FECHAALTA", "FECHA_INGRESO"),
    "MORTALIDAD_BINARIA": ("TIPOALTA",),
    "ESTADO_CENSURA": ("TIPOALTA",),
    **{e: tuple(esquema.DIAG_SEC) for e in _ELIX},
    "PUNTAJE_VAN_WALRAVEN": tuple(esquema.DIAG_SEC),
    "PUNTAJE_VAN_WALRAVEN_SIN_5": tuple(esquema.DIAG_SEC),
    "N_COMORB_ELIX": tuple(esquema.DIAG_SEC),
    "N_DX_TOTAL": _DIAG,
    "GRUPO_CLINICO": ("DIAGNOSTICO1",),
    "CIE10_3C": ("DIAGNOSTICO1",),
    "DERIVADO_OTRO_HOSPITAL": ("HOSPPROCEDENCIA",),
    "NIVEL_CUIDADO_INGRESO": ("SERVICIOINGRESO",),
    "ESPECIALIDAD_MACRO": ("ESPECIALIDAD_MEDICA",),
    "PROCEDENCIA_AGR": ("TIPO_PROCEDENCIA",),
    "GRD_BASE": ("IR_29301_COD_GRD",),
    "MDC": ("IR_29301_COD_GRD",),
    "PUEBLO_ORIGINARIO": ("ETNIA",),
    "ES_EXTRANJERO": ("NACIONALIDAD",),
    "TRAMO_FONASA": ("PREVISION",),
    "INDICE_PRIVACION": ("COMUNA",),
    "HISTORIAL_INCOMPLETO": ("FECHA_INGRESO",),
    "ES_TRASLADO_ENTRANTE": ("HOSPPROCEDENCIA", "CIP_ENCRIPTADO"),
    # variables del panel que el documento de derivadas no trae
    "N_DX_SEC": tuple(esquema.DIAG_SEC),
    "ENTRANTE_TIPO_PROCEDENCIA": ("TIPO_PROCEDENCIA",),
    "N_PROCEDIMIENTOS": tuple(esquema.PROC),
    "N_TRASLADOS_INTERNOS": tuple(esquema.FECHAS_TRASLADO),
    # nombres que solo menciona reglas_variables.csv
    "ANIO_INGRESO": ("FECHA_INGRESO",),
    "IR_29301_PESO_NUM": ("IR_29301_PESO",),
    "ESTANCIA_DIAS": ("FECHAALTA", "FECHA_INGRESO"),
    "INGRESO_CRITICO": ("SERVICIOINGRESO",),
}

_VACIO_EXPLICITO = frozenset({"HOSPPROCEDENCIA", "FECHAINTERV1", "ESPECIALIDADINTERVENCION", "MEDICOINTERV1_ENCRIPTADO", "USOSPABELLON"})


def vacio_es_ausencia_cruda(columna: str) -> bool:
    """Verdadero si el vacío de la columna cruda significa ausencia del evento y no un dato faltante.

    Es el criterio de la Fase 1 para el umbral ``faltante_alto``: HOSPPROCEDENCIA, DIAGNOSTICO2 a 35, PROCEDIMIENTO1 a 30, los traslados
    internos, FECHAINTERV1, ESPECIALIDADINTERVENCION, MEDICOINTERV1_ENCRIPTADO, USOSPABELLON y el bloque de recién nacidos.
    """
    return (
        columna in _VACIO_EXPLICITO
        or columna in esquema.DIAG_SEC
        or columna in esquema.PROC
        or columna in esquema.FECHAS_TRASLADO
        or columna in esquema.SERVICIOS_TRASLADO
        or columna in esquema.BLOQUE_RN
    )


# ------------------------------------------------------------------------------------ carga de documentos
@lru_cache(maxsize=None)
def _reglas() -> dict[str, str]:
    """``reglas_variables.csv``: nombre -> categoría. Una categoría desconocida es un error fatal."""
    t = config.cargar_reglas_variables()
    faltan = {"variable", "categoria"} - set(t.columns)
    if faltan:
        raise ValueError(f"reglas_variables.csv no trae las columnas {sorted(faltan)}")
    malas = sorted(set(t["categoria"]) - ROLES_CONOCIDOS)
    if malas:
        raise ValueError(f"reglas_variables.csv trae categorías desconocidas: {malas}")
    return dict(zip(t["variable"], t["categoria"]))


@lru_cache(maxsize=None)
def _diccionario() -> dict[str, dict[str, str]]:
    """Diccionario de la Fase 2: nombre -> categoría, tipo de dato y uso. Sus 129 variables deben ser exactamente ``esquema.COLUMNAS``."""
    t = config.leer_csv_texto(rutas.DIR_FASE2 / "Fase_2__Diccionario_de_Base_de_Datos.csv")
    if set(t["Variable"]) != set(esquema.COLUMNAS) or t.height != len(esquema.COLUMNAS):
        raise ValueError("el diccionario de la Fase 2 debe tener exactamente las 129 variables de esquema.COLUMNAS")
    malas = sorted(set(t["Categoría Fase 1"]) - ROLES_CONOCIDOS)
    if malas:
        raise ValueError(f"el diccionario de la Fase 2 trae categorías desconocidas: {malas}")
    return {
        r["Variable"]: {"categoria": r["Categoría Fase 1"], "tipo_doc": r["Tipo de dato"], "uso": r["Uso en el proyecto"]}
        for r in t.iter_rows(named=True)
    }


def _expandir_nombre(texto: str) -> list[str]:
    """Nombres de variable de una fila del documento de derivadas ("ELIX_03 a ELIX_31" se expande a 29)."""
    texto = texto.strip()
    m = re.fullmatch(r"ELIX_(\d\d) a ELIX_(\d\d)", texto)
    if m:
        return [f"ELIX_{i:02d}" for i in range(int(m.group(1)), int(m.group(2)) + 1)]
    nombre = texto.split(" — ")[0].strip()
    return [ALIAS_NOMBRE.get(nombre, nombre)]


@lru_cache(maxsize=None)
def _derivadas() -> tuple[tuple[str, str], ...]:
    """Documento de derivadas expandido: (nombre, "¿Entra al modelo?") en el orden del documento."""
    t = config.leer_csv_texto(rutas.DIR_FASE2 / "Fase_2__Variables_Derivadas.csv")
    faltan = {"Variable nueva", "¿Entra al modelo?"} - set(t.columns)
    if faltan:
        raise ValueError(f"Fase_2__Variables_Derivadas.csv no trae las columnas {sorted(faltan)}")
    filas: list[tuple[str, str]] = []
    for r in t.iter_rows(named=True):
        uso = r["¿Entra al modelo?"].strip()
        traducir_prefijo(uso)          # un prefijo desconocido es un error fatal aunque la variable no se consulte
        filas += [(n, uso) for n in _expandir_nombre(r["Variable nueva"])]
    return tuple(filas)


def limpiar_cache() -> None:
    """Olvida los documentos leídos (las pruebas que cambian un documento llaman a esta función)."""
    _reglas.cache_clear()
    _diccionario.cache_clear()
    _derivadas.cache_clear()


def traducir_prefijo(uso: str) -> str:
    """Rol que corresponde a un valor de "¿Entra al modelo?", probando los prefijos del más largo al más corto."""
    for prefijo in sorted(PREFIJOS_ENTRA_AL_MODELO, key=len, reverse=True):
        if uso.startswith(prefijo):
            return PREFIJOS_ENTRA_AL_MODELO[prefijo]
    raise ValueError(f"decision.traducir_prefijo: el valor {uso!r} de '¿Entra al modelo?' no calza con ningún prefijo conocido")


# ----------------------------------------------------------------------------------------- rol y universo
def rol_fase1(nombre: str) -> tuple[str, str]:
    """Rol de la Fase 1 de una variable y la fuente de la que sale.

    Precedencia: (1) ``reglas_variables.csv``; (2) columna "Categoría Fase 1" del diccionario; (3) para derivadas, la columna
    "¿Entra al modelo?" traducida por prefijo; (4) herencia explícita ``HEREDA_ROL``. Una fila de derivadas declara su propio rol antes de heredar
    el de su fuente (MES_INGRESO es FEATURE_BASAL aunque FECHA_INGRESO sea FUENTE_DERIVADA). Un nombre sin rol lanza ``ValueError``.
    Devuelve ``(rol, fuente_rol)`` con ``fuente_rol`` igual a "reglas_variables", "diccionario", "derivadas" o "hereda:<variable>".
    """
    nombre = ALIAS_NOMBRE.get(nombre, nombre)
    reglas = _reglas()
    if nombre in reglas:
        return reglas[nombre], "reglas_variables"
    dic = _diccionario()
    if nombre in dic:
        return dic[nombre]["categoria"], "diccionario"
    for n, uso in _derivadas():
        if n == nombre:
            return traducir_prefijo(uso), "derivadas"
    if nombre in HEREDA_ROL:
        return rol_fase1(HEREDA_ROL[nombre])[0], f"hereda:{HEREDA_ROL[nombre]}"
    raise ValueError(f"decision.rol_fase1: no hay rol de la Fase 1 para {nombre!r}")


@lru_cache(maxsize=None)
def _universo() -> pl.DataFrame:
    dic = _diccionario()
    derivadas = _derivadas()
    filas = [
        {"variable": c, "origen": "cruda", "tipo_doc": dic[c]["tipo_doc"], "uso_modelo_doc": dic[c]["uso"]} for c in esquema.COLUMNAS
    ]
    filas += [{"variable": n, "origen": "derivada", "tipo_doc": "", "uso_modelo_doc": uso} for n, uso in derivadas]
    vistos = {f["variable"] for f in filas}
    filas += [{"variable": n, "origen": "derivada", "tipo_doc": "", "uso_modelo_doc": ""} for n in VARIABLES_PANEL_SIN_DOCUMENTO if n not in vistos]
    vistos = {f["variable"] for f in filas}
    filas += [{"variable": n, "origen": "regla_fase1", "tipo_doc": "", "uso_modelo_doc": ""} for n in _reglas() if n not in vistos]
    duplicados = sorted(n for n, k in Counter(f["variable"] for f in filas).items() if k > 1)
    if duplicados:
        raise ValueError(f"decision.cargar_universo: nombres duplicados en el universo: {duplicados}")
    return pl.DataFrame(filas, schema={"variable": pl.String, "origen": pl.String, "tipo_doc": pl.String, "uso_modelo_doc": pl.String})


def cargar_universo() -> pl.DataFrame:
    """Universo de variables que debe cubrir la matriz de decisión (DT09): 199 filas con la documentación actual.

    Las 129 columnas crudas (en el orden de ``esquema.COLUMNAS``), las 62 variables derivadas del documento (con "ELIX_03 a ELIX_31" expandido a 29
    nombres y SCORE_VANWALRAVEN renombrado a PUNTAJE_VAN_WALRAVEN), las 4 variables del panel que ese documento no trae y los 4 nombres que solo
    menciona ``reglas_variables.csv`` (``origen = "regla_fase1"``). Las columnas auxiliares del panel no pertenecen al universo.
    Columnas: ``variable``, ``origen``, ``tipo_doc`` y ``uso_modelo_doc`` (el "Uso en el proyecto" de las crudas o el "¿Entra al modelo?" de las
    derivadas, textual).
    """
    return _universo().clone()


# ------------------------------------------------------------------------------- no evaluables y motivos
_RE_PROC = re.compile(r"^PROCEDIMIENTO\d+$")
_RE_TRASLADO = re.compile(r"^(FECHA|SERVICIO)TRASLADO\d+$")
_RE_RN = re.compile(r"^(CONDICIONDEALTANEONATO|PESORN|SEXORN)\d$|^RN\dESTADO$")
_RE_DIAG_SEC = re.compile(r"^DIAGNOSTICO(?:[2-9]|[12]\d|3[0-5])$")

# (patrón, motivo, equivalente en el panel). Se prueban en orden; la primera que calza decide.
_NO_EVALUABLES: tuple[tuple[re.Pattern[str], str, str], ...] = tuple(
    (re.compile(p), m, e)
    for p, m, e in (
        (r"^FECHA_NACIMIENTO$", "se evalúa como EDAD_ANIOS", "EDAD_ANIOS"),
        (r"^FECHA_INGRESO$", "se evalúa como EDAD_ANIOS, MES_INGRESO, DIA_SEMANA y ES_FIN_SEMANA", "EDAD_ANIOS; MES_INGRESO; DIA_SEMANA; ES_FIN_SEMANA"),
        (r"^FECHAALTA$", "información del egreso (regla R1): solo se usa para calcular la estadía", "ESTANCIA_DIAS"),
        (r"^TIPOALTA$", "variable de resultado: define el desenlace y no es candidata", "y"),
        (r"^CIP_ENCRIPTADO$", "identificador: solo para el historial", "HISTORIA_DISPONIBLE"),
        (r"^TIPO_ACTIVIDAD$", "constante en la cohorte D: filtro de cohorte", ""),
        (r"^IR_29301_COD_GRD$", "filtro de cohorte: el agrupador se calcula con datos del alta", ""),
        (r"^DIAGNOSTICO1$", "se evalúa como GRUPO_CLINICO y CIE10_3C", "GRUPO_CLINICO; CIE10_3C"),
        (_RE_DIAG_SEC.pattern, "se evalúa mediante las categorías de Elixhauser y N_DX_SEC", "ELIX_01 a ELIX_31; N_DX_SEC"),
        (_RE_PROC.pattern, "se resume en N_PROCEDIMIENTOS (solo fuga)", "N_PROCEDIMIENTOS"),
        (_RE_TRASLADO.pattern, "se resume en N_TRASLADOS_INTERNOS (solo fuga)", "N_TRASLADOS_INTERNOS"),
        (r"^HOSPPROCEDENCIA$", "se evalúa como DERIVADO_OTRO_HOSPITAL", "DERIVADO_OTRO_HOSPITAL"),
        (r"^(FECHAINTERV1|ESPECIALIDADINTERVENCION|MEDICOINTERV1_ENCRIPTADO)$", "ocurre durante la hospitalización: rol PROHIBIDA_FUGA, no se construye en el panel", ""),
        (r"^MEDICOALTA_ENCRIPTADO$", "se genera al alta: rol PROHIBIDA_FUGA, no se construye en el panel", ""),
        (r"^FECHAPROCEDIMIENTO1$", "descartada por la Fase 1: solo 9 valores y con estructura de RUT", ""),
        (_RE_RN.pattern, "bloque de recién nacidos descartado por la Fase 1: no entra al panel", ""),
        (r"^ID_EPISODIO$", "identificador interno: solo sirve para cruces", ""),
        (r"^ANIO_EGRESO$", "llave de partición temporal: no es predictor", ""),
        (r"^ANIO_INGRESO$", "llave de partición temporal: un año como predictor no extrapola al año de prueba", ""),
        (r"^FLAG_PANDEMIA$", "solo análisis de sensibilidad: no se construye en el panel", ""),
        (r"^N_DX_TOTAL$", "solo auditoría: el panel construye N_DX_SEC", "N_DX_SEC"),
        (r"^ESTANCIA$", "variable de resultado: no es candidata", "ESTANCIA_DIAS"),
        (r"^MORTALIDAD_BINARIA$", "variable de resultado: es la etiqueta y del panel", "y"),
        (r"^ESTADO_CENSURA$", "variable de resultado: define qué episodios están en la cohorte D", "y"),
        (r"^(GRD_BASE|MDC)$", "filtro de cohorte: no es predictor", ""),
        (r"^HISTORIAL_INCOMPLETO$", "pendiente: depende de la aprobación del historial; no se construye en el panel", "HISTORIA_DISPONIBLE"),
        (r"^ES_TRASLADO_ENTRANTE$", "marca de cohorte (decisión pendiente): el panel usa ENTRANTE_TIPO_PROCEDENCIA", "ENTRANTE_TIPO_PROCEDENCIA"),
        (r"^IR_29301_PESO_NUM$", "se evalúa como IR_29301_PESO (solo fuga)", "IR_29301_PESO"),
        (r"^INGRESO_CRITICO$", "requiere la verificación f2_06 antes de usarse: no se construye en el panel", ""),
        (r"^ESPECIALIDAD_MACRO$", "no existe un mapeo de las especialidades a macro grupos", ""),
        (r"^TRAMO_FONASA$", "mapeo de PREVISION a tramos no definido en la documentación; solo las categorías institucionales A a D son directas", ""),
        (r"^INDICE_PRIVACION$", "exige una fuente externa de vulnerabilidad por comuna", ""),
    )
)


def _regla_no_evaluable(nombre: str) -> tuple[str, str] | None:
    for patron, motivo, equivalente in _NO_EVALUABLES:
        if patron.fullmatch(nombre):
            return motivo, equivalente
    return None


def motivo_no_evaluable(nombre: str, origen: str) -> str:
    """Motivo por el que una variable que el panel no construye como candidata no es evaluable (todo "no" lleva motivo).

    Aplica reglas por nombre o prefijo (todas las crudas que el panel resume en otra variable, las derivadas sin construcción y los nombres que solo
    menciona ``reglas_variables.csv``). Un nombre sin regla lanza ``ValueError``: un documento nuevo exige decidir su motivo, no inventarlo.
    """
    hallado = _regla_no_evaluable(ALIAS_NOMBRE.get(nombre, nombre))
    if hallado is None:
        raise ValueError(f"decision.motivo_no_evaluable: no hay motivo definido para {nombre!r} (origen {origen!r})")
    return hallado[0]


def equivalente_panel(nombre: str) -> str:
    """Variable(s) del panel que cubren a una columna cruda o derivada que el panel no analiza por separado (vacío si no hay)."""
    hallado = _regla_no_evaluable(ALIAS_NOMBRE.get(nombre, nombre))
    return hallado[1] if hallado else ""


# --------------------------------------------------------------------------------------- disponibilidad
ESQUEMA_DISPONIBILIDAD_BASE = {
    "variable": pl.String, "origen": pl.String, "tipo": pl.String, "tipo_doc": pl.String, "rol_fase1": pl.String, "fuente_rol": pl.String,
    "uso_modelo_doc": pl.String, "evaluable": pl.String, "motivo_no_evaluable": pl.String, "uso_panel": pl.String, "equivalente_panel": pl.String,
    "vacio_es_ausencia": pl.String,
}


def columnas_disponibilidad(anios: list[int]) -> list[str]:
    """Columnas de ``disponibilidad_variables.csv`` en su orden."""
    return [*ESQUEMA_DISPONIBILIDAD_BASE, *[f"faltante_{a}" for a in anios], "faltante_max_t22", "fuente_faltante"]


def _faltante_por_columna(perfil: pl.DataFrame, anios: list[int]) -> dict[str, dict[int, float | None]]:
    """``pct_faltante / 100`` de ``t22_perfil_columnas.csv`` por columna y año (nulo donde el perfil lo suprimió o no existe)."""
    out: dict[str, dict[int, float | None]] = {}
    for r in perfil.iter_rows(named=True):
        pct = r["pct_faltante"]
        out.setdefault(str(r["columna"]), {})[int(r["anio"])] = None if pct is None or (isinstance(pct, float) and np.isnan(pct)) else float(pct) / 100.0
    return out


def construir_disponibilidad(ctx, registro: Mapping[str, Any], perfil: pl.DataFrame | None = None) -> pl.DataFrame:
    """``disponibilidad_variables.csv``: una fila por variable del universo con rol, evaluabilidad y faltante por año.

    ``registro`` es el registro de variables del panel (``panel.registro_variables``; se usa por duck typing para no importar ``panel``).
    El faltante de las crudas es ``pct_faltante / 100`` de ``t22_perfil_columnas.csv`` (ese archivo está en porcentaje de 0 a 100; las tablas nuevas
    usan fracción de 0 a 1); el de las derivadas, el máximo del de sus fuentes sin contar las que tienen ``vacio_es_ausencia`` (si no queda
    ninguna, 0). ``faltante_max_t22`` es el máximo de los años de resultado y ``faltante_2024`` solo se muestra: ninguna regla lo usa. Si
    ``perfil`` es ``None`` se lee de la carpeta de tablas del contexto cuando existe; sin el archivo, las columnas de faltante quedan nulas y
    ``fuente_faltante = "no disponible"``.
    """
    particion = ctx.umbrales.particion
    anios = [int(a) for a in particion["anios_estructura_calidad"]]
    anios_resultado = {int(a) for a in particion["anios_resultado"]}
    if perfil is None and (ctx.dir_tablas / "t22_perfil_columnas.csv").is_file():
        perfil = ctx.leer_tabla("t22_perfil_columnas.csv", columnas_texto=["columna"])
    falt = _faltante_por_columna(perfil, anios) if perfil is not None else None

    def faltante_cruda(col: str, anio: int) -> float | None:
        return None if falt is None else falt.get(col, {}).get(anio)

    def faltante_derivada(fuentes: tuple[str, ...], anio: int) -> float | None:
        validas = [c for c in fuentes if not vacio_es_ausencia_cruda(c)]
        if falt is None:
            return None
        if not validas:
            return 0.0
        vals = [faltante_cruda(c, anio) for c in validas]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    filas = []
    for u in cargar_universo().iter_rows(named=True):
        nombre, origen = u["variable"], u["origen"]
        rol, fuente_rol = rol_fase1(nombre)
        reg = registro.get(nombre)
        if reg is not None and getattr(reg, "uso", "") == "resultado":
            evaluable, motivo = "no", "variable de resultado: no es candidata"
        elif reg is not None and getattr(reg, "es_referencia", False):
            evaluable, motivo = "no", "referencia del análisis de proxy, no es candidata"
        elif reg is not None:
            evaluable, motivo = reg.evaluable, reg.motivo
        else:
            evaluable, motivo = "no", motivo_no_evaluable(nombre, origen)
        if evaluable != "si" and not motivo:
            raise ValueError(f"decision.construir_disponibilidad: la variable {nombre!r} no es evaluable y no tiene motivo")
        vacio = vacio_es_ausencia_cruda(nombre) or bool(getattr(reg, "vacio_es_ausencia", False))
        if origen != "cruda" and nombre not in FUENTES_DERIVADAS:
            raise ValueError(f"decision.construir_disponibilidad: falta FUENTES_DERIVADAS para {nombre!r}")
        por_anio = {}
        for a in anios:
            por_anio[a] = faltante_cruda(nombre, a) if origen == "cruda" else faltante_derivada(FUENTES_DERIVADAS[nombre], a)
        maximo = [v for a, v in por_anio.items() if a in anios_resultado and v is not None]
        fila: dict[str, Any] = {
            "variable": nombre, "origen": origen, "tipo": getattr(reg, "tipo", "") if reg is not None else "", "tipo_doc": u["tipo_doc"],
            "rol_fase1": rol, "fuente_rol": fuente_rol, "uso_modelo_doc": u["uso_modelo_doc"], "evaluable": evaluable,
            "motivo_no_evaluable": motivo if evaluable != "si" else "", "uso_panel": getattr(reg, "uso", "") if reg is not None else "",
            "equivalente_panel": "" if reg is not None else equivalente_panel(nombre), "vacio_es_ausencia": "si" if vacio else "no",
        }
        fila.update({f"faltante_{a}": por_anio[a] for a in anios})
        fila["faltante_max_t22"] = max(maximo) if maximo else None
        fila["fuente_faltante"] = "no disponible" if falt is None else "t22"
        filas.append(fila)
    esquema_df = {**ESQUEMA_DISPONIBILIDAD_BASE, **{f"faltante_{a}": pl.Float64 for a in anios}, "faltante_max_t22": pl.Float64, "fuente_faltante": pl.String}
    return pl.DataFrame(filas, schema=esquema_df)


# =============================================================================================================================================
# Parte 2 (Tarea 11): árbol de decisión, matriz, publicación y escritura. La matriz recomienda; la decisión es humana.
# =============================================================================================================================================
CODIGOS_ARBOL = ("R_FASE1", "FUGA", "PROXY", "PROXY_MOD", "CALIDAD", "REDUNDANTE", "BAJA_SENAL", "CLINICO", "MANTENER")
"""Los nueve códigos de ``config/arbol_decision.csv`` en el orden en que se evalúan. Si el CSV no los trae exactamente así, ``aplicar_arbol`` lanza ``ValueError``."""
REGLA_FASE1_CONSERVAR = {f"ELIX_{i:02d}": "R3" for i in range(1, 32)}
"""La regla R3 de la Fase 1 dice que las 31 categorías de Elixhauser deben estar en el modelo principal; ``reglas_variables.csv`` no la codifica (todas son FEATURE_BASAL)."""
LEYENDA_R3 = "R3: las 31 categorías deben estar en el modelo principal"
ALIAS_CONOCIMIENTO = {"EDAD_ANIOS": "EDAD", "GRUPO_CLINICO": "CAPITULO_CIE10", "ESTANCIA": "ESTANCIA_DIAS"}
"""Nombre de la ficha de ``conocimiento_clinico.csv`` cuando difiere del de la variable."""
SIN_FICHA = "sin ficha"
PREFIJO_SIN_MEDICION = "sin medición: "
"""Comienzo de ``motivo_no_evaluable`` en una candidata evaluable cuya medición quedó incompleta; el resto del texto es la causa."""
SIN_MEDICION = "(ninguna)"
RECOMENDACION_NO_EVALUABLE = "NO EVALUABLE"
RECOMENDACION_NO_EVALUABLE_EXTRACTO = "NO EVALUABLE en extracto"
ETIQUETAS_CONTRASTE = (
    "coinciden", "fase1_excluye_estadistica_no_objeta", "fase1_excluye_sin_evidencia", "fase1_exige_conservar_estadistica_objeta", "fase1_exige_conservar_estadistica_no_objeta",
    "fase1_exige_conservar_sin_evidencia", "fase1_permite_estadistica_objeta", "fase1_permite_estadistica_no_objeta", "fase1_permite_no_evaluable",
)
"""Las nueve etiquetas de ``contraste_fase1``, exhaustivas y excluyentes."""
REGLAS_ADVERSAS = ("FUGA", "PROXY", "PROXY_MOD", "CALIDAD", "REDUNDANTE", "BAJA_SENAL")
"""La estadística objeta si alguna de estas reglas (2 a 7 del árbol) está activada, aunque CLINICO la compense después."""
BINES_RECOMENDACION = {
    "R_FASE1": "excluir_fase1", "FUGA": "excluir_fuga", "PROXY": "excluir_proxy", "PROXY_MOD": "condicional_proxy", "CALIDAD": "condicional_calidad", "REDUNDANTE": "redundante",
    "BAJA_SENAL": "baja_senal", "CLINICO": "mantener_clinico", "MANTENER": "mantener", SIN_MEDICION: "no_evaluable",
}
"""Cómo se agrupan las recomendaciones (por ``regla_decisiva``) en el resumen: ``n_recomendacion_<bin>``."""
V_RANKING, V_CONFIRMACION = "2019-2022", "2023"
"""Etiquetas de las dos ventanas de ``t23_ranking_variables.csv``. Son las del panel (``decision`` no lo importa): una prueba las compara con ``panel.V_RANK`` y ``panel.V_CONF``."""
COMP_CONFIRMACION, COMP_RESTO = "2019-2022 contra 2023", "2019 contra 2020-2022"
"""Etiquetas de las dos comparaciones de PSI de ``t23_estabilidad.csv`` (una prueba las compara con las que escribe ``seleccion.tabla_estabilidad``)."""

ESQUEMA_MATRIZ = {
    "variable": pl.String, "origen": pl.String, "tipo": pl.String, "rol_fase1": pl.String, "fuente_rol": pl.String, "restriccion_fase1": pl.String, "evaluable": pl.String,
    "motivo_no_evaluable": pl.String, "estado_medicion": pl.String, "medicion_completa": pl.String, "auc_ranking": pl.Float64, "auc_senal": pl.Float64, "auc_confirmacion": pl.Float64,
    "incremento_auc": pl.Float64, "asociacion_hospital": pl.Float64, "psi_vs_2023": pl.Float64, "psi_2019_vs_resto": pl.Float64, "psi_max": pl.Float64, "cambio_signo": pl.String,
    "faltante_max": pl.Float64, "fuente_faltante": pl.String, "faltante_informativo": pl.String, "cluster": pl.Int64, "tamano_cluster": pl.Int64, "es_representante": pl.String,
    "representante_cluster": pl.String, "margen_auc_vs_representante": pl.Float64, "certeza_clinica": pl.String, "regla_decisiva": pl.String, "reglas_activadas": pl.String,
    "recomendacion": pl.String, "contraste_fase1": pl.String, "valor_suprimido": pl.String, "evidencia": pl.String,
}
COLUMNAS_MATRIZ = list(ESQUEMA_MATRIZ)
COLUMNAS_ENTRADA_ARBOL = (
    "variable", "rol_fase1", "evaluable", "motivo_no_evaluable", "vacio_es_ausencia", "uso_panel", "analiza_resultado", "estado_medicion", "auc_ranking", "auc_senal", "auc_confirmacion",
    "incremento_auc", "incremento_estado", "asociacion_hospital", "psi_vs_2023", "psi_2019_vs_resto", "cambio_signo", "faltante_max", "cluster", "tamano_cluster",
)
"""Columnas que ``aplicar_arbol`` exige en la tabla de entrada (una fila por variable). ``origen``, ``tipo``, ``fuente_rol``, ``fuente_faltante``, ``faltante_informativo``
y ``evidencia`` son opcionales: pasan tal cual a la salida."""
_OPCIONALES_ARBOL = ("origen", "tipo", "fuente_rol", "fuente_faltante", "faltante_informativo", "evidencia")
_UMBRALES_ARBOL = (
    "auc_sospecha_fuga", "proxy_hospital_alto", "proxy_hospital_moderado", "faltante_alto", "psi_alto", "auc_senal_baja", "incremento_auc_minimo",
)


def _cargar_arbol(arbol: pl.DataFrame | None) -> dict[str, str]:
    """Textos de recomendación por código, leídos de ``arbol_decision.csv`` ordenado por ``orden``; exige los nueve códigos en el orden implementado."""
    t = arbol if arbol is not None else config.cargar_arbol_decision()
    t = t.sort(pl.col("orden").cast(pl.Int64))
    if tuple(t["codigo"].to_list()) != CODIGOS_ARBOL:
        raise ValueError("el árbol de decisión del CSV no coincide con las reglas implementadas")
    return dict(zip(t["codigo"].to_list(), t["recomendacion"].to_list()))


def _certezas(conocimiento: pl.DataFrame | None) -> dict[str, str]:
    t = conocimiento if conocimiento is not None else modulo_conocimiento.cargar()
    return dict(zip(t["variable"].to_list(), t["certeza"].to_list()))


def _num(x: Any) -> float | None:
    """Número finito o ``None``."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _pub(x: Any) -> float | None:
    """Valor redondeado a los decimales con que se publican las tablas: contra él se comparan todos los umbrales."""
    x = _num(x)
    return None if x is None else float(est.redondear_publicado(x))


def _medicion(f: Mapping[str, Any]) -> tuple[bool, str]:
    """``(completa, causa)``: si existe la evidencia que el árbol necesita para esa variable y, si no, por qué falta.

    Candidata que analiza el resultado: ``estado_medicion = "ok"``, ``auc_senal`` presente, ``incremento_auc`` presente o ``incremento_estado`` que empieza con "no aplica"
    (EDAD_ANIOS y SEXO, del modelo base), Y valores reales de ``asociacion_hospital`` y ``psi_vs_2023`` (NIT 11b: MANTENER no sale de una variable cuyo proxy o deriva no se
    pudo medir). Variable de solo fuga: ``auc_ranking`` o ``auc_confirmacion`` presente. ETNIA, PUEBLO_ORIGINARIO, COD_HOSPITAL, las que tienen ``evaluable = "no"`` y las que no
    están en el panel: no. La separación de un nivel raro no interviene.
    """
    if f["evaluable"] == "no":
        return False, ""
    if f["uso_panel"] == "solo_fuga":
        if _num(f["auc_ranking"]) is not None or _num(f["auc_confirmacion"]) is not None:
            return True, ""
        return False, str(f["estado_medicion"] or "sin AUC")
    if f["uso_panel"] != "candidata" or f["analiza_resultado"] != "si":
        return False, ""
    if f["estado_medicion"] != "ok":
        return False, str(f["estado_medicion"])
    if _num(f["auc_senal"]) is None:
        return False, "sin AUC"
    if _num(f["incremento_auc"]) is None and not str(f.get("incremento_estado") or "").startswith("no aplica"):
        return False, f"incremento {f.get('incremento_estado') or 'no calculado'}"
    if _num(f["asociacion_hospital"]) is None:
        return False, "sin asociación con el hospital"
    if _num(f["psi_vs_2023"]) is None:
        return False, "sin PSI contra 2023"
    return True, ""


def aplicar_arbol(base: pl.DataFrame, u, arbol: pl.DataFrame | None = None, conocimiento: pl.DataFrame | None = None) -> pl.DataFrame:
    """Aplica el árbol de decisión de ``config/arbol_decision.csv`` a una tabla con una fila por variable y devuelve la matriz con las columnas de ``COLUMNAS_MATRIZ``.

    Es la pieza más probada: pura (no lee datos ni escribe). Se evalúan TODAS las reglas y se guardan las activadas en el orden del árbol (``reglas_activadas``). Todas las
    comparaciones usan el valor redondeado a los decimales publicados (``redondear_publicado``): 1 R_FASE1 (rol excluyente; prevalece, también en no evaluables); 2 FUGA (el
    mayor entre ``auc_ranking`` y ``auc_confirmacion`` alcanza ``auc_sospecha_fuga``); 3 PROXY (``asociacion_hospital`` desde ``proxy_hospital_alto``); 4 PROXY_MOD (desde
    ``proxy_hospital_moderado`` y bajo el alto); 5 CALIDAD (faltante desde ``faltante_alto`` salvo que el vacío sea ausencia, PSI contra 2023 desde ``psi_alto`` o cambio de signo);
    6 REDUNDANTE (elegible, cluster de más de un miembro y no es el representante); 7 BAJA_SENAL (``auc_senal`` bajo ``auc_senal_baja`` e incremento bajo ``incremento_auc_minimo``,
    ambos presentes); 8 CLINICO (BAJA_SENAL y certeza "verificada en fuente": refinamiento de la 7); 9 MANTENER (ninguna de 1 a 8 y medición completa). Las reglas 2 a 9 no se activan
    en una variable con ``evaluable = "no"``.

    ``regla_decisiva``: R_FASE1 si está activa; si no, la primera activa entre FUGA, PROXY, PROXY_MOD, CALIDAD y REDUNDANTE; si no, CLINICO si está activa o BAJA_SENAL; si no, MANTENER
    con medición completa y, en otro caso, "(ninguna)" con recomendación "NO EVALUABLE" ("NO EVALUABLE en extracto" si el motivo empieza con "no evaluable en extracto"). El
    representante de un cluster de más de un miembro es el elegible (``evaluable != "no"`` y sin R_FASE1, FUGA ni PROXY) de mayor ``auc_senal`` publicado, luego menor
    ``faltante_max`` y luego nombre; los demás elegibles quedan REDUNDANTE y un cluster sin elegibles no activa REDUNDANTE. ``contraste_fase1`` toma una de las nueve etiquetas de
    ``ETIQUETAS_CONTRASTE``; ninguna variable sin medición completa recibe una etiqueta "no_objeta". La restricción R3 (``restriccion_fase1``) no cambia el texto de la
    recomendación ni el orden del árbol.
    """
    recomendaciones = _cargar_arbol(arbol)
    certezas = _certezas(conocimiento)
    faltan = [c for c in COLUMNAS_ENTRADA_ARBOL if c not in base.columns]
    if faltan:
        raise ValueError(f"aplicar_arbol: a la tabla de entrada le faltan las columnas {faltan}")
    umbral = {k: float(u[k]) for k in _UMBRALES_ARBOL}
    filas = base.to_dicts()
    salida: list[dict[str, Any]] = []
    for f in filas:
        nombre = f["variable"]
        completa, causa = _medicion(f)
        candidata_medible = f["evaluable"] != "no" and ((f["uso_panel"] == "candidata" and f["analiza_resultado"] == "si") or f["uso_panel"] == "solo_fuga")
        motivo = f["motivo_no_evaluable"] or ""
        if candidata_medible and not completa:
            motivo = f"{PREFIJO_SIN_MEDICION}{causa}"
        evaluable = f["evaluable"] != "no"
        adversos = [v for v in (_pub(f["auc_ranking"]), _pub(f["auc_confirmacion"])) if v is not None]
        ah, psi23, faltante = _pub(f["asociacion_hospital"]), _pub(f["psi_vs_2023"]), _pub(f["faltante_max"])
        auc_senal, incremento = _pub(f["auc_senal"]), _pub(f["incremento_auc"])
        certeza = certezas.get(ALIAS_CONOCIMIENTO.get(nombre, nombre), SIN_FICHA)
        r_fase1 = f["rol_fase1"] in ROLES_EXCLUYENTES
        fuga = evaluable and bool(adversos) and max(adversos) >= umbral["auc_sospecha_fuga"]
        proxy = evaluable and ah is not None and ah >= umbral["proxy_hospital_alto"]
        proxy_mod = evaluable and ah is not None and umbral["proxy_hospital_moderado"] <= ah < umbral["proxy_hospital_alto"]
        calidad = evaluable and (
            (faltante is not None and faltante >= umbral["faltante_alto"] and f["vacio_es_ausencia"] != "si")
            or (psi23 is not None and psi23 >= umbral["psi_alto"])
            or f["cambio_signo"] == "si"
        )
        baja = evaluable and auc_senal is not None and incremento is not None and auc_senal < umbral["auc_senal_baja"] and incremento < umbral["incremento_auc_minimo"]
        clinico = baja and certeza == modulo_conocimiento.CERTEZA_VERIFICADA
        psi_ambos = [v for v in (_num(f["psi_vs_2023"]), _num(f["psi_2019_vs_resto"])) if v is not None]
        salida.append({
            "variable": nombre, "origen": f.get("origen"), "tipo": f.get("tipo"), "rol_fase1": f["rol_fase1"], "fuente_rol": f.get("fuente_rol"),
            "restriccion_fase1": LEYENDA_R3 if nombre in REGLA_FASE1_CONSERVAR else "", "evaluable": f["evaluable"], "motivo_no_evaluable": motivo,
            "estado_medicion": f["estado_medicion"], "medicion_completa": "si" if completa else "no", "auc_ranking": _num(f["auc_ranking"]), "auc_senal": _num(f["auc_senal"]),
            "auc_confirmacion": _num(f["auc_confirmacion"]), "incremento_auc": _num(f["incremento_auc"]), "asociacion_hospital": _num(f["asociacion_hospital"]),
            "psi_vs_2023": _num(f["psi_vs_2023"]), "psi_2019_vs_resto": _num(f["psi_2019_vs_resto"]), "psi_max": max(psi_ambos) if psi_ambos else None,
            "cambio_signo": f["cambio_signo"], "faltante_max": _num(f["faltante_max"]), "fuente_faltante": f.get("fuente_faltante"), "faltante_informativo": f.get("faltante_informativo"),
            "cluster": f["cluster"], "tamano_cluster": f["tamano_cluster"], "certeza_clinica": certeza, "valor_suprimido": "no", "evidencia": f.get("evidencia") or "",
            "_evaluable": evaluable, "_r_fase1": r_fase1, "_fuga": fuga, "_proxy": proxy, "_proxy_mod": proxy_mod, "_calidad": calidad, "_baja": baja, "_clinico": clinico,
            "_completa": completa,
        })
    # --- representantes de los clusters de más de un miembro (entre los elegibles) --------------------------------------------------------------
    miembros: dict[Any, list[int]] = {}
    for i, f in enumerate(salida):
        if f["cluster"] is not None and (f["tamano_cluster"] or 0) > 1:
            miembros.setdefault(f["cluster"], []).append(i)
    for f in salida:
        f.update({"es_representante": "no aplica", "representante_cluster": "", "margen_auc_vs_representante": None, "_redundante": False})
        f["_elegible"] = f["_evaluable"] and not (f["_r_fase1"] or f["_fuga"] or f["_proxy"])
    for indices in miembros.values():
        elegibles = [i for i in indices if salida[i]["_elegible"]]
        rep = None
        if elegibles:
            def clave(i: int) -> tuple[float, float, str]:
                a, fm = _pub(salida[i]["auc_senal"]), _pub(salida[i]["faltante_max"])
                return (math.inf if a is None else -a, math.inf if fm is None else fm, salida[i]["variable"])

            rep = min(elegibles, key=clave)
        for i in indices:
            f = salida[i]
            f["representante_cluster"] = salida[rep]["variable"] if rep is not None else "(ninguno elegible)"
            if not f["_elegible"]:
                f["es_representante"] = "no elegible"
            elif i == rep:
                f["es_representante"] = "si"
            else:
                f["es_representante"], f["_redundante"] = "no", True
            if rep is not None and f["auc_senal"] is not None and salida[rep]["auc_senal"] is not None:
                f["margen_auc_vs_representante"] = f["auc_senal"] - salida[rep]["auc_senal"]
    # --- decisión, recomendación y contraste con la Fase 1 --------------------------------------------------------------------------------------
    for f in salida:
        activas = {
            "R_FASE1": f["_r_fase1"], "FUGA": f["_fuga"], "PROXY": f["_proxy"], "PROXY_MOD": f["_proxy_mod"], "CALIDAD": f["_calidad"], "REDUNDANTE": f["_redundante"],
            "BAJA_SENAL": f["_baja"], "CLINICO": f["_clinico"],
        }
        activas["MANTENER"] = f["_evaluable"] and f["_completa"] and not any(activas.values())
        f["reglas_activadas"] = "; ".join(c for c in CODIGOS_ARBOL if activas[c])
        if activas["R_FASE1"]:
            decisiva = "R_FASE1"
        else:
            primera = next((c for c in ("FUGA", "PROXY", "PROXY_MOD", "CALIDAD", "REDUNDANTE") if activas[c]), None)
            if primera is not None:
                decisiva = primera
            elif activas["CLINICO"]:
                decisiva = "CLINICO"
            elif activas["BAJA_SENAL"]:
                decisiva = "BAJA_SENAL"
            elif activas["MANTENER"]:
                decisiva = "MANTENER"
            else:
                decisiva = SIN_MEDICION
        f["regla_decisiva"] = decisiva
        if decisiva == SIN_MEDICION:
            f["recomendacion"] = RECOMENDACION_NO_EVALUABLE_EXTRACTO if f["motivo_no_evaluable"].startswith("no evaluable en extracto") else RECOMENDACION_NO_EVALUABLE
        else:
            f["recomendacion"] = recomendaciones[decisiva]
        objeta = any(activas[c] for c in REGLAS_ADVERSAS)
        if f["_r_fase1"]:
            etiqueta = "coinciden" if objeta else ("fase1_excluye_estadistica_no_objeta" if f["_completa"] else "fase1_excluye_sin_evidencia")
        elif f["restriccion_fase1"]:
            etiqueta = "fase1_exige_conservar_estadistica_objeta" if objeta else (
                "fase1_exige_conservar_estadistica_no_objeta" if f["_completa"] else "fase1_exige_conservar_sin_evidencia"
            )
        else:
            etiqueta = "fase1_permite_estadistica_objeta" if objeta else ("fase1_permite_estadistica_no_objeta" if decisiva == "MANTENER" else "fase1_permite_no_evaluable")
        f["contraste_fase1"] = etiqueta
    columnas = [{c: f.get(c) for c in COLUMNAS_MATRIZ} for f in salida]
    return pl.DataFrame(columnas, schema=ESQUEMA_MATRIZ)


# ------------------------------------------------------------------------------------------------------------------------- publicación
def publicar_matriz(matriz: pl.DataFrame, fuentes_pub: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Versión publicable de la matriz: anula cada estadístico cuya fila fuente quedó suprimida (celda pequeña) en su tabla fuente y marca ``valor_suprimido = "si"``.

    La matriz real decide con los valores reales (nunca con los CSV ya suprimidos); aquí solo se decide qué cifras se vuelven a publicar. ``fuentes_pub`` trae las versiones
    publicables ``ranking``, ``estabilidad``, ``proxy``, ``importancia`` y ``fuga`` (las que falten no anulan nada). Un estadístico se anula si la matriz lo trae y la celda fuente
    quedó nula: ``auc_ranking`` y ``auc_senal`` salen de la fila de 2019-2022 del ranking (o del bloque de 2019-2022 de la fuga en las variables de solo fuga); ``auc_confirmacion``
    de la fila de 2023 (o del bloque de 2023 de la fuga); ``incremento_auc`` del bloque de 2023 de la importancia; ``asociacion_hospital`` del bloque de 2019-2022 del proxy;
    ``psi_vs_2023`` y ``psi_2019_vs_resto`` de su fila de PSI y ``psi_max`` de cualquiera de las dos; ``faltante_max`` de las dos filas del ranking; y
    ``margen_auc_vs_representante`` de la fila de la variable o la de su representante. La recomendación, la regla decisiva y las demás columnas de decisión no cambian.
    """
    ranking = _indexar(fuentes_pub, "ranking", ("variable", "ventana"))
    proxy = _indexar(fuentes_pub, "proxy", ("variable",), "variable")
    importancia = _indexar(fuentes_pub, "importancia", ("variable",), "variable")
    fuga = _indexar(fuentes_pub, "fuga", ("variable",), "variable")
    psi = _indexar(fuentes_pub, "estabilidad", ("variable", "comparacion"), "psi")
    solo_fuga = {k[0] for k, f in fuga.items() if f.get("categoria") == "solo_fuga"}
    v_rank, v_conf = V_RANKING, V_CONFIRMACION
    comp_conf, comp_resto = COMP_CONFIRMACION, COMP_RESTO
    senal_pub = {k[0]: f.get("auc_senal") for k, f in ranking.items() if k[1] == v_rank}      # {variable: auc_senal publicable}; ``None`` = fila suprimida
    filas = matriz.to_dicts()
    for f in filas:
        nombre = f["variable"]
        anulados: list[str] = []

        def anular(columna: str, fuente: dict | None, campo: str) -> None:
            """Anula ``columna`` si la matriz trae el valor real y la celda de la fuente publicable quedó nula (fila suprimida)."""
            if f[columna] is not None and fuente is not None and fuente.get(campo) is None:
                f[columna] = None
                anulados.append(columna)

        es_solo_fuga = nombre in solo_fuga
        anular("auc_ranking", fuga.get((nombre,)) if es_solo_fuga else ranking.get((nombre, v_rank)), "auc_ranking" if es_solo_fuga else "auc")
        if not es_solo_fuga:
            anular("auc_senal", ranking.get((nombre, v_rank)), "auc_senal")
        anular("auc_confirmacion", fuga.get((nombre,)) if es_solo_fuga else ranking.get((nombre, v_conf)), "auc_confirmacion" if es_solo_fuga else "auc")
        anular("incremento_auc", importancia.get((nombre,)), "incremento_auc")
        anular("asociacion_hospital", proxy.get((nombre,)), "asociacion_hospital")
        anular("psi_vs_2023", psi.get((nombre, comp_conf)), "psi")
        anular("psi_2019_vs_resto", psi.get((nombre, comp_resto)), "psi")

        def celda_suprimida(fila: dict | None, campo: str) -> bool:
            return fila is not None and fila.get(campo) is None

        if f["psi_max"] is not None and (celda_suprimida(psi.get((nombre, comp_conf)), "psi") or celda_suprimida(psi.get((nombre, comp_resto)), "psi")):
            f["psi_max"] = None
            anulados.append("psi_max")
        if f["faltante_max"] is not None and f["fuente_faltante"] == "panel" and any(celda_suprimida(ranking.get((nombre, v)), "prop_faltante") for v in (v_rank, v_conf)):
            f["faltante_max"] = None
            anulados.append("faltante_max")
        if f["margen_auc_vs_representante"] is not None:
            rep = f["representante_cluster"]
            if (nombre in senal_pub and senal_pub[nombre] is None) or (rep in senal_pub and senal_pub[rep] is None):
                f["margen_auc_vs_representante"] = None
                anulados.append("margen_auc_vs_representante")
        f["valor_suprimido"] = "si" if anulados else "no"
    return pl.DataFrame(filas, schema=ESQUEMA_MATRIZ)


# ------------------------------------------------------------------------------------------------------------------ armado y escritura
_DOCUMENTO_ROL = {
    "reglas_variables": "reglas_variables.csv", "diccionario": "Fase_2__Diccionario_de_Base_de_Datos.csv", "derivadas": "Fase_2__Variables_Derivadas.csv",
}
HISTORIALES = ("N_EGRESOS_12M", "DIAS_DESDE_EGRESO_PREVIO", "HISTORIA_DISPONIBLE")


def _documento_del_rol(fuente_rol: str) -> str:
    """Documento de la Fase 1 del que sale el rol de una variable (la herencia remite al documento de la variable de la que hereda)."""
    if fuente_rol.startswith("hereda:"):
        return _documento_del_rol(rol_fase1(fuente_rol.split(":", 1)[1])[1])
    return _DOCUMENTO_ROL.get(fuente_rol, "")


def _indexar(evidencia: Mapping[str, pl.DataFrame], clave: str, columnas: Sequence[str], tipo_fila: str | None = None) -> dict[tuple, dict[str, Any]]:
    """Filas de ``evidencia[clave]`` por los valores de ``columnas`` (vacío si la tabla no está o no tiene filas); con ``tipo_fila`` filtra las filas de ese tipo."""
    t = evidencia.get(clave)
    if t is None or t.height == 0:
        return {}
    if tipo_fila is not None:
        t = t.filter(pl.col("tipo_fila") == tipo_fila)
    return {tuple(f[c] for c in columnas): f for f in t.iter_rows(named=True)}


def armar_matriz(ctx, panel, evidencia: Mapping[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Arma ``disponibilidad_variables.csv`` y la matriz de decisión (real y publicable) a partir de la evidencia ya calculada. No escribe ni lee datos.

    ``evidencia`` trae las tablas reales ``ranking``, ``estabilidad``, ``proxy``, ``importancia``, ``fuga`` y ``clusters`` (la de la Tarea 10) y sus versiones publicables con la
    clave terminada en ``_pub``. Devuelve ``disponibilidad``, ``matriz`` (con los valores reales, la que decide) y ``matriz_pub`` (la que se escribe). ``estado_medicion`` es el
    ``estado`` del ranking en 2019-2022, el de ``t23_fuga.csv`` en las variables de solo fuga, "no analiza resultado" en ETNIA y PUEBLO_ORIGINARIO, "referencia" en COD_HOSPITAL y
    "sin fila de medición" en otro caso. ``faltante_max`` es el máximo entre el faltante del ranking en 2019-2022 y en 2023 en las candidatas del panel (``fuente_faltante = "panel"``) y
    ``faltante_max_t22`` en el resto, incluidas ETNIA y PUEBLO_ORIGINARIO (``fuente_faltante = "t22"``).
    """
    disponibilidad = evidencia.get("disponibilidad")
    if disponibilidad is None:
        disponibilidad = construir_disponibilidad(ctx, panel.variables)
    ranking = _indexar(evidencia, "ranking", ("variable", "ventana"))
    proxy = {k[0]: f for k, f in _indexar(evidencia, "proxy", ("variable",), "variable").items()}
    importancia = {k[0]: f for k, f in _indexar(evidencia, "importancia", ("variable",), "variable").items()}
    fuga = {k[0]: f for k, f in _indexar(evidencia, "fuga", ("variable",), "variable").items()}
    resumen = {k[0]: f for k, f in _indexar(evidencia, "estabilidad", ("variable",), "resumen").items()}
    psi = _indexar(evidencia, "estabilidad", ("variable", "comparacion"), "psi")
    en_estabilidad = {k[0] for k in _indexar(evidencia, "estabilidad", ("variable", "tipo_fila", "anio", "comparacion"))}
    clusters = {k[0]: f for k, f in _indexar(evidencia, "clusters", ("variable",)).items()}
    modo_completo = ctx.etiqueta == "completo"
    filas: list[dict[str, Any]] = []
    for d in disponibilidad.iter_rows(named=True):
        nombre = d["variable"]
        reg = panel.variables.get(nombre)
        uso = getattr(reg, "uso", "") if reg is not None else ""
        analiza = ("si" if getattr(reg, "analiza_resultado", True) else "no") if reg is not None else ""
        r_rank, r_conf = ranking.get((nombre, V_RANKING)), ranking.get((nombre, V_CONFIRMACION))
        if uso == "solo_fuga":
            estado = fuga[nombre]["estado"] if nombre in fuga else "sin fila de medición"
        elif uso == "referencia":
            estado = "referencia"
        elif uso == "candidata" and analiza == "no":
            estado = "no analiza resultado"
        elif uso == "candidata" and r_rank is not None:
            estado = r_rank["estado"]
        else:
            estado = "sin fila de medición"
        if uso == "solo_fuga":
            auc_ranking = fuga.get(nombre, {}).get("auc_ranking")
            auc_conf = fuga.get(nombre, {}).get("auc_confirmacion")
            auc_senal = None
        else:
            auc_ranking = r_rank["auc"] if r_rank is not None else None
            auc_conf = r_conf["auc"] if r_conf is not None else None
            auc_senal = r_rank["auc_senal"] if r_rank is not None else None
        en_panel_ranking = uso == "candidata" and r_rank is not None
        if en_panel_ranking:
            vals = [x for x in (r_rank["prop_faltante"], r_conf["prop_faltante"] if r_conf is not None else None) if x is not None]
            faltante_max, fuente_faltante = (max(vals) if vals else None), "panel"
        else:
            faltante_max, fuente_faltante = d["faltante_max_t22"], d["fuente_faltante"]
        c = clusters.get(nombre)
        tablas = ["disponibilidad_variables.csv", _documento_del_rol(d["fuente_rol"])]
        if (nombre, V_RANKING) in ranking:
            tablas.append("t23_ranking_variables.csv")
        if nombre in en_estabilidad:
            tablas.append("t23_estabilidad.csv")
        if nombre in proxy:
            tablas.append("t23_proxy_hospital.csv")
        if c is not None:
            tablas += ["matriz_asociacion.csv", "clusters_variables.csv"]
        if nombre in importancia:
            tablas.append("t23_importancia.csv")
        if nombre in REGLA_FASE1_CONSERVAR:
            tablas += ["t23_or_elix.csv", "t23_sensibilidad_elix.csv"]
        if nombre in fuga:
            tablas.append("t23_fuga.csv")
        if nombre == "DERIVADO_OTRO_HOSPITAL":
            tablas.append("t23_procedencia_autorref.csv")
        if modo_completo and nombre in HISTORIALES and d["evaluable"] != "no":
            tablas.append("t23_historial_factibilidad.csv")
        filas.append({
            "variable": nombre, "origen": d["origen"], "tipo": d["tipo"], "rol_fase1": d["rol_fase1"], "fuente_rol": d["fuente_rol"], "evaluable": d["evaluable"],
            "motivo_no_evaluable": d["motivo_no_evaluable"], "vacio_es_ausencia": d["vacio_es_ausencia"], "uso_panel": uso, "analiza_resultado": analiza, "estado_medicion": estado,
            "auc_ranking": auc_ranking, "auc_senal": auc_senal, "auc_confirmacion": auc_conf, "incremento_auc": importancia.get(nombre, {}).get("incremento_auc"),
            "incremento_estado": importancia.get(nombre, {}).get("incremento_estado"), "asociacion_hospital": proxy.get(nombre, {}).get("asociacion_hospital"),
            "psi_vs_2023": psi.get((nombre, COMP_CONFIRMACION), {}).get("psi"), "psi_2019_vs_resto": psi.get((nombre, COMP_RESTO), {}).get("psi"),
            "cambio_signo": resumen.get(nombre, {}).get("cambio_signo"), "faltante_max": faltante_max, "fuente_faltante": fuente_faltante,
            "faltante_informativo": r_rank["faltante_informativo"] if r_rank is not None else None, "cluster": c["cluster"] if c is not None else None,
            "tamano_cluster": c["tamano_cluster"] if c is not None else None, "evidencia": "; ".join(t for t in dict.fromkeys(tablas) if t),
        })
    esquema_base = {
        "variable": pl.String, "origen": pl.String, "tipo": pl.String, "rol_fase1": pl.String, "fuente_rol": pl.String, "evaluable": pl.String, "motivo_no_evaluable": pl.String,
        "vacio_es_ausencia": pl.String, "uso_panel": pl.String, "analiza_resultado": pl.String, "estado_medicion": pl.String, "auc_ranking": pl.Float64, "auc_senal": pl.Float64,
        "auc_confirmacion": pl.Float64, "incremento_auc": pl.Float64, "incremento_estado": pl.String, "asociacion_hospital": pl.Float64, "psi_vs_2023": pl.Float64,
        "psi_2019_vs_resto": pl.Float64, "cambio_signo": pl.String, "faltante_max": pl.Float64, "fuente_faltante": pl.String, "faltante_informativo": pl.String, "cluster": pl.Int64,
        "tamano_cluster": pl.Int64, "evidencia": pl.String,
    }
    base = pl.DataFrame(filas, schema=esquema_base)
    matriz = aplicar_arbol(base, ctx.umbrales)
    fuentes_pub = {k: evidencia[f"{k}_pub"] for k in ("ranking", "estabilidad", "proxy", "importancia", "fuga") if evidencia.get(f"{k}_pub") is not None}
    return {"disponibilidad": disponibilidad, "matriz": matriz, "matriz_pub": publicar_matriz(matriz, fuentes_pub)}


def ejecutar(ctx, panel, evidencia: Mapping[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Escribe ``disponibilidad_variables.csv`` y ``matriz_decision_variables.csv`` (la versión publicable, sin ``cols_n``: la matriz no tiene conteos de casos y hereda la supresión de
    sus fuentes) y devuelve lo que arma ``armar_matriz``. Recibe la evidencia ya calculada; no la calcula, para no importar ``seleccion`` y evitar un ciclo."""
    r = armar_matriz(ctx, panel, evidencia)
    ctx.escribir_tabla(
        "disponibilidad_variables.csv", r["disponibilidad"], tarea="T11",
        descripcion="Universo de variables (crudas, derivadas y de reglas de la Fase 1) con su rol, si son evaluables y por qué no, y el faltante por año de la Tarea 4",
    )
    ctx.escribir_tabla(
        "matriz_decision_variables.csv", r["matriz_pub"], tarea="T11",
        descripcion="Matriz de decisión por variable: reglas del árbol activadas, regla decisiva, recomendación y contraste con la Fase 1 (la matriz recomienda; la decisión es humana)",
    )
    return r


# --------------------------------------------------------------------------------------------------------------------------- resumen
def resumen_matriz(matriz: pl.DataFrame, matriz_pub: pl.DataFrame | None = None) -> dict[str, int]:
    """Conteos de la matriz para ``t23_resumen_seleccion.csv``: ``n_matriz``, ``n_recomendacion_<grupo>`` (los diez de ``BINES_RECOMENDACION``), ``n_contraste_<etiqueta>`` (las nueve
    de ``ETIQUETAS_CONTRASTE``), ``n_restriccion_r3``, ``n_valor_suprimido`` y ``n_sin_medicion``.

    Los conteos de variables no son conteos de casos, así que salen de la matriz real; ``n_valor_suprimido`` cuenta las filas de la versión publicable con alguna cifra anulada.
    ``n_recomendacion_no_evaluable`` incluye las candidatas sin medición completa y ``n_sin_medicion`` las separa (``motivo_no_evaluable`` empieza con ``PREFIJO_SIN_MEDICION``).
    """
    publicable = matriz if matriz_pub is None else matriz_pub
    por_regla = Counter(matriz["regla_decisiva"].to_list())
    por_etiqueta = Counter(matriz["contraste_fase1"].to_list())
    r: dict[str, int] = {"n_matriz": matriz.height}
    for regla, grupo in BINES_RECOMENDACION.items():
        r[f"n_recomendacion_{grupo}"] = por_regla.get(regla, 0)
    for etiqueta in ETIQUETAS_CONTRASTE:
        r[f"n_contraste_{etiqueta}"] = por_etiqueta.get(etiqueta, 0)
    r["n_restriccion_r3"] = sum(1 for x in matriz["restriccion_fase1"].to_list() if x)
    r["n_valor_suprimido"] = sum(1 for x in publicable["valor_suprimido"].to_list() if x == "si")
    r["n_sin_medicion"] = sum(1 for x in matriz["motivo_no_evaluable"].to_list() if x and x.startswith(PREFIJO_SIN_MEDICION))
    return r


# ----------------------------------------------------------------------------------------------------------------------------- informe
_TEXTOS_MATRIZ = [c for c, t in ESQUEMA_MATRIZ.items() if t == pl.String]
_ENCABEZADOS_CONTRASTE = {
    "variable": "Variable", "rol_fase1": "Rol en la Fase 1", "reglas_activadas": "Reglas activadas", "regla_decisiva": "Regla decisiva", "recomendacion": "Recomendación",
}


def agregar_secciones(inf, ctx, n0: int = 1) -> None:
    """Agrega la sección 8 del informe de la Tarea 11 (``N_SECCIONES = 1``): la matriz de decisión y su contraste con la Fase 1.

    Las cifras salen de ``t23_resumen_seleccion.csv`` (``cifra``) y de la propia matriz, los umbrales de ``descripcion.Cifra_umbral`` y las tablas se incrustan con
    ``inf.tabla(df=...)`` desde ``matriz_decision_variables.csv``. Contiene el conteo por recomendación (con las candidatas que quedaron sin medición y sus causas), las tres
    tablas de contraste con la Fase 1, la frase de la restricción R3, la evaluación de H01c (en extracto no es evaluable; con los historiales evaluables, una tabla con su factibilidad,
    incremento y proxy) y la frase "la matriz recomienda; la decisión es humana". La prosa son plantillas con marcadores: sin dígitos sueltos.
    """
    from .descripcion import Cifra_umbral

    def rs(col: str):
        return cifra(ctx, "t23_resumen_seleccion.csv", col)

    matriz = ctx.leer_tabla("matriz_decision_variables.csv", columnas_texto=_TEXTOS_MATRIZ)

    # --- el árbol y las recomendaciones ------------------------------------------------------------------------------------------------------
    inf.seccion(f"{n0}. Matriz de decisión y contraste con la Fase 1")
    inf.parrafo(
        "La matriz de decisión (`matriz_decision_variables.csv`) tiene una fila por cada una de las {n_matriz} variables del universo y aplica el árbol de `arbol_decision.csv`. "
        "Se evalúan las nueve reglas para cada variable y se guardan todas las que se activan; la regla decisiva es la primera que corresponde en este orden: la exclusión por la "
        "Fase 1, la sospecha de fuga, el proxy del hospital (alto y moderado), la calidad o la estabilidad, la redundancia, la baja señal y, como refinamiento de esta, el criterio "
        "clínico. Si no se activa ninguna y la medición está completa, la variable se mantiene. Todas las comparaciones usan los valores redondeados que se publican en las tablas. "
        "La matriz recomienda; la decisión es humana: ninguna columna se llama decisión y todo lo clínico queda pendiente de revisión clínica (decisión 12).",
        n_matriz=rs("n_matriz"),
    )
    inf.parrafo("La tabla siguiente reparte las variables según la recomendación de la matriz; el texto de cada recomendación sale de `arbol_decision.csv`.")
    textos = _cargar_arbol(None)
    por_recomendacion = [(textos[c], c, rs(f"n_recomendacion_{g}").valor) for c, g in BINES_RECOMENDACION.items() if c != SIN_MEDICION]
    por_recomendacion.append((RECOMENDACION_NO_EVALUABLE, SIN_MEDICION, rs("n_recomendacion_no_evaluable").valor))
    inf.tabla(
        "matriz_decision_variables.csv",
        df=pl.DataFrame({"recomendacion": [f[0] for f in por_recomendacion], "regla": [f[1] for f in por_recomendacion], "n_variables": [f[2] for f in por_recomendacion]}),
        columnas=["recomendacion", "regla", "n_variables"], encabezados={"recomendacion": "Recomendación", "regla": "Regla decisiva", "n_variables": "Variables"},
        leyenda="Variables por recomendación de la matriz",
    )

    # --- candidatas que no se pudieron medir -------------------------------------------------------------------------------------------------------------
    sin_medicion = matriz.filter(pl.col("motivo_no_evaluable").fill_null("").str.starts_with(PREFIJO_SIN_MEDICION))
    if sin_medicion.height:
        inf.parrafo(
            "Entre las variables no evaluables hay candidatas del panel que no se pudieron medir por completo en esta corrida (en total, {n_sin}): la variable es constante, no tiene "
            "ambos desenlaces o le falta algún componente de la medición. No reciben «mantener» ni la lectura de que la estadística no objeta, porque no se midieron; si alguna "
            "regla adversa se activa con la evidencia que sí existe, esa regla decide. Las causas están en la tabla siguiente.",
            n_sin=rs("n_sin_medicion"),
        )
        causas = (
            sin_medicion.with_columns(pl.col("motivo_no_evaluable").str.strip_prefix(PREFIJO_SIN_MEDICION).alias("causa"))
            .group_by("causa")
            .agg(pl.len().alias("n_variables"), pl.col("variable").sort().str.join(", ").alias("variables"))
            .sort(["n_variables", "causa"], descending=[True, False])
        )
        inf.tabla(
            "matriz_decision_variables.csv", df=causas, columnas=["causa", "n_variables", "variables"],
            encabezados={"causa": "Causa de la falta de medición", "n_variables": "Variables", "variables": "Miembros"}, leyenda="Candidatas sin medición completa, por causa",
        )
    else:
        inf.parrafo("Todas las candidatas evaluables del panel se midieron por completo en esta corrida.")

    # --- contraste con la Fase 1 ----------------------------------------------------------------------------------------------------------------------
    inf.parrafo(
        "El contraste con la Fase 1 compara lo que esa fase ya decidió con lo que la estadística objeta. Se dice que la estadística objeta cuando se activa alguna de las reglas de "
        "fuga, proxy, calidad, redundancia o baja señal, aunque el criterio clínico la compense después. Variables que la Fase 1 excluye y la estadística también objeta "
        "(coinciden): {coin}. Variables que la Fase 1 excluye y la estadística no objeta, solo las medidas por completo: {c1}. Variables que la Fase 1 permite y la estadística "
        "objeta: {c2}. Categorías de Elixhauser que la Fase 1 exige conservar y la estadística objeta: {c3}. Las tablas siguientes muestran cada caso.",
        coin=rs("n_contraste_coinciden"), c1=rs("n_contraste_fase1_excluye_estadistica_no_objeta"), c2=rs("n_contraste_fase1_permite_estadistica_objeta"),
        c3=rs("n_contraste_fase1_exige_conservar_estadistica_objeta"),
    )
    for etiqueta, leyenda, tope in (
        ("fase1_excluye_estadistica_no_objeta", "La Fase 1 excluye y la estadística no objeta (solo variables medidas por completo)", 30),
        ("fase1_permite_estadistica_objeta", "La Fase 1 permite y la estadística objeta", 30),
        ("fase1_exige_conservar_estadistica_objeta", "La Fase 1 exige conservar y la estadística objeta", None),
    ):
        filtrada = matriz.filter(pl.col("contraste_fase1") == etiqueta)
        if filtrada.height:
            inf.tabla(
                "matriz_decision_variables.csv", df=filtrada, columnas=list(_ENCABEZADOS_CONTRASTE), encabezados=_ENCABEZADOS_CONTRASTE, max_filas=tope, leyenda=leyenda,
            )
    inf.parrafo(
        "Las categorías de Elixhauser llevan en `restriccion_fase1` la regla R3 de la Fase 1: las {n_r3} deben estar en el modelo principal. Es una exigencia de la Fase 1 y no una "
        "discrepancia con la estadística: aunque la matriz proponga eliminar una de ellas por baja señal o condicionarla por calidad, el texto de la recomendación no cambia por "
        "esta restricción y la etiqueta de contraste le indica a quien decide que debe conservarse por la Fase 1.",
        n_r3=rs("n_restriccion_r3"),
    )
    if rs("n_valor_suprimido").valor:
        inf.parrafo(
            "Filas con alguna cifra anulada en la matriz publicada: {n_sup}. La fila de su tabla fuente quedó suprimida por tener pocos casos; la regla decisiva y la "
            "recomendación de esas filas se calcularon con los valores reales y no cambian.",
            n_sup=rs("n_valor_suprimido"),
        )

    # --- H01c: el historial ---------------------------------------------------------------------------------------------------------------------------------
    historiales = matriz.filter(pl.col("variable").is_in(HISTORIALES))
    evaluables = historiales.filter(pl.col("evaluable") != "no")
    if evaluables.height == 0:
        motivo = cifra(ctx, "matriz_decision_variables.csv", "motivo_no_evaluable", columnas_texto=_TEXTOS_MATRIZ, variable=HISTORIALES[0])
        inf.parrafo(
            "La hipótesis H01c pregunta si `N_EGRESOS_12M` y `DIAS_DESDE_EGRESO_PREVIO` aportan señal sobre la edad y el sexo sin ser un proxy del hospital. En esta corrida no es "
            "evaluable; el motivo registrado en la matriz es «{motivo|texto}». El criterio de la hipótesis pide que la factibilidad del historial se cumpla, que el incremento de AUC "
            "supere {inc|dec3} y que la asociación con el hospital sea menor que {prox|dec2}; con los datos completos se evalúa a partir de esta matriz y de la factibilidad de la Tarea 9.",
            motivo=motivo, inc=Cifra_umbral(ctx, "incremento_auc_minimo"), prox=Cifra_umbral(ctx, "proxy_hospital_moderado"),
        )
    else:
        tabla_h = historiales.select("variable", "incremento_auc", "asociacion_hospital", "regla_decisiva", "recomendacion")
        cifras_h: dict[str, Any] = {"inc": Cifra_umbral(ctx, "incremento_auc_minimo"), "prox": Cifra_umbral(ctx, "proxy_hospital_moderado")}
        frase_fact = ""
        if (ctx.dir_tablas / "t23_historial_factibilidad.csv").is_file():
            fact = ctx.leer_tabla("t23_historial_factibilidad.csv", columnas_texto=["regimen", "cumple_minimo"]).sort("anio")
            if fact.height:
                anio = int(fact["anio"][-1])
                cifras_h.update(
                    anio=cifra(ctx, "t23_historial_factibilidad.csv", "anio", columnas_texto=["regimen", "cumple_minimo"], anio=anio),
                    fact=cifra(ctx, "t23_historial_factibilidad.csv", "factibilidad", columnas_texto=["regimen", "cumple_minimo"], anio=anio),
                    minimo=Cifra_umbral(ctx, "historial_factibilidad_minima"),
                )
                frase_fact = " La factibilidad del historial en {anio|texto} es {fact|prop1}, frente a un mínimo de {minimo|prop1}."
        inf.parrafo(
            "La hipótesis H01c pregunta si `N_EGRESOS_12M` y `DIAS_DESDE_EGRESO_PREVIO` aportan señal sobre la edad y el sexo sin ser un proxy del hospital. La tabla reúne, para cada "
            "variable de historial, el incremento de AUC sobre la edad y el sexo y la asociación con el hospital." + frase_fact + " El criterio de la hipótesis recomienda incluirlas "
            "solo si la factibilidad se cumple, el incremento supera {inc|dec3} y la asociación con el hospital es menor que {prox|dec2}; el veredicto lo calcula la Tarea 12 con "
            "ese criterio.",
            **cifras_h,
        )
        inf.tabla(
            "matriz_decision_variables.csv", df=tabla_h, columnas=["variable", "incremento_auc", "asociacion_hospital", "regla_decisiva", "recomendacion"],
            encabezados={"variable": "Variable", "incremento_auc": "Incremento de AUC", "asociacion_hospital": "Asociación con el hospital", "regla_decisiva": "Regla decisiva",
                         "recomendacion": "Recomendación"},
            formatos={"incremento_auc": "dec3", "asociacion_hospital": "dec3"}, leyenda="Variables de historial: incremento, proxy y recomendación de la matriz",
        )
    inf.parrafo("La matriz recomienda; la decisión es humana.")


# FIN-decision
