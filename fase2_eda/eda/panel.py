"""Panel compartido de las Tareas 10 y 11: una sola lectura de 2019 a 2023 y las variables canónicas del análisis.

El panel lee una vez los años de resultado (nunca el año expuesto), reutiliza ``cohorte.tipar``, ``cohorte.agregar_comorbilidad`` y ``cohorte.marcar`` para
quedarse con la cohorte D (adultos hospitalizados de la cohorte dura con desenlace conocido), muestrea de forma reproducible hasta
``max_filas_asociacion_por_anio`` filas por año, calcula las variables, aplica el tope de ``cardinalidad_max_categorica`` niveles a las categóricas y conserva
un registro único de metadatos por variable que usan las dos tareas y el generador de ``disponibilidad_variables.csv``.

Reglas que este módulo hace cumplir:

- Es el ÚNICO punto de lectura de datos de las Tareas 10 y 11: ``asociacion.py`` y ``seleccion.py`` solo consumen el panel, de modo que la regla de que 2024 no
  se lee se cumple y se prueba en un lugar.
- El hospital se conserva SOLO para el análisis de proxy; nunca es un predictor.
- Nada se imputa (regla R5): los nulos de numéricas y binarias se conservan y el faltante de una categórica es el nivel "(sin dato)".
- El panel no conserva identificadores, fechas, nombres de hospital ni textos libres: ``id_num``, ``ing`` y ``alta`` viven solo durante la construcción.
- Los niveles de las categóricas se fijan con la ventana de ranking (2019 a 2022); un nivel que solo aparece en 2023 pasa a OTROS.

Dos mapeos quedan "pendientes de revisión clínica (decisión 12)" y se marcan así en el código y en los informes: ``MAPA_PROCEDENCIA_AGR`` (qué procedencias
son urgencia, derivación o programación) y ``PATRONES_CUIDADO_ESPECIAL`` (qué unidades cuentan como cuidados especiales). También lo está la codificación de
PUEBLO_ORIGINARIO a partir de ETNIA.

Tablas: ``t23_panel_muestreo.csv`` y ``t23_panel_niveles.csv``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from . import catalogos, cohorte, config, decision, elixhauser, esquema, hospitales, supresion
from . import transformaciones as tr
from .contexto import Contexto

TAREA = "T10"
SIN_DATO = "(sin dato)"
SIN_MAPEO = "(sin mapeo)"
OTROS = "OTROS"
OTROS_AGRUPADOS = "OTROS (agrupados)"

_PARTICION = config.cargar_umbrales().particion
V_RANK = cohorte.etiqueta_periodo(_PARTICION["anios_ranking"])
"""Etiqueta nominal de la ventana de ranking ("2019-2022"): la de la partición por defecto, aunque la fuente tenga menos años."""
V_CONF = str(_PARTICION["anio_confirmacion"])
"""Etiqueta de la ventana de confirmación ("2023")."""

MIN_FILAS_HOSPITAL = 100
"""Un hospital entra a las medidas de variación entre hospitales solo con al menos este número de filas en la ventana (mismo orden de magnitud que los
filtros de las Tareas 8 y 9). Vive aquí porque la cardinalidad de ESPECIALIDAD_MEDICA por hospital se calcula al construir el panel; ``sospechas`` lo reexporta."""

# Columnas que lee el panel, en el orden del diseño. Nunca se piden FECHAPROCEDIMIENTO1 (valores con estructura de RUT), MEDICOINTERV1_ENCRIPTADO,
# MEDICOALTA_ENCRIPTADO ni el bloque de recién nacidos, y CIP_ENCRIPTADO solo se pide cuando hay historial.
_COLUMNAS_PANEL_SIN_ID = tuple(
    dict.fromkeys(
        [
            *cohorte.COLS_BASE,
            *cohorte.COLS_RESULTADO,
            "COD_HOSPITAL",
            "HOSPPROCEDENCIA",
            "SERVICIOINGRESO",
            "ESPECIALIDAD_MEDICA",
            "PREVISION",
            "NACIONALIDAD",
            "ETNIA",
            "PROVINCIA",
            "COMUNA",
            "SERVICIO_SALUD",
            "IR_29301_SEVERIDAD",
            "IR_29301_MORTALIDAD",
            "IR_29301_PESO",
            "SERVICIOALTA",
            "USOSPABELLON",
            *esquema.PROC,
            *esquema.FECHAS_TRASLADO,
            *esquema.DIAG_SEC,
        ]
    )
)
COLUMNAS_PANEL = _COLUMNAS_PANEL_SIN_ID
"""Columnas crudas que lee el panel sin historial (el identificador del paciente se agrega solo con historial: ver ``columnas_panel``)."""


def columnas_panel(con_historial: bool) -> list[str]:
    """Columnas crudas por año: ``COLUMNAS_PANEL`` más ``CIP_ENCRIPTADO`` (tras HOSPPROCEDENCIA) solo si hay historial."""
    cols = list(COLUMNAS_PANEL)
    if con_historial:
        cols.insert(cols.index("HOSPPROCEDENCIA") + 1, "CIP_ENCRIPTADO")
    return cols


# ----------------------------------------------------------------------------------------- contrato
@dataclass(frozen=True)
class Variable:
    """Metadatos de una variable del panel (el registro único de verdad)."""

    nombre: str
    tipo: str                           # "num" | "bin" | "cat"
    grupo: str                          # mismos nombres de grupo que conocimiento_clinico.csv cuando existen
    rol_fase1: str
    momento: str                        # "basal" | "alta" | "durante"
    evaluable: str                      # "si" | "no" | "parcial"
    motivo: str = ""                    # obligatorio si evaluable != "si"
    uso: str = "candidata"              # "candidata" | "solo_fuga" | "referencia" | "resultado" | "auxiliar" | "no_construible"
    fuentes: tuple[str, ...] = ()       # columnas crudas de las que sale
    analiza_resultado: bool = True      # False: ETNIA y PUEBLO_ORIGINARIO (decisión 4)
    vacio_es_ausencia: bool = False     # el vacío significa ausencia del evento (regla CALIDAD, umbral faltante_alto)
    es_referencia: bool = False         # COD_HOSPITAL


@dataclass
class Panel:
    """Cohorte D muestreada con las variables del análisis, sus niveles y los metadatos de cada variable."""

    datos: pl.DataFrame                       # una fila por egreso de la cohorte D muestreada: anio, y y una columna por variable construida
    variables: dict[str, Variable]            # registro completo, incluye no evaluables y no construibles, en orden estable
    niveles: dict[str, list[str]]             # niveles finales de cada categórica (con OTROS, "(sin mapeo)" y "(sin dato)")
    muestreo: pl.DataFrame                    # una fila por año, sin suprimir
    niveles_tabla: pl.DataFrame               # formato largo (variable, nivel, orden, tipo_nivel, ventana, n, prop), sin suprimir
    regimenes: list[list[int]]                # espacios de identificadores (vacío si no hay historial)
    con_historial: bool
    anios_ranking: list[int]
    anio_confirmacion: int | None
    cardinalidad_especialidad: dict[str, Any] = field(default_factory=dict)
    _cache: dict[str, tuple[np.ndarray, list[str]]] = field(default_factory=dict, init=False, repr=False, compare=False)

    def ventana(self, anios: Sequence[int]) -> pl.DataFrame:
        """Filas de los años dados (en el orden del panel)."""
        return self.datos.filter(pl.col("anio").is_in([int(a) for a in anios]))

    def nombres(self, *, uso: str | None = None, tipo: str | None = None, evaluable: str | None = None) -> list[str]:
        """Nombres del registro (en su orden) que cumplen los filtros. Incluye variables sin columna (no evaluables y no construibles)."""
        return [
            n for n, v in self.variables.items()
            if (uso is None or v.uso == uso) and (tipo is None or v.tipo == tipo) and (evaluable is None or v.evaluable == evaluable)
        ]

    def codigos(self, nombre: str) -> tuple[np.ndarray, list[str]]:
        """Códigos ``int32`` por fila y niveles de una binaria o categórica (se cachea).

        En una binaria los niveles son ``["0", "1"]`` y el nulo vale -1. En una categórica el nivel es la posición en ``niveles[nombre]``.
        """
        if nombre in self._cache:
            return self._cache[nombre]
        if nombre not in self.variables:
            raise KeyError(f"Panel.codigos: la variable {nombre!r} no está en el registro")
        v = self.variables[nombre]
        if nombre not in self.datos.columns:
            raise ValueError(f"Panel.codigos: la variable {nombre!r} no tiene columna en el panel ({v.evaluable}: {v.motivo})")
        col = self.datos[nombre]
        if v.tipo == "bin":
            codigos, niveles = col.fill_null(-1).cast(pl.Int32).to_numpy(), ["0", "1"]
        elif v.tipo == "cat":
            niveles = list(self.niveles[nombre])
            codigos = col.replace_strict({n: i for i, n in enumerate(niveles)}, default=-1, return_dtype=pl.Int32).to_numpy()
        else:
            raise ValueError(f"Panel.codigos: {nombre!r} es numérica; solo las binarias y las categóricas tienen códigos")
        self._cache[nombre] = (codigos, niveles)
        return codigos, niveles

    @property
    def hospital(self) -> pl.Series:
        """Alias de ``datos["COD_HOSPITAL"]``: el hospital se usa SOLO para el análisis de proxy."""
        return self.datos["COD_HOSPITAL"]


# ------------------------------------------------------------------------------------ registro de variables
MOTIVO_HISTORIAL_EXTRACTO = "no evaluable en extracto: el ingreso previo casi nunca está en el mismo extracto"
MOTIVO_ETNIA = "decisión 4: no se analiza frente a resultados"
MOTIVO_REFERENCIA = "referencia del análisis de proxy, no es candidata"
MOTIVO_HISTORIAL_PARCIAL = "solo vale donde HISTORIA_DISPONIBLE es 1"


def registro_variables(con_historial: bool) -> dict[str, Variable]:
    """Registro completo de variables del panel, en orden estable y sin leer datos (función pura).

    Incluye las candidatas, las de solo fuga, la referencia, las de resultado, las auxiliares y las que no se pueden construir (con su motivo). El rol de la Fase 1
    lo resuelve ``decision.rol_fase1``. Las tres variables de historial son "no" con motivo en un extracto y "parcial", "parcial" y "si" en modo completo.
    """
    out: dict[str, Variable] = {}

    def agregar(nombre: str, tipo: str, grupo: str, momento: str, *, uso: str = "candidata", evaluable: str = "si", motivo: str = "", **extra: Any) -> None:
        if nombre in out:
            raise ValueError(f"registro_variables: variable repetida {nombre!r}")
        fuentes = extra.pop("fuentes", decision.FUENTES_DERIVADAS.get(nombre, (nombre,)))
        out[nombre] = Variable(
            nombre=nombre, tipo=tipo, grupo=grupo, rol_fase1=decision.rol_fase1(nombre)[0], momento=momento, evaluable=evaluable, motivo=motivo, uso=uso,
            fuentes=tuple(fuentes), **extra,
        )

    elix, gi = "comorbilidad (Elixhauser)", "ingreso y procedencia"
    gp, hist = "geografía y previsión", "historial"
    ev_hist = ("parcial", MOTIVO_HISTORIAL_PARCIAL) if con_historial else ("no", MOTIVO_HISTORIAL_EXTRACTO)
    agregar("EDAD_ANIOS", "num", "demográfica", "basal")
    agregar("N_DX_SEC", "num", "codificación", "alta")
    agregar("N_COMORB_ELIX", "num", elix, "alta")
    agregar("PUNTAJE_VAN_WALRAVEN", "num", elix, "alta")
    agregar("PUNTAJE_VAN_WALRAVEN_SIN_5", "num", elix, "alta")
    for n in ("N_EGRESOS_12M", "DIAS_DESDE_EGRESO_PREVIO"):
        agregar(n, "num", hist, "basal", evaluable=ev_hist[0], motivo=ev_hist[1])
    for i in range(1, 32):
        agregar(f"ELIX_{i:02d}", "bin", elix, "alta")
    agregar("ES_FIN_SEMANA", "bin", "tiempo", "basal")
    agregar("NIVEL_CUIDADO_INGRESO", "bin", gi, "basal")
    agregar("DERIVADO_OTRO_HOSPITAL", "bin", gi, "basal")
    agregar("ENTRANTE_TIPO_PROCEDENCIA", "bin", gi, "basal")
    agregar("HISTORIA_DISPONIBLE", "bin", hist, "basal", evaluable="si" if con_historial else "no", motivo="" if con_historial else MOTIVO_HISTORIAL_EXTRACTO)
    agregar("ES_EXTRANJERO", "bin", gp, "basal")
    agregar("PUEBLO_ORIGINARIO", "bin", gp, "basal", evaluable="parcial", motivo=MOTIVO_ETNIA, analiza_resultado=False)
    agregar("SEXO", "cat", "demográfica", "basal", fuentes=("SEXO",))
    agregar("TIPO_INGRESO", "cat", gi, "basal")
    agregar("PROCEDENCIA_AGR", "cat", gi, "basal")
    agregar("TIPO_PROCEDENCIA", "cat", gi, "basal")
    agregar("GRUPO_CLINICO", "cat", "diagnóstico principal", "alta")
    agregar("CIE10_3C", "cat", "diagnóstico principal", "alta")
    agregar("MES_INGRESO", "cat", "tiempo", "basal")
    agregar("DIA_SEMANA", "cat", "tiempo", "basal")
    agregar("ESPECIALIDAD_MEDICA", "cat", gi, "basal")
    agregar("SERVICIOINGRESO", "cat", gi, "basal")
    for n in ("PREVISION", "NACIONALIDAD", "PROVINCIA", "COMUNA"):
        agregar(n, "cat", gp, "basal")
    agregar("ETNIA", "cat", gp, "basal", evaluable="parcial", motivo=MOTIVO_ETNIA, analiza_resultado=False)
    agregar("SERVICIO_SALUD", "cat", gp, "basal")
    agregar("COD_HOSPITAL", "cat", "hospital", "basal", uso="referencia", evaluable="no", motivo=MOTIVO_REFERENCIA, es_referencia=True)
    for n in ("IR_29301_SEVERIDAD", "IR_29301_MORTALIDAD"):
        agregar(n, "cat", "gravedad del GRD", "alta", uso="solo_fuga")
    agregar("IR_29301_PESO", "num", "gravedad del GRD", "alta", uso="solo_fuga")
    agregar("SERVICIOALTA", "cat", "durante y alta", "alta", uso="solo_fuga")
    agregar("USOSPABELLON", "cat", "durante y alta", "durante", uso="solo_fuga", vacio_es_ausencia=True)
    for n in ("N_PROCEDIMIENTOS", "N_TRASLADOS_INTERNOS"):
        agregar(n, "num", "durante y alta", "durante", uso="solo_fuga")
    agregar("ESTANCIA_DIAS", "num", "actividad y estadía", "durante", uso="resultado")
    agregar("ES_COHORTE_E", "bin", "actividad y estadía", "durante", uso="auxiliar", fuentes=("TIPOALTA", "FECHAALTA", "FECHA_INGRESO"))
    agregar("SERVICIO_INGRESO_IGUAL_ALTA", "bin", "durante y alta", "alta", uso="auxiliar", fuentes=("SERVICIOINGRESO", "SERVICIOALTA"))
    for n, tipo, grupo in (("ESPECIALIDAD_MACRO", "cat", gi), ("TRAMO_FONASA", "cat", gp), ("INDICE_PRIVACION", "num", gp)):
        agregar(n, tipo, grupo, "basal", uso="no_construible", evaluable="no", motivo=decision.motivo_no_evaluable(n, "derivada"))
    return out


# ------------------------------------------------------------------------------------- textos y centinelas
_TILDES = str.maketrans("ÓÉÍÁÚ", "OEIAU")


def _es_centinela_texto(texto: str) -> bool:
    """Misma regla que ``esquema.es_centinela`` aplicada a un texto: sin espacios extremos, en mayúsculas y sin tildes."""
    return texto.strip().upper().translate(_TILDES) in esquema.CENTINELAS


def _vacio_o_centinela(texto: str) -> bool:
    return texto.strip() == "" or _es_centinela_texto(texto)


def _por_distintos(valores: pl.Series, funcion: Callable[[str], Any], dtype: pl.DataType) -> pl.Series:
    """Aplica ``funcion`` solo a los valores distintos no nulos y devuelve una serie del mismo largo (nulo donde el valor es nulo o la función da ``None``)."""
    s = valores.cast(pl.String)
    mapa = {d: r for d in s.drop_nulls().unique().to_list() if (r := funcion(d)) is not None}
    if not mapa:
        return pl.Series(valores.name, [None] * len(s), dtype=dtype)
    return s.replace_strict(mapa, default=None, return_dtype=dtype).alias(valores.name)


def limpiar_categorica(valores: pl.Series, *, extra_sin_dato: Sequence[str] = ()) -> pl.Series:
    """Texto de una categórica cruda: sin espacios extremos; nulo, vacío y centinela pasan al nivel "(sin dato)" (nunca se imputan).

    No se cambia mayúsculas ni tildes: la normalización agresiva solo se usa para comparar con diccionarios. ``extra_sin_dato`` agrega valores que también
    significan "sin dato" (el "SIN_DATO" que deja ``cohorte.tipar`` en SEXO). Esto unifica "OTRO " con "OTRO" en ETNIA.
    """
    s = valores.cast(pl.String).str.strip_chars()
    extra = {x.strip() for x in extra_sin_dato}
    malos = [d for d in s.drop_nulls().unique().to_list() if d == "" or _es_centinela_texto(d) or d in extra]
    if malos:
        s = s.to_frame("v").select(pl.when(pl.col("v").is_in(malos)).then(pl.lit(SIN_DATO)).otherwise(pl.col("v")).alias("v"))["v"]
    return s.fill_null(SIN_DATO).alias(valores.name)


# ----------------------------------------------------------------------------------- PROCEDENCIA_AGR
_PROCEDENCIA_CRUDA: dict[str, str] = {
    "SERVICIO EMERGENCIA (DOMICILIO)": "URGENCIA",
    "APS URGENCIA (SAPU, SUR, SUC)": "URGENCIA",
    "OTROS HOSPITALES DE LA RED": "DERIVADO_OTRO_HOSPITAL",
    "OTROS HOSPITALES RED NACIONAL": "DERIVADO_OTRO_HOSPITAL",
    "CENTRO ESPECIALIDADES (CDT, CRS, CONSULTORIO ADOS. ESP)": "PROGRAMADO",
    "APS CONSULTORIO (CESFAM)": "PROGRAMADO",
    "CONSULTA PRIVADA": "PROGRAMADO",
    "PLAN DE RESOLUCIÓN LE": "PROGRAMADO",
    "CIRUGÍA MAYOR AMBULATORIA (CMA)": "PROGRAMADO",
    "CARDIOCIRUGÍA PAGO GRD": "PROGRAMADO",
}
"""Textos de TIPO_PROCEDENCIA tal como vienen en los datos y su grupo. PENDIENTE DE REVISIÓN CLÍNICA (decisión 12). Lo que no está aquí, y no es un
valor sin dato, queda en el nivel visible "(sin mapeo)": las dos "OTRAS INSTITUCIONES" (truncadas en el origen), "ESTRATEGIA CRR", "HOSPITALIZACIÓN
DOMICILIARIA", "HOSPITALIZACIÓN DIURNA", "POSTA RURAL", "UGCC" y cualquier valor nuevo."""


def construir_mapa_procedencia(crudo: Mapping[str, str]) -> dict[str, str]:
    """Normaliza las claves con ``hospitales.normalizar_nombre`` (mayúsculas, sin tildes y con cada secuencia de signos reducida a un espacio).

    Lanza ``ValueError`` si dos textos distintos se normalizan igual con grupos distintos: el mapeo dejaría de ser una función.
    """
    out: dict[str, str] = {}
    origen: dict[str, str] = {}
    for texto, grupo in crudo.items():
        clave = hospitales.normalizar_nombre(texto)
        if clave in out and out[clave] != grupo:
            raise ValueError(f"construir_mapa_procedencia: {texto!r} y {origen[clave]!r} se normalizan igual ({clave!r}) con grupos distintos ({grupo} y {out[clave]})")
        out[clave] = grupo
        origen.setdefault(clave, texto)
    return out


MAPA_PROCEDENCIA_AGR = construir_mapa_procedencia(_PROCEDENCIA_CRUDA)
"""Mapa con las claves ya normalizadas. PENDIENTE DE REVISIÓN CLÍNICA (decisión 12)."""


def agrupar_procedencia(valores: pl.Series) -> pl.Series:
    """PROCEDENCIA_AGR: URGENCIA, DERIVADO_OTRO_HOSPITAL o PROGRAMADO; lo no listado va a "(sin mapeo)" y nulos, vacíos y centinelas a "(sin dato)"."""
    def grupo(t: str) -> str:
        return SIN_DATO if _vacio_o_centinela(t) else MAPA_PROCEDENCIA_AGR.get(hospitales.normalizar_nombre(t), SIN_MAPEO)

    return _por_distintos(valores, grupo, pl.String).fill_null(SIN_DATO)


def entrante_tipo_procedencia(valores: pl.Series) -> pl.Series:
    """ENTRANTE_TIPO_PROCEDENCIA: 1 si el TIPO_PROCEDENCIA normalizado contiene el texto de traslado entrante de la Tarea 9; 0 en cualquier otro caso (nunca nulo)."""
    def marca(t: str) -> int:
        return int(hospitales.TEXTO_ENTRANTE in hospitales.normalizar_nombre(t))

    return _por_distintos(valores, marca, pl.Int8).fill_null(0)


# ---------------------------------------------------------------------------------- binarias derivadas
PATRONES_CUIDADO_ESPECIAL = (r"\bUCI\b", r"\bUTI\b", r"CUIDADOS? INTENSIVOS?", r"TRATAMIENTOS? INTERMEDIOS?")
"""Patrones sobre el SERVICIOINGRESO normalizado que indican cuidados especiales. Quedan fuera por decisión de diseño las unidades "de agudos", los "cuidados
medios o básicos", la recuperación de pabellones y las urgencias, porque su nombre no dice que sean cuidados especiales. PENDIENTE DE REVISIÓN CLÍNICA (decisión 12)."""
_RE_CUIDADO = tuple(re.compile(p) for p in PATRONES_CUIDADO_ESPECIAL)


def nivel_cuidado_ingreso(valores: pl.Series) -> pl.Series:
    """NIVEL_CUIDADO_INGRESO: 1 si el servicio de ingreso es de cuidados especiales, 0 si está informado y no coincide, nulo si es nulo, vacío o centinela."""
    def marca(t: str) -> int | None:
        if _vacio_o_centinela(t):
            return None
        n = hospitales.normalizar_nombre(t)
        return int(any(p.search(n) for p in _RE_CUIDADO))

    return _por_distintos(valores, marca, pl.Int8)


def es_extranjero(valores: pl.Series) -> pl.Series:
    """ES_EXTRANJERO: 0 si la NACIONALIDAD normalizada empieza con "CHILE" (cubre "CHILE" y "CHILENA"), 1 en cualquier otro valor informado, nulo si es nula o centinela."""
    def marca(t: str) -> int | None:
        return None if _vacio_o_centinela(t) else int(not hospitales.normalizar_nombre(t).startswith("CHILE"))

    return _por_distintos(valores, marca, pl.Int8)


_NO_ORIGINARIO = frozenset({"NINGUNO", "NINGUNA", "OTRO"})


def es_pueblo_originario(valores: pl.Series) -> pl.Series:
    """PUEBLO_ORIGINARIO: 0 si la ETNIA normalizada es NINGUNO, NINGUNA u OTRO (la Fase 1 dice que OTRO reemplaza a NINGUNO desde 2022), 1 en cualquier otro
    valor informado, nulo si es nula o centinela. PENDIENTE DE REVISIÓN (decisiones 4 y 12)."""
    def marca(t: str) -> int | None:
        return None if _vacio_o_centinela(t) else int(hospitales.normalizar_nombre(t) not in _NO_ORIGINARIO)

    return _por_distintos(valores, marca, pl.Int8)


# ------------------------------------------------------------------------------------------- muestreo
def muestrear_indices(n: int, tope: int, semilla: int, anio: int) -> np.ndarray:
    """Posiciones de las filas de D que entran al panel: todas si ``tope <= 0`` (sin muestreo) o ``n <= tope``; si no, una muestra sin reemplazo con
    ``default_rng(semilla + anio)``, ordenada."""
    if tope <= 0 or n <= tope:
        return np.arange(n)
    return np.sort(np.random.default_rng(int(semilla) + int(anio)).choice(n, size=int(tope), replace=False))


# ----------------------------------------------------------------------------------- niveles de categóricas
@dataclass(frozen=True)
class NivelesCategorica:
    """Niveles finales de una categórica y el nombre del nivel de agrupación, si se usó."""

    niveles: list[str]
    otros: str | None = None            # "OTROS" u "OTROS (agrupados)" si algún valor se agrupó; None si no


def calcular_niveles(valores: pl.Series, anios: pl.Series, anios_ranking: Sequence[int], tope: int) -> NivelesCategorica:
    """Niveles de una categórica, calculados con la ventana de ranking.

    Se cuentan los niveles de ``anios_ranking`` excluyendo los protegidos "(sin dato)" y "(sin mapeo)", que nunca se fusionan y no ocupan cupo; se ordenan por
    frecuencia descendente y, en empate, por texto ascendente; se conservan los primeros ``tope`` y todo lo demás pasa a OTROS, incluidos los niveles que solo
    aparecen fuera de la ventana de ranking (un nivel nuevo en 2023 es evidencia de deriva, aunque la variable tenga menos de ``tope`` niveles). Si ya existe un
    nivel real llamado "OTROS", el agrupado se llama "OTROS (agrupados)". El orden final es: los niveles conservados, "(sin mapeo)", OTROS y "(sin dato)" (los
    tres últimos solo si se usan). Fijar los niveles con 2019 a 2023 usaría información de 2023 y contradice la partición.
    """
    df = pl.DataFrame({"v": valores.cast(pl.String), "a": anios.cast(pl.Int64)})
    protegidos = [SIN_DATO, SIN_MAPEO]
    conteo = (
        df.filter(pl.col("a").is_in([int(a) for a in anios_ranking]) & ~pl.col("v").is_in(protegidos) & pl.col("v").is_not_null())
        .group_by("v")
        .agg(pl.len().alias("n"))
        .sort(["n", "v"], descending=[True, False])
    )
    conservados = conteo["v"].head(int(tope)).to_list()
    cons = set(conservados)
    presentes = set(df["v"].drop_nulls().unique().to_list())
    hay_otros = any(v not in cons and v not in protegidos for v in presentes)
    nombre_otros = OTROS_AGRUPADOS if OTROS in presentes else OTROS
    niveles = list(conservados)
    if SIN_MAPEO in presentes:
        niveles.append(SIN_MAPEO)
    if hay_otros:
        niveles.append(nombre_otros)
    if SIN_DATO in presentes:
        niveles.append(SIN_DATO)
    return NivelesCategorica(niveles=niveles, otros=nombre_otros if hay_otros else None)


def calcular_niveles_referencia(valores: pl.Series) -> NivelesCategorica:
    """Niveles de la referencia (COD_HOSPITAL): todos, sin tope, en orden de texto: la medida de proxy necesita todos los hospitales."""
    return NivelesCategorica(niveles=sorted(valores.cast(pl.String).drop_nulls().unique().to_list()), otros=None)


def recodificar(valores: pl.Series, niveles: NivelesCategorica) -> pl.Series:
    """Lleva al nivel de agrupación todo valor que no está entre los niveles finales (el nulo pasa a "(sin dato)")."""
    s = valores.cast(pl.String)
    if niveles.otros is None:
        return s.fill_null(SIN_DATO).alias(valores.name)
    return s.to_frame("v").select(
        pl.when(pl.col("v").is_null()).then(pl.lit(SIN_DATO)).when(pl.col("v").is_in(niveles.niveles)).then(pl.col("v")).otherwise(pl.lit(niveles.otros)).alias(valores.name)
    )[valores.name]


def _tipo_nivel(nivel: str, niveles: NivelesCategorica) -> str:
    if nivel == SIN_DATO:
        return "sin_dato"
    if nivel == SIN_MAPEO:
        return "sin_mapeo"
    return "otros" if niveles.otros is not None and nivel == niveles.otros else "real"


ESQUEMA_NIVELES_TABLA = {"variable": pl.String, "nivel": pl.String, "orden": pl.Int64, "tipo_nivel": pl.String, "ventana": pl.String, "n": pl.Int64, "prop": pl.Float64}


def tabla_de_niveles(
    variable: str, valores: pl.Series, anios: pl.Series, niveles: NivelesCategorica, ventanas: Sequence[tuple[str, Sequence[int]]]
) -> pl.DataFrame:
    """Filas de ``t23_panel_niveles.csv`` de una categórica (valores ya recodificados): una por nivel y ventana, con ``n = 0`` si el nivel no aparece.

    ``prop`` es ``n`` dividido por las filas de la variable en esa ventana. Una ventana sin filas no genera filas (su proporción no existe).
    """
    df = pl.DataFrame({"v": valores.cast(pl.String), "a": anios.cast(pl.Int64)})
    filas: list[dict[str, Any]] = []
    for etiqueta, anios_v in ventanas:
        sub = df.filter(pl.col("a").is_in([int(a) for a in anios_v]))
        total = sub.height
        if total == 0:
            continue
        cuentas = dict(sub.group_by("v").agg(pl.len().alias("n")).iter_rows())
        for orden, nivel in enumerate(niveles.niveles, start=1):
            n = int(cuentas.get(nivel, 0))
            filas.append({"variable": variable, "nivel": nivel, "orden": orden, "tipo_nivel": _tipo_nivel(nivel, niveles), "ventana": etiqueta, "n": n, "prop": n / total})
    return pl.DataFrame(filas, schema=ESQUEMA_NIVELES_TABLA)


ESQUEMA_MUESTREO = {
    "anio": pl.Int64, "ventana": pl.String, "n_cohorte_d": pl.Int64, "n_muestra": pl.Int64, "n_eventos_muestra": pl.Int64, "muestreado": pl.String, "tope": pl.Int64,
    "fraccion_muestreada": pl.Float64,
}


def tabla_muestreo(filas: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    """``t23_panel_muestreo.csv`` (sin suprimir): una fila por año de resultado con la cohorte D, la muestra y si se muestreó.

    Cada elemento trae ``anio``, ``ventana`` ("ranking" o "confirmacion"), ``n_cohorte_d``, ``n_muestra``, ``n_eventos_muestra`` y ``tope``. Un año sin cohorte D
    queda con ceros y sin fracción.
    """
    salida = []
    for f in sorted(filas, key=lambda x: int(x["anio"])):
        n_d, n_m = int(f["n_cohorte_d"]), int(f["n_muestra"])
        salida.append({
            "anio": int(f["anio"]), "ventana": f["ventana"], "n_cohorte_d": n_d, "n_muestra": n_m, "n_eventos_muestra": int(f["n_eventos_muestra"]),
            "muestreado": "si" if (int(f["tope"]) > 0 and n_d > int(f["tope"])) else "no", "tope": int(f["tope"]), "fraccion_muestreada": (n_m / n_d) if n_d else None,
        })
    return pl.DataFrame(salida, schema=ESQUEMA_MUESTREO)


# ------------------------------------------------------------------------------------------ historial
_DIA_MAX = 99_999            # la clave compuesta usa 100.000 por identificador: una fecha fuera de [0, 99.999] días desde 1970 no es una fecha de egreso plausible
_ORIGEN_EPOCA = np.datetime64("1970-01-01")


def historial_por_episodio(episodios: pl.DataFrame, regimenes: Sequence[Sequence[int]], ventana: int) -> pl.DataFrame:
    """N_EGRESOS_12M, DIAS_DESDE_EGRESO_PREVIO y HISTORIA_DISPONIBLE de cada episodio, con numpy.

    ``episodios`` trae TODAS las filas de los años de resultado (no solo la cohorte D) con ``fila`` (clave del episodio), ``anio``, ``id_num`` (nulo si el
    identificador es inválido), ``ing`` y ``alta``. Por cada espacio de identificadores ``reg`` (lista de años consecutivos que comparten pacientes), con
    ``inicio = date(min(reg), 1, 1)``:

    - ``viable``: el episodio es del espacio, tiene identificador válido, ``ing`` no nulo e ``ing - ventana >= inicio`` (igual que ``hospitales.historial_regimen``).
    - un egreso previo es válido si es del mismo espacio y del mismo identificador, con ``ing`` y ``alta`` no nulos y ``alta >= ing``. Para el episodio ``e`` cuenta si
      ``ing_e - ventana <= alta_p <= ing_e - 1``: estrictamente anterior (365 días exactos entran, 366 no) y el reingreso del mismo día no cuenta.

    ``n_previos`` es el número de previos que cuentan (nulo si el episodio no es viable) y ``dias_previo`` los días desde el alta más reciente que cuenta (nulo si no
    es viable o no hay previo). Los espacios de identificadores distintos no se cruzan.

    Los identificadores llegan a 15 dígitos, así que una clave ``id * 100000 + día`` desbordaría ``int64``: los identificadores se reindexan a un índice denso antes
    de formar la clave. Una fecha fuera del rango de la clave (anterior a 1970 o posterior al año 2243) se trata como inválida. Para cada episodio el conteo es
    la diferencia de dos ``np.searchsorted`` sobre las claves ordenadas de los previos.
    """
    n = episodios.height
    anio = episodios["anio"].to_numpy().astype(np.int64)
    id_ok = episodios["id_num"].is_not_null().to_numpy()
    ids = episodios["id_num"].fill_null(0).to_numpy().astype(np.int64)
    ing_d = episodios["ing"].dt.epoch("d").fill_null(-1).to_numpy().astype(np.int64)
    alta_d = episodios["alta"].dt.epoch("d").fill_null(-1).to_numpy().astype(np.int64)
    ing_ok = (ing_d >= 0) & (ing_d <= _DIA_MAX)
    alta_ok = (alta_d >= 0) & (alta_d <= _DIA_MAX)
    viable = np.zeros(n, dtype=bool)
    n_prev = np.zeros(n, dtype=np.int64)
    dias = np.zeros(n, dtype=np.int64)
    con_dias = np.zeros(n, dtype=bool)
    for reg in regimenes:
        reg = [int(a) for a in reg]
        if not reg:
            continue
        en_reg = np.isin(anio, reg) & id_ok
        inicio = int((np.datetime64(f"{min(reg):04d}-01-01") - _ORIGEN_EPOCA).astype(np.int64))
        _, denso_reg = np.unique(ids[en_reg], return_inverse=True)
        denso = np.full(n, -1, dtype=np.int64)
        denso[en_reg] = denso_reg
        es_viable = en_reg & ing_ok & (ing_d - ventana >= inicio)
        es_previo = en_reg & ing_ok & alta_ok & (alta_d >= ing_d)
        claves = np.sort(denso[es_previo] * 100_000 + alta_d[es_previo])
        v = np.flatnonzero(es_viable)
        base = denso[v] * 100_000
        izq = np.searchsorted(claves, base + (ing_d[v] - ventana), side="left")
        der = np.searchsorted(claves, base + (ing_d[v] - 1), side="right")
        cuenta = der - izq
        viable[v] = True
        n_prev[v] = cuenta
        con = cuenta > 0
        if con.any():
            ultimo = claves[der[con] - 1] - base[con]
            dias[v[con]] = ing_d[v[con]] - ultimo
            con_dias[v[con]] = True
    salida = pl.DataFrame({"fila": episodios["fila"], "viable": viable, "n_previos": n_prev, "dias_previo": dias, "_con_dias": con_dias})
    return salida.with_columns(
        pl.when(pl.col("viable")).then(pl.col("n_previos")).otherwise(None).alias("n_previos"),
        pl.when(pl.col("_con_dias")).then(pl.col("dias_previo")).otherwise(None).alias("dias_previo"),
    ).drop("_con_dias")


# ---------------------------------------------------------------------------------- cardinalidad de especialidad
def cardinalidad_especialidad(texto: Sequence[str | None] | pl.Series, hospital: Sequence[str | None] | pl.Series, min_filas: int = MIN_FILAS_HOSPITAL) -> dict[str, Any]:
    """Cardinalidad de ESPECIALIDAD_MEDICA por hospital (H02b), calculada sobre el TEXTO sin tope de niveles y sin OTROS.

    Debe recibir el texto antes de aplicar el tope de 60 niveles: con el tope, ``n_niveles_totales`` nunca pasaría de 60 y los niveles raros, que son los que con más
    frecuencia pertenecen a un solo hospital, desaparecerían en OTROS. "Niveles reales" son los textos no nulos, no vacíos y no centinela.
    Devuelve solo agregados, y todos con una misma base: los hospitales con al menos ``min_filas`` filas (DIS 10.4, "por hospital con al menos MIN_FILAS_HOSPITAL
    filas"). ``n_hospitales`` es cuántos son; ``card_minima``, ``card_mediana`` y ``card_maxima`` resumen los niveles reales distintos de cada uno (nulos si no hay
    ninguno); ``n_niveles_totales`` son los niveles reales distintos que usa alguno de ellos y ``n_niveles_en_un_hospital`` los que usa uno solo de ellos.
    Un nivel que solo usan hospitales con menos filas que ``min_filas`` no entra a ninguna cifra.

    Base resuelta en el cierre de la etapa A: es la definición literal de DIS 10.4 (hospitales con al menos ``min_filas`` filas, niveles reales sin centinelas). La cifra
    que midió la revisión del diseño en los extractos (un nivel y un exclusivo más) no es una especificación y se reproduce por dos caminos que esta definición excluye,
    cada uno con un nivel de menos de diez filas: contar el centinela como nivel (lo usa un hospital grande) o contar los niveles sobre los hospitales chicos. Los dos
    juntos sumarían dos niveles y dos exclusivos. Las pruebas fijan las dos exclusiones; el detalle está en el registro de evidencia de la etapa A del plan.
    """
    if not isinstance(min_filas, (int, np.integer)) or isinstance(min_filas, bool) or min_filas < 1:
        raise ValueError(f"cardinalidad_especialidad: 'min_filas' debe ser un entero igual o mayor que 1 (recibió {min_filas!r})")
    t = (texto.cast(pl.String) if isinstance(texto, pl.Series) else pl.Series(list(texto), dtype=pl.String)).alias("t")
    h = (hospital.cast(pl.String) if isinstance(hospital, pl.Series) else pl.Series(list(hospital), dtype=pl.String)).alias("h")
    if len(t) != len(h):
        raise ValueError(f"cardinalidad_especialidad: 'texto' y 'hospital' tienen largos distintos ({len(t)} y {len(h)})")
    df = pl.DataFrame({"t": t.str.strip_chars(), "h": h})
    filas_h = df.group_by("h").agg(pl.len().alias("n")).filter(pl.col("n") >= int(min_filas))
    hospitales_ok = filas_h["h"].to_list()
    grandes = df.join(filas_h.select("h"), on="h", how="semi", nulls_equal=True)             # solo los hospitales con al menos ``min_filas`` filas
    reales = grandes.filter(pl.col("t").is_not_null() & (pl.col("t") != "") & ~esquema.es_centinela(pl.col("t")))
    por_nivel = reales.group_by("t").agg(pl.col("h").n_unique().alias("nh"))
    k_por_hospital = dict(reales.group_by("h").agg(pl.col("t").n_unique().alias("k")).iter_rows())
    cards = np.array([k_por_hospital.get(x, 0) for x in hospitales_ok], dtype=np.int64)
    return {
        "n_hospitales": len(hospitales_ok),
        "card_minima": int(cards.min()) if cards.size else None,
        "card_mediana": float(np.median(cards)) if cards.size else None,
        "card_maxima": int(cards.max()) if cards.size else None,
        "n_niveles_totales": int(por_nivel.height),
        "n_niveles_en_un_hospital": int((por_nivel["nh"] == 1).sum()),
    }


# =================================================================================================== construcción
def _contar_informadas(df: pl.DataFrame, columnas: Sequence[str]) -> pl.Series:
    """Número de columnas informadas (no nulas y no vacías) por fila, como Float64."""
    exprs = [(pl.col(c).is_not_null() & (pl.col(c).str.strip_chars() != "")).cast(pl.Int32) for c in columnas]
    return df.select(pl.sum_horizontal(exprs).cast(pl.Float64).alias("n"))["n"]


def _variables_anio(cr: pl.DataFrame, bs: pl.DataFrame, dx: pl.DataFrame, e: elixhauser.Elixhauser) -> pl.DataFrame:
    """Variables canónicas (sin historial) de las filas elegidas de un año, con las categóricas todavía sin tope de niveles.

    ``cr`` son las columnas crudas, ``bs`` la base tipada de ``cohorte.tipar`` (con los indicadores de cohorte) y ``dx`` los diagnósticos secundarios, las tres
    con las mismas filas. Nada se imputa: el nulo de una numérica o binaria se conserva y el de una categórica pasa al nivel "(sin dato)".
    """
    ing = cr.select(tr.a_fecha(pl.col("FECHA_INGRESO")).alias("ing"))["ing"]
    dia = ing.dt.weekday() - 1                                      # 0 lunes a 6 domingo (polars devuelve 1 a 7)
    flags = elixhauser.calcular_flags(dx, esquema.DIAG_SEC, e).rename({"SCORE_VANWALRAVEN": "PUNTAJE_VAN_WALRAVEN"})
    hospproc = bs["hospproc"]
    cie = cr.select(esquema.normalizar_cie10(pl.col("DIAGNOSTICO1")).str.slice(0, 3).alias("c"))["c"]
    # SERVICIO_INGRESO_IGUAL_ALTA se calcula con los textos crudos, no con los niveles ya recortados: dos servicios raros distintos caerían ambos en OTROS y parecerían iguales
    igual_alta = cr.select(
        a=pl.col("SERVICIOINGRESO").str.strip_chars().str.to_uppercase(), b=pl.col("SERVICIOALTA").str.strip_chars().str.to_uppercase()
    ).select(
        pl.when(
            pl.col("a").is_not_null() & pl.col("b").is_not_null() & (pl.col("a") != "") & (pl.col("b") != "")
            & ~esquema.es_centinela(pl.col("a")) & ~esquema.es_centinela(pl.col("b"))
        )
        .then((pl.col("a") == pl.col("b")).cast(pl.Int8))
        .otherwise(None)
        .alias("x")
    )["x"]
    columnas: dict[str, pl.Series] = {
        "EDAD_ANIOS": bs["edad"].cast(pl.Float64),
        "N_DX_SEC": _contar_informadas(dx, esquema.DIAG_SEC),
        "N_COMORB_ELIX": flags["N_COMORB_ELIX"].cast(pl.Float64),
        "PUNTAJE_VAN_WALRAVEN": flags["PUNTAJE_VAN_WALRAVEN"].cast(pl.Float64),
        "PUNTAJE_VAN_WALRAVEN_SIN_5": flags["PUNTAJE_VAN_WALRAVEN_SIN_5"].cast(pl.Float64),
    }
    for i in elixhauser.IDS:
        columnas[i] = flags[i].cast(pl.Int8)
    columnas["ES_FIN_SEMANA"] = (dia >= 5).cast(pl.Int8)
    columnas["NIVEL_CUIDADO_INGRESO"] = nivel_cuidado_ingreso(cr["SERVICIOINGRESO"])
    columnas["DERIVADO_OTRO_HOSPITAL"] = (hospproc.is_not_null() & (hospproc != "")).cast(pl.Int8)
    columnas["ENTRANTE_TIPO_PROCEDENCIA"] = entrante_tipo_procedencia(cr["TIPO_PROCEDENCIA"])
    columnas["ES_EXTRANJERO"] = es_extranjero(cr["NACIONALIDAD"])
    columnas["PUEBLO_ORIGINARIO"] = es_pueblo_originario(cr["ETNIA"])
    columnas["SEXO"] = limpiar_categorica(bs["sexo"], extra_sin_dato=("SIN_DATO",))
    for nombre in ("TIPO_INGRESO", "TIPO_PROCEDENCIA"):
        columnas[nombre] = limpiar_categorica(cr[nombre])
    columnas["PROCEDENCIA_AGR"] = agrupar_procedencia(cr["TIPO_PROCEDENCIA"])
    columnas["GRUPO_CLINICO"] = limpiar_categorica(bs["capitulo"])
    columnas["CIE10_3C"] = limpiar_categorica(cie)
    columnas["MES_INGRESO"] = limpiar_categorica(ing.dt.month().cast(pl.String))
    columnas["DIA_SEMANA"] = limpiar_categorica(dia.cast(pl.String))
    for nombre in ("ESPECIALIDAD_MEDICA", "SERVICIOINGRESO", "PREVISION", "NACIONALIDAD", "PROVINCIA", "COMUNA", "ETNIA", "SERVICIO_SALUD"):
        columnas[nombre] = limpiar_categorica(cr[nombre])
    columnas["COD_HOSPITAL"] = limpiar_categorica(bs["hospital"])
    for nombre in ("IR_29301_SEVERIDAD", "IR_29301_MORTALIDAD", "SERVICIOALTA", "USOSPABELLON"):
        columnas[nombre] = limpiar_categorica(cr[nombre])
    columnas["IR_29301_PESO"] = cr.select(tr.a_decimal(pl.col("IR_29301_PESO")).alias("x"))["x"]
    columnas["N_PROCEDIMIENTOS"] = _contar_informadas(cr, esquema.PROC)
    columnas["N_TRASLADOS_INTERNOS"] = _contar_informadas(cr, esquema.FECHAS_TRASLADO)
    columnas["ESTANCIA_DIAS"] = bs["estancia"].cast(pl.Float64)
    columnas["ES_COHORTE_E"] = bs["E"].cast(pl.Int8)
    columnas["SERVICIO_INGRESO_IGUAL_ALTA"] = igual_alta
    return pl.DataFrame({nombre: serie.alias(nombre) for nombre, serie in columnas.items()})


def _validar_regimenes(regimenes: Any, resultado: Sequence[int]) -> list[list[int]]:
    """Espacios de identificadores dados por quien llama: listas de años enteros, ordenados y todos años de resultado."""
    if isinstance(regimenes, (str, bytes)) or not hasattr(regimenes, "__iter__"):
        raise ValueError("construir_panel: 'regimenes' debe ser una lista de listas de años")
    salida = []
    for reg in regimenes:
        if isinstance(reg, (str, bytes)) or not hasattr(reg, "__iter__"):
            raise ValueError(f"construir_panel: cada espacio de 'regimenes' debe ser una lista de años (recibió {reg!r})")
        anios = list(reg)
        if not all(isinstance(a, (int, np.integer)) and not isinstance(a, bool) for a in anios):
            raise ValueError(f"construir_panel: los años de 'regimenes' deben ser enteros (recibió {anios!r})")
        anios = [int(a) for a in anios]
        if anios != sorted(set(anios)):
            raise ValueError(f"construir_panel: los años de cada espacio de 'regimenes' deben estar ordenados y sin repetir (recibió {anios!r})")
        fuera = [a for a in anios if a not in resultado]
        if fuera:
            raise ValueError(f"construir_panel: 'regimenes' trae años que no son de resultado: {fuera}")
        salida.append(anios)
    return salida


def construir_panel(ctx: Contexto, cat: catalogos.Catalogos | None = None, historial: bool | None = None, *, regimenes: Sequence[Sequence[int]] | None = None, tarea: str = TAREA) -> Panel:
    """Lee 2019 a 2023 una sola vez y arma el panel (cohorte D muestreada con las variables canónicas).

    Es el único punto de lectura de las Tareas 10 y 11. Lee solo los años de ``anios_resultado`` con ``ctx.lector.leer(anio, columnas, proposito="seleccion",
    tarea=...)`` y lanza ``RuntimeError("la partición temporal está rota")`` ANTES de leer nada si el año expuesto aparece entre ellos. Nunca lee 2024.

    ``historial`` es ``None`` (calcula el historial solo si la etiqueta del contexto es "completo"), ``True`` o ``False`` (lo fuerzan; con extractos ``True`` solo sirve
    para probar la mecánica). ``regimenes`` permite pasar los espacios de identificadores ya calculados (misma forma que en ``hospitales.calcular``); si es ``None`` se calculan con
    ``hospitales.regimenes_de`` sobre los identificadores válidos de los años leídos.

    Por cada año: ``cohorte.tipar``, ``cohorte.agregar_comorbilidad`` y ``cohorte.marcar`` (no se reimplementa ninguna regla de cohorte), muestreo reproducible sobre la cohorte D
    (``muestrear_indices`` con ``max_filas_asociacion_por_anio``) y cálculo de las variables. Después del último año: niveles de las categóricas con la ventana de ranking
    (tope ``cardinalidad_max_categorica`` y OTROS), cardinalidad de ESPECIALIDAD_MEDICA por hospital sobre el texto SIN tope y antes de recortar, historial y tablas de apoyo.
    Un año de resultado sin cohorte D genera una fila de muestreo con ceros y no aporta filas. Sin filas de ranking, o sin los dos desenlaces en la ventana de ranking, lanza ``ValueError``;
    si falta el año de confirmación, ``Panel.anio_confirmacion`` es ``None`` y todo lo de 2023 queda sin datos. El panel no conserva identificadores, fechas ni nombres de hospital.
    """
    u = ctx.umbrales
    part = u.particion
    resultado = [int(a) for a in part["anios_resultado"]]
    ranking_cfg = [int(a) for a in part["anios_ranking"]]
    conf_cfg = int(part["anio_confirmacion"])
    expuesto = int(part["anio_expuesto"])
    if historial is None:
        con_historial = ctx.etiqueta == "completo"
    elif isinstance(historial, bool):
        con_historial = historial
    else:
        raise TypeError(f"construir_panel: 'historial' debe ser True, False o None (recibió {type(historial).__name__})")
    anios = [a for a in ctx.fuente.anios() if a in resultado]
    if expuesto in anios or expuesto in resultado:
        raise RuntimeError("la partición temporal está rota")
    if regimenes is not None:
        regimenes = _validar_regimenes(regimenes, resultado)
    cat = cat or catalogos.cargar_catalogos()
    e, e3 = elixhauser.cargar(), elixhauser.cargar(solo_3c=True)
    tope_filas = int(u["max_filas_asociacion_por_anio"])
    tope_niveles = int(u["cardinalidad_max_categorica"])
    edad_adulto = int(u["edad_adulto"])
    columnas = columnas_panel(con_historial)

    partes: list[pl.DataFrame] = []
    filas_muestreo: list[dict[str, Any]] = []
    episodios: list[pl.DataFrame] = []
    ids_validos: dict[int, pl.Series] = {}
    especialidad: list[pl.DataFrame] = []
    desplazamiento = 0
    for anio in anios:
        crudo = ctx.lector.leer(anio, columnas, proposito="seleccion", tarea=tarea)
        dx = crudo.select(esquema.DIAG_SEC)
        base = cohorte.tipar(crudo, anio, cat, u)
        base = cohorte.agregar_comorbilidad(base, dx, e, e3)
        base = cohorte.marcar(base, edad_adulto)
        en_d = np.flatnonzero(base["D"].to_numpy())
        elegidas = en_d[muestrear_indices(en_d.size, tope_filas, ctx.semilla, anio)]
        if con_historial:
            episodios.append(
                base.select(
                    pl.Series("fila", desplazamiento + np.arange(base.height, dtype=np.int64)), pl.col("anio").cast(pl.Int32), "id_num", "ing", "alta"
                )
            )
            ids_validos[anio] = base["id_num"].drop_nulls().unique()
        n_eventos = 0
        if elegidas.size:
            cr, bs, dxs = crudo[elegidas], base[elegidas], dx[elegidas]
            variables = _variables_anio(cr, bs, dxs, e)
            y = (bs["rol"] == "EVENTO").cast(pl.Int8)
            n_eventos = int(y.sum())
            extra = [pl.Series("anio", np.full(elegidas.size, anio, dtype=np.int32), dtype=pl.Int32), y.alias("y")]
            if con_historial:
                extra.append(pl.Series("__fila", desplazamiento + elegidas, dtype=pl.Int64))
            partes.append(variables.select(*extra, pl.all()))
            if anio in ranking_cfg:
                especialidad.append(pl.DataFrame({"t": cr["ESPECIALIDAD_MEDICA"].cast(pl.String).str.strip_chars(), "h": bs["hospital"].cast(pl.String)}))
            del cr, bs, dxs, variables
        filas_muestreo.append({
            "anio": anio, "ventana": "ranking" if anio in ranking_cfg else "confirmacion", "n_cohorte_d": int(en_d.size), "n_muestra": int(elegidas.size),
            "n_eventos_muestra": n_eventos, "tope": tope_filas,
        })
        desplazamiento += base.height
        del crudo, dx, base

    if not partes:
        raise ValueError("construir_panel: ningún año de resultado tiene filas de la cohorte D")
    datos = pl.concat(partes)
    en_ranking = datos["anio"].is_in(ranking_cfg)
    if not en_ranking.any():
        raise ValueError("construir_panel: la ventana de ranking no tiene filas de la cohorte D")
    eventos = int(datos.filter(en_ranking)["y"].sum())
    if eventos == 0 or eventos == int(en_ranking.sum()):
        raise ValueError("la ventana de ranking no tiene ambos desenlaces")
    anios_con_filas = sorted(set(datos["anio"].to_list()))
    anios_ranking = [a for a in ranking_cfg if a in anios_con_filas]
    anio_confirmacion = conf_cfg if conf_cfg in anios_con_filas else None

    # --- historial (necesita todos los egresos, no solo la cohorte D)
    regs: list[list[int]] = []
    if con_historial:
        base_regs = regimenes if regimenes is not None else hospitales.regimenes_de({a: ids_validos[a] for a in anios}, u)
        regs = [r for r in ([a for a in reg if a in anios] for reg in base_regs) if r]
        h = historial_por_episodio(pl.concat(episodios), regs, int(u["ventana_historial_dias"]))
        pos = datos["__fila"].to_numpy()
        datos = datos.drop("__fila").with_columns(
            h["n_previos"].gather(pos).cast(pl.Float64).alias("N_EGRESOS_12M"),
            h["dias_previo"].gather(pos).cast(pl.Float64).alias("DIAS_DESDE_EGRESO_PREVIO"),
            h["viable"].gather(pos).cast(pl.Int8).alias("HISTORIA_DISPONIBLE"),
        )

    # --- niveles de las categóricas (con la ventana de ranking) y recodificado
    registro = registro_variables(con_historial)
    niveles: dict[str, list[str]] = {}
    tablas_niveles: list[pl.DataFrame] = []
    ventanas = [(V_RANK, anios_ranking)] + ([(V_CONF, [anio_confirmacion])] if anio_confirmacion is not None else [])
    for nombre, v in registro.items():
        if v.tipo != "cat" or nombre not in datos.columns:
            continue
        if v.es_referencia:
            niv = calcular_niveles_referencia(datos[nombre])
        else:
            niv = calcular_niveles(datos[nombre], datos["anio"], ranking_cfg, tope_niveles)
        datos = datos.with_columns(recodificar(datos[nombre], niv).alias(nombre))
        niveles[nombre] = niv.niveles
        if not v.es_referencia:
            tablas_niveles.append(tabla_de_niveles(nombre, datos[nombre], datos["anio"], niv, ventanas))
    orden = ["anio", "y"] + [n for n in registro if n in datos.columns]
    datos = datos.select(orden)

    # --- cardinalidad de ESPECIALIDAD_MEDICA por hospital: sobre el texto SIN tope de niveles y antes de recortar (H02b)
    esp = pl.concat(especialidad) if especialidad else pl.DataFrame({"t": [], "h": []}, schema={"t": pl.String, "h": pl.String})
    cardinalidad = cardinalidad_especialidad(esp["t"], esp["h"], MIN_FILAS_HOSPITAL)
    return Panel(
        datos=datos, variables=registro, niveles=niveles, muestreo=tabla_muestreo(filas_muestreo), niveles_tabla=pl.concat(tablas_niveles), regimenes=regs,
        con_historial=con_historial, anios_ranking=anios_ranking, anio_confirmacion=anio_confirmacion, cardinalidad_especialidad=cardinalidad,
    )


# =================================================================================================== escritura
def escribir_tablas(ctx: Contexto, panel: Panel, tarea: str = TAREA) -> None:
    """Escribe ``t23_panel_muestreo.csv`` y ``t23_panel_niveles.csv`` y registra las métricas del panel. Idempotente.

    Si una tabla ya figura en el índice de tablas del contexto no se reescribe y conserva la ``tarea`` del primer escritor (la Tarea 11 llama a esta función con
    ``tarea="T11"`` para que una corrida solo de T11 tenga igualmente los niveles). Las métricas ``panel.n_filas_ranking`` y ``panel.n_filas_2023`` (más las de eventos)
    llevan ``conteo=True`` y alcance explícito; ninguna cubre 2024.
    """
    existentes = set(ctx.indice_tablas()["nombre"].to_list())
    if "t23_panel_muestreo.csv" not in existentes:
        ctx.escribir_tabla(
            "t23_panel_muestreo.csv", panel.muestreo, tarea=tarea, descripcion="Cohorte D por año de resultado, tamaño de la muestra del panel y si se muestreó (tope por año)",
            cols_n=["n_cohorte_d", "n_muestra", "n_eventos_muestra"], dependientes=["fraccion_muestreada"],
        )
        m = panel.muestreo
        rank, conf = m.filter(pl.col("ventana") == "ranking"), m.filter(pl.col("ventana") == "confirmacion")
        hay_conf = panel.anio_confirmacion is not None and conf.height > 0
        for clave, columna in (("n_filas", "n_muestra"), ("n_eventos", "n_eventos_muestra")):
            ctx.registrar_metrica(f"panel.{clave}_ranking", int(rank[columna].sum()), tarea=tarea, tabla="t23_panel_muestreo.csv", conteo=True, alcance=V_RANK)
            ctx.registrar_metrica(
                f"panel.{clave}_{V_CONF}", int(conf[columna].sum()) if hay_conf else None, tarea=tarea, tabla="t23_panel_muestreo.csv", conteo=True, alcance=V_CONF,
                nota="" if hay_conf else "confirmación no disponible",
            )
    if "t23_panel_niveles.csv" not in existentes:
        ctx.escribir_tabla(
            "t23_panel_niveles.csv", panel.niveles_tabla, tarea=tarea, descripcion="Niveles de cada categórica del panel por ventana (conteos y proporción; incluye OTROS, (sin mapeo) y (sin dato))",
            cols_n=["n"], dependientes=["prop"],
        )


def escribir_por_bloques(
    ctx: Contexto, nombre: str, df: pl.DataFrame, bloques: Sequence[tuple[Sequence[str], Sequence[str]]], *, tarea: str, descripcion: str
) -> Any:
    """Escribe una tabla en la que una misma fila mezcla conteos de bases distintas (por ejemplo 2019-2022 y 2023), con supresión por bloques.

    DESVIACIÓN DECLARADA de la regla de privacidad del encargo ("si una tabla mezcla conteos de bases distintas, sepárala en varias tablas"): esta función la
    reemplaza porque el análisis exige la asociación de 2023 en columna aparte de la misma fila (pares redundantes, proxy, importancia, fuga). Lo que compensa: cada bloque es un
    par ``(conteos, cifras dependientes)`` y un conteo entre 1 y 9 anula ese conteo y solo las cifras de SU bloque (``supresion.suprimir_bloques`` con ``minimo=ctx.celda_minima``),
    de modo que no se anulan cifras ajenas; y luego se llama a ``ctx.escribir_tabla`` con TODOS los conteos de todos los bloques en ``cols_n`` y todas sus cifras en
    ``dependientes``, así el escáner de privacidad (que solo revisa las columnas registradas) vigila también los conteos de la segunda base. Tras el primer paso no queda ninguna
    celda pequeña en esas columnas, así que ``escribir_tabla`` no cambia nada y actúa solo como red de seguridad: si el paso por bloques fallara, anularía la fila entera, que es el lado seguro.

    Las celdas anuladas en el primer paso se suman a ``celdas_suprimidas`` de la entrada de la tabla en el índice (``escribir_tabla`` solo cuenta las que anula ella), para que
    la auditoría no subinforme. Lanza ``ValueError`` si alguna columna de algún bloque no existe en ``df``: ``suprimir`` y ``escribir_tabla`` ignoran en silencio las ausentes y un
    error de tipeo desactivaría la supresión.
    """
    bloques = [(list(cols_n), list(deps)) for cols_n, deps in bloques]
    ausentes = sorted({c for cols_n, deps in bloques for c in [*cols_n, *deps] if c not in df.columns})
    if ausentes:
        raise ValueError(f"escribir_por_bloques: las columnas {ausentes} de los bloques no existen en la tabla {nombre}")
    protegida, n_celdas = supresion.suprimir_bloques(df, bloques, minimo=ctx.celda_minima)
    cols_n = list(dict.fromkeys(c for cols, _ in bloques for c in cols))
    dependientes = [c for c in dict.fromkeys(c for _, deps in bloques for c in deps) if c not in cols_n]
    ruta = ctx.escribir_tabla(nombre, protegida, tarea=tarea, descripcion=descripcion, cols_n=cols_n, dependientes=dependientes)
    ctx._tablas[nombre]["celdas_suprimidas"] += int(n_celdas)
    return ruta


# FIN-panel
