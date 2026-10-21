"""Generador de informes en Markdown. Falla si una cifra no viene de un CSV.

Reglas:
- La prosa se escribe como plantilla con marcadores ``{nombre}`` o ``{nombre|formato}``; cada marcador
  se resuelve con una ``Cifra`` leída de un CSV de ``salidas/<fuente>/tablas``.
- La prosa no puede contener cifras escritas a mano: ``validar_narrativa`` rechaza cualquier dígito
  fuera de un conjunto cerrado de patrones (años, números de sección, identificadores como R1 o
  ELIX_04, números de decisión, etc.).
- Las tablas se renderizan desde DataFrames leídos del CSV y las figuras se incrustan desde el
  índice de figuras (cada una con su leyenda y su CSV de origen).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl


MARCADOR_TABLA = "(tabla)"
MARCADOR_TABLA_DERIVADA = "(tabla derivada)"
"""Marcadores de las tablas en el registro de cifras: una tabla derivada tiene columnas calculadas a partir de su CSV (conteos por grupo, uniones) que no están en su encabezado."""


class CifraSinRespaldo(ValueError):
    """Hay una cifra en el texto que no proviene de un CSV."""


# ------------------------------------------------------------------------------ formatos
def fmt_entero(v: float | int) -> str:
    return f"{int(round(float(v))):,}".replace(",", ".")


def fmt_dec(v: float | int, d: int) -> str:
    s = f"{float(v):,.{d}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


FORMATOS = {
    "entero": lambda v: fmt_entero(v),
    "dec1": lambda v: fmt_dec(v, 1),
    "dec2": lambda v: fmt_dec(v, 2),
    "dec3": lambda v: fmt_dec(v, 3),
    "dec4": lambda v: fmt_dec(v, 4),
    "pct1": lambda v: fmt_dec(v, 1) + " %",
    "pct2": lambda v: fmt_dec(v, 2) + " %",
    "prop1": lambda v: fmt_dec(float(v) * 100, 1) + " %",
    "prop2": lambda v: fmt_dec(float(v) * 100, 2) + " %",
    "texto": lambda v: str(v),
}


@dataclass(frozen=True)
class Cifra:
    nombre: str
    valor: object
    csv: str
    columna: str
    filtro: str

    def texto(self, formato: str = "entero") -> str:
        if self.valor is None:
            return "n/d"
        return FORMATOS[formato](self.valor)


def leer_cifra(df: pl.DataFrame, csv: str, columna: str, nombre: str = "", **filtro) -> Cifra:
    """Extrae un valor de una tabla. Debe quedar exactamente una fila tras filtrar."""
    sel = df
    for k, v in filtro.items():
        if k not in sel.columns:
            raise CifraSinRespaldo(f"{csv}: no existe la columna de filtro {k}")
        sel = sel.filter(pl.col(k).cast(pl.String) == str(v))
    if columna not in sel.columns:
        raise CifraSinRespaldo(f"{csv}: no existe la columna {columna}")
    if sel.height != 1:
        raise CifraSinRespaldo(f"{csv}: el filtro {filtro} devolvió {sel.height} filas y se esperaba 1")
    return Cifra(nombre or columna, sel[columna][0], csv, columna, json.dumps(filtro, sort_keys=True, ensure_ascii=False))


# --------------------------------------------------------------------------- validación
_PERMITIDOS = [
    r"\b20(?:19|2[0-4])\b",  # años de los datos
    r"\b[Ff]ase \d\b",
    r"\b[Tt]area \d+\b",
    r"\b[Tt]areas \d+(?:(?:, | y | a )\d+)*\b",
    r"(?:secci[oó]n|secciones|§|apartado)\s?\d+(?:\.\d+)*",
    r"\b[Dd]ecisi[oó]n \d+\b",
    r"\b[Dd]ecisiones \d+(?:(?:, | y | a )\d+)*\b",
    r"\bR[1-8]\b",
    r"\bP\d\b",
    r"\bELIX_\d{2}\b",
    r"\bCIE-\d+(?:-MC)?\b",
    r"\bCRISP-DM\b",
    r"\bMDC ?\d+\b",
    r"\bIR_29301_[A-Z_]+\b",
    r"\bIR-GRD\b",
    r"\b(?:DIAGNOSTICO|PROCEDIMIENTO|FECHATRASLADO|SERVICIOTRASLADO|PESORN|SEXORN|RN\d?ESTADO|CONDICIONDEALTANEONATO)\d*(?:ESTADO)?\b",
    r"\b(?:RN\d(?:ESTADO)?)\b",
    r"\b(?:F|T|H|S|DT)\d{2}[a-c]?(?:_\d{2})?\b",
    r"\bt\d{2}_[a-z_]+\b",
    r"\b[1-6]\.\d\b",  # tareas de CRISP-DM (2.1 a 2.4)
    r"(?<![\d,.])95 %",
    r"(?<![\d,.])99,8 %",
    r"\bn entre 1 y 9\b",
    r"\bSHA-256\b",
    r"\bp\d{2}(?:,\d)?\b",  # percentiles nombrados
    r"\b\d+[a-z]*_[A-Za-z_]+\b",
    r"\b[A-Za-z]+\d+[A-Za-z_]*\b",  # identificadores alfanuméricos (COD_GRD6, etc.)
]
_RE_PERMITIDOS = re.compile("|".join(f"(?:{p})" for p in _PERMITIDOS))
_RE_MARCADOR = re.compile(r"\{(\w+)(?:\|(\w+))?\}")


def validar_narrativa(texto: str) -> None:
    """Rechaza dígitos en la prosa que no formen parte de un patrón permitido."""
    sin_marcadores = _RE_MARCADOR.sub(" ", texto)
    sin_codigo = re.sub(r"`[^`]*`", " ", sin_marcadores)
    restante = _RE_PERMITIDOS.sub(" ", sin_codigo)
    m = re.search(r"\d", restante)
    if m:
        a, b = max(0, m.start() - 25), min(len(restante), m.end() + 25)
        raise CifraSinRespaldo(f"cifra sin respaldo en la prosa: ...{restante[a:b].strip()}...")


def _limpiar_linea_para_escaneo(linea: str) -> str | None:
    """Prepara una línea de un informe ya generado para el escaneo de cifras (None = se ignora)."""
    s = linea.rstrip()
    if not s or s.startswith("|") or s.startswith("![") or s.startswith("```"):
        return None
    s = re.sub(r"^#+\s*(?:\d+(?:\.\d+)*\.?\s+)?", "", s)
    s = re.sub(r"^\s*(?:\d+\.|[-*])\s+", "", s)
    return s


def escanear_cifras_md(texto: str, cifras_renderizadas: Iterable[str]) -> list[str]:
    """Devuelve las líneas de prosa con dígitos que no son cifras registradas ni patrones permitidos."""
    tokens = sorted({c for c in cifras_renderizadas if c}, key=len, reverse=True)
    malas: list[str] = []
    en_codigo = False
    for linea in texto.splitlines():
        if linea.strip().startswith("```"):
            en_codigo = not en_codigo
            continue
        if en_codigo:
            continue
        s = _limpiar_linea_para_escaneo(linea)
        if s is None:
            continue
        for t in tokens:
            s = re.sub(r"(?<![\w.,])" + re.escape(t) + r"(?![\w]|[.,]\d)", " ", s)
        s = re.sub(r"\*\*|\*|_(?=\s|$)", "", s)
        try:
            validar_narrativa(s)
        except CifraSinRespaldo:
            malas.append(linea.strip())
    return malas


# --------------------------------------------------------------------------- tablas md
COLUMNAS_SIN_MILES = frozenset(
    {"anio", "posicion", "decision", "tarea", "orden", "n_decision", "cod_hospital", "mes", "id", "cluster", "rango"}
)


def _fmt_celda(v, nombre_col: str, cols_n: set[str], formato: str | None) -> str:
    if v is None:
        return "<10" if nombre_col in cols_n else "n/d"
    if formato:
        return FORMATOS[formato](v)
    if nombre_col in COLUMNAS_SIN_MILES:
        return str(v)
    if isinstance(v, bool):
        return "sí" if v else "no"
    if isinstance(v, int):
        return fmt_entero(v)
    if isinstance(v, float):
        return fmt_dec(v, 2)
    return str(v).replace("|", "/").replace("\n", " ")


class Informe:
    """Constructor de un informe Markdown."""

    def __init__(self, ctx, nombre_archivo: str, titulo: str) -> None:
        self.ctx = ctx
        self.nombre_archivo = nombre_archivo
        self.partes: list[str] = []
        self.cifras: list[dict] = []
        self._renderizadas: list[str] = []
        validar_narrativa(titulo)
        self.partes.append(f"# {titulo}\n")
        self.nota_fuente()

    # -- estructura -------------------------------------------------------------------
    def nota_fuente(self) -> None:
        if self.ctx.etiqueta == "extracto":
            txt = (
                "**Fuente de los datos: extracto.** Los extractos son una fracción menor de los datos y no son "
                "una muestra aleatoria simple: sirven para desarrollar y probar, no para conteos absolutos."
            )
        else:
            txt = "**Fuente de los datos: completo.** Todas las cifras provienen de los archivos completos."
        self.partes.append(txt + "\n")

    def seccion(self, titulo: str, nivel: int = 2) -> None:
        validar_narrativa(re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", titulo))
        self.partes.append(f"{'#' * nivel} {titulo}\n")

    def parrafo(self, plantilla: str, **cifras: Cifra) -> None:
        self.partes.append(self._resolver(plantilla, cifras) + "\n")

    def lista(self, items: Sequence[str], **cifras: Cifra) -> None:
        self.partes.append("\n".join(f"- {self._resolver(i, cifras)}" for i in items) + "\n")

    def _resolver(self, plantilla: str, cifras: Mapping[str, Cifra]) -> str:
        validar_narrativa(plantilla)

        def sustituir(m: re.Match) -> str:
            nombre, fmt = m.group(1), m.group(2) or "entero"
            if nombre not in cifras:
                raise CifraSinRespaldo(f"el marcador {{{nombre}}} no tiene cifra asociada")
            if fmt not in FORMATOS:
                raise CifraSinRespaldo(f"formato desconocido {fmt!r}")
            c = cifras[nombre]
            txt = c.texto(fmt)
            self.cifras.append(
                {
                    "informe": self.nombre_archivo,
                    "marcador": nombre,
                    "csv": c.csv,
                    "columna": c.columna,
                    "filtro": c.filtro,
                    "formato": fmt,
                    "texto": txt,
                }
            )
            self._renderizadas.append(txt)
            return txt

        return _RE_MARCADOR.sub(sustituir, plantilla)

    # -- tablas y figuras -----------------------------------------------------------------
    def tabla(
        self,
        csv: str,
        *,
        columnas: Sequence[str] | None = None,
        encabezados: Mapping[str, str] | None = None,
        formatos: Mapping[str, str] | None = None,
        max_filas: int | None = None,
        columnas_texto: Sequence[str] = (),
        leyenda: str | None = None,
        df: pl.DataFrame | None = None,
    ) -> None:
        tabla = df if df is not None else self.ctx.leer_tabla(csv, columnas_texto=columnas_texto)
        cols = list(columnas) if columnas else list(tabla.columns)
        faltan = [c for c in cols if c not in tabla.columns]
        if faltan:
            raise CifraSinRespaldo(f"{csv}: faltan las columnas {faltan}")
        indice = self.ctx.indice_tablas()
        fila_idx = indice.filter(pl.col("nombre") == csv)
        cols_n = set(fila_idx["columnas_conteo"][0].split(";")) if fila_idx.height else set()
        formatos = dict(formatos or {})
        encabezados = dict(encabezados or {})
        n_total = tabla.height
        mostrar = tabla.head(max_filas) if max_filas else tabla
        filas = ["| " + " | ".join(encabezados.get(c, c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        for fila in mostrar.select(cols).iter_rows():
            filas.append(
                "| " + " | ".join(_fmt_celda(v, c, cols_n, formatos.get(c)) for v, c in zip(fila, cols)) + " |"
            )
        pie = f"*Tabla completa en `tablas/{csv}`*"
        if max_filas and n_total > max_filas:
            pie = f"*Se muestran las primeras {fmt_entero(max_filas)} de {fmt_entero(n_total)} filas. Tabla completa en `tablas/{csv}`*"
            self._renderizadas += [fmt_entero(max_filas), fmt_entero(n_total)]
        if leyenda:
            validar_narrativa(leyenda)
            pie = f"*{leyenda}.* " + pie
        self.partes.append("\n".join(filas) + "\n\n" + pie + "\n")
        # una tabla con columnas calculadas (agregaciones, uniones) se registra como derivada de su CSV: sus columnas no están en el encabezado y el verificador DT02 solo
        # exige que el CSV de origen exista; una tabla directa debe tener todas sus columnas en el encabezado
        derivada = False
        ruta_csv = self.ctx.dir_tablas / csv
        if df is not None and ruta_csv.is_file():
            encabezado = pl.read_csv(ruta_csv, n_rows=0, infer_schema=False).columns
            derivada = any(c not in encabezado for c in cols)
        self.cifras.append(
            {"informe": self.nombre_archivo, "marcador": MARCADOR_TABLA_DERIVADA if derivada else MARCADOR_TABLA, "csv": csv, "columna": ";".join(cols),
             "filtro": "{}", "formato": "tabla", "texto": ""}
        )

    def figura(self, id_figura: str) -> None:
        idx = self.ctx.figuras.indice().filter(pl.col("id_figura") == id_figura)
        if idx.height != 1:
            raise CifraSinRespaldo(f"la figura {id_figura} no está indexada")
        f = idx.row(0, named=True)
        self.partes.append(
            f"![{f['descripcion_alternativa']}](../figuras/{f['archivo_png']})\n\n"
            f"*Figura {id_figura}. {f['titulo']}. Datos de origen: `tablas/{f['csv']}`.*\n"
        )

    def texto_libre_sin_cifras(self, texto: str) -> None:
        validar_narrativa(texto)
        self.partes.append(texto + "\n")

    # -- cierre -----------------------------------------------------------------------------
    def a_markdown(self) -> str:
        return "\n".join(self.partes).rstrip() + "\n"

    def guardar(self) -> Path:
        destino = self.ctx.dir_informes / self.nombre_archivo
        destino.write_text(self.a_markdown(), encoding="utf8")
        malas = escanear_cifras_md(self.a_markdown(), self._renderizadas)
        if malas:
            raise CifraSinRespaldo(f"{self.nombre_archivo}: líneas con cifras sin respaldo: {malas[:3]}")
        self.ctx.registrar_cifras_informe(self.cifras, self._renderizadas)
        return destino


def cifra(ctx, csv: str, columna: str, nombre: str = "", columnas_texto: Sequence[str] = (), **filtro) -> Cifra:
    """Atajo: lee la tabla ``csv`` del contexto y extrae un valor único."""
    return leer_cifra(ctx.leer_tabla(csv, columnas_texto=columnas_texto), csv, columna, nombre, **filtro)
