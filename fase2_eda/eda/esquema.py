"""Esquema canónico de los archivos de egresos GRD de FONASA (129 columnas, por posición)."""

from __future__ import annotations

import polars as pl

# --- Nombres canónicos por posición (encabezado de los archivos 2019 a 2023) -------------
_INICIO = [
    "COD_HOSPITAL",
    "CIP_ENCRIPTADO",
    "SEXO",
    "FECHA_NACIMIENTO",
    "ETNIA",
    "PROVINCIA",
    "COMUNA",
    "NACIONALIDAD",
    "PREVISION",
    "SERVICIO_SALUD",
    "TIPO_PROCEDENCIA",
    "TIPO_INGRESO",
    "ESPECIALIDAD_MEDICA",
    "TIPO_ACTIVIDAD",
    "FECHA_INGRESO",
    "SERVICIOINGRESO",
]
_TRASLADOS = [f"{p}{i}" for i in range(1, 10) for p in ("FECHATRASLADO", "SERVICIOTRASLADO")]
_ALTA = ["FECHAALTA", "SERVICIOALTA", "TIPOALTA"]
# Bloque de recién nacidos: para cada i, condición de alta, peso, sexo y estado.
_RN: list[str] = []
for _i in range(1, 5):
    _RN += [f"CONDICIONDEALTANEONATO{_i}", f"PESORN{_i}", f"SEXORN{_i}", f"RN{_i}ESTADO"]
_DIAG = [f"DIAGNOSTICO{i}" for i in range(1, 36)]
_PROC = [f"PROCEDIMIENTO{i}" for i in range(1, 31)]
_FINAL = [
    "MEDICOINTERV1_ENCRIPTADO",
    "FECHAPROCEDIMIENTO1",
    "FECHAINTERV1",
    "ESPECIALIDADINTERVENCION",
    "MEDICOALTA_ENCRIPTADO",
    "USOSPABELLON",
    "IR_29301_COD_GRD",
    "IR_29301_PESO",
    "IR_29301_SEVERIDAD",
    "IR_29301_MORTALIDAD",
    "HOSPPROCEDENCIA",
]

COLUMNAS: tuple[str, ...] = tuple(_INICIO + _TRASLADOS + _ALTA + _RN + _DIAG + _PROC + _FINAL)
N_COLUMNAS: int = len(COLUMNAS)
assert N_COLUMNAS == 129, N_COLUMNAS

POS: dict[str, int] = {c: i for i, c in enumerate(COLUMNAS)}
COL_ID: str = "CIP_ENCRIPTADO"
POS_ID: int = POS[COL_ID]
ALIAS_ID: tuple[str, ...] = ("CIP_ENCRIPTADO", "ID_BENEFICIARIO")

DIAG: tuple[str, ...] = tuple(_DIAG)
DIAG_SEC: tuple[str, ...] = tuple(_DIAG[1:])
PROC: tuple[str, ...] = tuple(_PROC)
FECHAS_TRASLADO: tuple[str, ...] = tuple(f"FECHATRASLADO{i}" for i in range(1, 10))
SERVICIOS_TRASLADO: tuple[str, ...] = tuple(f"SERVICIOTRASLADO{i}" for i in range(1, 10))
BLOQUE_RN: tuple[str, ...] = tuple(_RN)
FECHAS: tuple[str, ...] = (
    "FECHA_NACIMIENTO",
    "FECHA_INGRESO",
    "FECHAALTA",
    "FECHAINTERV1",
    "FECHAPROCEDIMIENTO1",
    *FECHAS_TRASLADO,
)

# Valores centinela: faltantes disfrazados (comparación sin tildes ni mayúsculas).
CENTINELAS: frozenset[str] = frozenset(
    {
        "DESCONOCIDO",
        "DESCONOCIDA",
        "NO IDENTIFICADA",
        "NO IDENTIFICADO",
        "NO CONSIGNADO",
        "NO CONSIGNADA",
        "SIN INFORMACION",
        "NO INFORMADO",
        "NO INFORMADA",
        "NULL",
        "NA",
        "N/A",
        "NAN",
    }
)

# Máscaras de patrón: dígitos -> 9, letras -> A (idempotentes).
_RE_DIGITO = r"\p{N}"
_RE_LETRA = r"\p{L}"


def mascara_patron(expr: pl.Expr) -> pl.Expr:
    """Reemplaza dígitos por 9 y letras por A. Conserva puntuación y espacios."""
    return expr.str.replace_all(_RE_DIGITO, "9").str.replace_all(_RE_LETRA, "A")


# Columnas de resultado de texto categórico: en el año expuesto solo se entrega si hay valor o no
# (la longitud o la forma de la etiqueta podría insinuar el desenlace).
COLUMNAS_SOLO_PRESENCIA: tuple[str, ...] = ("TIPOALTA", "SERVICIOALTA")


def mascara_presencia(expr: pl.Expr) -> pl.Expr:
    """``A`` si hay valor y nulo si no hay. No conserva ninguna característica del valor."""
    return pl.when(expr.is_not_null()).then(pl.lit("A")).otherwise(None)


def es_centinela(expr: pl.Expr) -> pl.Expr:
    """True si el valor (sin tildes, en mayúsculas y sin espacios extremos) es un centinela."""
    norm = (
        expr.str.strip_chars()
        .str.to_uppercase()
        .str.replace_all("Ó", "O")
        .str.replace_all("É", "E")
        .str.replace_all("Í", "I")
        .str.replace_all("Á", "A")
        .str.replace_all("Ú", "U")
    )
    return norm.is_in(sorted(CENTINELAS)).fill_null(False)


def normalizar_cie10(expr: pl.Expr) -> pl.Expr:
    """Normaliza un código CIE-10: mayúsculas, sin espacios ni puntos."""
    return expr.str.strip_chars().str.to_uppercase().str.replace_all(".", "", literal=True)


def nombre_unico(nombres: list[str]) -> bool:
    return len(nombres) == len(set(nombres))
