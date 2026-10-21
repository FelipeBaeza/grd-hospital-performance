"""Cortafuegos del año expuesto (2024) con registro de accesos.

2024 ya fue evaluado por el pipeline del repositorio (Fase 2, sección 4.2). Esta suite solo lo usa
para estructura y calidad. Política:

- Año distinto del expuesto: se permite y se registra.
- Año expuesto con un propósito fuera de ``propositos_permitidos`` (por ejemplo ``resultado`` o
  ``seleccion``): se BLOQUEA, se registra y se lanza ``AccesoBloqueado``.
- Año expuesto con columnas de resultado (TIPOALTA, FECHAALTA, ...):
    * propósito ``estructura``: se entregan ENMASCARADAS (solo patrón de caracteres);
    * propósito ``calidad`` con reglas autorizadas (lista blanca): se entregan en claro, pero las
      reglas solo devuelven indicadores de violación, nunca distribuciones de resultado;
    * propósito ``calidad`` sin reglas autorizadas: ENMASCARADAS.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import polars as pl

PERMITIDO = "PERMITIDO"
ENMASCARADO = "PERMITIDO_ENMASCARADO"
REGLA_CALIDAD = "PERMITIDO_REGLA_CALIDAD"
BLOQUEADO = "BLOQUEADO"

ESQUEMA_REGISTRO = {
    "secuencia": pl.Int64,
    "tarea": pl.String,
    "anio": pl.Int64,
    "proposito": pl.String,
    "n_columnas": pl.Int64,
    "columnas_resultado": pl.String,
    "reglas": pl.String,
    "decision": pl.String,
    "motivo": pl.String,
}


class AccesoBloqueado(PermissionError):
    """Se intentó leer el año expuesto con un propósito o columnas no permitidos."""


@dataclass(frozen=True)
class Autorizacion:
    decision: str
    columnas_enmascaradas: tuple[str, ...] = ()


class Cortafuegos:
    def __init__(
        self,
        anio_expuesto: int = 2024,
        propositos_permitidos: Iterable[str] = ("estructura", "calidad"),
        columnas_resultado: Iterable[str] = (),
        reglas_autorizadas: Iterable[str] = (),
    ) -> None:
        self.anio_expuesto = int(anio_expuesto)
        self.propositos_permitidos = frozenset(propositos_permitidos)
        self.columnas_resultado = frozenset(columnas_resultado)
        self.reglas_autorizadas = frozenset(reglas_autorizadas)
        self._entradas: list[dict] = []

    @classmethod
    def desde_umbrales(cls, umbrales, reglas_autorizadas: Iterable[str] = ()) -> "Cortafuegos":
        return cls(
            anio_expuesto=umbrales.particion["anio_expuesto"],
            propositos_permitidos=umbrales.cortafuegos["propositos_permitidos"],
            columnas_resultado=umbrales.cortafuegos["columnas_resultado"],
            reglas_autorizadas=reglas_autorizadas,
        )

    # ------------------------------------------------------------------ política
    def autorizar(
        self,
        anio: int,
        columnas: Iterable[str],
        proposito: str,
        tarea: str = "",
        reglas: Iterable[str] = (),
    ) -> Autorizacion:
        columnas = list(columnas)
        reglas = [r for r in reglas]
        cols_res = sorted(set(columnas) & self.columnas_resultado)

        def registrar(decision: str, motivo: str) -> None:
            self._entradas.append(
                {
                    "secuencia": len(self._entradas) + 1,
                    "tarea": tarea,
                    "anio": int(anio),
                    "proposito": proposito,
                    "n_columnas": len(columnas),
                    "columnas_resultado": ";".join(cols_res),
                    "reglas": ";".join(reglas),
                    "decision": decision,
                    "motivo": motivo,
                }
            )

        if int(anio) != self.anio_expuesto:
            registrar(PERMITIDO, "año no expuesto")
            return Autorizacion(PERMITIDO)

        if proposito not in self.propositos_permitidos:
            motivo = f"el año {self.anio_expuesto} solo admite {sorted(self.propositos_permitidos)}"
            registrar(BLOQUEADO, motivo)
            raise AccesoBloqueado(f"{motivo}; se pidió {proposito!r} (tarea {tarea or '-'})")

        if not cols_res:
            registrar(PERMITIDO, "año expuesto sin columnas de resultado")
            return Autorizacion(PERMITIDO)

        if proposito == "calidad" and reglas and all(r in self.reglas_autorizadas for r in reglas):
            registrar(REGLA_CALIDAD, "regla de calidad autorizada: solo indicadores de violación")
            return Autorizacion(REGLA_CALIDAD)

        registrar(ENMASCARADO, "columnas de resultado entregadas solo como patrón")
        return Autorizacion(ENMASCARADO, tuple(cols_res))

    # ------------------------------------------------------------------ registro
    def registro(self) -> pl.DataFrame:
        if not self._entradas:
            return pl.DataFrame(schema=ESQUEMA_REGISTRO)
        return pl.DataFrame(self._entradas, schema=ESQUEMA_REGISTRO)

    def columnas_protegidas(self, anio: int) -> frozenset[str]:
        """Columnas de resultado que llegan enmascaradas en ese año (vacío si el año no está expuesto)."""
        return self.columnas_resultado if int(anio) == self.anio_expuesto else frozenset()

    def n_bloqueos(self) -> int:
        return sum(1 for e in self._entradas if e["decision"] == BLOQUEADO)

    def n_lecturas_en_claro_expuestas(self) -> int:
        """Accesos al año expuesto con columnas de resultado en claro (solo reglas autorizadas)."""
        return sum(
            1
            for e in self._entradas
            if e["anio"] == self.anio_expuesto and e["decision"] == REGLA_CALIDAD
        )

    def violaciones(self) -> int:
        """Entradas permitidas que contradicen la política (el invariante es que sea 0)."""
        malas = 0
        for e in self._entradas:
            if e["anio"] != self.anio_expuesto or e["decision"] == BLOQUEADO:
                continue
            if e["proposito"] not in self.propositos_permitidos:
                malas += 1
            elif e["columnas_resultado"] and e["decision"] == PERMITIDO:
                malas += 1
            elif e["decision"] == REGLA_CALIDAD:
                reglas = [r for r in e["reglas"].split(";") if r]
                if not reglas or not all(r in self.reglas_autorizadas for r in reglas):
                    malas += 1
        return malas
