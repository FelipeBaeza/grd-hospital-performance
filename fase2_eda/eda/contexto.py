"""Contexto de ejecución: une fuente de datos, cortafuegos, escritor de tablas y registro de figuras."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from . import config, rutas
from .cortafuegos import Cortafuegos
from .graficos import RegistroFiguras
from .lector import Fuente, Lector, resolver_fuente
from .manifiesto import sha256_archivo
from .supresion import suprimir

ESQUEMA_TABLAS = {
    "nombre": pl.String,
    "tarea": pl.String,
    "descripcion": pl.String,
    "filas": pl.Int64,
    "columnas_conteo": pl.String,
    "celdas_suprimidas": pl.Int64,
    "fuente": pl.String,
    "sha256": pl.String,
}

ESQUEMA_METRICAS = {
    "clave": pl.String,
    "valor": pl.Float64,
    "ic_bajo": pl.Float64,
    "ic_alto": pl.Float64,
    "tarea": pl.String,
    "tabla": pl.String,
    "nivel": pl.String,
    "alcance": pl.String,
    "nota": pl.String,
}


def _limpiar_numericos(df: pl.DataFrame, decimales: int = 6) -> pl.DataFrame:
    """NaN e infinitos pasan a nulo y los decimales se redondean (salida determinista)."""
    exprs = []
    for nombre, tipo in df.schema.items():
        if tipo in (pl.Float32, pl.Float64):
            c = pl.col(nombre)
            exprs.append(
                pl.when(c.is_nan() | c.is_infinite()).then(None).otherwise(c.round(decimales)).alias(nombre)
            )
    return df.with_columns(exprs) if exprs else df


@dataclass
class Contexto:
    fuente: Fuente
    umbrales: config.Umbrales
    dir_salida: Path
    lector: Lector
    cortafuegos: Cortafuegos
    figuras: RegistroFiguras
    dir_tablas: Path
    dir_figuras: Path
    dir_informes: Path
    semilla: int
    _tablas: dict[str, dict[str, Any]] = field(default_factory=dict)
    _metricas: list[dict[str, Any]] = field(default_factory=list)
    _cifras_informe: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ creación
    @classmethod
    def crear(
        cls,
        etiqueta: str,
        *,
        data_root: Path | str | None = None,
        dir_salida_base: Path | str | None = None,
        umbrales: config.Umbrales | None = None,
        reglas_autorizadas: Iterable[str] | None = None,
        anios: Iterable[int] | None = None,
        archivos_explicitos: dict[int, Path | str] | None = None,
        fuente: Fuente | None = None,
    ) -> "Contexto":
        umbrales = umbrales or config.cargar_umbrales()
        if reglas_autorizadas is None:
            # Reglas de calidad que pueden leer columnas de resultado del año expuesto (config/reglas_calidad.toml).
            from . import calidad

            reglas_autorizadas = calidad.ids_autorizados_expuesto(calidad.cargar_reglas())
        fuente = fuente or resolver_fuente(
            etiqueta, data_root=data_root, anios=anios, archivos_explicitos=archivos_explicitos
        )
        raiz = rutas.dir_salida(etiqueta, Path(dir_salida_base) if dir_salida_base else None)
        sub = rutas.subdirs_salida(raiz)
        cf = Cortafuegos.desde_umbrales(umbrales, reglas_autorizadas=reglas_autorizadas)
        return cls(
            fuente=fuente,
            umbrales=umbrales,
            dir_salida=raiz,
            lector=Lector(fuente, cf),
            cortafuegos=cf,
            figuras=RegistroFiguras(sub["figuras"], sub["tablas"], etiqueta),
            dir_tablas=sub["tablas"],
            dir_figuras=sub["figuras"],
            dir_informes=sub["informes"],
            semilla=int(umbrales.meta.get("semilla", 0)),
        )

    @property
    def etiqueta(self) -> str:
        return self.fuente.etiqueta

    @property
    def celda_minima(self) -> int:
        return int(self.umbrales["celda_minima"])

    # -------------------------------------------------------------------- tablas
    def escribir_tabla(
        self,
        nombre: str,
        df: pl.DataFrame,
        *,
        tarea: str | int,
        descripcion: str,
        cols_n: Sequence[str] = (),
        dependientes: Sequence[str] = (),
    ) -> Path:
        """Escribe una tabla CSV con supresión de celdas pequeñas y la registra en el índice.

        ``cols_n`` son las columnas de conteo que se protegen; ``dependientes`` son columnas
        derivadas de esos conteos que se anulan junto con ellos.
        """
        if not nombre.endswith(".csv"):
            raise ValueError("las tablas se escriben como .csv")
        protegida, n_sup = suprimir(df, cols_n, minimo=self.celda_minima, dependientes=dependientes)
        protegida = _limpiar_numericos(protegida)
        destino = self.dir_tablas / nombre
        protegida.write_csv(destino, float_scientific=False)
        self._tablas[nombre] = {
            "nombre": nombre,
            "tarea": str(tarea),
            "descripcion": descripcion,
            "filas": protegida.height,
            "columnas_conteo": ";".join(c for c in cols_n if c in df.columns),
            "celdas_suprimidas": int(n_sup),
            "fuente": self.etiqueta,
            "sha256": sha256_archivo(destino),
        }
        return destino

    def leer_tabla(self, nombre: str, columnas_texto: Sequence[str] = ()) -> pl.DataFrame:
        """Lee una tabla ya escrita (la fuente de verdad de figuras e informes)."""
        ruta = self.dir_tablas / nombre
        if not ruta.is_file():
            raise FileNotFoundError(f"no existe la tabla {nombre}")
        sobre = {c: pl.String for c in columnas_texto}
        return pl.read_csv(ruta, infer_schema_length=100000, schema_overrides=sobre or None)

    def indice_tablas(self) -> pl.DataFrame:
        if not self._tablas:
            return pl.DataFrame(schema=ESQUEMA_TABLAS)
        return pl.DataFrame([self._tablas[k] for k in sorted(self._tablas)], schema=ESQUEMA_TABLAS)

    # ------------------------------------------------------------------- métricas
    def registrar_metrica(
        self,
        clave: str,
        valor: float | int | None,
        *,
        tarea: str | int,
        tabla: str,
        ic_bajo: float | None = None,
        ic_alto: float | None = None,
        nivel: str | None = None,
        nota: str = "",
        conteo: bool = False,
        alcance: str = "",
    ) -> None:
        """Registra una cifra clave para el ledger. ``nivel`` es ``extracto`` o ``completo``.

        ``alcance`` son los años que cubre la medición (por ejemplo ``2019-2023``); vacío significa que cubre
        los mismos años que la cifra documentada. Si ``conteo`` es verdadero y el valor está entre 1 y 9, se
        registra sin valor (regla P2).
        """
        if conteo and valor is not None and 0 < float(valor) < self.celda_minima:
            valor = None
            nota = (nota + " " if nota else "") + "[suprimido: menos de diez casos]"
        self._metricas = [m for m in self._metricas if m["clave"] != clave]
        self._metricas.append(
            {
                "clave": clave,
                "valor": None if valor is None else float(valor),
                "ic_bajo": None if ic_bajo is None else float(ic_bajo),
                "ic_alto": None if ic_alto is None else float(ic_alto),
                "tarea": str(tarea),
                "tabla": tabla,
                "nivel": nivel or self.etiqueta,
                "alcance": alcance,
                "nota": nota,
            }
        )

    def metricas(self) -> pl.DataFrame:
        if not self._metricas:
            return pl.DataFrame(schema=ESQUEMA_METRICAS)
        return pl.DataFrame(sorted(self._metricas, key=lambda m: m["clave"]), schema=ESQUEMA_METRICAS)

    # -------------------------------------------------------------- cifras de informes
    def registrar_cifras_informe(self, cifras: list[dict[str, Any]], renderizadas: list[str]) -> None:
        """Registra las cifras (marcador, CSV, columna, filtro) usadas por un informe."""
        if not cifras:
            return
        informe = cifras[0]["informe"]
        self._cifras_informe = [c for c in self._cifras_informe if c["informe"] != informe]
        self._cifras_informe.extend(cifras)

    def cifras_informe(self) -> pl.DataFrame:
        esquema = {k: pl.String for k in ("informe", "marcador", "csv", "columna", "filtro", "formato", "texto")}
        if not self._cifras_informe:
            return pl.DataFrame(schema=esquema)
        return pl.DataFrame(self._cifras_informe, schema=esquema)

    # -------------------------------------------------------------------- cierre
    def escribir_registros(self) -> dict[str, Path]:
        """Escribe el registro de accesos, el índice de figuras, el de tablas y las métricas."""
        out: dict[str, Path] = {}
        reg = self.cortafuegos.registro()
        out["registro_accesos"] = self.dir_tablas / "registro_accesos.csv"
        reg.write_csv(out["registro_accesos"])
        out["figuras_indice"] = self.dir_tablas / "figuras_indice.csv"
        self.figuras.escribir_indice(out["figuras_indice"])
        out["metricas_eda"] = self.dir_tablas / "metricas_eda.csv"
        _limpiar_numericos(self.metricas()).write_csv(out["metricas_eda"], float_scientific=False)
        out["tablas_indice"] = self.dir_tablas / "tablas_indice.csv"
        self.indice_tablas().write_csv(out["tablas_indice"])
        out["cifras_informe"] = self.dir_tablas / "cifras_informe.csv"
        self.cifras_informe().write_csv(out["cifras_informe"])
        return out
