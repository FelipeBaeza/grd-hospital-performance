"""Lector seguro de los archivos anuales de egresos (texto delimitado por ``|``).

Garantías:
- Todo se lee como TEXTO (se conservan ceros a la izquierda, comas decimales y centinelas).
- Soporta BOM UTF-8 y finales de línea CRLF o LF.
- Las columnas se identifican por POSICIÓN. El nombre de la columna 2 puede ser
  CIP_ENCRIPTADO o ID_BENEFICIARIO: se normaliza al nombre canónico (alias por posición).
- Una línea con un número de campos distinto del esperado se RECHAZA y se registra; se verifica que
  ``leídas + rechazadas = líneas de datos``.
- Todo acceso pasa por el cortafuegos del año expuesto y queda registrado.

Memoria (modo completo): con la base completa (~5,8 millones de egresos, ~3 GB de texto) releer el texto en cada
llamada obliga a parsear el archivo entero y a tener muchas columnas como texto a la vez. Por eso, con la fuente
``completo`` el lector convierte UNA vez cada archivo anual a Parquet (capa Bronce, todo como texto, mismas filas y
mismo orden) en modo streaming, sin cargar el archivo en memoria, y desde ahí lee solo las columnas pedidas. El
archivo original no se modifica (R4). La caché se guarda junto a los datos (``<data-root>/_cache_parquet_fase2``),
nunca en el repositorio, y su nombre incluye la huella SHA-256 del archivo de origen: si el archivo cambia, se
regenera. ``FASE2_CACHE_PARQUET`` permite elegir otra carpeta o desactivarla (``0``).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import polars as pl

from . import config, esquema, rutas
from .cortafuegos import Cortafuegos, ENMASCARADO


class EsquemaInvalido(ValueError):
    """El archivo no tiene el número de columnas esperado o el encabezado es ambiguo."""


class ArchivoNoEncontrado(FileNotFoundError):
    """No se encontró el archivo de un año."""


BOM_UTF8 = b"\xef\xbb\xbf"
VERSION_CACHE = 1
"""Versión del formato de la caché Parquet: si cambia la forma de convertir, se regeneran todos los archivos."""


def carpeta_cache(fuente: "Fuente") -> Path | None:
    """Carpeta de la caché Parquet de una fuente, o ``None`` si no se usa.

    Por defecto solo se usa con la fuente ``completo`` y vive en ``<raíz de datos>/_cache_parquet_fase2``. La variable de
    entorno ``FASE2_CACHE_PARQUET`` la redirige a otra carpeta, o la desactiva con ``0``.
    """
    env = os.environ.get("FASE2_CACHE_PARQUET")
    if env is not None:
        return None if env.strip() in ("", "0") else Path(env)
    if fuente.etiqueta != "completo":
        return None
    return fuente.raiz / "_cache_parquet_fase2"


@dataclass(frozen=True)
class Fuente:
    etiqueta: str
    raiz: Path
    archivos: dict[int, Path]

    def anios(self) -> list[int]:
        return sorted(self.archivos)


def resolver_fuente(
    etiqueta: str,
    data_root: Path | str | None = None,
    fuentes: dict | None = None,
    anios: Iterable[int] | None = None,
    archivos_explicitos: dict[int, Path | str] | None = None,
) -> Fuente:
    """Construye la fuente de datos: año -> archivo, según config/fuentes.toml."""
    if etiqueta not in rutas.FUENTES_VALIDAS:
        raise ValueError(f"etiqueta de fuente inválida: {etiqueta!r}")
    fuentes = fuentes or config.cargar_fuentes_toml()
    anios = sorted(anios) if anios is not None else list(fuentes["meta"]["anios"])

    if etiqueta == "extracto":
        raiz = Path(data_root) if data_root else rutas.RAIZ_PROYECTO / fuentes["extracto"]["raiz_relativa_al_proyecto"]
    else:
        if not data_root:
            raise ValueError("el modo completo requiere --data-root")
        raiz = Path(data_root)
    if not raiz.is_dir():
        raise ArchivoNoEncontrado(f"la raíz de datos no existe: {raiz.name}")

    candidatos = fuentes[etiqueta]["archivos"]
    encontrados: dict[int, Path] = {}
    faltan: list[int] = []
    for anio in anios:
        if archivos_explicitos and anio in archivos_explicitos:
            p = Path(archivos_explicitos[anio])
            if p.is_file():
                encontrados[anio] = p
            else:
                faltan.append(anio)
            continue
        elegido: Path | None = None
        for rel in candidatos.get(str(anio), []):
            p = raiz / rel
            if p.is_file():
                elegido = p
                break
        if elegido is None:
            coincidencias = sorted(
                p for p in raiz.rglob(f"*{anio}*.txt") if p.is_file() and "GRD" in p.name.upper()
            )
            if len(coincidencias) == 1:
                elegido = coincidencias[0]
            elif len(coincidencias) > 1:
                raise ArchivoNoEncontrado(
                    f"hay {len(coincidencias)} archivos candidatos para {anio}; use fuentes.toml o --archivo"
                )
        if elegido is None:
            faltan.append(anio)
        else:
            encontrados[anio] = elegido
    if faltan:
        raise ArchivoNoEncontrado(f"faltan archivos para los años {faltan} en la fuente {etiqueta!r}")
    return Fuente(etiqueta=etiqueta, raiz=raiz, archivos=encontrados)


# ----------------------------------------------------------------------------- escaneo
@dataclass
class EscaneoArchivo:
    nombre: str
    tamano_bytes: int
    sha256: str
    tiene_bom: bool
    encabezado: list[str]
    n_campos_encabezado: int
    lineas_datos: int
    lineas_crlf: int
    lineas_lf: int
    ultima_linea_con_salto: bool
    lineas_vacias: int
    lineas_utf8_invalidas: int
    n_campos_esperado: int
    dist_campos: dict[int, int] = field(default_factory=dict)
    irregulares: list[int] = field(default_factory=list)  # números de línea (1 = encabezado)

    @property
    def lineas_total(self) -> int:
        return self.lineas_datos + 1

    @property
    def rechazadas(self) -> int:
        return len(self.irregulares)

    @property
    def leidas_esperadas(self) -> int:
        return self.lineas_datos - self.rechazadas

    @property
    def fin_de_linea(self) -> str:
        if self.lineas_crlf and self.lineas_lf:
            return "MIXTO"
        if self.lineas_crlf:
            return "CRLF"
        if self.lineas_lf:
            return "LF"
        return "SIN SALTO"

    @property
    def moda_campos(self) -> int:
        return max(self.dist_campos.items(), key=lambda kv: (kv[1], kv[0]))[0] if self.dist_campos else 0


def leer_encabezado(ruta: Path | str) -> list[str]:
    """Nombres de columna del encabezado (sin BOM ni fin de línea)."""
    with open(ruta, "rb") as f:
        linea = f.readline()
    return linea.decode("utf-8-sig").rstrip("\r\n").split("|")


def sha256_archivo(ruta: Path | str) -> str:
    with open(ruta, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def escanear_archivo(ruta: Path | str, n_campos_esperado: int = esquema.N_COLUMNAS) -> EscaneoArchivo:
    """Recorre el archivo una vez y resume su forma. No guarda ningún valor de dato."""
    ruta = Path(ruta)
    sha = sha256_archivo(ruta)
    dist: Counter[int] = Counter()
    irregulares: list[int] = []
    crlf = lf = vacias = invalidas = 0
    con_salto = True
    with open(ruta, "rb") as f:
        primera = f.readline()
        tiene_bom = primera.startswith(BOM_UTF8)
        encabezado = primera.decode("utf-8-sig", errors="replace").rstrip("\r\n").split("|")
        n_datos = 0
        for nro, linea in enumerate(f, start=2):
            n_datos += 1
            if linea.endswith(b"\r\n"):
                crlf += 1
            elif linea.endswith(b"\n"):
                lf += 1
            else:
                con_salto = False
            if not linea.strip():
                vacias += 1
            try:
                linea.decode("utf-8")
            except UnicodeDecodeError:
                invalidas += 1
            n = linea.count(b"|") + 1
            dist[n] += 1
            if n != n_campos_esperado:
                irregulares.append(nro)
    return EscaneoArchivo(
        nombre=ruta.name,
        tamano_bytes=ruta.stat().st_size,
        sha256=sha,
        tiene_bom=tiene_bom,
        encabezado=encabezado,
        n_campos_encabezado=len(encabezado),
        lineas_datos=n_datos,
        lineas_crlf=crlf,
        lineas_lf=lf,
        ultima_linea_con_salto=con_salto,
        lineas_vacias=vacias,
        lineas_utf8_invalidas=invalidas,
        n_campos_esperado=n_campos_esperado,
        dist_campos=dict(dist),
        irregulares=irregulares,
    )


# ------------------------------------------------------------------------------- lector
class Lector:
    """Lee columnas de los archivos anuales aplicando el cortafuegos de 2024."""

    def __init__(self, fuente: Fuente, cortafuegos: Cortafuegos | None = None, cache: Path | None | str = "auto") -> None:
        self.fuente = fuente
        self.cortafuegos = cortafuegos or Cortafuegos()
        self._escaneos: dict[int, EscaneoArchivo] = {}
        self.contabilidad: dict[int, dict[str, int]] = {}
        self.cache: Path | None = carpeta_cache(fuente) if cache == "auto" else (Path(cache) if cache else None)
        self._parquets: dict[int, Path] = {}

    # -- escaneo -------------------------------------------------------------------
    def escaneo(self, anio: int) -> EscaneoArchivo:
        if anio not in self._escaneos:
            self._escaneos[anio] = self._escaneo_con_cache(anio)
        return self._escaneos[anio]

    def _escaneo_con_cache(self, anio: int) -> EscaneoArchivo:
        """Sin caché, escanea. Con caché, reutiliza el escaneo guardado si la huella SHA-256 del archivo no cambió."""
        ruta = self.fuente.archivos[anio]
        if self.cache is None:
            return escanear_archivo(ruta)
        sha = sha256_archivo(ruta)
        json_ruta = self.cache / f"escaneo_{anio}_{sha[:16]}_v{VERSION_CACHE}.json"
        if json_ruta.is_file():
            datos = json.loads(json_ruta.read_text(encoding="utf8"))
            datos["dist_campos"] = {int(k): int(v) for k, v in datos["dist_campos"].items()}
            esc = EscaneoArchivo(**datos)
            if esc.sha256 == sha and esc.nombre == ruta.name:
                return esc
        esc = escanear_archivo(ruta)
        self.cache.mkdir(parents=True, exist_ok=True)
        tmp = json_ruta.with_suffix(".tmp")
        tmp.write_text(json.dumps(esc.__dict__, ensure_ascii=False), encoding="utf8")
        tmp.replace(json_ruta)
        return esc

    def parquet(self, anio: int) -> Path:
        """Archivo Parquet (capa Bronce) del año: se crea la primera vez en streaming, con todo como texto."""
        if anio in self._parquets:
            return self._parquets[anio]
        assert self.cache is not None
        esc = self.escaneo(anio)
        destino = self.cache / f"bronce_{anio}_{esc.sha256[:16]}_v{VERSION_CACHE}.parquet"
        if destino.is_file():
            try:
                n = pl.scan_parquet(destino).select(pl.len()).collect().item()
            except Exception:
                n = -1
            if n == esc.leidas_esperadas:
                self._parquets[anio] = destino
                return destino
            destino.unlink()
        self.cache.mkdir(parents=True, exist_ok=True)
        convertir_a_parquet(self.fuente.archivos[anio], esc, destino)
        self._parquets[anio] = destino
        return destino

    def preparar_cache(self, anios: Iterable[int] | None = None) -> list[Path]:
        """Convierte de antemano los años pedidos (todos por defecto). No hace nada si la caché está desactivada."""
        if self.cache is None:
            return []
        return [self.parquet(a) for a in (anios if anios is not None else self.fuente.anios())]

    def escanear_todo(self) -> dict[int, EscaneoArchivo]:
        return {a: self.escaneo(a) for a in self.fuente.anios()}

    # -- lectura -------------------------------------------------------------------
    def leer(
        self,
        anio: int,
        columnas: Sequence[str] | None = None,
        *,
        proposito: str,
        tarea: str = "",
        reglas: Iterable[str] = (),
        con_anio: bool = False,
        muestra: int | None = None,
        semilla: int = 0,
    ) -> pl.DataFrame:
        """Devuelve las columnas pedidas (todas como texto) de un año.

        ``columnas=None`` pide las 129. Las columnas de resultado del año expuesto pueden venir
        enmascaradas (ver ``cortafuegos``). Lanza ``AccesoBloqueado`` si la política lo impide.
        """
        columnas = list(columnas) if columnas is not None else list(esquema.COLUMNAS)
        desconocidas = [c for c in columnas if c not in esquema.POS]
        if desconocidas:
            raise KeyError(f"columnas desconocidas: {desconocidas}")
        auth = self.cortafuegos.autorizar(anio, columnas, proposito, tarea=tarea, reglas=reglas)

        esc = self.escaneo(anio)
        if esc.n_campos_encabezado != esquema.N_COLUMNAS:
            raise EsquemaInvalido(
                f"el archivo de {anio} tiene {esc.n_campos_encabezado} columnas y se esperaban {esquema.N_COLUMNAS}"
            )
        if len(set(esc.encabezado)) != len(esc.encabezado):
            raise EsquemaInvalido(f"el encabezado de {anio} tiene nombres de columna repetidos")

        idxs = sorted({esquema.POS[c] for c in columnas})
        if self.cache is not None:
            df = pl.read_parquet(self.parquet(anio), columns=idxs)
        else:
            df = self._parsear(self.fuente.archivos[anio], esc, idxs)
        # renombrar por posición al nombre canónico
        renombre = {esc.encabezado[i]: esquema.COLUMNAS[i] for i in idxs if esc.encabezado[i] != esquema.COLUMNAS[i]}
        if renombre:
            df = df.rename(renombre)
        df = df.select(columnas)

        self.contabilidad[anio] = {
            "lineas_datos": esc.lineas_datos,
            "rechazadas": esc.rechazadas,
            "leidas": df.height,
        }
        if df.height != esc.leidas_esperadas:
            raise EsquemaInvalido(
                f"{anio}: se leyeron {df.height} filas y se esperaban {esc.leidas_esperadas} "
                f"(líneas de datos {esc.lineas_datos} menos rechazadas {esc.rechazadas})"
            )

        if auth.decision == ENMASCARADO and auth.columnas_enmascaradas:
            df = df.with_columns(
                [
                    (
                        esquema.mascara_presencia(pl.col(c))
                        if c in esquema.COLUMNAS_SOLO_PRESENCIA
                        else esquema.mascara_patron(pl.col(c))
                    ).alias(c)
                    for c in auth.columnas_enmascaradas
                ]
            )

        if muestra is not None and df.height > muestra:
            rng = np.random.default_rng(int(semilla) + int(anio))
            idx = np.sort(rng.choice(df.height, size=int(muestra), replace=False))
            df = df[idx.tolist()]
        if con_anio:
            df = df.with_columns(pl.lit(int(anio), dtype=pl.Int32).alias("anio"))
        return df

    def leer_por_bloques(
        self,
        anio: int,
        columnas: Sequence[str] | None = None,
        *,
        bloque: int = 16,
        **kw,
    ):
        """Lee las columnas de un año en bloques de ``bloque`` columnas (en el orden pedido) y entrega cada bloque.

        Sirve para recorrer las 129 columnas de un año de la base completa sin tenerlas todas en memoria a la vez.
        Cada bloque pasa por el cortafuegos igual que ``leer``.
        """
        columnas = list(columnas) if columnas is not None else list(esquema.COLUMNAS)
        for i in range(0, len(columnas), int(bloque)):
            yield self.leer(anio, columnas[i:i + int(bloque)], **kw)

    def leer_varios(
        self,
        anios: Iterable[int],
        columnas: Sequence[str] | None = None,
        **kw,
    ) -> pl.DataFrame:
        partes = [self.leer(a, columnas, con_anio=True, **kw) for a in anios]
        return pl.concat(partes) if partes else pl.DataFrame()

    # -- parseo --------------------------------------------------------------------
    @staticmethod
    def _parsear(ruta: Path, esc: EscaneoArchivo, idxs: list[int]) -> pl.DataFrame:
        opciones = dict(
            separator="|",
            has_header=True,
            infer_schema=False,
            quote_char=None,
            encoding="utf8",
            columns=idxs,
        )
        if not esc.irregulares:
            return pl.read_csv(ruta, **opciones)
        malas = set(esc.irregulares)
        buenas: list[bytes] = []
        with open(ruta, "rb") as f:
            for nro, linea in enumerate(f, start=1):
                if nro == 1:
                    buenas.append(linea[len(BOM_UTF8):] if linea.startswith(BOM_UTF8) else linea)
                elif nro not in malas:
                    buenas.append(linea)
        return pl.read_csv(io.BytesIO(b"".join(buenas)), **opciones)


# ------------------------------------------------------------------------------ caché
def convertir_a_parquet(ruta: Path, esc: EscaneoArchivo, destino: Path) -> None:
    """Convierte un archivo anual a Parquet sin cargarlo en memoria (motor streaming de Polars).

    Se leen las mismas filas que ``Lector._parsear``: todo como texto, sin comillas, con BOM retirado y sin las líneas
    con un número de campos distinto del esperado. El resultado se escribe primero en un archivo temporal y se renombra
    al final, de modo que una conversión interrumpida nunca deja un Parquet incompleto con el nombre definitivo.
    """
    opciones = dict(separator="|", has_header=True, infer_schema=False, quote_char=None, encoding="utf8")
    tmp = destino.with_name(destino.name + ".tmp")
    limpio: Path | None = None
    try:
        origen = ruta
        if esc.irregulares:
            malas = set(esc.irregulares)
            fd, nombre = tempfile.mkstemp(suffix=".txt", dir=destino.parent)
            limpio = Path(nombre)
            with os.fdopen(fd, "wb") as out, open(ruta, "rb") as f:
                for nro, linea in enumerate(f, start=1):
                    if nro == 1:
                        out.write(linea[len(BOM_UTF8):] if linea.startswith(BOM_UTF8) else linea)
                    elif nro not in malas:
                        out.write(linea)
            origen = limpio
        pl.scan_csv(origen, **opciones).sink_parquet(tmp, compression="zstd", row_group_size=100_000, maintain_order=True)
        n = pl.scan_parquet(tmp).select(pl.len()).collect().item()
        if n != esc.leidas_esperadas:
            raise EsquemaInvalido(f"{ruta.name}: la conversión a Parquet dejó {n} filas y se esperaban {esc.leidas_esperadas}")
        tmp.replace(destino)
    finally:
        if tmp.exists():
            tmp.unlink()
        if limpio is not None and limpio.exists():
            limpio.unlink()
