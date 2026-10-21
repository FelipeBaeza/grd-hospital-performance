"""Capa Bronce: lee cada archivo anual de FONASA tal como viene (todo como texto) y lo guarda en Parquet.

Garantías (Fase 1, sección 6):
- Se valida que el archivo tenga exactamente las 129 columnas; si no, el proceso se detiene.
- Las líneas con otro número de campos se rechazan y se guardan aparte (solo número de línea y número de campos, nunca el contenido).
- Filas leídas + filas rechazadas = líneas de datos del archivo.
- ID_EPISODIO = año + línea del archivo + huella del contenido de la fila (determinista: misma entrada, mismo identificador).
- Se busca en las 129 columnas cualquier valor con forma de RUT. Solo se guardan conteos. Las columnas con hallazgos conocidos
  (config ``sensibles.columnas``) se descartan en los seis años; si aparece un hallazgo en otra columna el proceso se detiene.
- El archivo original no se modifica (regla R4).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from . import esquema
from .config import Umbrales, cargar_umbrales

BOM_UTF8 = b"\xef\xbb\xbf"
VERSION_BRONCE = 1


class EsquemaInvalido(ValueError):
    """El archivo no tiene las 129 columnas esperadas."""


class HallazgoSensibleNuevo(RuntimeError):
    """Se encontró un valor con forma de RUT en una columna que no estaba en la lista de columnas sensibles."""


@dataclass
class Escaneo:
    nombre: str
    sha256: str
    tamano_bytes: int
    tiene_bom: bool
    encabezado: list[str]
    lineas_datos: int
    dist_campos: dict[int, int] = field(default_factory=dict)
    irregulares: list[int] = field(default_factory=list)   # números de línea (1 = encabezado)
    campos_irregulares: list[int] = field(default_factory=list)

    @property
    def rechazadas(self) -> int:
        return len(self.irregulares)

    @property
    def leidas_esperadas(self) -> int:
        return self.lineas_datos - self.rechazadas


def escanear(ruta: Path | str) -> Escaneo:
    """Recorre el archivo una vez. No guarda ningún valor de dato."""
    ruta = Path(ruta)
    h = hashlib.sha256()
    dist: Counter[int] = Counter()
    irreg: list[int] = []
    campos_irreg: list[int] = []
    with open(ruta, "rb") as f:
        primera = f.readline()
        h.update(primera)
        bom = primera.startswith(BOM_UTF8)
        enc = primera.decode("utf-8-sig", errors="replace").rstrip("\r\n").split("|")
        n_datos = 0
        for nro, linea in enumerate(f, start=2):
            h.update(linea)
            n_datos += 1
            n = linea.count(b"|") + 1
            dist[n] += 1
            if n != esquema.N_COLUMNAS:
                irreg.append(nro)
                campos_irreg.append(n)
    return Escaneo(ruta.name, h.hexdigest(), ruta.stat().st_size, bom, enc, n_datos, dict(dist), irreg, campos_irreg)


def validar_esquema(esc: Escaneo) -> None:
    """Exactamente 129 columnas, y la columna 2 debe ser CIP_ENCRIPTADO o su alias histórico ID_BENEFICIARIO."""
    if len(esc.encabezado) != esquema.N_COLUMNAS:
        raise EsquemaInvalido(f"{esc.nombre}: {len(esc.encabezado)} columnas en el encabezado; se esperaban {esquema.N_COLUMNAS}")
    canon = list(esquema.COLUMNAS)
    for i, (visto, esperado) in enumerate(zip(esc.encabezado, canon)):
        if visto == esperado:
            continue
        if i == esquema.POS_ID and visto in esquema.ALIAS_ID:
            continue
        raise EsquemaInvalido(f"{esc.nombre}: la columna {i + 1} se llama {visto!r} y se esperaba {esperado!r}")


def _huella_hex(df: pl.DataFrame) -> list[str]:
    """Huella de contenido de cada fila (BLAKE2b de 64 bits, en hexadecimal). No depende de la versión de Polars ni de ninguna semilla."""
    texto = df.select(pl.concat_str([pl.col(c).fill_null("") for c in df.columns], separator="|").alias("t"))["t"].to_list()
    return [hashlib.blake2b(s.encode("utf8"), digest_size=8).hexdigest() for s in texto]


def _lineas_origen(k0: int, n: int, malas: list[int]) -> np.ndarray:
    """Número de línea del archivo original de las filas k0..k0+n-1 de la lectura sin las líneas rechazadas (1 = encabezado)."""
    linea = np.arange(k0, k0 + n, dtype=np.int64) + 2
    for b in sorted(malas):
        linea = linea + (linea >= b)
    return linea


def _convertir_texto(ruta: Path, esc: Escaneo, destino: Path) -> None:
    """Archivo de texto a Parquet, todo como texto, en streaming; sin las líneas rechazadas. 129 columnas con nombre canónico."""
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
        lf = pl.scan_csv(origen, **opciones)
        lf = lf.rename(dict(zip(lf.collect_schema().names(), esquema.COLUMNAS)))
        lf.sink_parquet(tmp, compression="zstd", row_group_size=100_000, maintain_order=True)
        n = pl.scan_parquet(tmp).select(pl.len()).collect().item()
        if n != esc.leidas_esperadas:
            raise EsquemaInvalido(f"{ruta.name}: la lectura dejó {n} filas y se esperaban {esc.leidas_esperadas}")
        tmp.replace(destino)
    finally:
        if tmp.exists():
            tmp.unlink()
        if limpio is not None and limpio.exists():
            limpio.unlink()


def construir_bronce(anio: int, ruta_txt: Path | str, dir_bronce: Path | str, u: Umbrales | None = None, lote: int | None = None) -> dict:
    """Construye la capa Bronce de un año. Devuelve el reporte (sin ningún valor de dato)."""
    u = u or cargar_umbrales()
    lote = int(lote or u["tamano_lote_filas"])
    ruta_txt, dir_bronce = Path(ruta_txt), Path(dir_bronce)
    dir_bronce.mkdir(parents=True, exist_ok=True)
    esc = escanear(ruta_txt)
    validar_esquema(esc)
    sensibles = list(u.sensibles["columnas"])
    patron = re.compile(u.sensibles["patron_rut"])
    crudo = dir_bronce / f"_crudo_{anio}.parquet"
    _convertir_texto(ruta_txt, esc, crudo)
    final = dir_bronce / f"bronce_{anio}.parquet"
    tmp = final.with_name(final.name + ".tmp")
    hallazgos: Counter[str] = Counter()
    huellas: list[np.ndarray] = []
    escritor = None
    k0 = 0
    try:
        pf = pq.ParquetFile(crudo)
        for batch in pf.iter_batches(batch_size=lote):
            df = pl.from_arrow(batch)
            n = df.height
            # 1) hallazgos con forma de RUT en las 129 columnas (solo conteos)
            cuentas = df.select([pl.col(c).str.contains(patron.pattern).sum().alias(c) for c in df.columns]).row(0)
            for c, v in zip(df.columns, cuentas):
                if v:
                    hallazgos[c] += int(v)
            # 2) huella e identificador (con las 129 columnas, antes de descartar nada)
            hex_ = _huella_hex(df)
            huellas.append(np.array([int(h, 16) for h in hex_], dtype=np.uint64))
            linea = _lineas_origen(k0, n, esc.irregulares)
            ids = [f"{anio}-{int(l):08d}-{h[:8]}" for l, h in zip(linea, hex_)]
            out = df.drop(sensibles).with_columns(
                pl.Series("ID_EPISODIO", ids),
                pl.Series("LINEA_ORIGEN", linea, dtype=pl.Int64),
                pl.lit(anio, dtype=pl.Int32).alias("ANIO_ARCHIVO"),
            )
            out = out.select(["ID_EPISODIO", "ANIO_ARCHIVO", "LINEA_ORIGEN", *[c for c in esquema.COLUMNAS if c not in sensibles]])
            tabla = out.to_arrow()
            if escritor is None:
                escritor = pq.ParquetWriter(tmp, tabla.schema, compression="zstd")
            escritor.write_table(tabla)
            k0 += n
        if escritor is not None:
            escritor.close()
        nuevos = sorted(c for c in hallazgos if c not in sensibles)
        if nuevos:
            raise HallazgoSensibleNuevo(f"{anio}: valores con forma de RUT en columnas no declaradas como sensibles: {nuevos}")
        filas = pq.ParquetFile(tmp).metadata.num_rows
        if filas != esc.leidas_esperadas:
            raise EsquemaInvalido(f"{anio}: Bronce quedó con {filas} filas y se esperaban {esc.leidas_esperadas}")
        tmp.replace(final)
    finally:
        if escritor is not None:
            try:
                escritor.close()
            except Exception:
                pass
        if tmp.exists():
            tmp.unlink()
        if crudo.exists():
            crudo.unlink()
    todas = np.concatenate(huellas) if huellas else np.array([], dtype=np.uint64)
    _, cnt = np.unique(todas, return_counts=True)
    repetidas = int((cnt[cnt > 1] - 1).sum())          # filas que repiten exactamente a otra del mismo año
    (dir_bronce / f"rechazadas_{anio}.csv").write_text(
        "linea,n_campos\n" + "".join(f"{l},{c}\n" for l, c in zip(esc.irregulares, esc.campos_irregulares)), encoding="utf8"
    )
    reporte = {
        "anio": anio,
        "archivo": esc.nombre,
        "sha256_origen": esc.sha256,
        "bytes_origen": esc.tamano_bytes,
        "lineas_datos": esc.lineas_datos,
        "filas_leidas": filas,
        "filas_rechazadas": esc.rechazadas,
        "cuadra": filas + esc.rechazadas == esc.lineas_datos,
        "columnas_origen": esquema.N_COLUMNAS,
        "columnas_bronce": 3 + esquema.N_COLUMNAS - len(sensibles),
        "columnas_descartadas": sensibles,
        "hallazgos_rut": {c: int(v) for c, v in sorted(hallazgos.items())},
        "filas_repetidas_exactas": repetidas,
        "tiene_bom": esc.tiene_bom,
        "version_bronce": VERSION_BRONCE,
    }
    (dir_bronce / f"reporte_bronce_{anio}.json").write_text(json.dumps(reporte, ensure_ascii=False, indent=1), encoding="utf8")
    return reporte
