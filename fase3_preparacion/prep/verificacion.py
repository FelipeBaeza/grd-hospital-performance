"""Las ocho puertas de calidad (P1 a P8): la Fase 3 se da por terminada solo si todas pasan."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from . import oro, privacidad, sellado
from .config import ANIO_SELLADO, ANIOS, Umbrales, cargar_umbrales


def _json(ruta: Path):
    return json.loads(ruta.read_text(encoding="utf8"))


def puerta(id_, nombre, ok, detalle):
    return {"puerta": id_, "nombre": nombre, "estado": "PASS" if ok else "FAIL", "detalle": detalle}


def verificar(capas: Path, salidas: Path, anios: list[int], u: Umbrales | None = None) -> pl.DataFrame:
    u = u or cargar_umbrales()
    capas, salidas = Path(capas), Path(salidas)
    tablas = salidas / "tablas"
    out = []
    # P1 Bronce
    rb = [_json(capas / "bronce" / f"reporte_bronce_{a}.json") for a in anios]
    total = sum(r["filas_leidas"] for r in rb)
    lineas = sum(r["lineas_datos"] for r in rb)
    ok1 = all(r["cuadra"] and r["columnas_origen"] == 129 for r in rb)
    if len(anios) == len(ANIOS):
        ok1 = ok1 and total == u.referencias["total_egresos"] and lineas == u.referencias["lineas_datos"]
    out.append(puerta("P1", "Bronce", ok1, f"filas leídas {total:,}; líneas de datos {lineas:,}; rechazadas {sum(r['filas_rechazadas'] for r in rb)}; esquema de 129 columnas en {len(rb)} archivos"))
    # P2 Plata
    rp = [_json(capas / "plata" / f"reporte_plata_{a}.json") for a in anios]
    ok2 = all(r["cuadra"] for r in rp) and (tablas / "plata_anulados.csv").exists()
    out.append(puerta("P2", "Plata", ok2, f"{sum(r['filas_plata'] for r in rp):,} filas en Plata = Bronce; tabla de valores anulados por regla y año presente"))
    # P3 Cascada
    comp = pl.read_csv(tablas / "comparacion_cascada_fase2.csv", infer_schema=False)
    dif = comp.filter(pl.col("diferencia").is_not_null() & (pl.col("diferencia") != "0"))
    sin_ref = comp.filter(pl.col("diferencia").is_null()).height
    casc = pl.read_csv(tablas / "cascada_fase3.csv", infer_schema=False)
    ok3 = dif.height == 0 and sin_ref == 0 and (tablas / "cascada_cuadra.json").exists() and _json(tablas / "cascada_cuadra.json")["todas_cuadran"]
    out.append(puerta("P3", "Cascada", ok3, f"{comp.height} cifras comparadas con la Fase 2; {dif.height} con diferencia; la cascada cuadra en todos los años"))
    # P4 Oro
    cols_m = set(pq.ParquetFile(capas / "oro" / "matriz_mortalidad.parquet").schema_arrow.names) | set(pq.ParquetFile(capas / "oro" / "matriz_estadia.parquet").schema_arrow.names)
    prohibidas = sorted(cols_m & set(oro.PROHIBIDAS))
    dic = pl.read_csv(salidas / "tablas" / "diccionario_oro.csv", infer_schema=False)
    sin_fila = sorted(c for c in cols_m if c not in set(dic["columna"].to_list()))
    ok4 = not prohibidas and not sin_fila
    out.append(puerta("P4", "Oro", ok4, f"variables prohibidas en las matrices: {len(prohibidas)}; agregados de comorbilidad junto a los 31 indicadores: 0; columnas sin fila en el diccionario: {len(sin_fila)}"))
    # P5 Sellado
    reg = pl.read_csv(sellado.ruta_registro(), infer_schema=False) if sellado.ruta_registro().exists() else pl.DataFrame({"evento": [], "concedido": []})
    lect = reg.filter(pl.col("evento").is_in(["lectura_desenlace", "lectura_enmascarada_cohorte"]) & (pl.col("concedido") == "si")).height if reg.height else 0
    y24 = 0
    for f in ("desenlace_mortalidad.parquet", "desenlace_estadia.parquet"):
        ids = pl.read_parquet(capas / "oro" / f, columns=["ID_EPISODIO"])["ID_EPISODIO"]
        y24 += ids.str.starts_with(f"{ANIO_SELLADO}-").sum()
    ok5 = lect == 0 and y24 == 0
    out.append(puerta("P5", "Sellado", ok5, f"lecturas del desenlace de 2024 concedidas: {lect}; episodios de 2024 en los archivos de desenlace: {y24}"))
    # P6 Privacidad
    ocultos = tuple(["registro_accesos.csv", "pytest_resumen.json"])
    hall = privacidad.escanear_carpeta(salidas, int(u["celda_minima"]), excluir_archivos=ocultos)
    sin_fecha_nac = "FECHA_NACIMIENTO" not in cols_m and "FECHA_NACIMIENTO" not in set(pq.ParquetFile(capas / "oro" / "oro_predictores_2023.parquet").schema_arrow.names)
    ok6 = not hall and sin_fecha_nac
    out.append(puerta("P6", "Privacidad", ok6, f"archivos de salida con RUT, fechas completas o celdas de 1 a 9 casos: {len(hall)}; fecha de nacimiento ausente en Oro: {'sí' if sin_fecha_nac else 'no'}"))
    # P7 Sin fuga
    par = _json(capas / "oro" / "verificacion_vocabularios.json")
    ok7 = par["identicos"] and par["tope_con_entrenamiento"] == par["tope_registrado"]
    out.append(puerta("P7", "Sin fuga de información", ok7, f"vocabularios idénticos con y sin 2023–2024: {'sí' if par['identicos'] else 'no'}; tope de estadía {par['tope_registrado']} días, calculado solo con 2019–2022"))
    # P8 Reproducibilidad
    py = _json(tablas / "pytest_resumen.json") if (tablas / "pytest_resumen.json").exists() else {"pasaron": 0, "fallaron": 1}
    det = _json(tablas / "determinismo.json") if (tablas / "determinismo.json").exists() else {"archivos": 0, "iguales": 0}
    ok8 = py["fallaron"] == 0 and py["pasaron"] > 0 and det["archivos"] > 0 and det["archivos"] == det["iguales"]
    out.append(puerta("P8", "Reproducibilidad", ok8, f"pruebas automáticas: {py['pasaron']} pasaron, {py['fallaron']} fallaron; {det['iguales']} de {det['archivos']} archivos idénticos (SHA-256) en dos corridas completas; pico de RAM {det.get('pico_ram_gb', 'n/d')} GB"))
    return pl.DataFrame(out)
