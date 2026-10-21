"""Capa Oro: variables finales del modelo, matrices de mortalidad y de estadía, y particiones.

Predictores del modelo (Fase 1 v3, sección 7 y decisiones confirmadas el 18 de octubre de 2026):
  EDAD_ANIOS, SEXO, ES_FIN_SEMANA, TIPO_INGRESO, TIPO_PROCEDENCIA (y su versión agrupada PROCEDENCIA_AGR), NIVEL_CUIDADO_INGRESO,
  CIE10_3C, GRUPO_CLINICO y las 31 categorías de Elixhauser (ELIX_01 a ELIX_31, solo diagnósticos secundarios, regla R3).
Fuera del modelo: ESPECIALIDAD_MEDICA y SERVICIOINGRESO (solo se usa SERVICIOINGRESO para derivar NIVEL_CUIDADO_INGRESO), MES_INGRESO,
DIA_SEMANA, el historial del paciente, la severidad/peso/riesgo de mortalidad del GRD y todo lo que ocurre durante la hospitalización.

Reglas:
- Los vocabularios (niveles de las categóricas; los poco frecuentes pasan a RARO) se calculan SOLO con la cohorte C de 2019 a 2022.
- La fecha de nacimiento no pasa a Oro (se usó para la edad en Plata).
- Los agregados de comorbilidad (N_COMORB_ELIX, PUNTAJE_VAN_WALRAVEN) y N_DX_SECUNDARIOS van a una tabla aparte, nunca junto a los 31 indicadores.
- Los desenlaces van en archivos distintos de los predictores. El de 2024 sigue sellado: no se lee.
- No se imputa nada: lo que falta queda vacío (nulo) o en el nivel «(sin dato)».
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from . import definiciones as dfn
from . import elixhauser, esquema
from .config import ANIO_SELLADO, ANIOS_RESULTADO, Umbrales, anio_split, cargar_umbrales

CATEGORICAS = ("SEXO", "TIPO_INGRESO", "TIPO_PROCEDENCIA", "PROCEDENCIA_AGR", "GRUPO_CLINICO", "CIE10_3C")
ELIX = elixhauser.IDS
PREDICTORES = ("EDAD_ANIOS", "SEXO", "ES_FIN_SEMANA", "TIPO_INGRESO", "TIPO_PROCEDENCIA", "PROCEDENCIA_AGR", "NIVEL_CUIDADO_INGRESO", "CIE10_3C", "GRUPO_CLINICO", *ELIX)
CLAVES = ("ID_EPISODIO", "ANIO_EGRESO", "SPLIT", "COD_HOSPITAL")
PROHIBIDAS = ("IR_29301_MORTALIDAD", "IR_29301_SEVERIDAD", "IR_29301_PESO", "FECHA_NACIMIENTO", "FECHAALTA", "TIPOALTA", "ESTANCIA_DIAS",
              "N_PROCEDIMIENTOS", "N_TRASLADOS_INTERNOS", "ESPECIALIDAD_MEDICA", "SERVICIOINGRESO", "MES_INGRESO", "DIA_SEMANA", "N_COMORB_ELIX",
              "PUNTAJE_VAN_WALRAVEN", "N_DX_SECUNDARIOS")
"""Columnas que nunca pueden estar en una matriz de predictores."""
COLS_PLATA_ORO = ["ID_EPISODIO", "ANIO_EGRESO", "COD_HOSPITAL", "SEXO", "FECHA_INGRESO", "EDAD_ANIOS", "TIPO_INGRESO", "TIPO_PROCEDENCIA", "SERVICIOINGRESO",
                  "N_DX_SECUNDARIOS", *esquema.DIAG]


def predictores_crudos(df: pl.DataFrame, capitulos: dict[str, str], e: elixhauser.Elixhauser) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Un bloque de Plata (filas de la cohorte C) -> (predictores con las categóricas aún sin agrupar, tabla de línea base)."""
    flags = elixhauser.calcular_flags(df, esquema.DIAG_SEC, e)
    cat3 = df.select(esquema.normalizar_cie10(pl.col("DIAGNOSTICO1")).str.slice(0, 3).alias("c"))["c"]
    cie = dfn.limpiar_categorica(cat3.alias("CIE10_3C"))
    capitulo = pl.Series("GRUPO_CLINICO", [
        dfn.SIN_DATO if v == dfn.SIN_DATO else capitulos.get(v, "sin capítulo") for v in cie.to_list()
    ])
    dia = df["FECHA_INGRESO"].dt.weekday()
    pred = pl.DataFrame({
        "ID_EPISODIO": df["ID_EPISODIO"], "ANIO_EGRESO": df["ANIO_EGRESO"], "COD_HOSPITAL": df["COD_HOSPITAL"],
        "EDAD_ANIOS": df["EDAD_ANIOS"],
        "SEXO": df["SEXO"],
        "ES_FIN_SEMANA": (dia >= 6).cast(pl.Int8),
        "TIPO_INGRESO": dfn.limpiar_categorica(df["TIPO_INGRESO"]),
        "TIPO_PROCEDENCIA": dfn.limpiar_categorica(df["TIPO_PROCEDENCIA"]),
        "PROCEDENCIA_AGR": dfn.agrupar_procedencia(df["TIPO_PROCEDENCIA"]),
        "NIVEL_CUIDADO_INGRESO": dfn.nivel_cuidado_ingreso(df["SERVICIOINGRESO"]),
        "CIE10_3C": cie, "GRUPO_CLINICO": capitulo,
        **{i: flags[i].cast(pl.Int8) for i in ELIX},
    })
    base = pl.DataFrame({
        "ID_EPISODIO": df["ID_EPISODIO"], "N_COMORB_ELIX": flags["N_COMORB_ELIX"].cast(pl.Int32),
        "PUNTAJE_VAN_WALRAVEN": flags["SCORE_VANWALRAVEN"].cast(pl.Int32), "N_DX_SECUNDARIOS": df["N_DX_SECUNDARIOS"],
    })
    return pred, base


def calcular_vocabularios(pred: pl.DataFrame, min_filas: int) -> dict[str, list[str]]:
    """Niveles que se conservan de cada categórica. ``pred`` debe contener SOLO filas de los años de entrenamiento.

    Un nivel con menos de ``min_filas`` egresos pasa a RARO. «(sin dato)» y «(sin mapeo)» nunca se agrupan.
    """
    voc: dict[str, list[str]] = {}
    for v in CATEGORICAS:
        cnt = pred.group_by(v).len().sort(["len", v], descending=[True, False])
        voc[v] = [r[0] for r in cnt.iter_rows() if r[0] in (dfn.SIN_DATO, dfn.SIN_MAPEO) or r[1] >= min_filas]
    return voc


def aplicar_vocabularios(pred: pl.DataFrame, voc: dict[str, list[str]]) -> pl.DataFrame:
    return pred.with_columns([
        pl.when(pl.col(v).is_in(voc[v])).then(pl.col(v)).otherwise(pl.lit("RARO")).alias(v) for v in CATEGORICAS
    ])


def percentil_estancia(dias: np.ndarray, q: float) -> int:
    """Menor valor v tal que al menos q % de los datos es ≤ v (percentil por rango más cercano)."""
    d = np.sort(np.asarray(dias))
    k = int(np.ceil(q / 100.0 * len(d))) - 1
    return int(d[max(k, 0)])


def construir_oro(anios: list[int], dir_plata: Path | str, dir_cohortes: Path | str, dir_oro: Path | str, ruta_cie10: Path | str,
                  u: Umbrales | None = None, e: elixhauser.Elixhauser | None = None) -> dict:
    u = u or cargar_umbrales()
    e = e or elixhauser.cargar()
    dir_plata, dir_cohortes, dir_oro = Path(dir_plata), Path(dir_cohortes), Path(dir_oro)
    dir_oro.mkdir(parents=True, exist_ok=True)
    capitulos = dfn.cargar_capitulos_cie10(str(ruta_cie10))
    entren = list(u.particion["anios_entrenamiento"])
    # ---- pasada 1: predictores crudos de la cohorte C de cada año
    base_total = []
    for a in anios:
        coh = pl.read_parquet(dir_cohortes / f"cohorte_{a}.parquet")
        c_ids = coh.filter(pl.col("C"))
        plata = pl.read_parquet(dir_plata / f"plata_{a}.parquet", columns=COLS_PLATA_ORO)
        if plata["ID_EPISODIO"].to_list() != coh["ID_EPISODIO"].to_list():
            raise ValueError(f"{a}: Plata y cohorte no tienen las mismas filas")
        sel = plata.filter(coh["C"])
        del plata
        pred, base = predictores_crudos(sel, capitulos, e)
        flags_c = c_ids.select("D", "E") if "D" in c_ids.columns else None
        if flags_c is not None:
            pred = pred.with_columns(flags_c["D"].alias("EN_COHORTE_D"), flags_c["E"].alias("EN_COHORTE_E"))
        else:
            pred = pred.with_columns(pl.lit(None, dtype=pl.Boolean).alias("EN_COHORTE_D"), pl.lit(None, dtype=pl.Boolean).alias("EN_COHORTE_E"))
        pred = pred.with_columns(pl.lit(anio_split(a, u)).alias("SPLIT"))
        pred.write_parquet(dir_oro / f"_crudo_{a}.parquet", compression="zstd")
        base.write_parquet(dir_oro / f"linea_base_{a}.parquet", compression="zstd")
        base_total.append(a)
        del sel, pred, base
    # ---- vocabularios: solo años de entrenamiento
    crudos = pl.concat([pl.read_parquet(dir_oro / f"_crudo_{a}.parquet", columns=list(CATEGORICAS)) for a in anios if a in entren])
    voc = calcular_vocabularios(crudos, int(u["min_filas_nivel"]))
    tabla_voc = pl.concat([
        crudos.group_by(v).len().rename({v: "nivel", "len": "n_entrenamiento"}).with_columns(pl.lit(v).alias("variable"),
        pl.col("nivel").is_in(voc[v]).alias("conservado")) for v in CATEGORICAS
    ]).select("variable", "nivel", "n_entrenamiento", "conservado").sort(["variable", "n_entrenamiento"], descending=[False, True])
    del crudos
    # ---- pasada 2: aplicar vocabularios y escribir
    orden = [*CLAVES, *PREDICTORES, "EN_COHORTE_D", "EN_COHORTE_E"]
    for a in anios:
        pred = pl.read_parquet(dir_oro / f"_crudo_{a}.parquet")
        pred = aplicar_vocabularios(pred, voc).select(orden)
        pred.write_parquet(dir_oro / f"oro_predictores_{a}.parquet", compression="zstd")
        (dir_oro / f"_crudo_{a}.parquet").unlink()
    return {"anios": anios, "vocabularios": {k: len(v) for k, v in voc.items()}, "tabla_vocabularios": tabla_voc}


def construir_matrices(anios: list[int], dir_plata: Path | str, dir_oro: Path | str, u: Umbrales | None = None) -> dict:
    """Matrices analíticas (predictores) y desenlaces separados. Solo 2019 a 2023; el desenlace de 2024 no se toca."""
    u = u or cargar_umbrales()
    dir_plata, dir_oro = Path(dir_plata), Path(dir_oro)
    libres = [a for a in anios if a in ANIOS_RESULTADO]
    preds = pl.concat([pl.read_parquet(dir_oro / f"oro_predictores_{a}.parquet") for a in libres])
    des = pl.concat([pl.read_parquet(dir_plata / f"desenlace_{a}.parquet", columns=["ID_EPISODIO", "ROL", "ESTANCIA_DIAS"]) for a in libres])
    d_mask = preds["EN_COHORTE_D"].fill_null(False)
    e_mask = preds["EN_COHORTE_E"].fill_null(False)
    cols = [*CLAVES, *PREDICTORES]
    mort = preds.filter(d_mask).select(cols)
    est = preds.filter(e_mask).select(cols)
    y_mort = des.join(mort.select("ID_EPISODIO"), on="ID_EPISODIO", how="inner").select(
        "ID_EPISODIO", (pl.col("ROL") == "EVENTO").cast(pl.Int8).alias("MORTALIDAD_BINARIA"))
    y_est = des.join(est.select("ID_EPISODIO"), on="ID_EPISODIO", how="inner").select("ID_EPISODIO", "ESTANCIA_DIAS")
    # el tope de estadía (percentil 99) se calcula UNA vez con los sobrevivientes de 2019 a 2022
    entren = list(u.particion["anios_entrenamiento"])
    ids_entren = est.filter(pl.col("ANIO_EGRESO").is_in(entren)).select("ID_EPISODIO")
    dias_entren = y_est.join(ids_entren, on="ID_EPISODIO", how="semi")["ESTANCIA_DIAS"].to_numpy()
    tope = percentil_estancia(dias_entren, float(u["percentil_estancia"]))
    y_est = y_est.with_columns(pl.min_horizontal(pl.col("ESTANCIA_DIAS"), pl.lit(tope, dtype=pl.Int32)).alias("ESTANCIA_DIAS_TOPE"))
    mort.write_parquet(dir_oro / "matriz_mortalidad.parquet", compression="zstd")
    est.write_parquet(dir_oro / "matriz_estadia.parquet", compression="zstd")
    y_mort.write_parquet(dir_oro / "desenlace_mortalidad.parquet", compression="zstd")
    y_est.write_parquet(dir_oro / "desenlace_estadia.parquet", compression="zstd")
    info = {"tope_estancia_dias": tope, "n_entrenamiento_estadia": int(len(dias_entren)),
            "n_afectados_tope_entrenamiento": int((dias_entren > tope).sum()), "filas_mortalidad": mort.height, "filas_estadia": est.height}
    if ANIO_SELLADO in anios:
        p24 = pl.read_parquet(dir_oro / f"oro_predictores_{ANIO_SELLADO}.parquet").select(cols)
        p24.write_parquet(dir_oro / "predictores_2024_cohorte_C.parquet", compression="zstd")
        info["filas_2024_cohorte_C"] = p24.height
    (dir_oro / "parametros_oro.json").write_text(json.dumps(info, indent=1), encoding="utf8")
    return info
