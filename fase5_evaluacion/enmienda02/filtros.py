"""Episodios que cada análisis de la enmienda 02 saca de las cohortes, con las mismas reglas que
fase3_preparacion/prep/cohortes.py con las opciones de la enmienda 02 activas.

Las cohortes D y E de la enmienda 02 son las de la Fase 3 menos estos episodios: con ``mdc14_completo`` salen todos
los del MDC 14 (los que no tenían comorbilidades ya estaban fuera); con ``excluir_tipoalta_e`` salen de E los tipos de
alta con rol_estancia = INCLUIDO_SENSIBILIDAD. Las categorías de las variables y el tope de 53 días quedan como en la
Fase 3, para que la comparación aísle el efecto de la regla. Una prueba comprueba la equivalencia con prep/cohortes.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from evaluacion.config import FASE3

DIAGNOSTICOS = [f"DIAGNOSTICO{i}" for i in range(1, 36)]
ALTA_DOMICILIARIA = "HOSPITALIZACIÓN DOMICILIARIA"


def etiquetas_fuera_de_e() -> tuple[str, ...]:
    from prep import cohortes
    return cohortes.tipoalta_fuera_de_e(pl.read_csv(FASE3 / "config" / "tipoalta_roles.csv"))


def _desenlace(capas: Path, trabajo5: Path, anio: int) -> pl.DataFrame:
    ruta = trabajo5 / "evaluacion_2024" / "desenlace_2024.parquet" if anio == 2024 else capas / "plata" / f"desenlace_{anio}.parquet"
    d = pl.read_parquet(ruta)
    if "ETIQUETA_TIPOALTA" not in d.columns:
        raise ValueError(f"{ruta.name} no trae ETIQUETA_TIPOALTA: no se puede aplicar la regla de tipo de alta")
    return d.select("ID_EPISODIO", "ROL", "ETIQUETA_TIPOALTA")


def episodios(capas: Path, trabajo5: Path, anio: int) -> pd.DataFrame:
    """Una fila por egreso de la cohorte C del año: COD_HOSPITAL, MDC, si es de la cohorte D y las marcas que usan los análisis."""
    dx = pl.col
    p = pl.read_parquet(capas / "plata" / f"plata_{anio}.parquet", columns=["ID_EPISODIO", "COD_HOSPITAL", "MDC", *DIAGNOSTICOS])
    c = pl.read_parquet(capas / "cohortes" / f"cohorte_{anio}.parquet", columns=["ID_EPISODIO", "C"])
    z = pl.any_horizontal([dx(k).cast(pl.String).str.replace_all(r"[.\s]", "").str.starts_with("Z515").fill_null(False) for k in DIAGNOSTICOS])
    d = (p.with_columns(z.alias("Z515")).drop(DIAGNOSTICOS).join(c, on="ID_EPISODIO").filter(dx("C"))
         .join(_desenlace(capas, trabajo5, anio), on="ID_EPISODIO", how="left"))
    fuera_e = list(etiquetas_fuera_de_e())
    d = d.with_columns((dx("MDC") == "14").alias("MDC14"), dx("ETIQUETA_TIPOALTA").is_in(fuera_e).fill_null(False).alias("FUERA_E"),
                       (dx("ETIQUETA_TIPOALTA") == ALTA_DOMICILIARIA).fill_null(False).alias("ALTA_DOMICILIARIA"),
                       dx("ROL").is_in(["EVENTO", "NO_EVENTO"]).fill_null(False).alias("EN_D"))
    return d.select("ID_EPISODIO", "COD_HOSPITAL", "MDC14", "FUERA_E", "ALTA_DOMICILIARIA", "Z515", "EN_D", "ROL").to_pandas()


def quitar(ep: pd.DataFrame, analisis_id: str, tarea: str) -> np.ndarray:
    """IDs que el análisis saca de la cohorte de la tarea."""
    k = np.zeros(len(ep), bool)
    if analisis_id in ("E2.2", "E2.3"):
        k |= ep["MDC14"].to_numpy()
    if analisis_id in ("E2.1", "E2.3") and tarea == "estadia":
        k |= ep["FUERA_E"].to_numpy()
    if analisis_id == "E2.6b" and tarea == "mortalidad":
        k |= ep["Z515"].to_numpy()
    return ep.loc[k, "ID_EPISODIO"].to_numpy()


def muertes_obstetricas(eps: dict[int, pd.DataFrame], anios=(2019, 2020, 2021, 2022, 2023)) -> int:
    """Muertes de la cohorte D del MDC 14 que la regla conservadora saca, sumadas en los años indicados."""
    return int(sum(((e["MDC14"]) & (e["ROL"] == "EVENTO")).sum() for a, e in eps.items() if a in anios))


def tasas_contexto(ep: pd.DataFrame) -> pd.DataFrame:
    """Conteos y tasas por hospital sobre la cohorte C (uso interno: los conteos no se publican)."""
    g = ep.groupby("COD_HOSPITAL").agg(egresos_c=("ID_EPISODIO", "size"), z515=("Z515", "sum"), alta_domiciliaria=("ALTA_DOMICILIARIA", "sum"),
                                       voluntaria_fuga=("FUERA_E", "sum")).reset_index().rename(columns={"COD_HOSPITAL": "cod_hospital"})
    for k in ("z515", "alta_domiciliaria", "voluntaria_fuga"):
        g[f"tasa_{k}"] = g[k] / g["egresos_c"]
    g["cod_hospital"] = g["cod_hospital"].astype(str)
    return g
