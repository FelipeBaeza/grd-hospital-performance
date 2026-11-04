"""Cohortes D y E con las reglas de la Fase 3 (prep/cohortes.py): D = C con desenlace conocido; E = D, sobrevivientes con estadía mayor que cero, con tope de 53 días."""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLES_D = ("EVENTO", "NO_EVENTO")


def construir(pred_c: pd.DataFrame, desenlace: pd.DataFrame, tope: int, opciones=None, marcas: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``pred_c``: predictores de la cohorte C (ID_EPISODIO, COD_HOSPITAL, 40 predictores). ``desenlace``: ID_EPISODIO, ROL, ESTANCIA_DIAS.
    Devuelve (mortalidad, estadía), cada una con la columna objetivo ``y``.

    ``opciones`` (``prep.cohortes.Opciones``, enmienda 03): aplica las reglas de la cohorte de la Fase 3 v7. Necesita ETIQUETA_TIPOALTA en
    ``desenlace`` y, si la regla lo pide, ``marcas`` con ID_EPISODIO, DUP_Q15 y ENTRANTE_ENLAZADO. Sin opciones, el comportamiento es el de la
    versión 1.0 del protocolo (D = C con desenlace conocido)."""
    cols = ["ID_EPISODIO", "ROL", "ESTANCIA_DIAS"] + (["ETIQUETA_TIPOALTA"] if opciones is not None and opciones.necesita_etiqueta else [])
    d = pred_c.merge(desenlace[cols], on="ID_EPISODIO", how="inner")
    if len(d) != len(pred_c):
        raise ValueError("el desenlace no cubre todos los egresos de la cohorte C")
    rol = d["ROL"]
    fuera_d = pd.Series(False, index=d.index)
    fuera_e = pd.Series(False, index=d.index)
    if opciones is not None and opciones.activas:
        if opciones.tipoalta_vivos:
            vivo = d["ETIQUETA_TIPOALTA"].isin(list(opciones.tipoalta_vivos))
            rol = rol.where(~vivo, "NO_EVENTO")
            fuera_e |= vivo
        if opciones.excluir_tipoalta_e:
            fuera_e |= d["ETIQUETA_TIPOALTA"].isin(list(opciones.excluir_tipoalta_e))
        if opciones.deduplicar_q15 or opciones.excluir_entrantes_enlazados:
            if marcas is None:
                raise ValueError("las reglas de filas repetidas y de entrantes enlazados necesitan las marcas")
            m = d[["ID_EPISODIO"]].merge(marcas, on="ID_EPISODIO", how="left")
            if opciones.deduplicar_q15:
                fuera_d |= m["DUP_Q15"].fillna(False).astype(bool).to_numpy()
            if opciones.excluir_entrantes_enlazados:
                fuera_d |= m["ENTRANTE_ENLAZADO"].fillna(False).astype(bool).to_numpy()
    en_d = rol.isin(ROLES_D) & ~fuera_d
    mort = d[en_d].copy()
    mort["y"] = (mort["ROL"] == "EVENTO").astype(np.int8)
    est = d[en_d & (rol == "NO_EVENTO") & (d["ESTANCIA_DIAS"] > 0) & ~fuera_e].copy()
    est["y"] = np.minimum(est["ESTANCIA_DIAS"].to_numpy(), tope).astype(np.int32)
    quitar = [c for c in ("ROL", "ESTANCIA_DIAS", "ETIQUETA_TIPOALTA") if c in d.columns]
    return mort.drop(columns=quitar).reset_index(drop=True), est.drop(columns=quitar).reset_index(drop=True)


def marcas_revision(capas, anio: int, desenlace, u=None):
    """Filas repetidas (Q15) y entrantes enlazados de un año, con el código de la Fase 3. ``desenlace`` (polars): el de Plata para 2019–2023 o el
    leído en la apertura autorizada para 2024 (ID_EPISODIO, FECHAALTA, ES_TRASLADO_SALIENTE). Para 2024 el enlace usa también los salientes de
    2023 (mismo régimen de identificadores)."""
    import polars as pl
    from prep import cohortes as c3
    from prep.config import cargar_umbrales
    u = u or cargar_umbrales()
    d_p = capas / "plata"
    clave = pl.read_parquet(d_p / f"plata_{anio}.parquet", columns=["ID_EPISODIO", *[k for k in c3.CLAVE_Q15 if k != "FECHAALTA"]])
    # mismo tipo que Plata (Date): el desenlace llega desde pandas como Datetime
    desenlace = desenlace.with_columns(pl.col("FECHAALTA").cast(pl.Date))
    des = clave.select("ID_EPISODIO").join(desenlace.select("ID_EPISODIO", "FECHAALTA", "ES_TRASLADO_SALIENTE"), on="ID_EPISODIO", how="left",
                                           maintain_order="left")
    dup = c3.marcar_q15(clave.drop("ID_EPISODIO"), des)
    enl = c3.entrantes_enlazados(anio, d_p, u, desenlaces={anio: des})
    return pl.DataFrame({"ID_EPISODIO": clave["ID_EPISODIO"], "DUP_Q15": dup, "ENTRANTE_ENLAZADO": enl}).to_pandas()
