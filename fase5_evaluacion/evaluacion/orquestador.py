"""Etapas de la Fase 5. Preparación (huellas, planes B, pre-registro) y ensayo con 2023 se ejecutan ahora; la apertura de 2024 exige autorización escrita."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

from . import analisis, apertura, cohortes, embudo, figuras, huellas, planes_b
from .config import DIR_MODELOS_F4, DIR_PLANES_B, DIR_SALIDAS, FASE4, RAIZ, REGISTRO_FASE3, cargar_protocolo, sha256_archivo

RUTA_PREAPERTURA = RAIZ / "documentos" / "registro_preapertura.json"


def jw(ruta: Path, obj) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding="utf8")


def etapa_huellas(trabajo: Path) -> dict:
    r = huellas.verificar_manifiesto(DIR_MODELOS_F4)
    tabla = pd.DataFrame([{"archivo": k, "coincide": ok} for k, (ok, _, _) in r.items()])
    tabla.to_csv(DIR_SALIDAS / "tablas" / "huellas_fase4.csv", index=False)
    if not tabla["coincide"].all():
        raise RuntimeError("hay archivos de la Fase 4 distintos del manifiesto")
    return r


def etapa_planes_b(trabajo_f4: Path) -> dict:
    m = planes_b.congelar(trabajo_f4, DIR_PLANES_B)
    huellas.exigir_intacto(DIR_PLANES_B)
    return m


def etapa_preregistro(prot) -> Path:
    from .config import FASE3
    hu = {"manifiesto_fase4": sha256_archivo(DIR_MODELOS_F4 / "manifiesto.json"), "manifiesto_planes_b": sha256_archivo(DIR_PLANES_B / "manifiesto.json"),
          "registro_accesos_fase3": sha256_archivo(REGISTRO_FASE3) if REGISTRO_FASE3.exists() else "no existe"}
    return apertura.registrar_preapertura(RUTA_PREAPERTURA, prot, hu)


def _entrantes(capas: Path, anio: int) -> pd.DataFrame:
    return pl.read_parquet(capas / "plata" / f"plata_{anio}.parquet", columns=["ID_EPISODIO", "ES_TRASLADO_ENTRANTE"]).to_pandas()


def _cohortes_anio(capas: Path, anio: int, tope: int, desenlace: pd.DataFrame | None = None):
    """``desenlace``: para 2024 se entrega el leído en la apertura; para los demás años se lee de Plata."""
    pred = pl.read_parquet(capas / "oro" / f"oro_predictores_{anio}.parquet").drop(["EN_COHORTE_D", "EN_COHORTE_E", "SPLIT"]).to_pandas()
    if desenlace is None:
        desenlace = pl.read_parquet(capas / "plata" / f"desenlace_{anio}.parquet", columns=["ID_EPISODIO", "ROL", "ESTANCIA_DIAS", "ES_TRASLADO_SALIENTE"]).to_pandas()
    mort, est = cohortes.construir(pred, desenlace, tope)
    return mort, est, desenlace[["ID_EPISODIO", "ES_TRASLADO_SALIENTE"]]


def etapa_ensayo(capas: Path, prot, B: int) -> dict:
    """Ensayo general con 2023 (desenlace conocido). 2023 sirvió para calibrar: estos resultados NO se reportan como evaluación."""
    from modelado.config import cargar_reglas
    from modelado.exhaustividad import cargar_anio, vocabulario_cie
    anio = prot.ensayo["anio"]
    tope = json.loads((capas / "oro" / "parametros_oro.json").read_text())["tope_estancia_dias"]
    mort, est, tras = _cohortes_anio(capas, anio, tope)
    mm = pl.read_parquet(capas / "oro" / "matriz_mortalidad.parquet", columns=["ID_EPISODIO", "ANIO_EGRESO"]).filter(pl.col("ANIO_EGRESO") == anio)
    me = pl.read_parquet(capas / "oro" / "matriz_estadia.parquet", columns=["ID_EPISODIO", "ANIO_EGRESO"]).filter(pl.col("ANIO_EGRESO") == anio)
    cuadra = {"mortalidad": [len(mort), mm.height], "estadia": [len(est), me.height],
              "mismos_ids_mortalidad": bool(set(mort["ID_EPISODIO"]) == set(mm["ID_EPISODIO"])), "mismos_ids_estadia": bool(set(est["ID_EPISODIO"]) == set(me["ID_EPISODIO"]))}
    reglas4 = cargar_reglas(FASE4 / "config" / "reglas_fase4.toml")
    voc = vocabulario_cie(RAIZ.parent / "fase3_preparacion" / "salidas" / "tablas" / "oro_vocabularios.csv")
    df_exh = cargar_anio(capas, anio, reglas4, voc)
    destino = DIR_SALIDAS / "ensayo_2023"
    r = analisis.evaluar(mort, est, df_exh, tras, prot, destino, B=B, entrantes=_entrantes(capas, anio))
    for nombre, tarea, ejes in (("mortalidad", "mortalidad", "muertes esperadas del hospital (escala log)"), ("estadia", "estadia", "días esperados del hospital (escala log)")):
        tab = pd.read_csv(destino / "tablas" / f"oe_hospital_{nombre}.csv")
        res = r["resumen"][f"hospitales_{nombre}"]
        figuras.grafico_embudo(tab, res["tau2"], res["dispersion"], f"ENSAYO CON 2023: O/E de {nombre} por hospital", ejes, destino / "figuras" / f"embudo_{nombre}.png",
                               aviso="Ensayo de código: 2023 se usó para calibrar; no es una evaluación.")
    jw(destino / "resumen_ensayo.json", {"cuadra_con_fase3": cuadra, **r["resumen"]})
    return {"cuadra": cuadra, **r["resumen"]}


def etapa_validar_embudo(prot) -> dict:
    """Con hospitales sin diferencia real, ~5 % queda fuera del 95 % y ~0,2 % fuera del 99,8 % (con un margen por el ajuste de la sobredispersión)."""
    rng = np.random.default_rng(3)
    esp = np.exp(rng.uniform(np.log(15), np.log(900), 65))
    r = embudo.simular_nulo(esp, 400, 1)
    jw(DIR_SALIDAS / "tablas" / "validacion_embudo.json", r)
    return r


def etapa_apertura(capas: Path, trabajo5: Path, prot) -> Path:
    """NO se ejecuta sin la autorización escrita del profesor guía."""
    destino = trabajo5 / "evaluacion_2024" / "desenlace_2024.parquet"
    apertura.abrir_2024(capas / "sellado", prot, RUTA_PREAPERTURA, RAIZ / prot.apertura["archivo_autorizacion"], REGISTRO_FASE3, lambda: huellas.exigir_intacto(DIR_MODELOS_F4), destino)
    return destino


def etapa_evaluacion_2024(capas: Path, trabajo5: Path, prot, B: int) -> dict:
    """Evaluación real: usa el desenlace guardado por la apertura (no vuelve a leer el archivo sellado) y los modelos congelados."""
    from modelado.config import cargar_reglas
    from modelado.exhaustividad import cargar_anio, vocabulario_cie
    huellas.exigir_intacto(DIR_MODELOS_F4)
    huellas.exigir_intacto(DIR_PLANES_B)
    ruta = trabajo5 / "evaluacion_2024" / "desenlace_2024.parquet"
    des = pl.read_parquet(ruta, columns=["ID_EPISODIO", "ROL", "ESTANCIA_DIAS", "ES_TRASLADO_SALIENTE"]).to_pandas()
    tope = json.loads((capas / "oro" / "parametros_oro.json").read_text())["tope_estancia_dias"]
    mort, est, tras = _cohortes_anio(capas, 2024, tope, des)
    reglas4 = cargar_reglas(FASE4 / "config" / "reglas_fase4.toml")
    voc = vocabulario_cie(RAIZ.parent / "fase3_preparacion" / "salidas" / "tablas" / "oro_vocabularios.csv")
    df_exh = cargar_anio(capas, 2024, reglas4, voc)
    destino = DIR_SALIDAS / "evaluacion_2024"
    r = analisis.evaluar(mort, est, df_exh, tras, prot, destino, B=B, entrantes=_entrantes(capas, 2024))
    for nombre, ejes in (("mortalidad", "muertes esperadas del hospital (escala log)"), ("estadia", "días esperados del hospital (escala log)")):
        tab = pd.read_csv(destino / "tablas" / f"oe_hospital_{nombre}.csv")
        res = r["resumen"][f"hospitales_{nombre}"]
        figuras.grafico_embudo(tab, res["tau2"], res["dispersion"], f"2024: O/E de {nombre} por hospital", ejes, destino / "figuras" / f"embudo_{nombre}.png")
    n_c = int(pl.read_parquet(capas / "oro" / "oro_predictores_2024.parquet", columns=["ID_EPISODIO"]).height)
    jw(destino / "resumen_evaluacion_2024.json", {"n_cohorte_c": n_c, "bootstrap": B, **r["resumen"]})
    return r["resumen"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--capas", required=True)
    p.add_argument("--trabajo", required=True)
    p.add_argument("--trabajo-fase4", required=True)
    p.add_argument("--etapas", nargs="+", default=["huellas", "planes_b", "preregistro", "validar_embudo", "ensayo", "verificacion"])
    p.add_argument("--bootstrap", type=int, default=None)
    a = p.parse_args(argv)
    capas, trabajo = Path(a.capas), Path(a.trabajo)
    prot = cargar_protocolo()
    (DIR_SALIDAS / "tablas").mkdir(parents=True, exist_ok=True)
    tiempos = []
    for e in a.etapas:
        t0 = time.time()
        print(f"[{e}] inicio", flush=True)
        if e == "huellas":
            etapa_huellas(trabajo)
        elif e == "planes_b":
            etapa_planes_b(Path(a.trabajo_fase4))
        elif e == "preregistro":
            etapa_preregistro(prot)
        elif e == "validar_embudo":
            print(etapa_validar_embudo(prot))
        elif e == "ensayo":
            r = etapa_ensayo(capas, prot, a.bootstrap or 100)
            print(r["cuadra"])
        elif e == "apertura":
            etapa_apertura(capas, trabajo, prot)
        elif e == "evaluacion_2024":
            print(etapa_evaluacion_2024(capas, trabajo, prot, a.bootstrap or prot.incertidumbre["bootstrap"]))
        elif e == "cierre":
            from .verificacion import verificar_cierre
            print(verificar_cierre(prot, trabajo).to_string())
        elif e == "verificacion":
            from .verificacion import verificar_preparacion
            print(verificar_preparacion(prot, trabajo).to_string())
        tiempos.append({"etapa": e, "segundos": round(time.time() - t0, 1)})
        print(f"[{e}] listo en {tiempos[-1]['segundos']} s", flush=True)
    pd.DataFrame(tiempos).to_csv(DIR_SALIDAS / "tablas" / ("tiempos_etapas_fase5_evaluacion.csv" if "evaluacion_2024" in a.etapas else "tiempos_etapas_fase5_preparacion.csv"), index=False)
    return 0
