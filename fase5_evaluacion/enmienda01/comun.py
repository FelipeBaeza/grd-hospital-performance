"""Piezas comunes de la enmienda 01: guardia de registro, tablas por hospital, comparación con la base, regla de lectura,
embudo con truncamiento, simulación P5b e intervalos bootstrap por hospital."""
from __future__ import annotations

import json
import tomllib
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evaluacion import embudo, hospitales
from evaluacion.config import RAIZ, cargar_protocolo, sha256_archivo

FUERA = ["alerta alta", "alerta baja", "señal alta", "señal baja"]
SEMILLA_MODELO = 20261018            # receta congelada de la Fase 4
SEMILLA_BOOTSTRAP = 20261019         # protocolo v1.0 y enmienda 01
ARCHIVOS_CODIGO = ["run_enmienda01.py", "enmienda01/__init__.py", "enmienda01/comun.py", "enmienda01/modelo.py",
                   "enmienda01/indicador.py", "enmienda01/publicar.py"]


class EnmiendaNoAutorizada(RuntimeError):
    pass


# ------------------------------------------------------------------ guardia
def verificar_autorizacion(raiz: Path = RAIZ, archivos: list[str] = ARCHIVOS_CODIGO) -> dict:
    """Exige la enmienda registrada y sin cambios, y el código igual al registrado antes de la primera corrida."""
    reg = raiz / "documentos" / "registro_enmienda_01.json"
    cor = raiz / "documentos" / "registro_corrida_enmienda_01.json"
    if not reg.exists():
        raise EnmiendaNoAutorizada("la enmienda 01 no está registrada (falta documentos/registro_enmienda_01.json)")
    if not cor.exists():
        raise EnmiendaNoAutorizada("falta registrar el código antes de correr: python registrar_enmienda.py antes-de-correr ...")
    firmado, corrida = json.loads(reg.read_text(encoding="utf8")), json.loads(cor.read_text(encoding="utf8"))
    for ruta, h in firmado["huellas"].items():
        if sha256_archivo(raiz / ruta) != h:
            raise EnmiendaNoAutorizada(f"{ruta} cambió después de firmarse")
    registrados = corrida["scripts"]
    faltan = [a for a in archivos if a not in registrados]
    if faltan:
        raise EnmiendaNoAutorizada(f"código sin registrar: {faltan}")
    distintos = [a for a in archivos if sha256_archivo(raiz / a) != registrados[a]]
    if distintos:
        raise EnmiendaNoAutorizada(f"el código cambió después de registrarse: {distintos}")
    return {"registro_enmienda": sha256_archivo(reg), "registro_corrida": sha256_archivo(cor), "momento_corrida_registrada": corrida["momento_utc"]}


def cargar_enmienda(raiz: Path = RAIZ) -> dict:
    with open(raiz / "config" / "protocolo_fase5_v2.toml", "rb") as f:
        return tomllib.load(f)


def analisis(enm: dict, id_: str) -> dict:
    return next(a for a in enm["analisis"] if a["id"] == id_)


def expandir_columnas(lista: list[str]) -> list[str]:
    """«ELIX_01..ELIX_31» se expande a las 31 columnas."""
    out = []
    for c in lista:
        if ".." in c:
            a, b = c.split("..")
            pre = a.rstrip("0123456789")
            out += [f"{pre}{i:02d}" for i in range(int(a[len(pre):]), int(b[len(pre):]) + 1)]
        else:
            out.append(c)
    return out


def protocolo(winsor: float | None = None):
    prot = cargar_protocolo(RAIZ / "config" / "protocolo_fase5.toml")
    if winsor is None:
        return prot
    return types.SimpleNamespace(hospital={**prot.hospital, "winsor": winsor})


# ------------------------------------------------------------------ tablas por hospital (uso interno: no se publican)
def tabla_interna(h, y, p, tarea: str, prot, tau2_forzado: float | None = None) -> tuple[pd.DataFrame, dict]:
    """Zona del embudo y O/E sin suprimir por hospital elegible. ``tau2_forzado`` reemplaza el τ² del protocolo (truncamiento)."""
    tab, res = hospitales.tabla(h, y, p, tarea, prot)
    g = pd.DataFrame({"cod_hospital": np.asarray(h), "y": np.asarray(y, float), "p": np.asarray(p, float)}).groupby("cod_hospital").agg(
        obs=("y", "sum"), esp=("p", "sum"), n=("y", "size")).reset_index()
    tab = tab[["cod_hospital", "zona"]].merge(g, on="cod_hospital", how="left")
    if tau2_forzado is not None:
        c = embudo.clasificar(tab["obs"].to_numpy(), tab["esp"].to_numpy(), tau2_forzado, prot.hospital["z_alerta"], prot.hospital["z_senal"], res["dispersion"])
        tab["zona"] = c["zona"].to_numpy()
        res = {**res, "tau2": float(tau2_forzado), "pct_fuera_95": float(tab["zona"].isin(FUERA).mean()),
               "pct_senal_998": float(tab["zona"].isin(["señal alta", "señal baja"]).mean())}
    tab["oe_interno"] = tab["obs"] / tab["esp"]
    tab["suprimido"] = (tarea == "mortalidad") & (tab["obs"] > 0) & (tab["obs"] < prot.hospital["celda_minima"])
    return tab, res


def comparar(base: pd.DataFrame, otra: pd.DataFrame) -> dict:
    m = base.merge(otra, on="cod_hospital", suffixes=("_b", "_v"))
    rho = float(spearmanr(m["oe_interno_b"], m["oe_interno_v"]).statistic) if len(m) > 2 else float("nan")
    cambia = m[m["zona_b"] != m["zona_v"]]
    return {"n_comunes": int(len(m)), "pct_misma_zona": float((m["zona_b"] == m["zona_v"]).mean()) if len(m) else float("nan"), "rho_oe": rho,
            "n_cambian": int(len(cambia)),
            "fuera_a_dentro": int((cambia["zona_b"].isin(FUERA) & (cambia["zona_v"] == "dentro de lo esperado")).sum()),
            "dentro_a_fuera": int(((cambia["zona_b"] == "dentro de lo esperado") & cambia["zona_v"].isin(FUERA)).sum()),
            "cambios": "; ".join(f"{c.cod_hospital}: {c.zona_b} → {c.zona_v}" for c in cambia.sort_values("cod_hospital").itertuples())}


def cumple_regla(comp: dict, enm: dict) -> bool:
    r = enm["regla_lectura"]
    return bool(comp["pct_misma_zona"] >= r["min_misma_zona"] and comp["rho_oe"] >= r["min_rho"])


def resumen_res(res: dict) -> dict:
    return {"n_elegibles": int(res["n_elegibles"]), "tau2": float(res["tau2"]), "pct_fuera_95": float(res["pct_fuera_95"]),
            "pct_fuera_998": float(res["pct_senal_998"])}


# ------------------------------------------------------------------ embudo: truncamiento al estilo SHMI y simulación P5b
def tau2_truncado(obs: np.ndarray, esp: np.ndarray, disp: float = 1.0, frac: float = 0.10) -> float:
    """τ² de DerSimonian-Laird calculado sin los hospitales del ``frac`` de cada cola de z (se excluyen, no se recortan)."""
    obs, esp = np.asarray(obs, float), np.asarray(esp, float)
    z = embudo.z_scores(obs, esp, disp)
    lo, hi = np.quantile(z, [frac, 1 - frac])
    k = (z >= lo) & (z <= hi)
    zk, w = z[k], esp[k] / disp
    n = int(k.sum())
    denom = float(np.sum(w) - np.sum(w ** 2) / np.sum(w)) if n else 0.0
    if n < 3 or denom <= 0:
        return 0.0
    return max(0.0, (float(np.sum(zk ** 2)) - (n - 1)) / denom)


def simular_p5b(esp: np.ndarray, disp: float, tau: float, winsor: float, rep: int = 400, semilla: int = 7) -> float:
    """Fracción fuera del 95 % cuando todos los hospitales son «normales» con variación real ``tau`` (log-normal)."""
    rng = np.random.default_rng(semilla)
    f = 0.0
    for _ in range(rep):
        mu = esp * np.exp(rng.normal(0, tau, len(esp)))
        obs = rng.poisson(mu) if disp == 1 else rng.gamma(mu / disp, disp)
        t2 = embudo.tau2(obs, esp, disp, winsor)
        f += float((embudo.clasificar(obs, esp, t2, 1.96, 3.09, disp)["zona"] != "dentro de lo esperado").mean())
    return f / rep


# ------------------------------------------------------------------ intervalos por bootstrap de hospitales completos
def oe_con_ic(h, y, p, B: int = 1000, semilla: int = SEMILLA_BOOTSTRAP, nivel: float = 0.95) -> tuple[float, float, float]:
    """O/E global y su intervalo, remuestreando hospitales completos (con las sumas por hospital: equivalente y rápido)."""
    g = pd.DataFrame({"h": np.asarray(h), "y": np.asarray(y, float), "p": np.asarray(p, float)}).groupby("h")[["y", "p"]].sum()
    o, e = g["y"].to_numpy(), g["p"].to_numpy()
    rng = np.random.default_rng(semilla)
    idx = rng.integers(0, len(o), (B, len(o)))
    oe_b = o[idx].sum(1) / e[idx].sum(1)
    a = (1 - nivel) / 2
    return float(o.sum() / e.sum()), float(np.quantile(oe_b, a)), float(np.quantile(oe_b, 1 - a))
