"""Criterios globales de aceptación (Fase 1, sección 12) con intervalos por bootstrap de hospitales completos."""
from __future__ import annotations

import numpy as np
import pandas as pd

from modelado.metricas import metricas_estadia, metricas_mortalidad


def remuestras(hospital: np.ndarray, B: int, semilla: int):
    """Genera B vectores de índices de fila, cada uno con hospitales completos remuestreados con reposición."""
    rng = np.random.default_rng(semilla)
    cod, inicio = np.unique(hospital, return_inverse=True)
    orden = np.argsort(inicio, kind="stable")
    cuentas = np.bincount(inicio)
    desplaz = np.concatenate([[0], np.cumsum(cuentas)])
    for _ in range(B):
        eleg = rng.integers(0, len(cod), len(cod))
        yield np.concatenate([orden[desplaz[h]:desplaz[h + 1]] for h in eleg])


def _veredicto(valor: float, lo: float, hi: float, minimo: float | None = None, maximo: float | None = None) -> str:
    def ok(x):
        return (minimo is None or x > minimo) and (maximo is None or x < maximo)
    def oki(x):                                      # umbral inclusivo en el rango [min, max]
        return (minimo is None or x >= minimo) and (maximo is None or x <= maximo)
    if minimo is not None and maximo is not None:
        punto, entero = oki(valor), oki(lo) and oki(hi)
    else:
        punto, entero = ok(valor), ok(lo) and ok(hi)
    return "cumple" if punto and entero else ("cumple con incertidumbre" if punto else "no cumple")


def _ic(x: np.ndarray, nivel: float) -> tuple[float, float]:
    a = (1 - nivel) / 2
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


def evaluar_mortalidad(y, p_modelo, p_logistica, hospital, prot, B: int | None = None) -> pd.DataFrame:
    B = B or prot.incertidumbre["bootstrap"]
    prev_ref = float(np.mean(y))
    def todas(idx):
        a, b = metricas_mortalidad(y[idx], p_modelo[idx], prev_ref), metricas_mortalidad(y[idx], p_logistica[idx], prev_ref)
        return a, b
    base, base_l = todas(np.arange(len(y)))
    bs = [todas(i) for i in remuestras(hospital, B, prot.semilla)]
    arr = lambda k, j: np.array([r[j][k] for r in bs])  # noqa: E731
    dif_pr = arr("pr_auc", 0) - arr("pr_auc", 1)
    c = prot.criterios
    filas = []
    lo, hi = _ic(dif_pr, prot.incertidumbre["nivel"])
    d0 = base["pr_auc"] - base_l["pr_auc"]
    filas.append(("Discriminación de mortalidad: PR AUC del modelo menos PR AUC de la logística", d0, lo, hi, "> 0", _veredicto(d0, lo, hi, minimo=0.0)))
    filas.append(("PR AUC del modelo", base["pr_auc"], *_ic(arr("pr_auc", 0), 0.95), "informativa", "informativa"))
    filas.append(("Ganancia relativa de PR AUC sobre la prevalencia", base["pr_auc"] / base["prevalencia"], *_ic(arr("pr_auc", 0) / np.array([r[0]["prevalencia"] for r in bs]), 0.95), "informativa", "informativa"))
    for nombre, k, u in (("Brier Skill Score", "bss", (c["bss_minimo"], None)), ):
        lo, hi = _ic(arr(k, 0), 0.95)
        filas.append((nombre, base[k], lo, hi, f"> {u[0]}", _veredicto(base[k], lo, hi, minimo=u[0])))
    lo, hi = _ic(arr("pendiente", 0), 0.95)
    filas.append(("Pendiente de calibración", base["pendiente"], lo, hi, f"{c['pendiente'][0]} a {c['pendiente'][1]}", _veredicto(base["pendiente"], lo, hi, *c["pendiente"])))
    lo, hi = _ic(arr("oe_global", 0), 0.95)
    filas.append(("O/E global de mortalidad", base["oe_global"], lo, hi, f"{c['oe_global'][0]} a {c['oe_global'][1]}", _veredicto(base["oe_global"], lo, hi, *c["oe_global"])))
    for k in ("auc", "brier", "log_loss", "citl"):
        filas.append((f"{k} (informativa)", base[k], *_ic(arr(k, 0), 0.95), "informativa", "informativa"))
    return pd.DataFrame(filas, columns=["criterio", "valor", "ic_inf", "ic_sup", "umbral", "veredicto"])


def evaluar_estadia(y, mu_modelo, mu_bases: dict, hospital, prot, B: int | None = None) -> pd.DataFrame:
    B = B or prot.incertidumbre["bootstrap"]
    def todas(idx):
        return metricas_estadia(y[idx], mu_modelo[idx]), {k: metricas_estadia(y[idx], v[idx]) for k, v in mu_bases.items()}
    base, bases = todas(np.arange(len(y)))
    bs = [todas(i) for i in remuestras(hospital, B, prot.semilla)]
    c = prot.criterios
    filas = []
    for k in mu_bases:
        d = np.array([r[0]["rmse"] - r[1][k]["rmse"] for r in bs])
        lo, hi = _ic(d, 0.95)
        d0 = base["rmse"] - bases[k]["rmse"]
        filas.append((f"Error de estadía: RMSE del modelo menos RMSE de la regresión {k}", d0, lo, hi, "< 0", _veredicto(-d0, -hi, -lo, minimo=0.0)))
    arr = lambda k: np.array([r[0][k] for r in bs])  # noqa: E731
    lo, hi = _ic(arr("oe_global"), 0.95)
    filas.append(("O/E global de estadía", base["oe_global"], lo, hi, f"{c['oe_global'][0]} a {c['oe_global'][1]}", _veredicto(base["oe_global"], lo, hi, *c["oe_global"])))
    for k in ("rmse", "mae", "devianza_poisson"):
        filas.append((f"{k} del modelo (informativa)", base[k], *_ic(arr(k), 0.95), "informativa", "informativa"))
    return pd.DataFrame(filas, columns=["criterio", "valor", "ic_inf", "ic_sup", "umbral", "veredicto"])
