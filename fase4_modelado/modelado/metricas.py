"""Métricas de mortalidad (clasificación) y de estadía (conteo/regresión)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, mean_poisson_deviance, roc_auc_score

EPS = 1e-9


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def pendiente_calibracion(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Pendiente e intercepto de la regresión logística de y sobre logit(p) (Newton–Raphson, 2 parámetros)."""
    z = _logit(p)
    if np.ptp(z) < 1e-9:                                           # predicción constante: la pendiente no está definida
        return float("nan"), float("nan")
    a, b = 0.0, 1.0
    for _ in range(50):
        mu = _sigmoid(a + b * z)
        w = mu * (1 - mu) + 1e-12
        g = np.array([np.sum(y - mu), np.sum((y - mu) * z)])
        h = np.array([[np.sum(w), np.sum(w * z)], [np.sum(w * z), np.sum(w * z * z)]])
        paso = np.linalg.solve(h, g)
        a, b = a + paso[0], b + paso[1]
        if np.max(np.abs(paso)) < 1e-9:
            break
    return float(b), float(a)


def intercepto_en_grande(y: np.ndarray, p: np.ndarray) -> float:
    """Calibration-in-the-large: desplazamiento del logit que iguala lo observado con lo esperado (pendiente fija en 1)."""
    z = _logit(p)
    a = 0.0
    for _ in range(50):
        mu = _sigmoid(a + z)
        paso = np.sum(y - mu) / (np.sum(mu * (1 - mu)) + 1e-12)
        a += paso
        if abs(paso) < 1e-10:
            break
    return float(a)


def metricas_mortalidad(y: np.ndarray, p: np.ndarray, prevalencia_ref: float | None = None) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    brier = float(np.mean((p - y) ** 2))
    ref = float(y.mean()) if prevalencia_ref is None else prevalencia_ref
    brier_ref = float(np.mean((ref - y) ** 2))
    pend, inter = pendiente_calibracion(y, p)
    return {"n": int(len(y)), "prevalencia": float(y.mean()), "pr_auc": float(average_precision_score(y, p)), "auc": float(roc_auc_score(y, p)),
            "brier": brier, "bss": 1 - brier / brier_ref, "log_loss": float(log_loss(y, p)), "pendiente": pend, "intercepto": inter,
            "citl": intercepto_en_grande(y, p), "oe_global": float(y.sum() / p.sum())}


def metricas_estadia(y: np.ndarray, mu: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-6, None)
    return {"n": int(len(y)), "media_obs": float(y.mean()), "media_pred": float(mu.mean()), "devianza_poisson": float(mean_poisson_deviance(y, mu)),
            "rmse": float(np.sqrt(np.mean((y - mu) ** 2))), "mae": float(np.mean(np.abs(y - mu))), "oe_global": float(y.sum() / mu.sum())}


def tabla_calibracion(y: np.ndarray, pred: np.ndarray, grupos: int = 10) -> pd.DataFrame:
    """Observado y predicho promedio por decil de la predicción."""
    d = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(pred, dtype=float)})
    d["decil"] = pd.qcut(d["p"].rank(method="first"), grupos, labels=False) + 1
    t = d.groupby("decil").agg(n=("y", "size"), observado=("y", "mean"), predicho=("p", "mean")).reset_index()
    t["razon_oe"] = t["observado"] / t["predicho"]
    return t
