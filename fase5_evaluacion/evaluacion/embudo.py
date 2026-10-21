"""Gráfico de embudo con corrección de sobredispersión (Spiegelhalter, 2005), aditiva con winsorización."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def z_scores(obs: np.ndarray, esp: np.ndarray, disp: float = 1.0) -> np.ndarray:
    """Puntaje z de cada hospital con varianza de Poisson escalada por ``disp`` (1 para mortalidad; dispersión global para estadía)."""
    return (obs - esp) / np.sqrt(disp * esp)


def tau2(obs: np.ndarray, esp: np.ndarray, disp: float = 1.0, winsor: float = 0.10) -> float:
    """Varianza adicional entre hospitales (DerSimonian-Laird sobre z winsorizados al ``winsor`` en cada cola)."""
    z = z_scores(obs, esp, disp)
    lo, hi = np.quantile(z, [winsor, 1 - winsor])
    zw = np.clip(z, lo, hi)
    n = len(z)
    w = esp / disp                                           # inverso de la varianza de O/E
    q = float(np.sum(zw ** 2))
    denom = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    if n < 3 or denom <= 0:                                  # con menos de 3 hospitales no se puede estimar la variación entre ellos
        return 0.0
    return max(0.0, (q - (n - 1)) / denom)


def limites(esp: np.ndarray, t2: float, z: float, disp: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Límites inferior y superior del O/E alrededor de 1 para cada volumen esperado."""
    s = np.sqrt(disp / esp + t2)
    return 1 - z * s, 1 + z * s


def clasificar(obs: np.ndarray, esp: np.ndarray, t2: float, z_alerta: float, z_senal: float, disp: float = 1.0) -> pd.DataFrame:
    oe = obs / esp
    l1, u1 = limites(esp, t2, z_alerta, disp)
    l2, u2 = limites(esp, t2, z_senal, disp)
    zona = np.where(oe > u2, "señal alta", np.where(oe > u1, "alerta alta", np.where(oe < l2, "señal baja", np.where(oe < l1, "alerta baja", "dentro de lo esperado"))))
    return pd.DataFrame({"oe": oe, "lim_inf_95": l1, "lim_sup_95": u1, "lim_inf_998": l2, "lim_sup_998": u2, "zona": zona})


def ic_poisson(obs: np.ndarray, esp: np.ndarray, nivel: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Intervalo exacto de Poisson para el O/E (límites sobre las muertes observadas dividido por las esperadas)."""
    a = (1 - nivel) / 2
    obs = np.asarray(obs, float)
    inf = np.where(obs > 0, chi2.ppf(a, 2 * obs) / 2, 0.0)
    sup = chi2.ppf(1 - a, 2 * (obs + 1)) / 2
    return inf / esp, sup / esp


def simular_nulo(esp: np.ndarray, repeticiones: int = 400, semilla: int = 1, tau: float = 0.0, disp: float = 1.0) -> dict:
    """Validación del método: hospitales sin diferencia real (o con variación real ``tau``); fracción que queda fuera de los límites."""
    rng = np.random.default_rng(semilla)
    f95 = f998 = 0.0
    for _ in range(repeticiones):
        real = np.exp(rng.normal(0, tau, len(esp))) if tau > 0 else np.ones(len(esp))
        obs = rng.poisson(esp * real)
        t2 = tau2(obs, esp, disp)
        c = clasificar(obs, esp, t2, 1.96, 3.09, disp)
        f95 += float((c["zona"] != "dentro de lo esperado").mean())
        f998 += float(c["zona"].isin(["señal alta", "señal baja"]).mean())
    return {"fuera_95": f95 / repeticiones, "fuera_998": f998 / repeticiones}
