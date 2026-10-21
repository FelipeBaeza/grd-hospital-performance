"""Recalibración global con 2023. Nunca por hospital: el recalibrador solo recibe la predicción del modelo."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .metricas import EPS, _logit, _sigmoid, metricas_estadia, metricas_mortalidad


class FueraDeAnio(ValueError):
    """El recalibrador recibió filas que no son de 2023."""


def _exigir_anio(anios, anio: int) -> None:
    if set(np.unique(anios)) != {anio}:
        raise FueraDeAnio(f"el recalibrador solo se ajusta con filas de {anio}")


class Platt:
    """p' = sigmoide(a + b · logit(p)): dos parámetros."""
    nombre = "platt"

    def fit(self, p, y):
        from .metricas import pendiente_calibracion
        self.b, self.a = pendiente_calibracion(np.asarray(y, float), np.asarray(p, float))
        return self

    def transformar(self, p):
        return _sigmoid(self.a + self.b * _logit(np.asarray(p, float)))


class IsotonicaProb:
    nombre = "isotonica"

    def fit(self, p, y):
        self.iso = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip", increasing=True).fit(np.asarray(p, float), np.asarray(y, float))
        return self

    def transformar(self, p):
        return self.iso.predict(np.asarray(p, float))


class FactorUnico:
    """mu' = f · mu, con f = suma observada / suma predicha."""
    nombre = "factor"

    def fit(self, mu, y):
        self.f = float(np.sum(y) / np.sum(mu))
        return self

    def transformar(self, mu):
        return np.asarray(mu, float) * self.f


class IsotonicaMedia:
    nombre = "isotonica"

    def fit(self, mu, y):
        self.iso = IsotonicRegression(y_min=0.05, out_of_bounds="clip", increasing=True).fit(np.asarray(mu, float), np.asarray(y, float))
        return self

    def transformar(self, mu):
        return self.iso.predict(np.asarray(mu, float))


METODOS = {"mortalidad": [Platt, IsotonicaProb], "estadia": [FactorUnico, IsotonicaMedia]}


def bloques_temporales(fechas: pd.Series, k: int) -> np.ndarray:
    """k bloques contiguos de igual tamaño según la fecha de ingreso."""
    orden = pd.Series(np.asarray(fechas)).rank(method="first")
    return (pd.qcut(orden, k, labels=False)).to_numpy()


def validacion_cruzada_bloques(tarea: str, pred: np.ndarray, y: np.ndarray, bloques: np.ndarray, clase) -> np.ndarray:
    """Ajusta con k-1 bloques y predice el restante. Todas las predicciones resultantes son 'fuera de muestra'."""
    oof = np.zeros(len(pred))
    for b in np.unique(bloques):
        fuera = bloques == b
        r = clase().fit(pred[~fuera], y[~fuera])
        oof[fuera] = r.transformar(pred[fuera])
    return oof


def _metricas(tarea: str, y, p):
    return metricas_mortalidad(y, p) if tarea == "mortalidad" else metricas_estadia(y, p)


def comparar_metodos(tarea: str, pred: np.ndarray, y: np.ndarray, bloques: np.ndarray, empate: float) -> dict:
    """Mide cada método fuera de muestra (y 'sin calibrar') y aplica la regla: gana el de mejor Brier (mortalidad) o devianza (estadía);
    si la diferencia relativa es menor al empate, se queda el más simple (el primero de la lista)."""
    clave = "brier" if tarea == "mortalidad" else "devianza_poisson"
    resultados = {"sin_calibrar": _metricas(tarea, y, pred)}
    oofs = {}
    for clase in METODOS[tarea]:
        oofs[clase.nombre] = validacion_cruzada_bloques(tarea, pred, y, bloques, clase)
        resultados[clase.nombre] = _metricas(tarea, y, oofs[clase.nombre])
    simple, complejo = (c.nombre for c in METODOS[tarea])
    s, c = resultados[simple][clave], resultados[complejo][clave]
    elegido = complejo if (s - c) / s > empate else simple
    return {"metricas": resultados, "elegido": elegido, "oof": oofs}


def ajustar_final(tarea: str, nombre: str, pred_2023: np.ndarray, y_2023: np.ndarray, anios: np.ndarray):
    """Recalibrador definitivo: se ajusta con todo 2023 (y solo con 2023)."""
    _exigir_anio(anios, 2023)
    clase = next(c for c in METODOS[tarea] if c.nombre == nombre)
    return clase().fit(pred_2023, y_2023)
