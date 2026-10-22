"""Prohíbe entrenar o recalibrar mientras se evalúa (plan de corrección 4.3; auditoría §3.5).

La puerta G4 original buscaba en el código el texto de una llamada a fit. No detectaba ``fit_transform`` ni un
entrenamiento llamado desde otro módulo. Esta guardia intercepta la operación misma: dentro de
``prohibir_ajustes()``, cualquier ``fit``, ``partial_fit``, ``fit_transform``, ``fit_predict`` o ``refit`` de un
estimador de scikit-learn, LightGBM o XGBoost, ``lightgbm.train``, ``xgboost.train`` y el ``fit`` de los
recalibradores de la Fase 4 lanzan ``AjusteProhibido``.

``prueba_humo()`` corre la evaluación completa (analisis.evaluar) con los modelos congelados sobre datos sintéticos
y la guardia activa. La usan la puerta G4 y las pruebas.
"""
from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

METODOS = ("fit", "partial_fit", "fit_transform", "fit_predict", "refit")
# Codificadores de etiquetas: las métricas de scikit-learn (log_loss, por ejemplo) los «ajustan» por dentro para
# conocer las clases. No aprenden nada del modelo, así que se permiten.
PERMITIDOS = ("sklearn.preprocessing._label",)


class AjusteProhibido(RuntimeError):
    pass


def _subclases(raiz: type) -> set[type]:
    vistas, pila = set(), [raiz]
    while pila:
        c = pila.pop()
        for s in c.__subclasses__():
            if s not in vistas:
                vistas.add(s)
                pila.append(s)
    return vistas | {raiz}


def _objetivos() -> list[tuple[object, str]]:
    import lightgbm
    import sklearn.base
    import xgboost
    from modelado import calibracion

    clases = {c for c in _subclases(sklearn.base.BaseEstimator) if not c.__module__.startswith(PERMITIDOS)} | {lightgbm.Booster}
    clases |= {v for v in vars(calibracion).values() if isinstance(v, type) and v.__module__ == calibracion.__name__}
    obj = [(c, m) for c in clases for m in METODOS if m in vars(c)]
    obj += [(lightgbm, "train"), (lightgbm, "cv"), (xgboost, "train"), (xgboost, "cv")]
    return obj


@contextlib.contextmanager
def prohibir_ajustes():
    """Dentro del bloque, cualquier entrenamiento o recalibración lanza AjusteProhibido."""
    originales = []
    for dueno, nombre in _objetivos():
        original = vars(dueno)[nombre] if isinstance(dueno, type) else getattr(dueno, nombre)

        def bloqueo(*a, _n=f"{getattr(dueno, '__name__', dueno)}.{nombre}", **k):
            raise AjusteProhibido(f"se intentó entrenar o recalibrar durante la evaluación: {_n}")
        originales.append((dueno, nombre, original))
        setattr(dueno, nombre, bloqueo)
    try:
        yield
    finally:
        for dueno, nombre, original in reversed(originales):
            setattr(dueno, nombre, original)


# ------------------------------------------------------------------ evaluación completa sobre datos sintéticos
def _categorias(tarea: str) -> dict[str, list]:
    import joblib

    from .prediccion import carpeta_final
    from modelado.datos import CATEGORICAS
    obj = joblib.load(carpeta_final(tarea) / "modelo_completo.joblib").objeto
    return dict(zip(CATEGORICAS, obj.booster_.pandas_categorical))


def datos_sinteticos(hospitales: int = 12, por_hospital: int = 900, semilla: int = 3) -> dict[str, pd.DataFrame]:
    """Cohortes D, E y C con el esquema de los modelos congelados y valores tomados de su vocabulario."""
    from modelado.datos import BINARIAS, CATEGORICAS
    rng = np.random.default_rng(semilla)
    cats = _categorias("mortalidad")
    n = hospitales * por_hospital
    d = pd.DataFrame({"ID_EPISODIO": np.arange(1, n + 1), "COD_HOSPITAL": np.repeat([f"9{i:05d}" for i in range(hospitales)], por_hospital),
                      "EDAD_ANIOS": rng.uniform(40, 95, n)})
    for c in CATEGORICAS:
        d[c] = rng.choice([v for v in cats[c] if v not in ("RARO", "(sin dato)", "(sin mapeo)", "SIN_DATO")][:12], n)
    for c in BINARIAS:
        d[c] = (rng.random(n) < (0.25 if c == "NIVEL_CUIDADO_INGRESO" else 0.10)).astype(np.int8)
    mort = d.assign(y=(rng.random(n) < 0.08).astype(np.int8))
    est = d.assign(y=rng.integers(1, 20, n))
    exh = d[["ID_EPISODIO", "COD_HOSPITAL", "CIE10_3C", "SEXO"]].assign(banda_edad="[60.0, 70.0)", y=rng.poisson(4, n))
    tras = d[["ID_EPISODIO"]].assign(ES_TRASLADO_SALIENTE=rng.random(n) < 0.03)
    ent = d[["ID_EPISODIO"]].assign(ES_TRASLADO_ENTRANTE=rng.random(n) < 0.05)
    return {"mort": mort, "est": est, "exh": exh, "tras": tras, "ent": ent}


def prueba_humo(B: int = 5) -> dict:
    """Corre analisis.evaluar completo con la guardia activa. Devuelve el resumen; falla si algo entrena."""
    from . import analisis
    from .config import RUTA_PROTOCOLO, cargar_protocolo
    prot = cargar_protocolo(RUTA_PROTOCOLO)
    s = datos_sinteticos()
    with tempfile.TemporaryDirectory() as tmp, prohibir_ajustes():
        r = analisis.evaluar(s["mort"], s["est"], s["exh"], s["tras"], prot, Path(tmp), B=B, entrantes=s["ent"])
    return r["resumen"]
