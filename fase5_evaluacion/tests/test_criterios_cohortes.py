import numpy as np
import pandas as pd
from evaluacion import cohortes, criterios
from evaluacion.config import cargar_protocolo


def test_remuestreo_usa_hospitales_completos():
    h = np.array([1] * 3 + [2] * 5 + [3] * 2)
    for idx in criterios.remuestras(h, 5, 1):
        cont = pd.Series(h[idx]).value_counts()
        assert all(cont[k] % n == 0 for k, n in ((1, 3), (2, 5), (3, 2)) if k in cont)


def test_veredictos():
    assert criterios._veredicto(1.0, 0.97, 1.03, 0.95, 1.05) == "cumple"
    assert criterios._veredicto(1.0, 0.90, 1.10, 0.95, 1.05) == "cumple con incertidumbre"
    assert criterios._veredicto(1.2, 1.15, 1.25, 0.95, 1.05) == "no cumple"
    assert criterios._veredicto(0.02, 0.01, 0.03, minimo=0.0) == "cumple"
    assert criterios._veredicto(-0.01, -0.02, 0.0, minimo=0.0) == "no cumple"


def _datos(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.integers(0, 12, n)
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-3 + 1.2 * x)))
    y = rng.binomial(1, p)
    return y, p, np.clip(p * 0.5 + 0.02, 0.001, 0.999), h


def test_criterios_mortalidad_modelo_mejor_que_su_base():
    y, p, base, h = _datos()
    t = criterios.evaluar_mortalidad(y, p, base, h, cargar_protocolo(), B=30)
    f = t[t["criterio"].str.startswith("Discriminación")].iloc[0]
    assert f["valor"] >= 0 and "veredicto" in t.columns
    assert set(t["veredicto"]) >= {"informativa"}


def test_cohortes_reglas_de_la_fase_3():
    pred = pd.DataFrame({"ID_EPISODIO": list("abcdef"), "COD_HOSPITAL": 1})
    des = pd.DataFrame({"ID_EPISODIO": list("abcdef"), "ROL": ["EVENTO", "NO_EVENTO", "NO_EVENTO", "CENSURADO", "SIN_DATO", "NO_EVENTO"],
                        "ESTANCIA_DIAS": [3, 0, 10, 4, 5, 80]})
    m, e = cohortes.construir(pred, des, 53)
    assert list(m["ID_EPISODIO"]) == ["a", "b", "c", "f"]            # D: solo evento o no evento
    assert list(m["y"]) == [1, 0, 0, 0]
    assert list(e["ID_EPISODIO"]) == ["c", "f"]                      # E: sobrevivientes con estadía mayor que cero
    assert list(e["y"]) == [10, 53]                                  # con tope de 53 días
