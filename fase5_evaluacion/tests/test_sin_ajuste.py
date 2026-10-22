"""Plan 4.3: la evaluación corre completa con los ajustes prohibidos, y la guardia detecta entrenamientos que la
puerta G4 original (expresión regular) no veía."""
import re

import numpy as np
import pytest

from evaluacion import sin_ajuste


def test_bloquea_sklearn_lightgbm_y_recalibradores():
    import lightgbm
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from modelado.calibracion import FactorUnico, Platt
    X, y = np.random.default_rng(0).normal(size=(50, 2)), np.r_[np.zeros(25), np.ones(25)]
    with sin_ajuste.prohibir_ajustes():
        for f in (lambda: LogisticRegression().fit(X, y), lambda: StandardScaler().fit_transform(X),
                  lambda: lightgbm.LGBMClassifier(n_estimators=2, verbose=-1).fit(X, y), lambda: lightgbm.train({}, lightgbm.Dataset(X, y)),
                  lambda: Platt().fit(np.full(50, 0.5), y), lambda: FactorUnico().fit(np.full(50, 2.0), y)):
            with pytest.raises(sin_ajuste.AjusteProhibido):
                f()
    LogisticRegression().fit(X, y)                      # fuera del bloque todo vuelve a funcionar
    assert StandardScaler().fit_transform(X).shape == (50, 2)


def test_canario_que_la_puerta_por_texto_no_detectaba(tmp_path, monkeypatch):
    """Un módulo que recalibra con fit_transform desde otro archivo: la expresión regular de G4 no lo ve; la guardia sí."""
    codigo = "from sklearn.preprocessing import StandardScaler\n\ndef recalibrar(x):\n    return StandardScaler().fit_transform(x)\n"
    (tmp_path / "modulo_canario.py").write_text(codigo)
    assert not re.search(r"\.fit\(|ajustar_final|entrenar\(", codigo)
    monkeypatch.syspath_prepend(str(tmp_path))
    import modulo_canario
    with sin_ajuste.prohibir_ajustes(), pytest.raises(sin_ajuste.AjusteProhibido, match="StandardScaler"):
        modulo_canario.recalibrar(np.ones((3, 1)))


def test_la_evaluacion_completa_corre_con_los_ajustes_prohibidos():
    r = sin_ajuste.prueba_humo(B=5)
    assert r["n_mortalidad"] > 0 and r["hospitales_mortalidad"]["n_elegibles"] > 0 and r["hospitales_estadia"]["n_elegibles"] > 0
    assert 0 < r["oe_global_mortalidad"] < 10


def test_prediccion_congelada_no_depende_del_orden_ni_de_las_categorias():
    """Auditoría §3.5, prueba 5: LightGBM vuelve a mapear las categorías con las del entrenamiento. La predicción de
    cada egreso no cambia si se reordenan las filas o si pandas ve otro conjunto de categorías."""
    import pandas as pd
    from evaluacion import prediccion
    from modelado.datos import CATEGORICAS
    d = sin_ajuste.datos_sinteticos(hospitales=2, por_hospital=300)["mort"]
    _, p = prediccion.predecir(d, prediccion.carpeta_final("mortalidad"))
    orden = np.random.default_rng(1).permutation(len(d))
    _, p2 = prediccion.predecir(d.iloc[orden].reset_index(drop=True), prediccion.carpeta_final("mortalidad"))
    assert np.allclose(p[orden], p2, rtol=0, atol=1e-12)
    d3 = d.copy()
    for c in CATEGORICAS:                                   # categorías en otro orden y con un nivel que no existe
        cats = sorted(d3[c].astype(str).unique(), reverse=True) + ["NIVEL_INEXISTENTE"]
        d3[c] = pd.Categorical(d3[c].astype(str), categories=cats)
    _, p3 = prediccion.predecir(d3, prediccion.carpeta_final("mortalidad"))
    assert np.allclose(p, p3, rtol=0, atol=1e-12)
