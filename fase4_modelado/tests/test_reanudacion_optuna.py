"""Plan de corrección 4.8 (H-19): las reanudaciones de Optuna quedan registradas y, con semilla por prueba, una
corrida cortada y retomada da las mismas pruebas que una de corrido."""
import types

import numpy as np
import optuna
import pandas as pd

from modelado import tuning


def _reglas():
    return types.SimpleNamespace(ajuste={"pruebas_mortalidad": 12, "pruebas_estadia": 12, "submuestra_ajuste": 1.0}, semilla={"maestra": 20261018})


def _parchar(monkeypatch):
    monkeypatch.setattr(tuning, "pliegues", lambda df, reglas: [(None, None), (None, None)])

    def evaluar(tipo, tarea, df, tr, va, params, reglas, semilla, fraccion):     # puntaje determinista según los parámetros
        v = 0.3 - (np.log10(params["learning_rate"]) + 1.2) ** 2 - 0.1 * (params["subsample"] - 0.8) ** 2 + 0.001 * (semilla % 2)
        return {"pr_auc": v}, 10
    monkeypatch.setattr(tuning, "_evaluar_pliegue", evaluar)


def _pruebas(r):
    t = r["tabla"].sort_values("number")[["learning_rate", "subsample", "num_leaves", "state"]].reset_index(drop=True)
    return t.astype({"num_leaves": int}).round(12)


def test_corrida_cortada_y_retomada_da_las_mismas_pruebas(tmp_path, monkeypatch):
    _parchar(monkeypatch)
    df = pd.DataFrame({"y": [0, 1]})
    seguida = tuning.ajustar_gbm("mortalidad", "lightgbm", df, _reglas(), tmp_path / "a", avisar=lambda *_: None, semilla_por_prueba=True)
    assert not (tmp_path / "a" / tuning.REGISTRO_REANUDACIONES).exists()
    # corte: 5 pruebas y una sexta que queda abierta
    tuning.ajustar_gbm("mortalidad", "lightgbm", df, _reglas(), tmp_path / "b", n_pruebas=5, avisar=lambda *_: None, semilla_por_prueba=True)
    est = optuna.load_study(study_name="mortalidad_lightgbm", storage=f"sqlite:///{tmp_path / 'b' / 'mortalidad_lightgbm.db'}")
    tuning.espacio(est.ask(), "lightgbm", "mortalidad")          # la prueba abierta ya tiene sus parámetros, como en un corte real
    retomada = tuning.ajustar_gbm("mortalidad", "lightgbm", df, _reglas(), tmp_path / "b", avisar=lambda *_: None, semilla_por_prueba=True)
    pd.testing.assert_frame_equal(_pruebas(seguida), _pruebas(retomada))
    reg = pd.read_csv(tmp_path / "b" / tuning.REGISTRO_REANUDACIONES)
    assert reg["pruebas_terminadas"].tolist() == [5] and str(reg["pruebas_abiertas"].iloc[0]) == "5"


def test_sin_semilla_por_prueba_la_reanudacion_igual_queda_registrada(tmp_path, monkeypatch):
    _parchar(monkeypatch)
    df = pd.DataFrame({"y": [0, 1]})
    tuning.ajustar_gbm("mortalidad", "lightgbm", df, _reglas(), tmp_path, n_pruebas=3, avisar=lambda *_: None)
    tuning.ajustar_gbm("mortalidad", "lightgbm", df, _reglas(), tmp_path, n_pruebas=4, avisar=lambda *_: None)
    reg = pd.read_csv(tmp_path / tuning.REGISTRO_REANUDACIONES)
    assert len(reg) == 1 and reg["pruebas_terminadas"].iloc[0] == 3 and not reg["semilla_por_prueba"].iloc[0]
