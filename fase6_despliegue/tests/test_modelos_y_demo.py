"""F3: modelos intactos, demo correcta frente a los modelos de la Fase 4, SHAP coherente."""
import json
import math
import shutil
import sys

import joblib
import numpy as np
import pytest

from app.motor import ModelosAlterados, MotorPrediccion
from conftest import DIR_DATOS, DIR_MODELOS, RAIZ

FASE4 = RAIZ.parent / "fase4_modelado"


@pytest.fixture(scope="module")
def motor():
    return MotorPrediccion(DIR_MODELOS, DIR_DATOS)


def test_huellas_coinciden_con_manifiesto(motor):
    assert all(v["coincide"] for v in motor.huellas.values())


def test_modelo_alterado_se_rechaza(tmp_path):
    copia = tmp_path / "modelos"
    shutil.copytree(DIR_MODELOS, copia)
    arch = copia / "mortalidad" / "modelo_lightgbm.txt"
    arch.write_text(arch.read_text() + "\n# alterado\n")
    with pytest.raises(ModelosAlterados):
        MotorPrediccion(copia, DIR_DATOS)


def test_recalibracion_alterada_se_rechaza(tmp_path):
    d = tmp_path / "datos"
    shutil.copytree(DIR_DATOS, d)
    r = json.loads((d / "recalibracion.json").read_text())
    r["estadia"]["fuente_sha256"] = "0" * 64
    (d / "recalibracion.json").write_text(json.dumps(r))
    with pytest.raises(ModelosAlterados):
        MotorPrediccion(DIR_MODELOS, d)


def test_shap_suma_el_resultado_final(motor, perfil):
    r = motor.predecir(perfil)
    m, e = r["mortalidad"], r["estadia"]
    assert sum(c["efecto"] for c in m["contribuciones"]) + m["base_logit"] == pytest.approx(m["logit"], abs=1e-9)
    assert sum(c["efecto"] for c in e["contribuciones"]) + e["base_log_dias"] == pytest.approx(e["log_dias"], abs=1e-9)
    assert m["probabilidad"] == pytest.approx(1 / (1 + math.exp(-m["logit"])))
    assert e["dias_esperados"] == pytest.approx(math.exp(e["log_dias"]))


@pytest.mark.skipif(not (FASE4 / "modelado").exists(), reason="no está el código de la Fase 4")
def test_demo_coincide_con_modelos_pickle_de_fase4(motor, perfil):
    sys.path.insert(0, str(FASE4))
    r = motor.predecir(perfil)
    df = motor.marco(perfil)
    mort = joblib.load(DIR_MODELOS / "mortalidad" / "modelo_completo.joblib")
    rec = joblib.load(DIR_MODELOS / "mortalidad" / "recalibrador.joblib")
    p_ref = float(np.asarray(rec.transformar(np.asarray(mort.predecir(df)))).ravel()[0])
    assert r["mortalidad"]["probabilidad"] == pytest.approx(p_ref, rel=1e-6, abs=1e-9)
    est = joblib.load(DIR_MODELOS / "estadia" / "modelo_completo.joblib")
    rec2 = joblib.load(DIR_MODELOS / "estadia" / "recalibrador.joblib")
    d_ref = float(np.asarray(rec2.transformar(np.asarray(est.predecir(df)))).ravel()[0])
    assert r["estadia"]["dias_esperados"] == pytest.approx(d_ref, rel=1e-6)


def test_mas_edad_y_comorbilidad_no_baja_el_riesgo_base(motor, perfil):
    joven = motor.predecir({**perfil, "edad": 30, "comorbilidades": []})
    viejo = motor.predecir({**perfil, "edad": 90, "comorbilidades": []})
    assert viejo["mortalidad"]["probabilidad"] > joven["mortalidad"]["probabilidad"]
