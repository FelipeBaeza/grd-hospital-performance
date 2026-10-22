"""Piezas de la enmienda 01 sobre datos sintéticos: guardia de autorización, embudo truncado, comparación, regla de lectura,
intervalos por hospital y supresión de estratos."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from enmienda01 import comun, publicar
from evaluacion import embudo, enmienda
from evaluacion.config import RAIZ


def test_expandir_columnas():
    assert comun.expandir_columnas(["ELIX_01..ELIX_03", "SEXO"]) == ["ELIX_01", "ELIX_02", "ELIX_03", "SEXO"]
    assert len(comun.expandir_columnas(["ELIX_01..ELIX_31"])) == 31


def test_truncamiento_sin_colas_es_dersimonian_laird():
    rng = np.random.default_rng(3)
    esp = rng.uniform(50, 500, 60)
    obs = rng.poisson(esp * np.exp(rng.normal(0, 0.2, 60)))
    assert comun.tau2_truncado(obs, esp, 1.0, 0.0) == pytest.approx(embudo.tau2(obs, esp, 1.0, 0.0))
    extremos = obs.copy().astype(float)
    extremos[:3] = esp[:3] * 4                                  # tres hospitales muy alejados inflan τ² sin truncar
    assert comun.tau2_truncado(extremos, esp, 1.0, 0.10) < embudo.tau2(extremos, esp, 1.0, 0.0)


def test_comparar_y_regla():
    base = pd.DataFrame({"cod_hospital": list("abcde"), "zona": ["dentro de lo esperado"] * 4 + ["alerta alta"], "oe_interno": [1, 1.1, 0.9, 1.05, 1.4]})
    otra = base.assign(zona=["dentro de lo esperado"] * 5, oe_interno=[1, 1.1, 0.9, 1.05, 1.3])
    c = comun.comparar(base, otra)
    assert c["pct_misma_zona"] == pytest.approx(0.8) and c["n_cambian"] == 1 and c["fuera_a_dentro"] == 1 and c["rho_oe"] == pytest.approx(1.0)
    assert c["cambios"] == "e: alerta alta → dentro de lo esperado"
    enm = {"regla_lectura": {"min_misma_zona": 0.80, "min_rho": 0.90}}
    assert comun.cumple_regla(c, enm)
    assert not comun.cumple_regla({**c, "pct_misma_zona": 0.79}, enm)
    assert not comun.cumple_regla({**c, "rho_oe": 0.89}, enm)


def test_oe_con_ic_contiene_el_punto_y_es_reproducible():
    rng = np.random.default_rng(5)
    h = np.repeat(np.arange(30), 200)
    p = np.full(len(h), 0.05)
    y = (rng.random(len(h)) < 0.05).astype(float)
    a, b = comun.oe_con_ic(h, y, p, B=300), comun.oe_con_ic(h, y, p, B=300)
    assert a == b and a[1] <= a[0] <= a[2]


def test_estratos_con_suprimido_no_publican_oe():
    e = pd.DataFrame({"tarea": ["mortalidad", "mortalidad", "estadia"], "contiene_suprimido": [True, False, True],
                      "oe": [0.5, 1.0, 0.9], "ic_inf": [0.3, 0.9, 0.8], "ic_sup": [0.8, 1.1, 1.0]})
    s = publicar.suprimir_estratos(e)
    assert np.isnan(s.loc[0, "oe"]) and s.loc[1, "oe"] == 1.0 and s.loc[2, "oe"] == 0.9   # en estadía no hay conteos de personas


@pytest.fixture
def raiz(tmp_path: Path) -> Path:
    for rel in (enmienda.PROTOCOLO_V1, enmienda.PROTOCOLO_V2, enmienda.TEXTO):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(RAIZ / rel, tmp_path / rel)
    (tmp_path / "evaluacion").mkdir()
    (tmp_path / "evaluacion" / "m.py").write_text("x = 1\n")
    (tmp_path / "script.py").write_text("print(1)\n")
    return tmp_path


def test_guardia_exige_registros_y_codigo_intacto(raiz):
    with pytest.raises(comun.EnmiendaNoAutorizada, match="no está registrada"):
        comun.verificar_autorizacion(raiz, ["script.py"])
    enmienda.firmar("A", "P", "2026-10-21", raiz=raiz, sin_constancia=True)
    with pytest.raises(comun.EnmiendaNoAutorizada, match="antes de correr"):
        comun.verificar_autorizacion(raiz, ["script.py"])
    enmienda.antes_de_correr(["script.py"], raiz=raiz)
    assert comun.verificar_autorizacion(raiz, ["script.py"])["registro_corrida"]
    with pytest.raises(comun.EnmiendaNoAutorizada, match="código sin registrar"):
        comun.verificar_autorizacion(raiz, ["script.py", "otro.py"])
    (raiz / "script.py").write_text("print(2)\n")
    with pytest.raises(comun.EnmiendaNoAutorizada, match="cambió después de registrarse"):
        comun.verificar_autorizacion(raiz, ["script.py"])


def test_el_codigo_real_de_la_enmienda_existe():
    for a in comun.ARCHIVOS_CODIGO:
        assert (RAIZ / a).is_file(), a


def test_punta_a_punta_con_datos_sinteticos(tmp_path, monkeypatch):
    """Corre las etapas modelo, indicador y publicar sobre capas sintéticas (sin la guardia, que se prueba aparte)."""
    from enmienda01 import indicador, modelo
    from evaluacion import prediccion
    import importlib.util
    spec = importlib.util.spec_from_file_location("sintetico_enmienda", Path(__file__).with_name("sintetico_enmienda.py"))
    sint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sint)
    rutas = sint.crear(tmp_path)
    monkeypatch.setattr(prediccion, "predecir", sint.prediccion_falsa)
    enm = comun.cargar_enmienda()
    interna = tmp_path / "interna"
    m = modelo.correr(rutas["capas"], rutas["trabajo4"], rutas["trabajo5"], interna, enm, log=lambda *_: None)
    assert set(m["analisis"]) == {"base", "2.1", "2.2", "2.3", "2.4", "2.5"} and len(m) == 2 * 2 * 6
    assert m["cumple_regla"].dropna().isin([True, False]).all()
    r = indicador.correr(rutas["capas"], rutas["trabajo5"], interna, enm, comun.RAIZ.parent, log=lambda *_: None)
    assert set(r["indicador"]["analisis"]) == {"base", "2.6", "3.1a", "3.1b", "3.3", "3.4a", "3.4b"}
    assert len(r["estabilidad"]) == 2 and len(r["validacion_p5b"]) == 4
    pub = tmp_path / "pub"
    tablas = publicar.escribir(m, r, pub, {"prueba": True})
    assert (pub / "resumen_modelo.csv").exists() and "metadatos.json" in [p.name for p in pub.iterdir()]
    assert "calibracion_por_estrato.csv" in tablas
