"""Enmienda 02 (borrador) sobre datos sintéticos: guardia, filtros equivalentes a prep/cohortes.py, publicación y
corrida de punta a punta usando como base la enmienda 01."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from enmienda01 import comun as c1
from enmienda02 import comun, filtros, publicar

spec = importlib.util.spec_from_file_location("sintetico_enmienda", Path(__file__).with_name("sintetico_enmienda.py"))
sint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sint)


def test_protocolo_v3_bien_formado():
    enm = comun.cargar_enmienda()
    ids = [a["id"] for a in enm["analisis"]]
    assert ids == ["E2.1", "E2.2", "E2.3", "E2.4a", "E2.4b", "E2.5", "E2.6a", "E2.6b"] and len(set(ids)) == len(ids)
    assert enm["huella_base"] == c1.sha256_archivo(comun.RAIZ / "config" / "protocolo_fase5_v2.toml")
    assert enm["regla_lectura"]["min_misma_zona"] == 0.80 and enm["regla_lectura"]["min_rho"] == 0.90


def test_el_codigo_de_la_enmienda_02_existe():
    for a in comun.ARCHIVOS_CODIGO:
        assert (comun.RAIZ / a).is_file(), a


def test_sin_registro_no_corre(tmp_path):
    with pytest.raises(comun.EnmiendaNoAutorizada, match="enmienda 02 no está registrada"):
        comun.verificar_autorizacion(tmp_path, ["x.py"])


def test_filtros_equivalen_a_las_opciones_de_la_fase_3():
    """Las cohortes D y E de la Fase 3 menos los episodios de filtros.quitar son las de prep/cohortes.py con las opciones."""
    from prep import cohortes
    rng = np.random.default_rng(1)
    n = 400
    b = pl.DataFrame({"MDC": rng.choice(["04", "14"], n).tolist(), "GRD_ESTADO": ["agrupable"] * n, "EDAD_ANIOS": [50] * n,
                      "TIPO_ACTIVIDAD": ["HOSPITALIZACIÓN"] * n, "N_COMORB_MDC14": rng.integers(0, 3, n).tolist(),
                      "ROL": rng.choice(["EVENTO", "NO_EVENTO", "CENSURADO"], n).tolist(), "ESTANCIA_DIAS": rng.integers(0, 6, n).tolist(),
                      "ETIQUETA_TIPOALTA": rng.choice(["DOMICILIO", "ALTA VOLUNTARIA", "FUGA DEL PACIENTE"], n).tolist()},
                     schema_overrides={"N_COMORB_MDC14": pl.Int32, "EDAD_ANIOS": pl.Int32, "ESTANCIA_DIAS": pl.Int32}).with_row_index("ID_EPISODIO")
    b = b.with_columns(pl.when(pl.col("MDC") == "14").then(pl.col("N_COMORB_MDC14")).alias("N_COMORB_MDC14"))
    hoy = cohortes.marcar(b)
    op = cohortes.opciones_enmienda02(pl.read_csv(comun.FASE3 / "config" / "tipoalta_roles.csv"))
    nueva = cohortes.marcar(b, opciones=op)
    ep = pd.DataFrame({"ID_EPISODIO": b["ID_EPISODIO"].to_numpy(), "MDC14": (b["MDC"] == "14").to_numpy(),
                       "FUERA_E": b["ETIQUETA_TIPOALTA"].is_in(list(op.excluir_tipoalta_e)).to_numpy(), "Z515": False})
    ids_d = set(hoy.filter(pl.col("D"))["ID_EPISODIO"]) - set(filtros.quitar(ep, "E2.3", "mortalidad"))
    ids_e = set(hoy.filter(pl.col("E"))["ID_EPISODIO"]) - set(filtros.quitar(ep, "E2.3", "estadia"))
    assert ids_d == set(nueva.filter(pl.col("D"))["ID_EPISODIO"])
    assert ids_e == set(nueva.filter(pl.col("E"))["ID_EPISODIO"])


def test_tasas_con_conteo_de_1_a_9_se_suprimen():
    t = pd.DataFrame({"diseno": "anidado", "anio": 2023, "cod_hospital": ["a", "b", "c"], "egresos_c": [1000, 1000, 1000],
                      "z515": [0, 5, 40], "alta_domiciliaria": [10, 0, 9], "voluntaria_fuga": [20, 30, 40]})
    for k in ("z515", "alta_domiciliaria", "voluntaria_fuga"):
        t[f"tasa_{k}"] = t[k] / t["egresos_c"]
    p = publicar.tasas_publicables(t)
    assert p["tasa_z515_pct"].tolist()[0] == 0.0 and np.isnan(p["tasa_z515_pct"][1]) and p["tasa_z515_pct"][2] == 4.0
    assert np.isnan(p["tasa_alta_domiciliaria_pct"][2]) and p["nota"][0] == ""
    assert "egresos_c" not in p.columns and not any(c in p.columns for c in ("z515", "alta_domiciliaria", "voluntaria_fuga"))


def test_punta_a_punta_con_datos_sinteticos(tmp_path, monkeypatch):
    from enmienda01 import indicador as i1, modelo as m1
    from enmienda02 import indicador, modelo
    from evaluacion import prediccion
    rutas = sint.crear(tmp_path)
    monkeypatch.setattr(prediccion, "predecir", sint.prediccion_falsa)
    int01, int02 = tmp_path / "interna01", tmp_path / "interna02"
    m1.correr(rutas["capas"], rutas["trabajo4"], rutas["trabajo5"], int01, c1.cargar_enmienda(), log=lambda *_: None)
    enm = comun.cargar_enmienda()
    eps = {a: filtros.episodios(rutas["capas"], rutas["trabajo5"], a) for a in range(2019, 2025)}
    assert eps[2023]["Z515"].any() and eps[2023]["FUERA_E"].any() and eps[2023]["MDC14"].any()
    m = modelo.correr(rutas["capas"], rutas["trabajo4"], rutas["trabajo5"], int01, int02, enm, eps, log=lambda *_: None)
    assert sorted(m.groupby("tarea")["analisis"].apply(lambda s: tuple(sorted(set(s))))) == [("E2.1", "E2.2", "E2.3"), ("E2.2", "E2.3")]
    assert (m["egresos_que_salen_de_la_prueba" if "egresos_que_salen_de_la_prueba" in m else "n_fuera_prueba"] > 0).all()
    r = indicador.correr(rutas["capas"], rutas["trabajo5"], int01, int02, enm, eps, comun.RAIZ.parent, usar_deis=False, log=lambda *_: None)
    ind = r["indicador"]
    assert (ind.loc[ind["analisis"].isin(["E2.4a", "E2.4b"]), "estado"].str.startswith("pendiente")).all()
    assert set(ind.loc[ind["estado"] == "corrido", "analisis"]) == {"E2.5", "E2.6b"}
    assert len(r["correlaciones_contexto"]) == 2 * 2 * 3
    pub = tmp_path / "pub"
    publicar.escribir(m, r, filtros.muertes_obstetricas(eps), pub, {"prueba": True})
    assert (pub / "tasas_contexto_hospital.csv").exists()
    assert pd.read_csv(pub / "muertes_obstetricas_excluidas.csv")["muertes"].astype(str).iloc[0] != ""


def test_e24_con_catalogo_deis_sintetico(tmp_path, monkeypatch):
    """E2.4a y E2.4b con un catálogo de tipos sobre los hospitales sintéticos (3 institutos, como el real)."""
    from enmienda01 import modelo as m1
    from enmienda02 import indicador
    from evaluacion import prediccion
    rutas = sint.crear(tmp_path)
    monkeypatch.setattr(prediccion, "predecir", sint.prediccion_falsa)
    int01, int02 = tmp_path / "interna01", tmp_path / "interna02"
    m1.correr(rutas["capas"], rutas["trabajo4"], rutas["trabajo5"], int01, c1.cargar_enmienda(), log=lambda *_: None)
    tipos = {c: ("instituto monográfico" if c in ("110110", "112103", "112104") else "general de mediana complejidad" if c.endswith("1")
                 else "general de alta complejidad") for c in sint.HOSPITALES}
    cat = tmp_path / "tipo_establecimiento.csv"
    pd.DataFrame({"cod_hospital": list(tipos), "tipo": list(tipos.values())}).to_csv(cat, index=False)
    monkeypatch.setattr(comun, "CATALOGO_DEIS", cat)
    eps = {a: filtros.episodios(rutas["capas"], rutas["trabajo5"], a) for a in range(2019, 2025)}
    r = indicador.correr(rutas["capas"], rutas["trabajo5"], int01, int02, comun.cargar_enmienda(), eps, comun.RAIZ.parent, usar_deis=True, log=lambda *_: None)
    ind = r["indicador"]
    assert set(ind.loc[ind["estado"] == "corrido", "analisis"]) == {"E2.4a", "E2.4b", "E2.5", "E2.6b"}
    assert set(r["marcas"].loc[r["marcas"]["analisis"] == "E2.4b", "cod_hospital"]) <= {"110110", "112103", "112104"}
