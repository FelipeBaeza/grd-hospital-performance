import numpy as np
import pandas as pd
from evaluacion import embudo, hospitales
from evaluacion.config import cargar_protocolo


def test_embudo_sin_diferencias_reales_marca_poco():
    esp = np.exp(np.random.default_rng(0).uniform(np.log(15), np.log(900), 65))
    r = embudo.simular_nulo(esp, 300, 2)
    assert 0.02 <= r["fuera_95"] <= 0.09 and r["fuera_998"] <= 0.01


def test_sobredispersion_ensancha_limites():
    esp = np.array([50.0, 200.0, 800.0])
    lo0, hi0 = embudo.limites(esp, 0.0, 1.96)
    lo1, hi1 = embudo.limites(esp, 0.02, 1.96)
    assert (hi1 > hi0).all() and (lo1 < lo0).all()


def test_tau2_cero_sin_variacion_real_y_positiva_con_ella():
    rng = np.random.default_rng(1)
    esp = np.full(60, 300.0)
    assert embudo.tau2(rng.poisson(esp), esp) < 0.01
    real = np.exp(rng.normal(0, 0.2, 60))
    assert embudo.tau2(rng.poisson(esp * real), esp) > 0.01


def test_ic_poisson_contiene_el_observado():
    obs, esp = np.array([20, 100]), np.array([22.0, 90.0])
    lo, hi = embudo.ic_poisson(obs, esp)
    assert (lo < obs / esp).all() and (hi > obs / esp).all()


def test_tabla_hospitales_volumen_minimo_y_sin_ranking():
    rng = np.random.default_rng(2)
    h = np.concatenate([np.full(2000, 1), np.full(1500, 2), np.full(100, 3)])
    e = np.full(len(h), 0.05)
    y = rng.binomial(1, e)
    t, r = hospitales.tabla(h, y, e, "mortalidad", cargar_protocolo())
    assert list(t["cod_hospital"]) == [1, 2]                      # el hospital de 100 egresos queda fuera
    assert r["n_bajo_volumen"] == 1 and r["egresos_bajo_volumen"] == 100
    assert t["cod_hospital"].is_monotonic_increasing             # orden por código, no por O/E


def test_oe_de_mortalidad_es_observado_sobre_esperado():
    h = np.full(1000, 7)
    e = np.full(1000, 0.1)
    y = np.zeros(1000); y[:80] = 1
    t, _ = hospitales.tabla(h, y, e, "mortalidad", cargar_protocolo())
    assert abs(t["oe"].iloc[0] - 0.8) < 1e-9


def test_subgrupos_sin_celdas_pequenas_ni_complementos(tmp_path):
    """Incidente I-11: los grupos pequeños se combinan, no queda ninguna celda oculta y el escáner por propiedad no
    reconstruye ningún conteo; con agrupar_pequenos=False se reproduce el método anterior."""
    from evaluacion import privacidad_propiedad as pp
    from evaluacion import subgrupos
    rng = np.random.default_rng(9)
    n = 6000
    df = pd.DataFrame({"EDAD_ANIOS": rng.uniform(18, 95, n), "SEXO": rng.choice(["HOMBRE", "MUJER"], n),
                       "GRUPO_CLINICO": rng.choice(["I", "II", "III"], n), "COD_HOSPITAL": rng.choice(["a", "b", "c"], n), "ELIX_01": 0})
    df.loc[:4, "SEXO"] = "SIN_DATO"                                # 5 egresos: grupo omitido con el método anterior
    df.loc[5:60, "GRUPO_CLINICO"] = "RARO"                         # 56 egresos con pocas muertes
    y = (rng.random(n) < 0.05).astype(float)
    y[5:60] = 0
    y[5:8] = 1                                                     # 3 muertes en el grupo raro
    p = np.full(n, 0.05)
    antes = subgrupos.tabla(df, y, p, "mortalidad", 10, agrupar_pequenos=False)
    assert antes["observado"].isna().any() or (antes.groupby("variable")["n"].sum().nunique() > 1)
    t = subgrupos.tabla(df, y, p, "mortalidad", 10)
    assert t["observado"].notna().all() and (t["observado"] >= 10).all() and (t["n"] >= 10).all()
    assert t.groupby("variable")["n"].sum().nunique() == 1 and t.groupby("variable")["observado"].sum().nunique() == 1
    assert "HOMBRE o SIN_DATO" in set(t["grupo"]) and any(" o RARO" in g or "RARO o " in g for g in t["grupo"])
    t.to_csv(tmp_path / "subgrupos_mortalidad.csv", index=False)
    assert pp.reconstrucciones(tmp_path) == []
    antes.to_csv(tmp_path / "subgrupos_mortalidad.csv", index=False)
    assert pp.reconstrucciones(tmp_path) != []                    # el método anterior sí deja conteos reconstruibles
