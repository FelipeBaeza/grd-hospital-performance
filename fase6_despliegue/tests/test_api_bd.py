"""F3: consistencia API / base de datos / tablas de la Fase 5, ausencia de ranking y de nombres, validación de entradas."""
import csv

import pytest

from conftest import DIR_DATOS


def _csv(n):
    with open(DIR_DATOS / n, newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("tarea", ["mortalidad", "estadia"])
def test_api_coincide_con_tablas_fase5(cliente, tarea):
    api = cliente.get(f"/indicadores/{tarea}").json()
    ref = {f["cod_hospital"]: f for f in _csv("indicador_hospital.csv") if f["tarea"] == tarea}
    assert api["n_hospitales"] == len(api["puntos"]) == len(ref) == len(_csv("hospital.csv"))
    for p in api["puntos"]:
        f = ref[p["cod_hospital"]]
        assert p["n_egresos"] == int(f["n_egresos"]) and p["n_egresos"] % 10 == 0           # redondeado a la decena
        assert not {"observado_pub", "esperado", "lim_inf_95", "lim_sup_95", "lim_inf_998", "lim_sup_998"} & set(p)
        if f["oe"] == "":
            assert p["oe"] is None
        else:
            assert p["oe"] == pytest.approx(float(f["oe"]))
            assert p["zona"] == f["zona"]


def test_ficha_coincide_con_lista(cliente):
    cods = cliente.get("/hospitales").json()["hospitales"]
    assert len(cods) == len(_csv("hospital.csv"))
    for c in cods[:10]:
        f = cliente.get(f"/hospitales/{c}").json()
        assert set(f["indicadores"]) == {"mortalidad", "estadia"}


def test_hospital_inexistente_404(cliente):
    assert cliente.get("/hospitales/999999").status_code == 404
    assert cliente.get("/indicadores/otra").status_code == 404


def test_orden_por_codigo_y_sin_ranking(cliente):
    for t in ("mortalidad", "estadia"):
        d = cliente.get(f"/indicadores/{t}").json()
        cods = [p["cod_hospital"] for p in d["puntos"]]
        assert cods == sorted(cods)
        for p in d["puntos"]:
            assert not {"ranking", "posicion", "rank", "lugar", "puesto"} & set(p)
    assert cliente.get("/hospitales").json()["hospitales"] == sorted(cliente.get("/hospitales").json()["hospitales"])


def test_nombres_vienen_del_catalogo_y_sin_rut(cliente):
    """Decisión E2 (08-11-2026): se muestra el nombre del hospital, tomado del catálogo; los hospitales sin nombre no se muestran."""
    cat = {f["cod_hospital"]: f["nombre"] for f in _csv("hospital.csv")}
    h = cliente.get("/hospitales").json()
    assert h["nombres"] == cat and all(cat.values())
    assert {f["cod_hospital"] for f in _csv("indicador_hospital.csv")} == set(cat)
    for p in cliente.get("/indicadores/mortalidad").json()["puntos"]:
        assert p["nombre"] == cat[p["cod_hospital"]]
    c = h["hospitales"][0]
    assert cliente.get(f"/hospitales/{c}").json()["nombre"] == cat[c]
    cuerpo = "".join(cliente.get(r).text.lower() for r in ("/hospitales", "/indicadores/mortalidad", f"/hospitales/{c}", "/metodologia"))
    assert "rut" not in cuerpo


def test_advertencias_visibles(cliente):
    adv = cliente.get("/metodologia").json()["advertencias"]
    assert len(adv) >= 4 and all(a["texto"] for a in adv)


def test_demo_valida_entradas_sin_ecoar(cliente, perfil):
    malo = {**perfil, "edad": 400, "diagnostico_principal": "SECRETO_XYZ"}
    r = cliente.post("/demo/riesgo", json=malo)
    assert r.status_code == 422 and "SECRETO_XYZ" not in r.text
    r = cliente.post("/demo/riesgo", json={**perfil, "diagnostico_principal": "ZZZ"})
    assert r.status_code == 422 and "ZZZ" not in r.text


def test_demo_sin_cache(cliente, perfil):
    r = cliente.post("/demo/riesgo", json=perfil)
    assert r.status_code == 200 and "no-store" in r.headers["cache-control"]


def test_no_se_publican_observado_esperado_ni_limites():
    """Enmienda 03 (incidente I-12): restando dos versiones de la cohorte se reconstruyen conteos de 1 a 9; por eso no se publican."""
    prohibidas = {"observado", "observado_pub", "esperado", "lim_inf_95", "lim_sup_95", "lim_inf_998", "lim_sup_998"}
    for archivo in ("indicador_hospital.csv", "subgrupo.csv"):
        filas = _csv(archivo)
        assert filas and not prohibidas & set(filas[0]), archivo
        assert all(int(f["n_egresos" if "n_egresos" in f else "n"]) % 10 == 0 for f in filas), archivo


def test_api_no_expone_curva_ni_esperados(cliente):
    d = cliente.get("/indicadores/mortalidad").json()
    assert "curva" not in d
    ficha = cliente.get("/hospitales/" + d["puntos"][0]["cod_hospital"]).json()
    for f in ficha["indicadores"].values():
        assert not {"observado_pub", "esperado", "lim_inf_95", "lim_sup_95"} & set(f)
    assert not {"observado_pub", "esperado"} & set(cliente.get("/subgrupos/mortalidad").json()["subgrupos"][0])
